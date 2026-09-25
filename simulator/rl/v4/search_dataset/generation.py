"""Timing and simulator dataset generation orchestration."""

from __future__ import annotations

from typing import Any
import numpy as np
from simulator.ruleset import load_fixed_ruleset
from simulator.rl.v4.distillation import DistillationSample
from simulator.rl.foundation.opponent_pool import OpponentPool
from simulator.rl.v4.timing_curriculum import TimingConfig, generate_timing_pool
from .config import SearchDatasetConfig, _hash_order
from .inference import _predict_records
from .records import _base_label, _new_sim_env, _record_from_env, _disagrees, _spell_opp
from .targets import _adopt_target, _to_search_sample
from .active import _generate_active_search_dataset
from .selection import (
    _select_sim_records,
    _select_candidates,
    _search_stats,
    _assert_search_balance,
)
from .workers import TimingSearchTask, SimSearchTask, run_search_tasks


def _build_timing_tasks(
    timing_pool: list[dict[str, Any]], timing_config: TimingConfig, config: SearchDatasetConfig,
) -> tuple[list[TimingSearchTask], set[str]]:
    """Prioritize transition and disagreement sequences for deterministic replay."""
    # Transition sequences from the timing pool metadata.
    by_seq: dict[str, list[int]] = {}
    for idx, record in enumerate(timing_pool):
        by_seq.setdefault(str(record["provenance"]["seq_id"]), []).append(idx)
    transition_seqs: set[str] = set()
    for seq_id, indices in by_seq.items():
        ordered = sorted(indices, key=lambda i: timing_pool[i]["offset"])
        modes = [timing_pool[i]["target"].mode for i in ordered]
        if any(a == 0 and b == 1 for a, b in zip(modes, modes[1:])):
            transition_seqs.add(seq_id)
    # Pass-2 replay selection: transitions first, then disagreement-heavy
    # and stratified filler sequences, all deterministic.
    seq_stats: list[tuple[int, str]] = []
    for seq_id, indices in by_seq.items():
        priority = (3 if seq_id in transition_seqs else 0) + (
            2 if any(_disagrees(timing_pool[i]) for i in indices) else 0
        )
        seq_stats.append((-priority, seq_id))
    seq_stats.sort(key=lambda item: (item[0], _hash_order(config.seed, item[1])))
    replay_count = min(config.replay_sequences, len(seq_stats))
    replay_seqs = {seq_id for _, seq_id in seq_stats[:replay_count]}

    # Group requested replay offsets by sequence.  Every capture of a
    # selected sequence is searched, exactly as the serial hook would.
    seq_offsets: dict[str, list[int]] = {}
    for idx, record in enumerate(timing_pool):
        seq_id = str(record["provenance"]["seq_id"])
        if seq_id in replay_seqs:
            seq_offsets.setdefault(seq_id, []).append(idx)
    timing_tasks: list[TimingSearchTask] = []
    for seq_id in sorted(replay_seqs):
        order = sorted(seq_offsets.get(seq_id, []), key=lambda i: timing_pool[i]["offset"])
        if not order:
            continue
        family, seq_idx = seq_id.split(":", 1)[0], int(seq_id.split(":", 1)[1])
        timing_tasks.append(
            TimingSearchTask(
                family=family,
                seq_idx=seq_idx,
                timing_config=timing_config,
                offsets=tuple(int(timing_pool[i]["offset"]) for i in order),
                records=tuple(timing_pool[i] for i in order),
                champs=tuple(timing_pool[i]["champ"] for i in order),
                horizon=config.effective_horizon,
                max_branches=config.max_branches,
                seed=config.seed,
                continuation_mode=config.continuation_mode,
                opponent_strategy=config.opponent_strategy,
            )
        )
    return timing_tasks, transition_seqs


def generate_search_dataset(
    config: SearchDatasetConfig,
    champion: Any,
    *,
    workers: int = 1,
    progress: Any = None,
) -> tuple[list[DistillationSample], dict[str, Any]]:
    """Build the Stage-2 search dataset (deterministic end to end).

    ``workers`` selects inline serial execution (``1``, the reference
    path) or a spawn-context process pool over rebuildable search tasks;
    ``progress``, when given, is called as ``progress(done, total)`` as
    search tasks complete.
    """

    if config.is_active:
        # Active trajectories retain recurrent history and live simulator
        # forks, so they are deliberately generated in one process.  The
        # legacy state/replay path below remains parallelizable and unchanged.
        return _generate_active_search_dataset(config, champion, progress=progress)

    # Pass 1a: timing pool records (no searching yet).
    timing_config = TimingConfig(n_states=config.timing_pool_states, seed=config.seed,
                                 family_weights=dict(config.timing_family_weights)
                                 if config.timing_family_weights else None)
    timing_pool = generate_timing_pool(timing_config)
    # Pass 1b: sim-natural records (envs rebuilt for search in pass 2).
    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=config.seed)
    sim_mix = dict(config.sim_family_mix) if config.sim_family_mix else None
    sim_records: list[dict[str, Any]] = []
    for index in range(config.sim_candidates):
        env, family, source = _new_sim_env(ruleset, pool, config.seed, index, config.seed, sim_mix)
        try:
            record = _record_from_env(env, family, source, ruleset)
        finally:
            del env
        if record is not None:
            record["sim_index"] = index
            sim_records.append(record)
    # Batched champion inference joined to every record.
    timing_preds = _predict_records(champion, timing_pool)
    sim_preds = _predict_records(champion, sim_records)
    for record, pred in zip(timing_pool, timing_preds):
        record["champ"] = pred
        record["base_target"] = _base_label(record)
    for record, pred in zip(sim_records, sim_preds):
        record["champ"] = pred
        record["base_target"] = _base_label(record)

    timing_tasks, transition_seqs = _build_timing_tasks(timing_pool, timing_config, config)
    sim_to_search = _select_sim_records(sim_records, config)
    sim_tasks = [
        SimSearchTask(
            index=int(record["sim_index"]),
            record=record,
            champ=record["champ"],
            horizon=config.effective_horizon,
            max_branches=config.max_branches,
            seed=config.seed,
            pool_seed=config.seed,
            distill_seed=config.seed,
            sim_family_mix=sim_mix,
            continuation_mode=config.continuation_mode,
            opponent_strategy=config.opponent_strategy,
        )
        for record in sim_to_search
    ]
    timing_results, sim_results = run_search_tasks(
        timing_tasks, sim_tasks, workers=workers, progress=progress,
    )
    searched: list[dict[str, Any]] = []
    for task in timing_tasks:
        for offset, rec in zip(task.offsets, task.records):
            key = (task.family, int(task.seq_idx), int(offset))
            if key not in timing_results:
                continue  # mid-sequence TeacherError: keep partials, as serial
            searched.append({"record": rec, "search": timing_results[key], "sim": False})
    for task in sim_tasks:
        searched.append({"record": task.record, "search": sim_results[int(task.index)], "sim": True})

    # Adoption + oversampled stratified selection.
    candidates: list[dict[str, Any]] = []
    for entry in searched:
        record = entry["record"]
        base = record["base_target"]
        target, adopted, regret_rule, placement_gap = _adopt_target(
            record, entry["search"]["result"], base
        )
        candidates.append(
            {
                "record": record,
                "target": target,
                "adopted": adopted,
                "regret_rule": regret_rule,
                "placement_gap": placement_gap,
                "search": entry["search"],
                "base": base,
                "sim": entry["sim"],
                "spell": _spell_opp(record),
                "transition": (
                    str(record["provenance"]["seq_id"]) in transition_seqs
                    if not entry["sim"]
                    else False
                ),
            }
        )
    selected = _select_candidates(candidates, config)
    n_total = min(config.n_states, len(candidates))
    stats = _search_stats(selected)
    _assert_search_balance(stats, config, n_total)
    stats["n_searched"] = len(candidates)
    stats["adopted_rate_searched"] = (
        sum(1 for c in candidates if c["adopted"]) / max(1, len(candidates))
    )
    stats["mean_regret_searched"] = float(
        np.mean([c["regret_rule"] for c in candidates])
    ) if candidates else 0.0
    stats["timing_family_weights"] = (
        dict(config.timing_family_weights) if config.timing_family_weights else "default"
    )
    stats["sim_family_mix"] = (
        dict(config.sim_family_mix) if config.sim_family_mix else "default"
    )
    stats["horizon"] = int(config.effective_horizon)
    stats["max_branches"] = int(config.max_branches)
    stats["continuation_mode"] = config.continuation_mode
    stats["opponent_strategy"] = config.opponent_strategy
    stats["placement_target_mode"] = config.placement_target_mode
    stats["active"] = False
    stats["sequential"] = False
    samples = [
        _to_search_sample(
            c["record"], c["target"], c["search"], c["base"],
            c["adopted"], c["regret_rule"], c["placement_gap"], index,
            config.placement_target_mode,
        )
        for index, c in enumerate(selected)
    ]
    return samples, stats

