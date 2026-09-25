"""Sequential active-match collection and search."""

from __future__ import annotations

from typing import Any
import numpy as np
from simulator.ruleset import load_fixed_ruleset
from simulator.rl.v4.distillation import DistillationSample
from simulator.rl.foundation.opponent_pool import OpponentPool
from simulator.rl.v4.simulator_teacher import TeacherError
from .config import _SPELL_COL, _stable_seed, SearchDatasetConfig
from .inference import _predict_records
from .records import _base_label, _new_sim_env, _record_from_env, _spell_opp
from .targets import _search_live, _adopt_target, _to_search_sample
from .selection import _search_stats


def _sync_public_memory(env: Any, estimator: Any) -> Any:
    """Synchronize public event memory for active-match records.

    ``sync_from_env`` is the one authoritative adapter.  Keeping this call
    strict matters: a fallback event translation cannot reproduce public card
    identity, cycle state, or the estimator's elixir bookkeeping faithfully.
    """

    sync = getattr(estimator, "sync_from_env", None)
    if not callable(sync):
        raise TeacherError("active records require PublicStateEstimator.sync_from_env")
    snapshot = sync(env, viewer=0)
    if snapshot is None:
        raise TeacherError("PublicStateEstimator.sync_from_env returned no snapshot")
    return snapshot


def _active_record_memory(record: dict[str, Any], snapshot: Any, *, match_id: str, offset: int) -> None:
    """Attach estimator memory and sequence provenance to one record."""

    record["event_history"] = np.asarray(snapshot.event_history, dtype=np.float32)
    record["opp_hand_probs"] = np.asarray(snapshot.opp_hand_probs, dtype=np.float32)
    record["opp_out_of_cycle"] = np.asarray(snapshot.opp_out_of_cycle, dtype=bool)
    record["opp_elixir_interval"] = np.asarray(
        [snapshot.opp_elixir_lo, snapshot.opp_elixir_hi], dtype=np.float32
    )
    record["seq_id"] = str(match_id)
    record["offset"] = int(offset)
    record["sequence_offset"] = int(offset)
    prov = record["provenance"]
    tick_seconds = float(prov.get("tick_seconds", 0.0))
    hand = list(record.get("hand_keys", ()))
    spell_in_hand = any(
        bool(record["hand_tokens"][slot, _SPELL_COL] > 0.5)
        for slot in range(min(len(hand), record["hand_tokens"].shape[0]))
    )
    phase = "opening" if tick_seconds < 60.0 else ("mid-match" if tick_seconds < 180.0 else "spell-endgame")
    tags = []
    if spell_in_hand:
        tags.append("card-preservation")
    if int(np.asarray(snapshot.opp_out_of_cycle, dtype=bool).sum()) > 0:
        tags.append("outcycling")
    if phase == "spell-endgame":
        tags.append("spell-endgame")
    prov.update(
        {
            "seq_id": str(match_id),
            "offset": int(offset),
            "sequence_offset": int(offset),
            "active_match": True,
            "active_match_id": str(match_id),
            "sequence_offset": int(offset),
            "active_phase": phase,
            "strategic_tags": tags,
            "out_of_cycle_count": int(np.asarray(snapshot.opp_out_of_cycle, dtype=bool).sum()),
            "opp_elixir_interval": [
                round(float(snapshot.opp_elixir_lo), 4),
                round(float(snapshot.opp_elixir_hi), 4),
            ],
        }
    )


def _generate_active_records(config: SearchDatasetConfig) -> list[dict[str, Any]]:
    """Generate sequential public records from active two-sided matches.

    The learner side follows the public rule teacher only to make a stable
    trajectory.  Each captured state keeps an isolated simulator fork for
    outcome search; the training target still comes from the public
    observation and the counterfactual evaluator.  This is a controller
    baseline, not a claim of expert play.
    """

    try:
        from ....actions import WaitAction
        from ....public_state_estimator import PublicStateEstimator
        from ...foundation.opponent_pool import make_opponent_controller
        from ..search_teacher import _public_rule_action
    except ImportError:  # pragma: no cover - top-level ``rl`` imports
        from simulator.actions import WaitAction
        from simulator.public_state_estimator import PublicStateEstimator
        from simulator.rl.foundation.opponent_pool import make_opponent_controller
        from simulator.rl.v4.search_teacher import _public_rule_action
    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=config.seed)
    family_mix = dict(config.sim_family_mix) if config.sim_family_mix else None
    target_records = max(int(config.n_states), 1)
    match_count = max(
        1,
        (target_records + int(config.active_match_decisions) - 1)
        // int(config.active_match_decisions),
    )
    records: list[dict[str, Any]] = []
    for match_index in range(match_count):
        env, family, source = _new_sim_env(
            ruleset, pool, config.seed, match_index, config.seed, family_mix
        )
        match_id = f"active:{config.seed}:{match_index}:{family}"
        estimator = PublicStateEstimator()
        estimator.reset()
        controller = make_opponent_controller(
            config.opponent_strategy,
            seed=_stable_seed(config.seed, "active-opponent", match_index),
        )
        try:
            for offset in range(int(config.active_match_decisions)):
                if env.state is None or env.state.terminal or len(records) >= target_records:
                    break
                snapshot = _sync_public_memory(env, estimator)
                record = _record_from_env(
                    env,
                    family,
                    source,
                    ruleset,
                    allow_partial_hand=True,
                )
                if record is None:
                    break
                record["sim_index"] = match_index
                record["active_state_fork"] = env.fork()
                _active_record_memory(record, snapshot, match_id=match_id, offset=offset)
                record["provenance"]["opponent_strategy"] = config.opponent_strategy
                records.append(record)
                mine = _public_rule_action(env, 0)
                opponent = controller.choose_action(env.engine, env.state, 1)
                if mine is None:
                    mine = WaitAction(0)
                env.step_v2((mine, opponent))
                # Sync after the decision so the next record contains public
                # deployment/death/damage history from this transition.
                _sync_public_memory(env, estimator)
                if env.state is None or env.state.terminal:
                    break
        finally:
            del env
        if len(records) >= target_records:
            break
    return records[:target_records]


def _generate_active_search_dataset(
    config: SearchDatasetConfig,
    champion: Any,
    *,
    progress: Any = None,
) -> tuple[list[DistillationSample], dict[str, Any]]:
    """Build the opt-in active sequential Stage-2 dataset in-process."""

    records = _generate_active_records(config)
    if not records:
        raise TeacherError("active match generation produced no usable records")
    preds = _predict_records(champion, records)
    candidates: list[dict[str, Any]] = []
    for index, (record, pred) in enumerate(zip(records, preds)):
        record["champ"] = pred
        record["base_target"] = _base_label(record)
        env = record.pop("active_state_fork")
        try:
            search = _search_live(
                env,
                record,
                pred,
                horizon=config.effective_horizon,
                max_branches=config.max_branches,
                seed=_stable_seed(config.seed, "active-search", index),
                continuation_mode=config.continuation_mode,
                opponent_strategy=config.opponent_strategy,
            )
        finally:
            del env
        target, adopted, regret, placement_gap = _adopt_target(
            record, search["result"], record["base_target"]
        )
        candidates.append(
            {
                "record": record,
                "target": target,
                "adopted": adopted,
                "regret_rule": regret,
                "placement_gap": placement_gap,
                "search": search,
                "base": record["base_target"],
                "sim": True,
                "spell": _spell_opp(record),
                "transition": False,
            }
        )
        if progress is not None:
            progress(index + 1, len(records))
    # Active records are already in sequential match order.  Keep the same
    # deterministic hash order used by legacy selection only for the optional
    # quota swaps, so paired state order remains reproducible.
    n_total = min(config.n_states, len(candidates))
    selected = candidates[:n_total]
    stats = _search_stats(selected)
    stats.update(
        {
            "n_searched": len(candidates),
            "active": True,
            "continuation_mode": config.continuation_mode,
            "opponent_strategy": config.opponent_strategy,
            "horizon": int(config.effective_horizon),
            "max_branches": int(config.max_branches),
            "active_match_decisions": int(config.active_match_decisions),
            "n_active_matches": len({c["record"]["seq_id"] for c in selected}),
            "sequential": True,
        }
    )
    samples = [
        _to_search_sample(
            c["record"],
            c["target"],
            c["search"],
            c["base"],
            c["adopted"],
            c["regret_rule"],
            c["placement_gap"],
            index,
            config.placement_target_mode,
        )
        for index, c in enumerate(selected)
    ]
    return samples, stats

