"""Stage-2 search dataset: outcome-labeled real simulator states.

Pipeline (all deterministic per seed):

* Pass 1 records a timing pool plus self-built sim-natural resets, with
  batched champion inference joined afterwards (no per-sample inference).
* Pass 2 replays selected timing sequences (bit-identical states, verified
  by state hash) and rebuilds selected sim states, running counterfactual
  search on the live environments with the champion's top cell included.
* Adoption: the search-best becomes the training target only on a material
  outcome gap (action margin 0.15, same-card placement margin 0.05), and
  never overturns a timing-hold (timing WAIT where the base rule plays:
  a short horizon cannot price building lifetime or cycle consequences,
  so those stay rule-labeled and contribute regret statistics only).
* Oversampled stratified selection emits the final joint-ready samples.

Near-ties keep rule labels by design: a model that disagrees with search
within noise is not wrong, and validated pressure behavior is preserved.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import numpy as np

try:
    from ..observation_v3 import (
        BELIEF_CARD_COUNT,
        EVENT_DIM,
        EVENT_HISTORY_LEN,
        HAND_TOKEN_FEATURES,
    )
    from ..roster import PLAYER_DECK as ROSTER_PLAYER_DECK
    from ..ruleset import load_fixed_ruleset
    from .distillation import DistillationConfig, DistillationSample, family_for_index
    from .opponent_pool import OpponentPool
    from .simulator_distillation import (
        FAMILY_TO_SOURCE,
        SOURCE_ARCHETYPE,
        build_hand_tokens,
    )
    from .simulator_teacher import (
        TeacherError,
        TeacherTarget,
        soft_placement_target,
        teacher_label,
        wait_duration_for_elixir,
    )
    from .timing_curriculum import (
        TimingConfig,
        generate_timing_pool,
        replay_timing_sequences,
    )
    from .timing_teacher import TIMING_TEACHER_VERSION, timing_teacher_label
    from .search_teacher import (
        HORIZON_DECISIONS,
        MAX_BRANCHES,
        SEARCH_TEACHER_VERSION,
        BranchAction,
        SearchResult,
        score_branch,
        search_state,
    )
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.observation_v3 import (
        BELIEF_CARD_COUNT,
        EVENT_DIM,
        EVENT_HISTORY_LEN,
        HAND_TOKEN_FEATURES,
    )
    from simulator.roster import PLAYER_DECK as ROSTER_PLAYER_DECK
    from simulator.ruleset import load_fixed_ruleset
    from simulator.rl.distillation import DistillationConfig, DistillationSample, family_for_index
    from simulator.rl.opponent_pool import OpponentPool
    from simulator.rl.simulator_distillation import (
        FAMILY_TO_SOURCE,
        SOURCE_ARCHETYPE,
        build_hand_tokens,
    )
    from simulator.rl.simulator_teacher import (
        TeacherError,
        TeacherTarget,
        soft_placement_target,
        teacher_label,
        wait_duration_for_elixir,
    )
    from simulator.rl.timing_curriculum import (
        TimingConfig,
        generate_timing_pool,
        replay_timing_sequences,
    )
    from simulator.rl.timing_teacher import TIMING_TEACHER_VERSION, timing_teacher_label
    from simulator.rl.search_teacher import (
        HORIZON_DECISIONS,
        MAX_BRANCHES,
        SEARCH_TEACHER_VERSION,
        BranchAction,
        SearchResult,
        score_branch,
        search_state,
    )


SEARCH_GENERATOR_VERSION: str = "search-v4-0"

ADOPT_MARGIN_ACTION: float = 0.15
"""Minimum regret(rule) to adopt a different mode/card from search."""

ADOPT_MARGIN_PLACE: float = 0.05
"""Minimum regret(rule) to adopt a different cell for the same card."""

OPPONENT_STRATEGY: str = "deterministic-cycle"

_SPELL_COL = HAND_TOKEN_FEATURES.index("is_spell")


def load_champion_actor(path: str, config=None):
    """Load a frozen champion checkpoint into a fresh V4 actor.

    The placement head materializes its cell-bias parameter lazily on
    first forward, so a dummy forward runs before the strict load.
    """

    import torch

    try:
        from .model_v4 import ModelConfigV4, RecurrentV4Policy
    except ImportError:  # pragma: no cover
        from simulator.rl.model_v4 import ModelConfigV4, RecurrentV4Policy
    policy = RecurrentV4Policy(config or ModelConfigV4())
    policy.eval()
    with torch.no_grad():
        zeros = torch.zeros(1, 1, 21, 32, 18)
        globals_768 = torch.zeros(1, 1, 768)
        entities = torch.zeros(1, 1, 128, 32)
        entity_mask = torch.zeros(1, 1, 128, dtype=torch.bool)
        hand = torch.zeros(1, 1, 4, 16)
        probs = torch.full((1, 1, 128), 1.0 / 128)
        out = torch.zeros(1, 1, 128, dtype=torch.bool)
        interval = torch.zeros(1, 1, 2)
        events = torch.zeros(1, 1, 16, 8)
        reset = torch.ones(1, 1, dtype=torch.bool)
        policy(
            zeros, globals_768, entities, entity_mask, hand, probs, out,
            interval, events, reset,
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class SearchDatasetConfig:
    """Sizing and quota contract for the Stage-2 search dataset."""

    n_states: int = 2000
    seed: int = 0
    timing_pool_states: int = 1200
    sim_candidates: int = 1200
    horizon: int = HORIZON_DECISIONS
    max_branches: int = MAX_BRANCHES
    replay_sequences: int = 260
    min_adopted_share: float = 0.30
    min_spell_share: float = 0.15
    min_transition_share: float = 0.15
    min_sim_share: float = 0.25
    family_min_share: float = 0.08

    def __post_init__(self) -> None:
        if type(self.n_states) is not int or self.n_states <= 0:
            raise ValueError("n_states must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


def _hash_order(seed: int, key: str) -> int:
    return _stable_seed(seed, "search-order", key)


def _hand_keys(record: dict[str, Any]) -> list[str]:
    if "hand_keys" in record:
        return list(record["hand_keys"])
    return list(record["provenance"]["target_hand"])


def _enemy_count(record: dict[str, Any]) -> int:
    mask = np.asarray(record["entity_mask"])
    tokens = np.asarray(record["entity_tokens"])
    return int(((tokens[:, 1] > 0.5) & mask).sum())


def _spell_legal(record: dict[str, Any]) -> bool:
    hand = np.asarray(record["hand_tokens"])
    legal = np.asarray(record["legal_play"]).reshape(4, -1).any(axis=1)
    return bool(((hand[:, _SPELL_COL] > 0.5) & legal).any())


def _base_label(record: dict[str, Any]) -> TeacherTarget:
    return teacher_label(
        hand_tokens=np.asarray(record["hand_tokens"]),
        entity_tokens=np.asarray(record["entity_tokens"]),
        entity_mask=np.asarray(record["entity_mask"]),
        legal_play=np.asarray(record["legal_play"]),
        legal_wait=bool(record["legal_wait"]),
        own_elixir=float(record["own_elixir"]),
    )


def _record_to_inference_sample(record: dict[str, Any]) -> DistillationSample:
    return DistillationSample(
        family=record["family"],
        raster=np.ascontiguousarray(record["raster"], dtype=np.float32),
        global_features=np.ascontiguousarray(record["global_features"], dtype=np.float32),
        entity_tokens=np.ascontiguousarray(record["entity_tokens"], dtype=np.float32),
        entity_mask=np.ascontiguousarray(record["entity_mask"], dtype=bool),
        hand_tokens=np.ascontiguousarray(record["hand_tokens"], dtype=np.float32),
        opp_hand_probs=np.full((BELIEF_CARD_COUNT,), 1.0 / BELIEF_CARD_COUNT, dtype=np.float32),
        opp_out_of_cycle=np.zeros((BELIEF_CARD_COUNT,), dtype=bool),
        opp_elixir_interval=np.asarray([0.0, 10.0], dtype=np.float32),
        event_history=np.zeros((EVENT_HISTORY_LEN, EVENT_DIM), dtype=np.float32),
        legal_play=np.ascontiguousarray(record["legal_play"], dtype=bool),
        legal_wait=bool(record["legal_wait"]),
        own_elixir=float(record["own_elixir"]),
        target=TeacherTarget(mode=0, card_slot=0, row=0, col=0, wait_duration_idx=0),
        soft_placement=np.zeros((32, 18), dtype=np.float32),
        provenance={"generator": SEARCH_GENERATOR_VERSION, "index": -1},
    )


def _predict_records(
    policy: Any, records: list[dict[str, Any]], device: Any = None
) -> list[dict[str, int]]:
    """Batched champion inference: one (mode, card, row, col) per record.

    ``device`` defaults to whatever device the policy lives on, keeping
    model and batch tensors consistent without caller plumbing.
    """

    import torch

    try:
        from .distillation import move_batch_to_device, to_torch_batch
    except ImportError:  # pragma: no cover
        from simulator.rl.distillation import move_batch_to_device, to_torch_batch
    if device is None:
        try:
            device = next(policy.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    preds: list[dict[str, int]] = []
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(records), 256):
            chunk = [_record_to_inference_sample(r) for r in records[start : start + 256]]
            batch = move_batch_to_device(to_torch_batch(chunk), device)
            logits, _, _ = policy(
                batch["raster"],
                batch["global_features"],
                batch["entities"],
                batch["entity_mask"],
                batch["hand_tokens"],
                batch["opp_hand_probs"],
                batch["opp_out_of_cycle"],
                batch["opp_elixir_interval"],
                batch["event_history"],
                batch["reset_mask"],
            )
            _, cols = logits.placement.shape[-2:]
            for i in range(len(chunk)):
                mode = int(logits.mode[i, 0].argmax())
                card = int(logits.card[i, 0].argmax())
                flat = logits.placement[i, 0, card].reshape(-1)
                mask = batch["masks"].placement[i, 0, card].reshape(-1)
                safe = torch.where(mask, flat, torch.full_like(flat, float("-inf")))
                cell = int(safe.argmax())
                preds.append(
                    {"mode": mode, "card": card, "row": cell // cols, "col": cell % cols}
                )
    policy.train()
    return preds


def _new_sim_env(context_ruleset: Any, pool: OpponentPool, seed: int, index: int, distill_seed: int):
    """Deterministic sim-natural reset (replayable by index)."""

    try:
        from ..engine import BattleEngine
        from ..env import SimulatorEnv
    except ImportError:  # pragma: no cover
        from simulator.engine import BattleEngine
        from simulator.env import SimulatorEnv
    distill_config = DistillationConfig(n_states=max(1, index + 1), seed=distill_seed)
    family = family_for_index(distill_config, index)
    source = FAMILY_TO_SOURCE[family]
    opponent = pool.sample(index, archetype=SOURCE_ARCHETYPE[source], strategy=OPPONENT_STRATEGY)
    env = SimulatorEnv(
        engine=BattleEngine(context_ruleset, validate_every_tick=False),
        decision_interval_us=250_000,
    )
    env.reset_v2(
        seed=_stable_seed(seed, "search-sim", family, index),
        decks=(tuple(ROSTER_PLAYER_DECK), tuple(opponent.deck.cards)),
        shuffle_decks=True,
    )
    return env, family, source


def _record_from_env(env: Any, family: str, source: str, context_ruleset: Any) -> dict[str, Any] | None:
    """Observe + label a live reset state (shared by sim build and replay)."""

    state = env.state
    if state is None or state.terminal:
        return None
    try:
        observation = env.observe_v2_for_viewer(0)
    except (ValueError, TypeError):
        return None
    player_state = state.players[0]
    hand = list(player_state.hand)
    if len(hand) != 4:
        return None
    elixir = float(player_state.elixir_milli) / 1000.0
    try:
        hand_tokens = build_hand_tokens(hand, elixir)
    except TeacherError:
        return None
    legal_play = np.array(observation.legal_play, dtype=bool)
    tick_seconds = float(state.tick) * float(context_ruleset.tick_us) / 1_000_000.0
    try:
        timing = timing_teacher_label(
            hand_tokens=hand_tokens,
            entity_tokens=np.asarray(observation.entity_tokens),
            entity_mask=np.asarray(observation.entity_mask),
            legal_play=legal_play,
            legal_wait=bool(observation.legal_wait),
            own_elixir=elixir,
            tick_seconds=tick_seconds,
        )
    except TeacherError:
        return None
    legal_cards = int(legal_play.reshape(4, -1).any(axis=1).sum())
    return {
        "family": family,
        "raster": np.array(observation.board, dtype=np.float32),
        "global_features": np.array(observation.global_vector, dtype=np.float32),
        "entity_tokens": np.array(observation.entity_tokens, dtype=np.float32),
        "entity_mask": np.array(observation.entity_mask, dtype=bool),
        "hand_tokens": hand_tokens,
        "legal_play": legal_play,
        "legal_wait": bool(observation.legal_wait),
        "own_elixir": float(elixir),
        "target": timing,
        "mode": int(timing.mode),
        "hand_keys": hand,
        "seq_id": "",
        "offset": 0,
        "legal_cards": legal_cards,
        "play_legal": bool(legal_cards > 0),
        "provenance": {
            "generator": SEARCH_GENERATOR_VERSION,
            "timing_family": family,
            "state_hash": state.state_hash(),
            "source": source,
            "target_elixir_milli": int(player_state.elixir_milli),
            "target_hand": list(hand),
            "legal_cards": legal_cards,
            "tick_seconds": round(tick_seconds, 3),
        },
    }


def _search_live(
    env: Any,
    record: dict[str, Any],
    actor_pred: dict[str, int] | None,
    *,
    horizon: int,
    max_branches: int,
    seed: int,
    actor_cells: Sequence[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """Run search on a live env and score the actor's action in-set.

    ``actor_cells`` optionally overrides the candidate cells outright (used
    for multi-model union cells in regret evaluation); when omitted they
    are derived from ``actor_pred`` exactly as before.
    """

    legal = np.asarray(record["legal_play"], dtype=bool)
    rule = record["target"]
    if actor_cells is None:
        actor_cells = []
        if actor_pred is not None and actor_pred["mode"] == 1:
            slot = int(actor_pred["card"])
            if 0 <= slot < legal.shape[0] and bool(legal[slot].any()):
                actor_cells = [(slot, int(actor_pred["row"]), int(actor_pred["col"]))]
    result = search_state(
        env,
        legal_play=legal,
        rule_action=rule if rule.mode == 1 else None,
        actor_cells=actor_cells,
        hand_keys=_hand_keys(record),
        horizon=horizon,
        max_branches=max_branches,
        seed=seed,
        source=str(record["provenance"].get("source", "")),
        family=str(record["family"]),
    )
    if result.state_hash != record["provenance"]["state_hash"]:
        raise TeacherError("replay state hash mismatch; generation is not deterministic")
    if not bool(record["legal_wait"]):
        kept = [b for b in result.branches if b.action.kind == "play"]
        if not kept:
            raise TeacherError("search found no legal branch")
        result.branches = kept
        order = sorted(range(len(kept)), key=lambda i: (-kept[i].score, i))
        result.best_index = int(order[0])
        result.second_gap = float(kept[order[0]].score - kept[order[1]].score) if len(order) > 1 else 0.0
    actor_score: float | None = None
    if actor_pred is not None:
        if actor_pred["mode"] == 0:
            wait_branch = next((b for b in result.branches if b.action.kind == "wait"), None)
            actor_score = float(wait_branch.score) if wait_branch is not None else None
        else:
            match = next(
                (
                    b
                    for b in result.branches
                    if b.action.kind == "play"
                    and b.action.slot == actor_pred["card"]
                    and b.action.row == actor_pred["row"]
                    and b.action.col == actor_pred["col"]
                ),
                None,
            )
            if match is None:
                extra = score_branch(
                    env,
                    BranchAction(
                        kind="play",
                        slot=int(actor_pred["card"]),
                        row=int(actor_pred["row"]),
                        col=int(actor_pred["col"]),
                    ),
                    horizon=horizon,
                )
                actor_score = float(extra.score)
            else:
                actor_score = float(match.score)
    return {"result": result, "actor_score": actor_score}


def _adopt_target(
    record: dict[str, Any],
    result: SearchResult,
    base_target: TeacherTarget,
) -> tuple[TeacherTarget, bool, float, float]:
    """Decide the training target: search-best on material gap, else rule.

    Returns ``(target, adopted, regret_rule, placement_gap)``.  Timing
    holds (timing WAIT where the base rule plays) are never overturned:
    the horizon cannot price building lifetime or cycle consequences.
    """

    timing: TeacherTarget = record["target"]
    legal = np.asarray(record["legal_play"], dtype=bool)
    rule_score: float | None = None
    if timing.mode == 1:
        match = next(
            (
                b
                for b in result.branches
                if b.action.kind == "play"
                and b.action.slot == timing.card_slot
                and b.action.row == timing.row
                and b.action.col == timing.col
            ),
            None,
        )
        rule_score = float(match.score) if match is not None else None
    else:
        wait_branch = next((b for b in result.branches if b.action.kind == "wait"), None)
        rule_score = float(wait_branch.score) if wait_branch is not None else None
    best = result.best
    regret_rule = float(best.score - (rule_score if rule_score is not None else best.score))
    timing_hold = timing.mode == 0 and base_target.mode == 1
    if timing_hold:
        return timing, False, regret_rule, 0.0
    if best.action.kind == "wait":
        if timing.mode == 0 or regret_rule < ADOPT_MARGIN_ACTION:
            return timing, False, regret_rule, 0.0
        return (
            TeacherTarget(
                mode=0,
                card_slot=0,
                row=0,
                col=0,
                wait_duration_idx=wait_duration_for_elixir(float(record["own_elixir"])),
            ),
            True,
            regret_rule,
            0.0,
        )
    hand_keys = _hand_keys(record)
    try:
        slot = hand_keys.index(best.action.card_key)
    except ValueError:
        return timing, False, regret_rule, 0.0
    if not bool(legal[slot, best.action.row, best.action.col]):
        return timing, False, regret_rule, 0.0
    same_card = timing.mode == 1 and timing.card_slot == slot
    margin = ADOPT_MARGIN_PLACE if same_card else ADOPT_MARGIN_ACTION
    if regret_rule < margin:
        return timing, False, regret_rule, 0.0
    placement_gap = 0.0
    if same_card:
        placement_gap = float(best.score - (rule_score if rule_score is not None else best.score))
    return (
        TeacherTarget(
            mode=1,
            card_slot=int(slot),
            row=int(best.action.row),
            col=int(best.action.col),
            wait_duration_idx=0,
        ),
        True,
        regret_rule,
        placement_gap,
    )


def _to_search_sample(
    record: dict[str, Any],
    target: TeacherTarget,
    search: dict[str, Any],
    base_target: TeacherTarget,
    adopted: bool,
    regret_rule: float,
    placement_gap: float,
    index: int,
) -> DistillationSample:
    result: SearchResult = search["result"]
    legal = np.asarray(record["legal_play"], dtype=bool)
    if target.mode == 1:
        soft = soft_placement_target(target.row, target.col, np.asarray(legal[target.card_slot]))
    else:
        soft = np.zeros((32, 18), dtype=np.float32)
    branches = [
        {
            "kind": b.action.kind,
            "slot": int(b.action.slot),
            "row": int(b.action.row),
            "col": int(b.action.col),
            "card": b.action.card_key,
            "score": round(float(b.score), 4),
            "terms": {k: round(float(v), 4) for k, v in b.terms.items()},
        }
        for b in result.branches
    ]
    best = result.best
    provenance = dict(record["provenance"])
    provenance.update(
        {
            "generator": SEARCH_GENERATOR_VERSION,
            "teacher": SEARCH_TEACHER_VERSION if adopted else TIMING_TEACHER_VERSION,
            "index": index,
            "adopted_search": bool(adopted),
            "rule_mode": int(record["target"].mode),
            "rule_card_slot": int(record["target"].card_slot),
            "base_mode": int(base_target.mode),
            "base_card_slot": int(base_target.card_slot),
            "timing_hold": bool(record["target"].mode == 0 and base_target.mode == 1),
            "search_best_kind": best.action.kind,
            "search_best_slot": int(best.action.slot),
            "search_best_row": int(best.action.row),
            "search_best_col": int(best.action.col),
            "search_best_card": best.action.card_key,
            "search_best_score": round(float(best.score), 4),
            "search_second_gap": round(float(result.second_gap), 4),
            "search_branch_count": int(result.branch_count),
            "search_horizon": int(result.horizon),
            "search_branches": branches,
            "regret_rule": round(float(regret_rule), 4),
            "regret_actor": (
                round(float(result.best.score - search["actor_score"]), 4)
                if search["actor_score"] is not None
                else None
            ),
            "placement_gap": round(float(placement_gap), 4),
        }
    )
    return DistillationSample(
        family=record["family"],
        raster=np.ascontiguousarray(record["raster"], dtype=np.float32),
        global_features=np.ascontiguousarray(record["global_features"], dtype=np.float32),
        entity_tokens=np.ascontiguousarray(record["entity_tokens"], dtype=np.float32),
        entity_mask=np.ascontiguousarray(record["entity_mask"], dtype=bool),
        hand_tokens=np.ascontiguousarray(record["hand_tokens"], dtype=np.float32),
        opp_hand_probs=np.full((BELIEF_CARD_COUNT,), 1.0 / BELIEF_CARD_COUNT, dtype=np.float32),
        opp_out_of_cycle=np.zeros((BELIEF_CARD_COUNT,), dtype=bool),
        opp_elixir_interval=np.asarray([0.0, 10.0], dtype=np.float32),
        event_history=np.zeros((EVENT_HISTORY_LEN, EVENT_DIM), dtype=np.float32),
        legal_play=np.ascontiguousarray(record["legal_play"], dtype=bool),
        legal_wait=bool(record["legal_wait"]),
        own_elixir=float(record["own_elixir"]),
        target=target,
        soft_placement=soft,
        provenance=provenance,
    )


def _disagrees(record: dict[str, Any]) -> bool:
    pred, target = record["champ"], record["target"]
    if pred["mode"] != target.mode:
        return True
    return bool(pred["mode"] == 1 and pred["card"] != target.card_slot)


def _spell_opp(record: dict[str, Any]) -> bool:
    return _spell_legal(record) and _enemy_count(record) >= 1


def _ambiguous(record: dict[str, Any]) -> bool:
    return record["target"].mode == 0 and bool(record["play_legal"])


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

    # Pass 1a: timing pool records (no searching yet).
    timing_config = TimingConfig(n_states=config.timing_pool_states, seed=config.seed)
    timing_pool = generate_timing_pool(timing_config)
    # Pass 1b: sim-natural records (envs rebuilt for search in pass 2).
    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=config.seed)
    sim_records: list[dict[str, Any]] = []
    sim_indices: list[int] = []
    for index in range(config.sim_candidates):
        env, family, source = _new_sim_env(ruleset, pool, config.seed, index, config.seed)
        try:
            record = _record_from_env(env, family, source, ruleset)
        finally:
            del env
        if record is not None:
            record["sim_index"] = index
            sim_records.append(record)
            sim_indices.append(index)
    # Batched champion inference joined to every record.
    timing_preds = _predict_records(champion, timing_pool)
    sim_preds = _predict_records(champion, sim_records)
    for record, pred in zip(timing_pool, timing_preds):
        record["champ"] = pred
        record["base_target"] = _base_label(record)
    for record, pred in zip(sim_records, sim_preds):
        record["champ"] = pred
        record["base_target"] = _base_label(record)

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

    searched: list[dict[str, Any]] = []

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
                horizon=config.horizon,
                max_branches=config.max_branches,
                seed=config.seed,
            )
        )
    # Sim states: select indices, then rebuild + search while envs are live.
    caps = {"spell": 300, "ambiguous": 500, "disagree": 400, "fill": 600}
    counts = dict.fromkeys(caps, 0)
    ordered_sim = sorted(
        sim_records, key=lambda r: _hash_order(config.seed, str(r["provenance"]["state_hash"]))
    )
    sim_to_search: list[dict[str, Any]] = []
    for record in ordered_sim:
        classes = []
        if _spell_opp(record):
            classes.append("spell")
        if _ambiguous(record):
            classes.append("ambiguous")
        if _disagrees(record):
            classes.append("disagree")
        classes.append("fill")
        for choice in classes:
            if counts[choice] < caps[choice]:
                counts[choice] += 1
                sim_to_search.append(record)
                break
    sim_tasks = [
        SimSearchTask(
            index=int(record["sim_index"]),
            record=record,
            champ=record["champ"],
            horizon=config.horizon,
            max_branches=config.max_branches,
            seed=config.seed,
            pool_seed=config.seed,
            distill_seed=config.seed,
        )
        for record in sim_to_search
    ]
    timing_results, sim_results = run_search_tasks(
        timing_tasks, sim_tasks, workers=workers, progress=progress,
    )
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
    n_total = min(config.n_states, len(candidates))
    by_adopted = [c for c in candidates if c["adopted"]]
    need_adopted = min(len(by_adopted), int(round(n_total * config.min_adopted_share)))
    selected: list[dict[str, Any]] = sorted(
        by_adopted, key=lambda c: _hash_order(config.seed, c["record"]["provenance"]["state_hash"])
    )[:need_adopted]
    taken = {id(c) for c in selected}
    rest = sorted(
        [c for c in candidates if id(c) not in taken],
        key=lambda c: _hash_order(config.seed, c["record"]["provenance"]["state_hash"]),
    )
    for entry in rest:
        if len(selected) >= n_total:
            break
        selected.append(entry)
    quotas = [
        (config.min_spell_share, lambda c: c["spell"]),
        (config.min_transition_share, lambda c: c["transition"]),
        (config.min_sim_share, lambda c: c["sim"]),
    ]
    pool_left = [c for c in rest if id(c) not in {id(s) for s in selected}]

    def share(pred) -> float:
        return sum(1 for c in selected if pred(c)) / max(1, len(selected))

    for minimum, pred in quotas:
        while share(pred) < minimum and pool_left:
            found = next((c for c in pool_left if pred(c)), None)
            if found is None:
                break
            pool_left.remove(found)
            if len(selected) < n_total:
                selected.append(found)
            else:
                victim = next((s for s in selected if not pred(s)), None)
                if victim is None:
                    break
                selected.remove(victim)
                pool_left.append(victim)
                selected.append(found)
    stats = _search_stats(selected)
    _assert_search_balance(stats, config, n_total)
    samples = [
        _to_search_sample(
            c["record"], c["target"], c["search"], c["base"],
            c["adopted"], c["regret_rule"], c["placement_gap"], index,
        )
        for index, c in enumerate(selected)
    ]
    return samples, stats


def _search_stats(selected: list[dict[str, Any]]) -> dict[str, Any]:
    families = Counter(
        c["record"]["family"] if not c["sim"] else f"sim:{c['record']['family']}"
        for c in selected
    )
    waits = [c for c in selected if c["target"].mode == 0]
    regrets = [c["regret_rule"] for c in selected]
    return {
        "n": len(selected),
        "n_wait": len(waits),
        "wait_share": len(waits) / max(1, len(selected)),
        "adopted_share": sum(1 for c in selected if c["adopted"]) / max(1, len(selected)),
        "spell_share": sum(1 for c in selected if c["spell"]) / max(1, len(selected)),
        "transition_share": sum(1 for c in selected if c["transition"]) / max(1, len(selected)),
        "sim_share": sum(1 for c in selected if c["sim"]) / max(1, len(selected)),
        "family_n": dict(families),
        "mean_regret_rule": float(np.mean(regrets)) if regrets else 0.0,
    }


def _assert_search_balance(stats: dict[str, Any], config: SearchDatasetConfig, n_total: int) -> None:
    problems: list[str] = []
    if stats["n"] != n_total:
        problems.append(f"selected {stats['n']} != {n_total} requested")
    if stats["adopted_share"] < config.min_adopted_share - 1e-9:
        problems.append(f"adopted {stats['adopted_share']:.3f} below minimum")
    if stats["spell_share"] < config.min_spell_share - 1e-9:
        problems.append(f"spell {stats['spell_share']:.3f} below minimum")
    if stats["transition_share"] < config.min_transition_share - 1e-9:
        problems.append(f"transition {stats['transition_share']:.3f} below minimum")
    if stats["sim_share"] < config.min_sim_share - 1e-9:
        problems.append(f"sim {stats['sim_share']:.3f} below minimum")
    if problems:
        raise TeacherError("search balance unsatisfiable: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# Parallel search execution.
#
# Counterfactual search over independent simulator states is embarrassingly
# parallel: every state rebuilds deterministically from seeds, branches are
# scored in isolation, and results join back in canonical order.  Workers
# run simulator physics only (CPU); torch inference stays in the parent.
# The spawn start method is used deliberately: the parent process has
# usually run torch inference (batched champion predictions) before the
# pool starts, and forking a process with active BLAS/OpenMP thread pools
# risks deadlocks.  ``workers=1`` executes the same task list inline and
# is the bit-identical reference path.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimingSearchTask:
    """Replay one timing sequence, searching the requested offsets.

    All fields are picklable; the worker rebuilds an isolated simulator
    instance from ``timing_config`` + ``(family, seq_idx)`` and runs the
    exact same ``_search_live`` call the serial path makes.  ``champs``
    parallels ``offsets`` (``None`` entries allowed); ``cells_list``
    optionally overrides candidate cells per offset (regret union cells).
    """

    family: str
    seq_idx: int
    timing_config: Any
    offsets: tuple[int, ...]
    records: tuple[dict[str, Any], ...]
    champs: tuple[Any, ...]
    horizon: int
    max_branches: int
    seed: int
    cells_list: tuple[Any, ...] = ()


@dataclass(frozen=True)
class SimSearchTask:
    """Rebuild one sim-natural reset and search it."""

    index: int
    record: dict[str, Any]
    champ: Any
    horizon: int
    max_branches: int
    seed: int
    pool_seed: int
    distill_seed: int
    actor_cells: Any = ()


def _worker_init() -> None:
    """Keep per-process thread pools at one thread per worker.

    Simulator stepping is single-threaded Python plus small-array numpy;
    without this, every worker would inherit (spawn) or share (fork) a
    full-size BLAS/torch thread pool and oversubscribe the machine.
    Environment variables are best-effort for libraries initialized after
    this runs; the torch calls take effect immediately.
    """

    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass


def _run_timing_task(task: TimingSearchTask) -> list[tuple[int, dict[str, Any]]]:
    """Replay one sequence in this process, searching requested offsets.

    Mirrors the serial ``replay_timing_sequences`` semantics exactly: an
    unknown family raises, while per-sequence build errors (including a
    failing hook capture) keep partial results and move on.
    """

    from .timing_curriculum import TIMING_FAMILIES

    if task.family not in TIMING_FAMILIES:
        raise TeacherError(f"unknown timing family for replay: {task.family!r}")
    wanted = {int(offset) for offset in task.offsets}
    cell_lookup = {
        int(offset): (tuple(c) if c is not None else None)
        for offset, c in zip(task.offsets, task.cells_list)
    } if task.cells_list else {}
    by_offset = {int(offset): (record, champ) for offset, record, champ in zip(task.offsets, task.records, task.champs)}
    found: list[tuple[int, dict[str, Any]]] = []

    def hook(env: Any, record: dict[str, Any], offset: int) -> None:
        if int(offset) not in wanted:
            return
        rec, champ = by_offset[int(offset)]
        search = _search_live(
            env, rec, champ,
            horizon=task.horizon, max_branches=task.max_branches, seed=task.seed,
            actor_cells=cell_lookup.get(int(offset)),
        )
        found.append((int(offset), search))

    try:
        replay_timing_sequences(
            task.timing_config, [(task.family, int(task.seq_idx))], live_hook=hook
        )
    except TeacherError:
        pass
    found.sort(key=lambda item: item[0])
    return found


def _run_sim_task(task: SimSearchTask) -> dict[str, Any]:
    """Rebuild one sim reset in this process and search it."""

    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=task.pool_seed)
    env, _, _ = _new_sim_env(ruleset, pool, task.seed, int(task.index), task.distill_seed)
    try:
        return _search_live(
            env, task.record, task.champ,
            horizon=task.horizon, max_branches=task.max_branches, seed=task.seed,
            actor_cells=tuple(task.actor_cells) if task.actor_cells else None,
        )
    finally:
        del env


def run_search_tasks(
    timing_tasks: Sequence[TimingSearchTask],
    sim_tasks: Sequence[SimSearchTask],
    *,
    workers: int = 1,
    progress: Any = None,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[int, dict[str, Any]]]:
    """Execute search tasks, joining results in canonical task order.

    Returns ``(timing_results, sim_results)`` where timing keys are
    ``(family, seq_idx, offset)`` and sim keys are reset indices.  With
    ``workers=1`` tasks run inline in order (the reference path); with
    ``workers>1`` a spawn-context process pool distributes them while the
    parent reports ``progress(done, total)`` as results arrive.  Either way
    the joined dictionaries are identical for identical inputs.
    """

    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    timing_tasks = list(timing_tasks)
    sim_tasks = list(sim_tasks)
    total = len(timing_tasks) + len(sim_tasks)
    timing_out: dict[tuple[str, int, int], dict[str, Any]] = {}
    sim_out: dict[int, dict[str, Any]] = {}
    if total == 0:
        return timing_out, sim_out
    done = 0

    def note() -> None:
        if progress is not None:
            progress(done, total)

    if workers == 1:
        for task in timing_tasks:
            for offset, search in _run_timing_task(task):
                timing_out[(task.family, int(task.seq_idx), int(offset))] = search
            done += 1
            note()
        for task in sim_tasks:
            sim_out[int(task.index)] = _run_sim_task(task)
            done += 1
            note()
        return timing_out, sim_out

    import concurrent.futures
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_worker_init
    ) as pool_executor:
        timing_futures = [pool_executor.submit(_run_timing_task, task) for task in timing_tasks]
        sim_futures = [pool_executor.submit(_run_sim_task, task) for task in sim_tasks]
        for task, future in zip(timing_tasks, timing_futures):
            for offset, search in future.result():
                timing_out[(task.family, int(task.seq_idx), int(offset))] = search
            done += 1
            note()
        for task, future in zip(sim_tasks, sim_futures):
            sim_out[int(task.index)] = future.result()
            done += 1
            note()
    return timing_out, sim_out


__all__ = [
    "ADOPT_MARGIN_ACTION",
    "ADOPT_MARGIN_PLACE",
    "SEARCH_GENERATOR_VERSION",
    "SearchDatasetConfig",
    "SimSearchTask",
    "TimingSearchTask",
    "generate_search_dataset",
    "load_champion_actor",
    "run_search_tasks",
]
