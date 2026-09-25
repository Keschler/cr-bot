"""Live counterfactual search, target adoption and sample materialization."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
from simulator.observation_v3 import BELIEF_CARD_COUNT, EVENT_DIM, EVENT_HISTORY_LEN
from simulator.rl.v4.distillation import DistillationSample
from simulator.rl.v4.simulator_teacher import (
    TeacherError,
    TeacherTarget,
    soft_placement_target,
    wait_duration_for_elixir,
)
from simulator.rl.v4.timing_teacher import TIMING_TEACHER_VERSION
from simulator.rl.v4.search_teacher import (
    SEARCH_TEACHER_VERSION,
    BranchAction,
    SearchResult,
    outcome_tested_placement_target,
    score_branch,
    search_state,
)
from .config import SEARCH_GENERATOR_VERSION, ADOPT_MARGIN_ACTION, ADOPT_MARGIN_PLACE, _stable_seed
from .records import _hand_keys


def _search_live(
    env: Any,
    record: dict[str, Any],
    actor_pred: dict[str, int] | None,
    *,
    horizon: int,
    max_branches: int,
    seed: int,
    actor_cells: Sequence[tuple[int, int, int]] | None = None,
    continuation_mode: str = "wait",
    opponent_strategy: str = "deterministic-cycle",
    learner_controller: Any = None,
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
        continuation_mode=continuation_mode,
        opponent_strategy=opponent_strategy,
        learner_controller=learner_controller,
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
                # The champion's argmax cell is decoded without a card mask,
                # so on low-elixir states it can name an unaffordable slot
                # (all--inf placement row argmaxes to (0,0)).  Production
                # decoding masks illegal slots, so an illegal actor action
                # carries no outcome: record None (regret consumers skip it)
                # instead of crashing dataset generation on engine rejection.
                slot, row, col = (
                    int(actor_pred["card"]), int(actor_pred["row"]), int(actor_pred["col"])
                )
                rows, cols = legal.shape[1], legal.shape[2]
                if (
                    0 <= slot < legal.shape[0]
                    and 0 <= row < rows
                    and 0 <= col < cols
                    and bool(legal[slot, row, col])
                ):
                    extra = score_branch(
                        env,
                        BranchAction(
                            kind="play",
                            slot=slot,
                            row=row,
                            col=col,
                            card_key=(
                                _hand_keys(record)[slot]
                                if slot < len(_hand_keys(record))
                                else ""
                            ),
                        ),
                        continuation_mode=continuation_mode,
                        continuation_seed=int(
                            _stable_seed(seed, "continuation", result.state_hash)
                        ),
                        opponent_strategy=opponent_strategy,
                        horizon=horizon,
                    )
                    actor_score = float(extra.score)
                else:
                    actor_score = None
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
    placement_target_mode: str = "tested",
) -> DistillationSample:
    result: SearchResult = search["result"]
    legal = np.asarray(record["legal_play"], dtype=bool)
    if target.mode == 1:
        if placement_target_mode == "tested":
            soft = outcome_tested_placement_target(
                search["result"],
                slot=int(target.card_slot),
                legal=np.asarray(legal[target.card_slot]),
                fallback=(int(target.row), int(target.col)),
            )
        else:
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
    # Active records keep sequence metadata on the live record while timing
    # records already carry it in provenance.  Materialize both forms here so
    # torch checkpoints and downstream sequence batching retain the same
    # recurrent boundaries after serialization.
    seq_id = str(record.get("seq_id", provenance.get("seq_id", "")))
    offset = int(record.get("offset", provenance.get("offset", 0)))
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
            "placement_target_mode": str(placement_target_mode),
            "seq_id": seq_id,
            "offset": offset,
            "sequence_offset": int(record.get("sequence_offset", offset)),
            "tested_placement_cells": sorted(
                {
                    (int(b.action.row), int(b.action.col))
                    for b in result.branches
                    if b.action.kind == "play"
                    and int(b.action.slot) == int(target.card_slot)
                    and bool(legal[target.card_slot, b.action.row, b.action.col])
                }
            ),
        }
    )
    return DistillationSample(
        family=record["family"],
        raster=np.ascontiguousarray(record["raster"], dtype=np.float32),
        global_features=np.ascontiguousarray(record["global_features"], dtype=np.float32),
        entity_tokens=np.ascontiguousarray(record["entity_tokens"], dtype=np.float32),
        entity_mask=np.ascontiguousarray(record["entity_mask"], dtype=bool),
        hand_tokens=np.ascontiguousarray(record["hand_tokens"], dtype=np.float32),
        opp_hand_probs=np.asarray(
            record.get("opp_hand_probs", np.full((BELIEF_CARD_COUNT,), 1.0 / BELIEF_CARD_COUNT)),
            dtype=np.float32,
        ),
        opp_out_of_cycle=np.asarray(
            record.get("opp_out_of_cycle", np.zeros((BELIEF_CARD_COUNT,), dtype=bool)),
            dtype=bool,
        ),
        opp_elixir_interval=np.asarray(
            record.get("opp_elixir_interval", [0.0, 10.0]), dtype=np.float32
        ),
        event_history=np.asarray(
            record.get("event_history", np.zeros((EVENT_HISTORY_LEN, EVENT_DIM), dtype=np.float32)),
            dtype=np.float32,
        ),
        legal_play=np.ascontiguousarray(record["legal_play"], dtype=bool),
        legal_wait=bool(record["legal_wait"]),
        own_elixir=float(record["own_elixir"]),
        target=target,
        soft_placement=soft,
        provenance=provenance,
    )

