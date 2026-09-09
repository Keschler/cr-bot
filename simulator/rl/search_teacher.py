"""Counterfactual search teacher for V4 Stage 2.

Given a real simulator state, the search teacher forks the environment per
candidate branch (WAIT plus legal card plays at candidate cells), rolls each
branch forward a short deterministic horizon with a frozen WAIT opponent,
and scores the resulting simulator outcomes.  It produces the best action,
per-branch scores, and regrets — outcome-derived supervision that moves
beyond rule imitation.

Information contract (paper asymmetric actor-critic, Stage-2 scope):

* actor inputs and search *labels* use public observations only;
* the branch *scores* may use authoritative simulator outcomes (tower HP,
  threat HP, deployed-unit survival, elixir spent).  This is the explicitly
  allowed critic/evaluator privilege, identical in spirit to PPO's
  privileged critic: scores never enter the actor's observation.

Calls with ``target_player != 0`` are rejected: branch cells are indexed in
world coordinates and viewer mirroring for player 1 is deferred work.  All
Stage-2 data uses target/viewer 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

try:
    from .simulator_teacher import TeacherError, TeacherTarget
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.rl.simulator_teacher import TeacherError, TeacherTarget


SEARCH_TEACHER_VERSION: str = "search-v4-0"

HORIZON_DECISIONS: int = 28
"""Default rollout horizon per branch (7s of game time).

Calibrated so consequences materialize: deployed attackers connect with
towers (Hog needs ~4 decisions of deploy freeze plus ~10 decisions of
walking before its hits register), threats advance into tower range, and
elixir commitment is priced.  Shorter horizons (8-20) recommend WAIT
nearly everywhere because nothing has happened yet.
"""

MAX_BRANCHES: int = 24
"""Hard cap on evaluated branches per state (compute control)."""

# Branch scoring weights.  Small, few, and documented (Stage-2 mandate):
# tower consequences dominate; removed threat is the defensive signal;
# deployed-survivor value counts only when the branch accomplished
# something (threat removed or tower damaged), otherwise pointless cycling
# would outscore WAIT; elixir spent prices commitment.
W_TOWER_DEALT: float = 2.0
W_TOWER_TAKEN: float = 2.0
W_THREAT_REMOVED: float = 1.0
W_SURVIVOR: float = 0.25
W_ELIXIR_SPENT: float = 1.0


@dataclass(frozen=True, slots=True)
class BranchAction:
    """One counterfactual branch: WAIT or PLAY (slot, row, col)."""

    kind: str  # "wait" | "play"
    slot: int = 0
    row: int = 0
    col: int = 0
    card_key: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("wait", "play"):
            raise ValueError(f"branch kind must be wait|play, got {self.kind!r}")


@dataclass(frozen=True, slots=True)
class BranchScore:
    """Scored branch with its outcome-term breakdown."""

    action: BranchAction
    score: float
    terms: dict[str, float]
    steps_run: int


@dataclass(slots=True)
class SearchResult:
    """Full provenance for one searched state."""

    state_hash: str
    seed: int
    source: str
    family: str
    horizon: int
    branches: list[BranchScore] = field(default_factory=list)
    best_index: int = 0
    second_gap: float = 0.0

    @property
    def best(self) -> BranchScore:
        return self.branches[self.best_index]

    @property
    def branch_count(self) -> int:
        return len(self.branches)


def _tower_totals(state: Any) -> dict[int, tuple[int, int]]:
    totals: dict[int, list[int]] = {0: [0, 0], 1: [0, 0]}
    for entity in state.entities.values():
        if entity.kind == "tower":
            totals[entity.owner][0] += max(0, int(entity.hp))
            totals[entity.owner][1] += max(1, int(entity.max_hp))
    return {owner: (hp, maximum) for owner, (hp, maximum) in totals.items()}


def _threat_health(state: Any, owner: int = 1) -> dict[int, tuple[int, int]]:
    return {
        entity.uid: (max(0, int(entity.hp)), max(1, int(entity.max_hp)))
        for entity in state.entities.values()
        if entity.alive and entity.owner == owner and entity.kind != "tower"
    }


def score_outcome(
    *,
    before_towers: dict[int, tuple[int, int]],
    after_towers: dict[int, tuple[int, int]],
    before_threats: dict[int, tuple[int, int]],
    after_state: Any,
    deployed_uids: Sequence[int],
    elixir_spent_milli: int,
    player: int = 0,
) -> tuple[float, dict[str, float]]:
    """Score a branch outcome from authoritative snapshots.

    All fractions use pre-branch maxima as denominators so branches from
    the same state are directly comparable.  Returns ``(score, terms)``.
    """

    opponent = 1 - player
    _, enemy_max = before_towers[opponent]
    _, own_max = before_towers[player]
    dealt = max(0, before_towers[opponent][0] - after_towers[opponent][0]) / max(1, enemy_max)
    taken = max(0, before_towers[player][0] - after_towers[player][0]) / max(1, own_max)
    start_hp = sum(hp for hp, _ in before_threats.values())
    if start_hp > 0:
        end_hp = sum(
            max(0, int(after_state.entities[uid].hp))
            if uid in after_state.entities and after_state.entities[uid].alive
            else 0
            for uid in before_threats
        )
        removed = max(0.0, (start_hp - end_hp) / start_hp)
    else:
        removed = 0.0
    survived = 0.0
    if deployed_uids and (removed > 0.0 or dealt > 0.0):
        current = [
            after_state.entities[uid]
            for uid in deployed_uids
            if uid in after_state.entities and after_state.entities[uid].alive
        ]
        if current:
            survived = sum(max(0, int(e.hp)) for e in current) / max(
                1, sum(max(1, int(e.max_hp)) for e in current)
            )
    spent = max(0, int(elixir_spent_milli)) / 10_000.0
    terms = {
        "tower_dealt": float(dealt),
        "tower_taken": float(taken),
        "threat_removed": float(removed),
        "deployed_survived": float(survived),
        "elixir_spent": float(spent),
    }
    score = (
        W_TOWER_DEALT * dealt
        - W_TOWER_TAKEN * taken
        + W_THREAT_REMOVED * removed
        + W_SURVIVOR * survived
        - W_ELIXIR_SPENT * spent
    )
    return float(score), terms


def _spiral_cells(center: tuple[int, int], radius: int = 2) -> list[tuple[int, int]]:
    row, col = center
    cells = [(row, col)]
    for ring in range(1, radius + 1):
        for dr in range(-ring, ring + 1):
            for dc in range(-ring, ring + 1):
                if max(abs(dr), abs(dc)) == ring:
                    cells.append((row + dr, col + dc))
    return cells


def candidate_actions(
    *,
    legal_play: np.ndarray,
    rule_action: TeacherTarget | None,
    actor_cells: Sequence[tuple[int, int, int]] = (),
    lane_anchors: Sequence[tuple[int, int]] | None = None,
    hand_keys: Sequence[str] = (),
    max_branches: int = MAX_BRANCHES,
) -> list[BranchAction]:
    """Build a bounded, deduplicated, all-legal candidate list.

    Priority order (deterministic): WAIT, the rule-teacher action, actor
    cells, then per-card round-robin over (teacher cell, spiral ring,
    lane anchors) so one card cannot consume the whole branch budget.
    Every candidate is verified against the legal mask, so search never
    evaluates an illegal branch.
    """

    if type(max_branches) is not int or max_branches < 2:
        raise ValueError("max_branches must be an integer >= 2")
    slots = legal_play.shape[0]
    rows, cols = legal_play.shape[1], legal_play.shape[2]
    anchors = list(lane_anchors) if lane_anchors else [(rows // 2, 4), (rows // 2, cols - 5)]
    keys = list(hand_keys) + [""] * max(0, slots - len(hand_keys))
    ordered: list[BranchAction] = [BranchAction(kind="wait")]
    seen: set[tuple] = {("wait",)}

    def add(action: BranchAction) -> None:
        key = (action.kind, action.slot, action.row, action.col)
        if key in seen or len(ordered) >= max_branches:
            return
        if action.kind == "play":
            if not (0 <= action.slot < slots and 0 <= action.row < rows and 0 <= action.col < cols):
                return
            if not bool(legal_play[action.slot, action.row, action.col]):
                return
        seen.add(key)
        ordered.append(action)

    def keyed(slot: int, row: int, col: int) -> BranchAction:
        return BranchAction(kind="play", slot=slot, row=row, col=col, card_key=keys[slot])

    if rule_action is not None and rule_action.mode == 1:
        add(keyed(rule_action.card_slot, rule_action.row, rule_action.col))
    for slot, row, col in actor_cells:
        if len(ordered) >= max_branches:
            break
        add(keyed(int(slot), int(row), int(col)))
    per_slot: list[list[BranchAction]] = []
    for slot in range(slots):
        legal = np.asarray(legal_play[slot])
        if not bool(legal.any()):
            continue
        centers: list[tuple[int, int]] = []
        if rule_action is not None and rule_action.mode == 1 and rule_action.card_slot == slot:
            centers.append((rule_action.row, rule_action.col))
        centers.extend(anchors)
        # Anchor cells first (strategically distinct positions are always
        # evaluated), then the spiral ring for local refinement.
        priority: list[tuple[int, int]] = []
        for center in centers:
            if 0 <= center[0] < rows and 0 <= center[1] < cols and bool(legal[center]):
                priority.append(center)
        cells: list[BranchAction] = []
        cell_seen: set[tuple[int, int]] = set()
        for row, col in priority:
            cell_seen.add((row, col))
            cells.append(keyed(slot, row, col))
            if len(cells) >= 8:
                break
        if len(cells) < 8:
            for center in centers:
                for row, col in _spiral_cells(center):
                    if (row, col) in cell_seen or len(cells) >= 8:
                        continue
                    if 0 <= row < rows and 0 <= col < cols and bool(legal[row, col]):
                        cell_seen.add((row, col))
                        cells.append(keyed(slot, row, col))
        per_slot.append(cells)
    round_idx = 0
    while len(ordered) < max_branches and any(len(cells) > round_idx for cells in per_slot):
        for cells in per_slot:
            if len(ordered) >= max_branches:
                break
            if len(cells) > round_idx:
                add(cells[round_idx])
        round_idx += 1
    return ordered
    return ordered


def _apply_branch(env: Any, action: BranchAction, player: int) -> tuple[list[int], int]:
    """Apply one branch action; return (deployed uids, elixir spent)."""

    try:
        from .actions import PlayCardAction
    except ImportError:  # pragma: no cover
        from simulator.actions import PlayCardAction
    state = env.state
    if action.kind == "wait":
        return [], 0
    before = set(state.entities)
    result = env.engine.apply_actions(
        state, (PlayCardAction(player, action.slot, (action.col, action.row)),)
    )
    if len(result) != 1 or not result[0].accepted:
        raise TeacherError(
            f"search branch rejected: slot={action.slot} cell=({action.col},{action.row}) "
            f"reason={result[0].reason if result else None}"
        )
    deployed = sorted(set(state.entities) - before)
    try:
        cost = int(env.engine.ruleset.card(action.card_key).elixir_milli) if action.card_key else 0
    except (KeyError, ValueError, AttributeError):
        cost = 0
    return deployed, cost


def _rollout(env: Any, horizon: int) -> int:
    """Advance ``horizon`` decisions with both players WAITing.

    Steps the engine tick-by-tick directly instead of going through
    ``step_v2``: branch rollouts need physics only, and observation
    building (rasters, entity rows, memories) would dominate the cost.
    Both sides play WAIT (empty action tuple) every tick.  Returns
    decisions actually run.
    """

    ticks_per_decision = int(getattr(env, "decision_interval_ticks", 5) or 5)
    ran = 0
    for _ in range(horizon):
        state = env.state
        if state is None or state.terminal:
            break
        for _ in range(ticks_per_decision):
            if state.terminal:
                break
            env.engine.step(state, ())
        ran += 1
        if state.terminal:
            break
    return ran


def score_branch(
    env: Any,
    action: BranchAction,
    *,
    player: int = 0,
    horizon: int = HORIZON_DECISIONS,
) -> BranchScore:
    """Fork, apply one branch, roll out, and score (child discarded)."""

    if player != 0:
        raise ValueError("search branching currently supports target_player 0 only")
    child = env.fork()
    before_towers = _tower_totals(child.state)
    before_threats = _threat_health(child.state, owner=1 - player)
    deployed, cost = _apply_branch(child, action, player)
    steps_run = _rollout(child, horizon)
    after_towers = _tower_totals(child.state)
    score, terms = score_outcome(
        before_towers=before_towers,
        after_towers=after_towers,
        before_threats=before_threats,
        after_state=child.state,
        deployed_uids=deployed,
        elixir_spent_milli=cost,
        player=player,
    )
    return BranchScore(action=action, score=score, terms=terms, steps_run=steps_run)


def search_state(
    env: Any,
    *,
    legal_play: np.ndarray,
    rule_action: TeacherTarget | None,
    actor_cells: Sequence[tuple[int, int, int]] = (),
    hand_keys: Sequence[str] = (),
    player: int = 0,
    horizon: int = HORIZON_DECISIONS,
    max_branches: int = MAX_BRANCHES,
    seed: int = 0,
    source: str = "",
    family: str = "",
) -> SearchResult:
    """Run the full counterfactual search on one live environment state."""

    if env.state is None or env.state.terminal:
        raise TeacherError("cannot search a missing or terminal state")
    state_hash = env.state.state_hash()
    branches = candidate_actions(
        legal_play=np.asarray(legal_play, dtype=bool),
        rule_action=rule_action,
        actor_cells=actor_cells,
        hand_keys=list(hand_keys),
        max_branches=max_branches,
    )
    scored = [score_branch(env, action, player=player, horizon=horizon) for action in branches]
    order = sorted(range(len(scored)), key=lambda i: (-scored[i].score, i))
    best = order[0]
    second_gap = scored[best].score - scored[order[1]].score if len(order) > 1 else 0.0
    return SearchResult(
        state_hash=state_hash,
        seed=int(seed),
        source=str(source),
        family=str(family),
        horizon=int(horizon),
        branches=scored,
        best_index=int(best),
        second_gap=float(second_gap),
    )


__all__ = [
    "HORIZON_DECISIONS",
    "MAX_BRANCHES",
    "SEARCH_TEACHER_VERSION",
    "W_ELIXIR_SPENT",
    "W_SURVIVOR",
    "W_THREAT_REMOVED",
    "W_TOWER_DEALT",
    "W_TOWER_TAKEN",
    "BranchAction",
    "BranchScore",
    "SearchResult",
    "candidate_actions",
    "score_branch",
    "score_outcome",
    "search_state",
]
