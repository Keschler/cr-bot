"""Counterfactual search teacher for V4 Stage 2.

Given a real simulator state, the search teacher forks the environment per
candidate branch (WAIT plus legal card plays at candidate cells), rolls each
branch forward with either the legacy WAIT continuation or paired active controllers,
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
import hashlib
from typing import Any, Callable, Sequence

import numpy as np

try:
    from ...actions import PlayCardAction
    from .simulator_teacher import TeacherError, TeacherTarget
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.actions import PlayCardAction
    from simulator.rl.v4.simulator_teacher import TeacherError, TeacherTarget


SEARCH_TEACHER_VERSION: str = "search-v4-1"

# ``wait`` is intentionally the compatibility default.  The active modes
# below are an opt-in continuation teacher: the first action is still
# evaluated from the exact same fork, then both sides keep making actions for
# the remainder of the branch.  Keeping the mode explicit makes reports and
# old checkpoints unambiguous.
CONTINUATION_MODES: tuple[str, ...] = ("wait", "active", "active-rule", "active-policy")
ACTIVE_CONTINUATION_HORIZON: int = 64

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


def _stable_seed(seed: int, *parts: object) -> int:
    """Derive a process-independent seed for paired active continuations."""

    payload = "\x1f".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


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
    continuation_mode: str = "wait"
    opponent_strategy: str = "wait"

    @property
    def best(self) -> BranchScore:
        return self.branches[self.best_index]

    @property
    def branch_count(self) -> int:
        return len(self.branches)


def _normalize_continuation_mode(value: str) -> str:
    """Return a canonical continuation mode or fail closed.

    ``active`` and ``active-rule`` are equivalent aliases.  The shorter name
    is convenient for the Stage-2 CLI; the longer name documents that the
    learner side uses the public rule teacher when no policy callback is
    supplied.
    """

    if not isinstance(value, str):
        raise ValueError("continuation_mode must be a string")
    mode = value.strip().casefold().replace("_", "-")
    if mode == "active":
        return "active-rule"
    if mode not in CONTINUATION_MODES:
        raise ValueError(
            f"continuation_mode must be one of {CONTINUATION_MODES}, got {value!r}"
        )
    return mode


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
    deployed_max_hp: int | None = None,
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
        # Keep the denominator tied to the units deployed by this branch.
        # Filtering dead units before summing max HP makes a destroyed swarm
        # look like a full-health survivor whenever one member remains.
        recorded_max_hp = 0
        deployed_current_hp = 0
        for uid in deployed_uids:
            entity = after_state.entities.get(uid)
            if entity is None:
                continue
            recorded_max_hp += max(1, int(entity.max_hp))
            if entity.alive:
                deployed_current_hp += max(0, int(entity.hp))
        denominator = int(deployed_max_hp) if deployed_max_hp is not None else recorded_max_hp
        if denominator:
            survived = deployed_current_hp / denominator
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


def tested_placement_target(
    result: SearchResult,
    *,
    slot: int,
    legal: np.ndarray,
    fallback: tuple[int, int] | None = None,
    temperature: float = 0.25,
) -> np.ndarray:
    """Build a placement distribution from outcome-tested cells only.

    A common source of accidental label leakage is to turn one selected cell
    into a small geometric neighborhood.  Those neighboring cells may never
    have been simulated and therefore have no outcome evidence.  This helper
    uses the scored branches for ``slot`` as the support of the distribution;
    scores become a deterministic softmax weight.  If the search did not
    include that slot, the explicit fallback is represented as a one-hot
    target.  The fallback is useful for legacy records and is still safe: it
    never invents neighboring support.

    ``legal`` is checked even though search candidates are already legal.  It
    keeps this public boundary fail-closed for callers that deserialize or
    filter branch records before constructing a target.
    """

    mask = np.asarray(legal, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("legal placement mask must be two-dimensional")
    if type(slot) is not int or not 0 <= slot:
        raise ValueError("slot must be a non-negative integer")
    if not np.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    rows, cols = mask.shape
    tested: dict[tuple[int, int], float] = {}
    for branch in result.branches:
        action = branch.action
        if action.kind != "play" or int(action.slot) != slot:
            continue
        cell = (int(action.row), int(action.col))
        if not (0 <= cell[0] < rows and 0 <= cell[1] < cols and bool(mask[cell])):
            continue
        # Duplicate cells can arise when the rule and actor proposals agree;
        # retaining the best observed score keeps the evidence monotonic.
        tested[cell] = max(float(branch.score), tested.get(cell, -float("inf")))
    if not tested:
        if fallback is None:
            raise TeacherError(f"search did not test any legal placement for slot {slot}")
        cell = (int(fallback[0]), int(fallback[1]))
        if not (0 <= cell[0] < rows and 0 <= cell[1] < cols and bool(mask[cell])):
            raise TeacherError("fallback placement is not legal")
        out = np.zeros_like(mask, dtype=np.float32)
        out[cell] = np.float32(1.0)
        return out
    values = np.asarray(list(tested.values()), dtype=np.float64)
    values = np.exp((values - float(values.max())) / float(temperature))
    values /= max(float(values.sum()), np.finfo(np.float64).tiny)
    out = np.zeros((rows, cols), dtype=np.float32)
    for cell, weight in zip(tested, values):
        out[cell] = np.float32(weight)
    # Float32 conversion can leave a tiny normalization error.  Normalize a
    # second time so the training loss receives a proper probability table.
    total = float(out.sum())
    if total <= 0.0:  # pragma: no cover - guarded by tested above
        raise TeacherError("tested placement distribution has no mass")
    return out / np.float32(total)


# A descriptive alias for callers that prefer the provenance terminology.
outcome_tested_placement_target = tested_placement_target


def _fork_child(env: Any) -> Any:
    """Fork preferring the lightweight physics-only path.

    The search path uses ``fork_for_search`` when available: identical
    physics state/RNG isolation without copying observation memories. Active
    continuations rebuild public observations from the child event history.
    Falls back to full :meth:`fork` for foreign env implementations.
    """

    fast = getattr(env, "fork_for_search", None)
    if callable(fast):
        return fast()
    return env.fork()


def _elixir_costs(hand_keys: Sequence[str], ruleset: Any) -> dict[str, int]:
    """Resolve elixir costs once per state (not once per PLAY branch)."""

    costs: dict[str, int] = {}
    for key in dict.fromkeys(hand_keys):
        if not key:
            continue
        try:
            costs[key] = int(ruleset.card(key).elixir_milli)
        except (KeyError, ValueError, AttributeError):
            costs[key] = 0
    return costs


def _apply_branch(
    env: Any,
    action: BranchAction,
    player: int,
    elixir_costs: dict[str, int] | None = None,
) -> tuple[list[int], int]:
    """Apply one branch action; return (deployed uids, elixir spent)."""

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
    if elixir_costs is not None:
        cost = int(elixir_costs.get(action.card_key, 0)) if action.card_key else 0
    else:
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
    for decision in range(horizon):
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


def _partial_hand_tokens(hand: Sequence[str], elixir: float) -> np.ndarray:
    """Encode a transient public hand without importing the model stack.

    During an active continuation a card draw can happen between decisions,
    so the hand can briefly contain fewer than four entries.  The regular
    distillation encoder intentionally rejects that state.  Missing rows are
    zero and therefore unavailable to the public rule teacher.
    """

    try:
        from .simulator_distillation import build_hand_tokens
    except ImportError:  # pragma: no cover - top-level ``rl`` imports
        from simulator.rl.v4.simulator_distillation import build_hand_tokens
    tokens = np.zeros((4, 16), dtype=np.float32)
    for slot, card in enumerate(list(hand)[:4]):
        row = np.asarray(build_hand_tokens([card] * 4, float(elixir))[0], dtype=np.float32)
        row = np.array(row, copy=True)
        row[2] = np.float32(slot / 3.0)
        tokens[slot] = row
    return tokens


def _public_rule_action(env: Any, player: int = 0) -> Any:
    """Choose one active learner action from the public observation only."""

    try:
        from ...actions import PlayCardAction, WaitAction
        from .simulator_teacher import teacher_label
    except ImportError:  # pragma: no cover - top-level ``rl`` imports
        from simulator.actions import PlayCardAction, WaitAction
        from simulator.rl.v4.simulator_teacher import teacher_label

    observation = env.observe_v2_for_viewer(int(player))
    state = env.state
    player_state = state.players[int(player)]
    hand = list(player_state.hand)
    elixir = float(player_state.elixir_milli) / 1000.0
    target = teacher_label(
        hand_tokens=_partial_hand_tokens(hand, elixir),
        entity_tokens=np.asarray(observation.entity_tokens),
        entity_mask=np.asarray(observation.entity_mask, dtype=bool),
        legal_play=np.asarray(observation.legal_play, dtype=bool),
        legal_wait=bool(observation.legal_wait),
        own_elixir=elixir,
    )
    if int(target.mode) == 0:
        return WaitAction(int(player))
    return PlayCardAction(
        int(player), int(target.card_slot), (int(target.col), int(target.row))
    )


def _active_policy_action(
    env: Any,
    observation: Any,
    player: int,
    learner_controller: Any,
) -> Any:
    """Call a public-policy continuation callback with a small API surface.

    The callback receives ``(observation, player)`` when callable.  A
    controller object with ``choose_public_action`` is also accepted.  We do
    not pass authoritative state to this side of the branch, preserving the
    actor's public information boundary.
    """

    if learner_controller is None:
        return _public_rule_action(env, player)
    chooser = getattr(learner_controller, "choose_public_action", None)
    if callable(chooser):
        return chooser(observation, player=int(player))
    if callable(learner_controller):
        try:
            return learner_controller(observation, player=int(player))
        except TypeError:
            return learner_controller(observation, int(player))
    raise TypeError(
        "learner_controller must be callable or expose choose_public_action(observation, player=...)"
    )


def _active_opponent(
    *,
    strategy: str,
    seed: int,
    factory: Callable[[str, int], Any] | None = None,
) -> Any:
    """Build an isolated deterministic opponent for one branch."""

    if factory is not None:
        return factory(str(strategy), int(seed))
    try:
        from ..foundation.opponent_pool import make_opponent_controller
    except ImportError:  # pragma: no cover - top-level ``rl`` imports
        from simulator.rl.foundation.opponent_pool import make_opponent_controller
    return make_opponent_controller(str(strategy), seed=int(seed))


def _rollout_active(
    env: Any,
    horizon: int,
    *,
    continuation_seed: int = 0,
    opponent_strategy: str = "deterministic-cycle",
    learner_controller: Any = None,
    opponent_factory: Callable[[str, int], Any] | None = None,
) -> int:
    """Advance a branch with both players active for ``horizon`` decisions.

    Every branch receives the same ``continuation_seed`` and starts with a
    fresh opponent controller.  A deterministic controller therefore sees
    common random numbers across branches, while branch physics still differ
    only because of the forced first action.  Opponent decisions use the
    authoritative simulator state; learner continuation uses a public
    observation or an explicitly supplied public-policy callback.
    """

    try:
        from ...actions import WaitAction
    except ImportError:  # pragma: no cover - top-level ``rl`` imports
        from simulator.actions import WaitAction
    if type(horizon) is not int or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    opponent = _active_opponent(
        strategy=opponent_strategy,
        seed=int(continuation_seed),
        factory=opponent_factory,
    )
    from copy import deepcopy

    learner_controller = deepcopy(learner_controller)
    ran = 0
    for decision in range(horizon):
        state = getattr(env, "state", None)
        if state is None or state.terminal:
            break
        observation = env.observe_v2_for_viewer(0)
        # The forced branch action already occupies the first decision.
        mine = WaitAction(0) if decision == 0 else _active_policy_action(env, observation, 0, learner_controller)
        if mine is None:
            mine = WaitAction(0)
        opponent_action = opponent.choose_action(env.engine, state, 1)
        if hasattr(env, "step_v2"):
            result = env.step_v2((mine, opponent_action))
            terminated = bool(getattr(result, "terminated", False))
            truncated = bool(getattr(result, "truncated", False))
        else:  # pragma: no cover - foreign env compatibility
            env.step((mine, opponent_action))
            terminated = bool(getattr(env.state, "terminal", False))
            truncated = False
        ran += 1
        if terminated or truncated or env.state is None or env.state.terminal:
            break
    return ran


def _continuation_seed(seed: int, state_hash: str) -> int:
    """Return the paired-randomness seed used by every branch of a state."""

    return int(_stable_seed(int(seed), "continuation", str(state_hash)))


def score_branch(
    env: Any,
    action: BranchAction,
    *,
    player: int = 0,
    horizon: int = HORIZON_DECISIONS,
    before_towers: dict[int, tuple[int, int]] | None = None,
    before_threats: dict[int, tuple[int, int]] | None = None,
    elixir_costs: dict[str, int] | None = None,
    continuation_mode: str = "wait",
    continuation_seed: int = 0,
    opponent_strategy: str = "deterministic-cycle",
    learner_controller: Any = None,
    opponent_factory: Callable[[str, int], Any] | None = None,
) -> BranchScore:
    """Fork, apply one branch, roll out, and score (child discarded).

    ``before_towers``/``before_threats``/``elixir_costs`` are hoisted
    per-state by :func:`search_state` (they are identical for every branch
    of the same parent state); direct callers may omit them to preserve
    the original per-branch computation exactly.
    """

    if player != 0:
        raise ValueError("search branching currently supports target_player 0 only")
    mode = _normalize_continuation_mode(continuation_mode)
    if mode == "active-policy" and learner_controller is None:
        raise ValueError("active-policy requires learner_controller")
    child = _fork_child(env)
    if before_towers is None:
        before_towers = _tower_totals(child.state)
    if before_threats is None:
        before_threats = _threat_health(child.state, owner=1 - player)
    deployed, cost = _apply_branch(child, action, player, elixir_costs)
    # Snapshot this before the rollout can remove dead entities from the
    # state's entity map.  The survival denominator must include every unit
    # deployed by the branch, including units that die later.
    deployed_max_hp = sum(
        max(1, int(child.state.entities[uid].max_hp))
        for uid in deployed
        if uid in child.state.entities
    )
    if mode == "wait":
        steps_run = _rollout(child, horizon)
    else:
        steps_run = _rollout_active(
            child,
            horizon,
            continuation_seed=int(continuation_seed),
            opponent_strategy=str(opponent_strategy),
            learner_controller=learner_controller,
            opponent_factory=opponent_factory,
        )
    after_towers = _tower_totals(child.state)
    score, terms = score_outcome(
        before_towers=before_towers,
        after_towers=after_towers,
        before_threats=before_threats,
        after_state=child.state,
        deployed_uids=deployed,
        elixir_spent_milli=cost,
        deployed_max_hp=deployed_max_hp,
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
    continuation_mode: str = "wait",
    opponent_strategy: str = "deterministic-cycle",
    learner_controller: Any = None,
    opponent_factory: Callable[[str, int], Any] | None = None,
) -> SearchResult:
    """Run the full counterfactual search on one live environment state."""

    if env.state is None or env.state.terminal:
        raise TeacherError("cannot search a missing or terminal state")
    mode = _normalize_continuation_mode(continuation_mode)
    state_hash = env.state.state_hash()
    branches = candidate_actions(
        legal_play=np.asarray(legal_play, dtype=bool),
        rule_action=rule_action,
        actor_cells=actor_cells,
        hand_keys=list(hand_keys),
        max_branches=max_branches,
    )
    # Hoisted per-state snapshots: the fork is exact, so reading the parent
    # gives bit-identical values to reading each child before its branch
    # action, while saving 23/24 tower/threat scans and up to 24
    # ruleset/card cost resolutions per searched state.
    parent_towers = _tower_totals(env.state)
    parent_threats = _threat_health(env.state, owner=1 - player)
    try:
        costs = _elixir_costs(list(hand_keys), env.engine.ruleset)
    except AttributeError:
        costs = None
    # One continuation seed is shared by every branch.  Deriving it from the
    # parent state (rather than the candidate action) is what makes active
    # branch comparisons paired/common-random-number comparisons.
    continuation_seed = _continuation_seed(int(seed), state_hash)
    scored = [
        score_branch(
            env, action, player=player, horizon=horizon,
            before_towers=parent_towers, before_threats=parent_threats,
            elixir_costs=costs,
            continuation_mode=mode,
            continuation_seed=continuation_seed,
            opponent_strategy=opponent_strategy,
            learner_controller=learner_controller,
            opponent_factory=opponent_factory,
        )
        for action in branches
    ]
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
        continuation_mode=mode,
        opponent_strategy=("wait" if mode == "wait" else str(opponent_strategy)),
    )


__all__ = [
    "ACTIVE_CONTINUATION_HORIZON",
    "CONTINUATION_MODES",
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
    "_continuation_seed",
    "outcome_tested_placement_target",
    "score_branch",
    "score_outcome",
    "search_state",
    "tested_placement_target",
]
