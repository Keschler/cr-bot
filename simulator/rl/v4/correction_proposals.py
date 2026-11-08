"""Bounded public action proposals for independently evaluated replay repairs.

These proposals are hypotheses, not action masks or training labels.  They
include the recorded action, WAIT and the existing teachers, then add visible
siege interception, air defense and ranged-ground alternatives.  Only a
counterfactual evaluator can establish whether any alternative is useful.
No engine, target identity, target lock or future observation is accepted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from .frontier_eval import BranchSpec, action_key
from .hand_features import V4_LEARNER_DECK
from .simulator_teacher import TeacherError, cycle_teacher_label, defense_teacher_label


PROPOSAL_VERSION = "public-replay-correction-proposals-v2"
_ROWS, _COLS = 32, 18
_HAND = {card[1]: (card[0], card[2]) for card in V4_LEARNER_DECK}
_SIEGE = frozenset(("x-bow", "mortar"))


@dataclass(frozen=True, slots=True)
class _Body:
    name: str
    x: float
    y: float
    air: bool
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _Public:
    entities: np.ndarray
    entity_mask: np.ndarray
    hand: np.ndarray
    legal: np.ndarray
    wait: bool
    elixir: float
    names: tuple[str | None, ...]
    costs: tuple[int, ...]
    enemies: tuple[_Body, ...]


@dataclass(frozen=True, slots=True)
class _Geometry:
    source: str
    category: str
    role: str
    slot: int
    cells: tuple[tuple[int, int], ...]


@dataclass(slots=True)
class _Alternative:
    branch: BranchSpec
    # A single cell can be a credible proposal for several visible bodies.
    # These are public proposal groups, never engine entity identities.
    groups: set[tuple[str, str | int]]


@lru_cache(maxsize=1)
def _cards() -> tuple[int, dict[int, tuple[str, Mapping[str, Any]]]]:
    from cr_bot.domain.card_metadata import CARD_METADATA

    return max(int(row["id"]) for row in CARD_METADATA.values()), {
        int(row["id"]): (name, row) for name, row in CARD_METADATA.items()
    }


def _metadata(name: str) -> Mapping[str, Any]:
    return next(metadata for card_name, metadata in _cards()[1].values() if card_name == name)


def _public(inputs: Mapping[str, Any], own_elixir: float | None) -> _Public:
    entities = np.asarray(inputs["entities"], dtype=np.float32)
    mask = np.asarray(inputs["entity_mask"], dtype=bool)
    hand = np.asarray(inputs["hand_tokens"], dtype=np.float32)
    legal = np.asarray(inputs["legal_play"], dtype=bool)
    if entities.shape != (128, 32) or mask.shape != (128,):
        raise ValueError("entities/entity_mask must use the V4 [128,32]/[128] contract")
    if hand.shape != (4, 16) or legal.shape != (4, _ROWS, _COLS):
        raise ValueError("hand_tokens/legal_play must use the V4 [4,16]/[4,32,18] contract")
    if not np.isfinite(entities[:, :11]).all() or not np.isfinite(hand).all():
        raise ValueError("public identity, geometry and hand features must be finite")
    if own_elixir is None:
        global_features = np.asarray(inputs["global_features"])
        if global_features.shape != (768,):
            raise ValueError("global_features must use the V4 [768] contract")
        own_elixir = float(global_features[0]) * 10.0
    elixir = float(own_elixir)
    if not np.isfinite(elixir) or not 0 <= elixir <= 10:
        raise ValueError("own_elixir must be finite and between zero and ten")
    names, costs = [], []
    for row in hand:
        identity = _HAND.get(int(round(float(row[0]) * 127)))
        names.append(identity[0] if identity else None)
        costs.append(identity[1] if identity else 0)
    maximum, catalog = _cards()
    enemies = []
    for row, present in zip(entities, mask, strict=True):
        if (not present or row[1] <= .5 or row[4] <= 0 or row[7] > .5
                or row[8] > .5 or row[9] <= .5 or row[10] <= .5
                or not (0 <= row[2] <= 1 and 0 <= row[3] <= 1)):
            continue
        identity = catalog.get(int(round(float(row[0]) * maximum)))
        if identity is None:
            continue
        name, metadata = identity
        if (metadata.get("kind") not in ("troop", "building")
                or bool(row[5] > .5) != bool(metadata.get("is_air"))
                or bool(row[6] > .5) != (metadata.get("kind") == "building")):
            continue
        enemies.append(_Body(name, float(row[2]) * _COLS, float(row[3]) * _ROWS,
                             bool(row[5] > .5), metadata))
    # Observed urgency then card/geometry ordering makes proposals independent
    # of entity-table ordering.  The tail includes no data required here.
    enemies.sort(key=lambda body: (-body.y, body.name, body.x))
    selected = np.zeros_like(entities)
    selected[:, :11] = entities[:, :11]
    return _Public(selected, mask, hand, legal, bool(inputs["legal_wait"]), elixir,
                   tuple(names), tuple(costs), tuple(enemies))


def _recorded(record: BranchSpec | Mapping[str, Any]) -> BranchSpec:
    if isinstance(record, BranchSpec):
        return BranchSpec("sim", record.mode, record.slot, record.row, record.col)
    if "mode" not in record:
        raise ValueError("recorded learner action requires mode")
    return BranchSpec("sim", int(record["mode"]), int(record.get("slot", 0)),
                      int(record.get("row", 0)), int(record.get("col", 0)))


def _playable(state: _Public, slot: int) -> bool:
    return (state.names[slot] is not None and state.costs[slot] <= state.elixir + 1e-6
            and bool(state.hand[slot, 3] > .5) and bool(state.hand[slot, 4] > .5)
            and bool(state.legal[slot].any()))


def _threat_category(body: _Body) -> str:
    if body.name in _SIEGE:
        return "siege"
    if body.air:
        return "air"
    return "ranged-ground" if float(body.metadata.get("range") or 0) >= 4 else "ground"


def _categories(state: _Public, recorded: BranchSpec) -> tuple[str, ...]:
    categories = []
    if any(body.name in _SIEGE for body in state.enemies):
        categories.append("siege")
    if any(body.air for body in state.enemies):
        categories.append("air")
    if any(not body.air and body.name not in _SIEGE
           and float(body.metadata.get("range") or 0) >= 4 for body in state.enemies):
        categories.append("ranged-ground")
    if recorded.mode == 1 and 0 <= recorded.slot < 4:
        # This is a candidate flag, not a verdict about the factual spend.
        # Keep valid tanks/stalls available and compare WAIT counterfactually.
        reserves = []
        if "air" in categories:
            reserves.extend(state.costs[slot] for slot, name in enumerate(state.names)
                            if name in ("musketeer", "fireball"))
        if "siege" in categories:
            reserves.extend(state.costs[slot] for slot, name in enumerate(state.names)
                            if name in ("cannon", "musketeer"))
        if reserves and state.elixir - state.costs[recorded.slot] < min(reserves):
            categories.append("upstream-spend")
        if (state.names[recorded.slot] == "cannon"
                and any(body.name == "princess" for body in state.enemies)):
            categories.append("cannon-princess-placement")
        # Distinguish public requests by family and lane, rather than calling
        # each member of one swarm a separate threat. Nearby ground bodies
        # remain visible alongside the air/siege request they compete with.
        requests = {(_threat_category(body), body.x < _COLS / 2)
                    for body in state.enemies
                    if _threat_category(body) != "ground" or body.y >= 13}
        if len(requests) >= 2:
            categories.append("multi-threat-spend")
    return tuple(categories)


def candidate_categories(
    inputs: Mapping[str, Any], recorded_action: BranchSpec | Mapping[str, Any], *,
    own_elixir: float | None = None,
) -> tuple[str, ...]:
    """Identify visible repair contexts, including possible reserve spending.

    A pre-threat upstream window needs caller-selected earlier observations;
    this function never examines a future Balloon deployment or outcome.
    """

    return _categories(_public(inputs, own_elixir), _recorded(recorded_action))


def _cell(
    legal: np.ndarray, anchor: tuple[float, float],
    credible: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[int, int] | None:
    rows, cols = np.nonzero(legal)
    if not len(rows):
        return None
    keep = credible(rows + .5, cols + .5)
    rows, cols = rows[keep], cols[keep]
    if not len(rows):
        return None
    distance = (rows + .5 - anchor[0]) ** 2 + (cols + .5 - anchor[1]) ** 2
    # nonzero is row-major and argmin preserves its deterministic ties.
    index = int(np.argmin(distance))
    return int(rows[index]), int(cols[index])


def _geometric(state: _Public, body: _Body, name: str) -> _Geometry | None:
    slot = state.names.index(name) if name in state.names else -1
    if slot < 0 or not _playable(state, slot):
        return None
    legal = state.legal[slot]
    siege = body.name in _SIEGE
    category = _threat_category(body)
    # Metadata range uses tile units. Center distance is deliberately
    # conservative: public tokens do not contain collision footprints.
    def within(radius: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return lambda r, c: (r - body.y) ** 2 + (c - body.x) ** 2 <= radius ** 2
    inward = 1 if body.x < _COLS / 2 else -1
    if name == "fireball":
        anchor, credible = (body.y, body.x), within(float(_metadata(name)["radius"]))
        role = "burst-damage"
    elif name == "log":
        if body.air or siege or body.metadata.get("kind") != "troop":
            return None
        # Aim just behind a visible ground body. No unobserved future path
        # or assumed target lock is needed for an immediate spell candidate.
        anchor = (min(31.5, body.y + 1), body.x)
        half_width = float(_metadata(name)["width"]) / 2
        credible = lambda r, c: ((r >= body.y) & (r - body.y <= 3)
                                 & (np.abs(c - body.x) <= half_width))
        role = "ground-clear"
    elif name == "musketeer":
        anchor = (max(17.5, body.y + 3), body.x + inward * 2)
        credible = within(float(_metadata(name)["range"]))
        role = "sustained-damage"
    elif name == "cannon":
        if body.air:
            if not body.metadata.get("targets_buildings_only"):
                return None
            anchor = (max(18.5, min(22.5, body.y + 2)), 7.5 if body.x < 9 else 10.5)
            credible = within(6.0)  # pull hypothesis, never claimed air damage
            role = "pull"
        else:
            anchor = (max(17.5, body.y + 2), body.x)
            credible = within(float(_metadata(name)["range"]))  # must be able to reach
            role = "stationary-defense"
    elif name in ("ice-golem", "skeletons", "hog-rider"):
        if body.air or (not siege and body.metadata.get("kind") != "troop"):
            return None
        if name == "hog-rider" and not siege:
            return None  # does not attack troops; factual/teacher pressure survives
        anchor = (max(16.5, body.y + 1), body.x)
        reach = float(body.metadata.get("range") or 0) if siege else 3.0
        # For siege these bodies are visible interception/approach candidates;
        # they do not imply an already connected X-Bow will switch targets.
        credible = within(reach)
        role = {"ice-golem": "tank", "skeletons": "surround", "hog-rider": "building-pressure"}[name]
    elif name == "ice-spirit":
        anchor = (max(16.5, body.y + 1), body.x)
        credible = within(float(_metadata(name)["range"]) + 1)
        role = "stall"
    else:
        return None
    cell = _cell(legal, anchor, credible)
    if cell is None:
        return None
    # Test actual adjacent cells against the same public purpose. Neighbors
    # never displace another card/body's primary hypothesis in a tight budget.
    cells = [cell]
    for dr, dc in ((0, inward), (1, 0), (-1, 0), (0, -inward),
                   (1, inward), (-1, inward), (1, -inward), (-1, -inward)):
        row, col = cell[0] + dr, cell[1] + dc
        if (0 <= row < _ROWS and 0 <= col < _COLS and legal[row, col]
                and bool(credible(np.asarray(row + .5), np.asarray(col + .5)))):
            cells.append((row, col))
    return _Geometry(f"{category}-geometry", category, role, slot, tuple(cells))


def _geometry_pools(state: _Public) -> tuple[list[_Alternative], list[_Alternative]]:
    """Build all primary groups before spending a bounded proposal budget."""

    primary: dict[tuple[int, int, int, int], _Alternative] = {}
    neighbors: dict[tuple[int, int, int, int], _Alternative] = {}
    for index, body in enumerate(state.enemies):
        category = _threat_category(body)
        if category == "ground" and body.y < 13:
            continue
        if category == "siege":
            names = ("cannon", "musketeer", "ice-golem", "skeletons", "hog-rider", "fireball", "ice-spirit")
        elif category == "air":
            names = ("musketeer", "fireball", "ice-spirit", "cannon")
        else:
            names = ("log", "musketeer", "skeletons", "ice-golem", "cannon", "fireball", "ice-spirit")
        for name in names:
            geometry = _geometric(state, body, name)
            if geometry is None:
                continue
            groups = {("category", geometry.category), ("threat", index),
                      ("card", geometry.slot), ("role", geometry.role)}
            for ordinal, (row, col) in enumerate(geometry.cells):
                branch = BranchSpec(geometry.source, 1, geometry.slot, row, col)
                pool = primary if ordinal == 0 else neighbors
                key = action_key(branch)
                if key not in pool:
                    pool[key] = _Alternative(branch, set())
                pool[key].groups.update(groups)
    # Credit every supported public purpose even when one group's neighbor is
    # another group's primary cell, or a teacher already proposed that action.
    for key in primary.keys() & neighbors.keys():
        primary[key].groups.update(neighbors[key].groups)
        neighbors[key].groups.update(primary[key].groups)
    return list(primary.values()), list(neighbors.values())


def _quiet_pools(state: _Public) -> tuple[list[_Alternative], list[_Alternative]]:
    """Offer public cycle/setup hypotheses when there is no reachable request.

    These placements claim no immediate defensive value. They let an earlier
    spending decision compete with WAIT and alternate setup/cycle choices;
    no future enemy deployment is used to choose their card or lane.
    """

    primary, neighbors = [], []
    for slot, name in enumerate(state.names):
        if not _playable(state, slot) or name in ("log", "fireball"):
            continue
        if name == "hog-rider":
            role, anchors = "pressure", ((17.5, 3.5), (17.5, 14.5))
        elif name == "cannon":
            role, anchors = "defensive-setup", ((20.5, 7.5), (20.5, 10.5))
        else:
            role = "cycle" if state.costs[slot] <= 2 else "defensive-setup"
            anchors = ((27.5, 2.5), (27.5, 15.5))
        for index, anchor in enumerate(anchors):
            credible = lambda r, c: (r - anchor[0]) ** 2 + (c - anchor[1]) ** 2 <= 2 ** 2
            cell = _cell(state.legal[slot], anchor, credible)
            if cell is None:
                continue
            branch = BranchSpec("quiet-setup", 1, slot, *cell)
            groups = {("category", "quiet"), ("threat", f"quiet-lane-{index}"),
                      ("card", slot), ("role", role)}
            (primary if index == 0 else neighbors).append(_Alternative(branch, groups))
    return primary, neighbors


def propose_correction_branches(
    inputs: Mapping[str, Any], recorded_action: BranchSpec | Mapping[str, Any], *,
    own_elixir: float | None = None, max_branches: int = 12,
) -> list[BranchSpec]:
    """Return deterministic legal hypotheses, preserving the exact baseline.

    ``max_branches`` bounds the whole list, including the factual action and
    WAIT. Existing teachers are proposal sources only; their choices and all
    geometric alternatives still require independent outcome validation.
    Primary alternatives balance visible threats, cards, roles and families
    before nearby placements consume any remaining budget.
    """

    if type(max_branches) is not int or max_branches < 2:
        raise ValueError("max_branches must be an integer >= 2")
    state, recorded = _public(inputs, own_elixir), _recorded(recorded_action)
    if recorded.mode == 0:
        if not state.wait:
            raise ValueError("recorded WAIT is not legal in its public origin")
    elif recorded.mode == 1:
        if (not 0 <= recorded.slot < 4 or not 0 <= recorded.row < _ROWS
                or not 0 <= recorded.col < _COLS or not _playable(state, recorded.slot)
                or not state.legal[recorded.slot, recorded.row, recorded.col]):
            raise ValueError("recorded PLAY is not legal in its public origin")
    else:
        raise ValueError("recorded action mode must be WAIT=0 or PLAY=1")
    branches = [recorded]
    seen = {action_key(recorded)}

    def add(source: str, mode: int, slot: int = 0, row: int = 0, col: int = 0) -> None:
        branch = BranchSpec(source, mode, slot, row, col)
        key = action_key(branch)
        if len(branches) >= max_branches or key in seen:
            return
        if mode == 0:
            if not state.wait:
                return
        elif (not 0 <= slot < 4 or not 0 <= row < _ROWS or not 0 <= col < _COLS
              or not _playable(state, slot) or not state.legal[slot, row, col]):
            return
        seen.add(key)
        branches.append(branch)

    add("wait", 0)
    kwargs = dict(hand_tokens=state.hand, entity_tokens=state.entities,
                  entity_mask=state.entity_mask, legal_play=state.legal,
                  legal_wait=state.wait, own_elixir=state.elixir)
    for source, labeler in (("cycle-teacher", cycle_teacher_label),
                            ("defense-teacher", defense_teacher_label)):
        try:
            target = labeler(**kwargs)
        except TeacherError:
            continue
        add(source, int(target.mode), int(target.card_slot), int(target.row), int(target.col))
    primary, neighbors = _geometry_pools(state)
    if not primary:
        primary, neighbors = _quiet_pools(state)
    support: dict[tuple[int, int, int, int], set[tuple[str, str | int]]] = {}
    for alternative in (*primary, *neighbors):
        support.setdefault(action_key(alternative.branch), set()).update(alternative.groups)
    counts: Counter[tuple[str, str | int]] = Counter()
    for branch in branches:
        counts.update(support.get(action_key(branch), ()))
    for pool in (primary, neighbors):
        remaining = [alternative for alternative in pool if action_key(alternative.branch) not in seen]
        while remaining and len(branches) < max_branches:
            def coverage(alternative: _Alternative) -> tuple[int, float]:
                groups = support[action_key(alternative.branch)]
                return (sum(counts[group] == 0 for group in groups),
                        sum(1 / (1 + counts[group]) for group in sorted(groups)))

            # A new body/card/role/family wins over another cell for an
            # already represented group. Stable pool order resolves ties.
            index = max(range(len(remaining)), key=lambda i: coverage(remaining[i]))
            alternative = remaining.pop(index)
            branch = alternative.branch
            add(branch.source, branch.mode, branch.slot, branch.row, branch.col)
            counts.update(support[action_key(branch)])
    return branches


__all__ = ["PROPOSAL_VERSION", "candidate_categories", "propose_correction_branches"]
