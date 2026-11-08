"""Small public-state rules teacher used by V4 bootstrap and frontier audit.

The teacher maps a public observation to a supervised target
``(mode, card, placement, WAIT duration)`` using validated tactical rules
plus simulator legality.  It is deliberately *not* a strategy oracle: each
rule keys off immediate public facts (threat position, cluster density,
elixir, affordability) and every proposal is clamped to the legal mask.
It is deliberately a baseline labeler rather than a policy or an oracle.  The
from-scratch bootstrap uses it to produce an initial compatible checkpoint;
frontier audit uses it as one candidate branch for comparison.

Entity feature order follows ``observation_v2.ENTITY_TOKEN_FEATURES``;
hand-token order follows ``observation_v3.HAND_TOKEN_FEATURES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np


class TeacherError(ValueError):
    """Raised when no legal supervised target exists for a state."""


# observation_v2 entity columns.
_COL_SIDE = 1
_COL_X = 2
_COL_Y = 3
_COL_HP = 4
_COL_IS_AIR = 5
_COL_IS_BUILDING = 6

# observation_v3 hand-token columns.
_HAND_COST = 1
_HAND_AFFORDABLE = 3
_HAND_PLAYABLE = 4
_HAND_WIN_CONDITION = 5
_HAND_SPELL = 6
_HAND_BUILDING = 7
_HAND_AIR_ATTACKER = 9


@dataclass(frozen=True, slots=True)
class TeacherTarget:
    """One supervised decision target (all indices are integer labels)."""

    mode: int  # 0 = WAIT, 1 = PLAY
    card_slot: int
    row: int
    col: int
    wait_duration_idx: int  # index into model_v4.WAIT_DURATIONS


def _playable(hand_tokens: np.ndarray, card_mask: np.ndarray) -> np.ndarray:
    affordable = hand_tokens[:, _HAND_AFFORDABLE] > 0.5
    playable = hand_tokens[:, _HAND_PLAYABLE] > 0.5
    return affordable & playable & card_mask


def nearest_legal_cell(
    row: int, col: int, legal: np.ndarray, *, max_radius: int = 4
) -> tuple[int, int] | None:
    """Spiral-search the nearest legal cell (Chebyshev rings, row-major)."""

    rows, cols = legal.shape
    for radius in range(max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                candidate = (row + dr, col + dc)
                if 0 <= candidate[0] < rows and 0 <= candidate[1] < cols:
                    if bool(legal[candidate]):
                        return candidate
    return None


def soft_placement_target(
    row: int, col: int, legal: np.ndarray
) -> np.ndarray:
    """Spread mass uniformly over legal cells within Chebyshev distance 1.

    Near-equivalent placements share supervision instead of forcing one
    arbitrary cell; cells outside the neighborhood get zero mass.
    """

    rows, cols = legal.shape
    probs = np.zeros((rows, cols), dtype=np.float32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            candidate = (row + dr, col + dc)
            if 0 <= candidate[0] < rows and 0 <= candidate[1] < cols:
                if bool(legal[candidate]):
                    probs[candidate] = 1.0
    total = float(probs.sum())
    if total <= 0.0:
        raise TeacherError("no legal cell in the placement neighborhood")
    return probs / np.float32(total)


def _to_cell(y_norm: float, x_norm: float, rows: int, cols: int) -> tuple[int, int]:
    row = int(min(max(float(y_norm), 0.0), 1.0) * rows)
    col = int(min(max(float(x_norm), 0.0), 1.0) * cols)
    return (min(row, rows - 1), min(col, cols - 1))


def wait_duration_for_elixir(elixir: float) -> int:
    """Map elixir to a WAIT-duration index (longer waits when broke)."""

    if elixir < 3.0:
        return 3
    if elixir < 6.0:
        return 2
    if elixir < 8.0:
        return 1
    return 0


def teacher_label(
    *,
    hand_tokens: np.ndarray,
    entity_tokens: np.ndarray,
    entity_mask: np.ndarray,
    legal_play: np.ndarray,
    legal_wait: bool,
    own_elixir: float,
) -> TeacherTarget:
    """Produce the supervised target for one public state.

    Rule priority: air-aware defense of the most advanced threat, spell
    value on clusters, win-condition pressure when safe and rich, cheap
    cycling when floating elixir, otherwise WAIT with an elixir-aware
    duration.  Every PLAY proposal is resolved through the legal mask;
    when the preferred rule has no legal cell the teacher falls through
    to WAIT (or raises if even WAIT is illegal).
    """

    rows, cols = legal_play.shape[1], legal_play.shape[2]
    card_mask = legal_play.reshape(4, -1).any(axis=1)
    playable = _playable(hand_tokens, card_mask)
    enemies = entity_tokens[entity_mask & (entity_tokens[:, _COL_SIDE] > 0.5)]
    threats = enemies[enemies[:, _COL_Y] > 0.45] if enemies.size else enemies[:0]

    def play(slot: int, row: int, col: int) -> TeacherTarget | None:
        cell = nearest_legal_cell(row, col, np.asarray(legal_play[int(slot)]))
        if cell is None:
            return None
        return TeacherTarget(
            mode=1,
            card_slot=int(slot),
            row=int(cell[0]),
            col=int(cell[1]),
            wait_duration_idx=0,
        )

    # 1. Defense: answer the most advanced threat, preferring an
    #    air-attacker against air and a building/swarm otherwise.
    if len(threats) and bool(playable.any()):
        order = np.argsort(-threats[:, _COL_Y], kind="stable")
        target = threats[order[0]]
        need_air = bool(target[_COL_IS_AIR] > 0.5)
        candidates = np.flatnonzero(playable)
        if need_air:
            air_ok = candidates[hand_tokens[candidates, _HAND_AIR_ATTACKER] > 0.5]
            if len(air_ok):
                candidates = air_ok
        else:
            ground = candidates[
                (hand_tokens[candidates, _HAND_BUILDING] > 0.5)
                | (hand_tokens[candidates, _HAND_COST] <= 0.3)
            ]
            if len(ground):
                candidates = ground
        threat_cell = _to_cell(float(target[_COL_Y]), float(target[_COL_X]), rows, cols)
        # Kite/pull toward our side: two rows below the threat.
        anchor = (min(threat_cell[0] + 2, rows - 1), threat_cell[1])
        for slot in candidates:
            if hand_tokens[int(slot), _HAND_SPELL] > 0.5:
                continue
            decided = play(int(slot), *anchor)
            if decided is not None:
                return decided

    # 2. Spell value: a cluster of two or more enemies within a small box.
    if len(enemies) >= 2 and bool(playable.any()):
        positions = enemies[:, (_COL_Y, _COL_X)]
        spread = np.abs(positions[:, None, :] - positions[None, :, :]).max(axis=-1)
        close_counts = (spread < 0.12).sum(axis=1)
        clustered = bool(close_counts.max() >= 2)
        if clustered:
            # Aim at the densest local group.  Averaging all enemies lets a
            # distant third unit pull a valid two-unit spell target away from
            # the cluster that triggered this branch.
            cluster_index = int(np.argmax(close_counts))
            cluster = positions[spread[cluster_index] < 0.12]
            centroid = cluster.mean(axis=0)
            anchor = _to_cell(float(centroid[0]), float(centroid[1]), rows, cols)
            spells = np.flatnonzero(
                playable & (hand_tokens[:, _HAND_SPELL] > 0.5)
            )
            for slot in spells:
                decided = play(int(slot), *anchor)
                if decided is not None:
                    return decided

    # 3. Win-condition pressure when safe and rich.
    if not len(threats) and own_elixir >= 6.0 and bool(playable.any()):
        win = np.flatnonzero(
            playable & (hand_tokens[:, _HAND_WIN_CONDITION] > 0.5)
        )
        if len(win):
            left = int((enemies[:, _COL_X] < 0.5).sum()) if len(enemies) else 0
            right = int((enemies[:, _COL_X] >= 0.5).sum()) if len(enemies) else 0
            lane_col = 4 if left <= right else cols - 5
            for slot in win:
                decided = play(int(slot), rows // 2, lane_col)
                if decided is not None:
                    return decided

    # 4. Cheap cycle when floating elixir with nothing to answer.
    if not len(threats) and own_elixir >= 8.0 and bool(playable.any()):
        cheap = np.flatnonzero(
            playable & (hand_tokens[:, _HAND_COST] <= 0.2)
        )
        for slot in cheap:
            decided = play(int(slot), rows - 2, 2)
            if decided is not None:
                return decided

    # 5. WAIT with an elixir-aware duration.
    if not bool(legal_wait):
        raise TeacherError("teacher found no legal WAIT or PLAY action")
    return TeacherTarget(
        mode=0,
        card_slot=0,
        row=0,
        col=0,
        wait_duration_idx=wait_duration_for_elixir(float(own_elixir)),
    )


@dataclass(frozen=True, slots=True)
class TeacherSpendingContext:
    """Public evidence used by the opt-in spending teacher and its audit."""

    threats: int
    air_threats: int
    covered_threats: int
    urgent_threats: int


@lru_cache(maxsize=1)
def _public_card_tables() -> tuple[int, dict[int, dict[str, object]]]:
    # Entity tokens use the public metadata maximum, while hand tokens use
    # 127.  Those are separate namespaces; neither scale may be substituted.
    from cr_bot.domain.card_metadata import CARD_METADATA

    maximum = max(int(row["id"]) for row in CARD_METADATA.values())
    return maximum, {
        int(row["id"]): dict(row, name=name)
        for name, row in CARD_METADATA.items()
    }


def _entity_card(row: np.ndarray) -> dict[str, object]:
    maximum, cards = _public_card_tables()
    return cards.get(int(round(float(row[0]) * maximum)), {})


def _hand_card(row: np.ndarray) -> str | None:
    from .hand_features import V4_LEARNER_DECK

    number = int(round(float(row[0]) * 127))
    return next((card[0] for card in V4_LEARNER_DECK if card[1] == number), None)


def _spending_entities(
    tokens: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    live = (
        np.asarray(mask, dtype=bool)
        & (tokens[:, _COL_HP] > 0)
        & (tokens[:, 9] > 0.5)  # public visibility
        & (tokens[:, 10] > 0.5)  # public targetability
        & (tokens[:, 7] < 0.5)  # towers are not troop threats/defenders
        & (tokens[:, 8] < 0.5)  # spell effects are not troop bodies
    )
    return tokens[live & (tokens[:, _COL_SIDE] > 0.5)], tokens[
        live & (tokens[:, _COL_SIDE] < 0.5)
    ]


def _covered_threat_mask(threats: np.ndarray, allies: np.ndarray) -> np.ndarray:
    covered = np.zeros(len(threats), dtype=bool)
    for index, threat in enumerate(threats):
        # Quantized positions and missing target locks do not prove coverage.
        # Only an isolated, nonurgent ground body can use this conservative
        # heuristic; air and near-tower threats always retain a defense answer.
        if threat[_COL_Y] >= np.float32(0.70) or threat[_COL_IS_AIR] > 0.5:
            continue
        same_lane = np.abs(threats[:, _COL_X] - threat[_COL_X]) <= 3 / 18
        if int(same_lane.sum()) != 1:
            continue
        for ally in allies:
            card = _entity_card(ally)
            if card.get("name") not in ("musketeer", "cannon"):
                continue
            if not card.get("can_attack_ground") or ally[_COL_HP] < 0.4:
                continue
            dx = abs(float(ally[_COL_X] - threat[_COL_X])) * 18
            dy = abs(float(ally[_COL_Y] - threat[_COL_Y])) * 32
            if (
                dx <= 3
                and dx * dx + dy * dy <= 16
                and ally[_COL_Y] >= threat[_COL_Y] - 1 / 32
            ):
                covered[index] = True
                break
    return covered


def saving_teacher_context(
    entity_tokens: np.ndarray, entity_mask: np.ndarray
) -> TeacherSpendingContext:
    enemies, allies = _spending_entities(entity_tokens, entity_mask)
    threats = enemies[enemies[:, _COL_Y] > 0.45]
    covered = _covered_threat_mask(threats, allies)
    return TeacherSpendingContext(
        threats=len(threats),
        air_threats=int((threats[:, _COL_IS_AIR] > 0.5).sum()),
        covered_threats=int(covered.sum()),
        urgent_threats=int((threats[:, _COL_Y] >= np.float32(0.70)).sum()),
    )


def saving_teacher_label(
    *,
    hand_tokens: np.ndarray,
    entity_tokens: np.ndarray,
    entity_mask: np.ndarray,
    legal_play: np.ndarray,
    legal_wait: bool,
    own_elixir: float,
) -> TeacherTarget:
    """Opt-in public teacher for the catalog initializer experiment.

    Air answers and spell target classes are explicit.  Healthy nearby
    Musketeers/Cannons can suppress repeated isolated ground defense, with
    urgent and crowded threats bypassing that hypothesis.  Whole-match
    evidence must decide whether this variant is useful; it is not an oracle.
    The historical :func:`teacher_label` remains the comparison control.
    """

    rows, cols = legal_play.shape[1:]
    playable = _playable(hand_tokens, legal_play.reshape(4, -1).any(axis=1))
    names = [_hand_card(row) for row in hand_tokens]
    enemies, allies = _spending_entities(entity_tokens, entity_mask)
    threats = enemies[enemies[:, _COL_Y] > 0.45]
    uncovered = threats[~_covered_threat_mask(threats, allies)]

    def play(slot: int, row: int, col: int) -> TeacherTarget | None:
        cell = nearest_legal_cell(row, col, np.asarray(legal_play[slot]))
        return None if cell is None else TeacherTarget(1, slot, *cell, 0)

    def choose(card_names: tuple[str, ...], anchor: tuple[int, int]) -> TeacherTarget | None:
        for name in card_names:
            for slot in range(4):
                if names[slot] == name and playable[slot]:
                    target = play(slot, *anchor)
                    if target is not None:
                        return target
        return None

    def cluster_spell(group: np.ndarray, *, urgent: bool = False) -> TeacherTarget | None:
        if len(group) < 2:
            return None
        positions = group[:, (_COL_Y, _COL_X)]
        spread = np.abs(positions[:, None] - positions[None]).max(axis=-1)
        counts = (spread < 0.12).sum(axis=1)
        center = int(np.argmax(group[:, _COL_Y])) if urgent else int(np.argmax(counts))
        if counts[center] < 2:
            return None
        cluster = group[spread[center] < 0.12]
        anchor = _to_cell(*cluster[:, (_COL_Y, _COL_X)].mean(axis=0), rows, cols)
        # Log cannot damage air.  Prefer it for ground troop clusters; an
        # all-building cluster instead receives the general-purpose Fireball.
        if bool((cluster[:, _COL_IS_AIR] > 0.5).any()):
            return choose(("fireball",), anchor)
        priority = ("log", "fireball") if bool((cluster[:, _COL_IS_BUILDING] < 0.5).any()) else ("fireball",)
        return choose(priority, anchor)

    urgent = uncovered[uncovered[:, _COL_Y] >= np.float32(0.70)]
    # A remote swarm must not spend the elixir needed for a near-tower threat.
    # If urgency is present, only a cluster containing the foremost urgent
    # threat can take priority over that threat's individual defense answer.
    decided = cluster_spell(urgent, urgent=True) if len(urgent) else cluster_spell(uncovered)
    if decided is not None:
        return decided
    for threat in sorted(uncovered, key=lambda row: -float(row[_COL_Y])):
        cell = _to_cell(float(threat[_COL_Y]), float(threat[_COL_X]), rows, cols)
        anchor = (min(cell[0] + 2, rows - 1), cell[1])
        if threat[_COL_IS_AIR] > 0.5:
            decided = choose(("musketeer", "ice-spirit"), anchor)
            if decided is None:
                decided = choose(("fireball",), cell)
        else:
            card = _entity_card(threat)
            durable = float(card.get("hitpoints", 0)) * float(threat[_COL_HP]) >= 800
            order = (
                ("cannon", "musketeer", "ice-golem", "skeletons", "ice-spirit")
                if durable or card.get("targets_buildings_only")
                else ("skeletons", "ice-spirit", "ice-golem", "cannon", "musketeer")
            )
            decided = choose(order, anchor)
        if decided is not None:
            return decided

    if not len(uncovered):
        # Keep four elixir available for a useful Hog window.  Slot legality,
        # not a forced PLAY quota, determines whether pressure is possible.
        if own_elixir >= 4:
            left = int((enemies[:, _COL_X] < 0.5).sum())
            right = len(enemies) - left
            decided = choose(("hog-rider",), (rows // 2, 4 if left <= right else cols - 5))
            if decided is not None:
                return decided
        if own_elixir >= 8:
            decided = cluster_spell(enemies)
            if decided is None:
                decided = choose(("skeletons", "ice-spirit", "ice-golem"), (rows - 2, 2))
            if decided is not None:
                return decided
    if not legal_wait:
        raise TeacherError("saving teacher found no legal WAIT or PLAY action")
    return TeacherTarget(0, 0, 0, 0, wait_duration_for_elixir(float(own_elixir)))


@dataclass(frozen=True, slots=True)
class TeacherDefenseContext:
    """Conservative public threat counts for the opt-in defense teacher."""

    threats: int
    air_threats: int
    siege_threats: int
    covered_threats: int
    covered_air_threats: int
    urgent_threats: int


def _defense_threats(
    enemies: np.ndarray, entity_tokens: np.ndarray, entity_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Prefer live public tower rows when the upstream producer supplies them.
    # The current adapter omits tower cards from its card-ID namespace, so
    # the fixed public arena sites are a conservative fallback. This never
    # consults authoritative tower HP, target locks, or private state.
    own_towers = entity_tokens[
        np.asarray(entity_mask, dtype=bool)
        & (entity_tokens[:, _COL_SIDE] < 0.5)
        & (entity_tokens[:, _COL_HP] > 0)
        & (entity_tokens[:, 7] > 0.5)
        & (entity_tokens[:, 9] > 0.5)
        & (entity_tokens[:, 10] > 0.5)
    ]
    tower_points = (
        own_towers[:, (_COL_X, _COL_Y)] * np.asarray([18, 32])
        if len(own_towers)
        else np.asarray([(3.5, 25.5), (14.5, 25.5), (9.0, 28.5)])
    )
    siege = np.zeros(len(enemies), dtype=bool)
    for index, row in enumerate(enemies):
        card = _entity_card(row)
        if card.get("name") != "x-bow":
            continue
        point = row[[_COL_X, _COL_Y]] * np.asarray([18, 32])
        # X-Bow has 11.5-tile public range. Two tiles account for the two
        # visible building footprints in an edge-to-edge range test.
        reach = float(card.get("range", 0)) + 2
        siege[index] = bool((((tower_points - point) ** 2).sum(axis=1) <= reach ** 2).any())
    air = (enemies[:, _COL_IS_AIR] > 0.5) & (
        enemies[:, _COL_Y] >= np.float32(13 / 32)
    )
    selected = (enemies[:, _COL_Y] > 0.45) | air | siege
    return enemies[selected], siege[selected]


def _musketeer_in_range(threat: np.ndarray, allies: np.ndarray) -> bool:
    for ally in allies:
        if _entity_card(ally).get("name") != "musketeer" or ally[_COL_HP] < 0.45:
            continue
        dx = abs(float(ally[_COL_X] - threat[_COL_X])) * 18
        dy = abs(float(ally[_COL_Y] - threat[_COL_Y])) * 32
        if (
            dx <= 4
            and dx * dx + dy * dy <= 5.5 * 5.5
            and ally[_COL_Y] >= threat[_COL_Y] - 1 / 32
        ):
            return True
    return False


def _air_covered_mask(threats: np.ndarray, allies: np.ndarray) -> np.ndarray:
    """Cover only an isolated, nonurgent air body with sustained damage.

    Public positions cannot establish target locks or enough damage for a
    push. A healthy Musketeer already within a conservative shooting radius
    is evidence for one approaching body, not for a crowded aerial lane.
    """

    covered = np.zeros(len(threats), dtype=bool)
    air = threats[:, _COL_IS_AIR] > 0.5
    for index, threat in enumerate(threats):
        if not air[index] or threat[_COL_Y] >= np.float32(0.70):
            continue
        same_lane = np.abs(threats[:, _COL_X] - threat[_COL_X]) <= 4 / 18
        if int(same_lane.sum()) == 1:
            covered[index] = _musketeer_in_range(threat, allies)
    return covered


def defense_teacher_context(
    entity_tokens: np.ndarray, entity_mask: np.ndarray
) -> TeacherDefenseContext:
    """Describe only public entities; a siege count is not a target-lock claim."""

    enemies, allies = _spending_entities(entity_tokens, entity_mask)
    threats, siege = _defense_threats(enemies, entity_tokens, entity_mask)
    air_covered = _air_covered_mask(threats, allies)
    covered = _covered_threat_mask(threats, allies) | air_covered
    # A nearby body does not prove that a remote X-Bow has stopped shooting
    # a tower. Preserve the siege response without consulting private locks.
    covered[siege] = False
    return TeacherDefenseContext(
        threats=len(threats),
        air_threats=int((threats[:, _COL_IS_AIR] > 0.5).sum()),
        siege_threats=int(siege.sum()),
        covered_threats=int(covered.sum()),
        covered_air_threats=int(air_covered.sum()),
        urgent_threats=int((threats[:, _COL_Y] >= np.float32(0.70)).sum()),
    )


def defense_teacher_label(
    *,
    hand_tokens: np.ndarray,
    entity_tokens: np.ndarray,
    entity_mask: np.ndarray,
    legal_play: np.ndarray,
    legal_wait: bool,
    own_elixir: float,
    _air_cycle_rescue: bool = False,
) -> TeacherTarget:
    """Opt-in public defense teacher for the targeted repair experiment.

    Recognize bridge X-Bow geometrically, deploy sustained aerial defense
    before the river, and avoid repeatedly paying for an already covered
    isolated air body. The previous teachers remain unchanged controls.
    These rules are candidates to evaluate in complete matches, not an oracle.
    """

    rows, cols = legal_play.shape[1:]
    playable = _playable(hand_tokens, legal_play.reshape(4, -1).any(axis=1))
    names = [_hand_card(row) for row in hand_tokens]
    enemies, allies = _spending_entities(entity_tokens, entity_mask)
    threats, siege = _defense_threats(enemies, entity_tokens, entity_mask)
    air_covered = _air_covered_mask(threats, allies)
    covered = _covered_threat_mask(threats, allies) | air_covered
    covered[siege] = False

    def play(slot: int, anchor: tuple[int, int]) -> TeacherTarget | None:
        cell = nearest_legal_cell(*anchor, np.asarray(legal_play[slot]))
        return None if cell is None else TeacherTarget(1, slot, *cell, 0)

    def choose(card_names: tuple[str, ...], anchor: tuple[int, int]) -> TeacherTarget | None:
        for name in card_names:
            for slot in range(4):
                if names[slot] == name and playable[slot]:
                    target = play(slot, anchor)
                    if target is not None:
                        return target
        return None

    def wait() -> TeacherTarget:
        if not legal_wait:
            raise TeacherError("defense teacher found no legal WAIT or PLAY action")
        # A short revisit is useful while a visible threat is approaching;
        # an unaffordable four-elixir answer must not be delayed by 2s WAITs.
        return TeacherTarget(0, 0, 0, 0, 0 if len(threats) else wait_duration_for_elixir(float(own_elixir)))

    def ranged_anchor(threat: np.ndarray) -> tuple[int, int]:
        row, col = _to_cell(float(threat[_COL_Y]), float(threat[_COL_X]), rows, cols)
        inward = 3 if col < cols // 2 else -3
        offset_col = (
            min(col + inward, cols // 2 - 1)
            if col < cols // 2
            else max(col + inward, cols // 2)
        )
        return min(max(row + 3, 18), 26), min(max(offset_col, 0), cols - 1)

    def pull_anchor(threat: np.ndarray) -> tuple[int, int]:
        row, col = _to_cell(float(threat[_COL_Y]), float(threat[_COL_X]), rows, cols)
        # Central placement lengthens the path and lets both towers help;
        # after a threat has passed that anchor, keep the building in reach.
        if row >= 21:
            # Once the body has passed the usual pull, an overly central
            # Cannon may be farther away than the Princess Tower. Move the
            # anchor toward this advanced body's lane before legal clamping.
            return min(row + 1, 23), col + 1 if col < cols // 2 else col - 1
        return 20, 7 if col < cols // 2 else cols - 8

    def air_answer(threat: np.ndarray) -> TeacherTarget | None:
        cell = _to_cell(float(threat[_COL_Y]), float(threat[_COL_X]), rows, cols)
        positions = enemies[:, (_COL_X, _COL_Y)] * np.asarray([18, 32])
        point = threat[[_COL_X, _COL_Y]] * np.asarray([18, 32])
        nearby = enemies[((positions - point) ** 2).sum(axis=1) <= 2.4 ** 2]
        fragile_air = sum(
            row[_COL_IS_AIR] > 0.5
            and _entity_card(row).get("name") in ("bats", "minions", "minion-horde")
            for row in nearby
        )
        if fragile_air >= 2:
            centroid = nearby[:, (_COL_Y, _COL_X)].mean(axis=0)
            decided = choose(("fireball",), _to_cell(*centroid, rows, cols))
            if decided is not None:
                return decided
        # Near-tower supplemental damage is more useful than another delayed
        # Musketeer if a healthy same-lane Musketeer is already in range.
        if threat[_COL_Y] >= np.float32(0.70) and _musketeer_in_range(threat, allies):
            decided = choose(("fireball",), cell)
            if decided is not None:
                return decided
            if _entity_card(threat).get("targets_buildings_only"):
                # A central pull can buy more time for the existing ranged
                # defender, but never replaces the first sustained air answer.
                decided = choose(("cannon",), pull_anchor(threat))
                if decided is not None:
                    return decided
        decided = choose(("musketeer",), ranged_anchor(threat))
        if decided is not None:
            return decided
        # Fireball offers actual damage; Ice Spirit is one brief stall.
        # A lone nonurgent tank should wait for a sustained answer instead
        # of spending its last four elixir on one low-value Fireball.
        if threat[_COL_Y] >= np.float32(0.60) or len(nearby) >= 2:
            decided = choose(("fireball",), cell)
            if decided is not None:
                return decided
        return choose(("ice-spirit",), (max(17, min(cell[0] + 1, 26)), cell[1]))

    def siege_answer(threat: np.ndarray) -> TeacherTarget | None:
        cell = _to_cell(float(threat[_COL_Y]), float(threat[_COL_X]), rows, cols)
        if threat[_COL_HP] <= 0.40:
            decided = choose(("fireball",), cell)
            if decided is not None:
                return decided
        # Same-lane bridge pressure attacks the visible building directly.
        decided = choose(("hog-rider",), (17, cell[1]))
        if decided is not None:
            return decided
        # A ranged defender can damage an already tower-locked X-Bow;
        # a new ground distraction alone cannot prove that it will retarget.
        decided = choose(("musketeer", "cannon"), (18, cell[1]))
        if decided is not None:
            return decided
        decided = choose(("fireball",), cell)
        if decided is not None:
            return decided
        return choose(("ice-golem", "skeletons"), (17, cell[1]))

    unanswered = np.flatnonzero(~covered)
    # Tower-adjacent threats precede remote siege; among other requests an
    # X-Bow has priority because the old forward-only rule omitted it.
    priority = sorted(
        unanswered,
        key=lambda index: (
            0 if threats[index, _COL_Y] >= np.float32(0.70) else 1 if siege[index] else 2,
            -float(threats[index, _COL_Y]),
            int(index),
        ),
    )
    for index in priority:
        threat = threats[index]
        if threat[_COL_IS_AIR] > 0.5:
            decided = air_answer(threat)
        elif siege[index]:
            decided = siege_answer(threat)
        else:
            card = _entity_card(threat)
            if card.get("targets_buildings_only"):
                decided = choose(("cannon",), pull_anchor(threat))
            else:
                decided = None
            if decided is None:
                # Preserve ground clusters and the existing ground choices,
                # excluding the air/siege bodies handled by this variant.
                ground_mask = np.array(entity_mask, dtype=bool, copy=True)
                for body, is_siege in zip(threats, siege, strict=True):
                    if body[_COL_IS_AIR] > 0.5 or is_siege:
                        ground_mask[np.all(entity_tokens == body, axis=1)] = False
                decided = saving_teacher_label(
                    hand_tokens=hand_tokens,
                    entity_tokens=entity_tokens,
                    entity_mask=ground_mask,
                    legal_play=legal_play,
                    legal_wait=True,
                    own_elixir=own_elixir,
                )
                if decided.mode == 0 or names[decided.card_slot] == "hog-rider":
                    decided = None
        if decided is not None:
            return decided
    if (
        _air_cycle_rescue
        and bool((threats[unanswered, _COL_IS_AIR] > 0.5).any())
        and not any(name in ("musketeer", "fireball", "ice-spirit") for name in names)
    ):
        # V2 only: draw toward an aerial answer after ordinary defense has
        # failed. Ground cards here are cycle plays, not direct air damage.
        # Keep four elixir for the next answer and honor every public mask.
        for name in ("skeletons", "ice-golem", "log", "cannon", "hog-rider"):
            for slot in range(4):
                cost = round(float(hand_tokens[slot, _HAND_COST]) * 10)
                if names[slot] == name and playable[slot] and own_elixir >= cost + 4:
                    decided = play(slot, (rows - 3, 5))
                    if decided is not None:
                        return decided
    if len(unanswered):
        return wait()
    if bool(air_covered.any()) and own_elixir < 7:
        return wait()
    # Outside the focused cases, retain the existing saving teacher exactly.
    # Exclude air bodies whose defense is already evidenced by a Musketeer.
    fallback_mask = np.array(entity_mask, dtype=bool, copy=True)
    for threat in threats[air_covered]:
        fallback_mask[np.all(entity_tokens == threat, axis=1)] = False
    return saving_teacher_label(
        hand_tokens=hand_tokens,
        entity_tokens=entity_tokens,
        entity_mask=fallback_mask,
        legal_play=legal_play,
        legal_wait=legal_wait,
        own_elixir=own_elixir,
    )


def cycle_teacher_label(
    *,
    hand_tokens: np.ndarray,
    entity_tokens: np.ndarray,
    entity_mask: np.ndarray,
    legal_play: np.ndarray,
    legal_wait: bool,
    own_elixir: float,
) -> TeacherTarget:
    """Opt-in V2 defense teacher with a four-elixir air-cycle reserve.

    Preserve ordinary defense priority. Only when it finds no answer to a
    visible air threat, and the public hand has no Musketeer, Fireball or
    Ice Spirit, cycle the cheapest legal ground card while keeping four
    elixir. This candidate still needs complete-match evaluation; the V1
    :func:`defense_teacher_label` keeps its previous behavior by default.
    """

    return defense_teacher_label(
        hand_tokens=hand_tokens,
        entity_tokens=entity_tokens,
        entity_mask=entity_mask,
        legal_play=legal_play,
        legal_wait=legal_wait,
        own_elixir=own_elixir,
        _air_cycle_rescue=True,
    )


__all__ = [
    "TeacherError",
    "TeacherTarget",
    "nearest_legal_cell",
    "soft_placement_target",
    "teacher_label",
    "TeacherSpendingContext",
    "TeacherDefenseContext",
    "defense_teacher_context",
    "defense_teacher_label",
    "cycle_teacher_label",
    "saving_teacher_context",
    "saving_teacher_label",
    "wait_duration_for_elixir",
]
