"""Structure-of-arrays state used by the policy observation hot path.

The authoritative simulator remains the readable :class:`BattleState` object
graph.  This module provides a reusable, numeric column store for the fields
needed by the policy projection.  It is deliberately a sidecar rather than a
second rules implementation: the sidecar is rebuilt from authoritative state
and the existing observation path remains available for parity tests and
callers that do not opt into it.

Keeping this boundary narrow is important.  Entity physics still needs the
full object graph for now; moving it into the same arrays is the next step
before introducing a JIT compiler.  The policy-side arrays already remove the
temporary ``GameState``/``Detection`` object construction from the common
observation path and provide a stable SoA layout for later batched stepping.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np

from cr_bot.domain.card_metadata import CARD_METADATA

from .fixed import POSITION_SCALE, distance_mtile
from .geometry import (
    GRID_COLS,
    GRID_ROWS,
    cell_center_mtile,
    is_basic_deploy_cell,
    is_ground_cell,
    position_to_cell,
)
from .ruleset import CardDefinition, Ruleset, normalize_identifier
from .state import BattleState


FeatureCardName = Callable[[str], str]

_HIDDEN_STATUS_KINDS = frozenset(
    {
        "concealed",
        "concealment",
        "burrow",
        "burrowed",
        "hidden",
        "invisible",
        "invisibility",
        "stealth",
    }
)


def is_public_observation_entity(entity: object) -> bool:
    """Return whether an entity may be rendered in the public observation.

    Visibility is deliberately fail-closed for both sides.  The public V1
    contract has no provenance bit that distinguishes an actor's known hidden
    unit from an opponent's hidden unit, so all active stealth/burrow/
    concealment forms use the same omission rule across reference, SoA, and
    V2 projections.  Card-play events are handled separately by
    :class:`ObservationMemory` and are not affected by this predicate.
    """

    kind = getattr(entity, "kind", None)
    if kind not in {"troop", "building"}:
        return False
    if not bool(getattr(entity, "alive", False)) or int(getattr(entity, "hp", 0)) <= 0:
        return False
    if bool(getattr(entity, "stealth_active", False)):
        return False
    if bool(getattr(entity, "burrow_active", False)):
        return False
    if bool(getattr(entity, "concealed_active", False)):
        return False
    for status in getattr(entity, "statuses", ()):
        status_kind = getattr(status, "kind", None)
        remaining_us = getattr(status, "remaining_us", 0)
        if (
            isinstance(status_kind, str)
            and type(remaining_us) is int
            and remaining_us > 0
            and normalize_identifier(status_kind) in _HIDDEN_STATUS_KINDS
        ):
            return False
    return True

_POLICY_GRID_CELLS: tuple[tuple[int, int], ...] = tuple(
    (col, row)
    for row in range(GRID_ROWS)
    for col in range(GRID_COLS)
)
_GROUND_CELLS: tuple[tuple[int, int], ...] = tuple(
    cell for cell in _POLICY_GRID_CELLS if is_ground_cell(cell)
)
_BASIC_DEPLOY_CELLS: tuple[frozenset[tuple[int, int]], ...] = tuple(
    frozenset(cell for cell in _GROUND_CELLS if is_basic_deploy_cell(player, cell))
    for player in (0, 1)
)
_BASIC_DEPLOY_CANDIDATES: tuple[tuple[tuple[int, int], ...], ...] = tuple(
    tuple(cell for cell in _POLICY_GRID_CELLS if cell in _BASIC_DEPLOY_CELLS[player])
    for player in (0, 1)
)
_RESTRICTED_SPELL_CELLS: tuple[tuple[tuple[int, int], ...], ...] = tuple(
    tuple(
        (col, row)
        for col, row in _POLICY_GRID_CELLS
        if (row >= 17 if player == 0 else row <= 14)
    )
    for player in (0, 1)
)


@lru_cache(maxsize=512)
def _blocked_cells(
    obstacles: tuple[tuple[int, int, int], ...],
    radius: int,
) -> frozenset[tuple[int, int]]:
    if not obstacles:
        return frozenset()
    blocked: set[tuple[int, int]] = set()
    offset = POSITION_SCALE // 2
    for col, row in _POLICY_GRID_CELLS:
        x = col * POSITION_SCALE + offset
        y = row * POSITION_SCALE + offset
        for obstacle_x, obstacle_y, obstacle_radius in obstacles:
            if distance_mtile(x, y, obstacle_x, obstacle_y) < radius + obstacle_radius:
                blocked.add((col, row))
                break
    return frozenset(blocked)


@lru_cache(maxsize=4)
def _building_cells(player: int) -> tuple[tuple[int, int], ...]:
    allowed = _BASIC_DEPLOY_CELLS[player]
    return tuple(
        (col, row)
        for col, row in _POLICY_GRID_CELLS
        if all(
            (col + dcol, row + drow) in allowed
            for drow in range(-1, 2)
            for dcol in range(-1, 2)
        )
    )


@lru_cache(maxsize=512)
def _cells_to_mask(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Convert one immutable world-cell tuple into a read-only grid mask."""

    mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    for col, row in cells:
        mask[row, col] = True
    mask.setflags(write=False)
    return mask


@lru_cache(maxsize=512)
def _threat_weight(card_name: str) -> float:
    metadata = CARD_METADATA[card_name]
    damage = metadata.get("damage")
    hit_speed = metadata.get("hit_speed")
    if (
        type(damage) not in (int, float)
        or type(hit_speed) not in (int, float)
        or damage <= 0
        or hit_speed <= 0
    ):
        return 0.0
    dps = float(damage) / max(float(hit_speed), 0.1)
    return min(dps / 1_000.0, 5.0)


def _ruleset_threat_weight(
    card_id: str,
    card_name: str,
    ruleset: Ruleset,
) -> float:
    """Return threat from the active card definition, with legacy fallback.

    The external vision metadata is a Level-16 catalog.  Simulator
    observations must instead use the loaded ruleset's damage and attack
    cadence, otherwise a policy sees a threat value for a different level.
    Minimal fake rulesets used by adapter tests do not carry those fields, so
    they retain the metadata-compatible behavior.
    """

    cards = getattr(ruleset, "cards", None)
    definition = cards.get(card_id) if hasattr(cards, "get") else None
    damage = getattr(definition, "damage", None)
    attack_interval_us = getattr(definition, "attack_interval_us", None)
    if (
        type(damage) not in (int, float)
        or type(attack_interval_us) not in (int, float)
        or damage <= 0
        or attack_interval_us <= 0
    ):
        return _threat_weight(card_name)
    dps = float(damage) * 1_000_000.0 / float(attack_interval_us)
    return min(dps / 1_000.0, 5.0)


@lru_cache(maxsize=512)
def _is_air_card(card_name: str) -> bool:
    return bool(CARD_METADATA[card_name].get("is_air"))


class ObservationSoA:
    """Reusable structure-of-arrays projection for one simulator lane.

    The arrays are capacity-backed so a normal observation does not allocate a
    new Python object for every living entity.  Variable-length state such as
    statuses remains in the authoritative state and is intentionally not
    copied: it is not part of the current public policy observation.
    """

    def __init__(self, capacity: int = 32) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self.count = 0
        self.uids = np.empty(capacity, dtype=np.int64)
        self.owners = np.empty(capacity, dtype=np.int8)
        self.x_mtile = np.empty(capacity, dtype=np.int64)
        self.y_mtile = np.empty(capacity, dtype=np.int64)
        self.hp = np.empty(capacity, dtype=np.int64)
        self.max_hp = np.empty(capacity, dtype=np.int64)
        self.cell_cols = np.empty(capacity, dtype=np.int16)
        self.cell_rows = np.empty(capacity, dtype=np.int16)
        self.hp_fraction = np.empty(capacity, dtype=np.float64)
        self.threat = np.empty(capacity, dtype=np.float64)
        self.is_air = np.zeros(capacity, dtype=bool)
        self.renderable = np.zeros(capacity, dtype=bool)
        self.card_names: list[str | None] = [None] * capacity

        # Tower data is kept in canonical owner/site order.  Sites are
        # left/king/right; the viewer transform is applied during projection.
        self.tower_hp = np.zeros((2, 3), dtype=np.int64)
        self.tower_alive = np.zeros((2, 3), dtype=bool)
        self._static_card_cells_cache: dict[
            tuple[str, int, str, tuple[tuple[int, int, int], ...]],
            tuple[tuple[int, int], ...],
        ] = {}

    def _grow(self, required: int) -> None:
        if required <= self._capacity:
            return
        capacity = self._capacity
        while capacity < required:
            capacity *= 2

        for name, dtype in (
            ("uids", np.int64),
            ("owners", np.int8),
            ("x_mtile", np.int64),
            ("y_mtile", np.int64),
            ("hp", np.int64),
            ("max_hp", np.int64),
            ("cell_cols", np.int16),
            ("cell_rows", np.int16),
            ("hp_fraction", np.float64),
            ("threat", np.float64),
            ("is_air", bool),
            ("renderable", bool),
        ):
            old = getattr(self, name)
            new = np.empty(capacity, dtype=dtype)
            new[: self.count] = old[: self.count]
            setattr(self, name, new)
        self.card_names.extend([None] * (capacity - self._capacity))
        self._capacity = capacity

    def sync(
        self,
        state: BattleState,
        feature_card_name: FeatureCardName,
        ruleset: Ruleset | None = None,
    ) -> None:
        """Refresh columns from one authoritative state in stable UID order."""

        entity_uids = sorted(state.entities)
        self._grow(len(entity_uids))
        self.count = len(entity_uids)
        self.renderable[: self.count] = False
        self.is_air[: self.count] = False
        self.tower_hp.fill(0)
        self.tower_alive.fill(False)

        for index, uid in enumerate(entity_uids):
            entity = state.entities[uid]
            self.uids[index] = entity.uid
            self.owners[index] = entity.owner
            self.x_mtile[index] = entity.x_mtile
            self.y_mtile[index] = entity.y_mtile
            self.hp[index] = entity.hp
            self.max_hp[index] = entity.max_hp
            self.card_names[index] = None

            is_tower = entity.kind == "tower"
            if is_tower:
                site = self._tower_site(entity.role, entity.card_id, entity.x_mtile)
                if entity.owner in (0, 1):
                    self.tower_hp[entity.owner, site] = max(0, entity.hp)
                    self.tower_alive[entity.owner, site] = bool(
                        entity.alive and entity.hp > 0
                    )
                continue
            if entity.kind == "spell" or not is_public_observation_entity(entity):
                continue

            # Match the existing observation adapter's failure behavior for a
            # live unsupported entity rather than silently dropping it.
            card_name = feature_card_name(entity.card_id)
            self.card_names[index] = card_name
            cell = position_to_cell(entity.x_mtile, entity.y_mtile)
            if cell is None:
                continue
            self.cell_cols[index], self.cell_rows[index] = cell
            self.hp_fraction[index] = (
                0.0
                if entity.max_hp <= 0
                else min(1.0, max(0.0, entity.hp / entity.max_hp))
            )
            threat_weight = (
                _ruleset_threat_weight(entity.card_id, card_name, ruleset)
                if ruleset is not None
                else _threat_weight(card_name)
            )
            self.threat[index] = self.hp_fraction[index] * threat_weight
            self.is_air[index] = _is_air_card(card_name)
            self.renderable[index] = True

    @staticmethod
    def _tower_site(role: str | None, card_id: str, x_mtile: int) -> int:
        if role == "king":
            return 1
        return 0 if x_mtile < GRID_COLS * 1_000 // 2 else 2

    def viewer_towers(self, viewer: int) -> tuple[np.ndarray, np.ndarray]:
        """Return tower HP/alive arrays in the viewer's left/king/right order."""

        if viewer not in (0, 1):
            raise ValueError("viewer must be 0 or 1")
        hp = np.zeros((2, 3), dtype=np.int64)
        alive = np.zeros((2, 3), dtype=bool)
        for owner in (0, 1):
            relative_team = 0 if owner == viewer else 1
            for canonical_site in (0, 1, 2):
                viewed_site = canonical_site
                if viewer == 1 and canonical_site != 1:
                    viewed_site = 2 - canonical_site
                hp[relative_team, viewed_site] = self.tower_hp[owner, canonical_site]
                alive[relative_team, viewed_site] = self.tower_alive[owner, canonical_site]
        return hp, alive

    def legal_action_cells_if_static(
        self,
        state: BattleState,
        ruleset: Ruleset,
        player: int,
    ) -> tuple[tuple[tuple[int, int], ...], ...] | None:
        """Return exact legality for the common static-territory case.

        The authoritative engine still owns the general legality function.
        This path is used only when every Princess/King Tower is alive and no
        live building has been deployed, so territory and obstacles are fixed
        apart from the card/deck/elixir columns.  A ``None`` result asks the
        caller to use the engine for states with dynamic pockets or building
        obstacles.
        """

        if type(player) is not int or player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if state.terminal:
            return ((), (), (), ())

        obstacles: list[tuple[int, int, int]] = []
        for uid in sorted(state.entities):
            entity = state.entities[uid]
            if entity.kind == "tower":
                if not entity.alive or entity.hp <= 0:
                    return None
                definition = ruleset.towers.get(entity.card_id)
                if definition is None:
                    return None
                obstacles.append(
                    (
                        entity.x_mtile,
                        entity.y_mtile,
                        int(definition.collision_radius_mtile),
                    )
                )
            elif entity.kind == "building" and entity.alive and entity.hp > 0:
                return None

        player_state = state.players[player]
        previous = player_state.last_played_card_id
        obstacle_key = tuple(obstacles)
        legal_by_slot: list[tuple[tuple[int, int], ...]] = []
        for raw_card_id in player_state.hand[:4]:
            card = ruleset.card(raw_card_id)
            placement_card = card
            if card.card_id == "mirror":
                if previous is None or previous == "mirror":
                    legal_by_slot.append(())
                    continue
                placement_card = ruleset.card(previous)
            effective_cost = card.elixir_milli
            if card.card_id == "mirror" and previous is not None:
                previous_card = ruleset.card(previous)
                effective_cost = min(
                    ruleset.match.max_elixir_milli,
                    previous_card.elixir_milli + 1_000,
                )
            if player_state.elixir_milli < effective_cost:
                legal_by_slot.append(())
                continue

            cache_key = (
                ruleset.content_hash,
                player,
                placement_card.card_id,
                obstacle_key,
            )
            candidates = self._static_card_cells_cache.get(cache_key)
            if candidates is None:
                candidates = self._build_static_card_cells(
                    placement_card,
                    player,
                    obstacle_key,
                )
                self._static_card_cells_cache[cache_key] = candidates
            legal_by_slot.append(candidates)

        while len(legal_by_slot) < 4:
            legal_by_slot.append(())
        return tuple(legal_by_slot)

    def legal_action_masks_if_static(
        self,
        state: BattleState,
        ruleset: Ruleset,
        player: int,
    ) -> tuple[np.ndarray, ...] | None:
        """Return cached world-grid legality masks for static states."""

        legal_cells = self.legal_action_cells_if_static(state, ruleset, player)
        if legal_cells is None:
            return None
        return tuple(_cells_to_mask(cells) for cells in legal_cells)

    def _build_static_card_cells(
        self,
        card: CardDefinition,
        player: int,
        obstacles: tuple[tuple[int, int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        """Build one static card row; callers cache the immutable result."""

        placement = card.mechanics.get("placement_class")
        if placement == "spell_anywhere":
            return _POLICY_GRID_CELLS
        if placement in {"restricted_spell", "own_ground_spell", "spells"}:
            return _RESTRICTED_SPELL_CELLS[player]
        if placement == "miner_anywhere":
            candidates = _GROUND_CELLS
        elif card.kind == "building":
            candidates = _building_cells(player)
        else:
            candidates = _BASIC_DEPLOY_CANDIDATES[player]

        if card.kind not in {"troop", "building"}:
            return candidates
        radius = int(card.collision_radius_mtile or 0)
        blocked = _blocked_cells(obstacles, radius)
        if not blocked:
            return candidates
        return tuple(cell for cell in candidates if cell not in blocked)

    @staticmethod
    def _effective_cost(
        card_id: str,
        previous_card_id: str | None,
        ruleset: Ruleset,
    ) -> int:
        card = ruleset.card(card_id)
        if card.card_id != "mirror" or previous_card_id is None:
            return card.elixir_milli
        previous = ruleset.card(previous_card_id)
        return min(ruleset.match.max_elixir_milli, previous.elixir_milli + 1_000)


__all__ = ["ObservationSoA"]
