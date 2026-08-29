"""Trusted producer for the public V2 entity-token rows.

The existing :mod:`simulator.observation` module is the public boundary for
authoritative simulator state.  This module composes that boundary with the
additive :class:`~simulator.observation_v2.PolicyObservationV2` container:

``BattleState`` -> sanitized public ``BattleState`` -> ``GameState`` -> V2

The sanitization step is deliberately conservative.  It is allowed to read
authoritative visibility flags only to *drop* a row; it never copies target
IDs, cooldowns, exact opponent resources, statuses, or other private fields
into a feature.  The row producer itself consumes only the public
``GameState`` detections returned by ``battle_state_to_observed_game_state``.

The public ``GameState`` is already viewer-local.  In particular, the
existing projection rotates positions and teams for viewer 1.  This adapter
does not mirror those coordinates a second time.  Tower-distance reference
points are transformed into the same viewer-local frame before distances are
computed.

Several V2 features have no representation in the current public detection
DTO (velocity, recent damage, temporal age, target state, and status flags).
The trusted producer encodes those fields as zero rather than recovering them
from ``BattleState``.  This makes the output honest and keeps the module
fail-closed until a future public observation contract exposes those facts.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math
from typing import Callable, Collection, Final

import numpy as np

from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.domain.game_state import GameState, Match
from cr_bot.features.action_space import ACTION_GRID

from .actions import PlayCardAction
from .geometry import (
    TOWER_SITES,
    mirror_position,
    position_to_cell,
)
from .observation import (
    LegalActionCellsCallback,
    ObservationMemory,
    PolicyObservationV1,
    battle_state_to_observed_game_state,
    build_policy_observation,
)
from .observation_v2 import (
    ENTITY_TOKEN_DIM,
    ENTITY_TOKEN_FEATURES,
    PolicyObservationV2,
)
from .ruleset import Ruleset, normalize_identifier
from .soa import ObservationSoA, is_public_observation_entity
from .state import BattleState, EntityState


_FEATURE_INDEX: Final = {name: index for index, name in enumerate(ENTITY_TOKEN_FEATURES)}
_MAX_CARD_ID: Final = max(
    int(metadata["id"])
    for metadata in CARD_METADATA.values()
    if isinstance(metadata, dict) and type(metadata.get("id")) is int
)
_UNIT_SQUARE_DIAGONAL: Final = math.sqrt(2.0)
# These are the same public feature aliases used by the existing V1
# projection.  They describe the visible form, not private simulator
# provenance, and are kept local so this additive module does not depend on a
# private symbol from observation.py.
_PUBLIC_CARD_ALIASES: Final = {
    "bush-goblin": "goblins",
    "barbarian": "barbarians",
    "cursed-hog": "hog-rider",
    "goblin": "goblins",
    "goblin-brawler": "goblins",
    "phoenix-egg": "phoenix",
    "golemite": "golem",
    "elixir-golemite": "elixir-golem",
    "elixir-blob": "elixir-golem",
    "lava-pup": "lava-hound",
    "rascal-boy": "barbarians",
    "rascal-girl": "spear-goblins",
    "spear-goblin": "spear-goblins",
    "cannon-cart-building": "cannon-cart",
}


def build_policy_observation_v2(
    state: BattleState,
    ruleset: Ruleset,
    *,
    viewer: int = 0,
    memory: ObservationMemory | None = None,
    legality_callback: Callable[[BattleState, PlayCardAction], bool] | None = None,
    legal_action_cells_callback: LegalActionCellsCallback | None = None,
    soa_state: ObservationSoA | None = None,
    _soa_already_synced: bool = False,
) -> PolicyObservationV2:
    """Project a simulator state into the trusted public V2 contract.

    ``state`` is never mutated.  The V1 tensors and entity rows are built
    from a shallow state copy whose entity mapping excludes entities that the
    public contract cannot safely expose.  Player state and event history are
    retained so the existing public observation memory can continue to
    estimate opponent resources from public card-play events; the opponent's
    exact hand and elixir are never read by this adapter.

    ``memory`` is viewer-local in the same way as the V1 observation API.  A
    caller that evaluates a sequence should reuse it; a one-shot call may
    leave it as ``None``.
    """

    _validate_viewer(viewer)
    if not isinstance(state, BattleState):
        raise TypeError("state must be a BattleState")
    if not isinstance(ruleset, Ruleset):
        raise TypeError("ruleset must be a Ruleset")

    public_uids = frozenset(
        entity.uid
        for entity in state.entities.values()
        if _is_conservatively_public(entity)
    )
    public_entities = {
        uid: entity
        for uid, entity in state.entities.items()
        if uid in public_uids or entity.kind == "tower"
    }
    public_state = replace(state, entities=public_entities)

    if memory is None:
        memory = ObservationMemory(viewer=viewer)
    observation_v1 = build_policy_observation(
        public_state,
        ruleset,
        viewer=viewer,
        memory=memory,
        legality_callback=legality_callback,
        legal_action_cells_callback=legal_action_cells_callback,
        soa_state=soa_state,
        allow_unrepresented_hand=True,
        _soa_already_synced=_soa_already_synced,
    )
    if soa_state is None:
        projected = battle_state_to_observed_game_state(
            public_state,
            ruleset,
            viewer=viewer,
            memory=memory,
            allow_unrepresented_hand=True,
        )
        rows = build_public_entity_rows(
            projected,
            viewer=viewer,
            public_entity_uids=public_uids,
        )
    else:
        rows = build_public_entity_rows_from_soa(
            public_state,
            viewer=viewer,
            soa_state=soa_state,
            public_entity_uids=public_uids,
        )
    return PolicyObservationV2.from_v1(
        observation_v1,
        public_entity_rows=rows,
    )


def build_public_entity_rows_from_soa(
    public_state: BattleState,
    *,
    viewer: int,
    soa_state: ObservationSoA,
    public_entity_uids: Collection[int],
) -> np.ndarray:
    """Build V2 rows directly from the already-synchronized SoA columns."""

    _validate_viewer(viewer)
    if not isinstance(public_state, BattleState):
        raise TypeError("public_state must be a BattleState")
    if not isinstance(soa_state, ObservationSoA):
        raise TypeError("soa_state must be an ObservationSoA")
    allowed_uids = _normalize_uid_allow_list(public_entity_uids)
    tower_points = _viewer_local_tower_points(viewer)
    rows: list[tuple[int, np.ndarray]] = []

    for index in range(soa_state.count):
        uid = int(soa_state.uids[index])
        if uid not in allowed_uids:
            continue
        entity = public_state.entities.get(uid)
        card_name = soa_state.card_names[index]
        if entity is None or card_name is None:
            continue
        metadata = _public_card_metadata(card_name)
        if metadata is None:
            continue

        x_mtile = int(soa_state.x_mtile[index])
        y_mtile = int(soa_state.y_mtile[index])
        if viewer == 1:
            x_mtile, y_mtile = mirror_position(x_mtile, y_mtile)
        cell = position_to_cell(x_mtile, y_mtile)
        if cell is None:
            continue
        raw_x, raw_y = ACTION_GRID.cell_to_norm_center(*cell)
        x = _canonicalize_grid_coordinate(raw_x, ACTION_GRID.x0, ACTION_GRID.width)
        y = _canonicalize_grid_coordinate(raw_y, ACTION_GRID.y0, ACTION_GRID.height)

        row = np.zeros((ENTITY_TOKEN_DIM,), dtype=np.float32)
        row[_FEATURE_INDEX["card_id"]] = np.float32(
            float(metadata["id"]) / float(_MAX_CARD_ID)
        )
        row[_FEATURE_INDEX["side"]] = np.float32(entity.owner != viewer)
        row[_FEATURE_INDEX["x"]] = np.float32(x)
        row[_FEATURE_INDEX["y"]] = np.float32(y)
        row[_FEATURE_INDEX["hp_fraction"]] = np.float32(
            _clip_unit(float(soa_state.hp_fraction[index]))
        )
        row[_FEATURE_INDEX["is_air"]] = np.float32(bool(metadata.get("is_air")))
        kind = str(metadata.get("kind") or "")
        row[_FEATURE_INDEX["is_building"]] = np.float32(kind == "building")
        row[_FEATURE_INDEX["is_tower"]] = np.float32(kind == "tower")
        row[_FEATURE_INDEX["is_spell"]] = np.float32(kind == "spell")
        row[_FEATURE_INDEX["is_visible"]] = 1.0
        row[_FEATURE_INDEX["is_targetable"]] = 1.0
        row[_FEATURE_INDEX["lane"]] = np.float32(_lane_value(x))
        row[_FEATURE_INDEX["distance_to_own_tower"]] = np.float32(
            _nearest_distance(x, y, tower_points["own"])
        )
        row[_FEATURE_INDEX["distance_to_enemy_tower"]] = np.float32(
            _nearest_distance(x, y, tower_points["enemy"])
        )
        row[_FEATURE_INDEX["distance_to_own_king"]] = np.float32(
            _distance(x, y, tower_points["own_king"])
        )
        row[_FEATURE_INDEX["distance_to_enemy_king"]] = np.float32(
            _distance(x, y, tower_points["enemy_king"])
        )
        row[_FEATURE_INDEX["confidence"]] = 1.0
        rows.append((uid, row))

    if len(rows) > 128:
        raise ValueError("public entity rows exceed the V2 NMAX=128 bound")
    if not rows:
        return np.zeros((0, ENTITY_TOKEN_DIM), dtype=np.float32)
    rows.sort(key=lambda item: item[0])
    return np.ascontiguousarray(np.stack([row for _, row in rows], axis=0), dtype=np.float32)


def build_public_entity_rows(
    public_state: GameState,
    *,
    viewer: int = 0,
    public_entity_uids: Collection[int] | None = None,
) -> np.ndarray:
    """Build normalized rows from a viewer-local public ``GameState``.

    ``public_state`` must already be in viewer-local coordinates, as returned
    by ``battle_state_to_observed_game_state`` or by the live visual
    observation boundary.  ``public_entity_uids`` is an optional allow-list
    used by the simulator adapter after its conservative visibility gate.
    When omitted, every detection in the public DTO is considered eligible;
    no authoritative state is consulted by this function.

    Rows are deterministic and bounded by the V2 ``NMAX`` limit.  Malformed
    or unknown public detections are omitted rather than converted using
    guessed simulator metadata.
    """

    _validate_viewer(viewer)
    if not isinstance(public_state, GameState):
        raise TypeError("public_state must be a GameState")
    allowed_uids = None if public_entity_uids is None else _normalize_uid_allow_list(public_entity_uids)

    matches: list[Match] = [*public_state.own_units, *public_state.enemy_units]
    rows_with_order: list[tuple[tuple[int, int, str], np.ndarray]] = []
    seen_uids: set[int] = set()
    for index, match in enumerate(matches):
        detection = getattr(match, "troop", None)
        if detection is None:
            continue
        track_id = getattr(detection, "track_id", None)
        if allowed_uids is not None:
            if type(track_id) is not int or track_id not in allowed_uids:
                continue
            # A duplicate track is not a second entity.  Dropping it avoids
            # making row order depend on a malformed public projection.
            if track_id in seen_uids:
                continue
            seen_uids.add(track_id)
        row = _row_from_public_detection(detection, viewer=viewer)
        if row is None:
            continue
        sort_key = (
            0 if type(track_id) is int else 1,
            track_id if type(track_id) is int else index,
            str(getattr(detection, "class_name", "")),
        )
        rows_with_order.append((sort_key, row))

    rows_with_order.sort(key=lambda item: item[0])
    if len(rows_with_order) > 128:
        raise ValueError("public entity rows exceed the V2 NMAX=128 bound")
    if not rows_with_order:
        return np.zeros((0, ENTITY_TOKEN_DIM), dtype=np.float32)
    return np.ascontiguousarray(
        np.stack([row for _, row in rows_with_order], axis=0),
        dtype=np.float32,
    )


def public_entity_rows_from_game_state(
    public_state: GameState,
    *,
    viewer: int = 0,
    public_entity_uids: Collection[int] | None = None,
) -> np.ndarray:
    """Descriptive alias for :func:`build_public_entity_rows`."""

    return build_public_entity_rows(
        public_state,
        viewer=viewer,
        public_entity_uids=public_entity_uids,
    )


def _is_conservatively_public(entity: EntityState) -> bool:
    """Return whether an entity is safe to pass through the public boundary.

    Visibility flags are used only as a negative filter.  This intentionally
    drops flagged entities for both teams: the public DTO has no explicit
    visibility provenance, and retaining an own-side hidden entity would make
    the same row contract unsafe for an opponent-side projection.
    """

    return is_public_observation_entity(entity)


def _row_from_public_detection(detection: object, *, viewer: int) -> np.ndarray | None:
    class_name = getattr(detection, "class_name", None)
    team = getattr(detection, "team", None)
    if not isinstance(class_name, str) or not isinstance(team, str):
        return None
    relative_team = team.strip().casefold()
    if relative_team not in {"ally", "enemy"}:
        return None

    metadata = _public_card_metadata(class_name)
    if metadata is None:
        return None
    x = _finite_unit_value(getattr(detection, "center_x", None))
    y = _finite_unit_value(getattr(detection, "center_y", None))
    confidence = _finite_unit_value(getattr(detection, "confidence", None))
    if x is None or y is None or confidence is None:
        return None
    hp_fraction_raw = getattr(detection, "estimated_hp", None)
    hp_fraction = (
        0.0
        if hp_fraction_raw is None
        else _finite_unit_value(hp_fraction_raw)
    )
    if hp_fraction is None:
        return None

    # ``Detection`` coordinates use the legacy action-grid normalization,
    # whose bounds include a small crop margin outside [0, 1].  V2 features
    # use canonical arena normalization so a viewer-1 rotation is exactly
    # ``x -> 1-x, y -> 1-y`` rather than inheriting that compatibility
    # padding.
    x = _canonicalize_grid_coordinate(x, ACTION_GRID.x0, ACTION_GRID.width)
    y = _canonicalize_grid_coordinate(y, ACTION_GRID.y0, ACTION_GRID.height)
    confidence = _clip_unit(confidence)
    hp_fraction = _clip_unit(hp_fraction)
    tower_points = _viewer_local_tower_points(viewer)

    row = np.zeros((ENTITY_TOKEN_DIM,), dtype=np.float32)
    row[_FEATURE_INDEX["card_id"]] = np.float32(
        float(metadata["id"]) / float(_MAX_CARD_ID)
    )
    row[_FEATURE_INDEX["side"]] = np.float32(relative_team == "enemy")
    row[_FEATURE_INDEX["x"]] = np.float32(x)
    row[_FEATURE_INDEX["y"]] = np.float32(y)
    row[_FEATURE_INDEX["hp_fraction"]] = np.float32(hp_fraction)
    row[_FEATURE_INDEX["is_air"]] = np.float32(bool(metadata.get("is_air")))
    kind = str(metadata.get("kind") or "")
    row[_FEATURE_INDEX["is_building"]] = np.float32(kind == "building")
    row[_FEATURE_INDEX["is_tower"]] = np.float32(kind == "tower")
    row[_FEATURE_INDEX["is_spell"]] = np.float32(kind == "spell")
    # A row is admitted only after the conservative visibility gate.  The
    # current public DTO does not expose targetability separately, so this is
    # the public visibility/targetability approximation, not a private engine
    # target flag.
    row[_FEATURE_INDEX["is_visible"]] = 1.0
    row[_FEATURE_INDEX["is_targetable"]] = 1.0
    row[_FEATURE_INDEX["lane"]] = np.float32(_lane_value(x))
    row[_FEATURE_INDEX["distance_to_own_tower"]] = np.float32(
        _nearest_distance(x, y, tower_points["own"])
    )
    row[_FEATURE_INDEX["distance_to_enemy_tower"]] = np.float32(
        _nearest_distance(x, y, tower_points["enemy"])
    )
    row[_FEATURE_INDEX["distance_to_own_king"]] = np.float32(
        _distance(x, y, tower_points["own_king"])
    )
    row[_FEATURE_INDEX["distance_to_enemy_king"]] = np.float32(
        _distance(x, y, tower_points["enemy_king"])
    )
    row[_FEATURE_INDEX["confidence"]] = np.float32(confidence)
    # The remaining fields are intentionally zero.  They require a public
    # temporal/status contract that the current GameState DTO does not carry.
    return row


@lru_cache(maxsize=512)
def _public_card_metadata(card_name: str) -> dict[str, object] | None:
    normalized = normalize_identifier(card_name)
    key = _PUBLIC_CARD_ALIASES.get(normalized, normalized)
    if key == "goblin-gang-goblin":
        key = "goblins"
    metadata = CARD_METADATA.get(key)
    if not isinstance(metadata, dict) or type(metadata.get("id")) is not int:
        return None
    return metadata


@lru_cache(maxsize=2)
def _viewer_local_tower_points(viewer: int) -> dict[str, tuple[tuple[float, float], ...] | tuple[float, float]]:
    points: dict[str, list[tuple[float, float]]] = {
        "own": [],
        "enemy": [],
    }
    kings: dict[str, tuple[float, float]] = {}
    for site in TOWER_SITES:
        x_mtile, y_mtile = site.x_mtile, site.y_mtile
        if viewer == 1:
            x_mtile, y_mtile = mirror_position(x_mtile, y_mtile)
        cell = position_to_cell(x_mtile, y_mtile)
        if cell is None:
            continue
        raw_point = ACTION_GRID.cell_to_norm_center(*cell)
        point = (
            _canonicalize_grid_coordinate(raw_point[0], ACTION_GRID.x0, ACTION_GRID.width),
            _canonicalize_grid_coordinate(raw_point[1], ACTION_GRID.y0, ACTION_GRID.height),
        )
        team = "own" if site.owner == viewer else "enemy"
        points[team].append(point)
        if site.role == "king":
            kings[team] = point
    return {
        "own": tuple(points["own"]),
        "enemy": tuple(points["enemy"]),
        "own_king": kings["own"],
        "enemy_king": kings["enemy"],
    }


def _nearest_distance(x: float, y: float, points: tuple[tuple[float, float], ...]) -> float:
    if not points:
        return 1.0
    return min(_distance(x, y, point) for point in points)


def _distance(x: float, y: float, point: tuple[float, float]) -> float:
    return _clip_unit(math.hypot(x - point[0], y - point[1]) / _UNIT_SQUARE_DIAGONAL)


def _lane_value(x: float) -> float:
    if x < (1.0 / 3.0):
        return 0.0
    if x < (2.0 / 3.0):
        return 0.5
    return 1.0


def _finite_unit_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _clip_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _canonicalize_grid_coordinate(value: float, origin: float, extent: float) -> float:
    return _clip_unit((float(value) - origin) / extent)


def _normalize_uid_allow_list(values: Collection[int]) -> frozenset[int]:
    result: set[int] = set()
    for value in values:
        if type(value) is not int or value < 0:
            raise TypeError("public_entity_uids must contain non-negative integers")
        result.add(value)
    return frozenset(result)


def _validate_viewer(viewer: int) -> None:
    if type(viewer) is not int or viewer not in (0, 1):
        raise ValueError(f"viewer must be 0 or 1, got {viewer!r}")


__all__ = [
    "build_policy_observation_v2",
    "build_public_entity_rows",
    "public_entity_rows_from_game_state",
]
