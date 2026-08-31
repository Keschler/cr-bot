"""Small, fail-closed adapter from the live vision stack to a policy.

The existing visual extractor exposes a ``cr_bot`` ``GameState`` through
``MatchSessionStep``.  This module joins that public boundary to both the
legacy V1 feature tensors and the recurrent prototype's V2 tensor contract:

``process_frame`` -> ``MatchSession.process`` -> ``game_state`` -> policy tensors

It deliberately does not own a camera loop, infer card identities, or send raw
ADB coordinates.  A caller can pass the resulting placement command to the
reviewed ``AutonomousPhone.select_and_place`` API, which performs a fresh card
identity check before tapping.

The live observation is viewer-local (the same convention as the existing
``cr_bot`` feature stack).  Therefore the returned ``(column, row)`` cell must
be sent to a calibration artifact for that same phone; callers must not mirror
it a second time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, Protocol

import numpy as np

from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.domain.game_state import Action as PolicyAction
from cr_bot.features.action_masks import get_action_mask
from cr_bot.features.board_rasterizer import build_board
from cr_bot.features.global_features import build_global_vector
from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

from ..observation import ACTION_MASK_SHAPE, PolicyObservationV1, policy_card_name
from ..observation_v2 import PolicyObservationV2
from ..observation_v2_adapter import build_public_entity_rows
from ..geometry import building_footprint_fits, is_basic_deploy_cell, is_spell_cell
from .schema import PhysicalLabError


_MASK_2D_SHAPE = ACTION_MASK_SHAPE[1:]
_LOGGER = logging.getLogger(__name__)
_LOGGED_UNSUPPORTED_VISUAL_LABELS: set[str] = set()

# KataCR reports the detector's unit labels (for example ``skeleton`` and
# ``hog``), while the feature stack's metadata is keyed by the playable card
# (``skeletons`` and ``hog-rider``).  Keep this normalization at the visual
# policy boundary: the extractor's GameState remains untouched, but board and
# public-token features receive the metadata vocabulary they require.
_VISUAL_ENTITY_FEATURE_ALIASES: dict[str, str] = {
    "bat": "bats",
    "bat-evolution": "bats",
    "bush-goblin": "goblins",
    "barbarian": "barbarians",
    "barbarian-evolution": "barbarians",
    "cursed-hog": "hog-rider",
    "elixir-golem-small": "elixir-golem",
    "elixir-golem-mid": "elixir-golem",
    "goblin": "goblins",
    "goblin-brawler": "goblins",
    "phoenix-egg": "phoenix",
    "phoenix-small": "phoenix",
    "golemite": "golem",
    "elixir-golemite": "elixir-golem",
    "elixir-blob": "elixir-golem",
    "lava-pup": "lava-hound",
    "minion": "minions",
    "rascal-boy": "barbarians",
    "rascal-girl": "spear-goblins",
    "spear-goblin": "spear-goblins",
    "cannon-cart-building": "cannon-cart",
}


def _visual_feature_card_name(value: object) -> str | None:
    """Resolve a detector unit label to a card-metadata feature key."""

    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().casefold().replace("_", "-")
    if normalized in CARD_METADATA:
        return normalized

    mapped = DIRECT_UNIT_TO_CARD.get(normalized)
    if mapped is None:
        mapped = _VISUAL_ENTITY_FEATURE_ALIASES.get(normalized)
    if isinstance(mapped, str) and mapped in CARD_METADATA:
        return mapped
    # Some YOLO labels are transient effects/projectiles (for example
    # ``bomb`` or ``axe``), not card-backed units.  They cannot contribute a
    # trustworthy card feature and are omitted by the caller.
    return None


def _canonicalize_visual_match(match: object) -> object | None:
    """Copy one visual match with a metadata-compatible troop label."""

    troop = getattr(match, "troop", None)
    raw_name = getattr(troop, "class_name", None)
    feature_name = _visual_feature_card_name(raw_name)
    if feature_name is None:
        label = repr(raw_name)
        if label not in _LOGGED_UNSUPPORTED_VISUAL_LABELS:
            _LOGGED_UNSUPPORTED_VISUAL_LABELS.add(label)
            _LOGGER.warning(
                "visual feature omitted unsupported detector label=%s "
                "(team=%s track=%s); no card metadata mapping",
                label,
                getattr(troop, "team", None),
                getattr(troop, "track_id", None),
            )
        return None
    if feature_name == raw_name:
        return match
    try:
        return replace(match, troop=replace(troop, class_name=feature_name))
    except TypeError:
        # The production extractor uses the slotted dataclasses.  Preserve
        # compatibility with light-weight test doubles that are not dataclass
        # instances; the original value will then fail closed if unsupported.
        return match


def _canonicalize_visual_game_state(game_state: Any) -> Any:
    """Return a non-mutating metadata-normalized view of a visual state."""

    own_units = getattr(game_state, "own_units", None)
    enemy_units = getattr(game_state, "enemy_units", None)
    if not isinstance(own_units, list) or not isinstance(enemy_units, list):
        return game_state
    normalized_own = [
        normalized
        for match in own_units
        if (normalized := _canonicalize_visual_match(match)) is not None
    ]
    normalized_enemy = [
        normalized
        for match in enemy_units
        if (normalized := _canonicalize_visual_match(match)) is not None
    ]
    if normalized_own == own_units and normalized_enemy == enemy_units:
        return game_state
    try:
        return replace(
            game_state,
            own_units=normalized_own,
            enemy_units=normalized_enemy,
        )
    except TypeError:
        # Keep the adapter safe for non-dataclass test doubles.  Real
        # extractor GameState objects always take the replace path above.
        return game_state


class PolicyBridgeError(PhysicalLabError):
    """Raised when a visual state or policy action cannot be used safely."""


@dataclass(frozen=True, slots=True)
class PlacementCommand:
    """A viewer-local, identity-checked placement request.

    ``arena_cell`` remains in the shared ``(column, row)`` action-grid
    convention.  It is intentionally not converted to pixels here.
    """

    card_id: str
    card_slot: int
    arena_cell: tuple[int, int]


class _PhonePlacer(Protocol):
    def select_and_place(
        self,
        card_id: str,
        *,
        calibration: Any,
        arena_cell: tuple[int, int],
        expected_slot: int | None = None,
        capture: Any = None,
    ) -> tuple[int, Any, Any]: ...


def _bool_mask(value: object, *, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.bool_)
    if array.shape != _MASK_2D_SHAPE:
        raise PolicyBridgeError(
            f"{field_name} must have shape {_MASK_2D_SHAPE}, got {array.shape}"
        )
    return np.ascontiguousarray(array, dtype=bool)


def _visual_legal_mask(card_name: str) -> np.ndarray:
    """Build the simulator's center-cell legality convention for one card.

    ``cr_bot.features.action_masks.get_action_mask`` is retained as the
    compatibility/debug ``spatial_masks`` input.  Its historical building
    mask uses top-left footprint anchors, while the simulator's policy action
    contract uses deployment centers.  The policy-facing ``legal_play`` mask
    therefore uses the center semantics directly.
    """

    metadata = CARD_METADATA.get(card_name)
    if not isinstance(metadata, dict):
        return np.zeros((32, 18), dtype=bool)
    placement = metadata.get("placement_class")
    result = np.zeros((32, 18), dtype=bool)
    for row in range(32):
        for col in range(18):
            cell = (col, row)
            if placement == "building":
                # Tesla is the only ordinary building in the fixed visual
                # contract with a 2x2 footprint; the other structures use 3x3.
                footprint_size = 2 if card_name == "tesla" else 3
                allowed = building_footprint_fits(0, cell, footprint_size)
            elif placement == "spell_anywhere":
                allowed = is_spell_cell(cell)
            elif placement == "spells":
                # The fixed policy deck's Log is the only restricted spell.
                allowed = is_spell_cell(cell) and row >= 17
            elif placement == "global_target":
                allowed = is_spell_cell(cell)
            else:
                allowed = is_basic_deploy_cell(0, cell)
            result[row, col] = allowed
    return result


def observation_from_game_state(
    game_state: Any,
    *,
    arena_px: tuple[float, float, float, float],
    legal_wait: bool | None = None,
) -> PolicyObservationV1:
    """Project a visual ``GameState`` into the pinned policy observation.

    This is intentionally conservative.  Unknown hand cards are represented
    with no legal play actions, and a card is playable only when its detected
    elixir is sufficient.  The card placer still performs the final fresh
    screenshot/identity check before any tap.
    """

    if game_state is None or not hasattr(game_state, "hud"):
        raise PolicyBridgeError("a live in-game GameState is required")
    hud = game_state.hud
    raw_hand = getattr(hud, "hand_cards", None)
    if not isinstance(raw_hand, (list, tuple)) or len(raw_hand) < 4:
        raise PolicyBridgeError("visual state does not contain four hand cards")

    feature_state = _canonicalize_visual_game_state(game_state)
    try:
        board = np.ascontiguousarray(build_board(feature_state, arena_px), dtype=np.float32)
        global_vector = np.ascontiguousarray(build_global_vector(feature_state), dtype=np.float32)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise PolicyBridgeError(f"visual state cannot be rasterized safely: {error}") from error

    spatial_masks = np.zeros(ACTION_MASK_SHAPE, dtype=bool)
    legal_play = np.zeros(ACTION_MASK_SHAPE, dtype=bool)
    try:
        elixir = float(hud.elixir_self)
    except (TypeError, ValueError):
        elixir = -1.0

    for slot, raw_card in enumerate(tuple(raw_hand)[:4]):
        card_name = policy_card_name(raw_card)
        if card_name is None:
            continue
        try:
            spatial_masks[slot] = _bool_mask(
                get_action_mask(card_name, feature_state),
                field_name=f"spatial mask for hand slot {slot}",
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PolicyBridgeError(
                f"cannot build compatibility mask for hand slot {slot}: {error}"
            ) from error
        metadata = CARD_METADATA.get(card_name)
        cost = metadata.get("elixir_cost") if isinstance(metadata, dict) else None
        if isinstance(cost, (int, float)) and elixir + 1e-6 >= float(cost):
            legal_play[slot] = _visual_legal_mask(card_name)

    if legal_wait is None:
        started = bool(getattr(game_state, "started", True))
        try:
            time_left = float(hud.time_left_s)
        except (TypeError, ValueError):
            time_left = 0.0
        legal_wait = started and time_left > 0.0

    for array in (board, global_vector, spatial_masks, legal_play):
        array.setflags(write=False)
    return PolicyObservationV1(
        board=board,
        global_vector=global_vector,
        spatial_masks=spatial_masks,
        legal_play=np.ascontiguousarray(legal_play, dtype=bool),
        legal_wait=bool(legal_wait),
    )


def observation_v2_from_game_state(
    game_state: Any,
    *,
    arena_px: tuple[float, float, float, float],
    legal_wait: bool | None = None,
) -> PolicyObservationV2:
    """Project one visual ``GameState`` into the public recurrent V2 contract.

    The V1 tensors are built by the existing visual feature stack.  Entity
    rows are built from the same viewer-local ``GameState`` and contain only
    detector-visible public entities.  No simulator ``BattleState`` or
    training-only critic features are available on this path.
    """

    observation_v1 = observation_from_game_state(
        game_state,
        arena_px=arena_px,
        legal_wait=legal_wait,
    )
    try:
        public_entity_rows = build_public_entity_rows(
            _canonicalize_visual_game_state(game_state),
            viewer=0,
        )
        return PolicyObservationV2.from_v1(
            observation_v1,
            public_entity_rows=public_entity_rows,
        )
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        raise PolicyBridgeError(
            f"visual state cannot be converted to public V2 observation: {error}"
        ) from error


def observation_from_match_step(step: Any) -> PolicyObservationV1 | None:
    """Convert one ``MatchSessionStep`` when it is safe to make a decision.

    The helper intentionally returns ``None`` during lobby/countdown frames or
    tracker frames that are not marked for emission.  A caller should simply
    wait for the next frame in those cases.
    """

    if (
        step is None
        or not bool(getattr(step, "in_game", False))
        or not bool(getattr(step, "should_emit", False))
        or getattr(step, "game_state", None) is None
    ):
        return None
    analysis = getattr(step, "analysis", None)
    arena_px = getattr(analysis, "arena_px", None)
    if arena_px is None:
        raise PolicyBridgeError("match step does not contain arena calibration bounds")
    # MatchSession already classified this frame as in-game.  A transient OCR
    # miss must not make both WAIT and PLAY illegal; waiting is always the safe
    # action while the match boundary is still active.
    return observation_from_game_state(
        step.game_state,
        arena_px=tuple(arena_px),
        legal_wait=True,
    )


def observation_v2_from_match_step(step: Any) -> PolicyObservationV2 | None:
    """Convert an emitted visual match step into a public V2 observation.

    Lobby/countdown/end frames return ``None``.  The caller owns recurrent
    reset handling; a non-emitted frame never advances the policy hidden state.
    """

    if (
        step is None
        or not bool(getattr(step, "in_game", False))
        or not bool(getattr(step, "should_emit", False))
        or getattr(step, "game_state", None) is None
    ):
        return None
    analysis = getattr(step, "analysis", None)
    arena_px = getattr(analysis, "arena_px", None)
    if arena_px is None:
        raise PolicyBridgeError("match step does not contain arena calibration bounds")
    return observation_v2_from_game_state(
        step.game_state,
        arena_px=tuple(arena_px),
        legal_wait=True,
    )


def placement_command_from_policy_action(
    action: PolicyAction,
    game_state: Any,
    *,
    observation: PolicyObservationV1 | PolicyObservationV2 | None = None,
) -> PlacementCommand | None:
    """Resolve a policy action to a verified hand card and local grid cell.

    ``observation`` is optional for callers that perform their own legality
    gating.  Supplying it is recommended: stale or unaffordable placements
    are then rejected before the phone is tapped.
    """

    if not isinstance(action, PolicyAction):
        raise PolicyBridgeError("policy action must be cr_bot.domain.game_state.Action")
    kind = action.kind.strip().casefold().replace("_", "-")
    if kind in {"wait", "noop", "no-op"}:
        if action.card_idx is not None or action.cell is not None:
            raise PolicyBridgeError("wait action must not carry a slot or cell")
        return None
    if kind != "play" or action.card_idx is None or action.cell is None:
        raise PolicyBridgeError("only Wait and Play actions are supported")
    slot = action.card_idx
    if type(slot) is not int or not 0 <= slot < 4:
        raise PolicyBridgeError("policy card slot must be in [0, 3]")
    cell = action.cell
    if (
        not isinstance(cell, tuple)
        or len(cell) != 2
        or type(cell[0]) is not int
        or type(cell[1]) is not int
        or not (0 <= cell[0] < 18 and 0 <= cell[1] < 32)
    ):
        raise PolicyBridgeError(f"policy cell is outside the 18x32 grid: {cell!r}")
    if observation is not None and not bool(observation.legal_play[slot, cell[1], cell[0]]):
        raise PolicyBridgeError(f"policy action is not legal in the supplied visual observation: {cell!r}")

    try:
        raw_card = game_state.hud.hand_cards[slot]
    except (AttributeError, IndexError, TypeError) as error:
        raise PolicyBridgeError(f"visual state has no hand card at slot {slot}") from error
    card_id = policy_card_name(raw_card)
    if card_id is None:
        raise PolicyBridgeError(f"hand slot {slot} is not a supported base policy card: {raw_card!r}")
    return PlacementCommand(card_id=card_id, card_slot=slot, arena_cell=cell)


def dispatch_policy_action(
    phone: _PhonePlacer,
    action: PolicyAction,
    game_state: Any,
    *,
    calibration: Any,
    observation: PolicyObservationV1 | PolicyObservationV2 | None = None,
    capture: Any = None,
) -> tuple[int, Any, Any] | None:
    """Send one validated placement through ``AutonomousPhone``.

    Wait returns ``None``.  Play delegates all pixel conversion and fresh card
    identity checks to the existing phone API.
    """

    command = placement_command_from_policy_action(
        action,
        game_state,
        observation=observation,
    )
    if command is None:
        return None
    return phone.select_and_place(
        command.card_id,
        calibration=calibration,
        arena_cell=command.arena_cell,
        expected_slot=command.card_slot,
        capture=capture,
    )


__all__ = [
    "PlacementCommand",
    "PolicyBridgeError",
    "dispatch_policy_action",
    "observation_from_game_state",
    "observation_from_match_step",
    "observation_v2_from_game_state",
    "observation_v2_from_match_step",
    "placement_command_from_policy_action",
]
