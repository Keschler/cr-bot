"""Vision-policy compatible observations for authoritative simulator states.

The simulator deliberately keeps its authoritative :class:`BattleState`
separate from the lossy state exposed to a policy.  This adapter is the only
place where authoritative state is projected into the existing vision feature
contract.  In particular, it does not expose an entity's target, attack
cooldown, either draw pile other than the viewer's next card, the opponent's
hand, or the opponent's exact elixir.

``vision_v1_exact`` preserves the current ``cr_bot.features`` builders.  The
legacy spatial building mask uses top-left footprint anchors; the simulator's
actions use ``(column, row)`` *center* cells.  Consequently ``spatial_masks``
reproduces the legacy model input while ``legal_play`` is independently
generated with the authoritative center-cell convention. Consumers should
train on ``legal_play``; ``spatial_masks`` is a compatibility/debug channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from typing import Callable, Final

import numpy as np

from cr_bot.domain.game_state import (
    Action as PolicyAction,
    Detection,
    GameState,
    HudState,
    Match,
    PrincessTowerState,
)
from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.domain.constants import (
    FEATURE_MAX_KING_TOWER_HP,
    MAX_ELIXIR,
    PRINCESS_TOWER_HP,
    TOTAL_MATCH_SECONDS,
)
from cr_bot.features.action_space import ACTION_GRID, get_card_deploy_mask, map_ground
from cr_bot.features.board_rasterizer import (
    ENEMY_KING_TOWER_SITE,
    ENEMY_LEFT_PRINCESS_TOWER_SITE,
    ENEMY_RIGHT_PRINCESS_TOWER_SITE,
    KERNEL_3X3,
    OWN_KING_TOWER_SITE,
    OWN_LEFT_PRINCESS_TOWER_SITE,
    OWN_RIGHT_PRINCESS_TOWER_SITE,
    build_board,
    build_static_board,
)
from cr_bot.features.channels import (
    DYNAMIC_CHANNEL_IDX,
    DYNAMIC_CHANNELS,
    GLOBAL_SCALAR_FEATURES,
    STATIC_CHANNELS,
)
from cr_bot.features.global_features import (
    CARD_COUNT,
    encode_hand_cards,
    encode_next_card,
    encode_seen_enemy_cards,
    build_global_vector,
    one_hot_card,
)

from .actions import PlayCardAction, SimAction, UseAbilityAction, WaitAction
from .geometry import (
    GRID_COLS,
    GRID_ROWS,
    building_footprint_fits,
    is_basic_deploy_cell,
    is_spell_cell,
    mirror_cell,
    mirror_position,
    position_to_cell,
    validate_cell,
)
from .ruleset import CardDefinition, Ruleset, normalize_identifier
from .soa import ObservationSoA, is_public_observation_entity
from .state import BattleState, EntityState


VISION_V1_EXACT: Final = "vision_v1_exact"
OBSERVATION_SCHEMA_VERSION: Final = "vision-v1-exact-2"
BOARD_SHAPE: Final = (21, GRID_ROWS, GRID_COLS)
GLOBAL_VECTOR_SHAPE: Final = (768,)
ACTION_MASK_SHAPE: Final = (4, GRID_ROWS, GRID_COLS)

# These IDs are part of the existing policy checkpoint contract.  They must
# not be regenerated from the versioned Level-11 ruleset: changing a policy ID
# would silently reinterpret historical feature vectors.
BASE_POLICY_CARD_IDS: Final[dict[str, int]] = {
    "fireball": 28,
    "hog-rider": 49,
    "ice-golem": 51,
    "ice-spirit": 52,
    "log": 59,
    "musketeer": 72,
    "skeletons": 96,
    "cannon": 114,
}

# The legacy policy vocabulary contains only the eight playable cards in the
# fixed player deck.  The simulator nevertheless exposes every opponent card
# and every internal child form through the same board/feature boundary.  A
# child form is rendered with its nearest public-card feature profile; its
# authoritative card_id, HP, targeting, and split/death behavior remain
# unchanged in BattleState.
_ENTITY_FEATURE_ALIASES: Final = {
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


def observation_contract_manifest() -> dict[str, object]:
    """Describe every imported policy feature convention that affects bytes."""

    card_features = {
        card_id: {
            key: CARD_METADATA[card_id].get(key)
            for key in ("id", "placement_class", "damage", "hit_speed", "is_air")
        }
        for card_id in sorted(BASE_POLICY_CARD_IDS)
    }
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "board_shape": list(BOARD_SHAPE),
        "global_vector_shape": list(GLOBAL_VECTOR_SHAPE),
        "action_mask_shape": list(ACTION_MASK_SHAPE),
        "static_channels": list(STATIC_CHANNELS),
        "dynamic_channels": list(DYNAMIC_CHANNELS),
        "global_scalar_features": list(GLOBAL_SCALAR_FEATURES),
        "base_policy_card_ids": dict(sorted(BASE_POLICY_CARD_IDS.items())),
        "base_card_features": card_features,
        "entity_feature_aliases": dict(sorted(_ENTITY_FEATURE_ALIASES.items())),
        "normalizers": {
            "max_elixir": MAX_ELIXIR,
            "total_match_seconds": TOTAL_MATCH_SECONDS,
            # The simulator's fixed runtime ruleset is Level 11.  The
            # imported cr_bot feature builders use Level-16 tower caps, so
            # these values are part of this simulator-owned contract.
            "princess_tower_hp": 3_052,
            "king_tower_hp": 4_824,
            "tower_hp_source": "simulator.rulesets.v1",
            "threat_weight_source": "active-ruleset-damage-over-attack-interval",
        },
        "action_grid": {
            "columns": ACTION_GRID.cols,
            "rows": ACTION_GRID.rows,
            "x0": ACTION_GRID.x0,
            "y0": ACTION_GRID.y0,
            "x1": ACTION_GRID.x1,
            "y1": ACTION_GRID.y1,
            "cell_order": "column,row",
            "tensor_order": "row,column",
        },
        "ground_mask_sha256": hashlib.sha256(map_ground.tobytes(order="C")).hexdigest(),
        "viewer_one_transform": "rotate-180",
    }


def calculate_observation_contract_hash() -> str:
    encoded = json.dumps(
        observation_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


PINNED_OBSERVATION_CONTRACT_HASH: Final = (
    "sha256:c9a9d0455d9ef70707051153c88348c26c533bb7479d37e83eb947041aa793ec"
)


def verify_observation_contract() -> None:
    actual = calculate_observation_contract_hash()
    if actual != PINNED_OBSERVATION_CONTRACT_HASH:
        raise RuntimeError(
            "vision policy feature contract changed; review compatibility and bump "
            f"the observation schema/hash (pinned={PINNED_OBSERVATION_CONTRACT_HASH}, actual={actual})"
        )


verify_observation_contract()

_POLICY_ALIASES: Final = {
    "the-log": "log",
}
# The legacy policy vocabulary contains only the eight playable cards in the
# fixed player deck.  The simulator nevertheless exposes every opponent card
# and every internal child form through the same board/feature boundary.  A
# child form is therefore rendered with its nearest public-card feature
# profile; its authoritative card_id, HP, targeting, and split/death behavior
# remain unchanged in BattleState.  This avoids leaking privileged simulator
# fields while preventing a Golemite/Lava Pup from crashing the legacy board
# rasterizer (which obtains air/threat metadata from CARD_METADATA).
_PUBLIC_CARD_PLAY_EVENT: Final = "card_played"
_FINISHED_PHASES: Final = frozenset(
    {"complete", "completed", "ended", "finished", "game-over", "match-over", "terminal"}
)
_NON_PLAYING_PHASES: Final = _FINISHED_PHASES | frozenset(
    {"", "countdown", "not-started", "pre-match", "pregame", "tiebreak"}
)
_ARENA_PX: Final = (0.0, 0.0, 1.0, 1.0)
_SOA_STATIC_BOARD = np.ascontiguousarray(build_static_board(), dtype=np.float32)
_SOA_STATIC_BOARD.setflags(write=False)


def _clip_unit(value: float) -> float:
    """Fast finite scalar equivalent of the feature-stack unit clip."""

    return min(1.0, max(0.0, float(value)))


def _normalize_soa_tower_hp(
    values: np.ndarray,
    ruleset: Ruleset | None = None,
) -> tuple[float, float, float]:
    left, king, right = (int(value) for value in values)
    towers = getattr(ruleset, "towers", None) if ruleset is not None else None
    princess = towers.get("princess-tower") if hasattr(towers, "get") else None
    king_tower = towers.get("king-tower") if hasattr(towers, "get") else None
    princess_hp = getattr(princess, "hitpoints", None)
    king_hp = getattr(king_tower, "hitpoints", None)
    if type(princess_hp) is not int or princess_hp <= 0:
        princess_hp = PRINCESS_TOWER_HP
    if type(king_hp) is not int or king_hp <= 0:
        king_hp = int(FEATURE_MAX_KING_TOWER_HP)
    return (
        _clip_unit(left / princess_hp),
        _clip_unit(king / king_hp),
        _clip_unit(right / princess_hp),
    )


LegalityCallback = Callable[[BattleState, PlayCardAction], bool]
LegalActionCellsCallback = Callable[
    [BattleState, int], tuple[tuple[tuple[int, int], ...], ...]
]


class UnsupportedPolicyFormError(ValueError):
    """Raised when ``vision_v1_exact`` cannot represent a visible card form."""


@dataclass(frozen=True, slots=True)
class PolicyObservationV1:
    """The exact array boundary consumed by the current vision policy."""

    board: np.ndarray
    global_vector: np.ndarray
    spatial_masks: np.ndarray
    legal_play: np.ndarray
    legal_wait: bool
    compatibility_mode: str = VISION_V1_EXACT
    contract_hash: str = PINNED_OBSERVATION_CONTRACT_HASH

    def __post_init__(self) -> None:
        _validate_array("board", self.board, BOARD_SHAPE, np.dtype(np.float32))
        _validate_array(
            "global_vector",
            self.global_vector,
            GLOBAL_VECTOR_SHAPE,
            np.dtype(np.float32),
        )
        _validate_array("spatial_masks", self.spatial_masks, ACTION_MASK_SHAPE, np.dtype(bool))
        _validate_array("legal_play", self.legal_play, ACTION_MASK_SHAPE, np.dtype(bool))
        if not isinstance(self.legal_wait, (bool, np.bool_)):
            raise TypeError("legal_wait must be boolean")
        if self.compatibility_mode != VISION_V1_EXACT:
            raise ValueError(f"unsupported compatibility mode: {self.compatibility_mode!r}")
        if self.contract_hash != PINNED_OBSERVATION_CONTRACT_HASH:
            raise ValueError("observation contract hash does not match vision_v1_exact")

    def as_dict(self, *, copy: bool = False) -> dict[str, np.ndarray | bool | str]:
        """Return a model-friendly mapping, optionally copying array storage."""

        maybe_copy = (lambda value: value.copy()) if copy else (lambda value: value)
        return {
            "board": maybe_copy(self.board),
            "global_vector": maybe_copy(self.global_vector),
            "spatial_masks": maybe_copy(self.spatial_masks),
            "legal_play": maybe_copy(self.legal_play),
            "legal_wait": bool(self.legal_wait),
            "compatibility_mode": self.compatibility_mode,
            "contract_hash": self.contract_hash,
        }

    def exact_policy_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return only the three tensors produced by the live vision feature stack."""

        return self.board, self.global_vector, self.spatial_masks


@dataclass(slots=True)
class ObservationMemory:
    """Public, viewer-local temporal state used by the observation adapter.

    The opponent estimate starts from the public match rule, regenerates from
    public match time, and deducts costs only after a public ``card_played``
    event.  It never reads ``BattleState.players[opponent].elixir_milli``.
    A :class:`Fraction` keeps sparse and dense observation schedules exactly
    equivalent without introducing floating-point state into the simulator.
    """

    viewer: int = 0
    seen_opponent_cards: list[str] = field(default_factory=list)
    last_elapsed_us: int = 0
    last_event_sequence: int = -1
    ruleset_hash: str | None = None
    _opponent_elixir_milli: Fraction = field(default_factory=Fraction, repr=False)
    _initialized: bool = field(default=False, repr=False)
    _battle_seed: int | None = field(default=None, repr=False)
    # ``event_sequence`` is the authoritative append revision: every event
    # emitted by the engine increments it.  The retained list and its O(1)
    # mutation revision catch the exceptional case where a caller rewrites an
    # already-consumed history without advancing that public sequence.  This
    # avoids reserializing the complete event log on every observation.
    _processed_event_revision: int | None = field(default=None, repr=False)
    _processed_event_list: object | None = field(default=None, repr=False)
    _processed_event_mutation_revision: int | None = field(default=None, repr=False)
    _processed_event_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        _validate_viewer(self.viewer)

    @property
    def opponent_elixir_milli_est(self) -> int:
        """Floor of the deterministic public estimate in milli-elixir."""

        return self._opponent_elixir_milli.numerator // self._opponent_elixir_milli.denominator

    @property
    def opponent_elixir_est(self) -> float:
        return float(self._opponent_elixir_milli / 1_000)

    def reset(self, ruleset: Ruleset, *, battle_seed: int | None = None) -> None:
        self.seen_opponent_cards.clear()
        self.last_elapsed_us = 0
        self.last_event_sequence = -1
        self.ruleset_hash = ruleset.content_hash
        self._opponent_elixir_milli = Fraction(ruleset.match.initial_elixir_milli, 1)
        self._initialized = True
        self._battle_seed = battle_seed
        self._processed_event_revision = None
        self._processed_event_list = None
        self._processed_event_mutation_revision = None
        self._processed_event_count = 0

    def update(self, state: BattleState, ruleset: Ruleset) -> None:
        """Consume previously unseen public events through ``state.elapsed_us``."""

        _validate_state_ruleset(state, ruleset)
        if state.elapsed_us < 0:
            raise ValueError("battle elapsed_us cannot be negative")
        if (
            not self._initialized
            or self.ruleset_hash != ruleset.content_hash
            or self._battle_seed != state.seed
            or state.elapsed_us < self.last_elapsed_us
            or state.event_sequence < self.last_event_sequence
        ):
            self.reset(ruleset, battle_seed=state.seed)
            history_rewritten = False
        elif (
            self._processed_event_revision is not None
            and self._processed_event_list is not None
        ):
            current_mutation_revision = getattr(state.events, "mutation_revision", None)
            history_rewritten = (
                state.event_sequence == self._processed_event_revision
                and (
                    state.events is not self._processed_event_list
                    or (
                        self._processed_event_mutation_revision is not None
                        and current_mutation_revision
                        != self._processed_event_mutation_revision
                    )
                )
            )
        else:
            history_rewritten = False
        if history_rewritten:
            # A restored or externally synchronized state may replace an
            # already-consumed event at the same sequence number.  Sequence
            # monotonicity alone cannot detect that rewrite.  Replay the
            # authoritative public history so card memory and elixir belief do
            # not silently describe a different episode.  Normal append-only
            # transport changes the sequence revision, so it takes the
            # incremental suffix path below.
            self.reset(ruleset, battle_seed=state.seed)

        # Engine histories are append-only and sequence ordered.  The common
        # path can therefore consume an event suffix directly instead of
        # scanning and sorting the complete replay log on every observation.
        # Persistent/process transports may replace the list object while
        # retaining the complete prefix; use a small binary search for that
        # case.  History rewrites retain the defensive full sort below.
        if history_rewritten:
            new_events = sorted(
                (
                    event
                    for event in state.events
                    if event.sequence > self.last_event_sequence
                ),
                key=lambda event: (event.tick, event.sequence),
            )
        elif (
            self._processed_event_list is state.events
            and self._processed_event_count <= len(state.events)
        ):
            new_events = state.events[self._processed_event_count :]
        else:
            low = 0
            high = len(state.events)
            while low < high:
                middle = (low + high) // 2
                if state.events[middle].sequence <= self.last_event_sequence:
                    low = middle + 1
                else:
                    high = middle
            new_events = state.events[low:]
        for event in new_events:
            # Engine events are emitted during tick ``t`` after that tick's
            # resource update. Their public boundary is therefore the end of
            # the tick, not its beginning. This matters at the elixir cap:
            # regeneration is discarded before a play rather than banked
            # after the cost deduction.
            event_us = max(
                0,
                min(state.elapsed_us, (int(event.tick) + 1) * ruleset.tick_us),
            )
            # A late-arriving event remains usable without pretending we can
            # reconstruct past cap ordering.  Deduct it at the current public
            # estimate instead.
            event_us = max(self.last_elapsed_us, event_us)
            self._advance_elixir(event_us, ruleset)
            if event.kind == _PUBLIC_CARD_PLAY_EVENT:
                self._consume_card_play(event, ruleset)
            elif event.kind in {"elixir_generated", "elixir_awarded"}:
                self._consume_elixir_gain(event, ruleset)
            self.last_event_sequence = max(self.last_event_sequence, event.sequence)

        self._advance_elixir(state.elapsed_us, ruleset)
        self._processed_event_revision = state.event_sequence
        self._processed_event_list = state.events
        mutation_revision = getattr(state.events, "mutation_revision", None)
        self._processed_event_mutation_revision = (
            mutation_revision if type(mutation_revision) is int else None
        )
        self._processed_event_count = len(state.events)

    def _advance_elixir(self, target_elapsed_us: int, ruleset: Ruleset) -> None:
        if target_elapsed_us <= self.last_elapsed_us:
            return
        cursor = self.last_elapsed_us
        for segment_end, interval_us in _elixir_segments(ruleset):
            if cursor >= target_elapsed_us:
                break
            if cursor >= segment_end:
                continue
            end = min(target_elapsed_us, segment_end)
            self._opponent_elixir_milli += Fraction((end - cursor) * 1_000, interval_us)
            self._opponent_elixir_milli = min(
                self._opponent_elixir_milli,
                Fraction(ruleset.match.max_elixir_milli, 1),
            )
            cursor = end
        self.last_elapsed_us = target_elapsed_us

    def _consume_card_play(self, event: object, ruleset: Ruleset) -> None:
        # SimEvent.get is intentionally used rather than looking at PlayerState.
        player_raw = event.get("player", event.get("owner"))  # type: ignore[attr-defined]
        if not isinstance(player_raw, int) or isinstance(player_raw, bool):
            return
        if player_raw != 1 - self.viewer:
            return
        card_raw = event.get("card_id", event.get("card"))  # type: ignore[attr-defined]
        if not isinstance(card_raw, str):
            return
        card = _lookup_card_definition(ruleset, card_raw)
        if card is None:
            # An unrepresented form must never be charged at a base-card cost.
            return
        policy_name = policy_card_name(card_raw)
        if policy_name is not None:
            observed_name = policy_name
        else:
            # V1 keeps the *player* hand/action vocabulary fixed, but its
            # opponent interaction set deliberately contains every eligible
            # base card.  Those cards must still be charged and remembered at
            # the public boundary; otherwise playing Furnace/Baby Dragon (or
            # any other non-deck opponent card) would crash observation and
            # make the all-opponent-card training surface unusable.  Excluded
            # Hero/Evolution/ability forms remain fail-closed.
            canonical = getattr(card, "card_id", None)
            interaction_set = getattr(ruleset, "interaction_set", None)
            if not isinstance(canonical, str) or not isinstance(interaction_set, tuple):
                raise UnsupportedPolicyFormError(
                    f"vision_v1_exact cannot represent public opponent card form {card_raw!r}"
                )
            if canonical not in interaction_set:
                raise UnsupportedPolicyFormError(
                    f"vision_v1_exact cannot represent public opponent card form {card_raw!r}"
                )
            observed_name = canonical
        if observed_name not in self.seen_opponent_cards:
            self.seen_opponent_cards.append(observed_name)
        # ``card_played`` carries the effective cost.  This is different from
        # the catalog cost for Mirror (and for future dynamic-cost cards), so
        # charging the definition's base cost makes the public belief drift
        # upward after every such play.
        raw_cost = event.get("cost_milli")  # type: ignore[attr-defined]
        cost_milli = (
            int(raw_cost)
            if isinstance(raw_cost, int) and not isinstance(raw_cost, bool) and raw_cost >= 0
            else int(card.elixir_milli)
        )
        self._opponent_elixir_milli = max(
            Fraction(0, 1),
            self._opponent_elixir_milli - cost_milli,
        )

    def _consume_elixir_gain(self, event: object, ruleset: Ruleset) -> None:
        """Apply public bonus-elixir events to the opponent estimate.

        Normal regeneration is reconstructed from elapsed time.  Building
        generation and death rewards are additive public events, and their
        payload already reports the amount that survived the ten-elixir cap.
        """

        player_raw = event.get("player", event.get("owner"))  # type: ignore[attr-defined]
        if not isinstance(player_raw, int) or isinstance(player_raw, bool):
            return
        if player_raw != 1 - self.viewer:
            return
        raw_amount = event.get("amount_milli")  # type: ignore[attr-defined]
        if not isinstance(raw_amount, int) or isinstance(raw_amount, bool) or raw_amount <= 0:
            return
        self._opponent_elixir_milli = min(
            Fraction(ruleset.match.max_elixir_milli, 1),
            self._opponent_elixir_milli + int(raw_amount),
        )


def build_policy_observation(
    state: BattleState,
    ruleset: Ruleset,
    *,
    viewer: int = 0,
    memory: ObservationMemory | None = None,
    compatibility_mode: str = VISION_V1_EXACT,
    legality_callback: LegalityCallback | None = None,
    legal_action_cells_callback: LegalActionCellsCallback | None = None,
    soa_state: ObservationSoA | None = None,
    allow_unrepresented_hand: bool = False,
    _soa_already_synced: bool = False,
) -> PolicyObservationV1:
    """Project an authoritative state into the current policy boundary.

    ``vision_v1_exact`` remains fail-closed by default.  The additive V2
    adapter may opt into ``allow_unrepresented_hand`` for the non-learning
    simulator side: opponent decks can contain cards outside the fixed
    player's eight-card action vocabulary, and those cards are represented as
    empty hand slots rather than crashing the actor-facing observation.  The
    authoritative opponent controller still sees its real hand in
    ``BattleState``.
    """

    _validate_viewer(viewer)
    _validate_state_ruleset(state, ruleset)
    if type(allow_unrepresented_hand) is not bool:
        raise TypeError("allow_unrepresented_hand must be boolean")
    if compatibility_mode != VISION_V1_EXACT:
        raise ValueError(f"unsupported compatibility mode: {compatibility_mode!r}")
    if legality_callback is not None and legal_action_cells_callback is not None:
        raise ValueError("legality_callback and legal_action_cells_callback are mutually exclusive")
    if memory is None:
        memory = ObservationMemory(viewer=viewer)
    elif memory.viewer != viewer:
        raise ValueError(f"observation memory belongs to viewer {memory.viewer}, not {viewer}")
    memory.update(state, ruleset)

    if soa_state is not None:
        if not isinstance(soa_state, ObservationSoA):
            raise TypeError("soa_state must be an ObservationSoA instance")
        return _build_policy_observation_soa(
            state,
            ruleset,
            viewer=viewer,
            memory=memory,
            legality_callback=legality_callback,
            legal_action_cells_callback=legal_action_cells_callback,
            soa_state=soa_state,
            allow_unrepresented_hand=allow_unrepresented_hand,
            soa_already_synced=_soa_already_synced,
        )

    observed = battle_state_to_observed_game_state(
        state,
        ruleset,
        viewer=viewer,
        memory=memory,
        allow_unrepresented_hand=allow_unrepresented_hand,
    )
    if _supports_runtime_projection(ruleset):
        # The imported feature stack is a Level-16 snapshot.  Use the local
        # projection whenever an authoritative ruleset is available so board
        # threat and global tower HP describe the same runtime as physics.
        runtime_soa = ObservationSoA()
        runtime_soa.sync(state, _feature_card_name, ruleset)
        board = np.ascontiguousarray(_build_soa_board(runtime_soa, viewer), dtype=np.float32)
        global_vector = _build_soa_global_vector(
            state,
            ruleset,
            viewer=viewer,
            memory=memory,
            soa_state=runtime_soa,
            allow_unrepresented_hand=allow_unrepresented_hand,
        )
    elif _requires_null_safe_board(observed):
        # The external rasterizer assumes every card has an attack cadence.
        # Keep its ordinary output byte-compatible for minimal adapter test
        # doubles, but use the simulator-local renderer for passive/resource
        # cards whose threat is zero by definition.
        safe_soa = ObservationSoA()
        safe_soa.sync(state, _feature_card_name)
        board = np.ascontiguousarray(_build_soa_board(safe_soa, viewer), dtype=np.float32)
        global_vector = np.ascontiguousarray(build_global_vector(observed), dtype=np.float32)
    else:
        board = np.ascontiguousarray(build_board(observed, _ARENA_PX), dtype=np.float32)
        global_vector = np.ascontiguousarray(build_global_vector(observed), dtype=np.float32)
    spatial_masks = _build_spatial_masks(observed)
    if legal_action_cells_callback is not None:
        legal_play = _build_legal_play_from_cells(
            state,
            viewer=viewer,
            legal_action_cells_callback=legal_action_cells_callback,
        )
    else:
        legal_play = _build_legal_play(
            state,
            ruleset,
            observed,
            spatial_masks,
            viewer=viewer,
            legality_callback=legality_callback,
        )

    # Feature arrays are immutable snapshots.  This prevents an agent or a
    # vectorized environment from corrupting a replayable BattleState view.
    for array in (board, global_vector, spatial_masks, legal_play):
        array.setflags(write=False)
    return PolicyObservationV1(
        board=board,
        global_vector=global_vector,
        spatial_masks=spatial_masks,
        legal_play=legal_play,
        legal_wait=not state.terminal and _phase_is_active(state.phase),
        compatibility_mode=compatibility_mode,
    )


def _build_policy_observation_soa(
    state: BattleState,
    ruleset: Ruleset,
    *,
    viewer: int,
    memory: ObservationMemory,
    legality_callback: LegalityCallback | None,
    legal_action_cells_callback: LegalActionCellsCallback | None,
    soa_state: ObservationSoA,
    allow_unrepresented_hand: bool,
    soa_already_synced: bool,
) -> PolicyObservationV1:
    """Build the policy tensors directly from reusable SoA columns.

    Legality remains delegated to the exact engine callback.  Only the
    projection of public board/global features uses the SoA path, so this
    optimization cannot alter action acceptance or simulator mechanics.
    """

    if not soa_already_synced:
        soa_state.sync(state, _feature_card_name, ruleset)
    board = _build_soa_board(soa_state, viewer)
    global_vector = _build_soa_global_vector(
        state,
        ruleset,
        viewer=viewer,
        memory=memory,
        soa_state=soa_state,
        allow_unrepresented_hand=allow_unrepresented_hand,
    )
    spatial_masks = _build_soa_spatial_masks(state, viewer=viewer, soa_state=soa_state)
    if legal_action_cells_callback is not None:
        legal_play = _build_legal_play_from_cells(
            state,
            viewer=viewer,
            legal_action_cells_callback=legal_action_cells_callback,
            ruleset=ruleset,
            soa_state=soa_state,
        )
    else:
        # Preserve the legacy callback/conservative fallback semantics for
        # callers that did not provide the exact engine legality callback.
        observed = battle_state_to_observed_game_state(
            state,
            ruleset,
            viewer=viewer,
            memory=memory,
            allow_unrepresented_hand=allow_unrepresented_hand,
        )
        legal_play = _build_legal_play(
            state,
            ruleset,
            observed,
            spatial_masks,
            viewer=viewer,
            legality_callback=legality_callback,
        )

    for array in (board, global_vector, spatial_masks, legal_play):
        array.setflags(write=False)
    return PolicyObservationV1(
        board=board,
        global_vector=global_vector,
        spatial_masks=spatial_masks,
        legal_play=legal_play,
        legal_wait=not state.terminal and _phase_is_active(state.phase),
        compatibility_mode=VISION_V1_EXACT,
    )


def _feature_card_name(card_id: str) -> str:
    return _cached_feature_card_name(card_id)


@lru_cache(maxsize=512)
def _cached_feature_card_name(card_id: str) -> str:
    normalized = normalize_identifier(card_id)
    card_name = normalized if normalized in CARD_METADATA else _ENTITY_FEATURE_ALIASES.get(normalized)
    # This private mixed-form ID is deliberately kept out of the public
    # contract manifest; it renders with the same public profile as Goblins.
    if card_name is None and normalized == "goblin-gang-goblin":
        card_name = "goblins"
    if card_name is None:
        raise UnsupportedPolicyFormError(
            f"vision_v1_exact cannot represent visible entity form {card_id!r}"
        )
    return card_name


def _requires_null_safe_board(observed: GameState) -> bool:
    """Detect public cards the external rasterizer cannot threat-score."""

    for match in (*observed.own_units, *observed.enemy_units):
        detection = getattr(match, "troop", None)
        if detection is None:
            continue
        metadata = CARD_METADATA.get(getattr(detection, "class_name", ""))
        if metadata is None:
            continue
        if (
            type(metadata.get("damage")) not in (int, float)
            or type(metadata.get("hit_speed")) not in (int, float)
        ):
            return True
    return False


def _supports_runtime_projection(ruleset: Ruleset) -> bool:
    """Whether ``ruleset`` has the fields needed for active feature values."""

    cards = getattr(ruleset, "cards", None)
    towers = getattr(ruleset, "towers", None)
    return (
        hasattr(cards, "get")
        and hasattr(towers, "get")
        and cards.get("hog-rider") is not None
        and towers.get("princess-tower") is not None
        and towers.get("king-tower") is not None
    )


def _build_soa_board(soa_state: ObservationSoA, viewer: int) -> np.ndarray:
    cached = soa_state.board_snapshot(viewer)
    if cached is not None:
        return cached
    board = np.zeros(BOARD_SHAPE, dtype=np.float32)
    static_count = len(STATIC_CHANNELS)
    board[:static_count] = _SOA_STATIC_BOARD
    dynamic = board[static_count:]

    _, tower_alive = soa_state.viewer_towers_view(viewer)
    own_tower_mask = dynamic[DYNAMIC_CHANNEL_IDX["own_alive_tower_mask"]]
    enemy_tower_mask = dynamic[DYNAMIC_CHANNEL_IDX["enemy_alive_tower_mask"]]
    # The legacy rasterizer always exposes king tower sites; the alive flag
    # only controls princess tower sites.
    own_tower_mask[OWN_KING_TOWER_SITE] = 1.0
    enemy_tower_mask[ENEMY_KING_TOWER_SITE] = 1.0
    if tower_alive[0, 0]:
        own_tower_mask[OWN_LEFT_PRINCESS_TOWER_SITE] = 1.0
    if tower_alive[0, 2]:
        own_tower_mask[OWN_RIGHT_PRINCESS_TOWER_SITE] = 1.0
    if tower_alive[1, 0]:
        enemy_tower_mask[ENEMY_LEFT_PRINCESS_TOWER_SITE] = 1.0
    if tower_alive[1, 2]:
        enemy_tower_mask[ENEMY_RIGHT_PRINCESS_TOWER_SITE] = 1.0

    for index in range(soa_state.count):
        if not soa_state.renderable[index]:
            continue
        if viewer == 1:
            x_mtile, y_mtile = mirror_position(
                int(soa_state.x_mtile[index]),
                int(soa_state.y_mtile[index]),
            )
            cell = position_to_cell(x_mtile, y_mtile)
            if cell is None:
                continue
            col, row = cell
        else:
            col = int(soa_state.cell_cols[index])
            row = int(soa_state.cell_rows[index])
        ally = int(soa_state.owners[index]) == viewer
        air = bool(soa_state.is_air[index])
        if ally:
            presence_name = "ally_air_presence" if air else "ally_ground_presence"
            hp_name = "ally_hp_mass"
            threat_name = "ally_threat_mass"
        else:
            presence_name = "enemy_air_presence" if air else "enemy_ground_presence"
            hp_name = "enemy_hp_mass"
            threat_name = "enemy_threat_mass"
        _splat_soa(dynamic[DYNAMIC_CHANNEL_IDX[presence_name]], row, col, 1.0)
        _splat_soa(
            dynamic[DYNAMIC_CHANNEL_IDX[hp_name]],
            row,
            col,
            float(soa_state.hp_fraction[index]),
        )
        _splat_soa(
            dynamic[DYNAMIC_CHANNEL_IDX[threat_name]],
            row,
            col,
            float(soa_state.threat[index]),
        )
    return soa_state.cache_board(viewer, board)


def _splat_soa(channel: np.ndarray, row: int, col: int, value: float) -> None:
    # Slice the fixed 3x3 kernel in one NumPy operation.  The clipped slice
    # has the same row-major order as the old scalar loop, so overlapping
    # entities still accumulate in deterministic UID order while avoiding
    # nine Python-level bounds checks per entity.
    row_start = max(0, row - 1)
    row_end = min(GRID_ROWS, row + 2)
    col_start = max(0, col - 1)
    col_end = min(GRID_COLS, col + 2)
    kernel_row_start = row_start - row + 1
    kernel_row_end = kernel_row_start + (row_end - row_start)
    kernel_col_start = col_start - col + 1
    kernel_col_end = kernel_col_start + (col_end - col_start)
    channel[row_start:row_end, col_start:col_end] += value * KERNEL_3X3[
        kernel_row_start:kernel_row_end,
        kernel_col_start:kernel_col_end,
    ]


def _build_soa_spatial_masks(
    state: BattleState,
    *,
    viewer: int,
    soa_state: ObservationSoA,
) -> np.ndarray:
    masks = np.zeros(ACTION_MASK_SHAPE, dtype=bool)
    _, tower_alive = soa_state.viewer_towers_view(viewer)
    for slot, card_name in enumerate(state.players[viewer].hand[:4]):
        policy_name = policy_card_name(card_name)
        if policy_name is None:
            continue
        masks[slot] = _cached_spatial_mask(
            policy_name,
            not tower_alive[0, 0],
            not tower_alive[0, 2],
            not tower_alive[1, 0],
            not tower_alive[1, 2],
        )
    return np.ascontiguousarray(masks, dtype=bool)


def _build_soa_global_vector(
    state: BattleState,
    ruleset: Ruleset,
    *,
    viewer: int,
    memory: ObservationMemory,
    soa_state: ObservationSoA,
    allow_unrepresented_hand: bool,
) -> np.ndarray:
    own = state.players[viewer]
    tower_hp, tower_alive = soa_state.viewer_towers_view(viewer)
    regulation_us = ruleset.match.regulation_us
    total_duration_us = regulation_us + ruleset.match.overtime_us
    total_remaining_s = max(
        0.0,
        (total_duration_us - state.elapsed_us) / 1_000_000.0,
    )
    overtime = state.phase.casefold() == "overtime"
    phase_end_us = total_duration_us if overtime else regulation_us
    time_left_s = max(0.0, (phase_end_us - state.elapsed_us) / 1_000_000.0)

    global_scalars = np.zeros(len(GLOBAL_SCALAR_FEATURES), dtype=np.float32)
    global_scalars[0] = _clip_unit((own.elixir_milli / 1_000.0) / MAX_ELIXIR)
    global_scalars[1] = _clip_unit(memory.opponent_elixir_est / MAX_ELIXIR)
    global_scalars[2] = _clip_unit(total_remaining_s / TOTAL_MATCH_SECONDS)
    global_scalars[3] = float(overtime)
    global_scalars[4] = float(own.king_active)
    global_scalars[5] = float(state.players[1 - viewer].king_active)
    global_scalars[6] = float(tower_alive[0, 0])
    global_scalars[7] = float(tower_alive[0, 2])
    global_scalars[8] = float(tower_alive[1, 0])
    global_scalars[9] = float(tower_alive[1, 2])
    global_scalars[10] = float(not tower_alive[1, 0])
    global_scalars[11] = float(not tower_alive[1, 2])
    own_left, own_king, own_right = _normalize_soa_tower_hp(tower_hp[0], ruleset)
    enemy_left, enemy_king, enemy_right = _normalize_soa_tower_hp(tower_hp[1], ruleset)
    global_scalars[12:15] = (own_left, own_king, own_right)
    global_scalars[15:18] = (enemy_left, enemy_king, enemy_right)

    if allow_unrepresented_hand:
        hand = [policy_card_name(card) or "" for card in own.hand[:4]]
    else:
        hand = [
            _require_policy_feature_name(card, context="viewer hand")
            for card in own.hand[:4]
        ]
    hand.extend([""] * (4 - len(hand)))
    if own.draw_pile:
        next_card = (
            policy_card_name(own.draw_pile[0]) or ""
            if allow_unrepresented_hand
            else _require_policy_feature_name(own.draw_pile[0], context="viewer next card")
        )
    else:
        next_card = ""
    seen_ids = sorted(
        {
            int(CARD_METADATA[name]["id"])
            for name in memory.seen_opponent_cards
            if name in CARD_METADATA and isinstance(CARD_METADATA[name].get("id"), int)
        }
    )
    # The feature-stack helpers each allocate and then concatenate several
    # one-hot vectors.  The contract is fixed-width, so fill one output row
    # directly and reuse immutable card encodings from bounded caches.  This
    # preserves the exact feature ordering/dtype while removing a handful of
    # temporary arrays from every viewer projection.
    vector = np.empty(GLOBAL_VECTOR_SHAPE, dtype=np.float32)
    scalar_width = len(GLOBAL_SCALAR_FEATURES)
    vector[:scalar_width] = global_scalars
    hand_offset = scalar_width
    hand_width = 4 * CARD_COUNT
    vector[hand_offset : hand_offset + hand_width] = _cached_hand_encoding(tuple(hand))
    next_offset = hand_offset + hand_width
    vector[next_offset : next_offset + CARD_COUNT] = _cached_one_hot_card(next_card)
    seen_offset = next_offset + CARD_COUNT
    vector[seen_offset : seen_offset + CARD_COUNT] = _cached_seen_encoding(tuple(seen_ids))
    return vector


@lru_cache(maxsize=512)
def _cached_one_hot_card(card_name: str) -> np.ndarray:
    """Return one immutable card row for the fixed global-vector ABI."""

    encoded = np.ascontiguousarray(one_hot_card(card_name), dtype=np.float32)
    encoded.setflags(write=False)
    return encoded


@lru_cache(maxsize=512)
def _cached_hand_encoding(hand: tuple[str, ...]) -> np.ndarray:
    """Return the four-slot hand block without per-slot concatenation."""

    padded = hand + ("",) * max(0, 4 - len(hand))
    encoded = np.empty(4 * CARD_COUNT, dtype=np.float32)
    for slot in range(4):
        start = slot * CARD_COUNT
        encoded[start : start + CARD_COUNT] = _cached_one_hot_card(padded[slot])
    encoded.setflags(write=False)
    return encoded


@lru_cache(maxsize=512)
def _cached_seen_encoding(seen_ids: tuple[int, ...]) -> np.ndarray:
    """Return the immutable seen-card block for one sorted ID tuple."""

    encoded = np.zeros(CARD_COUNT, dtype=np.float32)
    for card_id in seen_ids:
        encoded[card_id] = 1.0
    encoded.setflags(write=False)
    return encoded


def battle_state_to_observed_game_state(
    state: BattleState,
    ruleset: Ruleset,
    *,
    viewer: int,
    memory: ObservationMemory,
    allow_unrepresented_hand: bool = False,
) -> GameState:
    """Build the lossy DTO used by the existing vision feature builders.

    When ``allow_unrepresented_hand`` is enabled, unsupported hand/next-card
    forms become empty public slots.  This is only appropriate for the V2
    bridge, where the actor's action vocabulary is independently masked and
    simulator-side opponents retain authoritative state.
    """

    _validate_viewer(viewer)
    _validate_state_ruleset(state, ruleset)
    if type(allow_unrepresented_hand) is not bool:
        raise TypeError("allow_unrepresented_hand must be boolean")
    if memory.viewer != viewer:
        raise ValueError(f"observation memory belongs to viewer {memory.viewer}, not {viewer}")
    if len(state.players) != 2:
        raise ValueError("vision_v1_exact requires exactly two players")

    own = state.players[viewer]
    if allow_unrepresented_hand:
        own_hand = [policy_card_name(card) or "" for card in own.hand[:4]]
    else:
        own_hand = [
            _require_policy_feature_name(card, context="viewer hand")
            for card in own.hand[:4]
        ]
    own_hand.extend([""] * (4 - len(own_hand)))
    if own.draw_pile:
        next_card = (
            policy_card_name(own.draw_pile[0]) or ""
            if allow_unrepresented_hand
            else _require_policy_feature_name(own.draw_pile[0], context="viewer next card")
        )
    else:
        next_card = ""

    tower_hp, tower_alive = _viewer_tower_state(state, viewer)
    princess_towers = PrincessTowerState(
        own_left_alive=tower_alive[0]["left"],
        own_right_alive=tower_alive[0]["right"],
        enemy_left_alive=tower_alive[1]["left"],
        enemy_right_alive=tower_alive[1]["right"],
    )

    regulation_us = ruleset.match.regulation_us
    total_duration_us = regulation_us + ruleset.match.overtime_us
    total_remaining_s = max(0.0, (total_duration_us - state.elapsed_us) / 1_000_000.0)
    overtime = state.phase.casefold() == "overtime"
    phase_end_us = total_duration_us if overtime else regulation_us
    time_left_s = max(0.0, (phase_end_us - state.elapsed_us) / 1_000_000.0)

    # ``CARD_COUNT`` in the existing feature stack already has stable IDs for
    # the complete top-level card catalog.  Keep the eight player-deck IDs
    # pinned for action/hand encoding, while allowing the opponent seen-card
    # channel to carry every eligible base card without changing tensor shape.
    seen_ids = sorted(
        {
            int(CARD_METADATA[name]["id"])
            for name in memory.seen_opponent_cards
            if name in CARD_METADATA and isinstance(CARD_METADATA[name].get("id"), int)
        }
    )
    own_units: list[Match] = []
    enemy_units: list[Match] = []
    for uid in sorted(state.entities):
        entity = state.entities[uid]
        if (
            not is_public_observation_entity(entity)
            or _is_tower(entity)
            or entity.kind == "spell"
        ):
            continue
        match = _entity_to_match(entity, viewer)
        # A known entity can leave the finite vision arena before the
        # authoritative engine removes it.  That is an out-of-view entity,
        # not an unsupported policy form; the raster contract should omit it
        # rather than make a long evaluation crash.  Unknown forms still raise
        # from ``_entity_to_match`` below.
        if match is None:
            continue
        (own_units if entity.owner == viewer else enemy_units).append(match)

    return GameState(
        hud=HudState(
            time_left_s=time_left_s,
            overtime=overtime,
            elixir_self=own.elixir_milli / 1_000.0,
            hand_cards=own_hand,
            next_card=next_card,
            tower_hp_self=[
                tower_hp[0]["left"],
                tower_hp[0]["king"],
                tower_hp[0]["right"],
            ],
            tower_hp_enemy=[
                tower_hp[1]["left"],
                tower_hp[1]["king"],
                tower_hp[1]["right"],
            ],
            princess_towers=princess_towers,
        ),
        total_remaining_s=total_remaining_s,
        own_units=own_units,
        enemy_units=enemy_units,
        seen_enemy_cards=seen_ids,
        elixir_enemy_est=memory.opponent_elixir_est,
        own_king_active=own.king_active,
        enemy_king_active=state.players[1 - viewer].king_active,
        started=_phase_is_active(state.phase),
    )


def policy_card_name(card_id_or_alias: str | None) -> str | None:
    """Resolve only forms represented by the existing policy vocabulary."""

    if not isinstance(card_id_or_alias, str) or not card_id_or_alias.strip():
        return None
    return _cached_policy_card_name(card_id_or_alias)


@lru_cache(maxsize=512)
def _cached_policy_card_name(card_id_or_alias: str) -> str | None:
    normalized = normalize_identifier(card_id_or_alias)
    normalized = _POLICY_ALIASES.get(normalized, normalized)
    return normalized if normalized in BASE_POLICY_CARD_IDS else None


def policy_card_id(card_id_or_alias: str | None) -> int | None:
    name = policy_card_name(card_id_or_alias)
    return None if name is None else BASE_POLICY_CARD_IDS[name]


def decode_policy_action(action: PolicyAction, *, viewer: int = 0) -> SimAction:
    """Decode a legacy policy action into authoritative world coordinates."""

    _validate_viewer(viewer)
    kind = action.kind.strip().casefold().replace("_", "-")
    if kind in {"wait", "noop", "no-op"}:
        if action.card_idx is not None or action.cell is not None:
            raise ValueError("wait action must not carry a card slot or cell")
        return WaitAction(viewer)
    if kind != "play":
        raise ValueError(f"unsupported policy action kind: {action.kind!r}")
    if action.card_idx is None or not 0 <= action.card_idx < 4:
        raise ValueError("play action card_idx must be a hand slot in [0, 3]")
    if action.cell is None:
        raise ValueError("play action requires a (column, row) cell")
    validate_cell(action.cell)
    world_cell = mirror_cell(action.cell) if viewer == 1 else action.cell
    return PlayCardAction(viewer, action.card_idx, world_cell)


def encode_sim_action(action: SimAction, *, viewer: int = 0) -> PolicyAction:
    """Encode an authoritative base-policy action into viewer coordinates."""

    _validate_viewer(viewer)
    if action.player != viewer:
        raise ValueError(f"cannot encode player {action.player} action for viewer {viewer}")
    if isinstance(action, WaitAction):
        return PolicyAction(kind="Wait")
    if isinstance(action, PlayCardAction):
        if not 0 <= action.card_slot < 4:
            raise ValueError("play action card_slot must be in [0, 3]")
        validate_cell(action.cell)
        relative_cell = mirror_cell(action.cell) if viewer == 1 else action.cell
        return PolicyAction(kind="Play", card_idx=action.card_slot, cell=relative_cell)
    if isinstance(action, UseAbilityAction):
        raise ValueError("vision policy Action has no lossless ability representation")
    raise TypeError(f"unsupported simulator action: {type(action).__name__}")


# Readable aliases for callers that treat the vision policy as an adapter.
domain_action_to_sim_action = decode_policy_action
sim_action_to_domain_action = encode_sim_action


def _validate_array(name: str, value: np.ndarray, shape: tuple[int, ...], dtype: np.dtype) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")


def _validate_viewer(viewer: int) -> None:
    if viewer not in (0, 1):
        raise ValueError(f"viewer must be 0 or 1, got {viewer!r}")


def _validate_state_ruleset(state: BattleState, ruleset: Ruleset) -> None:
    if state.ruleset_id != ruleset.ruleset_id:
        raise ValueError(
            f"state ruleset ID {state.ruleset_id!r} does not match {ruleset.ruleset_id!r}"
        )
    if state.ruleset_hash != ruleset.content_hash:
        raise ValueError("state ruleset hash does not match the supplied immutable ruleset")


def _require_policy_feature_name(card_id_or_alias: str | None, *, context: str) -> str:
    name = policy_card_name(card_id_or_alias)
    if name is None:
        raise UnsupportedPolicyFormError(
            f"vision_v1_exact cannot represent {context} form {card_id_or_alias!r}"
        )
    return name


def _phase_is_active(phase: str) -> bool:
    normalized = normalize_identifier(phase)
    return normalized not in _NON_PLAYING_PHASES


def _elixir_segments(ruleset: Ruleset) -> tuple[tuple[int, int], ...]:
    """Return public regeneration-rate segments through maximum overtime."""

    regulation = ruleset.match.regulation_us
    overtime = ruleset.match.overtime_us
    one_minute = 60_000_000
    regulation_double_start = max(0, regulation - one_minute)
    overtime_end = regulation + overtime
    overtime_triple_start = max(regulation, overtime_end - one_minute)
    return (
        (regulation_double_start, ruleset.match.normal_elixir_interval_us),
        (overtime_triple_start, ruleset.match.double_elixir_interval_us),
        (overtime_end, ruleset.match.triple_elixir_interval_us),
        (2**63 - 1, ruleset.match.triple_elixir_interval_us),
    )


def _lookup_card_definition(ruleset: Ruleset, card_name: str) -> CardDefinition | None:
    try:
        return ruleset.card(card_name)
    except (KeyError, TypeError, ValueError):
        policy_name = policy_card_name(card_name)
        if policy_name is None:
            return None
        try:
            return ruleset.card(policy_name)
        except (KeyError, TypeError, ValueError):
            return None


def _viewer_position(entity: EntityState, viewer: int) -> tuple[int, int]:
    if viewer == 0:
        return entity.x_mtile, entity.y_mtile
    return mirror_position(entity.x_mtile, entity.y_mtile)


def _is_tower(entity: EntityState) -> bool:
    return entity.kind == "tower"


def _entity_to_match(entity: EntityState, viewer: int) -> Match | None:
    normalized = normalize_identifier(entity.card_id)
    card_name = normalized if normalized in CARD_METADATA else _ENTITY_FEATURE_ALIASES.get(normalized)
    if card_name is None and normalized == "goblin-gang-goblin":
        card_name = "goblins"
    if card_name is None:
        raise UnsupportedPolicyFormError(
            f"vision_v1_exact cannot represent visible entity form {entity.card_id!r}"
        )
    x_mtile, y_mtile = _viewer_position(entity, viewer)
    cell = position_to_cell(x_mtile, y_mtile)
    if cell is None:
        return None
    center_x, center_y = ACTION_GRID.cell_to_norm_center(*cell)
    hp_fraction = 0.0 if entity.max_hp <= 0 else min(1.0, max(0.0, entity.hp / entity.max_hp))
    detection = Detection(
        track_id=entity.uid,
        class_name=card_name,
        team="ally" if entity.owner == viewer else "enemy",
        confidence=1.0,
        x1=center_x,
        y1=center_y,
        x2=center_x,
        y2=center_y,
        center_x=center_x,
        center_y=center_y,
        estimated_hp=hp_fraction,
    )
    return Match(troop=detection, bar=None)


def _viewer_tower_state(
    state: BattleState,
    viewer: int,
) -> tuple[list[dict[str, int]], list[dict[str, bool]]]:
    hp = [
        {"left": 0, "king": 0, "right": 0},
        {"left": 0, "king": 0, "right": 0},
    ]
    alive = [
        {"left": False, "king": False, "right": False},
        {"left": False, "king": False, "right": False},
    ]
    for uid in sorted(state.entities):
        entity = state.entities[uid]
        if not _is_tower(entity):
            continue
        relative_team = 0 if entity.owner == viewer else 1
        x_mtile, _ = _viewer_position(entity, viewer)
        if entity.role == "king":
            role = "king"
        else:
            role = "left" if x_mtile < GRID_COLS * 1_000 // 2 else "right"
        hp[relative_team][role] = max(0, entity.hp)
        alive[relative_team][role] = bool(entity.alive and entity.hp > 0)
    return hp, alive


def _build_spatial_masks(observed: GameState) -> np.ndarray:
    masks = np.zeros(ACTION_MASK_SHAPE, dtype=bool)
    tower_flags = observed.hud.princess_towers.as_deploy_kwargs()
    for slot, card_name in enumerate(observed.hud.hand_cards[:4]):
        policy_name = policy_card_name(card_name)
        if policy_name is None:
            continue
        masks[slot] = _cached_spatial_mask(
            policy_name,
            tower_flags["own_left_princess_down"],
            tower_flags["own_right_princess_down"],
            tower_flags["enemy_left_princess_down"],
            tower_flags["enemy_right_princess_down"],
        )
    return np.ascontiguousarray(masks, dtype=bool)


@lru_cache(maxsize=128)
def _cached_spatial_mask(
    card_name: str,
    own_left_princess_down: bool,
    own_right_princess_down: bool,
    enemy_left_princess_down: bool,
    enemy_right_princess_down: bool,
) -> np.ndarray:
    """Cache the legacy static deployment mask by its complete input."""

    mask = np.ascontiguousarray(
        get_card_deploy_mask(
            card_name,
            own_left_princess_down=own_left_princess_down,
            own_right_princess_down=own_right_princess_down,
            enemy_left_princess_down=enemy_left_princess_down,
            enemy_right_princess_down=enemy_right_princess_down,
        ),
        dtype=bool,
    )
    mask.setflags(write=False)
    return mask


def _build_legal_play(
    state: BattleState,
    ruleset: Ruleset,
    observed: GameState,
    spatial_masks: np.ndarray,
    *,
    viewer: int,
    legality_callback: LegalityCallback | None,
) -> np.ndarray:
    legal = np.zeros(ACTION_MASK_SHAPE, dtype=bool)
    if state.terminal or not _phase_is_active(state.phase):
        return legal

    elixir_milli = state.players[viewer].elixir_milli
    for slot, raw_card_name in enumerate(state.players[viewer].hand[:4]):
        policy_name = policy_card_name(raw_card_name)
        if policy_name is None:
            continue
        card = _lookup_card_definition(ruleset, raw_card_name)
        if card is None or card.elixir_milli > elixir_milli:
            continue
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                relative_cell = (col, row)
                world_cell = mirror_cell(relative_cell) if viewer == 1 else relative_cell
                action = PlayCardAction(viewer, slot, world_cell)
                allowed = (
                    legality_callback(state, action)
                    if legality_callback is not None
                    else _conservative_cell_is_legal(card, viewer, world_cell)
                )
                if allowed:
                    legal[slot, row, col] = True
    return np.ascontiguousarray(legal, dtype=bool)


def _build_legal_play_from_cells(
    state: BattleState,
    *,
    viewer: int,
    legal_action_cells_callback: LegalActionCellsCallback,
    ruleset: Ruleset | None = None,
    soa_state: ObservationSoA | None = None,
) -> np.ndarray:
    """Build the policy mask from an exact, already-batched legality query."""

    legal = np.zeros(ACTION_MASK_SHAPE, dtype=bool)
    if state.terminal or not _phase_is_active(state.phase):
        return legal

    legal_cells_by_slot = None
    legal_masks_by_slot = None
    if (
        ruleset is not None
        and soa_state is not None
        and _is_engine_legal_action_cells_callback(legal_action_cells_callback)
    ):
        legal_masks_by_slot = soa_state.legal_action_masks_if_static(
            state,
            ruleset,
            viewer,
        )
    if legal_masks_by_slot is not None:
        for slot, raw_card_name in enumerate(state.players[viewer].hand[:4]):
            if policy_card_name(raw_card_name) is None:
                continue
            if viewer == 0:
                legal[slot] = legal_masks_by_slot[slot]
            else:
                legal[slot] = legal_masks_by_slot[slot][::-1, ::-1]
        return np.ascontiguousarray(legal, dtype=bool)

    if legal_cells_by_slot is None:
        legal_cells_by_slot = legal_action_cells_callback(state, viewer)
    if len(legal_cells_by_slot) != 4:
        raise ValueError("legal action cell callback must return four hand-slot rows")
    for slot, raw_card_name in enumerate(state.players[viewer].hand[:4]):
        # V2 can expose an opponent hand containing cards outside the fixed
        # actor vocabulary.  The engine callback is authoritative about
        # simulator legality, but those forms must remain empty in this mask.
        if policy_card_name(raw_card_name) is None:
            continue
        for world_col, world_row in legal_cells_by_slot[slot]:
            if viewer == 1:
                col = GRID_COLS - 1 - world_col
                row = GRID_ROWS - 1 - world_row
            else:
                col, row = world_col, world_row
            legal[slot, row, col] = True
    return np.ascontiguousarray(legal, dtype=bool)


def _is_engine_legal_action_cells_callback(callback: LegalActionCellsCallback) -> bool:
    """Identify the built-in engine callback without changing custom hooks."""

    from .engine import BattleEngine

    return getattr(callback, "__func__", None) is BattleEngine.legal_action_cells


def _conservative_cell_is_legal(
    card: CardDefinition,
    player: int,
    world_cell: tuple[int, int],
) -> bool:
    mechanics = getattr(card, "mechanics", {})
    placement = mechanics.get("placement_class") if hasattr(mechanics, "get") else None
    if card.kind == "building":
        footprint_size = int(mechanics.get("building_footprint_size") or 3)
        return building_footprint_fits(player, world_cell, footprint_size)
    if card.kind == "spell":
        if placement == "spell_anywhere" or (placement is None and card.card_id != "log"):
            return is_spell_cell(world_cell)
        _, row = world_cell
        return is_spell_cell(world_cell) and (row >= 17 if player == 0 else row <= 14)
    return is_basic_deploy_cell(player, world_cell)
