"""Public-observation counter policy for the pinned deterministic opponent.

This module is a deployment-safe counterpart to :mod:`rl.expert`.  The expert
may inspect authoritative state while producing training demonstrations.  The
policy here deliberately does not: it uses only the public V2 observation,
including the public hand encoding, visible entity tokens, tower-health
scalars, and the legality mask.

The pinned opponent is intentionally predictable: it plays the cheapest
affordable card and alternates lanes.  A small stateful policy is therefore a
useful reliability guard while the neural actor is still being trained.  Its
only state is the number of this player's publicly observable card plays,
which is used to reproduce the opponent's alternating-lane cadence for the
fallback cycle.  It is not a simulator-state shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.domain.game_state import Action as PolicyAction
from cr_bot.features.channels import GLOBAL_SCALAR_IDX
from cr_bot.features.global_features import CARD_COUNT

try:
    from ..observation_v2 import ENTITY_TOKEN_FEATURES, PolicyObservationV2
except ImportError:  # pragma: no cover - top-level ``rl`` layout
    from simulator.observation_v2 import ENTITY_TOKEN_FEATURES, PolicyObservationV2


_FEATURE_INDEX = {name: index for index, name in enumerate(ENTITY_TOKEN_FEATURES)}
_SCALAR_COUNT = len(GLOBAL_SCALAR_IDX)
_HAND_OFFSET = _SCALAR_COUNT
_MAX_CARD_ID = max(
    int(metadata["id"])
    for metadata in CARD_METADATA.values()
    if isinstance(metadata, dict) and type(metadata.get("id")) is int
)

_CARD_IDS = {
    card_name: int(CARD_METADATA[card_name]["id"])
    for card_name in (
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
        "fireball",
        "log",
    )
}
_CARD_COSTS = {
    card_name: int(CARD_METADATA[card_name]["elixir_cost"])
    for card_name in _CARD_IDS
}
_CARD_BY_ID = {card_id: card_name for card_name, card_id in _CARD_IDS.items()}

_CANNON_CARD_ID = _CARD_IDS["cannon"]


def _feature(name: str) -> int:
    return _FEATURE_INDEX[name]


@dataclass(slots=True)
class PublicCounterController:
    """Choose legal local-coordinate actions from one public V2 stream.

    ``plays`` and the previous public hand are the only persistent policy
    state.  The hand transition lets the controller count cards that were
    actually accepted by the environment, which keeps its lane cadence
    correct when it is used as a DAgger teacher and the actor executes some
    actions instead.  A new controller should be created for each match; the
    callback adapter below does this automatically for newly-created
    environments.
    """

    plays: int = 0
    _last_hand: tuple[str | None, ...] | None = None

    def reset(self) -> None:
        """Reset the public action-history state for a new match."""

        self.plays = 0
        self._last_hand = None

    @staticmethod
    def _validate_observation(observation: Any) -> PolicyObservationV2:
        if not isinstance(observation, PolicyObservationV2):
            raise TypeError("public counter requires a PolicyObservationV2")
        return observation

    @staticmethod
    def _hand(observation: PolicyObservationV2) -> tuple[str | None, ...]:
        """Decode the actor's own four public one-hot hand segments."""

        vector = observation.global_vector
        result: list[str | None] = []
        for slot in range(4):
            start = _HAND_OFFSET + slot * CARD_COUNT
            segment = vector[start : start + CARD_COUNT]
            if segment.size != CARD_COUNT or float(segment.max(initial=0.0)) < 0.5:
                result.append(None)
                continue
            card_id = int(np.argmax(segment))
            result.append(_CARD_BY_ID.get(card_id))
        return tuple(result)

    @staticmethod
    def _enemy_pressure(
        observation: PolicyObservationV2,
    ) -> tuple[bool, int | None]:
        """Return whether a visible enemy troop crossed and its local lane."""

        tokens = observation.entity_tokens
        mask = observation.entity_mask
        crossed: list[tuple[float, int]] = []
        for index in np.flatnonzero(mask):
            row = tokens[int(index)]
            # V2 rows are viewer-local: side=1 is the opponent.  Buildings
            # and spells are not treated as a troop crossing the river.
            if row[_feature("side")] < 0.5:
                continue
            if row[_feature("is_building")] >= 0.5 or row[_feature("is_spell")] >= 0.5:
                continue
            if row[_feature("is_visible")] < 0.5:
                continue
            y = float(row[_feature("y")])
            if y < 0.47:
                continue
            x = float(row[_feature("x")])
            # Keep the most advanced crossed troop; ties prefer the lane with
            # the smaller x so the choice remains deterministic.
            crossed.append((y, 0 if x < (1.0 / 2.0) else 1))
        if not crossed:
            return False, None
        _, lane = max(crossed, key=lambda item: (item[0], -item[1]))
        return True, lane

    @staticmethod
    def _enemy_has_air_pressure(
        observation: PolicyObservationV2,
        *,
        minimum_y: float = 0.47,
    ) -> bool:
        """Return whether a visible opponent air troop is at ``minimum_y``."""

        tokens = observation.entity_tokens
        mask = observation.entity_mask
        for index in np.flatnonzero(mask):
            row = tokens[int(index)]
            if row[_feature("side")] < 0.5:
                continue
            if row[_feature("is_building")] >= 0.5 or row[_feature("is_spell")] >= 0.5:
                continue
            if row[_feature("is_visible")] < 0.5:
                continue
            if row[_feature("y")] < minimum_y:
                continue
            if row[_feature("is_air")] >= 0.5:
                return True
        return False

    @staticmethod
    def _enemy_immediate_pressure(observation: PolicyObservationV2) -> bool:
        """Return whether a visible enemy troop is close to an own tower."""

        tokens = observation.entity_tokens
        mask = observation.entity_mask
        for index in np.flatnonzero(mask):
            row = tokens[int(index)]
            if row[_feature("side")] < 0.5:
                continue
            if row[_feature("is_building")] >= 0.5 or row[_feature("is_spell")] >= 0.5:
                continue
            if row[_feature("is_visible")] < 0.5:
                continue
            # Viewer-local y increases toward the controller's own towers.
            # The conservative threshold leaves time for a counter-push after
            # a river crossing but switches to defense for a troop already in
            # the tower approach.
            if row[_feature("y")] >= StrategicCounterController._IMMEDIATE_PRESSURE_Y:
                return True
        return False

    @staticmethod
    def _has_live_cannon(observation: PolicyObservationV2) -> bool:
        tokens = observation.entity_tokens
        mask = observation.entity_mask
        expected = float(_CANNON_CARD_ID) / float(_MAX_CARD_ID)
        for index in np.flatnonzero(mask):
            row = tokens[int(index)]
            if row[_feature("side")] >= 0.5:
                continue
            if row[_feature("is_building")] < 0.5:
                continue
            if abs(float(row[_feature("card_id")]) - expected) > 0.01:
                continue
            if row[_feature("hp_fraction")] > 0.0:
                return True
        return False

    @staticmethod
    def _has_live_musketeer(observation: PolicyObservationV2) -> bool:
        """Return whether one of our visible Musketeers is still alive."""

        tokens = observation.entity_tokens
        mask = observation.entity_mask
        expected = float(_CARD_IDS["musketeer"]) / float(_MAX_CARD_ID)
        for index in np.flatnonzero(mask):
            row = tokens[int(index)]
            if row[_feature("side")] >= 0.5:
                continue
            if row[_feature("is_building")] >= 0.5:
                continue
            if abs(float(row[_feature("card_id")]) - expected) > 0.01:
                continue
            if row[_feature("hp_fraction")] > 0.0:
                return True
        return False

    @staticmethod
    def _weakest_enemy_tower(observation: PolicyObservationV2) -> str:
        vector = observation.global_vector
        left = float(vector[GLOBAL_SCALAR_IDX["tower_hp_enemy_left"]])
        right = float(vector[GLOBAL_SCALAR_IDX["tower_hp_enemy_right"]])
        return "left" if left <= right else "right"

    @staticmethod
    def _first_legal_cell(
        legal_play: np.ndarray,
        slot: int,
        preferred: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Return a preferred (column,row), recovering through the mask."""

        if not 0 <= slot < legal_play.shape[0]:
            return None
        col, row = preferred
        if (
            0 <= row < legal_play.shape[1]
            and 0 <= col < legal_play.shape[2]
            and bool(legal_play[slot, row, col])
        ):
            return preferred
        legal = np.argwhere(legal_play[slot])
        if legal.size == 0:
            return None
        # np.argwhere is row-major, matching the stable fallback used by the
        # simulator's deterministic controller.
        fallback_row, fallback_col = (int(value) for value in legal[0])
        return fallback_col, fallback_row

    def _play(
        self,
        observation: PolicyObservationV2,
        hand: tuple[str | None, ...],
        card_name: str,
        preferred: tuple[int, int],
    ) -> PolicyAction | None:
        for slot, held_card in enumerate(hand):
            if held_card != card_name:
                continue
            cell = self._first_legal_cell(observation.legal_play, slot, preferred)
            if cell is None:
                return None
            return PolicyAction(kind="Play", card_idx=slot, cell=cell)
        return None

    def _prepare_observation(
        self, observation: Any
    ) -> tuple[PolicyObservationV2, tuple[str | None, ...]]:
        """Validate and account for one public observation before acting."""

        observation = self._validate_observation(observation)
        hand = self._hand(observation)
        # A changed public hand is the observable consequence of the previous
        # accepted card play.  Count that transition before choosing the next
        # fallback lane; do not count the action merely proposed by this
        # controller, because DAgger may execute the actor's action instead.
        if self._last_hand is not None and hand != self._last_hand:
            self.plays += 1
        self._last_hand = hand
        return observation, hand

    def choose_action(self, observation: Any, *, player: int = 0) -> PolicyAction:
        """Choose a public, legality-masked action in viewer-local cells."""

        del player  # The observation is already in the target viewer's frame.
        observation, hand = self._prepare_observation(observation)
        crossed, enemy_lane = self._enemy_pressure(observation)
        cannon_alive = self._has_live_cannon(observation)

        # Once a visible troop is across the river, place Cannon in the center
        # and use the lane-specific defensive cells for the remaining cards.
        # These are local viewer coordinates, so player 1 needs no second
        # mirror; SimulatorEnv performs that boundary conversion exactly once.
        if crossed and not cannon_alive:
            cannon = self._play(observation, hand, "cannon", (8, 21))
            if cannon is not None:
                return cannon
            lane_col = 3 if enemy_lane == 0 else 14
            for card_name, row in (
                ("musketeer", 23),
                ("skeletons", 22),
                ("log", 21),
                ("fireball", 21),
            ):
                action = self._play(observation, hand, card_name, (lane_col, row))
                if action is not None:
                    return action

        # Fireball is the only reliable direct tower-damage finisher in the
        # small pinned deck.  The tower target is also public in the scalar
        # vector.  Coordinates are the viewer-local spell target cells.
        tower_lane = self._weakest_enemy_tower(observation)
        fireball_cell = (3, 6) if tower_lane == "left" else (14, 6)
        action = self._play(observation, hand, "fireball", fireball_cell)
        if action is not None:
            return action

        # Preserve elixir for Hog Rider whenever there is no immediate troop
        # to answer.  This prevents cheap-cycle cards from consuming the
        # attack budget before Hog becomes affordable.
        hog_held = "hog-rider" in hand
        if not crossed:
            action = self._play(observation, hand, "hog-rider", (3, 17))
            if action is not None:
                return action
            if hog_held:
                return PolicyAction(kind="Wait")

        # Reproduce the deterministic cycle's cheapest-card choice, but use
        # the public legality mask rather than assuming a placement succeeds.
        affordable_order = sorted(
            (
                (cost, slot, card_name)
                for slot, card_name in enumerate(hand)
                if card_name is not None
                for cost in (_CARD_COSTS.get(card_name, 99),)
            ),
            key=lambda item: (item[0], item[1]),
        )
        use_left = self.plays % 2 == 0
        for _cost, _slot, card_name in affordable_order:
            if card_name in {"hog-rider", "fireball"}:
                continue
            if card_name == "cannon":
                preferred = (8, 20)
            elif card_name == "log":
                preferred = (3 if use_left else 14, 19)
            else:
                preferred = (3 if use_left else 14, 23)
            action = self._play(observation, hand, card_name, preferred)
            if action is not None:
                return action
        return PolicyAction(kind="Wait")


class StrategicCounterController(PublicCounterController):
    """A stronger public-only demonstration policy for actor warm starts.

    The original :class:`PublicCounterController` intentionally mirrors the
    pinned deterministic opponent and is retained as a regression baseline.
    This variant is less tied to that opponent's cheap-card cadence: it uses
    Hog Rider as the primary pressure card, holds Fireball for a damaged
    tower or an observed crossing, and spends defensive cards when a visible
    troop is already threatening the player's side.

    It still sees only ``PolicyObservationV2``.  In particular, no
    authoritative simulator state, opponent hand, or hidden deck rotation is
    consulted.  The policy is useful for supervised warm starts and grave
    action/placement audits; it is not intended to replace PPO exploration or
    to be presented as learned strategy.
    """

    # Tower HP scalars are normalized to the initial tower HP by the V2
    # contract.  In this provisional ruleset a single Hog connection can be
    # decisive in a tiebreak, so Fireball must be available as soon as a
    # tower has taken any damage.  Waiting for 0.85 caused the teacher to
    # reserve Fireball until the final few decisions and produced a weak
    # imitation target.  The per-match baseline below still prevents a full
    # tower from being treated as damaged.
    _FINISH_TOWER_FRACTION = 0.99
    # Viewer-local ``y`` is normalized to the arena height and increases
    # toward the controller's own towers.  Reserve the immediate-response
    # path for a troop that is genuinely deep in the tower approach; a lower
    # threshold makes the teacher spend every cycle defending and starves
    # Hog counterpressure.
    _IMMEDIATE_PRESSURE_Y = 0.75
    # A damaged own tower is public state too.  The old policy waited for a
    # visible troop to reach the immediate-pressure threshold, which allowed
    # chip damage to accumulate until a tiebreak loss was already inevitable.
    _EMERGENCY_TOWER_FRACTION = 0.60

    def reset(self) -> None:
        super().reset()
        self._strategic_initial_tower_hp = None

    def _tower_fraction(
        self,
        observation: PolicyObservationV2,
        tower_lane: str,
    ) -> float:
        """Compare tower HP with this match's observed starting HP.

        The feature contract normalizes against a static catalog maximum,
        while the pinned Level-11 ruleset's current princess-tower HP is
        lower than that maximum.  An absolute threshold would therefore
        mistake a full-health tower for a damaged one.  Establishing a
        per-match public baseline avoids that version/level mismatch.
        """

        vector = observation.global_vector
        values = (
            float(vector[GLOBAL_SCALAR_IDX["tower_hp_enemy_left"]]),
            float(vector[GLOBAL_SCALAR_IDX["tower_hp_enemy_right"]]),
        )
        baseline = getattr(self, "_strategic_initial_tower_hp", None)
        if not isinstance(baseline, tuple) or len(baseline) != 2:
            baseline = tuple(max(1e-6, value) for value in values)
        else:
            baseline = tuple(
                max(previous, current, 1e-6)
                for previous, current in zip(baseline, values, strict=True)
            )
        self._strategic_initial_tower_hp = baseline
        current = values[0 if tower_lane == "left" else 1]
        return current / baseline[0 if tower_lane == "left" else 1]

    @classmethod
    def _critical_own_tower(
        cls,
        observation: PolicyObservationV2,
    ) -> int | None:
        """Return the viewer-local lane index of a damaged own tower.

        ``tower_hp_self_*`` is normalized to the pinned princess-tower
        maximum, so unlike the enemy-tower baseline helper above it can use a
        fixed fraction.  A zero value is treated as unavailable/dead rather
        than as an emergency: the simulator terminates when a princess tower
        is destroyed, and hand-built test observations may omit self HP.
        """

        vector = observation.global_vector
        values = (
            float(vector[GLOBAL_SCALAR_IDX["tower_hp_self_left"]]),
            float(vector[GLOBAL_SCALAR_IDX["tower_hp_self_right"]]),
        )
        candidates = [
            (value, lane)
            for value, lane in zip(values, (0, 1), strict=True)
            if value > 0.0
        ]
        if not candidates:
            return None
        value, lane = min(candidates, key=lambda item: (item[0], item[1]))
        if value > cls._EMERGENCY_TOWER_FRACTION:
            return None
        return lane

    @staticmethod
    def _pressure_cell(
        observation: PolicyObservationV2,
        lane: int | None,
    ) -> tuple[int, int]:
        """Convert the most advanced visible enemy token to a viewer cell."""

        tokens = observation.entity_tokens
        mask = observation.entity_mask
        candidates: list[tuple[float, float, float]] = []
        for index in np.flatnonzero(mask):
            row = tokens[int(index)]
            if row[_feature("side")] < 0.5:
                continue
            if row[_feature("is_building")] >= 0.5 or row[_feature("is_spell")] >= 0.5:
                continue
            if row[_feature("is_visible")] < 0.5:
                continue
            x = float(row[_feature("x")])
            y = float(row[_feature("y")])
            token_lane = 0 if x < 0.5 else 1
            if lane is not None and token_lane != lane:
                continue
            candidates.append((y, x, float(index)))
        if candidates:
            _y, x, _index = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
            return (
                max(0, min(17, int(round(x * 17.0)))),
                max(0, min(31, int(round(_y * 31.0)))),
            )
        return (3, 21) if lane != 1 else (14, 21)

    def choose_action(self, observation: Any, *, player: int = 0) -> PolicyAction:
        """Choose pressure-first public actions with explicit defensive recovery."""

        del player
        observation, hand = self._prepare_observation(observation)
        crossed, enemy_lane = self._enemy_pressure(observation)
        cannon_alive = self._has_live_cannon(observation)
        immediate = self._enemy_immediate_pressure(observation)
        air_pressure = self._enemy_has_air_pressure(observation)
        air_near_tower = self._enemy_has_air_pressure(observation, minimum_y=0.72)
        musketeer_alive = self._has_live_musketeer(observation)
        emergency_lane = self._critical_own_tower(observation)
        emergency = emergency_lane is not None
        # Own-tower HP activates emergency mode, but a visible crossed troop
        # identifies the lane that needs the answer.  Otherwise a damaged
        # tower on the opposite lane can make the controller defend the wrong
        # side while the active push continues unchecked.
        if enemy_lane is not None:
            lane = enemy_lane
        elif emergency_lane is not None:
            lane = emergency_lane
        else:
            lane = 0 if self.plays % 2 == 0 else 1

        emergency_response = emergency and (crossed or immediate or not cannon_alive)
        if emergency_response:
            # Once one own tower is below the recovery threshold, spending
            # the next action on pressure is worse than buying time when a
            # public troop is actually threatening the damaged lane.  A low
            # tower with no visible pressure is different: locking the
            # teacher into cheap defense there starves the counterpush and was
            # measurably worse against long beatdown matches.  A live Cannon
            # is not treated as sufficient by itself once a troop has
            # crossed; the cheap Ice Spirit fallback prevents a final-step
            # WAIT when the building cannot answer the threat.
            defensive_priority = (
                (
                    "musketeer",
                    "ice-spirit",
                    "fireball",
                    "ice-golem",
                    "skeletons",
                    "cannon",
                    "log",
                )
                if air_pressure
                else (
                    "cannon",
                    "musketeer",
                    "ice-spirit",
                    "ice-golem",
                    "skeletons",
                    "log",
                    "fireball",
                )
            )
            defensive_cells = {
                "cannon": (8, 21),
                "musketeer": (3 if lane == 0 else 14, 23),
                "ice-spirit": (3 if lane == 0 else 14, 22),
                "ice-golem": (3 if lane == 0 else 14, 23),
                "skeletons": (3 if lane == 0 else 14, 22),
                "log": (3 if lane == 0 else 14, 20),
                "fireball": self._pressure_cell(observation, lane),
            }
            for card_name in defensive_priority:
                if card_name == "cannon" and cannon_alive:
                    continue
                action = self._play(observation, hand, card_name, defensive_cells[card_name])
                if action is not None:
                    return action
            # Do not fall through to the offensive Hog/Fireball branches when
            # no defensive card is currently legal; preserve elixir instead.
            return PolicyAction(kind="Wait")

        if air_pressure and not musketeer_alive:
            # Air pressure is a separate threat from a ground troop crossing.
            # The public token already identifies a crossed air unit; waiting
            # for it to reach the deeper near-tower threshold can spend the
            # only affordable action on a ground cycle card.  Prefer the only
            # reusable ranged air answer as soon as the legality mask says it
            # can be played, even when a live Cannon is covering the ground.
            lane_col = 3 if (enemy_lane if enemy_lane is not None else 0) == 0 else 14
            air_defensive_cells = (
                ("musketeer", (lane_col, 23)),
                ("fireball", self._pressure_cell(observation, enemy_lane)),
                ("ice-spirit", (lane_col, 22)),
            )
            for card_name, cell in air_defensive_cells:
                action = self._play(observation, hand, card_name, cell)
                if action is not None:
                    return action
            # If Musketeer is in hand but not legal yet, keep the elixir for
            # it instead of launching a Hog into an unresolved air threat.
            if "musketeer" in hand and not immediate:
                return PolicyAction(kind="Wait")

        if crossed and not cannon_alive:
            # Cannon cannot target air.  Against an air crossing, deploy the
            # Musketeer first when it is available; otherwise fall back to the
            # center Cannon and let the remaining cards/king tower absorb the
            # provisional simulator's simplified air interaction.
            defensive_priority = (
                ("musketeer", "fireball", "cannon", "skeletons", "log")
                if air_pressure
                else ("cannon", "musketeer", "skeletons", "log", "fireball")
            )
            lane_col = 3 if lane == 0 else 14
            defensive_cells = {
                "cannon": (8, 21),
                "musketeer": (lane_col, 23),
                "skeletons": (lane_col, 22),
                "log": (lane_col, 20),
                "fireball": self._pressure_cell(observation, enemy_lane),
            }
            for card_name in defensive_priority:
                cell = defensive_cells[card_name]
                action = self._play(observation, hand, card_name, cell)
                if action is not None:
                    return action

        if crossed and immediate:
            # Once a troop is in the tower approach, spend the current action
            # on recovery even when a previous Cannon is still alive.  This
            # branch is intentionally after the no-Cannon placement above so
            # it can add a ranged/support answer instead of replacing a live
            # building.
            lane_col = 3 if lane == 0 else 14
            defensive_cells = (
                ("musketeer", (lane_col, 23)),
                ("skeletons", (lane_col, 22)),
                ("log", (lane_col, 20)),
                ("fireball", self._pressure_cell(observation, enemy_lane)),
            )
            for card_name, cell in defensive_cells:
                action = self._play(observation, hand, card_name, cell)
                if action is not None:
                    return action

        weakest = self._weakest_enemy_tower(observation)
        vector = observation.global_vector
        weakest_fraction = self._tower_fraction(observation, weakest)
        if 0.0 < weakest_fraction <= self._FINISH_TOWER_FRACTION:
            fireball_cell = (3, 6) if weakest == "left" else (14, 6)
            action = self._play(observation, hand, "fireball", fireball_cell)
            if action is not None:
                return action

        # This is the central 2.6-Hog sanity check: when a safe Hog is held,
        # do not burn its elixir on a one-cost cycle card.
        # A river crossing is not automatically an emergency.  With a live
        # Cannon (or a troop still far from the tower), keep the initiative
        # and send Hog instead of cycling four cheap cards into a draw.
        if not immediate:
            hog_cell = (3, 17) if weakest == "left" else (14, 17)
            action = self._play(observation, hand, "hog-rider", hog_cell)
            if action is not None:
                return action
            if "hog-rider" in hand:
                return PolicyAction(kind="Wait")

        affordable_order = sorted(
            (
                (_CARD_COSTS.get(card_name, 99), slot, card_name)
                for slot, card_name in enumerate(hand)
                if card_name is not None and card_name not in {"hog-rider", "fireball"}
            ),
            key=lambda item: (item[0], item[1]),
        )
        for _cost, _slot, card_name in affordable_order:
            if card_name == "cannon":
                preferred = (8, 20)
            elif card_name == "log":
                preferred = (3 if lane == 0 else 14, 19)
            else:
                preferred = (3 if lane == 0 else 14, 23)
            action = self._play(observation, hand, card_name, preferred)
            if action is not None:
                return action
        return PolicyAction(kind="Wait")


def _public_counter_callback(
    environment: Any,
    observation: Any,
    player: int,
    *,
    attribute: str,
    controller_type: type[PublicCounterController],
) -> PolicyAction:
    """Run a public controller with per-environment episode isolation."""

    controller = getattr(environment, attribute, None)
    if controller is None:
        controller = controller_type()
        setattr(environment, attribute, controller)
        setattr(environment, f"{attribute}_last_time", None)
    time_left = float(observation.global_vector[GLOBAL_SCALAR_IDX["time_left_norm"]])
    last_time = getattr(environment, f"{attribute}_last_time", None)
    if isinstance(last_time, float) and time_left > last_time + 1e-6:
        controller.reset()
    setattr(environment, f"{attribute}_last_time", time_left)
    return controller.choose_action(observation, player=player)


def public_counter_action(environment: Any, observation: Any, player: int) -> PolicyAction:
    """Collector callback for the public counter policy.

    ``environment`` is used only as a per-lane storage object.  No simulator
    state is read; action selection is entirely a function of ``observation``
    and the controller's public play count.
    """

    return _public_counter_callback(
        environment,
        observation,
        player,
        attribute="_public_counter_controller",
        controller_type=PublicCounterController,
    )


def strategic_counter_action(environment: Any, observation: Any, player: int) -> PolicyAction:
    """Collector callback for the public-only strategic warm-start teacher."""

    return _public_counter_callback(
        environment,
        observation,
        player,
        attribute="_strategic_counter_controller",
        controller_type=StrategicCounterController,
    )


def public_defensive_threat_observed(observation: Any) -> bool:
    """Return whether the public stream contains a defensive threat.

    This is a label-quality gate for teacher-guided training, not an action
    rule.  It deliberately uses only the same public observation available to
    the actor.  In particular, it prevents a teacher's generic opening Hog
    preference from overwriting a useful learner policy while retaining labels
    for visible crossings, air threats, immediate pressure, and damaged own
    towers.
    """

    observation = PublicCounterController._validate_observation(observation)
    crossed, _lane = PublicCounterController._enemy_pressure(observation)
    return bool(
        crossed
        or PublicCounterController._enemy_has_air_pressure(observation)
        or PublicCounterController._enemy_immediate_pressure(observation)
        or StrategicCounterController._critical_own_tower(observation) is not None
    )


__all__ = [
    "PublicCounterController",
    "StrategicCounterController",
    "public_counter_action",
    "public_defensive_threat_observed",
    "strategic_counter_action",
]
