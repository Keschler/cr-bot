"""Deterministic short-state curriculum environments for Phase 1.

The strategic curriculum originally attached the Phase-1 percentages only to
opponent/deck labels while every lane still started as a complete empty-arena
match.  This module makes those labels executable without introducing a
teacher action: it creates a public, simulator-valid initial state, lets the
actor choose every learner action, and scores only resulting state changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

try:
    from ..actions import PlayCardAction
    from ..fixed import DeterministicRng
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.actions import PlayCardAction
    from simulator.fixed import DeterministicRng


BASIC_SCENARIO_SCHEMA_VERSION = 1
BASIC_SCENARIO_REWARD_VERSION = "basic-mechanics-state-potential-v2"
BASIC_MECHANICS_SOURCES = (
    "isolated-offense",
    "ground-defense",
    "air-defense",
    "spell-situations",
    "kiting-cycling-elixir",
)
_BASIC_SOURCE_SET = frozenset(BASIC_MECHANICS_SOURCES)


class BasicScenarioError(ValueError):
    """Raised when a short curriculum state cannot be represented safely."""


def _stable_seed(seed: int, *parts: object) -> int:
    if type(seed) is not int:
        raise BasicScenarioError("scenario seed must be an integer")
    payload = "\x1f".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _source(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BasicScenarioError("source must be a non-empty string")
    normalized = value.strip().casefold().replace("_", "-").replace(" ", "-")
    if normalized not in _BASIC_SOURCE_SET:
        raise BasicScenarioError(f"unsupported basic-mechanics source: {value!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class BasicScenarioConfig:
    """One lane's deterministic state-generation and result objective."""

    source: str
    target_player: int
    decision_limit: int = 64
    state_reward_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _source(self.source))
        if type(self.target_player) is not int or self.target_player not in (0, 1):
            raise BasicScenarioError("target_player must be 0 or 1")
        if type(self.decision_limit) is not int or self.decision_limit <= 0:
            raise BasicScenarioError("decision_limit must be a positive integer")
        if (
            isinstance(self.state_reward_weight, bool)
            or not isinstance(self.state_reward_weight, (int, float))
            or not isfinite(float(self.state_reward_weight))
            or float(self.state_reward_weight) <= 0.0
        ):
            raise BasicScenarioError(
                "state_reward_weight must be a finite positive number"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": BASIC_SCENARIO_SCHEMA_VERSION,
            "source": self.source,
            "target_player": self.target_player,
            "decision_limit": self.decision_limit,
            "reward_version": BASIC_SCENARIO_REWARD_VERSION,
            "state_reward_weight": float(self.state_reward_weight),
            "learner_action_source": "public-actor",
            "success_definition": "resulting-game-state",
        }


def phase_one_rehearsal_source(episode_index: int) -> str:
    """Select a concrete Phase-1 state for later rehearsal deterministically."""

    if type(episode_index) is not int or episode_index < 0:
        raise BasicScenarioError("episode_index must be a non-negative integer")
    return BASIC_MECHANICS_SOURCES[episode_index % len(BASIC_MECHANICS_SOURCES)]


def basic_scenario_source(
    sampling_source: str | None,
    *,
    episode_index: int,
) -> str | None:
    """Resolve curriculum provenance into an executable short-state kind."""

    if sampling_source in _BASIC_SOURCE_SET:
        return str(sampling_source)
    if sampling_source == "phase-1-rehearsal":
        return phase_one_rehearsal_source(episode_index)
    return None


def _shuffled_deck(
    deck: Sequence[str],
    rng: DeterministicRng,
    required_hand: Iterable[str],
) -> tuple[list[str], list[str]]:
    cards = list(deck)
    rng.shuffle(cards)
    required = [card for card in dict.fromkeys(required_hand) if card in cards]
    hand_size = 4
    for card in required[:hand_size]:
        index = cards.index(card)
        if index >= hand_size:
            replace_index = rng.randbelow(hand_size)
            while cards[replace_index] in required:
                replace_index = (replace_index + 1) % hand_size
            cards[index], cards[replace_index] = cards[replace_index], cards[index]
    hand = cards[:hand_size]
    rng.shuffle(hand)
    return hand, cards[hand_size:]


def _required_target_cards(source: str, deck: Sequence[str]) -> tuple[str, ...]:
    preferences: Mapping[str, tuple[str, ...]] = {
        "isolated-offense": ("hog-rider",),
        "ground-defense": ("cannon", "musketeer"),
        "air-defense": ("musketeer", "fireball"),
        "spell-situations": ("fireball", "log"),
        "kiting-cycling-elixir": ("ice-golem", "cannon", "skeletons"),
    }
    return tuple(card for card in preferences[source] if card in deck)


def _candidate_setup_cards(
    environment: Any,
    source: str,
    opponent_deck: Sequence[str],
    rng: DeterministicRng,
) -> tuple[str, ...]:
    definitions = [environment.engine.ruleset.card(card) for card in opponent_deck]
    bodies = [card for card in definitions if card.kind in {"troop", "building"}]
    ground = [
        card
        for card in bodies
        if str(card.mechanics.get("movement_layer") or "ground") == "ground"
        and card.kind == "troop"
    ]
    air = [
        card
        for card in bodies
        if str(card.mechanics.get("movement_layer") or "ground") == "air"
    ]
    if source == "isolated-offense":
        return ()
    if source == "air-defense":
        candidates = air or bodies
        count = 1
    elif source == "spell-situations":
        candidates = bodies
        count = min(3, len(candidates))
    else:
        candidates = ground or bodies
        count = 1
    if not candidates:
        raise BasicScenarioError(
            f"opponent deck has no setup body for {source}: {tuple(opponent_deck)!r}"
        )
    # Prefer meaningful threats while varying among the top candidates.  This
    # selects the state, not the learner's response.
    ordered = sorted(candidates, key=lambda card: (-card.elixir_milli, card.card_id))
    offset = rng.randbelow(min(3, len(ordered)))
    rotated = ordered[offset:] + ordered[:offset]
    return tuple(card.card_id for card in rotated[:count])


def _put_card_in_hand(state: Any, player: int, card_id: str) -> int:
    player_state = state.players[player]
    if card_id in player_state.hand:
        return player_state.hand.index(card_id)
    try:
        draw_index = player_state.draw_pile.index(card_id)
    except ValueError as error:  # pragma: no cover - validated deck invariant
        raise BasicScenarioError(f"setup card {card_id!r} is absent from the deck") from error
    slot = 0
    player_state.hand[slot], player_state.draw_pile[draw_index] = (
        player_state.draw_pile[draw_index],
        player_state.hand[slot],
    )
    return slot


def _setup_cell(
    environment: Any,
    state: Any,
    player: int,
    card_id: str,
    *,
    lane_column: int,
    ordinal: int,
) -> tuple[int, int]:
    legal = environment.engine.legal_cells(state, player, card_id)
    if not legal:
        raise BasicScenarioError(f"setup card {card_id!r} has no legal placement")
    desired_row = 17 if player == 0 else 14
    desired_column = max(0, min(17, lane_column + (ordinal % 3) - 1))
    return min(
        legal,
        key=lambda cell: (
            abs(cell[0] - desired_column) + abs(cell[1] - desired_row),
            abs(cell[0] - desired_column),
            abs(cell[1] - desired_row),
            cell[1],
            cell[0],
        ),
    )


def _tower_totals(state: Any) -> tuple[int, int]:
    totals = [0, 0]
    for entity in state.entities.values():
        if entity.kind == "tower":
            totals[entity.owner] += max(0, int(entity.hp))
    return totals[0], totals[1]


class BasicMechanicsScenarioEnv:
    """Wrap a simulator lane with deterministic short-state episodes."""

    def __init__(self, environment: Any, config: BasicScenarioConfig) -> None:
        if not isinstance(config, BasicScenarioConfig):
            raise TypeError("config must be a BasicScenarioConfig")
        if not hasattr(environment, "engine") or not hasattr(environment, "reset_v2"):
            raise TypeError("environment must expose the simulator V2 boundary")
        self.environment = environment
        self.config = config
        self._decision_count = 0
        self._episode_count = 0
        self._last_potential = 0.0
        self._initial_tower_hp = (0, 0)
        self._tower_reference_hp = (1, 1)
        self._threat_hp: dict[int, tuple[int, int]] = {}
        self._history_digest = hashlib.sha256(b"").hexdigest()
        self._sample_metadata: list[dict[str, object]] = []
        self._latest_metadata: dict[str, object] | None = None

    @property
    def state(self) -> Any:
        return self.environment.state

    @property
    def engine(self) -> Any:
        return self.environment.engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self.environment, name)

    def reset(self, **kwargs: Any) -> Any:
        self.environment.reset(**kwargs)
        self._configure_episode(int(kwargs.get("seed", 0)))
        return self.environment.observe()

    def reset_v2(self, **kwargs: Any) -> Any:
        self.environment.reset_v2(**kwargs)
        self._configure_episode(int(kwargs.get("seed", 0)))
        return self.environment.observe_v2()

    def observe(self) -> Any:
        return self.environment.observe()

    def observe_v2(self) -> Any:
        return self.environment.observe_v2()

    def observe_v2_for_viewer(self, viewer: int) -> Any:
        method = getattr(self.environment, "observe_v2_for_viewer", None)
        if callable(method):
            return method(viewer)
        return self.environment.observe_v2()[viewer]

    def step(self, actions: Any) -> Any:
        return self._scenario_step(self.environment.step(actions))

    def step_v2(self, actions: Any) -> Any:
        return self._scenario_step(self.environment.step_v2(actions))

    def step_v2_for_viewer(self, actions: Any, *, viewer: int) -> Any:
        method = getattr(self.environment, "step_v2_for_viewer", None)
        if callable(method):
            return self._scenario_step(method(actions, viewer=viewer))
        return self._scenario_step(self.environment.step_v2(actions))

    def _configure_episode(self, seed: int) -> None:
        state = self.state
        if state is None:  # pragma: no cover - reset invariant
            raise BasicScenarioError("base environment reset produced no state")
        source = self.config.source
        target = self.config.target_player
        opponent = 1 - target
        rng = DeterministicRng(_stable_seed(seed, source, target, self._episode_count))
        lane_column = (3, 14)[rng.randbelow(2)]

        target_state = state.players[target]
        hand, draw = _shuffled_deck(
            target_state.deck,
            rng,
            _required_target_cards(source, target_state.deck),
        )
        target_state.hand = hand
        target_state.draw_pile = draw
        target_state.next_card_cooldown_us = 0

        tower_hp_permille: dict[str, int] = {}
        for entity in sorted(state.entities.values(), key=lambda item: item.uid):
            if entity.kind != "tower" or entity.role == "king":
                continue
            permille = 650 + rng.randbelow(351)
            entity.hp = max(1, entity.max_hp * permille // 1_000)
            tower_hp_permille[str(entity.uid)] = permille

        setup_cards = _candidate_setup_cards(
            self.environment,
            source,
            state.players[opponent].deck,
            rng,
        )
        setup_placements: list[list[int]] = []
        before_uids = set(state.entities)
        for ordinal, card_id in enumerate(setup_cards):
            state.players[opponent].elixir_milli = self.engine.ruleset.match.max_elixir_milli
            state.players[opponent].elixir_remainder = 0
            state.players[opponent].next_card_cooldown_us = 0
            slot = _put_card_in_hand(state, opponent, card_id)
            cell = _setup_cell(
                self.environment,
                state,
                opponent,
                card_id,
                lane_column=lane_column,
                ordinal=ordinal,
            )
            result = self.engine.apply_actions(
                state,
                (PlayCardAction(opponent, slot, cell),),
            )
            if len(result) != 1 or not result[0].accepted:
                reason = None if not result else result[0].reason
                raise BasicScenarioError(
                    f"failed to deploy setup card {card_id!r}: {reason}"
                )
            setup_placements.append([cell[0], cell[1]])

        decision_ticks = int(
            getattr(
                self.environment,
                "decision_interval_ticks",
                getattr(getattr(self.environment, "environment", None), "decision_interval_ticks", 1),
            )
        )
        prelude_decisions = rng.randbelow(9) if setup_cards else rng.randbelow(3)
        prelude_ticks = prelude_decisions * max(1, decision_ticks)
        for _ in range(prelude_ticks):
            if state.terminal:
                raise BasicScenarioError("short-scenario prelude unexpectedly ended the match")
            self.engine.step(state, ())

        target_elixir_ranges = {
            "isolated-offense": (6_000, 10_000),
            "ground-defense": (5_000, 10_000),
            "air-defense": (5_000, 10_000),
            "spell-situations": (5_000, 10_000),
            "kiting-cycling-elixir": (3_000, 7_000),
        }
        low, high = target_elixir_ranges[source]
        target_state.elixir_milli = low + rng.randbelow(high - low + 1)
        target_state.elixir_remainder = 0
        opponent_state = state.players[opponent]
        opponent_state.elixir_milli = 2_000 + rng.randbelow(6_001)
        opponent_state.elixir_remainder = 0

        self._decision_count = 0
        self._initial_tower_hp = _tower_totals(state)
        tower_reference = [0, 0]
        for entity in state.entities.values():
            if entity.kind == "tower" and entity.role != "king":
                tower_reference[entity.owner] = max(
                    tower_reference[entity.owner],
                    int(entity.max_hp),
                )
        # Tower damage and removed setup-threat HP use the same Princess Tower
        # HP denominator. The old unit-fraction bonus could value killing one
        # cheap setup troop more than thousands of tower HP, allowing an
        # all-WAIT learner to "win" while its tower was being destroyed.
        self._tower_reference_hp = (
            max(1, tower_reference[0]),
            max(1, tower_reference[1]),
        )
        self._threat_hp = {
            uid: (max(0, int(state.entities[uid].hp)), max(1, int(state.entities[uid].max_hp)))
            for uid in sorted(set(state.entities) - before_uids)
            if state.entities[uid].owner == opponent and state.entities[uid].kind != "tower"
        }
        self._last_potential = self._state_potential()
        metadata: dict[str, object] = {
            **self.config.as_dict(),
            "episode_index": self._episode_count,
            "seed": seed,
            "lane": "left" if lane_column == 3 else "right",
            "target_hand": list(target_state.hand),
            "target_elixir_milli": target_state.elixir_milli,
            "opponent_elixir_milli": opponent_state.elixir_milli,
            "tower_hp_permille": tower_hp_permille,
            "setup_cards": list(setup_cards),
            "setup_placements": setup_placements,
            "prelude_ticks": prelude_ticks,
            "threat_uids": sorted(self._threat_hp),
            "initial_state_hash": state.state_hash(),
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self._history_digest = hashlib.sha256(
            bytes.fromhex(self._history_digest) + encoded
        ).hexdigest()
        self._latest_metadata = metadata
        if len(self._sample_metadata) < 8:
            self._sample_metadata.append(metadata)
        self._episode_count += 1
        cache_owner = self.environment
        while hasattr(cache_owner, "environment"):
            cache_owner = cache_owner.environment
        if hasattr(cache_owner, "_persistent_observation_cache"):
            cache_owner._persistent_observation_cache = None
        self.engine.validate_state(state)

    def _state_potential(self) -> float:
        state = self.state
        target = self.config.target_player
        opponent = 1 - target
        current_hp = _tower_totals(state)
        enemy_damage = (
            self._initial_tower_hp[opponent] - current_hp[opponent]
        ) / self._tower_reference_hp[opponent]
        own_damage = (
            self._initial_tower_hp[target] - current_hp[target]
        ) / self._tower_reference_hp[target]
        threat_progress = 0.0
        if self._threat_hp:
            removed_hp = 0
            for uid, (initial_hp, _maximum_hp) in self._threat_hp.items():
                entity = state.entities.get(uid)
                remaining = (
                    max(0, int(entity.hp))
                    if entity is not None and entity.alive
                    else 0
                )
                removed_hp += max(0, initial_hp - remaining)
            threat_progress = removed_hp / self._tower_reference_hp[target]
        # Threat removal is useful defensive progress, but tower preservation
        # remains the dominant objective. This scores resulting state only;
        # it does not prescribe a card, cell, or timing decision.
        defense_weight = 0.0 if self.config.source == "isolated-offense" else 0.25
        return enemy_damage - own_damage + defense_weight * threat_progress

    def _scenario_step(self, result: Any) -> Any:
        self._decision_count += 1
        potential = self._state_potential()
        delta = (potential - self._last_potential) * float(self.config.state_reward_weight)
        self._last_potential = potential
        rewards = [0.0, 0.0]
        target = self.config.target_player
        rewards[target] = delta
        rewards[1 - target] = -delta
        if result.terminated and not result.truncated:
            rewards[0] += float(result.rewards[0])
            rewards[1] += float(result.rewards[1])

        boundary = (
            self._decision_count >= self.config.decision_limit
            and not result.terminated
            and not result.truncated
        )
        info = dict(result.info)
        info.update(
            {
                "episode_kind": "basic-mechanics-short-scenario",
                "scenario_source": self.config.source,
                "scenario_decision": self._decision_count,
                "scenario_decision_limit": self.config.decision_limit,
                "scenario_reward_version": BASIC_SCENARIO_REWARD_VERSION,
                "scenario_state_potential": potential,
            }
        )
        terminated = bool(result.terminated)
        truncated = bool(result.truncated)
        if boundary:
            terminated = True
            truncated = False
            if potential > 1e-9:
                winner = target
                outcome = "win"
            elif potential < -1e-9:
                winner = 1 - target
                outcome = "loss"
            else:
                winner = None
                outcome = "draw"
            info.update(
                {
                    "winner": winner,
                    "terminal_reason": "basic_mechanics_scenario_horizon",
                    "scenario_outcome": outcome,
                }
            )
        return type(result)(
            observations=result.observations,
            rewards=(float(rewards[0]), float(rewards[1])),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def scenario_audit(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(),
            "episodes_generated": self._episode_count,
            "history_sha256": self._history_digest,
            "sampled_initial_states": list(self._sample_metadata),
            "latest": self._latest_metadata,
        }


__all__ = [
    "BASIC_MECHANICS_SOURCES",
    "BASIC_SCENARIO_REWARD_VERSION",
    "BASIC_SCENARIO_SCHEMA_VERSION",
    "BasicMechanicsScenarioEnv",
    "BasicScenarioConfig",
    "BasicScenarioError",
    "basic_scenario_source",
    "phase_one_rehearsal_source",
]
