"""Reproducible opponent decks and simulator-only strategy controllers.

This module is deliberately on the authoritative simulator side of the
observation boundary.  The controllers below receive ``BattleState`` and
``BattleEngine`` because they are opponents used by the simulator, not actor
policies.  A learner or actor should continue to receive only the public
observation produced by the environment.

The pool is intentionally self-contained so it can be used by training and
evaluation orchestration without changing the engine's pinned controller or
the policy implementation.  Every card placement goes through
``validate_action`` and, when a preferred location is unavailable, through
``legal_cells``.  Thus a strategy can be heuristic without emitting illegal
actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Callable, Mapping, Protocol

try:
    from ..actions import PlayCardAction, SimAction, WaitAction
    from ..engine import BattleEngine
    from ..ruleset import Ruleset, load_fixed_ruleset
    from ..roster import PLAYER_DECK
except ImportError:  # pragma: no cover - top-level ``rl`` layout
    from simulator.actions import PlayCardAction, SimAction, WaitAction
    from simulator.engine import BattleEngine
    from simulator.ruleset import Ruleset, load_fixed_ruleset
    from simulator.roster import PLAYER_DECK


DECK_SIZE = 8
OPPONENT_POOL_SCHEMA_VERSION = 1


class OpponentController(Protocol):
    """The controller shape accepted by :meth:`BattleEngine.run_match`."""

    def choose_action(self, engine: BattleEngine, state: Any, player: int) -> SimAction:
        ...


class OpponentPoolError(ValueError):
    """Raised when a pool request cannot be represented by a ruleset."""


def _stable_seed(seed: int, *parts: object) -> int:
    """Derive a stable 64-bit seed without using Python's salted hash()."""

    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _normalize_name(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpponentPoolError(f"{field} must be a non-empty string")
    return value.strip().casefold().replace("_", "-").replace(" ", "-")


def _validate_index(value: int, *, field: str) -> None:
    if type(value) is not int or value < 0:
        raise OpponentPoolError(f"{field} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class OpponentDeckSpec:
    """One canonical, simulator-valid opponent deck."""

    deck_id: str
    archetype: str
    cards: tuple[str, ...]
    tags: tuple[str, ...] = ()
    source: str = "curated"

    def __post_init__(self) -> None:
        if not isinstance(self.deck_id, str) or not self.deck_id.strip():
            raise OpponentPoolError("deck_id must be a non-empty string")
        if not isinstance(self.archetype, str) or not self.archetype.strip():
            raise OpponentPoolError("archetype must be a non-empty string")
        if len(self.cards) != DECK_SIZE:
            raise OpponentPoolError(f"cards must contain exactly {DECK_SIZE} cards")
        if any(not isinstance(card, str) or not card.strip() for card in self.cards):
            raise OpponentPoolError("cards must contain non-empty strings")
        if len(set(self.cards)) != len(self.cards):
            raise OpponentPoolError("cards must not contain duplicates")
        if len(set(self.tags)) != len(self.tags):
            raise OpponentPoolError("tags must not contain duplicates")
        if any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise OpponentPoolError("tags must contain non-empty strings")
        if not isinstance(self.source, str) or not self.source.strip():
            raise OpponentPoolError("source must be a non-empty string")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPPONENT_POOL_SCHEMA_VERSION,
            "deck_id": self.deck_id,
            "archetype": self.archetype,
            "cards": list(self.cards),
            "tags": list(self.tags),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class OpponentScenario:
    """A reproducible deck/controller selection for one episode."""

    episode_index: int
    selection_seed: int
    deck: OpponentDeckSpec
    strategy: str
    controller_seed: int

    def __post_init__(self) -> None:
        _validate_index(self.episode_index, field="episode_index")
        if type(self.selection_seed) is not int:
            raise OpponentPoolError("selection_seed must be an integer")
        if type(self.controller_seed) is not int:
            raise OpponentPoolError("controller_seed must be an integer")
        if not isinstance(self.deck, OpponentDeckSpec):
            raise OpponentPoolError("deck must be an OpponentDeckSpec")
        _normalize_name(self.strategy, field="strategy")

    def build_controller(self) -> OpponentController:
        """Construct a fresh, state-isolated controller for this scenario."""

        return make_opponent_controller(self.strategy, seed=self.controller_seed)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPPONENT_POOL_SCHEMA_VERSION,
            "episode_index": self.episode_index,
            "selection_seed": self.selection_seed,
            "deck": self.deck.as_dict(),
            "strategy": self.strategy,
            "controller_seed": self.controller_seed,
        }


# These are curated representatives, not claims that the simulator's
# provisional mechanics reproduce tournament-strength decks.  They make the
# training/evaluation distribution cover pressure, defense, beatdown, air,
# siege, and bait patterns while retaining a pinned deterministic regression
# deck.
_DECK_TEMPLATES: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "deterministic-cycle": (
        PLAYER_DECK,
    ),
    "aggressive-pressure": (
        (
            "hog-rider",
            "earthquake",
            "firecracker",
            "ice-spirit",
            "skeletons",
            "ice-golem",
            "log",
            "cannon",
        ),
        (
            "battle-ram",
            "bandit",
            "royal-ghost",
            "firecracker",
            "barbarian-barrel",
            "poison",
            "bats",
            "tesla",
        ),
        (
            "ram-rider",
            "wall-breakers",
            "miner",
            "spear-goblins",
            "fireball",
            "zap",
            "bomb-tower",
            "bats",
        ),
    ),
    "defensive-cycle": (
        (
            "x-bow",
            "tesla",
            "archers",
            "knight",
            "skeletons",
            "ice-spirit",
            "fireball",
            "log",
        ),
        (
            "mortar",
            "cannon",
            "musketeer",
            "valkyrie",
            "skeletons",
            "fireball",
            "log",
            "ice-spirit",
        ),
        (
            "goblin-drill",
            "bomb-tower",
            "archers",
            "knight",
            "goblin-gang",
            "poison",
            "log",
            "ice-spirit",
        ),
    ),
    "beatdown": (
        (
            "golem",
            "night-witch",
            "baby-dragon",
            "lumberjack",
            "tornado",
            "lightning",
            "mega-minion",
            "barbarians",
        ),
        (
            "giant",
            "witch",
            "musketeer",
            "mini-pekka",
            "bats",
            "fireball",
            "zap",
            "elixir-collector",
        ),
        (
            "electro-giant",
            "bowler",
            "electro-dragon",
            "tornado",
            "lightning",
            "mega-minion",
            "goblin-cage",
            "rage",
        ),
    ),
    "air-beatdown": (
        (
            "lava-hound",
            "balloon",
            "mega-minion",
            "baby-dragon",
            "skeleton-dragons",
            "miner",
            "arrows",
            "tombstone",
        ),
        (
            "lava-hound",
            "flying-machine",
            "skeleton-dragons",
            "mega-minion",
            "bats",
            "fireball",
            "tombstone",
            "zap",
        ),
    ),
    "siege-bait": (
        (
            "mortar",
            "goblin-barrel",
            "princess",
            "goblin-gang",
            "skeleton-army",
            "rocket",
            "log",
            "ice-spirit",
        ),
        (
            "x-bow",
            "tesla",
            "firecracker",
            "knight",
            "skeletons",
            "rocket",
            "log",
            "ice-spirit",
        ),
        (
            "goblin-drill",
            "goblin-barrel",
            "spear-goblins",
            "wall-breakers",
            "fireball",
            "barbarian-barrel",
            "bomb-tower",
            "princess",
        ),
    ),
    # This archetype has no fixed recipe: sample_deck draws a legal random
    # eight-card deck from the selected ruleset's interaction set.
    "random-legal": (),
}

# Held-out evaluation needs more than the finite number of hand-written
# recipes above.  These counts retain the archetype's recognizable pressure
# core while allowing the remaining slots to be replaced deterministically
# from the ruleset interaction set.  Training keeps the default exact-template
# behavior; variants are opt-in at the held-out matrix boundary.
_VARIANT_CORE_COUNTS: Mapping[str, int] = {
    "aggressive-pressure": 3,
    "defensive-cycle": 2,
    "beatdown": 3,
    "air-beatdown": 2,
    "siege-bait": 3,
}

ARCHETYPE_NAMES: tuple[str, ...] = tuple(_DECK_TEMPLATES)

_STRATEGY_ALIASES: Mapping[str, str] = {
    "deterministic-cycle": "deterministic-cycle",
    "cycle": "deterministic-cycle",
    "aggressive-pressure": "aggressive-pressure",
    "aggressive": "aggressive-pressure",
    "pressure": "aggressive-pressure",
    "defensive-cycle": "defensive-cycle",
    "defensive": "defensive-cycle",
    "beatdown": "beatdown",
    "tank-support": "beatdown",
    "beatdown-tank-support": "beatdown",
    "air-beatdown": "beatdown",
    "siege-bait": "siege-bait",
    "siege": "siege-bait",
    "bait": "siege-bait",
    "random-legal": "random-legal",
    "random": "random-legal",
}


class OpponentPool:
    """Sample deck/scenario variants reproducibly for a pinned ruleset."""

    def __init__(self, ruleset: Ruleset | None = None, *, seed: int = 0) -> None:
        if type(seed) is not int:
            raise OpponentPoolError("seed must be an integer")
        self.ruleset = ruleset or load_fixed_ruleset()
        self.seed = seed
        self._available_cards = tuple(sorted(self.ruleset.interaction_set))
        deck_size = int(self.ruleset.match.deck_size)
        if deck_size != DECK_SIZE:
            raise OpponentPoolError(
                f"the opponent pool supports {DECK_SIZE}-card decks, not {deck_size}"
            )
        if len(self._available_cards) < deck_size:
            raise OpponentPoolError("ruleset interaction set is smaller than one deck")

    @property
    def archetypes(self) -> tuple[str, ...]:
        return ARCHETYPE_NAMES

    def sample_deck(
        self,
        episode_index: int = 0,
        *,
        archetype: str | None = None,
        allow_variants: bool = False,
    ) -> OpponentDeckSpec:
        """Return one legal deck for ``episode_index``.

        With no archetype, the sampler covers all named archetypes including
        ``random-legal``.  Template cards unavailable in a narrower ruleset
        are removed and deterministically replaced from that ruleset's
        interaction set, so sampling remains fail-closed.
        """

        _validate_index(episode_index, field="episode_index")
        if type(allow_variants) is not bool:
            raise OpponentPoolError("allow_variants must be boolean")
        requested = None if archetype is None else _normalize_name(archetype, field="archetype")
        if requested is not None and requested not in _DECK_TEMPLATES:
            raise OpponentPoolError(f"unknown archetype: {archetype!r}")
        selection_seed = _stable_seed(self.seed, "deck", episode_index, requested or "any")
        rng = random.Random(selection_seed)
        selected = requested or ARCHETYPE_NAMES[rng.randrange(len(ARCHETYPE_NAMES))]
        if selected == "random-legal":
            cards = tuple(rng.sample(self._available_cards, DECK_SIZE))
            source = "random-interaction-set"
            tags = ("random", "legal")
        else:
            templates = _DECK_TEMPLATES[selected]
            raw = templates[rng.randrange(len(templates))]
            cards_list = [card for card in raw if card in self._available_cards]
            cards_list = list(dict.fromkeys(cards_list))
            for card in self._available_cards:
                if len(cards_list) >= DECK_SIZE:
                    break
                if card not in cards_list:
                    cards_list.append(card)
            cards = tuple(cards_list[:DECK_SIZE])
            source = "curated-template"
            tags = (selected,)
            if allow_variants and selected in _VARIANT_CORE_COUNTS:
                core_count = _VARIANT_CORE_COUNTS[selected]
                core = tuple(cards[:core_count])
                variant_rng = random.Random(
                    _stable_seed(self.seed, "variant", episode_index, selected)
                )
                remaining = tuple(card for card in self._available_cards if card not in core)
                support = tuple(variant_rng.sample(remaining, DECK_SIZE - len(core)))
                cards = tuple((*core, *support))
                source = "curated-variant"
                tags = (selected, "variant")
        if len(cards) != DECK_SIZE or len(set(cards)) != DECK_SIZE:
            raise OpponentPoolError(
                f"archetype {selected!r} cannot form a legal {DECK_SIZE}-card deck"
            )
        digest = hashlib.sha256(
            (selected + "\x1f" + "\x1f".join(cards)).encode("utf-8")
        ).hexdigest()[:12]
        return OpponentDeckSpec(
            deck_id=f"{selected}-{digest}",
            archetype=selected,
            cards=cards,
            tags=tags,
            source=source,
        )

    def sample_decks(
        self,
        count: int,
        *,
        start_index: int = 0,
        archetype: str | None = None,
        unique: bool = False,
    ) -> tuple[OpponentDeckSpec, ...]:
        if type(count) is not int or count < 0:
            raise OpponentPoolError("count must be a non-negative integer")
        _validate_index(start_index, field="start_index")
        if type(unique) is not bool:
            raise OpponentPoolError("unique must be boolean")
        if not unique:
            return tuple(
                self.sample_deck(start_index + index, archetype=archetype)
                for index in range(count)
            )

        # A matrix axis must not silently contain the same deck more than
        # once.  Keep the ordinary sampler's stream untouched and search
        # forward through its stable episode indices only when callers opt in
        # to this stronger guarantee.
        decks: list[OpponentDeckSpec] = []
        seen_cards: set[frozenset[str]] = set()
        candidate_index = start_index
        attempts = 0
        max_attempts = max(1_024, count * 256)
        while len(decks) < count and attempts < max_attempts:
            deck = self.sample_deck(candidate_index, archetype=archetype)
            deck_key = frozenset(deck.cards)
            if deck_key not in seen_cards:
                decks.append(deck)
                seen_cards.add(deck_key)
            candidate_index += 1
            attempts += 1
        if len(decks) != count:
            requested = "any archetype" if archetype is None else repr(archetype)
            raise OpponentPoolError(
                f"could not sample {count} unique decks for {requested} "
                f"from {attempts} deterministic candidates"
            )
        return tuple(decks)

    def sample(
        self,
        episode_index: int = 0,
        *,
        archetype: str | None = None,
        strategy: str | None = None,
    ) -> OpponentScenario:
        """Sample a deck and a matching reproducible simulator controller."""

        _validate_index(episode_index, field="episode_index")
        deck = self.sample_deck(episode_index, archetype=archetype)
        selection_seed = _stable_seed(self.seed, "scenario", episode_index)
        if strategy is None:
            selected_strategy = _STRATEGY_ALIASES.get(deck.archetype, "random-legal")
        else:
            normalized = _normalize_name(strategy, field="strategy")
            try:
                selected_strategy = _STRATEGY_ALIASES[normalized]
            except KeyError as error:
                raise OpponentPoolError(f"unknown strategy: {strategy!r}") from error
        controller_seed = _stable_seed(selection_seed, "controller", selected_strategy)
        return OpponentScenario(
            episode_index=episode_index,
            selection_seed=selection_seed,
            deck=deck,
            strategy=selected_strategy,
            controller_seed=controller_seed,
        )


def _clamp_cell(cell: tuple[int, int]) -> tuple[int, int]:
    return max(0, min(17, int(cell[0]))), max(0, min(31, int(cell[1])))


def _player_rows(player: int) -> tuple[int, int, int, int]:
    if player == 0:
        return 23, 21, 20, 17
    if player == 1:
        return 8, 10, 11, 14
    raise ValueError("player must be 0 or 1")


def _enemy_entities(state: Any, player: int) -> list[Any]:
    return [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.owner != player and entity.kind != "tower"
    ]


def _enemy_troops(state: Any, player: int) -> list[Any]:
    return [entity for entity in _enemy_entities(state, player) if entity.kind == "troop"]


def _enemy_towers(state: Any, player: int) -> list[Any]:
    return [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.owner != player and entity.kind == "tower"
    ]


def _own_entities(state: Any, player: int) -> list[Any]:
    return [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.owner == player
    ]


def _is_crossed(engine: BattleEngine, entity: Any, player: int) -> bool:
    river_min = int(engine.ruleset.arena.river_y_min_mtile)
    river_max = int(engine.ruleset.arena.river_y_max_mtile)
    return entity.y_mtile >= river_min if player == 0 else entity.y_mtile <= river_max


def _lane_for_x(engine: BattleEngine, x_mtile: int) -> int:
    return 0 if x_mtile < int(engine.ruleset.arena.width_mtile) // 2 else 1


def _lane_col(lane: int) -> int:
    return 3 if lane == 0 else 14


def _cell_from_entity(entity: Any) -> tuple[int, int]:
    return _clamp_cell((entity.x_mtile // 1_000, entity.y_mtile // 1_000))


def _target_cell(engine: BattleEngine, state: Any, player: int, *, crossed_only: bool = False) -> tuple[int, int]:
    candidates = _enemy_troops(state, player)
    if crossed_only:
        candidates = [entity for entity in candidates if _is_crossed(engine, entity, player)]
    if candidates:
        if player == 0:
            target = max(candidates, key=lambda entity: (entity.y_mtile, -entity.x_mtile, entity.uid))
        else:
            target = min(candidates, key=lambda entity: (entity.y_mtile, entity.x_mtile, entity.uid))
        return _cell_from_entity(target)
    towers = [tower for tower in _enemy_towers(state, player) if tower.role != "king"]
    if towers:
        target = min(towers, key=lambda tower: (tower.hp, tower.role, tower.uid))
        return _cell_from_entity(target)
    _, _, _, bridge_row = _player_rows(player)
    return (_lane_col(0), bridge_row)


def _threat_lane(engine: BattleEngine, state: Any, player: int) -> int:
    crossed = [entity for entity in _enemy_troops(state, player) if _is_crossed(engine, entity, player)]
    if not crossed:
        return 0 if (state.players[player].cards_played % 2 == 0) else 1
    if player == 0:
        target = max(crossed, key=lambda entity: (entity.y_mtile, -entity.x_mtile, entity.uid))
    else:
        target = min(crossed, key=lambda entity: (entity.y_mtile, entity.x_mtile, entity.uid))
    return _lane_for_x(engine, target.x_mtile)


def _has_live_card(state: Any, player: int, card_ids: set[str]) -> bool:
    return any(
        entity.card_id in card_ids
        for entity in _own_entities(state, player)
    )


class _HeuristicController:
    """Common legal-action and placement helpers for simulator opponents."""

    def _try_card(
        self,
        engine: BattleEngine,
        state: Any,
        player: int,
        card_id: str,
        preferred_cell: tuple[int, int],
    ) -> SimAction | None:
        player_state = state.players[player]
        for slot, held_card in enumerate(player_state.hand):
            if held_card != card_id:
                continue
            card = engine.ruleset.card(held_card)
            if card.card_id == "mirror" and player_state.last_played_card_id in {None, "mirror"}:
                return None
            if engine._effective_card_cost(player_state, card) > player_state.elixir_milli:
                continue
            preferred = _clamp_cell(preferred_cell)
            action = PlayCardAction(player, slot, preferred)
            if engine.validate_action(state, action) is None:
                return action
            legal = engine.legal_cells(state, player, card.card_id)
            if not legal:
                continue
            # Choose the legal point nearest the strategy's preference.  The
            # tuple tie-break keeps the action deterministic across runs.
            cell = min(
                legal,
                key=lambda value: (
                    abs(value[0] - preferred[0]) + abs(value[1] - preferred[1]),
                    value[1],
                    value[0],
                ),
            )
            candidate = PlayCardAction(player, slot, cell)
            if engine.validate_action(state, candidate) is None:
                return candidate
        return None

    def _try_priority(
        self,
        engine: BattleEngine,
        state: Any,
        player: int,
        priority: tuple[str, ...],
        cells: Mapping[str, tuple[int, int]],
        *,
        skip: frozenset[str] = frozenset(),
    ) -> SimAction | None:
        for card_id in priority:
            if card_id in skip:
                continue
            action = self._try_card(
                engine,
                state,
                player,
                card_id,
                cells.get(card_id, cells.get("default", (8, _player_rows(player)[0]))),
            )
            if action is not None:
                return action
        return None

    def _fallback(self, engine: BattleEngine, state: Any, player: int) -> SimAction:
        """Play the cheapest available card, with a legal placement fallback."""

        player_state = state.players[player]
        rows = _player_rows(player)
        lane = _threat_lane(engine, state, player)
        ordered = sorted(
            enumerate(player_state.hand),
            key=lambda item: (
                engine._effective_card_cost(player_state, engine.ruleset.card(item[1])),
                item[0],
            ),
        )
        for _slot, card_id in ordered:
            if card_id == "mirror":
                continue
            card = engine.ruleset.card(card_id)
            if card.kind == "spell":
                cell = _target_cell(engine, state, player)
            elif card.kind == "building":
                cell = (8, rows[2])
            else:
                cell = (_lane_col(lane), rows[0])
            action = self._try_card(engine, state, player, card_id, cell)
            if action is not None:
                return action
        return WaitAction(player)


class DeterministicCycleController:
    """The pinned cheapest-affordable-card cycle controller.

    This mirrors the engine's smoke-test controller in this independent
    module, allowing pool callers to obtain every opponent implementation from
    one import path while retaining the exact published behavior.
    """

    def __init__(self, *, lane: str = "alternate") -> None:
        if lane not in {"left", "right", "alternate"}:
            raise OpponentPoolError("lane must be left, right, or alternate")
        self.lane = lane

    def choose_action(self, engine: BattleEngine, state: Any, player: int) -> SimAction:
        player_state = state.players[player]
        affordable: list[int] = []
        for slot, card_id in enumerate(player_state.hand):
            if card_id == "mirror" and player_state.last_played_card_id in {None, "mirror"}:
                continue
            card = engine.ruleset.card(card_id)
            if engine._effective_card_cost(player_state, card) <= player_state.elixir_milli:
                affordable.append(slot)
        if not affordable:
            return WaitAction(player)
        slot = min(
            affordable,
            key=lambda index: (
                engine._effective_card_cost(player_state, engine.ruleset.card(player_state.hand[index])),
                index,
            ),
        )
        card = engine.ruleset.card(player_state.hand[slot])
        use_left = self.lane == "left" or (
            self.lane == "alternate" and player_state.cards_played % 2 == 0
        )
        col = 3 if use_left else 14
        rows = _player_rows(player)
        if card.kind == "spell" and card.card_id == "fireball":
            row = 6 if player == 0 else 25
        elif card.kind == "spell":
            row = 19 if player == 0 else 12
        elif card.kind == "building":
            col, row = 8, rows[2]
        else:
            row = rows[0]
        action = PlayCardAction(player, slot, (col, row))
        if engine.validate_action(state, action) is None:
            return action
        legal = engine.legal_cells(state, player, card.card_id)
        if not legal:
            return WaitAction(player)
        return PlayCardAction(player, slot, legal[0])


class AggressivePressureController(_HeuristicController):
    """Prioritize bridge pressure and tower-targeting win conditions."""

    _WIN_CONDITIONS = (
        "hog-rider",
        "battle-ram",
        "ram-rider",
        "royal-hogs",
        "balloon",
        "wall-breakers",
        "goblin-drill",
        "goblin-barrel",
        "miner",
    )
    _DEFENSE = (
        "fireball",
        "poison",
        "arrows",
        "barbarian-barrel",
        "log",
        "tesla",
        "cannon",
        "valkyrie",
        "musketeer",
        "archers",
        "bats",
        "skeletons",
    )
    _SUPPORT = (
        "firecracker",
        "dart-goblin",
        "musketeer",
        "electro-wizard",
        "ice-spirit",
        "spear-goblins",
        "bats",
        "goblins",
    )

    def choose_action(self, engine: BattleEngine, state: Any, player: int) -> SimAction:
        rows = _player_rows(player)
        lane = _threat_lane(engine, state, player)
        target = _target_cell(engine, state, player, crossed_only=True)
        crossed = any(_is_crossed(engine, entity, player) for entity in _enemy_troops(state, player))
        cells = {
            "default": (_lane_col(lane), rows[3]),
            "hog-rider": (_lane_col(lane), rows[3]),
            "battle-ram": (_lane_col(lane), rows[3]),
            "ram-rider": (_lane_col(lane), rows[3]),
            "royal-hogs": (_lane_col(lane), rows[3]),
            "balloon": (_lane_col(lane), rows[0]),
            "wall-breakers": (_lane_col(lane), rows[3]),
            "goblin-drill": target,
            "goblin-barrel": target,
            "miner": target,
            "fireball": target,
            "poison": target,
            "arrows": target,
            "barbarian-barrel": target,
            "log": target,
        }
        if crossed:
            action = self._try_priority(engine, state, player, self._DEFENSE, cells)
            if action is not None:
                return action
        action = self._try_priority(engine, state, player, self._WIN_CONDITIONS, cells)
        if action is not None:
            return action
        action = self._try_priority(engine, state, player, self._SUPPORT, cells)
        if action is not None:
            return action
        return self._fallback(engine, state, player)


class DefensiveCycleController(_HeuristicController):
    """Hold a defensive building/counter and cycle cheap cards safely."""

    _DEFENSIVE = (
        "cannon",
        "tesla",
        "bomb-tower",
        "goblin-cage",
        "tombstone",
        "mortar",
        "fireball",
        "poison",
        "arrows",
        "tornado",
        "log",
        "valkyrie",
        "mini-pekka",
        "musketeer",
        "archers",
        "knight",
        "skeletons",
        "bats",
    )
    _CYCLE = (
        "ice-spirit",
        "electro-spirit",
        "fire-spirit",
        "skeletons",
        "goblins",
        "spear-goblins",
        "bats",
        "archers",
        "knight",
        "musketeer",
    )

    def choose_action(self, engine: BattleEngine, state: Any, player: int) -> SimAction:
        rows = _player_rows(player)
        lane = _threat_lane(engine, state, player)
        target = _target_cell(engine, state, player, crossed_only=True)
        crossed = [entity for entity in _enemy_troops(state, player) if _is_crossed(engine, entity, player)]
        cells = {
            "default": (8, rows[1]),
            "cannon": (8, rows[1]),
            "tesla": (8, rows[1]),
            "bomb-tower": (8, rows[1]),
            "goblin-cage": (8, rows[1]),
            "tombstone": (_lane_col(lane), rows[1]),
            "mortar": (8, rows[2]),
            "fireball": target,
            "poison": target,
            "arrows": target,
            "tornado": target,
            "log": target,
            "valkyrie": (_lane_col(lane), rows[1]),
            "mini-pekka": (_lane_col(lane), rows[1]),
            "musketeer": (_lane_col(lane), rows[0]),
            "archers": (_lane_col(lane), rows[0]),
            "knight": (_lane_col(lane), rows[1]),
            "skeletons": (_lane_col(lane), rows[1]),
            "bats": (_lane_col(lane), rows[0]),
        }
        if crossed:
            action = self._try_priority(engine, state, player, self._DEFENSIVE, cells)
            if action is not None:
                return action
        action = self._try_priority(engine, state, player, self._CYCLE, cells)
        if action is not None:
            return action
        return self._fallback(engine, state, player)


class BeatdownTankSupportController(_HeuristicController):
    """Build a tank push, then add support behind the surviving tank."""

    _TANKS = (
        "golem",
        "electro-giant",
        "goblin-giant",
        "giant",
        "giant-skeleton",
        "lava-hound",
        "pekka",
        "mega-knight",
    )
    _SUPPORT = (
        "night-witch",
        "witch",
        "baby-dragon",
        "electro-dragon",
        "musketeer",
        "executioner",
        "mega-minion",
        "skeleton-dragons",
        "lumberjack",
        "bowler",
        "wizard",
        "bats",
    )
    _DEFENSE = (
        "lightning",
        "fireball",
        "poison",
        "tornado",
        "arrows",
        "mini-pekka",
        "valkyrie",
        "musketeer",
        "baby-dragon",
        "mega-minion",
    )

    def choose_action(self, engine: BattleEngine, state: Any, player: int) -> SimAction:
        rows = _player_rows(player)
        lane = _threat_lane(engine, state, player)
        target = _target_cell(engine, state, player, crossed_only=True)
        crossed = [entity for entity in _enemy_troops(state, player) if _is_crossed(engine, entity, player)]
        own_tank = _has_live_card(state, player, set(self._TANKS))
        cells = {
            "default": (_lane_col(lane), rows[0]),
            "golem": (_lane_col(lane), rows[0] + (2 if player == 0 else -2)),
            "electro-giant": (_lane_col(lane), rows[0] + (2 if player == 0 else -2)),
            "goblin-giant": (_lane_col(lane), rows[0] + (2 if player == 0 else -2)),
            "giant": (_lane_col(lane), rows[0] + (2 if player == 0 else -2)),
            "giant-skeleton": (_lane_col(lane), rows[0] + (2 if player == 0 else -2)),
            "lava-hound": (_lane_col(lane), rows[0] + (2 if player == 0 else -2)),
            "pekka": (_lane_col(lane), rows[0]),
            "mega-knight": (_lane_col(lane), rows[1]),
            "night-witch": (_lane_col(lane), rows[0] + (4 if player == 0 else -4)),
            "witch": (_lane_col(lane), rows[0] + (4 if player == 0 else -4)),
            "baby-dragon": (_lane_col(lane), rows[0] + (4 if player == 0 else -4)),
            "electro-dragon": (_lane_col(lane), rows[0] + (4 if player == 0 else -4)),
            "musketeer": (_lane_col(lane), rows[0] + (4 if player == 0 else -4)),
            "executioner": (_lane_col(lane), rows[0] + (4 if player == 0 else -4)),
            "mega-minion": (_lane_col(lane), rows[0] + (4 if player == 0 else -4)),
            "skeleton-dragons": (_lane_col(lane), rows[0] + (4 if player == 0 else -4)),
            "lumberjack": (_lane_col(lane), rows[0] + (2 if player == 0 else -2)),
            "bowler": (_lane_col(lane), rows[0] + (4 if player == 0 else -4)),
            "wizard": (_lane_col(lane), rows[0] + (4 if player == 0 else -4)),
            "bats": (_lane_col(lane), rows[0] + (4 if player == 0 else -4)),
            "lightning": target,
            "fireball": target,
            "poison": target,
            "tornado": target,
            "arrows": target,
            "mini-pekka": (_lane_col(lane), rows[1]),
            "valkyrie": (_lane_col(lane), rows[1]),
        }
        if crossed:
            action = self._try_priority(engine, state, player, self._DEFENSE, cells)
            if action is not None:
                return action
        if not own_tank:
            action = self._try_priority(engine, state, player, self._TANKS, cells)
            if action is not None:
                return action
        action = self._try_priority(engine, state, player, self._SUPPORT, cells)
        if action is not None:
            return action
        return self._fallback(engine, state, player)


class SiegeBaitController(_HeuristicController):
    """Alternate siege commitments with cheap bait and spell pressure."""

    _SIEGE = ("x-bow", "mortar", "goblin-drill")
    _BAIT = (
        "goblin-barrel",
        "skeleton-barrel",
        "wall-breakers",
        "goblin-gang",
        "skeleton-army",
        "princess",
        "spear-goblins",
        "firecracker",
    )
    _SPELLS = (
        "rocket",
        "fireball",
        "poison",
        "barbarian-barrel",
        "log",
        "arrows",
        "zap",
    )
    _DEFENSE = (
        "tesla",
        "bomb-tower",
        "cannon",
        "tombstone",
        "valkyrie",
        "knight",
        "fireball",
        "poison",
        "log",
        "arrows",
        "skeleton-army",
    )

    def choose_action(self, engine: BattleEngine, state: Any, player: int) -> SimAction:
        rows = _player_rows(player)
        lane = _threat_lane(engine, state, player)
        target = _target_cell(engine, state, player, crossed_only=True)
        crossed = [entity for entity in _enemy_troops(state, player) if _is_crossed(engine, entity, player)]
        cells = {
            "default": (8, rows[2]),
            "x-bow": (8, rows[2]),
            "mortar": (8, rows[2]),
            "goblin-drill": target,
            "goblin-barrel": target,
            "skeleton-barrel": target,
            "wall-breakers": (_lane_col(lane), rows[3]),
            "goblin-gang": (_lane_col(lane), rows[3]),
            "skeleton-army": (_lane_col(lane), rows[1]),
            "princess": (_lane_col(lane), rows[0]),
            "spear-goblins": (_lane_col(lane), rows[0]),
            "firecracker": (_lane_col(lane), rows[0]),
            "rocket": target,
            "fireball": target,
            "poison": target,
            "barbarian-barrel": target,
            "log": target,
            "arrows": target,
            "zap": target,
            "tesla": (8, rows[1]),
            "bomb-tower": (8, rows[1]),
            "cannon": (8, rows[1]),
            "tombstone": (_lane_col(lane), rows[1]),
            "valkyrie": (_lane_col(lane), rows[1]),
            "knight": (_lane_col(lane), rows[1]),
        }
        if crossed:
            action = self._try_priority(engine, state, player, self._DEFENSE, cells)
            if action is not None:
                return action
        action = self._try_priority(engine, state, player, self._SIEGE, cells)
        if action is not None:
            return action
        action = self._try_priority(engine, state, player, self._BAIT, cells)
        if action is not None:
            return action
        action = self._try_priority(engine, state, player, self._SPELLS, cells)
        if action is not None:
            return action
        return self._fallback(engine, state, player)


class RandomLegalController(_HeuristicController):
    """Seeded random controller that samples only validated legal actions."""

    def __init__(self, *, seed: int = 0, wait_probability: float = 0.05) -> None:
        if type(seed) is not int:
            raise OpponentPoolError("seed must be an integer")
        if isinstance(wait_probability, bool) or not 0.0 <= float(wait_probability) <= 1.0:
            raise OpponentPoolError("wait_probability must be in [0, 1]")
        self.seed = seed
        self.wait_probability = float(wait_probability)
        self._rng = random.Random(seed)

    def choose_action(self, engine: BattleEngine, state: Any, player: int) -> SimAction:
        if self._rng.random() < self.wait_probability:
            return WaitAction(player)
        player_state = state.players[player]
        candidates: list[SimAction] = []
        for slot, card_id in enumerate(player_state.hand):
            card = engine.ruleset.card(card_id)
            if card_id == "mirror" and player_state.last_played_card_id in {None, "mirror"}:
                continue
            if engine._effective_card_cost(player_state, card) > player_state.elixir_milli:
                continue
            legal = engine.legal_cells(state, player, card_id)
            if not legal:
                continue
            cell = legal[self._rng.randrange(len(legal))]
            action = PlayCardAction(player, slot, cell)
            if engine.validate_action(state, action) is None:
                candidates.append(action)
        if not candidates:
            return WaitAction(player)
        return candidates[self._rng.randrange(len(candidates))]


_CONTROLLER_FACTORIES: Mapping[str, Callable[..., OpponentController]] = {
    "deterministic-cycle": DeterministicCycleController,
    "aggressive-pressure": AggressivePressureController,
    "defensive-cycle": DefensiveCycleController,
    "beatdown": BeatdownTankSupportController,
    "siege-bait": SiegeBaitController,
    "random-legal": RandomLegalController,
}


def make_opponent_controller(
    strategy: str,
    *,
    seed: int = 0,
) -> OpponentController:
    """Construct a named simulator-only opponent controller."""

    normalized = _normalize_name(strategy, field="strategy")
    try:
        canonical = _STRATEGY_ALIASES[normalized]
    except KeyError as error:
        raise OpponentPoolError(f"unknown strategy: {strategy!r}") from error
    try:
        factory = _CONTROLLER_FACTORIES[canonical]
    except KeyError as error:
        raise OpponentPoolError(f"unknown strategy: {strategy!r}") from error
    if canonical == "deterministic-cycle":
        return factory(lane="alternate")
    if canonical == "random-legal":
        return factory(seed=seed)
    return factory()


__all__ = [
    "ARCHETYPE_NAMES",
    "AggressivePressureController",
    "BeatdownTankSupportController",
    "DECK_SIZE",
    "DefensiveCycleController",
    "DeterministicCycleController",
    "OpponentController",
    "OpponentDeckSpec",
    "OpponentPool",
    "OpponentPoolError",
    "OpponentScenario",
    "RandomLegalController",
    "SiegeBaitController",
    "make_opponent_controller",
]
