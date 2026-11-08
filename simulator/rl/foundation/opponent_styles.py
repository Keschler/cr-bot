"""Explicit, versioned training opponents over the frozen heuristic benchmark.

Styles vary meaningful commitments rather than injecting random mistakes. They
use own elixir, visible bodies and tower health; enemy hand/elixir are never
consulted. No-style callers keep using ``opponent_pool`` directly. The seed
chooses stable lane ties and formation variants, so forks need no RNG history.
These controllers are training infrastructure, not evidence of human strength.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Mapping

from .opponent_pool import (
    AggressivePressureController,
    BeatdownTankSupportController,
    OpponentPoolError,
    SiegeBaitController,
    _HeuristicController,
    _STRATEGY_ALIASES,
    _cell_from_entity,
    _enemy_entities,
    _enemy_towers,
    _enemy_troops,
    _is_crossed,
    _lane_col,
    _lane_for_x,
    _normalize_name,
    _own_entities,
    _player_rows,
    _stable_seed,
    _tower_connection_cell,
    make_opponent_controller,
)
from simulator.actions import PlayCardAction, WaitAction


OPPONENT_STYLE_VERSION = "training-opponent-styles-v1"
TRAIN_STYLE_IDS = ("tempo", "patient", "counterpush")
DEV_STYLE_IDS = ("adaptive", "split-pressure")
STYLE_ARCHETYPES = (
    "aggressive-pressure", "defensive-cycle", "beatdown", "air-beatdown", "siege-bait",
)
_LANE_POLICIES = frozenset({"weak-tower", "counterpush", "opposite-commitment", "adaptive"})


def normalize_style_strategy(strategy: str) -> str:
    """Normalize every historical alias, retaining air-beatdown provenance."""

    normalized = _normalize_name(strategy, field="strategy")
    if normalized == "air-beatdown":
        return normalized
    canonical = _STRATEGY_ALIASES.get(normalized)
    if canonical not in STYLE_ARCHETYPES:
        raise OpponentPoolError(f"unsupported styled strategy: {strategy!r}")
    return canonical


@dataclass(frozen=True, slots=True)
class OpponentStyleConfig:
    """Exact style parameters; integer units avoid ambiguous run identities."""

    style_id: str
    reserve_milli: int
    cycle_floor_milli: int
    punish_commitment_milli: int
    formation_depth_rows: int
    lane_policy: str
    version: str = OPPONENT_STYLE_VERSION

    def __post_init__(self) -> None:
        style_id = _normalize_name(self.style_id, field="style_id")
        lane_policy = _normalize_name(self.lane_policy, field="lane_policy")
        object.__setattr__(self, "style_id", style_id)
        object.__setattr__(self, "lane_policy", lane_policy)
        if self.version != OPPONENT_STYLE_VERSION:
            raise OpponentPoolError(f"unsupported opponent style version: {self.version!r}")
        if style_id not in (*TRAIN_STYLE_IDS, *DEV_STYLE_IDS):
            raise OpponentPoolError(f"unknown opponent style: {style_id!r}")
        if lane_policy not in _LANE_POLICIES:
            raise OpponentPoolError(f"unknown style lane_policy: {lane_policy!r}")
        for name, low, high in (
            ("reserve_milli", 0, 3_000),
            ("cycle_floor_milli", 1_000, 8_000),
            ("punish_commitment_milli", 4_000, 12_000),
            ("formation_depth_rows", 0, 2),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise OpponentPoolError(f"{name} must be an integer in [{low}, {high}]")

    @property
    def split(self) -> str:
        return "train" if self.style_id in TRAIN_STYLE_IDS else "dev"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_STYLE_CONFIGS = {
    "tempo": OpponentStyleConfig("tempo", 0, 3_000, 6_000, 0, "weak-tower"),
    "patient": OpponentStyleConfig("patient", 2_000, 7_000, 8_000, 2, "weak-tower"),
    "counterpush": OpponentStyleConfig("counterpush", 1_000, 5_000, 6_000, 1, "counterpush"),
    "adaptive": OpponentStyleConfig("adaptive", 1_500, 6_000, 7_000, 2, "adaptive"),
    "split-pressure": OpponentStyleConfig(
        "split-pressure", 500, 4_000, 5_000, 1, "opposite-commitment"
    ),
}


def style_ids(split: str = "train") -> tuple[str, ...]:
    """Return disjoint style pools without changing any deck/catalog split."""

    if split == "train":
        return TRAIN_STYLE_IDS
    if split == "dev":
        return DEV_STYLE_IDS
    raise OpponentPoolError("style split must be 'train' or 'dev'")


def resolve_style_config(
    style: str | OpponentStyleConfig | None,
    config: Mapping[str, Any] | None = None,
) -> OpponentStyleConfig:
    """Resolve a named preset plus validated overrides, or an exact config."""

    if config is not None and not isinstance(config, Mapping):
        raise OpponentPoolError("style_config must be a mapping")
    overrides = dict(config or {})
    unknown = set(overrides) - set(OpponentStyleConfig.__dataclass_fields__)
    if unknown:
        raise OpponentPoolError(f"unknown style_config fields: {sorted(unknown)}")
    if isinstance(style, OpponentStyleConfig):
        base = style
    else:
        requested = style if style is not None else overrides.get("style_id")
        key = _normalize_name(requested, field="style")
        try:
            base = _STYLE_CONFIGS[key]
        except KeyError as error:
            raise OpponentPoolError(f"unknown opponent style: {requested!r}") from error
    if "style_id" in overrides:
        if _normalize_name(overrides["style_id"], field="style_id") != base.style_id:
            raise OpponentPoolError("style and style_config.style_id disagree")
    return replace(base, **overrides)


def style_identity(
    strategy: str,
    style: str | OpponentStyleConfig | None,
    *,
    seed: int = 0,
    config: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Return full pinned provenance, including all seed-resolved variation."""

    if type(seed) is not int:
        raise OpponentPoolError("style seed must be an integer")
    canonical = normalize_style_strategy(strategy)
    resolved = resolve_style_config(style, config)
    variation = _stable_seed(seed, OPPONENT_STYLE_VERSION, canonical, resolved.style_id)
    identity: dict[str, object] = {
        "version": OPPONENT_STYLE_VERSION,
        "strategy": canonical,
        "style_id": resolved.style_id,
        "split": resolved.split,
        "seed": seed,
        "parameters": resolved.as_dict(),
        "preferred_lane": variation % 2,
        "formation_variant": (variation >> 1) % 2,
    }
    identity["sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity


_PRIMARY_CARDS = {
    "aggressive-pressure": AggressivePressureController._WIN_CONDITIONS,
    "defensive-cycle": ("x-bow", "mortar", "goblin-drill"),
    "beatdown": BeatdownTankSupportController._TANKS,
    "air-beatdown": BeatdownTankSupportController._TANKS,
    "siege-bait": (*SiegeBaitController._SIEGE, "goblin-barrel", "skeleton-barrel"),
}
_DIRECT_ATTACKS = frozenset({"miner", "goblin-barrel", "goblin-drill"})
_BRIDGE_ATTACKS = frozenset({
    "hog-rider", "battle-ram", "ram-rider", "royal-hogs", "wall-breakers", "skeleton-barrel",
})
_TANKS = frozenset(BeatdownTankSupportController._TANKS) - {"balloon"}
_COUNTERPUSH_SUPPORT = frozenset({
    "musketeer", "archers", "firecracker", "knight", "valkyrie", "mini-pekka",
    "mega-minion", "baby-dragon", "skeleton-dragons", "night-witch", "witch",
    "bowler", "executioner", "electro-dragon", "lumberjack", "bandit", "royal-ghost",
})


class StyledOpponentController(_HeuristicController):
    """Legal heuristic style that preserves urgent defense and primary plans.

    Reserve and cycle floors apply only on a quiet defensive half. Visible
    expensive commitments or surviving support release reserve for a punish or
    counterpush. At max elixir the controller always releases the reserve.
    """

    def __init__(
        self, strategy: str, style: str | OpponentStyleConfig | None, *,
        seed: int = 0, config: Mapping[str, Any] | None = None,
    ) -> None:
        self.strategy = normalize_style_strategy(strategy)
        self.config = resolve_style_config(style, config)
        self.identity = style_identity(self.strategy, self.config, seed=seed)
        self.base = make_opponent_controller(self.strategy, seed=seed)

    def _visible_commitment(self, engine: Any, state: Any, player: int) -> tuple[int, int | None]:
        """Count visible card deployments once, including multi-body swarms."""

        by_play: dict[tuple[str, int], Any] = {}
        for entity in _enemy_entities(state, player):
            if not _is_crossed(engine, entity, player):
                by_play.setdefault((entity.card_id, entity.spawn_tick), entity)
        commitment = [
            (engine.ruleset.card(entity.card_id).elixir_milli, entity)
            for entity in by_play.values()
        ]
        if not commitment:
            return 0, None
        # Lane of the largest commitment, rather than hidden enemy elixir.
        costs = [0, 0]
        for cost, entity in commitment:
            costs[_lane_for_x(engine, entity.x_mtile)] += cost
        lane = 0 if costs[0] > costs[1] else 1
        return sum(costs), lane

    def _support_anchor(self, engine: Any, state: Any, player: int) -> Any:
        own = [
            entity for entity in _own_entities(state, player)
            if entity.kind == "troop" and entity.card_id in _COUNTERPUSH_SUPPORT
            and entity.hp * 3 >= entity.max_hp
            and entity.deploy_remaining_us == 0
            and (entity.y_mtile <= 21_000 if player == 0 else entity.y_mtile >= 10_000)
        ]
        if not own:
            return None
        # Most advanced healthy support can turn defense into an attack.
        return min(own, key=lambda entity: (
            entity.y_mtile if player == 0 else -entity.y_mtile, entity.uid
        ))

    def _attack_lane(
        self, engine: Any, state: Any, player: int, commitment: int,
        committed_lane: int | None, support: Any,
    ) -> int:
        policy = self.config.lane_policy
        punish = commitment >= self.config.punish_commitment_milli
        if policy in {"opposite-commitment", "adaptive"} and punish and committed_lane is not None:
            return 1 - committed_lane
        if policy in {"counterpush", "adaptive"} and support is not None:
            return _lane_for_x(engine, support.x_mtile)
        towers = [tower for tower in _enemy_towers(state, player) if tower.role != "king"]
        if towers:
            preferred = int(self.identity["preferred_lane"])
            target = min(towers, key=lambda tower: (
                tower.hp * 1_000 // tower.max_hp,
                _lane_for_x(engine, tower.x_mtile) != preferred, tower.uid,
            ))
            return _lane_for_x(engine, target.x_mtile)
        return int(self.identity["preferred_lane"])

    def choose_action(self, engine: Any, state: Any, player: int) -> Any:
        # Existing benchmark defense remains the immediate response to a
        # crossed visible threat. This also preserves concealment semantics.
        base_action = self.base.choose_action(engine, state, player)
        if any(_is_crossed(engine, entity, player) for entity in _enemy_troops(state, player)):
            return base_action
        if not isinstance(base_action, PlayCardAction):
            return base_action
        own = state.players[player]
        card_id = own.hand[base_action.card_slot]
        card = engine.ruleset.card(card_id)
        primary = card_id in _PRIMARY_CARDS[self.strategy]
        commitment, committed_lane = self._visible_commitment(engine, state, player)
        support = self._support_anchor(engine, state, player)
        punish = commitment >= self.config.punish_commitment_milli
        own_tanks = [entity for entity in _own_entities(state, player) if entity.card_id in _TANKS]
        live_push = bool(own_tanks) or support is not None
        cost = engine._effective_card_cost(own, card)
        reserve = self.config.reserve_milli
        if punish or live_push:
            reserve = 0
        maximum = int(engine.ruleset.match.max_elixir_milli)
        if own.elixir_milli < maximum:
            if primary and own.elixir_milli < min(maximum, cost + reserve):
                return WaitAction(player)
            if not primary and not punish and not live_push:
                if own.elixir_milli < self.config.cycle_floor_milli:
                    return WaitAction(player)

        lane = self._attack_lane(engine, state, player, commitment, committed_lane, support)
        rows = _player_rows(player)
        depth = self.config.formation_depth_rows
        if depth:
            depth += int(self.identity["formation_variant"])
        rearward = 1 if player == 0 else -1
        preferred = base_action.cell
        if card_id in {"x-bow", "mortar"}:
            # Siege must still connect to a tower after any lane change.
            preferred = _tower_connection_cell(engine, state, player, card_id, lane)
            if preferred is None:
                return base_action
        elif card_id in _DIRECT_ATTACKS or card.kind == "spell":
            towers = _enemy_towers(state, player)
            if towers:
                target = min(towers, key=lambda tower: (
                    tower.role == "king", _lane_for_x(engine, tower.x_mtile) != lane,
                    tower.hp, tower.uid,
                ))
                preferred = _cell_from_entity(target)
        elif card_id in _BRIDGE_ATTACKS:
            preferred = (_lane_col(lane), rows[3])
        elif card_id in _TANKS:
            preferred = (_lane_col(lane), rows[0] + depth * rearward)
        elif card.kind == "troop":
            if own_tanks:
                # Keep support behind the live primary tank, including Balloon.
                anchor = max(own_tanks, key=lambda entity: (entity.spawn_tick, entity.uid))
                preferred = (
                    anchor.x_mtile // 1_000,
                    anchor.y_mtile // 1_000 + (2 + depth) * rearward,
                )
            else:
                preferred = (_lane_col(lane), rows[0] + depth * rearward)
        action = self._try_card(engine, state, player, card_id, preferred)
        return action if action is not None else base_action


def make_styled_opponent(
    strategy: str, style: str | OpponentStyleConfig | None, *,
    seed: int = 0, config: Mapping[str, Any] | None = None,
) -> StyledOpponentController:
    return StyledOpponentController(strategy, style, seed=seed, config=config)


__all__ = [
    "OPPONENT_STYLE_VERSION", "TRAIN_STYLE_IDS", "DEV_STYLE_IDS", "STYLE_ARCHETYPES",
    "OpponentStyleConfig", "StyledOpponentController", "make_styled_opponent",
    "normalize_style_strategy", "resolve_style_config", "style_identity", "style_ids",
]
