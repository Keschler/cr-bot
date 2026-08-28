"""Deterministic, headless Clash Royale battle engine.

This module is an executable Level-11 *model* for the pinned interaction set.
It deliberately keeps mechanics generic and data-driven.  Values marked as
uncertain by the ruleset remain fidelity targets; deterministic execution does
not promote them to ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from .actions import PlayCardAction, SimAction, UseAbilityAction, WaitAction
from .events import SimEvent
from .fixed import (
    ELIXIR_SCALE,
    PERMILLE,
    SECOND_US,
    DeterministicRng,
    ceil_div,
    distance_mtile,
    move_towards,
)
from .geometry import (
    GRID_COLS,
    GRID_ROWS,
    TOWER_SITES,
    cell_center_mtile,
    is_basic_deploy_cell,
    is_ground_cell,
    is_spell_cell,
    position_to_cell,
)
from .navigation import (
    NavigationObstacle,
    plan_route,
    point_is_walkable,
    segment_is_walkable,
)
from .ruleset import CardDefinition, Ruleset, RulesetError, TowerDefinition, load_ruleset
from .roster import PLAYER_DECK
from .state import (
    AreaEffectState,
    BattleState,
    EntityState,
    PlayerState,
    ProjectileState,
    StatusState,
)


# Keep the engine default, policy observation slots, scenario factory, and
# physical Testspiel fixed-deck order on one canonical contract.  In fixed
# order the first four cards are the opening hand and the remaining four are
# the replacement sequence.
BASE_HOG_CYCLE_DECK = PLAYER_DECK
# Behavior-changing mechanics (fixed V1 roster dispatch, persistent effects,
# phase/fuse entities, and structure-safe navigation) are part of this engine
# identity.  Replays and mined evidence must never be silently interpreted by
# a newer algorithm.
ENGINE_VERSION = "reference-0.31.0"


@dataclass(frozen=True, slots=True)
class ActionResult:
    player: int
    accepted: bool
    reason: str | None = None
    card_id: str | None = None


class Controller(Protocol):
    def choose_action(self, engine: "BattleEngine", state: BattleState, player: int) -> SimAction: ...


class BattleEngine:
    """Stateless rules processor operating on an explicit :class:`BattleState`."""

    def __init__(
        self,
        ruleset: Ruleset | None = None,
        *,
        validate_every_tick: bool = True,
    ) -> None:
        self.ruleset = ruleset or load_ruleset()
        self.ruleset.verify_hash()
        if type(validate_every_tick) is not bool:
            raise TypeError("validate_every_tick must be boolean")
        self.validate_every_tick = validate_every_tick

    @property
    def decision_interval_ticks(self) -> int:
        """Default 250 ms policy cadence, separate from the physics tick."""

        return max(1, 250_000 // self.ruleset.tick_us)

    def new_battle(
        self,
        decks: tuple[Iterable[str], Iterable[str]] | None = None,
        *,
        seed: int = 0,
        shuffle_decks: bool = True,
    ) -> BattleState:
        source_decks = tuple(decks or (BASE_HOG_CYCLE_DECK, BASE_HOG_CYCLE_DECK))
        if len(source_decks) != 2:
            raise ValueError("a battle requires exactly two decks")
        rng = DeterministicRng(seed & ((1 << 64) - 1))
        players: list[PlayerState] = []
        for raw_deck in source_decks:
            deck = [self.ruleset.resolve_card_id(card) for card in raw_deck]
            self._validate_deck(deck)
            draw_order = list(deck)
            if shuffle_decks:
                rng.shuffle(draw_order)
            hand_size = self.ruleset.match.hand_size
            # Mirror is the one base card which is explicitly excluded from
            # an opening hand. Preserve deterministic order by swapping it
            # with the first later card instead of reshuffling the deck.
            if "mirror" in draw_order[:hand_size]:
                mirror_index = draw_order.index("mirror")
                replacement_index = next(
                    index
                    for index in range(hand_size, len(draw_order))
                    if draw_order[index] != "mirror"
                )
                draw_order[mirror_index], draw_order[replacement_index] = (
                    draw_order[replacement_index], draw_order[mirror_index]
                )
            players.append(
                PlayerState(
                    deck=tuple(deck),
                    hand=draw_order[:hand_size],
                    draw_pile=draw_order[hand_size:],
                    elixir_milli=self.ruleset.match.initial_elixir_milli,
                )
            )
        state = BattleState(
            schema_version=1,
            engine_version=ENGINE_VERSION,
            ruleset_id=self.ruleset.ruleset_id,
            ruleset_hash=self.ruleset.content_hash,
            seed=seed,
            rng_state=rng.state,
            tick=0,
            elapsed_us=0,
            phase="regulation",
            players=players,
            entities={},
            projectiles={},
            next_uid=1,
            effects={},
        )
        for site in TOWER_SITES:
            tower_id = "king-tower" if site.role == "king" else "princess-tower"
            definition = self.ruleset.tower(tower_id)
            entity = EntityState(
                uid=self._allocate_uid(state),
                card_id=tower_id,
                owner=site.owner,
                kind="tower",
                x_mtile=site.x_mtile,
                y_mtile=site.y_mtile,
                hp=definition.hitpoints,
                max_hp=definition.hitpoints,
                spawn_tick=0,
                role=site.role,
            )
            state.entities[entity.uid] = entity
        self._emit(
            state,
            "match_started",
            seed=seed,
            engine_version=ENGINE_VERSION,
            ruleset_id=self.ruleset.ruleset_id,
        )
        self.validate_state(state)
        return state

    def _validate_deck(self, deck: list[str]) -> None:
        if len(deck) != self.ruleset.match.deck_size:
            raise ValueError(f"deck must contain {self.ruleset.match.deck_size} cards")
        if len(set(deck)) != len(deck):
            raise ValueError("deck cards must be unique")
        unsupported = sorted(set(deck) - set(self.ruleset.interaction_set))
        if unsupported:
            raise ValueError(f"cards outside declared interaction set: {unsupported}")

    def step(self, state: BattleState, actions: Iterable[SimAction] = ()) -> tuple[SimEvent, ...]:
        """Advance exactly one canonical physics tick and return new events."""

        if state.terminal:
            return ()
        self._verify_state_ruleset(state)
        event_start = len(state.events)
        self._regenerate_elixir(state)
        self.apply_actions(state, actions)
        self._advance_deployments(state)
        self._advance_area_effects(state)
        self._advance_statuses_and_lifetimes(state)
        self._advance_concealment(state)
        self._sync_carried_entities(state)
        self._invalidate_and_acquire_targets(state)
        self._move_entities(state)
        self._sync_carried_entities(state)
        self._separate_entities(state)
        self._advance_attacks(state)
        self._advance_projectiles(state)
        destroyed_towers = self._resolve_deaths(state)
        self._resolve_tower_outcomes(state, destroyed_towers)
        self._advance_match_clock(state)
        state.tick += 1
        if self.validate_every_tick:
            self.validate_state(state)
        return tuple(state.events[event_start:])

    def apply_actions(self, state: BattleState, actions: Iterable[SimAction]) -> tuple[ActionResult, ...]:
        """Apply at most one action per player in stable player order."""

        by_player: dict[int, SimAction] = {}
        duplicate_players: set[int] = set()
        for action in actions:
            if action.player in by_player:
                duplicate_players.add(action.player)
            else:
                by_player[action.player] = action
        results: list[ActionResult] = []
        for player in sorted(duplicate_players):
            self._emit(state, "action_rejected", player=player, reason="multiple_actions_in_tick")
            results.append(ActionResult(player, False, "multiple_actions_in_tick"))
            by_player.pop(player, None)
        for player in sorted(by_player):
            action = by_player[player]
            reason = self.validate_action(state, action)
            if reason is not None:
                self._emit(state, "action_rejected", player=player, reason=reason)
                results.append(ActionResult(player, False, reason))
                continue
            if isinstance(action, WaitAction):
                results.append(ActionResult(player, True))
                continue
            if isinstance(action, UseAbilityAction):
                # The action exists from schema v1 so current forms can extend
                # the engine without an action-space migration. Base v0.1 has
                # no ability-bearing definition and therefore fails closed.
                self._emit(state, "action_rejected", player=player, reason="ability_not_supported")
                results.append(ActionResult(player, False, "ability_not_supported"))
                continue
            card_id = state.players[player].hand[action.card_slot]
            self._play_card(state, action, card_id)
            results.append(ActionResult(player, True, card_id=card_id))
        return tuple(results)

    def validate_action(self, state: BattleState, action: SimAction) -> str | None:
        if type(action.player) is not int or action.player not in (0, 1):
            return "invalid_player"
        if state.terminal:
            return "match_ended"
        if isinstance(action, WaitAction):
            return None
        if isinstance(action, UseAbilityAction):
            return "ability_not_supported"
        if not isinstance(action, PlayCardAction):
            return "unknown_action"
        player = state.players[action.player]
        if type(action.card_slot) is not int or not (0 <= action.card_slot < len(player.hand)):
            return "invalid_card_slot"
        card = self.ruleset.card(player.hand[action.card_slot])
        placement_card = card
        if card.card_id == "mirror":
            if player.last_played_card_id is None:
                return "mirror_no_target"
            placement_card = self.ruleset.card(player.last_played_card_id)
        effective_cost = self._effective_card_cost(player, card)
        if player.elixir_milli < effective_cost:
            return "insufficient_elixir"
        if card.card_id == "mirror" and player.last_played_card_id == "mirror":
            return "mirror_chain_not_allowed"
        if not self._legal_deployment(state, action.player, placement_card, action.cell):
            return "illegal_placement"
        return None

    def legal_cells(self, state: BattleState, player: int, card_id: str) -> tuple[tuple[int, int], ...]:
        card = self.ruleset.card(card_id)
        if card.card_id == "mirror":
            previous = state.players[player].last_played_card_id
            if previous is None or previous == "mirror":
                return ()
            card = self.ruleset.card(previous)
        return tuple(
            (col, row)
            for row in range(GRID_ROWS)
            for col in range(GRID_COLS)
            if self._legal_deployment(state, player, card, (col, row))
        )

    def run_match(
        self,
        state: BattleState,
        controllers: tuple[Controller | Callable[["BattleEngine", BattleState, int], SimAction], Controller | Callable[["BattleEngine", BattleState, int], SimAction]] | None = None,
        *,
        decision_interval_ticks: int | None = None,
        max_ticks: int | None = None,
    ) -> BattleState:
        """Run until a King falls, regulation resolves, or tiebreak completes."""

        cadence = self.decision_interval_ticks if decision_interval_ticks is None else decision_interval_ticks
        if cadence <= 0:
            raise ValueError("decision_interval_ticks must be positive")
        total_us = self.ruleset.match.regulation_us + self.ruleset.match.overtime_us
        hard_limit = total_us // self.ruleset.tick_us + 2 if max_ticks is None else max_ticks
        if hard_limit <= 0:
            raise ValueError("max_ticks must be positive")
        while not state.terminal and state.tick < hard_limit:
            actions: list[SimAction] = []
            if controllers is not None and state.tick % cadence == 0:
                for player, controller in enumerate(controllers):
                    chooser = controller.choose_action if hasattr(controller, "choose_action") else controller
                    actions.append(chooser(self, state, player))
            self.step(state, actions)
        if not state.terminal:
            self._end_match(state, None, "runner_tick_limit")
        return state

    def _play_card(self, state: BattleState, action: PlayCardAction, card_id: str) -> None:
        player = state.players[action.player]
        card = self.ruleset.card(card_id)
        previous_card_id = player.last_played_card_id
        effective_cost = self._effective_card_cost(player, card)
        player.elixir_milli -= effective_cost
        used = player.hand.pop(action.card_slot)
        next_card = player.draw_pile.pop(0)
        player.hand.append(next_card)
        player.draw_pile.append(used)
        player.cards_played += 1
        opponent_seen = state.players[1 - action.player].seen_enemy_cards
        if card_id not in opponent_seen:
            opponent_seen.append(card_id)
        col, row = action.cell
        self._emit(
            state,
            "card_played",
            player=action.player,
            card_id=card_id,
            card_slot=action.card_slot,
            col=col,
            row=row,
            cost_milli=effective_cost,
        )
        if card.card_id == "mirror":
            if previous_card_id is None:
                self._emit(state, "mirror_no_target", player=action.player)
            else:
                mirrored = self.ruleset.card(previous_card_id)
                if mirrored.kind == "spell":
                    self._spawn_spell(
                        state, action.player, mirrored, action.cell,
                        level_multiplier_permille=1_100,
                    )
                else:
                    self._spawn_card_entities(
                        state, action.player, mirrored, action.cell,
                        level_multiplier_permille=1_100,
                    )
                self._emit(
                    state,
                    "card_mirrored",
                    player=action.player,
                    source_card_id=previous_card_id,
                    cost_milli=effective_cost,
                    level_delta=1,
                )
            # Mirror consumes itself.  It is deliberately the new previous
            # card, so a second Mirror cannot mirror a Mirror (or recursively
            # manufacture an unbounded chain).  A normal card played after it
            # clears this guard in the ordinary branch below.
            player.last_played_card_id = card.card_id
        elif card.kind == "spell":
            self._spawn_spell(state, action.player, card, action.cell)
            player.last_played_card_id = card.card_id
        else:
            self._spawn_card_entities(state, action.player, card, action.cell)
            player.last_played_card_id = card.card_id

    def _effective_card_cost(self, player: PlayerState, card: CardDefinition) -> int:
        """Return a card's payable cost, including Mirror's dynamic surcharge."""

        if card.card_id != "mirror" or player.last_played_card_id is None:
            return card.elixir_milli
        previous = self.ruleset.card(player.last_played_card_id)
        return min(self.ruleset.match.max_elixir_milli, previous.elixir_milli + 1_000)

    def _spawn_card_entities(
        self,
        state: BattleState,
        player: int,
        card: CardDefinition,
        cell: tuple[int, int],
        *,
        level_multiplier_permille: int = PERMILLE,
    ) -> None:
        center_x, center_y = cell_center_mtile(cell)
        raw_layout = card.mechanics.get("spawn_layout_mtile")
        if raw_layout:
            layout = tuple((int(offset[0]), int(offset[1])) for offset in raw_layout)
        else:
            layout = self._default_spawn_layout(card)
        if len(layout) != card.spawn_count:
            raise ValueError(f"{card.card_id}: spawn layout does not match spawn_count")
        if player == 1 and card.mechanics.get("mirror_spawn_layout"):
            layout = tuple((-offset_x, -offset_y) for offset_x, offset_y in layout)
        spawn_stagger_us = int(card.mechanics.get("spawn_stagger_us") or 0)
        mixed_children = card.mechanics.get("spawn_children")
        if mixed_children:
            layout_index = 0
            for child_spec in mixed_children:
                child = self.ruleset.card(str(child_spec["card_id"]))
                count = int(child_spec["count"])
                explicit_offsets = child_spec.get("offsets_mtile")
                offsets = (
                    tuple((int(point[0]), int(point[1])) for point in explicit_offsets)
                    if explicit_offsets is not None
                    else tuple(layout[layout_index + index] for index in range(count))
                )
                if len(offsets) != count:
                    raise ValueError(f"{card.card_id}: mixed child offset/count mismatch")
                if (
                    explicit_offsets is not None
                    and player == 1
                    and card.mechanics.get("mirror_spawn_layout")
                ):
                    offsets = tuple((-offset_x, -offset_y) for offset_x, offset_y in offsets)
                for child_index, (offset_x, offset_y) in enumerate(offsets):
                    x = min(self.ruleset.arena.width_mtile - 1, max(0, center_x + offset_x))
                    y = min(self.ruleset.arena.height_mtile - 1, max(0, center_y + offset_y))
                    self._spawn_single_at(
                        state,
                        child,
                        owner=player,
                        x_mtile=x,
                        y_mtile=y,
                        event_kind="entity_created",
                        deploy_remaining_us=(
                            child.deploy_time_us
                            + (layout_index + child_index) * spawn_stagger_us
                        ),
                        level_multiplier_permille=level_multiplier_permille,
                    )
                layout_index += count
            if layout_index != card.spawn_count:
                raise ValueError(f"{card.card_id}: mixed child count does not match spawn_count")
            return
        for spawn_index, (offset_x, offset_y) in enumerate(layout):
            x = min(self.ruleset.arena.width_mtile - 1, max(0, center_x + offset_x))
            y = min(self.ruleset.arena.height_mtile - 1, max(0, center_y + offset_y))
            uid = self._allocate_uid(state)
            burrow = card.mechanics.get("burrow")
            shield = card.mechanics.get("shield")
            stealth = bool(card.mechanics.get("stealth"))
            concealment = card.mechanics.get("concealment")
            entity = EntityState(
                uid=uid,
                card_id=card.card_id,
                owner=player,
                kind=card.kind,
                x_mtile=x,
                y_mtile=y,
                hp=self._scale_level_value(int(card.hitpoints or 0), level_multiplier_permille),
                max_hp=self._scale_level_value(int(card.hitpoints or 0), level_multiplier_permille),
                spawn_tick=state.tick,
                level_multiplier_permille=level_multiplier_permille,
                deploy_remaining_us=(
                    int(burrow.get("duration_us"))
                    if hasattr(burrow, "get")
                    else card.deploy_time_us + spawn_index * spawn_stagger_us
                ),
                lifetime_remaining_us=card.lifetime_us,
                spawn_cooldown_us=(
                    int(card.mechanics["spawn"].get("start_delay_us", 0))
                    if card.mechanics.get("spawn")
                    else int(card.mechanics["elixir_generation"].get("interval_us", 0))
                    if card.mechanics.get("elixir_generation")
                    else 0
                ),
                shield_hp=(
                    self._scale_level_value(int(shield["hitpoints"]), level_multiplier_permille)
                    if hasattr(shield, "get") else 0
                ),
                shield_max_hp=(
                    self._scale_level_value(int(shield["hitpoints"]), level_multiplier_permille)
                    if hasattr(shield, "get") else 0
                ),
                stealth_active=stealth,
                stealth_remaining_us=0,
                burrow_active=burrow is not None,
                concealed_active=bool(
                    concealment and concealment.get("starts_concealed", False)
                ),
            )
            state.entities[uid] = entity
            if entity.kind == "building":
                state.navigation_revision += 1
            self._emit(
                state,
                "entity_created",
                uid=uid,
                player=player,
                card_id=card.card_id,
                x_mtile=x,
                y_mtile=y,
            )
            if burrow is not None:
                self._emit(
                    state,
                    "burrow_started",
                    uid=uid,
                    player=player,
                    card_id=card.card_id,
                    x_mtile=x,
                    y_mtile=y,
                    duration_us=int(burrow["duration_us"]),
                )
            self._spawn_carried_children(state, entity)

    def _spawn_carried_children(self, state: BattleState, carrier: EntityState) -> None:
        """Create the attached bodies declared by a carrier component."""

        definition = self._definition(carrier)
        if carrier.kind == "tower" or not isinstance(definition, CardDefinition):
            return
        raw_carrier = definition.mechanics.get("carrier")
        if raw_carrier is None:
            return
        child_id = str(raw_carrier["child_card_id"])
        child = self.ruleset.card(child_id)
        offsets = tuple(
            (int(offset[0]), int(offset[1]))
            for offset in raw_carrier["offsets_mtile"]
        )
        expected = int(raw_carrier["count"])
        if len(offsets) != expected:
            raise ValueError(f"{carrier.card_id}: carrier offset/count mismatch")
        for offset_x, offset_y in offsets:
            self._spawn_single_at(
                state,
                child,
                owner=carrier.owner,
                x_mtile=carrier.x_mtile + offset_x,
                y_mtile=carrier.y_mtile + offset_y,
                parent_uid=carrier.uid,
                event_kind="carrier_child_created",
                deploy_remaining_us=carrier.deploy_remaining_us,
                carried_by_uid=carrier.uid,
                carried_offset_mtile=(offset_x, offset_y),
                level_multiplier_permille=carrier.level_multiplier_permille,
            )

    @staticmethod
    def _default_spawn_layout(card: CardDefinition) -> tuple[tuple[int, int], ...]:
        if card.spawn_count == 1:
            return ((0, 0),)
        radius = int(card.collision_radius_mtile or 400)
        candidates = (
            (-radius, 0),
            (radius, 0),
            (0, radius),
            (0, -radius),
            (-radius, radius),
            (radius, radius),
            (-radius, -radius),
            (radius, -radius),
        )
        if card.spawn_count > len(candidates):
            raise ValueError(f"no generic formation for {card.spawn_count} spawns")
        return candidates[: card.spawn_count]

    def _spawn_spell(
        self,
        state: BattleState,
        player: int,
        card: CardDefinition,
        cell: tuple[int, int],
        *,
        level_multiplier_permille: int = PERMILLE,
    ) -> None:
        if card.projectile is None:
            raise ValueError(f"spell {card.card_id} lacks an executable projectile")
        target_x, target_y = cell_center_mtile(cell)
        mechanics = card.mechanics
        mode = mechanics.get("projectile_mode")
        impact_mode = mechanics.get("impact_mode")
        origin = mechanics.get("spell_origin")

        # Spell origin is an executable part of the card definition.  A
        # selected-position origin is used by The Log; rolling spells also
        # begin at their selected endpoint and then continue along their
        # authored travel direction.  Ballistic spells originate at the
        # player's King Tower.  Keeping this dispatch here (rather than
        # inferring it from the card id) makes mirrored and future spells use
        # the same deterministic path.
        if mode == "rolling_linear" or origin == "selected-position":
            start_x, start_y = target_x, target_y
        elif origin == "own-king-tower":
            king = self._tower(state, player, "king")
            start_x, start_y = king.x_mtile, king.y_mtile
        else:
            raise RulesetError(
                f"{card.card_id}: unsupported spell origin {origin!r}"
            )

        if mode == "rolling_linear":
            direction = -1 if player == 0 else 1
            target_y = min(
                self.ruleset.arena.height_mtile - 1,
                max(
                    0,
                    start_y
                    + direction
                    * int(card.mechanics.get("rolling_range_mtile") or card.range_mtile or 0),
                ),
            )
        raw_status = card.mechanics.get("status")
        # Continuous impact modes are swept along the projectile path.  The
        # explicit component is authoritative even if an older generated
        # card omitted the redundant ``piercing`` boolean.
        continuous = impact_mode in {"continuous", "continuous_path"}
        projectile = ProjectileState(
            uid=self._allocate_uid(state),
            source_uid=None,
            source_card_id=card.card_id,
            owner=player,
            x_mtile=start_x,
            y_mtile=start_y,
            target_x_mtile=target_x,
            target_y_mtile=target_y,
            damage=self._scale_level_value(int(card.damage or 0), level_multiplier_permille),
            crown_damage=self._scale_level_value(
                int(card.crown_tower_damage if card.crown_tower_damage is not None else card.damage or 0),
                level_multiplier_permille,
            ),
            speed_mtile_per_s=card.projectile.speed_mtile_per_s,
            speed_code=(
                int(card.mechanics["projectile_speed_code"])
                if card.mechanics.get("projectile_speed_code") is not None
                else None
            ),
            homing=card.projectile.homing,
            radius_mtile=int(card.area_radius_mtile or card.projectile.radius_mtile),
            status_kind=None if not raw_status else str(raw_status.get("kind")),
            status_duration_us=0 if not raw_status else int(raw_status.get("duration_us") or 0),
            status_magnitude_permille=(
                PERMILLE
                if not raw_status
                else int(
                    raw_status.get("speed_multiplier_milli")
                    if raw_status.get("speed_multiplier_milli") is not None
                    else PERMILLE
                )
            ),
            status_damage_per_tick=0
            if not raw_status
            else int(raw_status.get("damage_per_tick") or 0),
            status_tick_interval_us=0
            if not raw_status
            else int(raw_status.get("tick_interval_us") or 0),
            knockback_mtile=int(mechanics.get("knockback_mtile") or 0),
            piercing=bool(mechanics.get("piercing")) or continuous,
            allowed_targets=tuple(
                str(value) for value in mechanics.get("impact_targets", ())
            ),
            origin_x_mtile=start_x,
            origin_y_mtile=start_y,
            line_end_x_mtile=target_x,
            line_end_y_mtile=target_y,
            direction_x_mtile=target_x - start_x,
            direction_y_mtile=target_y - start_y,
            level_multiplier_permille=level_multiplier_permille,
        )
        state.projectiles[projectile.uid] = projectile
        self._emit(
            state,
            "projectile_spawned",
            uid=projectile.uid,
            player=player,
            card_id=card.card_id,
            source_uid=None,
            target_uid=None,
            projectile_speed_code=projectile.speed_code,
        )

    def _legal_deployment(
        self,
        state: BattleState,
        player: int,
        card: CardDefinition,
        cell: tuple[int, int],
    ) -> bool:
        placement = card.mechanics.get("placement_class")
        if placement == "spell_anywhere":
            return is_spell_cell(cell)
        if placement in {"restricted_spell", "own_ground_spell", "spells"}:
            return self._restricted_spell_cell(state, player, cell)
        if placement == "miner_anywhere":
            if not is_ground_cell(cell):
                return False
        elif not self._territory_cell(state, player, cell):
            return False
        # Clash Royale does not allow a troop (ground *or* air) to be dropped
        # on top of an existing structure.  This is independent of ownership:
        # the same exclusion applies to friendly buildings and to enemy
        # buildings in a temporarily opened deployment pocket.  Spells are
        # handled by their own placement masks and may target structures.
        if card.kind == "troop":
            x, y = cell_center_mtile(cell)
            radius = int(card.collision_radius_mtile or 0)
            for entity in state.entities.values():
                if not entity.alive or entity.kind not in {"building", "tower"}:
                    continue
                if distance_mtile(x, y, entity.x_mtile, entity.y_mtile) < radius + self._collision_radius(entity):
                    return False
        if card.kind == "building":
            if not self._building_footprint_fits(state, player, cell, 3):
                return False
            x, y = cell_center_mtile(cell)
            radius = int(card.collision_radius_mtile or 0)
            for entity in state.entities.values():
                if not entity.alive or entity.kind not in {"building", "tower"}:
                    continue
                other_radius = self._collision_radius(entity)
                if distance_mtile(x, y, entity.x_mtile, entity.y_mtile) < radius + other_radius:
                    return False
        return True

    def _building_footprint_fits(
        self,
        state: BattleState,
        player: int,
        cell: tuple[int, int],
        size: int,
    ) -> bool:
        """Apply dynamic post-tower territory to every footprint cell."""

        low = -(size // 2)
        high = size - size // 2
        col, row = cell
        return all(
            self._territory_cell(state, player, (col + dcol, row + drow))
            for drow in range(low, high)
            for dcol in range(low, high)
        )

    def _restricted_spell_cell(
        self,
        state: BattleState,
        player: int,
        cell: tuple[int, int],
    ) -> bool:
        if not is_spell_cell(cell):
            return False
        col, row = cell
        if row >= 17 if player == 0 else row <= 14:
            return True
        enemy = 1 - player
        for tower in self._towers_for(state, enemy):
            if tower.alive or tower.role == "king":
                continue
            tower_col = tower.x_mtile // 1_000
            same_lane = (col < GRID_COLS // 2) == (tower_col < GRID_COLS // 2)
            forward = 11 <= row <= 16 if player == 0 else 15 <= row <= 20
            if same_lane and forward:
                return True
        return False

    def _territory_cell(self, state: BattleState, player: int, cell: tuple[int, int]) -> bool:
        if is_basic_deploy_cell(player, cell):
            return True
        if not is_ground_cell(cell):
            return False
        col, row = cell
        # Princess loss opens the destroyed site; taking an enemy Princess
        # Tower opens the corresponding forward pocket, matching policy-v1's
        # coarse deployment-state contract with center-cell semantics.
        for tower in self._towers_for(state, player):
            if tower.alive or tower.role == "king":
                continue
            if distance_mtile(*cell_center_mtile(cell), tower.x_mtile, tower.y_mtile) <= 2_000:
                return True
        enemy = 1 - player
        for tower in self._towers_for(state, enemy):
            if tower.alive or tower.role == "king":
                continue
            same_lane = abs(cell_center_mtile(cell)[0] - tower.x_mtile) <= 5_000
            forward = 11 <= row <= 16 if player == 0 else 15 <= row <= 20
            if same_lane and forward:
                return True
        return False

    def _regenerate_elixir(self, state: BattleState) -> None:
        interval = self._elixir_interval(state.elapsed_us)
        for player in state.players:
            if player.elixir_milli >= self.ruleset.match.max_elixir_milli:
                player.elixir_milli = self.ruleset.match.max_elixir_milli
                player.elixir_remainder = 0
                continue
            numerator = self.ruleset.tick_us * ELIXIR_SCALE + player.elixir_remainder
            gain, player.elixir_remainder = divmod(numerator, interval)
            player.elixir_milli = min(self.ruleset.match.max_elixir_milli, player.elixir_milli + gain)
            if player.elixir_milli == self.ruleset.match.max_elixir_milli:
                player.elixir_remainder = 0

    def _elixir_interval(self, elapsed_us: int) -> int:
        regulation = self.ruleset.match.regulation_us
        total = regulation + self.ruleset.match.overtime_us
        if elapsed_us >= total - 60 * SECOND_US:
            return self.ruleset.match.triple_elixir_interval_us
        if elapsed_us >= regulation - 60 * SECOND_US:
            return self.ruleset.match.double_elixir_interval_us
        return self.ruleset.match.normal_elixir_interval_us

    def _advance_deployments(self, state: BattleState) -> None:
        dt = self.ruleset.tick_us
        for entity in self._alive_entities(state):
            if entity.deploy_remaining_us <= 0:
                continue
            entity.deploy_remaining_us = max(0, entity.deploy_remaining_us - dt)
            if entity.deploy_remaining_us == 0:
                definition = self._definition(entity)
                mechanics = {} if entity.kind == "tower" else definition.mechanics
                if entity.burrow_active:
                    entity.burrow_active = False
                    self._emit(
                        state,
                        "burrow_emerged",
                        uid=entity.uid,
                        player=entity.owner,
                        card_id=entity.card_id,
                        x_mtile=entity.x_mtile,
                        y_mtile=entity.y_mtile,
                    )
                self._emit(
                    state,
                    "entity_deployed",
                    uid=entity.uid,
                    player=entity.owner,
                    card_id=entity.card_id,
                )
                # Clone bodies still deploy normally, but cloned deployment
                # pulses are suppressed. This covers the copied Electro/Ice
                # Wizard, Mega Knight, and Battle Healer interactions while
                # preserving the ordinary body after the deploy event.
                deploy_effect = None if entity.is_clone else mechanics.get("deploy_effect")
                if deploy_effect is not None:
                    self._impact_area(
                        state,
                        owner=entity.owner,
                        source_uid=entity.uid,
                        source_card_id=entity.card_id,
                        x=entity.x_mtile,
                        y=entity.y_mtile,
                        damage=self._scale_level_value(
                            int(deploy_effect.get("damage") or 0),
                            entity.level_multiplier_permille,
                        ),
                        crown_damage=self._scale_level_value(
                            int(deploy_effect.get("crown_tower_damage") or 0),
                            entity.level_multiplier_permille,
                        ),
                        radius=int(deploy_effect.get("radius_mtile") or 0),
                        status=deploy_effect,
                        knockback=int(deploy_effect.get("knockback_mtile") or 0),
                        primary_target_uid=None,
                        allowed_targets=tuple(str(value) for value in deploy_effect.get("targets", ())),
                    )
                    self._emit(
                        state,
                        "deployment_effect",
                        uid=entity.uid,
                        card_id=entity.card_id,
                        effect_kind=str(deploy_effect.get("kind")),
                    )
                jump = mechanics.get("jump")
                if (
                    not entity.is_clone
                    and jump is not None
                    and bool(jump.get("spawn_damage", True))
                ):
                    self._impact_area(
                        state,
                        owner=entity.owner,
                        source_uid=entity.uid,
                        source_card_id=entity.card_id,
                        x=entity.x_mtile,
                        y=entity.y_mtile,
                        damage=self._scale_level_value(
                            int(jump.get("damage") or 0), entity.level_multiplier_permille
                        ),
                        crown_damage=self._scale_level_value(
                            int(jump.get("damage") or 0), entity.level_multiplier_permille
                        ),
                        radius=int(jump.get("radius_mtile") or 0),
                        status=None,
                        knockback=0,
                        primary_target_uid=None,
                        allowed_targets=tuple(str(value) for value in mechanics.get("impact_targets", ())) or None,
                    )
                    entity.attack_cooldown_us = int(definition.attack_interval_us or 0)
                    self._emit(
                        state,
                        "landing_attack",
                        uid=entity.uid,
                        card_id=entity.card_id,
                        x_mtile=entity.x_mtile,
                        y_mtile=entity.y_mtile,
                    )

    def _advance_area_effects(self, state: BattleState) -> None:
        """Advance persistent area components in stable UID order.

        An effect is applied immediately when created, then once per declared
        interval.  The remaining lifetime is reduced by the exact fixed-point
        tick duration; interval remainders prevent drift at non-divisible
        physics frequencies.  Effects are retained after expiry so replay
        hashes and first-divergence reports preserve their lifecycle.
        """

        dt = self.ruleset.tick_us
        for effect in [state.effects[uid] for uid in sorted(state.effects)]:
            if not effect.alive:
                continue
            if effect.max_pulses is not None and effect.pulses_applied >= effect.max_pulses:
                effect.alive = False
                self._emit(
                    state,
                    "area_effect_expired",
                    uid=effect.uid,
                    card_id=effect.source_card_id,
                )
                continue
            if effect.initial_delay_remaining_us > 0:
                effect.initial_delay_remaining_us = max(
                    0, effect.initial_delay_remaining_us - dt
                )
                effect.remaining_us = max(0, effect.remaining_us - dt)
                if effect.initial_delay_remaining_us == 0 and effect.remaining_us > 0:
                    self._apply_area_effect_tick(state, effect)
                if effect.remaining_us == 0:
                    effect.alive = False
                    self._emit(
                        state,
                        "area_effect_expired",
                        uid=effect.uid,
                        card_id=effect.source_card_id,
                    )
                continue
            numerator = dt + effect.tick_remainder_us
            ticks, effect.tick_remainder_us = divmod(numerator, effect.tick_interval_us)
            for _ in range(ticks):
                if effect.alive:
                    self._apply_area_effect_tick(state, effect)
                if effect.max_pulses is not None and effect.pulses_applied >= effect.max_pulses:
                    break
            effect.remaining_us = max(0, effect.remaining_us - dt)
            if effect.remaining_us == 0 or (
                effect.max_pulses is not None and effect.pulses_applied >= effect.max_pulses
            ):
                effect.alive = False
                self._emit(
                    state,
                    "area_effect_expired",
                    uid=effect.uid,
                    card_id=effect.source_card_id,
                )

    def _apply_area_effect_tick(self, state: BattleState, effect: AreaEffectState) -> None:
        """Apply one persistent-area pulse and its optional spawn component."""

        if effect.max_pulses is not None and effect.pulses_applied >= effect.max_pulses:
            return

        allowed_targets = effect.allowed_targets or self.ruleset.cards[
            effect.source_card_id
        ].targets
        candidates = [
            target
            for target in self._alive_entities(state)
            if target.owner != effect.owner
            and self._spell_can_hit(
                effect.source_card_id,
                target,
                allowed_targets=allowed_targets,
            )
            and distance_mtile(
                effect.x_mtile,
                effect.y_mtile,
                target.x_mtile,
                target.y_mtile,
            )
            <= effect.radius_mtile + self._collision_radius(target)
        ]
        status = None
        if effect.status_kind:
            status = {
                "kind": effect.status_kind,
                "duration_us": effect.status_duration_us,
                "speed_multiplier_milli": effect.status_magnitude_permille,
                "hit_speed_multiplier_milli": effect.status_hit_speed_magnitude_permille,
                "damage_per_tick": effect.status_damage_per_tick,
                "tick_interval_us": effect.status_tick_interval_us,
                "on_death_spawn_card_id": effect.status_on_death_spawn_card_id,
                "on_death_spawn_count": effect.status_on_death_spawn_count,
                # A plain status (Poison/Freeze/Rage) has no death child and
                # therefore must carry a null owner.  The owner is meaningful
                # only for Goblin Curse-style death transforms; assigning it
                # unconditionally leaves an invalid owner on every ordinary
                # status and fails strict authoritative-state validation.
                "on_death_spawn_owner": (
                    effect.owner
                    if effect.status_on_death_spawn_card_id is not None
                    else None
                ),
                "source_level_multiplier_permille": effect.level_multiplier_permille,
            }
        raw_effect = self.ruleset.cards[effect.source_card_id].mechanics.get(
            "persistent_effect", {}
        )
        target_count_bucket = None
        if hasattr(raw_effect, "get") and (
            raw_effect.get("damage_by_target_count")
            or raw_effect.get("crown_damage_by_target_count")
        ):
            count = len(candidates)
            target_count_bucket = "1" if count <= 1 else "2-4" if count <= 4 else "5+"
        pulse_index = effect.pulses_applied
        scheduled_damage = (
            effect.damage_schedule[pulse_index]
            if pulse_index < len(effect.damage_schedule)
            else 0
            if effect.damage_schedule
            else effect.damage_per_tick
        )
        scheduled_crown_damage = (
            effect.crown_damage_schedule[pulse_index]
            if pulse_index < len(effect.crown_damage_schedule)
            else 0
            if effect.crown_damage_schedule
            else effect.crown_damage_per_tick
        )
        body_damage = scheduled_damage
        crown_damage = scheduled_crown_damage
        for target in candidates:
            if target.kind == "tower":
                damage_map = raw_effect.get("crown_damage_by_target_count", {}) if hasattr(raw_effect, "get") else {}
                damage = (
                    self._scale_level_value(
                        int(damage_map[target_count_bucket]),
                        effect.level_multiplier_permille,
                    )
                    if target_count_bucket in damage_map
                    else crown_damage
                )
            else:
                damage_map = raw_effect.get("damage_by_target_count", {}) if hasattr(raw_effect, "get") else {}
                damage = (
                    self._scale_level_value(
                        int(damage_map[target_count_bucket]),
                        effect.level_multiplier_permille,
                    )
                    if target_count_bucket in damage_map
                    else body_damage
                )
                if target.kind == "building" and hasattr(raw_effect, "get"):
                    if raw_effect.get("building_damage_per_tick") is not None:
                        damage = self._scale_level_value(
                            int(raw_effect.get("building_damage_per_tick") or 0),
                            effect.level_multiplier_permille,
                        )
            # A curse must be attached before a lethal pulse so the death
            # conversion still fires.  Ordinary statuses retain the legacy
            # post-damage ordering, which keeps existing projectile timing
            # fixtures unchanged.
            curse_status = bool(
                status is not None and status.get("on_death_spawn_card_id")
            )
            if curse_status and target.hp > 0:
                self._apply_status(state, target, status)
            self._deal_damage(
                state,
                target,
                damage,
                effect.source_uid,
                effect.source_card_id,
            )
            if target.hp > 0 and status is not None and not curse_status:
                self._apply_status(state, target, status)
            if target.hp > 0 and effect.knockback_mtile:
                self._apply_knockback(
                    state,
                    target,
                    effect.x_mtile,
                    effect.y_mtile,
                    effect.knockback_mtile,
                )
            if target.hp > 0 and effect.pull_to_center_mtile:
                self._apply_pull_to_center(state, target, effect)
        if effect.friendly_status_kind and effect.friendly_status_duration_us > 0:
            friendly_status = {
                "kind": effect.friendly_status_kind,
                "duration_us": effect.friendly_status_duration_us,
                "speed_multiplier_milli": effect.friendly_status_magnitude_permille,
                "hit_speed_multiplier_milli": effect.friendly_status_magnitude_permille,
            }
            for target in self._alive_entities(state):
                if (
                    target.owner != effect.owner
                    or target.kind == "tower"
                    or not self._spell_can_hit(
                        effect.source_card_id,
                        target,
                        allowed_targets=effect.friendly_allowed_targets,
                    )
                    or distance_mtile(
                        effect.x_mtile,
                        effect.y_mtile,
                        target.x_mtile,
                        target.y_mtile,
                    )
                    > effect.radius_mtile + self._collision_radius(target)
                ):
                    continue
                self._apply_status(state, target, friendly_status)
        if (
            effect.spawn_card_id is not None
            and effect.spawn_count > 0
            and effect.spawned_count < effect.max_spawns
        ):
            child = self.ruleset.card(effect.spawn_card_id)
            for _ in range(
                min(effect.spawn_count, effect.max_spawns - effect.spawned_count)
            ):
                spawn_x, spawn_y = effect.x_mtile, effect.y_mtile
                source_definition = self.ruleset.cards.get(effect.source_card_id)
                persistent = (
                    None
                    if source_definition is None
                    else source_definition.mechanics.get("persistent_effect")
                )
                spawn_spec = None if not persistent else persistent.get("spawn")
                offsets = None if not spawn_spec else spawn_spec.get("offsets_mtile")
                if offsets:
                    offset = offsets[effect.spawned_count % len(offsets)]
                    offset_x, offset_y = int(offset[0]), int(offset[1])
                    if effect.owner == 1:
                        offset_x, offset_y = -offset_x, -offset_y
                    spawn_x += offset_x
                    spawn_y += offset_y
                self._spawn_single_at(
                    state,
                    child,
                    owner=effect.owner,
                    x_mtile=spawn_x,
                    y_mtile=spawn_y,
                    parent_uid=effect.source_uid,
                    level_multiplier_permille=effect.level_multiplier_permille,
                )
                effect.spawned_count += 1
        self._emit(
            state,
            "area_effect_pulse",
            uid=effect.uid,
            card_id=effect.source_card_id,
            pulse_index=pulse_index,
            target_count=len(candidates),
            damage=body_damage,
            crown_damage=crown_damage,
            spawned_count=effect.spawned_count,
        )
        effect.pulses_applied += 1

    def _apply_pull_to_center(
        self,
        state: BattleState,
        target: EntityState,
        effect: AreaEffectState,
    ) -> None:
        """Move a valid target toward an effect center without tunneling."""

        if target.kind in {"tower", "building"} or not target.alive:
            return
        dx = effect.x_mtile - target.x_mtile
        dy = effect.y_mtile - target.y_mtile
        distance = distance_mtile(0, 0, dx, dy)
        if distance <= 0:
            return
        destination = move_towards(
            target.x_mtile,
            target.y_mtile,
            effect.x_mtile,
            effect.y_mtile,
            min(effect.pull_to_center_mtile, distance),
        )
        if self._position_clear_of_structures(
            state,
            target,
            *destination,
            exclude_target=False,
        ):
            target.x_mtile, target.y_mtile = destination
            target.navigation_waypoints.clear()
            target.navigation_cursor = 0
            target.navigation_revision = -1

    def _advance_statuses_and_lifetimes(self, state: BattleState) -> None:
        dt = self.ruleset.tick_us
        for entity in self._alive_entities(state):
            if not entity.stealth_active and entity.stealth_remaining_us > 0:
                entity.stealth_remaining_us = max(0, entity.stealth_remaining_us - dt)
                if entity.stealth_remaining_us == 0:
                    entity.stealth_active = True
                    self._emit(
                        state,
                        "stealth_started",
                        uid=entity.uid,
                        card_id=entity.card_id,
                    )
            if entity.jump_remaining_us > 0:
                entity.jump_remaining_us = max(0, entity.jump_remaining_us - dt)
                if entity.jump_remaining_us == 0:
                    landing_x = entity.jump_landing_x_mtile
                    landing_y = entity.jump_landing_y_mtile
                    if self._position_clear_of_structures(
                        state,
                        entity,
                        landing_x,
                        landing_y,
                        exclude_target=True,
                    ):
                        entity.x_mtile, entity.y_mtile = landing_x, landing_y
                    jump = self.ruleset.cards[entity.card_id].mechanics.get("jump", {})
                    self._impact_area(
                        state,
                        owner=entity.owner,
                        source_uid=entity.uid,
                        source_card_id=entity.card_id,
                        x=entity.x_mtile,
                        y=entity.y_mtile,
                        damage=self._scale_level_value(
                            int(jump.get("damage") or 0), entity.level_multiplier_permille
                        ),
                        crown_damage=self._scale_level_value(
                            int(jump.get("damage") or 0), entity.level_multiplier_permille
                        ),
                        radius=int(jump.get("radius_mtile") or 0),
                        status=None,
                        knockback=0,
                        primary_target_uid=None,
                        allowed_targets=tuple(str(value) for value in self.ruleset.cards[entity.card_id].mechanics.get("impact_targets", ())) or None,
                    )
                    entity.jump_target_uid = None
                    entity.attack_cooldown_us = int(self.ruleset.cards[entity.card_id].attack_interval_us or 0)
                    self._emit(
                        state,
                        "jump_landed",
                        uid=entity.uid,
                        card_id=entity.card_id,
                        x_mtile=entity.x_mtile,
                        y_mtile=entity.y_mtile,
                    )
            # Some cards change their kind at a health boundary rather than
            # dying or spawning a second UID (Cannon Cart's post-May-2025
            # shared-health rework).  Re-check at tick entry as well as on
            # every damage event so replay/state loading cannot leave a
            # below-threshold cart in its mobile form.
            self._maybe_transform_health(state, entity)
            definition = self._definition(entity)
            mechanics = {} if entity.kind == "tower" else definition.mechanics
            threshold = mechanics.get("charge_threshold_permille")
            if (
                threshold is not None
                and not entity.charge_active
                and entity.max_hp > 0
                and entity.hp * PERMILLE <= entity.max_hp * int(threshold)
            ):
                entity.charge_active = True
                duration = mechanics.get("charge_duration_us")
                entity.charge_remaining_us = None if duration is None else int(duration)
                self._emit(
                    state,
                    "phase_changed",
                    uid=entity.uid,
                    card_id=entity.card_id,
                    phase="charge",
                )
            if entity.charge_active and entity.charge_remaining_us is not None:
                entity.charge_remaining_us = max(0, entity.charge_remaining_us - dt)
                if entity.charge_remaining_us == 0:
                    entity.hp = 0
                    self._emit(
                        state,
                        "fuse_expired",
                        uid=entity.uid,
                        card_id=entity.card_id,
                    )
            remaining_statuses: list[StatusState] = []
            for status in entity.statuses:
                if status.damage_per_tick > 0 and status.tick_interval_us > 0:
                    numerator = dt + status.tick_remainder_us
                    ticks, status.tick_remainder_us = divmod(
                        numerator, status.tick_interval_us
                    )
                    for _ in range(ticks):
                        if not entity.alive or entity.hp <= 0:
                            break
                        self._deal_damage(
                            state,
                            entity,
                            status.damage_per_tick,
                            source_uid=None,
                            source_card_id=f"status:{status.kind}",
                        )
                status.remaining_us = max(0, status.remaining_us - dt)
                if status.remaining_us:
                    remaining_statuses.append(status)
                else:
                    self._emit(state, "status_expired", uid=entity.uid, status=status.kind)
            entity.statuses = remaining_statuses
            if entity.lifetime_remaining_us is None:
                continue
            if (
                entity.deploy_remaining_us > 0
                and mechanics.get("lifetime_start") != "placement"
            ):
                continue
            entity.lifetime_remaining_us = max(0, entity.lifetime_remaining_us - dt)
            lifetime_us = getattr(definition, "lifetime_us", None)
            if (
                lifetime_us is not None
                and mechanics.get("lifetime_decay") == "linear_hp"
            ):
                numerator = entity.max_hp * dt + entity.lifetime_decay_remainder
                decay, entity.lifetime_decay_remainder = divmod(numerator, lifetime_us)
                entity.hp = max(0, entity.hp - decay)
            if entity.lifetime_remaining_us == 0:
                entity.hp = 0
                if mechanics.get("revive_egg") is not None:
                    entity.hatch_due = True
                    self._emit(
                        state,
                        "egg_ready_to_hatch",
                        uid=entity.uid,
                        card_id=entity.card_id,
                    )
                self._emit(state, "building_expired", uid=entity.uid, card_id=entity.card_id)
        self._advance_spawners(state, dt)

    def _advance_spawners(self, state: BattleState, dt: int) -> None:
        """Advance data-driven building and active-troop spawners in UID order."""

        for parent in self._alive_entities(state):
            if parent.kind == "tower":
                continue
            if parent.deploy_remaining_us > 0:
                continue
            clock_progress = self._spawn_time_progress(parent, dt)
            definition = self._definition(parent)
            generation = definition.mechanics.get("elixir_generation")
            if generation:
                parent.spawn_cooldown_us = max(0, parent.spawn_cooldown_us - clock_progress)
                if parent.spawn_cooldown_us == 0:
                    player = state.players[parent.owner]
                    before = player.elixir_milli
                    player.elixir_milli = min(
                        self.ruleset.match.max_elixir_milli,
                        player.elixir_milli + int(generation["amount_milli"]),
                    )
                    if player.elixir_milli != before:
                        self._emit(
                            state,
                            "elixir_generated",
                            uid=parent.uid,
                            player=parent.owner,
                            amount_milli=player.elixir_milli - before,
                        )
                    parent.spawn_cooldown_us = int(generation["interval_us"])
            raw_spawn = definition.mechanics.get("spawn")
            if not raw_spawn or parent.kind not in {"building", "troop"}:
                continue
            spawn = raw_spawn
            activation_range = int(spawn.get("activation_range_mtile") or 0)
            if activation_range > 0:
                visible_enemy = any(
                    target.owner != parent.owner
                    and target.kind != "tower"
                    and self._targetable_for_acquisition(state, target)
                    and distance_mtile(
                        parent.x_mtile,
                        parent.y_mtile,
                        target.x_mtile,
                        target.y_mtile,
                    ) <= activation_range + self._collision_radius(target)
                    for target in self._alive_entities(state)
                )
                if visible_enemy != parent.spawner_active:
                    parent.spawner_active = visible_enemy
                    self._emit(
                        state,
                        "spawner_activation_changed",
                        uid=parent.uid,
                        card_id=parent.card_id,
                        active=visible_enemy,
                    )
                if not visible_enemy:
                    continue
            parent.spawn_cooldown_us = max(0, parent.spawn_cooldown_us - clock_progress)
            if parent.spawn_cooldown_us > 0:
                continue
            child_card_id = str(spawn["card_id"])
            if child_card_id not in self.ruleset.cards:
                raise ValueError(
                    f"{parent.card_id} spawner references unknown child {child_card_id!r}"
                )
            raw_max_alive = spawn.get("max_alive")
            max_alive = None if raw_max_alive is None else int(raw_max_alive)
            alive_children = sum(
                1
                for entity in self._alive_entities(state)
                if entity.owner == parent.owner and entity.card_id == child_card_id
            )
            # ``None`` is an explicit unbounded stream.  It is needed for the
            # post-2025 Furnace rework: the official notes specify one Fire
            # Spirit per cadence but do not publish a maximum number alive.
            # Other spawners retain their sourced finite caps.
            if max_alive is None or alive_children < max_alive:
                for _ in range(int(spawn["count"])):
                    if max_alive is not None and alive_children >= max_alive:
                        break
                    self._spawn_single_child(
                        state,
                        parent,
                        self.ruleset.card(child_card_id),
                        deploy_remaining_us=(
                            int(spawn["child_deploy_time_us"])
                            if spawn.get("child_deploy_time_us") is not None
                            else None
                        ),
                    )
                    alive_children += 1
                    parent.spawned_count += 1
            # A blocked spawner still waits one complete interval; this avoids
            # a death burst when a crowded lane suddenly becomes available.
            parent.spawn_cooldown_us = int(spawn["interval_us"])

    def _advance_concealment(self, state: BattleState) -> None:
        """Raise and lower Tesla-style structures from visible enemy sight."""

        for entity in self._alive_entities(state):
            definition = self.ruleset.cards.get(entity.card_id)
            if definition is None:
                continue
            component = definition.mechanics.get("concealment")
            if not component:
                continue
            if (
                bool(component.get("freeze_suppresses_reveal"))
                and self._is_frozen(entity)
            ):
                should_conceal = True
            else:
                reveal_range = int(component.get("reveal_range_mtile") or 0)
                should_conceal = not any(
                    target.owner != entity.owner
                    and target.kind != "tower"
                    and self._targetable_for_acquisition(state, target)
                    and distance_mtile(
                        entity.x_mtile,
                        entity.y_mtile,
                        target.x_mtile,
                        target.y_mtile,
                    ) <= reveal_range + self._collision_radius(target)
                    for target in self._alive_entities(state)
                )
            if should_conceal == entity.concealed_active:
                continue
            entity.concealed_active = should_conceal
            if should_conceal:
                entity.target_uid = None
                entity.pending_target_uid = None
                entity.windup_remaining_us = 0
            self._emit(
                state,
                "entity_concealment_changed",
                uid=entity.uid,
                card_id=entity.card_id,
                concealed=should_conceal,
            )

    def _spawn_single_child(
        self,
        state: BattleState,
        parent: EntityState,
        child: CardDefinition,
        *,
        offset_mtile: tuple[int, int] = (0, 0),
        deploy_remaining_us: int | None = None,
    ) -> None:
        # Clone provenance applies to the whole lifecycle. Death payloads,
        # spawner waves, and status conversions produced by a copied body are
        # copied bodies too; they keep the one-HP clone cap and clone shield
        # semantics instead of silently becoming full-stat children.
        is_clone = bool(parent.is_clone)
        parent_mechanics = self._definition(parent).mechanics
        authored_child_hp = parent_mechanics.get("spawn_child_hitpoints")
        child_hp_override = (
            self._scale_level_value(
                int(authored_child_hp), parent.level_multiplier_permille
            )
            if authored_child_hp is not None and not is_clone
            else (1 if is_clone else None)
        )
        self._spawn_single_at(
            state,
            child,
            owner=parent.owner,
            x_mtile=parent.x_mtile + offset_mtile[0],
            y_mtile=parent.y_mtile + offset_mtile[1],
            parent_uid=parent.uid,
            event_kind="entity_spawned",
            is_clone=is_clone,
            hp_override=child_hp_override,
            max_hp_override=child_hp_override,
            deploy_remaining_us=deploy_remaining_us,
            level_multiplier_permille=parent.level_multiplier_permille,
        )

    @staticmethod
    def _death_spawn_offsets(count: int) -> tuple[tuple[int, int], ...]:
        """Return deterministic separation offsets for one death stream.

        Child bodies are not all created at the exact parent center in the
        game.  In particular, a Battle Ram breaking at a building edge must
        release Barbarians without leaving them intersecting the building. A
        conservative 0.8-tile ring keeps strict state validation valid while
        preserving a deterministic placeholder until card-specific footage
        supplies exact offsets.  The helper is intentionally shared by death
        streams; card-specific layouts can replace it later without changing
        UID ordering or event semantics.
        """

        if count <= 0:
            return ()
        candidates = (
            (-800, 0),
            (800, 0),
            (0, 800),
            (0, -800),
            (-800, 800),
            (800, 800),
            (-800, -800),
            (800, -800),
        )
        return tuple(candidates[index % len(candidates)] for index in range(count))

    def _spawn_single_at(
        self,
        state: BattleState,
        child: CardDefinition,
        *,
        owner: int,
        x_mtile: int,
        y_mtile: int,
        parent_uid: int | None = None,
        event_kind: str = "entity_spawned",
        is_clone: bool = False,
        hp_override: int | None = None,
        max_hp_override: int | None = None,
        revive_eligible: bool | None = None,
        deploy_remaining_us: int | None = None,
        carried_by_uid: int | None = None,
        carried_offset_mtile: tuple[int, int] = (0, 0),
        level_multiplier_permille: int = PERMILLE,
    ) -> EntityState:
        uid = self._allocate_uid(state)
        x = min(self.ruleset.arena.width_mtile - 1, max(0, x_mtile))
        y = min(self.ruleset.arena.height_mtile - 1, max(0, y_mtile))
        maximum_hp = self._scale_level_value(
            int(child.hitpoints or 1), level_multiplier_permille
        )
        burrow = child.mechanics.get("burrow")
        shield = child.mechanics.get("shield")
        stealth = bool(child.mechanics.get("stealth"))
        concealment = child.mechanics.get("concealment")
        entity = EntityState(
            uid=uid,
            card_id=child.card_id,
            owner=owner,
            kind=child.kind,
            x_mtile=x,
            y_mtile=y,
            hp=maximum_hp if hp_override is None else int(hp_override),
            max_hp=maximum_hp if max_hp_override is None else int(max_hp_override),
            spawn_tick=state.tick,
            level_multiplier_permille=level_multiplier_permille,
            deploy_remaining_us=(
                int(deploy_remaining_us)
                if deploy_remaining_us is not None
                else (
                    int(burrow.get("duration_us"))
                    if hasattr(burrow, "get")
                    else child.deploy_time_us
                )
            ),
            lifetime_remaining_us=child.lifetime_us,
            is_clone=is_clone,
            revive_eligible=(not is_clone) if revive_eligible is None else revive_eligible,
            carried_by_uid=carried_by_uid,
            carried_offset_x_mtile=int(carried_offset_mtile[0]),
            carried_offset_y_mtile=int(carried_offset_mtile[1]),
            shield_hp=(
                # Cloned shielded troops keep the shield layer, but the
                # shield itself is capped at one HP just like the copied
                # body (Guards, Dark Prince, Royal Recruits, ...).
                1
                if is_clone and hasattr(shield, "get")
                else (
                    self._scale_level_value(int(shield["hitpoints"]), level_multiplier_permille)
                    if hasattr(shield, "get") else 0
                )
            ),
            shield_max_hp=(
                1
                if is_clone and hasattr(shield, "get")
                else (
                    self._scale_level_value(int(shield["hitpoints"]), level_multiplier_permille)
                    if hasattr(shield, "get") else 0
                )
            ),
            stealth_active=stealth,
            stealth_remaining_us=0,
            burrow_active=burrow is not None,
            concealed_active=bool(
                concealment and concealment.get("starts_concealed", False)
            ),
        )
        if entity.max_hp <= 0 or not 0 < entity.hp <= entity.max_hp:
            raise ValueError(f"{child.card_id}: invalid spawned HP override")
        state.entities[uid] = entity
        if entity.kind == "building":
            state.navigation_revision += 1
        self._emit(
            state,
            event_kind,
            uid=uid,
            parent_uid=parent_uid,
            player=owner,
            card_id=child.card_id,
            x_mtile=x,
            y_mtile=y,
            carried_by_uid=carried_by_uid,
        )
        if burrow is not None:
            self._emit(
                state,
                "burrow_started",
                uid=uid,
                player=owner,
                card_id=child.card_id,
                x_mtile=x,
                y_mtile=y,
                duration_us=int(burrow["duration_us"]),
            )
        return entity

    def _invalidate_and_acquire_targets(self, state: BattleState) -> None:
        for entity in self._alive_entities(state):
            if entity.deploy_remaining_us > 0:
                continue
            if entity.target_uid is not None:
                current = state.entities.get(entity.target_uid)
                invalid = not self._valid_target(state, entity, entity.target_uid)
                definition = self._definition(entity)
                if (
                    not invalid
                    and current is not None
                    and current.kind != "tower"
                    and self._edge_distance(entity, current) > self._sight_range(entity) * 2
                ):
                    invalid = True
                if invalid:
                    old_target = entity.target_uid
                    self._reset_attack_charge(state, entity, reason="target_invalidated")
                    self._reset_dash(state, entity, reason="target_invalidated")
                    self._reset_attack_ramp(state, entity, reason="target_invalidated")
                    entity.target_uid = None
                    self._emit(state, "target_changed", uid=entity.uid, old_target=old_target, target_uid=None)
                elif current is not None and current.kind == "tower" and entity.kind != "tower":
                    # A newly deployed defender/building must be able to pull a
                    # unit which was previously pathing to a Crown Tower.
                    nearby = self._nearby_non_tower_targets(state, entity)
                    replacement = self._preferred_target_uid(state, entity, nearby)
                    if replacement is not None:
                        old_target = entity.target_uid
                        self._reset_attack_charge(state, entity, reason="retargeted")
                        self._reset_dash(state, entity, reason="retargeted")
                        self._reset_attack_ramp(state, entity, reason="retargeted")
                        entity.target_uid = replacement
                        self._emit(
                            state,
                            "target_changed",
                            uid=entity.uid,
                            old_target=old_target,
                            target_uid=replacement,
                        )
            if entity.target_uid is None:
                target_uid = self._choose_target(state, entity)
                if target_uid is not None:
                    entity.target_uid = target_uid
                    self._emit(state, "target_changed", uid=entity.uid, old_target=None, target_uid=target_uid)

    def _valid_target(self, state: BattleState, source: EntityState, target_uid: int) -> bool:
        target = state.entities.get(target_uid)
        return bool(
            target
            and target.alive
            and target.hp > 0
            and target.owner != source.owner
            and self._targetable_for_acquisition(state, target)
            and self._target_allowed(source, target)
        )

    def _targetable_for_acquisition(
        self,
        state: BattleState | EntityState,
        target: EntityState | None = None,
    ) -> bool:
        # Keep the pre-state-aware helper signature usable by research
        # fixtures; all engine call sites pass (state, target).
        if target is None:
            target = state  # type: ignore[assignment]
            state = None  # type: ignore[assignment]
        definition = self.ruleset.cards.get(target.card_id)
        if target.concealed_active:
            return False
        if target.stealth_active or (
            definition is not None
            and definition.mechanics.get("stealth")
            and definition.mechanics.get("stealth_recloak_us") is None
        ):
            return False
        if target.burrow_active:
            return bool(
                definition is not None
                and definition.mechanics.get("burrow", {}).get(
                    "targetable_during_burrow", False
                )
            )
        if target.deploy_remaining_us <= 0:
            if target.kind != "tower":
                definition = self.ruleset.cards[target.card_id]
                if definition.mechanics.get("stealth") and target.stealth_active:
                    return False
            return True
        if target.kind == "tower":
            return True
        definition = self.ruleset.cards[target.card_id]
        if definition.mechanics.get("stealth") and target.stealth_active:
            return False
        return bool(definition.mechanics.get("targetable_during_deploy"))

    def _choose_target(self, state: BattleState, source: EntityState) -> int | None:
        definition = self._definition(source)
        if source.concealed_active:
            return None
        # Passive collectors/spawners have no attack cadence or range.  They
        # remain valid entities in the roster-complete ruleset but must not be
        # assigned a target (which would otherwise reach ``int(None)`` in the
        # attack scheduler).
        if (
            definition.attack_interval_us is None
            or definition.damage is None
            or definition.range_mtile is None
            or definition.sight_range_mtile is None
        ):
            return None
        sight = self._sight_range(source)
        if source.kind == "tower":
            if source.role == "king" and not state.players[source.owner].king_active:
                return None
            nearby = [
                target
                for target in self._alive_entities(state)
                if target.owner != source.owner
                and target.kind != "tower"
                and self._targetable_for_acquisition(state, target)
                and self._target_allowed(source, target)
                and self._edge_distance(source, target) <= sight
            ]
            return self._nearest_uid(source, nearby)
        nearby = self._nearby_non_tower_targets(state, source)
        nearby.extend(
            target
            for target in self._alive_entities(state)
            if target.owner != source.owner
            and target.kind == "tower"
            and self._target_allowed(source, target)
            and self._edge_distance(source, target) <= sight
            and self._edge_distance(source, target)
            >= int(definition.mechanics.get("min_attack_range_mtile") or 0)
        )
        if nearby:
            return self._preferred_target_uid(state, source, nearby)
        towers = [
            target
            for target in self._alive_entities(state)
            if target.owner != source.owner
            and target.kind == "tower"
            and target.role != "king"
            and self._target_allowed(source, target)
            and self._edge_distance(source, target)
            >= int(definition.mechanics.get("min_attack_range_mtile") or 0)
        ]
        if not towers:
            towers = [
                target
                for target in self._alive_entities(state)
                if target.owner != source.owner
                and target.kind == "tower"
                and self._target_allowed(source, target)
                and self._edge_distance(source, target)
                >= int(definition.mechanics.get("min_attack_range_mtile") or 0)
            ]
        return self._nearest_uid(source, towers)

    def _nearby_non_tower_targets(
        self,
        state: BattleState,
        source: EntityState,
    ) -> list[EntityState]:
        definition = self._definition(source)
        if definition.sight_range_mtile is None or definition.damage is None:
            return []
        sight = self._sight_range(source)
        return [
            target
            for target in self._alive_entities(state)
            if target.owner != source.owner
            and target.kind != "tower"
            and self._targetable_for_acquisition(state, target)
            and self._target_allowed(source, target)
            and self._edge_distance(source, target) <= sight
            and self._edge_distance(source, target)
            >= int(definition.mechanics.get("min_attack_range_mtile") or 0)
        ]

    @staticmethod
    def _nearest_uid(source: EntityState, candidates: list[EntityState]) -> int | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda target: (
                (source.x_mtile - target.x_mtile) ** 2 + (source.y_mtile - target.y_mtile) ** 2,
                target.uid,
            ),
        ).uid

    def _preferred_target_uid(
        self,
        state: BattleState,
        source: EntityState,
        candidates: list[EntityState],
    ) -> int | None:
        if not candidates:
            return None
        definition = self._definition(source)
        if source.kind == "tower" or not definition.mechanics.get("spread_targets"):
            return self._nearest_uid(source, candidates)
        sibling_targets: dict[int, int] = {}
        for sibling in self._alive_entities(state):
            if (
                sibling.owner == source.owner
                and sibling.card_id == source.card_id
                and sibling.spawn_tick == source.spawn_tick
                and sibling.uid != source.uid
                and sibling.target_uid is not None
            ):
                sibling_targets[sibling.target_uid] = (
                    sibling_targets.get(sibling.target_uid, 0) + 1
                )
        return min(
            candidates,
            key=lambda target: (
                sibling_targets.get(target.uid, 0),
                (source.x_mtile - target.x_mtile) ** 2
                + (source.y_mtile - target.y_mtile) ** 2,
                target.uid,
            ),
        ).uid

    def _target_allowed(self, source: EntityState, target: EntityState) -> bool:
        definition = self._definition(source)
        mechanics = {} if source.kind == "tower" else definition.mechanics
        authored_primary = mechanics.get("primary_targets")
        targets = set(authored_primary if authored_primary is not None else definition.targets)
        if source.charge_active and mechanics.get("charge_threshold_permille") is not None:
            # Goblin Demolisher's low-health phase becomes building-only.
            targets = {"building", "crown_tower"}
        if target.kind == "tower":
            # The August 2026 Spirit rules explicitly remove an unassisted
            # Crown-Tower connection.  This is distinct from the authored
            # movement/impact target classes: Spirits may still acquire and
            # attack ordinary ground/building targets, but a bare Crown Tower
            # must not be selected as their fallback target.
            if mechanics.get("crown_tower_connection") == "expected-no-unassisted-connection":
                return False
            return "crown_tower" in targets or "ground" in targets or "building" in targets
        if target.kind == "building":
            return "building" in targets or "ground" in targets
        target_definition = self.ruleset.cards[target.card_id]
        layer = self._movement_layer(target)
        return str(layer) in targets

    def _sync_carried_entities(self, state: BattleState) -> None:
        """Keep attached bodies at their carrier-relative offsets."""

        for child in self._alive_entities(state):
            carrier_uid = child.carried_by_uid
            if carrier_uid is None:
                continue
            carrier = state.entities.get(carrier_uid)
            if carrier is None or not carrier.alive:
                # A carrier is normally released by the death queue.  Leaving
                # the relation intact here would make a malformed replay
                # silently drag a child behind a dead/missing parent.
                child.carried_by_uid = None
                continue
            child.x_mtile = min(
                self.ruleset.arena.width_mtile - 1,
                max(0, carrier.x_mtile + child.carried_offset_x_mtile),
            )
            child.y_mtile = min(
                self.ruleset.arena.height_mtile - 1,
                max(0, carrier.y_mtile + child.carried_offset_y_mtile),
            )
            child.navigation_waypoints.clear()
            child.navigation_cursor = 0
            child.navigation_revision = -1

    def _move_entities(self, state: BattleState) -> None:
        dt = self.ruleset.tick_us
        for entity in self._alive_entities(state):
            if (
                entity.kind != "troop"
                or entity.carried_by_uid is not None
                or entity.deploy_remaining_us > 0
                or entity.burrow_active
                or entity.target_uid is None
            ):
                continue
            if entity.jump_remaining_us > 0:
                continue
            if self._is_frozen(entity):
                continue
            target = state.entities.get(entity.target_uid)
            if target is None or not target.alive:
                continue
            definition = self.ruleset.cards[entity.card_id]
            jump = definition.mechanics.get("jump")
            if jump is not None:
                edge_distance = self._edge_distance(entity, target)
                if (
                    int(jump.get("min_range_mtile") or 0)
                    <= edge_distance
                    <= int(jump.get("max_range_mtile") or 0)
                ):
                    entity.jump_remaining_us = int(jump.get("duration_us") or 1)
                    entity.jump_target_uid = target.uid
                    entity.jump_landing_x_mtile = target.x_mtile
                    entity.jump_landing_y_mtile = target.y_mtile
                    entity.navigation_waypoints.clear()
                    entity.navigation_cursor = 0
                    entity.navigation_revision = -1
                    self._emit(
                        state,
                        "jump_started",
                        uid=entity.uid,
                        card_id=entity.card_id,
                        target_uid=target.uid,
                        landing_x_mtile=target.x_mtile,
                        landing_y_mtile=target.y_mtile,
                    )
                    continue
            charge_attack = definition.mechanics.get("charge_attack")
            dash = definition.mechanics.get("dash")
            if dash is not None and not entity.dash_attack_active:
                edge_distance = self._edge_distance(entity, target)
                dash_range = int(dash.get("dash_range_mtile") or 0)
                minimum = int(dash.get("min_dash_distance_mtile") or 0)
                if minimum <= edge_distance <= dash_range:
                    center_distance = distance_mtile(
                        entity.x_mtile,
                        entity.y_mtile,
                        target.x_mtile,
                        target.y_mtile,
                    )
                    landing_gap = int(definition.range_mtile or 0) + self._collision_radius(entity) + self._collision_radius(target)
                    travel = max(0, center_distance - landing_gap)
                    old_position = (entity.x_mtile, entity.y_mtile)
                    entity.x_mtile, entity.y_mtile = move_towards(
                        entity.x_mtile,
                        entity.y_mtile,
                        target.x_mtile,
                        target.y_mtile,
                        travel,
                    )
                    entity.dash_attack_active = True
                    entity.navigation_waypoints.clear()
                    entity.navigation_cursor = 0
                    entity.navigation_revision = -1
                    self._emit(
                        state,
                        "dash_started",
                        uid=entity.uid,
                        card_id=entity.card_id,
                        target_uid=target.uid,
                        from_x_mtile=old_position[0],
                        from_y_mtile=old_position[1],
                        to_x_mtile=entity.x_mtile,
                        to_y_mtile=entity.y_mtile,
                    )
                    continue
            in_range = self._in_attack_range(entity, target)
            # Some contact-spawn bodies publish their trigger reach as a
            # semantic range (currently Suspicious Bush's ``long`` range)
            # instead of a separate attack weapon.  Read that authored field
            # explicitly so a future numeric range can extend the trigger
            # without turning the parent into a damaging attack.
            authored_spawn_range = definition.mechanics.get("spawn_range")
            if (
                definition.mechanics.get("trigger_on_target")
                and authored_spawn_range is not None
            ):
                if isinstance(authored_spawn_range, (int, float)):
                    spawn_range_mtile = int(authored_spawn_range)
                elif str(authored_spawn_range).lower() == "long":
                    # The card's Level-11 range scalar is the normalized
                    # world-space value for the authored Long trigger.
                    spawn_range_mtile = int(definition.range_mtile or 0)
                else:
                    spawn_range_mtile = 0
                in_range = in_range or (
                    spawn_range_mtile > 0
                    and self._edge_distance(entity, target) <= spawn_range_mtile
                )
            # Suicide contact troops (Suspicious Bush/Wall Breakers) stop at
            # the navigation collision boundary, which can leave a small
            # fixed-point gap while their authored attack range is zero.  A
            # quarter-tile contact tolerance represents the same physical
            # collision envelope and prevents a troop from parking forever in
            # front of its building-only target.
            trigger_limit_mtile = 250
            if (
                definition.mechanics.get("trigger_on_target")
                and authored_spawn_range is not None
            ):
                if isinstance(authored_spawn_range, (int, float)):
                    trigger_limit_mtile = max(250, int(authored_spawn_range))
                elif str(authored_spawn_range).lower() == "long":
                    trigger_limit_mtile = max(
                        250, int(definition.range_mtile or 0)
                    )
            trigger_contact = bool(
                definition.mechanics.get("trigger_on_target")
                and target.kind in {"building", "tower"}
                and self._edge_distance(entity, target) <= trigger_limit_mtile
            )
            if trigger_contact and definition.mechanics.get("trigger_on_target"):
                # Contact-trigger carriers (Skeleton Barrel and Suspicious
                # Bush) are consumed at their authored trigger reach. Their
                # melee fields must not turn the transport into a normal
                # damaging attack before the payload drops.
                entity.hp = 0
                self._emit(
                    state,
                    "entity_triggered",
                    uid=entity.uid,
                    card_id=entity.card_id,
                    target_uid=target.uid,
                )
                continue
            if in_range or trigger_contact:
                if (
                    entity.charge_active
                    and definition.mechanics.get("trigger_on_building_contact")
                    and target.kind in {"building", "tower"}
                    and trigger_contact
                ):
                    entity.hp = 0
                    self._emit(
                        state,
                        "entity_triggered",
                        uid=entity.uid,
                        card_id=entity.card_id,
                        target_uid=target.uid,
                    )
                continue
            speed = int(definition.move_speed_mtile_per_s or 0)
            if entity.charge_active:
                speed = int(definition.mechanics.get("charged_speed_mtile_per_s") or speed)
            if charge_attack is not None and entity.attack_charge_active:
                speed = int(charge_attack.get("charged_speed_mtile_per_s") or speed)
            speed = speed * self._speed_multiplier(entity) // PERMILLE
            numerator = speed * dt + entity.movement_remainder
            travel, entity.movement_remainder = divmod(numerator, SECOND_US)
            waypoint_x, waypoint_y = self._movement_waypoint(state, entity, target)
            old_x, old_y = entity.x_mtile, entity.y_mtile
            entity.x_mtile, entity.y_mtile = move_towards(
                entity.x_mtile,
                entity.y_mtile,
                waypoint_x,
                waypoint_y,
                travel,
            )
            river_airborne = bool(
                definition.mechanics.get("river_jump")
                and self.ruleset.arena.river_y_min_mtile
                < entity.y_mtile
                < self.ruleset.arena.river_y_max_mtile
            )
            if river_airborne != entity.river_airborne_active:
                entity.river_airborne_active = river_airborne
                self._emit(
                    state,
                    "river_airborne_changed",
                    uid=entity.uid,
                    card_id=entity.card_id,
                    airborne=river_airborne,
                )
            if charge_attack is not None and not entity.attack_charge_active:
                moved = distance_mtile(old_x, old_y, entity.x_mtile, entity.y_mtile)
                if moved > 0:
                    entity.attack_charge_distance_mtile += moved
                    threshold = int(charge_attack.get("charge_distance_mtile") or 0)
                    if threshold > 0 and entity.attack_charge_distance_mtile >= threshold:
                        entity.attack_charge_active = True
                        self._emit(
                            state,
                            "charge_started",
                            uid=entity.uid,
                            card_id=entity.card_id,
                            distance_mtile=entity.attack_charge_distance_mtile,
                        )

    def _movement_waypoint(
        self,
        state: BattleState,
        entity: EntityState,
        target: EntityState,
    ) -> tuple[int, int]:
        start = (entity.x_mtile, entity.y_mtile)
        goal = (target.x_mtile, target.y_mtile)
        # Air troops use a separate navigation layer. They do not route through
        # bridges or around ground-only arena terrain/units; they fly directly
        # toward their target while still respecting attack-range stopping.
        if self._movement_layer(entity) == "air":
            entity.navigation_waypoints = [goal]
            entity.navigation_cursor = 0
            entity.navigation_target_uid = target.uid
            entity.navigation_revision = state.navigation_revision
            entity.navigation_goal_x_mtile = target.x_mtile
            entity.navigation_goal_y_mtile = target.y_mtile
            return goal
        radius = self._collision_radius(entity)
        obstacles = self._navigation_obstacles(state, target.uid)
        if self._definition(entity).mechanics.get("river_jump") and segment_is_walkable(
            self.ruleset.arena,
            start,
            goal,
            agent_radius_mtile=radius,
            obstacles=obstacles,
            allow_river_crossing=True,
        ):
            entity.navigation_waypoints = [goal]
            entity.navigation_cursor = 0
            entity.navigation_target_uid = target.uid
            entity.navigation_revision = state.navigation_revision
            entity.navigation_goal_x_mtile = target.x_mtile
            entity.navigation_goal_y_mtile = target.y_mtile
            return goal
        cache_valid = (
            entity.navigation_target_uid == target.uid
            and entity.navigation_revision == state.navigation_revision
            and entity.navigation_cursor < len(entity.navigation_waypoints)
            and distance_mtile(
                entity.navigation_goal_x_mtile,
                entity.navigation_goal_y_mtile,
                target.x_mtile,
                target.y_mtile,
            ) <= 500
        )
        if cache_valid:
            while (
                entity.navigation_cursor < len(entity.navigation_waypoints)
                and entity.navigation_waypoints[entity.navigation_cursor] == start
            ):
                entity.navigation_cursor += 1
            if entity.navigation_cursor < len(entity.navigation_waypoints):
                cached_waypoint = entity.navigation_waypoints[entity.navigation_cursor]
                if segment_is_walkable(
                    self.ruleset.arena,
                    start,
                    cached_waypoint,
                    agent_radius_mtile=radius,
                    obstacles=obstacles,
                ):
                    return cached_waypoint
            else:
                return start

        if segment_is_walkable(
            self.ruleset.arena,
            start,
            goal,
            agent_radius_mtile=radius,
            obstacles=obstacles,
        ):
            entity.navigation_waypoints = [goal]
            entity.navigation_cursor = 0
            entity.navigation_target_uid = target.uid
            entity.navigation_revision = state.navigation_revision
            entity.navigation_goal_x_mtile = target.x_mtile
            entity.navigation_goal_y_mtile = target.y_mtile
            return goal

        route = plan_route(
            self.ruleset.arena,
            start,
            goal,
            agent_radius_mtile=radius,
            obstacles=obstacles,
        )
        entity.navigation_waypoints = list(route[1:])
        entity.navigation_cursor = 0
        entity.navigation_target_uid = target.uid
        entity.navigation_revision = state.navigation_revision
        entity.navigation_goal_x_mtile = target.x_mtile
        entity.navigation_goal_y_mtile = target.y_mtile
        while (
            entity.navigation_cursor < len(entity.navigation_waypoints)
            and entity.navigation_waypoints[entity.navigation_cursor] == start
        ):
            entity.navigation_cursor += 1
        if entity.navigation_cursor >= len(entity.navigation_waypoints):
            return start
        return entity.navigation_waypoints[entity.navigation_cursor]

    def _navigation_obstacles(
        self,
        state: BattleState,
        target_uid: int | None,
    ) -> tuple[NavigationObstacle, ...]:
        return tuple(
            NavigationObstacle(
                uid=entity.uid,
                x_mtile=entity.x_mtile,
                y_mtile=entity.y_mtile,
                radius_mtile=self._collision_radius(entity),
            )
            for entity in self._alive_entities(state)
            if entity.kind in {"building", "tower"} and entity.uid != target_uid
        )

    def _position_clear_of_structures(
        self,
        state: BattleState,
        entity: EntityState,
        x: int,
        y: int,
        *,
        exclude_target: bool = True,
        excluded_structure_uid: int | None = None,
    ) -> bool:
        radius = self._collision_radius(entity)
        if self._movement_layer(entity) == "air":
            return 0 <= x < self.ruleset.arena.width_mtile and 0 <= y < self.ruleset.arena.height_mtile
        if not point_is_walkable(self.ruleset.arena, x, y, radius):
            return False
        for obstacle in self._navigation_obstacles(
            state,
            (
                excluded_structure_uid
                if excluded_structure_uid is not None
                else entity.target_uid if exclude_target else None
            ),
        ):
            if distance_mtile(x, y, obstacle.x_mtile, obstacle.y_mtile) < (
                radius + obstacle.radius_mtile
            ):
                return False
        return True

    def _separate_entities(self, state: BattleState) -> None:
        """Resolve troop/structure overlap with stable symmetric iterations."""

        movable = [
            entity
            for entity in self._alive_entities(state)
            if (
                entity.kind == "troop"
                and entity.carried_by_uid is None
                and entity.deploy_remaining_us <= 0
            )
        ]
        for _ in range(3):
            displacement = {entity.uid: [0, 0] for entity in movable}
            for index, left in enumerate(movable):
                for right in movable[index + 1 :]:
                    if self._movement_layer(left) != self._movement_layer(right):
                        continue
                    dx = right.x_mtile - left.x_mtile
                    dy = right.y_mtile - left.y_mtile
                    distance = distance_mtile(0, 0, dx, dy)
                    minimum = self._collision_radius(left) + self._collision_radius(right)
                    if distance >= minimum:
                        continue
                    overlap = minimum - distance
                    left_mass = self._mass(left)
                    right_mass = self._mass(right)
                    total_mass = left_mass + right_mass
                    # Displacement is inversely proportional to mass. Stable
                    # remainder assignment preserves the exact overlap while
                    # making a heavier tank push a lighter troop farther.
                    left_push = overlap * right_mass // total_mass
                    right_push = overlap - left_push
                    if distance == 0:
                        direction = -1 if (left.uid + right.uid) % 2 else 1
                        unit_x, unit_y, denominator = direction, 0, 1
                    else:
                        unit_x, unit_y, denominator = dx, dy, distance
                    displacement[left.uid][0] -= unit_x * left_push // denominator
                    displacement[left.uid][1] -= unit_y * left_push // denominator
                    displacement[right.uid][0] += unit_x * right_push // denominator
                    displacement[right.uid][1] += unit_y * right_push // denominator

            changed = False
            for entity in movable:
                dx, dy = displacement[entity.uid]
                if not (dx or dy):
                    continue
                candidate_x = min(
                    self.ruleset.arena.width_mtile - 1,
                    max(0, entity.x_mtile + dx),
                )
                candidate_y = min(
                    self.ruleset.arena.height_mtile - 1,
                    max(0, entity.y_mtile + dy),
                )
                if self._position_clear_of_structures(
                    state,
                    entity,
                    candidate_x,
                    candidate_y,
                    exclude_target=False,
                ):
                    entity.x_mtile = candidate_x
                    entity.y_mtile = candidate_y
                    changed = True
            if not changed:
                break
        # A building may be deployed underneath a moving troop. Visibility
        # planning cannot start from inside an inflated obstacle, so project
        # any remaining troop/structure overlap out before the next tick.
        structures = [
            entity
            for entity in self._alive_entities(state)
            if entity.kind in {"building", "tower"}
        ]
        for troop in movable:
            if self._movement_layer(troop) == "air":
                continue
            for _ in range(max(1, len(structures) * 2)):
                overlap = next(
                    (
                        structure
                        for structure in structures
                        if distance_mtile(
                            troop.x_mtile,
                            troop.y_mtile,
                            structure.x_mtile,
                            structure.y_mtile,
                        )
                        < self._collision_radius(troop) + self._collision_radius(structure)
                    ),
                    None,
                )
                if overlap is None:
                    break
                dx = troop.x_mtile - overlap.x_mtile
                dy = troop.y_mtile - overlap.y_mtile
                distance = distance_mtile(0, 0, dx, dy)
                if distance == 0:
                    dx = -1 if (troop.uid + overlap.uid) % 2 else 1
                    dy = 0
                    distance = 1
                minimum = self._collision_radius(troop) + self._collision_radius(overlap) + 1
                candidate_x = overlap.x_mtile + dx * minimum // distance
                candidate_y = overlap.y_mtile + dy * minimum // distance
                if not point_is_walkable(
                    self.ruleset.arena,
                    candidate_x,
                    candidate_y,
                    self._collision_radius(troop),
                ):
                    break
                troop.x_mtile = candidate_x
                troop.y_mtile = candidate_y
            remaining = [
                structure
                for structure in structures
                if distance_mtile(
                    troop.x_mtile,
                    troop.y_mtile,
                    structure.x_mtile,
                    structure.y_mtile,
                )
                < self._collision_radius(troop) + self._collision_radius(structure)
            ]
            if remaining:
                candidates: list[tuple[int, int]] = []
                troop_radius = self._collision_radius(troop)
                for structure in structures:
                    radius = troop_radius + self._collision_radius(structure) + 1
                    diagonal = (radius * 708 + 999) // 1_000
                    for dx, dy in (
                        (-radius, 0),
                        (-diagonal, -diagonal),
                        (0, -radius),
                        (diagonal, -diagonal),
                        (radius, 0),
                        (diagonal, diagonal),
                        (0, radius),
                        (-diagonal, diagonal),
                    ):
                        x = structure.x_mtile + dx
                        y = structure.y_mtile + dy
                        if self._position_clear_of_structures(
                            state,
                            troop,
                            x,
                            y,
                            exclude_target=False,
                        ):
                            candidates.append((x, y))
                if candidates:
                    goal = (
                        troop.navigation_goal_x_mtile,
                        troop.navigation_goal_y_mtile,
                    )
                    troop.x_mtile, troop.y_mtile = min(
                        candidates,
                        key=lambda point: (
                            distance_mtile(troop.x_mtile, troop.y_mtile, *point),
                            distance_mtile(*point, *goal),
                            point,
                        ),
                    )

    def _advance_attacks(self, state: BattleState) -> None:
        dt = self.ruleset.tick_us
        self._advance_attack_ramps(state, dt)
        # A wind-up attack which resolves at the start of this tick must not
        # immediately start a second attack in the same tick.  This matters
        # for Sparky: its cooldown is deliberately cleared after a shot so
        # the next four-second charge can begin on the following tick.
        resolved_this_tick: set[int] = set()
        # Complete attacks which were already winding up.
        for entity in self._alive_entities(state):
            if entity.windup_remaining_us <= 0 or self._is_frozen(entity):
                continue
            progress = self._attack_time_progress(entity, dt)
            entity.windup_remaining_us = max(0, entity.windup_remaining_us - progress)
            if entity.windup_remaining_us == 0:
                self._resolve_attack(state, entity)
                resolved_this_tick.add(entity.uid)
        # Cooldowns and new attack starts are stable by UID.
        for entity in self._alive_entities(state):
            # Preserve the legacy scheduler's same-tick cooldown accounting
            # for ordinary attacks.  Only a recharge-style wind-up (Sparky)
            # needs the guard; its cooldown is intentionally zero after the
            # shot and must not start a second charge in this same tick.
            if (
                entity.uid in resolved_this_tick
                and entity.kind != "tower"
                and self._definition(entity).mechanics.get("attack_windup_mode") == "recharge"
            ):
                continue
            if entity.deploy_remaining_us > 0 or entity.concealed_active or self._is_frozen(entity):
                continue
            if entity.attack_cooldown_us > 0:
                progress = self._attack_time_progress(entity, dt)
                entity.attack_cooldown_us = max(0, entity.attack_cooldown_us - progress)
            if entity.windup_remaining_us > 0 or entity.attack_cooldown_us > 0:
                continue
            if entity.target_uid is None:
                continue
            target = state.entities.get(entity.target_uid)
            if target is None or not target.alive or not self._in_attack_range(entity, target):
                continue
            definition = self._definition(entity)
            if (
                entity.kind != "tower"
                and
                definition.mechanics.get("trigger_on_target")
                and target.kind in {"building", "tower"}
            ):
                # Transport cards resolve through their contact trigger.  Do
                # not allow the generic attack scheduler to fire while they
                # are still inside their authored attack range.
                continue
            if (
                definition.attack_interval_us is None
                or definition.damage is None
                or definition.range_mtile is None
            ):
                continue
            interval = int(definition.attack_interval_us)
            delay = int(definition.first_hit_delay_us)
            if entity.stealth_active:
                self._break_stealth(state, entity)
            entity.attack_cooldown_us = interval
            entity.windup_remaining_us = delay
            entity.pending_target_uid = target.uid
            self._emit(
                state,
                "attack_started",
                uid=entity.uid,
                card_id=entity.card_id,
                target_uid=target.uid,
                attack_number=entity.attack_count + 1,
            )
            if delay == 0:
                self._resolve_attack(state, entity)
                resolved_this_tick.add(entity.uid)
        self._advance_secondary_attacks(state, dt)

    def _break_stealth(self, state: BattleState, entity: EntityState) -> None:
        """Reveal a stealth troop for its attack/re-cloak lifecycle."""

        if not entity.stealth_active:
            return
        definition = self.ruleset.cards.get(entity.card_id)
        if definition is None or not definition.mechanics.get("stealth"):
            return
        entity.stealth_active = False
        entity.stealth_remaining_us = int(
            definition.mechanics.get("stealth_recloak_us") or 1_500_000
        )
        self._emit(
            state,
            "stealth_broken",
            uid=entity.uid,
            card_id=entity.card_id,
            recloak_us=entity.stealth_remaining_us,
        )

    def _advance_secondary_attacks(self, state: BattleState, dt: int) -> None:
        """Advance independent weapon channels (currently Goblin Machine).

        A secondary weapon deliberately does not reuse ``target_uid`` or the
        primary attack cooldown: both weapons can be active simultaneously and
        the rocket has a blind inner range.  The state fields are serialized so
        stopping and resuming a replay cannot shift the rocket cadence.
        """

        for entity in self._alive_entities(state):
            if entity.deploy_remaining_us > 0 or entity.concealed_active or self._is_frozen(entity):
                continue
            definition = self._definition(entity)
            if entity.kind == "tower":
                continue
            raw = definition.mechanics.get("secondary_attack")
            if not raw:
                continue
            if entity.secondary_windup_remaining_us <= 0:
                continue
            progress = self._secondary_attack_time_progress(entity, dt)
            entity.secondary_windup_remaining_us = max(
                0, entity.secondary_windup_remaining_us - progress
            )
            if entity.secondary_windup_remaining_us == 0:
                self._resolve_secondary_attack(state, entity)

        for entity in self._alive_entities(state):
            if entity.deploy_remaining_us > 0 or entity.concealed_active or self._is_frozen(entity):
                continue
            definition = self._definition(entity)
            if entity.kind == "tower":
                continue
            raw = definition.mechanics.get("secondary_attack")
            if not raw:
                continue
            if entity.secondary_attack_cooldown_us > 0:
                progress = self._secondary_attack_time_progress(entity, dt)
                entity.secondary_attack_cooldown_us = max(
                    0, entity.secondary_attack_cooldown_us - progress
                )
            if (
                entity.secondary_windup_remaining_us > 0
                or entity.secondary_attack_cooldown_us > 0
            ):
                continue
            target_uid = self._choose_secondary_target(state, entity, raw)
            if target_uid is None:
                entity.secondary_pending_target_uid = None
                continue
            entity.secondary_attack_cooldown_us = int(raw["attack_interval_us"])
            entity.secondary_windup_remaining_us = int(raw.get("first_hit_delay_us") or 0)
            entity.secondary_pending_target_uid = target_uid
            self._emit(
                state,
                "secondary_attack_started",
                uid=entity.uid,
                card_id=entity.card_id,
                player=entity.owner,
                target_uid=target_uid,
                attack_number=entity.secondary_attack_count + 1,
            )
            if entity.secondary_windup_remaining_us == 0:
                self._resolve_secondary_attack(state, entity)

    def _secondary_attack_time_progress(self, entity: EntityState, dt: int) -> int:
        multiplier = self._hit_speed_multiplier(entity)
        numerator = dt * multiplier + entity.secondary_attack_time_remainder
        progress, entity.secondary_attack_time_remainder = divmod(numerator, PERMILLE)
        return progress

    def _choose_secondary_target(
        self,
        state: BattleState,
        source: EntityState,
        raw: object,
    ) -> int | None:
        if not hasattr(raw, "get"):
            return None
        primary_uid = source.target_uid
        candidates = []
        for target in self._alive_entities(state):
            if target.uid == primary_uid:
                continue
            if self._valid_secondary_target(state, source, target, raw):
                candidates.append((self._edge_distance(source, target), target.uid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def _valid_secondary_target(
        self,
        state: BattleState,
        source: EntityState,
        target: EntityState,
        raw: object,
    ) -> bool:
        if not hasattr(raw, "get"):
            return False
        if target.owner == source.owner or not target.alive or target.hp <= 0:
            return False
        if bool(raw.get("troops_only")) and target.kind != "troop":
            return False
        if not self._targetable_for_acquisition(state, target):
            return False
        allowed = tuple(str(value) for value in raw.get("targets", ()))
        if not self._spell_can_hit(source.card_id, target, allowed_targets=allowed):
            return False
        distance = self._edge_distance(source, target)
        return (
            int(raw.get("min_range_mtile") or 0)
            <= distance
            <= int(raw.get("max_range_mtile") or 0)
        )

    def _resolve_secondary_attack(self, state: BattleState, source: EntityState) -> None:
        target_uid = source.secondary_pending_target_uid
        source.secondary_pending_target_uid = None
        if target_uid is None or not source.alive:
            return
        target = state.entities.get(target_uid)
        definition = self._definition(source)
        raw = definition.mechanics.get("secondary_attack")
        if target is None or not target.alive or not raw:
            return
        if not self._valid_secondary_target(state, source, target, raw):
            # The target may have moved into the blind range or died while the
            # rocket was winding up.  A cancelled shot does not retarget at
            # impact; the next cadence will acquire a fresh target.
            return
        source.secondary_attack_count += 1
        raw_status = raw.get("status")
        projectile = ProjectileState(
            uid=self._allocate_uid(state),
            source_uid=source.uid,
            source_card_id=source.card_id,
            owner=source.owner,
            x_mtile=source.x_mtile,
            y_mtile=source.y_mtile,
            target_x_mtile=target.x_mtile,
            target_y_mtile=target.y_mtile,
            target_uid=target.uid,
            damage=self._scale_level_value(int(raw["damage"]), source.level_multiplier_permille),
            crown_damage=self._scale_level_value(
                int(raw["crown_tower_damage"]), source.level_multiplier_permille
            ),
            speed_mtile_per_s=int(raw["projectile_speed_mtile_per_s"]),
            speed_code=None,
            homing=False,
            radius_mtile=int(raw["area_radius_mtile"]),
            allowed_targets=tuple(str(value) for value in raw["targets"]),
            level_multiplier_permille=source.level_multiplier_permille,
            status_kind=None if not raw_status else str(raw_status.get("kind") or "slow"),
            status_duration_us=0 if not raw_status else int(raw_status.get("duration_us") or 0),
            status_magnitude_permille=(
                PERMILLE if not raw_status else int(raw_status.get("speed_multiplier_milli") or PERMILLE)
            ),
            status_hit_speed_magnitude_permille=(
                PERMILLE if not raw_status else int(raw_status.get("hit_speed_multiplier_milli") or PERMILLE)
            ),
        )
        state.projectiles[projectile.uid] = projectile
        self._emit(
            state,
            "projectile_spawned",
            uid=projectile.uid,
            player=source.owner,
            card_id=source.card_id,
            source_uid=source.uid,
            target_uid=target.uid,
            attack_kind="secondary",
            projectile_speed_code=projectile.speed_code,
        )

    def _attack_time_progress(self, entity: EntityState, dt: int) -> int:
        multiplier = self._hit_speed_multiplier(entity)
        numerator = dt * multiplier + entity.attack_time_remainder
        progress, entity.attack_time_remainder = divmod(numerator, PERMILLE)
        return progress

    def _resolve_attack(self, state: BattleState, source: EntityState) -> None:
        target_uid = source.pending_target_uid
        source.pending_target_uid = None
        if target_uid is None or not source.alive or source.concealed_active:
            return
        target = state.entities.get(target_uid)
        if target is None or not target.alive:
            return
        definition = self._definition(source)
        mechanics = {} if source.kind == "tower" else definition.mechanics
        # Keep the lifecycle correct even when a deterministic fixture calls
        # the impact resolver directly instead of going through the normal
        # attack scheduler.  The scheduler also calls this helper before the
        # attack starts; the idempotent guard avoids duplicate events.
        if source.stealth_active:
            self._break_stealth(state, source)
        source.attack_count += 1
        projectile_definition = definition.projectile
        bayonet = mechanics.get("bayonet")
        bayonet_active = bool(
            bayonet
            and self._edge_distance(source, target) <= int(bayonet.get("range_mtile") or 0)
            and self._spell_can_hit(
                source.card_id,
                target,
                allowed_targets=tuple(str(value) for value in bayonet.get("targets", ())),
            )
        )
        status = None if source.kind == "tower" else definition.mechanics.get("status")
        snare = None if source.kind == "tower" else definition.mechanics.get("snare")
        if status is None and snare is not None:
            status = {
                "kind": "slow",
                "duration_us": int(snare.get("duration_us") or 0),
                "speed_multiplier_milli": int(snare.get("speed_multiplier_milli") or 1_000),
                "hit_speed_multiplier_milli": int(snare.get("hit_speed_multiplier_milli") or 1_000),
            }
        if status is not None and source.kind != "tower":
            status = {
                **dict(status),
                "source_level_multiplier_permille": source.level_multiplier_permille,
            }
        charge_attack = (
            None
            if source.kind == "tower"
            else definition.mechanics.get("charge_attack")
        )
        dash = None if source.kind == "tower" else definition.mechanics.get("dash")
        ramp_attack = None if source.kind == "tower" else definition.mechanics.get("ramp_attack")
        attack_damage = int(definition.damage or 0)
        if bayonet_active:
            attack_damage = int(bayonet.get("damage") or attack_damage)
        if charge_attack is not None and source.attack_charge_active:
            attack_damage = int(charge_attack.get("charge_damage") or attack_damage)
        if dash is not None and source.dash_attack_active:
            attack_damage = int(dash.get("dash_damage") or attack_damage)
        if ramp_attack is not None:
            schedule = ramp_attack.get("damage_schedule", ())
            if schedule:
                stage = min(source.ramp_stage, len(schedule) - 1)
                attack_damage = int(schedule[stage])
        definition_crown_damage = getattr(definition, "crown_tower_damage", None)
        crown_damage = int(
            definition_crown_damage
            if definition_crown_damage is not None
            else attack_damage
        )
        if bayonet_active:
            crown_damage = int(bayonet.get("crown_tower_damage") or attack_damage)
            projectile_definition = None
            self._emit(
                state,
                "bayonet_attack",
                uid=source.uid,
                card_id=source.card_id,
                target_uid=target.uid,
            )
        if source.kind != "tower":
            attack_damage = self._scale_level_value(
                attack_damage, source.level_multiplier_permille
            )
            crown_damage = self._scale_level_value(
                crown_damage, source.level_multiplier_permille
            )
        if projectile_definition is None:
            hook = None if source.kind == "tower" else definition.mechanics.get("hook")
            if hook is not None:
                self._apply_hook(state, source, target, hook)
            multi_target = definition.mechanics.get("multi_target_attack")
            if multi_target is not None:
                self._impact_multi_target(
                    state,
                    source=source,
                    primary_target_uid=target.uid,
                    raw_component=multi_target,
                    status=status,
                    reset_attack=bool(definition.mechanics.get("reset_attack")),
                )
            else:
                self._impact_area(
                    state,
                    owner=source.owner,
                    source_uid=source.uid,
                    source_card_id=source.card_id,
                    x=target.x_mtile,
                    y=target.y_mtile,
                    damage=attack_damage,
                    crown_damage=crown_damage,
                    radius=int(getattr(definition, "area_radius_mtile", 0) or 0),
                    status=status,
                    knockback=0,
                    primary_target_uid=target.uid,
                    allowed_targets=(
                        tuple(str(value) for value in mechanics.get("impact_targets", ()))
                        if mechanics.get("impact_targets") is not None
                        else None
                    ),
                    knockback_direction=(
                        None
                        if mechanics.get("knockback_direction") != "projectile_travel"
                        else (target.x_mtile - source.x_mtile, target.y_mtile - source.y_mtile)
                    ),
                )
            if source.card_id == "battle-healer":
                self._apply_battle_healer_heal(state, source)
        else:
            projectile_x, projectile_y = move_towards(
                source.x_mtile,
                source.y_mtile,
                target.x_mtile,
                target.y_mtile,
                min(
                    projectile_definition.start_radius_mtile,
                    distance_mtile(
                        source.x_mtile,
                        source.y_mtile,
                        target.x_mtile,
                        target.y_mtile,
                    ),
                ),
            )
            # Hunter launches a fan of primary pellets.  Firecracker's five
            # pellets are a post-impact shrapnel stream, so its attack still
            # creates one homing primary projectile here; the shrapnels are
            # materialized by ``_impact_projectile`` at the burst point.
            raw_pellets = mechanics.get("pellets")
            primary_pellets = raw_pellets if source.card_id != "firecracker" else None
            projectile_count = (
                int(primary_pellets.get("count", 1))
                if hasattr(primary_pellets, "get")
                else 1
            )
            pellet_spread = (
                int(primary_pellets.get("spread_mtile", 0))
                if hasattr(primary_pellets, "get")
                else 0
            )
            base_dx = target.x_mtile - source.x_mtile
            base_dy = target.y_mtile - source.y_mtile
            base_distance = max(1, distance_mtile(source.x_mtile, source.y_mtile, target.x_mtile, target.y_mtile))
            perp_x, perp_y = -base_dy, base_dx
            if projectile_count > 1:
                pellet_offsets = tuple(
                    (pellet_spread * (2 * index - (projectile_count - 1)) // (projectile_count - 1))
                    for index in range(projectile_count)
                )
            else:
                pellet_offsets = (0,)
            for pellet_index, offset in enumerate(pellet_offsets):
                end_x = target.x_mtile + perp_x * offset // base_distance
                end_y = target.y_mtile + perp_y * offset // base_distance
                # Magic Archer travels through the acquired target and keeps
                # a fixed line.  The line endpoint is deliberately capped by
                # the authored component rather than by target distance.
                line_component = mechanics.get("line_piercing")
                if hasattr(line_component, "get"):
                    line_length = int(line_component.get("length_mtile") or base_distance)
                    end_x = source.x_mtile + base_dx * line_length // base_distance
                    end_y = source.y_mtile + base_dy * line_length // base_distance
                end_x = min(self.ruleset.arena.width_mtile - 1, max(0, end_x))
                end_y = min(self.ruleset.arena.height_mtile - 1, max(0, end_y))
                projectile = ProjectileState(
                uid=self._allocate_uid(state),
                source_uid=source.uid,
                source_card_id=source.card_id,
                owner=source.owner,
                x_mtile=projectile_x,
                y_mtile=projectile_y,
                target_x_mtile=end_x,
                target_y_mtile=end_y,
                target_uid=target.uid,
                damage=attack_damage,
                crown_damage=crown_damage,
                speed_mtile_per_s=projectile_definition.speed_mtile_per_s,
                speed_code=(
                    int(mechanics["projectile_speed_code"])
                    if mechanics.get("projectile_speed_code") is not None
                    else None
                ),
                    # A Hunter shot is a fan of independent pellets.  The
                    # card's generic projectile definition is homing because
                    # it is also used by single-target ranged troops, but a
                    # pellet must retain its authored spread after launch.
                    # Otherwise the per-pellet endpoint is overwritten on the
                    # next tick and every pellet collapses onto the acquired
                    # target.
                    homing=projectile_definition.homing and projectile_count == 1,
                radius_mtile=int(getattr(definition, "area_radius_mtile", 0) or projectile_definition.radius_mtile),
                status_kind=None if not status else str(status.get("kind")),
                status_duration_us=0 if not status else int(status.get("duration_us") or 0),
                status_magnitude_permille=PERMILLE if not status else int(status.get("speed_multiplier_milli") or 0),
                status_hit_speed_magnitude_permille=(
                    PERMILLE if not status else int(status.get("hit_speed_multiplier_milli") or PERMILLE)
                ),
                status_damage_per_tick=0 if not status else int(status.get("damage_per_tick") or 0),
                status_tick_interval_us=0 if not status else int(status.get("tick_interval_us") or 0),
                knockback_mtile=int(mechanics.get("knockback_mtile") or 0),
                piercing=bool(
                    mechanics.get("piercing")
                    or mechanics.get("returning_projectile") is not None
                ),
                origin_x_mtile=source.x_mtile,
                origin_y_mtile=source.y_mtile,
                line_end_x_mtile=end_x,
                line_end_y_mtile=end_y,
                direction_x_mtile=base_dx,
                direction_y_mtile=base_dy,
                returning=bool(mechanics.get("returning_projectile")),
                pellet_index=pellet_index,
                level_multiplier_permille=source.level_multiplier_permille,
            )
                state.projectiles[projectile.uid] = projectile
                self._emit(
                    state,
                    "projectile_spawned",
                    uid=projectile.uid,
                    player=source.owner,
                    card_id=source.card_id,
                    source_uid=source.uid,
                    target_uid=target.uid,
                    pellet_index=pellet_index,
                    projectile_speed_code=projectile.speed_code,
                )
        if charge_attack is not None and bool(charge_attack.get("reset_on_hit")):
            self._reset_attack_charge(state, source, reason="hit_consumed")
        if dash is not None and bool(dash.get("reset_on_hit")):
            self._reset_dash(state, source, reason="hit_consumed")
        if source.kind != "tower" and bool(definition.mechanics.get("suicide_on_attack")):
            source.hp = 0
        # Sparky's four-second interval is the complete charge/recharge cycle,
        # not an additional cooldown after its four-second wind-up.  Clearing
        # the cooldown here lets the next tick start the next charge while the
        # resolved-this-tick guard above prevents a zero-time double shot.
        if (
            source.kind != "tower"
            and definition.mechanics.get("attack_windup_mode") == "recharge"
            and source.alive
        ):
            source.attack_cooldown_us = 0

    def _advance_attack_ramps(self, state: BattleState, dt: int) -> None:
        """Advance target-locked ramp attacks in deterministic UID order.

        Inferno Dragon and Inferno Tower keep their acquired target while the
        beam ramps.  Losing the target, leaving attack range, deployment, or
        hard crowd control clears the timer; otherwise the stage is selected
        from the integer threshold schedule before the attack scheduler runs.
        """

        for entity in self._alive_entities(state):
            ramp = self._ramp_component(entity)
            if ramp is None:
                continue
            if entity.deploy_remaining_us > 0 or self._is_frozen(entity):
                self._reset_attack_ramp(state, entity, reason="not_active")
                continue
            target = state.entities.get(entity.target_uid) if entity.target_uid is not None else None
            if (
                target is None
                or not self._valid_target(state, entity, target.uid)
                or not self._in_attack_range(entity, target)
            ):
                self._reset_attack_ramp(state, entity, reason="target_lost")
                continue
            thresholds = tuple(int(value) for value in ramp.get("stage_thresholds_us", ()))
            if not thresholds:
                self._reset_attack_ramp(state, entity, reason="invalid_schedule")
                continue
            old_stage = entity.ramp_stage
            entity.ramp_elapsed_us += dt
            stage = 0
            for index, threshold in enumerate(thresholds):
                if entity.ramp_elapsed_us >= threshold:
                    stage = index
                else:
                    break
            entity.ramp_stage = min(stage, len(thresholds) - 1)
            if entity.ramp_stage != old_stage:
                self._emit(
                    state,
                    "ramp_stage_changed",
                    uid=entity.uid,
                    card_id=entity.card_id,
                    stage=entity.ramp_stage,
                    elapsed_us=entity.ramp_elapsed_us,
                )

    def _apply_hook(
        self,
        state: BattleState,
        source: EntityState,
        target: EntityState,
        raw_hook: object,
    ) -> None:
        """Reel a hooked troop toward Fisherman before his melee impact."""

        if not hasattr(raw_hook, "get"):
            return
        if bool(raw_hook.get("pull_troops_only")) and target.kind != "troop":
            self._emit(
                state,
                "hook_noop",
                uid=source.uid,
                card_id=source.card_id,
                target_uid=target.uid,
                reason="target_class_not_pullable",
            )
            return
        pull_distance = int(raw_hook.get("pull_distance_mtile") or 0)
        center_distance = distance_mtile(
            source.x_mtile,
            source.y_mtile,
            target.x_mtile,
            target.y_mtile,
        )
        desired_gap = pull_distance + self._collision_radius(source) + self._collision_radius(target)
        travel = max(0, center_distance - desired_gap)
        jump_was_active = target.jump_remaining_us > 0
        if travel <= 0:
            if jump_was_active:
                # Fisherman's hook is the documented exception that can
                # cancel a Mega Knight already in flight.  A hook that lands
                # at the existing reel gap still cancels the pending landing
                # pulse; otherwise the jump would explode at its stale
                # pre-hook coordinates on the next tick.
                target.jump_remaining_us = 0
                target.jump_target_uid = None
                target.jump_landing_x_mtile = 0
                target.jump_landing_y_mtile = 0
                self._emit(
                    state,
                    "jump_cancelled",
                    uid=target.uid,
                    card_id=target.card_id,
                    reason="hooked",
                    source_uid=source.uid,
                )
            return
        old_x, old_y = target.x_mtile, target.y_mtile
        target.x_mtile, target.y_mtile = move_towards(
            target.x_mtile,
            target.y_mtile,
            source.x_mtile,
            source.y_mtile,
            travel,
        )
        target.navigation_waypoints.clear()
        target.navigation_cursor = 0
        target.navigation_revision = -1
        self._reset_attack_charge(state, target, reason="hooked")
        self._reset_dash(state, target, reason="hooked")
        if jump_was_active:
            # The hook interrupts the airborne phase rather than allowing the
            # old landing target/coordinates to survive the reel.  Clearing
            # all jump state also suppresses the landing splash, matching the
            # Fisherman-versus-Mega-Knight interaction.
            target.jump_remaining_us = 0
            target.jump_target_uid = None
            target.jump_landing_x_mtile = 0
            target.jump_landing_y_mtile = 0
            self._emit(
                state,
                "jump_cancelled",
                uid=target.uid,
                card_id=target.card_id,
                reason="hooked",
                source_uid=source.uid,
            )
        self._emit(
            state,
            "hook_pulled",
            uid=source.uid,
            card_id=source.card_id,
            target_uid=target.uid,
            from_x_mtile=old_x,
            from_y_mtile=old_y,
            to_x_mtile=target.x_mtile,
            to_y_mtile=target.y_mtile,
        )

    def _apply_battle_healer_heal(self, state: BattleState, source: EntityState) -> None:
        """Heal nearby friendly troops after a Battle Healer attack.

        The August 2026 rework explicitly excludes the healer herself and
        other Battle Healers.  Buildings and Crown Towers are not troop
        recipients.  Applying this at the melee impact keeps the event order
        deterministic and makes the heal observable in replay traces.
        """

        definition = self.ruleset.cards[source.card_id]
        amount = self._scale_level_value(
            int(definition.mechanics.get("heal_amount") or 0),
            source.level_multiplier_permille,
        )
        radius = int(definition.mechanics.get("heal_radius_mtile") or 0)
        if amount <= 0 or radius <= 0:
            return
        for target in self._alive_entities(state):
            if (
                target.owner != source.owner
                or target.kind != "troop"
                or target.card_id == "battle-healer"
                or target.uid == source.uid
                or distance_mtile(source.x_mtile, source.y_mtile, target.x_mtile, target.y_mtile)
                > radius + self._collision_radius(target)
            ):
                continue
            before = target.hp
            target.hp = min(target.max_hp, target.hp + amount)
            healed = target.hp - before
            if healed:
                self._emit(
                    state,
                    "healing_applied",
                    source_uid=source.uid,
                    source_card_id=source.card_id,
                    target_uid=target.uid,
                    amount=healed,
                    hp_after=target.hp,
                )

    def _apply_impact_heal(
        self,
        state: BattleState,
        *,
        owner: int,
        source_uid: int | None,
        source_card_id: str,
        x: int,
        y: int,
        raw_component: object,
        level_multiplier_permille: int = PERMILLE,
    ) -> None:
        """Apply a one-shot friendly troop heal at a projectile impact.

        Heal Spirit is the first consumer.  Its body is a suicide troop, so
        the source UID can already be dead when the jump resolves.  Recipient
        eligibility is therefore derived from owner and the target card's
        movement layer, never from source-body liveness.  Buildings and Crown
        Towers are intentionally excluded even when they are inside the
        impact radius.
        """

        if not hasattr(raw_component, "get"):
            return
        amount = self._scale_level_value(
            int(raw_component.get("amount") or 0), level_multiplier_permille
        )
        radius = int(raw_component.get("radius_mtile") or 0)
        allowed_layers = {
            str(value) for value in raw_component.get("targets", ("air", "ground"))
        }
        if amount <= 0 or radius < 0 or not allowed_layers:
            return
        exclude_source = bool(raw_component.get("exclude_source", True))
        healed_targets = 0
        for target in self._alive_entities(state):
            if target.owner != owner or target.kind != "troop":
                continue
            if exclude_source and source_uid is not None and target.uid == source_uid:
                continue
            if self._movement_layer(target) not in allowed_layers:
                continue
            if (
                distance_mtile(x, y, target.x_mtile, target.y_mtile)
                > radius + self._collision_radius(target)
            ):
                continue
            before = target.hp
            target.hp = min(target.max_hp, target.hp + amount)
            healed = target.hp - before
            if not healed:
                continue
            healed_targets += 1
            self._emit(
                state,
                "healing_applied",
                source_uid=source_uid,
                source_card_id=source_card_id,
                target_uid=target.uid,
                amount=healed,
                hp_after=target.hp,
            )
        self._emit(
            state,
            "healing_impact_resolved",
            source_uid=source_uid,
            source_card_id=source_card_id,
            owner=owner,
            radius_mtile=radius,
            recipient_count=healed_targets,
        )

    def _advance_projectiles(self, state: BattleState) -> None:
        dt = self.ruleset.tick_us
        for projectile in [state.projectiles[uid] for uid in sorted(state.projectiles)]:
            if not projectile.alive:
                continue
            if projectile.chain_next_index < len(projectile.chain_target_uids):
                projectile.chain_delay_remaining_us = max(
                    0, projectile.chain_delay_remaining_us - dt
                )
                if projectile.chain_delay_remaining_us == 0:
                    target = state.entities.get(
                        projectile.chain_target_uids[projectile.chain_next_index]
                    )
                    if target is not None and target.alive:
                        self._apply_chain_hit(
                            state,
                            projectile,
                            target,
                            projectile.chain_next_index + 1,
                        )
                    projectile.chain_next_index += 1
                    projectile.chain_delay_remaining_us = projectile.chain_delay_us
                if projectile.chain_next_index >= len(projectile.chain_target_uids):
                    projectile.alive = False
                    self._emit(
                        state,
                        "projectile_resolved",
                        uid=projectile.uid,
                        card_id=projectile.source_card_id,
                    )
                continue
            if projectile.target_uid is not None:
                target = state.entities.get(projectile.target_uid)
                if target is not None and target.alive and projectile.homing:
                    projectile.target_x_mtile = target.x_mtile
                    projectile.target_y_mtile = target.y_mtile
            if projectile.return_phase and projectile.source_uid is not None:
                source = state.entities.get(projectile.source_uid)
                if source is not None and source.alive:
                    projectile.target_x_mtile = source.x_mtile
                    projectile.target_y_mtile = source.y_mtile
            old_x, old_y = projectile.x_mtile, projectile.y_mtile
            numerator = projectile.speed_mtile_per_s * dt + projectile.movement_remainder
            travel, projectile.movement_remainder = divmod(numerator, SECOND_US)
            remaining = distance_mtile(
                old_x,
                old_y,
                projectile.target_x_mtile,
                projectile.target_y_mtile,
            )
            projectile.x_mtile, projectile.y_mtile = move_towards(
                old_x,
                old_y,
                projectile.target_x_mtile,
                projectile.target_y_mtile,
                travel,
            )
            if projectile.piercing:
                self._impact_piercing_projectile(state, projectile)
            if remaining <= travel or projectile.speed_mtile_per_s <= 0:
                if not projectile.piercing:
                    self._impact_projectile(state, projectile)
                    if projectile.chain_next_index < len(projectile.chain_target_uids):
                        continue
                if projectile.returning and not projectile.return_phase:
                    source = state.entities.get(projectile.source_uid) if projectile.source_uid is not None else None
                    if source is not None and source.alive:
                        raw_return = self.ruleset.cards[projectile.source_card_id].mechanics.get("returning_projectile", {})
                        # The return pass starts at the outbound endpoint and
                        # is allowed to hit the same bodies again.  Resetting
                        # the swept-path bookkeeping is therefore part of the
                        # authoritative projectile transition, not a render
                        # detail.
                        projectile.origin_x_mtile = projectile.x_mtile
                        projectile.origin_y_mtile = projectile.y_mtile
                        projectile.hit_uids.clear()
                        projectile.return_phase = True
                        projectile.target_uid = source.uid
                        projectile.target_x_mtile = source.x_mtile
                        projectile.target_y_mtile = source.y_mtile
                        projectile.speed_mtile_per_s = int(raw_return.get("return_speed_mtile_per_s") or projectile.speed_mtile_per_s)
                        projectile.movement_remainder = 0
                        self._emit(
                            state,
                            "projectile_return_started",
                            uid=projectile.uid,
                            card_id=projectile.source_card_id,
                            source_uid=projectile.source_uid,
                        )
                        continue
                if projectile.piercing:
                    definition = self.ruleset.cards.get(projectile.source_card_id)
                    spawn = (
                        None
                        if definition is None
                        else definition.mechanics.get("spawn_on_impact")
                    )
                    if spawn:
                        child = self.ruleset.card(str(spawn["card_id"]))
                        for _ in range(int(spawn["count"])):
                            self._spawn_single_at(
                                state,
                                child,
                                owner=projectile.owner,
                                x_mtile=projectile.target_x_mtile,
                                y_mtile=projectile.target_y_mtile,
                                parent_uid=projectile.source_uid,
                                level_multiplier_permille=projectile.level_multiplier_permille,
                            )
                projectile.alive = False
                self._emit(
                    state,
                    "projectile_resolved",
                    uid=projectile.uid,
                    card_id=projectile.source_card_id,
                )

    def _impact_projectile(self, state: BattleState, projectile: ProjectileState) -> None:
        # Component-boundary fixtures sometimes call the terminal impact
        # helper directly instead of advancing a projectile through the
        # physics loop.  Preserve the same swept-path semantics for piercing
        # projectiles by resolving them at their authored endpoint.
        if projectile.piercing:
            if (
                projectile.x_mtile == projectile.origin_x_mtile
                and projectile.y_mtile == projectile.origin_y_mtile
            ):
                projectile.x_mtile = projectile.target_x_mtile
                projectile.y_mtile = projectile.target_y_mtile
            self._impact_piercing_projectile(state, projectile)
            return
        status = None
        if projectile.status_kind:
            status = {
                "kind": projectile.status_kind,
                "duration_us": projectile.status_duration_us,
                "speed_multiplier_milli": projectile.status_magnitude_permille,
                "hit_speed_multiplier_milli": projectile.status_hit_speed_magnitude_permille,
                "damage_per_tick": projectile.status_damage_per_tick,
                "tick_interval_us": projectile.status_tick_interval_us,
                "on_death_spawn_card_id": (
                    self.ruleset.cards[projectile.source_card_id].mechanics.get("status", {}).get("on_death_spawn_card_id")
                    if projectile.source_card_id in self.ruleset.cards
                    and hasattr(self.ruleset.cards[projectile.source_card_id].mechanics.get("status"), "get")
                    else None
                ),
                "on_death_spawn_count": (
                    int(self.ruleset.cards[projectile.source_card_id].mechanics.get("status", {}).get("on_death_spawn_count") or 0)
                    if projectile.source_card_id in self.ruleset.cards
                    and hasattr(self.ruleset.cards[projectile.source_card_id].mechanics.get("status"), "get")
                    else 0
                ),
                "on_death_spawn_owner": (
                    projectile.owner
                    if projectile.source_card_id in self.ruleset.cards
                    and hasattr(self.ruleset.cards[projectile.source_card_id].mechanics.get("status"), "get")
                    and self.ruleset.cards[projectile.source_card_id].mechanics.get("status", {}).get("on_death_spawn_card_id") is not None
                    else None
                ),
                "source_level_multiplier_permille": projectile.level_multiplier_permille,
            }
        definition = self.ruleset.cards.get(projectile.source_card_id)
        clone = None if definition is None else definition.mechanics.get("clone")
        if clone is not None:
            self._impact_clone(
                state,
                owner=projectile.owner,
                source_uid=projectile.source_uid,
                source_card_id=projectile.source_card_id,
                x=projectile.target_x_mtile,
                y=projectile.target_y_mtile,
                radius=projectile.radius_mtile,
                raw_clone=clone,
                level_multiplier_permille=projectile.level_multiplier_permille,
            )
            return
        chain_attack = None if definition is None else definition.mechanics.get("chain_attack")
        if chain_attack is not None:
            self._impact_chain_projectile(
                state,
                projectile=projectile,
                raw_component=chain_attack,
                status=status,
                reset_attack=bool(definition.mechanics.get("reset_attack")),
            )
            return
        persistent = None if definition is None else definition.mechanics.get("persistent_effect")
        if persistent:
            self._create_area_effect(
                state,
                owner=projectile.owner,
                source_uid=projectile.source_uid,
                source_card_id=projectile.source_card_id,
                x_mtile=projectile.target_x_mtile,
                y_mtile=projectile.target_y_mtile,
                default_radius=projectile.radius_mtile,
                default_damage=projectile.damage,
                default_crown_damage=projectile.crown_damage,
                default_status=status,
                default_knockback=projectile.knockback_mtile,
                raw_effect=persistent,
                level_multiplier_permille=projectile.level_multiplier_permille,
            )
        else:
            self._impact_area(
                state,
                owner=projectile.owner,
                source_uid=projectile.source_uid,
                source_card_id=projectile.source_card_id,
                x=projectile.target_x_mtile,
                y=projectile.target_y_mtile,
                damage=projectile.damage,
                crown_damage=projectile.crown_damage,
                radius=projectile.radius_mtile,
                status=status,
                knockback=projectile.knockback_mtile,
                primary_target_uid=projectile.target_uid,
                allowed_targets=projectile.allowed_targets or None,
                knockback_direction=(
                    (projectile.direction_x_mtile, projectile.direction_y_mtile)
                    if definition is not None
                    and definition.mechanics.get("knockback_direction") == "projectile_travel"
                    else None
                ),
                target_limit=(
                    None
                    if definition is None
                    else (
                        int(definition.mechanics["target_limit"])
                        if definition.mechanics.get("target_limit") is not None
                        else None
                    )
                ),
                target_selection=(
                    None
                    if definition is None
                    else definition.mechanics.get("target_selection")
                ),
                reset_attack=bool(
                    definition is not None and definition.mechanics.get("reset_attack")
                ),
            )
            if definition is not None:
                spawn = definition.mechanics.get("spawn_on_impact")
                if spawn:
                    child = self.ruleset.card(str(spawn["card_id"]))
                    for _ in range(int(spawn["count"])):
                        self._spawn_single_at(
                            state,
                            child,
                            owner=projectile.owner,
                            x_mtile=projectile.target_x_mtile,
                            y_mtile=projectile.target_y_mtile,
                            parent_uid=projectile.source_uid,
                            level_multiplier_permille=projectile.level_multiplier_permille,
                        )
        if (
            definition is not None
            and projectile.source_card_id == "firecracker"
            and projectile.target_uid is not None
        ):
            self._spawn_firecracker_shrapnels(state, projectile, definition)
        if definition is not None:
            heal_on_impact = definition.mechanics.get("heal_on_impact")
            if heal_on_impact is not None:
                self._apply_impact_heal(
                    state,
                    owner=projectile.owner,
                    source_uid=projectile.source_uid,
                    source_card_id=projectile.source_card_id,
                    x=projectile.target_x_mtile,
                    y=projectile.target_y_mtile,
                    raw_component=heal_on_impact,
                    level_multiplier_permille=projectile.level_multiplier_permille,
                )
        if definition is not None and projectile.source_uid is not None:
            recoil = int(definition.mechanics.get("recoil_mtile") or 0)
            source = state.entities.get(projectile.source_uid)
            # Shrapnel projectiles retain the source UID for attribution but
            # have no acquired target.  Only the primary burst may recoil the
            # Firecracker; applying recoil once per shrapnel would move the
            # source five extra times.
            if recoil > 0 and projectile.target_uid is not None and source is not None and source.alive:
                before = (source.x_mtile, source.y_mtile)
                self._apply_knockback(
                    state,
                    source,
                    projectile.target_x_mtile,
                    projectile.target_y_mtile,
                    recoil,
                )
                self._emit(
                    state,
                    "recoil_applied",
                    source_uid=source.uid,
                    source_card_id=source.card_id,
                    from_x_mtile=before[0],
                    from_y_mtile=before[1],
                    to_x_mtile=source.x_mtile,
                    to_y_mtile=source.y_mtile,
                    distance_mtile=recoil,
                )

    def _spawn_firecracker_shrapnels(
        self,
        state: BattleState,
        projectile: ProjectileState,
        definition: CardDefinition | TowerDefinition,
    ) -> None:
        """Launch Firecracker's five non-homing swept shrapnels.

        The primary firework resolves its normal splash at the acquired
        target.  Its fragments then travel away from the attacker in a small
        fan, each retaining a swept collision path so bodies behind the
        primary target can be hit once.  The acquired target is seeded into
        every fragment's hit set: it already received the primary burst and
        must not be double-counted by the fragment paths.
        """

        mechanics = definition.mechanics
        raw_pellets = mechanics.get("pellets")
        line = mechanics.get("line_piercing")
        if not hasattr(raw_pellets, "get") or not hasattr(line, "get"):
            return
        count = int(raw_pellets.get("count") or 0)
        spread = int(raw_pellets.get("spread_mtile") or 0)
        length = int(line.get("length_mtile") or 0)
        if count <= 0 or length <= 0:
            return

        origin_x = projectile.target_x_mtile
        origin_y = projectile.target_y_mtile
        base_dx = origin_x - projectile.origin_x_mtile
        base_dy = origin_y - projectile.origin_y_mtile
        base_distance = distance_mtile(0, 0, base_dx, base_dy)
        if base_distance <= 0:
            base_dx = 0
            base_dy = -1 if projectile.owner == 0 else 1
            base_distance = 1
        perp_x, perp_y = -base_dy, base_dx
        primary_uid = projectile.target_uid
        for index in range(count):
            offset = (
                spread * (2 * index - (count - 1)) // (count - 1)
                if count > 1
                else 0
            )
            endpoint_x = origin_x + base_dx * length // base_distance + perp_x * offset // base_distance
            endpoint_y = origin_y + base_dy * length // base_distance + perp_y * offset // base_distance
            endpoint_x = min(self.ruleset.arena.width_mtile - 1, max(0, endpoint_x))
            endpoint_y = min(self.ruleset.arena.height_mtile - 1, max(0, endpoint_y))
            shrapnel = ProjectileState(
                uid=self._allocate_uid(state),
                source_uid=projectile.source_uid,
                source_card_id=projectile.source_card_id,
                owner=projectile.owner,
                x_mtile=origin_x,
                y_mtile=origin_y,
                target_x_mtile=endpoint_x,
                target_y_mtile=endpoint_y,
                target_uid=None,
                damage=projectile.damage,
                crown_damage=projectile.crown_damage,
                speed_mtile_per_s=projectile.speed_mtile_per_s,
                speed_code=projectile.speed_code,
                homing=False,
                radius_mtile=0,
                allowed_targets=projectile.allowed_targets,
                hit_uids=[] if primary_uid is None else [primary_uid],
                piercing=True,
                origin_x_mtile=origin_x,
                origin_y_mtile=origin_y,
                line_end_x_mtile=endpoint_x,
                line_end_y_mtile=endpoint_y,
                direction_x_mtile=base_dx,
                direction_y_mtile=base_dy,
                pellet_index=index + 1,
                level_multiplier_permille=projectile.level_multiplier_permille,
            )
            state.projectiles[shrapnel.uid] = shrapnel
            self._emit(
                state,
                "projectile_spawned",
                uid=shrapnel.uid,
                player=projectile.owner,
                card_id=projectile.source_card_id,
                source_uid=projectile.source_uid,
                target_uid=None,
                attack_kind="shrapnel",
                pellet_index=shrapnel.pellet_index,
                projectile_speed_code=shrapnel.speed_code,
            )

    def _impact_multi_target(
        self,
        state: BattleState,
        *,
        source: EntityState,
        primary_target_uid: int,
        raw_component: object,
        status: object,
        reset_attack: bool,
    ) -> None:
        """Resolve a discrete multi-target attack (Electro Wizard).

        This is deliberately separate from splash damage: the component picks
        at most ``max_targets`` legal victims in the attacker's range and
        applies one damage instance to each.  The primary target acquired by
        the normal targeting engine is always first; remaining ties are
        resolved by distance and UID so replays remain bit-identical.
        """

        if not hasattr(raw_component, "get"):
            raise ValueError(f"{source.card_id}: multi_target_attack must be an object")
        definition = self._definition(source)
        max_targets = int(raw_component.get("max_targets") or 0)
        range_mtile = int(raw_component.get("range_mtile") or definition.range_mtile or 0)
        if max_targets < 2 or range_mtile <= 0:
            raise ValueError(f"{source.card_id}: invalid multi-target component")
        candidates = [
            target
            for target in self._alive_entities(state)
            if target.owner != source.owner
            and self._spell_can_hit(source.card_id, target)
            and distance_mtile(source.x_mtile, source.y_mtile, target.x_mtile, target.y_mtile)
            <= range_mtile + self._collision_radius(target)
        ]
        candidates.sort(
            key=lambda target: (
                0 if target.uid == primary_target_uid else 1,
                distance_mtile(source.x_mtile, source.y_mtile, target.x_mtile, target.y_mtile),
                target.uid,
            )
        )
        for index, target in enumerate(candidates[:max_targets], start=1):
            dealt = (
                int(definition.crown_tower_damage)
                if target.kind == "tower" and definition.crown_tower_damage is not None
                else int(definition.damage or 0)
            )
            dealt = self._scale_level_value(dealt, source.level_multiplier_permille)
            self._deal_damage(state, target, dealt, source.uid, source.card_id)
            if status and target.hp > 0:
                self._apply_status(state, target, status)
            if reset_attack and target.hp > 0:
                target.attack_cooldown_us = 0
                target.windup_remaining_us = 0
                target.pending_target_uid = None
            self._emit(
                state,
                "multi_target_hit",
                source_uid=source.uid,
                source_card_id=source.card_id,
                target_uid=target.uid,
                target_index=index,
            )

    def _impact_chain_projectile(
        self,
        state: BattleState,
        *,
        projectile: ProjectileState,
        raw_component: object,
        status: object,
        reset_attack: bool,
    ) -> None:
        """Resolve a bounded nearest-neighbour chain projectile.

        The first target is the homing projectile's acquired target.  Each
        subsequent target must be an enemy legal for the card and lie within
        the component's hop radius of the previous target.  A target is never
        hit twice by one chain.  This captures Electro Dragon's strategic
        behavior while retaining explicit events for later frame-level timing
        calibration.
        """

        if not hasattr(raw_component, "get"):
            raise ValueError(f"{projectile.source_card_id}: chain_attack must be an object")
        definition = self.ruleset.card(projectile.source_card_id)
        max_targets = int(raw_component.get("max_targets") or 0)
        chain_range = int(raw_component.get("chain_range_mtile") or 0)
        if max_targets < 2 or chain_range <= 0:
            raise ValueError(f"{projectile.source_card_id}: invalid chain component")
        first = state.entities.get(projectile.target_uid) if projectile.target_uid is not None else None
        selected: list[EntityState] = []
        if (
            first is not None
            and first.alive
            and first.owner != projectile.owner
            and self._spell_can_hit(projectile.source_card_id, first)
        ):
            selected.append(first)
        anchor_x = first.x_mtile if first is not None else projectile.target_x_mtile
        anchor_y = first.y_mtile if first is not None else projectile.target_y_mtile
        while len(selected) < max_targets:
            candidates = [
                target
                for target in self._alive_entities(state)
                if target.owner != projectile.owner
                and target.uid not in {row.uid for row in selected}
                and self._spell_can_hit(projectile.source_card_id, target)
                and distance_mtile(anchor_x, anchor_y, target.x_mtile, target.y_mtile)
                <= chain_range + self._collision_radius(target)
            ]
            if not candidates:
                break
            candidates.sort(
                key=lambda target: (
                    distance_mtile(anchor_x, anchor_y, target.x_mtile, target.y_mtile),
                    target.uid,
                )
            )
            selected.append(candidates[0])
            anchor_x, anchor_y = selected[-1].x_mtile, selected[-1].y_mtile
        delay = int(raw_component.get("chain_delay_us") or 0)
        if not selected:
            return
        if delay <= 0:
            for index, target in enumerate(selected, start=1):
                self._apply_chain_hit(state, projectile, target, index)
            return
        projectile.chain_target_uids = [target.uid for target in selected]
        projectile.chain_next_index = 1
        projectile.chain_delay_us = delay
        projectile.chain_delay_remaining_us = delay
        self._apply_chain_hit(state, projectile, selected[0], 1)

    def _apply_chain_hit(
        self,
        state: BattleState,
        projectile: ProjectileState,
        target: EntityState,
        target_index: int,
    ) -> None:
        dealt = projectile.crown_damage if target.kind == "tower" else projectile.damage
        self._deal_damage(
            state, target, dealt, projectile.source_uid, projectile.source_card_id
        )
        if projectile.status_kind and target.hp > 0:
            self._apply_status(
                state,
                target,
                {
                    "kind": projectile.status_kind,
                    "duration_us": projectile.status_duration_us,
                    "speed_multiplier_milli": projectile.status_magnitude_permille,
                    "hit_speed_multiplier_milli": projectile.status_hit_speed_magnitude_permille,
                    "source_level_multiplier_permille": projectile.level_multiplier_permille,
                },
            )
        definition = self.ruleset.cards[projectile.source_card_id]
        if definition.mechanics.get("reset_attack") and target.hp > 0:
            target.attack_cooldown_us = 0
            target.windup_remaining_us = 0
            target.pending_target_uid = None
        self._emit(
            state,
            "chain_hit",
            source_uid=projectile.source_uid,
            source_card_id=projectile.source_card_id,
            target_uid=target.uid,
            target_index=target_index,
        )

    def _impact_clone(
        self,
        state: BattleState,
        *,
        owner: int,
        source_uid: int | None,
        source_card_id: str,
        x: int,
        y: int,
        radius: int,
        raw_clone: object,
        level_multiplier_permille: int = PERMILLE,
    ) -> None:
        """Copy eligible friendly troop bodies at a Clone impact.

        Clone deliberately bypasses :meth:`_impact_area`: that generic helper
        selects enemy victims and applies damage/status.  The spell instead
        snapshots friendly troops in stable UID order, excludes buildings and
        already-cloned bodies, and creates ordinary one-HP card entities so
        their normal movement, targeting, death effects, and spawner streams
        continue to work.  The exact visual offset behind an original is still
        an explicit fidelity unknown; starting at the source body and letting
        deterministic collision separation resolve overlap is the conservative
        V1 assumption.
        """

        if not hasattr(raw_clone, "get"):
            raise ValueError(f"{source_card_id}: clone component must be an object")
        clone_hp = int(raw_clone.get("clone_hp") or 1)
        clone_max_hp = int(raw_clone.get("clone_max_hp") or clone_hp)
        copy_kind = str(raw_clone.get("copy_kind") or "troop")
        exclude_clones = bool(raw_clone.get("exclude_clones", True))
        if copy_kind != "troop":
            raise ValueError(f"{source_card_id}: unsupported clone copy_kind {copy_kind!r}")
        originals = [
            entity
            for entity in self._alive_entities(state)
            if entity.owner == owner
            and entity.kind == copy_kind
            and (not exclude_clones or not entity.is_clone)
            and distance_mtile(x, y, entity.x_mtile, entity.y_mtile)
            <= radius + self._collision_radius(entity)
        ]
        cloned = 0
        for original in originals:
            child = self.ruleset.card(original.card_id)
            self._spawn_single_at(
                state,
                child,
                owner=owner,
                x_mtile=original.x_mtile,
                y_mtile=original.y_mtile,
                parent_uid=original.uid,
                event_kind="entity_cloned",
                is_clone=True,
                hp_override=clone_hp,
                max_hp_override=clone_max_hp,
                level_multiplier_permille=level_multiplier_permille,
            )
            cloned += 1
        self._emit(
            state,
            "clone_impact",
            uid=source_uid,
            player=owner,
            card_id=source_card_id,
            x_mtile=x,
            y_mtile=y,
            cloned_count=cloned,
        )

    def _create_area_effect(
        self,
        state: BattleState,
        *,
        owner: int,
        source_uid: int | None,
        source_card_id: str,
        x_mtile: int,
        y_mtile: int,
        default_radius: int,
        default_damage: int,
        default_crown_damage: int,
        default_status: object,
        default_knockback: int,
        raw_effect: object,
        level_multiplier_permille: int = PERMILLE,
    ) -> None:
        """Create and immediately pulse a data-driven persistent effect."""

        if not hasattr(raw_effect, "get"):
            raise ValueError(f"{source_card_id}: persistent_effect must be an object")
        effect = raw_effect
        duration_us = int(effect.get("duration_us") or 0)
        interval_us = int(effect.get("tick_interval_us") or 0)
        initial_delay_us = int(effect.get("initial_delay_us") or 0)
        if duration_us <= 0 or interval_us <= 0:
            raise ValueError(f"{source_card_id}: persistent effect has invalid timing")
        status = effect.get("status")
        if status is None:
            status = default_status
        status_kind = None if not status else str(status.get("kind"))
        status_duration = 0 if not status else int(status.get("duration_us") or 0)
        status_magnitude = (
            1_000 if not status else int(status.get("speed_multiplier_milli") or 1_000)
        )
        status_hit_speed_magnitude = (
            1_000
            if not status
            else int(status.get("hit_speed_multiplier_milli") or 1_000)
        )
        status_on_death_spawn_card_id = (
            None if not status else status.get("on_death_spawn_card_id")
        )
        status_on_death_spawn_count = (
            0 if not status else int(status.get("on_death_spawn_count") or 0)
        )
        raw_allowed = effect.get("targets")
        allowed_targets = tuple(
            str(item)
            for item in (
                raw_allowed
                if raw_allowed is not None
                else self.ruleset.cards[source_card_id].targets
            )
        )
        spawn = effect.get("spawn")
        spawn_card_id = None if not spawn else str(spawn.get("card_id"))
        spawn_count = 0 if not spawn else int(spawn.get("count") or 0)
        max_spawns = 0 if not spawn else int(spawn.get("max_spawns") or 0)
        def _schedule(name: str) -> tuple[int, ...]:
            values = effect.get(name)
            if values is None:
                return ()
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{source_card_id}: {name} must be a non-empty sequence")
            parsed = tuple(
                self._scale_level_value(int(value), level_multiplier_permille)
                for value in values
            )
            if any(value < 0 for value in parsed):
                raise ValueError(f"{source_card_id}: {name} contains negative damage")
            return parsed
        damage_schedule = _schedule("damage_schedule")
        crown_damage_schedule = _schedule("crown_damage_schedule")
        friendly_status = effect.get("friendly_status")
        friendly_status_kind = None if not friendly_status else str(friendly_status.get("kind"))
        friendly_status_duration = (
            0 if not friendly_status else int(friendly_status.get("duration_us") or 0)
        )
        friendly_status_magnitude = (
            1_000
            if not friendly_status
            else int(friendly_status.get("speed_multiplier_milli") or 1_000)
        )
        friendly_status_linger = (
            0 if not friendly_status else int(friendly_status.get("linger_us") or 0)
        )
        friendly_targets = tuple(str(item) for item in (effect.get("friendly_targets") or ()))
        duration_anchor = str(effect.get("duration_anchor") or "after_immediate")
        if duration_anchor not in {"after_immediate", "creation"}:
            raise ValueError(f"{source_card_id}: invalid duration_anchor {duration_anchor!r}")
        uid = self._allocate_uid(state)
        area = AreaEffectState(
            uid=uid,
            source_uid=source_uid,
            source_card_id=source_card_id,
            owner=owner,
            x_mtile=x_mtile,
            y_mtile=y_mtile,
            radius_mtile=int(effect.get("radius_mtile") or default_radius),
            # Most legacy persistent components model the immediate pulse as
            # consuming the first interval.  Effects with a non-integral
            # published lifetime (Tornado is the first) can explicitly anchor
            # their duration at creation while retaining the same immediate
            # pulse behavior.
            remaining_us=(
                duration_us
                if duration_anchor == "creation"
                else max(0, duration_us - interval_us)
            ),
            tick_interval_us=interval_us,
            initial_delay_remaining_us=initial_delay_us,
            damage_per_tick=(
                default_damage
                if effect.get("damage_per_tick") is None
                else self._scale_level_value(
                    int(effect.get("damage_per_tick") or 0), level_multiplier_permille
                )
            ),
            crown_damage_per_tick=(
                default_crown_damage
                if effect.get("crown_damage_per_tick") is None
                else self._scale_level_value(
                    int(effect.get("crown_damage_per_tick") or 0), level_multiplier_permille
                )
            ),
            status_kind=status_kind,
            status_duration_us=status_duration,
            status_magnitude_permille=status_magnitude,
            status_hit_speed_magnitude_permille=status_hit_speed_magnitude,
            status_damage_per_tick=0 if not status else int(status.get("damage_per_tick") or 0),
            status_tick_interval_us=0 if not status else int(status.get("tick_interval_us") or 0),
            knockback_mtile=int(effect.get("knockback_mtile") or default_knockback),
            pull_to_center_mtile=int(effect.get("pull_to_center_mtile") or 0),
            allowed_targets=allowed_targets,
            spawn_card_id=spawn_card_id,
            spawn_count=spawn_count,
            max_spawns=max_spawns,
            damage_schedule=damage_schedule,
            crown_damage_schedule=crown_damage_schedule,
            friendly_status_kind=friendly_status_kind,
            friendly_status_duration_us=friendly_status_duration,
            friendly_status_magnitude_permille=friendly_status_magnitude,
            friendly_status_linger_us=friendly_status_linger,
            friendly_allowed_targets=friendly_targets,
            status_on_death_spawn_card_id=(
                None
                if status_on_death_spawn_card_id is None
                else str(status_on_death_spawn_card_id)
            ),
            status_on_death_spawn_count=status_on_death_spawn_count,
            max_pulses=(
                None
                if effect.get("max_pulses") is None
                else int(effect.get("max_pulses"))
            ),
            level_multiplier_permille=level_multiplier_permille,
        )
        state.effects[uid] = area
        self._emit(
            state,
            "area_effect_created",
            uid=uid,
            player=owner,
            card_id=source_card_id,
            x_mtile=x_mtile,
            y_mtile=y_mtile,
        )
        if initial_delay_us == 0:
            self._apply_area_effect_tick(state, area)
        if area.remaining_us == 0:
            area.alive = False
            self._emit(
                state,
                "area_effect_expired",
                uid=area.uid,
                card_id=area.source_card_id,
            )

    def _impact_piercing_projectile(self, state: BattleState, projectile: ProjectileState) -> None:
        mechanics = self.ruleset.cards[projectile.source_card_id].mechanics
        line = mechanics.get("line_piercing")
        returning = mechanics.get("returning_projectile")
        impact_mode = mechanics.get("impact_mode")
        # Executioner's axe uses the same swept-path collision model as a
        # line projectile, but has a separately sourced width and a second
        # pass on the way back.  Rolling spell components opt into the same
        # sweep through their continuous impact mode; ordinary radial
        # projectiles retain point-impact behavior.
        swept_path = (
            line is not None
            or returning is not None
            or impact_mode in {"continuous", "continuous_path"}
        )
        line_width = (
            int(line.get("width_mtile") or projectile.radius_mtile)
            if hasattr(line, "get")
            else int(returning.get("return_radius_mtile") or projectile.radius_mtile)
            if hasattr(returning, "get")
            else projectile.radius_mtile
        )
        ax, ay = projectile.origin_x_mtile, projectile.origin_y_mtile
        # Older replay fixtures (and a few component-level callers) construct
        # ``ProjectileState`` directly, before the fixed line-origin fields
        # were added.  Their zero-valued origin is not an authored launch
        # point; the current projectile position is the only available start
        # coordinate.  Use it as the fallback so a direct impact resolves at
        # the endpoint/point instead of sweeping an accidental diagonal from
        # the arena origin through unrelated bodies.  Engine-created
        # projectiles always carry explicit origin metadata and are unchanged.
        if (
            ax == 0
            and ay == 0
            and (projectile.x_mtile != 0 or projectile.y_mtile != 0)
        ):
            ax, ay = projectile.x_mtile, projectile.y_mtile
        bx, by = projectile.x_mtile, projectile.y_mtile
        vx, vy = bx - ax, by - ay
        denominator = vx * vx + vy * vy
        for target in self._alive_entities(state):
            if target.owner == projectile.owner or target.uid in projectile.hit_uids:
                continue
            if not self._spell_can_hit(
                projectile.source_card_id,
                target,
                allowed_targets=projectile.allowed_targets or None,
            ):
                continue
            if line is None and not swept_path:
                if distance_mtile(
                    projectile.x_mtile,
                    projectile.y_mtile,
                    target.x_mtile,
                    target.y_mtile,
                ) > projectile.radius_mtile + self._collision_radius(target):
                    continue
            elif denominator == 0:
                if distance_mtile(ax, ay, target.x_mtile, target.y_mtile) > line_width + self._collision_radius(target):
                    continue
            else:
                projection = (
                    (target.x_mtile - ax) * vx + (target.y_mtile - ay) * vy
                )
                projection = max(0, min(denominator, projection))
                nearest_x = ax + vx * projection // denominator
                nearest_y = ay + vy * projection // denominator
                if distance_mtile(nearest_x, nearest_y, target.x_mtile, target.y_mtile) > line_width + self._collision_radius(target):
                    continue
            projectile.hit_uids.append(target.uid)
            damage = projectile.crown_damage if target.kind == "tower" else projectile.damage
            self._deal_damage(state, target, damage, projectile.source_uid, projectile.source_card_id)
            direction: tuple[int, int] | None = None
            if mechanics.get("knockback_direction") == "projectile_travel":
                direction = (
                    projectile.direction_x_mtile,
                    projectile.direction_y_mtile,
                )
                if direction == (0, 0):
                    direction = (
                        projectile.target_x_mtile - projectile.x_mtile,
                        projectile.target_y_mtile - projectile.y_mtile,
                    )
                if direction == (0, 0):
                    direction = (0, -1 if projectile.owner == 0 else 1)
            self._apply_knockback(
                state,
                target,
                projectile.x_mtile,
                projectile.y_mtile,
                projectile.knockback_mtile,
                direction=direction,
            )
            self._emit(
                state,
                "piercing_hit",
                source_uid=projectile.source_uid,
                source_card_id=projectile.source_card_id,
                target_uid=target.uid,
                return_phase=projectile.return_phase,
            )

    def _impact_area(
        self,
        state: BattleState,
        *,
        owner: int,
        source_uid: int | None,
        source_card_id: str,
        x: int,
        y: int,
        damage: int,
        crown_damage: int,
        radius: int,
        status: object,
        knockback: int,
        primary_target_uid: int | None,
        allowed_targets: tuple[str, ...] | None = None,
        target_limit: int | None = None,
        target_selection: str | None = None,
        reset_attack: bool = False,
        knockback_direction: tuple[int, int] | None = None,
    ) -> None:
        candidates: list[EntityState] = []
        if radius <= 0 and primary_target_uid is not None:
            target = state.entities.get(primary_target_uid)
            if target is not None and target.alive and target.owner != owner:
                candidates = [target]
        else:
            for target in self._alive_entities(state):
                if target.owner == owner or not self._spell_can_hit(
                    source_card_id,
                    target,
                    allowed_targets=allowed_targets,
                ):
                    continue
                if distance_mtile(x, y, target.x_mtile, target.y_mtile) <= radius + self._collision_radius(target):
                    candidates.append(target)
        if target_limit is not None and len(candidates) > target_limit:
            if target_selection == "highest_hp":
                candidates.sort(key=lambda target: (-target.hp, target.uid))
            else:
                candidates.sort(
                    key=lambda target: (
                        distance_mtile(x, y, target.x_mtile, target.y_mtile),
                        target.uid,
                    )
                )
            candidates = candidates[:target_limit]
        for target in candidates:
            dealt = crown_damage if target.kind == "tower" else damage
            curse_status = bool(status and hasattr(status, "get") and status.get("on_death_spawn_card_id"))
            if curse_status and target.hp > 0:
                self._apply_status(state, target, status)
            self._deal_damage(state, target, dealt, source_uid, source_card_id)
            if status and target.hp > 0 and not curse_status:
                self._apply_status(state, target, status)
            if reset_attack and target.hp > 0:
                target.attack_cooldown_us = 0
                target.windup_remaining_us = 0
                target.pending_target_uid = None
            if target.hp > 0:
                self._apply_knockback(
                    state,
                    target,
                    x,
                    y,
                    knockback,
                    direction=knockback_direction,
                    excluded_structure_uid=source_uid,
                )

    def _spell_can_hit(
        self,
        card_id: str,
        target: EntityState,
        *,
        allowed_targets: tuple[str, ...] | None = None,
    ) -> bool:
        if target.concealed_active and card_id not in {"earthquake", "freeze"}:
            return False
        if allowed_targets is not None or card_id in self.ruleset.cards:
            authored_impact_targets = (
                None
                if card_id not in self.ruleset.cards
                else self.ruleset.cards[card_id].mechanics.get("impact_targets")
            )
            targets = set(
                allowed_targets
                if allowed_targets is not None
                else authored_impact_targets
                if authored_impact_targets is not None
                else self.ruleset.cards[card_id].targets
            )
            if target.kind == "tower":
                return (
                    "crown_tower" in targets
                    or "building" in targets
                    or "ground" in targets
                )
            if target.kind == "building":
                return "building" in targets or "ground" in targets
            layer = self._movement_layer(target)
            return str(layer) in targets
        return True

    def _deal_damage(
        self,
        state: BattleState,
        target: EntityState,
        damage: int,
        source_uid: int | None,
        source_card_id: str,
    ) -> None:
        if damage <= 0 or not target.alive or target.hp <= 0:
            return
        source_definition = self.ruleset.cards.get(source_card_id)
        if (
            source_definition is not None
            and source_definition.mechanics.get("spirit_one_shot")
            and target.card_id in {
                "electro-spirit",
                "fire-spirit",
                "heal-spirit",
                "ice-spirit",
            }
        ):
            # August 2026's Archer interaction is authored as a mechanic on
            # the attacker rather than as a Spirit stat override.  Resolve
            # it at the common damage boundary so direct projectile impacts,
            # replay-loaded projectiles, and normal attack scheduling agree.
            damage = max(damage, target.hp + target.shield_hp)
        if target.shield_hp > 0:
            before_shield = target.shield_hp
            target.shield_hp = max(0, target.shield_hp - damage)
            absorbed = before_shield - target.shield_hp
            self._emit(
                state,
                "shield_damaged",
                source_uid=source_uid,
                source_card_id=source_card_id,
                target_uid=target.uid,
                damage=damage,
                absorbed=absorbed,
                shield_hp_after=target.shield_hp,
            )
            if target.shield_hp == 0:
                self._emit(
                    state,
                    "shield_broken",
                    source_uid=source_uid,
                    source_card_id=source_card_id,
                    target_uid=target.uid,
                )
            # Clash Royale shield damage is a complete hit transaction: any
            # excess damage is discarded rather than spilling into body HP.
            return
        before = target.hp
        target.hp = max(0, target.hp - damage)
        self._emit(
            state,
            "damage_applied",
            source_uid=source_uid,
            source_card_id=source_card_id,
            target_uid=target.uid,
            damage=before - target.hp,
            hp_after=target.hp,
        )
        self._maybe_transform_health(state, target)
        if target.hp > 0:
            self._maybe_reflect_damage(
                state,
                target=target,
                source_uid=source_uid,
                source_card_id=source_card_id,
            )
        if target.kind == "tower" and target.role == "king":
            self._activate_king(state, target.owner, "damaged")

    def _maybe_transform_health(self, state: BattleState, entity: EntityState) -> None:
        """Apply a data-driven health-threshold form change in place.

        A transformation keeps the UID and the shared remaining health.  The
        destination card supplies the stationary/building combat definition;
        movement, target locks, attack wind-up, and navigation caches are
        reset because they belong to the pre-transform form.  The component
        disappears with the source card, making the transition one-shot
        without a second boolean in authoritative state.
        """

        if not entity.alive or entity.hp <= 0:
            return
        if entity.kind == "tower":
            return
        source = self._definition(entity)
        component = source.mechanics.get("health_transform")
        if not hasattr(component, "get"):
            return
        threshold = int(component.get("threshold_permille") or 0)
        if threshold <= 0 or entity.max_hp <= 0:
            return
        if entity.hp * PERMILLE > entity.max_hp * threshold:
            return
        target_card_id = str(component.get("target_card_id") or "")
        if not target_card_id:
            raise ValueError(f"{entity.card_id}: health transform lacks target card")
        target_definition = self.ruleset.card(target_card_id)
        before_card_id = entity.card_id
        before_kind = entity.kind
        before_hp = entity.hp
        before_max_hp = entity.max_hp
        preserve_hp = bool(component.get("preserve_hp", True))
        preserve_max_hp = bool(component.get("preserve_max_hp", True))
        # The May-2025 Cannon Cart rework keeps the same target lock when the
        # wheel form becomes a stationary building.  Snapshot both channels
        # before replacing the card definition; validation is performed after
        # the destination kind/targets are installed below.
        preserved_target_uid = entity.target_uid
        preserved_pending_target_uid = entity.pending_target_uid
        preserved_windup_remaining_us = entity.windup_remaining_us
        preserved_attack_cooldown_us = entity.attack_cooldown_us

        entity.card_id = target_definition.card_id
        entity.kind = target_definition.kind
        if not preserve_max_hp:
            entity.max_hp = int(target_definition.hitpoints or before_max_hp)
        if preserve_hp:
            entity.hp = min(before_hp, entity.max_hp)
        else:
            entity.hp = min(int(target_definition.hitpoints or before_hp), entity.max_hp)
        if entity.hp <= 0:
            # Defensive guard for malformed custom rulesets.  The component
            # is only legal for a live target, so a zero result is a hard
            # configuration error rather than a silently dead transformation.
            raise ValueError(f"{before_card_id}: health transform produced non-positive HP")

        entity.role = None
        entity.target_uid = None
        entity.pending_target_uid = None
        entity.secondary_pending_target_uid = None
        entity.deploy_remaining_us = int(target_definition.deploy_time_us)
        entity.attack_cooldown_us = int(target_definition.first_hit_delay_us or 0)
        entity.windup_remaining_us = 0
        entity.secondary_attack_cooldown_us = 0
        entity.secondary_windup_remaining_us = 0
        entity.lifetime_remaining_us = int(
            component.get("lifetime_us")
            or target_definition.lifetime_us
            or 0
        ) or None
        entity.lifetime_decay_remainder = 0
        entity.spawn_cooldown_us = 0
        entity.spawn_time_remainder = 0
        entity.spawned_count = 0
        entity.movement_remainder = 0
        entity.attack_time_remainder = 0
        entity.navigation_target_uid = None
        entity.navigation_revision = -1
        entity.navigation_goal_x_mtile = entity.x_mtile
        entity.navigation_goal_y_mtile = entity.y_mtile
        entity.navigation_cursor = 0
        entity.navigation_waypoints.clear()
        entity.charge_active = False
        entity.charge_remaining_us = None
        entity.attack_charge_active = False
        entity.attack_charge_distance_mtile = 0
        entity.dash_attack_active = False
        entity.ramp_elapsed_us = 0
        entity.ramp_stage = 0
        entity.secondary_attack_count = 0
        entity.attack_count = 0
        entity.revive_eligible = False
        if preserved_target_uid is not None:
            preserved_target = state.entities.get(preserved_target_uid)
            if (
                preserved_target is not None
                and preserved_target.alive
                and self._valid_target(state, entity, preserved_target_uid)
            ):
                entity.target_uid = preserved_target_uid
                entity.attack_cooldown_us = preserved_attack_cooldown_us
                if (
                    preserved_pending_target_uid == preserved_target_uid
                    and preserved_windup_remaining_us > 0
                ):
                    entity.pending_target_uid = preserved_pending_target_uid
                    entity.windup_remaining_us = preserved_windup_remaining_us
        state.navigation_revision += 1
        self._emit(
            state,
            "entity_transformed",
            uid=entity.uid,
            source_card_id=before_card_id,
            target_card_id=target_definition.card_id,
            source_kind=before_kind,
            target_kind=target_definition.kind,
            threshold_permille=threshold,
            hp=entity.hp,
            max_hp=entity.max_hp,
            lifetime_remaining_us=entity.lifetime_remaining_us,
        )

    def _maybe_reflect_damage(
        self,
        state: BattleState,
        *,
        target: EntityState,
        source_uid: int | None,
        source_card_id: str,
    ) -> None:
        """Apply a reactive damage/stun pulse from a reflecting entity.

        Reflection is triggered by a concrete attacker UID, not by area/spell
        damage with no source body.  The synthetic ``:reflection`` source tag
        prevents two Electro Giants from recursively reflecting one another's
        zaps.  Radius and target legality are evaluated at the time the hit is
        received, and the result is represented as ordinary damage/status
        events for replay and sim-to-real comparison.
        """

        if source_uid is None or source_card_id.endswith(":reflection"):
            return
        if target.kind != "troop":
            return
        definition = self.ruleset.cards.get(target.card_id)
        if definition is None:
            return
        raw_reflection = definition.mechanics.get("reflection")
        if not hasattr(raw_reflection, "get"):
            return
        attacker = state.entities.get(source_uid)
        if attacker is None or not attacker.alive or attacker.owner == target.owner:
            return
        reflection = raw_reflection
        radius = int(reflection.get("radius_mtile") or 0)
        if distance_mtile(target.x_mtile, target.y_mtile, attacker.x_mtile, attacker.y_mtile) > radius + self._collision_radius(attacker):
            return
        allowed = tuple(str(value) for value in reflection.get("targets", ()))
        if not self._spell_can_hit(target.card_id, attacker, allowed_targets=allowed):
            return
        damage = (
            int(reflection.get("crown_tower_damage") or 0)
            if attacker.kind == "tower"
            else int(reflection.get("damage") or 0)
        )
        damage = self._scale_level_value(
            damage, target.level_multiplier_permille
        )
        self._deal_damage(
            state,
            attacker,
            damage,
            source_uid=target.uid,
            source_card_id=f"{target.card_id}:reflection",
        )
        stun_duration = int(reflection.get("stun_duration_us") or 0)
        if attacker.alive and attacker.hp > 0 and stun_duration > 0:
            self._apply_status(
                state,
                attacker,
                {
                    "kind": "stun",
                    "duration_us": stun_duration,
                    "speed_multiplier_milli": 0,
                    "hit_speed_multiplier_milli": 0,
                },
            )
            attacker.attack_cooldown_us = 0
            attacker.windup_remaining_us = 0
            attacker.pending_target_uid = None
        self._emit(
            state,
            "reflected_damage",
            source_uid=target.uid,
            source_card_id=target.card_id,
            target_uid=attacker.uid,
            damage=damage,
            crown_tower_damage=(
                self._scale_level_value(
                    int(reflection.get("crown_tower_damage") or 0),
                    target.level_multiplier_permille,
                )
                if attacker.kind == "tower"
                else 0
            ),
        )

    def _apply_status(self, state: BattleState, target: EntityState, raw_status: object) -> None:
        if not hasattr(raw_status, "get"):
            return
        status = raw_status
        kind = str(status.get("kind"))
        duration = int(status.get("duration_us") or 0)
        magnitude = int(status.get("speed_multiplier_milli") or 0)
        hit_speed_magnitude = int(
            status.get("hit_speed_multiplier_milli")
            if status.get("hit_speed_multiplier_milli") is not None
            else magnitude
        )
        if not kind or duration <= 0:
            return
        if kind in {"stun", "freeze"}:
            self._reset_attack_charge(state, target, reason=kind)
            self._reset_dash(state, target, reason=kind)
            self._reset_attack_ramp(state, target, reason=kind)
            # A hard CC on an Inferno beam's victim breaks the lock just as a
            # retarget does.  Reset every attacker currently locked to this
            # target in stable UID order; the next acquisition starts stage 1.
            for attacker in self._alive_entities(state):
                if attacker.target_uid == target.uid:
                    self._reset_attack_ramp(
                        state,
                        attacker,
                        reason=f"target_{kind}",
                    )
            # A hard crowd-control effect resets Sparky's charged shot.  The
            # generic scheduler otherwise retains the four-second wind-up and
            # would allow a shot to fire immediately after a Zap/Freeze.
            if target.card_id == "sparky":
                target.attack_cooldown_us = 0
                target.windup_remaining_us = 0
                target.pending_target_uid = None
        on_death_spawn_card_id = status.get("on_death_spawn_card_id")
        on_death_spawn_count = int(status.get("on_death_spawn_count") or 0)
        on_death_spawn_owner = status.get("on_death_spawn_owner")
        source_level_multiplier = int(
            status.get("source_level_multiplier_permille") or PERMILLE
        )
        if on_death_spawn_card_id is not None:
            on_death_spawn_card_id = str(on_death_spawn_card_id)
            if on_death_spawn_count <= 0:
                raise ValueError(
                    f"status {kind!r} has a child card but no positive spawn count"
                )
            if on_death_spawn_owner not in (0, 1):
                raise ValueError(
                    f"status {kind!r} has an invalid child owner"
                )
        existing = next((row for row in target.statuses if row.kind == kind), None)
        if existing is None:
            target.statuses.append(
                StatusState(
                    kind,
                    duration,
                    magnitude,
                    int(status.get("damage_per_tick") or 0),
                    int(status.get("tick_interval_us") or 0),
                    0,
                    on_death_spawn_card_id,
                    on_death_spawn_count,
                    on_death_spawn_owner,
                    hit_speed_magnitude_permille=hit_speed_magnitude,
                    source_level_multiplier_permille=source_level_multiplier,
                )
            )
        else:
            existing.remaining_us = max(existing.remaining_us, duration)
            existing.magnitude_permille = min(existing.magnitude_permille, magnitude)
            existing.hit_speed_magnitude_permille = min(
                (
                    existing.hit_speed_magnitude_permille
                    if existing.hit_speed_magnitude_permille is not None
                    else existing.magnitude_permille
                ),
                hit_speed_magnitude,
            )
            existing.damage_per_tick = max(
                existing.damage_per_tick,
                int(status.get("damage_per_tick") or 0),
            )
            existing.tick_interval_us = max(
                existing.tick_interval_us,
                int(status.get("tick_interval_us") or 0),
            )
            if on_death_spawn_card_id is not None:
                existing.on_death_spawn_card_id = on_death_spawn_card_id
                existing.on_death_spawn_count = max(
                    existing.on_death_spawn_count,
                    on_death_spawn_count,
                )
                existing.on_death_spawn_owner = on_death_spawn_owner
                existing.source_level_multiplier_permille = max(
                    existing.source_level_multiplier_permille,
                    source_level_multiplier,
                )
        target.statuses.sort(key=lambda row: row.kind)
        self._emit(state, "status_applied", uid=target.uid, status=kind, duration_us=duration)

    def _apply_knockback(
        self,
        state: BattleState,
        target: EntityState,
        source_x: int,
        source_y: int,
        distance: int,
        *,
        direction: tuple[int, int] | None = None,
        excluded_structure_uid: int | None = None,
    ) -> None:
        if distance <= 0 or target.kind in {"tower", "building"} or not target.alive or target.hp <= 0:
            return
        origin_x, origin_y = target.x_mtile, target.y_mtile
        if direction is None:
            dx = target.x_mtile - source_x
            dy = target.y_mtile - source_y
        else:
            dx, dy = direction
        if dx == 0 and dy == 0:
            dy = 1 if target.owner == 0 else -1
        far_x = origin_x + dx * 100
        far_y = origin_y + dy * 100

        def candidate(travel: int) -> tuple[int, int]:
            return move_towards(origin_x, origin_y, far_x, far_y, travel)

        def swept_clear(travel: int) -> bool:
            if travel <= 0:
                return True
            steps = max(1, ceil_div(travel, 50))
            return all(
                self._position_clear_of_structures(
                    state,
                    target,
                    *candidate(travel * index // steps),
                    exclude_target=False,
                    excluded_structure_uid=excluded_structure_uid,
                )
                for index in range(1, steps + 1)
            )

        destination = candidate(distance)
        if not swept_clear(distance):
            # Knockback cannot tunnel through a building, tower, arena edge,
            # or river bank. Sweeping the entire ray is essential: checking
            # only the destination would allow a long push to emerge on the
            # far side of a structure. Find the furthest legal integer
            # displacement with deterministic binary refinement.
            low = 0
            high = distance
            while low < high:
                middle = (low + high + 1) // 2
                if swept_clear(middle):
                    low = middle
                else:
                    high = middle - 1
            destination = candidate(low)
        target.x_mtile, target.y_mtile = destination
        if destination != (origin_x, origin_y):
            self._reset_attack_charge(state, target, reason="knockback")
            self._reset_dash(state, target, reason="knockback")
            self._reset_attack_ramp(state, target, reason="knockback")
            self._emit(
                state,
                "knockback_applied",
                target_uid=target.uid,
                from_x_mtile=origin_x,
                from_y_mtile=origin_y,
                to_x_mtile=destination[0],
                to_y_mtile=destination[1],
                distance_mtile=distance,
            )
        target.navigation_waypoints.clear()
        target.navigation_cursor = 0
        target.navigation_revision = -1

    def _resolve_deaths(self, state: BattleState) -> list[EntityState]:
        destroyed_towers: list[EntityState] = []
        while True:
            dead = [
                entity
                for entity in state.entities.values()
                if entity.alive and entity.hp <= 0
            ]
            if not dead:
                break
            for entity in sorted(dead, key=lambda row: row.uid):
                entity.alive = False
                entity.hp = 0
                entity.target_uid = None
                if entity.kind in {"building", "tower"}:
                    state.navigation_revision += 1
                self._emit(
                    state,
                    "entity_died",
                    uid=entity.uid,
                    player=entity.owner,
                    card_id=entity.card_id,
                )
                if entity.kind == "tower":
                    destroyed_towers.append(entity)
                    self._emit(
                        state,
                        "tower_destroyed",
                        uid=entity.uid,
                        player=entity.owner,
                        role=entity.role,
                    )
                    continue
                self._apply_status_death_transform(state, entity)
                self._apply_death_effect(state, entity)
                self._apply_revive(state, entity)
                self._release_carried_children(state, entity)
                # Statuses are attached to the living body.  Keeping a live
                # Poison/Freeze/Rage record on a dead entity makes the
                # authoritative replay retain effects that can no longer
                # tick, and strict state validation quite rightly rejects it.
                # Death transforms above intentionally run first because a
                # curse can inspect its victim's status before the body is
                # removed from play.
                entity.statuses.clear()
        for entity in state.entities.values():
            if entity.target_uid is not None and not state.entities[entity.target_uid].alive:
                entity.target_uid = None
            if entity.pending_target_uid is not None and not state.entities[entity.pending_target_uid].alive:
                entity.pending_target_uid = None
                entity.windup_remaining_us = 0
            if (
                entity.secondary_pending_target_uid is not None
                and not state.entities[entity.secondary_pending_target_uid].alive
            ):
                entity.secondary_pending_target_uid = None
                entity.secondary_windup_remaining_us = 0
        return destroyed_towers

    def _release_carried_children(self, state: BattleState, carrier: EntityState) -> None:
        """Detach a carrier's surviving child bodies after its death."""

        definition = self.ruleset.cards.get(carrier.card_id)
        if definition is None:
            return
        raw_carrier = definition.mechanics.get("carrier")
        if not raw_carrier or not bool(raw_carrier.get("release_on_death", True)):
            return
        released = 0
        for child in sorted(state.entities.values(), key=lambda row: row.uid):
            if child.carried_by_uid != carrier.uid:
                continue
            child.carried_by_uid = None
            if child.alive:
                child.x_mtile = min(
                    self.ruleset.arena.width_mtile - 1,
                    max(0, carrier.x_mtile + child.carried_offset_x_mtile),
                )
                child.y_mtile = min(
                    self.ruleset.arena.height_mtile - 1,
                    max(0, carrier.y_mtile + child.carried_offset_y_mtile),
                )
                child.deploy_remaining_us = 0
                child.navigation_waypoints.clear()
                child.navigation_cursor = 0
                child.navigation_revision = -1
                released += 1
            child.carried_offset_x_mtile = 0
            child.carried_offset_y_mtile = 0
            self._emit(
                state,
                "carrier_child_released",
                uid=child.uid,
                parent_uid=carrier.uid,
                parent_card_id=carrier.card_id,
                card_id=child.card_id,
                alive=child.alive,
            )
        if released:
            self._emit(
                state,
                "carrier_released",
                parent_uid=carrier.uid,
                parent_card_id=carrier.card_id,
                child_count=released,
            )

    def _apply_status_death_transform(
        self,
        state: BattleState,
        entity: EntityState,
    ) -> None:
        """Materialize a child produced by an active death-transform status.

        Goblin Curse owns this behavior today.  Keeping it on the status
        rather than on the victim card means a unit can be converted no
        matter whether the lethal hit came from the curse, a troop, a tower,
        or another spell, matching the game's curse semantics.
        """

        if entity.kind != "troop":
            return
        transforms = [
            status
            for status in entity.statuses
            if status.on_death_spawn_card_id is not None
            and status.on_death_spawn_count > 0
        ]
        for status in transforms:
            child_id = status.on_death_spawn_card_id
            if child_id is None:
                continue
            if child_id not in self.ruleset.cards:
                raise ValueError(
                    f"death-transform references unknown child {child_id!r}"
                )
            child = self.ruleset.card(child_id)
            owner = (
                entity.owner
                if status.on_death_spawn_owner is None
                else status.on_death_spawn_owner
            )
            for _ in range(status.on_death_spawn_count):
                self._spawn_single_at(
                    state,
                    child,
                    owner=owner,
                    x_mtile=entity.x_mtile,
                    y_mtile=entity.y_mtile,
                    parent_uid=entity.uid,
                    event_kind="entity_transformed",
                    is_clone=entity.is_clone,
                    hp_override=1 if entity.is_clone else None,
                    max_hp_override=1 if entity.is_clone else None,
                    level_multiplier_permille=status.source_level_multiplier_permille,
                )
            self._emit(
                state,
                "death_transform",
                uid=entity.uid,
                source_card_id=(
                    "mother-witch"
                    if status.kind == "mother-witch-curse"
                    else "goblin-curse"
                ),
                child_card_id=child_id,
                child_count=status.on_death_spawn_count,
                owner=owner,
            )

    def _apply_death_effect(self, state: BattleState, entity: EntityState) -> None:
        if entity.death_effect_done:
            return
        entity.death_effect_done = True
        definition = self.ruleset.cards[entity.card_id]
        death = definition.mechanics.get("death")
        if death:
            death_damage = self._scale_level_value(
                int(death.get("damage") or 0), entity.level_multiplier_permille
            )
            death_crown_damage = self._scale_level_value(
                int(
                    death["crown_tower_damage"]
                    if death.get("crown_tower_damage") is not None
                    else death.get("damage") or 0
                ),
                entity.level_multiplier_permille,
            )
            owner_reward = int(death.get("owner_elixir_milli") or 0)
            if owner_reward > 0:
                player = state.players[entity.owner]
                before = player.elixir_milli
                player.elixir_milli = min(
                    self.ruleset.match.max_elixir_milli,
                    player.elixir_milli + owner_reward,
                )
                self._emit(
                    state,
                    "elixir_awarded",
                    uid=entity.uid,
                    card_id=entity.card_id,
                    player=entity.owner,
                    amount_milli=player.elixir_milli - before,
                )
            reward = int(death.get("opponent_elixir_milli") or 0)
            if reward > 0:
                recipient = 1 - entity.owner
                player = state.players[recipient]
                before = player.elixir_milli
                player.elixir_milli = min(
                    self.ruleset.match.max_elixir_milli,
                    player.elixir_milli + reward,
                )
                self._emit(
                    state,
                    "elixir_awarded",
                    uid=entity.uid,
                    card_id=entity.card_id,
                    player=recipient,
                    amount_milli=player.elixir_milli - before,
                )
            delay_us = int(death.get("delay_us") or 0)
            if delay_us > 0:
                effect = AreaEffectState(
                    uid=self._allocate_uid(state), source_uid=entity.uid,
                    source_card_id=entity.card_id, owner=entity.owner,
                    x_mtile=entity.x_mtile, y_mtile=entity.y_mtile,
                    radius_mtile=int(death.get("radius_mtile") or 0),
                    remaining_us=delay_us, tick_interval_us=delay_us,
                    damage_per_tick=death_damage,
                    crown_damage_per_tick=death_crown_damage,
                    knockback_mtile=int(death.get("knockback_mtile") or 0),
                    allowed_targets=tuple(str(item) for item in death.get("targets", ())),
                    max_pulses=1,
                    level_multiplier_permille=entity.level_multiplier_permille,
                )
                state.effects[effect.uid] = effect
                self._emit(
                    state, "death_effect_scheduled", uid=effect.uid,
                    source_uid=entity.uid, card_id=entity.card_id, delay_us=delay_us,
                )
            else:
                self._impact_area(
                    state,
                    owner=entity.owner,
                    source_uid=entity.uid,
                    source_card_id=entity.card_id,
                    x=entity.x_mtile,
                    y=entity.y_mtile,
                    damage=death_damage,
                    crown_damage=death_crown_damage,
                    radius=int(death.get("radius_mtile") or 0),
                    status=death.get("status"),
                    knockback=int(death.get("knockback_mtile") or 0),
                    primary_target_uid=None,
                    allowed_targets=tuple(str(item) for item in death.get("targets", ())),
                )
            spawn_card_id = death.get("spawn_card_id")
            if spawn_card_id is not None:
                child = self.ruleset.card(str(spawn_card_id))
                count = int(death.get("spawn_count") or 1)
                authored_offsets = death.get("spawn_offsets_mtile")
                if authored_offsets is not None:
                    offsets = tuple(
                        (int(pair[0]), int(pair[1])) for pair in authored_offsets
                    )
                    if len(offsets) != count:
                        raise RulesetError(
                            f"{entity.card_id}.mechanics.death.spawn_offsets_mtile "
                            "must contain exactly spawn_count entries"
                        )
                else:
                    offsets = self._death_spawn_offsets(count)
                if entity.owner == 1 and definition.mechanics.get("mirror_spawn_layout"):
                    offsets = tuple((-x, -y) for x, y in offsets)
                for offset in offsets:
                    self._spawn_single_child(state, entity, child, offset_mtile=offset)
                self._emit(
                    state,
                    "death_spawn",
                    parent_uid=entity.uid,
                    parent_card_id=entity.card_id,
                    child_card_id=child.card_id,
                    child_count=count,
                    owner=entity.owner,
                )
            for child_spec in death.get("spawn_children", ()):
                child_id = str(child_spec["card_id"])
                child = self.ruleset.card(child_id)
                count = int(child_spec["count"])
                # Carrier children are created at deployment and remain attached
                # until the parent dies.  The legacy death component is retained
                # as a fallback for hand-built/old serialized states, but must not
                # duplicate already materialized children in normal play.
                carrier = definition.mechanics.get("carrier")
                has_materialized_carrier_children = bool(
                    carrier
                    and str(carrier.get("child_card_id")) == child_id
                    and any(
                        candidate.carried_by_uid == entity.uid
                        for candidate in state.entities.values()
                    )
                )
                if has_materialized_carrier_children:
                    continue
                authored_offsets = child_spec.get("offsets_mtile")
                if authored_offsets is not None:
                    offsets = tuple(
                        (int(pair[0]), int(pair[1])) for pair in authored_offsets
                    )
                    if len(offsets) != count:
                        raise RulesetError(
                            f"{entity.card_id}.mechanics.death.spawn_children "
                            "offsets_mtile must contain exactly count entries"
                        )
                else:
                    offsets = self._death_spawn_offsets(count)
                if entity.owner == 1 and definition.mechanics.get("mirror_spawn_layout"):
                    offsets = tuple((-x, -y) for x, y in offsets)
                for offset in offsets:
                    self._spawn_single_child(state, entity, child, offset_mtile=offset)
                self._emit(
                    state,
                    "death_spawn",
                    parent_uid=entity.uid,
                    parent_card_id=entity.card_id,
                    child_card_id=child.card_id,
                    child_count=count,
                    owner=entity.owner,
                )
        death_rage = definition.mechanics.get("death_rage")
        if death_rage is not None:
            self._create_area_effect(
                state,
                owner=entity.owner,
                source_uid=entity.uid,
                source_card_id=entity.card_id,
                x_mtile=entity.x_mtile,
                y_mtile=entity.y_mtile,
                default_radius=int(death_rage.get("radius_mtile") or 0),
                default_damage=0,
                default_crown_damage=0,
                default_status=None,
                default_knockback=0,
                raw_effect={
                    "duration_us": int(death_rage["duration_us"]),
                    "duration_anchor": "creation",
                    "tick_interval_us": int(death_rage["tick_interval_us"]),
                    "radius_mtile": int(death_rage["radius_mtile"]),
                    "damage_per_tick": 0,
                    "crown_damage_per_tick": 0,
                    "targets": ["air", "ground", "building", "crown_tower"],
                    "friendly_status": {
                        "kind": "rage",
                        "duration_us": int(death_rage["duration_us"]),
                        "speed_multiplier_milli": int(death_rage["speed_multiplier_milli"]),
                        "hit_speed_multiplier_milli": int(death_rage["hit_speed_multiplier_milli"]),
                        "linger_us": 0,
                    },
                    "friendly_targets": list(death_rage["targets"]),
                    "damage_schedule": [0],
                    "crown_damage_schedule": [0],
                },
            )
            self._emit(
                state,
                "death_rage_created",
                uid=entity.uid,
                card_id=entity.card_id,
            )

    def _apply_revive(self, state: BattleState, entity: EntityState) -> None:
        """Spawn a Phoenix egg or hatch one whose timer completed."""

        definition = self.ruleset.cards[entity.card_id]
        revive = definition.mechanics.get("revive")
        if revive is not None and entity.revive_eligible:
            egg_id = str(revive.get("egg_card_id"))
            egg = self.ruleset.card(egg_id)
            self._spawn_single_at(
                state,
                egg,
                owner=entity.owner,
                x_mtile=entity.x_mtile,
                y_mtile=entity.y_mtile,
                parent_uid=entity.uid,
                event_kind="phoenix_egg_created",
                revive_eligible=False,
                level_multiplier_permille=entity.level_multiplier_permille,
            )
            self._emit(
                state,
                "phoenix_death_rebirth_started",
                uid=entity.uid,
                card_id=entity.card_id,
                egg_card_id=egg_id,
            )
            return
        egg_component = definition.mechanics.get("revive_egg")
        if egg_component is None or not entity.hatch_due:
            return
        hatch_card_id = str(egg_component.get("hatch_card_id"))
        phoenix = self.ruleset.card(hatch_card_id)
        # The egg's card definition carries the fixed Level-11 hatch values;
        # use the parent component only for the source body identity.
        source = self.ruleset.cards.get(hatch_card_id)
        revive_values = source.mechanics.get("revive") if source is not None else None
        if revive_values is None:
            raise ValueError(f"{entity.card_id}: hatch card lacks revive component")
        revived_hitpoints = self._scale_level_value(
            int(revive_values["revived_hitpoints"]),
            entity.level_multiplier_permille,
        )
        revived = self._spawn_single_at(
            state,
            phoenix,
            owner=entity.owner,
            x_mtile=entity.x_mtile,
            y_mtile=entity.y_mtile,
            parent_uid=entity.uid,
            event_kind="phoenix_reborn",
            hp_override=revived_hitpoints,
            max_hp_override=revived_hitpoints,
            revive_eligible=False,
            level_multiplier_permille=entity.level_multiplier_permille,
        )
        self._emit(
            state,
            "phoenix_egg_hatched",
            uid=entity.uid,
            card_id=entity.card_id,
            revived_uid=revived.uid,
            hitpoints=revived.hp,
            damage=self._scale_level_value(
                int(revive_values["revived_damage"]),
                entity.level_multiplier_permille,
            ),
        )

    def _resolve_tower_outcomes(self, state: BattleState, destroyed: list[EntityState]) -> None:
        if not destroyed or state.terminal:
            return
        king_deaths = {tower.owner for tower in destroyed if tower.role == "king"}
        for tower in destroyed:
            opponent = state.players[1 - tower.owner]
            if tower.role == "king":
                opponent.crowns = 3
            else:
                opponent.crowns += 1
                self._activate_king(state, tower.owner, "princess_tower_destroyed")
        if len(king_deaths) == 2:
            self._end_match(state, None, "simultaneous_king_destruction")
        elif king_deaths:
            self._end_match(state, 1 - next(iter(king_deaths)), "king_tower_destroyed")
        elif state.phase == "overtime":
            crowns = (state.players[0].crowns, state.players[1].crowns)
            if crowns[0] != crowns[1]:
                self._end_match(state, 0 if crowns[0] > crowns[1] else 1, "overtime_sudden_death")

    def _advance_match_clock(self, state: BattleState) -> None:
        if state.terminal:
            return
        state.elapsed_us += self.ruleset.tick_us
        regulation = self.ruleset.match.regulation_us
        total = regulation + self.ruleset.match.overtime_us
        if state.phase == "regulation" and state.elapsed_us >= regulation:
            crowns = (state.players[0].crowns, state.players[1].crowns)
            if crowns[0] != crowns[1]:
                self._end_match(state, 0 if crowns[0] > crowns[1] else 1, "regulation_crowns")
                return
            state.phase = "overtime"
            self._emit(state, "match_phase_changed", phase="overtime")
        if state.elapsed_us >= total:
            if self.ruleset.match.tiebreak_enabled:
                self._resolve_tiebreak(state)
            else:
                self._end_match(state, None, "time_draw")

    def _resolve_tiebreak(self, state: BattleState) -> None:
        alive_towers = [entity for entity in self._alive_entities(state) if entity.kind == "tower"]
        if not alive_towers:
            self._end_match(state, None, "tiebreak_draw")
            return
        # Tiebreak drains the same raw HP from every surviving Crown Tower;
        # therefore the lowest absolute current HP falls first.
        lowest_hp = min(tower.hp for tower in alive_towers)
        minimum = [tower for tower in alive_towers if tower.hp == lowest_hp]
        owners = {tower.owner for tower in minimum}
        if len(owners) != 1:
            self._end_match(state, None, "tiebreak_equal_lowest_hp")
        else:
            self._end_match(state, 1 - next(iter(owners)), "tiebreak_lowest_hp")

    def _activate_king(self, state: BattleState, player: int, reason: str) -> None:
        if state.players[player].king_active:
            return
        state.players[player].king_active = True
        self._emit(state, "king_activated", player=player, reason=reason)

    def _end_match(self, state: BattleState, winner: int | None, reason: str) -> None:
        if state.terminal:
            return
        state.terminal = True
        state.phase = "ended"
        state.winner = winner
        state.terminal_reason = reason
        self._emit(
            state,
            "match_ended",
            winner=winner,
            reason=reason,
            crowns_0=state.players[0].crowns,
            crowns_1=state.players[1].crowns,
        )

    def _definition(self, entity: EntityState) -> CardDefinition | TowerDefinition:
        if entity.kind == "tower":
            return self.ruleset.towers[entity.card_id]
        return self.ruleset.cards[entity.card_id]

    def _projectile_definition(self, projectile: ProjectileState):
        if projectile.source_card_id in self.ruleset.cards:
            definition = self.ruleset.cards[projectile.source_card_id]
        else:
            definition = self.ruleset.towers[projectile.source_card_id]
        if definition.projectile is None:
            raise ValueError(f"{projectile.source_card_id} has no projectile definition")
        return definition.projectile

    def _in_attack_range(self, source: EntityState, target: EntityState) -> bool:
        definition = self._definition(source)
        range_mtile = definition.range_mtile
        hook = definition.mechanics.get("hook") if source.kind != "tower" else None
        if hook is not None:
            range_mtile = int(hook.get("hook_range_mtile") or range_mtile or 0)
        if source.charge_active and definition.mechanics.get("charge_range_mtile") is not None:
            range_mtile = int(definition.mechanics["charge_range_mtile"])
        if range_mtile is None:
            return False
        distance = self._edge_distance(source, target)
        minimum = (
            0
            if source.kind == "tower"
            else int(definition.mechanics.get("min_attack_range_mtile") or 0)
        )
        return minimum <= distance <= int(range_mtile)

    def _sight_range(self, entity: EntityState) -> int:
        definition = self._definition(entity)
        sight = int(definition.sight_range_mtile or 0)
        if entity.kind != "tower":
            hook = definition.mechanics.get("hook")
            if hook is not None:
                sight = max(sight, int(hook.get("hook_range_mtile") or 0))
        return sight

    def _reset_attack_charge(
        self,
        state: BattleState,
        entity: EntityState,
        *,
        reason: str,
    ) -> None:
        """Clear a generic movement-charge run and record the reason.

        ``charge_active`` is deliberately not touched: it belongs to
        threshold/fuse mechanics such as Goblin Demolisher.  Generic charge
        attacks reset on retarget, hard crowd-control, knockback, and a
        consumed hit.  Emitting resets makes truth-mining able to distinguish
        a missed charge from a normal walk without inspecting hidden state.
        """

        if not entity.attack_charge_active and entity.attack_charge_distance_mtile <= 0:
            return
        was_active = entity.attack_charge_active
        distance = entity.attack_charge_distance_mtile
        entity.attack_charge_active = False
        entity.attack_charge_distance_mtile = 0
        self._emit(
            state,
            "charge_reset",
            uid=entity.uid,
            card_id=entity.card_id,
            reason=reason,
            was_active=was_active,
            distance_mtile=distance,
        )

    def _reset_dash(self, state: BattleState, entity: EntityState, *, reason: str) -> None:
        """Cancel a pending Bandit-style dash impact."""

        if not entity.dash_attack_active:
            return
        entity.dash_attack_active = False
        self._emit(
            state,
            "dash_reset",
            uid=entity.uid,
            card_id=entity.card_id,
            reason=reason,
        )

    def _ramp_component(self, entity: EntityState):
        if entity.kind == "tower":
            return None
        return self.ruleset.cards[entity.card_id].mechanics.get("ramp_attack")

    def _reset_attack_ramp(self, state: BattleState, entity: EntityState, *, reason: str) -> None:
        """Reset an Inferno beam's elapsed lock time and stage."""

        if entity.ramp_elapsed_us == 0 and entity.ramp_stage == 0:
            return
        previous_stage = entity.ramp_stage
        previous_elapsed = entity.ramp_elapsed_us
        entity.ramp_elapsed_us = 0
        entity.ramp_stage = 0
        self._emit(
            state,
            "ramp_reset",
            uid=entity.uid,
            card_id=entity.card_id,
            reason=reason,
            previous_stage=previous_stage,
            elapsed_us=previous_elapsed,
        )

    def _edge_distance(self, source: EntityState, target: EntityState) -> int:
        center = distance_mtile(source.x_mtile, source.y_mtile, target.x_mtile, target.y_mtile)
        return max(0, center - self._collision_radius(source) - self._collision_radius(target))

    def _collision_radius(self, entity: EntityState) -> int:
        return int(self._definition(entity).collision_radius_mtile or 0)

    def _mass(self, entity: EntityState) -> int:
        if entity.kind == "tower":
            return 1_000_000
        return max(1, int(self.ruleset.cards[entity.card_id].mass or 1))

    def _movement_layer(self, entity: EntityState) -> str:
        """Return the entity's physics navigation layer."""

        if entity.kind == "tower":
            return "ground"
        definition = self.ruleset.cards.get(entity.card_id)
        if definition is None:
            return "ground"
        if entity.river_airborne_active:
            return "air"
        return str(definition.mechanics.get("movement_layer") or "ground")

    @staticmethod
    def _is_frozen(entity: EntityState) -> bool:
        return any(
            status.kind in {"freeze", "stun"} and status.remaining_us > 0
            for status in entity.statuses
        )

    @staticmethod
    def _speed_multiplier(entity: EntityState) -> int:
        slow = [
            status.magnitude_permille
            for status in entity.statuses
            if status.kind in {"slow", "freeze"}
        ]
        rage = [status.magnitude_permille for status in entity.statuses if status.kind == "rage"]
        result = min(slow, default=PERMILLE)
        if rage:
            result = result * max(rage) // PERMILLE
        return result

    @staticmethod
    def _scale_level_value(value: int, multiplier_permille: int) -> int:
        """Scale an integer stat using the deterministic Clash level step."""

        if value <= 0:
            return value
        return max(1, (value * multiplier_permille + 500) // PERMILLE)

    @staticmethod
    def _hit_speed_multiplier(entity: EntityState) -> int:
        slow = [
            (
                status.hit_speed_magnitude_permille
                if status.hit_speed_magnitude_permille is not None
                else status.magnitude_permille
            )
            for status in entity.statuses
            if status.kind in {"slow", "freeze"}
        ]
        rage = [status.magnitude_permille for status in entity.statuses if status.kind == "rage"]
        result = min(slow, default=PERMILLE)
        if rage:
            result = result * max(rage) // PERMILLE
        return result

    @staticmethod
    def _spawn_time_progress(entity: EntityState, dt: int) -> int:
        """Advance a spawner clock under Rage without floating-point drift."""

        multiplier = BattleEngine._hit_speed_multiplier(entity)
        numerator = dt * multiplier + entity.spawn_time_remainder
        progress, entity.spawn_time_remainder = divmod(numerator, PERMILLE)
        return progress

    @staticmethod
    def _alive_entities(state: BattleState) -> list[EntityState]:
        return [
            state.entities[uid]
            for uid in sorted(state.entities)
            if state.entities[uid].alive and state.entities[uid].hp > 0
        ]

    @staticmethod
    def _towers_for(state: BattleState, player: int) -> list[EntityState]:
        return [
            state.entities[uid]
            for uid in sorted(state.entities)
            if state.entities[uid].kind == "tower" and state.entities[uid].owner == player
        ]

    def _tower(self, state: BattleState, player: int, role: str) -> EntityState:
        return next(tower for tower in self._towers_for(state, player) if tower.role == role)

    @staticmethod
    def _allocate_uid(state: BattleState) -> int:
        uid = state.next_uid
        state.next_uid += 1
        return uid

    @staticmethod
    def _emit(state: BattleState, kind: str, **data: str | int | bool | None) -> None:
        event = SimEvent.create(state.tick, state.event_sequence, kind, **data)
        state.event_sequence += 1
        state.events.append(event)

    def _verify_state_ruleset(self, state: BattleState) -> None:
        if state.engine_version != ENGINE_VERSION:
            raise ValueError(
                f"battle state engine version {state.engine_version!r} does not match {ENGINE_VERSION!r}"
            )
        if state.ruleset_id != self.ruleset.ruleset_id or state.ruleset_hash != self.ruleset.content_hash:
            raise ValueError("battle state ruleset ID/hash does not match engine")

    def validate_state(self, state: BattleState) -> None:
        self._verify_state_ruleset(state)
        if type(state.schema_version) is not int or state.schema_version != 1:
            raise ValueError("unsupported battle-state schema version")
        if type(state.seed) is not int:
            raise ValueError("battle seed must be an integer")
        if type(state.rng_state) is not int or not (0 <= state.rng_state < 1 << 64):
            raise ValueError("rng_state must be an unsigned 64-bit integer")
        if type(state.tick) is not int or state.tick < 0:
            raise ValueError("battle tick must be a non-negative integer")
        if type(state.elapsed_us) is not int or state.elapsed_us < 0:
            raise ValueError("elapsed_us must be a non-negative integer")
        if state.phase not in {"regulation", "overtime", "ended"}:
            raise ValueError("invalid battle phase")
        if type(state.terminal) is not bool:
            raise ValueError("terminal must be boolean")
        if state.winner is not None and (type(state.winner) is not int or state.winner not in (0, 1)):
            raise ValueError("winner must be player 0, player 1, or None")
        if state.terminal != (state.phase == "ended"):
            raise ValueError("terminal flag and ended phase disagree")
        if state.terminal_reason is not None and not isinstance(state.terminal_reason, str):
            raise ValueError("terminal_reason must be a string or None")
        if not state.terminal and (state.winner is not None or state.terminal_reason is not None):
            raise ValueError("non-terminal state carries a terminal outcome")
        if len(state.players) != 2:
            raise ValueError("battle state must have two players")
        if type(state.next_uid) is not int or state.next_uid <= 0:
            raise ValueError("next_uid must be a positive integer")
        if type(state.navigation_revision) is not int or state.navigation_revision < 0:
            raise ValueError("navigation_revision must be a non-negative integer")
        known_uids = set(state.entities)
        if len(known_uids) != len(state.entities):
            raise ValueError("duplicate entity UID")
        projectile_uids = set(state.projectiles)
        if known_uids & projectile_uids:
            raise ValueError("entity and projectile UIDs must be globally disjoint")
        effect_uids = set(state.effects)
        if (known_uids | projectile_uids) & effect_uids:
            raise ValueError("entity, projectile, and effect UIDs must be globally disjoint")
        all_uids = known_uids | projectile_uids | effect_uids
        if state.next_uid <= max(all_uids, default=0):
            raise ValueError("next_uid must be greater than every allocated UID")
        for player in state.players:
            if type(player.elixir_milli) is not int or not (
                0 <= player.elixir_milli <= self.ruleset.match.max_elixir_milli
            ):
                raise ValueError("elixir outside ruleset bounds")
            if type(player.elixir_remainder) is not int or player.elixir_remainder < 0:
                raise ValueError("elixir remainder must be a non-negative integer")
            if len(player.hand) != self.ruleset.match.hand_size:
                raise ValueError("invalid hand size")
            if type(player.crowns) is not int or not (0 <= player.crowns <= 3):
                raise ValueError("invalid crown count")
            if type(player.king_active) is not bool:
                raise ValueError("king_active must be boolean")
            if type(player.cards_played) is not int or player.cards_played < 0:
                raise ValueError("cards_played must be a non-negative integer")
            if sorted(player.hand + player.draw_pile) != sorted(player.deck):
                raise ValueError("hand/draw cycle does not contain exactly the deck")
            try:
                resolved_deck = [self.ruleset.resolve_card_id(card) for card in player.deck]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("player deck contains an unknown card") from error
            if tuple(resolved_deck) != player.deck:
                raise ValueError("authoritative deck IDs must be canonical")
            self._validate_deck(resolved_deck)
            try:
                if any(self.ruleset.resolve_card_id(card) != card for card in player.hand + player.draw_pile):
                    raise ValueError("hand/draw IDs must be canonical")
            except (KeyError, TypeError) as error:
                raise ValueError("hand/draw cycle contains an unknown card") from error
            try:
                if any(self.ruleset.resolve_card_id(card) != card for card in player.seen_enemy_cards):
                    raise ValueError("seen enemy card IDs must be canonical")
            except (KeyError, TypeError) as error:
                raise ValueError("seen enemy cards contain an unknown card") from error
            if player.last_played_card_id is not None:
                try:
                    if self.ruleset.resolve_card_id(player.last_played_card_id) != player.last_played_card_id:
                        raise ValueError("last played card ID must be canonical")
                except (KeyError, TypeError) as error:
                    raise ValueError("last played card ID is unknown") from error
        for uid, entity in state.entities.items():
            if type(uid) is not int or uid <= 0 or type(entity.uid) is not int:
                raise ValueError("entity UID must be a positive integer")
            if entity.uid != uid:
                raise ValueError("entity dictionary key/UID mismatch")
            if type(entity.owner) is not int or entity.owner not in (0, 1):
                raise ValueError("entity has invalid owner")
            integer_fields = (
                entity.x_mtile,
                entity.y_mtile,
                entity.hp,
                entity.max_hp,
                entity.spawn_tick,
                entity.deploy_remaining_us,
                entity.attack_cooldown_us,
                entity.windup_remaining_us,
                entity.secondary_attack_cooldown_us,
                entity.secondary_windup_remaining_us,
                entity.secondary_attack_time_remainder,
                entity.secondary_attack_count,
                entity.lifetime_decay_remainder,
                entity.spawn_cooldown_us,
                entity.spawn_time_remainder,
                entity.spawned_count,
                entity.movement_remainder,
                entity.attack_time_remainder,
                entity.attack_count,
                entity.navigation_revision,
                entity.navigation_goal_x_mtile,
                entity.navigation_goal_y_mtile,
                entity.navigation_cursor,
                entity.attack_charge_distance_mtile,
                entity.ramp_elapsed_us,
                entity.ramp_stage,
                entity.carried_offset_x_mtile,
                entity.carried_offset_y_mtile,
                entity.shield_hp,
                entity.shield_max_hp,
                entity.stealth_remaining_us,
                entity.jump_remaining_us,
                entity.jump_landing_x_mtile,
                entity.jump_landing_y_mtile,
            )
            if any(type(value) is not int for value in integer_fields):
                raise ValueError("entity fixed-point fields must be integers")
            if entity.lifetime_remaining_us is not None and type(entity.lifetime_remaining_us) is not int:
                raise ValueError("entity lifetime must be an integer or None")
            if entity.charge_remaining_us is not None and type(entity.charge_remaining_us) is not int:
                raise ValueError("entity charge lifetime must be an integer or None")
            if entity.target_uid is not None and type(entity.target_uid) is not int:
                raise ValueError("entity target UID must be an integer or None")
            if entity.pending_target_uid is not None and type(entity.pending_target_uid) is not int:
                raise ValueError("pending target UID must be an integer or None")
            if entity.secondary_pending_target_uid is not None and type(entity.secondary_pending_target_uid) is not int:
                raise ValueError("secondary pending target UID must be an integer or None")
            if entity.navigation_target_uid is not None and type(entity.navigation_target_uid) is not int:
                raise ValueError("navigation target UID must be an integer or None")
            if entity.navigation_target_uid is not None and entity.navigation_target_uid not in known_uids:
                raise ValueError("dangling navigation target")
            if entity.carried_by_uid is not None:
                if type(entity.carried_by_uid) is not int:
                    raise ValueError("carried_by_uid must be an integer or None")
                if entity.carried_by_uid not in known_uids:
                    raise ValueError("dangling carrier UID")
                if entity.carried_by_uid == entity.uid:
                    raise ValueError("entity cannot carry itself")
                carrier = state.entities[entity.carried_by_uid]
                if carrier.owner != entity.owner or not carrier.alive:
                    raise ValueError("carried entity has an invalid carrier")
                if carrier.kind == "tower":
                    raise ValueError("tower cannot carry an entity")
                carrier_definition = self.ruleset.cards[carrier.card_id]
                carrier_component = carrier_definition.mechanics.get("carrier")
                if not carrier_component or str(carrier_component.get("child_card_id")) != entity.card_id:
                    raise ValueError("entity is attached to a non-matching carrier")
            if not 0 <= entity.navigation_cursor <= len(entity.navigation_waypoints):
                raise ValueError("navigation cursor outside waypoint list")
            if any(
                not isinstance(point, (tuple, list))
                or len(point) != 2
                or any(type(value) is not int for value in point)
                for point in entity.navigation_waypoints
            ):
                raise ValueError("navigation waypoints must be integer coordinate pairs")
            if (
                type(entity.alive) is not bool
                or type(entity.death_effect_done) is not bool
                or type(entity.charge_active) is not bool
                or type(entity.attack_charge_active) is not bool
                or type(entity.dash_attack_active) is not bool
                or type(entity.revive_eligible) is not bool
                or type(entity.hatch_due) is not bool
                or type(entity.is_clone) is not bool
                or type(entity.stealth_active) is not bool
                or type(entity.spawner_active) is not bool
                or type(entity.concealed_active) is not bool
                or type(entity.river_airborne_active) is not bool
                or type(entity.burrow_active) is not bool
            ):
                raise ValueError("entity lifecycle flags must be boolean")
            try:
                if entity.kind == "tower":
                    self.ruleset.tower(entity.card_id)
                else:
                    self.ruleset.card(entity.card_id)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("entity references an unknown definition") from error
            if entity.target_uid is not None and entity.target_uid not in known_uids:
                raise ValueError("dangling entity target")
            if entity.pending_target_uid is not None and entity.pending_target_uid not in known_uids:
                raise ValueError("dangling pending attack target")
            if entity.secondary_pending_target_uid is not None and entity.secondary_pending_target_uid not in known_uids:
                raise ValueError("dangling secondary pending attack target")
            if entity.jump_target_uid is not None:
                if type(entity.jump_target_uid) is not int:
                    raise ValueError("jump target UID must be an integer or None")
                if entity.jump_target_uid not in known_uids:
                    raise ValueError("dangling jump target")
            if entity.alive and not (0 < entity.hp <= entity.max_hp):
                raise ValueError("living entity has invalid HP")
            if not entity.alive and entity.hp != 0:
                raise ValueError("dead entity must have zero HP")
            if entity.deploy_remaining_us < 0 or entity.attack_cooldown_us < 0 or entity.windup_remaining_us < 0:
                raise ValueError("entity clock cannot be negative")
            if (
                entity.shield_hp < 0
                or entity.shield_max_hp < 0
                or entity.shield_hp > entity.shield_max_hp
                or entity.stealth_remaining_us < 0
                or entity.jump_remaining_us < 0
                or type(entity.level_multiplier_permille) is not int
                or entity.level_multiplier_permille <= 0
            ):
                raise ValueError("entity special-mechanic state is outside bounds")
            if entity.shield_max_hp == 0 and entity.shield_hp != 0:
                raise ValueError("entity carries shield HP without a shield definition")
            if (
                entity.secondary_attack_cooldown_us < 0
                or entity.secondary_windup_remaining_us < 0
                or entity.secondary_attack_time_remainder < 0
                or entity.secondary_attack_count < 0
            ):
                raise ValueError("secondary entity clock cannot be negative")
            if entity.lifetime_remaining_us is not None and entity.lifetime_remaining_us < 0:
                raise ValueError("entity lifetime cannot be negative")
            if entity.charge_remaining_us is not None and entity.charge_remaining_us < 0:
                raise ValueError("entity charge lifetime cannot be negative")
            if entity.attack_charge_distance_mtile < 0:
                raise ValueError("entity attack charge distance cannot be negative")
            if entity.ramp_elapsed_us < 0 or entity.ramp_stage < 0:
                raise ValueError("entity ramp state cannot be negative")
            ramp = self._ramp_component(entity)
            if ramp is None and (entity.ramp_elapsed_us or entity.ramp_stage):
                raise ValueError("non-ramp entity carries ramp state")
            if ramp is not None:
                schedule = ramp.get("damage_schedule", ())
                if entity.ramp_stage >= len(schedule):
                    raise ValueError("entity ramp stage exceeds its damage schedule")
            if entity.hatch_due and self.ruleset.cards[entity.card_id].mechanics.get("revive_egg") is None:
                raise ValueError("non-egg entity carries hatch_due state")
            if entity.lifetime_decay_remainder < 0:
                raise ValueError("entity lifetime decay remainder cannot be negative")
            if entity.spawn_cooldown_us < 0 or entity.spawned_count < 0:
                raise ValueError("entity spawner counters cannot be negative")
            if entity.spawn_time_remainder < 0:
                raise ValueError("entity spawner time remainder cannot be negative")
            if any(
                not isinstance(status.kind, str)
                or type(status.remaining_us) is not int
                or status.remaining_us <= 0
                or type(status.magnitude_permille) is not int
                or not (
                    0 <= status.magnitude_permille
                    <= (2_000 if status.kind == "rage" else PERMILLE)
                )
                or (
                    status.hit_speed_magnitude_permille is not None
                    and (
                        type(status.hit_speed_magnitude_permille) is not int
                        or not (
                            0 <= status.hit_speed_magnitude_permille
                            <= (2_000 if status.kind == "rage" else PERMILLE)
                        )
                    )
                )
                or type(status.damage_per_tick) is not int
                or status.damage_per_tick < 0
                or type(status.tick_interval_us) is not int
                or status.tick_interval_us < 0
                or type(status.tick_remainder_us) is not int
                or status.tick_remainder_us < 0
                or type(status.on_death_spawn_count) is not int
                or status.on_death_spawn_count < 0
                or type(status.source_level_multiplier_permille) is not int
                or status.source_level_multiplier_permille <= 0
                or (
                    status.on_death_spawn_card_id is not None
                    and (
                        not isinstance(status.on_death_spawn_card_id, str)
                        or status.on_death_spawn_count <= 0
                        or type(status.on_death_spawn_owner) is not int
                        or status.on_death_spawn_owner not in (0, 1)
                    )
                )
                or (
                    status.on_death_spawn_card_id is None
                    and (
                        status.on_death_spawn_count != 0
                        or status.on_death_spawn_owner is not None
                    )
                )
                for status in entity.statuses
            ):
                raise ValueError("expired status retained in authoritative state")
            if not (0 <= entity.x_mtile < self.ruleset.arena.width_mtile):
                raise ValueError("entity x coordinate outside arena")
            if not (0 <= entity.y_mtile < self.ruleset.arena.height_mtile):
                raise ValueError("entity y coordinate outside arena")
            if position_to_cell(entity.x_mtile, entity.y_mtile) is None:
                raise ValueError("entity does not map to policy grid")
        living_structures = [
            entity
            for entity in state.entities.values()
            if entity.alive and entity.kind in {"building", "tower"}
        ]
        for troop in (
            entity
            for entity in state.entities.values()
            if entity.alive and entity.kind == "troop" and entity.deploy_remaining_us <= 0
        ):
            # Air units occupy a separate navigation/collision layer.  They
            # may pass over towers, buildings, terrain, and ground troops;
            # applying the ground overlap invariant to them would reject
            # perfectly valid states (and would make strict validation depend
            # on where a flying unit happens to be above the arena).
            if troop.carried_by_uid is not None or self._movement_layer(troop) == "air":
                continue
            for structure in living_structures:
                minimum = self._collision_radius(troop) + self._collision_radius(structure)
                if distance_mtile(
                    troop.x_mtile,
                    troop.y_mtile,
                    structure.x_mtile,
                    structure.y_mtile,
                ) < minimum:
                    raise ValueError("active troop overlaps a living structure")
        for uid, projectile in state.projectiles.items():
            if type(uid) is not int or uid <= 0 or type(projectile.uid) is not int:
                raise ValueError("projectile UID must be a positive integer")
            if projectile.uid != uid:
                raise ValueError("projectile dictionary key/UID mismatch")
            projectile_integer_fields = (
                projectile.x_mtile,
                projectile.y_mtile,
                projectile.target_x_mtile,
                projectile.target_y_mtile,
                projectile.damage,
                projectile.crown_damage,
                projectile.speed_mtile_per_s,
                projectile.radius_mtile,
                projectile.status_duration_us,
                projectile.status_magnitude_permille,
                projectile.status_hit_speed_magnitude_permille,
                projectile.status_damage_per_tick,
                projectile.status_tick_interval_us,
                projectile.knockback_mtile,
                projectile.movement_remainder,
                projectile.origin_x_mtile,
                projectile.origin_y_mtile,
                projectile.line_end_x_mtile,
                projectile.line_end_y_mtile,
                projectile.direction_x_mtile,
                projectile.direction_y_mtile,
                projectile.pellet_index,
                projectile.chain_next_index,
                projectile.chain_delay_us,
                projectile.chain_delay_remaining_us,
                projectile.level_multiplier_permille,
            )
            if any(type(value) is not int for value in projectile_integer_fields):
                raise ValueError("projectile fixed-point fields must be integers")
            if any(type(value) is not int or value <= 0 for value in projectile.chain_target_uids):
                raise ValueError("projectile chain targets must be positive integer UIDs")
            if (
                projectile.status_duration_us < 0
                or projectile.status_damage_per_tick < 0
                or projectile.status_tick_interval_us < 0
                or not 0 <= projectile.status_magnitude_permille <= PERMILLE
                or not 0 <= projectile.status_hit_speed_magnitude_permille <= PERMILLE
                or projectile.level_multiplier_permille <= 0
            ):
                raise ValueError("projectile status fields are outside bounds")
            for reference in (projectile.source_uid, projectile.target_uid):
                if reference is not None and type(reference) is not int:
                    raise ValueError("projectile references must be integers or None")
            if (
                type(projectile.alive) is not bool
                or type(projectile.piercing) is not bool
                or type(projectile.homing) is not bool
                or type(projectile.returning) is not bool
                or type(projectile.return_phase) is not bool
            ):
                raise ValueError("projectile flags must be boolean")
            if any(
                not isinstance(value, str)
                or value not in {"air", "ground", "building", "crown_tower"}
                for value in projectile.allowed_targets
            ):
                raise ValueError("projectile allowed target classes are invalid")
            if any(type(hit_uid) is not int for hit_uid in projectile.hit_uids):
                raise ValueError("projectile hit UIDs must be integers")
            if len(projectile.hit_uids) != len(set(projectile.hit_uids)):
                raise ValueError("projectile hit UIDs must be unique")
            if projectile.target_uid is not None and projectile.target_uid not in known_uids:
                raise ValueError("dangling projectile target")
            if projectile.owner not in (0, 1):
                raise ValueError("projectile has invalid owner")
            if projectile.source_uid is not None and projectile.source_uid not in known_uids:
                raise ValueError("dangling projectile source")
            if not (0 <= projectile.x_mtile < self.ruleset.arena.width_mtile):
                raise ValueError("projectile x coordinate outside arena")
            if not (0 <= projectile.y_mtile < self.ruleset.arena.height_mtile):
                raise ValueError("projectile y coordinate outside arena")
            if not (0 <= projectile.target_x_mtile < self.ruleset.arena.width_mtile):
                raise ValueError("projectile target x outside arena")
            if not (0 <= projectile.target_y_mtile < self.ruleset.arena.height_mtile):
                raise ValueError("projectile target y outside arena")
        for uid, effect in state.effects.items():
            if type(uid) is not int or uid <= 0 or type(effect.uid) is not int:
                raise ValueError("effect UID must be a positive integer")
            if effect.uid != uid:
                raise ValueError("effect dictionary key/UID mismatch")
            if effect.owner not in (0, 1):
                raise ValueError("effect has invalid owner")
            integer_fields = (
                effect.x_mtile,
                effect.y_mtile,
                effect.radius_mtile,
                effect.remaining_us,
                effect.tick_interval_us,
                effect.tick_remainder_us,
                effect.initial_delay_remaining_us,
                effect.damage_per_tick,
                effect.crown_damage_per_tick,
                effect.status_duration_us,
                effect.status_magnitude_permille,
                effect.status_hit_speed_magnitude_permille,
                effect.status_damage_per_tick,
                effect.status_tick_interval_us,
                effect.knockback_mtile,
                effect.pull_to_center_mtile,
                effect.friendly_status_duration_us,
                effect.friendly_status_magnitude_permille,
                effect.friendly_status_linger_us,
                effect.status_on_death_spawn_count,
                effect.spawn_count,
                effect.max_spawns,
                effect.spawned_count,
                effect.pulses_applied,
                effect.level_multiplier_permille,
            )
            if any(type(value) is not int for value in integer_fields):
                raise ValueError("effect fixed-point fields must be integers")
            if (
                effect.radius_mtile < 0
                or effect.remaining_us < 0
                or effect.tick_interval_us <= 0
                or effect.tick_remainder_us < 0
                or effect.initial_delay_remaining_us < 0
                or effect.damage_per_tick < 0
                or effect.crown_damage_per_tick < 0
                or effect.status_duration_us < 0
                or not 0 <= effect.status_magnitude_permille <= PERMILLE
                or not 0 <= effect.status_hit_speed_magnitude_permille <= PERMILLE
                or effect.status_damage_per_tick < 0
                or effect.status_tick_interval_us < 0
                or effect.knockback_mtile < 0
                or effect.pull_to_center_mtile < 0
                or effect.friendly_status_duration_us < 0
                or not 0 <= effect.friendly_status_magnitude_permille <= 2_000
                or effect.friendly_status_linger_us < 0
                or effect.status_on_death_spawn_count < 0
                or effect.spawn_count < 0
                or effect.max_spawns < 0
                or effect.spawned_count < 0
                or effect.spawned_count > effect.max_spawns
                or effect.pulses_applied < 0
                or effect.level_multiplier_permille <= 0
                or (
                    effect.max_pulses is not None
                    and (
                        type(effect.max_pulses) is not int
                        or effect.max_pulses <= 0
                        or effect.pulses_applied > effect.max_pulses
                    )
                )
            ):
                raise ValueError("effect fields are outside bounds")
            if type(effect.alive) is not bool:
                raise ValueError("effect lifecycle flag must be boolean")
            for schedule in (effect.damage_schedule, effect.crown_damage_schedule):
                if not isinstance(schedule, tuple) or any(
                    type(value) is not int or value < 0 for value in schedule
                ):
                    raise ValueError("effect damage schedules must be non-negative integer tuples")
            if effect.source_uid is not None and effect.source_uid not in known_uids:
                raise ValueError("dangling effect source")
            if effect.source_card_id not in self.ruleset.cards:
                raise ValueError("effect references an unknown source card")
            if effect.spawn_card_id is not None:
                if effect.spawn_card_id not in self.ruleset.cards:
                    raise ValueError("effect references an unknown spawn card")
                if effect.spawn_count <= 0 or effect.max_spawns <= 0:
                    raise ValueError("effect spawn configuration is incomplete")
            if any(
                not isinstance(target, str) or target not in {"air", "ground", "building", "crown_tower"}
                for target in effect.allowed_targets
            ):
                raise ValueError("effect target classes are invalid")
            if any(
                not isinstance(target, str)
                or target not in {"air", "ground", "building", "crown_tower"}
                for target in effect.friendly_allowed_targets
            ):
                raise ValueError("friendly effect target classes are invalid")
            if effect.friendly_status_kind is not None:
                if not isinstance(effect.friendly_status_kind, str) or not effect.friendly_status_kind:
                    raise ValueError("friendly effect status kind must be a non-empty string")
                if not effect.friendly_allowed_targets:
                    raise ValueError("friendly effect status requires target classes")
            if effect.status_on_death_spawn_card_id is not None:
                if (
                    not isinstance(effect.status_on_death_spawn_card_id, str)
                    or effect.status_on_death_spawn_count <= 0
                    or effect.status_on_death_spawn_card_id not in self.ruleset.cards
                ):
                    raise ValueError("effect death-transform child is invalid")
            elif effect.status_on_death_spawn_count != 0:
                raise ValueError("effect death-transform count lacks a child card")
            if not (0 <= effect.x_mtile < self.ruleset.arena.width_mtile):
                raise ValueError("effect x coordinate outside arena")
            if not (0 <= effect.y_mtile < self.ruleset.arena.height_mtile):
                raise ValueError("effect y coordinate outside arena")
        if type(state.event_sequence) is not int or state.event_sequence < 0:
            raise ValueError("event_sequence must be a non-negative integer")
        if state.events:
            for event in state.events:
                if type(event.tick) is not int or event.tick < 0:
                    raise ValueError("event ticks must be non-negative integers")
                if type(event.sequence) is not int or event.sequence < 0:
                    raise ValueError("event sequences must be non-negative integers")
                if not isinstance(event.kind, str) or not event.kind:
                    raise ValueError("event kinds must be non-empty strings")
                if event.data != tuple(sorted(event.data)) or len(dict(event.data)) != len(event.data):
                    raise ValueError("event data must have unique, sorted keys")
                if any(
                    not isinstance(key, str)
                    or value is not None
                    and not isinstance(value, (str, int, bool))
                    for key, value in event.data
                ):
                    raise ValueError("event data must contain JSON scalar values")
            sequences = [event.sequence for event in state.events]
            if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
                raise ValueError("event sequences must be unique and ordered")
            if sequences[-1] >= state.event_sequence:
                raise ValueError("event_sequence must exceed retained event IDs")
            if any(event.tick > state.tick for event in state.events):
                raise ValueError("event occurs after current battle tick")


class DeterministicCycleController:
    """Simple reproducible smoke-test controller for complete headless matches."""

    def __init__(self, *, lane: str = "alternate") -> None:
        if lane not in {"left", "right", "alternate"}:
            raise ValueError("lane must be left, right, or alternate")
        self.lane = lane

    def choose_action(self, engine: BattleEngine, state: BattleState, player: int) -> SimAction:
        hand = state.players[player].hand
        player_state = state.players[player]
        affordable = [
            slot
            for slot, card_id in enumerate(hand)
            if (
                card_id != "mirror"
                or player_state.last_played_card_id not in {None, "mirror"}
            )
            and engine._effective_card_cost(player_state, engine.ruleset.card(card_id))
            <= player_state.elixir_milli
        ]
        if not affordable:
            return WaitAction(player)
        slot = min(
            affordable,
            key=lambda index: (
                engine._effective_card_cost(player_state, engine.ruleset.card(hand[index])),
                index,
            ),
        )
        card = engine.ruleset.card(hand[slot])
        use_left = self.lane == "left" or (
            self.lane == "alternate" and state.players[player].cards_played % 2 == 0
        )
        col = 3 if use_left else 14
        if card.kind == "spell" and card.card_id == "fireball":
            row = 6 if player == 0 else 25
        elif card.kind == "spell":
            row = 19 if player == 0 else 12
        elif card.kind == "building":
            col, row = 8, (20 if player == 0 else 9)
        else:
            row = 23 if player == 0 else 8
        preferred = (col, row)
        action = PlayCardAction(player, slot, preferred)
        if engine.validate_action(state, action) is None:
            return action
        legal = engine.legal_cells(state, player, card.card_id)
        if not legal:
            return WaitAction(player)
        return PlayCardAction(player, slot, legal[0])
