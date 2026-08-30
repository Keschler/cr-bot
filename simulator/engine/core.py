"""BattleEngine composition root and authoritative state validation."""

from __future__ import annotations

from ._base import *
from .abilities import AbilitiesMixin
from .collision import CollisionMixin
from .combat import CombatMixin
from .deaths import DeathsMixin
from .deployment import DeploymentMixin
from .match import MatchMixin
from .movement import MovementMixin
from .projectiles import ProjectilesMixin
from .scheduler import SchedulerMixin
from .spawning import SpawningMixin
from .status import StatusMixin
from .targeting import TargetingMixin


class BattleEngine(
    SchedulerMixin,
    DeploymentMixin,
    TargetingMixin,
    MovementMixin,
    CollisionMixin,
    CombatMixin,
    ProjectilesMixin,
    StatusMixin,
    AbilitiesMixin,
    SpawningMixin,
    DeathsMixin,
    MatchMixin,
):
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
        # These values are immutable for a ruleset.  Keeping the narrow
        # numeric component columns on the engine avoids repeatedly walking
        # the card-definition mappings from the per-tick physics loops while
        # leaving the authoritative EntityState object graph unchanged.
        self._card_collision_radii = {
            card_id: int(definition.collision_radius_mtile or 0)
            for card_id, definition in self.ruleset.cards.items()
        }
        self._tower_collision_radii = {
            tower_id: int(definition.collision_radius_mtile or 0)
            for tower_id, definition in self.ruleset.towers.items()
        }
        self._card_masses = {
            card_id: max(1, int(definition.mass or 1))
            for card_id, definition in self.ruleset.cards.items()
        }
        self._card_movement_layers = {
            card_id: str(definition.mechanics.get("movement_layer") or "ground")
            for card_id, definition in self.ruleset.cards.items()
        }
        self._navigation_cache_state: BattleState | None = None
        self._navigation_cache_revision = -1
        self._navigation_cache: tuple[NavigationObstacle, ...] = ()


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
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        canonical_seed = seed & _SEED_MASK
        rng = DeterministicRng(canonical_seed)
        players: list[PlayerState] = []
        for raw_deck in source_decks:
            deck = [self.ruleset.resolve_card_id(card) for card in raw_deck]
            self._validate_deck(deck)
            draw_order = list(deck)
            if shuffle_decks:
                rng.shuffle(draw_order)
            hand_size = self.ruleset.match.hand_size
            # Mirror and Elixir Collector are explicitly excluded from an
            # opening hand. Preserve deterministic order by swapping each
            # excluded card with the first later eligible card instead of
            # reshuffling the deck.
            opening_hand_exclusions = {"mirror", "elixir-collector"}
            for opening_index, opening_card_id in enumerate(draw_order[:hand_size]):
                if opening_card_id not in opening_hand_exclusions:
                    continue
                replacement_index = next(
                    index
                    for index in range(hand_size, len(draw_order))
                    if draw_order[index] not in opening_hand_exclusions
                )
                draw_order[opening_index], draw_order[replacement_index] = (
                    draw_order[replacement_index],
                    draw_order[opening_index],
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
            seed=canonical_seed,
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
            seed=canonical_seed,
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


    def _definition(self, entity: EntityState) -> CardDefinition | TowerDefinition:
        if entity.kind == "tower":
            return self.ruleset.towers[entity.card_id]
        return self.ruleset.cards[entity.card_id]

    def _mechanic_flag(self, entity: EntityState, name: str) -> bool:
        if entity.kind == "tower":
            return False
        definition = self.ruleset.cards.get(entity.card_id)
        return bool(definition is not None and definition.mechanics.get(name))

    def _counts_as_troop(self, entity: EntityState) -> bool:
        return entity.kind == "troop" or self._mechanic_flag(entity, "counts_as_troop")

    def _hook_pullable(self, entity: EntityState) -> bool:
        return entity.kind == "troop" or self._mechanic_flag(entity, "hook_pullable")

    def _pullable_by_area_effect(self, entity: EntityState) -> bool:
        return entity.kind == "troop" or self._mechanic_flag(entity, "pullable_by_area_effect")


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
            if not 0 <= len(player.hand) <= self.ruleset.match.hand_size:
                raise ValueError("invalid hand size")
            if type(player.next_card_cooldown_us) is not int or player.next_card_cooldown_us < 0:
                raise ValueError("next-card cooldown must be a non-negative integer")
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
                entity.attack_load_remaining_us,
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
                entity.dash_remaining_us,
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
            if (
                entity.last_reflection_source_uid is not None
                and type(entity.last_reflection_source_uid) is not int
            ):
                raise ValueError("last reflection source UID must be an integer or None")
            if (
                entity.last_reflection_attack_instance_id is not None
                and type(entity.last_reflection_attack_instance_id) is not int
            ):
                raise ValueError(
                    "last reflection attack instance ID must be an integer or None"
                )
            if entity.lifetime_remaining_us is not None and type(entity.lifetime_remaining_us) is not int:
                raise ValueError("entity lifetime must be an integer or None")
            if entity.charge_remaining_us is not None and type(entity.charge_remaining_us) is not int:
                raise ValueError("entity charge lifetime must be an integer or None")
            if entity.target_uid is not None and type(entity.target_uid) is not int:
                raise ValueError("entity target UID must be an integer or None")
            if entity.pending_target_uid is not None and type(entity.pending_target_uid) is not int:
                raise ValueError("pending target UID must be an integer or None")
            if entity.parent_uid is not None:
                if type(entity.parent_uid) is not int:
                    raise ValueError("parent UID must be an integer or None")
                if entity.parent_uid not in known_uids:
                    raise ValueError("dangling parent UID")
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
            if (
                entity.deploy_remaining_us < 0
                or entity.attack_cooldown_us < 0
                or entity.attack_load_remaining_us < 0
                or entity.windup_remaining_us < 0
                or entity.dash_remaining_us < 0
            ):
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
                projectile.impact_delay_remaining_us,
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
            for coordinate in (
                projectile.previous_x_mtile,
                projectile.previous_y_mtile,
            ):
                if coordinate is not None and type(coordinate) is not int:
                    raise ValueError("projectile previous coordinates must be integers or None")
            if (
                projectile.attack_instance_id is not None
                and type(projectile.attack_instance_id) is not int
            ):
                raise ValueError("projectile attack instance ID must be an integer or None")
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
