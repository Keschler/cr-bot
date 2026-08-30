"""spawning mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *
from .status import StatusMixin


class SpawningMixin:
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
                is_clone=carrier.is_clone,
                hp_override=1 if carrier.is_clone else None,
                max_hp_override=1 if carrier.is_clone else None,
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

    def _advance_spawners(self, state: BattleState, dt: int) -> None:
        """Advance data-driven building and active-troop spawners in UID order."""

        # Spawners are iterated over the entry snapshot. New children are not
        # parents until the next tick, matching the old list-comprehension
        # boundary. Keep a second, append-only alive view for the two queries
        # which intentionally observe children spawned by earlier parents in
        # this same phase.
        parents = self._alive_entities(state)
        alive_entities = list(parents)
        alive_counts: dict[tuple[int, str], int] = {}
        for entity in alive_entities:
            if entity.parent_uid is None:
                continue
            key = (entity.parent_uid, entity.card_id)
            alive_counts[key] = alive_counts.get(key, 0) + 1

        for parent in parents:
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
                    for target in alive_entities
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
            # ``max_alive`` is a cap owned by this spawner instance.  The
            # previous owner/card aggregate made two independent buildings
            # suppress one another's waves and produced a materially wrong
            # board for RL rollouts.
            alive_children = alive_counts.get((parent.uid, child_card_id), 0)
            # ``None`` is an explicit unbounded stream.  It is needed for the
            # post-2025 Furnace rework: the official notes specify one Fire
            # Spirit per cadence but do not publish a maximum number alive.
            # Other spawners retain their sourced finite caps.
            if max_alive is None or alive_children < max_alive:
                for spawn_index in range(int(spawn["count"])):
                    if max_alive is not None and alive_children >= max_alive:
                        break
                    raw_child_deploy_time = spawn.get("child_deploy_time_us")
                    child_spawn_stagger_us = int(
                        spawn.get("child_spawn_stagger_us") or 0
                    )
                    if child_spawn_stagger_us:
                        # A stagger is added to the child's ordinary deploy
                        # clock.  Explicit child deployment overrides (for
                        # example Goblin Hut's 0.5 s release delay) remain
                        # authoritative; without one, use the child card's
                        # own deployment time rather than silently making the
                        # first body deploy instantly.
                        child_deploy_time_us = (
                            int(raw_child_deploy_time)
                            if raw_child_deploy_time is not None
                            else self.ruleset.card(child_card_id).deploy_time_us
                        ) + spawn_index * child_spawn_stagger_us
                    else:
                        child_deploy_time_us = (
                            int(raw_child_deploy_time)
                            if raw_child_deploy_time is not None
                            else None
                        )
                    child_entity = self._spawn_single_child(
                        state,
                        parent,
                        self.ruleset.card(child_card_id),
                        deploy_remaining_us=child_deploy_time_us,
                    )
                    alive_entities.append(child_entity)
                    child_key = (parent.uid, child_entity.card_id)
                    alive_counts[child_key] = alive_counts.get(child_key, 0) + 1
                    alive_children += 1
                    parent.spawned_count += 1
            # A blocked spawner still waits one complete interval; this avoids
            # a death burst when a crowded lane suddenly becomes available.
            parent.spawn_cooldown_us = int(spawn["interval_us"])

    def _spawn_single_child(
        self,
        state: BattleState,
        parent: EntityState,
        child: CardDefinition,
        *,
        offset_mtile: tuple[int, int] = (0, 0),
        deploy_remaining_us: int | None = None,
    ) -> EntityState:
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
        return self._spawn_single_at(
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
        spawn_cooldown_us: int | None = None,
        level_multiplier_permille: int = PERMILLE,
        require_legal_position: bool = False,
    ) -> EntityState:
        uid = self._allocate_uid(state)
        x = min(self.ruleset.arena.width_mtile - 1, max(0, x_mtile))
        y = min(self.ruleset.arena.height_mtile - 1, max(0, y_mtile))
        if require_legal_position:
            x, y = self._nearest_legal_spawn_position(state, child, x, y)
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
            spawn_cooldown_us=(
                0 if spawn_cooldown_us is None else int(spawn_cooldown_us)
            ),
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
            parent_uid=parent_uid,
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

    def _nearest_legal_spawn_position(
        self,
        state: BattleState,
        child: CardDefinition,
        x_mtile: int,
        y_mtile: int,
    ) -> tuple[int, int]:
        """Bump a derived ground spawn to the nearest legal free point.

        Persistent effects such as Graveyard author spawn offsets in world
        space.  The game does not materialize a Skeleton inside a tower, on
        the river bank, or outside the arena; it resolves that offset to the
        nearest legal point.  Keep the search deterministic and local so
        generated replays do not depend on a physics-library implementation.
        """

        radius = int(child.collision_radius_mtile or 0)
        movement_layer = str(child.mechanics.get("movement_layer") or "ground")
        if movement_layer == "air":
            return (
                min(self.ruleset.arena.width_mtile - 1, max(0, x_mtile)),
                min(self.ruleset.arena.height_mtile - 1, max(0, y_mtile)),
            )
        structures = [
            entity
            for entity in self._alive_entities(state)
            if entity.kind in {"building", "tower"}
        ]

        def legal(x: int, y: int) -> bool:
            if not point_is_walkable(self.ruleset.arena, x, y, radius):
                return False
            return all(
                distance_mtile(x, y, structure.x_mtile, structure.y_mtile)
                >= radius + self._collision_radius(structure)
                for structure in structures
            )

        if legal(x_mtile, y_mtile):
            return x_mtile, y_mtile

        # A 100-milli-tile lattice is finer than the placement grid and keeps
        # the worst-case river-to-bridge search bounded.  Search complete
        # square rings so the first successful ring is spatially local, then
        # use exact distance/coordinates as the deterministic tie-break.
        step = 100
        max_radius = max(self.ruleset.arena.width_mtile, self.ruleset.arena.height_mtile)
        max_ring = (max_radius + step - 1) // step
        for ring in range(1, max_ring + 1):
            candidates: list[tuple[int, int]] = []
            extent = ring * step
            for index in range(-ring, ring + 1):
                candidates.extend(
                    (
                        (x_mtile + index * step, y_mtile - extent),
                        (x_mtile + index * step, y_mtile + extent),
                        (x_mtile - extent, y_mtile + index * step),
                        (x_mtile + extent, y_mtile + index * step),
                    )
                )
            legal_candidates = [
                point
                for point in set(candidates)
                if legal(*point)
            ]
            if legal_candidates:
                return min(
                    legal_candidates,
                    key=lambda point: (
                        (point[0] - x_mtile) ** 2 + (point[1] - y_mtile) ** 2,
                        point,
                    ),
                )

        # The arena always contains legal ground cells for the fixed roster;
        # retain a bounded fallback for malformed custom arenas rather than
        # allocating an entity at an unbounded coordinate.
        return (
            min(self.ruleset.arena.width_mtile - 1, max(radius, x_mtile)),
            min(self.ruleset.arena.height_mtile - 1, max(radius, y_mtile)),
        )

    @staticmethod
    def _spawn_time_progress(entity: EntityState, dt: int) -> int:
        """Advance a spawner clock under Rage without floating-point drift."""

        multiplier = StatusMixin._hit_speed_multiplier(entity)
        numerator = dt * multiplier + entity.spawn_time_remainder
        progress, entity.spawn_time_remainder = divmod(numerator, PERMILLE)
        return progress
