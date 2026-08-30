"""movement mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *


class MovementMixin:
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
                or entity.dash_remaining_us > 0
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
                    # The dash itself is the loaded attack.  Starting the
                    # ordinary first-hit clock here would make Bandit wait a
                    # full Hit Speed after landing before applying the
                    # authored dash damage.
                    entity.pending_target_uid = target.uid
                    entity.windup_remaining_us = 0
                    entity.attack_load_remaining_us = 0
                    entity.dash_attack_active = True
                    entity.dash_remaining_us = max(
                        self.ruleset.tick_us,
                        int(dash.get("duration_us") or self.ruleset.tick_us),
                    )
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
            if definition.mechanics.get("trigger_on_target"):
                # The legacy range column on transport cards describes the
                # carrier's body, not the payload trigger distance.  Let the
                # explicit contact predicate below control stopping; otherwise
                # Skeleton Barrel parks at its ordinary melee edge forever and
                # never reaches the building it is meant to trigger on.
                in_range = False
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
                # The route's first edge was validated when it was created.
                # Ground units only move monotonically toward that waypoint,
                # and ``navigation_revision`` changes whenever a structure
                # obstacle is created, transformed, or destroyed. Therefore
                # rechecking the shrinking sub-segment on every physics tick
                # is redundant; a topology change will fall through to the
                # normal route rebuild on the next call.
                return entity.navigation_waypoints[entity.navigation_cursor]
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
        # Structure positions are immutable throughout movement/separation;
        # ``navigation_revision`` is incremented whenever a structure is
        # created, transformed, or destroyed. Keep one canonical obstacle
        # tuple for that state/revision and apply the tiny target exclusion at
        # the end. Holding the state reference also prevents an object-id
        # reuse from ever returning a stale tuple to an interleaved caller.
        if (
            self._navigation_cache_state is not state
            or self._navigation_cache_revision != state.navigation_revision
        ):
            self._navigation_cache_state = state
            self._navigation_cache_revision = state.navigation_revision
            self._navigation_cache = tuple(
                NavigationObstacle(
                    uid=entity.uid,
                    x_mtile=entity.x_mtile,
                    y_mtile=entity.y_mtile,
                    radius_mtile=self._collision_radius(entity),
                )
                for entity in self._alive_entities(state)
                if entity.kind in {"building", "tower"}
            )
        if target_uid is None:
            return self._navigation_cache
        return tuple(
            obstacle
            for obstacle in self._navigation_cache
            if obstacle.uid != target_uid
        )
