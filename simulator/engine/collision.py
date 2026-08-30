"""collision mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *


class CollisionMixin:
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

        alive_entities = self._alive_entities(state)
        movable = [
            entity
            for entity in alive_entities
            if (
                entity.kind == "troop"
                and entity.carried_by_uid is None
                and entity.deploy_remaining_us <= 0
            )
        ]
        if len(movable) > 1:
            # The pairwise pass is deliberately a scratch SoA, not a second
            # authoritative state.  ``movable`` is already in the canonical
            # UID order; keeping one column per immutable-in-this-phase value
            # removes repeated dataclass/ruleset lookups without changing the
            # pair sequence or any fixed-point arithmetic.
            uids = [entity.uid for entity in movable]
            layers = [self._movement_layer(entity) for entity in movable]
            radii = [self._collision_radius(entity) for entity in movable]
            masses = [self._mass(entity) for entity in movable]
            count = len(movable)

            for _ in range(3):
                # Positions are written back only after the complete pair pass,
                # exactly as in the object-based implementation.  Re-read them
                # per pass because the prior pass may have moved an entity.
                x_mtile = [entity.x_mtile for entity in movable]
                y_mtile = [entity.y_mtile for entity in movable]
                displacement_x = [0] * count
                displacement_y = [0] * count
                for left_index in range(count - 1):
                    for right_index in range(left_index + 1, count):
                        if layers[left_index] != layers[right_index]:
                            continue
                        dx = x_mtile[right_index] - x_mtile[left_index]
                        dy = y_mtile[right_index] - y_mtile[left_index]
                        distance = distance_mtile(0, 0, dx, dy)
                        minimum = radii[left_index] + radii[right_index]
                        if distance >= minimum:
                            continue
                        overlap = minimum - distance
                        left_mass = masses[left_index]
                        right_mass = masses[right_index]
                        total_mass = left_mass + right_mass
                        # Displacement is inversely proportional to mass. Stable
                        # remainder assignment preserves the exact overlap while
                        # making a heavier tank push a lighter troop farther.
                        left_push = overlap * right_mass // total_mass
                        right_push = overlap - left_push
                        if distance == 0:
                            direction = -1 if (uids[left_index] + uids[right_index]) % 2 else 1
                            unit_x, unit_y, denominator = direction, 0, 1
                        else:
                            unit_x, unit_y, denominator = dx, dy, distance

                        # A one-milli-tile diagonal overlap can otherwise
                        # round both component displacements to zero (for
                        # example, 565/799).  Once a positive share is
                        # assigned, make every non-zero axis advance by at
                        # least one lattice unit.  This preserves deterministic
                        # mass ordering while guaranteeing that a colliding
                        # pair makes progress on the next separation pass.
                        def axis_push(component: int, amount: int) -> int:
                            if component == 0 or amount == 0:
                                return 0
                            magnitude = abs(component) * amount // denominator
                            return (1 if component > 0 else -1) * max(1, magnitude)

                        left_dx = axis_push(unit_x, left_push)
                        left_dy = axis_push(unit_y, left_push)
                        right_dx = axis_push(unit_x, right_push)
                        right_dy = axis_push(unit_y, right_push)
                        displacement_x[left_index] -= left_dx
                        displacement_y[left_index] -= left_dy
                        displacement_x[right_index] += right_dx
                        displacement_y[right_index] += right_dy

                changed = False
                for index, entity in enumerate(movable):
                    dx = displacement_x[index]
                    dy = displacement_y[index]
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
        structures = [entity for entity in alive_entities if entity.kind in {"building", "tower"}]
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
        if (
            distance <= 0
            or target.kind in {"tower", "building"}
            or target.dash_remaining_us > 0
            or not target.alive
            or target.hp <= 0
        ):
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
            self._reset_attack_preload(target)
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

    def _edge_distance(self, source: EntityState, target: EntityState) -> int:
        center = distance_mtile(source.x_mtile, source.y_mtile, target.x_mtile, target.y_mtile)
        return max(0, center - self._collision_radius(source) - self._collision_radius(target))

    def _collision_radius(self, entity: EntityState) -> int:
        if entity.kind == "tower":
            return self._tower_collision_radii[entity.card_id]
        return self._card_collision_radii[entity.card_id]

    def _mass(self, entity: EntityState) -> int:
        if entity.kind == "tower":
            return 1_000_000
        return self._card_masses[entity.card_id]

    def _movement_layer(self, entity: EntityState) -> str:
        """Return the entity's physics navigation layer."""

        if entity.kind == "tower":
            return "ground"
        if entity.river_airborne_active:
            return "air"
        return self._card_movement_layers.get(entity.card_id, "ground")
