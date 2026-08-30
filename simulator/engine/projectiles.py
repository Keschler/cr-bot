"""projectiles mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *


class ProjectilesMixin:
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
            projectile.previous_x_mtile = old_x
            projectile.previous_y_mtile = old_y
            remaining = distance_mtile(
                old_x,
                old_y,
                projectile.target_x_mtile,
                projectile.target_y_mtile,
            )
            if projectile.impact_delay_remaining_us > 0:
                delay_before = projectile.impact_delay_remaining_us
                projectile.impact_delay_remaining_us = max(0, delay_before - dt)
                if delay_before > dt:
                    continue
                # The authored delay is the complete arrival schedule for a
                # delayed spell. Resolve at the selected point when it
                # expires instead of adding a second, artificial flight leg.
                travel = remaining
                projectile.x_mtile = projectile.target_x_mtile
                projectile.y_mtile = projectile.target_y_mtile
            else:
                numerator = projectile.speed_mtile_per_s * dt + projectile.movement_remainder
                travel, projectile.movement_remainder = divmod(numerator, SECOND_US)
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
                        projectile.previous_x_mtile = projectile.x_mtile
                        projectile.previous_y_mtile = projectile.y_mtile
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
                        child_deploy_time_us = (
                            int(spawn["child_deploy_time_us"])
                            if spawn.get("child_deploy_time_us") is not None
                            else None
                        )
                        for _ in range(int(spawn["count"])):
                            self._spawn_single_at(
                                state,
                                child,
                                owner=projectile.owner,
                                x_mtile=projectile.target_x_mtile,
                                y_mtile=projectile.target_y_mtile,
                                parent_uid=projectile.source_uid,
                                level_multiplier_permille=projectile.level_multiplier_permille,
                                deploy_remaining_us=child_deploy_time_us,
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
                attack_instance_id=projectile.attack_instance_id,
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
                    child_deploy_time_us = (
                        int(spawn["child_deploy_time_us"])
                        if spawn.get("child_deploy_time_us") is not None
                        else None
                    )
                    for _ in range(int(spawn["count"])):
                        self._spawn_single_at(
                            state,
                            child,
                            owner=projectile.owner,
                            x_mtile=projectile.target_x_mtile,
                            y_mtile=projectile.target_y_mtile,
                            parent_uid=projectile.source_uid,
                            level_multiplier_permille=projectile.level_multiplier_permille,
                            deploy_remaining_us=child_deploy_time_us,
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
                attack_instance_id=projectile.attack_instance_id,
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
            self._deal_damage(
                state,
                target,
                dealt,
                source.uid,
                source.card_id,
                source.attack_count,
            )
            if status and target.hp > 0:
                self._apply_status(state, target, status)
            if reset_attack and target.hp > 0:
                target.attack_cooldown_us = 0
                target.windup_remaining_us = 0
                target.pending_target_uid = None
                target.attack_load_remaining_us = 0
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
            state,
            target,
            dealt,
            projectile.source_uid,
            projectile.source_card_id,
            projectile.attack_instance_id,
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
            target.attack_load_remaining_us = 0
        self._emit(
            state,
            "chain_hit",
            source_uid=projectile.source_uid,
            source_card_id=projectile.source_card_id,
            target_uid=target.uid,
            target_index=target_index,
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
        if (
            projectile.previous_x_mtile is not None
            and projectile.previous_y_mtile is not None
        ):
            ax, ay = projectile.previous_x_mtile, projectile.previous_y_mtile
        else:
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
            self._deal_damage(
                state,
                target,
                damage,
                projectile.source_uid,
                projectile.source_card_id,
                projectile.attack_instance_id,
            )
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
        # The next tick (or a direct component-level helper call) must sweep
        # only the newly traversed segment.  Retaining the launch origin here
        # repeatedly re-hits every body that lies anywhere behind the current
        # projectile position.
        projectile.previous_x_mtile = bx
        projectile.previous_y_mtile = by

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
        attack_instance_id: int | None = None,
    ) -> None:
        candidates: list[EntityState] = []
        if radius <= 0 and primary_target_uid is not None:
            target = state.entities.get(primary_target_uid)
            # A projectile can outlive the target state used when it was
            # launched. Re-apply the impact target contract at contact time;
            # otherwise a ground-only projectile can damage a unit which has
            # become airborne during a river jump.
            if (
                target is not None
                and target.alive
                and target.owner != owner
                and self._spell_can_hit(
                    source_card_id,
                    target,
                    allowed_targets=allowed_targets,
                )
                and distance_mtile(x, y, target.x_mtile, target.y_mtile)
                <= radius + self._collision_radius(target)
            ):
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
            self._deal_damage(
                state,
                target,
                dealt,
                source_uid,
                source_card_id,
                attack_instance_id,
            )
            if status and target.hp > 0 and not curse_status:
                self._apply_status(state, target, status)
            if reset_attack and target.hp > 0:
                target.attack_cooldown_us = 0
                target.windup_remaining_us = 0
                target.pending_target_uid = None
                target.attack_load_remaining_us = 0
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
