"""combat mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *


class CombatMixin:
    def _advance_attacks(self, state: BattleState) -> None:
        dt = self.ruleset.tick_us
        self._advance_attack_ramps(state, dt)
        # A wind-up attack which resolves at the start of this tick must not
        # immediately start a second attack in the same tick.  This matters
        # for Sparky: its cooldown is deliberately cleared after a shot so
        # the next four-second charge can begin on the following tick.
        resolved_this_tick: set[int] = set()
        alive_entities = self._alive_entities(state)
        # Complete attacks which were already winding up.
        for entity in alive_entities:
            if entity.windup_remaining_us <= 0 or self._is_frozen(entity):
                continue
            progress = self._attack_time_progress(entity, dt)
            entity.windup_remaining_us = max(0, entity.windup_remaining_us - progress)
            if entity.windup_remaining_us == 0:
                self._resolve_attack(state, entity)
                resolved_this_tick.add(entity.uid)
        # Cooldowns and new attack starts are stable by UID.
        for entity in alive_entities:
            # A resolving attack may have reduced this entity to zero HP (or
            # consumed a suicide attacker). The next canonical alive scan
            # would omit it; keep the same filter while reusing the stable
            # UID-ordered snapshot.
            if not entity.alive or entity.hp <= 0:
                continue
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
            if (
                entity.deploy_remaining_us > 0
                or entity.concealed_active
                or entity.dash_remaining_us > 0
                or self._is_frozen(entity)
            ):
                continue
            if entity.attack_cooldown_us > 0:
                progress = self._attack_time_progress(entity, dt)
                entity.attack_cooldown_us = max(0, entity.attack_cooldown_us - progress)
            if entity.windup_remaining_us > 0 or entity.attack_cooldown_us > 0:
                continue
            if entity.target_uid is None:
                continue
            target = state.entities.get(entity.target_uid)
            if target is None or not target.alive:
                continue
            definition = self._definition(entity)
            if (
                definition.attack_interval_us is None
                or definition.damage is None
                or definition.range_mtile is None
            ):
                continue

            # A Bandit dash is an impact at the locked landing point, not a
            # deferred damage reservation.  If the target moved out of the
            # landing envelope while the dash was airborne, the special hit
            # misses and the body must return to its ordinary attack cycle.
            if entity.dash_attack_active and not self._in_attack_range(entity, target):
                self._reset_dash(state, entity, reason="target_out_of_range")
                if entity.pending_target_uid == target.uid:
                    entity.pending_target_uid = None
                    entity.attack_load_remaining_us = 0
                    entity.windup_remaining_us = 0

            # The game preloads the first attack while a troop is moving.  A
            # range-only scheduler starts the first-hit clock too late: a
            # troop that has already spent that interval walking still waits
            # the full first-hit delay after reaching its target.  Keep the
            # preload target in the normal pending-target field so the state
            # remains replayable without another hidden map.
            first_attack_ready = False
            if entity.attack_count == 0:
                if entity.pending_target_uid != target.uid:
                    entity.pending_target_uid = target.uid
                    entity.attack_load_remaining_us = max(
                        0, int(definition.first_hit_delay_us or 0)
                    )
                if entity.attack_load_remaining_us > 0:
                    progress = self._attack_time_progress(entity, dt)
                    entity.attack_load_remaining_us = max(
                        0, entity.attack_load_remaining_us - progress
                    )
                first_attack_ready = entity.attack_load_remaining_us == 0
                if not first_attack_ready:
                    continue

            if not self._in_attack_range(entity, target):
                continue
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
            interval = int(definition.attack_interval_us)
            # ``first_hit_delay`` is the acquisition/load time, not an extra
            # pause after every ordinary hit.  Recharge weapons such as
            # Sparky explicitly reuse it as their per-shot wind-up; preserve
            # that authored exception while keeping normal Hit Speed cadence
            # at one interval between impacts.
            recharge_windup = (
                entity.kind != "tower"
                and definition.mechanics.get("attack_windup_mode") == "recharge"
            )
            delay = (
                int(definition.first_hit_delay_us or 0)
                if recharge_windup and not (first_attack_ready and entity.attack_count == 0)
                else 0
            )
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
        self._advance_secondary_attacks(
            state,
            dt,
            alive_entities=alive_entities,
        )

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

    def _advance_secondary_attacks(
        self,
        state: BattleState,
        dt: int,
        *,
        alive_entities: list[EntityState] | None = None,
    ) -> None:
        """Advance independent weapon channels (currently Goblin Machine).

        A secondary weapon deliberately does not reuse ``target_uid`` or the
        primary attack cooldown: both weapons can be active simultaneously and
        the rocket has a blind inner range.  The state fields are serialized so
        stopping and resuming a replay cannot shift the rocket cadence.
        """

        if alive_entities is None:
            alive_entities = self._alive_entities(state)
        for entity in alive_entities:
            if not entity.alive or entity.hp <= 0:
                continue
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

        for entity in alive_entities:
            if not entity.alive or entity.hp <= 0:
                continue
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
            attack_instance_id=source.secondary_attack_count,
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
        source.attack_load_remaining_us = 0
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
            if (
                hook is not None
                and self._edge_distance(source, target)
                >= int(hook.get("min_hook_range_mtile") or 0)
            ):
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
                # Hunter's spread is an angular fan: the same cone occupies
                # less world space at close range and more at long range.
                # Fixed world-space offsets made every target receive the
                # same pellet count regardless of distance, removing the
                # card's defining close-range damage behavior.
                spread_reference = max(1, int(definition.range_mtile or base_distance))
                spread_distance = min(base_distance, spread_reference)
                pellet_offsets = tuple(
                    (
                        pellet_spread
                        * spread_distance
                        * (2 * index - (projectile_count - 1))
                        // ((projectile_count - 1) * spread_reference)
                    )
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
                # Firecracker's line-piercing component describes the five
                # fragments released behind the burst.  Its primary firework
                # still lands on the acquired target tile; applying this
                # component here incorrectly moves the primary impact point
                # past the target.
                if hasattr(line_component, "get") and source.card_id != "firecracker":
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
                attack_instance_id=source.attack_count,
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
        pullable = self._hook_pullable(target)
        if bool(raw_hook.get("pull_troops_only")) and not pullable:
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
        if pullable:
            desired_gap = (
                pull_distance
                + self._collision_radius(source)
                + self._collision_radius(target)
            )
        else:
            # Buildings do not move: Fisherman reels himself into normal melee
            # range instead of treating the hook as a no-op.
            desired_gap = (
                int(self._definition(source).range_mtile or 0)
                + self._collision_radius(source)
                + self._collision_radius(target)
            )
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
        if pullable:
            old_x, old_y = target.x_mtile, target.y_mtile
            target.x_mtile, target.y_mtile = move_towards(
                target.x_mtile,
                target.y_mtile,
                source.x_mtile,
                source.y_mtile,
                travel,
            )
            self._reset_attack_preload(target)
            target.navigation_waypoints.clear()
            target.navigation_cursor = 0
            target.navigation_revision = -1
            self._reset_attack_charge(state, target, reason="hooked")
            self._reset_dash(state, target, reason="hooked")
        else:
            old_x, old_y = source.x_mtile, source.y_mtile
            source.x_mtile, source.y_mtile = move_towards(
                source.x_mtile,
                source.y_mtile,
                target.x_mtile,
                target.y_mtile,
                travel,
            )
            self._reset_attack_preload(source)
            source.navigation_waypoints.clear()
            source.navigation_cursor = 0
            source.navigation_revision = -1
        if jump_was_active and pullable:
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

    def _reset_attack_preload(self, entity: EntityState) -> None:
        """Reset a first attack's movement-loaded clock after displacement."""

        if entity.attack_count > 0:
            return
        target_uid = entity.pending_target_uid or entity.target_uid
        if target_uid is None:
            return
        definition = self._definition(entity)
        interval = getattr(definition, "attack_interval_us", None)
        if interval is None:
            return
        # A displacement interrupts the loaded attack.  Retain the target
        # lock, but restart from the full Hit Time rather than the shortened
        # First Hit time.  This is the documented Fireball/knockback edge.
        entity.pending_target_uid = target_uid
        entity.windup_remaining_us = 0
        entity.attack_load_remaining_us = int(interval)

    def _reset_dash(self, state: BattleState, entity: EntityState, *, reason: str) -> None:
        """Cancel a pending Bandit-style dash impact."""

        if not entity.dash_attack_active and entity.dash_remaining_us == 0:
            return
        entity.dash_attack_active = False
        entity.dash_remaining_us = 0
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
