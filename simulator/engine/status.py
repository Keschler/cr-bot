"""status mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *


class StatusMixin:
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
                if (
                    target.kind == "building"
                    and not self._counts_as_troop(target)
                    and hasattr(raw_effect, "get")
                ):
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
                    require_legal_position=effect.source_card_id == "graveyard",
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

        if (
            target.kind == "tower"
            or (target.kind == "building" and not self._pullable_by_area_effect(target))
            or not target.alive
        ):
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
            self._reset_attack_preload(target)
            target.navigation_waypoints.clear()
            target.navigation_cursor = 0
            target.navigation_revision = -1

    def _advance_statuses_and_lifetimes(self, state: BattleState) -> None:
        dt = self.ruleset.tick_us
        for entity in self._alive_entities(state):
            if entity.dash_remaining_us > 0:
                entity.dash_remaining_us = max(0, entity.dash_remaining_us - dt)
                if entity.dash_remaining_us == 0:
                    self._emit(
                        state,
                        "dash_ended",
                        uid=entity.uid,
                        card_id=entity.card_id,
                    )
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
            lifetime_progress_us = dt
            if mechanics.get("revive_egg") is not None:
                # Rage is the one documented status that accelerates Phoenix
                # Egg hatching. Inspect the status at tick entry so a Rage
                # expiring on this boundary still advances the egg for the
                # interval during which it was active.
                rage_multiplier = max(
                    [
                        PERMILLE,
                        *(
                            int(status.magnitude_permille)
                            for status in entity.statuses
                            if status.kind == "rage" and status.remaining_us > 0
                        ),
                    ]
                )
                lifetime_progress_us = dt * rage_multiplier // PERMILLE
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
            entity.lifetime_remaining_us = max(
                0, entity.lifetime_remaining_us - lifetime_progress_us
            )
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

    def _advance_concealment(self, state: BattleState) -> None:
        """Raise and lower Tesla-style structures from visible enemy sight."""

        alive_entities = self._alive_entities(state)
        for entity in alive_entities:
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
                    for target in alive_entities
                )
            if should_conceal == entity.concealed_active:
                continue
            entity.concealed_active = should_conceal
            if should_conceal:
                entity.target_uid = None
                entity.pending_target_uid = None
                entity.windup_remaining_us = 0
                entity.attack_load_remaining_us = 0
            self._emit(
                state,
                "entity_concealment_changed",
                uid=entity.uid,
                card_id=entity.card_id,
                concealed=should_conceal,
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

    def _spell_can_hit(
        self,
        card_id: str,
        target: EntityState,
        *,
        allowed_targets: tuple[str, ...] | None = None,
    ) -> bool:
        if target.carried_by_uid is not None:
            # Attached carrier payloads are sheltered from direct and area
            # impacts.  They become ordinary spell targets only after the
            # carrier release transition.
            return False
        source_definition = self.ruleset.cards.get(card_id)
        if (
            target.jump_remaining_us > 0
            and source_definition is not None
            and source_definition.mechanics.get("cannot_hit_jumping")
        ):
            return False
        if card_id in self.ruleset.towers:
            if not self._counts_as_troop(target):
                return False
            targets = set(
                allowed_targets
                if allowed_targets is not None
                else self.ruleset.towers[card_id].targets
            )
            return str(self._movement_layer(target)) in targets
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
            if target.kind == "building" and self._counts_as_troop(target):
                return str(self._movement_layer(target)) in targets
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
        attack_instance_id: int | None = None,
    ) -> None:
        if (
            damage <= 0
            or not target.alive
            or target.hp <= 0
            or target.dash_remaining_us > 0
        ):
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
                attack_instance_id=attack_instance_id,
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
        entity.attack_load_remaining_us = 0
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
        attack_instance_id: int | None = None,
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
        if any(
            status.kind == "freeze" and status.remaining_us > 0
            for status in target.statuses
        ):
            # Freeze suppresses Electro Giant's reactive aura for the whole
            # frozen window.  A regular damage hit still lands normally.
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
        # The Zap Pack's ordinary target class is Air/Ground troops.  Crown
        # Towers are the one non-troop exception and use the reduced reflected
        # tower-damage value below; defensive buildings are not reflected
        # victims.  ``_spell_can_hit`` deliberately treats ``ground`` as a
        # valid building target for ordinary spells, so this card-specific
        # boundary must be explicit here.
        if attacker.kind == "building":
            return
        reflection = raw_reflection
        radius = int(reflection.get("radius_mtile") or 0)
        if distance_mtile(target.x_mtile, target.y_mtile, attacker.x_mtile, attacker.y_mtile) > radius + self._collision_radius(attacker):
            return
        allowed = tuple(str(value) for value in reflection.get("targets", ()))
        if not self._spell_can_hit(target.card_id, attacker, allowed_targets=allowed):
            return
        if attack_instance_id is not None:
            if (
                target.last_reflection_source_uid == source_uid
                and target.last_reflection_attack_instance_id == attack_instance_id
            ):
                return
            target.last_reflection_source_uid = source_uid
            target.last_reflection_attack_instance_id = attack_instance_id
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
            attacker.attack_load_remaining_us = 0
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
            if target.dash_remaining_us > 0:
                # Bandit is invulnerable and crowd-control immune during the
                # authored dash phase.  Do not reset/cancel the dash before
                # its landing attack has resolved.
                return
            self._reset_attack_charge(state, target, reason=kind)
            self._reset_dash(state, target, reason=kind)
            self._reset_attack_ramp(state, target, reason=kind)
            self._reset_attack_preload(target)
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
                target.attack_load_remaining_us = 0
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
            if status.kind in _SLOW_STATUS_KINDS
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
            if status.kind in _SLOW_STATUS_KINDS
        ]
        rage = [status.magnitude_permille for status in entity.statuses if status.kind == "rage"]
        result = min(slow, default=PERMILLE)
        if rage:
            result = result * max(rage) // PERMILLE
        return result
