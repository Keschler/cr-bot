"""deaths mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *


class DeathsMixin:
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
                entity.attack_load_remaining_us = 0
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
            rage_definition = self.ruleset.card("rage")
            rage_damage = int(rage_definition.damage or 0)
            rage_crown_damage = int(rage_definition.crown_tower_damage or 0)
            self._create_area_effect(
                state,
                owner=entity.owner,
                source_uid=entity.uid,
                source_card_id=entity.card_id,
                x_mtile=entity.x_mtile,
                y_mtile=entity.y_mtile,
                default_radius=int(death_rage.get("radius_mtile") or 0),
                default_damage=rage_damage,
                default_crown_damage=rage_crown_damage,
                default_status=None,
                default_knockback=0,
                raw_effect={
                    "duration_us": int(death_rage["duration_us"]),
                    "duration_anchor": "creation",
                    "tick_interval_us": int(death_rage["tick_interval_us"]),
                    "radius_mtile": int(death_rage["radius_mtile"]),
                    "damage_per_tick": rage_damage,
                    "crown_damage_per_tick": rage_crown_damage,
                    "targets": ["air", "ground", "building", "crown_tower"],
                    "friendly_status": {
                        "kind": "rage",
                        "duration_us": int(death_rage["duration_us"]),
                        "speed_multiplier_milli": int(death_rage["speed_multiplier_milli"]),
                        "hit_speed_multiplier_milli": int(death_rage["hit_speed_multiplier_milli"]),
                        "linger_us": 0,
                    },
                    "friendly_targets": list(death_rage["targets"]),
                    "damage_schedule": [rage_damage],
                    "crown_damage_schedule": [rage_crown_damage],
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
        revived_hitpoints = (
            1
            if entity.is_clone
            else self._scale_level_value(
                int(revive_values["revived_hitpoints"]),
                entity.level_multiplier_permille,
            )
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
            is_clone=entity.is_clone,
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
