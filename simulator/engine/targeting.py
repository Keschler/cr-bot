"""targeting mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *


class TargetingMixin:
    def _invalidate_and_acquire_targets(self, state: BattleState) -> None:
        # Target acquisition only mutates target/charge bookkeeping; entity
        # membership and HP do not change during this phase. Reuse the
        # canonical UID-ordered alive snapshot across all selectors instead
        # of sorting and filtering the entity mapping once per candidate.
        alive_entities = self._alive_entities(state)
        for entity in alive_entities:
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
                    and self._edge_distance(entity, current)
                    > self._sight_range(entity) * 3 // 2
                ):
                    invalid = True
                if invalid:
                    old_target = entity.target_uid
                    self._reset_attack_charge(state, entity, reason="target_invalidated")
                    self._reset_dash(state, entity, reason="target_invalidated")
                    self._reset_attack_ramp(state, entity, reason="target_invalidated")
                    entity.target_uid = None
                    if entity.pending_target_uid == old_target:
                        entity.pending_target_uid = None
                        entity.windup_remaining_us = 0
                        entity.attack_load_remaining_us = 0
                    self._emit(state, "target_changed", uid=entity.uid, old_target=old_target, target_uid=None)
                elif current is not None and entity.kind != "tower":
                    # Moving troops scan for a nearer valid target until they
                    # have entered attack range and committed to the current
                    # attack.  Building-targeters are the live-game
                    # exception: they keep selecting the closest building
                    # even after entering attack range.
                    building_targeter = bool(definition.mechanics.get("building_only")) or (
                        bool(definition.targets)
                        and set(definition.targets) <= {"building", "crown_tower"}
                    )
                    if building_targeter or not self._target_is_locked(entity, current):
                        replacement = self._choose_target(
                            state,
                            entity,
                            alive_entities=alive_entities,
                        )
                        if replacement is not None and replacement != current.uid:
                            old_target = entity.target_uid
                            self._reset_attack_charge(state, entity, reason="retargeted")
                            self._reset_dash(state, entity, reason="retargeted")
                            self._reset_attack_ramp(state, entity, reason="retargeted")
                            entity.target_uid = replacement
                            if entity.pending_target_uid == old_target:
                                entity.pending_target_uid = None
                                entity.windup_remaining_us = 0
                                entity.attack_load_remaining_us = 0
                            self._emit(
                                state,
                                "target_changed",
                                uid=entity.uid,
                                old_target=old_target,
                                target_uid=replacement,
                            )
            if entity.target_uid is None:
                target_uid = self._choose_target(
                    state,
                    entity,
                    alive_entities=alive_entities,
                )
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
            and not self._projectile_lethally_reserved(state, target)
        )

    def _target_is_locked(self, source: EntityState, target: EntityState) -> bool:
        """Return whether a primary target has entered attack commitment.

        A first-hit preload may exist while a troop is still walking, so
        ``pending_target_uid`` alone is not a lock.  An active wind-up, dash,
        movement charge, or a target already inside the ordinary attack range
        is committed; unlocked movers are allowed to reacquire each tick.
        """

        if source.dash_remaining_us > 0 or source.attack_charge_active:
            return True
        if source.windup_remaining_us > 0:
            return True
        return self._in_attack_range(source, target)

    @staticmethod
    def _projectile_lethally_reserved(
        state: BattleState,
        target: EntityState,
    ) -> bool:
        """Return whether already-launched shots reserve a lethal target.

        A projectile keeps its original target at contact, while the attacker
        may acquire another target as soon as the in-flight shot is guaranteed
        to be lethal. Shield damage is a complete hit transaction, so model it
        explicitly instead of summing raw damage.
        """

        remaining_hp = target.hp
        remaining_shield = target.shield_hp
        incoming: list[tuple[int, int]] = []
        for projectile in state.projectiles.values():
            if (
                projectile.alive
                and projectile.target_uid == target.uid
                and projectile.owner != target.owner
            ):
                damage = (
                    projectile.crown_damage
                    if target.kind == "tower"
                    else projectile.damage
                )
                if damage > 0:
                    incoming.append((projectile.uid, int(damage)))
        for _, damage in sorted(incoming):
            if remaining_shield > 0:
                remaining_shield = max(0, remaining_shield - damage)
            else:
                remaining_hp = max(0, remaining_hp - damage)
            if remaining_hp == 0:
                return True
        return False

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
        if target.carried_by_uid is not None:
            # Goblin Giant's backpack Spear Goblins attack independently but
            # are not independent target bodies until the carrier dies.
            return False
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
        # A normal troop is already a valid damage/target body during its
        # deployment animation.  ``deploy_remaining_us`` gates the target's
        # own movement and attacks, but it must not make towers and nearby
        # troops ignore it.  Tunneling, carried, concealed, and stealth bodies
        # have been filtered above because those are separate untargetable
        # mechanics.  Keep the explicit component for non-troop entities and
        # future card-specific exceptions.
        return target.kind == "troop" or bool(
            definition.mechanics.get("targetable_during_deploy")
        )

    def _choose_target(
        self,
        state: BattleState,
        source: EntityState,
        *,
        alive_entities: list[EntityState] | None = None,
    ) -> int | None:
        definition = self._definition(source)
        alive = self._alive_entities(state) if alive_entities is None else alive_entities
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
                for target in alive
                if target.owner != source.owner
                and target.kind != "tower"
                and self._targetable_for_acquisition(state, target)
                and self._target_allowed(source, target)
                and not self._projectile_lethally_reserved(state, target)
                and self._edge_distance(source, target) <= sight
            ]
            return self._nearest_uid(source, nearby)
        nearby = self._nearby_non_tower_targets(
            state,
            source,
            alive_entities=alive,
        )
        min_attack_range = int(definition.mechanics.get("min_attack_range_mtile") or 0)
        nearby.extend(
            target
            for target in alive
            if target.owner != source.owner
            and target.kind == "tower"
            and self._target_allowed(source, target)
            and self._edge_distance(source, target) <= sight
            and not self._projectile_lethally_reserved(state, target)
            and self._edge_distance(source, target)
            >= min_attack_range
        )
        if nearby:
            return self._preferred_target_uid(
                state,
                source,
                nearby,
                alive_entities=alive,
            )
        towers = [
            target
            for target in alive
            if target.owner != source.owner
            and target.kind == "tower"
            and target.role != "king"
            and self._target_allowed(source, target)
            and not self._projectile_lethally_reserved(state, target)
            and self._edge_distance(source, target) >= min_attack_range
        ]
        if not towers:
            towers = [
                target
                for target in alive
                if target.owner != source.owner
                and target.kind == "tower"
                and self._target_allowed(source, target)
                and not self._projectile_lethally_reserved(state, target)
                and self._edge_distance(source, target) >= min_attack_range
            ]
        return self._nearest_uid(source, towers)

    def _nearby_non_tower_targets(
        self,
        state: BattleState,
        source: EntityState,
        *,
        alive_entities: list[EntityState] | None = None,
    ) -> list[EntityState]:
        definition = self._definition(source)
        if definition.sight_range_mtile is None or definition.damage is None:
            return []
        sight = self._sight_range(source)
        min_attack_range = int(definition.mechanics.get("min_attack_range_mtile") or 0)
        alive = self._alive_entities(state) if alive_entities is None else alive_entities
        return [
            target
            for target in alive
            if target.owner != source.owner
            and target.kind != "tower"
            and self._targetable_for_acquisition(state, target)
            and self._target_allowed(source, target)
            and not self._projectile_lethally_reserved(state, target)
            and self._edge_distance(source, target) <= sight
            and self._edge_distance(source, target) >= min_attack_range
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
        *,
        alive_entities: list[EntityState] | None = None,
    ) -> int | None:
        if not candidates:
            return None
        definition = self._definition(source)
        if source.kind == "tower" or not definition.mechanics.get("spread_targets"):
            return self._nearest_uid(source, candidates)
        sibling_targets: dict[int, int] = {}
        alive = self._alive_entities(state) if alive_entities is None else alive_entities
        for sibling in alive:
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
        if (
            source.kind != "tower"
            and target.jump_remaining_us > 0
            and mechanics.get("cannot_hit_jumping")
        ):
            return False
        if source.charge_active and mechanics.get("charge_threshold_permille") is not None:
            # Goblin Demolisher's low-health phase becomes building-only.
            targets = {"building", "crown_tower"}
        if source.kind == "tower":
            # Crown Towers attack troops only. Their generic ``ground`` target
            # class must not make placed buildings valid victims.
            return self._counts_as_troop(target) and str(self._movement_layer(target)) in targets
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
            if self._counts_as_troop(target):
                return str(self._movement_layer(target)) in targets
            return "building" in targets or "ground" in targets
        target_definition = self.ruleset.cards[target.card_id]
        layer = self._movement_layer(target)
        return str(layer) in targets

    def _in_attack_range(self, source: EntityState, target: EntityState) -> bool:
        definition = self._definition(source)
        range_mtile = definition.range_mtile
        hook = definition.mechanics.get("hook") if source.kind != "tower" else None
        if hook is not None:
            distance = self._edge_distance(source, target)
            melee_range = int(definition.range_mtile or 0)
            hook_range = int(hook.get("hook_range_mtile") or melee_range)
            min_hook_range = int(hook.get("min_hook_range_mtile") or 0)
            if distance <= melee_range:
                return True
            return min_hook_range <= distance <= hook_range
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
