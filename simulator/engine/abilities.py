"""abilities mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *


class AbilitiesMixin:
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
            and (
                entity.kind == copy_kind
                or self._mechanic_flag(entity, "cloneable_by_clone")
            )
            # Carrier payloads are already represented as first-class bodies
            # attached to their original. Cloning those hidden bodies as
            # ordinary troops recreates the historical floating/bodyless
            # Goblin Giant bug and incorrectly doubles the payload.
            and entity.carried_by_uid is None
            and (not exclude_clones or not entity.is_clone)
            and distance_mtile(x, y, entity.x_mtile, entity.y_mtile)
            <= radius + self._collision_radius(entity)
        ]
        cloned = 0
        for original in originals:
            child = self.ruleset.card(original.card_id)
            spawn = child.mechanics.get("spawn")
            elixir_generation = child.mechanics.get("elixir_generation")
            initial_spawn_cooldown_us = (
                int(spawn.get("start_delay_us", 0))
                if hasattr(spawn, "get")
                else (
                    int(elixir_generation.get("interval_us", 0))
                    if hasattr(elixir_generation, "get")
                    else 0
                )
            )
            cloned_entity = self._spawn_single_at(
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
                # A clone skips the copied card's normal deploy animation;
                # its own periodic stream starts afresh.
                deploy_remaining_us=0,
                spawn_cooldown_us=initial_spawn_cooldown_us,
                level_multiplier_permille=level_multiplier_permille,
            )
            # Carriers materialize their payload as attached clone bodies.
            # This keeps the payload hidden/sheltered until release and lets
            # the death-spawn fallback avoid creating duplicates.
            self._spawn_carried_children(state, cloned_entity)
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
