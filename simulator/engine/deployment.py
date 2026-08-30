"""deployment mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *


class DeploymentMixin:
    def _legal_cells_for_card(
        self,
        state: BattleState,
        player: int,
        card: CardDefinition,
        *,
        territory_cells: frozenset[tuple[int, int]] | None = None,
        deployment_obstacles: tuple[tuple[int, int, int], ...] | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Evaluate one already-resolved card over the policy grid.

        The result is equivalent to calling ``_legal_deployment`` for every
        cell, but shared territory and obstacle predicates are computed once
        by ``legal_action_cells``.  This is deliberately kept inside the
        engine so the generic public ``validate_action`` path remains the
        authoritative single-action validator.
        """

        placement = card.mechanics.get("placement_class")
        if placement == "spell_anywhere":
            # Every coordinate in the policy grid is already bounds-checked.
            return _POLICY_GRID_CELLS
        if placement in {"restricted_spell", "own_ground_spell", "spells"}:
            return self._restricted_spell_cells(state, player)

        if placement == "miner_anywhere":
            candidates = _GROUND_CELLS
        else:
            if territory_cells is None:
                territory_cells = self._deployment_territory_cells(state, player)
            candidates = tuple(cell for cell in _POLICY_GRID_CELLS if cell in territory_cells)

        if deployment_obstacles is None:
            deployment_obstacles = self._deployment_obstacles(state)

        radius = int(card.collision_radius_mtile or 0)
        if card.kind == "troop":
            return self._cells_without_deployment_collision(
                candidates,
                radius,
                deployment_obstacles,
            )

        if card.kind == "building":
            if territory_cells is None:
                territory_cells = self._deployment_territory_cells(state, player)
            footprint_size = int(card.mechanics.get("building_footprint_size") or 3)
            candidates = _footprint_cells_in_allowed(territory_cells, footprint_size)
            return self._cells_without_deployment_collision(
                candidates,
                radius,
                deployment_obstacles,
            )

        return candidates

    def _restricted_spell_cells(
        self,
        state: BattleState,
        player: int,
    ) -> tuple[tuple[int, int], ...]:
        """Return restricted-spell cells without repeating tower scans."""

        destroyed_enemy_lanes = tuple(
            tower.x_mtile // 1_000
            for tower in self._towers_for(state, 1 - player)
            if not tower.alive and tower.role != "king"
        )
        cells: list[tuple[int, int]] = []
        for col, row in _POLICY_GRID_CELLS:
            if row >= 17 if player == 0 else row <= 14:
                cells.append((col, row))
                continue
            forward = 11 <= row <= 16 if player == 0 else 15 <= row <= 20
            if forward and any(
                (col < GRID_COLS // 2) == (tower_col < GRID_COLS // 2)
                for tower_col in destroyed_enemy_lanes
            ):
                cells.append((col, row))
        return tuple(cells)

    def _deployment_obstacles(
        self,
        state: BattleState,
    ) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (entity.x_mtile, entity.y_mtile, self._collision_radius(entity))
            for entity in state.entities.values()
            if entity.alive and entity.kind in {"building", "tower"}
        )

    @staticmethod
    def _cells_without_deployment_collision(
        candidates: tuple[tuple[int, int], ...],
        radius: int,
        obstacles: tuple[tuple[int, int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        blocked = _blocked_deployment_cells(obstacles, radius)
        if not blocked:
            return candidates
        return tuple(cell for cell in candidates if cell not in blocked)

    def _deployment_territory_cells(
        self,
        state: BattleState,
        player: int,
    ) -> frozenset[tuple[int, int]]:
        """Return the current deployment territory in one shared scan.

        Most states use the immutable basic-deployment map.  Destroyed
        Princess Towers add small dynamic pockets; those are layered on top
        only when present instead of asking ``_territory_cell`` to rescan the
        tower list for every card and every grid cell.
        """

        territory = set(_BASIC_DEPLOY_CELLS[player])
        own_towers = self._towers_for(state, player)
        enemy_towers = self._towers_for(state, 1 - player)
        dynamic_towers = tuple(
            tower
            for tower in (*own_towers, *enemy_towers)
            if not tower.alive and tower.role != "king"
        )
        if not dynamic_towers:
            return _BASIC_DEPLOY_CELLS[player]
        for tower in own_towers:
            if tower.alive or tower.role == "king":
                continue
            for cell in _GROUND_CELLS:
                if distance_mtile(
                    *cell_center_mtile(cell),
                    tower.x_mtile,
                    tower.y_mtile,
                ) <= 2_000:
                    territory.add(cell)
        for tower in enemy_towers:
            if tower.alive or tower.role == "king":
                continue
            for col, row in _GROUND_CELLS:
                forward = 11 <= row <= 16 if player == 0 else 15 <= row <= 20
                center_x = col * POSITION_SCALE + POSITION_SCALE // 2
                if forward and abs(center_x - tower.x_mtile) <= 5_000:
                    territory.add((col, row))
        return frozenset(territory)

    def _play_card(self, state: BattleState, action: PlayCardAction, card_id: str) -> None:
        player = state.players[action.player]
        card = self.ruleset.card(card_id)
        previous_card_id = player.last_played_card_id
        effective_cost = self._effective_card_cost(player, card)
        player.elixir_milli -= effective_cost
        used = player.hand.pop(action.card_slot)
        player.draw_pile.append(used)
        # The first card in the queue is ready only when its loading timer has
        # expired.  A rapid second/third play therefore leaves an empty hand
        # slot until the existing Next card finishes loading.
        if player.next_card_cooldown_us == 0 and player.draw_pile:
            player.hand.append(player.draw_pile.pop(0))
            player.next_card_cooldown_us = self._card_cycle_cooldown_us(state)
        player.cards_played += 1
        opponent_seen = state.players[1 - action.player].seen_enemy_cards
        if card_id not in opponent_seen:
            opponent_seen.append(card_id)
        col, row = action.cell
        self._emit(
            state,
            "card_played",
            player=action.player,
            card_id=card_id,
            card_slot=action.card_slot,
            col=col,
            row=row,
            cost_milli=effective_cost,
        )
        if card.card_id == "mirror":
            if previous_card_id is None:
                self._emit(state, "mirror_no_target", player=action.player)
            else:
                mirrored = self.ruleset.card(previous_card_id)
                if mirrored.kind == "spell":
                    self._spawn_spell(
                        state, action.player, mirrored, action.cell,
                        level_multiplier_permille=1_100,
                    )
                else:
                    self._spawn_card_entities(
                        state, action.player, mirrored, action.cell,
                        level_multiplier_permille=1_100,
                    )
                self._emit(
                    state,
                    "card_mirrored",
                    player=action.player,
                    source_card_id=previous_card_id,
                    cost_milli=effective_cost,
                    level_delta=1,
                )
            # Mirror consumes itself.  It is deliberately the new previous
            # card, so a second Mirror cannot mirror a Mirror (or recursively
            # manufacture an unbounded chain).  A normal card played after it
            # clears this guard in the ordinary branch below.
            player.last_played_card_id = card.card_id
        elif card.kind == "spell":
            self._spawn_spell(state, action.player, card, action.cell)
            player.last_played_card_id = card.card_id
        else:
            self._spawn_card_entities(state, action.player, card, action.cell)
            player.last_played_card_id = card.card_id

    def _spawn_spell(
        self,
        state: BattleState,
        player: int,
        card: CardDefinition,
        cell: tuple[int, int],
        *,
        level_multiplier_permille: int = PERMILLE,
    ) -> None:
        if card.projectile is None:
            raise ValueError(f"spell {card.card_id} lacks an executable projectile")
        target_x, target_y = cell_center_mtile(cell)
        mechanics = card.mechanics
        mode = mechanics.get("projectile_mode")
        impact_mode = mechanics.get("impact_mode")
        origin = mechanics.get("spell_origin")

        # Spell origin is an executable part of the card definition.  A
        # selected-position origin is used by The Log; rolling spells also
        # begin at their selected endpoint and then continue along their
        # authored travel direction.  Ballistic spells originate at the
        # player's King Tower.  Keeping this dispatch here (rather than
        # inferring it from the card id) makes mirrored and future spells use
        # the same deterministic path.
        if mode == "rolling_linear" or origin == "selected-position":
            start_x, start_y = target_x, target_y
        elif origin == "own-king-tower":
            king = self._tower(state, player, "king")
            start_x, start_y = king.x_mtile, king.y_mtile
        else:
            raise RulesetError(
                f"{card.card_id}: unsupported spell origin {origin!r}"
            )

        if mode == "rolling_linear":
            direction = -1 if player == 0 else 1
            target_y = min(
                self.ruleset.arena.height_mtile - 1,
                max(
                    0,
                    start_y
                    + direction
                    * int(card.mechanics.get("rolling_range_mtile") or card.range_mtile or 0),
                ),
            )
        raw_status = card.mechanics.get("status")
        # Continuous impact modes are swept along the projectile path.  The
        # explicit component is authoritative even if an older generated
        # card omitted the redundant ``piercing`` boolean.
        continuous = impact_mode in {"continuous", "continuous_path"}
        projectile = ProjectileState(
            uid=self._allocate_uid(state),
            source_uid=None,
            source_card_id=card.card_id,
            owner=player,
            x_mtile=start_x,
            y_mtile=start_y,
            target_x_mtile=target_x,
            target_y_mtile=target_y,
            damage=self._scale_level_value(int(card.damage or 0), level_multiplier_permille),
            crown_damage=self._scale_level_value(
                int(card.crown_tower_damage if card.crown_tower_damage is not None else card.damage or 0),
                level_multiplier_permille,
            ),
            speed_mtile_per_s=card.projectile.speed_mtile_per_s,
            impact_delay_remaining_us=int(mechanics.get("impact_delay_us") or 0),
            speed_code=(
                int(card.mechanics["projectile_speed_code"])
                if card.mechanics.get("projectile_speed_code") is not None
                else None
            ),
            homing=card.projectile.homing,
            radius_mtile=int(card.area_radius_mtile or card.projectile.radius_mtile),
            status_kind=None if not raw_status else str(raw_status.get("kind")),
            status_duration_us=0 if not raw_status else int(raw_status.get("duration_us") or 0),
            status_magnitude_permille=(
                PERMILLE
                if not raw_status
                else int(
                    raw_status.get("speed_multiplier_milli")
                    if raw_status.get("speed_multiplier_milli") is not None
                    else PERMILLE
                )
            ),
            status_damage_per_tick=0
            if not raw_status
            else int(raw_status.get("damage_per_tick") or 0),
            status_tick_interval_us=0
            if not raw_status
            else int(raw_status.get("tick_interval_us") or 0),
            knockback_mtile=int(mechanics.get("knockback_mtile") or 0),
            piercing=bool(mechanics.get("piercing")) or continuous,
            allowed_targets=tuple(
                str(value) for value in mechanics.get("impact_targets", ())
            ),
            origin_x_mtile=start_x,
            origin_y_mtile=start_y,
            line_end_x_mtile=target_x,
            line_end_y_mtile=target_y,
            direction_x_mtile=target_x - start_x,
            direction_y_mtile=target_y - start_y,
            level_multiplier_permille=level_multiplier_permille,
        )
        state.projectiles[projectile.uid] = projectile
        self._emit(
            state,
            "projectile_spawned",
            uid=projectile.uid,
            player=player,
            card_id=card.card_id,
            source_uid=None,
            target_uid=None,
            projectile_speed_code=projectile.speed_code,
        )

    def _legal_deployment(
        self,
        state: BattleState,
        player: int,
        card: CardDefinition,
        cell: tuple[int, int],
    ) -> bool:
        placement = card.mechanics.get("placement_class")
        if placement == "spell_anywhere":
            return is_spell_cell(cell)
        if placement in {"restricted_spell", "own_ground_spell", "spells"}:
            return self._restricted_spell_cell(state, player, cell)
        if placement == "miner_anywhere":
            if not is_ground_cell(cell):
                return False
        elif not self._territory_cell(state, player, cell):
            return False
        # Clash Royale does not allow a troop (ground *or* air) to be dropped
        # on top of an existing structure.  This is independent of ownership:
        # the same exclusion applies to friendly buildings and to enemy
        # buildings in a temporarily opened deployment pocket.  Spells are
        # handled by their own placement masks and may target structures.
        if card.kind == "troop":
            x, y = cell_center_mtile(cell)
            radius = int(card.collision_radius_mtile or 0)
            for entity in state.entities.values():
                if not entity.alive or entity.kind not in {"building", "tower"}:
                    continue
                if distance_mtile(x, y, entity.x_mtile, entity.y_mtile) < radius + self._collision_radius(entity):
                    return False
        if card.kind == "building":
            footprint_size = int(card.mechanics.get("building_footprint_size") or 3)
            if not self._building_footprint_fits(state, player, cell, footprint_size):
                return False
            x, y = cell_center_mtile(cell)
            radius = int(card.collision_radius_mtile or 0)
            for entity in state.entities.values():
                if not entity.alive or entity.kind not in {"building", "tower"}:
                    continue
                other_radius = self._collision_radius(entity)
                if distance_mtile(x, y, entity.x_mtile, entity.y_mtile) < radius + other_radius:
                    return False
        return True

    def _building_footprint_fits(
        self,
        state: BattleState,
        player: int,
        cell: tuple[int, int],
        size: int,
    ) -> bool:
        """Apply dynamic post-tower territory to every footprint cell."""

        low = -(size // 2)
        high = size - size // 2
        col, row = cell
        return all(
            self._territory_cell(state, player, (col + dcol, row + drow))
            for drow in range(low, high)
            for dcol in range(low, high)
        )

    def _restricted_spell_cell(
        self,
        state: BattleState,
        player: int,
        cell: tuple[int, int],
    ) -> bool:
        if not is_spell_cell(cell):
            return False
        col, row = cell
        if row >= 17 if player == 0 else row <= 14:
            return True
        enemy = 1 - player
        for tower in self._towers_for(state, enemy):
            if tower.alive or tower.role == "king":
                continue
            tower_col = tower.x_mtile // 1_000
            same_lane = (col < GRID_COLS // 2) == (tower_col < GRID_COLS // 2)
            forward = 11 <= row <= 16 if player == 0 else 15 <= row <= 20
            if same_lane and forward:
                return True
        return False

    def _territory_cell(self, state: BattleState, player: int, cell: tuple[int, int]) -> bool:
        if is_basic_deploy_cell(player, cell):
            return True
        if not is_ground_cell(cell):
            return False
        col, row = cell
        # Princess loss opens the destroyed site; taking an enemy Princess
        # Tower opens the corresponding forward pocket, matching policy-v1's
        # coarse deployment-state contract with center-cell semantics.
        for tower in self._towers_for(state, player):
            if tower.alive or tower.role == "king":
                continue
            if distance_mtile(*cell_center_mtile(cell), tower.x_mtile, tower.y_mtile) <= 2_000:
                return True
        enemy = 1 - player
        for tower in self._towers_for(state, enemy):
            if tower.alive or tower.role == "king":
                continue
            same_lane = abs(cell_center_mtile(cell)[0] - tower.x_mtile) <= 5_000
            forward = 11 <= row <= 16 if player == 0 else 15 <= row <= 20
            if same_lane and forward:
                return True
        return False

    def _advance_deployments(self, state: BattleState) -> None:
        dt = self.ruleset.tick_us
        for entity in self._alive_entities(state):
            if entity.deploy_remaining_us <= 0:
                continue
            entity.deploy_remaining_us = max(0, entity.deploy_remaining_us - dt)
            if entity.deploy_remaining_us == 0:
                definition = self._definition(entity)
                mechanics = {} if entity.kind == "tower" else definition.mechanics
                if entity.burrow_active:
                    entity.burrow_active = False
                    self._emit(
                        state,
                        "burrow_emerged",
                        uid=entity.uid,
                        player=entity.owner,
                        card_id=entity.card_id,
                        x_mtile=entity.x_mtile,
                        y_mtile=entity.y_mtile,
                    )
                self._emit(
                    state,
                    "entity_deployed",
                    uid=entity.uid,
                    player=entity.owner,
                    card_id=entity.card_id,
                )
                # Clone bodies still deploy normally, but cloned deployment
                # pulses are suppressed. This covers the copied Electro/Ice
                # Wizard, Mega Knight, and Battle Healer interactions while
                # preserving the ordinary body after the deploy event.
                deploy_effect = None if entity.is_clone else mechanics.get("deploy_effect")
                if deploy_effect is not None:
                    self._impact_area(
                        state,
                        owner=entity.owner,
                        source_uid=entity.uid,
                        source_card_id=entity.card_id,
                        x=entity.x_mtile,
                        y=entity.y_mtile,
                        damage=self._scale_level_value(
                            int(deploy_effect.get("damage") or 0),
                            entity.level_multiplier_permille,
                        ),
                        crown_damage=self._scale_level_value(
                            int(deploy_effect.get("crown_tower_damage") or 0),
                            entity.level_multiplier_permille,
                        ),
                        radius=int(deploy_effect.get("radius_mtile") or 0),
                        status=deploy_effect,
                        knockback=int(deploy_effect.get("knockback_mtile") or 0),
                        primary_target_uid=None,
                        allowed_targets=tuple(str(value) for value in deploy_effect.get("targets", ())),
                    )
                    self._emit(
                        state,
                        "deployment_effect",
                        uid=entity.uid,
                        card_id=entity.card_id,
                        effect_kind=str(deploy_effect.get("kind")),
                    )
                jump = mechanics.get("jump")
                if (
                    not entity.is_clone
                    and jump is not None
                    and bool(jump.get("spawn_damage", True))
                ):
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
                        allowed_targets=tuple(str(value) for value in mechanics.get("impact_targets", ())) or None,
                    )
                    entity.attack_cooldown_us = int(definition.attack_interval_us or 0)
                    self._emit(
                        state,
                        "landing_attack",
                        uid=entity.uid,
                        card_id=entity.card_id,
                        x_mtile=entity.x_mtile,
                        y_mtile=entity.y_mtile,
                    )
