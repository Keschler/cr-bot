from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cr_bot.features.action_space import (
    ACTION_GRID,
    BRIDGE_MASK,
    ENEMY_KING_TOWER_SITE,
    ENEMY_LEFT_PRINCESS_TOWER_SITE,
    ENEMY_PRINCESS_TOWER_SITES,
    ENEMY_RIGHT_PRINCESS_TOWER_SITE,
    LEGAL_GROUND,
    OWN_KING_TOWER_SITE,
    OWN_LEFT_PRINCESS_TOWER_SITE,
    OWN_PRINCESS_TOWER_SITES,
    OWN_RIGHT_PRINCESS_TOWER_SITE,
    RIVER_MASK,
)
from cr_bot.features.channels import DYNAMIC_CHANNELS, DYNAMIC_CHANNEL_IDX, STATIC_CHANNELS, STATIC_CHANNEL_IDX
from cr_bot.domain.troop_hp_level16 import get_unit_hp_level16
from cr_bot.domain.card_metadata import CARD_METADATA

if TYPE_CHECKING:
    from cr_bot.domain.game_state import GameState, Match

GRID_ROWS, GRID_COLS = LEGAL_GROUND.shape
BOARD_SHAPE = (GRID_ROWS, GRID_COLS)

KERNEL_3X3 = np.array(
  [
      [0.05, 0.10, 0.05],
      [0.10, 0.40, 0.10],
      [0.05, 0.10, 0.05],
  ],
  dtype=np.float32,
)

def build_static_board() -> np.ndarray:
    board = np.zeros((len(STATIC_CHANNELS), GRID_ROWS, GRID_COLS), dtype=np.float32)

    board[STATIC_CHANNEL_IDX["ground_walkable"]] = LEGAL_GROUND.astype(np.float32)
    board[STATIC_CHANNEL_IDX["river_mask"]] = RIVER_MASK.astype(np.float32)
    board[STATIC_CHANNEL_IDX["bridge_mask"]] = BRIDGE_MASK.astype(np.float32)
    board[STATIC_CHANNEL_IDX["own_princess_tower_sites"]] = OWN_PRINCESS_TOWER_SITES.astype(np.float32)
    board[STATIC_CHANNEL_IDX["own_king_tower_site"]] = OWN_KING_TOWER_SITE.astype(np.float32)
    board[STATIC_CHANNEL_IDX["enemy_princess_tower_sites"]] = ENEMY_PRINCESS_TOWER_SITES.astype(np.float32)
    board[STATIC_CHANNEL_IDX["enemy_king_tower_site"]] = ENEMY_KING_TOWER_SITE.astype(np.float32)

    return board

def build_dynamic_board(state: GameState, arena_px: tuple[float, float, float, float]):
    board = np.zeros((len(DYNAMIC_CHANNELS), GRID_ROWS, GRID_COLS), dtype=np.float32)

    rasterize_units(board, state.own_units, team="ally", arena_px=arena_px)
    rasterize_units(board, state.enemy_units, team="enemy", arena_px=arena_px)
    rasterize_alive_towers(board, state)

    return board

def rasterize_units(board: np.ndarray, units: list[Match], team: str, arena_px) -> None:
    for match in units:
        troop = match.troop
        cell = ACTION_GRID.pixel_to_cell(troop.center_x, troop.center_y, arena_px)
        if cell is None:
            continue

        col, row = cell
        hp_frac = estimate_hp_fraction(match)
        threat = estimate_threat_weight(troop.class_name) * hp_frac

        if team == "ally":
            presence_ch = "ally_air_presence" if is_air_unit(troop.class_name) else "ally_ground_presence"
            hp_ch = "ally_hp_mass"
            threat_ch = "ally_threat_mass"
        else:
            presence_ch = "enemy_air_presence" if is_air_unit(troop.class_name) else "enemy_ground_presence"
            hp_ch = "enemy_hp_mass"
            threat_ch = "enemy_threat_mass"
        splat(board[DYNAMIC_CHANNEL_IDX[presence_ch]], row, col, 1.0)
        splat(board[DYNAMIC_CHANNEL_IDX[hp_ch]], row, col, hp_frac)
        splat(board[DYNAMIC_CHANNEL_IDX[threat_ch]], row, col, threat)

def rasterize_alive_towers(board, state: GameState):
    towers = state.hud.princess_towers

    own_ch = board[DYNAMIC_CHANNEL_IDX["own_alive_tower_mask"]]
    enemy_ch = board[DYNAMIC_CHANNEL_IDX["enemy_alive_tower_mask"]]

    own_ch[OWN_KING_TOWER_SITE] = 1.0
    enemy_ch[ENEMY_KING_TOWER_SITE] = 1.0

    if towers.own_left_alive:
        own_ch[OWN_LEFT_PRINCESS_TOWER_SITE] = 1.0
    if towers.own_right_alive:
        own_ch[OWN_RIGHT_PRINCESS_TOWER_SITE] = 1.0
    if towers.enemy_left_alive:
        enemy_ch[ENEMY_LEFT_PRINCESS_TOWER_SITE] = 1.0
    if towers.enemy_right_alive:
        enemy_ch[ENEMY_RIGHT_PRINCESS_TOWER_SITE] = 1.0


# spreads a unit’s value onto a small 3x3 area of a board channel
def splat(channel: np.ndarray, row: int, col: int, value: float) -> None:
    radius = 1

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1): 
            rr = row + dr 
            cc = col + dc
            if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS: # Prevent writing outside of board (if the unit is near edge)
                channel[rr, cc] += value * KERNEL_3X3[dr + radius, dc + radius]

def estimate_hp_fraction(match: Match):
    hp = match.troop.estimated_hp
    if hp is None:
        return 1.0
    
    if 0.0 <= hp <= 1.0:
        return float(hp)
    
    max_hp = get_unit_hp_level16(match.troop.class_name)
    if max_hp:
        return min(1.0, max(0.0, float(hp) / max_hp))
    
    return 1.0

def is_air_unit(card_name):
    metadata = CARD_METADATA.get(card_name)
    return bool(metadata and metadata.get("is_air"))

def estimate_threat_weight(card_name) -> float:
    metadata = CARD_METADATA.get(card_name)
    if metadata is None:
        raise KeyError(f"{card_name} is not included in card metadata")
    
    damage = metadata.get("damage")
    hit_speed = metadata.get("hit_speed")
    if (
        not isinstance(damage, (int, float))
        or isinstance(damage, bool)
        or not isinstance(hit_speed, (int, float))
        or isinstance(hit_speed, bool)
        or hit_speed <= 0
    ):
        return 0.0
    dps = damage / max(hit_speed, 0.1)

    return min(dps / 1000.0, 5.0)


def build_board(state: GameState, arena_px: tuple[float, float, float, float]) -> np.ndarray:
    static = build_static_board()
    dynamic = build_dynamic_board(state, arena_px)
    return np.concatenate([static, dynamic], axis=0)
