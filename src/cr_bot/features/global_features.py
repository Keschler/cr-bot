import numpy as np

from cr_bot.domain.constants import FEATURE_MAX_KING_TOWER_HP, MAX_ELIXIR, PRINCESS_TOWER_HP, TOTAL_MATCH_SECONDS
from cr_bot.features.channels import GLOBAL_SCALAR_FEATURES, GLOBAL_SCALAR_IDX
from cr_bot.domain.game_state import GameState
from cr_bot.domain.card_metadata import CARD_METADATA

CARD_COUNT = max(card["id"] for card in CARD_METADATA.values()) + 1

def build_global_scalar_vector(state) -> np.ndarray:
    g = np.zeros(len(GLOBAL_SCALAR_FEATURES), dtype=np.float32)
    hud = state.hud
    towers = state.hud.princess_towers

    g[GLOBAL_SCALAR_IDX["elixir_self"]] = normalize_elixir(hud.elixir_self)
    g[GLOBAL_SCALAR_IDX["elixir_enemy_est"]] = normalize_elixir(state.elixir_enemy_est)
    g[GLOBAL_SCALAR_IDX["time_left_norm"]] = normalize_time(state.total_remaining_s)
    g[GLOBAL_SCALAR_IDX["overtime"]] = float(hud.overtime)

    g[GLOBAL_SCALAR_IDX["own_left_princess_alive"]] = float(towers.own_left_alive)
    g[GLOBAL_SCALAR_IDX["own_right_princess_alive"]] = float(towers.own_right_alive)
    g[GLOBAL_SCALAR_IDX["enemy_left_princess_alive"]] = float(towers.enemy_left_alive)
    g[GLOBAL_SCALAR_IDX["enemy_right_princess_alive"]] = float(towers.enemy_right_alive)


    own_left, own_king, own_right = normalize_tower_hp_triplet(hud.tower_hp_self)
    enemy_left, enemy_king, enemy_right = normalize_tower_hp_triplet(hud.tower_hp_enemy)

    g[GLOBAL_SCALAR_IDX["tower_hp_self_left"]] = own_left
    g[GLOBAL_SCALAR_IDX["tower_hp_self_king"]] = own_king
    g[GLOBAL_SCALAR_IDX["tower_hp_self_right"]] = own_right

    g[GLOBAL_SCALAR_IDX["tower_hp_enemy_left"]] = enemy_left
    g[GLOBAL_SCALAR_IDX["tower_hp_enemy_king"]] = enemy_king
    g[GLOBAL_SCALAR_IDX["tower_hp_enemy_right"]] = enemy_right

    g[GLOBAL_SCALAR_IDX["own_king_active"]] = float(state.own_king_active)
    g[GLOBAL_SCALAR_IDX["enemy_king_active"]] = float(state.enemy_king_active)

    g[GLOBAL_SCALAR_IDX["deploy_state_left"]] = float(not towers.enemy_left_alive)
    g[GLOBAL_SCALAR_IDX["deploy_state_right"]] = float(not towers.enemy_right_alive)

    return g


def normalize_elixir(elixir):
    return float(np.clip(elixir / MAX_ELIXIR, 0.0, 1.0))

def normalize_time(time_left_s):
    return float(np.clip(time_left_s / TOTAL_MATCH_SECONDS, 0.0, 1.0))

def normalize_hp(hp, max_hp):
    if hp is None:
        return 0.0
    return float(np.clip(hp / max_hp, 0.0, 1.0))

def normalize_tower_hp_triplet(values) -> tuple[float, float, float]:
    left, king, right = values

    return (
        normalize_hp(left, PRINCESS_TOWER_HP),
        normalize_hp(king, FEATURE_MAX_KING_TOWER_HP),
        normalize_hp(right, PRINCESS_TOWER_HP),
    )

def card_to_id(card_name):
    if not card_name:
        return None
    key = card_name.strip().lower().replace(" ", "-")
    metadata = CARD_METADATA.get(key)
    if metadata is None:
        return None
    return metadata["id"]

def one_hot_card(card_name) -> np.ndarray:
    vec = np.zeros(CARD_COUNT, dtype=np.float32)
    card_id = card_to_id(card_name)
    if card_id is not None:
        vec[card_id] = 1.0
    return vec

def encode_hand_cards(hand_cards: list[str]) -> np.ndarray:
    parts = []
    
    for i in range(4):
        card_name = hand_cards[i]
        parts.append(one_hot_card(card_name))
    
    return np.concatenate(parts)

def encode_next_card(next_card):
    return one_hot_card(next_card)

def encode_seen_enemy_cards(seen_enemy_cards):
    vec = np.zeros(CARD_COUNT, dtype=np.float32)

    for card_id in seen_enemy_cards:
        vec[card_id] = 1.0

    return vec


def build_global_vector(state):
    return np.concatenate(
        [
            build_global_scalar_vector(state),
            encode_hand_cards(state.hud.hand_cards),
            encode_next_card(state.hud.next_card),
            encode_seen_enemy_cards(state.seen_enemy_cards),
        ]
    ).astype(np.float32)
