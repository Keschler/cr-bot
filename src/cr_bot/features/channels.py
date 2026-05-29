STATIC_CHANNELS = [
    "ground_walkable",
    "river_mask",
    "bridge_mask",
    "own_princess_tower_sites",
    "own_king_tower_site",
    "enemy_princess_tower_sites",
    "enemy_king_tower_site",
]


DYNAMIC_CHANNELS = [
    "ally_ground_presence",
    "enemy_ground_presence",
    "ally_air_presence",
    "enemy_air_presence",
    "ally_building_presence",
    "enemy_building_presence",
    "ally_hp_mass",
    "enemy_hp_mass",
    "ally_threat_mass",
    "enemy_threat_mass",
    "recent_ally_spell_effect",
    "recent_enemy_spell_effect",
    "own_alive_tower_mask",
    "enemy_alive_tower_mask",
]

GLOBAL_SCALAR_FEATURES = [
      "elixir_self",
      "elixir_enemy_est",
      "time_left_norm",
      "overtime",
      "own_king_active",
      "enemy_king_active",
      "own_left_princess_alive",
      "own_right_princess_alive",
      "enemy_left_princess_alive",
      "enemy_right_princess_alive",
      "deploy_state_left",
      "deploy_state_right",
      "tower_hp_self_left",
      "tower_hp_self_king",
      "tower_hp_self_right",
      "tower_hp_enemy_left",
      "tower_hp_enemy_king",
      "tower_hp_enemy_right",
  ]




STATIC_CHANNEL_IDX = {name: i for i, name in enumerate(STATIC_CHANNELS)}
DYNAMIC_CHANNEL_IDX = {name: i for i, name in enumerate(DYNAMIC_CHANNELS)}
GLOBAL_SCALAR_IDX = {name: i for i, name in enumerate(GLOBAL_SCALAR_FEATURES)}

