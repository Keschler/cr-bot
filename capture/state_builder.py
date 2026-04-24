from constants import KING_TOWER_HP
from game_state import GameState, HudState, PrincessTowerState

def build_game_state(result, *, seen_enemy_cards=None, elixir_enemy_est=None):
    towers_hp = result["towers_hp"]
    hand = result["state"]

    def tower_alive(hp):
        return hp is not None and hp > 0

    def king_active(hp):
        return hp is not None and hp < KING_TOWER_HP

    princess_towers = PrincessTowerState(
        own_left_alive=tower_alive(towers_hp["own_support_left"]),
          own_right_alive=tower_alive(towers_hp["own_support_right"]),
          enemy_left_alive=tower_alive(towers_hp["enemy_support_left"]),
          enemy_right_alive=tower_alive(towers_hp["enemy_support_right"]),
    )

    hud = HudState(
        time_left_s=result["time_left_s"],
          overtime=result["overtime"],
          elixir_self=result["elixir"]["estimated_value"] + result["elixir"]["displayed_digit"][0],
          hand_cards=[
              hand["card_1"],
              hand["card_2"],
              hand["card_3"],
              hand["card_4"],
          ],
          next_card=hand["next_card"],
          tower_hp_self=[
              towers_hp["own_support_left"],
              towers_hp["own_king"],
              towers_hp["own_support_right"],
          ],
          tower_hp_enemy=[
              towers_hp["enemy_support_left"],
              towers_hp["enemy_king"],
              towers_hp["enemy_support_right"],
          ],
          princess_towers=princess_towers,
    )
    
    own_units = [m for m in result["matches"] if m.troop.team == "ally"]
    enemy_units = [m for m in result["matches"] if m.troop.team == "enemy"]

    return GameState(
        hud=hud,
        total_remaining_s=result["total_remaining_s"],
        own_units=own_units,
        enemy_units=enemy_units,
        seen_enemy_cards=seen_enemy_cards or [],
        elixir_enemy_est=0.0 if elixir_enemy_est is None else elixir_enemy_est,
        own_king_active=king_active(towers_hp["own_king"]),
        enemy_king_active=king_active(towers_hp["enemy_king"]),
    )
