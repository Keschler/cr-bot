from dataclasses import dataclass



@dataclass(slots=True)
class Detection:
    track_id: int | None
    class_name: str
    team: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    estimated_hp: float | None=None

@dataclass(slots=True)
class Match:
    troop: Detection
    bar: Detection | None


@dataclass(slots=True, frozen=True)
class PrincessTowerState:
    own_left_alive: bool
    own_right_alive: bool
    enemy_left_alive: bool
    enemy_right_alive: bool

    def as_deploy_kwargs(self) -> dict[str, bool]:
        return {
            "own_left_princess_down": not self.own_left_alive,
            "own_right_princess_down": not self.own_right_alive,
            "enemy_left_princess_down": not self.enemy_left_alive,
            "enemy_right_princess_down": not self.enemy_right_alive,
        }

@dataclass(slots=True)
class HudState:
    time_left_s: float
    overtime: bool
    elixir_self: float
    hand_cards: list[str]
    next_card: str
    tower_hp_self: list[float]
    tower_hp_enemy: list[float]
    princess_towers: PrincessTowerState


@dataclass(slots=True)
class GameState:
    hud: HudState
    total_remaining_s: float
    own_units: list[Match]
    enemy_units: list[Match]
    seen_enemy_cards: list[int]
    elixir_enemy_est: float 
    own_king_active: bool
    enemy_king_active: bool
    started: bool

Cell = tuple[int, int]

@dataclass(slots=True)
class Action:
    kind: str   # Wait or Play
    card_idx: int | None = None
    cell: Cell | None = None
