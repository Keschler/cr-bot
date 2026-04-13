from dataclasses import dataclass


Cell = tuple[int, int]

@dataclass(slots=True)
class Detection:
    class_name: str
    team: str
    confusion: float
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
    own_left: bool
    own_right: bool
    enemy_left: bool
    enemy_right: bool

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
    own_units: list[Match]
    enemy_unist: list[Match]
    seen_enemy_cards: list[int]
    #elixir_enemy: float 

Cell = tuple[int, int]

@dataclass(slots=True)
class Action:
    kind: str   # Wait or Play
    card_idx: int | None = None
    cell: None or Cell


