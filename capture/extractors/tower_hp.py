import cv2
from pathlib import Path
from constants import FULL_TOWER_HP, KING_TOWER_HP
from image_utils import crop, read_number_from_roi, preprocess_digit, detect_if_king_tower_activated, detect_if_support_tower_alive
from vision.yolo_runtime import parse_box_row, load_yolo_runtime

from rois import ROIS

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates" / "numbers"


def read_template(name: str):
    path = TEMPLATE_DIR / name
    template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Failed to read tower HP template: {path}")
    return template


def load_templates():
    raw_templates = {
        0: read_template("0.png"),
        1: read_template("1.png"),
        2: read_template("2.png"),
        3: read_template("3.png"),
        4: read_template("4.png"),
        5: read_template("5.png"),
        6: read_template("6.png"),
        7: read_template("7.png"),
        8: read_template("8.png"),
        9: read_template("9.png"),
    }

    return {
        digit: preprocess_digit(template)
        for digit, template in raw_templates.items()
    }


TEMPLATES = load_templates()

def extract_tower_hp(frame, yolo_boxes=None, debug_steps_by_tower=None):
    if yolo_boxes is None:
        towers_hp = {
            "enemy_king": {
                "image": crop(frame, ROIS["opponent_king_health_text"]),
                "value": None,
            },
            "own_king": {
                "image": crop(frame, ROIS["player_king_health_text"]),
                "value": None,
            },
            "enemy_support_left": {
                "image": crop(frame, ROIS["opponent_left_support_health_text"]),
                "value": None,
            },
            "enemy_support_right": {
                "image": crop(frame, ROIS["opponent_right_support_health_text"]),
                "value": None,
            },
            "own_support_left": {
                "image": crop(frame, ROIS["player_left_support_health_text"]),
                "value": None,
            },
            "own_support_right": {
                "image": crop(frame, ROIS["player_right_support_health_text"]),
                "value": None,
            },
        }

        for tower_name, tower_data in towers_hp.items():
            tower_debug = {} if debug_steps_by_tower is not None else None
            tower_data["value"] = read_number_from_roi(tower_data["image"], TEMPLATES, debug_steps=tower_debug)
            if debug_steps_by_tower is not None:
                debug_steps_by_tower[tower_name] = tower_debug


        king_tower_activated = detect_if_king_tower_activated(frame)

        if not king_tower_activated["own_king_activated"]:
            towers_hp["own_king"]["value"] = KING_TOWER_HP
        if not king_tower_activated["enemy_king_activated"]:
            towers_hp["enemy_king"]["value"] = KING_TOWER_HP

        support_tower_alive = detect_if_support_tower_alive(frame)

        if not support_tower_alive["support_left_activated"]:
            towers_hp["own_support_left"]["value"] = 0
        if not support_tower_alive["support_right_activated"]:
            towers_hp["own_support_right"]["value"] = 0
        if not support_tower_alive["enemy_support_left_activated"]:
            towers_hp["enemy_support_left"]["value"] = 0
        if not support_tower_alive["enemy_support_right_activated"]:
            towers_hp["enemy_support_right"]["value"] = 0

        return {
            tower_name: tower_data["value"]
            for tower_name, tower_data in towers_hp.items()
        }
    else:
        _, _, idx2unit = load_yolo_runtime()

        king_towers = []
        queen_towers = []
        tower_bars = []
        king_tower_bars = []
        
        for row in yolo_boxes:
            x1, y1, x2, y2, track_id, conf, cls, team = parse_box_row(row)
            class_name = idx2unit[int(cls)]

            det = {
              "class_name": class_name,
              "team": "enemy" if int(team) == 1 else "ally",
              "track_id": None if track_id is None else int(track_id),
              "confidence": float(conf),
              "x1": float(x1),
              "y1": float(y1),
              "x2": float(x2),
              "y2": float(y2),
              "cx": float((x1 + x2) / 2.0),
              "cy": float((y1 + y2) / 2.0),
          }


            if class_name == "king-tower":
               king_towers.append(det) 
            elif class_name == "queen-tower":
                queen_towers.append(det)
            elif class_name in ("tower-bar", "dagger-duchess-tower-bar"):
                tower_bars.append(det)
            elif class_name == "king-tower-bar":
                king_tower_bars.append(det)

        ally_king = [d for d in king_towers if d["team"] == "ally"]
        enemy_king = [d for d in king_towers if d["team"] == "enemy"]

        ally_queens = sorted(
          [d for d in queen_towers if d["team"] == "ally"],
          key=lambda d: d["cx"],
      )
        enemy_queens = sorted(
          [d for d in queen_towers if d["team"] == "enemy"],
          key=lambda d: d["cx"],
      )
        own_support_left = ally_queens[0] if len(ally_queens) > 0 else None
        own_support_right = ally_queens[1] if len(ally_queens) > 1 else None
        enemy_support_left = enemy_queens[0] if len(enemy_queens) > 0 else None
        enemy_support_right = enemy_queens[1] if len(enemy_queens) > 1 else None

        towers = {
                "own_king": ally_king[0] if ally_king else None,
                "enemy_king": enemy_king[0] if enemy_king else None,
                "own_support_left": own_support_left,
                "own_support_right": own_support_right,
                "enemy_support_left": enemy_support_left,
                "enemy_support_right": enemy_support_right,
                }
        king_matches = match_bars_to_towers([t for name, t in towers.items() if t is not None and "king" in name], king_tower_bars)
        queen_matches = match_bars_to_towers([t for name, t in towers.items() if t is not None and "support" in name], tower_bars)

        result = {}

        for tower_name, tower in towers.items():
            tower_debug = {} if debug_steps_by_tower is not None else None
            if tower is None:
                result[tower_name] = FULL_TOWER_HP[tower_name]
                if debug_steps_by_tower is not None:
                    debug_steps_by_tower[tower_name] = tower_debug
                continue

            if "king" in tower_name:
                bar = king_matches.get(id(tower))
            else:
                bar = queen_matches.get(id(tower))

            if bar is None:
                result[tower_name] = FULL_TOWER_HP[tower_name]
                if debug_steps_by_tower is not None:
                    debug_steps_by_tower[tower_name] = tower_debug
                continue

            bar_roi = (
                int(bar["x1"]),
                int(bar["y1"]),
                max(1, int(bar["x2"] - bar["x1"])),
                max(1, int(bar["y2"] - bar["y1"])),
            )
            bar_img = crop(frame, bar_roi)
            text_img = crop_tower_hp_text_area(bar_img, tower_name)
            value = read_number_from_roi(text_img, TEMPLATES, debug_steps=tower_debug, digit_mode="tower")
            if value in (None, 0):
                value = FULL_TOWER_HP[tower_name]
            if len(str(value)) > 4:
                value = str(value)[-4:]

            result[tower_name] = value
            if debug_steps_by_tower is not None:
                debug_steps_by_tower[tower_name] = tower_debug
        return result


def squared_distance(a, b):
      ax, ay = center(a)
      bx, by = center(b)
      return (ax - bx) ** 2 + (ay - by) ** 2

def center(det):
    return det["cx"], det["cy"]



def match_bars_to_towers(towers, bars):
      matches = {}
      remaining_bars = bars.copy()

      for tower in towers:
          candidates = [
              bar for bar in remaining_bars
              if bar["team"] == tower["team"]
          ]

          if not candidates:
              matches[id(tower)] = None
              continue

          best_bar = min(candidates, key=lambda bar: squared_distance(tower, bar))
          matches[id(tower)] = best_bar
          remaining_bars.remove(best_bar)

      return matches


def crop_tower_hp_text_area(bar_img, tower_name):
    h, w = bar_img.shape[:2]

    x0 = int(w * 0.22)
    x1 = int(w * 0.78)

    if tower_name.startswith("enemy_"):
      y0 = 3
      y1 = int(h * 0.6)
    else:
      x0 = int(w * 0.28)
      y0 = int(h * 0.2)
      y1 = h

    if tower_name.endswith("king"):
        x0 = int(w * 0.4)


    return bar_img[y0:y1, x0:x1]
