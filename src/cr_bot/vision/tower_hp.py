from dataclasses import dataclass

from cr_bot.domain.constants import FULL_TOWER_HP, KING_TOWER_HP
from cr_bot.vision.image_utils import crop, detect_if_king_tower_activated, detect_if_support_tower_alive
from cr_bot.vision.tower_hp_ocr import get_tower_hp_ocr
from cr_bot.vision.yolo_runtime import parse_box_row, load_yolo_runtime

from cr_bot.domain.rois import ROIS

OWN_TOWER_NAMES = {"own_king", "own_support_left", "own_support_right"}


@dataclass
class TowerHPCrop:
    tower_name: str
    image: object
    mode: str


def extract_tower_hp(frame, yolo_boxes=None, debug_steps_by_tower=None, support_tower_yolo_boxes=None, *, rois=None):
    if yolo_boxes is None: # Live gameplay
        pause_own_tower_hp = has_blocking_emotes(support_tower_yolo_boxes)
        if rois is None:
            tower_crops = extract_tower_hp_crops(frame)
        else:
            tower_crops = extract_tower_hp_crops(frame, rois=rois)
        towers_hp = {tower_crop.tower_name: None for tower_crop in tower_crops}
        pending_crops = {}

        for tower_crop in tower_crops:
            tower_debug = _tower_debug(debug_steps_by_tower, tower_crop.tower_name)
            if pause_own_tower_hp and tower_crop.tower_name in OWN_TOWER_NAMES:
                if tower_debug is not None:
                    tower_debug["ocr_missing_reason"] = "blocked_by_emote"
            else:
                pending_crops[tower_crop.tower_name] = tower_crop.image

        predictions = get_tower_hp_ocr().predict_batch(
            pending_crops,
            debug_steps_by_tower=debug_steps_by_tower,
        )
        for tower_name, prediction in predictions.items():
            towers_hp[tower_name] = prediction.value

        if rois is None:
            king_tower_activated = detect_if_king_tower_activated(frame)
        else:
            king_tower_activated = detect_if_king_tower_activated(frame, rois=rois)

        if not pause_own_tower_hp and not king_tower_activated["own_king_activated"]:
            towers_hp["own_king"] = KING_TOWER_HP
        if not king_tower_activated["enemy_king_activated"]:
            towers_hp["enemy_king"] = KING_TOWER_HP

        if support_tower_yolo_boxes is not None:
            support_tower_alive = detect_if_support_tower_alive_from_yolo(support_tower_yolo_boxes)
        else:
            if rois is None:
                support_tower_alive = detect_if_support_tower_alive(frame)
            else:
                support_tower_alive = detect_if_support_tower_alive(frame, rois=rois)

        if not pause_own_tower_hp and not support_tower_alive["support_left_activated"]:
            towers_hp["own_support_left"] = 0
        if not pause_own_tower_hp and not support_tower_alive["support_right_activated"]:
            towers_hp["own_support_right"] = 0
        if not support_tower_alive["enemy_support_left_activated"]:
            towers_hp["enemy_support_left"] = 0
        if not support_tower_alive["enemy_support_right_activated"]:
            towers_hp["enemy_support_right"] = 0

        return towers_hp

    crops, result, paused_towers = extract_tower_hp_crops_from_yolo(
        frame,
        yolo_boxes,
        debug_steps_by_tower=debug_steps_by_tower,
    )
    pending_crops = {
        tower_crop.tower_name: tower_crop.image
        for tower_crop in crops
        if tower_crop.tower_name not in paused_towers
    }
    predictions = get_tower_hp_ocr().predict_batch(
        pending_crops,
        debug_steps_by_tower=debug_steps_by_tower,
    )
    for tower_name, prediction in predictions.items():
        value = prediction.value
        if value in (None, 0):
            value = None if "support" in tower_name else FULL_TOWER_HP[tower_name]
        result[tower_name] = value
    return result


def _tower_debug(debug_steps_by_tower, tower_name):
    if debug_steps_by_tower is None:
        return None
    return debug_steps_by_tower.setdefault(tower_name, {})


def extract_tower_hp_crops(frame, *, rois=None):
    if rois is None:
        return [
            TowerHPCrop("enemy_king", crop(frame, ROIS["opponent_king_health_text"]), "fixed_roi"),
            TowerHPCrop("own_king", crop(frame, ROIS["player_king_health_text"]), "fixed_roi"),
            TowerHPCrop("enemy_support_left", crop(frame, ROIS["opponent_left_support_health_text"]), "fixed_roi"),
            TowerHPCrop("enemy_support_right", crop(frame, ROIS["opponent_right_support_health_text"]), "fixed_roi"),
            TowerHPCrop("own_support_left", crop(frame, ROIS["player_left_support_health_text"]), "fixed_roi"),
            TowerHPCrop("own_support_right", crop(frame, ROIS["player_right_support_health_text"]), "fixed_roi"),
        ]
    from cr_bot.vision.roi_adapt import resolve_crop as _resolve_crop

    return [
        TowerHPCrop("enemy_king", _resolve_crop(frame, "opponent_king_health_text", rois=rois), "fixed_roi"),
        TowerHPCrop("own_king", _resolve_crop(frame, "player_king_health_text", rois=rois), "fixed_roi"),
        TowerHPCrop("enemy_support_left", _resolve_crop(frame, "opponent_left_support_health_text", rois=rois), "fixed_roi"),
        TowerHPCrop("enemy_support_right", _resolve_crop(frame, "opponent_right_support_health_text", rois=rois), "fixed_roi"),
        TowerHPCrop("own_support_left", _resolve_crop(frame, "player_left_support_health_text", rois=rois), "fixed_roi"),
        TowerHPCrop("own_support_right", _resolve_crop(frame, "player_right_support_health_text", rois=rois), "fixed_roi"),
    ]


def extract_tower_hp_crops_from_yolo(frame, yolo_boxes, debug_steps_by_tower=None):
    _, _, idx2unit = load_yolo_runtime()
    pause_own_tower_hp = has_blocking_emotes(yolo_boxes, idx2unit=idx2unit)

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

    ally_queens = [d for d in queen_towers if d["team"] == "ally"]
    enemy_queens = [d for d in queen_towers if d["team"] == "enemy"]

    own_support_left, own_support_right = assign_support_towers_by_position(
        ally_queens,
        ROIS["player_left_support_tower"],
        ROIS["player_right_support_tower"],
    )
    enemy_support_left, enemy_support_right = assign_support_towers_by_position(
        enemy_queens,
        ROIS["opponent_left_support_tower"],
        ROIS["opponent_right_support_tower"],
    )

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
    crops = []
    paused_towers = set()

    for tower_name, tower in towers.items():
        tower_debug = _tower_debug(debug_steps_by_tower, tower_name)
        if pause_own_tower_hp and tower_name in OWN_TOWER_NAMES:
            result[tower_name] = None
            paused_towers.add(tower_name)
            if tower_debug is not None:
                tower_debug["ocr_missing_reason"] = "blocked_by_emote"
            continue

        if tower is None:
            if "support" in tower_name:
                result[tower_name] = 0
            else:
                result[tower_name] = FULL_TOWER_HP[tower_name]
            if tower_debug is not None:
                tower_debug["ocr_missing_reason"] = "no_tower_detection"
            continue

        if "king" in tower_name:
            bar = king_matches.get(id(tower))
        else:
            bar = queen_matches.get(id(tower))

        if bar is None:
            result[tower_name] = FULL_TOWER_HP[tower_name]
            if tower_debug is not None:
                tower_debug["ocr_missing_reason"] = "no_bar_detection"
            continue

        bar_roi = (
            int(bar["x1"]),
            int(bar["y1"]),
            max(1, int(bar["x2"] - bar["x1"])),
            max(1, int(bar["y2"] - bar["y1"])),
        )
        bar_img = crop(frame, bar_roi)
        text_img = crop_tower_hp_text_area(bar_img, tower_name)
        crops.append(TowerHPCrop(tower_name, text_img, "yolo_bar"))

    return crops, result, paused_towers


def has_blocking_emotes(yolo_boxes, idx2unit=None):
    if yolo_boxes is None:
        return False

    if idx2unit is None:
        _, _, idx2unit = load_yolo_runtime()

    emote_count = 0
    for row in yolo_boxes:
        x1, y1, x2, y2, track_id, conf, cls, team = parse_box_row(row)
        if idx2unit[int(cls)] == "emote":
            emote_count += 1
            if emote_count >= 2:
                return True

    return False


def detect_if_support_tower_alive_from_yolo(yolo_boxes):
    _, _, idx2unit = load_yolo_runtime()

    ally_queens = []
    enemy_queens = []

    for row in yolo_boxes:
        x1, y1, x2, y2, track_id, conf, cls, team = parse_box_row(row)
        class_name = idx2unit[int(cls)]
        if class_name != "queen-tower":
            continue

        det = {
            "team": "enemy" if int(team) == 1 else "ally",
            "cx": float((x1 + x2) / 2.0),
        }
        if det["team"] == "ally":
            ally_queens.append(det)
        else:
            enemy_queens.append(det)

    own_left, own_right = assign_support_towers_by_position(
        ally_queens,
        ROIS["player_left_support_tower"],
        ROIS["player_right_support_tower"],
    )
    enemy_left, enemy_right = assign_support_towers_by_position(
        enemy_queens,
        ROIS["opponent_left_support_tower"],
        ROIS["opponent_right_support_tower"],
    )

    return {
        "support_left_activated": own_left is not None,
        "support_right_activated": own_right is not None,
        "enemy_support_left_activated": enemy_left is not None,
        "enemy_support_right_activated": enemy_right is not None,
    }


def assign_support_towers_by_position(detections, left_roi, right_roi):
    left_cx = left_roi[0] + left_roi[2] / 2.0
    right_cx = right_roi[0] + right_roi[2] / 2.0

    left_det = None
    right_det = None
    left_distance = None
    right_distance = None

    for det in detections:
        dist_left = abs(det["cx"] - left_cx)
        dist_right = abs(det["cx"] - right_cx)

        if dist_left <= dist_right:
            if left_distance is None or dist_left < left_distance:
                left_det = det
                left_distance = dist_left
        else:
            if right_distance is None or dist_right < right_distance:
                right_det = det
                right_distance = dist_right

    return left_det, right_det


def squared_distance(a, b):
      ax, ay = center(a)
      bx, by = center(b)
      return (ax - bx) ** 2 + (ay - by) ** 2

def center(det):
    return det["cx"], det["cy"]



def match_bars_to_towers(towers, bars):
      max_match_distance_sq = 300 ** 2
      candidates = []

      for tower_idx, tower in enumerate(towers):
          for bar_idx, bar in enumerate(bars):
              if bar["team"] != tower["team"]:
                  continue

              distance = squared_distance(tower, bar)
              if distance > max_match_distance_sq:
                  continue

              candidates.append((distance, tower_idx, bar_idx))

      candidates.sort(key=lambda x: x[0])

      used_towers = set()
      used_bars = set()
      matches = {id(tower): None for tower in towers}

      for _, tower_idx, bar_idx in candidates:
          if tower_idx in used_towers or bar_idx in used_bars:
              continue

          matches[id(towers[tower_idx])] = bars[bar_idx]
          used_towers.add(tower_idx)
          used_bars.add(bar_idx)

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
