from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = ROOT / "capture"
sys.path.insert(0, str(CAPTURE_ROOT))
os.chdir(CAPTURE_ROOT)

from main import process_frame
from state_builder import build_game_state
from vision.yolo_runtime import build_detector
from extractors.match_state import in_game, game_end_from_result
from extractors.timer import total_remaining_seconds
from trackers.enemy_cards import EnemyCardTracker
from trackers.match_clock import MatchClockFilter
from trackers.tower_hp_filter import TowerHPFilter


def jpeg_roundtrip(frame):
  ok, encoded = cv2.imencode(".jpg", frame)
  if not ok:
      raise RuntimeError("Failed to encode frame as JPEG")
  decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
  if decoded is None:
      raise RuntimeError("Failed to decode JPEG frame")
  return decoded, encoded


def unit_to_dict(match):
  troop = match.troop
  return {
      "class_name": troop.class_name,
      "team": troop.team,
      "confidence": float(troop.confidence),
      "bbox": [
          float(troop.x1),
          float(troop.y1),
          float(troop.x2),
          float(troop.y2),
      ],
      "center": [
          float(troop.center_x),
          float(troop.center_y),
      ],
      "estimated_hp": None if troop.estimated_hp is None else float(troop.estimated_hp),
  }


def state_to_row(video_id, frame_idx, video_time_s, image_path, state):
  return {
      "video_id": video_id,
      "frame_idx": frame_idx,
      "video_time_s": video_time_s,
      "image_path": str(image_path),
      "started": state.started,
      "match_time_s": state.total_remaining_s,
      "hand": state.hud.hand_cards,
      "next_card": state.hud.next_card,
      "elixir_self": float(state.hud.elixir_self),
      "overtime": state.hud.overtime,
      "tower_hp_self": state.hud.tower_hp_self,
      "tower_hp_enemy": state.hud.tower_hp_enemy,
      "own_king_active": state.own_king_active,
      "enemy_king_active": state.enemy_king_active,
      "own_units": [unit_to_dict(m) for m in state.own_units],
      "enemy_units": [unit_to_dict(m) for m in state.enemy_units],
      "seen_enemy_cards": state.seen_enemy_cards,
      "elixir_enemy_est": state.elixir_enemy_est,
  }


def main():
  video_path = ROOT / "dataset_generation/data/video_clips/output_2m00s_2m30s_h264.mp4"
  out_path = ROOT / "dataset_generation/data/frame_states/clip/states.jsonl"
  frames_dir = ROOT / "dataset_generation/data/frame_states/clip/frames"

  out_path.parent.mkdir(parents=True, exist_ok=True)
  frames_dir.mkdir(parents=True, exist_ok=True)

  detector = build_detector()
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
      raise FileNotFoundError(f"Could not open video: {video_path}")

  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  frame_idx = 0
  game_started = False
  not_in_game_streak = 0
  enemy_card_tracker = EnemyCardTracker()
  match_clock_filter = MatchClockFilter()
  tower_hp_filter = TowerHPFilter()

  with out_path.open("w", encoding="utf-8") as f:
      while True:
          ok, frame = cap.read()
          if not ok:
              break


          # Start sparse while debugging. Later you can process every frame.
          if frame_idx % 20 != 0:
              frame_idx += 1
              continue
          frame = cv2.resize(frame, (1080, 2400), interpolation=cv2.INTER_LINEAR)
          frame, encoded_frame = jpeg_roundtrip(frame)
          video_time_s = frame_idx / fps

          if not game_started and not in_game(frame):
              frame_idx += 1
              continue

          result = process_frame(frame, detector, show_rois=False, yolo_tower_hp_detections=True)
          result["towers_hp"] = tower_hp_filter.update(result["towers_hp"])

          if not match_clock_filter.initialised:
              match_clock_filter.initialise(result["time_left_s"], video_time_s)

          filtered_time_left_s = match_clock_filter.update(result["time_left_s"], video_time_s)
          if filtered_time_left_s is None:
              frame_idx += 1
              continue
          result["time_left_s"] = filtered_time_left_s
          result["total_remaining_s"] = total_remaining_seconds(filtered_time_left_s, result["overtime"])

          if not game_started:
              game_started = True
              enemy_card_tracker.start_match(
                  result["time_left_s"],
                  result["total_remaining_s"],
                  now_s=video_time_s,
              )
          else:
              enemy_card_tracker.update(
                  result["total_remaining_s"],
                  result["matches"],
                  now_s=video_time_s,
              )

          if game_end_from_result(result):
              not_in_game_streak += 1
              if not_in_game_streak >= 20:
                  game_started = False
                  not_in_game_streak = 0
                  enemy_card_tracker = EnemyCardTracker()
                  match_clock_filter = MatchClockFilter()
                  tower_hp_filter = TowerHPFilter()
                  frame_idx += 1
                  continue
          else:
              not_in_game_streak = 0

          state = build_game_state(
              result,
              seen_enemy_cards=list(enemy_card_tracker.confirmed_seen_cards),
              elixir_enemy_est=enemy_card_tracker.elixir_enemy_est,
              game_started=game_started,
          )

          image_path = frames_dir / f"{frame_idx:06d}.jpg"
          image_path.write_bytes(encoded_frame.tobytes())

          row = state_to_row(
              video_id="clip",
              frame_idx=frame_idx,
              video_time_s=video_time_s,
              image_path=image_path.relative_to(ROOT),
              state=state,
          )
          row["arena_px"] = result["arena_px"]

          f.write(json.dumps(row) + "\n")
          frame_idx += 1

  cap.release()


if __name__ == "__main__":
    main()
