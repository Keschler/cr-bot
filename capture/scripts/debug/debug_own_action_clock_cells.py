from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractors.match_state import game_end_from_result, game_start
from extractors.timer import total_remaining_seconds
from features.action_space import ACTION_GRID
from main import has_visible_match_timer, normalize_frame, process_frame
from state_builder import build_game_state
from trackers.enemy_cards import EnemyCardTracker
from trackers.match_clock import MatchClockFilter
from trackers.own_actions import OwnActionTracker
from trackers.tower_hp_filter import TowerHPFilter
from vision.yolo_runtime import build_detector


TARGET_ACTIONS = [
    ("skeletons", 0, None, 292),
    ("hog-rider", 0, None, 288),
    ("ice-spirit", 1, None, 287),
    ("cannon-cart", 0, (1, 18), 286),
    ("ice-spirit", None, (2, 17), 285),
    ("ice-golem", 3, (4, 13), 283),
    ("cannon", 1, (8, 22), 279),
    ("ice-golem", 3, (10, 19), 275),
    ("musketeer", 0, (12, 18), 271),
    ("skeletons", 3, (9, 17), 269),
    ("musketeer", None, (7, 27), 269),
    ("fireball", 2, (3, 8), 258.9),
    ("ice-spirit", 3, (1, 21), 248),
    ("wizard", 2, (7, 18), 246),
    ("ice-spirit", None, (0, 23), 246),
    ("hog-rider", 0, (2, 14), 245),
]


def _safe_cell(cell):
    if cell is None:
        return "none"
    return f"{cell[0]}_{cell[1]}"


def _action_key(action):
    return (
        action["card"],
        action["slot_idx"],
        action["cell"],
        round(float(action["time_left_s"]), 1),
    )


def _matches_target(action, target):
    card, slot, cell, time_left = target
    return (
        action["card"] == card
        and action["slot_idx"] == slot
        and action["cell"] == cell
        and abs(float(action["time_left_s"]) - float(time_left)) <= 0.15
    )


def _draw_grid(img, arena_px):
    ax, ay, aw, ah = arena_px
    x0 = int(round(ax + ACTION_GRID.x0 * aw))
    y0 = int(round(ay + ACTION_GRID.y0 * ah))
    x1 = int(round(ax + ACTION_GRID.x1 * aw))
    y1 = int(round(ay + ACTION_GRID.y1 * ah))

    for col in range(ACTION_GRID.cols + 1):
        x = int(round(x0 + col / ACTION_GRID.cols * (x1 - x0)))
        cv2.line(img, (x, y0), (x, y1), (40, 40, 40), 1)
    for row in range(ACTION_GRID.rows + 1):
        y = int(round(y0 + row / ACTION_GRID.rows * (y1 - y0)))
        cv2.line(img, (x0, y), (x1, y), (40, 40, 40), 1)

    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 2)


def _draw_cell(img, arena_px, cell, label, color):
    if cell is None:
        return
    col, row = cell
    cx, cy = ACTION_GRID.cell_to_pixel_center(col, row, arena_px)
    ax, ay, aw, ah = arena_px
    cell_w = ACTION_GRID.width * aw / ACTION_GRID.cols
    cell_h = ACTION_GRID.height * ah / ACTION_GRID.rows
    x0 = int(round(cx - cell_w / 2))
    y0 = int(round(cy - cell_h / 2))
    x1 = int(round(cx + cell_w / 2))
    y1 = int(round(cy + cell_h / 2))
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
    cv2.drawMarker(img, (int(round(cx)), int(round(cy))), color, cv2.MARKER_CROSS, 22, 2)
    cv2.putText(img, label, (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_boxes(img, result, action):
    for clock in result["clock_boxes"]:
        color = (255, 255, 0) if clock["team"] == "ally" else (0, 160, 255)
        x1, y1, x2, y2 = (int(round(clock[key])) for key in ("x1", "y1", "x2", "y2"))
        cx, cy = int(round(clock["center_x"])), int(round(clock["center_y"]))
        cell = ACTION_GRID.pixel_to_cell(clock["center_x"], clock["center_y"], result["arena_px"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, (cx, cy), 5, color, -1)
        cv2.putText(img, f"clock:{clock['team']} {cell}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    for match in result["matches"]:
        troop = match.troop
        if troop.team != "ally":
            continue
        color = (0, 255, 0)
        x1, y1, x2, y2 = int(troop.x1), int(troop.y1), int(troop.x2), int(troop.y2)
        cell = ACTION_GRID.pixel_to_cell(troop.center_x, troop.center_y, result["arena_px"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, (int(troop.center_x), int(troop.center_y)), 5, color, -1)
        cv2.putText(
            img,
            f"id={troop.track_id} {troop.class_name} {cell}",
            (x1, min(img.shape[0] - 8, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )

    text = (
        f"action card={action['card']} slot={action['slot_idx']} "
        f"cell={action['cell']} time_left={action['time_left_s']}"
    )
    cv2.rectangle(img, (10, 8), (min(img.shape[1] - 10, 920), 48), (0, 0, 0), -1)
    cv2.putText(img, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def _render_action_debug(frame, result, action):
    overlay = frame.copy()
    _draw_grid(overlay, result["arena_px"])
    _draw_boxes(overlay, result, action)
    _draw_cell(overlay, result["arena_px"], action["cell"], "selected", (0, 0, 255))
    return overlay


def run(video_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector()
    enemy_card_tracker = EnemyCardTracker()
    own_action_tracker = OwnActionTracker()
    match_clock_filter = MatchClockFilter()
    tower_hp_filter = TowerHPFilter()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    game_started = False
    not_in_game_streak = 0
    frame_idx = 0
    saved = []
    seen_actions = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        frame = normalize_frame(frame)
        result = process_frame(frame, detector, show_rois=False)
        video_time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if not game_started and (game_start(frame) or has_visible_match_timer(result)):
            game_started = True
            result["towers_hp"] = tower_hp_filter.update(result["towers_hp"])
            match_clock_filter.initialise(result["time_left_s"], video_time_s)
            enemy_card_tracker.start_match(
                result["time_left_s"],
                result["total_remaining_s"],
                now_s=video_time_s,
            )
        elif not game_started:
            continue
        else:
            if match_clock_filter.initialised:
                filtered_time_left_s = match_clock_filter.update(result["time_left_s"], video_time_s)
                result["time_left_s"] = filtered_time_left_s
                result["total_remaining_s"] = total_remaining_seconds(filtered_time_left_s, result["overtime"])
            else:
                match_clock_filter.initialise(result["time_left_s"], video_time_s)
            result["towers_hp"] = tower_hp_filter.update(result["towers_hp"])

        game_state = build_game_state(
            result,
            seen_enemy_cards=list(enemy_card_tracker.confirmed_seen_cards),
            elixir_enemy_est=enemy_card_tracker.elixir_enemy_est,
            game_started=game_started,
        )
        before_len = len(own_action_tracker.actions)
        own_action_tracker.update(
            game_state,
            result["arena_px"],
            frame=frame,
            clock_boxes=result["clock_boxes"],
            elixir_change=result["elixir_change"],
            video_time_s=video_time_s,
        )
        new_actions = own_action_tracker.actions[before_len:]

        enemy_card_tracker.reconcile_own_actions(own_action_tracker.actions, arena_px=result["arena_px"])
        enemy_card_tracker.update(
            result["total_remaining_s"],
            result["matches"],
            now_s=video_time_s,
            clock_boxes=result["clock_boxes"],
            own_actions=own_action_tracker.actions,
            arena_px=result["arena_px"],
        )

        for action in new_actions:
            key = _action_key(action)
            if key in seen_actions:
                continue
            seen_actions.add(key)
            overlay = _render_action_debug(frame, result, action)
            target_suffix = "target" if any(_matches_target(action, target) for target in TARGET_ACTIONS) else "extra"
            filename = (
                f"{len(saved):02d}_frame_{frame_idx:04d}_tl_{float(action['time_left_s']):05.1f}_"
                f"{action['card']}_slot_{action['slot_idx']}_cell_{_safe_cell(action['cell'])}_{target_suffix}.jpg"
            )
            cv2.imwrite(str(output_dir / filename), overlay)
            saved.append((filename, action))

        if game_end_from_result(result):
            not_in_game_streak += 1
            if not_in_game_streak >= 20:
                break
        else:
            not_in_game_streak = 0

    cap.release()

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        for filename, action in saved:
            fh.write(
                f"{filename}: card={action['card']} slot={action['slot_idx']} "
                f"cell={action['cell']} time_left={action['time_left_s']}\n"
            )

    print(f"wrote {len(saved)} debug images to {output_dir}")
    print(f"summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "assets/pictures/10_fps_gameplay.mp4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "debug_output/own_action_clock_cells_10_fps_gameplay")
    args = parser.parse_args()
    run(args.video, args.output_dir)


if __name__ == "__main__":
    main()
