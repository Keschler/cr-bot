from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from card_metadata import CARD_METADATA
from features.action_space import ACTION_GRID
from main import normalize_frame, process_frame
from trackers.own_actions import OwnActionTracker
from vision.yolo_runtime import build_detector


TARGET_ACTIONS_BY_FRAME = {
    343: [
        {"card": "musketeer", "slot_idx": 3, "cell": (4, 18), "time_left_s": 128.1},
        {"card": "fireball", "slot_idx": 0, "cell": (15, 22), "time_left_s": 120.0},
        {"card": "giant-snowball", "slot_idx": 2, "cell": (15, 22), "time_left_s": 120.0},
    ],
    441: [
        {"card": "musketeer", "slot_idx": 3, "cell": (4, 18), "time_left_s": 128.1},
        {"card": "fireball", "slot_idx": 0, "cell": (15, 22), "time_left_s": 120.0},
        {"card": "giant-snowball", "slot_idx": 2, "cell": (15, 22), "time_left_s": 120.0},
        {"card": "cannon", "slot_idx": 1, "cell": (12, 29), "time_left_s": 120.0},
    ],
    470: [
        {"card": "musketeer", "slot_idx": 3, "cell": (4, 18), "time_left_s": 128.1},
        {"card": "fireball", "slot_idx": 0, "cell": (15, 22), "time_left_s": 120.0},
        {"card": "giant-snowball", "slot_idx": 2, "cell": (15, 22), "time_left_s": 120.0},
        {"card": "cannon", "slot_idx": 1, "cell": (12, 29), "time_left_s": 120.0},
        {"card": "skeletons", "slot_idx": 2, "cell": (9, 22), "time_left_s": 120.0},
    ],
}


def _seek_frame(cap, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx}")
    return normalize_frame(frame)


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
    aw = arena_px[2]
    ah = arena_px[3]
    cell_w = ACTION_GRID.width * aw / ACTION_GRID.cols
    cell_h = ACTION_GRID.height * ah / ACTION_GRID.rows
    x0 = int(round(cx - cell_w / 2))
    y0 = int(round(cy - cell_h / 2))
    x1 = int(round(cx + cell_w / 2))
    y1 = int(round(cy + cell_h / 2))
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
    cv2.drawMarker(img, (int(round(cx)), int(round(cy))), color, cv2.MARKER_CROSS, 22, 2)
    cv2.putText(img, label, (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _render_action_overlay(frame, arena_px, action):
    overlay = frame.copy()
    _draw_grid(overlay, arena_px)
    _draw_cell(overlay, arena_px, action["cell"], "selected", (0, 0, 255))
    text = (
        f"card={action['card']} slot={action['slot_idx']} "
        f"cell={action['cell']} time_left={action['time_left_s']:.1f}"
    )
    cv2.rectangle(overlay, (10, 8), (min(overlay.shape[1] - 10, 920), 48), (0, 0, 0), -1)
    cv2.putText(overlay, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def _choose_candidate(candidates, action_cell, arena_px):
    if not candidates:
        return None
    for candidate in candidates:
        cell = ACTION_GRID.pixel_to_cell(candidate.center_x, candidate.center_y, arena_px)
        if cell == action_cell:
            return candidate
    return candidates[0]


def _purple_region(candidate, arena_px, purple_mask):
    arena_x = int(round(arena_px[0]))
    arena_y = int(round(arena_px[1]))
    cx = float(candidate.center_x - arena_x)
    cy = float(candidate.center_y - arena_y)
    radius_x = float(candidate.radius_x_px or candidate.radius_px or 0.0)
    radius_y = float(candidate.radius_y_px or candidate.radius_px or 0.0)

    x0 = max(0, int(round(cx - radius_x * 0.58)))
    x1 = min(purple_mask.shape[1], int(round(cx + radius_x * 0.58)))
    y0 = max(0, int(round(cy - radius_y * 1.35)))
    y1 = min(purple_mask.shape[0], int(round(cy - radius_y * 0.02)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _write_spell_debug(output_dir, prefix, frame, arena_px, action, locator):
    cost = CARD_METADATA.get(action["card"], {}).get("elixir_cost")
    candidates, masks = locator.detect(frame, arena_px, action["card"], cost)
    purple_mask = masks.get("purple_mask")
    candidate = _choose_candidate(candidates, action["cell"], arena_px)
    overlay = _render_action_overlay(frame, arena_px, action)

    if candidate is None or purple_mask is None:
        cv2.imwrite(str(output_dir / f"{prefix}_overlay.jpg"), overlay)
        return None

    purple_score = locator._purple_release_score(purple_mask, arena_px, candidate)
    center = (int(round(candidate.center_x)), int(round(candidate.center_y)))
    radius_x = int(round(candidate.radius_x_px or candidate.radius_px or 0))
    radius_y = int(round(candidate.radius_y_px or candidate.radius_px or 0))
    cv2.ellipse(overlay, center, (radius_x, radius_y), 0, 0, 360, (0, 255, 0), 2)
    cv2.putText(
        overlay,
        f"purple={purple_score:.3f}",
        (center[0] - 70, max(22, center[1] - radius_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    region = _purple_region(candidate, arena_px, purple_mask)
    if region is not None:
        x0, y0, x1, y1 = region
        ax, ay = int(arena_px[0]), int(arena_px[1])
        cv2.rectangle(overlay, (ax + x0, ay + y0), (ax + x1, ay + y1), (0, 180, 255), 2)
        roi_crop = masks["roi"][y0:y1, x0:x1]
        roi_mask = cv2.cvtColor(purple_mask[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(output_dir / f"{prefix}_purple_crop.jpg"), roi_crop)
        cv2.imwrite(str(output_dir / f"{prefix}_purple_mask.jpg"), roi_mask)

    cv2.imwrite(str(output_dir / f"{prefix}_overlay.jpg"), overlay)
    return purple_score


def run(video_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector()
    locator = OwnActionTracker().spell_deploy_locator
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    summary = []
    for frame_idx, actions in TARGET_ACTIONS_BY_FRAME.items():
        frame = _seek_frame(cap, frame_idx)
        result = process_frame(frame, detector, show_rois=False)
        arena_px = result["arena_px"]

        frame_overlay = frame.copy()
        _draw_grid(frame_overlay, arena_px)
        cv2.putText(
            frame_overlay,
            f"frame={frame_idx}",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_dir / f"frame_{frame_idx:04d}_full_overlay.jpg"), frame_overlay)

        for action in actions:
            prefix = (
                f"frame_{frame_idx:04d}_{action['card']}_"
                f"slot_{action['slot_idx']}_cell_{action['cell'][0]}_{action['cell'][1]}"
            )
            if action["card"] in {"fireball", "giant-snowball"}:
                purple_score = _write_spell_debug(output_dir, prefix, frame, arena_px, action, locator)
            else:
                overlay = _render_action_overlay(frame, arena_px, action)
                cv2.imwrite(str(output_dir / f"{prefix}_overlay.jpg"), overlay)
                purple_score = None

            summary.append(
                f"frame={frame_idx} card={action['card']} slot={action['slot_idx']} "
                f"cell={action['cell']} time_left={action['time_left_s']:.1f} "
                f"purple_score={purple_score}"
            )

    cap.release()
    (output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"wrote {len(summary)} action debug entries to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "assets/pictures/10_fps_action.mp4")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "debug_output/actions_output_selected_actions",
    )
    args = parser.parse_args()
    run(args.video, args.output_dir)


if __name__ == "__main__":
    main()
