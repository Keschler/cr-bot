from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KATACR_ROOT = ROOT / "vendor/external/KataCR"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from extractors.spell_deploy import SpellDeployLocator
from features.action_space import ACTION_GRID
from katacr.build_dataset.utils.split_part import process_part, ratio2name
from main import normalize_frame


FAILED_WRONG_DETECTION_TARGETS = [
    ("poison", 4, 87),
    ("clone", 3, 295),
    ("arrows", 3, 431),
]

CONFIRMED_WRONG_DETECTION_TARGETS = [
    ("fireball", 4, 165),
    ("giant-snowball", 2, 364),
    ("zap", 2, 476),
]

PRESETS = {
    "failed-wrong-detections": FAILED_WRONG_DETECTION_TARGETS,
    "confirmed-wrong-detections": CONFIRMED_WRONG_DETECTION_TARGETS,
    "all-wrong-detections": FAILED_WRONG_DETECTION_TARGETS + CONFIRMED_WRONG_DETECTION_TARGETS,
}


def _parse_target(value: str):
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("target must be formatted as card:cost:frame")
    card_name, cost, frame_idx = parts
    try:
        return card_name, int(cost), int(frame_idx)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target cost and frame must be integers") from exc


def _arena_px(frame):
    if ratio2name(frame) is None:
        raise ValueError(f"unsupported frame shape: {frame.shape}")
    _, box_params = process_part(frame, 2, verbose=True)
    fx, fy, fw, fh = box_params
    frame_h, frame_w = frame.shape[:2]
    return (
        int(frame_w * fx),
        int(frame_h * fy),
        int(frame_w * fw),
        int(frame_h * fh),
    )


def _seek_frame(cap, target_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read frame {target_frame}")
    return normalize_frame(frame)


def _candidate_search_rect(locator, arena_px, purple_mask, candidate):
    region = locator._purple_elixir_search_region(purple_mask, arena_px, candidate)
    if region is None:
        return None
    return region


def _contour_scores(locator, purple_mask, arena_px, candidate, elixir_cost):
    region = _candidate_search_rect(locator, arena_px, purple_mask, candidate)
    if region is None:
        return []
    x0, y0, x1, y1, ellipse_crop = region
    roi = cv2.bitwise_and(purple_mask[y0:y1, x0:x1], ellipse_crop)
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expected_digit = str(int(elixir_cost))
    rows = []
    target_x = (x1 - x0) / 2.0
    target_y = (y1 - y0) * 0.48
    norm_x = max(1.0, (x1 - x0) * 0.50)
    norm_y = max(1.0, (y1 - y0) * 0.75)

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            aspect = 0.0
        else:
            aspect = w / h
        accepted = 12 <= area <= 1800 and h >= 8 and w >= 5 and 0.25 <= aspect <= 1.8
        template_score = 0.0
        score = 0.0
        if accepted:
            patch = roi[
                max(0, y - 3):min(roi.shape[0], y + h + 3),
                max(0, x - 3):min(roi.shape[1], x + w + 3),
            ]
            template_score = locator._digit_template_score(patch, expected_digit)
            area_score = min(1.0, area / 140.0)
            contour_cx = x + w / 2.0
            contour_cy = y + h / 2.0
            distance = np.hypot(
                (contour_cx - target_x) / norm_x,
                (contour_cy - target_y) / norm_y,
            )
            proximity = max(0.0, 1.0 - distance)
            score = 0.45 * template_score + 0.30 * area_score + 0.25 * proximity
        rows.append({
            "x": x0 + x,
            "y": y0 + y,
            "w": w,
            "h": h,
            "area": area,
            "aspect": aspect,
            "accepted": accepted,
            "template": template_score,
            "score": score,
        })
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def _write_labeled(path, image, label):
    output = image.copy()
    cv2.rectangle(output, (8, 8), (min(output.shape[1] - 8, 900), 44), (0, 0, 0), -1)
    cv2.putText(output, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), output)


def _draw_grid(overlay, arena_px):
    ax, ay, aw, ah = arena_px
    x0 = int(round(ax + ACTION_GRID.x0 * aw))
    y0 = int(round(ay + ACTION_GRID.y0 * ah))
    x1 = int(round(ax + ACTION_GRID.x1 * aw))
    y1 = int(round(ay + ACTION_GRID.y1 * ah))
    for col in range(ACTION_GRID.cols + 1):
        x = int(round(x0 + col / ACTION_GRID.cols * (x1 - x0)))
        cv2.line(overlay, (x, y0), (x, y1), (45, 45, 45), 1)
    for row in range(ACTION_GRID.rows + 1):
        y = int(round(y0 + row / ACTION_GRID.rows * (y1 - y0)))
        cv2.line(overlay, (x0, y), (x1, y), (45, 45, 45), 1)


def run(video_path: Path, output_dir: Path, targets):
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    locator = SpellDeployLocator()
    summary = []

    for card_name, elixir_cost, frame_idx in targets:
        frame = _seek_frame(cap, frame_idx)
        arena_px = _arena_px(frame)
        debug, candidates = locator.render_debug(frame, arena_px, card_name, elixir_cost)
        masks = locator._build_masks(frame, arena_px)
        purple_mask = masks["purple_mask"]
        confirmed = locator._confirm_with_elixir_used_display(frame, arena_px, candidates, elixir_cost)

        prefix = f"{frame_idx:04d}_{card_name}"
        for name, image in debug.items():
            _write_labeled(output_dir / f"{prefix}_{name}.jpg", image, f"{card_name} frame={frame_idx} {name}")

        overlay = frame.copy()
        _draw_grid(overlay, arena_px)
        cv2.rectangle(
            overlay,
            (int(arena_px[0]), int(arena_px[1])),
            (int(arena_px[0] + arena_px[2]), int(arena_px[1] + arena_px[3])),
            (255, 255, 0),
            2,
        )

        summary.append(f"{card_name} frame={frame_idx} elixir_cost={elixir_cost} candidates={len(candidates)}")
        summary.append(f"  confirmed_index={candidates.index(confirmed) if confirmed in candidates else None}")

        for idx, candidate in enumerate(candidates):
            score = locator._purple_elixir_score(purple_mask, arena_px, candidate, elixir_cost)
            region = _candidate_search_rect(locator, arena_px, purple_mask, candidate)
            if region is None:
                continue
            x0, y0, x1, y1, ellipse_crop = region
            ax, ay = int(arena_px[0]), int(arena_px[1])
            cell = ACTION_GRID.pixel_to_cell(candidate.center_x, candidate.center_y, arena_px)
            color = (0, 255, 0) if candidate is confirmed else ((0, 180, 255) if idx == 0 else (0, 120, 255))
            cv2.rectangle(overlay, (ax + x0, ay + y0), (ax + x1, ay + y1), color, 2)
            cv2.putText(
                overlay,
                f"{idx} p={score:.3f} cell={cell}",
                (ax + x0, max(22, ay + y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            crop = masks["roi"][y0:y1, x0:x1]
            crop = cv2.bitwise_and(crop, crop, mask=ellipse_crop)
            crop_mask = cv2.bitwise_and(purple_mask[y0:y1, x0:x1], ellipse_crop)
            crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)
            _write_labeled(output_dir / f"{prefix}_cand_{idx}_search_roi.jpg", crop, f"{card_name} cand={idx} score={score:.3f}")
            _write_labeled(output_dir / f"{prefix}_cand_{idx}_purple_roi.jpg", crop_mask, f"{card_name} cand={idx} purple score={score:.3f}")

            rows = _contour_scores(locator, purple_mask, arena_px, candidate, elixir_cost)
            summary.append(
                "  cand={idx} cell={cell} center=({cx:.1f},{cy:.1f}) conf={conf:.3f} "
                "arc={arc:.3f} purple={purple:.3f} recomputed_purple={score:.3f} "
                "search_roi=({x0},{y0})-({x1},{y1}) contours={contours}".format(
                    idx=idx,
                    cell=cell,
                    cx=candidate.center_x,
                    cy=candidate.center_y,
                    conf=candidate.confidence,
                    arc=candidate.arc_score,
                    purple=candidate.purple_score,
                    score=score,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    contours=len(rows),
                )
            )
            for row_idx, row in enumerate(rows[:6]):
                summary.append(
                    "    contour={idx} xywh=({x},{y},{w},{h}) area={area:.1f} aspect={aspect:.2f} "
                    "accepted={accepted} template={template:.3f} score={score:.3f}".format(
                        idx=row_idx,
                        **row,
                    )
                )

        _write_labeled(output_dir / f"{prefix}_purple_search_overlay.jpg", overlay, f"{card_name} frame={frame_idx} purple search")
        summary.append("")

    (output_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"wrote debug images to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("assets/pictures/10_fps_wrong_detections.mp4"))
    parser.add_argument("--output-dir", type=Path, default=Path("debug_output/spell_purple_wrong_detections"))
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="failed-wrong-detections",
        help="named set of spell deploy frames to render",
    )
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_target,
        help="explicit target formatted as card:cost:frame; can be passed multiple times",
    )
    args = parser.parse_args()
    targets = args.target if args.target else PRESETS[args.preset]
    run(args.video, args.output_dir, targets)


if __name__ == "__main__":
    main()
