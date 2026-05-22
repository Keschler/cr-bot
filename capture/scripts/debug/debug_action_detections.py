from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from card_metadata import CARD_METADATA
from features.action_space import ACTION_GRID
from main import normalize_frame, process_frame
from trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD
from vision.yolo_runtime import build_detector, load_yolo_runtime, parse_box_row


DEFAULT_VIDEO = REPO_ROOT / "dataset_generation/data/video_clips/10_fps_2.6HogCycle.mp4"
DEFAULT_OUTPUT_DIR = ROOT / "debug_output/action_detection_debug"


@dataclass(frozen=True)
class DebugEvent:
    side: str
    card: str
    status: str
    start_s: float | None = None
    end_s: float | None = None
    actual_card: str | None = None
    actual_time_left: float | None = None
    selected_cell: tuple[int, int] | None = None
    note: str = ""

    @property
    def slug(self) -> str:
        parts = [self.side, self.status, self.card]
        if self.actual_card:
            parts.append(f"actual_{self.actual_card}")
        if self.start_s is not None:
            parts.append(f"{self.start_s:g}_{self.end_s:g}s")
        elif self.actual_time_left is not None:
            parts.append(f"tl_{self.actual_time_left:g}")
        return "_".join(parts).replace("/", "_").replace(" ", "_")


EVENTS = [
    DebugEvent("enemy", "knight", "missing", 17, 19),
    DebugEvent("enemy", "log", "missing", 40, 41),
    DebugEvent("enemy", "dart-goblin", "missing", 42, 43),
    DebugEvent("enemy", "fireball", "missing", 44, 45),
    DebugEvent("enemy", "mega-knight", "missing", 65, 67),
    DebugEvent("enemy", "knight", "missing", 103, 105),
    DebugEvent("enemy", "fireball", "missing", 119, 121),
    DebugEvent("enemy", "ice-spirit", "missing", 122, 124),
    DebugEvent("enemy", "evo-dart-goblin", "missing", 140, 141),
    DebugEvent("enemy", "evo-knight", "missing", 141, 142),
    DebugEvent("enemy", "fireball", "missing", 154, 155),
    DebugEvent("enemy", "log", "missing", 159, 160),
    DebugEvent("enemy", "ice-spirit", "missing", 162, 163),
    DebugEvent("enemy", "zap", "wrong", 157.5, 158.0, actual_card="zap", actual_time_left=136),
    DebugEvent("own", "ice-spirit", "missing", 86, 87),
    DebugEvent("own", "cannon", "missing", 89, 91),
    DebugEvent("own", "ice-spirit", "missing", 91, 93),
    DebugEvent("own", "ice-golem", "missing", 133, 134),
    DebugEvent("own", "fireball", "missing", 157, 160),
    DebugEvent(
        "own",
        "ice-golem",
        "wrong",
        149.5,
        150.0,
        actual_card="old-musketeer",
        actual_time_left=144,
        selected_cell=(7, 19),
        note="wrong own action from log; likely caused by emote/HUD noise",
    ),
]


def canonical_card(card: str) -> str:
    card = card.replace("_", "-")
    if card.startswith("evo-"):
        return card[4:]
    return card


def expected_labels(card: str) -> set[str]:
    canonical = canonical_card(card)
    labels = {
        label
        for label, mapped_card in DIRECT_UNIT_TO_CARD.items()
        if mapped_card == canonical
    }
    if card.replace("_", "-").startswith("evo-"):
        labels = {label for label in labels if "evolution" in label} or labels
    return labels


def draw_grid(img, arena_px):
    ax, ay, aw, ah = arena_px
    x0 = int(round(ax + ACTION_GRID.x0 * aw))
    y0 = int(round(ay + ACTION_GRID.y0 * ah))
    x1 = int(round(ax + ACTION_GRID.x1 * aw))
    y1 = int(round(ay + ACTION_GRID.y1 * ah))

    for col in range(ACTION_GRID.cols + 1):
        x = int(round(x0 + col / ACTION_GRID.cols * (x1 - x0)))
        cv2.line(img, (x, y0), (x, y1), (35, 35, 35), 1)
    for row in range(ACTION_GRID.rows + 1):
        y = int(round(y0 + row / ACTION_GRID.rows * (y1 - y0)))
        cv2.line(img, (x0, y), (x1, y), (35, 35, 35), 1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 2)


def draw_cell(img, arena_px, cell, label, color):
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
    cv2.putText(img, label, (x0, max(22, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def put_lines(img, lines, origin=(12, 28), line_h=24):
    if not lines:
        return
    max_chars = max(len(line) for line in lines)
    x, y = origin
    w = min(img.shape[1] - x - 12, max(560, max_chars * 11))
    h = line_h * len(lines) + 12
    cv2.rectangle(img, (x - 6, y - 22), (x + w, y - 22 + h), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def yolo_rows(result):
    _, _, idx2unit = load_yolo_runtime()
    rows = result["yolo_boxes"].cpu().numpy() if hasattr(result["yolo_boxes"], "cpu") else result["yolo_boxes"]
    for row in rows:
        x1, y1, x2, y2, track_id, conf, cls, team = parse_box_row(row)
        label = idx2unit[int(cls)]
        yield {
            "label": label,
            "team": "enemy" if int(team) == 1 else "ally",
            "track_id": int(track_id) if track_id is not None else None,
            "confidence": float(conf),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "center_x": float((x1 + x2) / 2.0),
            "center_y": float((y1 + y2) / 2.0),
        }


def box_cell(box, arena_px):
    return ACTION_GRID.pixel_to_cell(box["center_x"], box["center_y"], arena_px)


def clock_match_score(clock, box, side):
    horizontal_gap = abs(clock["center_x"] - box["center_x"])
    vertical_gap = clock["center_y"] - box["center_y"]
    if side == "enemy":
        valid = horizontal_gap <= 90 and 10 <= vertical_gap <= 140
        target_vertical = 75
    else:
        valid = horizontal_gap <= 100 and -40 <= vertical_gap <= 220
        target_vertical = 80
    score = horizontal_gap + abs(vertical_gap - target_vertical) * 0.5
    return valid, score, horizontal_gap, vertical_gap


def best_clock_for_candidate(result, candidate, side):
    expected_team = "enemy" if side == "enemy" else "ally"
    best = None
    for clock in result["clock_boxes"]:
        if clock["team"] != expected_team or clock["confidence"] < 0.5:
            continue
        valid, score, horizontal_gap, vertical_gap = clock_match_score(clock, candidate, side)
        item = (valid, score, horizontal_gap, vertical_gap, clock)
        if best is None:
            best = item
            continue
        if valid != best[0]:
            best = item if valid else best
        elif score < best[1]:
            best = item
    return best


def selected_cell_for(event, result, candidates):
    if event.selected_cell is not None:
        return event.selected_cell, "logged selected cell"
    if not candidates:
        return None, "no candidate cell"

    if event.side == "enemy" and CARD_METADATA.get(canonical_card(event.card), {}).get("kind") == "spell":
        candidate = max(candidates, key=lambda item: item["confidence"])
        return box_cell(candidate, result["arena_px"]), "spell YOLO cell"

    best = None
    for candidate in candidates:
        clock = best_clock_for_candidate(result, candidate, event.side)
        if clock is None:
            continue
        valid, score, _horizontal_gap, _vertical_gap, clock_box = clock
        if not valid:
            continue
        if best is None or score < best[0]:
            best = (score, clock_box)
    if best is None:
        return None, "no selected clock cell"
    return ACTION_GRID.pixel_to_cell(best[1]["center_x"], best[1]["center_y"], result["arena_px"]), "best matching clock cell"


def reason_for_event(event, result, candidates, rows):
    team = "enemy" if event.side == "enemy" else "ally"
    clocks = [clock for clock in result["clock_boxes"] if clock["team"] == team and clock["confidence"] >= 0.5]
    emotes = [row for row in rows if row["label"] == "emote"]
    hand = result.get("state") or {}
    card = canonical_card(event.card)
    kind = CARD_METADATA.get(card, {}).get("kind")

    if event.status == "wrong":
        details = [f"logged wrong {event.actual_card or event.card}"]
        if event.note:
            details.append(event.note)
        if emotes:
            details.append(f"emotes visible={len(emotes)}")
        return "; ".join(details)

    if not candidates:
        labels = ", ".join(sorted(expected_labels(event.card))) or "none"
        return f"missing: no {team} YOLO candidate for labels [{labels}]"

    if event.side == "own":
        hand_cards = [value[0] if isinstance(value, tuple) else value for value in hand.values()]
        if len(emotes) >= 2:
            return f"own actions blocked: {len(emotes)} emote detections, pending actions cleared"
        if card not in hand_cards:
            return f"own card not in HUD hand at sample; hand={hand_cards}"
        if kind == "spell":
            return "own spell needs HUD drop + elixir + deploy/release marker; inspect purple/white marker outside YOLO"
        if not clocks:
            return "own unit candidate seen but no ally deploy clock >=0.5"
        return "own unit candidate seen; inspect clock link and HUD drop timing"

    if kind == "spell":
        return "enemy spell candidate seen; tracker needs enough stable frame-confirming detections"

    if not clocks:
        return "enemy unit candidate seen but no enemy deploy clock >=0.5"

    best_details = []
    for candidate in candidates:
        best = best_clock_for_candidate(result, candidate, event.side)
        if best is None:
            continue
        valid, _score, horizontal_gap, vertical_gap, _clock = best
        best_details.append(
            f"{candidate['label']} id={candidate['track_id']} valid_clock={valid} "
            f"dx={horizontal_gap:.1f} dy={vertical_gap:.1f}"
        )
    return "; ".join(best_details) or "candidate seen but no usable clock geometry"


def draw_boxes(overlay, result, event, rows):
    expected = expected_labels(event.card)
    actual = expected_labels(event.actual_card) if event.actual_card else set()

    for box in rows:
        label = box["label"]
        team = box["team"]
        is_clock = label == "clock"
        is_expected = label in expected and team == event.side.replace("own", "ally")
        is_actual = label in actual and team == event.side.replace("own", "ally")
        if is_expected:
            color = (0, 255, 255)
            thickness = 4
        elif is_actual:
            color = (0, 0, 255)
            thickness = 4
        elif is_clock:
            color = (255, 255, 0) if team == "ally" else (0, 165, 255)
            thickness = 3
        elif team == "ally":
            color = (0, 220, 0)
            thickness = 1
        else:
            color = (255, 80, 80)
            thickness = 1

        x1, y1, x2, y2 = (int(round(box[key])) for key in ("x1", "y1", "x2", "y2"))
        cell = box_cell(box, result["arena_px"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.drawMarker(overlay, (int(round(box["center_x"])), int(round(box["center_y"]))), color, cv2.MARKER_CROSS, 12, 2)
        text = f"{label}:{team} {box['confidence']:.2f} id={box['track_id']} cell={cell}"
        cv2.putText(overlay, text, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA)


def render_event_frame(frame, result, event, video_time_s, sample_label):
    rows = list(yolo_rows(result))
    labels = expected_labels(event.card)
    team = "enemy" if event.side == "enemy" else "ally"
    candidates = [
        row
        for row in rows
        if row["team"] == team and row["label"] in labels
    ]
    selected_cell, selected_reason = selected_cell_for(event, result, candidates)
    reason = reason_for_event(event, result, candidates, rows)

    overlay = frame.copy()
    draw_grid(overlay, result["arena_px"])
    draw_boxes(overlay, result, event, rows)
    draw_cell(overlay, result["arena_px"], selected_cell, f"selected: {selected_reason}", (0, 0, 255))
    put_lines(
        overlay,
        [
            f"{event.side} {event.status}: expected={event.card} actual={event.actual_card or '-'} sample={sample_label}",
            f"video_time={video_time_s:.2f}s timer={result['time']} total={result['total_remaining_s']} overtime={result['overtime']}",
            f"reason: {reason}",
        ],
    )
    return overlay, reason, selected_cell


def sample_times(event, samples_per_event):
    if event.start_s is None or event.end_s is None:
        return []
    if samples_per_event <= 1 or event.start_s == event.end_s:
        return [event.start_s]
    step = (event.end_s - event.start_s) / (samples_per_event - 1)
    return [event.start_s + idx * step for idx in range(samples_per_event)]


def process_at_time(cap, detector, video_time_s):
    cap.set(cv2.CAP_PROP_POS_MSEC, video_time_s * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None, None, None
    actual_time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    frame = normalize_frame(frame)
    result = process_frame(frame, detector, show_rois=False, yolo_tower_hp_detections=True)
    return frame, result, actual_time_s


def run(video_path: Path, output_dir: Path, samples_per_event: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = build_detector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    summary = []
    for event in EVENTS:
        times = sample_times(event, samples_per_event)
        if not times and event.actual_time_left is not None:
            # The log-only wrong events do not have a reliable video timestamp in the
            # prompt. Save an explanatory placeholder in summary instead.
            summary.append(f"{event.slug}: skipped image; no video_time window was provided")
            continue

        event_dir = output_dir / event.slug
        event_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx, requested_time_s in enumerate(times):
            frame, result, actual_time_s = process_at_time(cap, detector, requested_time_s)
            if frame is None:
                summary.append(f"{event.slug}: failed to read sample at {requested_time_s:.2f}s")
                continue

            sample_label = f"{sample_idx + 1}/{len(times)} requested={requested_time_s:.2f}s"
            overlay, reason, selected_cell = render_event_frame(frame, result, event, actual_time_s, sample_label)
            out_path = event_dir / f"sample_{sample_idx + 1:02d}_time_{actual_time_s:06.2f}.jpg"
            cv2.imwrite(str(out_path), overlay)
            summary.append(
                f"{out_path.relative_to(output_dir)}: reason={reason}; selected_cell={selected_cell}"
            )

    cap.release()
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"wrote action detection debug output to {output_dir}")
    print(f"summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--samples-per-event",
        type=int,
        default=1,
        help="Number of frames to sample across each event window. Keep this low for full-list runs.",
    )
    args = parser.parse_args()
    run(args.video, args.output_dir, max(1, args.samples_per_event))


if __name__ == "__main__":
    main()
