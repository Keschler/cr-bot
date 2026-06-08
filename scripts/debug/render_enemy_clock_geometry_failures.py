from __future__ import annotations

import argparse
import contextlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import types

import cv2
import numpy as np
import yaml


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "src" / "cr_bot").exists():
            return parent
    # Fallback keeps the same behavior as the original script if the file is moved elsewhere.
    return here.parents[2]


ROOT = find_repo_root()
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
KATACR_ROOT = ROOT / "vendor/external/KataCR"
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

# The capture pipeline imports KataCR modules that usually expect jax.
# For this debug renderer we only need enough of jax to let imports succeed.
if "jax" not in sys.modules:
    jax_stub = types.ModuleType("jax")
    jax_stub.numpy = np
    jax_stub.Array = np.ndarray
    jax_stub.jit = lambda func=None, **_: (lambda inner: inner) if func is None else func
    jax_stub.vmap = lambda func=None, **_: (lambda inner: inner) if func is None else func
    sys.modules["jax"] = jax_stub
    sys.modules["jax.numpy"] = np

try:
    import torch
    import torchvision
    from ultralytics.utils import LOGGER
    import ultralytics.utils.ops as ultralytics_ops
    import ultralytics.trackers.track as ultralytics_track
    import ultralytics.utils.plotting as ultralytics_plotting

    if not hasattr(ultralytics_track, "yaml_load"):
        def yaml_load_with_defaults(path):
            data = yaml.safe_load(Path(path).read_text())
            if isinstance(data, dict):
                data.setdefault("fuse_score", False)
            return data

        ultralytics_track.yaml_load = yaml_load_with_defaults
    if not hasattr(ultralytics_plotting, "contextlib"):
        ultralytics_plotting.contextlib = contextlib
    if not hasattr(ultralytics_ops, "torchvision"):
        ultralytics_ops.torchvision = torchvision
    if not hasattr(ultralytics_ops, "LOGGER"):
        ultralytics_ops.LOGGER = LOGGER
    if not hasattr(ultralytics_ops, "nms_rotated"):
        ultralytics_ops.nms_rotated = lambda boxes, scores, iou: torch.empty(
            0, dtype=torch.long, device=scores.device
        )
except Exception:
    pass

from cr_bot.features.action_space import ACTION_GRID


DEFAULT_VIDEO = (
    ROOT
    / "dataset_generation/data/video_clips/downloaded_videos/"
    / "HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].mp4"
)
DEFAULT_PREDICTIONS = ROOT / "outputs/video/capture/3400Ladder.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/debug/enemy_clock_association_examples"

FRAME_RE = re.compile(r"^frame\s+(?P<frame>\d+)\s+video_time=(?P<video_time>[0-9.]+)s")

# Also works if the line was copied from explain_action_misses.py and has a leading "L123:".
WAITING_RE = re.compile(
    r"(?:.*?L(?P<reported_line>\d+):\s*)?"
    r"\[enemy_cards\]\s+waiting\s+track=(?P<track_id>\S+)\s+"
    r"class=(?P<class_name>\S+)\s+"
    r"seen=(?P<seen>\d+)\s+"
    r"avg_conf=(?P<avg_conf>[-0-9.]+)\s+"
    r"team=(?P<team>\S+)\s+"
    r"team_ratio=(?P<team_ratio>[-0-9.]+)\s+"
    r"frame_class=(?P<frame_class>\S+)\s+"
    r"clock_reject=(?P<reject>.+)$"
)

CLOCK_CANDIDATE_RE = re.compile(
    r"(?:.*?L(?P<reported_line>\d+):\s*)?"
    r"\[enemy_cards\]\s+clock candidate\s+"
    r"track=(?P<track_id>\S+)\s+"
    r"class=(?P<class_name>\S+)\s+"
    r"source=(?P<source>\S+)\s+"
    r"status=(?P<status>\S+)\s+"
    r"troop_center=\((?P<troop_x>-?[0-9.]+),(?P<troop_y>-?[0-9.]+)\)\s+"
    r"clock_center=\((?P<clock_x>-?[0-9.]+),(?P<clock_y>-?[0-9.]+)\)\s+"
    r"clock_team=(?P<clock_team>\S+)\s+"
    r"clock_track=(?P<clock_track>\S+)\s+"
    r"clock_conf=(?P<clock_conf>\S+)\s+"
    r"dx=(?P<dx>-?[0-9.]+)\s+"
    r"dy=(?P<dy>-?[0-9.]+)\s+"
    r"reject=(?P<reject>.*?)\s+"
    r"consumed_by=(?P<consumed_by>\S+)\s*$"
)


@dataclass(frozen=True)
class Case:
    label: str
    expected_video_s: float
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class ClockCandidate:
    line_no: int
    frame_index: int
    video_time_s: float
    track_id: str
    class_name: str
    source: str
    status: str
    troop_x: float
    troop_y: float
    clock_x: float
    clock_y: float
    clock_team: str
    clock_track: str
    clock_conf: float | None
    dx: float
    dy: float
    reject_reason: str
    consumed_by: str

    @property
    def failure_kind(self) -> str:
        reason = self.reject_reason
        if "already consumed" in reason:
            return "clock_consumed"
        if "horizontal gap" in reason:
            return "horizontal_gap"
        if "vertical gap" in reason:
            return "vertical_gap"
        if "clock team" in reason:
            return "wrong_clock_team"
        if "clock confidence" in reason:
            return "low_clock_confidence"
        if self.status == "accepted":
            return "accepted"
        if self.status == "consumed":
            return "clock_consumed"
        if self.status == "skipped":
            return "skipped"
        if self.status == "rejected":
            return "rejected"
        return "other"


@dataclass(frozen=True)
class DebugHit:
    case: Case
    frame_index: int
    video_time_s: float
    line_no: int
    track_id: str
    class_name: str
    seen: int
    avg_conf: float
    team: str
    team_ratio: float
    frame_class: bool
    reject_reason: str
    candidates: tuple[ClockCandidate, ...]


CASES = (
    # Good examples from the 3400Ladder miss report:
    # - log cases show that frame-confirmation can wait while all clock candidates are rejected.
    # - musketeer cases show the "good" clock being consumed by another track.
    # - cannon_275 shows vertical-gap rejection.
    Case("log_039_90_horizontal", 39.90, ("the-log",)),
    Case("musketeer_042_10_consumed", 42.10, ("musketeer",)),
    Case("musketeer_119_60_consumed", 119.60, ("musketeer",)),
    Case("musketeer_196_10_consumed", 196.10, ("musketeer",)),
    Case("log_240_80_horizontal", 240.80, ("the-log",)),
    Case("cannon_275_60_vertical", 275.60, ("cannon",)),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render and explain enemy deploy-clock association failures."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-before", type=float, default=2.5)
    parser.add_argument("--window-after", type=float, default=4.0)
    parser.add_argument(
        "--candidate-window",
        type=float,
        default=0.75,
        help="Attach clock-candidate lines within this many seconds before the waiting line.",
    )
    parser.add_argument(
        "--max-candidates-per-hit",
        type=int,
        default=10,
        help="Keep at most this many candidate lines per example image/report.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Only write text reports; do not open the video or render images.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hits = find_debug_hits(
        args.predictions,
        window_before_s=args.window_before,
        window_after_s=args.window_after,
        candidate_window_s=args.candidate_window,
        max_candidates_per_hit=args.max_candidates_per_hit,
    )
    if not hits:
        raise RuntimeError("no matching enemy-card waiting/candidate examples found")

    report_lines = []
    for hit in hits:
        explanation_path = write_hit_report(args.output_dir, hit)
        report_lines.append(format_hit_summary(hit, explanation_path))

    if not args.no_render:
        render_hits(args.video, args.output_dir, hits, report_lines)

    summary = args.output_dir / "summary.txt"
    summary.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(hits)} example report(s) to {args.output_dir}")
    print(f"summary: {summary}")


def find_debug_hits(
    predictions: Path,
    *,
    window_before_s: float,
    window_after_s: float,
    candidate_window_s: float,
    max_candidates_per_hit: int,
) -> list[DebugHit]:
    hits_by_label: dict[str, DebugHit] = {}
    current_frame: int | None = None
    current_video_time: float | None = None
    candidate_history: dict[tuple[str, str], deque[ClockCandidate]] = defaultdict(deque)

    for line_no, line in enumerate(predictions.read_text(encoding="utf-8").splitlines(), start=1):
        frame_match = FRAME_RE.match(line)
        if frame_match:
            current_frame = int(frame_match.group("frame"))
            current_video_time = float(frame_match.group("video_time"))
            continue

        if current_frame is None or current_video_time is None:
            continue

        candidate_match = CLOCK_CANDIDATE_RE.search(line)
        if candidate_match:
            candidate = parse_clock_candidate(candidate_match, line_no, current_frame, current_video_time)
            key = (candidate.track_id, candidate.class_name)
            candidate_history[key].append(candidate)
            # Keep memory bounded. We filter by time again when a waiting line is hit.
            while len(candidate_history[key]) > 50:
                candidate_history[key].popleft()
            continue

        waiting_match = WAITING_RE.search(line)
        if not waiting_match:
            continue

        class_name = waiting_match.group("class_name")
        track_id = waiting_match.group("track_id")
        reject_reason = waiting_match.group("reject")

        for case in CASES:
            if case.label in hits_by_label:
                continue
            if class_name not in case.class_names:
                continue
            if not (
                case.expected_video_s - window_before_s
                <= current_video_time
                <= case.expected_video_s + window_after_s
            ):
                continue

            candidates = tuple(
                candidate
                for candidate in candidate_history.get((track_id, class_name), ())
                if 0 <= current_video_time - candidate.video_time_s <= candidate_window_s
                and candidate.line_no < line_no
            )
            candidates = candidates[-max_candidates_per_hit:]

            hits_by_label[case.label] = DebugHit(
                case=case,
                frame_index=current_frame,
                video_time_s=current_video_time,
                line_no=line_no,
                track_id=track_id,
                class_name=class_name,
                seen=int(waiting_match.group("seen")),
                avg_conf=float(waiting_match.group("avg_conf")),
                team=waiting_match.group("team"),
                team_ratio=float(waiting_match.group("team_ratio")),
                frame_class=waiting_match.group("frame_class") == "True",
                reject_reason=reject_reason,
                candidates=candidates,
            )
            break

    return [hits_by_label[case.label] for case in CASES if case.label in hits_by_label]


def parse_optional_float(value: str) -> float | None:
    if value == "-":
        return None
    return float(value)


def parse_clock_candidate(match: re.Match[str], line_no: int, frame_index: int, video_time_s: float) -> ClockCandidate:
    return ClockCandidate(
        line_no=line_no,
        frame_index=frame_index,
        video_time_s=video_time_s,
        track_id=match.group("track_id"),
        class_name=match.group("class_name"),
        source=match.group("source"),
        status=match.group("status"),
        troop_x=float(match.group("troop_x")),
        troop_y=float(match.group("troop_y")),
        clock_x=float(match.group("clock_x")),
        clock_y=float(match.group("clock_y")),
        clock_team=match.group("clock_team"),
        clock_track=match.group("clock_track"),
        clock_conf=parse_optional_float(match.group("clock_conf")),
        dx=float(match.group("dx")),
        dy=float(match.group("dy")),
        reject_reason=match.group("reject"),
        consumed_by=match.group("consumed_by"),
    )


def write_hit_report(output_dir: Path, hit: DebugHit) -> Path:
    path = output_dir / f"{hit.case.label}.txt"
    reason_counts = Counter(candidate.failure_kind for candidate in hit.candidates)
    lines = [
        f"case: {hit.case.label}",
        f"expected_video: {hit.case.expected_video_s:.2f}s",
        f"waiting_line: {hit.line_no}",
        f"debug_frame: {hit.frame_index}",
        f"debug_video: {hit.video_time_s:.2f}s",
        f"track: {hit.track_id}",
        f"class: {hit.class_name}",
        f"seen: {hit.seen}",
        f"avg_conf: {hit.avg_conf:.3f}",
        f"team: {hit.team}",
        f"team_ratio: {hit.team_ratio:.2f}",
        f"frame_class: {hit.frame_class}",
        f"waiting_clock_reject: {hit.reject_reason}",
        "",
        "why the clock did not associate:",
    ]
    if hit.candidates:
        for kind, count in reason_counts.most_common():
            lines.append(f"- {kind}: {count} candidate(s)")
        best = best_candidate(hit.candidates)
        if best is not None:
            lines.extend([
                "",
                "closest/highest-value candidate:",
                format_candidate(best),
            ])
    else:
        lines.append("- no clock-candidate lines were found in the candidate window before this waiting line")

    lines.extend(["", "all attached candidates:"])
    for candidate in hit.candidates:
        lines.append(format_candidate(candidate))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def best_candidate(candidates: tuple[ClockCandidate, ...]) -> ClockCandidate | None:
    if not candidates:
        return None

    def score(candidate: ClockCandidate) -> tuple[int, float, float, float]:
        wrong_team = 0 if candidate.clock_team in ("enemy", "-") else 1
        low_conf = 0 if candidate.clock_conf is None or candidate.clock_conf >= 0.5 else 1
        consumed = 1 if candidate.failure_kind == "clock_consumed" else 0
        geometry_error = max(0.0, abs(candidate.dx) - 90.0) + max(0.0, 10.0 - candidate.dy) + max(0.0, candidate.dy - 140.0)
        conf_score = -(candidate.clock_conf or 0.0)
        return wrong_team + low_conf + consumed, geometry_error, conf_score, candidate.line_no

    return min(candidates, key=score)


def format_candidate(candidate: ClockCandidate) -> str:
    conf = "-" if candidate.clock_conf is None else f"{candidate.clock_conf:.3f}"
    return (
        f"L{candidate.line_no}: source={candidate.source} status={candidate.status} "
        f"clock_track={candidate.clock_track} clock_team={candidate.clock_team} "
        f"clock_conf={conf} dx={candidate.dx:.1f} dy={candidate.dy:.1f} "
        f"consumed_by={candidate.consumed_by} reason={candidate.reject_reason} "
        f"troop=({candidate.troop_x:.1f},{candidate.troop_y:.1f}) "
        f"clock=({candidate.clock_x:.1f},{candidate.clock_y:.1f})"
    )


def format_hit_summary(hit: DebugHit, explanation_path: Path) -> str:
    reason_counts = Counter(candidate.failure_kind for candidate in hit.candidates)
    reasons = ", ".join(f"{kind}={count}" for kind, count in reason_counts.most_common()) or "no candidates"
    return (
        f"{hit.case.label}: expected={hit.case.expected_video_s:.2f}s "
        f"debug={hit.video_time_s:.2f}s frame={hit.frame_index} "
        f"track={hit.track_id} class={hit.class_name} waiting_reject={hit.reject_reason!r} "
        f"candidate_reasons=[{reasons}] report={explanation_path.name}"
    )


def render_hits(video_path: Path, output_dir: Path, hits: list[DebugHit], report_lines: list[str]) -> None:
    detector, normalize_frame, process_frame = load_detector_pipeline()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"could not open video; wrote text reports only: {video_path}")
        return

    try:
        for hit in hits:
            frame = read_video_time(cap, hit.video_time_s)
            if detector is not None and normalize_frame is not None and process_frame is not None:
                frame = normalize_frame(frame)
                result = process_frame(frame, detector, show_rois=False)
                overlay = render_hit(frame, result, hit)
            else:
                overlay = render_hit_without_detections(frame, hit)
            out_path = output_dir / f"{hit.case.label}_frame_{hit.frame_index:06d}.jpg"
            cv2.imwrite(str(out_path), overlay)
            report_lines.append(f"image={out_path.name}")
    finally:
        cap.release()


def load_detector_pipeline():
    try:
        from cr_bot.app.pipeline import normalize_frame, process_frame
        from cr_bot.vision.yolo_runtime import build_detector
    except Exception as exc:
        print(f"detector imports unavailable; writing candidate-only frame snapshots: {exc}")
        return None, None, None

    try:
        return build_detector(), normalize_frame, process_frame
    except Exception as exc:
        print(f"detector unavailable; writing candidate-only frame snapshots: {exc}")
        return None, None, None


def read_video_time(cap, video_time_s: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, video_time_s * 1000.0)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read frame at {video_time_s:.2f}s")
    return frame


def render_hit(frame, result, hit: DebugHit):
    overlay = frame.copy()
    draw_grid(overlay, result["arena_px"])
    target = find_target_match(result, hit)
    enemy_clocks = [
        clock
        for clock in result["clock_boxes"]
        if clock["team"] == "enemy" and clock["confidence"] >= 0.5
    ]

    # Draw live/reprocessed detector output lightly.
    for clock in result["clock_boxes"]:
        color = (0, 165, 255) if clock["team"] == "enemy" else (255, 255, 0)
        label = (
            f"clock:{clock['team']} conf={clock['confidence']:.2f} "
            f"cell={ACTION_GRID.pixel_to_cell(clock['center_x'], clock['center_y'], result['arena_px'])}"
        )
        draw_box(
            overlay,
            clock["x1"],
            clock["y1"],
            clock["x2"],
            clock["y2"],
            clock["center_x"],
            clock["center_y"],
            color,
            label,
        )

    for match in result["matches"]:
        troop = match.troop
        if troop.class_name not in hit.case.class_names:
            continue
        is_target = target is not None and troop is target
        color = (0, 0, 255) if is_target else (0, 255, 0) if troop.team == "enemy" else (255, 0, 255)
        label = (
            f"{'TARGET ' if is_target else ''}{troop.class_name}:{troop.team} "
            f"id={getattr(troop, 'track_id', None)} conf={troop.confidence:.2f}"
        )
        draw_troop(overlay, troop, color, label)

    # Draw the exact candidate geometry from the tracker debug lines. This is the important part:
    # it shows the clock that the tracker tried to associate and the exact reason it rejected it.
    for idx, candidate in enumerate(hit.candidates):
        draw_candidate_geometry(overlay, candidate, idx)

    draw_header(overlay, hit, target, len(enemy_clocks))
    draw_candidate_table(overlay, hit)
    return overlay


def render_hit_without_detections(frame, hit: DebugHit):
    overlay = frame.copy()
    for idx, candidate in enumerate(hit.candidates):
        draw_candidate_geometry(overlay, candidate, idx)
    draw_text_panel(
        overlay,
        [
            f"{hit.case.label}",
            f"expected={hit.case.expected_video_s:.2f}s debug_frame={hit.frame_index} debug_video={hit.video_time_s:.2f}s line={hit.line_no}",
            f"track={hit.track_id} class={hit.class_name} waiting_reject={hit.reject_reason}",
            "candidate-only snapshot: detector stack unavailable in this Python env",
        ],
        top=8,
        height=126,
    )
    draw_candidate_table(overlay, hit)
    return overlay


def find_target_match(result, hit: DebugHit):
    typed_matches = [
        match.troop
        for match in result["matches"]
        if match.troop.class_name in hit.case.class_names
    ]
    if not typed_matches:
        return None

    for troop in typed_matches:
        if str(getattr(troop, "track_id", "")) == hit.track_id:
            return troop

    enemy_matches = [troop for troop in typed_matches if troop.team == "enemy"]
    if enemy_matches:
        return max(enemy_matches, key=lambda troop: troop.confidence)
    return max(typed_matches, key=lambda troop: troop.confidence)


def draw_grid(img, arena_px):
    ax, ay, aw, ah = arena_px
    x0 = int(round(ax + ACTION_GRID.x0 * aw))
    y0 = int(round(ay + ACTION_GRID.y0 * ah))
    x1 = int(round(ax + ACTION_GRID.x1 * aw))
    y1 = int(round(ay + ACTION_GRID.y1 * ah))
    for col in range(ACTION_GRID.cols + 1):
        x = int(round(x0 + col / ACTION_GRID.cols * (x1 - x0)))
        cv2.line(img, (x, y0), (x, y1), (45, 45, 45), 1)
    for row in range(ACTION_GRID.rows + 1):
        y = int(round(y0 + row / ACTION_GRID.rows * (y1 - y0)))
        cv2.line(img, (x0, y), (x1, y), (45, 45, 45), 1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 2)


def draw_troop(img, troop, color, label):
    draw_box(
        img,
        troop.x1,
        troop.y1,
        troop.x2,
        troop.y2,
        troop.center_x,
        troop.center_y,
        color,
        label,
    )


def draw_box(img, x1, y1, x2, y2, center_x, center_y, color, label):
    x1, y1, x2, y2 = (int(round(value)) for value in (x1, y1, x2, y2))
    cx, cy = int(round(center_x)), int(round(center_y))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 24, 2)
    text_y = max(24, y1 - 8)
    cv2.putText(img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def candidate_color(candidate: ClockCandidate):
    if candidate.status == "accepted":
        return (0, 220, 0)
    if candidate.failure_kind == "clock_consumed":
        return (255, 0, 255)
    if candidate.failure_kind == "wrong_clock_team":
        return (255, 255, 0)
    if candidate.failure_kind == "low_clock_confidence":
        return (200, 200, 200)
    return (0, 0, 255)


def draw_candidate_geometry(img, candidate: ClockCandidate, idx: int):
    color = candidate_color(candidate)
    troop_center = (int(round(candidate.troop_x)), int(round(candidate.troop_y)))
    clock_center = (int(round(candidate.clock_x)), int(round(candidate.clock_y)))
    cv2.line(img, troop_center, clock_center, color, 3)
    cv2.drawMarker(img, troop_center, color, cv2.MARKER_TILTED_CROSS, 28, 2)
    cv2.drawMarker(img, clock_center, color, cv2.MARKER_CROSS, 28, 2)

    label = (
        f"C{idx + 1} {candidate.status} {candidate.failure_kind} "
        f"dx={candidate.dx:.1f} dy={candidate.dy:.1f} team={candidate.clock_team} "
        f"consumed={candidate.consumed_by}"
    )
    label_x = max(5, min(img.shape[1] - 900, (troop_center[0] + clock_center[0]) // 2))
    label_y = max(160, min(img.shape[0] - 20, (troop_center[1] + clock_center[1]) // 2))
    cv2.rectangle(img, (label_x - 4, label_y - 22), (min(img.shape[1] - 5, label_x + 880), label_y + 8), (0, 0, 0), -1)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def draw_header(img, hit: DebugHit, target, enemy_clock_count: int):
    target_text = "target not found in reprocessed frame"
    if target is not None:
        target_text = (
            f"target center=({target.center_x:.1f},{target.center_y:.1f}) "
            f"team={target.team} conf={target.confidence:.2f}"
        )
    lines = [
        f"{hit.case.label} expected={hit.case.expected_video_s:.2f}s frame={hit.frame_index} video={hit.video_time_s:.2f}s line={hit.line_no}",
        f"track={hit.track_id} class={hit.class_name} waiting_reject={hit.reject_reason}",
        f"{target_text}; enemy_clock_count={enemy_clock_count}; candidate_lines={len(hit.candidates)}",
    ]
    draw_text_panel(img, lines, top=8, height=96)


def draw_candidate_table(img, hit: DebugHit):
    lines = []
    for idx, candidate in enumerate(hit.candidates[:6]):
        conf = "-" if candidate.clock_conf is None else f"{candidate.clock_conf:.2f}"
        lines.append(
            f"C{idx + 1}: {candidate.status}/{candidate.failure_kind} "
            f"clock={candidate.clock_track} team={candidate.clock_team} conf={conf} "
            f"dx={candidate.dx:.1f} dy={candidate.dy:.1f} consumed={candidate.consumed_by}"
        )
    if not lines:
        lines = ["No candidate lines attached. Increase --candidate-window if needed."]
    draw_text_panel(img, lines, top=img.shape[0] - (len(lines) * 25 + 24), height=len(lines) * 25 + 16)


def draw_text_panel(img, lines: list[str], *, top: int, height: int):
    top = max(0, min(img.shape[0] - 1, top))
    bottom = max(top + 1, min(img.shape[0] - 1, top + height))
    cv2.rectangle(img, (8, top), (img.shape[1] - 8, bottom), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            img,
            line[:170],
            (18, top + 26 + idx * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    main()

