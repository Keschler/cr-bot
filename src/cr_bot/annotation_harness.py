from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable

import cv2
import numpy as np

from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.domain.rois import ROIS
from cr_bot.features.action_space import ACTION_GRID


ANNOTATION_TYPE = "blinded_codex_ground_truth"
PROMPT_VERSION = "codex-harness-v7"
PROCESSING_SIZE = (1080, 2400)
LABEL_MARGIN_PX = 76
REVIEW_MAX_FRAMES = {
    "full": 6,
    "arena": 20,
    "own_context": 9,
    "own_confirmation": 8,
    "identity": 6,
    "macro": 4,
    "grid": 4,
}
REVIEW_MAX_PIXELS = 8_000_000
EVIDENCE_KEYS = (
    "elixir_drop",
    "hand_transition",
    "deployment_onset",
    "first_visible_object",
    "side_direction",
    "impact_sequence",
)
OWN_CONFIRMATION_KEYS = frozenset(
    {
        "release_confirmed",
        "elixir_spend_persisted",
        "hand_cycle_completed",
        "post_release_effect",
    }
)
LOCATION_RULES = {
    "spawn_center",
    "deployment_center",
    "target_center",
    "impact_center",
    "initial_rolling_object_center",
    "unavailable",
}
AMBIGUITIES = {
    "none",
    "multiple_actions",
    "card_identity",
    "exact_frame",
    "spawn_center",
    "spell_side",
    "spell_target",
    "evolution_status",
    "insufficient_resolution",
    "unscorable",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if (frame.shape[1], frame.shape[0]) == PROCESSING_SIZE:
        return frame
    return cv2.resize(frame, PROCESSING_SIZE, interpolation=cv2.INTER_AREA)


def add_frame_identity(
    image: np.ndarray, *, source_frame_index: int, fps: float
) -> np.ndarray:
    canvas = np.zeros(
        (image.shape[0] + LABEL_MARGIN_PX, image.shape[1], image.shape[2]),
        dtype=image.dtype,
    )
    canvas[LABEL_MARGIN_PX:] = image
    full_label = (
        f"SOURCE FRAME {source_frame_index:06d}    "
        f"VIDEO TIME {source_frame_index / fps:.3f} s"
    )
    compact_label = (
        f"FRAME {source_frame_index:06d}    "
        f"{source_frame_index / fps:.3f} s"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.05
    max_label_width = max(1, image.shape[1] - 36)
    label = full_label
    label_width = cv2.getTextSize(label, font, font_scale, 2)[0][0]
    if label_width > max_label_width:
        label = compact_label
        label_width = cv2.getTextSize(label, font, font_scale, 2)[0][0]
    if label_width > max_label_width:
        font_scale *= max_label_width / label_width
    thickness = 2 if font_scale >= 0.65 else 1
    cv2.putText(
        canvas,
        label,
        (18, 49),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def frame_bounds(
    *, fps: float, frame_count: int, start_time_s: float, end_time_s: float | None
) -> tuple[int, int, float]:
    if fps <= 0:
        raise ValueError("source FPS must be positive")
    duration_s = frame_count / fps
    effective_end = duration_s if end_time_s is None else end_time_s
    if start_time_s < 0 or effective_end <= start_time_s or effective_end > duration_s + 1e-6:
        raise ValueError(
            f"invalid half-open segment [{start_time_s}, {effective_end}); "
            f"video duration is {duration_s:.6f}s"
        )
    start_frame = max(0, math.ceil(start_time_s * fps - 1e-9))
    end_frame = min(frame_count, math.ceil(effective_end * fps - 1e-9))
    return start_frame, end_frame, effective_end


def prepare_annotation_run(
    *,
    video_path: Path,
    output_dir: Path,
    start_time_s: float = 0.0,
    end_time_s: float | None = None,
    jpeg_quality: int = 92,
    own_change_threshold: float = 10.0,
    own_cluster_seconds: float = 0.7,
    enemy_window_seconds: float = 2.0,
    enemy_overlap_seconds: float = 0.0,
) -> dict[str, Any]:
    video_path = video_path.resolve()
    output_dir = output_dir.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"annotation run already exists: {output_dir}")
    if not 0 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in 0..100")
    if enemy_overlap_seconds >= enemy_window_seconds:
        raise ValueError("enemy overlap must be smaller than the window")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame, end_frame, effective_end = frame_bounds(
            fps=fps,
            frame_count=frame_count,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir()

        frame_records: list[dict[str, Any]] = []
        own_signals: list[dict[str, Any]] = []
        previous_arena: np.ndarray | None = None
        previous_hud: np.ndarray | None = None
        # Decode from frame zero so source indices do not depend on approximate
        # keyframe seeking. Frames before the requested segment are discarded.
        for source_index in range(end_frame):
            ok, source_frame = capture.read()
            if not ok:
                raise ValueError(
                    f"decode stopped at source frame {source_index}; expected through {end_frame - 1}"
                )
            if source_index < start_frame:
                continue
            normalized = normalize_frame(source_frame)
            relative_path = Path("frames") / f"frame_{source_index:06d}.jpg"
            labeled = add_frame_identity(
                normalized, source_frame_index=source_index, fps=fps
            )
            if not cv2.imwrite(
                str(output_dir / relative_path),
                labeled,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                raise OSError(f"failed to write {relative_path}")

            arena_gray = _difference_image(_crop(normalized, ROIS["battlefield"]))
            hud_gray = _difference_image(_own_hud_crop(normalized))
            arena_score = _mean_absolute_difference(previous_arena, arena_gray)
            own_score = _mean_absolute_difference(previous_hud, hud_gray)
            frame_records.append(
                {
                    "source_frame_index": source_index,
                    "video_time_s": source_index / fps,
                    "path": relative_path.as_posix(),
                    "arena_change_score": arena_score,
                    "own_hud_change_score": own_score,
                }
            )
            if own_score is not None and own_score >= own_change_threshold:
                own_signals.append(
                    {
                        "source_frame_index": source_index,
                        "change_score": own_score,
                    }
                )
            previous_arena = arena_gray
            previous_hud = hud_gray
    finally:
        capture.release()

    own_candidates = cluster_own_signals(
        own_signals,
        fps=fps,
        max_gap_frames=max(1, round(own_cluster_seconds * fps)),
        segment_start_frame=start_frame,
        segment_end_frame=end_frame,
    )
    enemy_windows = build_scan_windows(
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
        window_seconds=enemy_window_seconds,
        overlap_seconds=enemy_overlap_seconds,
    )
    run_id = f"{video_path.stem}.codex.{start_frame}-{end_frame}.{utc_now_iso()}"
    manifest = {
        "run_id": run_id,
        "annotation_type": ANNOTATION_TYPE,
        "workflow_version": 7,
        "prompt_version": PROMPT_VERSION,
        "created_at": utc_now_iso(),
        "video": str(video_path),
        "video_sha256": sha256_file(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "source_resolution": [source_width, source_height],
        "processing_resolution": list(PROCESSING_SIZE),
        "label_margin_px": LABEL_MARGIN_PX,
        "segment": {
            "start_time_s": start_time_s,
            "end_time_s": effective_end,
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame,
        },
        "blindness": {
            "semantic_inputs": ["source_video"],
            "semantic_models_used_locally": [],
        },
        "frames": frame_records,
        "candidate_discovery": {
            "own_change_threshold": own_change_threshold,
            "own_candidates": own_candidates,
            "enemy_scan_windows": enemy_windows,
        },
    }
    audit = {
        "run_id": run_id,
        "annotation_type": ANNOTATION_TYPE,
        "video": video_path.name,
        "fps": fps,
        "segment": manifest["segment"],
        "manifest": "manifest.json",
        "prompt_version": PROMPT_VERSION,
        "accepted_events": [],
        "rejected_candidates": [],
        "completeness_sweeps": [],
        "adjudications": [],
        "locked": False,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(output_dir / "audit.json", audit)
    from cr_bot.annotation_stages import write_stage_templates

    write_stage_templates(output_dir, manifest)
    return manifest


def cluster_own_signals(
    signals: list[dict[str, Any]],
    *,
    fps: float,
    max_gap_frames: int,
    segment_start_frame: int,
    segment_end_frame: int,
) -> list[dict[str, Any]]:
    if not signals:
        return []
    max_span_frames = max(1, REVIEW_MAX_FRAMES["own_context"] - 1)
    groups: list[list[dict[str, Any]]] = [[signals[0]]]
    for signal in signals[1:]:
        gap = signal["source_frame_index"] - groups[-1][-1]["source_frame_index"]
        span = signal["source_frame_index"] - groups[-1][0]["source_frame_index"]
        if gap <= max_gap_frames and span <= max_span_frames:
            groups[-1].append(signal)
        else:
            groups.append([signal])
    candidates = []
    before = max(1, round(0.3 * fps))
    review_frames = REVIEW_MAX_FRAMES["own_context"]
    for group in groups:
        peak = max(group, key=lambda item: item["change_score"])
        approximate = peak["source_frame_index"]
        inspection_start = max(segment_start_frame, approximate - before)
        inspection_end = min(
            segment_end_frame, inspection_start + review_frames
        )
        inspection_start = max(
            segment_start_frame, inspection_end - review_frames
        )
        candidates.append(
            {
                "candidate_id": f"own:{approximate:06d}",
                "side_hint": "own",
                "approximate_frame_index": approximate,
                "inspection_start_frame": inspection_start,
                "inspection_end_frame_exclusive": inspection_end,
                "sources": ["own_hud_pixel_change"],
                "signals": group,
            }
        )
    return candidates


def build_scan_windows(
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
    window_seconds: float,
    overlap_seconds: float,
) -> list[dict[str, Any]]:
    window = max(1, round(window_seconds * fps))
    step = max(1, round((window_seconds - overlap_seconds) * fps))
    windows = []
    cursor = start_frame
    pass_number = 1
    while cursor < end_frame:
        window_end = min(end_frame, cursor + window)
        windows.append(
            {
                "candidate_id": f"enemy-scan:{cursor:06d}-{window_end:06d}:p{pass_number}",
                "side_hint": "enemy",
                "inspection_start_frame": cursor,
                "inspection_end_frame_exclusive": window_end,
                "sources": [f"exhaustive_enemy_scan_pass_{pass_number}"],
            }
        )
        if window_end == end_frame:
            break
        cursor += step
    if overlap_seconds == 0:
        # The second pass protects only primary-window boundaries instead of
        # uploading the complete interval twice.
        boundary_radius = max(1, round(0.2 * fps))
        for boundary in range(start_frame + window, end_frame, window):
            window_start = max(start_frame, boundary - boundary_radius)
            window_end = min(end_frame, boundary + boundary_radius)
            windows.append(
                {
                    "candidate_id": (
                        f"enemy-boundary:{window_start:06d}-{window_end:06d}:p2"
                    ),
                    "side_hint": "enemy",
                    "inspection_start_frame": window_start,
                    "inspection_end_frame_exclusive": window_end,
                    "sources": ["enemy_scan_boundary_pass_2"],
                }
            )
        return windows

    shifted = start_frame + max(1, step // 2)
    pass_number = 2
    cursor = shifted
    while cursor < end_frame:
        window_end = min(end_frame, cursor + window)
        windows.append(
            {
                "candidate_id": f"enemy-scan:{cursor:06d}-{window_end:06d}:p{pass_number}",
                "side_hint": "enemy",
                "inspection_start_frame": cursor,
                "inspection_end_frame_exclusive": window_end,
                "sources": [f"exhaustive_enemy_scan_pass_{pass_number}"],
            }
        )
        if window_end == end_frame:
            break
        cursor += step
    return windows


def render_review_sheet(
    *,
    run_dir: Path,
    output_path: Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
    candidate_id: str | None = None,
    event_id: str | None = None,
    purpose: str = "arena",
    columns: int = 5,
    tile_width: int = 360,
    grid_center: tuple[int, int] | None = None,
    grid_radius: int = 3,
    focus_cell: tuple[int, int] | None = None,
    focus_radius: int = 4,
) -> Path:
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "manifest.json")
    segment = manifest["segment"]
    if candidate_id is not None:
        candidate = _find_candidate(manifest, candidate_id)
        start_frame = candidate["inspection_start_frame"]
        end_frame = candidate["inspection_end_frame_exclusive"]
    if start_frame is None or end_frame is None:
        raise ValueError("provide a candidate ID or a half-open frame range")
    if start_frame < segment["start_frame"] or end_frame > segment["end_frame_exclusive"]:
        raise ValueError("review range is outside the prepared segment")
    if start_frame >= end_frame:
        raise ValueError("review range must not be empty")
    if purpose not in {
        "full",
        "arena",
        "own_context",
        "own_confirmation",
        "identity",
        "macro",
        "grid",
    }:
        raise ValueError(f"unsupported purpose: {purpose}")
    frame_total = end_frame - start_frame
    max_frames = REVIEW_MAX_FRAMES[purpose]
    if manifest.get("workflow_version", 1) >= 3 and frame_total > max_frames:
        raise ValueError(
            f"{purpose} review contains {frame_total} frames; maximum is {max_frames}. "
            "Split it into smaller half-open ranges so every tile remains readable."
        )
    if manifest.get("workflow_version", 1) >= 2 and purpose in {"macro", "grid"}:
        if not event_id:
            raise ValueError(
                f"--event-id is required for staged {purpose} review"
            )
        from cr_bot.annotation_stages import require_verification_checkpoint

        require_verification_checkpoint(run_dir, event_id)
    if purpose == "identity":
        if not event_id:
            raise ValueError("--event-id is required for identity review")
    if purpose == "own_confirmation" and not event_id:
        raise ValueError("--event-id is required for own confirmation review")
    if columns <= 0 or tile_width <= 0:
        raise ValueError("columns and tile width must be positive")

    frames_by_index = {
        item["source_frame_index"]: item for item in manifest["frames"]
    }
    tiles = []
    for frame_index in range(start_frame, end_frame):
        record = frames_by_index.get(frame_index)
        if record is None:
            raise ValueError(f"frame {frame_index} was not prepared")
        labeled = cv2.imread(str(run_dir / record["path"]))
        if labeled is None:
            raise FileNotFoundError(run_dir / record["path"])
        content = labeled[manifest["label_margin_px"] :]
        view = _review_view(
            content,
            purpose=purpose,
            grid_center=grid_center,
            grid_radius=grid_radius,
            focus_cell=focus_cell,
            focus_radius=focus_radius,
        )
        view = add_frame_identity(
            view, source_frame_index=frame_index, fps=float(manifest["fps"])
        )
        scale = tile_width / view.shape[1]
        tile = cv2.resize(
            view,
            (tile_width, max(1, round(view.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        tiles.append(tile)
    sheet = _make_contact_sheet(tiles, columns=columns)
    if (
        manifest.get("workflow_version", 1) >= 3
        and sheet.shape[0] * sheet.shape[1] > REVIEW_MAX_PIXELS
    ):
        raise ValueError(
            f"review sheet is {sheet.shape[1]}x{sheet.shape[0]}; reduce --tile-width "
            "or use more columns"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError(f"failed to write {output_path}")
    if manifest.get("workflow_version", 1) >= 2:
        from cr_bot.annotation_stages import record_review

        record_review(
            run_dir=run_dir,
            output_path=output_path,
            purpose=purpose,
            start_frame=start_frame,
            end_frame=end_frame,
            candidate_id=candidate_id,
            event_id=event_id,
        )
    return output_path


def _review_view(
    frame: np.ndarray,
    *,
    purpose: str,
    grid_center: tuple[int, int] | None,
    grid_radius: int,
    focus_cell: tuple[int, int] | None,
    focus_radius: int,
) -> np.ndarray:
    if purpose == "full":
        return frame.copy()
    if purpose == "arena":
        return _crop(frame, ROIS["battlefield"]).copy()
    if purpose == "own_context":
        x, y, w, _ = ROIS["battlefield"]
        return frame[y : frame.shape[0], x : x + w].copy()
    if purpose == "own_confirmation":
        x, y, w, _ = ROIS["battlefield"]
        return frame[y : frame.shape[0], x : x + w].copy()
    if purpose == "identity":
        if focus_cell is None:
            return _crop(frame, ROIS["battlefield"]).copy()
        return _focus_crop(
            frame,
            focus_cell=focus_cell,
            focus_radius=focus_radius,
        )
    if purpose == "macro":
        overlay = frame.copy()
        _draw_macro_grid(overlay, ROIS["battlefield"])
        return _crop(overlay, ROIS["battlefield"]).copy()

    overlay = frame.copy()
    arena_px = ROIS["battlefield"]
    _draw_grid(overlay, arena_px)
    if grid_center is None:
        return overlay
    col, row = grid_center
    if not (0 <= col < ACTION_GRID.cols and 0 <= row < ACTION_GRID.rows):
        raise ValueError("grid center is outside the action grid")
    radius = max(0, grid_radius)
    min_col = max(0, col - radius)
    max_col = min(ACTION_GRID.cols - 1, col + radius)
    min_row = max(0, row - radius)
    max_row = min(ACTION_GRID.rows - 1, row + radius)
    x0, y0 = ACTION_GRID.cell_to_pixel_center(min_col, min_row, arena_px)
    x1, y1 = ACTION_GRID.cell_to_pixel_center(max_col, max_row, arena_px)
    cell_w = ACTION_GRID.width * arena_px[2] / ACTION_GRID.cols
    cell_h = ACTION_GRID.height * arena_px[3] / ACTION_GRID.rows
    left = max(0, round(x0 - cell_w / 2))
    top = max(0, round(y0 - cell_h / 2))
    right = min(overlay.shape[1], round(x1 + cell_w / 2))
    bottom = min(overlay.shape[0], round(y1 + cell_h / 2))
    return overlay[top:bottom, left:right].copy()


def _focus_crop(
    frame: np.ndarray,
    *,
    focus_cell: tuple[int, int],
    focus_radius: int,
) -> np.ndarray:
    col, row = focus_cell
    if not (0 <= col < ACTION_GRID.cols and 0 <= row < ACTION_GRID.rows):
        raise ValueError("focus cell is outside the action grid")
    radius = max(0, focus_radius)
    min_col = max(0, col - radius)
    max_col = min(ACTION_GRID.cols - 1, col + radius)
    min_row = max(0, row - radius)
    max_row = min(ACTION_GRID.rows - 1, row + radius)
    arena_px = ROIS["battlefield"]
    x0, y0 = ACTION_GRID.cell_to_pixel_center(min_col, min_row, arena_px)
    x1, y1 = ACTION_GRID.cell_to_pixel_center(max_col, max_row, arena_px)
    cell_w = ACTION_GRID.width * arena_px[2] / ACTION_GRID.cols
    cell_h = ACTION_GRID.height * arena_px[3] / ACTION_GRID.rows
    left = max(0, round(x0 - cell_w / 2))
    top = max(0, round(y0 - cell_h / 2))
    right = min(frame.shape[1], round(x1 + cell_w / 2))
    bottom = min(frame.shape[0], round(y1 + cell_h / 2))
    return frame[top:bottom, left:right].copy()


def _draw_grid(image: np.ndarray, arena_px: tuple[int, int, int, int]) -> None:
    ax, ay, aw, ah = arena_px
    gx0 = round(ax + ACTION_GRID.x0 * aw)
    gy0 = round(ay + ACTION_GRID.y0 * ah)
    gx1 = round(ax + ACTION_GRID.x1 * aw)
    gy1 = round(ay + ACTION_GRID.y1 * ah)
    for col in range(ACTION_GRID.cols + 1):
        x = round(gx0 + col / ACTION_GRID.cols * (gx1 - gx0))
        cv2.line(image, (x, gy0), (x, gy1), (30, 30, 30), 1)
    for row in range(ACTION_GRID.rows + 1):
        y = round(gy0 + row / ACTION_GRID.rows * (gy1 - gy0))
        cv2.line(image, (gx0, y), (gx1, y), (30, 30, 30), 1)
    for row in range(ACTION_GRID.rows):
        for col in range(ACTION_GRID.cols):
            x, y = ACTION_GRID.cell_to_pixel_center(col, row, arena_px)
            label = f"{col},{row}"
            cv2.putText(
                image,
                label,
                (round(x - 14), round(y + 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.27,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label,
                (round(x - 14), round(y + 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.27,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def _draw_macro_grid(image: np.ndarray, arena_px: tuple[int, int, int, int]) -> None:
    ax, ay, aw, ah = arena_px
    gx0 = round(ax + ACTION_GRID.x0 * aw)
    gy0 = round(ay + ACTION_GRID.y0 * ah)
    gx1 = round(ax + ACTION_GRID.x1 * aw)
    gy1 = round(ay + ACTION_GRID.y1 * ah)
    macro_cols, macro_rows = 3, 4
    for col in range(macro_cols + 1):
        x = round(gx0 + col / macro_cols * (gx1 - gx0))
        cv2.line(image, (x, gy0), (x, gy1), (0, 0, 0), 3)
    for row in range(macro_rows + 1):
        y = round(gy0 + row / macro_rows * (gy1 - gy0))
        cv2.line(image, (gx0, y), (gx1, y), (0, 0, 0), 3)
    for row in range(macro_rows):
        for col in range(macro_cols):
            x = round(gx0 + (col + 0.5) / macro_cols * (gx1 - gx0))
            y = round(gy0 + (row + 0.5) / macro_rows * (gy1 - gy0))
            for color, thickness in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                cv2.putText(
                    image,
                    f"M{col},{row}",
                    (x - 38, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )


def finalize_annotation(
    *,
    run_dir: Path,
    decisions_path: Path,
    output_path: Path,
    audit_output_path: Path,
) -> tuple[Path, Path, Path]:
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "manifest.json")
    staged_checkpoints = None
    if manifest.get("workflow_version", 1) >= 2:
        from cr_bot.annotation_stages import assemble_staged_decisions

        decisions, staged_checkpoints = assemble_staged_decisions(run_dir, manifest)
    else:
        decisions = _read_json(decisions_path)
    if decisions.get("run_id") != manifest["run_id"]:
        raise ValueError("decisions run_id does not match the manifest")
    lock_path = audit_output_path.with_suffix(audit_output_path.suffix + ".lock.json")
    for path in (output_path, audit_output_path, lock_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite locked artifact: {path}")

    events = validate_decisions(manifest, decisions)
    for event in events:
        hashed_artifacts = []
        for artifact_name in event["review_artifacts"]:
            artifact_path = Path(artifact_name)
            if not artifact_path.is_absolute():
                artifact_path = run_dir / artifact_path
            artifact_path = artifact_path.resolve()
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"missing review artifact for {event['candidate_id']}: {artifact_path}"
                )
            hashed_artifacts.append(
                {
                    "path": str(artifact_path),
                    "sha256": sha256_file(artifact_path),
                }
            )
        event["review_artifacts"] = hashed_artifacts
    compact_events = []
    for event in events:
        compact = {
            "side": event["side"],
            "card": event["card"],
            "frame_index": event["event_frame_index"],
        }
        if event.get("cell") is not None:
            compact["cell"] = event["cell"]
        compact_events.append(compact)
    compact_events.sort(
        key=lambda event: (
            event["frame_index"],
            event["side"],
            event["card"],
            event.get("cell") or [-1, -1],
        )
    )
    compact = {
        "video": Path(manifest["video"]).name,
        "fps": manifest["fps"],
        "notes": (
            "Blinded Codex visual annotation; no existing labels, detector output, "
            "replay cache, OCR, classifier, or tracker output inspected."
        ),
        "segment": {
            "start_time_s": manifest["segment"]["start_time_s"],
            "end_time_s": manifest["segment"]["end_time_s"],
        },
        "events": compact_events,
    }
    atomic_write_json(output_path, compact)
    final_sha = sha256_file(output_path)
    audit = {
        "run_id": manifest["run_id"],
        "annotation_type": ANNOTATION_TYPE,
        "video": Path(manifest["video"]).name,
        "video_sha256": manifest["video_sha256"],
        "fps": manifest["fps"],
        "segment": manifest["segment"],
        "manifest": str((run_dir / "manifest.json").resolve()),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "prompt_version": manifest["prompt_version"],
        "accepted_events": events,
        "rejected_candidates": decisions.get("rejected_candidates", []),
        "completeness_sweeps": decisions["completeness_sweeps"],
        "adjudications": decisions.get("adjudications", []),
        "stage_checkpoints": staged_checkpoints,
        "final_output": str(output_path.resolve()),
        "final_sha256": final_sha,
        "locked_at": utc_now_iso(),
        "locked": True,
    }
    atomic_write_json(audit_output_path, audit)
    lock = {
        "run_id": manifest["run_id"],
        "final_path": str(output_path.resolve()),
        "final_sha256": final_sha,
        "audit_path": str(audit_output_path.resolve()),
        "audit_sha256": sha256_file(audit_output_path),
        "locked_at": audit["locked_at"],
    }
    atomic_write_json(lock_path, lock)
    return output_path, audit_output_path, lock_path


def validate_decisions(
    manifest: dict[str, Any], decisions: dict[str, Any]
) -> list[dict[str, Any]]:
    sweeps = decisions.get("completeness_sweeps")
    if not isinstance(sweeps, list):
        raise ValueError("completeness_sweeps must be a list")
    completed_sides = {
        item.get("side")
        for item in sweeps
        if isinstance(item, dict) and item.get("completed") is True
    }
    if not {"own", "enemy"} <= completed_sides:
        raise ValueError("completed independent own and enemy completeness sweeps are required")

    events = decisions.get("events")
    if not isinstance(events, list):
        raise ValueError("events must be a list")
    start = manifest["segment"]["start_frame"]
    end = manifest["segment"]["end_frame_exclusive"]
    validated = []
    seen: set[tuple[str, str, int]] = set()
    for index, raw in enumerate(events):
        if not isinstance(raw, dict):
            raise ValueError(f"event {index} must be an object")
        event = dict(raw)
        if not isinstance(event.get("candidate_id"), str) or not event["candidate_id"]:
            raise ValueError(f"event {index}: candidate_id is required")
        side = event.get("side")
        if side not in {"own", "enemy"}:
            raise ValueError(f"event {index}: invalid side")
        card = event.get("card")
        base_card = card[4:] if isinstance(card, str) and card.startswith("evo-") else card
        if not isinstance(card, str) or base_card not in CARD_METADATA:
            raise ValueError(f"event {index}: unknown canonical card slug {card!r}")
        frame = event.get("event_frame_index")
        if not isinstance(frame, int) or isinstance(frame, bool) or not start <= frame < end:
            raise ValueError(f"event {index}: event_frame_index is outside the segment")
        location_frame = event.get("location_frame_index")
        if (
            not isinstance(location_frame, int)
            or isinstance(location_frame, bool)
            or not start <= location_frame < end
        ):
            raise ValueError(f"event {index}: invalid location_frame_index")
        if event.get("location_rule") not in LOCATION_RULES:
            raise ValueError(f"event {index}: invalid location_rule")
        if event.get("ambiguity") not in AMBIGUITIES:
            raise ValueError(f"event {index}: invalid ambiguity")
        if event["ambiguity"] not in {"none", "unscorable"}:
            raise ValueError(f"event {index}: unresolved ambiguity {event['ambiguity']!r}")
        cell = event.get("cell")
        if cell is not None and (
            not isinstance(cell, list)
            or len(cell) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in cell)
            or not 0 <= cell[0] < ACTION_GRID.cols
            or not 0 <= cell[1] < ACTION_GRID.rows
        ):
            raise ValueError(f"event {index}: invalid [column,row] cell")
        if (cell is None) != (event["location_rule"] == "unavailable"):
            raise ValueError(
                f"event {index}: cell must be null exactly when location_rule is unavailable"
            )
        evidence = event.get("evidence")
        if not isinstance(evidence, dict) or set(evidence) != set(EVIDENCE_KEYS):
            raise ValueError(
                f"event {index}: evidence must contain exactly {', '.join(EVIDENCE_KEYS)}"
            )
        if any(
            value is not True and value is not False and value is not None
            for value in evidence.values()
        ):
            raise ValueError(f"event {index}: evidence values must be true, false, or null")
        _validate_event_evidence(index, side, base_card, evidence)
        review_artifacts = event.get("review_artifacts")
        if (
            not isinstance(review_artifacts, list)
            or not review_artifacts
            or any(not isinstance(path, str) or not path for path in review_artifacts)
        ):
            raise ValueError(
                f"event {index}: review_artifacts must be a non-empty list of paths"
            )
        key = (side, card, frame)
        if key in seen:
            raise ValueError(f"event {index}: duplicate event {key}")
        seen.add(key)
        validated.append(event)
    return validated


def _validate_event_evidence(
    index: int, side: str, base_card: str, evidence: dict[str, bool | None]
) -> None:
    kind = CARD_METADATA[base_card]["kind"]
    if side == "own":
        if evidence["elixir_drop"] is not True or not (
            evidence["hand_transition"] is True
            or evidence["deployment_onset"] is True
        ):
            raise ValueError(
                f"event {index}: own event requires elixir_drop plus hand_transition "
                "or deployment_onset evidence"
            )
    elif kind == "spell":
        if evidence["side_direction"] is not True or evidence["impact_sequence"] is not True:
            raise ValueError(
                f"event {index}: enemy spell requires side_direction and impact_sequence evidence"
            )
    elif evidence["first_visible_object"] is not True or evidence["deployment_onset"] is not True:
        raise ValueError(
            f"event {index}: enemy unit/building requires first_visible_object "
            "and deployment_onset evidence"
        )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _crop(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    return frame[y : y + height, x : x + width]


def _own_hud_crop(frame: np.ndarray) -> np.ndarray:
    return frame[2000:2400, 30:1060]


def _difference_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (180, 240), interpolation=cv2.INTER_AREA)


def _mean_absolute_difference(
    previous: np.ndarray | None, current: np.ndarray
) -> float | None:
    if previous is None:
        return None
    return float(np.mean(cv2.absdiff(previous, current)))


def _find_candidate(manifest: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    discovery = manifest["candidate_discovery"]
    candidates: Iterable[dict[str, Any]] = (
        list(discovery["own_candidates"]) + list(discovery["enemy_scan_windows"])
    )
    for candidate in candidates:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    raise KeyError(f"unknown candidate_id: {candidate_id}")


def _make_contact_sheet(tiles: list[np.ndarray], *, columns: int) -> np.ndarray:
    if not tiles:
        raise ValueError("cannot render an empty contact sheet")
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = math.ceil(len(tiles) / columns)
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        sheet[
            row * tile_height : row * tile_height + tile.shape[0],
            col * tile_width : col * tile_width + tile.shape[1],
        ] = tile
    return sheet
