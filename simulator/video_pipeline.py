"""Autonomous, leakage-safe video truth mining contracts.

The expensive detector/tracker's job is to emit a small JSON observation
manifest.  This module performs the deterministic part of the pipeline:

* enforce the YersonCz, pre-2023-06-19 source boundary;
* detect/record the two known HUD profiles without guessing ambiguous frames;
* keep only isolated, high-confidence tracks suitable for movement/interaction
  truth, including detector-independent shape and speed-stability gates;
* assign whole videos to calibration/validation/held-out splits; and
* emit provenance and retention records before raw-video eviction is allowed.

No network, OpenCV, or neural model is imported here.  That keeps CI cheap and
means a detector upgrade can be evaluated against the same sealed manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping


VIDEO_PIPELINE_SCHEMA_VERSION = 1
SOURCE_CHANNEL_URL = "https://www.youtube.com/@yersoncz6334"
SOURCE_CHANNEL_KEY = "yersoncz6334"
PRE_EVOLUTION_CUTOFF = date(2023, 6, 19)
HUD_VARIANTS = ("standard", "alternative")
# The two HUD profiles decode the same video frames.  A small world-coordinate
# discrepancy is expected because each profile has its own calibrated crop and
# homography.  This tolerance is deliberately only a cross-check; it never
# lowers the detector confidence gate or turns two HUD runs into independent
# evidence.
HUD_AGREEMENT_TOLERANCE_MTILE = 250
HUD_AGREEMENT_MIN_OVERLAP_SAMPLES = 5
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Candidate rows are produced by the repository's offline card/action miner.
# They are *not* truth by themselves: this adapter only uses them to choose
# inexpensive windows for the vision extractor.  Keeping the selection here,
# next to source validation, makes the high-scale path reproducible without
# making the audio classifier an authority for card identity or placement.
ACTION_WINDOW_SCHEMA_VERSION = 1
DEFAULT_ACTION_WINDOW_BEFORE_S = 2.0
DEFAULT_ACTION_WINDOW_AFTER_S = 8.0


class VideoPipelineError(ValueError):
    """Raised when an observation manifest is unsafe or ambiguous."""


@dataclass(frozen=True, slots=True)
class HudProfile:
    name: str
    hand_y: int
    elixir_y: int
    tolerance: int = 45


HUD_PROFILES = {
    "standard": HudProfile("standard", hand_y=2_020, elixir_y=2_310),
    "alternative": HudProfile("alternative", hand_y=1_960, elixir_y=2_250),
}


def _parse_date(value: object) -> date | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise VideoPipelineError(f"invalid upload date: {value!r}")


def _video_id(value: object) -> str:
    text = str(value or "").strip()
    if not _VIDEO_ID_RE.fullmatch(text):
        raise VideoPipelineError("video_id must be an 11-character YouTube ID")
    return text


def _sha256(value: object, field: str) -> str:
    text = str(value or "")
    if not _SHA256_RE.fullmatch(text):
        raise VideoPipelineError(f"{field} must be sha256:<64 lowercase hex characters>")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _inspect_replay_cache_completeness(
    path: Path,
    *,
    expected_start_s: float | None,
    expected_duration_s: float | None,
    sample_interval_s: float | None,
) -> dict[str, Any] | None:
    """Inspect a sealed replay cache without loading the vision stack.

    An interrupted extractor can leave a perfectly readable gzip/pickle cache
    containing only the prefix of a requested window.  Treating that file as
    complete makes a resumed high-scale run silently skip the missing suffix.
    The extractor cache format is intentionally read lazily here; opaque files
    (including legacy test fixtures) return ``None`` and retain the historical
    "sealed artifact" behavior.
    """

    if (
        expected_start_s is None
        or expected_duration_s is None
        or expected_duration_s <= 0
        or sample_interval_s is None
        or sample_interval_s <= 0
    ):
        return None
    try:
        from cr_bot.replay.cache import ReplayCacheReader

        iterator = iter(ReplayCacheReader(path))
        first = next(iterator)
        count = 1
        last = first
        for frame in iterator:
            count += 1
            last = frame
    except (EOFError, OSError, TypeError, ValueError, AttributeError, StopIteration):
        # A non-replay file is kept as an opaque sealed artifact.  This is
        # important for callers which deliberately seed a cache path before a
        # scheduler run; only recognized replay caches can be proven partial.
        return None

    start = float(first.video_time_s)
    end = float(last.video_time_s)
    tolerance = max(0.25, float(sample_interval_s) * 2.0)
    expected_end = float(expected_start_s) + float(expected_duration_s)
    complete = (
        start <= float(expected_start_s) + tolerance
        and end >= expected_end - tolerance
    )
    return {
        "recognized": True,
        "complete": complete,
        "frame_count": count,
        "first_video_time_s": start,
        "last_video_time_s": end,
        "expected_start_time_s": float(expected_start_s),
        "expected_end_time_s": expected_end,
        "sample_interval_s": float(sample_interval_s),
    }


def validate_source_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one source row and fail closed outside the V1 source scope."""

    channel = str(entry.get("channel_url") or entry.get("source_channel") or "").rstrip("/")
    if channel not in {SOURCE_CHANNEL_URL, f"{SOURCE_CHANNEL_URL}/videos"}:
        raise VideoPipelineError(
            f"source channel is outside the YersonCz V1 boundary: {channel!r}"
        )
    video_id = _video_id(entry.get("video_id") or entry.get("id"))
    uploaded = _parse_date(entry.get("upload_date"))
    if uploaded is None:
        raise VideoPipelineError(f"{video_id}: exact upload_date is required")
    if uploaded >= PRE_EVOLUTION_CUTOFF:
        raise VideoPipelineError(
            f"{video_id}: upload date {uploaded.isoformat()} is not pre-evolution"
        )
    normalized = dict(entry)
    normalized.update(
        {
            "channel_url": SOURCE_CHANNEL_URL,
            "channel_key": SOURCE_CHANNEL_KEY,
            "video_id": video_id,
            "upload_date": uploaded.isoformat(),
            "source_eligible": True,
            "publication_cutoff_exclusive": PRE_EVOLUTION_CUTOFF.isoformat(),
        }
    )
    if entry.get("media_sha256") is not None:
        normalized["media_sha256"] = _sha256(entry["media_sha256"], "media_sha256")
    return normalized


def filter_source_manifest(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return accepted/rejected source rows deterministically.

    Rejections are retained as evidence rather than silently disappearing;
    this is useful when a remote manifest omits dates or includes post-cutoff
    videos.
    """

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        try:
            if not isinstance(entry, Mapping):
                raise VideoPipelineError("source entry must be an object")
            row = validate_source_entry(entry)
            if row["video_id"] in seen:
                raise VideoPipelineError("duplicate video_id")
            seen.add(row["video_id"])
            accepted.append(row)
        except (TypeError, VideoPipelineError) as error:
            rejected.append(
                {
                    "index": str(index),
                    "video_id": str(
                        entry.get("video_id") or entry.get("id") or ""
                        if isinstance(entry, Mapping)
                        else ""
                    ),
                    "reason": str(error),
                }
            )
    accepted.sort(key=lambda row: (row["upload_date"], row["video_id"]))
    return {
        "schema_version": VIDEO_PIPELINE_SCHEMA_VERSION,
        "kind": "simulator_video_source_manifest",
        "source_channel": SOURCE_CHANNEL_URL,
        "publication_cutoff_exclusive": PRE_EVOLUTION_CUTOFF.isoformat(),
        "accepted": accepted,
        "rejected": rejected,
    }


def _candidate_file_sha256(path: Path) -> str:
    """Hash an action-candidate file without importing the mining stack."""

    return _file_sha256(path)


def _load_action_candidates(
    path: Path,
    *,
    video_id: str,
    supported_cards: set[str],
    confidence_threshold: float,
    duration_s: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Read and confidence-gate one JSONL candidate file.

    Candidate files are intentionally treated as untrusted detector output.
    A malformed row or an unsupported form is recorded as a rejection, while
    clean rows remain available for window selection.  No row is promoted to
    a simulator observation by this function.
    """

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    if not path.is_file():
        return accepted, [{"video_id": video_id, "reason": "missing_candidate_file"}]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return accepted, [{"video_id": video_id, "reason": f"candidate_read_error:{error}"}]
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise VideoPipelineError("candidate row must be an object")
            row_video_id = str(row.get("video_id") or "").strip()
            if row_video_id != video_id:
                raise VideoPipelineError("candidate video_id does not match source")
            card_raw = str(row.get("card") or row.get("best_class") or "").strip()
            if not card_raw:
                raise VideoPipelineError("candidate card is missing")
            # Normalization is supplied by the fixed ruleset at the caller;
            # candidates with an evolution/hero suffix can never be silently
            # collapsed onto a base card.
            card_id = card_raw.lower().replace("_", "-")
            if card_id.endswith(("-evolution", "-hero", "-champion")):
                raise VideoPipelineError("excluded alternative card form")
            if card_id not in supported_cards:
                raise VideoPipelineError("card is outside the fixed V1 interaction set")
            confidence = row.get("avg_confidence", row.get("confidence"))
            if type(confidence) not in (int, float) or not 0 <= float(confidence) <= 1:
                raise VideoPipelineError("candidate confidence is invalid")
            confidence = float(confidence)
            if confidence < confidence_threshold:
                raise VideoPipelineError("candidate confidence below threshold")
            video_time = row.get("video_time_s")
            if type(video_time) not in (int, float) or not math.isfinite(float(video_time)):
                raise VideoPipelineError("candidate video_time_s is invalid")
            video_time = float(video_time)
            if video_time < 0 or (duration_s is not None and video_time >= duration_s):
                raise VideoPipelineError("candidate time is outside the source video")
            # A window is useful only when the card/action anchor is clocked.
            # ``frame_confirmed`` is accepted as a second, explicit timing
            # signal for old candidates whose clock OCR is unavailable.
            if not bool(row.get("clock_confirmed")) and not bool(row.get("frame_confirmed")):
                raise VideoPipelineError("candidate has no confirmed time anchor")
            event_id = str(row.get("event_id") or f"line-{line_index}").strip()
            if not event_id:
                raise VideoPipelineError("candidate event_id is empty")
            cell = row.get("cell")
            if cell is not None:
                if (
                    not isinstance(cell, list)
                    or len(cell) != 2
                    or any(type(value) is not int for value in cell)
                ):
                    raise VideoPipelineError("candidate cell must be [column,row]")
                cell = [int(cell[0]), int(cell[1])]
            accepted.append(
                {
                    "event_id": event_id,
                    "video_id": video_id,
                    "card_id": card_id,
                    "video_time_s": video_time,
                    "frame_index": (
                        int(row["frame_index"])
                        if type(row.get("frame_index")) is int
                        else None
                    ),
                    "cell": cell,
                    "confidence": confidence,
                    "clock_confirmed": bool(row.get("clock_confirmed")),
                    "frame_confirmed": bool(row.get("frame_confirmed")),
                    "quality_tier": str(row.get("quality_tier") or "unknown"),
                    "source_candidate_line": line_index + 1,
                }
            )
        except (TypeError, ValueError, VideoPipelineError, json.JSONDecodeError) as error:
            rejected.append(
                {
                    "video_id": video_id,
                    "candidate_line": str(line_index + 1),
                    "reason": str(error),
                }
            )
    accepted.sort(key=lambda row: (-float(row["confidence"]), float(row["video_time_s"]), str(row["event_id"])))
    return accepted, rejected


def _canonical_v1_candidate_cards() -> set[str]:
    """Load canonical card IDs lazily for action-window planning."""

    from .ruleset import load_ruleset

    return set(load_ruleset("v1").interaction_set)


def build_action_window_manifest(
    source_manifest: Mapping[str, Any],
    candidates_root: str | Path,
    *,
    confidence_threshold: float = 0.85,
    max_windows_per_video: int = 8,
    window_before_s: float = DEFAULT_ACTION_WINDOW_BEFORE_S,
    window_after_s: float = DEFAULT_ACTION_WINDOW_AFTER_S,
    minimum_window_separation_s: float = 3.0,
    split_salt: str = "simulator-v1-video-split",
    supported_cards: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Select reproducible interaction windows from mined action candidates.

    The result is an *extractor plan input*, not a truth corpus.  A fixed
    greedy selector first gives every supported card a chance, then fills the
    remaining budget with the highest-confidence events.  Windows from one
    source video never cross a split boundary, and every accepted row records
    the candidate-file hash and the exact detector events that caused the
    extraction request.  This lets a nightly job process millions of frames
    while retaining zero human labels in the selection loop.
    """

    if not 0 < float(confidence_threshold) <= 1:
        raise VideoPipelineError("confidence_threshold must be in (0, 1]")
    if type(max_windows_per_video) is not int or max_windows_per_video < 1:
        raise VideoPipelineError("max_windows_per_video must be positive")
    for value, name in (
        (window_before_s, "window_before_s"),
        (window_after_s, "window_after_s"),
        (minimum_window_separation_s, "minimum_window_separation_s"),
    ):
        if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0:
            raise VideoPipelineError(f"{name} must be finite and non-negative")
    if window_before_s + window_after_s <= 0:
        raise VideoPipelineError("action window must have positive duration")
    if not isinstance(split_salt, str) or not split_salt.strip():
        raise VideoPipelineError("split_salt must be non-empty")
    root = Path(candidates_root)
    cards = set(supported_cards) if supported_cards is not None else _canonical_v1_candidate_cards()
    if not cards:
        raise VideoPipelineError("supported_cards must not be empty")
    accepted_sources = source_manifest.get("accepted")
    if not isinstance(accepted_sources, list):
        raise VideoPipelineError("source manifest must contain an accepted array")

    windows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for raw_source in accepted_sources:
        source = validate_source_entry(raw_source)
        video_id = source["video_id"]
        candidate_path = root / f"{video_id}.jsonl"
        candidates, candidate_rejections = _load_action_candidates(
            candidate_path,
            video_id=video_id,
            supported_cards=cards,
            confidence_threshold=float(confidence_threshold),
            duration_s=(
                float(source["duration_s"])
                if type(source.get("duration_s")) in (int, float)
                and float(source["duration_s"]) > 0
                else None
            ),
        )
        rejected.extend(candidate_rejections)
        if not candidates:
            rejected.append({"video_id": video_id, "reason": "no_eligible_action_candidates"})
            continue

        # One high-confidence representative per card is selected first.  A
        # second pass then adds strategically important interactions while
        # respecting a minimum temporal separation.  The sort key is fully
        # explicit, so adding an unrelated JSONL line cannot reshuffle an
        # already selected window.
        by_card: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            by_card.setdefault(str(row["card_id"]), []).append(row)
        ordered: list[dict[str, Any]] = []
        for card_id in sorted(by_card):
            ordered.append(by_card[card_id][0])
        ordered.extend(
            row
            for row in candidates
            if row not in ordered
        )
        selected: list[dict[str, Any]] = []
        for row in ordered:
            if len(selected) >= max_windows_per_video:
                break
            time_s = float(row["video_time_s"])
            if any(abs(time_s - float(item["video_time_s"])) < float(minimum_window_separation_s) for item in selected):
                continue
            selected.append(row)
        # If all card representatives were clustered in one fight, fill the
        # remaining slots with the best separated events rather than silently
        # returning fewer windows than the source can support.
        if len(selected) < max_windows_per_video:
            for row in candidates:
                if row in selected:
                    continue
                time_s = float(row["video_time_s"])
                if any(abs(time_s - float(item["video_time_s"])) < float(minimum_window_separation_s) for item in selected):
                    continue
                selected.append(row)
                if len(selected) >= max_windows_per_video:
                    break
        selected.sort(key=lambda row: (float(row["video_time_s"]), str(row["event_id"])))
        for index, anchor in enumerate(selected):
            anchor_time = float(anchor["video_time_s"])
            start = max(0.0, anchor_time - float(window_before_s))
            end = anchor_time + float(window_after_s)
            duration = source.get("duration_s")
            if type(duration) in (int, float) and float(duration) > 0:
                end = min(end, float(duration))
            if end <= start:
                rejected.append({"video_id": video_id, "reason": "empty_action_window"})
                continue
            nearby = [
                row
                for row in candidates
                if start <= float(row["video_time_s"]) <= end
            ]
            window_id = f"{video_id}:action-window:{index:03d}"
            windows.append(
                {
                    **source,
                    "window_id": window_id,
                    "analysis_start_time_s": round(start, 3),
                    "analysis_duration_s": round(end - start, 3),
                    "split": assign_video_split(video_id, salt=split_salt),
                    "anchor_event_id": str(anchor["event_id"]),
                    "anchor_card_id": str(anchor["card_id"]),
                    "anchor_video_time_s": anchor_time,
                    "candidate_events": nearby,
                    "candidate_file": str(candidate_path),
                    "candidate_file_sha256": _candidate_file_sha256(candidate_path),
                    "selection_method": "round_robin_card_coverage_then_confidence_v1",
                }
            )
    windows.sort(key=lambda row: (str(row["video_id"]), float(row["analysis_start_time_s"]), str(row["window_id"])))
    return {
        "schema_version": ACTION_WINDOW_SCHEMA_VERSION,
        "kind": "simulator_video_action_window_manifest",
        "source_channel": SOURCE_CHANNEL_URL,
        "publication_cutoff_exclusive": PRE_EVOLUTION_CUTOFF.isoformat(),
        "confidence_threshold": float(confidence_threshold),
        "max_windows_per_video": max_windows_per_video,
        "window_before_s": float(window_before_s),
        "window_after_s": float(window_after_s),
        "minimum_window_separation_s": float(minimum_window_separation_s),
        "split_salt": split_salt,
        "accepted": windows,
        "rejected": rejected,
        "summary": {
            "source_count": len(accepted_sources),
            "window_count": len(windows),
            "source_windows": {
                video_id: sum(row["video_id"] == video_id for row in windows)
                for video_id in sorted({str(row["video_id"]) for row in windows})
            },
            "card_coverage": {
                card_id: sum(
                    any(str(event["card_id"]) == card_id for event in row["candidate_events"])
                    for row in windows
                )
                for card_id in sorted({
                    str(event["card_id"])
                    for row in windows
                    for event in row["candidate_events"]
                })
            },
            "rejected_candidate_count": len(rejected),
        },
    }


def build_action_window_extractor_jobs(
    window_manifest: Mapping[str, Any],
    *,
    output_root: str | Path = "outputs/simulator/fidelity_media/extractor-action-windows",
    sample_interval_s: float = 0.1,
    yolo_detections: bool = True,
) -> dict[str, Any]:
    """Build both-HUD extractor jobs for an action-window manifest."""

    rows = window_manifest.get("accepted")
    if not isinstance(rows, list):
        raise VideoPipelineError("action-window manifest must contain accepted rows")
    if sample_interval_s <= 0:
        raise VideoPipelineError("sample_interval_s must be positive")
    root = Path(output_root)
    jobs: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise VideoPipelineError("action-window row must be an object")
        source = validate_source_entry(raw)
        window_id = str(raw.get("window_id") or "").strip()
        if not window_id:
            raise VideoPipelineError("action-window row needs window_id")
        video_value = raw.get("analysis_video_path") or raw.get("video_path") or raw.get("raw_path")
        video_path = Path(str(video_value)) if video_value else None
        available = video_path is not None and video_path.is_file()
        start = raw.get("analysis_start_time_s")
        duration = raw.get("analysis_duration_s")
        if type(start) not in (int, float) or float(start) < 0:
            raise VideoPipelineError(f"{window_id}: invalid analysis_start_time_s")
        if type(duration) not in (int, float) or float(duration) <= 0:
            raise VideoPipelineError(f"{window_id}: invalid analysis_duration_s")
        for hud_variant in HUD_VARIANTS:
            replay_path = root / window_id / hud_variant / "replay-cache.json"
            command = (
                extractor_command(
                    video_path,
                    replay_path,
                    hud_variant=hud_variant,
                    sample_interval_s=sample_interval_s,
                    yolo_detections=yolo_detections,
                    video_start_time_s=float(start),
                    video_duration_s=float(duration),
                )
                if available
                else None
            )
            jobs.append(
                {
                    "job_id": f"{window_id}:{hud_variant}",
                    "window_id": window_id,
                    "video_id": source["video_id"],
                    "hud_variant": hud_variant,
                    "video_path": str(video_path) if video_path is not None else None,
                    "replay_cache_path": str(replay_path),
                    "command": command,
                    "status": "ready" if available else "missing_video_path",
                    "source": {
                        "channel_url": SOURCE_CHANNEL_URL,
                        "upload_date": source["upload_date"],
                        "media_sha256": source.get("media_sha256"),
                        "analysis_start_time_s": float(start),
                        "analysis_duration_s": float(duration),
                        "split": raw.get("split"),
                        "anchor_card_id": raw.get("anchor_card_id"),
                        "anchor_video_time_s": raw.get("anchor_video_time_s"),
                        "candidate_file_sha256": raw.get("candidate_file_sha256"),
                    },
                }
            )
    jobs.sort(key=lambda row: str(row["job_id"]))
    return {
        "schema_version": ACTION_WINDOW_SCHEMA_VERSION,
        "kind": "simulator_video_action_window_extractor_plan",
        "source_channel": SOURCE_CHANNEL_URL,
        "publication_cutoff_exclusive": PRE_EVOLUTION_CUTOFF.isoformat(),
        "hud_variants": list(HUD_VARIANTS),
        "sample_interval_s": sample_interval_s,
        "yolo_detections": bool(yolo_detections),
        "jobs": jobs,
        "ready_job_count": sum(row["status"] == "ready" for row in jobs),
        "missing_video_path_count": sum(row["status"] == "missing_video_path" for row in jobs),
    }


def discover_source_manifest(
    source: str | Path | None = None,
    *,
    max_videos: int | None = None,
    cookies_from_browser: str | None = None,
) -> dict[str, Any]:
    """Discover and seal eligible source metadata through the repo resolver.

    Discovery is deliberately an explicit operation: importing the simulator
    never performs network access.  The resolver fetches complete YouTube
    metadata with the exclusive pre-Evolution cutoff, after which this module
    applies its stricter channel, date, ID, and duplicate checks again.  A
    caller can therefore save the returned manifest and reproduce all later
    mining without contacting YouTube.
    """

    local_source = Path(source) if source is not None and Path(source).exists() else None
    if local_source is not None:
        # Preserve local-only fields such as ``analysis_video_path``.  The
        # repository resolver intentionally normalizes remote metadata and
        # does not promise to retain those pipeline hints.
        try:
            if local_source.suffix == ".jsonl":
                entries = [
                    json.loads(line)
                    for line in local_source.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                payload = json.loads(local_source.read_text(encoding="utf-8"))
                entries = (
                    payload.get("accepted")
                    or payload.get("entries")
                    or payload.get("videos")
                    or payload.get("selected")
                    if isinstance(payload, dict)
                    else payload
                )
            if not isinstance(entries, list):
                raise VideoPipelineError("local source manifest must contain an array")
        except (OSError, TypeError, json.JSONDecodeError) as error:
            raise VideoPipelineError(f"cannot read local source manifest: {error}") from error
        if max_videos is not None:
            entries = entries[: max(0, int(max_videos))]
    else:
        try:
            from cr_bot.mining.video_manifest import resolve_video_manifest
        except ImportError as error:  # pragma: no cover - only broken installations
            raise VideoPipelineError(
                "repository video-manifest resolver is unavailable; install cr-bot first"
            ) from error
        entries = resolve_video_manifest(
            source or SOURCE_CHANNEL_URL,
            before_date=PRE_EVOLUTION_CUTOFF.isoformat(),
            max_videos=max_videos,
            cookies_from_browser=cookies_from_browser,
        )
    return filter_source_manifest(entries)


def assign_video_split(video_id: str, *, salt: str = "simulator-v1-video-split") -> str:
    """Assign a complete source video to one split using a stable hash."""

    _video_id(video_id)
    digest = hashlib.sha256(f"{salt}:{video_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "calibration"
    if bucket < 85:
        return "validation"
    return "heldout"


def select_hud_variant(
    *,
    hand_y: int | None,
    elixir_y: int | None,
    tolerance: int = 45,
) -> str | None:
    """Select a HUD profile only when exactly one profile fits observations."""

    if hand_y is None and elixir_y is None:
        return None
    if type(tolerance) is not int or tolerance < 0:
        raise VideoPipelineError("HUD tolerance must be a non-negative integer")
    candidates: list[tuple[int, str]] = []
    for name, profile in HUD_PROFILES.items():
        errors = []
        if hand_y is not None:
            errors.append(abs(int(hand_y) - profile.hand_y))
        if elixir_y is not None:
            errors.append(abs(int(elixir_y) - profile.elixir_y))
        if errors and max(errors) <= tolerance:
            candidates.append((sum(errors), name))
    if len(candidates) != 1:
        return None
    return min(candidates)[1]


def _confidence(value: object, field: str) -> float:
    if type(value) not in (int, float) or not 0 <= float(value) <= 1:
        raise VideoPipelineError(f"{field} must be between zero and one")
    return float(value)


def _int(value: object, field: str) -> int:
    if type(value) is not int:
        raise VideoPipelineError(f"{field} must be an integer")
    return value


def _track_motion_quality(
    samples: list[Mapping[str, Any]],
    *,
    fps: float,
    minimum_displacement_mtile: int,
    minimum_elapsed_s: float,
    minimum_moving_interval_fraction: float,
    moving_speed_floor_mtile_per_s: int,
    maximum_step_speed_mtile_per_s: int,
    maximum_frame_gap_factor: float,
    maximum_frame_gap: int,
    maximum_path_to_displacement_ratio: float,
    maximum_speed_iqr_ratio: float,
) -> dict[str, Any]:
    """Reject detector tracks that are not trustworthy motion observations.

    Confidence and isolation alone do not make a track a movement oracle.  A
    detector can keep an ID on a stationary sprite after it has disappeared,
    bridge two occlusions with a teleport, or emit only a pair of frames.  The
    checks here deliberately use no card speed or simulator output: they are
    generic signal-quality gates and therefore remain valid for held-out
    fidelity evaluation.
    """

    if len(samples) < 2:
        raise VideoPipelineError("track has fewer than two ordered samples")
    if type(minimum_displacement_mtile) is not int or minimum_displacement_mtile < 0:
        raise VideoPipelineError("minimum_displacement_mtile must be non-negative")
    if (
        type(minimum_elapsed_s) not in (int, float)
        or not math.isfinite(float(minimum_elapsed_s))
        or float(minimum_elapsed_s) < 0
    ):
        raise VideoPipelineError("minimum_elapsed_s must be non-negative")
    if not 0 <= float(minimum_moving_interval_fraction) <= 1:
        raise VideoPipelineError("minimum_moving_interval_fraction must be in [0, 1]")
    if type(moving_speed_floor_mtile_per_s) is not int or moving_speed_floor_mtile_per_s < 0:
        raise VideoPipelineError("moving_speed_floor_mtile_per_s must be non-negative")
    if type(maximum_step_speed_mtile_per_s) is not int or maximum_step_speed_mtile_per_s <= 0:
        raise VideoPipelineError("maximum_step_speed_mtile_per_s must be positive")
    if (
        type(maximum_frame_gap_factor) not in (int, float)
        or not math.isfinite(float(maximum_frame_gap_factor))
        or float(maximum_frame_gap_factor) < 1
    ):
        raise VideoPipelineError("maximum_frame_gap_factor must be at least one")
    if type(maximum_frame_gap) is not int or maximum_frame_gap < 1:
        raise VideoPipelineError("maximum_frame_gap must be positive")
    if (
        type(maximum_path_to_displacement_ratio) not in (int, float)
        or not math.isfinite(float(maximum_path_to_displacement_ratio))
        or float(maximum_path_to_displacement_ratio) < 1
    ):
        raise VideoPipelineError(
            "maximum_path_to_displacement_ratio must be finite and at least one"
        )
    if (
        type(maximum_speed_iqr_ratio) not in (int, float)
        or not math.isfinite(float(maximum_speed_iqr_ratio))
        or float(maximum_speed_iqr_ratio) < 0
    ):
        raise VideoPipelineError(
            "maximum_speed_iqr_ratio must be finite and non-negative"
        )
    if not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise VideoPipelineError("track source fps must be positive")

    frame_indices = [int(sample["frame_idx"]) for sample in samples]
    if any(current <= previous for previous, current in zip(frame_indices, frame_indices[1:])):
        raise VideoPipelineError("track samples contain duplicate or decreasing frame indices")
    frame_gaps = [current - previous for previous, current in zip(frame_indices, frame_indices[1:])]
    median_gap = statistics.median(frame_gaps) if frame_gaps else 1
    allowed_gap = min(maximum_frame_gap, max(1, math.ceil(median_gap * maximum_frame_gap_factor)))
    if frame_gaps and max(frame_gaps) > allowed_gap:
        raise VideoPipelineError(
            f"track has an implausible frame gap ({max(frame_gaps)} > {allowed_gap})"
        )

    times: list[float] = []
    for sample in samples:
        value = sample.get("video_time_s")
        if value is None:
            value = float(sample["frame_idx"]) / float(fps)
        if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0:
            raise VideoPipelineError("track sample time must be finite and non-negative")
        times.append(float(value))
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise VideoPipelineError("track samples contain duplicate or decreasing times")
    elapsed_s = times[-1] - times[0]
    if elapsed_s < float(minimum_elapsed_s):
        raise VideoPipelineError(
            f"track elapsed time {elapsed_s:.3f}s is shorter than {float(minimum_elapsed_s):.3f}s"
        )

    interval_distances: list[float] = []
    interval_speeds: list[float] = []
    for previous, current, previous_time, current_time in zip(
        samples, samples[1:], times, times[1:]
    ):
        distance = math.hypot(
            int(current["x_mtile"]) - int(previous["x_mtile"]),
            int(current["y_mtile"]) - int(previous["y_mtile"]),
        )
        interval_distances.append(distance)
        interval_speeds.append(distance / (current_time - previous_time))
    maximum_step_speed = max(interval_speeds)
    if maximum_step_speed > maximum_step_speed_mtile_per_s:
        raise VideoPipelineError(
            "track contains an implausible motion step "
            f"({maximum_step_speed:.1f} > {maximum_step_speed_mtile_per_s} mtile/s)"
        )
    displacement = math.hypot(
        int(samples[-1]["x_mtile"]) - int(samples[0]["x_mtile"]),
        int(samples[-1]["y_mtile"]) - int(samples[0]["y_mtile"]),
    )
    if displacement < minimum_displacement_mtile:
        raise VideoPipelineError(
            f"track displacement {displacement:.1f} mtile is shorter than "
            f"{minimum_displacement_mtile} mtile"
        )
    moving_fraction = sum(
        speed >= moving_speed_floor_mtile_per_s for speed in interval_speeds
    ) / len(interval_speeds)
    if moving_fraction < float(minimum_moving_interval_fraction):
        raise VideoPipelineError(
            "track has too few moving intervals "
            f"({moving_fraction:.3f} < {float(minimum_moving_interval_fraction):.3f})"
        )
    path_length = sum(interval_distances)
    path_to_displacement_ratio = path_length / max(displacement, 1.0)
    if path_to_displacement_ratio > float(maximum_path_to_displacement_ratio):
        raise VideoPipelineError(
            "track path is too irregular for a displacement truth "
            f"({path_to_displacement_ratio:.3f} > "
            f"{float(maximum_path_to_displacement_ratio):.3f})"
        )
    median_step_speed = statistics.median(interval_speeds)
    if len(interval_speeds) >= 4:
        first_quartile, third_quartile = statistics.quantiles(
            interval_speeds, n=4, method="inclusive"
        )[0::2]
    else:
        first_quartile = min(interval_speeds)
        third_quartile = max(interval_speeds)
    speed_iqr = third_quartile - first_quartile
    speed_iqr_ratio = speed_iqr / max(median_step_speed, 1.0)
    if speed_iqr_ratio > float(maximum_speed_iqr_ratio):
        raise VideoPipelineError(
            "track step speed is too unstable for a movement truth "
            f"({speed_iqr_ratio:.3f} > {float(maximum_speed_iqr_ratio):.3f})"
        )
    return {
        "sample_count": len(samples),
        "elapsed_s": round(elapsed_s, 6),
        "frame_gap_max": max(frame_gaps) if frame_gaps else 0,
        "frame_gap_median": round(float(median_gap), 6),
        "displacement_mtile": round(displacement, 6),
        "path_length_mtile": round(path_length, 6),
        "path_to_displacement_ratio": round(path_to_displacement_ratio, 6),
        "moving_interval_fraction": round(float(moving_fraction), 6),
        "median_step_speed_mtile_per_s": round(float(median_step_speed), 6),
        "speed_iqr_mtile_per_s": round(float(speed_iqr), 6),
        "speed_iqr_ratio": round(float(speed_iqr_ratio), 6),
        "maximum_step_speed_mtile_per_s": round(float(maximum_step_speed), 6),
    }


def mine_clean_tracks(
    source_manifest: Mapping[str, Any],
    *,
    confidence_threshold: float = 0.98,
    minimum_track_frames: int = 20,
    split_salt: str = "simulator-v1-video-split",
    minimum_displacement_mtile: int = 500,
    minimum_elapsed_s: float = 0.25,
    minimum_moving_interval_fraction: float = 0.5,
    moving_speed_floor_mtile_per_s: int = 250,
    maximum_step_speed_mtile_per_s: int = 6_000,
    maximum_frame_gap_factor: float = 4.0,
    maximum_frame_gap: int = 60,
    maximum_path_to_displacement_ratio: float = 3.0,
    maximum_speed_iqr_ratio: float = 2.0,
) -> dict[str, Any]:
    """Filter detector output into truth candidates without human promotion.

    Expected input rows are ``source_manifest.accepted`` with optional
    ``tracks`` arrays.  Each track contains integer ``samples`` with
    ``frame_idx``, ``x_mtile``, ``y_mtile`` and confidence/occlusion metadata.
    Ambiguous tracks are discarded and listed with a reason.
    """

    if not 0 < confidence_threshold <= 1:
        raise VideoPipelineError("confidence_threshold must be in (0, 1]")
    if type(minimum_track_frames) is not int or minimum_track_frames < 2:
        raise VideoPipelineError("minimum_track_frames must be at least two")
    if not isinstance(split_salt, str) or not split_salt.strip():
        raise VideoPipelineError("split_salt must be a non-empty string")
    if type(minimum_displacement_mtile) is not int or minimum_displacement_mtile < 0:
        raise VideoPipelineError("minimum_displacement_mtile must be non-negative")
    if (
        type(minimum_elapsed_s) not in (int, float)
        or not math.isfinite(float(minimum_elapsed_s))
        or float(minimum_elapsed_s) < 0
    ):
        raise VideoPipelineError("minimum_elapsed_s must be non-negative")
    if not 0 <= float(minimum_moving_interval_fraction) <= 1:
        raise VideoPipelineError("minimum_moving_interval_fraction must be in [0, 1]")
    if type(moving_speed_floor_mtile_per_s) is not int or moving_speed_floor_mtile_per_s < 0:
        raise VideoPipelineError("moving_speed_floor_mtile_per_s must be non-negative")
    if type(maximum_step_speed_mtile_per_s) is not int or maximum_step_speed_mtile_per_s <= 0:
        raise VideoPipelineError("maximum_step_speed_mtile_per_s must be positive")
    if (
        type(maximum_frame_gap_factor) not in (int, float)
        or not math.isfinite(float(maximum_frame_gap_factor))
        or float(maximum_frame_gap_factor) < 1
    ):
        raise VideoPipelineError("maximum_frame_gap_factor must be at least one")
    if type(maximum_frame_gap) is not int or maximum_frame_gap < 1:
        raise VideoPipelineError("maximum_frame_gap must be positive")
    if (
        type(maximum_path_to_displacement_ratio) not in (int, float)
        or not math.isfinite(float(maximum_path_to_displacement_ratio))
        or float(maximum_path_to_displacement_ratio) < 1
    ):
        raise VideoPipelineError(
            "maximum_path_to_displacement_ratio must be finite and at least one"
        )
    if (
        type(maximum_speed_iqr_ratio) not in (int, float)
        or not math.isfinite(float(maximum_speed_iqr_ratio))
        or float(maximum_speed_iqr_ratio) < 0
    ):
        raise VideoPipelineError(
            "maximum_speed_iqr_ratio must be finite and non-negative"
        )
    accepted_sources = source_manifest.get("accepted")
    if not isinstance(accepted_sources, list):
        raise VideoPipelineError("source manifest must contain an accepted array")
    cases: list[dict[str, Any]] = []
    discarded: list[dict[str, str]] = []
    for raw_source in accepted_sources:
        source = validate_source_entry(raw_source)
        video_id = source["video_id"]
        source_group_id = _source_group_key(source)
        split = assign_video_split(video_id, salt=split_salt)
        tracks = raw_source.get("tracks", [])
        if not isinstance(tracks, list):
            raise VideoPipelineError(f"{video_id}: tracks must be an array")
        for track_index, raw_track in enumerate(tracks):
            try:
                if not isinstance(raw_track, Mapping):
                    raise VideoPipelineError("track must be an object")
                track_id = str(raw_track.get("track_id") or "").strip()
                card_id = str(raw_track.get("card_id") or "").strip()
                if not track_id or not card_id:
                    raise VideoPipelineError("track_id and card_id are required")
                owner = raw_track.get("owner", "ally")
                if owner not in {"ally", "enemy", 0, 1, "0", "1"}:
                    raise VideoPipelineError("track.owner must be ally/enemy or 0/1")
                track_confidence = _confidence(
                    raw_track.get("confidence", 0), "track.confidence"
                )
                if track_confidence < confidence_threshold:
                    raise VideoPipelineError("track confidence below threshold")
                if raw_track.get("occluded", False):
                    raise VideoPipelineError("track is occluded")
                samples = raw_track.get("samples", [])
                if not isinstance(samples, list) or len(samples) < minimum_track_frames:
                    raise VideoPipelineError("track is shorter than minimum frames")
                normalized_samples: list[dict[str, Any]] = []
                for sample_index, raw_sample in enumerate(samples):
                    if not isinstance(raw_sample, Mapping):
                        raise VideoPipelineError("sample must be an object")
                    frame_idx = _int(raw_sample.get("frame_idx"), "sample.frame_idx")
                    x = _int(raw_sample.get("x_mtile"), "sample.x_mtile")
                    y = _int(raw_sample.get("y_mtile"), "sample.y_mtile")
                    sample_confidence = _confidence(
                        raw_sample.get("confidence", track_confidence),
                        "sample.confidence",
                    )
                    if sample_confidence < confidence_threshold:
                        raise VideoPipelineError("sample confidence below threshold")
                    if raw_sample.get("nearby_entities", 0):
                        raise VideoPipelineError("track has nearby entities")
                    if raw_sample.get("occluded", False):
                        raise VideoPipelineError("sample is occluded")
                    normalized_samples.append(
                        {
                            "frame_idx": frame_idx,
                            "x_mtile": x,
                            "y_mtile": y,
                            "confidence": sample_confidence,
                        }
                    )
                    if raw_sample.get("video_time_s") is not None:
                        video_time = raw_sample["video_time_s"]
                        if (
                            type(video_time) not in (int, float)
                            or not float(video_time) >= 0
                        ):
                            raise VideoPipelineError(
                                "sample.video_time_s must be a non-negative number"
                            )
                        normalized_samples[-1]["video_time_s"] = float(video_time)
                normalized_samples.sort(key=lambda row: row["frame_idx"])
                source_fps = source.get("fps", 30.0)
                try:
                    source_fps = float(source_fps)
                except (TypeError, ValueError):
                    source_fps = 30.0
                motion_quality = _track_motion_quality(
                    normalized_samples,
                    fps=source_fps,
                    minimum_displacement_mtile=minimum_displacement_mtile,
                    minimum_elapsed_s=minimum_elapsed_s,
                    minimum_moving_interval_fraction=minimum_moving_interval_fraction,
                    moving_speed_floor_mtile_per_s=moving_speed_floor_mtile_per_s,
                    maximum_step_speed_mtile_per_s=maximum_step_speed_mtile_per_s,
                    maximum_frame_gap_factor=maximum_frame_gap_factor,
                    maximum_frame_gap=maximum_frame_gap,
                    maximum_path_to_displacement_ratio=maximum_path_to_displacement_ratio,
                    maximum_speed_iqr_ratio=maximum_speed_iqr_ratio,
                )
                hud_variant = raw_track.get("hud_variant")
                if hud_variant == "auto":
                    hud_variant = select_hud_variant(
                        hand_y=raw_track.get("hand_y"),
                        elixir_y=raw_track.get("elixir_y"),
                    )
                if hud_variant is None:
                    hud_variant = raw_track.get("hud_variant") or source.get("hud_variant")
                if hud_variant not in HUD_VARIANTS:
                    raise VideoPipelineError("HUD variant is ambiguous or unsupported")
                evidence = {
                    "source_id": f"yersoncz:{video_id}",
                    "source_group_id": source_group_id,
                    "method": "offline_detector_track_v1",
                    "media_sha256": source.get("media_sha256"),
                    "publication_date": source["upload_date"],
                    "source_channel": SOURCE_CHANNEL_URL,
                }
                # A dual-HUD comparison is a useful detector quality signal,
                # but it is explicitly not independent truth.  Preserve it
                # in provenance so candidate reports can be ranked without
                # silently weakening the confidence/held-out gates.
                hud_agreement = raw_track.get("hud_agreement")
                if isinstance(hud_agreement, Mapping):
                    evidence["hud_agreement"] = dict(hud_agreement)
                    evidence["hud_agreement_independent_evidence"] = False
                evidence["motion_quality"] = motion_quality
                cases.append(
                    {
                        "case_id": f"{source_group_id}:{track_id}",
                        "video_id": video_id,
                        "group_id": video_id,
                        "source_group_id": source_group_id,
                        "split": split,
                        "mechanic": f"{card_id}_isolated_movement_{hud_variant}",
                        "card_id": card_id,
                        "track_id": track_id,
                        "owner": owner,
                        "hud_variant": hud_variant,
                        "frame_start": normalized_samples[0]["frame_idx"],
                        "frame_end": normalized_samples[-1]["frame_idx"],
                        "fps": source.get("fps"),
                        "confidence": min(
                            track_confidence,
                            min(sample["confidence"] for sample in normalized_samples),
                        ),
                        "samples": normalized_samples,
                        "evidence": evidence,
                    }
                )
            except (TypeError, KeyError, VideoPipelineError) as error:
                discarded.append(
                    {
                        "video_id": video_id,
                        "track_index": str(track_index),
                        "reason": str(error),
                    }
                )
    cases.sort(key=lambda row: row["case_id"])
    return {
        "schema_version": VIDEO_PIPELINE_SCHEMA_VERSION,
        "kind": "simulator_video_truth_manifest",
        "source_channel": SOURCE_CHANNEL_URL,
        "publication_cutoff_exclusive": PRE_EVOLUTION_CUTOFF.isoformat(),
        "confidence_threshold": confidence_threshold,
        "minimum_track_frames": minimum_track_frames,
        "motion_quality_gate": {
            "minimum_displacement_mtile": minimum_displacement_mtile,
            "minimum_elapsed_s": minimum_elapsed_s,
            "minimum_moving_interval_fraction": minimum_moving_interval_fraction,
            "moving_speed_floor_mtile_per_s": moving_speed_floor_mtile_per_s,
            "maximum_step_speed_mtile_per_s": maximum_step_speed_mtile_per_s,
            "maximum_frame_gap_factor": maximum_frame_gap_factor,
            "maximum_frame_gap": maximum_frame_gap,
            "maximum_path_to_displacement_ratio": maximum_path_to_displacement_ratio,
            "maximum_speed_iqr_ratio": maximum_speed_iqr_ratio,
            "simulator_independent": True,
        },
        "split_salt": split_salt,
        "cases": cases,
        "discarded": discarded,
        "summary": {
            "accepted_case_count": len(cases),
            "discarded_track_count": len(discarded),
            "truth_ready": bool(cases),
            "split_counts": {
                split: sum(case["split"] == split for case in cases)
                for split in ("calibration", "validation", "heldout")
            },
        },
    }


def replay_cache_track_manifest(
    source_entry: Mapping[str, Any],
    replay_cache_path: str | Path,
    *,
    hud_variant: str,
    confidence_threshold: float = 0.98,
    minimum_track_frames: int = 20,
    isolation_radius_mtile: int = 3_500,
) -> dict[str, Any]:
    """Convert extractor replay frames into a confidence-gated track manifest.

    This is the bridge between the repository's existing detector/tracker
    cache and :func:`mine_clean_tracks`.  It deliberately keeps only stable
    troop IDs, converts pixel detections through the same calibrated arena
    mapping as replay mining, and annotates nearby detections so crowded clips
    are discarded instead of becoming fabricated movement truth.  Spells and
    instantaneous effects are not represented as tracks; their dedicated
    event miners remain the appropriate oracle.
    """

    if hud_variant not in HUD_VARIANTS:
        raise VideoPipelineError(f"unsupported HUD variant: {hud_variant!r}")
    if not 0 < confidence_threshold <= 1:
        raise VideoPipelineError("confidence_threshold must be in (0, 1]")
    if type(minimum_track_frames) is not int or minimum_track_frames < 2:
        raise VideoPipelineError("minimum_track_frames must be at least two")
    if type(isolation_radius_mtile) is not int or isolation_radius_mtile < 1:
        raise VideoPipelineError("isolation_radius_mtile must be positive")
    source = validate_source_entry(source_entry)
    cache_path = Path(replay_cache_path)
    if not cache_path.is_file():
        raise VideoPipelineError(f"replay cache is not a file: {cache_path}")

    # Imports stay inside the adapter: the simulator's normal import path does
    # not initialize OpenCV, detector models, or the repository feature stack.
    from cr_bot.replay.cache import ReplayCacheReader
    from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD
    from cr_bot.domain.card_metadata import CARD_METADATA
    from .mining import _detection_world_position
    from .ruleset import load_ruleset

    # This adapter emits movement truth only.  Buildings and spells may have
    # detector tracks, but treating a stationary Cannon or a projectile as a
    # troop trajectory creates a vacuous ``movement_speed`` measurement and
    # can falsely look like an engine pathfinding failure.  Their dedicated
    # lifetime/damage/projectile miners consume the same cache separately.
    ruleset = load_ruleset("v1")

    track_rows: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for replay_frame in ReplayCacheReader(cache_path):
        frame_detections: list[dict[str, Any]] = []
        for match in replay_frame.analysis.matches:
            detection = match.troop
            if detection.track_id is None or detection.team not in {"ally", "enemy"}:
                continue
            card_id = DIRECT_UNIT_TO_CARD.get(detection.class_name)
            if not card_id:
                continue
            try:
                definition = ruleset.card(str(card_id))
            except KeyError:
                continue
            if definition.kind != "troop":
                continue
            # ``matches`` includes decorative bars/towers in some detector
            # versions.  The direct card mapping above and a positive ID are
            # the conservative troop boundary for this generic track adapter.
            metadata = CARD_METADATA.get(card_id, {})
            x_mtile, y_mtile = _detection_world_position(
                detection,
                replay_frame.analysis.arena_px,
                ground_anchor=not bool(metadata.get("is_air", False)),
            )
            frame_detections.append(
                {
                    "track_key": (
                        int(detection.track_id),
                        str(card_id),
                        str(detection.team),
                    ),
                    "track_id": str(detection.track_id),
                    "card_id": str(card_id),
                    "owner": str(detection.team),
                    "x_mtile": x_mtile,
                    "y_mtile": y_mtile,
                    "confidence": float(detection.confidence),
                }
            )
        for row in frame_detections:
            if row["confidence"] < confidence_threshold:
                continue
            nearby = any(
                other is not row
                and (
                    (int(row["x_mtile"]) - int(other["x_mtile"])) ** 2
                    + (int(row["y_mtile"]) - int(other["y_mtile"])) ** 2
                )
                < isolation_radius_mtile**2
                for other in frame_detections
            )
            track_rows.setdefault(row["track_key"], []).append(
                {
                    "frame_idx": int(replay_frame.frame_idx),
                    "video_time_s": float(replay_frame.video_time_s),
                    "x_mtile": int(row["x_mtile"]),
                    "y_mtile": int(row["y_mtile"]),
                    "confidence": float(row["confidence"]),
                    "nearby_entities": int(nearby),
                }
            )

    tracks: list[dict[str, Any]] = []
    for (track_id, card_id, owner), samples in sorted(track_rows.items()):
        samples.sort(key=lambda row: (row["frame_idx"], row["video_time_s"]))
        # Do not throw away an otherwise useful track because one later frame
        # becomes crowded.  Split at every non-isolated sample and retain only
        # clean contiguous runs.  This is the autonomous equivalent of a
        # human selecting an easy movement clip, while the split boundary and
        # discarded samples remain reproducible in the manifest.
        segments: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for sample in samples:
            if int(sample["nearby_entities"]) == 0:
                current.append(sample)
                continue
            if len(current) >= minimum_track_frames:
                segments.append(current)
            current = []
        if len(current) >= minimum_track_frames:
            segments.append(current)
        for segment_index, segment in enumerate(segments):
            segment_id = (
                str(track_id)
                if len(segments) == 1
                else f"{track_id}.clean{segment_index}"
            )
            tracks.append(
                {
                    "track_id": segment_id,
                    "parent_track_id": str(track_id),
                    "segment_index": segment_index,
                    "card_id": card_id,
                    "owner": owner,
                    "hud_variant": hud_variant,
                    "confidence": min(row["confidence"] for row in segment),
                    "occluded": False,
                    "samples": segment,
                }
            )
    source = dict(source)
    if source.get("media_sha256") is None:
        # Retention deletes the downloaded/raw artifact, not a derived
        # OpenCV-normalized analysis file. Hash that same raw path whenever it
        # exists; otherwise fall back to the analysis path for local fixtures.
        media_values = (
            source.get("raw_path"),
            source.get("download_path"),
            source.get("video_path"),
            source.get("analysis_video_path"),
        )
        for media_value in media_values:
            if isinstance(media_value, str) and Path(media_value).is_file():
                source["media_sha256"] = _file_sha256(Path(media_value))
                source["media_hash_path"] = str(Path(media_value))
                break
    source.update(
        {
            "hud_variant": hud_variant,
            "replay_cache_path": str(cache_path),
            "replay_cache_sha256": _file_sha256(cache_path),
            "tracks": tracks,
        }
    )
    return filter_source_manifest([source])


def _source_group_key(source: Mapping[str, Any]) -> str:
    """Return the identity of one evidence unit within a source video.

    A normal extraction has one cache per video, so its group is simply the
    YouTube ID.  Action-window extraction deliberately creates several
    independent cache jobs from the same video; those jobs must not overwrite
    one another when manifests are merged.  ``source_group_id`` is preferred
    because callers can choose a stable namespace, while ``window_id`` keeps
    older action-window manifests readable.
    """

    raw = source.get("source_group_id") or source.get("window_id")
    text = str(raw or "").strip()
    return text or str(source["video_id"])


def _discover_replay_cache_jobs(
    source: Mapping[str, Any],
    root: Path,
    *,
    hud_variant: str,
) -> list[tuple[dict[str, Any], Path]]:
    """Discover direct and action-window cache layouts for one source.

    The extractor writes full-video jobs to ``<video>/<hud>/`` and bounded
    action jobs to ``<video>:action-window:<n>/<hud>/``.  Keeping discovery in
    one deterministic adapter lets a resumed/nightly run consume either
    layout without requiring a second manifest format.
    """

    video_id = str(source["video_id"])
    candidates: list[tuple[dict[str, Any], Path]] = []
    direct = root / video_id / hud_variant / "replay-cache.json"
    if direct.is_file():
        direct_source = dict(source)
        direct_source.setdefault("source_group_id", video_id)
        candidates.append((direct_source, direct))

    prefix = f"{video_id}:action-window:"
    for window_root in sorted(
        (path for path in root.glob(f"{prefix}*") if path.is_dir()),
        key=lambda path: path.name,
    ):
        cache_path = window_root / hud_variant / "replay-cache.json"
        if not cache_path.is_file():
            continue
        window_id = window_root.name
        window_source = dict(source)
        # These fields are provenance, not a replacement for video_id.  The
        # latter remains the original source identity for split assignment and
        # media-hash lookup.
        window_source.update(
            {
                "window_id": window_id,
                "source_group_id": window_id,
                "analysis_window_id": window_id,
            }
        )
        candidates.append((window_source, cache_path))
    return candidates


def batch_replay_cache_track_manifest(
    source_manifest: Mapping[str, Any],
    extractor_root: str | Path,
    *,
    hud_variant: str,
    confidence_threshold: float = 0.98,
    minimum_track_frames: int = 20,
    isolation_radius_mtile: int = 3_500,
) -> dict[str, Any]:
    """Mine all available cache jobs for one HUD profile deterministically."""

    if hud_variant not in HUD_VARIANTS:
        raise VideoPipelineError(f"unsupported HUD variant: {hud_variant!r}")
    accepted = source_manifest.get("accepted")
    if not isinstance(accepted, list):
        raise VideoPipelineError("source manifest accepted must be an array")
    root = Path(extractor_root)
    mined: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    discovered_jobs = 0
    for index, raw_source in enumerate(accepted):
        try:
            source = validate_source_entry(raw_source)
            jobs = _discover_replay_cache_jobs(source, root, hud_variant=hud_variant)
            if not jobs:
                video_id = source["video_id"]
                expected = root / video_id / hud_variant / "replay-cache.json"
                raise VideoPipelineError(f"missing replay cache: {expected}")
            discovered_jobs += len(jobs)
            for job_source, cache_path in jobs:
                try:
                    result = replay_cache_track_manifest(
                        job_source,
                        cache_path,
                        hud_variant=hud_variant,
                        confidence_threshold=confidence_threshold,
                        minimum_track_frames=minimum_track_frames,
                        isolation_radius_mtile=isolation_radius_mtile,
                    )
                    mined.extend(result["accepted"])
                except (OSError, TypeError, VideoPipelineError) as error:
                    rejected.append(
                        {
                            "index": str(index),
                            "video_id": str(job_source["video_id"]),
                            "source_group_id": _source_group_key(job_source),
                            "reason": str(error),
                        }
                    )
        except (OSError, TypeError, VideoPipelineError) as error:
            rejected.append(
                {
                    "index": str(index),
                    "video_id": str(raw_source.get("video_id", ""))
                    if isinstance(raw_source, Mapping)
                    else "",
                    "reason": str(error),
                }
            )
    mined.sort(
        key=lambda row: (
            str(row["video_id"]),
            _source_group_key(row),
            str(row.get("replay_cache_path") or ""),
        )
    )
    return {
        "schema_version": VIDEO_PIPELINE_SCHEMA_VERSION,
        "kind": "simulator_video_source_manifest",
        "source_channel": SOURCE_CHANNEL_URL,
        "publication_cutoff_exclusive": PRE_EVOLUTION_CUTOFF.isoformat(),
        "hud_variant": hud_variant,
        "accepted": mined,
        "rejected": rejected,
        "summary": {
            "source_count": len(accepted),
            "accepted_source_count": len(mined),
            "rejected_source_count": len(rejected),
            "discovered_cache_job_count": discovered_jobs,
            "track_count": sum(len(row.get("tracks", [])) for row in mined),
        },
    }


def merge_track_manifests(
    manifests: Iterable[Mapping[str, Any]],
    *,
    hud_variant: str,
) -> dict[str, Any]:
    """Merge repeated extractor roots for one HUD profile deterministically.

    Large mining runs are commonly resumed into a new output directory.  A
    later validation manifest must be able to name both roots without making
    duplicate evidence or depending on filesystem order.  For each source
    group (a full video or a bounded action window) we keep the candidate with
    the most clean tracks, then highest mean track confidence, then the
    earliest manifest order.  The discarded candidates remain in
    ``candidate_provenance`` for auditability.
    """

    if hud_variant not in HUD_VARIANTS:
        raise VideoPipelineError(f"unsupported HUD variant: {hud_variant!r}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    rejected: list[dict[str, str]] = []
    for manifest_index, raw_manifest in enumerate(manifests):
        if not isinstance(raw_manifest, Mapping):
            raise VideoPipelineError("track manifest must be an object")
        accepted = raw_manifest.get("accepted")
        if not isinstance(accepted, list):
            raise VideoPipelineError("track manifest accepted must be an array")
        for raw_source in accepted:
            source = validate_source_entry(raw_source)
            variant = str(source.get("hud_variant") or "")
            if variant != hud_variant:
                raise VideoPipelineError(
                    f"{source['video_id']}: expected HUD {hud_variant}, got {variant!r}"
                )
            tracks = source.get("tracks", [])
            if not isinstance(tracks, list):
                raise VideoPipelineError(f"{source['video_id']}: tracks must be an array")
            confidences = [
                float(track["confidence"])
                for track in tracks
                if isinstance(track, Mapping)
                and type(track.get("confidence")) in (int, float)
            ]
            candidate = dict(source)
            source_group_id = _source_group_key(source)
            candidate["_selection"] = {
                "manifest_index": manifest_index,
                "hud_variant": hud_variant,
                "source_group_id": source_group_id,
                "track_count": len(tracks),
                "mean_track_confidence": (
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
                "replay_cache_path": str(source.get("replay_cache_path") or ""),
                "replay_cache_sha256": source.get("replay_cache_sha256"),
            }
            grouped.setdefault(source_group_id, []).append(candidate)
        for raw_rejection in raw_manifest.get("rejected", []):
            if isinstance(raw_rejection, Mapping):
                rejected.append(
                    {
                        "index": str(raw_rejection.get("index", manifest_index)),
                        "video_id": str(raw_rejection.get("video_id", "")),
                        "reason": str(raw_rejection.get("reason", "")),
                    }
                )

    selected: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for source_group_id, candidates in sorted(grouped.items()):
        candidate = max(
            candidates,
            key=lambda row: (
                int(row["_selection"]["track_count"]),
                float(row["_selection"]["mean_track_confidence"]),
                -int(row["_selection"]["manifest_index"]),
            ),
        )
        candidate_provenance = [
            dict(row["_selection"])
            for row in sorted(
                candidates,
                key=lambda row: (
                    int(row["_selection"]["manifest_index"]),
                    str(row.get("replay_cache_path") or ""),
                ),
            )
        ]
        selected_row = {key: value for key, value in candidate.items() if key != "_selection"}
        selected_row["extractor_root_candidates"] = candidate_provenance
        selected.append(selected_row)
        provenance.append(
            {
                "video_id": str(candidate["video_id"]),
                "source_group_id": source_group_id,
                "selected": dict(candidate["_selection"]),
                "candidates": candidate_provenance,
            }
        )
    return {
        "schema_version": VIDEO_PIPELINE_SCHEMA_VERSION,
        "kind": "simulator_video_source_manifest",
        "source_channel": SOURCE_CHANNEL_URL,
        "publication_cutoff_exclusive": PRE_EVOLUTION_CUTOFF.isoformat(),
        "hud_variant": hud_variant,
        "accepted": selected,
        "rejected": rejected,
        "candidate_provenance": provenance,
        "summary": {
            "source_count": len(grouped),
            "accepted_source_count": len(selected),
            "rejected_source_count": len(rejected),
            "track_count": sum(len(row.get("tracks", [])) for row in selected),
            "candidate_source_count": sum(len(rows) for rows in grouped.values()),
        },
    }


def _track_frame_positions(track: Mapping[str, Any]) -> dict[int, tuple[int, int]]:
    """Return valid frame-indexed positions for a detector track.

    HUD agreement is a quality diagnostic, not a second parser boundary.  A
    malformed sample is therefore ignored here and remains visible to the
    normal confidence/validation stages instead of making the entire source
    manifest unusable.  Duplicate samples at one frame are ignored after the
    first deterministic occurrence.
    """

    raw_samples = track.get("samples", [])
    if not isinstance(raw_samples, list):
        return {}
    positions: dict[int, tuple[int, int]] = {}
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, Mapping):
            continue
        frame_idx = raw_sample.get("frame_idx")
        x_mtile = raw_sample.get("x_mtile")
        y_mtile = raw_sample.get("y_mtile")
        if (
            type(frame_idx) is not int
            or type(x_mtile) is not int
            or type(y_mtile) is not int
        ):
            continue
        positions.setdefault(frame_idx, (x_mtile, y_mtile))
    return positions


def _track_identity(track: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return a stable identity used only to constrain cross-HUD matching."""

    return (
        str(track.get("card_id") or ""),
        str(track.get("owner") or ""),
        str(track.get("parent_track_id") or track.get("track_id") or ""),
    )


def _compare_hud_tracks(
    first_tracks: list[Mapping[str, Any]],
    second_tracks: list[Mapping[str, Any]],
    *,
    first_variant: str,
    second_variant: str,
    tolerance_mtile: int,
    minimum_overlap_samples: int,
) -> dict[str, Any]:
    """Compare two detector interpretations without double-counting evidence.

    Track IDs are normally stable between the two repository extractor runs,
    but they are not trusted as the only key.  We constrain matches by card,
    owner, and parent track, then greedily select the largest frame overlap and
    smallest position error.  The ordering is fully deterministic and works
    for duplicate swarm cards without relying on list order.
    """

    pair_candidates: list[tuple[int, float, int, str, str, int, int]] = []
    prepared_first = [
        (track, _track_frame_positions(track))
        for track in first_tracks
        if isinstance(track, Mapping)
    ]
    prepared_second = [
        (track, _track_frame_positions(track))
        for track in second_tracks
        if isinstance(track, Mapping)
    ]
    for first_index, (first_track, first_positions) in enumerate(prepared_first):
        if not first_positions:
            continue
        first_card, first_owner, first_parent = _track_identity(first_track)
        for second_index, (second_track, second_positions) in enumerate(prepared_second):
            if not second_positions or _track_identity(second_track)[:2] != (
                first_card,
                first_owner,
            ):
                continue
            # Parent IDs are a helpful tie-breaker, not a hard requirement:
            # some tracker versions re-number IDs after a HUD crop change.
            common_frames = sorted(set(first_positions) & set(second_positions))
            if len(common_frames) < minimum_overlap_samples:
                continue
            errors = [
                (
                    (first_positions[frame][0] - second_positions[frame][0]) ** 2
                    + (first_positions[frame][1] - second_positions[frame][1]) ** 2
                )
                ** 0.5
                for frame in common_frames
            ]
            mean_error = sum(errors) / len(errors)
            max_error = max(errors)
            second_parent = _track_identity(second_track)[2]
            parent_penalty = 0 if first_parent and first_parent == second_parent else 1
            pair_candidates.append(
                (
                    -len(common_frames),
                    mean_error,
                    parent_penalty,
                    str(first_track.get("track_id") or first_index),
                    str(second_track.get("track_id") or second_index),
                    first_index,
                    second_index,
                )
            )

    pair_candidates.sort()
    used_first: set[int] = set()
    used_second: set[int] = set()
    matches: list[dict[str, Any]] = []
    for (
        negative_overlap,
        mean_error,
        _parent_penalty,
        _first_sort_id,
        _second_sort_id,
        first_index,
        second_index,
    ) in pair_candidates:
        if first_index in used_first or second_index in used_second:
            continue
        first_track = prepared_first[first_index][0]
        second_track = prepared_second[second_index][0]
        first_positions = prepared_first[first_index][1]
        second_positions = prepared_second[second_index][1]
        common_frames = sorted(set(first_positions) & set(second_positions))
        errors = [
            (
                (first_positions[frame][0] - second_positions[frame][0]) ** 2
                + (first_positions[frame][1] - second_positions[frame][1]) ** 2
            )
            ** 0.5
            for frame in common_frames
        ]
        max_error = max(errors)
        used_first.add(first_index)
        used_second.add(second_index)
        matches.append(
            {
                "first_track_id": str(first_track.get("track_id") or first_index),
                "second_track_id": str(second_track.get("track_id") or second_index),
                "card_id": str(first_track.get("card_id") or ""),
                "owner": str(first_track.get("owner") or ""),
                "overlap_sample_count": len(common_frames),
                "position_mae_mtile": round(float(mean_error), 6),
                "position_max_error_mtile": round(float(max_error), 6),
                "agrees": bool(max_error <= tolerance_mtile),
            }
        )

    if not first_tracks or not second_tracks:
        status = "insufficient_tracks"
    elif not matches:
        status = "no_matching_tracks"
    elif len(matches) < min(len(first_tracks), len(second_tracks)):
        status = "partial"
    elif not all(match["agrees"] for match in matches):
        status = "disagree"
    else:
        status = "agree"

    all_errors = [match["position_mae_mtile"] for match in matches]
    all_max_errors = [match["position_max_error_mtile"] for match in matches]
    return {
        "status": status,
        "independent_evidence": False,
        "first_variant": first_variant,
        "second_variant": second_variant,
        "tolerance_mtile": tolerance_mtile,
        "minimum_overlap_samples": minimum_overlap_samples,
        "first_track_count": len(first_tracks),
        "second_track_count": len(second_tracks),
        "matched_track_count": len(matches),
        "overlap_sample_count": sum(match["overlap_sample_count"] for match in matches),
        "mean_position_mae_mtile": round(
            sum(all_errors) / len(all_errors), 6
        )
        if all_errors
        else None,
        "max_position_error_mtile": round(max(all_max_errors), 6)
        if all_max_errors
        else None,
        "matches": matches,
    }


def merge_hud_track_manifests(
    manifests: Iterable[Mapping[str, Any]],
    *,
    agreement_tolerance_mtile: int = HUD_AGREEMENT_TOLERANCE_MTILE,
    minimum_overlap_samples: int = HUD_AGREEMENT_MIN_OVERLAP_SAMPLES,
) -> dict[str, Any]:
    """Select one HUD interpretation per source group without double counting.

    Both HUD profiles are deliberately extracted, but they are not independent
    evidence: they decode the same frames.  This deterministic selector keeps
    the candidate with the most clean tracks, then the highest mean track
    confidence, and finally prefers the standard profile on an exact tie.
    A compact candidate summary is retained so a later audit can reproduce the
    choice without rerunning neural inference.
    """

    if type(agreement_tolerance_mtile) is not int or agreement_tolerance_mtile < 0:
        raise VideoPipelineError("agreement_tolerance_mtile must be non-negative")
    if type(minimum_overlap_samples) is not int or minimum_overlap_samples < 1:
        raise VideoPipelineError("minimum_overlap_samples must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    rejected: list[dict[str, str]] = []
    for manifest_index, raw_manifest in enumerate(manifests):
        if not isinstance(raw_manifest, Mapping):
            raise VideoPipelineError("HUD track manifest must be an object")
        accepted = raw_manifest.get("accepted")
        if not isinstance(accepted, list):
            raise VideoPipelineError("HUD track manifest accepted must be an array")
        for raw_source in accepted:
            source = validate_source_entry(raw_source)
            video_id = source["video_id"]
            source_group_id = _source_group_key(source)
            tracks = raw_source.get("tracks", [])
            if not isinstance(tracks, list):
                raise VideoPipelineError(f"{video_id}: tracks must be an array")
            variant = str(raw_source.get("hud_variant") or "")
            if variant not in HUD_VARIANTS:
                raise VideoPipelineError(f"{video_id}: HUD variant is missing")
            confidences = [
                float(track["confidence"])
                for track in tracks
                if isinstance(track, Mapping)
                and type(track.get("confidence")) in (int, float)
            ]
            candidate = dict(raw_source)
            candidate["_selection"] = {
                "hud_variant": variant,
                "source_group_id": source_group_id,
                "track_count": len(tracks),
                "mean_track_confidence": (
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
                "manifest_index": manifest_index,
            }
            grouped.setdefault(source_group_id, []).append(candidate)
        for raw_rejection in raw_manifest.get("rejected", []):
            if isinstance(raw_rejection, Mapping):
                rejected.append(
                    {
                        "index": str(raw_rejection.get("index", manifest_index)),
                        "video_id": str(raw_rejection.get("video_id", "")),
                        "reason": str(raw_rejection.get("reason", "")),
                    }
                )

    selected: list[dict[str, Any]] = []
    for source_group_id, candidates in sorted(grouped.items()):
        # A resumed extraction may provide multiple candidates for one HUD
        # profile.  Collapse those first; otherwise a duplicate root could
        # overwrite the profile used for the cross-HUD comparison.
        by_variant: dict[str, dict[str, Any]] = {}
        for variant in HUD_VARIANTS:
            variant_candidates = [
                row for row in candidates if row["_selection"]["hud_variant"] == variant
            ]
            if variant_candidates:
                by_variant[variant] = max(
                    variant_candidates,
                    key=lambda row: (
                        int(row["_selection"]["track_count"]),
                        float(row["_selection"]["mean_track_confidence"]),
                        -int(row["_selection"]["manifest_index"]),
                    ),
                )
        candidate = max(
            by_variant.values(),
            key=lambda row: (
                int(row["_selection"]["track_count"]),
                float(row["_selection"]["mean_track_confidence"]),
                1 if row["_selection"]["hud_variant"] == "standard" else 0,
                -int(row["_selection"]["manifest_index"]),
            ),
        )
        summaries = [
            dict(row["_selection"])
            for row in sorted(
                candidates,
                key=lambda row: row["_selection"]["hud_variant"],
            )
        ]
        selected_row = {
            key: value for key, value in candidate.items() if key != "_selection"
        }
        variant_rows = by_variant
        if len(variant_rows) >= 2 and all(
            variant in variant_rows for variant in HUD_VARIANTS
        ):
            agreement = _compare_hud_tracks(
                list(variant_rows["standard"].get("tracks", [])),
                list(variant_rows["alternative"].get("tracks", [])),
                first_variant="standard",
                second_variant="alternative",
                tolerance_mtile=agreement_tolerance_mtile,
                minimum_overlap_samples=minimum_overlap_samples,
            )
        else:
            agreement = {
                "status": "single_variant",
                "independent_evidence": False,
                "first_variant": str(candidate["_selection"]["hud_variant"]),
                "second_variant": None,
                "tolerance_mtile": agreement_tolerance_mtile,
                "minimum_overlap_samples": minimum_overlap_samples,
                "first_track_count": len(candidate.get("tracks", [])),
                "second_track_count": 0,
                "matched_track_count": 0,
                "overlap_sample_count": 0,
                "mean_position_mae_mtile": None,
                "max_position_error_mtile": None,
                "matches": [],
            }
        # Do not mutate the original manifest row.  Selected tracks carry only
        # their own cross-HUD diagnostic, while the source-level object keeps
        # the complete comparison for later audits.
        selected_tracks = [
            dict(track) for track in selected_row.get("tracks", [])
            if isinstance(track, Mapping)
        ]
        selected_variant = str(selected_row.get("hud_variant") or "")
        for selected_track in selected_tracks:
            selected_id = str(selected_track.get("track_id") or "")
            selected_card = str(selected_track.get("card_id") or "")
            selected_owner = str(selected_track.get("owner") or "")
            for match in agreement["matches"]:
                match_id = (
                    match["first_track_id"]
                    if selected_variant == agreement.get("first_variant")
                    else match["second_track_id"]
                )
                if match_id == selected_id and match["card_id"] == selected_card and match[
                    "owner"
                ] == selected_owner:
                    selected_track["hud_agreement"] = {
                        "status": "agree" if match["agrees"] else "disagree",
                        "independent_evidence": False,
                        "variants": [
                            agreement["first_variant"],
                            agreement["second_variant"],
                        ],
                        "matched_track_id": (
                            match["second_track_id"]
                            if selected_variant == agreement.get("first_variant")
                            else match["first_track_id"]
                        ),
                        "overlap_sample_count": match["overlap_sample_count"],
                        "position_mae_mtile": match["position_mae_mtile"],
                        "position_max_error_mtile": match[
                            "position_max_error_mtile"
                        ],
                        "tolerance_mtile": agreement["tolerance_mtile"],
                    }
                    break
        selected_row["tracks"] = selected_tracks
        selected_row["hud_candidates"] = summaries
        selected_row["hud_selection"] = "track_count_then_confidence_then_standard"
        selected_row["hud_agreement"] = agreement
        selected_row.setdefault("source_group_id", source_group_id)
        selected.append(selected_row)

    agreement_status_counts: dict[str, int] = {}
    for row in selected:
        status = str(row.get("hud_agreement", {}).get("status", "unknown"))
        agreement_status_counts[status] = agreement_status_counts.get(status, 0) + 1

    return {
        "schema_version": VIDEO_PIPELINE_SCHEMA_VERSION,
        "kind": "simulator_video_source_manifest",
        "source_channel": SOURCE_CHANNEL_URL,
        "publication_cutoff_exclusive": PRE_EVOLUTION_CUTOFF.isoformat(),
        "hud_variants": list(HUD_VARIANTS),
        "accepted": selected,
        "rejected": rejected,
        "summary": {
            "source_count": len(grouped),
            "accepted_source_count": len(selected),
            "rejected_source_count": len(rejected),
            "track_count": sum(len(row.get("tracks", [])) for row in selected),
            "selected_hud_counts": {
                variant: sum(row.get("hud_variant") == variant for row in selected)
                for variant in HUD_VARIANTS
            },
            "hud_agreement_status_counts": dict(sorted(agreement_status_counts.items())),
        },
    }


def video_truth_to_observation_manifest(
    truth_manifest: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any] | None = None,
    eligible_card_ids: Iterable[str] | None = None,
    corpus_id: str = "yersoncz-video-truth-v1",
    tick_us: int = 50_000,
    position_tolerance_mtile: int = 200,
    speed_estimator: str = "endpoint",
) -> dict[str, Any]:
    """Turn mined movement truth into the simulator corpus compiler schema.

    The conversion creates a mid-track initial state and only x/y/speed
    measurements.  It never invents a target, card play, or combat event from
    a detector track.  Missing media hashes, malformed timestamps, and
    duplicate/empty cases fail closed so the resulting corpus can enter a
    held-out gate only with explicit provenance.
    """

    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise VideoPipelineError("corpus_id must be non-empty")
    if type(tick_us) is not int or tick_us <= 0:
        raise VideoPipelineError("tick_us must be positive")
    if type(position_tolerance_mtile) is not int or position_tolerance_mtile < 0:
        raise VideoPipelineError("position_tolerance_mtile must be non-negative")
    if speed_estimator not in {"endpoint", "path_length", "median_step"}:
        raise VideoPipelineError(
            "speed_estimator must be one of endpoint, path_length, median_step"
        )
    raw_cases = truth_manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise VideoPipelineError("truth manifest cases must be a non-empty array")
    if eligible_card_ids is None:
        from .roster import load_opponent_roster

        eligible = frozenset(load_opponent_roster().eligible_cards)
    else:
        eligible = frozenset(str(card_id) for card_id in eligible_card_ids)
    source_hashes: dict[str, str] = {}
    if source_manifest is not None:
        accepted = source_manifest.get("accepted")
        if not isinstance(accepted, list):
            raise VideoPipelineError("source manifest accepted must be an array")
        for raw_source in accepted:
            source = validate_source_entry(raw_source)
            media_hash = source.get("media_sha256")
            if isinstance(media_hash, str):
                source_hashes[source["video_id"]] = _sha256(media_hash, "media_sha256")

    clips: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen_case_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise VideoPipelineError(f"truth case {index} must be an object")
        case_id = str(raw_case.get("case_id") or "").strip()
        video_id = str(raw_case.get("video_id") or "").strip()
        card_id = str(raw_case.get("card_id") or "").strip()
        track_id = str(raw_case.get("track_id") or "").strip()
        if not case_id or not video_id or not card_id or not track_id:
            raise VideoPipelineError(f"truth case {index} lacks a stable identity")
        if card_id not in eligible:
            rejected.append(
                {
                    "case_id": case_id,
                    "reason": f"card_outside_v1_roster:{card_id}",
                }
            )
            continue
        if case_id in seen_case_ids:
            raise VideoPipelineError(f"duplicate truth case ID: {case_id}")
        seen_case_ids.add(case_id)
        evidence = raw_case.get("evidence")
        if not isinstance(evidence, Mapping):
            raise VideoPipelineError(f"truth case {case_id} evidence must be an object")
        media_hash = evidence.get("media_sha256") or source_hashes.get(video_id)
        if not isinstance(media_hash, str):
            raise VideoPipelineError(f"truth case {case_id} lacks media_sha256")
        media_hash = _sha256(media_hash, f"{case_id}.media_sha256")
        raw_samples = raw_case.get("samples")
        if not isinstance(raw_samples, list) or len(raw_samples) < 2:
            raise VideoPipelineError(f"truth case {case_id} needs at least two samples")
        fps_value = raw_case.get("fps")
        fps = float(fps_value) if isinstance(fps_value, (int, float)) else 30.0
        if fps <= 0:
            raise VideoPipelineError(f"truth case {case_id} fps must be positive")
        prepared: list[tuple[float, Mapping[str, Any]]] = []
        for sample_index, raw_sample in enumerate(raw_samples):
            if not isinstance(raw_sample, Mapping):
                raise VideoPipelineError(f"truth case {case_id} sample {sample_index} is invalid")
            frame_idx = raw_sample.get("frame_idx")
            x = raw_sample.get("x_mtile")
            y = raw_sample.get("y_mtile")
            confidence = raw_sample.get("confidence")
            if (
                type(frame_idx) is not int
                or type(x) is not int
                or type(y) is not int
                or type(confidence) not in (int, float)
                or not 0 <= float(confidence) <= 1
            ):
                raise VideoPipelineError(f"truth case {case_id} has malformed sample {sample_index}")
            video_time = raw_sample.get("video_time_s")
            if video_time is None:
                video_time = float(frame_idx) / fps
            if type(video_time) not in (int, float) or float(video_time) < 0:
                raise VideoPipelineError(f"truth case {case_id} has invalid sample time")
            prepared.append((float(video_time), raw_sample))
        prepared.sort(key=lambda item: (item[0], int(item[1]["frame_idx"])))
        first_time = prepared[0][0]
        by_tick: dict[int, Mapping[str, Any]] = {}
        for video_time, raw_sample in prepared:
            relative_tick = max(0, round((video_time - first_time) * 1_000_000 / tick_us))
            prior = by_tick.get(relative_tick)
            if prior is None or float(raw_sample["confidence"]) > float(prior["confidence"]):
                by_tick[relative_tick] = raw_sample
        if len(by_tick) < 2:
            raise VideoPipelineError(f"truth case {case_id} collapses to one simulator tick")
        ordered = sorted(by_tick.items())
        first_sample = ordered[0][1]
        last_tick, last_sample = ordered[-1]
        if last_tick <= 0:
            raise VideoPipelineError(f"truth case {case_id} has no elapsed movement time")
        owner_raw = raw_case.get("owner", 0)
        if owner_raw in {"ally", "0", 0}:
            owner = 0
        elif owner_raw in {"enemy", "1", 1}:
            owner = 1
        else:
            raise VideoPipelineError(f"truth case {case_id} has invalid owner")
        interval_distances: list[float] = []
        interval_speeds: list[float] = []
        for (previous_tick, previous), (current_tick, current) in zip(
            ordered, ordered[1:]
        ):
            elapsed_ticks = current_tick - previous_tick
            if elapsed_ticks <= 0:
                raise VideoPipelineError(
                    f"truth case {case_id} has non-increasing simulator ticks"
                )
            distance = math.hypot(
                int(current["x_mtile"]) - int(previous["x_mtile"]),
                int(current["y_mtile"]) - int(previous["y_mtile"]),
            )
            interval_distances.append(distance)
            interval_speeds.append(distance * 1_000_000 / (elapsed_ticks * tick_us))
        if not interval_distances:
            raise VideoPipelineError(f"truth case {case_id} has no measurable intervals")
        if speed_estimator == "endpoint":
            dx = int(last_sample["x_mtile"]) - int(first_sample["x_mtile"])
            dy = int(last_sample["y_mtile"]) - int(first_sample["y_mtile"])
            distance = math.hypot(dx, dy)
            observed_speed = max(1, round(distance * 1_000_000 / (last_tick * tick_us)))
        elif speed_estimator == "path_length":
            path_length = sum(interval_distances)
            observed_speed = max(
                1,
                round(path_length * 1_000_000 / (last_tick * tick_us)),
            )
        else:
            observed_speed = max(1, round(statistics.median(interval_speeds)))
        include_position = bool(raw_case.get("include_position", False))
        samples = [
            {
                "tick": tick,
                "x_mtile": int(sample["x_mtile"]),
                "y_mtile": int(sample["y_mtile"]),
                "confidence": float(sample["confidence"]),
            }
            for tick, sample in ordered
        ]
        clip = {
            "clip_id": case_id,
            "group_id": str(raw_case.get("group_id") or video_id),
            "split": str(raw_case.get("split") or "calibration"),
            "media_hash": media_hash,
            "frame_start": int(raw_case.get("frame_start", raw_samples[0]["frame_idx"])),
            "frame_end": int(raw_case.get("frame_end", raw_samples[-1]["frame_idx"])),
            # The estimator is part of the evidence method so the downstream
            # validation corpus preserves it even though the compiler only
            # accepts the common observation schema.
            "method": f"video_truth_to_observation_manifest_v2:{speed_estimator}",
            "confidence": float(raw_case.get("confidence", min(item[1]["confidence"] for item in prepared))),
            "initial": {
                "tick": 0,
                "elapsed_us": 0,
                "phase": "regulation",
                "towers": [],
                "entities": [
                    {
                        "track_id": track_id,
                        "card_id": card_id,
                        "owner": owner,
                        "x_mtile": int(first_sample["x_mtile"]),
                        "y_mtile": int(first_sample["y_mtile"]),
                        "confidence": float(first_sample["confidence"]),
                    }
                ],
            },
            "tracks": [
                {
                    "track_id": track_id,
                    "mechanic": str(raw_case.get("mechanic") or f"{card_id}_isolated_movement"),
                    "confidence": float(raw_case.get("confidence", min(item[1]["confidence"] for item in prepared))),
                    "displacement_speed": {
                        "start_tick": 0,
                    "end_tick": last_tick,
                    "observed_mtile_per_s": observed_speed,
                    "tolerance_mtile_per_s": max(120, observed_speed // 10),
                    "compare_to_card_base_speed": False,
                    "speed_estimator": speed_estimator,
                },
                    "samples": samples if include_position else [],
                }
            ],
        }
        if not include_position:
            # An isolated track does not reveal its unseen target.  Comparing
            # absolute x/y against a target chosen by the simulator would
            # manufacture a pathfinding failure, so use only level-invariant
            # displacement speed for this evidence class.
            clip["tracks"][0]["displacement_speed"][
                "compare_to_card_base_speed"
            ] = True
        clips.append(clip)
    clips.sort(key=lambda row: row["clip_id"])
    return {
        "schema_version": 1,
        "corpus_id": corpus_id,
        "split_salt": str(truth_manifest.get("split_salt") or "yersoncz-video-truth-v1"),
        "speed_estimator": speed_estimator,
        "confidence_threshold": float(truth_manifest.get("confidence_threshold", 0.98)),
        "position_tolerance_mtile": position_tolerance_mtile,
        "hp_tolerance": 0,
        "clips": clips,
        "rejected": rejected,
    }


def retention_records(
    source_manifest: Mapping[str, Any],
    *,
    truth_manifest_path: str | Path,
    raw_root_relative: str = "outputs/simulator/fidelity_media/raw",
    truth_manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create storage records; raw deletion is allowed only after this step."""

    truth_path = str(truth_manifest_path)
    cases = source_manifest.get("accepted", [])
    if not isinstance(cases, list):
        raise VideoPipelineError("source manifest accepted must be an array")
    extracted_video_ids: set[str] | None = None
    if truth_manifest is not None:
        truth_cases = truth_manifest.get("cases")
        if not isinstance(truth_cases, list):
            raise VideoPipelineError("truth manifest cases must be an array")
        extracted_video_ids = {
            str(case["video_id"])
            for case in truth_cases
            if isinstance(case, Mapping) and case.get("video_id")
        }
    records: list[dict[str, Any]] = []
    for raw in cases:
        source = validate_source_entry(raw)
        if extracted_video_ids is not None and source["video_id"] not in extracted_video_ids:
            continue
        raw_path = raw.get("raw_path") or raw.get("download_path")
        if not isinstance(raw_path, str) or not raw_path:
            # No raw path means there is nothing safe to evict; retain a
            # provenance record rather than inventing one.
            continue
        relative = str(Path(raw_root_relative) / Path(raw_path).name)
        records.append(
            {
                "artifact_id": f"yersoncz:{source['video_id']}",
                "path": relative,
                "video_id": source["video_id"],
                "source_channel": SOURCE_CHANNEL_URL,
                "publication_date": source["upload_date"],
                "media_sha256": source.get("media_sha256"),
                "source_media_path": raw_path,
                "truth_manifest": truth_path,
                "truth_extracted": True,
                "truth_extracted_at": datetime.now(timezone.utc).isoformat(),
                "eviction_eligible": True,
                "retention_reason": "truth manifest sealed; raw video reproducible by source/hash",
            }
        )
    return records


def extractor_command(
    video_path: str | Path,
    replay_cache_path: str | Path,
    *,
    hud_variant: str,
    sample_interval_s: float = 0.1,
    yolo_detections: bool = True,
    video_start_time_s: float | None = None,
    video_duration_s: float | None = None,
) -> list[str]:
    """Build the repository vision-extractor command for one HUD profile.

    The command is returned, rather than executed, so a batch scheduler can
    run standard and alternative profiles in parallel and feed both caches to
    the confidence selector.  It intentionally uses the current project
    interpreter and the public ``cr-bot`` CLI boundary.
    """

    if hud_variant not in HUD_VARIANTS:
        raise VideoPipelineError(f"unsupported HUD variant: {hud_variant!r}")
    if sample_interval_s <= 0:
        raise VideoPipelineError("sample_interval_s must be positive")
    if video_start_time_s is not None and video_start_time_s < 0:
        raise VideoPipelineError("video_start_time_s must be non-negative")
    if video_duration_s is not None and video_duration_s <= 0:
        raise VideoPipelineError("video_duration_s must be positive")
    command = [
        sys.executable,
        "-m",
        "cr_bot.app.cli",
        "--video",
        str(video_path),
        "--video-sample-interval",
        str(sample_interval_s),
        "--write-replay-cache",
        str(replay_cache_path),
    ]
    if yolo_detections:
        command.append("--yolo-detections")
    if hud_variant == "alternative":
        command.append("--alternative-rois")
    if video_start_time_s is not None:
        command.extend(["--video-start-time", str(video_start_time_s)])
    if video_duration_s is not None:
        # The application CLI currently interprets ``--video-duration`` as an
        # absolute timestamp even though its help text says "from the start".
        # Emit an absolute end time here so a bounded job remains correct when
        # it seeks into a long recording.
        end_time_s = (video_start_time_s or 0.0) + video_duration_s
        command.extend(["--video-end-time", str(end_time_s)])
    return command


def build_extractor_jobs(
    source_manifest: Mapping[str, Any],
    *,
    output_root: str | Path = "outputs/simulator/fidelity_media/extractor",
    sample_interval_s: float = 0.1,
    yolo_detections: bool = True,
    video_start_time_s: float | None = None,
    video_duration_s: float | None = None,
) -> dict[str, Any]:
    """Create deterministic standard/alternative HUD jobs for each source.

    A job is only *ready* when the sealed source row contains a local video
    path.  Remote-only rows remain in the plan with ``missing_video_path``;
    this prevents a scheduler from accidentally downloading an unverified or
    post-cutoff file.  The same source video is always assigned both HUD
    profiles, allowing the downstream confidence selector to reject an
    ambiguous layout rather than silently choosing one.
    """

    if sample_interval_s <= 0:
        raise VideoPipelineError("sample_interval_s must be positive")
    accepted = source_manifest.get("accepted")
    if not isinstance(accepted, list):
        raise VideoPipelineError("source manifest must contain an accepted array")
    root = Path(output_root)
    jobs: list[dict[str, Any]] = []
    for raw in accepted:
        source = validate_source_entry(raw)
        video_id = source["video_id"]
        video_value = (
            raw.get("analysis_video_path")
            or raw.get("raw_path")
            or raw.get("download_path")
            or raw.get("video_path")
        )
        video_path = Path(str(video_value)) if video_value else None
        video_available = video_path is not None and video_path.is_file()
        # A source row may carry a detector-selected gameplay window.  Use it
        # when present, while retaining the CLI-wide window as a reproducible
        # fallback for manifests that do not have an audio/gameplay index.
        source_start = raw.get("analysis_start_time_s", video_start_time_s)
        source_duration = raw.get("analysis_duration_s", video_duration_s)
        if source_start is not None:
            if type(source_start) not in (int, float) or float(source_start) < 0:
                raise VideoPipelineError(
                    f"{video_id}: analysis_start_time_s must be non-negative"
                )
            source_start = float(source_start)
        if source_duration is not None:
            if type(source_duration) not in (int, float) or float(source_duration) <= 0:
                raise VideoPipelineError(
                    f"{video_id}: analysis_duration_s must be positive"
                )
            source_duration = float(source_duration)
        for hud_variant in HUD_VARIANTS:
            replay_path = root / video_id / hud_variant / "replay-cache.json"
            command = (
                extractor_command(
                    video_path,
                    replay_path,
                    hud_variant=hud_variant,
                    sample_interval_s=sample_interval_s,
                    yolo_detections=yolo_detections,
                    video_start_time_s=source_start,
                    video_duration_s=source_duration,
                )
                if video_available
                else None
            )
            jobs.append(
                {
                    "job_id": f"{video_id}:{hud_variant}",
                    "video_id": video_id,
                    "hud_variant": hud_variant,
                    "video_path": str(video_path) if video_path is not None else None,
                    "replay_cache_path": str(replay_path),
                    "command": command,
                    "status": (
                        "ready"
                        if video_available
                        else "missing_video_path"
                    ),
                    "source": {
                        "channel_url": SOURCE_CHANNEL_URL,
                        "upload_date": source["upload_date"],
                        "media_sha256": source.get("media_sha256"),
                        "analysis_start_time_s": source_start,
                        "analysis_duration_s": source_duration,
                    },
                }
            )
    jobs.sort(key=lambda row: row["job_id"])
    return {
        "schema_version": VIDEO_PIPELINE_SCHEMA_VERSION,
        "kind": "simulator_video_extractor_job_plan",
        "source_channel": SOURCE_CHANNEL_URL,
        "publication_cutoff_exclusive": PRE_EVOLUTION_CUTOFF.isoformat(),
        "hud_variants": list(HUD_VARIANTS),
        "sample_interval_s": sample_interval_s,
        "yolo_detections": bool(yolo_detections),
        "video_start_time_s": video_start_time_s,
        "video_duration_s": video_duration_s,
        "jobs": jobs,
        "ready_job_count": sum(row["status"] == "ready" for row in jobs),
        "missing_video_path_count": sum(
            row["status"] == "missing_video_path" for row in jobs
        ),
    }


def run_extractor_jobs(
    plan: Mapping[str, Any],
    *,
    execute: bool = False,
    skip_existing: bool = True,
    stop_on_error: bool = False,
    workspace_root: str | Path | None = None,
    retention_manifest_path: str | Path = "outputs/simulator/fidelity_media/retention.json",
    raw_media_root: str | Path = "outputs/simulator/fidelity_media/raw",
    reserve_bytes: int = 1_000_000_000,
    evict: bool = False,
    job_timeout_s: float | None = 1_800.0,
) -> dict[str, Any]:
    """Run a sealed extractor plan, or return a scheduler-safe dry run.

    The default is a dry run so a report/plan command cannot unexpectedly
    spend hours running neural inference.  When ``execute`` is true, each
    command is run with captured stdout/stderr and an explicit return code;
    no source metadata is changed and failed jobs remain auditable. Existing
    replay caches are treated as immutable completed jobs by default. This
    makes interrupted batches resumable without silently replacing evidence;
    pass ``skip_existing=False`` only for an intentional extractor rerun.
    """

    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise VideoPipelineError("reserve_bytes must be a non-negative integer")
    if job_timeout_s is not None and (
        type(job_timeout_s) not in (int, float) or job_timeout_s <= 0
    ):
        raise VideoPipelineError("job_timeout_s must be positive or None")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise VideoPipelineError("extractor plan must contain a jobs array")
    results: list[dict[str, Any]] = []
    repository_root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[1]
    )
    budget_kwargs = {
        "workspace_root": repository_root,
        "manifest_path": Path(retention_manifest_path),
        "raw_media_root": Path(raw_media_root),
        "evict": evict,
    }
    for raw_job in jobs:
        if not isinstance(raw_job, Mapping):
            raise VideoPipelineError("extractor job must be an object")
        result = dict(raw_job)
        command = raw_job.get("command")
        if raw_job.get("status") != "ready" or not isinstance(command, list):
            result.update({"executed": False, "returncode": None})
            results.append(result)
            continue
        replay_cache_value = raw_job.get("replay_cache_path")
        replay_cache = Path(str(replay_cache_value)) if replay_cache_value else None
        if skip_existing and replay_cache is not None and replay_cache.is_file():
            source = raw_job.get("source")
            source = source if isinstance(source, Mapping) else {}
            cache_check = _inspect_replay_cache_completeness(
                replay_cache,
                expected_start_s=(
                    float(source["analysis_start_time_s"])
                    if type(source.get("analysis_start_time_s")) in (int, float)
                    else None
                ),
                expected_duration_s=(
                    float(source["analysis_duration_s"])
                    if type(source.get("analysis_duration_s")) in (int, float)
                    else None
                ),
                sample_interval_s=(
                    float(plan.get("sample_interval_s"))
                    if type(plan.get("sample_interval_s")) in (int, float)
                    else None
                ),
            )
            if cache_check is not None:
                result["replay_cache_completeness"] = cache_check
            if cache_check is None or cache_check["complete"]:
                result.update(
                    {
                        "executed": False,
                        "returncode": 0,
                        "status": "already_complete",
                        "replay_cache_bytes": replay_cache.stat().st_size,
                        "replay_cache_sha256": _file_sha256(replay_cache),
                    }
                )
                results.append(result)
                continue
            # A recognized but incomplete cache is safe to replace: it is a
            # generated prefix, not a sealed truth artifact.  Keep its hash
            # and observed coverage in the run report for auditability, then
            # execute the normal command even when --rerun-existing was not
            # requested so interrupted batches repair themselves on resume.
            result.update(
                {
                    "status": "incomplete_existing_cache",
                    "previous_replay_cache_bytes": replay_cache.stat().st_size,
                    "previous_replay_cache_sha256": _file_sha256(replay_cache),
                }
            )
        if not execute:
            result.update({"executed": False, "returncode": None})
            results.append(result)
            continue
        from .storage import enforce_workspace_budget

        budget_before = enforce_workspace_budget(
            **budget_kwargs,
            reserve_bytes=reserve_bytes,
        )
        result["budget_before"] = budget_before
        if not budget_before["passed"]:
            result.update(
                {
                    "executed": False,
                    "returncode": None,
                    "status": "blocked_workspace_budget",
                    "error": "workspace cap/reserve would be exceeded before extraction",
                }
            )
            results.append(result)
            if stop_on_error:
                break
            continue
        environment = os.environ.copy()
        src_path = str(repository_root / "src")
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            src_path
            if not existing_pythonpath
            else os.pathsep.join((src_path, existing_pythonpath))
        )
        result["job_timeout_s"] = job_timeout_s
        try:
            completed = subprocess.run(
                [str(part) for part in command],
                check=False,
                text=True,
                capture_output=True,
                cwd=repository_root,
                env=environment,
                timeout=job_timeout_s,
            )
        except subprocess.TimeoutExpired as error:
            result.update(
                {
                    "executed": True,
                    "returncode": None,
                    "stdout": str(error.stdout or "")[-4_000:],
                    "stderr": str(error.stderr or "")[-4_000:],
                    "status": "timeout",
                    "error": f"extractor job exceeded timeout ({job_timeout_s}s)",
                }
            )
            budget_after = enforce_workspace_budget(
                **budget_kwargs,
                reserve_bytes=0,
            )
            result["budget_after"] = budget_after
            if not budget_after["passed"]:
                result["status"] = "failed_workspace_budget"
                result["error"] = "workspace cap exceeded after timed-out extraction"
            results.append(result)
            if stop_on_error:
                break
            continue
        result.update(
            {
                "executed": True,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4_000:],
                "stderr": completed.stderr[-4_000:],
                "status": "completed" if completed.returncode == 0 else "failed",
            }
        )
        budget_after = enforce_workspace_budget(
            **budget_kwargs,
            reserve_bytes=0,
        )
        result["budget_after"] = budget_after
        if not budget_after["passed"]:
            result["status"] = "failed_workspace_budget"
            result["error"] = "workspace cap exceeded after extraction"
        results.append(result)
        if stop_on_error and (
            completed.returncode != 0 or not budget_after["passed"]
        ):
            break
    return {
        "schema_version": VIDEO_PIPELINE_SCHEMA_VERSION,
        "kind": "simulator_video_extractor_run",
        "source_channel": SOURCE_CHANNEL_URL,
        "execute": bool(execute),
        "reserve_bytes": reserve_bytes,
        "workspace_root": str(repository_root),
        "jobs": results,
        "completed_count": sum(
            row.get("status") in {"completed", "already_complete"}
            for row in results
        ),
        "failed_count": sum(
            row.get("status")
            in {"failed", "timeout", "failed_workspace_budget", "blocked_workspace_budget"}
            for row in results
        ),
        "skipped_count": sum(not row.get("executed") for row in results),
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


__all__ = [
    "HUD_AGREEMENT_MIN_OVERLAP_SAMPLES",
    "HUD_AGREEMENT_TOLERANCE_MTILE",
    "HUD_PROFILES",
    "HUD_VARIANTS",
    "ACTION_WINDOW_SCHEMA_VERSION",
    "DEFAULT_ACTION_WINDOW_AFTER_S",
    "DEFAULT_ACTION_WINDOW_BEFORE_S",
    "PRE_EVOLUTION_CUTOFF",
    "SOURCE_CHANNEL_URL",
    "VideoPipelineError",
    "assign_video_split",
    "batch_replay_cache_track_manifest",
    "merge_track_manifests",
    "build_action_window_extractor_jobs",
    "build_action_window_manifest",
    "build_extractor_jobs",
    "discover_source_manifest",
    "filter_source_manifest",
    "extractor_command",
    "mine_clean_tracks",
    "merge_hud_track_manifests",
    "replay_cache_track_manifest",
    "retention_records",
    "run_extractor_jobs",
    "select_hud_variant",
    "validate_source_entry",
    "video_truth_to_observation_manifest",
    "write_json",
]
