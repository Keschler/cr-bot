"""Extract, normalize, and seal physical-lab battle evidence.

The connected runner deliberately stops at a sealed candidate run.  This
module is the next boundary: it invokes the existing ``cr_bot`` vision
extractor, turns the recognized replay cache into a compact event/timeline
record, and writes an observation manifest that can be compared with the
deterministic simulator.

Two clocks are kept separate throughout this module.  Video timestamps and
workstation monotonic receipts are provenance; the clock drawn by the game is
retained as a diagnostic field only and never supplies ``match_time_us``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import os
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from ..ruleset import load_ruleset
from .artifacts import hash_file, seal_json
from .cache import seal_replay_cache
from .comparison import ComparisonReport, compare_observation_to_replay
from .identity import KnownCardIdentity, KnownPlacement
from .observation import ingest_for_experiment
from .replay import SimulatorReplay, action_match_time_us, run_simulator_replay
from .schema import ExperimentSpec, PhysicalLabError, canonical_hash


EXTRACTION_SCHEMA_VERSION = 1
DEFAULT_SAMPLE_INTERVAL_S = 0.1
DEFAULT_CONFIDENCE_THRESHOLD = 0.0
DEFAULT_JOB_TIMEOUT_S = 1_800.0
# The current cache sampling interval is about 100 ms.  Three consecutive
# misses therefore provide a 300 ms confirmation boundary, while a 500 ms
# continuity window absorbs a detector gap without merging a later deployment.
LIFECYCLE_CONFIRMATION_FRAMES = 3
LIFECYCLE_REAPPEARANCE_GAP_US = 500_000
LIFECYCLE_REAPPEARANCE_DISTANCE_MTILE = 5_000
CROSS_PHONE_CORROBORATION_WINDOW_US = 250_000


class PhysicalExtractionError(PhysicalLabError):
    """Raised when extractor output cannot be safely interpreted."""


@dataclass(frozen=True, slots=True)
class ExtractorJobResult:
    side: str
    command: tuple[str, ...]
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    replay_cache_path: str
    replay_cache_sha256: str | None
    replay_cache_recognized: bool
    source_media_path: str
    source_media_sha256: str | None
    screen_width_px: int | None
    screen_height_px: int | None
    frame_normalization: str
    hud_profile: str

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "command": list(self.command),
            "status": self.status,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "replay_cache_path": self.replay_cache_path,
            "replay_cache_sha256": self.replay_cache_sha256,
            "replay_cache_recognized": self.replay_cache_recognized,
            "source_media_path": self.source_media_path,
            "source_media_sha256": self.source_media_sha256,
            "screen_width_px": self.screen_width_px,
            "screen_height_px": self.screen_height_px,
            "frame_normalization": self.frame_normalization,
            "hud_profile": self.hud_profile,
        }


@dataclass(slots=True)
class _LifecycleTrack:
    canonical_key: tuple[str, str, str]
    owner: str
    card_id: str
    logical_id: str
    last_seen_video_us: int
    last_x_mtile: int
    last_y_mtile: int
    raw_track_ids: set[str]
    missing_frames: int = 0
    missing_since_video_us: int | None = None
    disappearance_event: dict[str, object] | None = None
    confirmed_dead: bool = False


def _json_scalar(value: object) -> object:
    """Convert NumPy/dataclass scalar values without importing NumPy."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_scalar(item())
        except (TypeError, ValueError):
            return None
    return str(value)


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return _json_scalar(value)


def _opposite(side: str) -> str:
    if side == "A":
        return "B"
    if side == "B":
        return "A"
    raise PhysicalExtractionError(f"unsupported physical side: {side!r}")


def _owner_for_team(viewer_side: str, team: object) -> str | None:
    if team == "ally":
        return viewer_side
    if team == "enemy":
        return _opposite(viewer_side)
    return None


def _video_us(video_time_s: object) -> int:
    if type(video_time_s) not in (int, float) or not math.isfinite(float(video_time_s)):
        raise PhysicalExtractionError(f"invalid replay video timestamp: {video_time_s!r}")
    if float(video_time_s) < 0:
        raise PhysicalExtractionError("replay video timestamp cannot be negative")
    return int(round(float(video_time_s) * 1_000_000))


def _internal_match_us(
    video_time_s: float,
    *,
    battle_start_video_s: float | None,
    capture_start_monotonic_us: int | None,
    battle_start_monotonic_us: int | None,
    stream_offset_us: int = 0,
) -> int | None:
    """Map a cache timestamp onto the runner's internal match-time axis.

    The video timestamp is a transport coordinate, not the game's clock.  A
    connected run supplies the monotonic timestamp at which its capture and
    reviewed battle barrier started.  That mapping is therefore stable even
    when the HUD clock is localized, occluded, or rounded between frames.  The
    extractor boundary remains a deterministic fallback for older manifests
    that predate the monotonic provenance fields.
    """

    if type(stream_offset_us) is not int:
        raise PhysicalExtractionError("stream synchronization offset must be an integer")
    if capture_start_monotonic_us is not None and battle_start_monotonic_us is not None:
        return max(
            0,
            int(round(capture_start_monotonic_us + video_time_s * 1_000_000))
            - battle_start_monotonic_us
            - stream_offset_us,
        )
    if battle_start_video_s is None:
        return None
    return max(
        0,
        int(round((video_time_s - battle_start_video_s) * 1_000_000)) - stream_offset_us,
    )


def _stream_alignment_offset_us(synchronization: object, side: str) -> int:
    if not isinstance(synchronization, Mapping):
        return 0
    alignments = synchronization.get("alignments")
    if not isinstance(alignments, list):
        return 0
    for alignment in alignments:
        if not isinstance(alignment, Mapping) or alignment.get("device_id") != side:
            continue
        value = alignment.get("offset_us")
        if type(value) is int:
            return value
    return 0


def _nearest_frame_index(rows: list[Mapping[str, Any]], video_time_s: float | None) -> int:
    if not rows:
        return 0
    if video_time_s is None:
        return int(rows[0]["frame_index"])
    return int(min(rows, key=lambda row: abs(float(row["video_time_s"]) - video_time_s))["frame_index"])


def _canonical_position_mtile(
    viewer_side: str,
    x_mtile: int,
    y_mtile: int,
) -> tuple[int, int]:
    """Convert a viewer-relative arena point to player-A-bottom coordinates."""

    if viewer_side == "A":
        return x_mtile, y_mtile
    if viewer_side == "B":
        return 18_000 - x_mtile, 32_000 - y_mtile
    raise PhysicalExtractionError(f"unsupported physical side: {viewer_side!r}")


def _troop_row(match: object, *, viewer_side: str, arena_px: object) -> dict[str, object] | None:
    troop = getattr(match, "troop", None)
    if troop is None:
        return None
    class_name = getattr(troop, "class_name", None)
    team = getattr(troop, "team", None)
    owner = _owner_for_team(viewer_side, team)
    if not isinstance(class_name, str) or owner is None:
        return None
    x_px = _json_scalar(getattr(troop, "center_x", None))
    y_px = _json_scalar(getattr(troop, "center_y", None))
    if type(x_px) not in (int, float) or type(y_px) not in (int, float):
        return None
    try:
        from cr_bot.features.action_space import ACTION_GRID
        from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

        card_id = DIRECT_UNIT_TO_CARD.get(class_name)
        cell = ACTION_GRID.pixel_to_cell(float(x_px), float(y_px), tuple(arena_px))
        x_mtile: int | None = None
        y_mtile: int | None = None
        if cell is not None:
            # The simulator uses 1,000 milli-tiles per policy-grid cell.  The
            # continuous conversion retains sub-cell motion where possible.
            ax, ay, aw, ah = (float(item) for item in arena_px)
            x_norm = (float(x_px) - ax) / aw
            y_norm = (float(y_px) - ay) / ah
            x0, y0, x1, y1 = (
                ACTION_GRID.x0,
                ACTION_GRID.y0,
                ACTION_GRID.x1,
                ACTION_GRID.y1,
            )
            x_mtile = int(round((x_norm - x0) / (x1 - x0) * 18_000))
            y_mtile = int(round((y_norm - y0) / (y1 - y0) * 32_000))
            # Both phones render their own towers at the bottom.  Simulator
            # coordinates keep player A at the bottom, so Phone B's arena is
            # canonicalized with a 180-degree rotation.
            x_mtile, y_mtile = _canonical_position_mtile(
                viewer_side,
                x_mtile,
                y_mtile,
            )
        if x_mtile is None or y_mtile is None:
            return None
    except (ImportError, TypeError, ValueError, ZeroDivisionError):
        return None
    return {
        "track_id": _json_scalar(getattr(troop, "track_id", None)),
        "class_name": class_name,
        "card_id": card_id,
        "raw_card_id": card_id,
        "team": team,
        "owner": owner,
        "confidence": float(_json_scalar(getattr(troop, "confidence", 0.0)) or 0.0),
        "center_px": [float(x_px), float(y_px)],
        "x_mtile": x_mtile,
        "y_mtile": y_mtile,
        "estimated_hp": _json_scalar(getattr(troop, "estimated_hp", None)),
    }


def _analysis_row(
    cache_row: object,
    *,
    viewer_side: str,
    battle_start_video_s: float | None,
    capture_start_monotonic_us: int | None,
    battle_start_monotonic_us: int | None,
    stream_offset_us: int = 0,
    identity_context: KnownCardIdentity | None = None,
) -> dict[str, object]:
    analysis = getattr(cache_row, "analysis", None)
    video_time_s = float(getattr(cache_row, "video_time_s"))
    match_time_us = _internal_match_us(
        video_time_s,
        battle_start_video_s=battle_start_video_s,
        capture_start_monotonic_us=capture_start_monotonic_us,
        battle_start_monotonic_us=battle_start_monotonic_us,
        stream_offset_us=stream_offset_us,
    )
    arena_px = tuple(getattr(analysis, "arena_px", (0, 0, 1, 1)))
    units: list[dict[str, object]] = []
    identity_rejections: list[dict[str, object]] = []
    for match in list(getattr(analysis, "matches", []) or []):
        unit = _troop_row(match, viewer_side=viewer_side, arena_px=arena_px)
        if unit is None:
            continue
        if identity_context is None:
            # Preserve the historical behavior for callers that use this
            # helper without a physical run manifest: unmapped detector
            # classes cannot become simulator cards.
            if unit.get("card_id") is None:
                continue
            units.append(unit)
            continue

        decision = identity_context.resolve(
            owner=str(unit.get("owner") or ""),
            raw_card_id=(
                str(unit["raw_card_id"])
                if unit.get("raw_card_id") is not None
                else None
            ),
            match_time_us=match_time_us,
        )
        if not decision.accepted:
            identity_rejections.append(
                {
                    "frame_index": int(getattr(cache_row, "frame_idx")),
                    "video_time_us": _video_us(video_time_s),
                    "match_time_us": match_time_us,
                    "owner": unit.get("owner"),
                    "raw_class_name": unit.get("class_name"),
                    "raw_card_id": decision.raw_card_id,
                    "confidence": unit.get("confidence"),
                    "reason": decision.reason,
                }
            )
            continue
        unit["card_id"] = decision.card_id
        unit["identity_source"] = decision.source
        unit["identity_reason"] = decision.reason
        if decision.matched_action_id is not None:
            unit["identity_action_id"] = decision.matched_action_id
        units.append(unit)
    hand_state = _json_value(getattr(analysis, "hand_state", {}))
    towers_hp = _json_value(getattr(analysis, "towers_hp", {}))
    elixir = _json_value(getattr(analysis, "elixir", {}))
    result: dict[str, object] = {
        "frame_index": int(getattr(cache_row, "frame_idx")),
        "video_time_us": _video_us(video_time_s),
        "match_time_us": match_time_us,
        "in_game_clock_text": getattr(analysis, "time", None),
        "in_game_time_left_s": _json_scalar(getattr(analysis, "time_left_s", None)),
        "in_game_total_remaining_s": _json_scalar(getattr(analysis, "total_remaining_s", None)),
        "overtime_diagnostic": bool(getattr(analysis, "overtime", False)),
        "hand_state": hand_state,
        "elixir": elixir,
        "towers_hp": towers_hp,
        "arena_px": [_json_scalar(item) for item in arena_px],
        "units": units,
    }
    if identity_rejections:
        result["identity_rejections"] = identity_rejections
    return result


def _event_row(
    *,
    event_id: str,
    kind: str,
    owner: str,
    card_id: str | None,
    video_time_s: float | None,
    match_time_s: float | None,
    frame_index: int,
    confidence: float,
    certainty: str,
    source_capture_side: str | None = None,
    cell: object = None,
    values: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "event_id": event_id,
        "kind": kind,
        "video_time_us": _video_us(0.0 if video_time_s is None else video_time_s),
        "match_time_us": (
            None
            if match_time_s is None
            else max(0, int(round(match_time_s * 1_000_000)))
        ),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "certainty": certainty,
        "source_frame_indices": [frame_index],
        "evidence_refs": [f"capture:{source_capture_side or owner}:{frame_index}"],
        "owner": owner,
        "values": dict(values or {}),
        "uncertainty_us": 100_000,
    }
    if card_id is not None:
        result["card_id"] = card_id
    if isinstance(cell, (tuple, list)) and len(cell) == 2:
        result["values"] = {
            **dict(result["values"]),
            "cell_col": int(cell[0]),
            "cell_row": int(cell[1]),
        }
    return result


def _event_time_us(event: Mapping[str, object]) -> int:
    match_time = event.get("match_time_us")
    if type(match_time) is int:
        return match_time
    video_time = event.get("video_time_us")
    return int(video_time) if type(video_time) is int else 0


def _event_capture_side(event: Mapping[str, object]) -> str | None:
    refs = event.get("evidence_refs")
    if not isinstance(refs, list):
        return None
    for ref in refs:
        if isinstance(ref, str) and ref.startswith("capture:"):
            parts = ref.split(":")
            if len(parts) >= 2 and parts[1] in {"A", "B"}:
                return parts[1]
    return None


def _identity_context_from_run(
    run: Mapping[str, object],
    spec: ExperimentSpec,
) -> KnownCardIdentity:
    """Build the known-card context without trusting requested actions alone."""

    placements: list[KnownPlacement] = []
    actions = run.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, Mapping) or action.get("accepted") is not True:
                continue
            side = str(action.get("side") or "").upper()
            card_id = str(action.get("card_id") or "").lower()
            match_time_us = action_match_time_us(run, action)
            if (
                side not in {"A", "B"}
                or not card_id
                or type(match_time_us) is not int
            ):
                continue
            raw_cell = action.get("arena_cell")
            arena_cell: tuple[int, int] | None = None
            if (
                isinstance(raw_cell, (list, tuple))
                and len(raw_cell) == 2
                and type(raw_cell[0]) is int
                and type(raw_cell[1]) is int
            ):
                arena_cell = (int(raw_cell[0]), int(raw_cell[1]))
            placements.append(
                KnownPlacement(
                    action_id=str(action.get("action_id") or "unknown-action"),
                    owner=side,
                    card_id=card_id,
                    match_time_us=match_time_us,
                    arena_cell=arena_cell,
                )
            )
    return KnownCardIdentity(
        decks=spec.initial_conditions.decks,
        placements=placements,
        placements_authoritative=True,
    )


def _cross_phone_event_key(event: Mapping[str, object]) -> tuple[object, ...] | None:
    kind = event.get("kind")
    if kind not in {
        "unit_spawn_observed",
        "unit_disappearance_observed",
        "tower_damage_observed",
    }:
        return None
    values = event.get("values")
    tower = values.get("tower") if isinstance(values, Mapping) else None
    return (kind, event.get("card_id"), event.get("owner"), tower)


def _merge_cross_phone_event(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> dict[str, object]:
    first_time = _event_time_us(first)
    second_time = _event_time_us(second)
    merged = dict(first)
    merged["event_id"] = f"cross-phone-{first.get('event_id')}-{second.get('event_id')}"
    merged["video_time_us"] = int(round((int(first.get("video_time_us", first_time)) + int(second.get("video_time_us", second_time))) / 2))
    if type(first.get("match_time_us")) is int and type(second.get("match_time_us")) is int:
        merged["match_time_us"] = int(round((first_time + second_time) / 2))
    elif type(first.get("match_time_us")) is int:
        merged["match_time_us"] = first_time
    elif type(second.get("match_time_us")) is int:
        merged["match_time_us"] = second_time
    merged["confidence"] = max(float(first.get("confidence", 0.0)), float(second.get("confidence", 0.0)))
    certainty_priority = {"tentative": 0, "inferred": 1, "direct": 2}
    first_certainty = str(first.get("certainty", "inferred"))
    second_certainty = str(second.get("certainty", "inferred"))
    certainty = max(
        (first_certainty, second_certainty),
        key=lambda value: certainty_priority.get(value, 0),
    )
    if certainty == "tentative":
        # Independent tentative observations are sufficient to confirm the
        # transition without trusting either phone's one-frame detector edge.
        certainty = "inferred"
    merged["certainty"] = certainty
    merged["source_frame_indices"] = sorted(
        {
            int(frame)
            for source in (first, second)
            for frame in source.get("source_frame_indices", [])
            if type(frame) is int
        }
    )
    merged["evidence_refs"] = sorted(
        {
            str(ref)
            for source in (first, second)
            for ref in source.get("evidence_refs", [])
            if isinstance(ref, str)
        }
    )
    values: dict[str, object] = {}
    for source in (first, second):
        source_values = source.get("values")
        if isinstance(source_values, Mapping):
            values.update({str(key): value for key, value in source_values.items()})
    first_side = _event_capture_side(first)
    second_side = _event_capture_side(second)
    sides = sorted({side for side in (first_side, second_side) if side is not None})
    values.update(
        {
            "cross_phone_corroborated": True,
            "cross_phone_sides": ",".join(sides),
            "corroborating_event_id": str(second.get("event_id")),
            "cross_phone_time_delta_us": abs(first_time - second_time),
        }
    )
    if first_certainty == "tentative" or second_certainty == "tentative":
        values["confirmation_state"] = "cross_phone_confirmed"
    merged["values"] = values
    return merged


def _merge_cross_phone_events(
    events_by_side: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    window_us: int = CROSS_PHONE_CORROBORATION_WINDOW_US,
) -> list[dict[str, object]]:
    """Deduplicate overlapping A/B visual events while retaining provenance."""

    events_a = [dict(event) for event in events_by_side.get("A", ())]
    events_b = [dict(event) for event in events_by_side.get("B", ())]
    used_b: set[int] = set()
    merged: list[dict[str, object]] = []
    for event_a in events_a:
        key = _cross_phone_event_key(event_a)
        if key is None:
            merged.append(event_a)
            continue
        candidates = [
            (index, event_b)
            for index, event_b in enumerate(events_b)
            if index not in used_b
            and _cross_phone_event_key(event_b) == key
            and abs(_event_time_us(event_a) - _event_time_us(event_b)) <= window_us
        ]
        if not candidates:
            merged.append(event_a)
            continue
        index, event_b = min(
            candidates,
            key=lambda item: abs(_event_time_us(event_a) - _event_time_us(item[1])),
        )
        used_b.add(index)
        merged.append(_merge_cross_phone_event(event_a, event_b))
    merged.extend(event for index, event in enumerate(events_b) if index not in used_b)
    return sorted(
        merged,
        key=lambda event: (_event_time_us(event), str(event.get("event_id", ""))),
    )


def _apply_event_identity(
    event: Mapping[str, object],
    *,
    identity_context: KnownCardIdentity | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Apply the known-card gate to a tracker-produced card event."""

    result = dict(event)
    if identity_context is None or result.get("card_id") is None or result.get("owner") is None:
        return result, None
    raw_card_id = str(result["card_id"])
    decision = identity_context.resolve(
        owner=str(result["owner"]),
        raw_card_id=raw_card_id,
        match_time_us=(
            int(result["match_time_us"])
            if type(result.get("match_time_us")) is int
            else None
        ),
    )
    if not decision.accepted:
        return None, {
            "record_type": "event",
            "event_id": result.get("event_id"),
            "kind": result.get("kind"),
            "owner": result.get("owner"),
            "raw_card_id": raw_card_id,
            "confidence": result.get("confidence"),
            "reason": decision.reason,
        }
    result["card_id"] = decision.card_id
    values = result.get("values")
    merged_values = dict(values) if isinstance(values, Mapping) else {}
    merged_values.update(
        {
            "raw_card_id": decision.raw_card_id,
            "identity_source": decision.source,
            "identity_reason": decision.reason,
        }
    )
    if decision.matched_action_id is not None:
        merged_values["identity_action_id"] = decision.matched_action_id
    result["values"] = merged_values
    return result, None


def _unit_identity_values(unit: Mapping[str, object]) -> dict[str, object]:
    values: dict[str, object] = {}
    for key in (
        "raw_class_name",
        "raw_card_id",
        "identity_source",
        "identity_reason",
        "identity_action_id",
    ):
        if key in unit:
            values[key] = unit[key]
    return values


def _entity_source_side(entity: Mapping[str, object]) -> str | None:
    samples = entity.get("samples")
    if not isinstance(samples, list):
        return None
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        source = sample.get("source_capture_id")
        if source == "capture-A":
            return "A"
        if source == "capture-B":
            return "B"
    return None


def _select_primary_view_entities(
    stream_payloads: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Use each owner's own-phone trajectory while retaining raw streams.

    The second phone remains lifecycle corroboration.  Counting both views as
    separate physical entities made one troop compete twice against the same
    simulator entity and amplified viewpoint calibration noise.
    """

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for payload in stream_payloads.values():
        rows = payload.get("entities")
        if not isinstance(rows, list):
            continue
        for entity in rows:
            if not isinstance(entity, Mapping):
                continue
            owner = str(entity.get("owner") or "")
            card_id = str(entity.get("card_id") or "")
            if owner in {"A", "B"} and card_id:
                grouped[(owner, card_id)].append(dict(entity))

    selected: list[dict[str, object]] = []
    for (owner, _card_id), entities in sorted(grouped.items()):
        owner_view = [entity for entity in entities if _entity_source_side(entity) == owner]
        selected.extend(owner_view or entities)
    return selected


def _sample_time_us(sample: Mapping[str, object]) -> int:
    match_time_us = sample.get("match_time_us")
    if type(match_time_us) is int:
        return match_time_us
    video_time_us = sample.get("video_time_us")
    return int(video_time_us) if type(video_time_us) is int else 0


def _merge_singleton_action_entities(
    entities: Sequence[Mapping[str, object]],
    *,
    identity_context: KnownCardIdentity,
    singleton_card_ids: frozenset[str],
) -> list[dict[str, object]]:
    """Stitch detector fragments for accepted one-entity deployments."""

    remaining = [dict(entity) for entity in entities]
    fused: list[dict[str, object]] = []
    placements = sorted(identity_context.placements, key=lambda row: row.match_time_us)
    for placement in placements:
        if placement.card_id not in singleton_card_ids:
            continue
        next_time = min(
            (
                other.match_time_us
                for other in placements
                if other.owner.upper() == placement.owner.upper()
                and other.card_id == placement.card_id
                and other.match_time_us > placement.match_time_us
            ),
            default=None,
        )
        selected_samples: list[dict[str, object]] = []
        next_remaining: list[dict[str, object]] = []
        for entity in remaining:
            if (
                str(entity.get("owner") or "").upper() != placement.owner.upper()
                or str(entity.get("card_id") or "").lower() != placement.card_id
            ):
                next_remaining.append(entity)
                continue
            samples = entity.get("samples")
            if not isinstance(samples, list):
                next_remaining.append(entity)
                continue
            inside: list[dict[str, object]] = []
            outside: list[dict[str, object]] = []
            for sample in samples:
                if not isinstance(sample, Mapping):
                    continue
                time_us = _sample_time_us(sample)
                in_window = time_us >= placement.match_time_us - 250_000 and (
                    next_time is None or time_us < next_time - 250_000
                )
                (inside if in_window else outside).append(dict(sample))
            selected_samples.extend(inside)
            if outside:
                remainder = dict(entity)
                remainder["samples"] = outside
                next_remaining.append(remainder)
        remaining = next_remaining
        if not selected_samples:
            continue

        # Multiple detector tracks can contain the same physical box in one
        # frame.  Keep the strongest row for that source frame.
        by_frame: dict[tuple[str, int], dict[str, object]] = {}
        for sample in selected_samples:
            key = (
                str(sample.get("source_capture_id") or "unknown"),
                int(sample.get("frame_index") or 0),
            )
            previous = by_frame.get(key)
            if previous is None or float(sample.get("confidence") or 0.0) > float(
                previous.get("confidence") or 0.0
            ):
                by_frame[key] = sample
        samples = sorted(
            by_frame.values(),
            key=lambda sample: (
                int(sample.get("video_time_us") or 0),
                int(sample.get("frame_index") or 0),
            ),
        )
        fused.append(
            {
                "stable_observation_id": f"action-{placement.action_id}-entity-000",
                "card_id": placement.card_id,
                "owner": placement.owner.upper(),
                "confidence": min(float(sample.get("confidence") or 0.0) for sample in samples),
                "source_card_id": placement.card_id,
                "samples": samples,
                "track_id_diagnostic": f"accepted-action:{placement.action_id}",
            }
        )
    return sorted([*remaining, *fused], key=lambda row: str(row.get("stable_observation_id", "")))


def _reconcile_singleton_lifecycle_events(
    events: Sequence[Mapping[str, object]],
    *,
    identity_context: KnownCardIdentity,
    singleton_card_ids: frozenset[str],
) -> list[dict[str, object]]:
    """Keep one spawn and the final disappearance per singleton action."""

    result = [dict(event) for event in events]
    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"spawn": [], "death": []}
    )
    for index, event in enumerate(result):
        kind = event.get("kind")
        family = (
            "spawn"
            if kind == "unit_spawn_observed"
            else "death"
            if kind == "unit_disappearance_observed"
            else None
        )
        card_id = str(event.get("card_id") or "").lower()
        owner = str(event.get("owner") or "").upper()
        if family is None or card_id not in singleton_card_ids or owner not in {"A", "B"}:
            continue
        lineage = identity_context.placement_lineage(
            owner=owner,
            card_id=card_id,
            match_time_us=_event_time_us(event),
        )
        if lineage is None:
            continue
        values = event.get("values")
        merged_values = dict(values) if isinstance(values, Mapping) else {}
        merged_values.setdefault("identity_action_id", lineage.action_id)
        event["values"] = merged_values
        grouped[(owner, lineage.action_id)][family].append(index)

    for (_owner, action_id), families in grouped.items():
        spawn_indices = sorted(families["spawn"], key=lambda index: _event_time_us(result[index]))
        death_indices = sorted(families["death"], key=lambda index: _event_time_us(result[index]))
        suppress = [*spawn_indices[1:], *death_indices[:-1]]
        for index in suppress:
            event = result[index]
            event["certainty"] = "tentative"
            values = event.get("values")
            merged_values = dict(values) if isinstance(values, Mapping) else {}
            merged_values.update(
                {
                    "confirmation_state": "duplicate_track_fragment",
                    "identity_action_id": action_id,
                }
            )
            event["values"] = merged_values
    return result


def _normalized_rows(
    cache_path: Path,
    *,
    viewer_side: str,
    capture_start_monotonic_us: int | None = None,
    battle_start_monotonic_us: int | None = None,
    stream_offset_us: int = 0,
    identity_context: KnownCardIdentity | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Read a replay cache through the existing ``cr_bot`` session tracker."""

    try:
        from cr_bot.app.match_session import MatchSession
        from cr_bot.replay.cache import ReplayCacheReader
    except ImportError as error:  # pragma: no cover - depends on project env
        raise PhysicalExtractionError(
            "the cr_bot source environment is required to read replay caches"
        ) from error

    try:
        cache_rows = list(ReplayCacheReader(cache_path))
    except (EOFError, OSError, TypeError, ValueError, AttributeError) as error:
        raise PhysicalExtractionError(f"cannot read replay cache {cache_path}: {error}") from error
    if not cache_rows:
        raise PhysicalExtractionError(f"replay cache is empty: {cache_path}")

    # The first session frame that passes the extractor's game-start gate is a
    # useful alignment marker.  Its position is retained, but the displayed
    # game clock is never used to compute match time.
    session = MatchSession(tracker_debug=False)
    battle_start_video_s: float | None = None
    timeline: list[dict[str, object]] = []
    battle_cache_rows: list[object] = []
    own_events: list[dict[str, object]] = []
    enemy_events: list[dict[str, object]] = []
    entity_samples: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    previous_towers: Mapping[str, object] | None = None
    previous_tracks: set[tuple[str, str, str]] = set()
    track_states: dict[tuple[str, str, str], _LifecycleTrack] = {}
    raw_track_aliases: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    next_logical_track = 0
    seen_own_actions = 0
    seen_enemy_events = 0
    identity_rejections: list[dict[str, object]] = []

    def match_time_s(video_time_s: float) -> float | None:
        value = _internal_match_us(
            video_time_s,
            battle_start_video_s=battle_start_video_s,
            capture_start_monotonic_us=capture_start_monotonic_us,
            battle_start_monotonic_us=battle_start_monotonic_us,
            stream_offset_us=stream_offset_us,
        )
        return None if value is None else value / 1_000_000

    for cache_row in cache_rows:
        frame = cache_row.decode_frame()
        video_time_s = float(cache_row.video_time_s)
        step = session.process(
            cache_row.analysis,
            frame=frame,
            now_s=video_time_s,
        )
        if step.in_game and battle_start_video_s is None:
            battle_start_video_s = video_time_s
        if step.in_game:
            battle_cache_rows.append(cache_row)
            row = _analysis_row(
                cache_row,
                viewer_side=viewer_side,
                battle_start_video_s=battle_start_video_s,
                capture_start_monotonic_us=capture_start_monotonic_us,
                battle_start_monotonic_us=battle_start_monotonic_us,
                stream_offset_us=stream_offset_us,
                identity_context=identity_context,
            )
            timeline.append(row)
            row_rejections = row.get("identity_rejections")
            if isinstance(row_rejections, list):
                identity_rejections.extend(
                    item for item in row_rejections if isinstance(item, Mapping)
                )
            current_towers = row.get("towers_hp")
            if isinstance(current_towers, Mapping) and isinstance(previous_towers, Mapping):
                for tower, before in previous_towers.items():
                    after = current_towers.get(tower)
                    if type(before) in (int, float) and type(after) in (int, float) and float(after) < float(before):
                        tower_owner = viewer_side if str(tower).startswith("own_") else _opposite(viewer_side)
                        damage = float(before) - float(after)
                        tower_event = _event_row(
                                event_id=f"{viewer_side}-tower-damage-{len(enemy_events)+1:06d}",
                                kind="tower_damage_observed",
                                owner=tower_owner,
                                card_id=None,
                                video_time_s=video_time_s,
                                match_time_s=match_time_s(video_time_s),
                                frame_index=int(cache_row.frame_idx),
                                confidence=0.90,
                                certainty="inferred",
                                source_capture_side=viewer_side,
                                values={
                                    "tower": str(tower),
                                    "hp_before": float(before),
                                    "hp_after": float(after),
                                    "damage": damage,
                                    "tower_damage": damage,
                                },
                            )
                        if identity_context is not None:
                            attacker = _opposite(tower_owner)
                            event_time_us = tower_event.get("match_time_us")
                            candidates = [
                                placement
                                for placement in identity_context.placements
                                if placement.owner.upper() == attacker
                                and type(event_time_us) is int
                                and placement.match_time_us <= event_time_us
                                and identity_context._can_have_a_tracked_entity(placement.card_id)
                            ]
                            if len(candidates) == 1:
                                tower_event["source_card_id"] = candidates[0].card_id
                                tower_event["target_card_id"] = (
                                    "king-tower" if "king" in str(tower) else "princess-tower"
                                )
                        enemy_events.append(
                            tower_event
                        )
            if isinstance(current_towers, Mapping):
                previous_towers = dict(current_towers)

            current_tracks: set[tuple[str, str, str]] = set()
            for unit_index, unit in enumerate(row["units"]):
                if not isinstance(unit, Mapping):
                    continue
                card_id = str(unit.get("card_id") or "")
                owner = str(unit.get("owner") or "")
                if not card_id or not owner:
                    continue
                raw_track_id = str(unit.get("track_id"))
                if raw_track_id in {"", "None", "nan"}:
                    raw_track_id = f"unknown-{row['frame_index']}-{unit_index}"
                raw_key = (owner, card_id, raw_track_id)
                key = raw_track_aliases.get(raw_key)
                state = track_states.get(key) if key is not None else None
                if state is not None and (state.confirmed_dead or key in current_tracks):
                    key = None
                    state = None
                if state is None:
                    candidates: list[tuple[float, _LifecycleTrack]] = []
                    x_mtile = int(unit["x_mtile"])
                    y_mtile = int(unit["y_mtile"])
                    for candidate in track_states.values():
                        if (
                            candidate.owner != owner
                            or candidate.card_id != card_id
                            or candidate.confirmed_dead
                            or candidate.canonical_key in current_tracks
                        ):
                            continue
                        gap_us = int(row["video_time_us"]) - candidate.last_seen_video_us
                        distance = math.hypot(
                            float(x_mtile - candidate.last_x_mtile),
                            float(y_mtile - candidate.last_y_mtile),
                        )
                        if gap_us <= LIFECYCLE_REAPPEARANCE_GAP_US and distance <= LIFECYCLE_REAPPEARANCE_DISTANCE_MTILE:
                            candidates.append((distance, candidate))
                    if candidates:
                        _distance, state = min(candidates, key=lambda item: item[0])
                        key = state.canonical_key
                        raw_track_aliases[raw_key] = key
                    else:
                        next_logical_track += 1
                        logical_id = f"track-{next_logical_track:06d}"
                        key = (owner, card_id, logical_id)
                        state = _LifecycleTrack(
                            canonical_key=key,
                            owner=owner,
                            card_id=card_id,
                            logical_id=logical_id,
                            last_seen_video_us=int(row["video_time_us"]),
                            last_x_mtile=int(unit["x_mtile"]),
                            last_y_mtile=int(unit["y_mtile"]),
                            raw_track_ids=set(),
                        )
                        track_states[key] = state
                        raw_track_aliases[raw_key] = key

                reappeared = state.missing_frames > 0 and not state.confirmed_dead
                if reappeared and state.disappearance_event is not None:
                    disappearance_values = state.disappearance_event.get("values")
                    values = dict(disappearance_values) if isinstance(disappearance_values, Mapping) else {}
                    values.update(
                        {
                            "confirmation_state": "reappeared_within_gap",
                            "reappearance_gap_us": int(row["video_time_us"]) - state.last_seen_video_us,
                            "missing_frame_count": state.missing_frames,
                        }
                    )
                    state.disappearance_event["values"] = values
                    state.disappearance_event["certainty"] = "tentative"
                    state.disappearance_event = None
                state.missing_frames = 0
                state.missing_since_video_us = None
                state.confirmed_dead = False
                state.raw_track_ids.add(raw_track_id)
                state.last_seen_video_us = int(row["video_time_us"])
                state.last_x_mtile = int(unit["x_mtile"])
                state.last_y_mtile = int(unit["y_mtile"])
                current_tracks.add(key)
                entity_samples[key].append(
                    {
                        "frame_index": row["frame_index"],
                        "video_time_us": row["video_time_us"],
                        "match_time_us": row["match_time_us"],
                        "x_mtile": unit["x_mtile"],
                        "y_mtile": unit["y_mtile"],
                        "confidence": unit["confidence"],
                        "source_capture_id": f"capture-{viewer_side}",
                        "uncertainty_us": 100_000,
                    }
                )
                if key not in previous_tracks and not reappeared:
                    enemy_events.append(
                        _event_row(
                            event_id=f"{viewer_side}-spawn-{len(enemy_events)+1:06d}",
                            kind="unit_spawn_observed",
                            owner=owner,
                            card_id=card_id,
                            video_time_s=video_time_s,
                            match_time_s=match_time_s(video_time_s),
                            frame_index=int(cache_row.frame_idx),
                            confidence=float(unit.get("confidence") or 0.0),
                            certainty="inferred",
                            source_capture_side=viewer_side,
                            values=_unit_identity_values(unit),
                        )
                    )
            for key in sorted(previous_tracks - current_tracks, key=str):
                state = track_states[key]
                state.missing_frames += 1
                if state.missing_since_video_us is None:
                    state.missing_since_video_us = int(row["video_time_us"])
                if state.disappearance_event is None:
                    state.disappearance_event = _event_row(
                        event_id=f"{viewer_side}-death-{len(enemy_events)+1:06d}",
                        kind="unit_disappearance_observed",
                        owner=state.owner,
                        card_id=state.card_id,
                        video_time_s=video_time_s,
                        match_time_s=match_time_s(video_time_s),
                        frame_index=int(cache_row.frame_idx),
                        confidence=0.70,
                        certainty="tentative",
                        source_capture_side=viewer_side,
                        values={
                            "confirmation_state": "tentative",
                            "missing_frame_count": state.missing_frames,
                            "logical_track_id": state.logical_id,
                        },
                    )
                    enemy_events.append(state.disappearance_event)
                else:
                    disappearance_values = state.disappearance_event.get("values")
                    values = dict(disappearance_values) if isinstance(disappearance_values, Mapping) else {}
                    values["missing_frame_count"] = state.missing_frames
                    state.disappearance_event["values"] = values
                gap_us = int(row["video_time_us"]) - state.last_seen_video_us
                confirmed = (
                    state.missing_frames >= LIFECYCLE_CONFIRMATION_FRAMES
                    or gap_us > LIFECYCLE_REAPPEARANCE_GAP_US
                )
                if confirmed and not state.confirmed_dead:
                    state.disappearance_event["certainty"] = "inferred"
                    disappearance_values = state.disappearance_event.get("values")
                    values = dict(disappearance_values) if isinstance(disappearance_values, Mapping) else {}
                    values["confirmation_state"] = "confirmed"
                    values["missing_frame_count"] = state.missing_frames
                    state.disappearance_event["values"] = values
                    state.confirmed_dead = True
                    for raw_key, canonical_key in tuple(raw_track_aliases.items()):
                        if canonical_key == key:
                            del raw_track_aliases[raw_key]
            # Keep unconfirmed absences in the comparison set so a second
            # consecutive missing frame can promote the tentative event.
            previous_tracks = current_tracks | {
                key
                for key, state in track_states.items()
                if state.missing_frames > 0 and not state.confirmed_dead
            }

            actions = list(getattr(session.own_action_tracker, "actions", []) or [])
            while seen_own_actions < len(actions):
                action = actions[seen_own_actions]
                action_video_s = _json_scalar(getattr(action, "video_time_s", None))
                if type(action_video_s) not in (int, float):
                    action_video_s = video_time_s
                cell = getattr(action, "cell", None)
                own_event = _event_row(
                        event_id=f"{viewer_side}-own-play-{seen_own_actions+1:06d}",
                        kind="own_card_play_observed",
                        owner=viewer_side,
                        card_id=str(getattr(action, "card", "unknown")),
                        video_time_s=float(action_video_s),
                        match_time_s=match_time_s(float(action_video_s)),
                        frame_index=_nearest_frame_index(
                            timeline,
                            float(action_video_s),
                        ),
                        confidence=0.90,
                        certainty="inferred",
                        source_capture_side=viewer_side,
                        cell=cell,
                        values={
                            "slot_idx": _json_scalar(getattr(action, "slot_idx", None)),
                            "played_via": getattr(action, "played_via", None),
                        },
                    )
                filtered_event, rejection = _apply_event_identity(
                    own_event,
                    identity_context=identity_context,
                )
                if filtered_event is not None:
                    own_events.append(filtered_event)
                if rejection is not None:
                    identity_rejections.append(rejection)
                seen_own_actions += 1

        finished_enemy = step.finished_enemy_plays or []
        while seen_enemy_events < len(finished_enemy):
            play = finished_enemy[seen_enemy_events]
            event_video_s = _json_scalar(getattr(play, "video_time_s", None))
            if type(event_video_s) not in (int, float):
                event_video_s = video_time_s
            enemy_owner = _opposite(viewer_side)
            confidence = float(_json_scalar(getattr(play, "avg_confidence", 0.0)) or 0.0)
            direct = bool(getattr(play, "clock_confirmed", False) or getattr(play, "frame_confirmed", False))
            enemy_event = _event_row(
                    event_id=f"{viewer_side}-enemy-play-{seen_enemy_events+1:06d}",
                    kind="enemy_card_play_observed",
                    owner=enemy_owner,
                    card_id=str(getattr(play, "card", "unknown")),
                    video_time_s=float(event_video_s),
                    match_time_s=match_time_s(float(event_video_s)),
                    frame_index=_nearest_frame_index(timeline, float(event_video_s)),
                    confidence=confidence,
                    certainty="direct" if direct else "inferred",
                    source_capture_side=viewer_side,
                    cell=getattr(play, "cell", None),
                    values={
                        "cost": _json_scalar(getattr(play, "cost", None)),
                        "track_id": _json_scalar(getattr(play, "track_id", None)),
                        "clock_confirmed": bool(getattr(play, "clock_confirmed", False)),
                        "frame_confirmed": bool(getattr(play, "frame_confirmed", False)),
                        "team_ratio": _json_scalar(getattr(play, "team_ratio", None)),
                    },
                )
            filtered_event, rejection = _apply_event_identity(
                enemy_event,
                identity_context=identity_context,
            )
            if filtered_event is not None:
                enemy_events.append(filtered_event)
            if rejection is not None:
                identity_rejections.append(rejection)
            seen_enemy_events += 1

    # If the extractor did not reach its result-frame finalization boundary,
    # retain any still-visible cards as events rather than dropping the run.
    trailing_actions = list(getattr(session.own_action_tracker, "actions", []) or [])
    while seen_own_actions < len(trailing_actions):
        action = trailing_actions[seen_own_actions]
        action_video_s = _json_scalar(getattr(action, "video_time_s", None))
        if type(action_video_s) not in (int, float):
            action_video_s = float(timeline[-1]["video_time_us"]) / 1_000_000 if timeline else 0.0
        own_event = _event_row(
                event_id=f"{viewer_side}-own-play-{seen_own_actions+1:06d}",
                kind="own_card_play_observed",
                owner=viewer_side,
                card_id=str(getattr(action, "card", "unknown")),
                video_time_s=float(action_video_s),
                match_time_s=match_time_s(float(action_video_s)),
                frame_index=_nearest_frame_index(timeline, float(action_video_s)),
                confidence=0.90,
                certainty="inferred",
                source_capture_side=viewer_side,
                cell=getattr(action, "cell", None),
                values={"slot_idx": _json_scalar(getattr(action, "slot_idx", None))},
            )
        filtered_event, rejection = _apply_event_identity(
            own_event,
            identity_context=identity_context,
        )
        if filtered_event is not None:
            own_events.append(filtered_event)
        if rejection is not None:
            identity_rejections.append(rejection)
        seen_own_actions += 1

    entities: list[dict[str, object]] = []
    for index, (key, samples) in enumerate(sorted(entity_samples.items())):
        owner, card_id, track_id = key
        if not samples:
            continue
        confidence = min(float(sample["confidence"]) for sample in samples)
        entities.append(
            {
                "stable_observation_id": f"{viewer_side}-entity-{index:06d}",
                "card_id": card_id,
                "owner": owner,
                "confidence": confidence,
                "source_card_id": card_id,
                "samples": samples,
                "track_id_diagnostic": track_id,
            }
        )

    events = sorted(
        [*own_events, *enemy_events],
        key=lambda row: (int(row["video_time_us"]), str(row["event_id"])),
    )
    return (
        {
            "side": viewer_side,
            "cache_path": str(cache_path),
            "battle_start_video_time_us": (
                None if battle_start_video_s is None else _video_us(battle_start_video_s)
            ),
            "clock_provenance": {
                "source": (
                    "workstation_monotonic_capture_start_to_battle_barrier"
                    if capture_start_monotonic_us is not None and battle_start_monotonic_us is not None
                    else "capture_video_time_relative_to_extractor_battle_boundary"
                ),
                "match_time_authoritative": "internal_monotonic_capture_axis",
                "in_game_clock_used_for_timing": False,
                "in_game_clock_retained_as_diagnostic": True,
                "stream_alignment_offset_us": stream_offset_us,
            },
            "timeline": timeline,
            "entities": entities,
            "events": events,
            "identity_filter": (
                None if identity_context is None else identity_context.to_dict()
            ),
            "identity_rejections": identity_rejections,
            "event_counts": {
                "own": len(own_events),
                "enemy": len([event for event in events if event["kind"] == "enemy_card_play_observed"]),
                "tentative": len([event for event in events if event.get("certainty") == "tentative"]),
                "total": len(events),
            },
        },
        {
            "battle_start_video_time_s": battle_start_video_s,
            "frame_count": len(cache_rows),
            "battle_frame_count": len(battle_cache_rows),
            "entity_count": len(entities),
            "event_count": len(events),
            "identity_rejection_count": len(identity_rejections),
        },
    )


def _extractor_command(
    repository_root: Path,
    media_path: Path,
    cache_path: Path,
    *,
    side: str,
    sample_interval_s: float,
) -> tuple[list[str], str, str]:
    from ..video_pipeline import extractor_command

    command = extractor_command(
        media_path,
        cache_path,
        hud_variant="standard" if side == "A" else "alternative",
        sample_interval_s=sample_interval_s,
        yolo_detections=True,
    )
    # A is the ASUS calibrated 1080x2400 path.  B records its native
    # 1080x2280 geometry in the device/capture provenance, but the existing
    # extractor only supports the calibrated 1080x2400 processing canvas.  Do
    # not pass ``--no-normalize`` here: the detector rejects the Samsung
    # aspect ratio before the alternative bottom-HUD ROIs can run.  The
    # normalized stream retains both the native geometry and the extractor
    # transform so simulator coordinates are never confused with raw pixels.
    if side == "B":
        transform = "normalized_1080x2400_from_native_1080x2280"
        hud_profile = "alternative_samsung_on_calibrated_canvas"
    else:
        transform = "native_1080x2400_identity"
        hud_profile = "standard_asus"
    return command, transform, hud_profile


def _run_one_extractor(
    *,
    repository_root: Path,
    side: str,
    media_path: Path,
    cache_path: Path,
    screen_width_px: int | None,
    screen_height_px: int | None,
    sample_interval_s: float,
    timeout_s: float,
) -> ExtractorJobResult:
    command, transform, hud_profile = _extractor_command(
        repository_root,
        media_path,
        cache_path,
        side=side,
        sample_interval_s=sample_interval_s,
    )
    environment = os.environ.copy()
    source_root = repository_root / "src"
    # The simulator is a writable workspace subtree while the existing
    # cr_bot extractor lives in the sibling project-level ``src`` directory.
    # Keep repository-root artifacts inside the simulator workspace, but use
    # the actual source tree when that layout is in effect.
    if not source_root.is_dir() and (repository_root.parent / "src").is_dir():
        source_root = repository_root.parent / "src"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else os.pathsep.join((str(source_root), existing_pythonpath))
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=repository_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        return ExtractorJobResult(
            side=side,
            command=tuple(str(part) for part in command),
            status="timeout",
            returncode=None,
            stdout=str(error.stdout or "")[-8_000:],
            stderr=str(error.stderr or "")[-8_000:],
            replay_cache_path=str(cache_path),
            replay_cache_sha256=None,
            replay_cache_recognized=False,
            source_media_path=str(media_path),
            source_media_sha256=hash_file(media_path) if media_path.is_file() else None,
            screen_width_px=screen_width_px,
            screen_height_px=screen_height_px,
            frame_normalization=transform,
            hud_profile=hud_profile,
        )
    recognized = False
    cache_hash: str | None = None
    if completed.returncode == 0 and cache_path.is_file():
        try:
            seal = seal_replay_cache(cache_path)
            recognized = seal.recognized
            cache_hash = seal.sha256
        except (PhysicalLabError, OSError):
            recognized = False
    status = "completed" if completed.returncode == 0 and recognized else "failed"
    return ExtractorJobResult(
        side=side,
        command=tuple(str(part) for part in command),
        status=status,
        returncode=completed.returncode,
        stdout=completed.stdout[-8_000:],
        stderr=completed.stderr[-8_000:],
        replay_cache_path=str(cache_path),
        replay_cache_sha256=cache_hash,
        replay_cache_recognized=recognized,
        source_media_path=str(media_path),
        source_media_sha256=hash_file(media_path) if media_path.is_file() else None,
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        frame_normalization=transform,
        hud_profile=hud_profile,
    )


def _run_action_events(
    run: Mapping[str, Any],
    *,
    stream_payloads: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Project acknowledged input receipts into direct-timing observations.

    The action log is the only source that knows when the operator's input was
    accepted on the internal runner clock.  Keeping these rows alongside the
    vision-derived events lets direct-timing measurements use the same axis as
    the simulator replay while preserving the extractor output unchanged.
    """

    actions = run.get("actions")
    if not isinstance(actions, list):
        return []
    result: list[dict[str, object]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping) or action.get("accepted") is not True:
            continue
        side = str(action.get("side", "")).upper()
        card_id = action.get("card_id")
        match_time_us = action_match_time_us(run, action)
        if side not in {"A", "B"} or not isinstance(card_id, str) or type(match_time_us) is not int:
            continue
        timeline = stream_payloads.get(side, {}).get("timeline", [])
        frame_index = 0
        video_time_us = 0
        if isinstance(timeline, list) and timeline:
            nearest = min(
                (row for row in timeline if isinstance(row, Mapping)),
                key=lambda row: abs(
                    int(row.get("match_time_us") or 0) - match_time_us
                ),
                default=None,
            )
            if nearest is not None:
                frame_index = int(nearest.get("frame_index", 0))
                video_time_us = int(nearest.get("video_time_us", 0))
        values: dict[str, object] = {
            "cell_col": action.get("arena_cell", [None, None])[0],
            "cell_row": action.get("arena_cell", [None, None])[1],
            "slot_idx": action.get("card_slot"),
            "action_id": action.get("action_id"),
            "input_receipt": "runner_action_log",
            "identity_source": "placement_receipt",
        }
        result.append(
            {
                "event_id": f"runner-{side}-own-play-{index + 1:06d}",
                "kind": "own_card_play_observed",
                "video_time_us": video_time_us,
                "match_time_us": match_time_us,
                "confidence": 1.0,
                "certainty": "direct",
                "source_frame_indices": [frame_index],
                "evidence_refs": [f"run-action:{action.get('action_id', index)}"],
                "owner": side,
                "card_id": card_id,
                "values": values,
                "uncertainty_us": 5_000,
            }
        )
    return result


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise PhysicalExtractionError(
                f"immutable physical extraction artifact already differs: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _workspace_budget(root: Path) -> dict[str, object]:
    """Check the whole repository before/after materializing extractor data."""

    from ..storage import enforce_workspace_budget

    return enforce_workspace_budget(
        root,
        manifest_path=root / "outputs/simulator/fidelity_media/retention.json",
        raw_media_root=root / "outputs/simulator/fidelity_media",
        max_bytes=200_000_000_000,
        low_water_bytes=190_000_000_000,
        reserve_bytes=0,
        evict=False,
    )


def extract_physical_run(
    run_manifest_path: str | Path,
    *,
    repository_root: str | Path,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    extractor_timeout_s: float = DEFAULT_JOB_TIMEOUT_S,
    run_extractor: bool = True,
) -> dict[str, object]:
    """Extract both captured streams and write the compact physical case.

    The run manifest is immutable input.  Existing caches are reused only if
    they are recognized by ``cr_bot.replay.cache.ReplayCacheReader``; opaque
    files cannot cross the observation admission boundary.
    """

    root = Path(repository_root).resolve()
    run_path = Path(run_manifest_path).resolve()
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhysicalExtractionError(f"cannot load physical run manifest {run_path}: {error}") from error
    if not isinstance(run, Mapping) or run.get("kind") != "physical_lab_run":
        raise PhysicalExtractionError("extraction input must be a physical_lab_run manifest")
    run_id = str(run.get("run_id") or "")
    experiment = run.get("experiment")
    if not isinstance(experiment, Mapping):
        raise PhysicalExtractionError("physical run lacks its experiment specification")
    spec = ExperimentSpec.from_dict(experiment)
    if run.get("experiment_hash") != spec.experiment_hash():
        raise PhysicalExtractionError("physical run experiment hash does not match its specification")
    captures = run.get("captures")
    devices = run.get("device_info")
    if not isinstance(captures, Mapping) or not isinstance(devices, Mapping):
        raise PhysicalExtractionError("physical run must contain device_info and captures")
    if sample_interval_s <= 0 or extractor_timeout_s <= 0:
        raise PhysicalExtractionError("extractor timing values must be positive")
    if not 0 <= confidence_threshold <= 1:
        raise PhysicalExtractionError("confidence_threshold must be between zero and one")

    budget_before = _workspace_budget(root)
    if not budget_before["passed"]:
        raise PhysicalExtractionError(
            "repository exceeds the hard 200 GB cap before extraction: "
            f"deficit={budget_before['deficit_bytes']} bytes"
        )

    output_root = run_path.parent
    clock = run.get("clock_provenance")
    battle_start_monotonic_us = (
        clock.get("battle_start_monotonic_us")
        if isinstance(clock, Mapping) and type(clock.get("battle_start_monotonic_us")) is int
        else None
    )
    jobs: dict[str, ExtractorJobResult] = {}
    stream_payloads: dict[str, dict[str, object]] = {}
    stream_summaries: dict[str, dict[str, object]] = {}
    identity_context = _identity_context_from_run(run, spec)
    synchronization = run.get("synchronization")
    for side in ("A", "B"):
        capture = captures.get(side)
        device = devices.get(side)
        if not isinstance(capture, Mapping) or not isinstance(device, Mapping):
            raise PhysicalExtractionError(f"physical run lacks capture/device row for {side}")
        media_value = capture.get("media_path")
        if not isinstance(media_value, str) or not media_value:
            raise PhysicalExtractionError(f"capture {side} lacks media_path")
        media_path = Path(media_value)
        if not media_path.is_absolute():
            media_path = (root / media_path).resolve()
        cache_path = output_root / f"replay-cache-{side}.pkl.gz"
        existing_job: ExtractorJobResult | None = None
        if cache_path.is_file() and not run_extractor:
            try:
                seal = seal_replay_cache(cache_path)
                existing_job = ExtractorJobResult(
                    side=side,
                    command=(),
                    status="already_complete" if seal.recognized else "failed",
                    returncode=0,
                    stdout="",
                    stderr="",
                    replay_cache_path=str(cache_path),
                    replay_cache_sha256=seal.sha256 if seal.recognized else None,
                    replay_cache_recognized=seal.recognized,
                    source_media_path=str(media_path),
                    source_media_sha256=capture.get("media_sha256"),
                    screen_width_px=device.get("screen_width_px"),
                    screen_height_px=device.get("screen_height_px"),
                    frame_normalization=(
                        "native_1080x2400_identity"
                        if side == "A"
                        else "normalized_1080x2400_from_native_1080x2280"
                    ),
                    hud_profile=(
                        "standard_asus"
                        if side == "A"
                        else "alternative_samsung_on_calibrated_canvas"
                    ),
                )
            except (PhysicalLabError, OSError) as error:
                raise PhysicalExtractionError(f"cannot reuse replay cache {cache_path}: {error}") from error
        job = existing_job or _run_one_extractor(
            repository_root=root,
            side=side,
            media_path=media_path,
            cache_path=cache_path,
            screen_width_px=device.get("screen_width_px") if type(device.get("screen_width_px")) is int else None,
            screen_height_px=device.get("screen_height_px") if type(device.get("screen_height_px")) is int else None,
            sample_interval_s=sample_interval_s,
            timeout_s=extractor_timeout_s,
        )
        jobs[side] = job
        if not job.replay_cache_recognized:
            continue
        capture_start_monotonic_us = (
            capture.get("started_at_monotonic_us")
            if type(capture.get("started_at_monotonic_us")) is int
            else None
        )
        payload, summary = _normalized_rows(
            cache_path,
            viewer_side=side,
            capture_start_monotonic_us=capture_start_monotonic_us,
            battle_start_monotonic_us=battle_start_monotonic_us,
            stream_offset_us=_stream_alignment_offset_us(synchronization, side),
            identity_context=identity_context,
        )
        stream_payloads[side] = payload
        stream_summaries[side] = summary

    primary = stream_payloads.get("A") or stream_payloads.get("B")
    primary_side = "A" if "A" in stream_payloads else "B" if "B" in stream_payloads else None
    if primary is None or primary_side is None:
        raise PhysicalExtractionError("neither physical stream produced a recognized extractor cache")
    missing_streams = [side for side in ("A", "B") if side not in stream_payloads]
    if missing_streams:
        raise PhysicalExtractionError(
            "both phone streams must produce recognized extractor caches; "
            f"missing: {', '.join(missing_streams)}"
        )

    normalized_stream_refs: dict[str, dict[str, object]] = {}
    for side in ("A", "B"):
        stream_path = output_root / f"normalized-stream-{side}.json"
        _immutable_json(stream_path, stream_payloads[side])
        normalized_stream_refs[side] = {
            "path": str(stream_path),
            "sha256": hash_file(stream_path),
            "timeline_frame_count": len(stream_payloads[side].get("timeline", [])),
            "entity_count": len(stream_payloads[side].get("entities", [])),
            "event_count": len(stream_payloads[side].get("events", [])),
        }

    # Raw phone-specific entities remain in normalized-stream-A/B.  The scored
    # observation uses each owner's own-phone trajectory so two viewpoints do
    # not masquerade as two physical troops.
    stream_events_by_side: dict[str, list[dict[str, object]]] = {}
    for side in ("A", "B"):
        payload = stream_payloads.get(side)
        if payload is None:
            continue
        stream_events_by_side[side] = [
            event
            for event in payload.get("events", [])
            if isinstance(event, Mapping)
            and not (
                event.get("kind") == "own_card_play_observed"
                and event.get("owner") in {"A", "B"}
            )
        ]
    ruleset = load_ruleset(spec.ruleset_id)
    singleton_card_ids = frozenset(
        card_id
        for deck in spec.initial_conditions.decks.values()
        for card_id in deck
        if ruleset.card(card_id).spawn_count == 1
    )
    merged_entities = _merge_singleton_action_entities(
        _select_primary_view_entities(stream_payloads),
        identity_context=identity_context,
        singleton_card_ids=singleton_card_ids,
    )
    # The runner's acknowledged input receipt is the authoritative direct
    # timing record for operator actions.  First reconcile the two visual
    # viewpoints; this retains Phone B as corroborating evidence without
    # counting the same lifecycle transition twice.
    merged_events = _merge_cross_phone_events(stream_events_by_side)
    merged_events.extend(_run_action_events(run, stream_payloads=stream_payloads))
    merged_events = _reconcile_singleton_lifecycle_events(
        merged_events,
        identity_context=identity_context,
        singleton_card_ids=singleton_card_ids,
    )
    merged_events.sort(key=lambda row: (int(row.get("video_time_us", 0)), str(row.get("event_id", ""))))

    sync = synchronization
    if not isinstance(sync, Mapping):
        sync = {"accepted": False, "rejection_reasons": ["physical run has no synchronization result"]}
    capture_ids = tuple(
        str(captures[side].get("capture_id"))
        for side in ("A", "B")
        if isinstance(captures.get(side), Mapping) and captures[side].get("capture_id")
    )
    media_hashes = {
        side: str(captures[side].get("media_sha256"))
        for side in ("A", "B")
        if isinstance(captures.get(side), Mapping) and isinstance(captures[side].get("media_sha256"), str)
    }
    primary_cache_hash = jobs[primary_side].replay_cache_sha256
    raw_observation = {
        "entities": merged_entities,
        "events": merged_events,
    }
    observation = ingest_for_experiment(
        raw_observation,
        spec=spec,
        run_id=run_id,
        synchronization=sync,
        confidence_threshold=confidence_threshold,
        capture_ids=capture_ids,
        media_hashes=media_hashes,
        replay_cache_hash=primary_cache_hash,
        replay_cache_error=(
            None if primary_cache_hash is not None else "primary replay cache was not recognized"
        ),
    )
    observation_path = output_root / "observation.json"
    observation.save(observation_path)

    replay: SimulatorReplay | None = None
    comparison: ComparisonReport | None = None
    replay_path = output_root / "simulator-replay.json"
    if isinstance(run.get("simulator_replay"), Mapping):
        try:
            action_times = {
                str(action.get("action_id")): int(match_time_us)
                for action in run.get("actions", [])
                if isinstance(action, Mapping)
                and action.get("accepted") is True
                and isinstance(action.get("action_id"), str)
                and (match_time_us := action_match_time_us(run, action)) is not None
            }
            replay = run_simulator_replay(spec, action_times=action_times)
            seal_json(replay_path, replay.to_dict())
            comparison = compare_observation_to_replay(observation, replay)
            _immutable_json(output_root / "comparison.json", comparison.to_dict())
        except (PhysicalLabError, KeyError, TypeError, ValueError) as error:
            _immutable_json(
                output_root / "comparison.json",
                {
                    "kind": "physical_lab_divergence_report",
                    "run_id": run_id,
                    "eligible": False,
                    "rejection_reasons": [f"comparison failed: {error}"],
                },
            )

    payload: dict[str, object] = {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "kind": "physical_lab_extracted_case",
        "run_id": run_id,
        "experiment_hash": spec.experiment_hash(),
        "status": observation.status.value,
        "clock_provenance": {
            "source": "workstation_monotonic_and_capture_video_time",
            "match_time_axis": "internal_capture_time_from_extractor_battle_boundary",
            "in_game_clock_used_for_timing": False,
            "in_game_clock_retained_as_diagnostic": True,
        },
        "device_geometry": {
            side: {
                "screen_width_px": devices[side].get("screen_width_px"),
                "screen_height_px": devices[side].get("screen_height_px"),
                "extractor_frame_normalization": jobs[side].frame_normalization,
                "hud_profile": jobs[side].hud_profile,
            }
            for side in ("A", "B")
            if side in jobs
        },
        "extractor_jobs": {side: job.to_dict() for side, job in sorted(jobs.items())},
        "stream_summaries": stream_summaries,
        "identity_filter": identity_context.to_dict(),
        "identity_rejection_count": sum(
            int(summary.get("identity_rejection_count", 0))
            for summary in stream_summaries.values()
            if isinstance(summary, Mapping)
        ),
        "normalized_streams": normalized_stream_refs,
        "stream_events": {
            side: {
                "battle_start_video_time_us": stream_payloads[side].get("battle_start_video_time_us"),
                "event_counts": stream_payloads[side].get("event_counts"),
            }
            for side in sorted(stream_payloads)
        },
        "observation": {
            "path": str(observation_path),
            "sha256": hash_file(observation_path),
            "event_count": len(observation.events),
            "entity_count": len(observation.entities),
            "rejected_count": len(observation.rejected),
            "replay_cache_hash": observation.replay_cache_hash,
        },
        "comparison": (
            None
            if comparison is None
            else {
                "path": str(output_root / "comparison.json"),
                "sha256": hash_file(output_root / "comparison.json"),
                "eligible": comparison.eligible,
                "metrics": comparison.metrics,
            }
        ),
        "admission": {
            "candidate_only_until_review": True,
            "reason": "physical extractor output is normalized evidence, not automatic ground truth",
        },
        "workspace_budget_before": budget_before,
    }
    budget_after = _workspace_budget(root)
    if not budget_after["passed"]:
        raise PhysicalExtractionError(
            "repository exceeds the hard 200 GB cap after extraction: "
            f"deficit={budget_after['deficit_bytes']} bytes"
        )
    payload["workspace_budget_after"] = budget_after
    payload["case_hash"] = canonical_hash(payload)
    extraction_path = output_root / "extracted-case.json"
    _immutable_json(extraction_path, payload)
    return {
        "path": str(extraction_path),
        "sha256": hash_file(extraction_path),
        "observation_path": str(observation_path),
        "comparison_path": str(output_root / "comparison.json") if (output_root / "comparison.json").is_file() else None,
        "status": observation.status.value,
        "event_count": len(observation.events),
        "entity_count": len(observation.entities),
        "job_status": {side: job.status for side, job in sorted(jobs.items())},
        "workspace_budget_after": budget_after,
    }


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_JOB_TIMEOUT_S",
    "DEFAULT_SAMPLE_INTERVAL_S",
    "EXTRACTION_SCHEMA_VERSION",
    "ExtractorJobResult",
    "PhysicalExtractionError",
    "extract_physical_run",
]
