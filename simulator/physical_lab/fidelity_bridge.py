"""Fail-closed bridge from admitted physical observations to fidelity corpora.

The physical lab and the simulator intentionally use different wire formats.
This module is the narrow, reviewable boundary between them.  It does not
infer labels, repair timing, or promote a candidate run.  It only accepts an
observation manifest that already passed physical admission and projects its
directly timed events/entity samples into the existing validation-corpus
schema.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from ..engine import ENGINE_VERSION
from ..ruleset import load_ruleset
from ..validation import validation_corpus_from_dict
from .observation import (
    EntityObservation,
    NormalizedEvent,
    ObservationCertainty,
    ObservationManifest,
)
from .replay import SimulatorReplay, action_match_time_us, run_simulator_replay
from .schema import EvidenceStatus, ExperimentSpec, PhysicalLabError, canonical_hash


_PHYSICAL_EVIDENCE_STATUSES = frozenset(
    {
        EvidenceStatus.CALIBRATED_ONLY,
        EvidenceStatus.VALIDATION,
        EvidenceStatus.HELDOUT,
        EvidenceStatus.REGRESSION,
    }
)
_RUN_HASH_FIELD = "run_hash"
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STABLE_EVENT_FIELDS = frozenset(
    {
        "card_id",
        "owner",
        "source_card_id",
        "source_owner",
        "source_role",
        "source_kind",
        "target_card_id",
        "target_owner",
        "target_role",
        "target_kind",
        "old_target_card_id",
        "old_target_owner",
        "old_target_role",
        "old_target_kind",
        "damage",
        "hp_after",
        "card_slot",
        "col",
        "row",
        "player",
        "attack_number",
        "pellet_index",
        "projectile_speed_code",
    }
)
_RESERVED_EVENT_FIELDS = frozenset(
    {"uid", "source_uid", "target_uid", "old_target", "sequence"}
)


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicalLabError(f"{field_name} must be an object")
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _tick(match_time_us: int, tick_us: int) -> int:
    if type(match_time_us) is not int or match_time_us < 0:
        raise PhysicalLabError("physical observations require non-negative match time")
    return _ceil_div(match_time_us, tick_us)


def _owner_number(owner: str | int | None) -> int | None:
    if owner is None:
        return None
    if owner in (0, 1):
        return int(owner)
    if owner == "A":
        return 0
    if owner == "B":
        return 1
    raise PhysicalLabError(f"unsupported physical owner value: {owner!r}")


def _run_hash_is_valid(run: Mapping[str, Any]) -> None:
    declared = run.get(_RUN_HASH_FIELD)
    if not isinstance(declared, str):
        raise PhysicalLabError("sealed physical run is missing run_hash")
    unsigned = dict(run)
    unsigned.pop(_RUN_HASH_FIELD, None)
    if canonical_hash(unsigned) != declared:
        raise PhysicalLabError("sealed physical run run_hash does not match its contents")


def _require_run_provenance(
    manifest: ObservationManifest,
    run: Mapping[str, Any],
    *,
    replay_cache_hash: str,
) -> ExperimentSpec:
    if manifest.status not in _PHYSICAL_EVIDENCE_STATUSES:
        raise PhysicalLabError(
            f"physical fidelity bridge requires admitted evidence, got {manifest.status.value}"
        )
    if manifest.rejected:
        raise PhysicalLabError("physical fidelity bridge refuses a manifest with rejected observations")
    if not _HASH_RE.fullmatch(replay_cache_hash):
        raise PhysicalLabError("recognized replay-cache attestation is not a sha256 hash")
    if manifest.replay_cache_hash != replay_cache_hash:
        raise PhysicalLabError("observation replay-cache hash does not match the recognized cache")
    if not isinstance(run.get("run_id"), str) or run["run_id"] != manifest.run_id:
        raise PhysicalLabError("observation and sealed run have different run IDs")
    if run.get("status") == EvidenceStatus.REJECTED.value:
        raise PhysicalLabError("physical fidelity bridge refuses a rejected run")
    _run_hash_is_valid(run)

    try:
        spec = ExperimentSpec.from_dict(_mapping(run.get("experiment"), "run.experiment"))
    except (KeyError, TypeError, ValueError) as error:
        raise PhysicalLabError(f"sealed run has an invalid experiment: {error}") from error
    if run.get("experiment_hash") != spec.experiment_hash():
        raise PhysicalLabError("sealed run experiment_hash does not match its experiment")
    if manifest.experiment_hash != spec.experiment_hash():
        raise PhysicalLabError("observation and sealed run have different experiment hashes")
    if manifest.capture_group_id != spec.capture_group_id:
        raise PhysicalLabError("observation capture group does not match the sealed experiment")
    if manifest.evidence_split.value != spec.evidence_split.value:
        raise PhysicalLabError("observation split does not match the sealed experiment")
    expected_status = {
        "calibration": EvidenceStatus.CALIBRATED_ONLY,
        "validation": EvidenceStatus.VALIDATION,
        "heldout": EvidenceStatus.HELDOUT,
        "regression": EvidenceStatus.REGRESSION,
    }[manifest.evidence_split.value]
    if manifest.status is not expected_status:
        raise PhysicalLabError("observation status does not match its preassigned evidence split")

    run_sync = _mapping(run.get("synchronization"), "run.synchronization")
    if run_sync.get("accepted") is not True:
        raise PhysicalLabError("sealed run synchronization was not accepted")
    if manifest.synchronization.get("accepted") is not True:
        raise PhysicalLabError("observation synchronization was not accepted")

    lifecycle = _mapping(run.get("lifecycle"), "run.lifecycle")
    if lifecycle.get("passed") is not True:
        raise PhysicalLabError("sealed run lifecycle was not passed")

    captures = _mapping(run.get("captures"), "run.captures")
    if set(captures) != {"A", "B"}:
        raise PhysicalLabError("physical fidelity bridge requires verified A/B captures")
    run_capture_ids: set[str] = set()
    run_media_hashes: dict[str, str] = {}
    for side in ("A", "B"):
        capture = _mapping(captures[side], f"run.captures.{side}")
        if capture.get("source_device") != side:
            raise PhysicalLabError(f"capture {side} is bound to the wrong source device")
        if capture.get("status") != "complete":
            raise PhysicalLabError(f"capture {side} is not complete")
        if capture.get("stream_verified") is not True:
            raise PhysicalLabError(f"capture {side} is not stream-verified")
        if type(capture.get("frame_count")) is not int or capture["frame_count"] <= 0:
            raise PhysicalLabError(f"capture {side} has no verified frames")
        capture_id = capture.get("capture_id")
        media_hash = capture.get("media_sha256")
        if not isinstance(capture_id, str) or not capture_id:
            raise PhysicalLabError(f"capture {side} is missing capture_id")
        if not isinstance(media_hash, str) or not _HASH_RE.fullmatch(media_hash):
            raise PhysicalLabError(f"capture {side} is missing media_sha256")
        run_capture_ids.add(capture_id)
        run_media_hashes[side] = media_hash

    if set(manifest.capture_ids) != run_capture_ids:
        raise PhysicalLabError("observation must bind both sealed capture IDs")
    if set(manifest.media_hashes) != {"A", "B"}:
        raise PhysicalLabError("observation must bind both sealed capture media hashes")
    for side, media_hash in manifest.media_hashes.items():
        if not _HASH_RE.fullmatch(media_hash) or run_media_hashes[side] != media_hash:
            raise PhysicalLabError(f"observation media hash for {side} does not match the sealed run")

    devices = _mapping(run.get("device_info"), "run.device_info")
    if set(devices) != {"A", "B"}:
        raise PhysicalLabError("sealed run is missing A/B device provenance")
    for side in ("A", "B"):
        device = _mapping(devices[side], f"run.device_info.{side}")
        expected_device = spec.devices[side]
        if device.get("connected") is not True or device.get("serial_hash") != expected_device.serial_hash:
            raise PhysicalLabError(f"device provenance for {side} does not match the experiment")

    actions = run.get("actions")
    if not isinstance(actions, list):
        raise PhysicalLabError("sealed run actions must be an array")
    if len(actions) != len(spec.actions):
        raise PhysicalLabError("sealed run does not acknowledge every experiment action")
    expected_action_ids = {action.action_id for action in spec.actions}
    actual_action_ids: set[str] = set()
    for index, action in enumerate(actions):
        row = _mapping(action, f"run.actions[{index}]")
        if row.get("accepted") is not True or type(row.get("actual_match_time_us")) is not int:
            raise PhysicalLabError(f"run action {index} is not an acknowledged timed action")
        if row["actual_match_time_us"] < 0 or not isinstance(row.get("action_id"), str):
            raise PhysicalLabError(f"run action {index} has invalid timing or identity")
        actual_action_ids.add(row["action_id"])
    if actual_action_ids != expected_action_ids:
        raise PhysicalLabError("sealed run action IDs do not match the experiment")
    return spec


def _sync_uncertainty_us(manifest: ObservationManifest, run: Mapping[str, Any]) -> int:
    values = [manifest.synchronization.get("uncertainty_us")]
    run_sync = run.get("synchronization")
    if isinstance(run_sync, Mapping):
        values.append(run_sync.get("uncertainty_us"))
    clean = [int(value) for value in values if type(value) is int and value >= 0]
    return max(clean, default=0)


def _timed_uncertainty_ticks(
    uncertainty_us: int,
    *,
    sync_uncertainty_us: int,
    tick_us: int,
) -> int:
    total = uncertainty_us + sync_uncertainty_us
    return _ceil_div(total, tick_us) if total else 0


def _event_values(event: NormalizedEvent) -> dict[str, Any]:
    values = {
        key: value
        for key, value in event.values.items()
        if key not in _RESERVED_EVENT_FIELDS and key in _STABLE_EVENT_FIELDS
    }
    if event.card_id is not None:
        values["card_id"] = event.card_id
    owner = _owner_number(event.owner)
    if owner is not None:
        values["owner"] = owner
    for key in ("source_card_id", "target_role", "target_card_id"):
        value = getattr(event, key)
        if value is not None:
            values[key] = value
    return dict(sorted(values.items()))


def _event_filter(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: values[key]
        for key in (
            "card_id",
            "owner",
            "source_card_id",
            "target_role",
            "target_card_id",
        )
        if key in values
    }


def _event_mechanic(event: NormalizedEvent) -> str:
    card_id = event.card_id or event.source_card_id or event.target_card_id
    if event.kind == "target_changed" and card_id == "hog-rider":
        return "hog_cannon_pull_targeting"
    if card_id == "hog-rider":
        return f"hog-rider_{event.kind}"
    if card_id:
        return f"{card_id}_{event.kind}"
    return f"physical_{event.kind}"


def _entity_mechanic(entity: EntityObservation, axis: str) -> str:
    return f"{entity.card_id}_isolated_movement_{axis}"


def _evidence(
    manifest: ObservationManifest,
    *,
    confidence: float | None,
    frame_start: int | None,
    frame_end: int | None,
) -> dict[str, Any]:
    return {
        "source_id": manifest.run_id,
        "group_id": manifest.capture_group_id,
        "method": "physical_lab_observation_manifest_v1",
        "confidence": confidence,
        "notes": "admitted physical observation; cache and run bindings verified",
        "media_hash": canonical_hash(manifest.media_hashes) if manifest.media_hashes else None,
        "frame_start": frame_start,
        "frame_end": frame_end,
    }


def build_fidelity_corpus_payload(
    manifest: ObservationManifest,
    run: Mapping[str, Any],
    *,
    replay_cache_hash: str,
    replay: SimulatorReplay | None = None,
    position_tolerance_mtile: int = 250,
) -> dict[str, object]:
    """Compile one admitted physical manifest into a validated corpus payload."""

    if type(position_tolerance_mtile) is not int or position_tolerance_mtile < 0:
        raise PhysicalLabError("position_tolerance_mtile must be a non-negative integer")
    spec = _require_run_provenance(
        manifest,
        run,
        replay_cache_hash=replay_cache_hash,
    )
    if replay is None:
        action_times = {
            str(row["action_id"]): int(match_time_us)
            for row in run["actions"]  # type: ignore[index]
            if isinstance(row, Mapping)
            and isinstance(row.get("action_id"), str)
            and (match_time_us := action_match_time_us(run, row)) is not None
        }
        try:
            replay = run_simulator_replay(spec, action_times=action_times)
        except PhysicalLabError:
            raise
        except (TypeError, ValueError) as error:
            raise PhysicalLabError(f"cannot construct simulator replay: {error}") from error
    if replay.experiment_hash != spec.experiment_hash():
        raise PhysicalLabError("supplied simulator replay does not match the physical run")

    ruleset = load_ruleset(spec.ruleset_id)
    if ruleset.content_hash != spec.ruleset_hash or replay.scenario.engine_version != ENGINE_VERSION:
        raise PhysicalLabError("simulator replay identity does not match the experiment")
    tick_us = ruleset.tick_us
    sync_uncertainty_us = _sync_uncertainty_us(manifest, run)

    all_confidences: list[float] = []
    frame_indices: list[int] = []
    measurements: list[dict[str, Any]] = []
    for entity in manifest.entities:
        all_confidences.append(entity.confidence)
        filters: dict[str, Any] = {
            "card_id": entity.card_id,
            "owner": _owner_number(entity.owner),
        }
        if entity.role is not None:
            filters["role"] = entity.role
        for sample_index, sample in enumerate(entity.samples):
            if sample.match_time_us is None:
                raise PhysicalLabError(
                    f"entity {entity.stable_observation_id!r} lacks direct match timing"
                )
            all_confidences.append(sample.confidence)
            frame_indices.append(sample.frame_index)
            sample_tick = _tick(sample.match_time_us, tick_us)
            tick_tolerance = _timed_uncertainty_ticks(
                sample.uncertainty_us,
                sync_uncertainty_us=sync_uncertainty_us,
                tick_us=tick_us,
            )
            evidence = _evidence(
                manifest,
                confidence=sample.confidence,
                frame_start=sample.frame_index,
                frame_end=sample.frame_index,
            )
            base_id = f"{manifest.run_id}:{entity.stable_observation_id}:{sample_index}"
            for axis, value in (("x", sample.x_mtile), ("y", sample.y_mtile)):
                measurements.append(
                    {
                        "sample_id": f"{base_id}:{axis}",
                        "mechanic": _entity_mechanic(entity, axis),
                        "observed_value": value,
                        "observed_tick": sample_tick,
                        "tolerance": {
                            "absolute": position_tolerance_mtile,
                            "relative": 0.0,
                            "ticks": tick_tolerance,
                        },
                        "extractor": {
                            "type": f"entity_{axis}_mtile_at_tick",
                            "tick": sample_tick,
                            "filters": filters,
                        },
                    }
                )

    grouped_events: dict[tuple[str, str], list[tuple[NormalizedEvent, dict[str, Any]]]] = {}
    for event in manifest.events:
        if event.certainty is ObservationCertainty.TENTATIVE:
            # A tentative lifecycle edge remains in the readable observation
            # manifest for auditability, but cannot become a corpus trace.
            continue
        if event.match_time_us is None:
            raise PhysicalLabError(f"event {event.event_id!r} lacks direct match timing")
        if event.certainty is not ObservationCertainty.DIRECT:
            raise PhysicalLabError(f"event {event.event_id!r} is not directly timed")
        values = _event_values(event)
        filters = _event_filter(values)
        key = (event.kind, json.dumps(filters, sort_keys=True, separators=(",", ":")))
        grouped_events.setdefault(key, []).append((event, values))
        all_confidences.append(event.confidence)
        frame_indices.extend(event.source_frame_indices)

    traces: list[dict[str, Any]] = []
    for trace_index, ((event_kind, filter_json), rows) in enumerate(sorted(grouped_events.items())):
        filters = json.loads(filter_json)
        trace_events = []
        mechanic = _event_mechanic(rows[0][0])
        for event, values in sorted(rows, key=lambda item: (item[0].match_time_us or 0, item[0].event_id)):
            assert event.match_time_us is not None
            trace_events.append(
                {
                    "tick": _tick(event.match_time_us, tick_us),
                    "kind": event.kind,
                    "values": values,
                    "tick_tolerance": _timed_uncertainty_ticks(
                        event.uncertainty_us,
                        sync_uncertainty_us=sync_uncertainty_us,
                        tick_us=tick_us,
                    ),
                    "value_tolerances": {},
                }
            )
        traces.append(
            {
                "trace_id": f"{manifest.run_id}:trace:{trace_index}",
                "mechanic": mechanic,
                "included_event_kinds": [event_kind],
                "filters": filters,
                "events": trace_events,
            }
        )

    if not measurements and not traces:
        raise PhysicalLabError("physical observation contains no timed events or entity samples")
    frame_start = min(frame_indices) if frame_indices else None
    frame_end = max(frame_indices) if frame_indices else None
    evidence = _evidence(
        manifest,
        confidence=min(all_confidences) if all_confidences else None,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    case_id = f"{manifest.run_id}:physical"
    payload: dict[str, object] = {
        "schema_version": 1,
        "corpus_id": f"physical-lab-{manifest.run_id}",
        "engine_version": replay.scenario.engine_version,
        "ruleset_id": replay.scenario.ruleset_id,
        "ruleset_hash": replay.scenario.ruleset_hash,
        "cases": [
            {
                "case_id": case_id,
                "split": manifest.evidence_split.value,
                "evidence": evidence,
                "scenario": replay.scenario.to_dict(),
                "measurements": measurements,
                "traces": traces,
            }
        ],
    }
    try:
        validation_corpus_from_dict(payload)
    except (TypeError, ValueError) as error:
        raise PhysicalLabError(f"physical observation cannot form a fidelity corpus: {error}") from error
    return payload


def write_fidelity_corpus_payload(path: str | Path, payload: Mapping[str, object]) -> str:
    """Validate and atomically write a bridge-produced corpus."""

    checked = validation_corpus_from_dict(dict(payload), base_dir=Path(path).resolve().parent)
    encoded = json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return checked.content_hash


__all__ = [
    "build_fidelity_corpus_payload",
    "write_fidelity_corpus_payload",
]
