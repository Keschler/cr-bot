#!/usr/bin/env python3
"""Build reviewed, hash-sealed prerequisites for a physical-lab run.

The input is intentionally explicit and machine-checkable.  A minimal input
document has this shape::

    {
      "schema_version": 1,
      "devices": {
        "A": {
          "device_id": "A",
          "device_label": "phone-a",
          "device_serial_hash": "sha256:<64 lowercase hex>",
          "screen_width_px": 1080,
          "screen_height_px": 2400,
          "calibration_id": "phone-a-calibration-2026-08-21",
          "arena_px": [0, 300, 1080, 1500],
          "hand_px": [0, 1900, 1080, 500],
          "reviewed": true,
          "reviewer": "reviewer-id",
          "reviewed_at": "2026-08-21T12:00:00Z",
          "game_patch": "patch-id",
          "level": 11,
          "source_screenshots": [
            {"state": "recovery", "path": "sources/A/recovery.png"},
            {"state": "lobby", "path": "sources/A/lobby.png"},
            {"state": "challenge_sent", "path": "sources/A/challenge_sent.png"},
            {"state": "challenge_accepted", "path": "sources/A/challenge_accepted.png"},
            {"state": "loading", "path": "sources/A/loading.png"},
            {"state": "battle", "path": "sources/A/battle.png"},
            {"state": "result", "path": "sources/A/result.png"},
            {"state": "archived", "path": "sources/A/archived.png"}
          ]
        },
        "B": {"...": "the same explicit fields for phone B"}
      }
    }

Paths in the input are resolved relative to the input JSON's directory.  An
optional ``sha256`` (or ``source_sha256``) on a screenshot row is checked
against the actual file; the builder always computes and records the actual
hash itself.  The generated directory contains one real
``CalibrationArtifact`` JSON and one ``TemplateLifecycleDetector``-compatible
manifest per device, plus a provenance sidecar.

This module never talks to ADB or any device.  It is a prerequisite compiler,
not an experiment runner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
from typing import Any, Mapping
import zlib

# Make direct invocation independent of the caller's working directory.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from simulator.physical_lab.artifacts import hash_file
from simulator.physical_lab.calibration import CalibrationArtifact, CalibrationError
from simulator.physical_lab.lifecycle import LifecycleState
from simulator.physical_lab.schema import PhysicalLabError, canonical_hash
from simulator.physical_lab.screen_state import (
    SCREEN_TEMPLATE_SCHEMA_VERSION,
    ScreenStateDetectionError,
    TemplateLifecycleDetector,
)


INPUT_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_EXPECTED_STATES = tuple(state.value for state in LifecycleState)
_DEFAULT_SCORE_THRESHOLD = 0.96
_DEFAULT_MARGIN_THRESHOLD = 0.02


class PrerequisiteBuilderError(PhysicalLabError):
    """Raised when prerequisite input cannot be admitted safely."""


# A short alias makes CLI/test error handling convenient without hiding the
# more descriptive public exception name.
BuilderError = PrerequisiteBuilderError


@dataclass(frozen=True, slots=True)
class SourceScreenshot:
    state: LifecycleState
    source_reference: str
    source_path: Path
    sha256: str
    width_px: int
    height_px: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class ValidatedDevice:
    device_id: str
    device_label: str
    device_serial_hash: str
    screen_width_px: int
    screen_height_px: int
    calibration_id: str
    arena_px: tuple[float, float, float, float]
    hand_px: tuple[float, float, float, float]
    hand_slot_count: int
    reviewer: str
    reviewed_at: str
    game_patch: str
    level: int
    reviewed: bool
    score_threshold: float
    margin_threshold: float
    screenshots: tuple[SourceScreenshot, ...]
    calibration: CalibrationArtifact


@dataclass(frozen=True, slots=True)
class RenderedDevice:
    device: ValidatedDevice
    calibration_relative_path: str
    calibration_file_sha256: str
    calibration_hash: str
    manifest_relative_path: str
    manifest_file_sha256: str
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class _LoadedInput:
    document: Mapping[str, Any]
    source_base_dir: Path
    metadata_sha256: str | None


def _fail(message: str) -> PrerequisiteBuilderError:
    return PrerequisiteBuilderError(message)


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{field_name} must be an object")
    return value


def _check_unknown(raw: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise _fail(f"unknown fields at {field_name}: {unknown}")


def _text(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise _fail(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise _fail(f"{field_name} must be a positive integer")
    return value


def _hash(value: object, field_name: str) -> str:
    if type(value) is not str or not _HASH_RE.fullmatch(value):
        raise _fail(f"{field_name} must be sha256:<64 lowercase hex characters>")
    return value


def _number(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise _fail(f"{field_name} must be a finite number")
    return float(value)


def _rect(
    value: object,
    field_name: str,
    *,
    screen_width_px: int,
    screen_height_px: int,
) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise _fail(f"{field_name} must be [x, y, width, height]")
    x, y, width, height = tuple(
        _number(item, f"{field_name}[{index}]") for index, item in enumerate(value)
    )
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise _fail(f"{field_name} must have non-negative origin and positive size")
    if x + width > screen_width_px or y + height > screen_height_px:
        raise _fail(f"{field_name} must lie inside the declared phone screen")
    return x, y, width, height


def _probability(value: object, field_name: str, default: float) -> float:
    if value is None:
        return default
    result = _number(value, field_name)
    if not 0 <= result <= 1:
        raise _fail(f"{field_name} must be between zero and one")
    return result


def _reviewed_at(value: object) -> str:
    result = _text(value, "reviewed_at")
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as error:
        raise _fail("reviewed_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _fail("reviewed_at must include an explicit timezone")
    return result


def _png_dimensions(payload: bytes, path: Path) -> tuple[int, int]:
    """Parse and structurally validate a PNG without depending on a decoder."""

    if not payload.startswith(_PNG_SIGNATURE):
        raise _fail(f"source screenshot is not a PNG: {path}")
    offset = len(_PNG_SIGNATURE)
    dimensions: tuple[int, int] | None = None
    found_idat = False
    found_iend = False
    while offset < len(payload):
        if len(payload) - offset < 12:
            raise _fail(f"truncated PNG chunk in source screenshot: {path}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_start = offset + 8
        chunk_end = chunk_start + length
        crc_end = chunk_end + 4
        if crc_end > len(payload):
            raise _fail(f"PNG chunk exceeds file length: {path}")
        chunk_data = payload[chunk_start:chunk_end]
        declared_crc = struct.unpack(">I", payload[chunk_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if declared_crc != actual_crc:
            raise _fail(f"PNG CRC mismatch in {path}")
        if dimensions is None:
            if chunk_type != b"IHDR" or length != 13:
                raise _fail(f"PNG must begin with a valid IHDR chunk: {path}")
            width, height = struct.unpack(">II", chunk_data[:8])
            if width == 0 or height == 0:
                raise _fail(f"PNG dimensions must be positive: {path}")
            dimensions = (width, height)
        elif chunk_type == b"IHDR":
            raise _fail(f"PNG contains duplicate IHDR chunks: {path}")
        if chunk_type == b"IDAT":
            found_idat = True
        if chunk_type == b"IEND":
            if length != 0:
                raise _fail(f"PNG IEND chunk must be empty: {path}")
            found_iend = True
            offset = crc_end
            if offset != len(payload):
                raise _fail(f"PNG contains data after IEND: {path}")
            break
        offset = crc_end
    if dimensions is None or not found_idat or not found_iend:
        raise _fail(f"PNG is missing IHDR, IDAT, or IEND: {path}")
    return dimensions


def _resolve_source_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path)
    try:
        return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()
    except OSError as error:
        raise _fail(f"cannot resolve screenshot path {raw_path!r}: {error}") from error


def _parse_screenshots(
    raw: Mapping[str, Any],
    *,
    device_id: str,
    screen_width_px: int,
    screen_height_px: int,
    source_base_dir: Path,
) -> tuple[SourceScreenshot, ...]:
    if "source_screenshots" in raw and "screenshots" in raw:
        raise _fail(f"devices.{device_id} must use only one screenshot-list field")
    field_name = "source_screenshots" if "source_screenshots" in raw else "screenshots"
    rows_raw = raw.get(field_name)
    if not isinstance(rows_raw, list):
        raise _fail(f"devices.{device_id}.{field_name} must be an array")
    if len(rows_raw) != len(_EXPECTED_STATES):
        raise _fail(
            f"devices.{device_id}.{field_name} must contain exactly {len(_EXPECTED_STATES)} entries"
        )

    parsed: list[SourceScreenshot] = []
    labels: list[str] = []
    for index, value in enumerate(rows_raw):
        row = _require_mapping(value, f"devices.{device_id}.{field_name}[{index}]")
        _check_unknown(
            row,
            {"state", "path", "sha256", "source_sha256"},
            f"devices.{device_id}.{field_name}[{index}]",
        )
        label = _text(row.get("state"), f"devices.{device_id}.{field_name}[{index}].state")
        labels.append(label)
        try:
            state = LifecycleState(label)
        except ValueError as error:
            raise _fail(f"unknown lifecycle state label {label!r}") from error
        raw_path = _text(row.get("path"), f"devices.{device_id}.{field_name}[{index}].path")
        if Path(raw_path).suffix.lower() != ".png":
            raise _fail(f"source screenshot path must have a .png suffix: {raw_path!r}")
        source_path = _resolve_source_path(raw_path, source_base_dir)
        if not source_path.is_file():
            raise _fail(f"source screenshot does not exist: {source_path}")
        try:
            payload = source_path.read_bytes()
        except OSError as error:
            raise _fail(f"cannot read source screenshot {source_path}: {error}") from error
        width_px, height_px = _png_dimensions(payload, source_path)
        if (width_px, height_px) != (screen_width_px, screen_height_px):
            raise _fail(
                f"{source_path} dimensions {width_px}x{height_px} do not match "
                f"declared phone dimensions {screen_width_px}x{screen_height_px}"
            )
        try:
            source_sha256 = hash_file(source_path)
        except PhysicalLabError as error:
            raise _fail(str(error)) from error
        supplied_hashes = [row[key] for key in ("sha256", "source_sha256") if key in row]
        if len(supplied_hashes) > 1:
            raise _fail(
                f"{field_name}[{index}] may contain only one of sha256/source_sha256"
            )
        if supplied_hashes:
            declared_hash = _hash(supplied_hashes[0], f"{field_name}[{index}].sha256")
            if declared_hash != source_sha256:
                raise _fail(
                    f"source screenshot hash mismatch for {source_path}: "
                    f"declared={declared_hash}, actual={source_sha256}"
                )
        parsed.append(
            SourceScreenshot(
                state=state,
                source_reference=Path(raw_path).as_posix(),
                source_path=source_path,
                sha256=source_sha256,
                width_px=width_px,
                height_px=height_px,
                payload=payload,
            )
        )

    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise _fail(f"devices.{device_id} has duplicate lifecycle state labels: {duplicates}")
    missing = sorted(set(_EXPECTED_STATES) - set(labels))
    if missing:
        raise _fail(f"devices.{device_id} is missing lifecycle state labels: {missing}")
    if set(labels) != set(_EXPECTED_STATES):
        # This is defensive after the enum conversion above and gives a stable
        # message if LifecycleState ever gains a member unexpectedly.
        raise _fail(f"devices.{device_id} lifecycle labels are not exactly the required eight states")
    parsed.sort(key=lambda item: _EXPECTED_STATES.index(item.state.value))
    return tuple(parsed)


def _parse_device(device_id: str, raw_value: object, *, source_base_dir: Path) -> ValidatedDevice:
    raw = _require_mapping(raw_value, f"devices.{device_id}")
    _check_unknown(
        raw,
        {
            "device_id",
            "device_label",
            "device_serial_hash",
            "screen_width_px",
            "screen_height_px",
            "calibration_id",
            "arena_px",
            "hand_px",
            "hand_slot_count",
            "reviewed",
            "reviewer",
            "reviewed_at",
            "game_patch",
            "level",
            "game_level",
            "score_threshold",
            "margin_threshold",
            "source_screenshots",
            "screenshots",
        },
        f"devices.{device_id}",
    )
    declared_device_id = _text(raw.get("device_id"), f"devices.{device_id}.device_id")
    if declared_device_id != device_id:
        raise _fail(
            f"devices.{device_id}.device_id must equal its explicit device key {device_id!r}"
        )
    device_label = _text(raw.get("device_label"), f"devices.{device_id}.device_label")
    device_serial_hash = _hash(
        raw.get("device_serial_hash"), f"devices.{device_id}.device_serial_hash"
    )
    screen_width_px = _positive_int(
        raw.get("screen_width_px"), f"devices.{device_id}.screen_width_px"
    )
    screen_height_px = _positive_int(
        raw.get("screen_height_px"), f"devices.{device_id}.screen_height_px"
    )
    calibration_id = _text(raw.get("calibration_id"), f"devices.{device_id}.calibration_id")
    arena_px = _rect(
        raw.get("arena_px"),
        f"devices.{device_id}.arena_px",
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
    )
    hand_px = _rect(
        raw.get("hand_px"),
        f"devices.{device_id}.hand_px",
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
    )
    hand_slot_count = raw.get("hand_slot_count", 4)
    if type(hand_slot_count) is not int or not 1 <= hand_slot_count <= 8:
        raise _fail(f"devices.{device_id}.hand_slot_count must be an integer from 1 through 8")

    if raw.get("reviewed") is not True:
        raise _fail(f"devices.{device_id}.reviewed must be explicitly true")
    reviewer = _text(raw.get("reviewer"), f"devices.{device_id}.reviewer")
    reviewed_at = _reviewed_at(raw.get("reviewed_at"))
    game_patch = _text(raw.get("game_patch"), f"devices.{device_id}.game_patch")
    level_fields = [field for field in ("level", "game_level") if field in raw]
    if len(level_fields) != 1:
        raise _fail(f"devices.{device_id} must contain exactly one of level/game_level")
    level = raw[level_fields[0]]
    if type(level) is not int or level != 11:
        raise _fail(f"devices.{device_id} is not explicitly Level-11")

    score_threshold = _probability(
        raw.get("score_threshold"),
        f"devices.{device_id}.score_threshold",
        _DEFAULT_SCORE_THRESHOLD,
    )
    margin_threshold = _probability(
        raw.get("margin_threshold"),
        f"devices.{device_id}.margin_threshold",
        _DEFAULT_MARGIN_THRESHOLD,
    )
    screenshots = _parse_screenshots(
        raw,
        device_id=device_id,
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        source_base_dir=source_base_dir,
    )
    try:
        calibration = CalibrationArtifact(
            calibration_id=calibration_id,
            device_label=device_label,
            screen_width_px=screen_width_px,
            screen_height_px=screen_height_px,
            arena_px=arena_px,
            hand_px=hand_px,
            device_serial_hash=device_serial_hash,
            hand_slot_count=hand_slot_count,
        )
    except CalibrationError as error:
        raise _fail(f"invalid calibration for device {device_id}: {error}") from error
    return ValidatedDevice(
        device_id=device_id,
        device_label=device_label,
        device_serial_hash=device_serial_hash,
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        calibration_id=calibration_id,
        arena_px=arena_px,
        hand_px=hand_px,
        hand_slot_count=hand_slot_count,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        game_patch=game_patch,
        level=level,
        reviewed=True,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        screenshots=screenshots,
        calibration=calibration,
    )


def _load_input(metadata: Mapping[str, Any] | str | Path, source_base_dir: str | Path | None) -> _LoadedInput:
    if isinstance(metadata, Mapping):
        document = metadata
        base_dir = Path(source_base_dir or Path.cwd()).resolve()
        metadata_sha256 = None
    else:
        metadata_path = Path(metadata).resolve()
        if not metadata_path.is_file():
            raise _fail(f"metadata input does not exist: {metadata_path}")
        try:
            raw_bytes = metadata_path.read_bytes()
            document = json.loads(raw_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise _fail(f"cannot load metadata input {metadata_path}: {error}") from error
        base_dir = Path(source_base_dir).resolve() if source_base_dir is not None else metadata_path.parent
        metadata_sha256 = hash_file(metadata_path)
    if not isinstance(document, Mapping):
        raise _fail("metadata input must be a JSON object")
    return _LoadedInput(document=document, source_base_dir=base_dir, metadata_sha256=metadata_sha256)


def _validate_document(loaded: _LoadedInput) -> tuple[ValidatedDevice, ValidatedDevice]:
    raw = loaded.document
    _check_unknown(raw, {"schema_version", "kind", "devices"}, "input")
    if raw.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise _fail(f"unsupported prerequisite input schema: {raw.get('schema_version')!r}")
    if "kind" in raw and raw["kind"] != "physical_lab_prerequisite_input":
        raise _fail("input.kind must be physical_lab_prerequisite_input when present")
    devices_raw = _require_mapping(raw.get("devices"), "input.devices")
    if set(devices_raw) != {"A", "B"}:
        raise _fail("input.devices must contain exactly the per-device keys A and B")
    devices = tuple(
        _parse_device(device_id, devices_raw[device_id], source_base_dir=loaded.source_base_dir)
        for device_id in ("A", "B")
    )
    return devices[0], devices[1]


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _fail(f"generated prerequisite JSON is not canonical: {error}") from error


def _render_device(device: ValidatedDevice, staging_root: Path) -> RenderedDevice:
    device_root = staging_root / device.device_id
    templates_root = device_root / "templates"
    templates_root.mkdir(parents=True, exist_ok=True)

    template_rows: dict[str, list[dict[str, str]]] = {}
    for screenshot in device.screenshots:
        template_path = templates_root / f"{screenshot.state.value}.png"
        try:
            template_path.write_bytes(screenshot.payload)
        except OSError as error:
            raise _fail(f"cannot stage {template_path}: {error}") from error
        staged_hash = hash_file(template_path)
        if staged_hash != screenshot.sha256:
            raise _fail(f"staged screenshot hash changed for {template_path}")
        template_rows[screenshot.state.value] = [
            {
                "path": f"templates/{screenshot.state.value}.png",
                "sha256": staged_hash,
            }
        ]
    if tuple(template_rows) != _EXPECTED_STATES:
        raise _fail(f"device {device.device_id} did not render all lifecycle states")

    manifest_without_hash: dict[str, Any] = {
        "schema_version": SCREEN_TEMPLATE_SCHEMA_VERSION,
        "device_id": device.device_id,
        "score_threshold": device.score_threshold,
        "margin_threshold": device.margin_threshold,
        "templates": template_rows,
    }
    manifest = dict(manifest_without_hash)
    manifest_hash = canonical_hash(manifest_without_hash)
    manifest["manifest_hash"] = manifest_hash
    manifest_path = device_root / "lifecycle-templates.json"
    manifest_path.write_bytes(_json_bytes(manifest))

    calibration_path = device_root / "calibration.json"
    try:
        # This is an explicitly constructed, bounded artifact.  The
        # provisional/offline constructor is intentionally not used here.
        device.calibration.save(calibration_path)
        restored = CalibrationArtifact.load(calibration_path)
    except (CalibrationError, OSError) as error:
        raise _fail(f"cannot seal calibration for device {device.device_id}: {error}") from error
    if restored.calibration_hash() != device.calibration.calibration_hash():
        raise _fail(f"calibration hash changed while sealing device {device.device_id}")

    # Loading through the existing detector is the final compatibility gate:
    # it checks the exact allowed manifest fields, all eight states, each file
    # hash, relative paths, and canonical manifest_hash.
    try:
        TemplateLifecycleDetector(
            lambda: None,
            manifest_path,
            expected_device_id=device.device_id,
        )
    except (ScreenStateDetectionError, OSError, ImportError) as error:
        raise _fail(
            f"generated lifecycle manifest is not detector-compatible for device "
            f"{device.device_id}: {error}"
        ) from error

    return RenderedDevice(
        device=device,
        calibration_relative_path=f"{device.device_id}/calibration.json",
        calibration_file_sha256=hash_file(calibration_path),
        calibration_hash=restored.calibration_hash(),
        manifest_relative_path=f"{device.device_id}/lifecycle-templates.json",
        manifest_file_sha256=hash_file(manifest_path),
        manifest_hash=manifest_hash,
    )


def _provenance_payload(
    loaded: _LoadedInput,
    rendered: tuple[RenderedDevice, RenderedDevice],
) -> dict[str, Any]:
    devices: dict[str, Any] = {}
    for item in rendered:
        device = item.device
        source_rows = [
            {
                "state": screenshot.state.value,
                "path": screenshot.source_reference,
                "sha256": screenshot.sha256,
                "width_px": screenshot.width_px,
                "height_px": screenshot.height_px,
            }
            for screenshot in device.screenshots
        ]
        devices[device.device_id] = {
            "device_id": device.device_id,
            "device_label": device.device_label,
            "device_serial_hash": device.device_serial_hash,
            "screen_width_px": device.screen_width_px,
            "screen_height_px": device.screen_height_px,
            "reviewed": device.reviewed,
            "reviewer": device.reviewer,
            "reviewed_at": device.reviewed_at,
            "game_patch": device.game_patch,
            "level": device.level,
            "source_screenshots": source_rows,
            "source_hashes": {
                row["state"]: row["sha256"] for row in source_rows
            },
            "calibration": {
                "path": item.calibration_relative_path,
                "file_sha256": item.calibration_file_sha256,
                "calibration_hash": item.calibration_hash,
            },
            "lifecycle_manifest": {
                "path": item.manifest_relative_path,
                "file_sha256": item.manifest_file_sha256,
                "manifest_hash": item.manifest_hash,
                "template_hashes": {
                    row["state"]: row["sha256"] for row in source_rows
                },
            },
        }
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": "physical_lab_prerequisite_review",
        "devices": devices,
    }
    if loaded.metadata_sha256 is not None:
        payload["input_metadata_sha256"] = loaded.metadata_sha256
    payload["provenance_hash"] = canonical_hash(payload)
    return payload


def _copy_staged_tree(staging_root: Path, output_root: Path) -> None:
    if output_root.exists() and not output_root.is_dir():
        raise _fail(f"output path is not a directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    for source in sorted(staging_root.rglob("*")):
        relative = source.relative_to(staging_root)
        destination = output_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(source, destination)
            except OSError as error:
                raise _fail(f"cannot publish generated artifact {destination}: {error}") from error


def _provenance_destination(output_root: Path, provenance_out: str | Path | None) -> Path:
    if provenance_out is None:
        return output_root / "prerequisite-review.json"
    candidate = Path(provenance_out)
    if not candidate.is_absolute():
        candidate = output_root / candidate
    return candidate.resolve()


def build_prerequisites(
    metadata: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    provenance_out: str | Path | None = None,
    source_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate input and publish a deterministic prerequisite artifact set.

    All source files are read, hashed, dimension-checked, and associated with
    both devices before any output is published.  The return value is a small
    summary; the complete review record is the generated sidecar.
    """

    loaded = _load_input(metadata, source_base_dir)
    devices = _validate_document(loaded)
    output_root = Path(output_dir).resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() and not output_root.is_dir():
        raise _fail(f"output path is not a directory: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise _fail(
            f"output directory must be empty so stale prerequisites cannot survive: {output_root}"
        )
    provenance_path = _provenance_destination(output_root, provenance_out)

    # Stage and detector-validate the complete set before touching the final
    # output directory.  TemporaryDirectory cleanup is safe and scoped to the
    # output parent's generated staging directory.
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.prerequisites-",
        dir=output_root.parent,
    ) as staging_name:
        staging_root = Path(staging_name)
        rendered = tuple(_render_device(device, staging_root) for device in devices)
        sidecar = _provenance_payload(loaded, rendered)  # type: ignore[arg-type]
        _copy_staged_tree(staging_root, output_root)

    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        provenance_path.write_bytes(_json_bytes(sidecar))
    except OSError as error:
        raise _fail(f"cannot write provenance sidecar {provenance_path}: {error}") from error

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": "physical_lab_prerequisites",
        "status": "ready",
        "output_dir": str(output_root),
        "provenance_path": str(provenance_path),
        "provenance_hash": sidecar["provenance_hash"],
        "devices": {
            item.device.device_id: {
                "calibration_path": str(output_root / item.calibration_relative_path),
                "calibration_hash": item.calibration_hash,
                "lifecycle_manifest_path": str(output_root / item.manifest_relative_path),
                "manifest_hash": item.manifest_hash,
            }
            for item in rendered
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build fail-closed physical-lab calibrations, lifecycle manifests, "
            "and review provenance from explicit reviewed screenshots"
        )
    )
    parser.add_argument(
        "--input",
        "--metadata",
        dest="metadata_path",
        type=Path,
        required=True,
        help="JSON prerequisite input containing explicit devices A and B",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for generated per-device artifacts and the sidecar",
    )
    parser.add_argument(
        "--provenance-out",
        type=Path,
        help="optional sidecar path; relative paths are below --output-dir",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = build_prerequisites(
            args.metadata_path,
            args.output_dir,
            provenance_out=args.provenance_out,
        )
    except (PrerequisiteBuilderError, OSError, ValueError) as error:
        print(f"rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI users
    raise SystemExit(main())


__all__ = [
    "BuilderError",
    "INPUT_SCHEMA_VERSION",
    "PrerequisiteBuilderError",
    "PROVENANCE_SCHEMA_VERSION",
    "build_prerequisites",
    "main",
]
