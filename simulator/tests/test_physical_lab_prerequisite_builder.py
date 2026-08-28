from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import zlib

import pytest

from scripts.build_physical_lab_prerequisites import BuilderError, build_prerequisites, main
from simulator.physical_lab import CalibrationArtifact, LifecycleState, TemplateLifecycleDetector
from simulator.physical_lab.artifacts import hash_file


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _synthetic_png(width: int, height: int, seed: int) -> bytes:
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend(((x * 17 + seed) % 256, (y * 29 + seed * 3) % 256, (x + y + seed) % 256))
        rows.append(bytes(row))
    raw = b"".join(rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(
        b"IDAT", zlib.compress(raw, level=9)
    ) + _png_chunk(b"IEND", b"")


def _device_metadata(tmp_path: Path, device_id: str, *, width: int = 32, height: int = 24) -> dict[str, object]:
    source_root = tmp_path / "sources" / device_id
    source_root.mkdir(parents=True)
    screenshots = []
    for index, state in enumerate(LifecycleState):
        path = source_root / f"{state.value}.png"
        path.write_bytes(_synthetic_png(width, height, seed=index + (1 if device_id == "A" else 20)))
        screenshots.append({"state": state.value, "path": str(path)})
    return {
        "device_id": device_id,
        "device_label": f"phone-{device_id.lower()}",
        "device_serial_hash": "sha256:" + ("a" if device_id == "A" else "b") * 64,
        "screen_width_px": width,
        "screen_height_px": height,
        "calibration_id": f"{device_id.lower()}-calibration-v1",
        "arena_px": [1, 2, width - 2, height - 10],
        "hand_px": [2, height - 7, width - 4, 5],
        "reviewed": True,
        "reviewer": "physical-lab-reviewer",
        "reviewed_at": "2026-08-21T12:00:00Z",
        "game_patch": "2026-08-21-test-patch",
        "level": 11,
        "source_screenshots": screenshots,
    }


def _input_document(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "physical_lab_prerequisite_input",
        "devices": {
            "A": _device_metadata(tmp_path, "A"),
            "B": _device_metadata(tmp_path, "B"),
        },
    }


def _write_input(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "prerequisites.json"
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_builder_is_deterministic_and_emits_detector_compatible_artifacts(tmp_path: Path) -> None:
    document = _input_document(tmp_path)
    input_path = _write_input(tmp_path, document)
    first_root = tmp_path / "out-first"
    second_root = tmp_path / "out-second"

    first = build_prerequisites(input_path, first_root)
    second = build_prerequisites(input_path, second_root)

    first_files = _file_bytes(first_root)
    second_files = _file_bytes(second_root)
    assert first_files == second_files
    assert first["provenance_hash"] == second["provenance_hash"]

    calibration = CalibrationArtifact.load(first_root / "A" / "calibration.json")
    assert calibration.arena_px == (1.0, 2.0, 30.0, 14.0)
    assert calibration.hand_px == (2.0, 17.0, 28.0, 5.0)
    assert calibration.device_serial_hash == "sha256:" + "a" * 64

    detector = TemplateLifecycleDetector(
        lambda: None,
        first_root / "A" / "lifecycle-templates.json",
        expected_device_id="A",
    )
    assert detector.provenance()["template_count"] == len(LifecycleState)
    manifest = json.loads((first_root / "A" / "lifecycle-templates.json").read_text())
    assert set(manifest["templates"]) == {state.value for state in LifecycleState}

    sidecar = json.loads((first_root / "prerequisite-review.json").read_text())
    assert sidecar["devices"]["A"]["level"] == 11
    assert len(sidecar["devices"]["A"]["source_screenshots"]) == len(LifecycleState)
    assert sidecar["devices"]["A"]["calibration"]["calibration_hash"] == calibration.calibration_hash()
    assert sidecar["devices"]["A"]["lifecycle_manifest"]["manifest_hash"] == manifest["manifest_hash"]
    assert sidecar["devices"]["A"]["source_hashes"]["battle"].startswith("sha256:")
    assert hash_file(first_root / "A" / "lifecycle-templates.json") == sidecar["devices"]["A"]["lifecycle_manifest"]["file_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda doc: doc["devices"]["A"]["source_screenshots"].pop(), "exactly 8 entries"),
        (
            lambda doc: doc["devices"]["A"]["source_screenshots"][0].update({"state": "lobby"}),
            "duplicate lifecycle",
        ),
        (
            lambda doc: doc["devices"]["A"]["source_screenshots"][0].update({"state": "not-a-state"}),
            "unknown lifecycle",
        ),
    ],
)
def test_builder_rejects_missing_duplicate_and_unknown_states(
    tmp_path: Path, mutation, message: str
) -> None:
    document = _input_document(tmp_path)
    mutation(document)
    with pytest.raises(BuilderError, match=message):
        build_prerequisites(document, tmp_path / "rejected", source_base_dir=tmp_path)
    assert not (tmp_path / "rejected").exists()


def test_builder_rejects_png_dimension_mismatch_without_output(tmp_path: Path) -> None:
    document = _input_document(tmp_path)
    document["devices"]["B"]["screen_width_px"] = 31
    with pytest.raises(BuilderError, match="dimensions .* do not match"):
        build_prerequisites(document, tmp_path / "rejected", source_base_dir=tmp_path)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize(
    "field_mutation",
    [
        lambda device: device.pop("reviewer"),
        lambda device: device.pop("reviewed_at"),
        lambda device: device.pop("game_patch"),
        lambda device: device.update({"level": 10}),
        lambda device: device.update({"reviewed": False}),
    ],
)
def test_builder_rejects_missing_review_metadata_or_non_level_11(tmp_path: Path, field_mutation) -> None:
    document = _input_document(tmp_path)
    field_mutation(document["devices"]["A"])
    with pytest.raises(BuilderError):
        build_prerequisites(document, tmp_path / "rejected", source_base_dir=tmp_path)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize(
    "field_mutation",
    [
        lambda device: device.update({"device_serial_hash": "sha256:bad"}),
        lambda device: device.update({"arena_px": [-1, 2, 10, 10]}),
        lambda device: device.update({"hand_px": [0, 0, 100, 100]}),
        lambda device: device["source_screenshots"][0].update({"sha256": "sha256:bad"}),
    ],
)
def test_builder_rejects_invalid_hashes_and_coordinates(tmp_path: Path, field_mutation) -> None:
    document = _input_document(tmp_path)
    field_mutation(document["devices"]["A"])
    with pytest.raises(BuilderError):
        build_prerequisites(document, tmp_path / "rejected", source_base_dir=tmp_path)
    assert not (tmp_path / "rejected").exists()


def test_builder_requires_both_explicit_device_keys(tmp_path: Path) -> None:
    document = _input_document(tmp_path)
    document["devices"] = {"A": document["devices"]["A"], "C": document["devices"]["B"]}
    with pytest.raises(BuilderError, match="exactly.*A and B"):
        build_prerequisites(document, tmp_path / "rejected", source_base_dir=tmp_path)


def test_cli_entrypoint_consumes_metadata_and_reports_ready(tmp_path: Path, capsys) -> None:
    input_path = _write_input(tmp_path, _input_document(tmp_path))
    output_root = tmp_path / "cli-output"
    assert main(["--input", str(input_path), "--output-dir", str(output_root)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "ready"
    assert (output_root / "A" / "calibration.json").is_file()
    assert (output_root / "B" / "lifecycle-templates.json").is_file()


def test_builder_rejects_nonempty_output_so_stale_artifacts_cannot_survive(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "existing-output"
    output_root.mkdir()
    stale = output_root / "stale-template.png"
    stale.write_bytes(b"must remain untouched")

    with pytest.raises(BuilderError, match="must be empty"):
        build_prerequisites(
            _input_document(tmp_path),
            output_root,
            source_base_dir=tmp_path,
        )

    assert stale.read_bytes() == b"must remain untouched"
    assert tuple(output_root.iterdir()) == (stale,)
