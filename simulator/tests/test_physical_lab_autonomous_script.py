from __future__ import annotations

import argparse

import pytest

import json
from pathlib import Path

from scripts.run_physical_lab_autonomous import (
    _controllers,
    _point,
    _summary_path,
    _validate_preparation,
)
from simulator.physical_lab.automation import AutomationError, FIXED_HOG_CYCLE_DECK
from simulator.physical_lab.devices import AdbPhoneController, sha256_bytes
from simulator.physical_lab.schema import canonical_hash


def test_single_phone_preparation_constructs_only_the_owned_controller() -> None:
    controllers = _controllers(
        argparse.Namespace(serial_a=None, serial_b="phone-b"),
        ("B",),
    )

    assert set(controllers) == {"B"}
    assert controllers["B"].serial == "phone-b"


def test_single_phone_preparation_rejects_a_missing_owned_serial() -> None:
    with pytest.raises(AutomationError, match="selected phone operator scope: B"):
        _controllers(
            argparse.Namespace(serial_a="phone-a", serial_b=None),
            ("B",),
        )


def test_single_phone_preparation_is_symmetric_for_side_a() -> None:
    controllers = _controllers(
        argparse.Namespace(serial_a="phone-a", serial_b=None),
        ("A",),
    )

    assert set(controllers) == {"A"}
    assert controllers["A"].serial == "phone-a"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("700,1600", (700, 1600)), (" 0, 12 ", (0, 12))],
)
def test_reviewed_ui_point_parser(raw: str, expected: tuple[int, int]) -> None:
    assert _point(raw, option="--fixed-deck-toggle-point") == expected


@pytest.mark.parametrize("raw", ["700", "x,1600", "-1,2"])
def test_reviewed_ui_point_parser_rejects_unsafe_values(raw: str) -> None:
    with pytest.raises(AutomationError):
        _point(raw, option="--fixed-deck-toggle-point")


def test_preparation_output_defaults_are_side_specific() -> None:
    root = Path("/repo")
    args = argparse.Namespace(
        json_out=None,
        prepare_only=True,
        prepare_side="A",
    )

    assert _summary_path(args, root) == root / "outputs/simulator/fidelity_media/physical_lab/preparation-A.json"


def test_preparation_manifest_is_bound_to_serial_and_fixed_deck(tmp_path: Path) -> None:
    serial = "phone-a"
    manifest = tmp_path / "preparation-A.json"
    deck = [
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
        "fireball",
        "log",
    ]
    unsigned = {
        "kind": "physical_lab_autonomous_preparation",
        "schema_version": 1,
        "status": "prepared",
        "devices": {
            "A": {
                "serial_hash": sha256_bytes(serial.encode("utf-8")),
            }
        },
        "decks": {"A": deck},
    }
    manifest.write_text(
        json.dumps({**unsigned, "manifest_hash": canonical_hash(unsigned)}),
        encoding="utf-8",
    )

    result = _validate_preparation(
        manifest,
        side="A",
        controller=AdbPhoneController(serial, device_label="A"),
    )

    assert result["side"] == "A"
    assert result["serial_hash"] == sha256_bytes(serial.encode("utf-8"))
    assert result["deck"] == deck


def test_preparation_manifest_rejects_tampering(tmp_path: Path) -> None:
    serial = "phone-a"
    manifest = tmp_path / "preparation-A.json"
    unsigned = {
        "kind": "physical_lab_autonomous_preparation",
        "schema_version": 1,
        "status": "prepared",
        "devices": {"A": {"serial_hash": sha256_bytes(serial.encode("utf-8"))}},
        "decks": {"A": [*FIXED_HOG_CYCLE_DECK]},
    }
    manifest.write_text(
        json.dumps({**unsigned, "manifest_hash": canonical_hash(unsigned | {"status": "tampered"})}),
        encoding="utf-8",
    )

    with pytest.raises(AutomationError, match="invalid hash"):
        _validate_preparation(
            manifest,
            side="A",
            controller=AdbPhoneController(serial, device_label="A"),
        )


def test_controllers_reject_duplicate_serials_before_device_access() -> None:
    with pytest.raises(AutomationError, match="distinct physical devices"):
        _controllers(
            argparse.Namespace(serial_a="same-phone", serial_b="same-phone"),
            ("A", "B"),
        )
