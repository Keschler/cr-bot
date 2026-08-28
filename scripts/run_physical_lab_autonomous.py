#!/usr/bin/env python3
"""Run the two-phone physical-lab workflow as one fail-closed command.

Preparation only (safe deck setup, no match):

    outputs/venv/bin/python scripts/run_physical_lab_autonomous.py \
      --serial-a SERIAL_A --serial-b SERIAL_B --prepare-only

Evidence-candidate run (reviewed prerequisites required):

    outputs/venv/bin/python scripts/run_physical_lab_autonomous.py \
      --serial-a SERIAL_A --serial-b SERIAL_B \
      --calibration-a path/to/A.json --calibration-b path/to/B.json \
      --lifecycle-templates-a path/to/A-templates.json \
      --lifecycle-templates-b path/to/B-templates.json \
      --json-out outputs/simulator/fidelity_media/physical_lab/automation.json

The script intentionally keeps raw serials out of JSON artifacts.  It never
force-stops Clash Royale; an active friendly battle is allowed to finish or
the run is rejected with its recordings preserved.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping

# Make the script a genuine one-command entry point when invoked directly
# from any working directory, without requiring the operator to manage
# PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.physical_lab import (
    EvidenceStatus,
    ExperimentSpec,
    PhysicalExtractionError,
    PhysicalLabError,
    extract_physical_run,
    hog_cannon_probe,
)
from simulator.physical_lab.automation import (
    AutonomousPhysicalLab,
    AutonomousPhone,
    AutonomousSessionConfig,
    AutomationError,
    CardVision,
    DEFAULT_FRIENDLY_TARGET_PLAYER_NAME,
    FIXED_DECK_LONG_PRESS_MS,
    FIXED_HOG_CYCLE_DECK,
    UiProfile,
    bind_spec_to_devices,
)
from simulator.physical_lab.calibration import CalibrationArtifact
from simulator.physical_lab.devices import AdbPhoneController, SCRCPY_LEGAL_MAX_TIME_LIMIT_S
from simulator.physical_lab.artifacts import hash_file, physical_output_root
from simulator.physical_lab.campaign import CampaignCase, InteractionCampaign, load_campaign
from simulator.physical_lab.planner import load_plan_line
from simulator.physical_lab.schema import canonical_hash
from simulator.storage import (
    DEFAULT_LOW_WATER_BYTES,
    DEFAULT_MAX_WORKSPACE_BYTES,
    enforce_workspace_budget,
)


PREPARATION_SCHEMA_VERSION = 1
DEFAULT_SUMMARY_PATH = Path("outputs/simulator/fidelity_media/physical_lab/autonomous-summary.json")
DEFAULT_PREPARATION_ROOT = Path("outputs/simulator/fidelity_media/physical_lab")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rooted(root: Path, value: Path | None) -> Path | None:
    if value is None or value.is_absolute():
        return value
    return root / value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous two-phone physical-fidelity lab")
    parser.add_argument("--serial-a", help="ADB serial for controlled player A")
    parser.add_argument("--serial-b", help="ADB serial for controlled opponent B")
    parser.add_argument("--experiment", type=Path, help="experiment JSON or JSONL plan")
    parser.add_argument("--plan-line", type=int, default=1)
    parser.add_argument("--capture-group-id", default="lab-session-autonomous-calibration")
    parser.add_argument("--evidence-split", default="calibration")
    parser.add_argument("--run-id", default="hog-cannon-autonomous-run")
    parser.add_argument(
        "--target-player-name",
        default=DEFAULT_FRIENDLY_TARGET_PLAYER_NAME,
        help="online account that Phone B must select before sending the friendly challenge",
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        help="immutable interaction campaign manifest containing the case to execute",
    )
    parser.add_argument(
        "--case-id",
        help="campaign case ID; required when --campaign contains more than one case",
    )
    parser.add_argument("--repository-root", type=Path, default=_repository_root())
    parser.add_argument(
        "--template-root",
        type=Path,
        default=Path("assets/templates/cr-api-assets/cards-gold"),
        help="reviewed card artwork root used only for UI card identification",
    )
    parser.add_argument("--calibration-a", type=Path)
    parser.add_argument("--calibration-b", type=Path)
    parser.add_argument("--lifecycle-templates-a", type=Path)
    parser.add_argument("--lifecycle-templates-b", type=Path)
    parser.add_argument(
        "--keep-awake",
        action="store_true",
        help="set both explicitly supplied phones to stay awake while powered",
    )
    parser.add_argument(
        "--max-workspace-bytes",
        type=int,
        default=DEFAULT_MAX_WORKSPACE_BYTES,
        help="hard repository cap; values above 200,000,000,000 are rejected",
    )
    parser.add_argument("--low-water-bytes", type=int, default=DEFAULT_LOW_WATER_BYTES)
    parser.add_argument(
        "--reserve-bytes",
        type=int,
        default=2_000_000_000,
        help="space reserved before starting captures (default: 2 GB)",
    )
    parser.add_argument(
        "--evict",
        action="store_true",
        help="evict only already-finalized registered raw media if the guard requires it",
    )
    parser.add_argument(
        "--retention-manifest",
        type=Path,
        default=Path("outputs/simulator/fidelity_media/retention.json"),
    )
    parser.add_argument(
        "--raw-media-root",
        type=Path,
        default=Path("outputs/simulator/fidelity_media"),
    )
    parser.add_argument(
        "--capture-time-limit-s",
        type=int,
        default=360,
        help="maximum local scrcpy recording duration (default: %(default)s seconds)",
    )
    parser.add_argument(
        "--extractor-sample-interval-s",
        type=float,
        default=0.1,
        help="existing cr_bot extractor sample interval used after both captures are sealed",
    )
    parser.add_argument(
        "--extractor-timeout-s",
        type=float,
        default=1800.0,
        help="maximum extractor time per phone after a capture",
    )
    parser.add_argument(
        "--max-collection-swipes",
        type=int,
        default=36,
        help="maximum verified collection scrolls per card before rejecting",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="configure and verify both active decks, then return to lobby without starting a match",
    )
    parser.add_argument(
        "--prepare-side",
        choices=("A", "B", "both"),
        default="both",
        help="limit --prepare-only to one delegated phone operator (default: both)",
    )
    parser.add_argument(
        "--no-prepare",
        action="store_true",
        help="required for a connected run after both delegated phone operators verify preparation",
    )
    parser.add_argument(
        "--fixed-deck-order",
        action="store_true",
        help=(
            "open Testspiel > Solokampf with a long press and enable fixed deck order; "
            "the deck's first four cards become the opening hand"
        ),
    )
    parser.add_argument(
        "--fixed-deck-toggle-point",
        type=str,
        help="reviewed pixel point X,Y for the fixed-deck-order toggle after the long press",
    )
    parser.add_argument(
        "--test-match-start-point",
        type=str,
        help="reviewed pixel point X,Y for starting/hosting the configured Testspiel",
    )
    parser.add_argument(
        "--start-test-match",
        action="store_true",
        help="after enabling fixed deck order, start/host the Testspiel and leave it waiting",
    )
    parser.add_argument(
        "--fixed-deck-long-press-ms",
        type=int,
        default=FIXED_DECK_LONG_PRESS_MS,
        help="hold duration for the Solokampf button (default: %(default)s ms)",
    )
    parser.add_argument(
        "--recover-editor-top",
        action="store_true",
        help="prepare-only: recover a positively recognized scrolled card editor to top Decks",
    )
    parser.add_argument(
        "--resume-remove-slot",
        type=int,
        help="prepare-only: remove the explicitly reviewed selected zero-based deck slot",
    )
    parser.add_argument(
        "--preparation-a",
        type=Path,
        help="sealed one-phone preparation manifest for logical side A",
    )
    parser.add_argument(
        "--preparation-b",
        type=Path,
        help="sealed one-phone preparation manifest for logical side B",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="summary path; one-phone preparation defaults to a side-specific file",
    )
    return parser


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _point(value: str | None, *, option: str) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise AutomationError(f"{option} must be formatted as X,Y")
    try:
        point = (int(parts[0].strip()), int(parts[1].strip()))
    except ValueError as error:
        raise AutomationError(f"{option} must be formatted as integer X,Y") from error
    if min(point) < 0:
        raise AutomationError(f"{option} coordinates must be non-negative")
    return point


def _summary_path(args: argparse.Namespace, repository_root: Path) -> Path:
    if args.json_out is not None:
        return _rooted(repository_root, args.json_out) or args.json_out
    if args.prepare_only and args.prepare_side != "both":
        return repository_root / DEFAULT_PREPARATION_ROOT / f"preparation-{args.prepare_side}.json"
    if args.prepare_only:
        return repository_root / DEFAULT_PREPARATION_ROOT / "preparation-both.json"
    return repository_root / DEFAULT_SUMMARY_PATH


def _validate_preparation(
    path: Path | None,
    *,
    side: str,
    controller: AdbPhoneController,
    expected_deck: tuple[str, ...] = FIXED_HOG_CYCLE_DECK,
) -> dict[str, object]:
    if path is None:
        raise AutomationError(f"--preparation-{side.lower()} is required for --no-prepare")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutomationError(f"cannot load preparation manifest for side {side}: {error}") from error
    if not isinstance(raw, Mapping) or raw.get("kind") != "physical_lab_autonomous_preparation":
        raise AutomationError(f"preparation manifest for side {side} has an unsupported kind")
    if raw.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise AutomationError(f"preparation manifest for side {side} has an unsupported schema")
    manifest_hash = raw.get("manifest_hash")
    unsigned = {key: value for key, value in raw.items() if key != "manifest_hash"}
    if not isinstance(manifest_hash, str) or manifest_hash != canonical_hash(unsigned):
        raise AutomationError(f"preparation manifest for side {side} has an invalid hash")
    if raw.get("status") != "prepared":
        raise AutomationError(f"preparation manifest for side {side} is not marked prepared")
    devices = raw.get("devices")
    device = devices.get(side) if isinstance(devices, Mapping) else None
    if not isinstance(device, Mapping):
        raise AutomationError(f"preparation manifest for side {side} lacks its device record")
    if device.get("serial_hash") != controller.serial_hash:
        raise AutomationError(f"preparation manifest for side {side} does not match the supplied serial")
    decks = raw.get("decks")
    deck = decks.get(side) if isinstance(decks, Mapping) else None
    expected = list(expected_deck)
    if deck != expected:
        raise AutomationError(f"preparation manifest for side {side} does not contain the requested campaign deck")
    return {
        "path": str(path),
        "sha256": hash_file(path),
        "side": side,
        "serial_hash": controller.serial_hash,
        "deck": deck,
    }


def _controllers(
    args: argparse.Namespace,
    sides: tuple[str, ...] = ("A", "B"),
) -> dict[str, AdbPhoneController]:
    serials = {"A": args.serial_a, "B": args.serial_b}
    missing = [side for side in sides if not serials[side]]
    if missing:
        raise AutomationError(
            "missing serial for selected phone operator scope: " + ", ".join(missing)
        )
    selected_serials = [str(serials[side]) for side in sides]
    if len(set(selected_serials)) != len(selected_serials):
        raise AutomationError("selected phone operators must use distinct physical devices")
    if any(any(character.isspace() for character in serial) for serial in selected_serials):
        raise AutomationError("selected phone serials must be non-empty tokens")
    return {
        side: AdbPhoneController(str(serials[side]), device_label=side)
        for side in sides
    }


def _campaign_case(
    args: argparse.Namespace,
    repository_root: Path,
) -> tuple[InteractionCampaign | None, CampaignCase | None]:
    if args.campaign is None:
        if args.case_id is not None:
            raise AutomationError("--case-id requires --campaign")
        return None, None
    campaign_path = _rooted(repository_root, args.campaign)
    assert campaign_path is not None
    campaign = load_campaign(campaign_path)
    if args.case_id is None:
        if len(campaign.cases) != 1:
            raise AutomationError("--case-id is required when --campaign contains multiple cases")
        return campaign, campaign.cases[0]
    try:
        return campaign, next(case for case in campaign.cases if case.case_id == args.case_id)
    except StopIteration as error:
        raise AutomationError(f"campaign has no case {args.case_id!r}") from error


def _prepare_only(
    args: argparse.Namespace,
    repository_root: Path,
    case: CampaignCase | None = None,
) -> dict[str, object]:
    sides = ("A", "B") if args.prepare_side == "both" else (args.prepare_side,)
    controllers = _controllers(args, sides)
    infos = {side: controllers[side].device_info() for side in sides}
    if any(not info.connected for info in infos.values()):
        raise AutomationError("every selected device must be connected for autonomous preparation")
    vision = CardVision(_rooted(repository_root, args.template_root) or args.template_root)
    decks: dict[str, tuple[str, ...]] = {}
    testspiel: dict[str, object] | None = None
    toggle_point = _point(args.fixed_deck_toggle_point, option="--fixed-deck-toggle-point")
    start_point = _point(args.test_match_start_point, option="--test-match-start-point")
    testspiel_side = args.prepare_side if args.prepare_side != "both" else "B"
    target_decks = (
        {side: tuple(case.spec.initial_conditions.decks[side]) for side in ("A", "B")}
        if case is not None
        else {"A": FIXED_HOG_CYCLE_DECK, "B": FIXED_HOG_CYCLE_DECK}
    )
    keep_awake: dict[str, object] = {}
    if args.keep_awake:
        for side in sides:
            keep_awake[side] = controllers[side].set_keep_awake()
    for side in sides:
        profile = UiProfile.for_device(side, infos[side])
        phone = AutonomousPhone(controllers[side], profile, vision)
        if args.recover_editor_top:
            phone.recover_editor_top()
        if args.resume_remove_slot is not None:
            phone.resume_open_remove_panel(args.resume_remove_slot)
        decks[side] = phone.configure_fixed_deck(
            target_deck=target_decks[side],
            max_swipes=args.max_collection_swipes,
        )
        if args.fixed_deck_order and side == testspiel_side:
            testspiel = phone.open_testspiel_solo(
                target_player_name=args.target_player_name,
                fixed_deck_order=True,
                fixed_deck_toggle_point=toggle_point,
                test_match_start_point=start_point if args.start_test_match else None,
                long_press_ms=args.fixed_deck_long_press_ms,
                opening_hand=target_decks[side][:4],
                replacement_order=target_decks[side][4:],
            )
            if not args.start_test_match:
                # The fixed-order options page is not a valid handoff state
                # for the later two-phone coordinator.  Return to the lobby
                # after recording the reviewed option.
                phone.return_to_lobby()
                testspiel["returned_to_lobby"] = True
        else:
            phone.return_to_lobby()
    payload: dict[str, object] = {
        "kind": "physical_lab_autonomous_preparation",
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "status": "prepared",
        "prepared_at_monotonic_us": max(
            int(infos[side].observed_at_monotonic_us or 0) for side in sides
        ),
        "devices": {
            side: {
                "device_id": infos[side].device_id,
                "serial_hash": infos[side].serial_hash,
                "model": infos[side].model,
                "screen_width_px": infos[side].screen_width_px,
                "screen_height_px": infos[side].screen_height_px,
            }
            for side in sides
        },
        "decks": {side: list(deck) for side, deck in decks.items()},
        "fixed_deck": {
            "enabled": bool(args.fixed_deck_order),
            "host_side": "B",
            "opening_hand": {side: list(deck[:4]) for side, deck in decks.items()},
            "replacement_order": {side: list(deck[4:]) for side, deck in decks.items()},
        },
        "campaign": (
            None
            if case is None
            else {
                "case_id": case.case_id,
                "case_hash": case.case_hash(),
                "experiment_hash": case.spec.experiment_hash(),
                "level": case.level,
            }
        ),
        "keep_awake": keep_awake,
        "testspiel": testspiel,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def _load_spec(
    args: argparse.Namespace,
    repository_root: Path,
    case: CampaignCase | None = None,
) -> ExperimentSpec:
    if not args.serial_a or not args.serial_b:
        raise AutomationError("connected runs require both --serial-a and --serial-b")
    if case is not None:
        spec = case.spec
    elif args.experiment is None:
        spec = hog_cannon_probe(
            capture_group_id=args.capture_group_id,
            evidence_split=args.evidence_split,
        )
    else:
        experiment_path = _rooted(repository_root, args.experiment)
        assert experiment_path is not None
        try:
            spec = ExperimentSpec.load(experiment_path)
        except PhysicalLabError:
            spec = load_plan_line(experiment_path, args.plan_line)
    spec = bind_spec_to_devices(spec, args.serial_a, args.serial_b)
    return replace(
        spec,
        metadata={
            **dict(spec.metadata),
            "autonomous_controller": "physical_lab.automation.v2",
            "action_boundary": "reviewed_after_observation_waiter_required",
            "truth_promoted": False,
            **({
                "campaign_case_id": case.case_id,
                "campaign_case_hash": case.case_hash(),
            } if case is not None else {}),
        },
    )


def _budget_kwargs(args: argparse.Namespace, repository_root: Path) -> dict[str, object]:
    if args.max_workspace_bytes > DEFAULT_MAX_WORKSPACE_BYTES:
        raise AutomationError(
            "--max-workspace-bytes cannot exceed the repository hard cap of "
            f"{DEFAULT_MAX_WORKSPACE_BYTES} bytes"
        )
    raw_media_root = _rooted(repository_root, args.raw_media_root)
    retention_manifest = _rooted(repository_root, args.retention_manifest)
    assert raw_media_root is not None and retention_manifest is not None
    return {
        "workspace_root": repository_root,
        "manifest_path": retention_manifest,
        "raw_media_root": raw_media_root,
        "max_bytes": args.max_workspace_bytes,
        "low_water_bytes": args.low_water_bytes,
        "evict": args.evict,
    }


def _check_workspace_budget(
    args: argparse.Namespace,
    repository_root: Path,
    *,
    reserve_bytes: int,
) -> dict[str, object]:
    if reserve_bytes < 0:
        raise AutomationError("--reserve-bytes must be non-negative")
    return enforce_workspace_budget(
        **_budget_kwargs(args, repository_root),
        reserve_bytes=reserve_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    try:
        if args.max_collection_swipes <= 0:
            raise AutomationError("--max-collection-swipes must be positive")
        if args.fixed_deck_long_press_ms <= 0:
            raise AutomationError("--fixed-deck-long-press-ms must be positive")
        if args.capture_time_limit_s < SCRCPY_LEGAL_MAX_TIME_LIMIT_S:
            raise AutomationError(
                f"--capture-time-limit-s must be at least {SCRCPY_LEGAL_MAX_TIME_LIMIT_S} seconds"
            )
        if args.extractor_sample_interval_s <= 0 or args.extractor_timeout_s <= 0:
            raise AutomationError("extractor timing values must be positive")
        if args.start_test_match and not args.fixed_deck_order:
            raise AutomationError("--start-test-match requires --fixed-deck-order")
        if args.test_match_start_point is not None and not args.start_test_match:
            raise AutomationError("--test-match-start-point requires --start-test-match")
        if not args.prepare_only and args.prepare_side != "both":
            raise AutomationError("--prepare-side is only valid with --prepare-only")
        if (args.recover_editor_top or args.resume_remove_slot is not None) and (
            not args.prepare_only or args.prepare_side == "both"
        ):
            raise AutomationError(
                "editor recovery options require --prepare-only and one explicit --prepare-side"
            )
        if not args.prepare_only and not args.no_prepare:
            raise AutomationError(
                "connected runs require independent --prepare-side A and --prepare-side B "
                "operator runs, followed by coordinator --no-prepare"
            )
        if not args.prepare_only and (not args.serial_a or not args.serial_b):
            raise AutomationError("connected runs require both --serial-a and --serial-b")
        if not args.prepare_only and args.fixed_deck_order:
            if args.fixed_deck_toggle_point is None or args.test_match_start_point is None:
                raise AutomationError(
                    "connected fixed-deck runs require --fixed-deck-toggle-point and "
                    "--test-match-start-point"
                )
        if args.prepare_only and args.fixed_deck_order and args.prepare_side == "A":
            raise AutomationError(
                "fixed deck order must be armed on Phone B, the friendly-match host; "
                "run this preparation as --prepare-side B or --prepare-side both"
            )
        if args.max_workspace_bytes <= 0:
            raise AutomationError("--max-workspace-bytes must be positive")
        if args.low_water_bytes < 0 or args.low_water_bytes > args.max_workspace_bytes:
            raise AutomationError("--low-water-bytes must be between zero and the workspace cap")
        if args.reserve_bytes < 0 or args.reserve_bytes > args.max_workspace_bytes:
            raise AutomationError("--reserve-bytes must be between zero and the workspace cap")
        budget_before = _check_workspace_budget(
            args,
            repository_root,
            reserve_bytes=args.reserve_bytes if not args.prepare_only else 0,
        )
        if not budget_before["passed"]:
            payload = {
                "kind": "physical_lab_autonomous_error",
                "status": "blocked_workspace_budget",
                "budget_before": budget_before,
            }
            _write_json(_summary_path(args, repository_root), payload)
            print(json.dumps(payload, sort_keys=True))
            return 2
        _campaign, case = _campaign_case(args, repository_root)
        if args.prepare_only:
            payload = _prepare_only(args, repository_root, case)
            payload["budget_before"] = budget_before
            _write_json(_summary_path(args, repository_root), payload)
            print(json.dumps(payload, sort_keys=True))
            return 0
        required = {
            "--calibration-a": args.calibration_a,
            "--calibration-b": args.calibration_b,
            "--lifecycle-templates-a": args.lifecycle_templates_a,
            "--lifecycle-templates-b": args.lifecycle_templates_b,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise AutomationError(
                "connected autonomous runs require reviewed prerequisites: " + ", ".join(missing)
            )
        spec = _load_spec(args, repository_root, case)
        run_id = args.run_id
        if case is not None and args.run_id == "hog-cannon-autonomous-run":
            run_id = f"{case.case_id}-run"
        controllers = _controllers(args)
        keep_awake: dict[str, object] = {}
        if args.keep_awake:
            keep_awake = {
                side: controllers[side].set_keep_awake() for side in ("A", "B")
            }
        preparations = {
            "A": _validate_preparation(
                _rooted(repository_root, args.preparation_a),
                side="A",
                controller=controllers["A"],
                expected_deck=tuple(spec.initial_conditions.decks["A"]),
            ),
            "B": _validate_preparation(
                _rooted(repository_root, args.preparation_b),
                side="B",
                controller=controllers["B"],
                expected_deck=tuple(spec.initial_conditions.decks["B"]),
            ),
        }
        calibrations = {
            "A": CalibrationArtifact.load(_rooted(repository_root, args.calibration_a)),
            "B": CalibrationArtifact.load(_rooted(repository_root, args.calibration_b)),
        }
        config = AutonomousSessionConfig(
            repository_root=repository_root,
            raw_media_root=(_rooted(repository_root, args.raw_media_root) or args.raw_media_root).resolve(),
            run_id=run_id,
            target_player_name=args.target_player_name,
            max_collection_swipes=args.max_collection_swipes,
            fixed_deck_order=args.fixed_deck_order,
            fixed_deck_toggle_point=_point(
                args.fixed_deck_toggle_point,
                option="--fixed-deck-toggle-point",
            ),
            test_match_start_point=_point(
                args.test_match_start_point,
                option="--test-match-start-point",
            ),
            fixed_deck_long_press_ms=args.fixed_deck_long_press_ms,
            capture_time_limit_s=args.capture_time_limit_s,
            retention_manifest=_rooted(repository_root, args.retention_manifest),
            decks={side: tuple(spec.initial_conditions.decks[side]) for side in ("A", "B")},
        )
        runner = AutonomousPhysicalLab(
            spec=spec,
            controller_a=controllers["A"],
            controller_b=controllers["B"],
            calibration_a=calibrations["A"],
            calibration_b=calibrations["B"],
            lifecycle_manifest_a=_rooted(repository_root, args.lifecycle_templates_a),
            lifecycle_manifest_b=_rooted(repository_root, args.lifecycle_templates_b),
            config=config,
            template_root=_rooted(repository_root, args.template_root) or args.template_root,
        )
        preparation = {
            "status": "independently_prepared",
            "required_operator_scopes": ["A", "B"],
            "manifests": preparations,
        }
        result = runner.run()
        budget_after_capture = _check_workspace_budget(
            args,
            repository_root,
            reserve_bytes=0,
        )
        if not budget_after_capture["passed"]:
            result = replace(
                result,
                status=EvidenceStatus.REJECTED,
                rejection_reasons=result.rejection_reasons
                + ("workspace cap exceeded after physical capture",),
            )
        payload = result.to_dict(include_hash=True)
        payload["preparation"] = preparation
        payload["keep_awake"] = keep_awake
        payload["budget_before"] = budget_before
        payload["budget_after_capture"] = budget_after_capture
        artifact_root = physical_output_root(
            repository_root=repository_root,
            run_id=result.run_id,
        )
        run_manifest_path = artifact_root / "run.json"
        handoff_path = artifact_root / "observation-handoff.json"
        payload["run_manifest_path"] = str(run_manifest_path)
        payload["observation_handoff_path"] = str(handoff_path)
        if run_manifest_path.is_file():
            payload["run_manifest_sha256"] = hash_file(run_manifest_path)
        if handoff_path.is_file():
            payload["observation_handoff_sha256"] = hash_file(handoff_path)
        extraction: dict[str, object] | None = None
        extraction_error: str | None = None
        capture_rows = result.captures
        if (
            result.status is not EvidenceStatus.REJECTED
            and set(capture_rows) == {"A", "B"}
            and all(
                capture.stream_verified
                and capture.status == "complete"
                and capture.media_path is not None
                for capture in capture_rows.values()
            )
        ):
            try:
                extraction = extract_physical_run(
                    run_manifest_path,
                    repository_root=repository_root,
                    sample_interval_s=args.extractor_sample_interval_s,
                    extractor_timeout_s=args.extractor_timeout_s,
                )
            except (PhysicalExtractionError, PhysicalLabError, OSError, ValueError) as error:
                extraction_error = str(error)
        payload["extraction"] = extraction
        if extraction_error is not None:
            payload["extraction_error"] = extraction_error
        # The summary is a convenience index.  Its hash is intentionally
        # separate from the sealed run_hash above.
        payload["summary_hash"] = canonical_hash(payload)
        output_path = _summary_path(args, repository_root)
        _write_json(output_path, payload)
        print(json.dumps({"status": result.status.value, "json_out": str(output_path)}))
        return 0 if result.status is EvidenceStatus.CANDIDATE_ONLY and budget_after_capture["passed"] else 2
    except (AutomationError, PhysicalLabError, OSError, ValueError, json.JSONDecodeError) as error:
        payload = {"kind": "physical_lab_autonomous_error", "status": "rejected", "error": str(error)}
        _write_json(_summary_path(args, repository_root), payload)
        print(json.dumps(payload, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
