"""Command-line entry points for the software-only and physical lab paths."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .automation import (
    AutonomousPhone,
    CardVision,
    DEFAULT_FRIENDLY_TARGET_PLAYER_NAME,
    FIXED_DECK_LONG_PRESS_MS,
    FIXED_HOG_CYCLE_DECK,
    FIXED_HOG_CYCLE_OPENING_HAND,
    FIXED_HOG_CYCLE_REPLACEMENT_ORDER,
    UiProfile,
)
from .calibration import CalibrationArtifact
from .artifacts import finalize_retention_records
from .comparison import compare_observation_to_replay
from .cache import ReplayCacheError, seal_replay_cache
from .campaign import (
    DEFAULT_CAMPAIGN_ID,
    DEFAULT_CAMPAIGN_ROOT,
    build_default_campaign,
    evaluate_campaign,
    load_campaign,
    materialize_campaign,
    write_campaign_evaluation,
)
from .devices import AdbPhoneController, AdbScreenCapture, LogicalPhone, sha256_bytes
from .fidelity_bridge import build_fidelity_corpus_payload, write_fidelity_corpus_payload
from .extraction import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_JOB_TIMEOUT_S,
    DEFAULT_SAMPLE_INTERVAL_S,
    extract_physical_run,
)
from .evaluation import evaluate_stored_cases, write_stored_evaluation
from .lifecycle import LIFECYCLE_PATH, LifecycleMachine, LifecyclePolicy, ScriptedLifecycleDetector
from .observation import (
    ObservationManifest,
    RejectedObservation,
    ingest_for_experiment,
)
from .planner import (
    load_plan_line,
    load_questions,
    hog_cannon_probe,
    plan_from_questions,
    plan_from_readiness,
    write_plan,
)
from .replay import run_simulator_replay
from .runner import PhysicalLabRunner, offline_runner, write_run_artifacts
from .schema import (
    EvidenceSplit,
    EvidenceStatus,
    ExperimentSpec,
    PhysicalLabError,
    canonical_hash,
)
from .split import SplitLock
from .screen_state import TemplateLifecycleDetector


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_questions_path() -> Path:
    return Path(__file__).resolve().parent.parent / "unknown_behaviors.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physical-lab", description="Physical differential-testing lab harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="emit canonical experiment JSONL before capture inspection")
    plan.add_argument("--unknown-behaviors", type=Path, default=_default_questions_path())
    plan.add_argument("--readiness", type=Path, help="optional readiness report; held-out rows are ignored")
    plan.add_argument("--json-out", type=Path, required=True)
    plan.add_argument("--capture-group-id", default="lab-session-offline-calibration")
    plan.add_argument("--evidence-split", choices=[item.value for item in EvidenceSplit], default="calibration")
    plan.add_argument("--limit", type=int, default=10)
    plan.add_argument("--hog-cannon-only", action="store_true")
    plan.add_argument(
        "--serial-a",
        help="bind a hashed physical device identity to side A before capture",
    )
    plan.add_argument(
        "--serial-b",
        help="bind a hashed physical device identity to side B before capture",
    )

    prepare = subparsers.add_parser(
        "prepare",
        help="configure one connected phone's fixed Hog deck and optional Testspiel order",
    )
    prepare.add_argument("--serial", required=True, help="ADB serial for the one phone being prepared")
    prepare.add_argument("--side", choices=("A", "B"), default="A", help="logical side recorded in the manifest")
    prepare.add_argument("--repository-root", type=Path, default=_default_repository_root())
    prepare.add_argument(
        "--template-root",
        type=Path,
        default=Path("assets/templates/cr-api-assets/cards-gold"),
        help="reviewed card artwork root used for deck verification",
    )
    prepare.add_argument("--max-collection-swipes", type=int, default=36)
    prepare.add_argument(
        "--fixed-deck-order",
        action="store_true",
        help="long-press Testspiel/Solokampf and enable the fixed deck-order option",
    )
    prepare.add_argument(
        "--fixed-deck-toggle-point",
        help="reviewed pixel point X,Y for the fixed-deck-order toggle",
    )
    prepare.add_argument(
        "--test-match-start-point",
        help="reviewed pixel point X,Y for the Testspiel start/host control",
    )
    prepare.add_argument(
        "--start-test-match",
        action="store_true",
        help="start/host the configured Testspiel after enabling fixed order",
    )
    prepare.add_argument(
        "--fixed-deck-long-press-ms",
        type=int,
        default=FIXED_DECK_LONG_PRESS_MS,
    )
    prepare.add_argument("--json-out", type=Path)
    prepare.add_argument(
        "--target-player-name",
        default=DEFAULT_FRIENDLY_TARGET_PLAYER_NAME,
        help="online account that the friendly challenge must target",
    )

    run = subparsers.add_parser("run", help="run a physical experiment or the offline fake harness")
    run.add_argument("--experiment", type=Path)
    run.add_argument("--plan-line", type=int, default=1)
    run.add_argument("--mode", choices=("offline", "adb"), default="offline")
    run.add_argument("--serial-a")
    run.add_argument("--serial-b")
    run.add_argument("--calibration-a", type=Path)
    run.add_argument("--calibration-b", type=Path)
    run.add_argument(
        "--lifecycle-templates-a",
        type=Path,
        help="sealed reviewed screen-state template manifest for physical device A",
    )
    run.add_argument(
        "--lifecycle-templates-b",
        type=Path,
        help="sealed reviewed screen-state template manifest for physical device B",
    )
    run.add_argument("--repository-root", type=Path, default=_default_repository_root())
    run.add_argument(
        "--retention-manifest",
        type=Path,
        default=Path("outputs/simulator/fidelity_media/retention.json"),
    )
    run.add_argument(
        "--raw-media-root",
        type=Path,
        default=Path("outputs/simulator/fidelity_media"),
        help="workspace-relative root containing public and physical disposable media",
    )
    run.add_argument("--max-workspace-bytes", type=int, default=200_000_000_000)
    run.add_argument("--low-water-bytes", type=int, default=190_000_000_000)
    run.add_argument(
        "--reserve-bytes",
        type=int,
        default=0,
        help="space to reserve before capture; set this to the expected recording size",
    )
    run.add_argument(
        "--evict",
        action="store_true",
        help="evict only previously finalized registered raw media when needed",
    )
    run.add_argument("--split-lock", type=Path)
    run.add_argument("--run-id")
    run.add_argument("--json-out", type=Path)

    ingest = subparsers.add_parser("ingest", help="ingest detector rows into a sealed observation manifest")
    ingest.add_argument("raw", type=Path, help="JSON detector rows or an already normalized manifest")
    ingest.add_argument("--run", type=Path, required=True, help="sealed run.json containing provenance")
    ingest.add_argument("--json-out", type=Path, required=True)
    ingest.add_argument("--confidence-threshold", type=float, default=0.98)
    ingest.add_argument("--direct-timing", action="store_true")
    ingest.add_argument("--replay-cache", type=Path)
    ingest.add_argument(
        "--retention-manifest",
        type=Path,
        help="finalize registered physical raw captures after successful ingest",
    )
    ingest.add_argument(
        "--repository-root",
        type=Path,
        default=_default_repository_root(),
    )
    ingest.add_argument(
        "--audit-path",
        type=Path,
        action="append",
        default=[],
        help="additional compact audit/scenario artifact to retain with the capture",
    )

    compare = subparsers.add_parser("compare", help="compare an observation manifest with a logical simulator replay")
    compare.add_argument("observation", type=Path)
    compare.add_argument("--experiment", type=Path)
    compare.add_argument("--run", type=Path)
    compare.add_argument("--json-out", type=Path, required=True)
    compare.add_argument("--position-tolerance-mtile", type=int, default=200)
    compare.add_argument("--timing-tolerance-us", type=int, default=10_000)

    fidelity = subparsers.add_parser(
        "fidelity",
        help="compile an admitted physical observation into the standard fidelity report",
    )
    fidelity.add_argument("observation", type=Path)
    fidelity.add_argument("--run", type=Path, required=True, help="sealed physical run.json")
    fidelity.add_argument(
        "--replay-cache",
        type=Path,
        required=True,
        help="recognized replay cache used for physical observation admission",
    )
    fidelity.add_argument("--corpus-out", type=Path)
    fidelity.add_argument("--json-out", type=Path, required=True)
    fidelity.add_argument("--position-tolerance-mtile", type=int, default=250)
    fidelity.add_argument("--min-observations", type=int, default=1)
    fidelity.add_argument("--min-agreement-rate", type=float)
    fidelity.add_argument("--require-mechanic", action="append", default=[])

    extract_run = subparsers.add_parser(
        "extract-run",
        help="run the existing cr_bot extractor on both physical capture streams",
    )
    extract_run.add_argument("--run", type=Path, required=True, help="sealed physical_lab_run run.json")
    extract_run.add_argument("--repository-root", type=Path, default=_default_repository_root())
    extract_run.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_S)
    extract_run.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    extract_run.add_argument("--extractor-timeout", type=float, default=DEFAULT_JOB_TIMEOUT_S)
    extract_run.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse only replay caches recognized by cr_bot.replay.cache",
    )
    extract_run.add_argument("--json-out", type=Path, required=True)

    status = subparsers.add_parser("status", help="probe ADB devices without running an experiment")
    status.add_argument("--serial-a")
    status.add_argument("--serial-b")
    status.add_argument("--json-out", type=Path)

    keep_awake = subparsers.add_parser(
        "keep-awake",
        help="set the explicitly supplied A/B phones to stay awake while powered",
    )
    keep_awake.add_argument("--serial-a", required=True)
    keep_awake.add_argument("--serial-b", required=True)
    keep_awake.add_argument("--json-out", type=Path)

    campaign_plan = subparsers.add_parser(
        "campaign-plan",
        help="create the immutable simple-to-complex interaction campaign",
    )
    campaign_plan.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    campaign_plan.add_argument("--evidence-split", default="calibration")
    campaign_plan.add_argument("--output-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    campaign_plan.add_argument("--json-out", type=Path)

    campaign_evaluate = subparsers.add_parser(
        "campaign-evaluate",
        help="re-evaluate all stored campaign corpora with the current simulator",
    )
    campaign_evaluate.add_argument("--campaign", type=Path, required=True)
    campaign_evaluate.add_argument("--results-root", type=Path, required=True)
    campaign_evaluate.add_argument("--split")
    campaign_evaluate.add_argument("--min-observations", type=int, default=1)
    campaign_evaluate.add_argument("--min-agreement-rate", type=float)
    campaign_evaluate.add_argument("--json-out", type=Path, required=True)

    stored_evaluate = subparsers.add_parser(
        "evaluate-stored",
        help="re-run the current simulator against every stored extracted physical case",
    )
    stored_evaluate.add_argument(
        "--cases-root",
        type=Path,
        default=Path("outputs/simulator/fidelity_media/physical_lab"),
    )
    stored_evaluate.add_argument("--repository-root", type=Path, default=_default_repository_root())
    stored_evaluate.add_argument("--json-out", type=Path, required=True)
    return parser


def _parse_ui_point(value: str | None, *, option: str) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise PhysicalLabError(f"{option} must be formatted as X,Y")
    try:
        point = (int(parts[0].strip()), int(parts[1].strip()))
    except ValueError as error:
        raise PhysicalLabError(f"{option} must be formatted as integer X,Y") from error
    if min(point) < 0:
        raise PhysicalLabError(f"{option} coordinates must be non-negative")
    return point


def _prepare_command(args: argparse.Namespace) -> int:
    if not args.serial or any(character.isspace() for character in args.serial):
        raise PhysicalLabError("--serial must be a non-empty token")
    if args.max_collection_swipes <= 0:
        raise PhysicalLabError("--max-collection-swipes must be positive")
    if args.fixed_deck_long_press_ms <= 0:
        raise PhysicalLabError("--fixed-deck-long-press-ms must be positive")

    toggle_point = _parse_ui_point(
        args.fixed_deck_toggle_point,
        option="--fixed-deck-toggle-point",
    )
    start_point = _parse_ui_point(
        args.test_match_start_point,
        option="--test-match-start-point",
    )
    if args.fixed_deck_order and toggle_point is None:
        raise PhysicalLabError(
            "--fixed-deck-order requires the reviewed --fixed-deck-toggle-point"
        )
    if not args.fixed_deck_order and toggle_point is not None:
        raise PhysicalLabError("--fixed-deck-toggle-point requires --fixed-deck-order")
    if args.start_test_match and not args.fixed_deck_order:
        raise PhysicalLabError("--start-test-match requires --fixed-deck-order")
    if args.start_test_match and start_point is None:
        raise PhysicalLabError(
            "--start-test-match requires the reviewed --test-match-start-point"
        )
    if not args.start_test_match and start_point is not None:
        raise PhysicalLabError("--test-match-start-point requires --start-test-match")

    repository_root = args.repository_root.resolve()
    controller = AdbPhoneController(args.serial, device_label=args.side)
    info = controller.device_info()
    if not info.connected:
        raise PhysicalLabError(f"device {args.side} is not connected")
    profile = UiProfile.for_device(args.side, info)
    vision = CardVision(
        args.template_root
        if args.template_root.is_absolute()
        else repository_root / args.template_root
    )
    phone = AutonomousPhone(controller, profile, vision)
    deck = phone.configure_fixed_deck(max_swipes=args.max_collection_swipes)
    if deck != FIXED_HOG_CYCLE_DECK:
        raise PhysicalLabError(f"prepared deck does not match the fixed Hog cycle: {deck!r}")

    testspiel: dict[str, object] | None = None
    if args.fixed_deck_order:
        testspiel = phone.open_testspiel_solo(
            target_player_name=args.target_player_name,
            fixed_deck_order=True,
            fixed_deck_toggle_point=toggle_point,
            test_match_start_point=start_point if args.start_test_match else None,
            long_press_ms=args.fixed_deck_long_press_ms,
        )
        if not args.start_test_match:
            # Enabling the option without hosting a match leaves the custom
            # Testspiel surface open.  A later two-phone coordinator requires
            # a positively recognized lobby, so close this staged workflow at
            # its safe handoff boundary.
            phone.return_to_lobby()
            testspiel["returned_to_lobby"] = True
    else:
        phone.return_to_lobby()

    payload: dict[str, object] = {
        "kind": "physical_lab_autonomous_preparation",
        "schema_version": 1,
        "status": "prepared",
        "prepared_at_monotonic_us": int(info.observed_at_monotonic_us or 0),
        "devices": {args.side: info.to_dict()},
        "decks": {args.side: list(deck)},
        "fixed_deck": {
            "enabled": bool(args.fixed_deck_order),
            "opening_hand": list(FIXED_HOG_CYCLE_OPENING_HAND),
            "replacement_order": list(FIXED_HOG_CYCLE_REPLACEMENT_ORDER),
        },
        "testspiel": testspiel,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    output_path = args.json_out
    if output_path is not None and not output_path.is_absolute():
        output_path = repository_root / output_path
    _write_json(output_path, payload)
    if output_path is not None:
        _write_json(None, {"status": "prepared", "json_out": str(output_path)})
    return 0


def _write_json(path: Path | None, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path is None:
        print(encoded, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _extract_run_command(args: argparse.Namespace) -> int:
    repository_root = args.repository_root.resolve()
    run_path = args.run if args.run.is_absolute() else repository_root / args.run
    result = extract_physical_run(
        run_path,
        repository_root=repository_root,
        sample_interval_s=args.sample_interval,
        confidence_threshold=args.confidence_threshold,
        extractor_timeout_s=args.extractor_timeout,
        run_extractor=not args.reuse_existing,
    )
    output_path = args.json_out if args.json_out.is_absolute() else repository_root / args.json_out
    _write_json(output_path, result)
    _write_json(None, result)
    return 0


def _bound_device_specs(
    serial_a: str | None,
    serial_b: str | None,
) -> dict[str, object] | None:
    """Build canonical device records without placing raw serials in a spec."""

    if (serial_a is None) != (serial_b is None):
        raise PhysicalLabError("device binding requires both --serial-a and --serial-b")
    if serial_a is None or serial_b is None:
        return None
    if (
        any(character.isspace() for character in serial_a)
        or any(character.isspace() for character in serial_b)
    ):
        raise PhysicalLabError("device serials must be non-empty tokens")
    if serial_a == serial_b:
        raise PhysicalLabError("physical sides A and B must use distinct devices")
    from .schema import DeviceSpec

    return {
        "A": DeviceSpec(sha256_bytes(serial_a.encode("utf-8")), "player", "A"),
        "B": DeviceSpec(sha256_bytes(serial_b.encode("utf-8")), "opponent", "B"),
    }


def _bind_planned_devices(item: object, devices: Mapping[str, object]) -> object:
    if isinstance(item, ExperimentSpec):
        return replace(item, devices=devices)
    spec = getattr(item, "spec", None)
    if isinstance(spec, ExperimentSpec):
        return replace(item, spec=replace(spec, devices=devices))
    raise PhysicalLabError("planner returned an unsupported experiment record")


def _load_experiment(path: Path | None, line: int) -> ExperimentSpec:
    if path is None:
        return hog_cannon_probe()
    try:
        return ExperimentSpec.load(path)
    except PhysicalLabError as direct_error:
        try:
            return load_plan_line(path, line)
        except PhysicalLabError as line_error:
            raise PhysicalLabError(f"cannot load experiment or plan line: {direct_error}; {line_error}") from line_error


def _adb_runner(args: argparse.Namespace, spec: ExperimentSpec):
    if not args.serial_a or not args.serial_b:
        raise PhysicalLabError("ADB mode requires --serial-a and --serial-b")
    if args.serial_a == args.serial_b:
        raise PhysicalLabError("physical sides A and B must use distinct devices")
    missing_calibrations = [
        name
        for name, value in (
            ("--calibration-a", args.calibration_a),
            ("--calibration-b", args.calibration_b),
        )
        if value is None
    ]
    if missing_calibrations:
        raise PhysicalLabError(
            "ADB mode requires explicit reviewed calibrations: "
            + ", ".join(missing_calibrations)
        )
    missing_lifecycle = [
        name
        for name, value in (
            ("--lifecycle-templates-a", args.lifecycle_templates_a),
            ("--lifecycle-templates-b", args.lifecycle_templates_b),
        )
        if value is None
    ]
    if missing_lifecycle:
        raise PhysicalLabError(
            "ADB mode requires explicit reviewed lifecycle manifests: "
            + ", ".join(missing_lifecycle)
        )
    controllers = {
        "A": AdbPhoneController(args.serial_a, device_label="A"),
        "B": AdbPhoneController(args.serial_b, device_label="B"),
    }
    calibrations = {
        "A": CalibrationArtifact.load(args.calibration_a),
        "B": CalibrationArtifact.load(args.calibration_b),
    }
    phones = {
        side: LogicalPhone(controllers[side], calibrations[side])
        for side in ("A", "B")
    }
    run_id = args.run_id or f"{spec.experiment_id}-run"
    capture_root = args.repository_root / "outputs/simulator/fidelity_media/physical_lab" / run_id / "raw"
    captures = {
        side: AdbScreenCapture(
            controllers[side],
            capture_root / f"{side}.mp4",
        )
        for side in ("A", "B")
    }
    template_paths = (args.lifecycle_templates_a, args.lifecycle_templates_b)
    detectors = {
        side: TemplateLifecycleDetector(
            controllers[side].screenshot,
            path,
            expected_device_id=side,
        )
        for side, path in zip(("A", "B"), template_paths, strict=True)
    }
    lifecycle = LifecycleMachine(
        detectors,
        policy=LifecyclePolicy(
            timeout_us={state: 1 for state in LIFECYCLE_PATH},
            poll_interval_us=1,
        ),
        sleep=lambda _interval: None,
    )
    return PhysicalLabRunner(
        phones,
        captures,
        lifecycle,
        split_lock=SplitLock(args.split_lock) if args.split_lock is not None else None,
    )


def _run_command(args: argparse.Namespace) -> int:
    spec = _load_experiment(args.experiment, args.plan_line)
    split_lock = SplitLock(args.split_lock) if args.split_lock is not None else None
    from simulator.storage import enforce_workspace_budget

    budget_kwargs = {
        "workspace_root": args.repository_root,
        "manifest_path": args.retention_manifest,
        "raw_media_root": args.raw_media_root,
        "max_bytes": args.max_workspace_bytes,
        "low_water_bytes": args.low_water_bytes,
        "evict": args.evict,
    }
    budget_before = enforce_workspace_budget(
        **budget_kwargs,
        reserve_bytes=args.reserve_bytes,
    )
    if not budget_before["passed"]:
        payload = {
            "kind": "physical_lab_run_blocked",
            "status": "blocked_workspace_budget",
            "experiment_hash": spec.experiment_hash(),
            "budget_before": budget_before,
        }
        _write_json(args.json_out, payload)
        return 2
    if args.mode == "offline":
        runner = offline_runner(
            observation_times={"hog_crosses_y_mtile": 17_000},
            split_lock=split_lock,
        )
    else:
        runner = _adb_runner(args, spec)
    result = runner.run(spec, run_id=args.run_id)
    budget_after_capture = enforce_workspace_budget(
        **budget_kwargs,
        reserve_bytes=0,
    )
    if not budget_after_capture["passed"]:
        result = replace(
            result,
            status=EvidenceStatus.REJECTED,
            rejection_reasons=result.rejection_reasons
            + ("workspace cap exceeded after physical capture",),
        )
    artifacts = write_run_artifacts(
        result,
        repository_root=args.repository_root,
        retention_manifest=args.retention_manifest,
    )
    payload = result.to_dict(include_hash=True)
    payload["artifacts_written"] = artifacts
    payload["budget_before"] = budget_before
    payload["budget_after_capture"] = budget_after_capture
    _write_json(args.json_out, payload)
    return 0 if result.status is EvidenceStatus.CANDIDATE_ONLY and budget_after_capture["passed"] else 2


def _ingest_command(args: argparse.Namespace) -> int:
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    run = json.loads(args.run.read_text(encoding="utf-8"))
    replay_cache_hash = None
    replay_cache_error = None
    if args.replay_cache is not None:
        try:
            replay_cache_hash = seal_replay_cache(args.replay_cache).sha256
        except ReplayCacheError as error:
            replay_cache_error = str(error)
            if args.replay_cache.is_file():
                from .artifacts import hash_file

                replay_cache_hash = hash_file(args.replay_cache)
    if isinstance(raw, dict) and raw.get("kind") == "physical_lab_observation_manifest":
        manifest = ObservationManifest.from_dict(raw)
        if replay_cache_error is not None:
            manifest = replace(
                manifest,
                status=EvidenceStatus.REJECTED,
                rejected=manifest.rejected
                + (
                    RejectedObservation(
                        "replay-cache",
                        "replay_cache",
                        replay_cache_error,
                    ),
                ),
            )
        elif replay_cache_hash is not None:
            manifest = replace(manifest, replay_cache_hash=replay_cache_hash)
    else:
        synchronization = run.get("synchronization") or {"accepted": False, "rejection_reasons": ["run lacks synchronization"]}
        captures = run.get("captures", {})
        capture_ids = tuple(
            str(item.get("capture_id"))
            for item in captures.values()
            if isinstance(item, dict) and item.get("capture_id")
        ) if isinstance(captures, dict) else ()
        media_hashes = {
            str(side): str(item["media_sha256"])
            for side, item in captures.items()
            if isinstance(item, dict) and item.get("media_sha256")
        } if isinstance(captures, dict) else {}
        spec = ExperimentSpec.from_dict(run["experiment"])
        manifest = ingest_for_experiment(
            raw,
            spec=spec,
            run_id=str(run["run_id"]),
            synchronization=synchronization,
            confidence_threshold=args.confidence_threshold,
            force_direct_timing=args.direct_timing,
            capture_ids=capture_ids,
            media_hashes=media_hashes,
            replay_cache_hash=replay_cache_hash,
            replay_cache_error=replay_cache_error,
        )
    manifest.save(args.json_out)
    retention = None
    retention_error = None
    if args.retention_manifest is not None and manifest.status is not EvidenceStatus.REJECTED:
        audit_paths = list(args.audit_path)
        if args.replay_cache is not None and args.replay_cache.is_file():
            audit_paths.append(args.replay_cache)
        try:
            retention = finalize_retention_records(
                args.retention_manifest,
                run_id=manifest.run_id,
                run_manifest_path=args.run,
                observation_manifest_path=args.json_out,
                workspace_root=args.repository_root,
                audit_paths=audit_paths,
            )
        except PhysicalLabError as error:
            retention_error = str(error)
    payload = manifest.to_dict(include_hash=True)
    if retention is not None:
        payload["retention_finalization"] = retention
    if retention_error is not None:
        payload["retention_finalization_error"] = retention_error
    _write_json(None, payload)
    return 0 if manifest.status is not EvidenceStatus.REJECTED and retention_error is None else 2


def _compare_command(args: argparse.Namespace) -> int:
    observation = ObservationManifest.load(args.observation)
    action_times = None
    if args.run is not None:
        run_raw = json.loads(args.run.read_text(encoding="utf-8"))
        spec = ExperimentSpec.from_dict(run_raw["experiment"])
        action_times = {
            str(item["action_id"]): int(item["actual_match_time_us"])
            for item in run_raw.get("actions", [])
            if item.get("actual_match_time_us") is not None
        }
    elif args.experiment is not None:
        spec = _load_experiment(args.experiment, 1)
    else:
        raise PhysicalLabError("compare requires --experiment or --run")
    replay = run_simulator_replay(spec, action_times=action_times)
    report = compare_observation_to_replay(
        observation,
        replay,
        position_tolerance_mtile=args.position_tolerance_mtile,
        timing_tolerance_us=args.timing_tolerance_us,
    )
    _write_json(args.json_out, report.to_dict())
    _write_json(None, report.to_dict())
    return 0 if report.eligible else 2


def _fidelity_command(args: argparse.Namespace) -> int:
    observation = ObservationManifest.load(args.observation)
    try:
        cache_seal = seal_replay_cache(args.replay_cache)
    except ReplayCacheError as error:
        raise PhysicalLabError(f"replay cache was not recognized: {error}") from error
    run_raw = json.loads(args.run.read_text(encoding="utf-8"))
    run = run_raw if isinstance(run_raw, Mapping) else None
    if run is None:
        raise PhysicalLabError("sealed run.json must contain an object")
    if observation.replay_cache_hash != cache_seal.sha256:
        raise PhysicalLabError(
            "observation replay-cache hash does not match the supplied recognized cache"
        )
    corpus_payload = build_fidelity_corpus_payload(
        observation,
        run,
        replay_cache_hash=cache_seal.sha256,
        position_tolerance_mtile=args.position_tolerance_mtile,
    )
    corpus_path = args.corpus_out
    if corpus_path is None:
        corpus_path = args.json_out.with_name(f"{args.json_out.stem}.corpus.json")
    corpus_hash = write_fidelity_corpus_payload(corpus_path, corpus_payload)

    from ..engine import BattleEngine
    from ..ruleset import load_ruleset
    from ..validation import apply_fidelity_gate, run_fidelity_corpus

    spec = ExperimentSpec.from_dict(run["experiment"])
    report = run_fidelity_corpus(
        BattleEngine(load_ruleset(spec.ruleset_id)),
        corpus_path,
        split=observation.evidence_split.value,
    )
    report = apply_fidelity_gate(
        report,
        min_observations=args.min_observations,
        min_agreement_rate=args.min_agreement_rate,
        required_mechanics=args.require_mechanic,
    )
    report.write_json(args.json_out)
    _write_json(
        None,
        {
            "kind": "physical_lab_fidelity",
            "corpus_path": str(corpus_path),
            "corpus_hash": corpus_hash,
            "report_path": str(args.json_out),
            "report": report.to_dict(),
        },
    )
    assert report.gate is not None
    return 0 if report.gate["passed"] else 2


def _status_command(args: argparse.Namespace) -> int:
    payload: dict[str, object] = {"kind": "physical_lab_device_status", "devices": {}}
    for side, serial in (("A", args.serial_a), ("B", args.serial_b)):
        if serial is None:
            payload["devices"][side] = {"connected": False, "reason": "serial not supplied"}  # type: ignore[index]
            continue
        info = AdbPhoneController(serial).device_info()
        payload["devices"][side] = info.to_dict()  # type: ignore[index]
    _write_json(args.json_out, payload)
    return 0 if all(bool(row.get("connected")) for row in payload["devices"].values()) else 2  # type: ignore[union-attr]


def _keep_awake_command(args: argparse.Namespace) -> int:
    if not args.serial_a or not args.serial_b:
        raise PhysicalLabError("keep-awake requires both --serial-a and --serial-b")
    if args.serial_a == args.serial_b:
        raise PhysicalLabError("keep-awake requires two distinct phone serials")
    if any(any(character.isspace() for character in serial) for serial in (args.serial_a, args.serial_b)):
        raise PhysicalLabError("phone serials must be non-empty tokens")
    payload = {
        "kind": "physical_lab_keep_awake",
        "devices": {
            side: controller.set_keep_awake()
            for side, controller in (
                ("A", AdbPhoneController(args.serial_a, device_label="A")),
                ("B", AdbPhoneController(args.serial_b, device_label="B")),
            )
        },
    }
    _write_json(args.json_out, payload)
    return 0


def _campaign_plan_command(args: argparse.Namespace) -> int:
    campaign = build_default_campaign(
        campaign_id=args.campaign_id,
        evidence_split=args.evidence_split,
    )
    materialized = materialize_campaign(campaign, args.output_root)
    payload = {
        "kind": "physical_lab_campaign_plan",
        "status": "planned",
        **materialized,
        "case_count": len(campaign.cases),
    }
    _write_json(args.json_out, payload)
    return 0


def _campaign_evaluate_command(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.campaign)
    payload = evaluate_campaign(
        campaign,
        results_root=args.results_root,
        split=args.split,
        min_observations=args.min_observations,
        min_agreement_rate=args.min_agreement_rate,
    )
    evaluation_hash = write_campaign_evaluation(args.json_out, payload)
    _write_json(None, {**payload, "evaluation_hash": evaluation_hash})
    return 0 if payload["missing_or_rejected_case_count"] == 0 else 2


def _stored_evaluate_command(args: argparse.Namespace) -> int:
    repository_root = args.repository_root.resolve()
    cases_root = args.cases_root if args.cases_root.is_absolute() else repository_root / args.cases_root
    payload = evaluate_stored_cases(cases_root, repository_root=repository_root)
    output_path = args.json_out if args.json_out.is_absolute() else repository_root / args.json_out
    evaluation_hash = write_stored_evaluation(output_path, payload)
    _write_json(None, {**payload, "evaluation_hash": evaluation_hash, "json_out": str(output_path)})
    return 0 if payload["rejected_artifact_count"] == 0 else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            if args.hog_cannon_only:
                planned = (hog_cannon_probe(capture_group_id=args.capture_group_id, evidence_split=args.evidence_split),)
            else:
                if args.readiness is not None:
                    planned = plan_from_readiness(
                        args.readiness,
                        capture_group_id=args.capture_group_id,
                        evidence_split=args.evidence_split,
                        limit=args.limit,
                    )
                else:
                    planned = plan_from_questions(
                        load_questions(args.unknown_behaviors),
                        capture_group_id=args.capture_group_id,
                        evidence_split=args.evidence_split,
                        limit=args.limit,
                    )
            bound_devices = _bound_device_specs(args.serial_a, args.serial_b)
            if bound_devices is not None:
                planned = tuple(
                    _bind_planned_devices(item, bound_devices) for item in planned
                )
            write_plan(args.json_out, planned)
            _write_json(
                None,
                {
                    "kind": "physical_lab_plan",
                    "count": len(planned),
                    "experiments": [
                        item.to_dict(include_hash=True) if isinstance(item, ExperimentSpec) else item.to_dict()
                        for item in planned
                    ],
                    "json_out": str(args.json_out),
                },
            )
            return 0
        if args.command == "prepare":
            return _prepare_command(args)
        if args.command == "run":
            return _run_command(args)
        if args.command == "ingest":
            return _ingest_command(args)
        if args.command == "compare":
            return _compare_command(args)
        if args.command == "fidelity":
            return _fidelity_command(args)
        if args.command == "extract-run":
            return _extract_run_command(args)
        if args.command == "status":
            return _status_command(args)
        if args.command == "keep-awake":
            return _keep_awake_command(args)
        if args.command == "campaign-plan":
            return _campaign_plan_command(args)
        if args.command == "campaign-evaluate":
            return _campaign_evaluate_command(args)
        if args.command == "evaluate-stored":
            return _stored_evaluate_command(args)
        raise PhysicalLabError(f"unknown command: {args.command}")
    except (PhysicalLabError, OSError, ValueError, json.JSONDecodeError) as error:
        _write_json(None, {"kind": "physical_lab_error", "error": str(error)})
        return 2


__all__ = ["main"]
