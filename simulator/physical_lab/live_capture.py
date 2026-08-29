"""Capture and exercise an already-running two-phone friendly battle.

This is the operator bridge for a battle that has already been started and
visually confirmed on both phones.  It deliberately does not navigate either
device, cancel a match, or infer the in-game clock.  The caller supplies the
campaign case and the two reviewed calibrations; the coordinator records a
workstation-monotonic barrier, executes the case actions, and writes the same
sealed ``run.json``/handoff layout as the full lifecycle runner.

The resulting run remains candidate evidence.  This mode is useful for
building a corpus from a live battle while the lifecycle templates are being
reviewed, and it keeps the admission boundary explicit instead of fabricating
the missing lobby/challenge transitions.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from .artifacts import hash_file, physical_output_root
from .automation import AutonomousPhone, CardVision, UiProfile, bind_spec_to_devices
from .calibration import CalibrationArtifact
from .campaign import load_campaign
from .devices import (
    AdbPhoneController,
    CaptureManifest,
    DeviceInfo,
    ScrcpyScreenCapture,
    monotonic_time_us,
)
from .lifecycle import LifecycleReport, LifecycleState
from .replay import run_simulator_replay
from .runner import (
    ActionLogEntry,
    ClockProvenance,
    PhysicalRunResult,
    write_run_artifacts,
)
from .schema import EvidenceStatus, PhysicalLabError, canonical_hash
from ..storage import DEFAULT_LOW_WATER_BYTES, DEFAULT_MAX_WORKSPACE_BYTES, enforce_workspace_budget


class LiveCaptureError(PhysicalLabError):
    """Raised when a current-battle capture cannot be sealed safely."""


def _budget(root: Path) -> dict[str, object]:
    return enforce_workspace_budget(
        root,
        manifest_path=root / "outputs/simulator/fidelity_media/retention.json",
        raw_media_root=root / "outputs/simulator/fidelity_media",
        max_bytes=DEFAULT_MAX_WORKSPACE_BYTES,
        low_water_bytes=DEFAULT_LOW_WATER_BYTES,
        reserve_bytes=0,
        evict=False,
    )


def _select_case(campaign_path: Path, case_id: str):
    campaign = load_campaign(campaign_path)
    try:
        return next(case for case in campaign.cases if case.case_id == case_id)
    except StopIteration as error:
        raise LiveCaptureError(f"campaign has no case {case_id!r}") from error


def _validate_device(
    side: str,
    info: DeviceInfo,
    calibration: CalibrationArtifact,
) -> None:
    if not info.connected:
        raise LiveCaptureError(f"Phone {side} is not connected")
    if info.screen_width_px != calibration.screen_width_px or info.screen_height_px != calibration.screen_height_px:
        raise LiveCaptureError(
            f"Phone {side} native screen {info.screen_width_px}x{info.screen_height_px} "
            f"does not match calibration {calibration.screen_width_px}x{calibration.screen_height_px}"
        )


def _action_log_entry(
    action,
    *,
    slot: int,
    selected,
    placed,
    actual_time_us: int,
) -> ActionLogEntry:
    return ActionLogEntry(
        action_id=action.action_id,
        side=action.side,
        card_id=action.card_id,
        arena_cell=action.arena_cell,
        card_slot=slot,
        requested_trigger=action.trigger.to_dict(),
        actual_match_time_us=actual_time_us,
        selected_card_receipt=selected,
        placement_receipt=placed,
        accepted=True,
    )


def capture_current_battle(
    *,
    repository_root: str | Path,
    campaign_path: str | Path,
    case_id: str,
    serial_a: str,
    serial_b: str,
    calibration_a: str | Path,
    calibration_b: str | Path,
    template_root: str | Path,
    run_id: str,
    duration_s: int = 150,
    action_delay_s: float = 1.0,
) -> dict[str, object]:
    """Capture a confirmed current battle and execute its reviewed actions.

    ``duration_s`` is the capture duration from the synchronized barrier.  A
    run is written even when an action or transport fails so that the failure
    remains auditable; the function raises only before a run can be sealed.
    """

    root = Path(repository_root).resolve()
    if duration_s < 1:
        raise LiveCaptureError("duration_s must be positive")
    if action_delay_s < 0:
        raise LiveCaptureError("action_delay_s must be non-negative")
    campaign = Path(campaign_path)
    if not campaign.is_absolute():
        campaign = root / campaign
    case = _select_case(campaign.resolve(), case_id)
    spec = bind_spec_to_devices(case.spec, serial_a, serial_b)
    spec = replace(
        spec,
        metadata={
            **dict(spec.metadata),
            "live_capture_mode": "confirmed_current_battle",
            "lifecycle_prerequisite": "operator_confirmed_battle_only",
            "truth_promoted": False,
            "campaign_case_id": case.case_id,
            "campaign_case_hash": case.case_hash(),
        },
    )
    output_root = physical_output_root(repository_root=root, run_id=run_id)
    raw_root = output_root / "raw"
    if output_root.exists() and any(output_root.iterdir()):
        raise LiveCaptureError(f"refusing to overwrite existing live-capture run: {output_root}")
    budget_before = _budget(root)
    if not budget_before["passed"]:
        raise LiveCaptureError(
            f"workspace is over the 200 GB cap before capture: {budget_before['deficit_bytes']} bytes"
        )

    controllers = {
        "A": AdbPhoneController(serial_a, device_label="A"),
        "B": AdbPhoneController(serial_b, device_label="B"),
    }
    infos = {side: controllers[side].device_info() for side in ("A", "B")}
    calibrations = {
        "A": CalibrationArtifact.load(Path(calibration_a)),
        "B": CalibrationArtifact.load(Path(calibration_b)),
    }
    for side in ("A", "B"):
        _validate_device(side, infos[side], calibrations[side])

    vision = CardVision(Path(template_root))
    phones = {
        side: AutonomousPhone(
            controllers[side],
            UiProfile.for_device(side, infos[side]),
            vision,
            device_model=infos[side].model,
        )
        for side in ("A", "B")
    }
    captures = {
        side: ScrcpyScreenCapture(
            controllers[side],
            raw_root / f"{side}.mp4",
            time_limit_s=max(330, duration_s + 30),
        )
        for side in ("A", "B")
    }
    actions: list[ActionLogEntry] = []
    rejection_reasons: list[str] = []
    capture_manifests: dict[str, CaptureManifest] = {}
    started = monotonic_time_us()
    battle_barrier: int | None = None
    try:
        # Start the two transports back-to-back, then anchor all action times
        # to the later start.  The video and receipt timestamps remain intact.
        handles = {side: captures[side].start() for side in ("A", "B")}
        battle_barrier = max(handle.started_at_monotonic_us for handle in handles.values())
        for side in ("A", "B"):
            phones[side].record(captures[side])

        if action_delay_s:
            time.sleep(action_delay_s)
        for action in spec.actions:
            if action.trigger.type.value != "match_time_us":
                raise LiveCaptureError(
                    f"current-battle coordinator only supports match-time actions; {action.action_id} uses "
                    f"{action.trigger.type.value}"
                )
            if action.side not in phones:
                raise LiveCaptureError(f"unsupported action side {action.side!r}")
            slot, selected, placed = phones[action.side].select_and_place(
                action.card_id,
                calibration=calibrations[action.side],
                arena_cell=action.arena_cell,
                expected_slot=action.card_slot,
                capture=captures[action.side],
            )
            if battle_barrier is None:
                raise LiveCaptureError("capture barrier was not established")
            if not placed.accepted or placed.completed_at_monotonic_us < battle_barrier:
                raise LiveCaptureError(
                    "placement receipt is not a valid accepted timestamp on the battle axis"
                )
            # Use the placement receipt itself.  Taking another ADB screenshot
            # here can add seconds of latency and makes the simulator replay
            # start from a different input boundary than the phone.
            actual_time_us = placed.completed_at_monotonic_us - battle_barrier
            actions.append(
                _action_log_entry(
                    action,
                    slot=slot,
                    selected=selected,
                    placed=placed,
                    actual_time_us=actual_time_us,
                )
            )
            # Record the other viewpoint immediately after the input so the
            # normalized streams retain a cross-phone action boundary even if
            # the later MP4 frame sampler is sparse.
            other_side = "B" if action.side == "A" else "A"
            phones[other_side].record(captures[other_side])

        deadline = time.monotonic() + float(duration_s)
        while time.monotonic() < deadline:
            time.sleep(min(30.0, max(0.0, deadline - time.monotonic())))
    except Exception as error:  # preserve the partial capture for diagnosis
        rejection_reasons.append(str(error))
    finally:
        for side in ("A", "B"):
            try:
                capture_manifests[side] = captures[side].stop()
            except Exception as error:
                rejection_reasons.append(f"capture {side} failed to stop: {error}")

    if capture_manifests:
        from .sync import estimate_clock_alignment, markers_from_captures

        synchronization = estimate_clock_alignment(
            markers_from_captures(capture_manifests),
            device_ids=("A", "B"),
            declared_tolerance_us=min(
                measurement.timing_tolerance_us for measurement in spec.measurements
            ),
        )
    else:
        synchronization = None
    if synchronization is not None and not synchronization.accepted:
        rejection_reasons.extend(synchronization.rejection_reasons)

    replay = None
    if not rejection_reasons and actions:
        try:
            replay = run_simulator_replay(
                spec,
                action_times={item.action_id: int(item.actual_match_time_us or 0) for item in actions},
            )
        except PhysicalLabError as error:
            rejection_reasons.append(f"simulator replay failed: {error}")
    status = EvidenceStatus.CANDIDATE_ONLY if not rejection_reasons else EvidenceStatus.REJECTED
    finished = monotonic_time_us()
    result = PhysicalRunResult(
        run_id=run_id,
        spec=spec,
        status=status,
        started_at_monotonic_us=started,
        finished_at_monotonic_us=max(started, finished),
        device_info=infos,
        lifecycle=LifecycleReport(
            initial_state=LifecycleState.BATTLE,
            final_state=LifecycleState.BATTLE,
            passed=False,
            observations=({"A": "battle", "B": "battle"},),
            detector_provenance={
                "A": {"mode": "operator_confirmed_current_battle", "reviewed": False},
                "B": {"mode": "operator_confirmed_current_battle", "reviewed": False},
            },
        ),
        captures=capture_manifests,
        synchronization=synchronization,
        actions=tuple(actions),
        replay=replay,
        rejection_reasons=tuple(rejection_reasons),
        clock_provenance=ClockProvenance(
            source="workstation_monotonic_us",
            match_time_axis="elapsed_from_confirmed_capture_barrier",
            in_game_clock_used_for_timing=False,
            in_game_clock_retained_as_diagnostic=True,
            battle_start_monotonic_us=battle_barrier,
            capture_start_monotonic_us={
                side: capture.started_at_monotonic_us
                for side, capture in capture_manifests.items()
            },
        ),
    )
    write_run_artifacts(
        result,
        repository_root=root,
        retention_manifest=root / "outputs/simulator/fidelity_media/retention.json",
    )
    budget_after = _budget(root)
    if not budget_after["passed"]:
        raise LiveCaptureError(
            f"workspace exceeded the 200 GB cap after capture: {budget_after['deficit_bytes']} bytes"
        )
    run_path = output_root / "run.json"
    payload = {
        "kind": "physical_lab_live_capture_summary",
        "run_id": run_id,
        "case_id": case_id,
        "status": result.status.value,
        "run_manifest_path": str(run_path),
        "run_manifest_sha256": hash_file(run_path),
        "actions": [item.to_dict() for item in actions],
        "rejection_reasons": list(rejection_reasons),
        "clock_provenance": result.clock_provenance.to_dict(),
        "synchronization": None if synchronization is None else synchronization.to_dict(),
        "budget_before": budget_before,
        "budget_after_capture": budget_after,
    }
    payload["summary_hash"] = canonical_hash(payload)
    summary_path = output_root / "live-capture-summary.json"
    summary_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    payload["summary_path"] = str(summary_path)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a confirmed current two-phone battle")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--serial-a", required=True)
    parser.add_argument("--serial-b", required=True)
    parser.add_argument("--calibration-a", type=Path, required=True)
    parser.add_argument("--calibration-b", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--duration-s", type=int, default=150)
    parser.add_argument("--action-delay-s", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = capture_current_battle(
            repository_root=args.repository_root,
            campaign_path=args.campaign,
            case_id=args.case_id,
            serial_a=args.serial_a,
            serial_b=args.serial_b,
            calibration_a=args.calibration_a,
            calibration_b=args.calibration_b,
            template_root=args.template_root,
            run_id=args.run_id,
            duration_s=args.duration_s,
            action_delay_s=args.action_delay_s,
        )
    except (LiveCaptureError, PhysicalLabError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == EvidenceStatus.CANDIDATE_ONLY.value else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LiveCaptureError", "capture_current_battle", "main"]
