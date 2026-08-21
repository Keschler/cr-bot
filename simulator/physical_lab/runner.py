"""Physical/offline experiment orchestration and per-run audit records."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Protocol

from .artifacts import ArtifactRef, artifact_manifest, hash_file, physical_output_root, register_retention_records, seal_json
from .calibration import CalibrationError
from .devices import (
    ActionReceipt,
    CaptureManifest,
    DeviceDisconnectedError,
    DeviceInfo,
    FakePhoneController,
    FakeScreenCapture,
    Frame,
    LogicalPhone,
    PhoneController,
    ScreenCapture,
    monotonic_time_us,
)
from .lifecycle import (
    LIFECYCLE_PATH,
    LifecycleMachine,
    LifecycleReport,
    LifecycleState,
    ScriptedLifecycleDetector,
)
from .replay import SimulatorReplay, replay_hash_pair, run_simulator_replay
from .schema import EvidenceStatus, ExperimentSpec, PhysicalAction, PhysicalLabError, TriggerType, canonical_hash
from .split import SplitLock
from .sync import SynchronizationResult, estimate_clock_alignment, markers_from_captures


class ObservationWaiter(Protocol):
    def __call__(self, event: str, value: int, timeout_us: int) -> int: ...


class CaptureFrameRecorder(Protocol):
    def record_frame(self, frame: Frame) -> None: ...


@dataclass(frozen=True, slots=True)
class ActionLogEntry:
    action_id: str
    side: str
    card_id: str
    arena_cell: tuple[int, int]
    card_slot: int | None
    requested_trigger: Mapping[str, object]
    actual_match_time_us: int | None
    selected_card_receipt: ActionReceipt | None
    placement_receipt: ActionReceipt | None
    accepted: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "action_id": self.action_id,
            "side": self.side,
            "card_id": self.card_id,
            "arena_cell": list(self.arena_cell),
            "requested_trigger": dict(self.requested_trigger),
            "actual_match_time_us": self.actual_match_time_us,
            "accepted": self.accepted,
        }
        if self.card_slot is not None:
            result["card_slot"] = self.card_slot
        if self.selected_card_receipt is not None:
            result["selected_card_receipt"] = self.selected_card_receipt.to_dict()
        if self.placement_receipt is not None:
            result["placement_receipt"] = self.placement_receipt.to_dict()
        if self.reason is not None:
            result["reason"] = self.reason
        return result


@dataclass(frozen=True, slots=True)
class PhysicalRunResult:
    run_id: str
    spec: ExperimentSpec
    status: EvidenceStatus | str
    started_at_monotonic_us: int
    finished_at_monotonic_us: int
    device_info: Mapping[str, DeviceInfo]
    lifecycle: LifecycleReport | None
    captures: Mapping[str, CaptureManifest]
    synchronization: SynchronizationResult | None
    actions: tuple[ActionLogEntry, ...]
    replay: SimulatorReplay | None
    rejection_reasons: tuple[str, ...] = ()
    artifact_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        try:
            status = self.status if isinstance(self.status, EvidenceStatus) else EvidenceStatus(self.status)
        except (TypeError, ValueError) as error:
            raise PhysicalLabError(f"invalid physical run status: {self.status!r}") from error
        object.__setattr__(self, "status", status)

    @property
    def experiment_hash(self) -> str:
        return self.spec.experiment_hash()

    def to_dict(self, *, include_hash: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 1,
            "kind": "physical_lab_run",
            "run_id": self.run_id,
            "status": self.status.value,
            "experiment_hash": self.experiment_hash,
            "experiment": self.spec.to_dict(include_hash=True),
            "started_at_monotonic_us": self.started_at_monotonic_us,
            "finished_at_monotonic_us": self.finished_at_monotonic_us,
            "device_info": {side: info.to_dict() for side, info in sorted(self.device_info.items())},
            "lifecycle": None if self.lifecycle is None else self.lifecycle.to_dict(),
            "captures": {side: capture.to_dict() for side, capture in sorted(self.captures.items())},
            "synchronization": None if self.synchronization is None else self.synchronization.to_dict(),
            "actions": [action.to_dict() for action in self.actions],
            "rejection_reasons": list(self.rejection_reasons),
            "artifacts": [artifact.to_dict() for artifact in self.artifact_refs],
        }
        if self.replay is not None:
            result["simulator_replay"] = self.replay.to_dict()
        if include_hash:
            result["run_hash"] = canonical_hash(result)
        return result

    def save(self, path: str | Path) -> ArtifactRef:
        return seal_json(path, self.to_dict(include_hash=True))


def _default_clock() -> int:
    return monotonic_time_us()


def _action_slot(spec: ExperimentSpec, action: PhysicalAction) -> int | None:
    if action.card_slot is not None:
        return action.card_slot
    return spec.initial_conditions.hand_slots.get(action.side, {}).get(action.card_id)


class PhysicalLabRunner:
    """Run an experiment against two logical phone adapters.

    The runner does not interpret pixels and does not assign truth.  It only
    records requested actions, lifecycle transitions, capture metadata, clock
    uncertainty, and explicit rejection reasons.  Observation extraction is a
    separate injected boundary so detector guesses cannot be promoted here.
    """

    def __init__(
        self,
        phones: Mapping[str, LogicalPhone],
        captures: Mapping[str, ScreenCapture],
        lifecycle: LifecycleMachine,
        *,
        observation_waiter: ObservationWaiter | None = None,
        match_time_provider: Callable[[], int] | None = None,
        wait_until_match_time: Callable[[int], None] | None = None,
        clock: Callable[[], int] = _default_clock,
        timing_tolerance_us: int | None = None,
        split_lock: SplitLock | None = None,
        run_id_factory: Callable[[ExperimentSpec], str] | None = None,
    ) -> None:
        if set(phones) != {"A", "B"} or set(captures) != {"A", "B"}:
            raise PhysicalLabError("physical runner requires phone and capture adapters for A and B")
        self.phones = dict(phones)
        self.captures = dict(captures)
        self.lifecycle = lifecycle
        self.observation_waiter = observation_waiter
        self.match_time_provider = match_time_provider
        self.wait_until_match_time = wait_until_match_time
        self.clock = clock
        self.timing_tolerance_us = timing_tolerance_us
        self.split_lock = split_lock
        self.run_id_factory = run_id_factory or (lambda spec: f"{spec.experiment_id}-run")

    def _record_screenshot(self, side: str) -> str | None:
        try:
            frame = self.phones[side].screenshot()
            capture = self.captures[side]
            record_frame = getattr(capture, "record_frame", None)
            if callable(record_frame):
                record_frame(frame)
            return None
        except (DeviceDisconnectedError, PhysicalLabError) as error:
            return f"screenshot {side} failed: {error}"

    def _preflight(self, spec: ExperimentSpec) -> tuple[dict[str, DeviceInfo], list[str]]:
        infos: dict[str, DeviceInfo] = {}
        reasons: list[str] = []
        for side in ("A", "B"):
            try:
                info = self.phones[side].device_info()
            except (DeviceDisconnectedError, PhysicalLabError) as error:
                reasons.append(f"device {side} info failed: {error}")
                continue
            infos[side] = info
            if not info.connected:
                reasons.append(f"device {side} is not connected")
            expected_hash = spec.devices[side].serial_hash
            if info.serial_hash != expected_hash:
                reasons.append(f"device {side} serial hash does not match experiment specification")
            calibration_hash = getattr(self.phones[side].calibration, "device_serial_hash", None)
            if calibration_hash is not None and calibration_hash != info.serial_hash:
                reasons.append(f"device {side} serial hash does not match calibration artifact")
        return infos, reasons

    def _execute_action(
        self,
        spec: ExperimentSpec,
        action: PhysicalAction,
        action_times: dict[str, int],
    ) -> ActionLogEntry:
        trigger = action.trigger
        actual_match_time: int | None
        if trigger.type is TriggerType.MATCH_TIME_US:
            if self.match_time_provider is None:
                # Offline runs use the canonical requested boundary.  A
                # connected runner can inject an OCR/countdown clock and must
                # then prove that the boundary was actually reached.
                actual_match_time = trigger.value
            else:
                if self.wait_until_match_time is not None:
                    self.wait_until_match_time(trigger.value)
                actual_match_time = self.match_time_provider()
                if type(actual_match_time) is not int or actual_match_time < trigger.value:
                    return ActionLogEntry(
                        action_id=action.action_id,
                        side=action.side,
                        card_id=action.card_id,
                        arena_cell=action.arena_cell,
                        card_slot=_action_slot(spec, action),
                        requested_trigger=trigger.to_dict(),
                        actual_match_time_us=actual_match_time,
                        selected_card_receipt=None,
                        placement_receipt=None,
                        accepted=False,
                        reason="match clock did not reach the requested action boundary",
                    )
        else:
            if self.observation_waiter is None:
                return ActionLogEntry(
                    action_id=action.action_id,
                    side=action.side,
                    card_id=action.card_id,
                    arena_cell=action.arena_cell,
                    card_slot=_action_slot(spec, action),
                    requested_trigger=trigger.to_dict(),
                    actual_match_time_us=None,
                    selected_card_receipt=None,
                    placement_receipt=None,
                    accepted=False,
                    reason="after_observation trigger has no observation waiter",
                )
            try:
                actual_match_time = self.observation_waiter(trigger.event or "", trigger.value, spec.duration_us)
            except (PhysicalLabError, TimeoutError) as error:
                return ActionLogEntry(
                    action_id=action.action_id,
                    side=action.side,
                    card_id=action.card_id,
                    arena_cell=action.arena_cell,
                    card_slot=_action_slot(spec, action),
                    requested_trigger=trigger.to_dict(),
                    actual_match_time_us=None,
                    selected_card_receipt=None,
                    placement_receipt=None,
                    accepted=False,
                    reason=f"observation trigger failed: {error}",
                )
            if type(actual_match_time) is not int or actual_match_time < 0:
                return ActionLogEntry(
                    action_id=action.action_id,
                    side=action.side,
                    card_id=action.card_id,
                    arena_cell=action.arena_cell,
                    card_slot=_action_slot(spec, action),
                    requested_trigger=trigger.to_dict(),
                    actual_match_time_us=None,
                    selected_card_receipt=None,
                    placement_receipt=None,
                    accepted=False,
                    reason="observation waiter returned invalid match time",
                )
        action_times[action.action_id] = actual_match_time
        slot = _action_slot(spec, action)
        if slot is None:
            return ActionLogEntry(
                action_id=action.action_id,
                side=action.side,
                card_id=action.card_id,
                arena_cell=action.arena_cell,
                card_slot=None,
                requested_trigger=trigger.to_dict(),
                actual_match_time_us=actual_match_time,
                selected_card_receipt=None,
                placement_receipt=None,
                accepted=False,
                reason="card slot is not present in the experiment specification",
            )
        try:
            selected = self.phones[action.side].select_card(slot=slot)
            if not selected.accepted:
                return ActionLogEntry(
                    action_id=action.action_id,
                    side=action.side,
                    card_id=action.card_id,
                    arena_cell=action.arena_cell,
                    card_slot=slot,
                    requested_trigger=trigger.to_dict(),
                    actual_match_time_us=actual_match_time,
                    selected_card_receipt=selected,
                    placement_receipt=None,
                    accepted=False,
                    reason=selected.reason or "card selection was not acknowledged",
                )
            placed = self.phones[action.side].place_card(action.card_id, arena_cell=action.arena_cell)
            return ActionLogEntry(
                action_id=action.action_id,
                side=action.side,
                card_id=action.card_id,
                arena_cell=action.arena_cell,
                card_slot=slot,
                requested_trigger=trigger.to_dict(),
                actual_match_time_us=actual_match_time,
                selected_card_receipt=selected,
                placement_receipt=placed,
                accepted=placed.accepted,
                reason=placed.reason if not placed.accepted else None,
            )
        except (CalibrationError, DeviceDisconnectedError, PhysicalLabError) as error:
            return ActionLogEntry(
                action_id=action.action_id,
                side=action.side,
                card_id=action.card_id,
                arena_cell=action.arena_cell,
                card_slot=slot,
                requested_trigger=trigger.to_dict(),
                actual_match_time_us=actual_match_time,
                selected_card_receipt=None,
                placement_receipt=None,
                accepted=False,
                reason=str(error),
            )

    def run(self, spec: ExperimentSpec, *, run_id: str | None = None) -> PhysicalRunResult:
        run_id = run_id or self.run_id_factory(spec)
        started = self.clock()
        infos, reasons = self._preflight(spec)
        if self.split_lock is not None:
            try:
                self.split_lock.lock(spec.capture_group_id, spec.evidence_split)
            except PhysicalLabError as error:
                reasons.append(f"capture-group split lock failed: {error}")
        if reasons:
            finished = self.clock()
            return PhysicalRunResult(
                run_id=run_id,
                spec=spec,
                status=EvidenceStatus.REJECTED,
                started_at_monotonic_us=started,
                finished_at_monotonic_us=max(started, finished),
                device_info=infos,
                lifecycle=None,
                captures={},
                synchronization=None,
                actions=(),
                replay=None,
                rejection_reasons=tuple(reasons),
            )

        captures: dict[str, CaptureManifest] = {}
        actions: list[ActionLogEntry] = []
        action_times: dict[str, int] = {}
        lifecycle_report: LifecycleReport | None = None
        synchronization: SynchronizationResult | None = None
        replay: SimulatorReplay | None = None
        try:
            for side in ("A", "B"):
                self.captures[side].start()
            # Capture begins before lifecycle navigation.  A frame at this
            # boundary is also the software harness's sync marker.
            for side in ("A", "B"):
                screenshot_error = self._record_screenshot(side)
                if screenshot_error is not None:
                    reasons.append(screenshot_error)
            lifecycle_report = self.lifecycle.run()
            if not lifecycle_report.passed:
                reasons.append("battle lifecycle failed")
            else:
                for action in spec.actions:
                    entry = self._execute_action(spec, action, action_times)
                    actions.append(entry)
                    screenshot_error = self._record_screenshot(action.side)
                    if screenshot_error is not None:
                        reasons.append(screenshot_error)
                    if not entry.accepted:
                        reasons.append(f"action {action.action_id} was not acknowledged: {entry.reason}")
        except (DeviceDisconnectedError, PhysicalLabError) as error:
            reasons.append(f"runner failure: {error}")
        finally:
            for side in ("A", "B"):
                try:
                    captures[side] = self.captures[side].stop()
                except (DeviceDisconnectedError, PhysicalLabError) as error:
                    reasons.append(f"capture {side} failed to stop: {error}")

        if captures:
            for side, capture in sorted(captures.items()):
                if not capture.stream_verified:
                    reasons.append(f"capture {side} does not contain a verified video stream")
                if not capture.frames:
                    reasons.append(f"capture {side} contains no frame records")
            tolerance = self.timing_tolerance_us
            if tolerance is None:
                tolerance = min(item.timing_tolerance_us for item in spec.measurements)
            synchronization = estimate_clock_alignment(
                markers_from_captures(captures),
                device_ids=("A", "B"),
                declared_tolerance_us=tolerance,
            )
            if not synchronization.accepted:
                reasons.extend(synchronization.rejection_reasons)
            for capture in captures.values():
                # The dataclass is immutable; the sealed run report retains
                # the synchronization result alongside each stream.
                _ = capture

        if not reasons and lifecycle_report is not None and lifecycle_report.passed:
            try:
                replay = run_simulator_replay(spec, action_times=action_times)
                # A physical run is not evidence until an extractor supplies
                # normalized observations.  The software replay is still
                # recorded and audited now, while status stays candidate-only.
                replay_hash_pair(spec, action_times=action_times)
            except PhysicalLabError as error:
                reasons.append(f"simulator replay failed: {error}")

        status = EvidenceStatus.CANDIDATE_ONLY if not reasons else EvidenceStatus.REJECTED
        finished = self.clock()
        return PhysicalRunResult(
            run_id=run_id,
            spec=spec,
            status=status,
            started_at_monotonic_us=started,
            finished_at_monotonic_us=max(started, finished),
            device_info=infos,
            lifecycle=lifecycle_report,
            captures=captures,
            synchronization=synchronization,
            actions=tuple(actions),
            replay=replay,
            rejection_reasons=tuple(reasons),
        )


def offline_runner(
    *,
    observation_times: Mapping[str, int] | None = None,
    split_lock: SplitLock | None = None,
    monotonic_clock: Callable[[], int] | None = None,
) -> PhysicalLabRunner:
    """Create the complete Phase-0 software harness without connected phones."""

    if monotonic_clock is None:
        class _OfflineClock:
            value = 0

            def __call__(self) -> int:
                current = self.value
                self.value += 100
                return current

        monotonic_clock = _OfflineClock()

    phone_a_controller = FakePhoneController("A", serial_label="A", monotonic_clock=monotonic_clock)
    phone_b_controller = FakePhoneController("B", serial_label="B", monotonic_clock=monotonic_clock)
    from .calibration import CalibrationArtifact

    calibration_a = CalibrationArtifact.for_screen(
        device_label="offline-A", screen_width_px=1080, screen_height_px=2400
    )
    calibration_b = CalibrationArtifact.for_screen(
        device_label="offline-B", screen_width_px=1080, screen_height_px=2400
    )
    phones = {
        "A": LogicalPhone(phone_a_controller, calibration_a),
        "B": LogicalPhone(phone_b_controller, calibration_b),
    }
    captures = {
        "A": FakeScreenCapture("A", frame_source=phone_a_controller.screenshot, monotonic_clock=monotonic_clock),
        "B": FakeScreenCapture("B", frame_source=phone_b_controller.screenshot, monotonic_clock=monotonic_clock),
    }
    detectors = {
        side: ScriptedLifecycleDetector(LIFECYCLE_PATH)
        for side in ("A", "B")
    }
    lifecycle = LifecycleMachine(
        detectors,
        clock=monotonic_clock,
        sleep=lambda _interval: None,
    )
    observation_times = dict(observation_times or {})

    def wait_for_observation(event: str, value: int, _timeout_us: int) -> int:
        return observation_times.get(event, value)

    return PhysicalLabRunner(
        phones,
        captures,
        lifecycle,
        observation_waiter=wait_for_observation,
        split_lock=split_lock,
        clock=monotonic_clock,
    )


def write_run_artifacts(
    result: PhysicalRunResult,
    *,
    repository_root: str | Path,
    retention_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Write only small JSON provenance artifacts under the ignored output root."""

    root = physical_output_root(repository_root=repository_root, run_id=result.run_id)
    root.mkdir(parents=True, exist_ok=True)
    run_path = root / "run.json"
    run_ref = result.save(run_path)
    refs = [run_ref]
    for side, capture in sorted(result.captures.items()):
        if capture.media_path is None:
            continue
        media_path = Path(capture.media_path)
        if not media_path.is_file():
            continue
        refs.append(
            ArtifactRef(
                artifact_id=f"capture-{side}",
                kind="raw_video",
                path=str(media_path),
                sha256=capture.media_sha256 or hash_file(media_path),
                size_bytes=media_path.stat().st_size,
            )
        )
    if result.replay is not None:
        replay_ref = seal_json(root / "simulator-replay.json", result.replay.to_dict())
        refs.append(replay_ref)
    manifest_payload = artifact_manifest(
        run_id=result.run_id,
        experiment_hash=result.experiment_hash,
        artifacts=refs,
        status=result.status.value,
    )
    manifest_ref = seal_json(root / "artifacts.json", manifest_payload)
    refs.append(manifest_ref)
    if retention_manifest is not None:
        register_retention_records(
            retention_manifest,
            run_id=result.run_id,
            experiment_hash=result.experiment_hash,
            artifacts=refs,
            workspace_root=repository_root,
        )
    return {
        "output_root": str(root),
        "run_path": str(run_path),
        "artifact_manifest": str(root / "artifacts.json"),
        "artifacts": [item.to_dict() for item in refs],
    }


__all__ = [
    "ActionLogEntry",
    "PhysicalLabRunner",
    "PhysicalRunResult",
    "offline_runner",
    "write_run_artifacts",
]
