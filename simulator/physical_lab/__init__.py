"""Phase-0 physical differential-testing lab.

The package is intentionally usable without connected hardware.  Use
``offline_runner`` to exercise canonical specs, lifecycle handling, capture
sealing, synchronization, and deterministic simulator replay.  ADB adapters
are available for a later connected-device session and fail closed when a
phone is absent.
"""

from .calibration import CalibrationArtifact, CalibrationError
from .automation import (
    AutonomousPhysicalLab,
    AutonomousPhone,
    AutonomousSessionConfig,
    AutomationError,
    CardMatch,
    CardVision,
    FIXED_HOG_CYCLE_DECK,
    UiProfile,
    bind_spec_to_devices,
)
from .artifacts import finalize_retention_records
from .cache import ReplayCacheError, ReplayCacheSeal, seal_replay_cache
from .comparison import ComparisonReport, compare_observation_to_replay
from .devices import (
    ActionReceipt,
    AdbPhoneController,
    AdbScreenCapture,
    CaptureHandle,
    CaptureManifest,
    DeviceDisconnectedError,
    DeviceCommandError,
    DeviceInfo,
    FakePhoneController,
    FakeScreenCapture,
    FrameBufferCapture,
    Frame,
    LogicalPhone,
    PhoneController,
    ScreenCapture,
    monotonic_time_us,
)
from .lifecycle import (
    LIFECYCLE_PATH,
    LifecycleFailure,
    LifecycleMachine,
    LifecyclePolicy,
    LifecycleReport,
    LifecycleState,
    LifecycleTransition,
    ScriptedLifecycleDetector,
)
from .screen_state import (
    SCREEN_TEMPLATE_SCHEMA_VERSION,
    ScreenStateDetectionError,
    TemplateLifecycleDetector,
)
from .observation import (
    EntityObservation,
    EntitySample,
    NormalizedEvent,
    ObservationCertainty,
    ObservationManifest,
    RejectedObservation,
    ingest_extracted_observations,
)
from .planner import (
    PlannedExperiment,
    PriorityInputs,
    hog_cannon_probe,
    plan_from_questions,
    plan_from_readiness,
)
from .replay import ReplayAction, SimulatorReplay, build_scenario, run_simulator_replay
from .runner import ActionLogEntry, PhysicalLabRunner, PhysicalRunResult, offline_runner, write_run_artifacts
from .schema import (
    DeviceSpec,
    EvidenceSplit,
    EvidenceStatus,
    ExperimentSpec,
    InitialConditions,
    MeasurementSpec,
    PhysicalAction,
    PhysicalLabError,
    Trigger,
    TriggerType,
)
from .sync import (
    DeviceClockAlignment,
    SynchronizationError,
    SynchronizationResult,
    SyncMarker,
    TimeMapping,
    estimate_clock_alignment,
)
from .split import SPLIT_LOCK_SCHEMA_VERSION, SplitLock, assign_capture_group_split


__all__ = [
    "ActionLogEntry",
    "ActionReceipt",
    "AdbPhoneController",
    "AdbScreenCapture",
    "AutonomousPhysicalLab",
    "AutonomousPhone",
    "AutonomousSessionConfig",
    "AutomationError",
    "CalibrationArtifact",
    "CalibrationError",
    "ReplayCacheError",
    "ReplayCacheSeal",
    "CaptureHandle",
    "CaptureManifest",
    "CardMatch",
    "CardVision",
    "ComparisonReport",
    "DeviceClockAlignment",
    "DeviceDisconnectedError",
    "DeviceCommandError",
    "DeviceInfo",
    "DeviceSpec",
    "EntityObservation",
    "EntitySample",
    "EvidenceSplit",
    "EvidenceStatus",
    "ExperimentSpec",
    "FakePhoneController",
    "FakeScreenCapture",
    "FrameBufferCapture",
    "Frame",
    "FIXED_HOG_CYCLE_DECK",
    "InitialConditions",
    "LIFECYCLE_PATH",
    "LifecycleFailure",
    "LifecycleMachine",
    "LifecyclePolicy",
    "LifecycleReport",
    "LifecycleState",
    "LifecycleTransition",
    "LogicalPhone",
    "MeasurementSpec",
    "NormalizedEvent",
    "ObservationCertainty",
    "ObservationManifest",
    "PhoneController",
    "PhysicalAction",
    "PhysicalLabError",
    "PhysicalLabRunner",
    "PhysicalRunResult",
    "PlannedExperiment",
    "PriorityInputs",
    "RejectedObservation",
    "ReplayAction",
    "ScriptedLifecycleDetector",
    "SCREEN_TEMPLATE_SCHEMA_VERSION",
    "ScreenStateDetectionError",
    "TemplateLifecycleDetector",
    "SPLIT_LOCK_SCHEMA_VERSION",
    "ScreenCapture",
    "SplitLock",
    "SimulatorReplay",
    "SynchronizationError",
    "SynchronizationResult",
    "SyncMarker",
    "TimeMapping",
    "Trigger",
    "TriggerType",
    "UiProfile",
    "bind_spec_to_devices",
    "build_scenario",
    "compare_observation_to_replay",
    "estimate_clock_alignment",
    "finalize_retention_records",
    "assign_capture_group_split",
    "hog_cannon_probe",
    "ingest_extracted_observations",
    "offline_runner",
    "monotonic_time_us",
    "plan_from_questions",
    "plan_from_readiness",
    "run_simulator_replay",
    "seal_replay_cache",
    "write_run_artifacts",
]
