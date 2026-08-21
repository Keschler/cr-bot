"""Screen-verified battle lifecycle state machine with explicit failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Callable, Iterable, Mapping, Protocol

from .schema import PhysicalLabError


class LifecycleState(str, Enum):
    RECOVERY = "recovery"
    LOBBY = "lobby"
    CHALLENGE_SENT = "challenge_sent"
    CHALLENGE_ACCEPTED = "challenge_accepted"
    LOADING = "loading"
    BATTLE = "battle"
    RESULT = "result"
    ARCHIVED = "archived"


LIFECYCLE_PATH: tuple[LifecycleState, ...] = (
    LifecycleState.RECOVERY,
    LifecycleState.LOBBY,
    LifecycleState.CHALLENGE_SENT,
    LifecycleState.CHALLENGE_ACCEPTED,
    LifecycleState.LOADING,
    LifecycleState.BATTLE,
    LifecycleState.RESULT,
    LifecycleState.ARCHIVED,
    LifecycleState.RECOVERY,
)


class LifecycleDetector(Protocol):
    """Return the state currently visible on one phone."""

    def detect(self) -> LifecycleState: ...


@dataclass(frozen=True, slots=True)
class LifecycleTransition:
    from_state: LifecycleState
    to_state: LifecycleState
    observed_at_monotonic_us: int
    device_states: Mapping[str, LifecycleState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "from": self.from_state.value,
            "to": self.to_state.value,
            "observed_at_monotonic_us": self.observed_at_monotonic_us,
            "device_states": {key: value.value for key, value in sorted(self.device_states.items())},
        }


@dataclass(frozen=True, slots=True)
class LifecycleFailure:
    state: LifecycleState
    reason: str
    observed_at_monotonic_us: int
    recovery_attempted: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "observed_at_monotonic_us": self.observed_at_monotonic_us,
            "recovery_attempted": self.recovery_attempted,
        }


@dataclass(frozen=True, slots=True)
class LifecycleReport:
    initial_state: LifecycleState
    final_state: LifecycleState
    passed: bool
    transitions: tuple[LifecycleTransition, ...] = ()
    failure: LifecycleFailure | None = None
    observations: tuple[Mapping[str, str], ...] = ()
    detector_provenance: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "initial_state": self.initial_state.value,
            "final_state": self.final_state.value,
            "passed": self.passed,
            "transitions": [item.to_dict() for item in self.transitions],
            "observations": [dict(item) for item in self.observations],
            "detector_provenance": {
                key: dict(value) for key, value in sorted(self.detector_provenance.items())
            },
        }
        if self.failure is not None:
            result["failure"] = self.failure.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class LifecyclePolicy:
    """Per-state screen wait limits and recovery actions."""

    timeout_us: Mapping[LifecycleState, int] = field(
        default_factory=lambda: {
            LifecycleState.LOBBY: 30_000_000,
            LifecycleState.CHALLENGE_SENT: 30_000_000,
            LifecycleState.CHALLENGE_ACCEPTED: 30_000_000,
            LifecycleState.LOADING: 60_000_000,
            LifecycleState.BATTLE: 30_000_000,
            LifecycleState.RESULT: 60_000_000,
            LifecycleState.ARCHIVED: 30_000_000,
            LifecycleState.RECOVERY: 30_000_000,
        }
    )
    poll_interval_us: int = 100_000

    def timeout_for(self, state: LifecycleState) -> int:
        timeout = self.timeout_us.get(state, 30_000_000)
        if type(timeout) is not int or timeout <= 0:
            raise PhysicalLabError(f"invalid timeout for lifecycle state {state.value}")
        return timeout


def _default_clock() -> int:
    return time.monotonic_ns() // 1_000


def _default_sleep(interval_us: int) -> None:
    time.sleep(max(0.0, interval_us / 1_000_000.0))


class LifecycleMachine:
    """Drive one or two screen detectors through the verified lifecycle."""

    def __init__(
        self,
        detectors: Mapping[str, LifecycleDetector] | LifecycleDetector,
        *,
        policy: LifecyclePolicy | None = None,
        recovery_actions: Mapping[LifecycleState, Callable[[], None]] | None = None,
        clock: Callable[[], int] = _default_clock,
        sleep: Callable[[int], None] = _default_sleep,
    ) -> None:
        if isinstance(detectors, Mapping):
            if not detectors:
                raise PhysicalLabError("at least one lifecycle detector is required")
            self.detectors = dict(detectors)
        else:
            self.detectors = {"device": detectors}
        detector_provenance: dict[str, Mapping[str, object]] = {}
        for device_id, detector in sorted(self.detectors.items()):
            provider = getattr(detector, "provenance", None)
            if callable(provider):
                value = provider()
                if not isinstance(value, Mapping):
                    raise PhysicalLabError(
                        f"detector {device_id} provenance must be a mapping"
                    )
                detector_provenance[device_id] = dict(value)
        self.detector_provenance = detector_provenance
        self.policy = policy or LifecyclePolicy()
        if self.policy.poll_interval_us <= 0:
            raise PhysicalLabError("lifecycle poll interval must be positive")
        self.recovery_actions = dict(recovery_actions or {})
        self.clock = clock
        self.sleep = sleep

    def _detect(self) -> tuple[LifecycleState | None, dict[str, LifecycleState]]:
        states: dict[str, LifecycleState] = {}
        for device_id, detector in sorted(self.detectors.items()):
            state = detector.detect()
            if not isinstance(state, LifecycleState):
                try:
                    state = LifecycleState(state)
                except (TypeError, ValueError) as error:
                    raise PhysicalLabError(f"detector {device_id} returned invalid state {state!r}") from error
            states[device_id] = state
        unique = set(states.values())
        return (next(iter(unique)) if len(unique) == 1 else None), states

    def run(self) -> LifecycleReport:
        current = LifecycleState.RECOVERY
        transitions: list[LifecycleTransition] = []
        observations: list[Mapping[str, str]] = []
        # The initial recovery state is a controller state.  Every transition,
        # including final ARCHIVED -> RECOVERY, must be seen on all devices.
        for target in LIFECYCLE_PATH[1:]:
            deadline = self.clock() + self.policy.timeout_for(target)
            recovery_attempted = False
            while True:
                observed, device_states = self._detect()
                observations.append(
                    {device_id: state.value for device_id, state in sorted(device_states.items())}
                )
                now = self.clock()
                if observed is not None and observed is target:
                    transitions.append(
                        LifecycleTransition(
                            from_state=current,
                            to_state=target,
                            observed_at_monotonic_us=now,
                            device_states=device_states,
                        )
                    )
                    current = target
                    break
                if now >= deadline:
                    action = self.recovery_actions.get(current)
                    if action is not None and not recovery_attempted:
                        action()
                        recovery_attempted = True
                    detail = (
                        "devices disagree: "
                        + ", ".join(f"{key}={value.value}" for key, value in sorted(device_states.items()))
                        if observed is None
                        else f"screen shows {observed.value}, expected {target.value}"
                    )
                    failure = LifecycleFailure(
                        state=target,
                        reason=f"lifecycle timeout after {self.policy.timeout_for(target)}us: {detail}",
                        observed_at_monotonic_us=now,
                        recovery_attempted=recovery_attempted,
                    )
                    return LifecycleReport(
                        initial_state=LifecycleState.RECOVERY,
                        final_state=current,
                        passed=False,
                        transitions=tuple(transitions),
                        failure=failure,
                        observations=tuple(observations),
                        detector_provenance=self.detector_provenance,
                    )
                self.sleep(self.policy.poll_interval_us)
        return LifecycleReport(
            initial_state=LifecycleState.RECOVERY,
            final_state=current,
            passed=True,
            transitions=tuple(transitions),
            observations=tuple(observations),
            detector_provenance=self.detector_provenance,
        )


class ScriptedLifecycleDetector:
    """Deterministic detector for offline tests and the fake runner."""

    def __init__(self, states: Iterable[LifecycleState | str]) -> None:
        parsed = []
        for state in states:
            parsed.append(state if isinstance(state, LifecycleState) else LifecycleState(state))
        if not parsed:
            raise PhysicalLabError("scripted lifecycle detector needs at least one state")
        self._states = tuple(parsed)
        self._index = 0

    def detect(self) -> LifecycleState:
        state = self._states[min(self._index, len(self._states) - 1)]
        if self._index < len(self._states) - 1:
            self._index += 1
        return state


__all__ = [
    "LIFECYCLE_PATH",
    "LifecycleDetector",
    "LifecycleFailure",
    "LifecycleMachine",
    "LifecyclePolicy",
    "LifecycleReport",
    "LifecycleState",
    "LifecycleTransition",
    "ScriptedLifecycleDetector",
]
