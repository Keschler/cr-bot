"""Observation-only inference over recorded or live visual match streams.

The recurrent PPO prototype normally receives observations from
``SimulatorEnv`` and advances a simulator after every action.  This module is
the separate shadow path for real footage:

``MP4/replay cache/V4L2 -> vision -> MatchSession -> public V2 -> actor -> JSON``

It never advances simulator physics, never constructs privileged critic
features, and never calls a phone action API.  A recorded MP4 cannot provide a
counterfactual win rate; this runner instead produces a timestamped trace of
the actions the actor would have selected.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

from ._compat import TORCH_AVAILABLE, TorchUnavailableError


SHADOW_SCHEMA_VERSION = 1
SHADOW_RUNNER_VERSION = "public-vision-shadow-v1"


class ShadowConfigurationError(ValueError):
    """Raised when a shadow source or inference contract is invalid."""


def _vision_dependency_error(error: ModuleNotFoundError) -> ShadowConfigurationError:
    missing = error.name or "a vision dependency"
    return ShadowConfigurationError(
        "shadow media input requires the project's vision dependencies; "
        f"missing {missing}. Install requirements.txt in the configured "
        "outputs/venv environment before using --video, --replay-cache, or "
        "--video-device."
    )


def _require_torch() -> Any:
    if not TORCH_AVAILABLE:
        raise TorchUnavailableError(
            "The shadow runner requires PyTorch. Use the configured outputs/venv "
            "Python or install the optional torch dependency."
        )
    import torch

    return torch


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ShadowConfigurationError(f"cannot hash shadow source {path}: {error}") from error
    return f"sha256:{digest.hexdigest()}"


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ShadowConfigurationError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ShadowConfigurationError(f"{name} must be a non-negative integer")


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShadowConfigurationError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ShadowConfigurationError(f"{name} must be a finite non-negative number")
    return result


class ShadowPolicyRunner:
    """Run one public recurrent actor over an emitted visual-step stream.

    A caller supplies the ``MatchSessionStep`` objects produced by the
    existing vision tracker.  Hidden state is reset after a non-playing frame
    and is otherwise carried from one valid observation to the next.
    """

    def __init__(self, learner: Any, *, source_id: str | None = None) -> None:
        torch = _require_torch()
        if learner is None or not (
            hasattr(learner, "policy") or hasattr(learner, "rollout_step")
        ):
            raise TypeError("learner must expose a recurrent policy or rollout_step")
        self.learner = learner
        self.policy = learner.policy.eval() if hasattr(learner, "policy") else None
        self._state = learner.initial_rollout_state(1)
        self.source_id = source_id
        self._reset_pending = True
        self._match_active = False
        self.matches_started = 0
        self.match_boundaries = 0
        self.emitted_observations = 0
        self.invalid_observations = 0
        self._decisions: list[dict[str, object]] = []
        self._last_frame_idx: int | None = None
        self._last_time_s: float | None = None
        # Keep the import local so ``import rl`` remains usable without torch.
        self._torch = torch

    @property
    def reset_pending(self) -> bool:
        """Whether the next valid observation starts a fresh recurrent episode."""

        return self._reset_pending

    def _validate_frame_coordinates(self, frame_idx: int, time_s: float) -> None:
        _nonnegative_int("frame_idx", frame_idx)
        time_s = _finite_nonnegative("time_s", time_s)
        if self._last_frame_idx is not None and frame_idx <= self._last_frame_idx:
            raise ShadowConfigurationError(
                f"shadow frame indices must increase strictly: {frame_idx} after {self._last_frame_idx}"
            )
        if self._last_time_s is not None and time_s + 1e-9 < self._last_time_s:
            raise ShadowConfigurationError(
                f"shadow timestamps must be monotonic: {time_s} after {self._last_time_s}"
            )
        self._last_frame_idx = frame_idx
        self._last_time_s = time_s

    def _mark_not_in_game(self) -> None:
        if self._match_active:
            self.match_boundaries += 1
            self._match_active = False
        self._reset_pending = True

    def _batch_observation(self, observation: Any) -> tuple[Any, Any, Any, Any, Any]:
        # Reuse the collector's exact V2 tensor and mask conversion.  This
        # helper only batches public observations; it does not touch an env or
        # ask for a privileged value estimate.
        from .collector import _batch_observations

        return _batch_observations([observation], device=self.learner.device)

    def _deterministic_actor_step(self, observation: Any) -> tuple[Any, Any, Any]:
        """Run only the actor, deliberately bypassing a privileged critic."""

        from .learner import RecurrentRolloutState, _deterministic_action

        raster, global_features, entities, entity_mask, masks = self._batch_observation(
            observation
        )
        reset_mask = self._torch.tensor(
            [[self._reset_pending]],
            dtype=self._torch.bool,
            device=self.learner.device,
        )
        if self.policy is None:
            # This narrow fallback is useful for contract tests and small
            # adapters that expose the learner's actor-step seam directly.
            # The production learner has ``policy`` and therefore takes the
            # branch below, which bypasses its privileged value head.
            step = self.learner.rollout_step(
                self._state,
                raster[:, 0],
                global_features[:, 0],
                entities[:, 0],
                entity_mask[:, 0],
                masks,
                reset_mask=reset_mask.reshape(-1),
                privileged_features=None,
                deterministic=True,
            )
            self._state = step.next_state
            self._reset_pending = False
            log_probs = getattr(
                step,
                "log_probs",
                self._torch.zeros((1, 1), dtype=self._torch.float32, device=self.learner.device),
            )
            entropy = getattr(
                step,
                "entropy",
                self._torch.zeros((1, 1), dtype=self._torch.float32, device=self.learner.device),
            )
            return step.actions, log_probs, entropy
        with self._torch.inference_mode():
            output = self.policy(
                raster,
                global_features,
                entities,
                entity_mask,
                reset_mask=reset_mask,
                hidden=self._state.hidden,
                action_masks=masks,
            )
            actions, log_probs, entropy = _deterministic_action(
                self.policy,
                output,
                masks,
            )
        self._state = RecurrentRolloutState(output.final_hidden.detach())
        self._reset_pending = False
        return actions, log_probs, entropy

    def consume(
        self,
        step: Any,
        *,
        frame_idx: int,
        time_s: float,
    ) -> "ShadowPrediction | None":
        """Consume one tracker step and return a prediction when available."""

        self._validate_frame_coordinates(frame_idx, time_s)
        if step is None or not bool(getattr(step, "in_game", False)):
            self._mark_not_in_game()
            return None
        if not bool(getattr(step, "should_emit", False)) or getattr(step, "game_state", None) is None:
            return None
        # MatchSession keeps emitting a short end-of-match grace period while
        # it waits for the result screen to be confirmed.  Never suggest an
        # action after regulation/overtime has ended.
        analysis = getattr(step, "analysis", None)
        remaining = getattr(analysis, "total_remaining_s", None)
        if remaining is not None:
            try:
                if not math.isfinite(float(remaining)) or float(remaining) <= 0.0:
                    self._mark_not_in_game()
                    return None
            except (TypeError, ValueError):
                self.invalid_observations += 1
                return None

        if not self._match_active:
            self.matches_started += 1
            self._match_active = True
        self.emitted_observations += 1

        try:
            try:
                from ..physical_lab.policy_bridge import observation_v2_from_match_step
                from ..physical_lab.schema import PhysicalLabError
            except ImportError:  # top-level ``rl`` layout from the simulator directory
                from simulator.physical_lab.policy_bridge import observation_v2_from_match_step
                from simulator.physical_lab.schema import PhysicalLabError

            observation = observation_v2_from_match_step(step)
            if observation is None:
                return None
            actions, log_probs, entropy = self._deterministic_actor_step(observation)
        except (PhysicalLabError, TypeError, ValueError, RuntimeError):
            # A malformed or incomplete visual observation must produce no
            # action.  Preserve the recurrent state and let the next valid
            # frame try again; callers receive the count in the report.
            self.invalid_observations += 1
            return None

        try:
            mode = int(actions.mode[0, 0].detach().cpu().item())
            action_log_prob = float(log_probs[0, 0].detach().cpu().item())
            action_entropy = float(entropy[0, 0].detach().cpu().item())
            if mode == 0:
                prediction = ShadowPrediction(
                    frame_idx=frame_idx,
                    time_s=float(time_s),
                    action_kind="WAIT",
                    card_slot=None,
                    card_id=None,
                    arena_cell=None,
                    action_log_prob=action_log_prob,
                    entropy=action_entropy,
                )
            elif mode == 1:
                card_slot = int(actions.card_slot[0, 0].detach().cpu().item())
                row = int(actions.placement[0, 0, 0].detach().cpu().item())
                column = int(actions.placement[0, 0, 1].detach().cpu().item())
                hand = getattr(getattr(step.game_state, "hud", None), "hand_cards", ())
                card_id = None
                if isinstance(hand, (list, tuple)) and 0 <= card_slot < len(hand):
                    try:
                        from ..observation import policy_card_name
                    except ImportError:  # top-level ``rl`` layout from the simulator directory
                        from simulator.observation import policy_card_name

                    card_id = policy_card_name(hand[card_slot])
                prediction = ShadowPrediction(
                    frame_idx=frame_idx,
                    time_s=float(time_s),
                    action_kind="PLAY",
                    card_slot=card_slot,
                    card_id=card_id,
                    arena_cell=(column, row),
                    action_log_prob=action_log_prob,
                    entropy=action_entropy,
                )
            else:
                self.invalid_observations += 1
                return None
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
            # A malformed model output is never forwarded as a placement.
            self.invalid_observations += 1
            return None
        self._decisions.append(prediction.to_trace_dict())
        return prediction

    def process_step(
        self,
        step: Any,
        *,
        frame_index: int,
        timestamp_s: float,
    ) -> dict[str, object] | None:
        """Compatibility-facing JSON decision wrapper around :meth:`consume`."""

        prediction = self.consume(
            step,
            frame_idx=frame_index,
            time_s=timestamp_s,
        )
        return None if prediction is None else prediction.to_trace_dict()

    def trace_payload(self) -> dict[str, object]:
        """Return the JSON-safe decision trace accumulated so far."""

        payload: dict[str, object] = {
            "kind": "recurrent_public_ppo_shadow_trace",
            "shadow_schema_version": SHADOW_SCHEMA_VERSION,
            "shadow_runner_version": SHADOW_RUNNER_VERSION,
            "decisions": list(self._decisions),
            "actor_privileged_inputs": False,
            "critic_privileged_inputs": False,
            "taps_sent": 0,
        }
        if self.source_id is not None:
            payload["source_id"] = self.source_id
        return payload


class ShadowPrediction:
    """One deterministic actor decision in viewer-local action coordinates."""

    __slots__ = (
        "frame_idx",
        "time_s",
        "action_kind",
        "card_slot",
        "card_id",
        "arena_cell",
        "action_log_prob",
        "entropy",
    )

    def __init__(
        self,
        *,
        frame_idx: int,
        time_s: float,
        action_kind: str,
        card_slot: int | None,
        card_id: str | None,
        arena_cell: tuple[int, int] | None,
        action_log_prob: float,
        entropy: float,
    ) -> None:
        _nonnegative_int("frame_idx", frame_idx)
        _finite_nonnegative("time_s", time_s)
        if action_kind not in {"WAIT", "PLAY"}:
            raise ShadowConfigurationError("action_kind must be WAIT or PLAY")
        if action_kind == "WAIT" and any(
            value is not None for value in (card_slot, card_id, arena_cell)
        ):
            raise ShadowConfigurationError("WAIT predictions cannot carry card or placement data")
        if action_kind == "PLAY":
            if type(card_slot) is not int or not 0 <= card_slot < 4:
                raise ShadowConfigurationError("PLAY card_slot must be in [0, 3]")
            if (
                not isinstance(arena_cell, tuple)
                or len(arena_cell) != 2
                or any(type(value) is not int for value in arena_cell)
                or not (0 <= arena_cell[0] < 18 and 0 <= arena_cell[1] < 32)
            ):
                raise ShadowConfigurationError("PLAY arena_cell must be a valid (column, row) cell")
        self.frame_idx = frame_idx
        self.time_s = float(time_s)
        self.action_kind = action_kind
        self.card_slot = card_slot
        self.card_id = card_id
        self.arena_cell = arena_cell
        self.action_log_prob = float(action_log_prob)
        self.entropy = float(entropy)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "frame_idx": self.frame_idx,
            "time_s": self.time_s,
            "action_kind": self.action_kind,
            "action_log_prob": self.action_log_prob,
            "entropy": self.entropy,
        }
        if self.card_slot is not None:
            result["card_slot"] = self.card_slot
        if self.card_id is not None:
            result["card_id"] = self.card_id
        if self.arena_cell is not None:
            result["arena_cell"] = list(self.arena_cell)
        return result

    def to_trace_dict(self) -> dict[str, object]:
        """Return the compact action shape used by shadow traces."""

        if self.action_kind == "WAIT":
            action: dict[str, object] = {"kind": "Wait"}
        else:
            action = {
                "kind": "Play",
                "card_slot": self.card_slot,
                "cell": list(self.arena_cell or ()),
            }
        return {
            "frame_index": self.frame_idx,
            "timestamp_s": self.time_s,
            "action": action,
        }


def _source_spec(
    *,
    video: str | Path | None,
    replay_cache: str | Path | None,
    video_device: str | int | None,
    allow_missing_files: bool = False,
) -> tuple[str, str | Path, str | None]:
    supplied = [value is not None for value in (video, replay_cache, video_device)]
    if sum(supplied) != 1:
        raise ShadowConfigurationError(
            "provide exactly one of video, replay_cache, or video_device"
        )
    if video is not None:
        path = Path(video)
        if not path.is_file() and not allow_missing_files:
            raise ShadowConfigurationError(f"shadow video is not a file: {path}")
        return "video", path, _file_sha256(path) if path.is_file() else None
    if replay_cache is not None:
        path = Path(replay_cache)
        if not path.is_file() and not allow_missing_files:
            raise ShadowConfigurationError(f"shadow replay cache is not a file: {path}")
        return "replay-cache", path, _file_sha256(path) if path.is_file() else None
    assert video_device is not None
    if isinstance(video_device, str) and not video_device.strip():
        raise ShadowConfigurationError("video_device must be a non-empty path or index")
    return "video-device", str(video_device), None


def _video_capture_source(value: str | Path | int, *, live: bool = False) -> Any:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise _vision_dependency_error(error) from error

    source: str | int = value
    if isinstance(value, str) and value.strip().isdigit():
        source = int(value.strip())
    capture = cv2.VideoCapture(source, cv2.CAP_V4L2) if live else cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise ShadowConfigurationError(f"could not open visual source: {value}")
    return capture


def _video_time(capture: Any, frame_idx: int, fps: float) -> float:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise _vision_dependency_error(error) from error

    raw = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if math.isfinite(raw) and raw >= 0.0:
        result = raw / 1000.0
        if result > 0.0 or frame_idx == 1:
            return result
    return max(0.0, (frame_idx - 1) / max(fps, 1.0))


def _iter_replay_steps(path: Path) -> Iterable[tuple[int, float, Any, Any]]:
    from dataclasses import replace

    try:
        from cr_bot.replay.cache import ReplayCacheReader
    except ModuleNotFoundError as error:
        raise _vision_dependency_error(error) from error

    for record in ReplayCacheReader(path):
        frame = record.decode_frame()
        analysis = record.analysis
        if getattr(analysis, "yolo_boxes", None) is None:
            analysis = replace(analysis, yolo_boxes=[])
        yield int(record.frame_idx), float(record.video_time_s), frame, analysis


def _iter_video_steps(
    path_or_device: str | Path | int,
    *,
    video_start_time_s: float | None,
    video_end_time_s: float | None,
    sample_interval_s: float,
    frame_stride: int,
    normalize: bool,
    detector: Any | None,
    yolo_detections: bool,
    live: bool,
    max_seconds: float | None,
) -> Iterable[tuple[int, float, Any, Any]]:
    try:
        import cv2
        from cr_bot.app.pipeline import normalize_frame, process_frame
    except ModuleNotFoundError as error:
        raise _vision_dependency_error(error) from error

    if detector is None:
        from cr_bot.vision.yolo_runtime import build_detector

        detector = build_detector()
    capture = _video_capture_source(path_or_device, live=live)
    session_clock = time.monotonic()
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if video_start_time_s is not None and not live:
        capture.set(cv2.CAP_PROP_POS_MSEC, video_start_time_s * 1000.0)
    frame_idx = 0
    next_sample = 0.0 if video_start_time_s is None else float(video_start_time_s)
    duration_origin_s = 0.0 if video_start_time_s is None else float(video_start_time_s)
    try:
        while True:
            if live and max_seconds is not None and time.monotonic() - session_clock >= max_seconds:
                break
            ok, frame = capture.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx > 1 and frame_stride > 1 and (frame_idx - 1) % frame_stride:
                continue
            timestamp = (
                time.monotonic() - session_clock
                if live
                else _video_time(capture, frame_idx, fps)
            )
            if not live:
                if video_start_time_s is not None and timestamp + 1e-9 < video_start_time_s:
                    continue
                if video_end_time_s is not None and timestamp > video_end_time_s + 1e-9:
                    break
                if timestamp + 1e-9 < next_sample:
                    continue
                intervals_elapsed = max(
                    1,
                    int((timestamp - next_sample) // sample_interval_s) + 1,
                )
                next_sample += intervals_elapsed * sample_interval_s
            elif timestamp + 1e-9 < next_sample:
                continue
            else:
                intervals_elapsed = max(
                    1,
                    int((timestamp - next_sample) // sample_interval_s) + 1,
                )
                next_sample += intervals_elapsed * sample_interval_s
            if (
                max_seconds is not None
                and timestamp - duration_origin_s > max_seconds + 1e-9
            ):
                break
            analyzed = normalize_frame(frame) if normalize else frame
            analysis = process_frame(
                analyzed,
                detector,
                show_rois=False,
                yolo_tower_hp_detections=yolo_detections,
            )
            yield frame_idx, timestamp, analyzed, analysis
    finally:
        capture.release()


def run_shadow_media(
    checkpoint: str | Path,
    *,
    video: str | Path | None = None,
    replay_cache: str | Path | None = None,
    video_device: str | int | None = None,
    sample_interval_s: float = 0.1,
    frame_stride: int = 1,
    video_start_time_s: float | None = None,
    video_end_time_s: float | None = None,
    normalize: bool = True,
    yolo_detections: bool = False,
    max_frames: int | None = None,
    max_seconds: float | None = None,
    device: str | None = None,
    allow_stale_ruleset: bool = False,
    detector: Any | None = None,
    runner: Any | None = None,
    source_factory: Callable[..., Iterable[tuple[Any, ...]]] | None = None,
    trace_path: str | Path | None = None,
) -> dict[str, object]:
    """Run public-only deterministic inference over one visual source."""

    if type(allow_stale_ruleset) is not bool:
        raise ShadowConfigurationError("allow_stale_ruleset must be boolean")

    _positive_int("frame_stride", frame_stride)
    sample_interval_s = _finite_nonnegative("sample_interval_s", sample_interval_s)
    if sample_interval_s <= 0.0:
        raise ShadowConfigurationError("sample_interval_s must be positive")
    if video_start_time_s is not None:
        _finite_nonnegative("video_start_time_s", video_start_time_s)
    if video_end_time_s is not None:
        _finite_nonnegative("video_end_time_s", video_end_time_s)
    if (
        video_start_time_s is not None
        and video_end_time_s is not None
        and video_end_time_s <= video_start_time_s
    ):
        raise ShadowConfigurationError("video_end_time_s must be greater than video_start_time_s")
    if max_frames is not None:
        _positive_int("max_frames", max_frames)
    if max_seconds is not None:
        max_seconds = _finite_nonnegative("max_seconds", max_seconds)
        if max_seconds <= 0.0:
            raise ShadowConfigurationError("max_seconds must be positive")

    source_kind, source, source_hash = _source_spec(
        video=video,
        replay_cache=replay_cache,
        video_device=video_device,
        allow_missing_files=source_factory is not None,
    )
    if source_kind == "video-device" and video_start_time_s is not None:
        raise ShadowConfigurationError("video_start_time_s is only valid for MP4 input")
    if source_kind == "video-device" and video_end_time_s is not None:
        raise ShadowConfigurationError("video_end_time_s is only valid for MP4 input")

    metadata: Mapping[str, object]
    if runner is None:
        from .prototype import load_shadow_prototype_checkpoint

        learner, _config, metadata = load_shadow_prototype_checkpoint(
            checkpoint,
            device=device,
            allow_stale_ruleset=allow_stale_ruleset,
        )
        # Keep construction to the required learner-only contract.  The
        # source is already recorded in the run report, and this also keeps
        # the media orchestration seam usable with small injected runners.
        runner = ShadowPolicyRunner(learner)
    else:
        metadata = {"checkpoint_format": "injected-shadow-runner"}

    session = None
    if source_factory is None:
        try:
            from cr_bot.app.match_session import MatchSession
        except ModuleNotFoundError as error:
            raise _vision_dependency_error(error) from error

        session = MatchSession()
    predictions: list[dict[str, object]] = []
    frames_seen = 0
    in_game_frames = 0
    interrupted = False
    duration_origin_s: float | None = None

    def consume_analysis(frame_idx: int, time_s: float, analysis: Any, frame: Any) -> None:
        nonlocal frames_seen, in_game_frames
        frames_seen += 1
        assert session is not None
        step = session.process(analysis, frame=frame, now_s=time_s)
        consume_step(frame_idx, time_s, step)

    def consume_step(frame_idx: int, time_s: float, step: Any) -> None:
        nonlocal in_game_frames
        if bool(getattr(step, "in_game", False)):
            in_game_frames += 1
        if hasattr(runner, "process_step"):
            decision = runner.process_step(
                step,
                frame_index=frame_idx,
                timestamp_s=time_s,
            )
            if decision is not None:
                predictions.append(dict(decision))
            return
        prediction = runner.consume(step, frame_idx=frame_idx, time_s=time_s)
        if prediction is not None:
            predictions.append(prediction.to_dict())

    try:
        if source_factory is not None:
            for item in source_factory(source_kind, source):
                if not isinstance(item, tuple) or len(item) not in {3, 4}:
                    raise ShadowConfigurationError(
                        "shadow source_factory items must be (frame, time, step) "
                        "or (frame, time, frame_data, analysis)"
                    )
                frame_idx, time_s = item[0], item[1]
                if duration_origin_s is None:
                    duration_origin_s = float(time_s)
                if (
                    max_seconds is not None
                    and float(time_s) - duration_origin_s > max_seconds + 1e-9
                ):
                    break
                if len(item) == 3:
                    frames_seen += 1
                    consume_step(int(frame_idx), float(time_s), item[2])
                else:
                    consume_analysis(int(frame_idx), float(time_s), item[3], item[2])
                if max_frames is not None and frames_seen >= max_frames:
                    break
        elif source_kind == "replay-cache":
            for frame_idx, time_s, frame, analysis in _iter_replay_steps(Path(source)):
                if duration_origin_s is None:
                    duration_origin_s = float(time_s)
                if (
                    max_seconds is not None
                    and float(time_s) - duration_origin_s > max_seconds + 1e-9
                ):
                    break
                consume_analysis(frame_idx, time_s, analysis, frame)
                if max_frames is not None and frames_seen >= max_frames:
                    break
        elif source_kind == "video":
            for frame_idx, time_s, frame, analysis in _iter_video_steps(
                Path(source),
                video_start_time_s=video_start_time_s,
                video_end_time_s=video_end_time_s,
                sample_interval_s=sample_interval_s,
                frame_stride=frame_stride,
                normalize=normalize,
                detector=detector,
                yolo_detections=yolo_detections,
                live=False,
                max_seconds=max_seconds,
            ):
                consume_analysis(frame_idx, time_s, analysis, frame)
                if max_frames is not None and frames_seen >= max_frames:
                    break
        else:
            for frame_idx, time_s, frame, analysis in _iter_video_steps(
                source,
                video_start_time_s=None,
                video_end_time_s=None,
                sample_interval_s=sample_interval_s,
                frame_stride=frame_stride,
                normalize=normalize,
                detector=detector,
                yolo_detections=yolo_detections,
                live=True,
                max_seconds=max_seconds,
            ):
                consume_analysis(frame_idx, time_s, analysis, frame)
                if max_frames is not None and frames_seen >= max_frames:
                    break
    except KeyboardInterrupt:
        interrupted = True

    if source_factory is not None and hasattr(runner, "trace_payload"):
        report = dict(runner.trace_payload())
        report.update(
            {
                "checkpoint": str(checkpoint),
                "source": str(source),
                "source_kind": source_kind,
                "source_sha256": source_hash,
            }
        )
        report.setdefault(
            "checkpoint_validation",
            {
                "mode": "injected",
                "status": "not-checked",
            },
        )
    else:
        checkpoint_ruleset_match = bool(
            metadata.get("_checkpoint_ruleset_match", True)
        )
        stale_ruleset_used = bool(metadata.get("_stale_ruleset_allowed", False))
        checkpoint_validation = {
            "mode": "shadow-stale-allowed" if stale_ruleset_used else "current",
            "status": "stale" if not checkpoint_ruleset_match else "current",
            "checkpoint_ruleset_id": metadata.get(
                "_checkpoint_ruleset_id", metadata.get("ruleset_id")
            ),
            "checkpoint_ruleset_hash": metadata.get(
                "_checkpoint_ruleset_hash", metadata.get("ruleset_hash")
            ),
            "runtime_ruleset_id": metadata.get(
                "_runtime_ruleset_id", metadata.get("ruleset_id")
            ),
            "runtime_ruleset_hash": metadata.get(
                "_runtime_ruleset_hash", metadata.get("ruleset_hash")
            ),
            "hash_match": checkpoint_ruleset_match,
            "stale_checkpoint_used": stale_ruleset_used,
        }
        report = {
            "kind": "recurrent_public_ppo_shadow",
            "checkpoint": str(checkpoint),
            "checkpoint_format": metadata["checkpoint_format"],
            "shadow_schema_version": SHADOW_SCHEMA_VERSION,
            "shadow_runner_version": SHADOW_RUNNER_VERSION,
            "source": str(source),
            "source_kind": source_kind,
            "source_sha256": source_hash,
            "checkpoint_validation": checkpoint_validation,
            "frames_seen": frames_seen,
            "in_game_frames": in_game_frames,
            "emitted_observations": int(getattr(runner, "emitted_observations", 0)),
            "invalid_observations": int(getattr(runner, "invalid_observations", 0)),
            "matches_started": int(getattr(runner, "matches_started", 0)),
            "match_boundaries": int(getattr(runner, "match_boundaries", 0)),
            "interrupted": interrupted,
            "predictions": predictions,
            "actor_privileged_inputs": False,
            # This run intentionally bypasses the critic.  The checkpoint may
            # contain a critic trained with privilege, but it is never called.
            "critic_privileged_inputs": False,
            "checkpoint_critic_privileged_inputs": bool(
                metadata.get("critic_observation", {}).get("privileged_inputs", False)
                if isinstance(metadata.get("critic_observation"), Mapping)
                else False
            ),
            "taps_sent": 0,
            "warning": (
                "Shadow mode only: predictions are logged and cannot change a recorded "
                "match; live V4L2 input also sends no phone actions."
                + (
                    " This checkpoint has an older content hash than the runtime "
                    "ruleset; it was accepted only for this read-only shadow run."
                    if stale_ruleset_used
                    else ""
                )
            ),
        }
    if trace_path is not None:
        destination = Path(trace_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        import json

        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


__all__ = [
    "SHADOW_RUNNER_VERSION",
    "SHADOW_SCHEMA_VERSION",
    "ShadowConfigurationError",
    "ShadowPolicyRunner",
    "ShadowPrediction",
    "run_shadow_media",
]
