from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from ..models.session import FrontendSession
from ..services.devices import resolve_inference_devices
from ..services.live_diagnostics import LiveDiagnostics, collect_runtime_diagnostics
from ..services.live_preview import LivePreviewProvider
from .pump import _run_pump_loop


def run_live_session(
    session: FrontendSession,
    *,
    serial: str,
    transport: str = "stream",
    stream_max_width_px: int | None = None,
    checkpoint: str,
    device: str = "auto",
    calibration: str | Path | None = None,
    execute: bool = False,
    confirm_live: bool = False,
    max_frames: int | None = None,
    yolo_image_size: int = 896,
    poll_interval_s: float = 0.0,
    min_action_interval_s: float = 0.75,
    post_action_delay_s: float = 0.35,
    yolo_tower_hp_detections: bool = False,
    normalize: bool = True,
    stop_event: threading.Event | None = None,
    **kwargs: Any,
) -> dict[str, int]:
    """Pump live ADB frames into ``session`` (dry-run unless explicitly armed).

    Never sends taps unless ``execute`` + ``confirm_live`` + ``calibration``
    are all present.
    """

    session.mode = "live"
    session.live_frame_clock = (
        "workstation-decode-publication" if transport == "stream"
        else "workstation-screenshot-receipt"
    )
    session.running = True
    session.error = None
    session.summary = None
    # A previous run's actor is stale once a new run starts; the new actor
    # is lent below after loading. Retained actors keep history revisable
    # after a run finishes (what-if only, never execute).
    session.actor = None
    session.set_live_preview_provider(None)
    session.clear()
    diagnostics = (
        LiveDiagnostics(transport=transport, execute=bool(execute))
        if os.environ.get("CR_BOT_LIVE_DIAGNOSTICS") == "1" else None
    )
    session.live_diagnostics = diagnostics
    frame_source: Any = None
    try:
        if diagnostics is not None:
            diagnostics.start()
        if execute and (calibration is None or not confirm_live):
            raise ValueError(
                "live execution requires calibration and confirm_live=True"
            )
        if not isinstance(serial, str) or not serial.strip():
            raise ValueError("serial must be a non-empty ADB device serial")
        if transport not in ("stream", "screenshot"):
            raise ValueError("transport must be 'stream' or 'screenshot'")
        if stream_max_width_px is not None and (
            type(stream_max_width_px) is not int or stream_max_width_px <= 0
        ):
            raise ValueError("stream_max_width_px must be a positive integer or None")
        if max_frames is not None and (
            type(max_frames) is not int or max_frames <= 0
        ):
            raise ValueError("max_frames must be a positive integer when supplied")
        if type(yolo_image_size) is not int or yolo_image_size <= 0:
            raise ValueError("yolo_image_size must be a positive integer")

        try:
            from simulator.physical_lab.workflow.prototype_controller import (
                AdbH264FrameSource,
                AdbScreenshotSource,
                CachedAdbPhoneController,
                LiveDetectionFilter,
                LiveHandStateFilter,
                LivePrototypeRunner,
                V4Actor,
                _bootstrap_extractor_runtime,
                _filter_live_analysis,
                action_to_dict,
                configure_detector_inference_size,
            )
        except ImportError:
            from physical_lab.workflow.prototype_controller import (  # type: ignore
                AdbH264FrameSource,
                AdbScreenshotSource,
                CachedAdbPhoneController,
                LiveDetectionFilter,
                LiveHandStateFilter,
                LivePrototypeRunner,
                V4Actor,
                _bootstrap_extractor_runtime,
                _filter_live_analysis,
                action_to_dict,
                configure_detector_inference_size,
            )
        _bootstrap_extractor_runtime()
        try:
            from cr_bot.app.match_session import MatchSession
            from cr_bot.app.pipeline import normalize_frame, process_frame
            from cr_bot.vision.yolo_runtime import build_detector
        except ImportError as error:
            raise RuntimeError(
                "the cr_bot visual extractor is not importable"
            ) from error
        try:
            from simulator.physical_lab.sensing.policy_bridge import (
                dispatch_policy_action,
                observation_v2_from_match_step,
            )
        except ImportError:
            from physical_lab.sensing.policy_bridge import (  # type: ignore
                dispatch_policy_action,
                observation_v2_from_match_step,
            )

        adb_executable = str(kwargs.get("adb_executable", "adb"))
        ffmpeg_executable = str(kwargs.get("ffmpeg_executable", "ffmpeg"))
        controller = CachedAdbPhoneController(
            serial.strip(),
            device_label="LIVE",
            adb_executable=adb_executable,
        )
        if transport == "stream":
            capture_kwargs = {"diagnostics_enabled": True} if diagnostics is not None else {}
            frame_source = AdbH264FrameSource(
                controller,
                ffmpeg_executable=ffmpeg_executable,
                stream_max_width_px=stream_max_width_px,
                **capture_kwargs,
            )
            if diagnostics is not None:
                diagnostics.set_capture_probe(lambda: frame_source.diagnostics().as_dict())
                diagnostics.set_phase("capture_start")
            session.set_live_preview_provider(
                LivePreviewProvider(
                    frame_source.latest_frame,
                    normalize_frame=normalize_frame if normalize else None,
                )
            )
            # Drain capture during model warmup and serve fresh preview pixels
            # without consuming the inference delivery cursor.
            frame_source.start()
        else:
            frame_source = AdbScreenshotSource(controller)

        if diagnostics is not None:
            diagnostics.set_phase("model_warmup")

        phone: Any = None
        calibration_obj: Any = None
        if execute:
            # Live taps stay behind calibration + explicit confirmation gates.
            try:
                from simulator.physical_lab.capture.calibration import CalibrationArtifact
                from simulator.physical_lab.workflow.prototype_controller import (
                    _REPOSITORY_ROOT,
                    _default_template_root,
                    _resolve,
                    _validate_live_setup,
                )
            except ImportError:
                from physical_lab.capture.calibration import CalibrationArtifact  # type: ignore
                from physical_lab.workflow.prototype_controller import (  # type: ignore
                    _REPOSITORY_ROOT,
                    _default_template_root,
                    _resolve,
                    _validate_live_setup,
                )
            calibration_path = Path(str(calibration)).expanduser()
            if not calibration_path.is_absolute():
                calibration_path = _REPOSITORY_ROOT / calibration_path
            calibration_obj = CalibrationArtifact.load(calibration_path)
            action_frame_provider = (
                frame_source.frame_for_action
                if transport == "stream"
                else None
            )
            phone, _info = _validate_live_setup(
                controller,
                calibration_obj,
                template_root=_default_template_root(_REPOSITORY_ROOT),
                action_frame_provider=action_frame_provider,
            )

        yolo_device_name, actor_device = resolve_inference_devices(device)
        detector = build_detector(device=yolo_device_name)
        configure_detector_inference_size(detector, yolo_image_size)
        actor = V4Actor(checkpoint, device=actor_device)
        if diagnostics is not None:
            try:
                diagnostics.set_runtime(collect_runtime_diagnostics(
                    requested_device=device,
                    yolo_device=yolo_device_name,
                    policy_device=actor_device,
                    yolo_image_size=yolo_image_size,
                ))
            except Exception:
                logging.getLogger("uvicorn.error").exception("LIVE_DIAGNOSTICS runtime inspection failed")
            diagnostics.set_phase("analyzing")
        # Lend the live actor to what-if re-evaluations (borrowed under
        # session.actor_lock with hidden-state save/restore).
        session.actor = actor
        match_session = MatchSession(tracker_debug=False)
        match_session.hand_state_filter = LiveHandStateFilter()
        detection_filter = LiveDetectionFilter()
        summary = _run_pump_loop(
            frontend_session=session,
            frame_source=frame_source,
            detector=detector,
            actor=actor,
            match_session=match_session,
            observation_builder=observation_v2_from_match_step,
            dispatch_fn=dispatch_policy_action,
            normalize_frame_fn=normalize_frame,
            process_frame_fn=process_frame,
            filter_live_analysis_fn=_filter_live_analysis,
            action_to_dict_fn=action_to_dict,
            record_fn=LivePrototypeRunner._record,
            detection_filter=detection_filter,
            execute=bool(execute),
            phone=phone,
            calibration=calibration_obj,
            max_frames=max_frames,
            poll_interval_s=poll_interval_s,
            min_action_interval_s=min_action_interval_s,
            post_action_delay_s=post_action_delay_s,
            stop_event=stop_event,
            yolo_tower_hp_detections=yolo_tower_hp_detections,
            normalize=normalize,
        )
        summary["devices"] = {
            "requested": device,
            "yolo": yolo_device_name,
            "actor": actor_device,
        }
        session.summary = summary
        return summary
    except Exception as error:
        # Fail closed: record the error for the UI; never half-arm taps.
        if isinstance(error, ValueError):
            session.error = str(error) or repr(error)
            raise
        session.error = str(error) or repr(error)
        raise
    finally:
        session.actor = None
        session.set_live_preview_provider(None)
        if frame_source is not None:
            try:
                frame_source.close()
            except Exception:
                pass
        session.running = False
        if diagnostics is not None:
            diagnostics.close(error=session.error)
