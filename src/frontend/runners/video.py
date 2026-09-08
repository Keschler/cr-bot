from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from ..models.session import FrontendSession
from ..services.devices import resolve_inference_devices
from .pump import _OffsetFrameSource, _run_pump_loop


def run_video_session(
    session: FrontendSession,
    *,
    video_path: str,
    checkpoint: str,
    device: str = "auto",
    frame_stride: int = 1,
    start_frame: int = 0,
    max_frames: int | None = None,
    yolo_image_size: int = 896,
    stop_event: threading.Event | None = None,
    poll_interval_s: float = 0.0,
    min_action_interval_s: float = 0.75,
    post_action_delay_s: float = 0.35,
    yolo_tower_hp_detections: bool = False,
    normalize: bool = True,
    adapt_rois: bool = False,
    roi_set: dict | None = None,
) -> dict[str, int]:
    """Pump a recorded video through extraction + policy into ``session``."""

    session.mode = "video"
    session.running = True
    session.error = None
    session.summary = None
    session.actor = None
    session.clear()
    frame_source: Any = None
    try:
        try:
            from simulator.physical_lab.prototype_controller import (
                LiveDetectionFilter,
                LiveHandStateFilter,
                LivePrototypeRunner,
                PrototypeActor,
                VideoFrameSource,
                _bootstrap_extractor_runtime,
                _filter_live_analysis,
                action_to_dict,
                configure_detector_inference_size,
            )
        except ImportError:  # direct-script execution fallback
            from physical_lab.prototype_controller import (  # type: ignore
                LiveDetectionFilter,
                LiveHandStateFilter,
                LivePrototypeRunner,
                PrototypeActor,
                VideoFrameSource,
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
            from simulator.physical_lab.policy_bridge import (
                dispatch_policy_action,
                observation_v2_from_match_step,
            )
        except ImportError:
            from physical_lab.policy_bridge import (  # type: ignore
                dispatch_policy_action,
                observation_v2_from_match_step,
            )

        if max_frames is not None and (
            type(max_frames) is not int or max_frames <= 0
        ):
            raise ValueError("max_frames must be a positive integer when supplied")
        if type(frame_stride) is not int or frame_stride <= 0:
            raise ValueError("frame_stride must be a positive integer")
        if type(start_frame) is not int or start_frame < 0:
            raise ValueError("start_frame must be a non-negative integer")
        if type(yolo_image_size) is not int or yolo_image_size <= 0:
            raise ValueError("yolo_image_size must be a positive integer")
        if type(adapt_rois) is not bool:
            raise ValueError("adapt_rois must be a bool")
        if roi_set is not None and not isinstance(roi_set, dict):
            raise ValueError("roi_set must be a dict or None")

        effective_rois: dict | None = None
        startup_ms: dict[str, float] = {}
        t_startup = time.monotonic()
        if adapt_rois:
            try:
                import cv2  # lazy: keeps module import light
            except ImportError as error:
                raise ValueError("adapt_rois requires OpenCV") from error
            capture = cv2.VideoCapture(str(video_path))
            try:
                if not capture.isOpened():
                    raise ValueError(f"could not open video: {video_path}")
                native_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                native_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            finally:
                try:
                    capture.release()
                except Exception:
                    pass
            if native_w <= 0 or native_h <= 0:
                raise ValueError(f"could not probe video size: {video_path}")
            try:
                from cr_bot.vision.roi_adapt import validate_and_merge
            except ImportError as error:
                raise RuntimeError("roi adaptation runtime is not importable") from error
            effective_rois = validate_and_merge(roi_set, native_w, native_h)
        startup_ms["adapt_probe"] = (time.monotonic() - t_startup) * 1000.0

        t_startup = time.monotonic()
        yolo_device_name, actor_device = resolve_inference_devices(device)
        detector = build_detector(device=yolo_device_name)
        startup_ms["build_detector"] = (time.monotonic() - t_startup) * 1000.0
        t_startup = time.monotonic()
        configure_detector_inference_size(detector, yolo_image_size)
        actor = PrototypeActor(checkpoint, device=actor_device)
        startup_ms["load_actor"] = (time.monotonic() - t_startup) * 1000.0
        # Lend the live actor to what-if re-evaluations (borrowed under
        # session.actor_lock with hidden-state save/restore).
        session.actor = actor
        t_startup = time.monotonic()
        match_session = MatchSession(tracker_debug=False)
        match_session.hand_state_filter = LiveHandStateFilter()
        detection_filter = LiveDetectionFilter()
        frame_source = VideoFrameSource(video_path, frame_stride=frame_stride)
        if start_frame > 0:
            frame_source = _OffsetFrameSource(frame_source, start_frame)
        startup_ms["session_setup"] = (time.monotonic() - t_startup) * 1000.0
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
            execute=False,
            phone=None,
            calibration=None,
            max_frames=max_frames,
            poll_interval_s=poll_interval_s,
            min_action_interval_s=min_action_interval_s,
            post_action_delay_s=post_action_delay_s,
            stop_event=stop_event,
            yolo_tower_hp_detections=yolo_tower_hp_detections,
            normalize=normalize,
            effective_rois=effective_rois,
            adapt_rois_enabled=bool(adapt_rois),
        )
        summary["startup_ms"] = startup_ms
        summary["devices"] = {
            "requested": device,
            "yolo": yolo_device_name,
            "actor": actor_device,
        }
        session.summary = summary
        return summary
    except Exception as error:
        session.error = str(error) or repr(error)
        if session.summary is None:
            session.summary = None
        raise
    finally:
        if frame_source is not None:
            try:
                frame_source.close()
            except Exception:
                pass
        session.running = False
