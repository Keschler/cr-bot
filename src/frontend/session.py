"""Frontend session state and frame-pump workers (lightweight import surface).

No heavy imports (torch/cv2/cr_bot) at module top.  All heavy runtime
imports happen lazily inside functions so the UI server can be imported
without GPU/CV dependencies installed.
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FrontendFrame:
    """One UI-facing frame with decision payload and optional JPEG preview."""

    frame_index: int
    timestamp_s: float
    jpeg_bytes: bytes | None
    record: dict = field(default_factory=dict)
    suggestions: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    in_game: bool = False
    emitted: bool = False
    # Tracker events newly confirmed on this frame (timeline markers).
    own_actions: list[dict] = field(default_factory=list)
    enemy_plays: list[dict] = field(default_factory=list)
    # Pixel dimensions of the coordinate space that detection ``center_px``
    # values refer to (the normalized frame passed to the extractor, before
    # JPEG downscaling). The UI uses these to map overlays onto the preview.
    frame_width: int | None = None
    frame_height: int | None = None


def _frame_dimensions(image: Any) -> tuple[int | None, int | None]:
    shape = getattr(image, "shape", None)
    try:
        if shape is not None and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])
            if width > 0 and height > 0:
                return width, height
    except (TypeError, ValueError):
        pass
    return None, None


class FrontendSession:
    """Thread-safe bounded store of recent :class:`FrontendFrame` objects."""

    def __init__(self, *, mode: str = "idle", maxlen: int = 512) -> None:
        self._lock = threading.Lock()
        self._frames: collections.deque[FrontendFrame] = collections.deque(
            maxlen=maxlen
        )
        self.latest: FrontendFrame | None = None
        self.running: bool = False
        self.mode: str = mode
        self.error: str | None = None
        self.summary: dict | None = None

    @property
    def history(self) -> list[FrontendFrame]:
        with self._lock:
            return list(self._frames)

    @property
    def maxlen(self) -> int | None:
        return self._frames.maxlen

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self.latest = None

    def push(self, frame: FrontendFrame) -> None:
        with self._lock:
            self._frames.append(frame)
            self.latest = frame

    def get_latest(self) -> FrontendFrame | None:
        with self._lock:
            return self.latest

    def get_since(
        self, since: int = 0, limit: int | None = 50
    ) -> list[FrontendFrame]:
        """Return frames with ``frame_index`` strictly greater than ``since``.

        Results are chronological (oldest first).  When ``limit`` is set,
        at most ``limit`` frames are returned.
        """

        try:
            since_int = int(since)
        except (TypeError, ValueError):
            since_int = 0
        with self._lock:
            matched = [f for f in self._frames if int(f.frame_index) > since_int]
        if limit is not None:
            try:
                limit_int = int(limit)
            except (TypeError, ValueError):
                limit_int = 50
            if limit_int is not None and limit_int >= 0:
                matched = matched[:limit_int]
        return matched

    def to_status_dict(self) -> dict[str, Any]:
        with self._lock:
            latest = self.latest
            count = len(self._frames)
            running = self.running
            mode = self.mode
            error = self.error
            summary = dict(self.summary) if isinstance(self.summary, dict) else None
        if latest is not None:
            latest_info: dict[str, Any] | None = {
                "frame_index": latest.frame_index,
                "timestamp_s": latest.timestamp_s,
                "in_game": latest.in_game,
                "emitted": latest.emitted,
                "has_image": latest.jpeg_bytes is not None,
                "record": latest.record,
                "suggestions": latest.suggestions,
                "diagnostics": latest.diagnostics,
                "frame_width": latest.frame_width,
                "frame_height": latest.frame_height,
                "own_actions": latest.own_actions,
                "enemy_plays": latest.enemy_plays,
            }
            latest_index: int | None = latest.frame_index
        else:
            latest_info = None
            latest_index = None
        return {
            "running": running,
            "mode": mode,
            "error": error,
            "summary": summary,
            "frames": count,
            "frame_count": count,
            "latest_frame_index": latest_index,
            "latest": latest_info,
        }


def encode_jpeg(bgr: Any, max_width: int = 720) -> bytes | None:
    """Encode a BGR image to JPEG bytes, downscaling to ``max_width``.

    Returns ``None`` when OpenCV is unavailable or encoding fails.  Never
    raises for UI-path robustness.
    """

    if bgr is None:
        return None
    try:
        import cv2  # lazy: keeps module import light
    except ImportError:
        return None
    try:
        height_width = getattr(bgr, "shape", None)
        if height_width is None or len(height_width) < 2:
            return None
        height, width = int(height_width[0]), int(height_width[1])
        if width <= 0 or height <= 0:
            return None
        image = bgr
        if isinstance(max_width, int) and max_width > 0 and width > max_width:
            scale = max_width / float(width)
            new_height = max(1, int(round(height * scale)))
            image = cv2.resize(
                bgr, (int(max_width), new_height), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok or buf is None:
            return None
        return bytes(buf.tobytes())
    except Exception:
        return None


def _try_import_decide_with_scores() -> Any | None:
    """Import the parallel-worker's scoring entry point, if present."""

    try:
        from .scoring import decide_with_scores  # type: ignore

        if callable(decide_with_scores):
            return decide_with_scores
    except Exception:
        pass
    try:
        from frontend.scoring import decide_with_scores  # type: ignore

        if callable(decide_with_scores):
            return decide_with_scores
    except Exception:
        pass
    return None


def _suggestion_to_dict(suggestion: Any) -> dict[str, Any]:
    if isinstance(suggestion, dict):
        out = dict(suggestion)
        cell = out.get("cell")
        if isinstance(cell, (tuple, list)):
            out["cell"] = [int(v) for v in cell]
        return out
    cell = getattr(suggestion, "cell", None)
    if isinstance(cell, (tuple, list)) and len(cell) == 2:
        try:
            cell_out = [int(cell[0]), int(cell[1])]
        except (TypeError, ValueError):
            cell_out = None
    else:
        cell_out = None
    probability = getattr(suggestion, "probability", None)
    log_prob = getattr(suggestion, "log_prob", None)
    try:
        probability = float(probability) if probability is not None else None
    except (TypeError, ValueError):
        probability = None
    try:
        log_prob = float(log_prob) if log_prob is not None else None
    except (TypeError, ValueError):
        log_prob = None
    card_slot = getattr(suggestion, "card_slot", None)
    try:
        card_slot = int(card_slot) if card_slot is not None else None
    except (TypeError, ValueError):
        pass
    return {
        "kind": getattr(suggestion, "kind", None),
        "card_slot": card_slot,
        "cell": cell_out,
        "probability": probability,
        "log_prob": log_prob,
        "card_name": getattr(suggestion, "card_name", None),
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    import math

    return result if math.isfinite(result) else None


def _cell_to_list(cell: Any) -> list[int] | None:
    if isinstance(cell, (tuple, list)) and len(cell) == 2:
        try:
            return [int(cell[0]), int(cell[1])]
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def _summarize_own_action(event: Any) -> dict[str, Any] | None:
    """Summarize one OwnActionEvent for the UI timeline (fail-soft)."""
    try:
        card = getattr(event, "card", None)
        if not isinstance(card, str) or not card.strip():
            return None
        slot = getattr(event, "slot_idx", None)
        return {
            "card": card,
            "slot_idx": int(slot) if isinstance(slot, int) else None,
            "cell": _cell_to_list(getattr(event, "cell", None)),
            "video_time_s": _finite_or_none(getattr(event, "video_time_s", None)),
            "time_left_s": _finite_or_none(getattr(event, "time_left_s", None)),
            "played_via": getattr(event, "played_via", None),
        }
    except (TypeError, ValueError, AttributeError):
        return None


def _summarize_enemy_play(play: Any) -> dict[str, Any] | None:
    """Summarize one confirmed EnemyCardPlay for the UI timeline (fail-soft)."""
    try:
        if not bool(getattr(play, "clock_confirmed", False)) and not bool(
            getattr(play, "frame_confirmed", False)
        ):
            return None
        card = getattr(play, "card", None)
        if not isinstance(card, str) or not card.strip():
            return None
        event_id = getattr(play, "event_id", None)
        cost = getattr(play, "cost", None)
        track_id = getattr(play, "track_id", None)
        return {
            "event_id": str(event_id) if event_id is not None else None,
            "card": card,
            "cost": int(cost) if isinstance(cost, int) else None,
            "cell": _cell_to_list(getattr(play, "cell", None)),
            "track_id": int(track_id) if isinstance(track_id, int) else None,
            "video_time_s": _finite_or_none(getattr(play, "video_time_s", None)),
            "time_left_s": _finite_or_none(getattr(play, "time_left_s", None)),
            "clock_confirmed": bool(getattr(play, "clock_confirmed", False)),
            "frame_confirmed": bool(getattr(play, "frame_confirmed", False)),
            "avg_confidence": _finite_or_none(getattr(play, "avg_confidence", None)),
            "is_spell": bool(getattr(play, "is_spell", False)),
            "played_via": getattr(play, "played_via", None),
        }
    except (TypeError, ValueError, AttributeError):
        return None


def _new_tracker_actions(
    match_session: Any,
    *,
    own_tracker_seen: Any,
    own_baseline: int,
    enemy_seen_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, int, set[str]]:
    """Diff tracker event lists since the previous frame (fail-soft).

    Own actions are append-only per tracker instance; a replaced tracker
    (match reset) restarts the baseline. Enemy plays are keyed by event_id
    because reconciliation can prune the list.
    """
    new_own: list[dict[str, Any]] = []
    new_enemy: list[dict[str, Any]] = []
    try:
        own_tracker = getattr(match_session, "own_action_tracker", None)
        actions = getattr(own_tracker, "actions", None)
        if own_tracker is not own_tracker_seen or not isinstance(actions, list):
            own_tracker_seen, own_baseline = own_tracker, 0
        if isinstance(actions, list):
            if own_baseline < 0:
                own_baseline = 0
            for event in actions[own_baseline:]:
                summary = _summarize_own_action(event)
                if summary is not None:
                    new_own.append(summary)
            own_baseline = len(actions)
    except (TypeError, AttributeError):
        pass
    try:
        enemy_tracker = getattr(match_session, "enemy_card_tracker", None)
        plays = getattr(enemy_tracker, "detected_card_plays", None)
        if isinstance(plays, list):
            current_ids: set[str] = set()
            for play in plays:
                summary = _summarize_enemy_play(play)
                if summary is None:
                    continue
                play_id = summary.get("event_id")
                key = play_id if isinstance(play_id, str) else repr(play)
                current_ids.add(key)
                if key not in enemy_seen_ids:
                    new_enemy.append(summary)
            enemy_seen_ids |= current_ids
    except (TypeError, AttributeError):
        pass
    return new_own, new_enemy, own_tracker_seen, own_baseline, enemy_seen_ids


def _decide_with_fallback(
    actor: Any,
    observation: Any,
    decide_with_scores: Any | None,
    action_to_dict: Any,
) -> tuple[Any, list[dict], dict]:
    """Run scored decoding when available, else plain ``actor.decide``."""

    if decide_with_scores is not None:
        try:
            scored: Any = None
            try:
                scored = decide_with_scores(actor, observation)
            except TypeError:
                # Alternate worker signature: observation-only.
                scored = decide_with_scores(observation)
            if isinstance(scored, tuple) and len(scored) == 3:
                action_obj, suggestions, diagnostics = scored
                suggestion_dicts = [
                    _suggestion_to_dict(s)
                    for s in (suggestions if isinstance(suggestions, (list, tuple)) else [])
                ]
                diag = (
                    dict(diagnostics)
                    if isinstance(diagnostics, dict)
                    else ({} if diagnostics is None else {"value": diagnostics})
                )
                return action_obj, suggestion_dicts, diag
        except Exception:
            pass
    action_obj = actor.decide(observation)
    return action_obj, [], {}


def _run_pump_loop(
    *,
    frontend_session: FrontendSession,
    frame_source: Any,
    detector: Any,
    actor: Any,
    match_session: Any,
    observation_builder: Any,
    dispatch_fn: Any,
    normalize_frame_fn: Any,
    process_frame_fn: Any,
    filter_live_analysis_fn: Any,
    action_to_dict_fn: Any,
    record_fn: Any,
    detection_filter: Any,
    execute: bool,
    phone: Any,
    calibration: Any,
    max_frames: int | None,
    poll_interval_s: float,
    min_action_interval_s: float,
    post_action_delay_s: float,
    stop_event: threading.Event | None,
    yolo_tower_hp_detections: bool = False,
    normalize: bool = True,
    effective_rois: dict | None = None,
    adapt_rois_enabled: bool = False,
) -> dict[str, int]:
    decide_with_scores = _try_import_decide_with_scores()
    frames = emitted = waits = proposed_plays = dispatched_plays = 0
    last_play_timestamp_s: float | None = None
    last_lobby_push_monotonic = 0.0
    own_tracker_seen: Any = None
    own_baseline = 0
    enemy_seen_ids: set[str] = set()
    try:
        poll_interval = float(poll_interval_s or 0.0)
    except (TypeError, ValueError):
        poll_interval = 0.0
    try:
        while max_frames is None or frames < max_frames:
            if stop_event is not None and stop_event.is_set():
                break
            started = time.monotonic()
            timing_ms: dict[str, float] = {}
            t_stage = time.monotonic()
            source_frame = frame_source.next_frame()
            timing_ms["fetch"] = (time.monotonic() - t_stage) * 1000.0
            if source_frame is None:
                break
            frames += 1
            native_image = source_frame.image
            image = native_image
            if normalize:
                t_stage = time.monotonic()
                image = normalize_frame_fn(native_image)
                timing_ms["normalize"] = (time.monotonic() - t_stage) * 1000.0
            t_stage = time.monotonic()
            if adapt_rois_enabled and effective_rois is not None:
                analysis = process_frame_fn(
                    image,
                    detector,
                    show_rois=False,
                    yolo_tower_hp_detections=yolo_tower_hp_detections,
                    rois=effective_rois,
                    native_frame=native_image,
                )
            else:
                analysis = process_frame_fn(
                    image,
                    detector,
                    show_rois=False,
                    yolo_tower_hp_detections=yolo_tower_hp_detections,
                    rois=None,
                    native_frame=None,
                )
            timing_ms["process_frame"] = (time.monotonic() - t_stage) * 1000.0
            t_stage = time.monotonic()
            if filter_live_analysis_fn is not None:
                analysis = filter_live_analysis_fn(analysis)
            if detection_filter is not None:
                update = getattr(detection_filter, "update", None)
                if callable(update):
                    analysis = update(
                        analysis, timestamp_s=source_frame.timestamp_s
                    )
            timing_ms["filter"] = (time.monotonic() - t_stage) * 1000.0
            t_stage = time.monotonic()
            step = match_session.process(
                analysis,
                frame=image,
                now_s=source_frame.timestamp_s,
            )
            timing_ms["match"] = (time.monotonic() - t_stage) * 1000.0
            in_game = bool(getattr(step, "in_game", False))
            should_emit = bool(getattr(step, "should_emit", False))
            if (
                detection_filter is not None
                and not in_game
                and hasattr(detection_filter, "reset")
            ):
                try:
                    detection_filter.reset()
                except Exception:
                    pass
            action_json: dict[str, Any] | None = None
            suggestions: list[dict] = []
            diagnostics: dict[str, Any] = {}
            if not in_game or not should_emit:
                try:
                    actor.reset()
                except Exception:
                    pass
                result = "not-in-game"
            else:
                emitted += 1
                hand_filter = getattr(match_session, "hand_state_filter", None)
                ready = bool(getattr(hand_filter, "ready", True))
                if not ready:
                    try:
                        actor.reset()
                    except Exception:
                        pass
                    action_json = {"kind": "wait"}
                    waits += 1
                    result = "hand-not-stable"
                else:
                    t_stage = time.monotonic()
                    observation = observation_builder(step)
                    timing_ms["observe"] = (time.monotonic() - t_stage) * 1000.0
                    if observation is None:
                        try:
                            actor.reset()
                        except Exception:
                            pass
                        result = "observation-not-ready"
                    else:
                        t_stage = time.monotonic()
                        action_obj, suggestions, diagnostics = _decide_with_fallback(
                            actor,
                            observation,
                            decide_with_scores,
                            action_to_dict_fn,
                        )
                        timing_ms["decide"] = (time.monotonic() - t_stage) * 1000.0
                        action_json = action_to_dict_fn(action_obj)
                        if action_json.get("kind") == "wait":
                            waits += 1
                            result = "wait"
                        else:
                            proposed_plays += 1
                            enough_time = (
                                last_play_timestamp_s is None
                                or source_frame.timestamp_s - last_play_timestamp_s
                                >= min_action_interval_s
                            )
                            if not enough_time:
                                result = "cooldown"
                            elif not execute:
                                result = "dry-run"
                            else:
                                dispatch_fn(
                                    phone,
                                    action_obj,
                                    step.game_state,
                                    calibration=calibration,
                                    observation=observation,
                                )
                                # Mirror live seeding so the policy sees its own
                                # dispatched card while the detector catches up.
                                try:
                                    hand_filter.expect_replacement(
                                        getattr(action_obj, "card_idx", -1)
                                    )
                                except Exception:
                                    pass
                                try:
                                    hud = getattr(
                                        getattr(step, "game_state", None), "hud", None
                                    )
                                    hand_cards = getattr(hud, "hand_cards", ())
                                    slot = getattr(action_obj, "card_idx", None)
                                    card_name = None
                                    if (
                                        isinstance(hand_cards, (list, tuple))
                                        and type(slot) is int
                                        and 0 <= slot < len(hand_cards)
                                    ):
                                        card_name = hand_cards[slot]
                                    detection_filter.notify_own_play(
                                        card_name=card_name,
                                        cell=getattr(action_obj, "cell", None),
                                        arena_px=getattr(analysis, "arena_px", None),
                                        timestamp_s=source_frame.timestamp_s,
                                    )
                                except Exception:
                                    pass
                                last_play_timestamp_s = source_frame.timestamp_s
                                dispatched_plays += 1
                                result = "dispatched"
                                if post_action_delay_s:
                                    if stop_event is not None:
                                        stop_event.wait(post_action_delay_s)
                                    else:
                                        time.sleep(post_action_delay_s)
            t_stage = time.monotonic()
            record_obj = record_fn(
                source_frame, step, action=action_json, result=result
            )
            try:
                record_dict = record_obj.as_dict()
            except Exception:
                record_dict = {
                    "frame_index": source_frame.frame_index,
                    "timestamp_s": source_frame.timestamp_s,
                    "in_game": in_game,
                    "emitted": should_emit,
                    "action": action_json,
                    "result": result,
                }
            timing_ms["record"] = (time.monotonic() - t_stage) * 1000.0
            t_stage = time.monotonic()
            jpeg_bytes = encode_jpeg(image)
            timing_ms["encode"] = (time.monotonic() - t_stage) * 1000.0
            timing_ms["total"] = (time.monotonic() - started) * 1000.0
            if isinstance(diagnostics, dict):
                diagnostics["timing_ms"] = dict(timing_ms)
            frame_width, frame_height = _frame_dimensions(image)
            new_own, new_enemy, own_tracker_seen, own_baseline, enemy_seen_ids = (
                _new_tracker_actions(
                    match_session,
                    own_tracker_seen=own_tracker_seen,
                    own_baseline=own_baseline,
                    enemy_seen_ids=enemy_seen_ids,
                )
            )
            frontend_frame = FrontendFrame(
                frame_index=int(source_frame.frame_index),
                timestamp_s=float(source_frame.timestamp_s),
                jpeg_bytes=jpeg_bytes,
                record=record_dict if isinstance(record_dict, dict) else {},
                suggestions=suggestions,
                diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
                in_game=in_game,
                emitted=should_emit,
                frame_width=frame_width,
                frame_height=frame_height,
                own_actions=new_own,
                enemy_plays=new_enemy,
            )
            if should_emit:
                frontend_session.push(frontend_frame)
            else:
                now_mono = time.monotonic()
                if now_mono - last_lobby_push_monotonic >= 1.0:
                    frontend_session.push(frontend_frame)
                    last_lobby_push_monotonic = now_mono
            elapsed = time.monotonic() - started
            if poll_interval:
                remaining = poll_interval - elapsed
                if remaining > 0:
                    if stop_event is not None:
                        stop_event.wait(remaining)
                    else:
                        time.sleep(remaining)
    finally:
        pass
    return {
        "frames": frames,
        "emitted_frames": emitted,
        "waits": waits,
        "proposed_plays": proposed_plays,
        "dispatched_plays": dispatched_plays,
    }


class _OffsetFrameSource:
    """Wrap a video FrameSource, discarding frames before ``start_frame``.

    Source frame indices and timestamps are preserved, so the timeline still
    shows real video positions. EOF during the skip ends the session with
    zero processed frames instead of failing.
    """

    def __init__(self, inner: Any, start_frame: int) -> None:
        self._inner = inner
        self._start_frame = start_frame
        self._primed = False
        self._pending: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_inner"], name)

    def next_frame(self) -> Any:
        if not self._primed:
            self._primed = True
            while True:
                candidate = self._inner.next_frame()
                if candidate is None or int(candidate.frame_index) >= self._start_frame:
                    self._pending = candidate
                    break
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return frame
        return self._inner.next_frame()

    def close(self) -> None:
        try:
            self._inner.close()
        except (AttributeError, OSError, RuntimeError):
            pass


def run_video_session(
    session: FrontendSession,
    *,
    video_path: str,
    checkpoint: str,
    device: str = "cpu",
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
        detector = build_detector()
        startup_ms["build_detector"] = (time.monotonic() - t_startup) * 1000.0
        t_startup = time.monotonic()
        configure_detector_inference_size(detector, yolo_image_size)
        actor = PrototypeActor(checkpoint, device=device)
        startup_ms["load_actor"] = (time.monotonic() - t_startup) * 1000.0
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


def run_live_session(
    session: FrontendSession,
    *,
    serial: str,
    transport: str = "stream",
    checkpoint: str,
    device: str = "cpu",
    calibration: str | Path | None = None,
    execute: bool = False,
    confirm_live: bool = False,
    max_frames: int | None = None,
    yolo_image_size: int = 896,
    poll_interval_s: float = 0.25,
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
    session.running = True
    session.error = None
    session.summary = None
    session.clear()
    frame_source: Any = None
    try:
        if execute and (calibration is None or not confirm_live):
            raise ValueError(
                "live execution requires calibration and confirm_live=True"
            )
        if not isinstance(serial, str) or not serial.strip():
            raise ValueError("serial must be a non-empty ADB device serial")
        if transport not in ("stream", "screenshot"):
            raise ValueError("transport must be 'stream' or 'screenshot'")
        if max_frames is not None and (
            type(max_frames) is not int or max_frames <= 0
        ):
            raise ValueError("max_frames must be a positive integer when supplied")
        if type(yolo_image_size) is not int or yolo_image_size <= 0:
            raise ValueError("yolo_image_size must be a positive integer")

        try:
            from simulator.physical_lab.prototype_controller import (
                AdbH264FrameSource,
                AdbScreenshotSource,
                CachedAdbPhoneController,
                LiveDetectionFilter,
                LiveHandStateFilter,
                LivePrototypeRunner,
                PrototypeActor,
                _bootstrap_extractor_runtime,
                _filter_live_analysis,
                action_to_dict,
                configure_detector_inference_size,
            )
        except ImportError:
            from physical_lab.prototype_controller import (  # type: ignore
                AdbH264FrameSource,
                AdbScreenshotSource,
                CachedAdbPhoneController,
                LiveDetectionFilter,
                LiveHandStateFilter,
                LivePrototypeRunner,
                PrototypeActor,
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

        adb_executable = str(kwargs.get("adb_executable", "adb"))
        ffmpeg_executable = str(kwargs.get("ffmpeg_executable", "ffmpeg"))
        controller = CachedAdbPhoneController(
            serial.strip(),
            device_label="LIVE",
            adb_executable=adb_executable,
        )
        if transport == "stream":
            frame_source = AdbH264FrameSource(
                controller,
                ffmpeg_executable=ffmpeg_executable,
            )
        else:
            frame_source = AdbScreenshotSource(controller)

        phone: Any = None
        calibration_obj: Any = None
        if execute:
            # Live taps stay behind calibration + explicit confirmation gates.
            try:
                from simulator.physical_lab.calibration import CalibrationArtifact
                from simulator.physical_lab.prototype_controller import (
                    _REPOSITORY_ROOT,
                    _default_template_root,
                    _resolve,
                    _validate_live_setup,
                )
            except ImportError:
                from physical_lab.calibration import CalibrationArtifact  # type: ignore
                from physical_lab.prototype_controller import (  # type: ignore
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

        detector = build_detector()
        configure_detector_inference_size(detector, yolo_image_size)
        actor = PrototypeActor(checkpoint, device=device)
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
        if frame_source is not None:
            try:
                frame_source.close()
            except Exception:
                pass
        session.running = False


__all__ = [
    "FrontendFrame",
    "FrontendSession",
    "encode_jpeg",
    "run_live_session",
    "run_video_session",
]
