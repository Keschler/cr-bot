from __future__ import annotations

import threading
import time
from typing import Any

from ..imaging import _frame_dimensions, encode_jpeg
from ..models.frames import FrontendFrame
from ..models.session import FrontendSession


def _try_import_decide_with_scores() -> Any | None:
    """Import the parallel-worker's scoring entry point, if present."""

    try:
        from ..scoring import decide_with_scores  # type: ignore

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


def _detection_row(match: Any) -> dict[str, Any] | None:
    """Summarize one analysis match as an editable detection row (fail-soft).

    Keeps the raw troop fields the label-correction flow needs to rebuild a
    what-if observation: box, class, team, confidence, track id, HP, center.
    """
    try:
        troop = getattr(match, "troop", None)
        if troop is None:
            return None
        class_name = getattr(troop, "class_name", None)
        team = getattr(troop, "team", None)
        if not isinstance(class_name, str) or not class_name.strip():
            return None
        if not isinstance(team, str) or not team.strip():
            return None
        track_id = getattr(troop, "track_id", None)
        try:
            track_id = int(track_id) if track_id is not None else None
        except (TypeError, ValueError, OverflowError):
            track_id = str(track_id)
        try:
            confidence = float(getattr(troop, "confidence", 0.0))
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        try:
            x1 = float(getattr(troop, "x1"))
            y1 = float(getattr(troop, "y1"))
            x2 = float(getattr(troop, "x2"))
            y2 = float(getattr(troop, "y2"))
            center_x = float(getattr(troop, "center_x"))
            center_y = float(getattr(troop, "center_y"))
        except (TypeError, ValueError, OverflowError):
            return None
        import math

        values = (confidence, x1, y1, x2, y2, center_x, center_y)
        if not all(math.isfinite(v) for v in values):
            return None
        try:
            hp_raw = getattr(troop, "estimated_hp", None)
            hp = float(hp_raw) if hp_raw is not None else None
            if hp is not None and not math.isfinite(hp):
                hp = None
        except (TypeError, ValueError, OverflowError):
            hp = None
        return {
            "class_name": class_name,
            "team": team,
            "confidence": confidence,
            "track_id": track_id,
            "box": [x1, y1, x2, y2],
            "center": [center_x, center_y],
            "estimated_hp": hp,
        }
    except AttributeError:
        return None


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
                        # Serialized with what-if re-evaluations borrowing the
                        # same actor (see reevaluate_frame).
                        with frontend_session.actor_lock:
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
            detection_rows: list[dict[str, Any]] = []
            try:
                matches = getattr(analysis, "matches", None)
                if isinstance(matches, (list, tuple)):
                    for match in matches:
                        row = _detection_row(match)
                        if row is not None:
                            detection_rows.append(row)
            except Exception:
                detection_rows = []
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
                detections=detection_rows,
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
    zero processed frames instead of failing. When the inner source offers
    ``fast_forward_to`` (keyframe seek), the skip decodes only the small
    remainder instead of every discarded frame.
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
            fast_forward = getattr(self._inner, "fast_forward_to", None)
            if callable(fast_forward):
                try:
                    fast_forward(self._start_frame)
                except Exception:
                    pass
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
