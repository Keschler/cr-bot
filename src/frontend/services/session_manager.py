from __future__ import annotations

import threading
from typing import Any

from ..models.session import FrontendSession


_lock = threading.Lock()
_session = FrontendSession(mode="idle")
_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None


def _get_session() -> FrontendSession:
    with _lock:
        return _session


def _stop_current_locked() -> None:
    global _thread, _stop_event
    event = _stop_event
    thread = _thread
    if event is not None:
        try:
            event.set()
        except Exception:
            pass
    if thread is not None and thread.is_alive():
        thread.join(timeout=5.0)
    _thread = None
    _stop_event = None


def stop_current_session() -> FrontendSession:
    """Stop the background worker (if any) and return the current session."""

    with _lock:
        _stop_current_locked()
        session = _session
        session.running = False
        return session


def _start_worker(
    *, mode: str, target: Any, kwargs: dict[str, Any]
) -> FrontendSession:
    global _session, _thread, _stop_event
    with _lock:
        _stop_current_locked()
        session = FrontendSession(mode=mode)
        event = threading.Event()
        kwargs = dict(kwargs)
        kwargs["stop_event"] = event

        def _runner() -> None:
            try:
                target(session, **kwargs)
            except Exception:
                # run_* already records session.error fail-closed.
                pass

        thread = threading.Thread(target=_runner, name=f"frontend-{mode}", daemon=True)
        _session = session
        _stop_event = event
        _thread = thread
        session.running = True
        thread.start()
        return session


def _frame_to_json(frame: Any) -> dict[str, Any]:
    return {
        "frame_index": frame.frame_index,
        "timestamp_s": frame.timestamp_s,
        "in_game": frame.in_game,
        "emitted": frame.emitted,
        "record": frame.record,
        "suggestions": frame.suggestions,
        "diagnostics": frame.diagnostics,
        "has_image": frame.jpeg_bytes is not None,
        "frame_width": getattr(frame, "frame_width", None),
        "frame_height": getattr(frame, "frame_height", None),
        "own_actions": getattr(frame, "own_actions", []),
        "enemy_plays": getattr(frame, "enemy_plays", []),
        "detections": getattr(frame, "detections", []),
        "corrected": getattr(frame, "corrected", None),
    }
