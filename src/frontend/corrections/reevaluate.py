from __future__ import annotations

from typing import Any

from ..models.session import FrontendSession
from ..runners.pump import _suggestion_to_dict
from ._common import CorrectionUnprocessableError, FrameNotFoundError, NoActorError
from .edits import _apply_correction_edits
from .observation import _corrected_observation
from .observation import _live_policy_imports


def reevaluate_frame(
    session: FrontendSession, frame_index: Any, edits: Any
) -> dict[str, Any]:
    """Run a what-if forward pass for label-corrected detections.

    Applies ``edits`` to the frame's retained detection rows, rebuilds the
    observation, and scores it with the live actor under ``actor_lock``
    (hidden state saved/restored, so the running session is undisturbed).
    Stores the result on the frame as its display correction and returns it.

    Never touches trackers, history, or the execute path. Raises
    ``FrameNotFoundError`` (evicted), ``NoActorError`` (no live actor),
    ``ValueError`` (malformed edits), or ``CorrectionUnprocessableError``.
    """
    frame = session.find_frame(frame_index)
    if frame is None:
        raise FrameNotFoundError(f"frame {frame_index!r} is not retained")
    if not frame.in_game or not frame.emitted:
        raise CorrectionUnprocessableError("only emitted in-game frames can be revised")
    actor = session.actor
    if actor is None:
        raise NoActorError("no live policy actor is available")
    rows, applied = _apply_correction_edits(
        frame.detections if isinstance(frame.detections, list) else [],
        edits,
        frame_width=frame.frame_width,
        frame_height=frame.frame_height,
    )
    observation = _corrected_observation(frame, rows)
    _, decide_fn, action_to_dict_fn = _live_policy_imports()
    with session.actor_lock:
        hidden = getattr(actor, "_hidden", None)
        try:
            try:
                scored = decide_fn(actor, observation)
            except TypeError:
                scored = decide_fn(observation)
            if not (isinstance(scored, tuple) and len(scored) == 3):
                raise CorrectionUnprocessableError("decision entry point misbehaved")
            action_obj, suggestions, diagnostics = scored
        finally:
            try:
                actor._hidden = hidden
            except (AttributeError, TypeError):
                pass
    suggestion_dicts = [
        _suggestion_to_dict(s)
        for s in (suggestions if isinstance(suggestions, (list, tuple)) else [])
    ]
    try:
        action_json = action_to_dict_fn(action_obj)
        if not isinstance(action_json, dict):
            action_json = {"kind": "wait"}
    except Exception as error:
        raise CorrectionUnprocessableError(
            f"decided action cannot be serialized: {error}"
        ) from error
    correction = {
        "suggestions": suggestion_dicts,
        "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
        "action": action_json,
        "applied": applied,
        "revised": True,
    }
    session.set_correction(frame.frame_index, correction)
    return correction


def clear_reevaluation(session: FrontendSession, frame_index: Any) -> bool:
    """Clear a frame's what-if correction (True when the frame exists)."""
    return session.set_correction(frame_index, None)


__all__ = [
    "CorrectionUnprocessableError",
    "FrameNotFoundError",
    "NoActorError",
    "clear_reevaluation",
    "reevaluate_frame",
]
