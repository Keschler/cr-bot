from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..models.requests import ReevaluateRequest
from .deps import _get_session

router = APIRouter()


@router.get("/api/labels")
def api_labels() -> dict[str, Any]:
    """Unit-class vocabulary for label correction (best-effort)."""
    from ..corrections.vocab import unit_label_vocabulary

    labels = unit_label_vocabulary()
    return {"labels": labels if labels is not None else [], "teams": ["ally", "enemy"]}


@router.post("/api/frame/{frame_index}/reevaluate")
def api_frame_reevaluate(frame_index: int, request: ReevaluateRequest) -> dict[str, Any]:
    """Run a what-if forward pass for label-corrected detections.

    Applies ``{updates, deletes, adds}`` to the frame's retained detection
    rows, rebuilds the observation, and scores it with the live actor
    (hidden state restored afterwards). The result is stored as the frame's
    display correction; trackers, history, and the execute path are never
    touched. Unknown/evicted indices answer 404, missing actor 409,
    malformed edits 400, unbuildable observations 422.
    """
    from ..corrections.reevaluate import (
        CorrectionUnprocessableError,
        FrameNotFoundError,
        NoActorError,
        reevaluate_frame,
    )

    session = _get_session()
    edits = {
        "updates": request.updates if request.updates is not None else [],
        "deletes": request.deletes if request.deletes is not None else [],
        "adds": request.adds if request.adds is not None else [],
    }
    try:
        correction = reevaluate_frame(session, frame_index, edits)
    except FrameNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except NoActorError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except CorrectionUnprocessableError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"frame_index": int(frame_index), "corrected": correction}


@router.delete("/api/frame/{frame_index}/reevaluate")
def api_frame_reevaluate_revert(frame_index: int) -> dict[str, Any]:
    """Clear a frame's what-if correction (404 when evicted)."""
    from ..corrections.reevaluate import FrameNotFoundError, clear_reevaluation

    session = _get_session()
    if session.find_frame(frame_index) is None:
        raise HTTPException(
            status_code=404, detail=f"frame {frame_index!r} is not retained"
        )
    reverted = clear_reevaluation(session, frame_index)
    return {"frame_index": int(frame_index), "reverted": bool(reverted)}
