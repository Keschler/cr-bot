from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..services import session_library

router = APIRouter()


@router.get("/api/sessions")
def api_sessions_list() -> dict[str, Any]:
    return {"sessions": session_library.list_entries()}


@router.post("/api/sessions")
def api_sessions_save(entry: dict[str, Any]) -> dict[str, Any]:
    try:
        saved = session_library.save_entry(entry)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(
            status_code=500, detail=f"could not store session: {error}"
        ) from error
    return {"session": saved}


@router.delete("/api/sessions/{name}")
def api_sessions_delete(name: str) -> dict[str, Any]:
    try:
        removed = session_library.delete_entry(name)
    except OSError as error:
        raise HTTPException(
            status_code=500, detail=f"could not delete session: {error}"
        ) from error
    if not removed:
        raise HTTPException(status_code=404, detail=f"unknown session: {name}")
    return {"deleted": True, "name": session_library.sanitize_entry_name(name)}
