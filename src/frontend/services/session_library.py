from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import UPLOAD_DIR

LIBRARY_FILENAME = "sessions.json"
MAX_ENTRIES = 50


_lock = threading.Lock()


def _library_path() -> Path:
    return UPLOAD_DIR / LIBRARY_FILENAME


def sanitize_entry_name(name: Any) -> str:
    raw = str(name or "").strip() or "replay"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(raw).name).strip("._") or "replay"
    return cleaned[:128]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_entries() -> list[dict[str, Any]]:
    """Return saved replay entries, newest first (best-effort, never raises)."""
    try:
        with _lock:
            return _read_locked()
    except Exception:
        return []


def _read_locked() -> list[dict[str, Any]]:
    path = _library_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def _write_locked(entries: list[dict[str, Any]]) -> None:
    path = _library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp.replace(path)


def validate_entry(entry: Any) -> dict[str, Any]:
    """Validate a client-supplied entry; raises ValueError describing problems."""
    if not isinstance(entry, dict):
        raise ValueError("entry must be an object")
    name = sanitize_entry_name(entry.get("name"))
    video_path = entry.get("video_path")
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("video_path must be a non-empty string")
    params = entry.get("params")
    if params is not None and not isinstance(params, dict):
        raise ValueError("params must be an object or omitted")
    cursor_frame = entry.get("cursor_frame")
    if cursor_frame is not None:
        try:
            cursor_frame = int(cursor_frame)
        except (TypeError, ValueError):
            raise ValueError("cursor_frame must be an integer or null") from None
        if cursor_frame < 0:
            raise ValueError("cursor_frame must be non-negative")
    frame_count = entry.get("frame_count")
    if frame_count is not None:
        try:
            frame_count = int(frame_count)
        except (TypeError, ValueError):
            raise ValueError("frame_count must be an integer or null") from None
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
    filename = entry.get("filename")
    if filename is not None and not isinstance(filename, str):
        raise ValueError("filename must be a string or omitted")
    return {
        "name": name,
        "video_path": video_path.strip(),
        "filename": (filename or "").strip() or Path(video_path.strip()).name,
        "params": dict(params or {}),
        "cursor_frame": cursor_frame,
        "frame_count": frame_count,
        "updated": _utcnow_iso(),
    }


def save_entry(entry: Any) -> dict[str, Any]:
    """Upsert one entry by name (newest first, capped). Returns the stored entry."""
    clean = validate_entry(entry)
    with _lock:
        entries = [e for e in _read_locked() if e.get("name") != clean["name"]]
        entries.insert(0, clean)
        del entries[MAX_ENTRIES:]
        _write_locked(entries)
    return clean


def delete_entry(name: Any) -> bool:
    """Delete one entry by name. Returns True when something was removed."""
    clean = sanitize_entry_name(name)
    with _lock:
        entries = _read_locked()
        kept = [e for e in entries if e.get("name") != clean]
        if len(kept) == len(entries):
            return False
        _write_locked(kept)
        return True
