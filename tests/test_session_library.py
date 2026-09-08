"""Tests for the frontend session library (recent-replay sidecar)."""

import pytest

from src.frontend.services import session_library


@pytest.fixture
def library_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(session_library, "UPLOAD_DIR", tmp_path)
    return tmp_path


def _entry(name="replay-a", **overrides):
    base = {
        "name": name,
        "video_path": "uploads/replay-a.mp4",
        "filename": "replay-a.mp4",
        "params": {"frame_stride": 1, "start_frame": 0},
        "cursor_frame": 12,
        "frame_count": 40,
    }
    base.update(overrides)
    return base


def test_save_and_list_round_trip(library_dir):
    assert session_library.list_entries() == []
    saved = session_library.save_entry(_entry())
    assert saved["name"] == "replay-a"
    assert saved["updated"]
    assert [e["name"] for e in session_library.list_entries()] == ["replay-a"]


def test_save_upserts_by_name_newest_first(library_dir):
    session_library.save_entry(_entry("a"))
    session_library.save_entry(_entry("b"))
    session_library.save_entry(_entry("a", cursor_frame=99))
    entries = session_library.list_entries()
    assert [e["name"] for e in entries] == ["a", "b"]
    assert entries[0]["cursor_frame"] == 99


def test_save_rejects_bad_entries(library_dir):
    with pytest.raises(ValueError):
        session_library.save_entry({"name": "x"})
    with pytest.raises(ValueError):
        session_library.save_entry(_entry(cursor_frame=-1))
    with pytest.raises(ValueError):
        session_library.save_entry(_entry(params=[1, 2]))
    with pytest.raises(ValueError):
        session_library.save_entry("not-an-object")


def test_delete_entry(library_dir):
    session_library.save_entry(_entry("a"))
    assert session_library.delete_entry("a") is True
    assert session_library.delete_entry("a") is False
    assert session_library.list_entries() == []


def test_corrupt_sidecar_lists_empty(library_dir):
    (library_dir / session_library.LIBRARY_FILENAME).write_text("{oops", encoding="utf-8")
    assert session_library.list_entries() == []


def test_api_delete_unknown_is_404():
    from fastapi import HTTPException

    from src.frontend.api import sessions as sessions_api

    with pytest.raises(HTTPException) as exc:
        sessions_api.api_sessions_delete("missing-never-saved")
    assert exc.value.status_code == 404
