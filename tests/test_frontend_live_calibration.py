from types import SimpleNamespace

import pytest

from src.frontend.api import live
from src.frontend.models.requests import LiveStartRequest
from src.frontend.services.paths import REPO_ROOT


@pytest.mark.parametrize("calibration", [None, "", "   ", "custom/profile.json"])
def test_live_calibration_default_and_override(monkeypatch, tmp_path, calibration):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    captured = {}

    def start_worker(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_status_dict=lambda: {"running": True})

    monkeypatch.setattr(live, "_start_worker", start_worker)
    monkeypatch.setattr(live, "resolve_inference_devices", lambda _: None)
    live.api_live_start(
        LiveStartRequest(
            serial="test-device",
            checkpoint=str(checkpoint),
            calibration=calibration,
            execute=True,
            confirm_live=True,
        )
    )
    expected = (calibration or "").strip() or live.DEFAULT_LIVE_CALIBRATION
    assert captured["kwargs"]["calibration"] == str(REPO_ROOT / expected)
    assert captured["kwargs"]["stream_max_width_px"] is None


@pytest.mark.parametrize("width", [480, 720, None])
def test_live_start_forwards_stream_capture_width(monkeypatch, tmp_path, width):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    captured = {}

    def start_worker(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_status_dict=lambda: {"running": True})

    monkeypatch.setattr(live, "_start_worker", start_worker)
    monkeypatch.setattr(live, "resolve_inference_devices", lambda _: None)
    live.api_live_start(LiveStartRequest(
        serial="test-device", checkpoint=str(checkpoint), stream_max_width_px=width,
    ))
    assert captured["kwargs"]["stream_max_width_px"] == width


@pytest.mark.parametrize("width", [0, -1, True, False, "720", 720.0])
def test_live_capture_width_rejects_invalid_request_types(width):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LiveStartRequest(serial="test-device", stream_max_width_px=width)


@pytest.mark.parametrize("width", [0, -1, True, "720", 720.0])
def test_live_runner_rejects_invalid_capture_width_before_capture(monkeypatch, width):
    from src.frontend.models.session import FrontendSession
    from src.frontend.runners.live import run_live_session

    monkeypatch.delenv("CR_BOT_LIVE_DIAGNOSTICS", raising=False)
    session = FrontendSession(mode="live")
    with pytest.raises(ValueError, match="stream_max_width_px"):
        run_live_session(
            session, serial="test-device", checkpoint="unused", stream_max_width_px=width,
        )
    assert session.running is False
    assert session.get_live_preview() is None
