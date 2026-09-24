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
