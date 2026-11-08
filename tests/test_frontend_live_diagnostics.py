import asyncio
import json
import threading

import pytest
from pydantic import ValidationError

from src.frontend.api import frames, live
from src.frontend.app import create_app
from src.frontend.models.frames import LivePreviewFrame
from src.frontend.models.requests import LivePreviewMetricsRequest
from src.frontend.models.session import FrontendSession
from src.frontend.services import live_diagnostics, session_manager
from src.frontend.services.live_diagnostics import LiveDiagnostics, LivePreviewTimingMiddleware


def test_rates_and_stall_clocks_do_not_claim_phone_age():
    clock = [0.0]
    diagnostics = LiveDiagnostics(transport="stream", clock=lambda: clock[0])
    diagnostics.set_capture_probe(lambda: {"clock": "workstation_decode_publication", "phone_to_output_age_ms": None})
    timing = {"total": 1000.0, "vision.detector": 800.0}
    for stamp, index in [(1.0, 20), (2.0, 40)]:
        clock[0] = stamp
        diagnostics.record_analysis(frame_index=index, publication_s=stamp - 0.5, timing_ms=timing)
    timing["total"] = 0.0
    snapshot = diagnostics.snapshot()
    assert snapshot["phone_to_display_age_ms"] is None
    assert snapshot["capture"]["phone_to_output_age_ms"] is None
    assert snapshot["analysis"]["recent_fps"] == 1.0
    assert snapshot["analysis"]["last_input_decode_publication_age_ms"] == 500.0
    assert snapshot["analysis"]["last_timing_ms"]["total"] == 1000.0
    snapshot["analysis"]["last_timing_ms"]["total"] = -1
    assert diagnostics.snapshot()["analysis"]["last_timing_ms"]["total"] == 1000.0

    clock[0] = 3.0
    diagnostics.begin_analysis(60)
    clock[0] = 9.0
    stalled = diagnostics.snapshot()["analysis"]
    assert stalled["completed_frames"] == 2
    assert stalled["recent_fps"] == 0.0
    assert stalled["active_frame_index"] == 60
    assert stalled["active_elapsed_ms"] == 6000.0


def test_repeated_preview_pixels_are_distinct_from_response_fps():
    clock = [0.0]
    diagnostics = LiveDiagnostics(transport="stream", clock=lambda: clock[0])
    for stamp, index in [(1.0, 10), (2.0, 10), (3.0, 11)]:
        clock[0] = stamp
        diagnostics.record_preview(frame_index=index, handler_ms=190.0, size_bytes=199802)
    diagnostics.record_preview_server_time(220.0)
    preview = diagnostics.snapshot()["preview"]
    assert preview["responses"] == 3
    assert preview["recent_response_fps"] == 1.0
    assert preview["recent_changed_frame_fps"] == 0.5
    assert preview["last_handler_ms"] == 190.0
    assert preview["last_dispatch_to_response_ms"] == 220.0
    assert preview["last_size_bytes"] == 199802
    clock[0] = 10.0
    assert diagnostics.snapshot()["preview"]["recent_response_fps"] == 0.0


def test_capture_probe_runs_outside_collector_lock_and_can_fail():
    diagnostics = LiveDiagnostics(transport="stream")

    def capture():
        diagnostics.set_phase("capturing")
        raise RuntimeError("capture closed")

    diagnostics.set_capture_probe(capture)
    assert diagnostics.snapshot()["capture"] == {"diagnostics_error": "capture closed"}
    assert diagnostics.snapshot()["phase"] == "capturing"


def test_reporter_keeps_logging_while_analysis_is_stalled_and_stops():
    lines = []
    reported = threading.Event()

    class Logger:
        def info(self, format_string, payload):
            assert format_string == "LIVE_DIAGNOSTICS %s"
            lines.append(json.loads(payload))
            if len(lines) >= 2:
                reported.set()

    diagnostics = LiveDiagnostics(transport="stream", logger=Logger(), report_interval_s=0.01)
    diagnostics.begin_analysis(77)
    diagnostics.start()
    try:
        assert reported.wait(timeout=1.0)
        assert lines[-1]["analysis"]["active_frame_index"] == 77
        assert lines[-1]["analysis"]["completed_frames"] == 0
    finally:
        diagnostics.close(error="test error")
    assert not diagnostics._thread.is_alive()
    assert lines[-1]["phase"] == "stopped"
    assert lines[-1]["error"] == "test error"


def _client_request(session_id):
    return LivePreviewMetricsRequest(
        session_id=session_id, window_s=2.0, displayed_frames=5, changed_frames=1,
        last_frame_index=77, response_headers_ms=190.0, body_ms=0.0,
        decode_ms=25.0, request_to_display_ms=215.0,
    )


def test_diagnostic_api_gates_client_reports_by_running_session(monkeypatch):
    session = FrontendSession(mode="live")
    monkeypatch.setattr(live, "_get_session", lambda: session)
    assert live.api_live_diagnostics() == {"enabled": False, "enable_with": "CR_BOT_LIVE_DIAGNOSTICS=1"}
    assert live.api_live_preview_metrics(_client_request("stale")).status_code == 204
    session.live_diagnostics = LiveDiagnostics(transport="stream")
    session.running = True
    live.api_live_preview_metrics(_client_request("stale"))
    assert live.api_live_diagnostics()["browser"] is None
    live.api_live_preview_metrics(_client_request(session.live_diagnostics.session_id))
    browser = live.api_live_diagnostics()["browser"]
    assert browser["displayed_fps"] == 2.5
    assert browser["changed_fps"] == 0.5
    assert browser["response_headers_ms"] == 190.0
    session.running = False
    request = _client_request(session.live_diagnostics.session_id)
    request.last_frame_index = 999
    live.api_live_preview_metrics(request)
    assert live.api_live_diagnostics()["browser"]["last_frame_index"] == 77


@pytest.mark.parametrize("field,value", [("window_s", 0), ("decode_ms", -1), ("body_ms", float("nan"))])
def test_browser_metrics_reject_invalid_timings(field, value):
    values = _client_request("session").model_dump()
    values[field] = value
    with pytest.raises(ValidationError):
        LivePreviewMetricsRequest(**values)


def test_preview_reports_server_work_and_identifies_timestamp_clock(monkeypatch):
    session = FrontendSession(mode="live")
    session.running = True
    session.live_frame_clock = "workstation-decode-publication"
    session.live_diagnostics = LiveDiagnostics(transport="stream")
    session.set_live_preview_provider(lambda: LivePreviewFrame(77, 1.0, b"jpeg"))
    monkeypatch.setattr(frames, "_get_session", lambda: session)
    response = frames.api_frame_preview()
    assert response.headers["x-frame-clock"] == "workstation-decode-publication"
    assert response.headers["x-live-diagnostics"] == session.live_diagnostics.session_id
    assert session.live_diagnostics.snapshot()["preview"]["responses"] == 1
    assert session.live_diagnostics.snapshot()["preview"]["last_size_bytes"] == 4


def test_preview_middleware_measures_dispatch_and_preserves_response(monkeypatch):
    session = FrontendSession(mode="live")
    session.live_diagnostics = LiveDiagnostics(transport="stream")
    clock = [10.0]
    monkeypatch.setattr(live_diagnostics.time, "monotonic", lambda: clock[0])
    messages = []

    async def app(scope, receive, send):
        scope["cr_bot.live_diagnostics"] = session.live_diagnostics
        clock[0] += 0.125
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-frame-index", b"77")]})
        await send({"type": "http.response.body", "body": b"jpeg"})

    async def send(message):
        messages.append(message)

    # A concurrent start/stop can hold this lock while joining a worker.
    # Preview middleware must never wait for it on the event loop.
    with session_manager._lock:
        asyncio.run(LivePreviewTimingMiddleware(app)({"type": "http", "path": "/api/frame/preview"}, None, send))
    assert messages[1]["body"] == b"jpeg"
    headers = dict(messages[0]["headers"])
    assert headers[b"x-frame-index"] == b"77"
    assert float(headers[b"x-preview-server-ms"]) == 125.0
    assert session.live_diagnostics.snapshot()["preview"]["last_dispatch_to_response_ms"] == 125.0


def test_preview_timing_middleware_is_opt_in(monkeypatch):
    monkeypatch.delenv("CR_BOT_LIVE_DIAGNOSTICS", raising=False)
    assert LivePreviewTimingMiddleware not in [m.cls for m in create_app().user_middleware]
    monkeypatch.setenv("CR_BOT_LIVE_DIAGNOSTICS", "1")
    assert LivePreviewTimingMiddleware in [m.cls for m in create_app().user_middleware]
