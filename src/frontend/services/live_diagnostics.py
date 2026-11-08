from __future__ import annotations

import collections
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LiveRuntimeDiagnostics:
    requested_device: str
    yolo_device: str
    policy_device: str
    card_classifier_selected_device: str
    tower_hp_selected_device: str
    torch_version: str
    torch_cuda_build: str | None
    cuda_available: bool
    cuda_device_name: str | None
    torch_cpu_threads: int
    torch_interop_threads: int
    opencv_threads: int
    yolo_image_size: int


def collect_runtime_diagnostics(
    *, requested_device: str, yolo_device: str, policy_device: str,
    yolo_image_size: int,
) -> LiveRuntimeDiagnostics:
    # Called after model initialization, never while importing the frontend.
    import cv2
    import torch
    from cr_bot.vision.model_loader import torch_inference_device

    cuda = bool(torch.cuda.is_available())
    return LiveRuntimeDiagnostics(
        requested_device=requested_device,
        yolo_device=yolo_device,
        policy_device=policy_device,
        card_classifier_selected_device=str(torch_inference_device("CARD_CLASSIFIER_DEVICE")),
        tower_hp_selected_device=str(torch_inference_device("TOWER_HP_OCR_DEVICE")),
        torch_version=str(torch.__version__),
        torch_cuda_build=torch.version.cuda,
        cuda_available=cuda,
        cuda_device_name=torch.cuda.get_device_name() if cuda else None,
        torch_cpu_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        opencv_threads=cv2.getNumThreads(),
        yolo_image_size=yolo_image_size,
    )


@dataclass(frozen=True, slots=True)
class LivePreviewClientMetrics:
    window_s: float
    displayed_frames: int
    changed_frames: int
    last_frame_index: int
    response_headers_ms: float
    body_ms: float
    decode_ms: float
    request_to_display_ms: float


class LiveDiagnostics:
    """Bounded, opt-in workstation timings, independent of inference progress.

    No phone capture timestamp exists in the elementary H.264 stream. These
    timings must never be presented as end-to-end gameplay age.
    """

    def __init__(
        self, *, transport: str, execute: bool = False,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
        report_interval_s: float = 2.0,
    ) -> None:
        self.session_id = uuid.uuid4().hex
        self.transport = transport
        self.execute = execute
        self._clock = clock
        self._logger = logger or logging.getLogger("uvicorn.error")
        self._report_interval_s = report_interval_s
        self._started_s = clock()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = "starting"
        self._error: str | None = None
        self._runtime: LiveRuntimeDiagnostics | None = None
        self._capture_probe: Callable[[], dict[str, Any]] | None = None
        self._analysis_times: collections.deque[float] = collections.deque(maxlen=512)
        self._preview_times: collections.deque[float] = collections.deque(maxlen=512)
        self._preview_changes: collections.deque[float] = collections.deque(maxlen=512)
        self._analysis_count = 0
        self._analysis_index: int | None = None
        self._analysis_in_game: bool | None = None
        self._analysis_publication_s: float | None = None
        self._active_analysis_s: float | None = None
        self._active_analysis_index: int | None = None
        self._timings: dict[str, float] = {}
        self._preview_count = 0
        self._preview_index: int | None = None
        self._preview_handler_ms: float | None = None
        self._preview_server_ms: float | None = None
        self._preview_bytes: int | None = None
        self._client_metrics: LivePreviewClientMetrics | None = None
        self._client_received_s: float | None = None

    def set_capture_probe(self, probe: Callable[[], dict[str, Any]]) -> None:
        with self._lock:
            self._capture_probe = probe

    def set_runtime(self, runtime: LiveRuntimeDiagnostics) -> None:
        with self._lock:
            self._runtime = runtime

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def begin_analysis(self, frame_index: int) -> None:
        with self._lock:
            self._active_analysis_s = self._clock()
            self._active_analysis_index = frame_index

    def record_analysis(
        self, *, frame_index: int, publication_s: float,
        timing_ms: dict[str, float],
        in_game: bool | None = None,
    ) -> None:
        with self._lock:
            self._analysis_count += 1
            self._analysis_times.append(self._clock())
            self._analysis_index = frame_index
            self._analysis_in_game = in_game
            self._analysis_publication_s = publication_s
            self._active_analysis_s = None
            self._active_analysis_index = None
            self._timings = dict(timing_ms)

    def record_preview(self, *, frame_index: int, handler_ms: float, size_bytes: int) -> None:
        with self._lock:
            now = self._clock()
            self._preview_count += 1
            self._preview_times.append(now)
            if frame_index != self._preview_index:
                self._preview_changes.append(now)
            self._preview_index = frame_index
            self._preview_handler_ms = handler_ms
            self._preview_bytes = size_bytes

    def record_preview_server_time(self, elapsed_ms: float) -> None:
        with self._lock:
            self._preview_server_ms = elapsed_ms

    def record_client(self, metrics: LivePreviewClientMetrics) -> None:
        with self._lock:
            self._client_metrics = metrics
            self._client_received_s = self._clock()

    def _recent_rate(self, events: collections.deque[float], now: float) -> float:
        while events and events[0] < now - 5.0:
            events.popleft()
        if not events:
            return 0.0
        # Intervals, including any current stall; a single event is insufficient
        # to establish FPS. This is not the inverse of one frame's latency.
        elapsed = now - events[0]
        return max(0, len(events) - 1) / elapsed if elapsed > 0.0 else 0.0

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()

        def age(timestamp: float | None) -> float | None:
            return None if timestamp is None else max(0.0, (now - timestamp) * 1000.0)

        with self._lock:
            capture_probe = self._capture_probe
            client = asdict(self._client_metrics) if self._client_metrics else None
            if client is not None:
                client["displayed_fps"] = client["displayed_frames"] / client["window_s"]
                client["changed_fps"] = client["changed_frames"] / client["window_s"]
                client["report_age_ms"] = age(self._client_received_s)
            result: dict[str, Any] = {
                "enabled": True,
                "session_id": self.session_id,
                "transport": self.transport,
                "execute": self.execute,
                "phase": self._phase,
                "elapsed_s": max(0.0, now - self._started_s),
                "error": self._error,
                "clock": "workstation_monotonic",
                "phone_to_display_age_ms": None,
                "runtime": asdict(self._runtime) if self._runtime else None,
                "analysis": {
                    "completed_frames": self._analysis_count,
                    "recent_fps": self._recent_rate(self._analysis_times, now),
                    "last_frame_index": self._analysis_index,
                    "in_game": self._analysis_in_game,
                    "last_input_decode_publication_age_ms": age(self._analysis_publication_s),
                    "active_frame_index": self._active_analysis_index,
                    "active_elapsed_ms": age(self._active_analysis_s),
                    "last_timing_ms": dict(self._timings),
                },
                "preview": {
                    "responses": self._preview_count,
                    "recent_response_fps": self._recent_rate(self._preview_times, now),
                    "recent_changed_frame_fps": self._recent_rate(self._preview_changes, now),
                    "last_frame_index": self._preview_index,
                    "last_handler_ms": self._preview_handler_ms,
                    "last_dispatch_to_response_ms": self._preview_server_ms,
                    "last_size_bytes": self._preview_bytes,
                },
                "browser": client,
            }
        # Do not nest the diagnostics lock with capture/preview locks.
        try:
            result["capture"] = capture_probe() if capture_probe else None
        except Exception as error:
            result["capture"] = {"diagnostics_error": str(error)}
        return result

    def _emit(self) -> None:
        self._logger.info("LIVE_DIAGNOSTICS %s", json.dumps(self.snapshot(), separators=(",", ":")))

    def start(self) -> None:
        self._emit()

        def report() -> None:
            while not self._stop.wait(self._report_interval_s):
                self._emit()

        self._thread = threading.Thread(target=report, name="live-diagnostics", daemon=True)
        self._thread.start()

    def close(self, *, error: str | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._lock:
            self._phase = "stopped"
            self._error = error
        self._emit()


class LivePreviewTimingMiddleware:
    """Time preview dispatch through response start, including threadpool wait."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("path") != "/api/frame/preview":
            await self.app(scope, receive, send)
            return
        started = time.monotonic()

        async def timed_send(message: Any) -> None:
            if message["type"] == "http.response.start":
                # The synchronous route attaches its actual collector after
                # resolving the session. Never take lifecycle locks on the
                # event loop or charge a replacement session's response to
                # the old collector.
                diagnostics = scope.get("cr_bot.live_diagnostics")
                if diagnostics is None:
                    await send(message)
                    return
                elapsed_ms = (time.monotonic() - started) * 1000.0
                diagnostics.record_preview_server_time(elapsed_ms)
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-preview-server-ms", str(elapsed_ms).encode("ascii")),
                ]
            await send(message)

        await self.app(scope, receive, timed_send)
