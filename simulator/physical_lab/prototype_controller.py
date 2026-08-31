"""Run the public prototype actor against live ADB screenshots or a video.

The module is deliberately a thin deployment boundary:

``ADB/video frame -> cr_bot visual extractor -> public V2 observation ->
prototype actor -> Wait/Play -> calibrated ADB taps``

The actor is loaded from the auditable recurrent-prototype checkpoint and only
receives the same public V2 tensors used by simulator training.  Live control
is opt-in.  A normal invocation is a dry-run that performs extraction and
prints or records the decisions without sending input to Android.

The live source uses serial-scoped ``AdbPhoneController`` operations.  By
default, ADB ``screenrecord`` is kept open as an H.264 stream and a bounded
latest-frame buffer prevents inference lag.  A play is sent through
``AutonomousPhone.select_and_place`` so a recent decoded stream frame must
still identify the selected card before either tap is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import textwrap
import time
from typing import Any, Callable, Protocol, TextIO


_MODULE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _MODULE_ROOT.parent

# This makes ``python physical_lab/prototype_controller.py`` usable directly
# from a checkout.  The normal editable-install/PYTHONPATH setup already
# contains these entries; the supported package entry point is the wrapper at
# ``run_prototype_live.py``.
for _path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


DEFAULT_CHECKPOINT = _MODULE_ROOT / "outputs/simulator/training/prototype-fast-current/prototype.pt"
LIVE_IGNORED_DETECTOR_LABELS: frozenset[str] = frozenset({"bomb"})
LIVE_OWN_DECK_CARD_NAMES: frozenset[str] = frozenset(
    {
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
        "fireball",
        "log",
    }
)
LIVE_OWN_DETECTOR_ALIASES: dict[str, str] = {
    "hog": "hog-rider",
    "skeleton": "skeletons",
    "skeleton-evolution": "skeletons",
    "ice-spirit-evolution": "ice-spirit",
    "the-log": "log",
    "old-musketeer": "musketeer",
}
LIVE_OWN_TROOP_CARD_NAMES: frozenset[str] = frozenset(
    {
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
    }
)
LIVE_ENEMY_CONFIRMATION_FRAMES = 2
LIVE_OWN_SEED_MAX_AGE_S = 2.0
LIVE_OWN_SEED_MATCH_RADIUS_PX = 280.0
_KATACR_ROOT_CANDIDATES = (
    _REPOSITORY_ROOT / "vendor/external/KataCR",
    _REPOSITORY_ROOT / "capture/vendor/external/KataCR",
)


def _bootstrap_extractor_runtime() -> None:
    """Make the checkout's vendored KataCR runtime visible to ``cr_bot``.

    ``cr_bot.paths`` points at the canonical repository-level vendor location,
    while this checkout keeps the populated KataCR tree under ``capture``.
    The extractor imports KataCR lazily, so repair both the import path and the
    path constants before importing ``pipeline`` or ``yolo_runtime``.
    """

    katacr_root = next(
        (
            candidate
            for candidate in _KATACR_ROOT_CANDIDATES
            if (candidate / "katacr").is_dir()
        ),
        None,
    )
    if katacr_root is None:
        locations = ", ".join(str(path) for path in _KATACR_ROOT_CANDIDATES)
        raise PrototypeControllerError(
            "the cr_bot visual extractor requires the vendored KataCR runtime; "
            f"none was found at {locations}"
        )

    if str(katacr_root) not in sys.path:
        sys.path.insert(0, str(katacr_root))

    try:
        import cr_bot.paths as cr_paths
    except ImportError as error:  # pragma: no cover - environment-specific
        raise PrototypeControllerError("the cr_bot package is not importable") from error

    # PyInstaller places ``cr_bot`` at the bundle root rather than below the
    # checkout's ``src`` directory.  Rebase all asset paths to the extracted
    # bundle so the frozen executable can find its embedded models/templates.
    if getattr(sys, "frozen", False):
        cr_paths.REPO_ROOT = _REPOSITORY_ROOT
        cr_paths.APP_ROOT = _REPOSITORY_ROOT
        cr_paths.ASSETS_DIR = _REPOSITORY_ROOT / "assets"
        cr_paths.MODELS_DIR = cr_paths.ASSETS_DIR / "models"
        cr_paths.TEMPLATES_DIR = cr_paths.ASSETS_DIR / "templates"
        cr_paths.PICTURES_DIR = cr_paths.ASSETS_DIR / "pictures"
        cr_paths.CACHE_DIR = _REPOSITORY_ROOT / "outputs" / "cache"

    dataset_root = katacr_root.parent / "Clash-Royale-Detection-Dataset"
    cr_paths.KATACR_ROOT = katacr_root
    if dataset_root.is_dir():
        cr_paths.KATACR_DATASET_ROOT = dataset_root

    # Be robust if a caller imported one of these modules before constructing
    # the runner.  They copy the constants with ``from cr_bot.paths import``.
    katacr_runtime = sys.modules.get("cr_bot.vision.katacr_runtime")
    if katacr_runtime is not None:
        katacr_runtime.KATACR_ROOT = katacr_root
        if dataset_root.is_dir():
            katacr_runtime.KATACR_DATASET_ROOT = dataset_root
    yolo_runtime = sys.modules.get("cr_bot.vision.yolo_runtime")
    if yolo_runtime is not None:
        yolo_runtime.KATACR_ROOT = katacr_root


class PrototypeControllerError(RuntimeError):
    """Raised when live inference cannot proceed safely."""


try:
    from .devices import AdbPhoneController
except ImportError:  # pragma: no cover - direct script execution
    from simulator.physical_lab.devices import AdbPhoneController


class CachedAdbPhoneController(AdbPhoneController):
    """ADB controller that avoids redundant connectivity probes before taps.

    ``AdbPhoneController.tap_screen`` normally calls ``device_info`` first;
    that is three additional ADB subprocesses for every tap.  A successful
    screenshot or decoded stream frame already proves that the selected serial
    is reachable, and the tap command itself remains serial-scoped and still
    fails closed if the device disappears. Cache that transport proof for a
    short interval, preserving the initial live preflight check.
    """

    def __init__(
        self,
        serial: str,
        *,
        connection_check_interval_s: float = 5.0,
        monotonic_fn: Callable[[], float] = time.monotonic,
        **kwargs: Any,
    ) -> None:
        if not math.isfinite(float(connection_check_interval_s)) or float(
            connection_check_interval_s
        ) < 0:
            raise PrototypeControllerError(
                "connection_check_interval_s must be finite and non-negative"
            )
        super().__init__(serial, **kwargs)
        self.connection_check_interval_s = float(connection_check_interval_s)
        self._connection_monotonic_fn = monotonic_fn
        self._last_successful_screenshot_s: float | None = None

    def screenshot(self) -> Any:
        frame = super().screenshot()
        self.mark_transport_alive()
        return frame

    def mark_transport_alive(self) -> None:
        """Record proof from a successful persistent stream frame."""

        self._last_successful_screenshot_s = self._connection_monotonic_fn()

    def _require_connected(self) -> None:
        last_screenshot = self._last_successful_screenshot_s
        if last_screenshot is not None:
            age_s = self._connection_monotonic_fn() - last_screenshot
            if 0.0 <= age_s <= self.connection_check_interval_s:
                return
        super()._require_connected()


def configure_detector_inference_size(detector: Any, image_size: int) -> Any:
    """Override the extractor YOLO input size without changing ``cr_bot``.

    The checked-in extractor hard-codes ``imgsz=896``.  The live controller
    can use a smaller, explicit size to reduce CPU latency; Ultralytics maps
    detections back to the original frame coordinates.  Keeping this adapter
    local makes the optimization opt-out and leaves the shared extractor
    unchanged for other callers.
    """

    if type(image_size) is not int or image_size <= 0:
        raise PrototypeControllerError("YOLO image size must be a positive integer")
    if image_size == 896:
        return detector
    models = getattr(detector, "models", None)
    if not isinstance(models, (list, tuple)) or not models:
        raise PrototypeControllerError(
            "the configured detector does not expose YOLO models for resizing"
        )
    previous_size = getattr(detector, "_prototype_inference_size", None)
    if previous_size is not None:
        if previous_size != image_size:
            raise PrototypeControllerError(
                "detector inference size was already configured with a different value"
            )
        return detector
    for model in models:
        original_predict = getattr(model, "predict", None)
        if not callable(original_predict):
            raise PrototypeControllerError("configured detector model has no predict method")

        def predict(
            *args: Any,
            _original_predict: Callable[..., Any] = original_predict,
            _image_size: int = image_size,
            **kwargs: Any,
        ) -> Any:
            kwargs["imgsz"] = _image_size
            return _original_predict(*args, **kwargs)

        # ``predict`` is stored on the model instance, so the closure retains
        # the original bound method and does not alter its call signature.
        model.predict = predict
    try:
        detector._prototype_inference_size = image_size
    except (AttributeError, TypeError) as error:
        raise PrototypeControllerError(
            "configured detector does not allow a speed profile"
        ) from error
    return detector


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """One decoded BGR frame and its source-time coordinate."""

    image: Any
    frame_index: int
    timestamp_s: float

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise PrototypeControllerError("source frame index must be non-negative")
        if not math.isfinite(float(self.timestamp_s)) or self.timestamp_s < 0:
            raise PrototypeControllerError("source frame timestamp must be finite and non-negative")


class FrameSource(Protocol):
    """Frame source used by :class:`LivePrototypeRunner`."""

    def next_frame(self) -> SourceFrame | None: ...

    def close(self) -> None: ...


class AdbH264FrameSource:
    """Continuously decode one explicitly selected phone's H.264 stream.

    Android's ``screenrecord`` writes an elementary H.264 stream to stdout.
    ``ffmpeg`` decodes that stream to raw BGR frames in a background thread.
    Only the newest decoded frame is retained, so a slow CPU inference pass
    cannot create an ever-growing latency backlog.  Android's screenrecord
    time limit is handled by restarting the segment after EOF.
    """

    def __init__(
        self,
        controller: Any,
        *,
        ffmpeg_executable: str = "ffmpeg",
        bit_rate: int = 8_000_000,
        restart_delay_s: float = 0.25,
        stream_read_timeout_s: float = 5.0,
        max_consecutive_failures: int = 3,
        action_frame_max_age_s: float = 1.0,
        action_frame_wait_timeout_s: float = 2.0,
        popen_factory: Callable[..., Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(ffmpeg_executable, str) or not ffmpeg_executable.strip():
            raise PrototypeControllerError("ffmpeg executable must be a non-empty token")
        if any(character.isspace() for character in ffmpeg_executable):
            raise PrototypeControllerError("ffmpeg executable must be a single token")
        if type(bit_rate) is not int or bit_rate <= 0:
            raise PrototypeControllerError("stream bit_rate must be a positive integer")
        for name, value in (
            ("restart_delay_s", restart_delay_s),
            ("stream_read_timeout_s", stream_read_timeout_s),
            ("action_frame_max_age_s", action_frame_max_age_s),
            ("action_frame_wait_timeout_s", action_frame_wait_timeout_s),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise PrototypeControllerError(f"{name} must be finite and non-negative")
        if type(max_consecutive_failures) is not int or max_consecutive_failures <= 0:
            raise PrototypeControllerError("max_consecutive_failures must be positive")

        self.controller = controller
        self.ffmpeg_executable = ffmpeg_executable
        self.bit_rate = bit_rate
        self.restart_delay_s = float(restart_delay_s)
        self.stream_read_timeout_s = float(stream_read_timeout_s)
        self.max_consecutive_failures = max_consecutive_failures
        self.action_frame_max_age_s = float(action_frame_max_age_s)
        self.action_frame_wait_timeout_s = float(action_frame_wait_timeout_s)
        self._popen_factory = popen_factory or subprocess.Popen
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn
        self._condition = threading.Condition()
        self._process_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._adb_process: Any | None = None
        self._decoder_process: Any | None = None
        self._stream_width: int | None = None
        self._stream_height: int | None = None
        self._latest: SourceFrame | None = None
        self._last_delivered_frame_index: int | None = None
        self._next_frame_index = 0
        self._error: Exception | None = None

    def _device_dimensions(self) -> tuple[int, int]:
        info = self.controller.device_info()
        if not bool(getattr(info, "connected", False)):
            raise PrototypeControllerError("selected ADB device is not connected")
        width = getattr(info, "screen_width_px", None)
        height = getattr(info, "screen_height_px", None)
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise PrototypeControllerError(
                "streaming requires positive dimensions from the selected ADB device"
            )
        return width, height

    def _start_segment(self) -> None:
        if self._stream_width is None or self._stream_height is None:
            self._stream_width, self._stream_height = self._device_dimensions()

        adb_command = [
            str(self.controller.adb_executable),
            "-s",
            str(self.controller.serial),
            "exec-out",
            "screenrecord",
            "--output-format",
            "h264",
            "--bit-rate",
            str(self.bit_rate),
            "-",
        ]
        decoder_command = [
            self.ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-flags",
            "low_delay",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-fps_mode",
            "passthrough",
            "pipe:1",
        ]
        adb_process: Any | None = None
        adb_stdout: Any | None = None
        try:
            adb_process = self._popen_factory(
                adb_command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            adb_stdout = getattr(adb_process, "stdout", None)
            if adb_stdout is None:
                raise PrototypeControllerError("ADB stream did not expose stdout")
            decoder_process = self._popen_factory(
                decoder_command,
                stdin=adb_stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            # The decoder owns the read end from this point onward. Closing
            # the parent's duplicate is required for clean EOF propagation.
            try:
                adb_stdout.close()
            except (AttributeError, OSError):
                pass
            with self._process_lock:
                if self._stop_event.is_set() or self._closed:
                    self._terminate_process(decoder_process)
                    self._terminate_process(adb_process)
                    return
                self._adb_process = adb_process
                self._decoder_process = decoder_process
        except Exception:
            if adb_stdout is not None:
                try:
                    adb_stdout.close()
                except (AttributeError, OSError):
                    pass
            if adb_process is not None:
                self._terminate_process(adb_process)
            raise

    @staticmethod
    def _terminate_process(process: Any | None) -> None:
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
        except (AttributeError, OSError):
            pass
        try:
            process.wait(timeout=1.0)
            return
        except (AttributeError, OSError, subprocess.TimeoutExpired):
            pass
        try:
            process.kill()
            process.wait(timeout=1.0)
        except (AttributeError, OSError, subprocess.TimeoutExpired):
            pass

    def _stop_segment(self) -> None:
        with self._process_lock:
            decoder = self._decoder_process
            adb_process = self._adb_process
            self._decoder_process = None
            self._adb_process = None
        self._terminate_process(decoder)
        self._terminate_process(adb_process)

    def _read_exact(self, stream: Any, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        stream_fd: int | None = None
        try:
            stream_fd = int(stream.fileno())
        except (AttributeError, OSError, TypeError, ValueError):
            # Small in-memory test streams and alternate decoders may not
            # expose a selectable file descriptor.
            pass
        while remaining > 0:
            if self._stop_event.is_set():
                return b""
            if stream_fd is not None:
                import select

                ready, _writeable, _exceptional = select.select(
                    [stream_fd],
                    [],
                    [],
                    self.stream_read_timeout_s,
                )
                if not ready:
                    raise PrototypeControllerError(
                        "ffmpeg H.264 stream produced no data before the read timeout"
                    )
            chunk = stream.read(remaining)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise PrototypeControllerError("ffmpeg stream returned a non-byte frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _publish(self, image: Any) -> None:
        timestamp_s = self._monotonic_fn()
        with self._condition:
            frame = SourceFrame(
                image=image,
                frame_index=self._next_frame_index,
                timestamp_s=timestamp_s,
            )
            self._next_frame_index += 1
            self._latest = frame
            self._condition.notify_all()
        # A live decoded frame is stronger transport evidence than the old
        # pre-tap device-info probe. The actual tap still carries -s <serial>
        # and fails closed if the device vanishes between frames.
        mark_alive = getattr(self.controller, "mark_transport_alive", None)
        if callable(mark_alive):
            mark_alive()

    def _set_error(self, error: Exception) -> None:
        with self._condition:
            self._error = error
            self._condition.notify_all()

    def _reader_loop(self) -> None:
        import numpy as np

        consecutive_failures = 0
        try:
            while not self._stop_event.is_set():
                try:
                    self._start_segment()
                    with self._process_lock:
                        decoder = self._decoder_process
                    if decoder is None or getattr(decoder, "stdout", None) is None:
                        raise PrototypeControllerError("ffmpeg decoder did not expose stdout")
                    if self._stream_width is None or self._stream_height is None:
                        raise PrototypeControllerError("stream dimensions were not initialized")
                    frame_bytes = self._stream_width * self._stream_height * 3
                    while not self._stop_event.is_set():
                        payload = self._read_exact(decoder.stdout, frame_bytes)
                        if len(payload) != frame_bytes:
                            if self._stop_event.is_set():
                                return
                            raise PrototypeControllerError(
                                "ffmpeg H.264 stream ended before a complete frame"
                            )
                        image = np.frombuffer(payload, dtype=np.uint8).reshape(
                            (self._stream_height, self._stream_width, 3)
                        ).copy()
                        self._publish(image)
                        consecutive_failures = 0
                except Exception as error:
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_consecutive_failures:
                        self._set_error(
                            PrototypeControllerError(
                                f"ADB H.264 stream failed {consecutive_failures} times: {error}"
                            )
                        )
                        return
                finally:
                    self._stop_segment()
                if self._stop_event.is_set():
                    return
                if self.restart_delay_s:
                    self._sleep_fn(self.restart_delay_s)
        finally:
            self._stop_segment()
            with self._condition:
                self._condition.notify_all()

    def _ensure_started(self) -> None:
        with self._condition:
            if self._closed:
                raise PrototypeControllerError("ADB H.264 frame source is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._reader_loop,
                    name="adb-h264-frame-reader",
                    daemon=True,
                )
                self._thread.start()

    def next_frame(self) -> SourceFrame:
        self._ensure_started()
        with self._condition:
            while True:
                if self._error is not None:
                    raise self._error
                latest = self._latest
                if latest is not None and latest.frame_index != self._last_delivered_frame_index:
                    self._last_delivered_frame_index = latest.frame_index
                    return latest
                if self._closed:
                    raise PrototypeControllerError("ADB H.264 frame source is closed")
                self._condition.wait(timeout=0.5)

    def _latest_for_action(self) -> SourceFrame:
        self._ensure_started()
        deadline = self._monotonic_fn() + self.action_frame_wait_timeout_s
        with self._condition:
            while True:
                if self._error is not None:
                    raise self._error
                latest = self._latest
                if latest is not None:
                    age_s = self._monotonic_fn() - latest.timestamp_s
                    if 0.0 <= age_s <= self.action_frame_max_age_s:
                        return latest
                remaining_s = deadline - self._monotonic_fn()
                if remaining_s <= 0:
                    raise PrototypeControllerError(
                        "no recent decoded stream frame is available for card verification"
                    )
                self._condition.wait(timeout=min(0.25, remaining_s))

    def frame_for_action(self) -> Any:
        """Encode a recent decoded frame for the existing card matcher."""

        source_frame = self._latest_for_action()
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - environment-specific
            raise PrototypeControllerError(
                "stream card verification requires OpenCV"
            ) from error
        encoded_ok, encoded = cv2.imencode(
            ".png",
            source_frame.image,
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
        if not encoded_ok:
            raise PrototypeControllerError("could not encode decoded stream frame for verification")
        try:
            from .devices import Frame
        except ImportError:  # pragma: no cover - direct script execution
            from simulator.physical_lab.devices import Frame
        return Frame(
            source_device=self.controller.device_id,
            frame_index=source_frame.frame_index,
            workstation_monotonic_us=int(round(source_frame.timestamp_s * 1_000_000)),
            payload=encoded.tobytes(),
            capture_uncertainty_us=25_000,
        )

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            self._condition.notify_all()
        self._stop_segment()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)


def decode_adb_screenshot(frame: Any) -> Any:
    """Decode an ``AdbPhoneController.screenshot`` PNG into a BGR image."""

    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment-specific
        raise PrototypeControllerError(
            "ADB inference requires OpenCV and NumPy"
        ) from error

    payload = getattr(frame, "payload", None)
    if not isinstance(payload, bytes) or not payload:
        raise PrototypeControllerError("ADB screenshot did not contain PNG bytes")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise PrototypeControllerError("ADB screenshot could not be decoded as an image")
    return image


class AdbScreenshotSource:
    """Poll one explicitly selected Android device through ADB."""

    def __init__(
        self,
        controller: Any,
        *,
        decode: Callable[[Any], Any] = decode_adb_screenshot,
        max_attempts: int = 3,
        retry_delay_s: float = 0.15,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if type(max_attempts) is not int or max_attempts <= 0:
            raise PrototypeControllerError("ADB screenshot max_attempts must be positive")
        if not math.isfinite(float(retry_delay_s)) or float(retry_delay_s) < 0:
            raise PrototypeControllerError(
                "ADB screenshot retry_delay_s must be finite and non-negative"
            )
        self.controller = controller
        self.decode = decode
        self.max_attempts = max_attempts
        self.retry_delay_s = float(retry_delay_s)
        self.sleep_fn = sleep_fn

    def next_frame(self) -> SourceFrame:
        last_error: PrototypeControllerError | None = None
        for attempt in range(self.max_attempts):
            frame = self.controller.screenshot()
            try:
                image = self.decode(frame)
            except PrototypeControllerError as error:
                last_error = error
                if attempt + 1 < self.max_attempts and self.retry_delay_s:
                    self.sleep_fn(self.retry_delay_s)
                continue
            timestamp_s = float(getattr(frame, "workstation_monotonic_us")) / 1_000_000.0
            return SourceFrame(
                image=image,
                frame_index=int(getattr(frame, "frame_index")),
                timestamp_s=timestamp_s,
            )
        raise PrototypeControllerError(
            f"ADB screenshot remained invalid after {self.max_attempts} attempts: {last_error}"
        ) from last_error

    def close(self) -> None:
        """ADB screenshots have no persistent process to release."""


class VideoFrameSource:
    """Read a recorded video for offline/dry-run policy inspection."""

    def __init__(self, path: str | Path, *, frame_stride: int = 1) -> None:
        if type(frame_stride) is not int or frame_stride <= 0:
            raise PrototypeControllerError("video frame_stride must be positive")
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - environment-specific
            raise PrototypeControllerError("video inference requires OpenCV") from error

        self.path = Path(path)
        if not self.path.is_file():
            raise PrototypeControllerError(f"video file does not exist: {self.path}")
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise PrototypeControllerError(f"could not open video: {self.path}")
        self._frame_stride = frame_stride
        self._read_index = 0
        raw_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None

    def next_frame(self) -> SourceFrame | None:
        while True:
            ok, image = self._capture.read()
            if not ok:
                return None
            source_index = self._read_index
            self._read_index += 1
            if source_index % self._frame_stride:
                continue
            raw_time = float(self._capture.get(self._cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not math.isfinite(raw_time) or raw_time < 0:
                raw_time = (
                    float(source_index) / self._fps
                    if self._fps is not None
                    else float(source_index)
                )
            result = SourceFrame(
                image=image,
                frame_index=source_index,
                timestamp_s=raw_time,
            )
            return result

    def close(self) -> None:
        self._capture.release()


def _array_tensor(torch: Any, value: Any, *, dtype: Any, device: Any) -> Any:
    """Copy one immutable public NumPy snapshot into ``[B=1,T=1,...]``."""

    import numpy as np

    array = np.array(value, dtype=dtype, order="C", copy=True)
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(1).to(device=device)


def observation_to_model_inputs(observation: Any, *, device: Any) -> tuple[Any, ...]:
    """Convert one ``PolicyObservationV2`` into recurrent actor inputs."""

    if not callable(getattr(observation, "structured_action_masks", None)):
        raise PrototypeControllerError(
            "prototype inference requires a public PolicyObservationV2"
        )
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment-specific
        raise PrototypeControllerError(
            "prototype inference requires PyTorch"
        ) from error
    try:
        try:
            from ..rl.trajectory import ActionMasks
        except ImportError:
            from rl.trajectory import ActionMasks
    except ImportError as error:  # pragma: no cover - environment-specific
        raise PrototypeControllerError(
            "could not import the recurrent action-mask contract"
        ) from error

    mode_mask, card_mask, placement_mask = observation.structured_action_masks()
    masks = ActionMasks(
        mode=_array_tensor(torch, mode_mask, dtype=bool, device=device),
        card=_array_tensor(torch, card_mask, dtype=bool, device=device),
        placement=_array_tensor(torch, placement_mask, dtype=bool, device=device),
    )
    return (
        _array_tensor(torch, observation.board, dtype="float32", device=device),
        _array_tensor(torch, observation.global_vector, dtype="float32", device=device),
        _array_tensor(torch, observation.entity_tokens, dtype="float32", device=device),
        _array_tensor(torch, observation.entity_mask, dtype=bool, device=device),
        masks,
    )


def policy_action_from_batch(actions: Any) -> Any:
    """Decode a one-row ``ActionBatch`` as the visual policy ``Action``."""

    mode_tensor = getattr(actions, "mode", None)
    card_tensor = getattr(actions, "card_slot", None)
    placement_tensor = getattr(actions, "placement", None)
    if mode_tensor is None or card_tensor is None or placement_tensor is None:
        raise PrototypeControllerError("actor did not return an ActionBatch")
    if int(mode_tensor.numel()) != 1 or int(card_tensor.numel()) != 1:
        raise PrototypeControllerError("actor action batch must contain one row")
    try:
        from cr_bot.domain.game_state import Action
    except ImportError as error:  # pragma: no cover - environment-specific
        raise PrototypeControllerError("cr_bot is required to decode actor actions") from error

    mode = int(mode_tensor.reshape(-1)[0].item())
    if mode == 0:
        return Action(kind="Wait")
    if mode != 1:
        raise PrototypeControllerError(f"actor returned unsupported mode: {mode}")
    card_slot = int(card_tensor.reshape(-1)[0].item())
    placement = placement_tensor.reshape(-1, 2)
    if placement.shape[0] != 1:
        raise PrototypeControllerError("actor placement batch must contain one row")
    row = int(placement[0, 0].item())
    column = int(placement[0, 1].item())
    if not 0 <= card_slot < 4 or not 0 <= column < 18 or not 0 <= row < 32:
        raise PrototypeControllerError(
            f"actor returned an out-of-range play: slot={card_slot}, cell={(column, row)}"
        )
    return Action(kind="Play", card_idx=card_slot, cell=(column, row))


class PrototypeActor:
    """Frozen recurrent actor loaded from a prototype checkpoint."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - environment-specific
            raise PrototypeControllerError("prototype inference requires PyTorch") from error
        try:
            try:
                from ..rl.prototype import load_shadow_prototype_checkpoint
            except ImportError:
                from rl.prototype import load_shadow_prototype_checkpoint
            learner, config, metadata = load_shadow_prototype_checkpoint(
                checkpoint,
                device=device,
            )
        except Exception as error:
            raise PrototypeControllerError(
                f"could not load prototype checkpoint {Path(checkpoint)}: {error}"
            ) from error
        self.learner = learner
        self.config = config
        self.metadata = metadata
        self.policy = learner.policy
        self.policy.eval()
        self._torch = torch
        self._hidden: Any | None = None

    @property
    def device(self) -> Any:
        return self.learner.device

    def reset(self) -> None:
        """Reset the GRU before the next detected match."""

        self._hidden = None

    def decide(self, observation: Any) -> Any:
        inputs = observation_to_model_inputs(observation, device=self.device)
        if self._hidden is None:
            hidden = self.policy.initial_hidden(1, device=self.device)
            reset_mask = self._torch.ones(
                (1, 1), dtype=self._torch.bool, device=self.device
            )
        else:
            hidden = self._hidden
            reset_mask = self._torch.zeros(
                (1, 1), dtype=self._torch.bool, device=self.device
            )
        with self._torch.inference_mode():
            actions, next_hidden = self.policy.act_deterministic(
                *inputs,
                reset_mask=reset_mask,
                hidden=hidden,
            )
        self._hidden = next_hidden.detach()
        return policy_action_from_batch(actions)


def _action_kind(action: Any) -> str:
    return str(getattr(action, "kind", "")).strip().casefold().replace("_", "-")


def action_to_dict(action: Any) -> dict[str, object]:
    """Serialize a visual policy action without serializing tensors."""

    kind = _action_kind(action)
    if kind in {"wait", "noop", "no-op"}:
        return {"kind": "wait"}
    if kind != "play":
        raise PrototypeControllerError(f"unsupported action kind: {kind!r}")
    slot = getattr(action, "card_idx", None)
    cell = getattr(action, "cell", None)
    if type(slot) is not int or not isinstance(cell, tuple) or len(cell) != 2:
        raise PrototypeControllerError("play action is missing card slot or cell")
    return {
        "kind": "play",
        "card_slot": slot,
        "cell": [int(cell[0]), int(cell[1])],
    }


def _filter_live_analysis(analysis: Any) -> Any:
    """Remove invalid live detections before policy/tracker processing.

    ``bomb`` is an object/projectile label in KataCR, not a card-backed unit
    in the prototype observation contract.  Ally detections are also
    restricted to the fixed 2.6 Hog deck, because an own-side detector label
    outside that deck cannot be a valid policy input for this live agent.
    Keep these exclusions local to this runner: the shared extractor output
    and simulator mechanics remain unchanged for replay, training, and other
    consumers.
    """

    matches = getattr(analysis, "matches", None)
    if not isinstance(matches, (list, tuple)):
        return analysis

    filtered_matches = []
    for match in matches:
        troop = getattr(match, "troop", None)
        label = getattr(troop, "class_name", None)
        normalized = (
            label.strip().casefold().replace("_", "-")
            if isinstance(label, str)
            else None
        )
        if normalized in LIVE_IGNORED_DETECTOR_LABELS:
            continue
        team = getattr(troop, "team", None)
        if (
            isinstance(team, str)
            and team.strip().casefold() == "ally"
            and _live_own_card_name(label) not in LIVE_OWN_DECK_CARD_NAMES
        ):
            continue
        filtered_matches.append(match)

    if len(filtered_matches) == len(matches):
        return analysis
    try:
        return replace(analysis, matches=filtered_matches)
    except TypeError:
        # Production FrameAnalysisResult is a dataclass.  Do not mutate an
        # arbitrary injected test double if it is not replace-compatible.
        return analysis


def _live_own_card_name(label: Any) -> str | None:
    if not isinstance(label, str):
        return None
    normalized = label.strip().casefold().replace("_", "-")
    if not normalized:
        return None
    return LIVE_OWN_DETECTOR_ALIASES.get(normalized, normalized)


@dataclass(slots=True)
class _LiveOwnSeed:
    """A just-dispatched own card waiting for the detector to see it."""

    card_name: str
    center_x: float
    center_y: float
    created_update: int
    created_timestamp_s: float | None


class LiveDetectionFilter:
    """Make live unit input conservative without changing the shared extractor.

    Own cards are constrained by the configured live deck.  After a successful
    dispatch, the selected card is also inserted at the exact policy cell for
    a short bridge period, so the policy does not see an empty board while the
    detector catches up with the new troop.  Enemy detections are admitted
    only after the same tracked label has appeared in two processed frames.

    This filter is intentionally a live-runner boundary.  Replays, training,
    and the common visual extractor continue to receive the unmodified match
    list.
    """

    def __init__(
        self,
        *,
        enemy_confirmation_frames: int = LIVE_ENEMY_CONFIRMATION_FRAMES,
        own_seed_max_age_s: float = LIVE_OWN_SEED_MAX_AGE_S,
        own_seed_match_radius_px: float = LIVE_OWN_SEED_MATCH_RADIUS_PX,
        own_seed_max_updates: int = 8,
    ) -> None:
        if (
            type(enemy_confirmation_frames) is not int
            or enemy_confirmation_frames <= 0
        ):
            raise PrototypeControllerError(
                "enemy_confirmation_frames must be a positive integer"
            )
        if not math.isfinite(float(own_seed_max_age_s)) or own_seed_max_age_s < 0:
            raise PrototypeControllerError(
                "own_seed_max_age_s must be finite and non-negative"
            )
        if (
            not math.isfinite(float(own_seed_match_radius_px))
            or own_seed_match_radius_px < 0
        ):
            raise PrototypeControllerError(
                "own_seed_match_radius_px must be finite and non-negative"
            )
        if type(own_seed_max_updates) is not int or own_seed_max_updates <= 0:
            raise PrototypeControllerError(
                "own_seed_max_updates must be a positive integer"
            )
        self.enemy_confirmation_frames = enemy_confirmation_frames
        self.own_seed_max_age_s = float(own_seed_max_age_s)
        self.own_seed_match_radius_px = float(own_seed_match_radius_px)
        self.own_seed_max_updates = own_seed_max_updates
        self.reset()

    def reset(self) -> None:
        self._update_index = 0
        # key -> (canonical label, consecutive observations, last update)
        self._enemy_candidates: dict[tuple[Any, ...], tuple[str, int, int]] = {}
        self._own_seeds: list[_LiveOwnSeed] = []

    def notify_own_play(
        self,
        *,
        card_name: Any,
        cell: Any,
        arena_px: Any,
        timestamp_s: float | None = None,
    ) -> None:
        """Seed the policy state after a card was actually dispatched."""

        if isinstance(card_name, (tuple, list)) and card_name:
            card_name = card_name[0]
        canonical_name = _live_own_card_name(card_name)
        if canonical_name not in LIVE_OWN_TROOP_CARD_NAMES:
            # Spells do not create a persistent unit to put into the board.
            return
        if not isinstance(cell, (tuple, list)) or len(cell) != 2:
            return
        try:
            col, row = int(cell[0]), int(cell[1])
            arena = tuple(float(value) for value in arena_px)
            if (
                len(arena) != 4
                or not all(math.isfinite(value) for value in arena)
                or arena[2] <= 0
                or arena[3] <= 0
            ):
                return
            from cr_bot.features.action_space import ACTION_GRID

            center_x, center_y = ACTION_GRID.cell_to_pixel_center(col, row, arena)
        except (TypeError, ValueError, OverflowError):
            # A malformed/incomplete frame must never create a fake unit.
            return

        self._own_seeds.append(
            _LiveOwnSeed(
                card_name=canonical_name,
                center_x=float(center_x),
                center_y=float(center_y),
                created_update=self._update_index,
                created_timestamp_s=_finite_float(timestamp_s),
            )
        )

    def update(self, analysis: Any, *, timestamp_s: float | None = None) -> Any:
        """Return the policy-facing analysis with live temporal gates applied."""

        matches = getattr(analysis, "matches", None)
        if not isinstance(matches, (list, tuple)):
            return analysis

        self._update_index += 1
        current_enemy_keys: set[tuple[Any, ...]] = set()
        seen_enemy_keys: set[tuple[Any, ...]] = set()
        filtered_matches: list[Any] = []
        for match in matches:
            troop = getattr(match, "troop", None)
            team = self._team_name(troop)
            if team != "enemy":
                # Ally matches have already passed the fixed-deck filter.  No
                # two-frame delay is needed on our side: a dispatched card is
                # known immediately, and existing own units can be used from
                # the first live frame.
                filtered_matches.append(match)
                continue

            key = self._enemy_key(troop)
            if key in seen_enemy_keys:
                # A duplicate tracker row must not count as another frame.
                continue
            seen_enemy_keys.add(key)
            current_enemy_keys.add(key)
            label = self._canonical_label(troop)
            candidate = self._enemy_candidates.get(key)
            if (
                candidate is not None
                and candidate[0] == label
                and candidate[2] == self._update_index - 1
            ):
                count = candidate[1] + 1
            else:
                count = 1
            self._enemy_candidates[key] = (label, count, self._update_index)
            if count >= self.enemy_confirmation_frames:
                filtered_matches.append(match)

        # A disappeared track must start over if it returns.  This prevents a
        # stale one-frame false positive from being promoted much later.
        self._enemy_candidates = {
            key: candidate
            for key, candidate in self._enemy_candidates.items()
            if key in current_enemy_keys
        }
        filtered_matches = self._add_own_seeds(
            filtered_matches,
            timestamp_s=_finite_float(timestamp_s),
        )

        original_matches = list(matches)
        if filtered_matches == original_matches:
            return analysis
        try:
            return replace(analysis, matches=filtered_matches)
        except TypeError:
            # Production FrameAnalysisResult is a dataclass.  Keep injected
            # non-dataclass test doubles non-mutating, as the existing filter
            # does.
            return analysis

    @staticmethod
    def _team_name(troop: Any) -> str:
        team = getattr(troop, "team", "")
        return team.strip().casefold() if isinstance(team, str) else ""

    @staticmethod
    def _canonical_label(troop: Any) -> str:
        label = getattr(troop, "class_name", "")
        canonical = _live_own_card_name(label)
        return canonical or ""

    @classmethod
    def _enemy_key(cls, troop: Any) -> tuple[Any, ...]:
        track_id = getattr(troop, "track_id", None)
        if track_id is not None:
            try:
                hash(track_id)
            except TypeError:
                pass
            else:
                return ("track", track_id)

        # ByteTrack normally supplies an id.  This fallback still requires two
        # observations for an untracked row, provided it remains in roughly
        # the same area of the frame.
        label = cls._canonical_label(troop)
        center_x = _finite_float(getattr(troop, "center_x", None))
        center_y = _finite_float(getattr(troop, "center_y", None))
        if center_x is None or center_y is None:
            return ("untracked", label)
        return ("position", label, round(center_x / 100.0), round(center_y / 100.0))

    def _add_own_seeds(
        self,
        matches: list[Any],
        *,
        timestamp_s: float | None,
    ) -> list[Any]:
        if not self._own_seeds:
            return matches

        raw_own_matches = [
            match
            for match in matches
            if self._team_name(getattr(match, "troop", None)) == "ally"
        ]
        retained_seeds: list[_LiveOwnSeed] = []
        synthetic_matches: list[Any] = []
        for seed in self._own_seeds:
            if self._seed_expired(seed, timestamp_s):
                continue
            if any(self._seed_matches_raw(seed, match) for match in raw_own_matches):
                # The detector has caught up.  Use its current position/HP and
                # do not duplicate it with the temporary placement seed.
                continue
            retained_seeds.append(seed)
            synthetic_matches.append(self._synthetic_match(seed))
        self._own_seeds = retained_seeds
        return matches + synthetic_matches

    def _seed_expired(
        self,
        seed: _LiveOwnSeed,
        timestamp_s: float | None,
    ) -> bool:
        if (
            timestamp_s is not None
            and seed.created_timestamp_s is not None
            and timestamp_s >= seed.created_timestamp_s
        ):
            return timestamp_s - seed.created_timestamp_s > self.own_seed_max_age_s
        return self._update_index - seed.created_update > self.own_seed_max_updates

    def _seed_matches_raw(self, seed: _LiveOwnSeed, match: Any) -> bool:
        troop = getattr(match, "troop", None)
        if self._canonical_label(troop) != seed.card_name:
            return False
        center_x = _finite_float(getattr(troop, "center_x", None))
        center_y = _finite_float(getattr(troop, "center_y", None))
        if center_x is None or center_y is None:
            return False
        return math.hypot(center_x - seed.center_x, center_y - seed.center_y) <= (
            self.own_seed_match_radius_px
        )

    @staticmethod
    def _synthetic_match(seed: _LiveOwnSeed) -> Any:
        from cr_bot.domain.card_metadata import CARD_METADATA
        from cr_bot.domain.game_state import Detection, Match

        metadata = CARD_METADATA.get(seed.card_name, {})
        hitpoints = _finite_float(metadata.get("hitpoints"))
        troop = Detection(
            track_id=None,
            class_name=seed.card_name,
            team="ally",
            confidence=1.0,
            x1=seed.center_x - 2.0,
            y1=seed.center_y - 2.0,
            x2=seed.center_x + 2.0,
            y2=seed.center_y + 2.0,
            center_x=seed.center_x,
            center_y=seed.center_y,
            estimated_hp=hitpoints,
        )
        return Match(troop=troop, bar=None)


class LiveHandStateFilter:
    """Debounce live hand classifier results before policy inference.

    The shared extractor filter protects against a full-screen classification
    failure, but a card animation usually changes only one slot and therefore
    passes that filter immediately. Live control needs the stronger rule:
    every new slot identity must persist for several observations, and a
    successfully played slot is filled from the last stable next-card preview.
    That preview is then held unknown until a different next card settles, so
    the same transition prediction cannot be consumed twice.
    """

    hand_slots = ("card_1", "card_2", "card_3", "card_4")
    tracked_slots = hand_slots + ("next_card",)

    def __init__(
        self,
        *,
        confirmation_frames: int = 3,
        initial_confirmation_frames: int = 3,
        minimum_confidence: float = 55.0,
    ) -> None:
        if type(confirmation_frames) is not int or confirmation_frames <= 0:
            raise PrototypeControllerError("confirmation_frames must be positive")
        if (
            type(initial_confirmation_frames) is not int
            or initial_confirmation_frames <= 0
        ):
            raise PrototypeControllerError(
                "initial_confirmation_frames must be positive"
            )
        if not math.isfinite(float(minimum_confidence)) or minimum_confidence < 0:
            raise PrototypeControllerError(
                "minimum_confidence must be finite and non-negative"
            )
        self.confirmation_frames = confirmation_frames
        self.initial_confirmation_frames = initial_confirmation_frames
        self.minimum_confidence = float(minimum_confidence)
        self.reset()

    @property
    def ready(self) -> bool:
        """Whether all four hand slots are known and stable this update."""

        return self._ready

    def reset(self) -> None:
        self.last_state: dict[str, Any] = {
            slot: None for slot in self.tracked_slots
        }
        self._candidates: dict[str, tuple[str, Any, int]] = {}
        self._forced_replacements: set[str] = set()
        self._expected_replacements: dict[str, Any] = {}
        self._next_card_refresh_pending_name: str | None = None
        self._ready = False

    def expect_replacement(self, slot: int) -> None:
        """Record the preview that should replace a successfully played slot."""

        if type(slot) is not int or not 0 <= slot < len(self.hand_slots):
            raise PrototypeControllerError("played hand slot must be from 0 through 3")
        key = self.hand_slots[slot]
        expected = (
            None
            if self._next_card_refresh_pending_name is not None
            else self.last_state.get("next_card")
        )
        expected_name = self._card_name_if_confident(expected)
        current_name = self._card_name(self.last_state.get(key))
        if (
            expected_name is not None
            and expected_name != current_name
            and not self._occupied_by_other_hand_slot(key, expected_name)
        ):
            # The preview is the card that enters the hand after this play.
            # Capture the already-stable value before processing the next
            # frame, where the preview may still be in its transition state.
            self._expected_replacements[key] = expected
        else:
            self._expected_replacements.pop(key, None)
        self._forced_replacements.add(key)
        self._candidates.pop(key, None)
        self._ready = False

    def update(self, state: Any) -> Any:
        if not isinstance(state, dict):
            self.reset()
            return state

        filtered = state.copy()
        unstable_hand = False
        for slot in self.tracked_slots:
            observed = state.get(slot)
            observed_name = self._card_name_if_confident(observed)
            previous = self.last_state.get(slot)
            previous_name = self._card_name(previous)
            forced_replacement = slot in self._forced_replacements

            if forced_replacement:
                expected = self._expected_replacements.get(slot)
                expected_name = self._card_name_if_confident(expected)
                if expected_name is not None and not self._occupied_by_other_hand_slot(
                    slot, expected_name
                ):
                    # The live UI can expose a blank/old card while the hand
                    # animation is running.  The stable preview captured at
                    # dispatch time is authoritative for this one replacement.
                    self.last_state[slot] = expected
                    self._candidates.pop(slot, None)
                    self._forced_replacements.discard(slot)
                    self._expected_replacements.pop(slot, None)
                    self._consume_next_card_preview(expected_name)
                    filtered[slot] = expected
                    continue

            if observed_name is None:
                if (
                    slot == "next_card"
                    and self._next_card_refresh_pending_name is not None
                ):
                    self.last_state[slot] = None
                self._candidates.pop(slot, None)
                filtered[slot] = (
                    None if previous is None or forced_replacement else previous
                )
                if (
                    slot == "next_card"
                    and self._next_card_refresh_pending_name is not None
                ):
                    filtered[slot] = None
                if slot in self.hand_slots:
                    unstable_hand = True
                continue

            if (
                slot == "next_card"
                and self._next_card_refresh_pending_name is not None
                and observed_name == self._next_card_refresh_pending_name
            ):
                # The preview often lingers on the card that just entered the
                # hand for a few decoded frames.  Do not accept that stale
                # value as the new preview.
                self.last_state[slot] = None
                self._candidates.pop(slot, None)
                filtered[slot] = None
                continue

            if not forced_replacement and previous_name == observed_name:
                self._candidates.pop(slot, None)
                self.last_state[slot] = observed
                filtered[slot] = observed
                continue

            if previous_name is not None and observed_name == previous_name:
                # A played slot is not allowed to remain the old card even if
                # the animation briefly makes the classifier repeat it.
                self._candidates.pop(slot, None)
                filtered[slot] = None if forced_replacement else previous
                if slot in self.hand_slots:
                    unstable_hand = True
                continue

            candidate = self._candidates.get(slot)
            if candidate is not None and candidate[0] == observed_name:
                candidate = (observed_name, observed, candidate[2] + 1)
            else:
                candidate = (observed_name, observed, 1)
            self._candidates[slot] = candidate

            required = (
                self.initial_confirmation_frames
                if previous_name is None
                else self.confirmation_frames
            )
            accepted = candidate[2] >= required
            if accepted and slot in self.hand_slots:
                if self._occupied_by_other_hand_slot(slot, observed_name):
                    accepted = False

            if accepted:
                self.last_state[slot] = candidate[1]
                self._candidates.pop(slot, None)
                self._forced_replacements.discard(slot)
                filtered[slot] = candidate[1]
                if slot == "next_card":
                    self._next_card_refresh_pending_name = None
            else:
                filtered[slot] = (
                    None if previous is None or forced_replacement else previous
                )
                if slot in self.hand_slots:
                    unstable_hand = True

        hand_names = [
            self._card_name(self.last_state.get(slot)) for slot in self.hand_slots
        ]
        self._ready = (
            not unstable_hand
            and all(name is not None for name in hand_names)
            and len(set(hand_names)) == len(hand_names)
        )
        return filtered

    def _occupied_by_other_hand_slot(self, slot: str, card_name: str) -> bool:
        return any(
            other != slot
            and self._card_name(self.last_state.get(other)) == card_name
            for other in self.hand_slots
        )

    def _consume_next_card_preview(self, card_name: str) -> None:
        self._next_card_refresh_pending_name = card_name
        self.last_state["next_card"] = None
        self._candidates.pop("next_card", None)

    def _card_name_if_confident(self, card: Any) -> str | None:
        name = self._card_name(card)
        if name is None:
            return None
        if isinstance(card, (tuple, list)) and len(card) >= 2:
            confidence = _finite_float(card[1])
            if confidence is None or confidence < self.minimum_confidence:
                return None
        return name

    @staticmethod
    def _card_name(card: Any) -> str | None:
        if isinstance(card, (tuple, list)) and card:
            card = card[0]
        if not isinstance(card, str):
            return None
        name = card.strip()
        if not name or name.casefold() == "none":
            return None
        return name


def _finite_float(value: Any) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def _safe_card_value(value: Any) -> str | None:
    return None if value is None else str(value)


def _summarize_visual_match(match: Any) -> dict[str, object] | None:
    troop = getattr(match, "troop", None)
    if troop is None:
        return None
    label = getattr(troop, "class_name", None)
    summary: dict[str, object] = {
        "label": str(label) if label is not None else "?",
        "team": str(getattr(troop, "team", "?")),
    }
    track_id = getattr(troop, "track_id", None)
    if track_id is not None:
        try:
            summary["track"] = int(track_id)
        except (TypeError, ValueError, OverflowError):
            summary["track"] = str(track_id)
    confidence = _finite_float(getattr(troop, "confidence", None))
    if confidence is not None:
        summary["confidence"] = round(confidence, 3)
    center_x = _finite_float(getattr(troop, "center_x", None))
    center_y = _finite_float(getattr(troop, "center_y", None))
    if center_x is not None and center_y is not None:
        summary["center_px"] = [round(center_x, 1), round(center_y, 1)]
    estimated_hp = _finite_float(getattr(troop, "estimated_hp", None))
    if estimated_hp is not None:
        summary["estimated_hp"] = round(estimated_hp, 1)
    return summary


def _summarize_extracted_visual_state(step: Any) -> dict[str, object]:
    """Make the extractor state readable without exposing NumPy/tensor objects."""

    analysis = getattr(step, "analysis", None)
    game_state = getattr(step, "game_state", None)
    hud = getattr(game_state, "hud", None)
    raw_matches = getattr(analysis, "matches", ())
    matches = raw_matches if isinstance(raw_matches, (list, tuple)) else ()
    detections = [
        summary
        for match in matches
        if (summary := _summarize_visual_match(match)) is not None
    ]

    raw_hand = getattr(hud, "hand_cards", ())
    hand = raw_hand if isinstance(raw_hand, (list, tuple)) else ()
    raw_seen = getattr(game_state, "seen_enemy_cards", ())
    seen = raw_seen if isinstance(raw_seen, (list, tuple, set)) else ()
    safe_seen: list[object] = []
    for card in sorted(seen, key=str):
        try:
            safe_seen.append(int(card))
        except (TypeError, ValueError, OverflowError):
            safe_seen.append(str(card))

    def safe_hp_triplet(value: Any) -> list[float | None]:
        if not isinstance(value, (list, tuple)):
            return []
        return [_finite_float(item) for item in value]

    return {
        "hand": [_safe_card_value(card) for card in hand],
        "next_card": _safe_card_value(getattr(hud, "next_card", None)),
        "elixir": _finite_float(getattr(hud, "elixir_self", None)),
        "enemy_elixir_est": _finite_float(
            getattr(game_state, "elixir_enemy_est", None)
        ),
        "time_left_s": _finite_float(getattr(hud, "time_left_s", None)),
        "total_remaining_s": _finite_float(
            getattr(game_state, "total_remaining_s", None)
        ),
        "overtime": bool(getattr(hud, "overtime", False)),
        "ally_units": [
            item for item in detections if item.get("team") == "ally"
        ],
        "enemy_units": [
            item for item in detections if item.get("team") == "enemy"
        ],
        "seen_enemy_cards": safe_seen,
        "tower_hp_self": safe_hp_triplet(getattr(hud, "tower_hp_self", None)),
        "tower_hp_enemy": safe_hp_triplet(getattr(hud, "tower_hp_enemy", None)),
        "detection_count": len(detections),
        "arena_px": list(getattr(analysis, "arena_px", ()))
        if isinstance(getattr(analysis, "arena_px", None), (list, tuple))
        else [],
    }


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """JSONL-friendly result of processing one source frame."""

    frame_index: int
    timestamp_s: float
    in_game: bool
    emitted: bool
    action: dict[str, object] | None
    result: str
    hand_cards: tuple[str, ...] = ()
    elixir: float | None = None
    time_left_s: float | None = None
    detection_count: int | None = None
    visual_state: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "in_game": self.in_game,
            "emitted": self.emitted,
            "action": self.action,
            "result": self.result,
            "hand_cards": list(self.hand_cards),
        }
        if self.elixir is not None:
            result["elixir"] = self.elixir
        if self.time_left_s is not None:
            result["time_left_s"] = self.time_left_s
        if self.detection_count is not None:
            result["detection_count"] = self.detection_count
        if self.visual_state is not None:
            result["visual_state"] = self.visual_state
        return result


def _format_card_list(cards: Any) -> str:
    if not isinstance(cards, (list, tuple)) or not cards:
        return "—"
    return " · ".join("?" if card is None else str(card) for card in cards)


def _format_number(value: Any, *, digits: int = 1, suffix: str = "") -> str:
    number = _finite_float(value)
    return "—" if number is None else f"{number:.{digits}f}{suffix}"


def _format_detection_list(detections: Any) -> str:
    if not isinstance(detections, (list, tuple)) or not detections:
        return "—"
    formatted: list[str] = []
    for detection in detections:
        if not isinstance(detection, dict):
            formatted.append(str(detection))
            continue
        label = str(detection.get("label", "?"))
        track = detection.get("track")
        track_text = f"#{track}" if track is not None else ""
        center = detection.get("center_px")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            position = f"@{center[0]},{center[1]}"
        else:
            position = ""
        health = _format_number(detection.get("estimated_hp"), digits=0)
        confidence = _format_number(detection.get("confidence"), digits=2)
        details = f"{label}{track_text} hp={health}"
        if position:
            details += f" {position}"
        formatted.append(f"{details} [{confidence}]")
    return "  ·  ".join(formatted)


def _format_tower_hp(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        return "—"
    return "[" + ", ".join(_format_number(item, digits=0) for item in value) + "]"


def _format_seen_enemy_cards(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        return "—"
    return "[" + ", ".join(str(card) for card in value) + "]"


def format_decision_record(record: DecisionRecord) -> str:
    """Render one extracted-state/action decision for the live terminal."""

    state = record.visual_state or {}
    action = record.action or {"kind": "none"}
    action_text = repr(action)
    state_line = (
        f"hand=[{_format_card_list(state.get('hand', record.hand_cards))}]  "
        f"next={state.get('next_card') or '—'}  "
        f"elixir={_format_number(state.get('elixir', record.elixir))}/10  "
        f"enemy≈{_format_number(state.get('enemy_elixir_est'))}/10  "
        f"clock={_format_number(state.get('time_left_s', record.time_left_s), suffix='s')}  "
        f"overtime={'yes' if state.get('overtime') else 'no'}"
    )
    visual_line = (
        f"ally=[{_format_detection_list(state.get('ally_units'))}]  "
        f"enemy=[{_format_detection_list(state.get('enemy_units'))}]  "
        f"detections={state.get('detection_count', record.detection_count) or 0}"
    )
    towers_line = (
        f"self={_format_tower_hp(state.get('tower_hp_self'))}  "
        f"enemy={_format_tower_hp(state.get('tower_hp_enemy'))}  "
        f"seen_enemy={_format_seen_enemy_cards(state.get('seen_enemy_cards'))}"
    )
    raw_lines = [
        f"╭─ frame={record.frame_index} · t={record.timestamp_s:.3f}s",
        f"│ EXTRACTED  {state_line}",
        f"│ VISUAL     {visual_line}",
        f"│ TOWERS     {towers_line}",
        f"│ DECISION   action={action_text}  result={record.result}",
        "╰─",
    ]
    wrapped: list[str] = []
    for line in raw_lines:
        if line.startswith("│ "):
            prefix, _, content = line.partition("  ")
            # Detection lists are intentionally given a wider compact row so
            # the usual handful of troops remains horizontal.  Very large
            # lists can still wrap safely at the terminal boundary.
            line_width = 180 if prefix == "│ VISUAL" else 116
            chunks = textwrap.wrap(
                content,
                width=line_width - len(prefix),
                initial_indent=prefix + "  ",
                subsequent_indent="│            ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped.extend(chunks or [prefix])
        else:
            wrapped.append(line)
    return "\n".join(wrapped)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Aggregate counters for one controller invocation."""

    frames: int
    emitted_frames: int
    waits: int
    proposed_plays: int
    dispatched_plays: int

    def as_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "emitted_frames": self.emitted_frames,
            "waits": self.waits,
            "proposed_plays": self.proposed_plays,
            "dispatched_plays": self.dispatched_plays,
        }


class LivePrototypeRunner:
    """Connect extraction, recurrent inference, and optional device actions."""

    def __init__(
        self,
        frame_source: FrameSource,
        detector: Any,
        actor: Any,
        *,
        execute: bool = False,
        phone: Any | None = None,
        calibration: Any | None = None,
        normalize: bool = True,
        yolo_tower_hp_detections: bool = False,
        poll_interval_s: float = 0.25,
        min_action_interval_s: float = 0.75,
        post_action_delay_s: float = 0.35,
        session: Any | None = None,
        process_frame_fn: Callable[..., Any] | None = None,
        normalize_frame_fn: Callable[[Any], Any] | None = None,
        observation_builder: Callable[[Any], Any] | None = None,
        dispatch_fn: Callable[..., Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(execute) is not bool:
            raise PrototypeControllerError("execute must be boolean")
        if execute and (phone is None or calibration is None):
            raise PrototypeControllerError(
                "live execution requires both an AutonomousPhone and calibration"
            )
        for name, value in (
            ("poll_interval_s", poll_interval_s),
            ("min_action_interval_s", min_action_interval_s),
            ("post_action_delay_s", post_action_delay_s),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise PrototypeControllerError(f"{name} must be finite and non-negative")
        live_hand_filter_enabled = session is None
        live_detection_filter_enabled = session is None
        if session is None or process_frame_fn is None or normalize_frame_fn is None:
            _bootstrap_extractor_runtime()
            try:
                from cr_bot.app.match_session import MatchSession
                from cr_bot.app.pipeline import normalize_frame, process_frame
            except ImportError as error:  # pragma: no cover - environment-specific
                raise PrototypeControllerError(
                    "the cr_bot visual extractor is not importable"
                ) from error
            # Live runs keep enemy-card tracking enabled, but do not emit the
            # tracker’s per-detection diagnostics to the terminal.
            session = (
                MatchSession(tracker_debug=False) if session is None else session
            )
            if live_hand_filter_enabled:
                session.hand_state_filter = LiveHandStateFilter()
            process_frame_fn = process_frame if process_frame_fn is None else process_frame_fn
            normalize_frame_fn = (
                normalize_frame if normalize_frame_fn is None else normalize_frame_fn
            )
        if observation_builder is None:
            try:
                from .policy_bridge import observation_v2_from_match_step
            except ImportError:
                from simulator.physical_lab.policy_bridge import observation_v2_from_match_step
            observation_builder = observation_v2_from_match_step
        if dispatch_fn is None:
            try:
                from .policy_bridge import dispatch_policy_action
            except ImportError:
                from simulator.physical_lab.policy_bridge import dispatch_policy_action
            dispatch_fn = dispatch_policy_action

        self.frame_source = frame_source
        self.detector = detector
        self.actor = actor
        self.execute = execute
        self.phone = phone
        self.calibration = calibration
        self.normalize = normalize
        self.yolo_tower_hp_detections = yolo_tower_hp_detections
        self.poll_interval_s = float(poll_interval_s)
        self.min_action_interval_s = float(min_action_interval_s)
        self.post_action_delay_s = float(post_action_delay_s)
        self.session = session
        self.process_frame_fn = process_frame_fn
        self.normalize_frame_fn = normalize_frame_fn
        self.observation_builder = observation_builder
        self.dispatch_fn = dispatch_fn
        self.sleep_fn = sleep_fn
        self.monotonic_fn = monotonic_fn
        self._last_play_timestamp_s: float | None = None
        self._live_hand_filter_enabled = live_hand_filter_enabled
        self._live_detection_filter_enabled = live_detection_filter_enabled
        self._live_detection_filter = (
            LiveDetectionFilter() if live_detection_filter_enabled else None
        )

    def _ensure_live_hand_filter(self) -> None:
        if not self._live_hand_filter_enabled:
            return
        if not isinstance(
            getattr(self.session, "hand_state_filter", None), LiveHandStateFilter
        ):
            # MatchSession.reset() creates its default filter after a finished
            # match; restore the live-only filter before the next frame.
            self.session.hand_state_filter = LiveHandStateFilter()

    def _live_hand_state_ready(self) -> bool:
        if not self._live_hand_filter_enabled:
            return True
        return bool(getattr(self.session.hand_state_filter, "ready", False))

    def _notify_live_play(
        self,
        action: Any,
        *,
        step: Any | None = None,
        arena_px: Any = None,
        timestamp_s: float | None = None,
    ) -> None:
        if self._live_hand_filter_enabled:
            hand_filter = getattr(self.session, "hand_state_filter", None)
            if isinstance(hand_filter, LiveHandStateFilter):
                hand_filter.expect_replacement(getattr(action, "card_idx", -1))

        if not self._live_detection_filter_enabled:
            return
        detection_filter = self._live_detection_filter
        if not isinstance(detection_filter, LiveDetectionFilter):
            return
        hud = getattr(getattr(step, "game_state", None), "hud", None)
        hand_cards = getattr(hud, "hand_cards", ())
        slot = getattr(action, "card_idx", None)
        card_name = None
        if (
            isinstance(hand_cards, (list, tuple))
            and type(slot) is int
            and 0 <= slot < len(hand_cards)
        ):
            card_name = hand_cards[slot]
        detection_filter.notify_own_play(
            card_name=card_name,
            cell=getattr(action, "cell", None),
            arena_px=arena_px,
            timestamp_s=timestamp_s,
        )

    def run(
        self,
        *,
        max_frames: int | None = None,
        on_record: Callable[[DecisionRecord], None] | None = None,
    ) -> RunSummary:
        if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
            raise PrototypeControllerError("max_frames must be a positive integer when supplied")
        if on_record is not None and not callable(on_record):
            raise PrototypeControllerError("on_record must be callable when supplied")

        frames = emitted = waits = proposed_plays = dispatched_plays = 0
        try:
            while max_frames is None or frames < max_frames:
                started = self.monotonic_fn()
                self._ensure_live_hand_filter()
                source_frame = self.frame_source.next_frame()
                if source_frame is None:
                    break
                frames += 1
                image = source_frame.image
                if self.normalize:
                    image = self.normalize_frame_fn(image)
                analysis = self.process_frame_fn(
                    image,
                    self.detector,
                    show_rois=False,
                    yolo_tower_hp_detections=self.yolo_tower_hp_detections,
                )
                analysis = _filter_live_analysis(analysis)
                if self._live_detection_filter_enabled:
                    detection_filter = self._live_detection_filter
                    if isinstance(detection_filter, LiveDetectionFilter):
                        analysis = detection_filter.update(
                            analysis,
                            timestamp_s=source_frame.timestamp_s,
                        )
                step = self.session.process(
                    analysis,
                    frame=image,
                    now_s=source_frame.timestamp_s,
                )
                if (
                    self._live_detection_filter_enabled
                    and not bool(getattr(step, "in_game", False))
                    and isinstance(self._live_detection_filter, LiveDetectionFilter)
                ):
                    # Do not carry a troop or enemy confirmation from a
                    # finished match into the next one.
                    self._live_detection_filter.reset()
                action: Any | None = None
                action_json: dict[str, object] | None = None
                if not bool(getattr(step, "in_game", False)) or not bool(
                    getattr(step, "should_emit", False)
                ):
                    self.actor.reset()
                    result = "not-in-game"
                else:
                    emitted += 1
                    if not self._live_hand_state_ready():
                        self.actor.reset()
                        action_json = {"kind": "wait"}
                        waits += 1
                        result = "hand-not-stable"
                    else:
                        observation = self.observation_builder(step)
                        if observation is None:
                            self.actor.reset()
                            result = "observation-not-ready"
                        else:
                            action = self.actor.decide(observation)
                            action_json = action_to_dict(action)
                            if action_json["kind"] == "wait":
                                waits += 1
                                result = "wait"
                            else:
                                proposed_plays += 1
                                enough_time = (
                                    self._last_play_timestamp_s is None
                                    or source_frame.timestamp_s - self._last_play_timestamp_s
                                    >= self.min_action_interval_s
                                )
                                if not enough_time:
                                    result = "cooldown"
                                elif not self.execute:
                                    result = "dry-run"
                                else:
                                    self.dispatch_fn(
                                        self.phone,
                                        action,
                                        step.game_state,
                                        calibration=self.calibration,
                                        observation=observation,
                                    )
                                    self._notify_live_play(
                                        action,
                                        step=step,
                                        arena_px=getattr(analysis, "arena_px", None),
                                        timestamp_s=source_frame.timestamp_s,
                                    )
                                    self._last_play_timestamp_s = source_frame.timestamp_s
                                    dispatched_plays += 1
                                    result = "dispatched"
                                    if self.post_action_delay_s:
                                        self.sleep_fn(self.post_action_delay_s)

                record = self._record(
                    source_frame,
                    step,
                    action=action_json,
                    result=result,
                )
                if on_record is not None:
                    on_record(record)
                elapsed = self.monotonic_fn() - started
                if self.poll_interval_s:
                    self.sleep_fn(max(0.0, self.poll_interval_s - elapsed))
        finally:
            self.frame_source.close()
        return RunSummary(
            frames=frames,
            emitted_frames=emitted,
            waits=waits,
            proposed_plays=proposed_plays,
            dispatched_plays=dispatched_plays,
        )

    @staticmethod
    def _record(
        source_frame: SourceFrame,
        step: Any,
        *,
        action: dict[str, object] | None,
        result: str,
    ) -> DecisionRecord:
        analysis = getattr(step, "analysis", None)
        state = getattr(step, "game_state", None)
        hud = getattr(state, "hud", None)
        hand = getattr(hud, "hand_cards", ())
        if not isinstance(hand, (list, tuple)):
            hand = ()
        safe_hand = tuple(str(card) for card in hand)

        detections = getattr(analysis, "matches", None)
        detection_count = len(detections) if isinstance(detections, (list, tuple)) else None
        return DecisionRecord(
            frame_index=source_frame.frame_index,
            timestamp_s=source_frame.timestamp_s,
            in_game=bool(getattr(step, "in_game", False)),
            emitted=bool(getattr(step, "should_emit", False)),
            action=action,
            result=result,
            hand_cards=safe_hand,
            elixir=_finite_float(getattr(hud, "elixir_self", None)),
            time_left_s=_finite_float(getattr(hud, "time_left_s", None)),
            detection_count=detection_count,
            visual_state=_summarize_extracted_visual_state(step),
        )


def _default_template_root(repository_root: Path) -> Path:
    candidates = (
        repository_root / "assets/templates/cr-api-assets/cards-gold",
        repository_root / "capture/templates/cr-api-assets/cards-gold",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value if value.is_absolute() else root / value


def _runtime_executable(value: str) -> str:
    """Prefer a helper embedded in a frozen one-file build when available."""

    if value not in {"adb", "ffmpeg"}:
        return value
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        bundled = Path(bundle_root) / value
        if bundled.is_file():
            return str(bundled)
    return value


def _validate_live_setup(
    controller: Any,
    calibration: Any,
    *,
    template_root: Path,
    action_frame_provider: Callable[[], Any] | None = None,
) -> tuple[Any, Any]:
    """Validate device identity/calibration before constructing the action sink."""

    info = controller.device_info()
    if not bool(getattr(info, "connected", False)):
        raise PrototypeControllerError("selected ADB device is not connected")
    if calibration.device_serial_hash != controller.serial_hash:
        raise PrototypeControllerError(
            "calibration serial hash does not match the explicitly selected ADB device"
        )
    if (
        calibration.screen_width_px != info.screen_width_px
        or calibration.screen_height_px != info.screen_height_px
    ):
        raise PrototypeControllerError(
            "calibration dimensions do not match the connected device: "
            f"calibration={calibration.screen_width_px}x{calibration.screen_height_px}, "
            f"device={info.screen_width_px}x{info.screen_height_px}"
        )
    if not template_root.is_dir():
        raise PrototypeControllerError(f"card template directory does not exist: {template_root}")
    try:
        from .automation import CARD_ASSET_NAMES, FIXED_HOG_CYCLE_DECK, AutonomousPhone, CardVision, UiProfile
    except ImportError:
        from simulator.physical_lab.automation import (
            CARD_ASSET_NAMES,
            FIXED_HOG_CYCLE_DECK,
            AutonomousPhone,
            CardVision,
            UiProfile,
        )
    missing = [
        card
        for card in FIXED_HOG_CYCLE_DECK
        if not (template_root / f"{CARD_ASSET_NAMES.get(card, card)}.png").is_file()
    ]
    if missing:
        raise PrototypeControllerError(
            f"card templates are missing from {template_root}: {', '.join(missing)}"
        )
    profile = UiProfile.for_device("LIVE", info)
    phone = AutonomousPhone(
        controller,
        profile,
        CardVision(template_root),
        device_model=info.model,
        action_frame_provider=action_frame_provider,
    )
    return phone, info


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the public Clash Royale prototype actor on ADB screenshots or video"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--serial", help="one explicitly selected ADB device serial")
    source.add_argument("--video", type=Path, help="recorded video (dry-run only)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="recurrent prototype checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device for actor inference (default: cpu)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="reviewed CalibrationArtifact JSON required for live execution",
    )
    parser.add_argument(
        "--template-root",
        type=Path,
        default=_default_template_root(_REPOSITORY_ROOT),
        help="card-art templates used for final live card identity verification",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send real card/placement taps; without this flag the run is a dry-run",
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="explicitly confirm that --execute may control the selected phone",
    )
    parser.add_argument(
        "--adb-transport",
        choices=("stream", "screenshot"),
        default="stream",
        help=(
            "live ADB frame transport; stream uses persistent H.264 with latest-frame "
            "processing (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--ffmpeg-executable",
        default="ffmpeg",
        help="ffmpeg executable used to decode the live H.264 stream",
    )
    parser.add_argument(
        "--adb-executable",
        default="adb",
        help="adb executable used for device control and screen capture",
    )
    parser.add_argument(
        "--stream-bit-rate",
        type=int,
        default=8_000_000,
        help="Android screenrecord H.264 bit rate (default: %(default)s)",
    )
    parser.add_argument(
        "--stream-restart-delay-s",
        type=float,
        default=0.25,
        help="delay before restarting an ended screenrecord segment (default: %(default)s)",
    )
    parser.add_argument(
        "--stream-action-frame-max-age-s",
        type=float,
        default=1.0,
        help="maximum age of a stream frame accepted for card verification (default: %(default)s)",
    )
    parser.add_argument(
        "--stream-action-frame-timeout-s",
        type=float,
        default=2.0,
        help="wait limit for a recent stream frame before failing a play (default: %(default)s)",
    )
    parser.add_argument("--max-frames", type=int, help="stop after this many source frames")
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.25,
        help="minimum live polling interval (default: %(default)s)",
    )
    parser.add_argument(
        "--min-action-interval-s",
        type=float,
        default=0.75,
        help="minimum time between dispatched plays (default: %(default)s)",
    )
    parser.add_argument(
        "--post-action-delay-s",
        type=float,
        default=0.35,
        help="settling delay after a dispatched play (default: %(default)s)",
    )
    parser.add_argument(
        "--connection-check-interval-s",
        type=float,
        default=5.0,
        help=(
            "reuse a successful screenshot as the ADB connectivity proof for this "
            "many seconds before a tap (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--yolo-image-size",
        type=int,
        default=896,
        help=(
            "YOLO inference edge size; use 640 for the faster experimental live "
            "profile (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="recorded-video frame stride (default: %(default)s)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="pass native frames to cr_bot instead of normalizing to 1080x2400",
    )
    parser.add_argument(
        "--yolo-detections",
        action="store_true",
        help="use YOLO tower-health detections in the existing extractor",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        help="write one extracted-state/action record per processed frame",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.confirm_live and not args.execute:
        parser.error("--confirm-live is only valid together with --execute")
    if args.execute and args.video is not None:
        parser.error("--execute requires --serial; recorded video is dry-run only")
    if args.execute and not args.confirm_live:
        parser.error("real taps require both --execute and --confirm-live")
    if args.execute and args.calibration is None:
        parser.error("--calibration is required for live execution")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    for option in (
        "interval_s",
        "min_action_interval_s",
        "post_action_delay_s",
        "connection_check_interval_s",
        "stream_restart_delay_s",
        "stream_action_frame_max_age_s",
        "stream_action_frame_timeout_s",
    ):
        if not math.isfinite(float(getattr(args, option))) or float(getattr(args, option)) < 0:
            parser.error(f"--{option.replace('_', '-')} must be finite and non-negative")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if args.yolo_image_size <= 0:
        parser.error("--yolo-image-size must be positive")
    if args.stream_bit_rate <= 0:
        parser.error("--stream-bit-rate must be positive")

    checkpoint = _resolve(_MODULE_ROOT, args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"prototype checkpoint does not exist: {checkpoint}")
    try:
        _bootstrap_extractor_runtime()
    except PrototypeControllerError as error:
        raise SystemExit(str(error)) from error

    controller = None
    phone = None
    calibration = None
    if args.serial is not None:
        try:
            try:
                from .calibration import CalibrationArtifact
            except ImportError:
                from simulator.physical_lab.calibration import CalibrationArtifact
            adb_executable = _runtime_executable(args.adb_executable)
            ffmpeg_executable = _runtime_executable(args.ffmpeg_executable)
            controller = CachedAdbPhoneController(
                args.serial,
                device_label="LIVE",
                connection_check_interval_s=args.connection_check_interval_s,
                adb_executable=adb_executable,
            )
            if args.adb_transport == "stream":
                source = AdbH264FrameSource(
                    controller,
                    ffmpeg_executable=ffmpeg_executable,
                    bit_rate=args.stream_bit_rate,
                    restart_delay_s=args.stream_restart_delay_s,
                    action_frame_max_age_s=args.stream_action_frame_max_age_s,
                    action_frame_wait_timeout_s=args.stream_action_frame_timeout_s,
                )
            else:
                source = AdbScreenshotSource(controller)
            if args.execute:
                calibration = CalibrationArtifact.load(
                    _resolve(_MODULE_ROOT, args.calibration)
                )
                phone, _info = _validate_live_setup(
                    controller,
                    calibration,
                    template_root=_resolve(_REPOSITORY_ROOT, args.template_root),
                    action_frame_provider=(
                        source.frame_for_action
                        if isinstance(source, AdbH264FrameSource)
                        else None
                    ),
                )
                print(
                    "LIVE CONTROL ENABLED for the explicitly selected device; "
                    f"card identity and calibration gates are active ({args.adb_transport} transport); "
                    "own-play seeding and 2-frame enemy confirmation are active."
                )
        except Exception as error:
            if isinstance(error, PrototypeControllerError):
                raise SystemExit(str(error)) from error
            raise SystemExit(f"could not initialize ADB source: {error}") from error
    else:
        source = VideoFrameSource(args.video, frame_stride=args.frame_stride)

    try:
        from cr_bot.vision.yolo_runtime import build_detector
        detector = build_detector()
        configure_detector_inference_size(detector, args.yolo_image_size)
        actor = PrototypeActor(checkpoint, device=args.device)
        runner = LivePrototypeRunner(
            source,
            detector,
            actor,
            execute=args.execute,
            phone=phone,
            calibration=calibration,
            normalize=not args.no_normalize,
            yolo_tower_hp_detections=args.yolo_detections,
            poll_interval_s=args.interval_s if args.serial is not None else 0.0,
            min_action_interval_s=args.min_action_interval_s,
            post_action_delay_s=args.post_action_delay_s,
        )
        output_handle: TextIO | None = None
        if args.jsonl_out is not None:
            output_path = _resolve(_MODULE_ROOT, args.jsonl_out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w", encoding="utf-8")

        def record(record: DecisionRecord) -> None:
            if output_handle is not None:
                output_handle.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")
                output_handle.flush()
            if record.emitted:
                print(format_decision_record(record), flush=True)

        try:
            summary = runner.run(max_frames=args.max_frames, on_record=record)
        finally:
            if output_handle is not None:
                output_handle.close()
        print(json.dumps({"summary": summary.as_dict()}, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        print("stopped by operator")
        return 130
    except Exception as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "AdbH264FrameSource",
    "AdbScreenshotSource",
    "CachedAdbPhoneController",
    "DEFAULT_CHECKPOINT",
    "DecisionRecord",
    "LiveDetectionFilter",
    "LivePrototypeRunner",
    "LiveHandStateFilter",
    "LIVE_ENEMY_CONFIRMATION_FRAMES",
    "LIVE_IGNORED_DETECTOR_LABELS",
    "LIVE_OWN_DECK_CARD_NAMES",
    "LIVE_OWN_DETECTOR_ALIASES",
    "LIVE_OWN_TROOP_CARD_NAMES",
    "PrototypeActor",
    "PrototypeControllerError",
    "RunSummary",
    "SourceFrame",
    "VideoFrameSource",
    "action_to_dict",
    "build_arg_parser",
    "decode_adb_screenshot",
    "configure_detector_inference_size",
    "format_decision_record",
    "main",
    "observation_to_model_inputs",
    "policy_action_from_batch",
]
