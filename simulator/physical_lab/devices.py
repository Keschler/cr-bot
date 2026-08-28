"""Phone and capture transport interfaces for the physical-fidelity lab.

The experiment layer depends on the small protocols in this module.  The
offline fake implementation is deterministic and is the default for software
verification.  The ADB implementation is lazy and reports a disconnected
device as a normal, auditable failure; importing the package never probes or
connects to a phone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, Iterable, Protocol

from .calibration import CalibrationArtifact, CalibrationError
from .schema import PhysicalLabError, canonical_hash


class DeviceDisconnectedError(PhysicalLabError):
    """Raised when a device operation cannot run because the phone is absent."""


class DeviceCommandError(PhysicalLabError):
    """Raised when an ADB command returns a non-zero result."""


def monotonic_time_us() -> int:
    return time.monotonic_ns() // 1_000


def sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Non-secret device metadata captured in a run manifest."""

    device_id: str
    serial_hash: str
    model: str | None = None
    os_version: str | None = None
    screen_width_px: int | None = None
    screen_height_px: int | None = None
    transport: str = "unknown"
    connected: bool = False
    observed_at_monotonic_us: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.device_id, str) or not self.device_id.strip():
            raise PhysicalLabError("device_id is required")
        if not isinstance(self.serial_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.serial_hash):
            raise PhysicalLabError("serial_hash must be a sha256 digest")
        for field_name in ("screen_width_px", "screen_height_px", "observed_at_monotonic_us"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise PhysicalLabError(f"{field_name} must be a non-negative integer when present")
        if self.screen_width_px == 0 or self.screen_height_px == 0:
            raise PhysicalLabError("screen dimensions must be positive when present")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "device_id": self.device_id,
            "serial_hash": self.serial_hash,
            "transport": self.transport,
            "connected": self.connected,
        }
        for key in ("model", "os_version", "screen_width_px", "screen_height_px", "observed_at_monotonic_us"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame boundary with all three time/index coordinates preserved."""

    source_device: str
    frame_index: int
    workstation_monotonic_us: int
    payload: bytes | None = None
    presentation_time_us: int | None = None
    device_time_us: int | None = None
    capture_uncertainty_us: int = 0
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_device, str) or not self.source_device.strip():
            raise PhysicalLabError("frame.source_device is required")
        for field_name in ("frame_index", "workstation_monotonic_us", "capture_uncertainty_us"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise PhysicalLabError(f"frame.{field_name} must be a non-negative integer")
        for field_name in ("presentation_time_us", "device_time_us"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise PhysicalLabError(f"frame.{field_name} must be a non-negative integer when present")
        if self.payload is not None and not isinstance(self.payload, bytes):
            raise PhysicalLabError("frame.payload must be bytes when present")

    @property
    def payload_hash(self) -> str | None:
        return None if self.payload is None else sha256_bytes(self.payload)

    def to_dict(self, *, include_payload: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "source_device": self.source_device,
            "frame_index": self.frame_index,
            "workstation_monotonic_us": self.workstation_monotonic_us,
            "capture_uncertainty_us": self.capture_uncertainty_us,
        }
        if self.presentation_time_us is not None:
            result["presentation_time_us"] = self.presentation_time_us
        if self.device_time_us is not None:
            result["device_time_us"] = self.device_time_us
        if self.source_path is not None:
            result["source_path"] = self.source_path
        if self.payload_hash is not None:
            result["payload_sha256"] = self.payload_hash
        if include_payload and self.payload is not None:
            # This is intended only for small fake fixtures, never for video.
            import base64

            result["payload_b64"] = base64.b64encode(self.payload).decode("ascii")
        return result


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    """Result of one low-level screen action."""

    receipt_id: str
    device_id: str
    accepted: bool
    requested_at_monotonic_us: int
    completed_at_monotonic_us: int
    x_px: int | None = None
    y_px: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.receipt_id or not self.device_id:
            raise PhysicalLabError("action receipt IDs are required")
        if type(self.requested_at_monotonic_us) is not int or type(self.completed_at_monotonic_us) is not int:
            raise PhysicalLabError("action receipt times must be integers")
        if self.completed_at_monotonic_us < self.requested_at_monotonic_us:
            raise PhysicalLabError("action receipt completed time precedes request")
        for name in ("x_px", "y_px"):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise PhysicalLabError(f"{name} must be an integer when present")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "receipt_id": self.receipt_id,
            "device_id": self.device_id,
            "accepted": self.accepted,
            "requested_at_monotonic_us": self.requested_at_monotonic_us,
            "completed_at_monotonic_us": self.completed_at_monotonic_us,
        }
        for key in ("x_px", "y_px", "reason"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True, slots=True)
class CaptureHandle:
    capture_id: str
    source_device: str
    started_at_monotonic_us: int
    transport: str

    def to_dict(self) -> dict[str, object]:
        return {
            "capture_id": self.capture_id,
            "source_device": self.source_device,
            "started_at_monotonic_us": self.started_at_monotonic_us,
            "transport": self.transport,
        }


@dataclass(frozen=True, slots=True)
class CaptureManifest:
    """Sealed metadata for one device stream."""

    capture_id: str
    source_device: str
    started_at_monotonic_us: int
    stopped_at_monotonic_us: int
    frames: tuple[Frame, ...] = ()
    media_path: str | None = None
    media_sha256: str | None = None
    stream_verified: bool = False
    status: str = "complete"
    synchronization_uncertainty_us: int | None = None
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.capture_id or not self.source_device:
            raise PhysicalLabError("capture IDs are required")
        if self.stopped_at_monotonic_us < self.started_at_monotonic_us:
            raise PhysicalLabError("capture stop time precedes start time")
        if any(not isinstance(frame, Frame) for frame in self.frames):
            raise PhysicalLabError("capture frames must be Frame records")
        frame_indices = [frame.frame_index for frame in self.frames]
        if frame_indices != sorted(set(frame_indices)):
            raise PhysicalLabError("capture frame indices must be sorted and unique")
        if self.synchronization_uncertainty_us is not None and self.synchronization_uncertainty_us < 0:
            raise PhysicalLabError("synchronization uncertainty must be non-negative")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "capture_id": self.capture_id,
            "source_device": self.source_device,
            "started_at_monotonic_us": self.started_at_monotonic_us,
            "stopped_at_monotonic_us": self.stopped_at_monotonic_us,
            "frames": [frame.to_dict() for frame in self.frames],
            "frame_count": len(self.frames),
            "stream_verified": self.stream_verified,
            "status": self.status,
            "rejection_reasons": list(self.rejection_reasons),
        }
        for key in ("media_path", "media_sha256", "synchronization_uncertainty_us"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


class PhoneController(Protocol):
    """Small low-level interface used by the experiment runner."""

    def screenshot(self) -> Frame: ...

    def tap_screen(self, x_px: int, y_px: int) -> ActionReceipt: ...

    def long_press_screen(
        self,
        x_px: int,
        y_px: int,
        *,
        duration_ms: int,
    ) -> ActionReceipt: ...

    def press_back(self) -> None: ...

    def device_info(self) -> DeviceInfo: ...


class ScreenCapture(Protocol):
    """Capture transport boundary; the experiment does not know its codec."""

    def start(self) -> CaptureHandle: ...

    def stop(self) -> CaptureManifest: ...


SCRCPY_LEGAL_MAX_TIME_LIMIT_S = 330


class _CaptureProcess(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class LogicalPhone:
    """Expose logical card/cell actions while hiding calibrated pixels."""

    def __init__(self, controller: PhoneController, calibration: CalibrationArtifact) -> None:
        self.controller = controller
        self.calibration = calibration
        self.selected_slot: int | None = None

    def device_info(self) -> DeviceInfo:
        return self.controller.device_info()

    def screenshot(self) -> Frame:
        return self.controller.screenshot()

    def select_card(self, *, slot: int) -> ActionReceipt:
        try:
            x_px, y_px = self.calibration.slot_to_pixel(slot)
        except CalibrationError:
            raise
        receipt = self.controller.tap_screen(x_px, y_px)
        if receipt.accepted:
            self.selected_slot = slot
        return receipt

    def place_card(self, card_id: str, *, arena_cell: tuple[int, int]) -> ActionReceipt:
        """Place a named card at a logical cell; ``card_id`` is audit context."""

        if not isinstance(card_id, str) or not card_id.strip():
            raise PhysicalLabError("card_id is required for place_card")
        if self.selected_slot is None:
            raise PhysicalLabError("select_card must succeed before place_card")
        x_px, y_px = self.calibration.cell_to_pixel(arena_cell)
        return self.controller.tap_screen(x_px, y_px)

    def long_press(self, point: tuple[int, int], *, duration_ms: int) -> ActionReceipt:
        """Hold a calibrated UI point without exposing ADB coordinates to callers."""

        if not isinstance(point, tuple) or len(point) != 2:
            raise PhysicalLabError("long-press point must be an (x, y) tuple")
        x_px, y_px = point
        return self.controller.long_press_screen(x_px, y_px, duration_ms=duration_ms)

    def press_back(self) -> None:
        self.controller.press_back()


class FakePhoneController:
    """Deterministic phone substitute used while physical devices are absent."""

    def __init__(
        self,
        device_id: str,
        *,
        screen_width_px: int = 1080,
        screen_height_px: int = 2400,
        connected: bool = True,
        serial_label: str | None = None,
        monotonic_clock: Callable[[], int] = monotonic_time_us,
    ) -> None:
        if not device_id:
            raise PhysicalLabError("fake device_id is required")
        self.device_id = device_id
        self.connected = connected
        self._clock = monotonic_clock
        self._serial_hash = canonical_hash({"offline_device": serial_label or device_id})
        self._screen_width_px = screen_width_px
        self._screen_height_px = screen_height_px
        self._frame_index = 0
        self.taps: list[tuple[int, int]] = []
        self.long_presses: list[tuple[int, int, int]] = []
        self.back_presses = 0

    def _require_connected(self) -> None:
        if not self.connected:
            raise DeviceDisconnectedError(f"device {self.device_id} is not connected")

    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            device_id=self.device_id,
            serial_hash=self._serial_hash,
            model="offline-fake-phone",
            os_version="offline",
            screen_width_px=self._screen_width_px,
            screen_height_px=self._screen_height_px,
            transport="fake",
            connected=self.connected,
            observed_at_monotonic_us=self._clock(),
        )

    def screenshot(self) -> Frame:
        self._require_connected()
        now = self._clock()
        frame = Frame(
            source_device=self.device_id,
            frame_index=self._frame_index,
            workstation_monotonic_us=now,
            payload=f"fake-frame:{self.device_id}:{self._frame_index}".encode("ascii"),
            capture_uncertainty_us=100,
        )
        self._frame_index += 1
        return frame

    def tap_screen(self, x_px: int, y_px: int) -> ActionReceipt:
        self._require_connected()
        if type(x_px) is not int or type(y_px) is not int:
            raise PhysicalLabError("fake tap coordinates must be integers")
        requested = self._clock()
        self.taps.append((x_px, y_px))
        completed = self._clock()
        return ActionReceipt(
            receipt_id=f"{self.device_id}-tap-{len(self.taps):04d}",
            device_id=self.device_id,
            accepted=True,
            requested_at_monotonic_us=requested,
            completed_at_monotonic_us=max(requested, completed),
            x_px=x_px,
            y_px=y_px,
        )

    def long_press_screen(
        self,
        x_px: int,
        y_px: int,
        *,
        duration_ms: int,
    ) -> ActionReceipt:
        self._require_connected()
        if type(x_px) is not int or type(y_px) is not int:
            raise PhysicalLabError("fake long-press coordinates must be integers")
        if type(duration_ms) is not int or duration_ms <= 0:
            raise PhysicalLabError("fake long-press duration must be a positive integer")
        requested = self._clock()
        self.long_presses.append((x_px, y_px, duration_ms))
        completed = self._clock()
        return ActionReceipt(
            receipt_id=f"{self.device_id}-long-press-{len(self.long_presses):04d}",
            device_id=self.device_id,
            accepted=True,
            requested_at_monotonic_us=requested,
            completed_at_monotonic_us=max(requested, completed),
            x_px=x_px,
            y_px=y_px,
        )

    def press_back(self) -> None:
        self._require_connected()
        self.back_presses += 1


class FakeScreenCapture:
    """Capture a deterministic frame list without creating media files."""

    def __init__(
        self,
        source_device: str,
        *,
        frame_source: Callable[[], Frame] | None = None,
        monotonic_clock: Callable[[], int] = monotonic_time_us,
    ) -> None:
        self.source_device = source_device
        self._frame_source = frame_source
        self._clock = monotonic_clock
        self._handle: CaptureHandle | None = None
        self._frames: list[Frame] = []

    def start(self) -> CaptureHandle:
        if self._handle is not None:
            raise PhysicalLabError(f"capture for {self.source_device} is already running")
        now = self._clock()
        self._handle = CaptureHandle(
            capture_id=f"{self.source_device}-capture-{now}",
            source_device=self.source_device,
            started_at_monotonic_us=now,
            transport="fake",
        )
        self._frames = []
        return self._handle

    def record_frame(self, frame: Frame) -> None:
        if self._handle is None:
            raise PhysicalLabError("capture must be started before recording frames")
        if frame.source_device != self.source_device:
            raise PhysicalLabError("frame source does not match capture source")
        self._frames.append(frame)

    def stop(self) -> CaptureManifest:
        if self._handle is None:
            raise PhysicalLabError(f"capture for {self.source_device} is not running")
        if self._frame_source is not None and not self._frames:
            self._frames.append(self._frame_source())
        stopped = max(self._clock(), self._handle.started_at_monotonic_us)
        manifest = CaptureManifest(
            capture_id=self._handle.capture_id,
            source_device=self.source_device,
            started_at_monotonic_us=self._handle.started_at_monotonic_us,
            stopped_at_monotonic_us=stopped,
            frames=tuple(sorted(self._frames, key=lambda frame: frame.frame_index)),
            stream_verified=True,
            status="complete",
        )
        self._handle = None
        return manifest


class FrameBufferCapture:
    """Transport-neutral frame recorder for adapters with external capture.

    It is useful for ADB screenshot smoke tests, but it is deliberately not
    marked as a verified video stream.  A connected physical run must replace
    it with a screenrecord/external-camera implementation before it can enter
    an evidence corpus.
    """

    def __init__(
        self,
        source_device: str,
        *,
        transport: str = "frame_buffer",
        stream_verified: bool = False,
        monotonic_clock: Callable[[], int] = monotonic_time_us,
    ) -> None:
        self.source_device = source_device
        self.transport = transport
        self.stream_verified = stream_verified
        self._clock = monotonic_clock
        self._handle: CaptureHandle | None = None
        self._frames: list[Frame] = []

    def start(self) -> CaptureHandle:
        if self._handle is not None:
            raise PhysicalLabError(f"capture for {self.source_device} is already running")
        now = self._clock()
        self._handle = CaptureHandle(
            capture_id=f"{self.source_device}-capture-{now}",
            source_device=self.source_device,
            started_at_monotonic_us=now,
            transport=self.transport,
        )
        self._frames = []
        return self._handle

    def record_frame(self, frame: Frame) -> None:
        if self._handle is None:
            raise PhysicalLabError("capture must be started before recording frames")
        if frame.source_device != self.source_device:
            raise PhysicalLabError("frame source does not match capture source")
        self._frames.append(frame)

    def stop(self) -> CaptureManifest:
        if self._handle is None:
            raise PhysicalLabError(f"capture for {self.source_device} is not running")
        stopped = max(self._clock(), self._handle.started_at_monotonic_us)
        manifest = CaptureManifest(
            capture_id=self._handle.capture_id,
            source_device=self.source_device,
            started_at_monotonic_us=self._handle.started_at_monotonic_us,
            stopped_at_monotonic_us=stopped,
            frames=tuple(sorted(self._frames, key=lambda frame: frame.frame_index)),
            stream_verified=self.stream_verified,
            status="complete" if self.stream_verified else "frames_only",
        )
        self._handle = None
        return manifest


class AdbScreenCapture:
    """ADB ``screenrecord`` transport with local hash sealing.

    The process is started only when :meth:`start` is called.  This keeps
    imports and experiment planning network/device-free.  The runner still
    records occasional ``screencap`` frames for synchronization and action
    provenance; the MP4 is the verified continuous stream.
    """

    def __init__(
        self,
        controller: "AdbPhoneController",
        output_path: str | Path,
        *,
        bit_rate: int = 8_000_000,
        time_limit_s: int = 180,
        monotonic_clock: Callable[[], int] = monotonic_time_us,
    ) -> None:
        if bit_rate <= 0 or time_limit_s <= 0:
            raise PhysicalLabError("screenrecord bit rate and time limit must be positive")
        self.controller = controller
        self.output_path = Path(output_path)
        self.bit_rate = bit_rate
        self.time_limit_s = time_limit_s
        self._clock = monotonic_clock
        self._handle: CaptureHandle | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._device_path: str | None = None
        self._frames: list[Frame] = []

    def start(self) -> CaptureHandle:
        if self._handle is not None or self._process is not None:
            raise PhysicalLabError(f"ADB capture for {self.controller.device_id} is already running")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        now = self._clock()
        capture_id = f"{self.controller.device_id}-capture-{now}"
        device_path = f"/sdcard/{capture_id}.mp4"
        command = [
            self.controller.adb_executable,
            "-s",
            self.controller.serial,
            "shell",
            "screenrecord",
            "--bit-rate",
            str(self.bit_rate),
            "--time-limit",
            str(self.time_limit_s),
            device_path,
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise DeviceDisconnectedError(f"cannot start ADB screenrecord: {error}") from error
        self._handle = CaptureHandle(capture_id, self.controller.device_id, now, "adb-screenrecord")
        self._process = process
        self._device_path = device_path
        self._frames = []
        return self._handle

    def record_frame(self, frame: Frame) -> None:
        if self._handle is None:
            raise PhysicalLabError("capture must be started before recording frames")
        if frame.source_device != self.controller.device_id:
            raise PhysicalLabError("frame source does not match ADB capture")
        self._frames.append(frame)

    def stop(self) -> CaptureManifest:
        if self._handle is None or self._process is None or self._device_path is None:
            raise PhysicalLabError(f"ADB capture for {self.controller.device_id} is not running")
        process = self._process
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        status = "failed"
        media_sha256: str | None = None
        try:
            self.controller._run("pull", self._device_path, str(self.output_path))
            self.controller._run("shell", "rm", "-f", self._device_path)
            if self.output_path.is_file() and self.output_path.stat().st_size > 0:
                digest = hashlib.sha256()
                with self.output_path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                media_sha256 = f"sha256:{digest.hexdigest()}"
                status = "complete"
        except (DeviceDisconnectedError, OSError):
            status = "failed"
        stopped = max(self._clock(), self._handle.started_at_monotonic_us)
        manifest = CaptureManifest(
            capture_id=self._handle.capture_id,
            source_device=self.controller.device_id,
            started_at_monotonic_us=self._handle.started_at_monotonic_us,
            stopped_at_monotonic_us=stopped,
            frames=tuple(sorted(self._frames, key=lambda frame: frame.frame_index)),
            media_path=str(self.output_path),
            media_sha256=media_sha256,
            stream_verified=status == "complete",
            status=status,
            rejection_reasons=() if status == "complete" else ("screenrecord stream was not sealed",),
        )
        self._handle = None
        self._process = None
        self._device_path = None
        return manifest


def _media_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class ScrcpyScreenCapture:
    """Long-running local MP4 capture through the installed ``scrcpy`` CLI.

    ``scrcpy`` owns the device stream and writes the recording directly to
    ``output_path``.  The process is started lazily, and the transport never
    exposes the raw serial through a capture handle or manifest.  A manifest
    is complete only when scrcpy exits cleanly and has produced a fresh,
    non-empty MP4 whose hash can be sealed.
    """

    def __init__(
        self,
        controller: "AdbPhoneController",
        output_path: str | Path,
        *,
        time_limit_s: int | None = None,
        scrcpy_executable: str = "scrcpy",
        stop_timeout_s: float = 10.0,
        popen_factory: Callable[..., _CaptureProcess] | None = None,
        monotonic_clock: Callable[[], int] = monotonic_time_us,
    ) -> None:
        if time_limit_s is not None and (
            type(time_limit_s) is not int or time_limit_s < SCRCPY_LEGAL_MAX_TIME_LIMIT_S
        ):
            raise PhysicalLabError(
                "scrcpy time_limit_s must be omitted or at least "
                f"{SCRCPY_LEGAL_MAX_TIME_LIMIT_S} seconds"
            )
        if not scrcpy_executable or any(character.isspace() for character in scrcpy_executable):
            raise PhysicalLabError("scrcpy executable must be a non-empty token")
        if stop_timeout_s <= 0:
            raise PhysicalLabError("scrcpy stop timeout must be positive")

        self.controller = controller
        self.output_path = Path(output_path)
        if self.output_path.suffix.lower() != ".mp4":
            raise PhysicalLabError("scrcpy capture output must be an .mp4 file")
        self.time_limit_s = time_limit_s
        self.scrcpy_executable = scrcpy_executable
        self.stop_timeout_s = stop_timeout_s
        self._popen_factory = popen_factory
        self._clock = monotonic_clock
        self._handle: CaptureHandle | None = None
        self._process: _CaptureProcess | None = None
        self._frames: list[Frame] = []
        self._preexisting_media_signature: tuple[int, int, int] | None = None

    def start(self) -> CaptureHandle:
        if self._handle is not None or self._process is not None:
            raise PhysicalLabError(f"scrcpy capture for {self.controller.device_id} is already running")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._preexisting_media_signature = _media_signature(self.output_path)
        now = self._clock()
        capture_id = f"{self.controller.device_id}-capture-{now}"
        command = [
            self.scrcpy_executable,
            "--serial",
            self.controller.serial,
            "--no-window",
            "--no-audio",
            "--record",
            str(self.output_path),
        ]
        if self.time_limit_s is not None:
            command.extend(("--time-limit", str(self.time_limit_s)))
        popen = self._popen_factory or subprocess.Popen
        try:
            process = popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.SubprocessError) as error:
            self._preexisting_media_signature = None
            raise DeviceDisconnectedError(f"cannot start scrcpy capture: {error}") from error
        self._handle = CaptureHandle(capture_id, self.controller.device_id, now, "scrcpy")
        self._process = process
        self._frames = []
        return self._handle

    def record_frame(self, frame: Frame) -> None:
        if self._handle is None or self._process is None:
            raise PhysicalLabError("capture must be started before recording frames")
        if frame.source_device != self.controller.device_id:
            raise PhysicalLabError("frame source does not match scrcpy capture")
        if self._process.poll() is not None:
            raise PhysicalLabError("scrcpy process exited before the frame was recorded")
        self._frames.append(frame)

    def stop(self) -> CaptureManifest:
        if self._handle is None or self._process is None:
            raise PhysicalLabError(f"scrcpy capture for {self.controller.device_id} is not running")

        handle = self._handle
        process = self._process
        frames = tuple(sorted(self._frames, key=lambda frame: frame.frame_index))
        rejection_reasons: list[str] = []
        exit_code: int | None = None
        try:
            try:
                exit_code = process.poll()
            except OSError as error:
                rejection_reasons.append(f"scrcpy process status unavailable: {error}")

            if exit_code is None and not rejection_reasons:
                try:
                    process.terminate()
                    exit_code = process.wait(timeout=self.stop_timeout_s)
                except subprocess.TimeoutExpired:
                    rejection_reasons.append("scrcpy process did not stop before the timeout")
                    try:
                        process.kill()
                        exit_code = process.wait(timeout=self.stop_timeout_s)
                    except (OSError, subprocess.TimeoutExpired) as error:
                        rejection_reasons.append(f"scrcpy process could not be killed: {error}")
                except OSError as error:
                    rejection_reasons.append(f"scrcpy process could not be stopped: {error}")

            if exit_code != 0:
                if exit_code is None:
                    rejection_reasons.append("scrcpy process did not report a clean exit")
                else:
                    rejection_reasons.append(f"scrcpy process exited with code {exit_code}")

            media_sha256: str | None = None
            media_sealed = False
            try:
                current_signature = _media_signature(self.output_path)
                if (
                    current_signature is not None
                    and current_signature[1] > 0
                    and current_signature != self._preexisting_media_signature
                ):
                    media_sha256 = _sha256_file(self.output_path)
                    media_sealed = True
                else:
                    rejection_reasons.append("scrcpy MP4 output is absent, empty, or unchanged")
            except OSError as error:
                rejection_reasons.append(f"scrcpy MP4 output could not be sealed: {error}")

            status = "complete" if exit_code == 0 and media_sealed and not rejection_reasons else "failed"
            stopped = max(self._clock(), handle.started_at_monotonic_us)
            return CaptureManifest(
                capture_id=handle.capture_id,
                source_device=handle.source_device,
                started_at_monotonic_us=handle.started_at_monotonic_us,
                stopped_at_monotonic_us=stopped,
                frames=frames,
                media_path=str(self.output_path),
                media_sha256=media_sha256 if status == "complete" else None,
                stream_verified=status == "complete",
                status=status,
                rejection_reasons=() if status == "complete" else tuple(rejection_reasons),
            )
        finally:
            self._handle = None
            self._process = None
            self._frames = []
            self._preexisting_media_signature = None


class AdbPhoneController:
    """Minimal ADB-backed controller with no import-time or constructor probe."""

    def __init__(
        self,
        serial: str,
        *,
        device_label: str | None = None,
        adb_executable: str = "adb",
        command_timeout_s: float = 10.0,
        monotonic_clock: Callable[[], int] = monotonic_time_us,
    ) -> None:
        if not serial or any(character.isspace() for character in serial):
            raise PhysicalLabError("ADB serial must be a non-empty token")
        if command_timeout_s <= 0:
            raise PhysicalLabError("command_timeout_s must be positive")
        self.serial = serial
        self.device_label = device_label or f"adb-{sha256_bytes(serial.encode('utf-8'))[7:19]}"
        self.adb_executable = adb_executable
        self.command_timeout_s = command_timeout_s
        self._clock = monotonic_clock
        self._frame_index = 0

    @property
    def device_id(self) -> str:
        return self.device_label

    @property
    def serial_hash(self) -> str:
        return sha256_bytes(self.serial.encode("utf-8"))

    def _run(self, *arguments: str, binary: bool = False) -> str | bytes:
        command = [self.adb_executable, "-s", self.serial, *arguments]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=self.command_timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DeviceDisconnectedError(f"ADB command unavailable for {self.serial}: {error}") from error
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise DeviceDisconnectedError(
                f"ADB device {self.serial} unavailable ({completed.returncode}): {detail or 'no details'}"
            )
        if binary:
            return completed.stdout
        return completed.stdout.decode("utf-8", errors="replace").strip()

    def device_info(self) -> DeviceInfo:
        observed = self._clock()
        try:
            model = str(self._run("shell", "getprop", "ro.product.model")) or None
            os_version = str(self._run("shell", "getprop", "ro.build.version.release")) or None
            size = str(self._run("shell", "wm", "size"))
            width = height = None
            if "Physical size:" in size:
                dimensions = size.split("Physical size:", 1)[1].strip().split("x")
                if len(dimensions) == 2:
                    width, height = int(dimensions[0]), int(dimensions[1])
            return DeviceInfo(
                device_id=self.device_id,
                serial_hash=self.serial_hash,
                model=model,
                os_version=os_version,
                screen_width_px=width,
                screen_height_px=height,
                transport="adb",
                connected=True,
                observed_at_monotonic_us=observed,
            )
        except (DeviceDisconnectedError, ValueError):
            return DeviceInfo(
                device_id=self.device_id,
                serial_hash=self.serial_hash,
                transport="adb",
                connected=False,
                observed_at_monotonic_us=observed,
            )

    def _require_connected(self) -> None:
        if not self.device_info().connected:
            raise DeviceDisconnectedError(f"ADB device {self.serial} is not connected")

    def screenshot(self) -> Frame:
        payload = self._run("exec-out", "screencap", "-p", binary=True)
        assert isinstance(payload, bytes)
        now = self._clock()
        frame = Frame(
            source_device=self.device_id,
            frame_index=self._frame_index,
            workstation_monotonic_us=now,
            payload=payload,
            capture_uncertainty_us=2_000,
        )
        self._frame_index += 1
        return frame

    def tap_screen(self, x_px: int, y_px: int) -> ActionReceipt:
        self._require_connected()
        requested = self._clock()
        self._run("shell", "input", "tap", str(x_px), str(y_px))
        completed = max(requested, self._clock())
        return ActionReceipt(
            receipt_id=f"{self.device_id}-tap-{requested}",
            device_id=self.device_id,
            accepted=True,
            requested_at_monotonic_us=requested,
            completed_at_monotonic_us=completed,
            x_px=x_px,
            y_px=y_px,
        )

    def long_press_screen(
        self,
        x_px: int,
        y_px: int,
        *,
        duration_ms: int,
    ) -> ActionReceipt:
        """Hold a screen point using ADB's explicit serial-scoped swipe input."""

        if type(x_px) is not int or type(y_px) is not int:
            raise PhysicalLabError("ADB long-press coordinates must be integers")
        if type(duration_ms) is not int or duration_ms <= 0:
            raise PhysicalLabError("ADB long-press duration must be a positive integer")
        self._require_connected()
        requested = self._clock()
        self._run(
            "shell",
            "input",
            "swipe",
            str(x_px),
            str(y_px),
            str(x_px),
            str(y_px),
            str(duration_ms),
        )
        completed = max(requested, self._clock())
        return ActionReceipt(
            receipt_id=f"{self.device_id}-long-press-{requested}",
            device_id=self.device_id,
            accepted=True,
            requested_at_monotonic_us=requested,
            completed_at_monotonic_us=completed,
            x_px=x_px,
            y_px=y_px,
        )

    def press_back(self) -> None:
        self._require_connected()
        self._run("shell", "input", "keyevent", "4")

    def set_keep_awake(self) -> dict[str, object]:
        """Keep this explicitly selected phone awake while it is powered.

        Android has no portable literal ``never`` timeout through ``settings``.
        The maximum timeout plus ``stay_on_while_plugged_in=3`` is the reviewed
        lab equivalent while a phone is connected over USB.  Every command is
        routed through :meth:`_run`, which always includes this controller's
        explicit ``-s <serial>`` selector.
        """

        self._require_connected()
        timeout_value = "2147483647"
        plugged_value = "3"
        self._run("shell", "settings", "put", "system", "screen_off_timeout", timeout_value)
        self._run("shell", "settings", "put", "global", "stay_on_while_plugged_in", plugged_value)
        self._run("shell", "svc", "power", "stayon", "true")
        timeout_readback = str(
            self._run("shell", "settings", "get", "system", "screen_off_timeout")
        ).strip()
        plugged_readback = str(
            self._run("shell", "settings", "get", "global", "stay_on_while_plugged_in")
        ).strip()
        try:
            plugged_mask = int(plugged_readback)
        except ValueError:
            plugged_mask = 0
        # Some Android builds normalize the requested mask to all supported
        # charging sources (15).  The lab is connected over USB, so any mask
        # containing the USB bit is sufficient and stronger values are valid.
        try:
            timeout_ms = int(timeout_readback)
        except ValueError:
            timeout_ms = 0
        # Some vendor builds clamp the maximum system timeout (the ASUS build
        # currently reports 600000 ms).  USB stay-on is the effective
        # never-sleep contract for the lab because both phones remain powered
        # over their explicit USB connection.  A numeric timeout is still
        # required so an unreadable settings surface cannot pass.
        usb_stay_on = bool(plugged_mask & 2)
        verified = timeout_ms > 0 and usb_stay_on
        if not verified:
            raise DeviceCommandError(
                f"ADB keep-awake settings were not verified for {self.serial}: "
                f"screen_off_timeout={timeout_readback!r}, "
                f"stay_on_while_plugged_in={plugged_readback!r}"
            )
        return {
            "device_id": self.device_id,
            "serial_hash": self.serial_hash,
            "screen_off_timeout_requested_ms": int(timeout_value),
            "screen_off_timeout_ms": timeout_ms,
            "stay_on_while_plugged_in": int(plugged_readback),
            "power_stayon": True,
            "effective_never_sleep_while_usb_powered": usb_stay_on,
            "timeout_clamped_by_firmware": timeout_ms != int(timeout_value),
            "verified": True,
        }


__all__ = [
    "ActionReceipt",
    "AdbPhoneController",
    "AdbScreenCapture",
    "CaptureHandle",
    "CaptureManifest",
    "DeviceCommandError",
    "DeviceDisconnectedError",
    "DeviceInfo",
    "FakePhoneController",
    "FakeScreenCapture",
    "FrameBufferCapture",
    "Frame",
    "LogicalPhone",
    "PhoneController",
    "ScreenCapture",
    "SCRCPY_LEGAL_MAX_TIME_LIMIT_S",
    "ScrcpyScreenCapture",
    "monotonic_time_us",
    "sha256_bytes",
]
