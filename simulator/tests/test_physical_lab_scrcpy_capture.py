from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simulator.physical_lab.devices import (
    AdbPhoneController,
    DeviceDisconnectedError,
    Frame,
    PhysicalLabError,
    SCRCPY_LEGAL_MAX_TIME_LIMIT_S,
    ScrcpyScreenCapture,
)
import simulator.physical_lab.devices as devices


class FakeScrcpyProcess:
    def __init__(self, *, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.terminate_calls = 0
        self.kill_calls = 0

    @property
    def returncode(self) -> int | None:
        return self.exit_code

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.exit_code = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.exit_code is None:
            self.exit_code = 0
        return self.exit_code


def _controller() -> AdbPhoneController:
    return AdbPhoneController("raw-device-serial", device_label="A", monotonic_clock=lambda: 100)


def _frame() -> Frame:
    return Frame(
        source_device="A",
        frame_index=0,
        workstation_monotonic_us=101,
        payload=b"frame",
    )


def test_scrcpy_capture_builds_explicit_command_and_seals_fresh_mp4(tmp_path: Path) -> None:
    output = tmp_path / "capture.mp4"
    commands: list[list[str]] = []
    process = FakeScrcpyProcess()

    def popen(command: list[str], **_kwargs: object) -> FakeScrcpyProcess:
        commands.append(command)
        Path(command[command.index("--record") + 1]).write_bytes(b"scrcpy-mp4")
        return process

    capture = ScrcpyScreenCapture(_controller(), output, popen_factory=popen, monotonic_clock=lambda: 200)
    handle = capture.start()
    capture.record_frame(_frame())
    manifest = capture.stop()

    assert commands == [[
        "scrcpy",
        "--serial",
        "raw-device-serial",
        "--no-window",
        "--no-audio",
        "--record",
        str(output),
    ]]
    assert process.terminate_calls == 1
    assert handle.transport == "scrcpy"
    assert manifest.status == "complete"
    assert manifest.stream_verified is True
    assert manifest.media_sha256 == "sha256:" + hashlib.sha256(b"scrcpy-mp4").hexdigest()
    assert manifest.media_path == str(output)
    assert "raw-device-serial" not in handle.to_dict()
    assert "raw-device-serial" not in manifest.to_dict()


@pytest.mark.parametrize("time_limit_s", [None, SCRCPY_LEGAL_MAX_TIME_LIMIT_S, 600])
def test_scrcpy_time_limit_is_optional_or_long_enough(
    tmp_path: Path,
    time_limit_s: int | None,
) -> None:
    command: list[str] = []
    process = FakeScrcpyProcess()

    def popen(argv: list[str], **_kwargs: object) -> FakeScrcpyProcess:
        command.extend(argv)
        Path(argv[argv.index("--record") + 1]).write_bytes(b"mp4")
        return process

    capture = ScrcpyScreenCapture(
        _controller(),
        tmp_path / f"capture-{time_limit_s}.mp4",
        time_limit_s=time_limit_s,
        popen_factory=popen,
    )
    capture.start()
    manifest = capture.stop()

    assert manifest.stream_verified is True
    if time_limit_s is None:
        assert "--time-limit" not in command
    else:
        assert command[-2:] == ["--time-limit", str(time_limit_s)]


def test_scrcpy_rejects_a_limit_shorter_than_the_legal_match(tmp_path: Path) -> None:
    with pytest.raises(PhysicalLabError, match="at least"):
        ScrcpyScreenCapture(
            _controller(),
            tmp_path / "capture.mp4",
            time_limit_s=SCRCPY_LEGAL_MAX_TIME_LIMIT_S - 1,
        )


def test_scrcpy_fails_closed_on_nonzero_process_exit(tmp_path: Path) -> None:
    process = FakeScrcpyProcess(exit_code=17)

    def popen(argv: list[str], **_kwargs: object) -> FakeScrcpyProcess:
        Path(argv[argv.index("--record") + 1]).write_bytes(b"partial-mp4")
        return process

    capture = ScrcpyScreenCapture(_controller(), tmp_path / "capture.mp4", popen_factory=popen)
    capture.start()
    manifest = capture.stop()

    assert manifest.status == "failed"
    assert manifest.stream_verified is False
    assert manifest.media_sha256 is None
    assert "scrcpy process exited with code 17" in manifest.rejection_reasons


def test_scrcpy_fails_closed_when_mp4_is_absent(tmp_path: Path) -> None:
    process = FakeScrcpyProcess()
    capture = ScrcpyScreenCapture(
        _controller(),
        tmp_path / "missing.mp4",
        popen_factory=lambda _argv, **_kwargs: process,
    )
    capture.start()
    manifest = capture.stop()

    assert manifest.status == "failed"
    assert manifest.stream_verified is False
    assert manifest.media_sha256 is None
    assert any("absent, empty, or unchanged" in reason for reason in manifest.rejection_reasons)


def test_scrcpy_does_not_seal_stale_media(tmp_path: Path) -> None:
    output = tmp_path / "stale.mp4"
    output.write_bytes(b"old-capture")
    process = FakeScrcpyProcess()

    capture = ScrcpyScreenCapture(
        _controller(),
        output,
        popen_factory=lambda _argv, **_kwargs: process,
    )
    capture.start()
    manifest = capture.stop()

    assert manifest.status == "failed"
    assert manifest.media_sha256 is None


def test_scrcpy_popen_failure_is_reported_without_device_access(tmp_path: Path) -> None:
    def popen(_argv: list[str], **_kwargs: object) -> FakeScrcpyProcess:
        raise OSError("scrcpy is unavailable")

    capture = ScrcpyScreenCapture(_controller(), tmp_path / "capture.mp4", popen_factory=popen)
    with pytest.raises(DeviceDisconnectedError, match="cannot start scrcpy"):
        capture.start()


def test_scrcpy_is_exported_from_the_devices_module() -> None:
    assert "ScrcpyScreenCapture" in devices.__all__


def test_adb_long_press_uses_a_serial_scoped_hold_gesture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AdbPhoneController("raw-device-serial", device_label="A")
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(controller, "_require_connected", lambda: None)
    monkeypatch.setattr(
        controller,
        "_run",
        lambda *arguments, **_kwargs: commands.append(arguments) or "",
    )

    receipt = controller.long_press_screen(540, 840, duration_ms=900)

    assert commands == [
        ("shell", "input", "swipe", "540", "840", "540", "840", "900")
    ]
    assert receipt.accepted is True
    assert receipt.x_px == 540
    assert receipt.y_px == 840
