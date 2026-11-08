from __future__ import annotations

from io import BytesIO
import os
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from cr_bot.domain.frame_analysis import FrameAnalysisResult
from cr_bot.domain.game_state import Action, Detection, Match
from simulator.physical_lab.workflow.automation import AutonomousPhone, CardMatch, UiProfile
from simulator.physical_lab.capture.calibration import CalibrationArtifact
from simulator.physical_lab.capture.devices import FakePhoneController, Frame
from simulator.physical_lab.workflow.prototype_controller import (
    AdbH264FrameSource,
    AdbScreenshotSource,
    CachedAdbPhoneController,
    DecisionRecord,
    LiveDetectionFilter,
    LiveHandStateFilter,
    LivePrototypeRunner,
    LIVE_OWN_DECK_CARD_NAMES,
    LIVE_IGNORED_DETECTOR_LABELS,
    LIVE_OWN_DETECTOR_ALIASES,
    PrototypeControllerError,
    SourceFrame,
    action_to_dict,
    build_arg_parser,
    configure_detector_inference_size,
    format_decision_record,
    _filter_live_analysis,
)


def test_adb_source_retries_an_empty_screenshot_payload() -> None:
    class Controller:
        def __init__(self) -> None:
            self.frames = iter(
                (
                    SimpleNamespace(
                        payload=b"",
                        frame_index=0,
                        workstation_monotonic_us=1_000_000,
                    ),
                    SimpleNamespace(
                        payload=b"valid",
                        frame_index=1,
                        workstation_monotonic_us=1_200_000,
                    ),
                )
            )

        def screenshot(self):
            return next(self.frames)

    sleeps: list[float] = []

    def decode(frame):
        if not frame.payload:
            raise PrototypeControllerError("empty screenshot")
        return frame.payload

    source = AdbScreenshotSource(
        Controller(),
        decode=decode,
        retry_delay_s=0.1,
        sleep_fn=sleeps.append,
    )
    result = source.next_frame()

    assert result.image == b"valid"
    assert result.frame_index == 1
    assert result.timestamp_s == 1.2
    assert sleeps == [0.1]


def test_cached_adb_controller_reuses_recent_screenshot_connectivity() -> None:
    now = [10.0]
    controller = CachedAdbPhoneController(
        "serial",
        connection_check_interval_s=5.0,
        monotonic_fn=lambda: now[0],
    )
    info_calls = 0

    def device_info():
        nonlocal info_calls
        info_calls += 1
        return SimpleNamespace(connected=True)

    controller.device_info = device_info
    controller._last_successful_screenshot_s = now[0]
    controller._require_connected()
    assert info_calls == 0

    now[0] = 16.0
    controller._require_connected()
    assert info_calls == 1


def test_h264_stream_is_serial_scoped_and_provides_recent_action_frames() -> None:
    pytest.importorskip("cv2")
    width, height = 2, 1
    raw_frame = bytes((10, 20, 30, 40, 50, 60))
    commands: list[list[str]] = []
    alive_marks = 0

    class Process:
        def __init__(self, stdout) -> None:
            self.stdout = stdout
            self.terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self) -> None:
            self.terminated = True

    def popen(command, **_kwargs):
        commands.append(list(command))
        if command[0] == "adb":
            return Process(BytesIO(b"h264-input"))
        return Process(BytesIO(raw_frame))

    def mark_alive() -> None:
        nonlocal alive_marks
        alive_marks += 1

    controller = SimpleNamespace(
        adb_executable="adb",
        serial="R7AIB700D744BX7",
        device_id="LIVE",
        device_info=lambda: SimpleNamespace(
            connected=True,
            screen_width_px=width,
            screen_height_px=height,
        ),
        mark_transport_alive=mark_alive,
    )
    source = AdbH264FrameSource(
        controller,
        popen_factory=popen,
        restart_delay_s=0.05,
        action_frame_max_age_s=1.0,
        action_frame_wait_timeout_s=1.0,
    )

    try:
        source_frame = source.next_frame()
        action_frame = source.frame_for_action()
    finally:
        source.close()

    assert source_frame.image.shape == (height, width, 3)
    assert source_frame.image.reshape(-1).tolist() == list(raw_frame)
    assert action_frame.source_device == "LIVE"
    assert action_frame.frame_index == source_frame.frame_index
    assert action_frame.payload
    assert alive_marks >= 1
    assert commands[0][:3] == ["adb", "-s", "R7AIB700D744BX7"]
    assert commands[0][3] == "exec-out"
    assert "--size" not in commands[0]
    assert commands[1][0] == "ffmpeg"


@pytest.mark.parametrize("maximum", [0, -1, True, False, 720.0, "720"])
def test_h264_stream_width_rejects_nonpositive_or_inexact_integers(maximum) -> None:
    with pytest.raises(PrototypeControllerError, match="stream_max_width_px"):
        AdbH264FrameSource(SimpleNamespace(), stream_max_width_px=maximum)


@pytest.mark.parametrize(
    ("native_width", "native_height", "maximum", "encoded_size"),
    [
        (1080, 2400, 720, (720, 1600)),
        (1080, 2400, 721, (720, 1600)),
        (1080, 2400, 2000, (1080, 2400)),
        (1920, 1080, 720, (720, 404)),
        (1081, 2401, 721, (720, 1598)),
        (64, 95, 720, (64, 94)),
    ],
)
def test_h264_sized_command_preserves_aspect_and_native_bounds(
    native_width, native_height, maximum, encoded_size,
) -> None:
    commands: list[list[str]] = []

    def popen(command, **_kwargs):
        commands.append(list(command))
        return SimpleNamespace(stdout=BytesIO(), poll=lambda: 0, wait=lambda timeout: 0)

    source = AdbH264FrameSource(
        _h264_test_controller(native_width, native_height),
        stream_max_width_px=maximum, popen_factory=popen,
    )
    try:
        source._start_segment()
        width, height = encoded_size
        assert commands[0][:4] == ["adb", "-s", "synthetic-only", "exec-out"]
        assert commands[0][-3:] == ["--size", f"{width}x{height}", "-"]
        assert width <= min(native_width, maximum) and height <= native_height
        assert width % 2 == 0 and height % 2 == 0
        assert abs(height - native_height * width / native_width) < 2.0
        snapshot = source.diagnostics()
        assert (snapshot.native_width_px, snapshot.native_height_px) == (native_width, native_height)
        assert (snapshot.stream_width_px, snapshot.stream_height_px) == encoded_size
    finally:
        source.close()


def test_h264_stream_width_too_small_for_even_dimensions_fails_before_subprocess() -> None:
    source = AdbH264FrameSource(
        _h264_test_controller(), stream_max_width_px=1,
        popen_factory=lambda *_args, **_kwargs: pytest.fail("invalid size must not start subprocesses"),
    )
    try:
        with pytest.raises(PrototypeControllerError, match="positive even stream size"):
            source._start_segment()
    finally:
        source.close()


def test_h264_sized_framing_restores_native_geometry_and_action_image(monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    encoded = np.zeros((6, 4, 3), dtype=np.uint8)
    encoded[:3, :2] = (255, 0, 0)
    encoded[:3, 2:] = (0, 255, 0)
    encoded[3:, :2] = (0, 0, 255)
    encoded[3:, 2:] = (255, 255, 255)
    raw = encoded.tobytes() + (255 - encoded).tobytes()
    commands: list[list[str]] = []

    def popen(command, **_kwargs):
        commands.append(list(command))
        payload = b"synthetic-h264" if command[0] == "adb" else raw
        return SimpleNamespace(stdout=BytesIO(payload), poll=lambda: 0, wait=lambda timeout: 0)

    source = AdbH264FrameSource(
        _h264_test_controller(8, 12), stream_max_width_px=4,
        popen_factory=popen, max_consecutive_failures=1,
        monotonic_fn=lambda: 10.0,
    )
    retained: list[SourceFrame] = []
    action_images = []
    publish = source._publish

    def retain(image):
        publish(image)
        retained.append(source.latest_frame())
        action = source.frame_for_action()
        action_images.append(cv2.imdecode(np.frombuffer(action.payload, dtype=np.uint8), cv2.IMREAD_COLOR))

    monkeypatch.setattr(source, "_publish", retain)
    monkeypatch.setattr(source, "_ensure_started", lambda: None)
    try:
        source._reader_loop()
        assert commands[0][-3:] == ["--size", "4x6", "-"]
        assert [frame.frame_index for frame in retained] == [0, 1]
        assert all(frame.timestamp_s == 10.0 for frame in retained)
        assert all(frame.image.shape == (12, 8, 3) for frame in retained)
        first = retained[0].image
        assert first[0, 0].tolist() == [255, 0, 0]
        assert first[0, -1].tolist() == [0, 255, 0]
        assert first[-1, 0].tolist() == [0, 0, 255]
        assert first[-1, -1].tolist() == [255, 255, 255]
        assert all(image.shape == (12, 8, 3) for image in action_images)
        assert all(np.array_equal(image, frame.image) for image, frame in zip(action_images, retained))
        snapshot = source.diagnostics()
        assert snapshot.decoded_frames == 2
        assert (snapshot.native_width_px, snapshot.native_height_px) == (8, 12)
        assert (snapshot.stream_width_px, snapshot.stream_height_px) == (4, 6)
    finally:
        source.close()


def test_h264_latest_preview_does_not_start_or_consume_capture(monkeypatch) -> None:
    source = AdbH264FrameSource(SimpleNamespace(), monotonic_fn=lambda: 10.0)
    assert source.latest_frame() is None
    assert source._thread is None
    # Delivering a published frame must never need an actual subprocess here.
    monkeypatch.setattr(source, "_ensure_started", lambda: None)
    try:
        source._publish(b"first")
        first = source.latest_frame()
        assert first is not None and first.image == b"first"
        assert source.latest_frame() is first
        assert source.next_frame() is first

        # The preview can read ahead without advancing next_frame's cursor.
        source._publish(b"intermediate")
        source._publish(b"newest")
        newest = source.latest_frame()
        assert newest is not None and newest.image == b"newest"
        assert newest.frame_index == 2
        assert source.next_frame() is newest
        assert source.latest_frame() is newest
    finally:
        source.close()
    assert source.latest_frame() is None


def test_h264_start_is_idempotent_and_does_not_wait_for_input(monkeypatch) -> None:
    source = AdbH264FrameSource(SimpleNamespace())
    entered = threading.Event()
    release = threading.Event()
    starts: list[bool] = []

    def reader() -> None:
        starts.append(True)
        entered.set()
        release.wait(5.0)

    monkeypatch.setattr(source, "_reader_loop", reader)
    try:
        source.start()
        assert entered.wait(2.0)
        source.start()
        assert starts == [True]
        assert source.latest_frame() is None
    finally:
        release.set()
        source.close()
    with pytest.raises(PrototypeControllerError, match="closed"):
        source.start()


@pytest.mark.parametrize("use_readinto", [True, False])
def test_h264_read_buffer_handles_fragments_partial_eof_and_retained_frames(use_readinto: bool) -> None:
    class FragmentedStream:
        def __init__(self) -> None:
            self.data = BytesIO(b"abcdefghijklmn")

        def read(self, size):
            return self.data.read(min(size, 2))

    stream = FragmentedStream()
    if use_readinto:
        def readinto(view):
            chunk = stream.read(len(view))
            view[:len(chunk)] = chunk
            return len(chunk)
        stream.readinto = readinto
    source = AdbH264FrameSource(SimpleNamespace())
    try:
        first = source._read_exact(stream, 6)
        second = source._read_exact(stream, 6)
        partial = source._read_exact(stream, 6)
        assert first == b"abcdef" and second == b"ghijkl" and partial == b"mn"
        assert isinstance(first, bytearray)
        first[0] = ord("X")
        assert second == b"ghijkl"
        assert source._read_exact(stream, 6) == b""
        source._stop_event.set()
        assert source._read_exact(BytesIO(b"unchanged"), 6) == b""
    finally:
        source.close()


@pytest.mark.parametrize("count", [-1, 7, True, "invalid"])
def test_h264_readinto_rejects_invalid_byte_counts(count) -> None:
    source = AdbH264FrameSource(SimpleNamespace())
    try:
        with pytest.raises(PrototypeControllerError, match="invalid read length"):
            source._read_exact(SimpleNamespace(readinto=lambda _view: count), 6)
    finally:
        source.close()


def test_h264_diagnostics_rates_use_decode_publications_and_include_stalls() -> None:
    now = [10.0]
    source = AdbH264FrameSource(SimpleNamespace(), monotonic_fn=lambda: now[0])
    try:
        initial = source.diagnostics()
        assert initial.decoded_frames == 0
        assert initial.decode_publication_age_ms is None
        assert initial.phone_to_output_age_ms is None
        assert initial.reader_started is False
        for timestamp in (10.0, 10.1, 10.2):
            now[0] = timestamp
            source._publish(b"synthetic")
        snapshot = source.diagnostics()
        assert snapshot.decoded_frames == 3
        assert snapshot.recent_decoded_fps == pytest.approx(10.0)
        assert snapshot.decode_publication_age_ms == pytest.approx(0.0)
        assert snapshot.last_decode_publication_monotonic_s == 10.2
        assert snapshot.as_dict()["clock"] == "workstation_decode_publication"

        now[0] = 10.4
        stalled = source.diagnostics()
        assert stalled.recent_decoded_fps == pytest.approx(5.0)
        assert stalled.decode_publication_age_ms == pytest.approx(200.0)
        now[0] = 16.0
        assert source.diagnostics().recent_decoded_fps == 0.0
        assert snapshot.decoded_frames == 3  # Snapshots do not mutate.

        # High-rate publication cannot grow the telemetry history unbounded.
        for index in range(1_000):
            now[0] = 16.0 + index / 1_000
            source._publish(b"synthetic")
        assert len(source._decode_publications) <= source._DIAGNOSTICS_FPS_MAX_SAMPLES
    finally:
        source.close()
    assert source.diagnostics().closed is True


def test_h264_diagnostics_records_read_timeout_and_conversion(monkeypatch) -> None:
    pytest.importorskip("numpy")
    now = [10.0]
    calls = [0]

    def popen(_command, **_kwargs):
        return SimpleNamespace(stdout=BytesIO(), poll=lambda: 0, wait=lambda timeout: 0)

    source = AdbH264FrameSource(
        _h264_test_controller(), popen_factory=popen,
        monotonic_fn=lambda: now[0], max_consecutive_failures=1,
    )

    def read(_stream, size):
        calls[0] += 1
        if calls[0] == 1:
            now[0] += 0.25
            return bytes(size)
        now[0] += 0.75
        raise PrototypeControllerError("synthetic output read timeout")

    monkeypatch.setattr(source, "_read_exact", read)
    try:
        source._reader_loop()
        snapshot = source.diagnostics()
        assert snapshot.decoded_frames == 1
        assert snapshot.output_read_wait_ms == pytest.approx(750.0)
        assert snapshot.output_read_in_progress_ms is None
        assert snapshot.image_conversion_ms == pytest.approx(0.0)
        assert snapshot.decode_publication_age_ms == pytest.approx(750.0)
        assert snapshot.total_failures == 1
        assert snapshot.consecutive_failures == 1
        assert snapshot.last_error == "synthetic output read timeout"
        assert "synthetic output read timeout" in snapshot.terminal_error
        assert (snapshot.native_width_px, snapshot.native_height_px) == (64, 96)
    finally:
        source.close()


def test_h264_diagnostics_exposes_an_in_progress_output_wait(monkeypatch) -> None:
    pytest.importorskip("numpy")
    now = [20.0]
    entered = threading.Event()
    release = threading.Event()

    def popen(_command, **_kwargs):
        return SimpleNamespace(stdout=BytesIO(), poll=lambda: 0, wait=lambda timeout: 0)

    source = AdbH264FrameSource(
        _h264_test_controller(), popen_factory=popen,
        monotonic_fn=lambda: now[0], max_consecutive_failures=1,
    )

    def read(_stream, _size):
        entered.set()
        release.wait(2.0)
        raise PrototypeControllerError("synthetic timeout")

    monkeypatch.setattr(source, "_read_exact", read)
    try:
        source.start()
        assert entered.wait(2.0)
        now[0] = 20.5
        snapshot = source.diagnostics()
        assert snapshot.reader_alive is True
        assert snapshot.decoded_frames == 0
        assert snapshot.output_read_in_progress_ms == pytest.approx(500.0)
        assert snapshot.output_read_wait_ms is None
    finally:
        release.set()
        source.close()
    assert source.diagnostics().reader_alive is False


def test_h264_diagnostics_records_a_real_stalled_output_pipe() -> None:
    pytest.importorskip("numpy")
    read_fd, write_fd = os.pipe()
    output = os.fdopen(read_fd, "rb", buffering=0)

    def popen(command, **_kwargs):
        stdout = BytesIO() if command[0] == "adb" else output
        return SimpleNamespace(stdout=stdout, poll=lambda: 0, wait=lambda timeout: 0)

    source = AdbH264FrameSource(
        _h264_test_controller(), popen_factory=popen,
        stream_read_timeout_s=0.05, max_consecutive_failures=1,
    )
    try:
        source._reader_loop()
        snapshot = source.diagnostics()
        assert snapshot.decoded_frames == 0
        assert snapshot.total_failures == 1
        assert "read timeout" in snapshot.last_error
        assert snapshot.output_read_wait_ms >= 40.0
        assert snapshot.output_read_in_progress_ms is None
    finally:
        source.close()
        output.close()
        os.close(write_fd)


def test_h264_diagnostics_errors_and_restarts_are_bounded(monkeypatch) -> None:
    pytest.importorskip("numpy")
    stderr_modes: list[int] = []

    def popen(_command, **kwargs):
        stderr_modes.append(kwargs["stderr"])
        return SimpleNamespace(stdout=BytesIO(), poll=lambda: 0, wait=lambda timeout: 0)

    source = AdbH264FrameSource(
        _h264_test_controller(), popen_factory=popen,
        restart_delay_s=0.0, max_consecutive_failures=12,
    )

    def fail(_stream, _size):
        raise PrototypeControllerError("synthetic failure " + "x" * 2_000)

    monkeypatch.setattr(source, "_read_exact", fail)
    try:
        source._reader_loop()
        snapshot = source.diagnostics()
        assert snapshot.segment_start_attempts == 12
        assert snapshot.segment_starts == 12
        assert snapshot.segment_restarts == 11
        assert snapshot.total_failures == 12
        assert len(snapshot.recent_errors) == source._DIAGNOSTICS_ERROR_LIMIT
        assert all(len(error) <= source._DIAGNOSTICS_ERROR_CHARS for error in snapshot.recent_errors)
        assert all(mode == subprocess.DEVNULL for mode in stderr_modes)
        assert source._stderr_drains == []
    finally:
        source.close()


def test_h264_diagnostics_drains_large_stderr_without_leaking_restart_threads() -> None:
    processes: list[subprocess.Popen] = []
    segment = [0]

    def popen(command, **kwargs):
        name = "adb" if command[0] == "adb" else "ffmpeg"
        if name == "adb":
            segment[0] += 1
            kwargs["stdin"] = subprocess.PIPE
        marker = f"{name}-segment-{segment[0]}-final"
        process = subprocess.Popen(
            [
                sys.executable, "-c",
                "import sys; "
                "sys.stderr.buffer.write(b'x'*300000 + sys.argv[1].encode()); "
                "sys.stderr.buffer.flush(); sys.stdin.buffer.read()",
                marker,
            ],
            **kwargs,
        )
        processes.append(process)
        return process

    source = AdbH264FrameSource(
        _h264_test_controller(), popen_factory=popen, diagnostics_enabled=True,
    )
    try:
        for number in range(1, 4):
            source._start_segment()
            drains = list(source._stderr_drains)
            assert len(drains) == 2
            deadline = time.monotonic() + 3.0
            while True:
                snapshot = source.diagnostics()
                if (
                    snapshot.adb_stderr_tail.endswith(f"adb-segment-{number}-final")
                    and snapshot.ffmpeg_stderr_tail.endswith(f"ffmpeg-segment-{number}-final")
                ):
                    break
                assert time.monotonic() < deadline, snapshot.as_dict()
                time.sleep(0.01)
            assert snapshot.adb_pid == processes[-2].pid
            assert snapshot.ffmpeg_pid == processes[-1].pid
            assert snapshot.segment_starts == number
            assert snapshot.stderr_capture_error is None
            assert len(snapshot.adb_stderr_tail.encode()) <= source._DIAGNOSTICS_STDERR_TAIL_BYTES
            assert len(snapshot.ffmpeg_stderr_tail.encode()) <= source._DIAGNOSTICS_STDERR_TAIL_BYTES
            source._stop_segment()
            assert all(not drain.thread.is_alive() and drain.stream.closed for drain in drains)
            assert source._stderr_drains == []
            assert source.diagnostics().adb_pid is None
        assert source.diagnostics().segment_restarts == 2
    finally:
        source.close()
        for process in processes:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()


def _synthetic_h264(
    ffmpeg: str, *, b_frames: int = 0, width: int = 64, height: int = 96, frames: int = 24,
) -> bytes:
    encoded = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=30",
            "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast",
            "-bf", str(b_frames),
            "-x264-params", "aud=1:keyint=24:min-keyint=24:scenecut=0",
            "-f", "h264", "pipe:1",
        ],
        capture_output=True,
        timeout=15.0,
    )
    if encoded.returncode and b"Unknown encoder 'libx264'" in encoded.stderr:
        pytest.skip("local ffmpeg does not provide the synthetic H.264 encoder")
    assert encoded.returncode == 0, encoded.stderr.decode(errors="replace")
    return encoded.stdout


def _h264_test_controller(width: int = 64, height: int = 96):
    return SimpleNamespace(
        adb_executable="adb",
        serial="synthetic-only",
        device_id="TEST",
        device_info=lambda: SimpleNamespace(
            connected=True, screen_width_px=width, screen_height_px=height
        ),
    )


@pytest.mark.parametrize("b_frames", [0, 2])
def test_h264_reduced_probing_preserves_every_decoded_frame(b_frames: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is unavailable")
    encoded = _synthetic_h264(ffmpeg, b_frames=b_frames)
    commands: list[list[str]] = []

    def record_command(command, **_kwargs):
        commands.append(list(command))
        return SimpleNamespace(
            stdout=BytesIO(), poll=lambda: 0, wait=lambda timeout: 0
        )

    source = AdbH264FrameSource(
        _h264_test_controller(), ffmpeg_executable=ffmpeg,
        popen_factory=record_command,
    )
    try:
        source._start_segment()
    finally:
        source.close()
    decoded = subprocess.run(
        commands[1], input=encoded, capture_output=True, timeout=15.0, check=True
    ).stdout
    reference = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-flags", "low_delay",
            "-f", "h264", "-i", "pipe:0", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-fps_mode", "passthrough", "pipe:1",
        ],
        input=encoded, capture_output=True, timeout=15.0, check=True,
    ).stdout
    assert len(decoded) == 24 * 64 * 96 * 3
    assert decoded == reference


def test_h264_decodes_while_stream_input_is_still_open(tmp_path) -> None:
    pytest.importorskip("numpy")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is unavailable")
    clip = tmp_path / "synthetic.h264"
    clip.write_bytes(_synthetic_h264(ffmpeg))
    producers: list[subprocess.Popen] = []

    def popen(command, **kwargs):
        if command[0] == "adb":
            # Replace the device with local H.264 bytes followed by an open
            # pipe. Waiting for EOF/probing the whole live stream cannot pass.
            kwargs["stdin"] = subprocess.PIPE
            command = [
                sys.executable, "-c",
                "import pathlib,sys; "
                "sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes()); "
                "sys.stdout.buffer.flush(); sys.stdin.buffer.read()",
                str(clip),
            ]
            process = subprocess.Popen(command, **kwargs)
            producers.append(process)
            return process
        return subprocess.Popen(command, **kwargs)

    source = AdbH264FrameSource(
        _h264_test_controller(), ffmpeg_executable=ffmpeg,
        popen_factory=popen, stream_read_timeout_s=2.0,
        max_consecutive_failures=1,
    )
    try:
        source.start()
        frame = source.next_frame()
        assert frame.image.shape == (96, 64, 3)
        assert producers[0].poll() is None
        assert source.latest_frame() is not None
    finally:
        source.close()
        for process in producers:
            if process.stdin is not None:
                process.stdin.close()


@pytest.mark.parametrize(
    ("width", "height", "stream_max_width", "stream_width", "stream_height"),
    [(1080, 2400, None, 1080, 2400), (64, 96, 32, 32, 48)],
)
def test_h264_native_capture_preserves_all_pixels_and_independent_writable_images(
    tmp_path, monkeypatch, width, height, stream_max_width, stream_width, stream_height,
) -> None:
    np = pytest.importorskip("numpy")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is unavailable")
    frame_count = 6
    encoded = _synthetic_h264(ffmpeg, width=stream_width, height=stream_height, frames=frame_count)
    clip = tmp_path / "native.h264"
    clip.write_bytes(encoded)

    def popen(command, **kwargs):
        if command[0] == "adb":
            # A local synthetic clip replaces every device operation.
            command = [sys.executable, "-c", "import pathlib,sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())", str(clip)]
        return subprocess.Popen(command, **kwargs)

    source = AdbH264FrameSource(
        _h264_test_controller(width, height), ffmpeg_executable=ffmpeg,
        stream_max_width_px=stream_max_width,
        popen_factory=popen, max_consecutive_failures=1,
    )
    retained: list[SourceFrame] = []
    publish = source._publish

    def retain(image):
        publish(image)
        retained.append(source.latest_frame())

    monkeypatch.setattr(source, "_publish", retain)
    try:
        source._reader_loop()
        reference = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-flags", "low_delay",
                "-f", "h264", "-i", "pipe:0", "-f", "rawvideo",
                "-pix_fmt", "bgr24", "-fps_mode", "passthrough", "pipe:1",
            ],
            input=encoded, capture_output=True, timeout=15.0, check=True,
        ).stdout
        expected = np.frombuffer(reference, dtype=np.uint8).reshape(frame_count, stream_height, stream_width, 3)
        if (stream_width, stream_height) != (width, height):
            cv2 = pytest.importorskip("cv2")
            expected = np.stack([
                cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
                for image in expected
            ])
        assert len(retained) == frame_count
        assert [frame.frame_index for frame in retained] == list(range(frame_count))
        assert [frame.timestamp_s for frame in retained] == sorted(frame.timestamp_s for frame in retained)
        for index, frame in enumerate(retained):
            assert frame.image.flags.writeable
            assert frame.image.shape == (height, width, 3)
            assert np.array_equal(frame.image, expected[index])
        assert not np.shares_memory(retained[0].image, retained[1].image)
        retained[0].image.fill(0)
        assert np.array_equal(retained[1].image, expected[1])
        snapshot = source.diagnostics()
        assert snapshot.decoded_frames == frame_count
        assert (snapshot.native_width_px, snapshot.native_height_px) == (width, height)
        assert (snapshot.stream_width_px, snapshot.stream_height_px) == (stream_width, stream_height)
    finally:
        source.close()


def test_autonomous_phone_can_verify_a_card_from_a_stream_frame() -> None:
    controller = FakePhoneController("A")
    screenshot_calls = 0

    def unexpected_screenshot():
        nonlocal screenshot_calls
        screenshot_calls += 1
        raise AssertionError("stream-backed action verification should not call ADB screenshot")

    controller.screenshot = unexpected_screenshot
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    frame = Frame("A", 10, 10, b"stream-frame")

    class Vision:
        threshold = 0.62

        @staticmethod
        def find_hand_card_in_slot(_frame, _profile, _card_id, _slot, *, hand_px):
            assert hand_px == (237.0, 1900.0, 808.0, 500.0)
            return CardMatch("hog-rider", 0.99, (338, 2179))

    phone = AutonomousPhone(
        controller,
        profile,
        Vision(),
        action_frame_provider=lambda: frame,
    )
    calibration = CalibrationArtifact.for_screen(
        device_label="A",
        screen_width_px=1080,
        screen_height_px=2400,
        hand_px=(237.0, 1900.0, 808.0, 500.0),
    )

    slot, _selected, _placed = phone.select_and_place(
        "hog-rider",
        calibration=calibration,
        arena_cell=(3, 20),
        expected_slot=0,
    )

    assert slot == 0
    assert screenshot_calls == 0
    assert len(controller.taps) == 2


def test_detector_speed_adapter_overrides_each_yolo_call_size() -> None:
    calls: list[int] = []

    class Model:
        def predict(self, *_args, **kwargs):
            calls.append(kwargs["imgsz"])
            return []

    detector = SimpleNamespace(models=[Model(), Model()])
    configure_detector_inference_size(detector, 640)

    detector.models[0].predict(object(), imgsz=896)
    detector.models[1].predict(object(), imgsz=896)

    assert calls == [640, 640]
    assert detector._prototype_inference_size == 640


def test_detector_resolution_defaults_to_original_extractor_size() -> None:
    parser = build_arg_parser()

    default_args = parser.parse_args(["--video", "gameplay.mp4"])
    fast_args = parser.parse_args(["--video", "gameplay.mp4", "--yolo-image-size", "640"])
    screenshot_args = parser.parse_args(
        ["--serial", "SERIAL", "--adb-transport", "screenshot"]
    )

    assert default_args.yolo_image_size == 896
    assert fast_args.yolo_image_size == 640
    assert default_args.adb_transport == "stream"
    assert screenshot_args.adb_transport == "screenshot"


def test_live_runner_omits_ignored_bomb_objects_before_session() -> None:
    def match(label: str, team: str = "ally") -> Match:
        return Match(
            troop=Detection(
                track_id=1,
                class_name=label,
                team=team,
                confidence=0.9,
                x1=0.0,
                y1=0.0,
                x2=1.0,
                y2=1.0,
                center_x=0.5,
                center_y=0.5,
            ),
            bar=None,
        )

    analysis = FrameAnalysisResult(
        rendered=None,
        elixir={},
        elixir_change=None,
        towers_hp={},
        time=None,
        time_left_s=None,
        total_remaining_s=None,
        overtime=False,
        hand_state={},
        yolo_boxes=None,
        clock_boxes=[],
        emote_boxes=[],
        matches=[
            match("bomb"),
            match("bomber"),
            match("hog"),
            match("skeleton-evolution"),
            match("giant", team="enemy"),
        ],
        arena_px=(0, 0, 1, 1),
        tower_hp_debug_steps={},
        timer_debug_steps={},
    )

    filtered = _filter_live_analysis(analysis)

    assert "bomb" in LIVE_IGNORED_DETECTOR_LABELS
    assert "hog-rider" in LIVE_OWN_DECK_CARD_NAMES
    assert LIVE_OWN_DETECTOR_ALIASES["skeleton-evolution"] == "skeletons"
    assert [item.troop.class_name for item in filtered.matches] == [
        "hog",
        "skeleton-evolution",
        "giant",
    ]
    assert [item.troop.class_name for item in analysis.matches] == [
        "bomb",
        "bomber",
        "hog",
        "skeleton-evolution",
        "giant",
    ]


def test_live_detection_filter_seeds_own_play_and_confirms_enemy_on_second_frame() -> None:
    def match(
        label: str,
        team: str,
        *,
        track_id: int | None,
        center: tuple[float, float],
    ) -> Match:
        center_x, center_y = center
        return Match(
            troop=Detection(
                track_id=track_id,
                class_name=label,
                team=team,
                confidence=0.9,
                x1=center_x - 5.0,
                y1=center_y - 5.0,
                x2=center_x + 5.0,
                y2=center_y + 5.0,
                center_x=center_x,
                center_y=center_y,
            ),
            bar=None,
        )

    def analysis(matches: list[Match]) -> FrameAnalysisResult:
        return FrameAnalysisResult(
            rendered=None,
            elixir={},
            elixir_change=None,
            towers_hp={},
            time=None,
            time_left_s=None,
            total_remaining_s=None,
            overtime=False,
            hand_state={},
            yolo_boxes=None,
            clock_boxes=[],
            emote_boxes=[],
            matches=matches,
            arena_px=(0, 0, 568, 896),
            tower_hp_debug_steps={},
            timer_debug_steps={},
        )

    detection_filter = LiveDetectionFilter()
    detection_filter.notify_own_play(
        card_name="hog-rider",
        cell=(3, 20),
        arena_px=(0, 0, 568, 896),
        timestamp_s=10.0,
    )
    enemy = match("giant", "enemy", track_id=17, center=(300.0, 300.0))

    first = detection_filter.update(analysis([enemy]), timestamp_s=10.25)
    first_classes = [item.troop.class_name for item in first.matches]
    assert first_classes == ["hog-rider"]
    assert first.matches[0].troop.team == "ally"

    second = detection_filter.update(analysis([enemy]), timestamp_s=10.50)
    assert [item.troop.class_name for item in second.matches] == [
        "giant",
        "hog-rider",
    ]
    assert second.matches[0].troop.team == "enemy"
    assert second.matches[1].troop.team == "ally"

    # Once the extractor catches the dispatched troop, the temporary seed is
    # removed and the real detection (including its position/HP) is retained.
    own_detection = match("hog", "ally", track_id=41, center=(300.0, 700.0))
    third = detection_filter.update(
        analysis([own_detection]),
        timestamp_s=10.75,
    )
    assert [item.troop.class_name for item in third.matches] == ["hog"]


def _hand_state(
    *cards: str | None,
    next_card: str | None = "cannon",
) -> dict[str, object]:
    return {
        **{
            f"card_{index}": (card, 99.0) if card is not None else None
            for index, card in enumerate(cards, start=1)
        },
        "next_card": (next_card, 99.0) if next_card is not None else None,
    }


def test_live_hand_filter_warms_up_and_rejects_duplicate_initial_predictions() -> None:
    filter_ = LiveHandStateFilter()

    duplicate = _hand_state("fire-spirit", "knight", "dart-goblin", "fire-spirit")
    for _ in range(2):
        filtered = filter_.update(duplicate)
        assert not filter_.ready
        assert filtered["card_1"] is None

    filtered = filter_.update(duplicate)

    assert not filter_.ready
    assert filtered["card_1"] == ("fire-spirit", 99.0)
    assert filtered["card_4"] is None


def test_live_hand_filter_uses_stable_next_card_for_replacement() -> None:
    filter_ = LiveHandStateFilter()
    before = _hand_state("log", "ice-spirit", "ice-golem", "hog-rider")
    after = _hand_state(
        "log", "cannon", "ice-golem", "hog-rider", next_card="fireball"
    )

    for _ in range(3):
        filter_.update(before)
    assert filter_.ready

    filter_.expect_replacement(1)
    held = filter_.update(before)
    assert filter_.ready
    assert held["card_2"] == ("cannon", 99.0)
    assert held["next_card"] is None

    first = filter_.update(after)
    second = filter_.update(after)
    third = filter_.update(after)
    assert first["card_2"] == ("cannon", 99.0)
    assert second["card_2"] == ("cannon", 99.0)
    assert third["card_2"] == ("cannon", 99.0)
    assert first["next_card"] is None
    assert second["next_card"] is None
    assert third["next_card"] == ("fireball", 99.0)
    assert filter_.ready


def test_runner_dispatches_play_but_never_dispatches_wait() -> None:
    class Source:
        def __init__(self) -> None:
            self.frames = [
                SourceFrame(object(), 0, 1.0),
                SourceFrame(object(), 1, 2.0),
            ]
            self.closed = False

        def next_frame(self):
            return self.frames.pop(0) if self.frames else None

        def close(self) -> None:
            self.closed = True

    class Actor:
        def __init__(self) -> None:
            self.actions = iter(
                (
                    Action(kind="Wait"),
                    Action(kind="Play", card_idx=0, cell=(9, 20)),
                )
            )
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def decide(self, _observation):
            return next(self.actions)

    state = SimpleNamespace(
        hud=SimpleNamespace(
            hand_cards=["hog-rider", "cannon", "musketeer", "skeletons"],
            elixir_self=10.0,
            time_left_s=120.0,
        )
    )
    step = SimpleNamespace(
        in_game=True,
        should_emit=True,
        game_state=state,
        analysis=SimpleNamespace(matches=[]),
    )
    dispatched: list[tuple[object, object]] = []

    runner = LivePrototypeRunner(
        Source(),
        detector=object(),
        actor=Actor(),
        execute=True,
        phone=object(),
        calibration=object(),
        normalize=False,
        poll_interval_s=0.0,
        min_action_interval_s=0.0,
        post_action_delay_s=0.0,
        session=SimpleNamespace(process=lambda *_args, **_kwargs: step),
        process_frame_fn=lambda *_args, **_kwargs: object(),
        normalize_frame_fn=lambda image: image,
        observation_builder=lambda _step: object(),
        dispatch_fn=lambda phone, action, *_args, **_kwargs: dispatched.append(
            (phone, action)
        ),
    )
    records = []
    summary = runner.run(on_record=records.append)

    assert summary.as_dict() == {
        "frames": 2,
        "emitted_frames": 2,
        "waits": 1,
        "proposed_plays": 1,
        "dispatched_plays": 1,
    }
    assert len(dispatched) == 1
    assert dispatched[0][1] == Action(kind="Play", card_idx=0, cell=(9, 20))
    assert [record.result for record in records] == ["wait", "dispatched"]
    assert records[0].visual_state == {
        "hand": ["hog-rider", "cannon", "musketeer", "skeletons"],
        "next_card": None,
        "elixir": 10.0,
        "enemy_elixir_est": None,
        "time_left_s": 120.0,
        "total_remaining_s": None,
        "overtime": False,
        "ally_units": [],
        "enemy_units": [],
        "seen_enemy_cards": [],
        "tower_hp_self": [],
        "tower_hp_enemy": [],
        "own_king_active": False,
        "enemy_king_active": False,
        "detection_count": 0,
        "arena_px": [],
    }


def test_decision_output_combines_frame_action_and_extracted_state() -> None:
    output = format_decision_record(
        DecisionRecord(
            frame_index=1884,
            timestamp_s=7457.318,
            in_game=True,
            emitted=True,
            action={"kind": "wait"},
            result="wait",
            visual_state={
                "hand": ["ice-golem", "cannon", "log", "musketeer"],
                "next_card": "hog-rider",
                "elixir": 8.5,
                "time_left_s": 274.2,
                "overtime": False,
                "ally_units": [
                    {
                        "label": "dark-prince",
                        "track": 644,
                        "center_px": [875.6, 851.2],
                        "estimated_hp": 1234,
                        "confidence": 0.56,
                    }
                ],
                "enemy_units": [
                    {
                        "label": "night-witch",
                        "track": 620,
                        "center_px": [378.4, 1552.2],
                        "estimated_hp": 987,
                        "confidence": 0.73,
                    },
                    {
                        "label": "electro-wizard",
                        "track": 640,
                        "center_px": [302.8, 1375.2],
                        "estimated_hp": 1456,
                        "confidence": 0.95,
                    },
                ],
                "detection_count": 3,
                "tower_hp_self": [1000, 3000, 1200],
                "tower_hp_enemy": [900, 3000, 1100],
                "seen_enemy_cards": [4, 17],
            },
        )
    )

    assert "frame=1884 · t=7457.318s" in output
    assert "hand=[ice-golem · cannon · log · musketeer]" in output
    visual_line = next(line for line in output.splitlines() if "VISUAL" in line)
    assert "dark-prince#644 hp=1234 @875.6,851.2 [0.56]" in visual_line
    assert "night-witch#620 hp=987 @378.4,1552.2 [0.73]" in visual_line
    assert "electro-wizard#640 hp=1456 @302.8,1375.2 [0.95]" in visual_line
    assert "action={'kind': 'wait'}  result=wait" in output


def test_runner_rejects_execution_without_action_sink() -> None:
    with pytest.raises(RuntimeError, match="requires both an AutonomousPhone"):
        LivePrototypeRunner(
            SimpleNamespace(next_frame=lambda: None, close=lambda: None),
            detector=object(),
            actor=object(),
            execute=True,
            normalize=False,
            session=object(),
            process_frame_fn=lambda *_args, **_kwargs: object(),
            normalize_frame_fn=lambda image: image,
        )
