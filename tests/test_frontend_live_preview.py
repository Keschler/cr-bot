"""Live pixels must stay fresh even when analyzed frames arrive slowly."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.frontend.api import frames
from src.frontend.models.frames import FrontendFrame, LivePreviewFrame
from src.frontend.models.session import FrontendSession
from src.frontend.services import live_preview


def test_preview_advances_without_analysis_and_history_stays_aligned(monkeypatch):
    session = FrontendSession(mode="live")
    session.running = True
    analyzed = FrontendFrame(frame_index=3, timestamp_s=5.0, jpeg_bytes=b"analyzed")
    session.push(analyzed)
    captured = [SimpleNamespace(frame_index=60, timestamp_s=9.98, image=b"capture-60")]
    session.set_live_preview_provider(live_preview.LivePreviewProvider(lambda: captured[0]))
    monkeypatch.setattr(live_preview, "encode_preview_jpeg", lambda image: image)
    monkeypatch.setattr(frames, "_get_session", lambda: session)
    monkeypatch.setattr(frames.time, "monotonic", lambda: 10.0)

    # Preview does not borrow the policy lock or wait for a decision.
    with session.actor_lock:
        response = frames.api_frame_preview()
    assert response.body == b"capture-60"
    assert response.headers["x-frame-index"] == "60"
    assert float(response.headers["x-capture-age-ms"]) == pytest.approx(20.0)
    assert float(response.headers["x-analysis-age-ms"]) == 5000.0
    assert response.headers["cache-control"] == "no-store"

    captured[0] = SimpleNamespace(frame_index=90, timestamp_s=10.0, image=b"capture-90")
    assert frames.api_frame_preview().body == b"capture-90"
    assert session.get_latest() is analyzed
    assert session.history == [analyzed]
    assert frames.api_frame_latest().body == b"analyzed"
    assert frames.api_frame_latest().headers["x-frame-index"] == "3"
    assert frames.api_frame_at(3).body == b"analyzed"


def test_preview_available_during_model_warmup(monkeypatch):
    session = FrontendSession(mode="live")
    session.running = True
    session.set_live_preview_provider(lambda: LivePreviewFrame(42, 9.0, b"live"))
    monkeypatch.setattr(frames, "_get_session", lambda: session)
    response = frames.api_frame_preview()
    assert response.status_code == 200
    assert response.body == b"live"
    assert response.headers["x-frame-index"] == "42"
    assert "x-analysis-age-ms" not in response.headers
    assert session.get_latest() is None


def test_preview_caches_only_one_frame_and_preserves_normalization(monkeypatch):
    source = [SimpleNamespace(frame_index=4, timestamp_s=2.0, image="native-4")]
    normalized = []
    encoded = []

    def normalize(image):
        normalized.append(image)
        return image + "-normalized"

    def encode(image):
        encoded.append(image)
        return image.encode()

    monkeypatch.setattr(live_preview, "encode_preview_jpeg", encode)
    provider = live_preview.LivePreviewProvider(lambda: source[0], normalize_frame=normalize)
    first = provider()
    assert first.jpeg_bytes == b"native-4-normalized"
    assert provider() is first
    assert normalized == ["native-4"]
    assert encoded == ["native-4-normalized"]
    source[0] = SimpleNamespace(frame_index=5, timestamp_s=3.0, image="native-5")
    assert provider().frame_index == 5
    assert encoded == ["native-4-normalized", "native-5-normalized"]
    source[0] = None
    assert provider() is None  # Never replay a cached picture after capture closes.


def test_preview_retries_jpeg_failure_instead_of_caching_it(monkeypatch):
    frame = SimpleNamespace(frame_index=4, timestamp_s=2.0, image=object())
    results = iter([None, b"jpeg"])
    monkeypatch.setattr(live_preview, "encode_preview_jpeg", lambda _: next(results))
    provider = live_preview.LivePreviewProvider(lambda: frame)
    assert provider() is None
    assert provider().jpeg_bytes == b"jpeg"


def test_live_jpeg_is_smaller_and_preserves_geometry_without_changing_analysis_image():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from src.frontend.imaging import encode_jpeg, encode_preview_jpeg

    image = np.zeros((2400, 1080, 3), dtype=np.uint8)
    image[:1200, :540] = (0, 0, 255)
    image[:1200, 540:] = (0, 255, 0)
    image[1200:, :540] = (255, 0, 0)
    image[1200:, 540:] = (255, 255, 255)
    original = image.copy()
    preview = cv2.imdecode(np.frombuffer(encode_preview_jpeg(image), dtype=np.uint8), cv2.IMREAD_COLOR)
    history = cv2.imdecode(np.frombuffer(encode_jpeg(image), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert preview.shape[:2] == (1067, 480)
    assert history.shape[:2] == (1600, 720)
    for y, x in [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]:
        actual = preview[int(y * preview.shape[0]), int(x * preview.shape[1])]
        expected = image[int(y * image.shape[0]), int(x * image.shape[1])]
        np.testing.assert_allclose(actual, expected, atol=3)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("mode,running", [("live", False), ("video", True), ("video", False)])
def test_nonlive_or_stopped_preview_uses_analyzed_pixels(monkeypatch, mode, running):
    session = FrontendSession(mode=mode)
    session.running = running
    session.push(FrontendFrame(frame_index=4, timestamp_s=2.0, jpeg_bytes=b"history"))

    def unexpected_capture():
        raise AssertionError("stopped/video sessions must not poll live capture")

    session.set_live_preview_provider(unexpected_capture)
    monkeypatch.setattr(frames, "_get_session", lambda: session)
    response = frames.api_frame_preview()
    assert response.body == b"history"
    assert response.headers["x-frame-index"] == "4"
    if mode == "video":
        assert "x-capture-age-ms" not in response.headers


def test_preview_without_capture_or_history_returns_no_content(monkeypatch):
    session = FrontendSession(mode="live")
    session.running = True
    session.set_live_preview_provider(lambda: None)
    monkeypatch.setattr(frames, "_get_session", lambda: session)
    response = frames.api_frame_preview()
    assert response.status_code == 204
    assert response.headers["cache-control"] == "no-store"


def test_javascript_live_preview_regressions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed only for the JavaScript regression tests")
    script = Path(__file__).with_name("frontend_live_preview.mjs")
    result = subprocess.run(
        [node, "--experimental-vm-modules", str(script)],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
