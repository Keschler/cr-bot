from __future__ import annotations

from typing import Any


def _frame_dimensions(image: Any) -> tuple[int | None, int | None]:
    shape = getattr(image, "shape", None)
    try:
        if shape is not None and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])
            if width > 0 and height > 0:
                return width, height
    except (TypeError, ValueError):
        pass
    return None, None


def encode_jpeg(bgr: Any, max_width: int = 720) -> bytes | None:
    """Encode a BGR image to JPEG bytes, downscaling to ``max_width``.

    Returns ``None`` when OpenCV is unavailable or encoding fails.  Never
    raises for UI-path robustness.
    """

    if bgr is None:
        return None
    try:
        import cv2  # lazy: keeps module import light
    except ImportError:
        return None
    try:
        height_width = getattr(bgr, "shape", None)
        if height_width is None or len(height_width) < 2:
            return None
        height, width = int(height_width[0]), int(height_width[1])
        if width <= 0 or height <= 0:
            return None
        image = bgr
        if isinstance(max_width, int) and max_width > 0 and width > max_width:
            scale = max_width / float(width)
            new_height = max(1, int(round(height * scale)))
            image = cv2.resize(
                bgr, (int(max_width), new_height), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok or buf is None:
            return None
        return bytes(buf.tobytes())
    except Exception:
        return None
