from __future__ import annotations

import re
from pathlib import Path

from .paths import VIDEO_EXTENSIONS


def _sanitize_upload_filename(filename: str | None) -> str:
    raw = (filename or "upload").strip() or "upload"
    name = Path(raw).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "upload"
    if Path(cleaned).suffix.lower() not in VIDEO_EXTENSIONS:
        cleaned += ".mp4"
    return cleaned[:128]
