"""Back-compat shim for the former flat ``server`` module (light import surface).

READ access note: ``_session``/``_lock``/``_thread``/``_stop_event`` live in
:mod:`src.frontend.services.session_manager`. Reads via
``frontend_server._session`` delegate to that module (see ``__getattr__``).
Rebinding ``frontend_server._session = ...`` does NOT propagate to the
session manager; poke
``src.frontend.services.session_manager._session`` instead.
"""

from __future__ import annotations

from typing import Any

from .api.assets import api_card_icon, api_grid
from .api.corrections import (
    api_frame_reevaluate,
    api_frame_reevaluate_revert,
    api_labels,
)
from .api.frames import api_frame_at, api_frame_latest, api_frames
from .api.live import api_live_start, api_stop
from .api.roi import api_roi_preview
from .api.stream import api_stream
from .api.system import api_checkpoints, api_health, api_status
from .api.video import api_upload, api_video_info, api_video_start
from .app import app, create_app
from .models.requests import (
    LiveStartRequest,
    ReevaluateRequest,
    StopResponse,
    VideoStartRequest,
)
from .services import session_manager as _session_manager
from .services.card_art import (
    CARD_ART_ALIASES,
    CARD_ART_CANDIDATES,
    _card_art_candidates,
    _card_art_dir,
)
from .services.checkpoints import (
    CHECKPOINT_GLOBS,
    _default_checkpoint,
    _list_checkpoints,
)
from .services.paths import (
    REPO_ROOT,
    STATIC_DIR,
    UPLOAD_DIR,
    VIDEO_EXTENSIONS,
    _resolve_against_repo,
)
from .services.session_manager import (
    _frame_to_json,
    _get_session,
    _start_worker,
    _stop_current_locked,
    stop_current_session,
)
from .services.uploads import _sanitize_upload_filename

__all__ = [
    "CARD_ART_ALIASES",
    "CARD_ART_CANDIDATES",
    "CHECKPOINT_GLOBS",
    "REPO_ROOT",
    "STATIC_DIR",
    "UPLOAD_DIR",
    "VIDEO_EXTENSIONS",
    "LiveStartRequest",
    "ReevaluateRequest",
    "StopResponse",
    "VideoStartRequest",
    "_card_art_candidates",
    "_card_art_dir",
    "_default_checkpoint",
    "_frame_to_json",
    "_get_session",
    "_list_checkpoints",
    "_resolve_against_repo",
    "_sanitize_upload_filename",
    "_start_worker",
    "_stop_current_locked",
    "api_card_icon",
    "api_checkpoints",
    "api_frame_at",
    "api_frame_latest",
    "api_frame_reevaluate",
    "api_frame_reevaluate_revert",
    "api_frames",
    "api_grid",
    "api_health",
    "api_labels",
    "api_live_start",
    "api_roi_preview",
    "api_status",
    "api_stop",
    "api_stream",
    "api_upload",
    "api_video_info",
    "api_video_start",
    "app",
    "create_app",
    "stop_current_session",
]


def __getattr__(name: str) -> Any:
    if name in {"_lock", "_session", "_thread", "_stop_event"}:
        return getattr(_session_manager, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
