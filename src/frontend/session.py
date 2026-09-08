"""Back-compat shim for the former flat ``session`` module (light import surface)."""

from __future__ import annotations

from .corrections._common import (
    CorrectionUnprocessableError,
    FrameNotFoundError,
    NoActorError,
)
from .corrections.edits import (
    _apply_correction_edits,
    _find_row_index,
    _match_edit_key,
)
from .corrections.observation import _corrected_observation, _live_policy_imports
from .corrections.reevaluate import clear_reevaluation, reevaluate_frame
from .corrections.vocab import unit_label_vocabulary
from .imaging import _frame_dimensions, encode_jpeg
from .models.frames import FrontendFrame
from .models.session import FrontendSession
from .runners.live import run_live_session
from .runners.pump import (
    _OffsetFrameSource,
    _cell_to_list,
    _decide_with_fallback,
    _detection_row,
    _finite_or_none,
    _new_tracker_actions,
    _run_pump_loop,
    _summarize_enemy_play,
    _summarize_own_action,
    _suggestion_to_dict,
    _try_import_decide_with_scores,
)
from .runners.video import run_video_session
from .services.devices import (
    INFERENCE_DEVICES,
    _cuda_available,
    resolve_inference_devices,
)

__all__ = [
    "CorrectionUnprocessableError",
    "FrameNotFoundError",
    "FrontendFrame",
    "FrontendSession",
    "INFERENCE_DEVICES",
    "NoActorError",
    "_OffsetFrameSource",
    "_apply_correction_edits",
    "_corrected_observation",
    "_cuda_available",
    "_cell_to_list",
    "_decide_with_fallback",
    "_detection_row",
    "_finite_or_none",
    "_find_row_index",
    "_frame_dimensions",
    "_live_policy_imports",
    "_match_edit_key",
    "_new_tracker_actions",
    "_run_pump_loop",
    "_summarize_enemy_play",
    "_summarize_own_action",
    "_suggestion_to_dict",
    "_try_import_decide_with_scores",
    "clear_reevaluation",
    "encode_jpeg",
    "reevaluate_frame",
    "resolve_inference_devices",
    "run_live_session",
    "run_video_session",
    "unit_label_vocabulary",
]
