"""Conservative, hash-verified screen-state detection for connected lab runs.

The physical runner must not infer a battle lifecycle from sleeps or from the
fact that an ADB command succeeded.  This module provides a small integration
boundary for reviewed screen templates.  Templates are loaded lazily, each
file is verified against the sealed manifest, and a state is returned only
when the best template is above the declared score and is separated from the
runner-up by the declared margin.

This detector is deliberately not an observation truth oracle.  It gates the
coarse lifecycle only; entity/event extraction still goes through the
confidence-gated observation and replay-cache pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .devices import Frame, sha256_bytes
from .lifecycle import LifecycleState
from .schema import PhysicalLabError, canonical_hash


SCREEN_TEMPLATE_SCHEMA_VERSION = 1
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ScreenStateDetectionError(PhysicalLabError):
    """Raised when a reviewed template set cannot classify a frame."""


@dataclass(frozen=True, slots=True)
class _Template:
    state: LifecycleState
    path: Path
    sha256: str
    image: Any


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _relative_template_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ScreenStateDetectionError("screen template path is required")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ScreenStateDetectionError(
            f"screen template path must stay below manifest directory: {value!r}"
        )
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ScreenStateDetectionError(
            f"screen template path escapes manifest directory: {value!r}"
        ) from error
    return resolved


def _read_image(path: Path) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment-specific
        raise ScreenStateDetectionError(
            "reviewed screen-state detection requires OpenCV and NumPy"
        ) from error
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ScreenStateDetectionError(f"cannot decode screen template {path}")
    return image


class TemplateLifecycleDetector:
    """Classify screenshots using a sealed, reviewed template manifest.

    The manifest is a JSON object of the following shape::

        {
          "schema_version": 1,
          "device_id": "A",
          "score_threshold": 0.96,
          "margin_threshold": 0.02,
          "templates": {
            "recovery": [{"path": "recovery.png", "sha256": "sha256:..."}],
            "lobby": [...],
            "challenge_sent": [...],
            "challenge_accepted": [...],
            "loading": [...],
            "battle": [...],
            "result": [...],
            "archived": [...]
          },
          "manifest_hash": "sha256:..."
        }

    At least one template is required for every lifecycle state.  Relative
    paths and file hashes make the detector reproducible and prevent an
    unreviewed replacement image from silently changing admission behavior.
    """

    def __init__(
        self,
        frame_source: Callable[[], Frame],
        manifest_path: str | Path,
        *,
        expected_device_id: str | None = None,
    ) -> None:
        if not callable(frame_source):
            raise ScreenStateDetectionError("screen-state frame source must be callable")
        self._frame_source = frame_source
        self.manifest_path = Path(manifest_path).resolve()
        try:
            raw_bytes = self.manifest_path.read_bytes()
            raw = json.loads(raw_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise ScreenStateDetectionError(
                f"cannot load screen-state template manifest {self.manifest_path}: {error}"
            ) from error
        if not isinstance(raw, Mapping):
            raise ScreenStateDetectionError("screen-state template manifest must be an object")
        unknown_manifest_fields = sorted(
            set(raw)
            - {
                "schema_version",
                "device_id",
                "score_threshold",
                "margin_threshold",
                "templates",
                "manifest_hash",
            }
        )
        if unknown_manifest_fields:
            raise ScreenStateDetectionError(
                f"unknown screen-state manifest fields: {unknown_manifest_fields}"
            )
        if raw.get("schema_version") != SCREEN_TEMPLATE_SCHEMA_VERSION:
            raise ScreenStateDetectionError(
                f"unsupported screen-state template schema: {raw.get('schema_version')!r}"
            )
        device_id = raw.get("device_id")
        if not isinstance(device_id, str) or not device_id.strip():
            raise ScreenStateDetectionError("screen-state template manifest needs device_id")
        self.device_id = device_id.strip()
        if expected_device_id is not None and self.device_id != expected_device_id:
            raise ScreenStateDetectionError(
                f"template device_id {self.device_id!r} does not match {expected_device_id!r}"
            )
        self.score_threshold = self._probability(raw.get("score_threshold", 0.96), "score_threshold")
        self.margin_threshold = self._probability(raw.get("margin_threshold", 0.02), "margin_threshold")
        declared_manifest_hash = raw.get("manifest_hash")
        if declared_manifest_hash is not None:
            if not isinstance(declared_manifest_hash, str) or not _HASH_RE.fullmatch(declared_manifest_hash):
                raise ScreenStateDetectionError("manifest_hash must be a sha256 digest")
            without_hash = dict(raw)
            without_hash.pop("manifest_hash", None)
            actual_manifest_hash = canonical_hash(without_hash)
            if declared_manifest_hash != actual_manifest_hash:
                raise ScreenStateDetectionError(
                    f"screen-state manifest hash mismatch: declared={declared_manifest_hash!r}, "
                    f"actual={actual_manifest_hash!r}"
                )
        templates_raw = raw.get("templates")
        if not isinstance(templates_raw, Mapping):
            raise ScreenStateDetectionError("screen-state template manifest needs templates")
        unknown_states = sorted(set(templates_raw) - {state.value for state in LifecycleState})
        if unknown_states:
            raise ScreenStateDetectionError(
                f"unknown lifecycle states in screen-state manifest: {unknown_states}"
            )
        root = self.manifest_path.parent
        loaded: dict[LifecycleState, tuple[_Template, ...]] = {}
        for state in LifecycleState:
            rows = templates_raw.get(state.value)
            if not isinstance(rows, list) or not rows:
                raise ScreenStateDetectionError(
                    f"screen-state manifest needs at least one template for {state.value}"
                )
            state_templates: list[_Template] = []
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise ScreenStateDetectionError(
                        f"templates.{state.value}[{index}] must be an object"
                    )
                unknown = sorted(set(row) - {"path", "sha256"})
                if unknown:
                    raise ScreenStateDetectionError(
                        f"unknown fields in templates.{state.value}[{index}]: {unknown}"
                    )
                path = _relative_template_path(root, row.get("path"))
                declared_hash = row.get("sha256")
                if not isinstance(declared_hash, str) or not _HASH_RE.fullmatch(declared_hash):
                    raise ScreenStateDetectionError(
                        f"templates.{state.value}[{index}].sha256 must be a sha256 digest"
                    )
                if not path.is_file():
                    raise ScreenStateDetectionError(f"screen template does not exist: {path}")
                actual_hash = _hash_file(path)
                if actual_hash != declared_hash:
                    raise ScreenStateDetectionError(
                        f"screen template hash mismatch for {path}: "
                        f"declared={declared_hash}, actual={actual_hash}"
                    )
                state_templates.append(
                    _Template(state=state, path=path, sha256=declared_hash, image=_read_image(path))
                )
            loaded[state] = tuple(state_templates)
        self._templates = loaded
        self._manifest_sha256 = sha256_bytes(raw_bytes)
        self._last_frame_index: int | None = None
        self._last_scores: Mapping[str, float] = {}

    @staticmethod
    def _probability(value: object, field_name: str) -> float:
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ScreenStateDetectionError(f"{field_name} must be finite")
        result = float(value)
        if not 0 <= result <= 1:
            raise ScreenStateDetectionError(f"{field_name} must be between zero and one")
        return result

    def provenance(self) -> dict[str, object]:
        return {
            "kind": "reviewed_screen_template_detector",
            "device_id": self.device_id,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self._manifest_sha256,
            "score_threshold": self.score_threshold,
            "margin_threshold": self.margin_threshold,
            "template_count": sum(len(rows) for rows in self._templates.values()),
            "template_hashes": {
                state.value: [template.sha256 for template in rows]
                for state, rows in sorted(self._templates.items(), key=lambda item: item[0].value)
            },
        }

    @property
    def last_scores(self) -> Mapping[str, float]:
        return dict(self._last_scores)

    @staticmethod
    def _score(frame_image: Any, template_image: Any) -> float:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - environment-specific
            raise ScreenStateDetectionError("OpenCV is required for screen-state detection") from error
        target_height, target_width = template_image.shape[:2]
        if frame_image.shape[:2] != (target_height, target_width):
            frame_image = cv2.resize(
                frame_image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        frame_gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        score = float(cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)[0, 0])
        if not math.isfinite(score):
            return 0.0
        return max(-1.0, min(1.0, score))

    def detect(self) -> LifecycleState:
        frame = self._frame_source()
        if not isinstance(frame, Frame):
            raise ScreenStateDetectionError("screen-state frame source returned a non-Frame value")
        if frame.source_device != self.device_id:
            raise ScreenStateDetectionError(
                f"screen frame source {frame.source_device!r} does not match template device {self.device_id!r}"
            )
        if self._last_frame_index is not None and frame.frame_index < self._last_frame_index:
            raise ScreenStateDetectionError("screen-state frames are out of order")
        self._last_frame_index = frame.frame_index
        if frame.payload is None:
            raise ScreenStateDetectionError("screen-state frame has no image payload")
        try:
            import cv2
            import numpy as np
        except ImportError as error:  # pragma: no cover - environment-specific
            raise ScreenStateDetectionError(
                "reviewed screen-state detection requires OpenCV and NumPy"
            ) from error
        encoded = np.frombuffer(frame.payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ScreenStateDetectionError("cannot decode screenshot payload for screen-state detection")
        scores = {
            state.value: max(self._score(image, template.image) for template in templates)
            for state, templates in self._templates.items()
        }
        self._last_scores = dict(sorted(scores.items()))
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_state, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        if best_score < self.score_threshold:
            raise ScreenStateDetectionError(
                f"no screen state reached score threshold: best={best_state}:{best_score:.4f}, "
                f"threshold={self.score_threshold:.4f}"
            )
        if best_score - second_score < self.margin_threshold:
            raise ScreenStateDetectionError(
                f"screen state is ambiguous: best={best_state}:{best_score:.4f}, "
                f"runner_up={ranked[1][0]}:{second_score:.4f}, "
                f"margin={best_score - second_score:.4f}, required={self.margin_threshold:.4f}"
            )
        return LifecycleState(best_state)


__all__ = [
    "SCREEN_TEMPLATE_SCHEMA_VERSION",
    "ScreenStateDetectionError",
    "TemplateLifecycleDetector",
]
