"""Versioned logical-to-screen calibration for physical lab devices.

The experiment planner only deals in the repository's ``(column, row)``
18x32 action grid.  This module is the sole place that turns a logical cell
or hand slot into pixels.  The normalized arena bounds intentionally mirror
``cr_bot.features.action_space.ACTION_GRID``; callers must not introduce a
second row origin or coordinate convention.

The current vision stack uses an affine crop calibration rather than a
general projective transform.  A calibration artifact records that contract
explicitly.  A future homography can be added as a new artifact version
without changing the experiment schema or silently reinterpreting old runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .schema import PhysicalLabError, canonical_hash


CALIBRATION_SCHEMA_VERSION = 1
GRID_COLS = 18
GRID_ROWS = 32

# These are the normalized coordinates from cr_bot.features.action_space's
# KATACR_GRID_XYXY / KATACR_BACKGROUND_SIZE.  Keeping the source values here
# avoids importing NumPy or the vision stack in the headless simulator.
ACTION_GRID_NORM_BOUNDS = (
    -0.9320463320463317 / 568.0,
    72.54622356495467 / 896.0,
    569.2610038610038 / 568.0,
    879.9748640483384 / 896.0,
)


class CalibrationError(PhysicalLabError):
    """Raised when a calibration artifact cannot safely map an action."""


def _number(value: object, field_name: str) -> float:
    if type(value) not in (int, float):
        raise CalibrationError(f"{field_name} must be numeric")
    import math

    if not math.isfinite(float(value)):
        raise CalibrationError(f"{field_name} must be finite")
    return float(value)


def _rect(value: object, field_name: str, *, positive: bool = True) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise CalibrationError(f"{field_name} must be [x, y, width, height]")
    result = tuple(_number(item, f"{field_name}[{index}]") for index, item in enumerate(value))
    if positive and (result[2] <= 0 or result[3] <= 0):
        raise CalibrationError(f"{field_name} width and height must be positive")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """A device-specific calibration with a stable coordinate contract."""

    calibration_id: str
    device_label: str
    screen_width_px: int
    screen_height_px: int
    arena_px: tuple[float, float, float, float]
    hand_px: tuple[float, float, float, float]
    device_serial_hash: str | None = None
    hand_slot_count: int = 4
    grid_norm_bounds: tuple[float, float, float, float] = ACTION_GRID_NORM_BOUNDS
    coordinate_system: str = "cr_bot_action_grid_v1"
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise CalibrationError(f"unsupported calibration schema: {self.schema_version}")
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise CalibrationError("calibration_id is required")
        if not isinstance(self.device_label, str) or not self.device_label.strip():
            raise CalibrationError("device_label is required")
        if self.device_serial_hash is not None:
            if not isinstance(self.device_serial_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.device_serial_hash):
                raise CalibrationError("device_serial_hash must be sha256:<64 lowercase hex characters>")
        if type(self.screen_width_px) is not int or self.screen_width_px <= 0:
            raise CalibrationError("screen_width_px must be positive")
        if type(self.screen_height_px) is not int or self.screen_height_px <= 0:
            raise CalibrationError("screen_height_px must be positive")
        object.__setattr__(self, "arena_px", _rect(self.arena_px, "arena_px"))
        object.__setattr__(self, "hand_px", _rect(self.hand_px, "hand_px"))
        if type(self.hand_slot_count) is not int or not 1 <= self.hand_slot_count <= 8:
            raise CalibrationError("hand_slot_count must be between 1 and 8")
        bounds = _rect(self.grid_norm_bounds, "grid_norm_bounds", positive=False)
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise CalibrationError("grid_norm_bounds must have increasing corners")
        object.__setattr__(self, "grid_norm_bounds", bounds)
        if self.coordinate_system != "cr_bot_action_grid_v1":
            raise CalibrationError(f"unsupported coordinate system: {self.coordinate_system!r}")

        ax, ay, aw, ah = self.arena_px
        if ax < 0 or ay < 0 or ax + aw > self.screen_width_px or ay + ah > self.screen_height_px:
            raise CalibrationError("arena_px must be inside the declared screen")

    @classmethod
    def for_screen(
        cls,
        *,
        device_label: str,
        device_serial_hash: str | None = None,
        screen_width_px: int,
        screen_height_px: int,
        arena_px: tuple[float, float, float, float] | None = None,
        hand_px: tuple[float, float, float, float] | None = None,
        hand_slot_count: int = 4,
        calibration_id: str | None = None,
    ) -> "CalibrationArtifact":
        """Build a clearly provisional calibration for fake/offline tests.

        Real devices should load a reviewed artifact produced by the grid
        calibration tool.  The full-screen default is useful only for the
        software harness and is intentionally labeled ``offline`` when used
        by the CLI.
        """

        if arena_px is None:
            arena_px = (0.0, 0.0, float(screen_width_px), float(screen_height_px))
        if hand_px is None:
            hand_px = (
                0.0,
                float(screen_height_px) * 0.80,
                float(screen_width_px),
                float(screen_height_px) * 0.20,
            )
        calibration_id = calibration_id or f"{device_label}-offline-calibration-v1"
        return cls(
            calibration_id=calibration_id,
            device_label=device_label,
            device_serial_hash=device_serial_hash,
            screen_width_px=screen_width_px,
            screen_height_px=screen_height_px,
            arena_px=arena_px,
            hand_px=hand_px,
            hand_slot_count=hand_slot_count,
        )

    def to_dict(self, *, include_hash: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "device_label": self.device_label,
            "screen_width_px": self.screen_width_px,
            "screen_height_px": self.screen_height_px,
            "arena_px": list(self.arena_px),
            "hand_px": list(self.hand_px),
            "hand_slot_count": self.hand_slot_count,
            "grid_norm_bounds": list(self.grid_norm_bounds),
            "coordinate_system": self.coordinate_system,
        }
        if self.device_serial_hash is not None:
            result["device_serial_hash"] = self.device_serial_hash
        if include_hash:
            result["calibration_hash"] = self.calibration_hash()
        return result

    def calibration_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(include_hash=True), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CalibrationArtifact":
        if not isinstance(raw, Mapping):
            raise CalibrationError("calibration document must be an object")
        allowed = {
            "schema_version",
            "calibration_id",
            "device_label",
            "device_serial_hash",
            "screen_width_px",
            "screen_height_px",
            "arena_px",
            "hand_px",
            "hand_slot_count",
            "grid_norm_bounds",
            "coordinate_system",
            "calibration_hash",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise CalibrationError(f"unknown calibration fields: {unknown}")
        artifact = cls(
            schema_version=raw.get("schema_version", CALIBRATION_SCHEMA_VERSION),
            calibration_id=raw.get("calibration_id"),
            device_label=raw.get("device_label"),
            device_serial_hash=raw.get("device_serial_hash"),
            screen_width_px=raw.get("screen_width_px"),
            screen_height_px=raw.get("screen_height_px"),
            arena_px=raw.get("arena_px"),
            hand_px=raw.get("hand_px"),
            hand_slot_count=raw.get("hand_slot_count", 4),
            grid_norm_bounds=raw.get("grid_norm_bounds", ACTION_GRID_NORM_BOUNDS),
            coordinate_system=raw.get("coordinate_system", "cr_bot_action_grid_v1"),
        )
        declared = raw.get("calibration_hash")
        if declared is not None and declared != artifact.calibration_hash():
            raise CalibrationError(
                f"calibration_hash mismatch: declared={declared!r}, actual={artifact.calibration_hash()!r}"
            )
        return artifact

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationArtifact":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError(f"cannot load calibration {source}: {error}") from error
        return cls.from_dict(raw)

    def _cell_to_norm_center(self, col: int, row: int) -> tuple[float, float]:
        if type(col) is not int or type(row) is not int or not (0 <= col < GRID_COLS and 0 <= row < GRID_ROWS):
            raise CalibrationError(f"cell out of bounds: {(col, row)!r}")
        x0, y0, x1, y1 = self.grid_norm_bounds
        return (
            x0 + (col + 0.5) / GRID_COLS * (x1 - x0),
            y0 + (row + 0.5) / GRID_ROWS * (y1 - y0),
        )

    def cell_to_pixel(self, cell: tuple[int, int]) -> tuple[int, int]:
        """Return the screen point for a logical action-grid cell center."""

        if not isinstance(cell, tuple) or len(cell) != 2:
            raise CalibrationError("cell must be a (column, row) tuple")
        x_norm, y_norm = self._cell_to_norm_center(cell[0], cell[1])
        ax, ay, aw, ah = self.arena_px
        x = int(round(ax + x_norm * aw))
        y = int(round(ay + y_norm * ah))
        if not (0 <= x < self.screen_width_px and 0 <= y < self.screen_height_px):
            raise CalibrationError("calibrated cell falls outside the screen")
        return x, y

    def pixel_to_cell(self, point: tuple[float, float]) -> tuple[int, int] | None:
        """Inverse mapping used by future capture extractors and calibration tests."""

        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise CalibrationError("point must contain x and y")
        x, y = _number(point[0], "point.x"), _number(point[1], "point.y")
        ax, ay, aw, ah = self.arena_px
        x_norm, y_norm = (x - ax) / aw, (y - ay) / ah
        x0, y0, x1, y1 = self.grid_norm_bounds
        if not (x0 <= x_norm < x1 and y0 <= y_norm < y1):
            return None
        return (
            min(GRID_COLS - 1, max(0, int((x_norm - x0) / (x1 - x0) * GRID_COLS))),
            min(GRID_ROWS - 1, max(0, int((y_norm - y0) / (y1 - y0) * GRID_ROWS))),
        )

    def slot_to_pixel(self, slot: int) -> tuple[int, int]:
        """Return the center of a hand slot without exposing it to planners."""

        if type(slot) is not int or not 0 <= slot < self.hand_slot_count:
            raise CalibrationError(f"card slot out of bounds: {slot!r}")
        hx, hy, hw, hh = self.hand_px
        return (
            int(round(hx + (slot + 0.5) / self.hand_slot_count * hw)),
            int(round(hy + hh * 0.5)),
        )


__all__ = [
    "ACTION_GRID_NORM_BOUNDS",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationArtifact",
    "CalibrationError",
    "GRID_COLS",
    "GRID_ROWS",
]
