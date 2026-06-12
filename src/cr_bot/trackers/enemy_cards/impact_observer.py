from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cr_bot.features.action_space import ACTION_GRID, GRID_SIZE

from .models import ProjectileTrajectoryConfig, RecentSpellTargetObservation


@dataclass(frozen=True, slots=True)
class ArenaFrameSample:
    time_left_s: float
    arena_bgr: np.ndarray


@dataclass(frozen=True, slots=True)
class QuadraticTrajectoryModel:
    coefficients: np.ndarray
    first_x: float
    first_y: float
    last_x: float
    last_y: float
    predicted_y: float
    last_row: int

    def predict_x(self, y_norm: float) -> float:
        return float(np.polyval(self.coefficients, y_norm))


@dataclass(frozen=True, slots=True)
class ImpactObservationDebug:
    candidate_count: int
    best_cell: tuple[int, int] | None
    best_score: float | None
    emitted: bool
    reject_reason: str | None


def build_quadratic_trajectory_model(event, time_left_s: float, arena_px) -> QuadraticTrajectoryModel | None:
    if not event.observed_centers:
        return None

    arena_x, arena_y, arena_w, arena_h = arena_px
    points = [
        (sample_time, (x - arena_x) / arena_w, (y - arena_y) / arena_h)
        for sample_time, x, y in event.observed_centers
    ]
    points.sort(key=lambda item: item[0], reverse=True)

    x_values = np.array([item[1] for item in points], dtype=np.float64)
    y_values = np.array([item[2] for item in points], dtype=np.float64)
    unique_y = np.unique(np.round(y_values, 4))
    if unique_y.size >= 3:
        coefficients = np.polyfit(y_values, x_values, deg=2)
    elif unique_y.size >= 2:
        linear = np.polyfit(y_values, x_values, deg=1)
        coefficients = np.array([0.0, linear[0], linear[1]], dtype=np.float64)
    else:
        coefficients = np.array([0.0, 0.0, x_values[-1]], dtype=np.float64)

    last_time, last_x, last_y = points[-1]
    if len(points) >= 2:
        prev_time, _, prev_y = points[-2]
        elapsed_s = max(0.05, prev_time - last_time)
        vertical_speed = max(0.01, (last_y - prev_y) / elapsed_s)
    else:
        vertical_speed = 0.22

    predicted_y = last_y + vertical_speed * max(0.0, last_time - time_left_s)
    predicted_y = min(0.98, max(last_y, predicted_y))
    last_cell = ACTION_GRID.pixel_to_cell(
        last_x * arena_w + arena_x,
        last_y * arena_h + arena_y,
        arena_px,
    )
    return QuadraticTrajectoryModel(
        coefficients=coefficients,
        first_x=points[0][1],
        first_y=points[0][2],
        last_x=last_x,
        last_y=last_y,
        predicted_y=predicted_y,
        last_row=last_cell[1] if last_cell is not None else 0,
    )


class EnemyProjectileImpactObserver:
    _SUPPORTED_CARDS = frozenset({"fireball", "rocket"})
    _MIN_IMPACT_SCORE = {
        "fireball": 0.18,
        "rocket": 0.18,
    }

    def supports(self, card: str) -> bool:
        return card in self._SUPPORTED_CARDS

    def observe_event_impact(
        self,
        *,
        card: str,
        event,
        current_sample: ArenaFrameSample,
        previous_sample: ArenaFrameSample | None,
        arena_px,
        config: ProjectileTrajectoryConfig,
    ) -> RecentSpellTargetObservation | None:
        observation, _ = self.inspect_event_impact(
            card=card,
            event=event,
            current_sample=current_sample,
            previous_sample=previous_sample,
            arena_px=arena_px,
            config=config,
        )
        return observation

    def inspect_event_impact(
        self,
        *,
        card: str,
        event,
        current_sample: ArenaFrameSample,
        previous_sample: ArenaFrameSample | None,
        arena_px,
        config: ProjectileTrajectoryConfig,
    ) -> tuple[RecentSpellTargetObservation | None, ImpactObservationDebug]:
        if not self.supports(card):
            return None, ImpactObservationDebug(0, None, None, False, "unsupported-card")

        model = build_quadratic_trajectory_model(event, current_sample.time_left_s, arena_px)
        if model is None:
            return None, ImpactObservationDebug(0, None, None, False, "no-trajectory-model")

        frame = current_sample.arena_bgr
        previous_frame = previous_sample.arena_bgr if previous_sample is not None else None
        local_arena = (0.0, 0.0, float(frame.shape[1]), float(frame.shape[0]))
        best: tuple[float, tuple[int, int]] | None = None
        candidate_count = 0

        for cell, lateral_error, row_penalty in self._candidate_cells(model, config):
            candidate_count += 1
            score = self._impact_score(
                frame=frame,
                previous_frame=previous_frame,
                cell=cell,
                arena_px=local_arena,
                lateral_error=lateral_error,
                row_penalty=row_penalty,
            )
            if best is None or score > best[0]:
                best = (score, cell)

        if best is None:
            return None, ImpactObservationDebug(candidate_count, None, None, False, "no-candidates")
        if best[0] < self._MIN_IMPACT_SCORE.get(card, 1.0):
            return None, ImpactObservationDebug(candidate_count, best[1], best[0], False, "below-score-threshold")

        score, cell = best
        if card == "rocket":
            _, row = cell
            y_norm = ACTION_GRID.cell_to_norm_center(0, row)[1]
            arena_x, arena_y, arena_w, arena_h = arena_px
            points = [
                ((x - arena_x) / arena_w, (y - arena_y) / arena_h)
                for _, x, y in event.observed_centers
            ]
            if len(points) >= 2:
                x_values = np.array([point[0] for point in points], dtype=np.float64)
                y_values = np.array([point[1] for point in points], dtype=np.float64)
                slope, intercept = np.polyfit(y_values, x_values, deg=1)
                predicted_x = float(slope * y_norm + intercept)
            else:
                predicted_x = model.predict_x(y_norm)
            predicted_col = max(
                0,
                min(GRID_SIZE[0] - 1, int(round(predicted_x * GRID_SIZE[0] - 0.5))),
            )
            cell = (predicted_col, row)
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(*cell, arena_px)
        return (
            RecentSpellTargetObservation(
                card=card,
                time_left_s=current_sample.time_left_s,
                cell=cell,
                phase="impact",
                quality=score,
                center_x=center_x,
                center_y=center_y,
                key=f"{card}:impact:{current_sample.time_left_s:.2f}:{cell[0]}:{cell[1]}",
            ),
            ImpactObservationDebug(candidate_count, cell, score, True, None),
        )

    def _candidate_cells(self, model: QuadraticTrajectoryModel, config: ProjectileTrajectoryConfig):
        cols, rows = GRID_SIZE
        predicted_row = int(round(model.predicted_y * rows - 0.5))
        row_start = max(model.last_row, predicted_row - 2)
        row_stop = min(rows - 1, predicted_row + 4)
        for row in range(row_start, row_stop + 1):
            y_norm = ACTION_GRID.cell_to_norm_center(0, row)[1]
            predicted_x = model.predict_x(y_norm)
            predicted_col = int(round(predicted_x * cols - 0.5))
            for col in range(max(0, predicted_col - 2), min(cols - 1, predicted_col + 2) + 1):
                cand_x, _ = ACTION_GRID.cell_to_norm_center(col, row)
                lateral_error = abs(cand_x - predicted_x)
                if lateral_error > config.corridor_width_norm * 1.2:
                    continue
                row_penalty = abs(y_norm - model.predicted_y)
                yield (col, row), lateral_error, row_penalty

    def _impact_score(
        self,
        *,
        frame,
        previous_frame,
        cell,
        arena_px,
        lateral_error: float,
        row_penalty: float,
    ) -> float:
        patch = self._cell_patch(frame, cell, arena_px)
        if patch.size == 0:
            return 0.0
        previous_patch = (
            self._cell_patch(previous_frame, cell, arena_px)
            if previous_frame is not None
            else None
        )

        patch_f = patch.astype(np.float32)
        b = patch_f[..., 0]
        g = patch_f[..., 1]
        r = patch_f[..., 2]
        brightness = float(((r + g + b) / (3.0 * 255.0)).mean())
        hot_ratio = float(((r > 150.0) & (g > 75.0) & (r > g) & (g > b)).mean())
        flare_ratio = float(((r > 120.0) & (b > 110.0)).mean())

        positive_delta = 0.0
        if previous_patch is not None and previous_patch.shape == patch.shape:
            prev_gray = previous_patch.astype(np.float32).mean(axis=2)
            curr_gray = patch_f.mean(axis=2)
            positive_delta = float(np.maximum(curr_gray - prev_gray, 0.0).mean() / 255.0)

        return (
            hot_ratio * 2.3
            + flare_ratio * 0.7
            + brightness * 0.25
            + positive_delta * 2.2
            - lateral_error * 3.5
            - row_penalty * 0.8
        )

    def _cell_patch(self, frame, cell, arena_px):
        if frame is None:
            return np.empty((0, 0, 3), dtype=np.uint8)
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(*cell, arena_px)
        cell_w = arena_px[2] * ACTION_GRID.width / GRID_SIZE[0]
        cell_h = arena_px[3] * ACTION_GRID.height / GRID_SIZE[1]
        radius_x = max(6, int(round(cell_w * 0.9)))
        radius_y = max(6, int(round(cell_h * 0.9)))
        x0 = max(0, int(round(center_x - radius_x)))
        y0 = max(0, int(round(center_y - radius_y)))
        x1 = min(frame.shape[1], int(round(center_x + radius_x)))
        y1 = min(frame.shape[0], int(round(center_y + radius_y)))
        return frame[y0:y1, x0:x1]
