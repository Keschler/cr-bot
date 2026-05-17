from dataclasses import dataclass

import cv2
import numpy as np

from card_metadata import CARD_METADATA
from features.action_space import ACTION_GRID


@dataclass(frozen=True)
class SpellDeployCandidate:
    center_x: float
    center_y: float
    radius_px: float | None = None
    radius_x_px: float | None = None
    radius_y_px: float | None = None
    confidence: float = 0.0
    elixir_cost_confirmed: bool = False
    arc_score: float = 0.0
    radius_score: float = 0.0
    purple_score: float = 0.0


class SpellDeployLocator:
    """Locate a pending spell's deploy center from cast UI elements.

    The intended flow is:
    1. A hand-slot HUD change creates a pending spell action.
    2. The white spell radius is detected and provides the deploy center.
    3. The purple elixir-used display confirms the correct candidate when
       several deploy effects are visible at once.

    The white radius is projected as an ellipse because arena cells are not
    square in screen space.
    """

    def locate(self, frame, arena_px, card_name: str, elixir_cost: int | float | None):
        candidates, _ = self.detect(frame, arena_px, card_name, elixir_cost)
        if not candidates:
            return None
        return candidates[0]

    def detect(
        self,
        frame,
        arena_px,
        card_name: str,
        elixir_cost: int | float | None,
    ):
        if frame is None or arena_px is None:
            return [], {}

        expected_radii_px = self._expected_radii_px(arena_px, card_name)
        masks = self._build_masks(frame, arena_px)
        candidates = self._detect_fixed_radius_candidates(
            frame,
            arena_px,
            expected_radii_px,
            masks,
        )
        if not candidates:
            return [], masks

        return self._score_candidates(candidates, masks, arena_px, elixir_cost), masks

    def _score_candidates(self, candidates, masks, arena_px, elixir_cost):
        scored = []
        for candidate in candidates:
            purple_score = 0.0
            confirmed = False
            if elixir_cost is not None:
                purple_score = self._purple_elixir_score(
                    masks["purple_mask"],
                    arena_px,
                    candidate,
                    elixir_cost,
                )
                confirmed = purple_score >= 0.35

            confidence = (
                0.70 * candidate.arc_score
                + 0.25 * candidate.radius_score
                + 0.05 * purple_score
            )
            scored.append(SpellDeployCandidate(
                center_x=candidate.center_x,
                center_y=candidate.center_y,
                radius_px=candidate.radius_px,
                radius_x_px=candidate.radius_x_px,
                radius_y_px=candidate.radius_y_px,
                confidence=confidence,
                elixir_cost_confirmed=confirmed,
                arc_score=candidate.arc_score,
                radius_score=candidate.radius_score,
                purple_score=purple_score,
            ))

        scored.sort(key=lambda candidate: candidate.confidence, reverse=True)
        return [
            candidate
            for candidate in scored
            if candidate.confidence >= 0.35 and candidate.arc_score >= 0.12
        ]

    def _detect_fixed_radius_candidates(
        self,
        frame,
        arena_px,
        expected_radii_px: tuple[float, float] | None,
        masks,
    ):
        if expected_radii_px is None:
            return []

        arena_x, arena_y, arena_w, arena_h = self._arena_rect(arena_px, frame.shape)
        edges = masks["edges"]
        radius_x = int(round(expected_radii_px[0]))
        radius_y = int(round(expected_radii_px[1]))
        grid_x0 = max(0, int(round(ACTION_GRID.x0 * arena_w)))
        grid_y0 = max(0, int(round(ACTION_GRID.y0 * arena_h)))
        grid_x1 = min(arena_w - 1, int(round(ACTION_GRID.x1 * arena_w)))
        grid_y1 = min(arena_h - 1, int(round(ACTION_GRID.y1 * arena_h)))

        coarse_step = 6
        coarse = []
        for cy in range(grid_y0, grid_y1 + 1, coarse_step):
            for cx in range(grid_x0, grid_x1 + 1, coarse_step):
                arc_score = self._ellipse_arc_score(edges, cx, cy, radius_x, radius_y)
                if arc_score >= 0.14:
                    coarse.append((arc_score, cx, cy))

        coarse.sort(reverse=True)
        refined = []
        dedupe_distance = max(18, min(radius_x, radius_y) * 0.35)
        for _, cx, cy in self._dedupe_centers(coarse, min_distance=dedupe_distance, limit=12):
            best = self._refine_fixed_radius_center(edges, cx, cy, radius_x, radius_y, coarse_step)
            refined.append(best)

        refined.sort(reverse=True)
        candidates = []
        for arc_score, cx, cy in self._dedupe_centers(refined, min_distance=dedupe_distance, limit=8):
            frame_x = float(arena_x + cx)
            frame_y = float(arena_y + cy)
            if ACTION_GRID.pixel_to_cell(frame_x, frame_y, arena_px) is None:
                continue
            candidates.append(SpellDeployCandidate(
                center_x=frame_x,
                center_y=frame_y,
                radius_px=float((radius_x + radius_y) / 2.0),
                radius_x_px=float(radius_x),
                radius_y_px=float(radius_y),
                confidence=0.0,
                arc_score=arc_score,
                radius_score=1.0,
            ))
        return candidates

    def _refine_fixed_radius_center(self, edges, cx, cy, radius_x, radius_y, coarse_step):
        best_score = -1.0
        best = (cx, cy)
        for step in (2, 1):
            search = range(-coarse_step, coarse_step + 1, step)
            for dy in search:
                for dx in search:
                    tx = best[0] + dx
                    ty = best[1] + dy
                    if not (0 <= tx < edges.shape[1] and 0 <= ty < edges.shape[0]):
                        continue
                    score = self._ellipse_arc_score(edges, tx, ty, radius_x, radius_y)
                    if score > best_score:
                        best_score = score
                        best = (tx, ty)
        return best_score, best[0], best[1]

    def _dedupe_centers(self, scored_centers, min_distance, limit):
        kept = []
        for score, cx, cy in scored_centers:
            if any(np.hypot(cx - kx, cy - ky) < min_distance for _, kx, ky in kept):
                continue
            kept.append((score, cx, cy))
            if len(kept) >= limit:
                break
        return kept

    def _confirm_with_elixir_used_display(
        self,
        frame,
        arena_px,
        candidates: list[SpellDeployCandidate],
        elixir_cost: int | float,
    ):
        """Pick the candidate confirmed by the purple elixir-used display.

        TODO: Detect/template-match the purple elixir cost and associate it
        with the closest radius candidate.
        """
        return None

    def _expected_radii_px(self, arena_px, card_name: str):
        radius_tiles = CARD_METADATA.get(card_name, {}).get("radius")
        if radius_tiles is None:
            return None

        _, _, arena_w, arena_h = arena_px
        cell_w = ACTION_GRID.width * arena_w / ACTION_GRID.cols
        cell_h = ACTION_GRID.height * arena_h / ACTION_GRID.rows
        return float(radius_tiles) * cell_w, float(radius_tiles) * cell_h

    def _build_masks(self, frame, arena_px):
        arena_x, arena_y, arena_w, arena_h = self._arena_rect(arena_px, frame.shape)
        roi = frame[arena_y:arena_y + arena_h, arena_x:arena_x + arena_w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 72, 255]))
        bright = cv2.inRange(v, 170, 255)
        white_mask = cv2.bitwise_and(white_mask, bright)
        kernel3 = np.ones((3, 3), np.uint8)
        white_mask = cv2.medianBlur(white_mask, 3)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel3)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel3)
        edges = cv2.Canny(white_mask, 40, 120)

        purple_mask = cv2.inRange(hsv, np.array([118, 55, 70]), np.array([168, 255, 255]))
        purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel3)

        return {
            "roi": roi,
            "white_mask": white_mask,
            "edges": edges,
            "purple_mask": purple_mask,
        }

    def _purple_elixir_score(self, purple_mask, arena_px, candidate, elixir_cost):
        arena_x, arena_y = int(round(arena_px[0])), int(round(arena_px[1]))
        arena_h, arena_w = purple_mask.shape[:2]
        cx = int(round(candidate.center_x - arena_x))
        cy = int(round(candidate.center_y - arena_y))
        radius = int(round(candidate.radius_px or 0))

        search_radius_x = max(35, int(radius * 0.75))
        y0 = max(0, cy - max(40, int(radius * 0.90)))
        y1 = min(arena_h, cy + max(35, int(radius * 0.45)))
        x0 = max(0, cx - search_radius_x)
        x1 = min(arena_w, cx + search_radius_x)
        if x1 <= x0 or y1 <= y0:
            return 0.0

        roi = purple_mask[y0:y1, x0:x1]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = 0.0
        expected_digit = str(int(elixir_cost))
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 12 <= area <= 1800:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h < 8 or w < 5:
                continue
            aspect = w / max(1, h)
            if not 0.25 <= aspect <= 1.8:
                continue

            patch = roi[max(0, y - 3):min(roi.shape[0], y + h + 3), max(0, x - 3):min(roi.shape[1], x + w + 3)]
            template_score = self._digit_template_score(patch, expected_digit)
            area_score = min(1.0, area / 140.0)
            contour_cx = x0 + x + w / 2.0
            contour_cy = y0 + y + h / 2.0
            distance = np.hypot((contour_cx - cx) / max(1, search_radius_x), (contour_cy - cy) / max(1, y1 - y0))
            proximity = max(0.0, 1.0 - distance)
            best = max(best, 0.45 * template_score + 0.30 * area_score + 0.25 * proximity)

        return best

    def _digit_template_score(self, patch, digit: str):
        if patch.size == 0:
            return 0.0
        patch = cv2.resize(patch, (24, 32), interpolation=cv2.INTER_AREA)
        _, patch = cv2.threshold(patch, 1, 255, cv2.THRESH_BINARY)
        template = np.zeros((32, 24), dtype=np.uint8)
        cv2.putText(
            template,
            digit,
            (3, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            255,
            2,
            cv2.LINE_AA,
        )
        _, template = cv2.threshold(template, 20, 255, cv2.THRESH_BINARY)
        result = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
        score = float(result[0, 0])
        if np.isnan(score):
            return 0.0
        return max(0.0, score)

    def _ellipse_arc_score(self, edges, cx, cy, radius_x, radius_y):
        samples = 96
        hits = 0
        for angle in np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False):
            x = int(round(cx + np.cos(angle) * radius_x))
            y = int(round(cy + np.sin(angle) * radius_y))
            if x < 1 or y < 1 or x >= edges.shape[1] - 1 or y >= edges.shape[0] - 1:
                continue
            if edges[y - 1:y + 2, x - 1:x + 2].any():
                hits += 1
        return hits / samples

    def render_debug(
        self,
        frame,
        arena_px,
        card_name: str,
        elixir_cost: int | float | None,
    ):
        candidates, masks = self.detect(frame, arena_px, card_name, elixir_cost)
        arena_x, arena_y, arena_w, arena_h = self._arena_rect(arena_px, frame.shape)
        overlay = frame.copy()
        cv2.rectangle(overlay, (arena_x, arena_y), (arena_x + arena_w, arena_y + arena_h), (255, 255, 0), 2)

        for idx, candidate in enumerate(candidates):
            center = (int(round(candidate.center_x)), int(round(candidate.center_y)))
            radius_x = int(round(candidate.radius_x_px or candidate.radius_px or 0))
            radius_y = int(round(candidate.radius_y_px or candidate.radius_px or 0))
            color = (0, 255, 0) if idx == 0 else (0, 180, 255)
            cv2.ellipse(overlay, center, (radius_x, radius_y), 0, 0, 360, color, 2)
            cv2.drawMarker(overlay, center, color, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                overlay,
                f"{idx} {candidate.confidence:.2f} a{candidate.arc_score:.2f} p{candidate.purple_score:.2f}",
                (center[0] - 50, max(20, center[1] - radius_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        debug = {"overlay": overlay}
        for key in ("white_mask", "edges", "purple_mask"):
            if key in masks:
                debug[key] = cv2.cvtColor(masks[key], cv2.COLOR_GRAY2BGR)
        if "roi" in masks:
            debug["roi"] = masks["roi"]
        return debug, candidates

    def _arena_rect(self, arena_px, frame_shape):
        ax, ay, aw, ah = arena_px
        frame_h, frame_w = frame_shape[:2]
        x = max(0, min(frame_w - 1, int(round(ax))))
        y = max(0, min(frame_h - 1, int(round(ay))))
        w = max(1, min(frame_w - x, int(round(aw))))
        h = max(1, min(frame_h - y, int(round(ah))))
        return x, y, w, h
