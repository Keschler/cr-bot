from dataclasses import dataclass

import cv2
import numpy as np

from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.features.action_space import ACTION_GRID


@dataclass(frozen=True)
class SpellDeployCandidate:
    center_x: float
    center_y: float
    radius_px: float | None = None
    radius_x_px: float | None = None
    radius_y_px: float | None = None
    confidence: float = 0.0
    arc_score: float = 0.0
    radius_score: float = 0.0


class SpellDeployLocator:
    """Locate a pending spell's deploy center from cast UI elements.

    The intended flow is:
    1. A hand-slot HUD change creates a pending spell action.
    2. The white spell radius is detected and provides the deploy center.

    The white radius is projected as an ellipse because arena cells are not
    square in screen space.
    """

    def locate(self, frame, arena_px, card_name: str, elixir_cost: int | float | None):
        candidates, _ = self.detect(frame, arena_px, card_name, elixir_cost)
        if not candidates:
            return None
        return candidates[0]

    def locate_released(self, frame, arena_px, card_name: str, elixir_cost: int | float | None):
        """Return the aimed spell ellipse only when the purple release marker is present."""
        candidates, masks = self.detect(frame, arena_px, card_name, elixir_cost)
        if not candidates:
            return None

        purple_mask = masks.get("purple_mask")
        if purple_mask is None:
            return None

        best = None
        for candidate in candidates[:5]:
            purple_score = self._purple_release_score(purple_mask, arena_px, candidate)
            if purple_score < 0.18:
                continue
            score = (purple_score, candidate.confidence)
            if best is None or score > best[0]:
                best = (score, candidate)

        if best is None:
            return None
        return best[1]

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

        return self._score_candidates(candidates), masks

    def _score_candidates(self, candidates):
        scored = []
        for candidate in candidates:
            confidence = (
                0.70 * candidate.arc_score
                + 0.25 * candidate.radius_score
            )
            scored.append(SpellDeployCandidate(
                center_x=candidate.center_x,
                center_y=candidate.center_y,
                radius_px=candidate.radius_px,
                radius_x_px=candidate.radius_x_px,
                radius_y_px=candidate.radius_y_px,
                confidence=confidence,
                arc_score=candidate.arc_score,
                radius_score=candidate.radius_score,
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
        """Find candidate spell centers from a fast ellipse-template response map."""
        if expected_radii_px is None:
            return []

        arena_x, arena_y, arena_w, arena_h = self._arena_rect(arena_px, frame.shape)
        edge_hits = masks["edge_hits"]
        radius_x = int(round(expected_radii_px[0]))
        radius_y = int(round(expected_radii_px[1]))

        # Restrict the search to the playable area
        grid_x0 = max(0, int(round(ACTION_GRID.x0 * arena_w)))
        grid_y0 = max(0, int(round(ACTION_GRID.y0 * arena_h)))
        grid_x1 = min(arena_w - 1, int(round(ACTION_GRID.x1 * arena_w)))
        grid_y1 = min(arena_h - 1, int(round(ACTION_GRID.y1 * arena_h)))

        response, template_center, scale = self._ellipse_match_response(
            edge_hits,
            radius_x,
            radius_y,
            grid_x0,
            grid_y0,
            grid_x1,
            grid_y1,
        )
        if response is None:
            return []

        coarse = self._response_candidates(response, template_center, grid_x0, grid_y0, scale)
        if not coarse:
            return []

        dedupe_distance = max(18, min(radius_x, radius_y) * 0.35)
        refined = []
        for _, cx, cy in self._dedupe_centers(coarse, min_distance=dedupe_distance, limit=12):
            best = self._refine_fixed_radius_center(
                edge_hits,
                cx,
                cy,
                radius_x,
                radius_y,
                coarse_step=4,
            )
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

    def _refine_fixed_radius_center(self, edge_hits, cx, cy, radius_x, radius_y, coarse_step):
        """Refine one proposal with a tiny local search around the coarse center."""
        best_score = -1.0
        best = (cx, cy)
        for step in (2, 1):
            search = range(-coarse_step, coarse_step + 1, step)
            for dy in search:
                for dx in search:
                    tx = best[0] + dx
                    ty = best[1] + dy
                    if not (0 <= tx < edge_hits.shape[1] and 0 <= ty < edge_hits.shape[0]):
                        continue
                    score = self._ellipse_arc_score(edge_hits, tx, ty, radius_x, radius_y)
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

    def _expected_radii_px(self, arena_px, card_name: str):
        radius_tiles = CARD_METADATA.get(card_name, {}).get("radius")
        if radius_tiles is None:
            return None

        _, _, arena_w, arena_h = arena_px
        cell_w = ACTION_GRID.width * arena_w / ACTION_GRID.cols
        cell_h = ACTION_GRID.height * arena_h / ACTION_GRID.rows
        return float(radius_tiles) * cell_w, float(radius_tiles) * cell_h

    def _build_masks(self, frame, arena_px):
        """Build the thresholded arena masks once so later stages stay in OpenCV/NumPy."""
        arena_x, arena_y, arena_w, arena_h = self._arena_rect(arena_px, frame.shape)
        roi = frame[arena_y:arena_y + arena_h, arena_x:arena_x + arena_w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)

        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 72, 255]))
        bright = cv2.inRange(v, 170, 255)
        white_mask = cv2.bitwise_and(white_mask, bright)
        kernel3 = np.ones((3, 3), np.uint8)
        white_mask = cv2.medianBlur(white_mask, 3)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel3)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel3)
        edges = cv2.Canny(white_mask, 40, 120)
        edge_hits = cv2.dilate(edges, kernel3)
        purple_mask = cv2.inRange(hsv, np.array([118, 55, 70]), np.array([168, 255, 255]))
        purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel3)

        return {
            "roi": roi,
            "white_mask": white_mask,
            "edges": edges,
            "edge_hits": edge_hits,
            "purple_mask": purple_mask,
        }

    def _ellipse_match_response(self, edge_hits, radius_x, radius_y, grid_x0, grid_y0, grid_x1, grid_y1):
        """Score the whole search region in one OpenCV call using an ellipse perimeter template."""
        template, template_center = self._ellipse_template(radius_x, radius_y)
        search = edge_hits[grid_y0:grid_y1 + 1, grid_x0:grid_x1 + 1]
        if (
            search.shape[0] < template.shape[0]
            or search.shape[1] < template.shape[1]
        ):
            return None, template_center, 1.0

        scale = 0.5
        if min(search.shape[:2]) < 160 or min(template.shape[:2]) < 24:
            scale = 1.0

        if scale != 1.0:
            search = cv2.resize(
                search,
                (max(1, int(round(search.shape[1] * scale))), max(1, int(round(search.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            template = cv2.resize(
                template,
                (max(1, int(round(template.shape[1] * scale))), max(1, int(round(template.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            _, template = cv2.threshold(template, 1, 255, cv2.THRESH_BINARY)
            template_center = (
                int(round(template_center[0] * scale)),
                int(round(template_center[1] * scale)),
            )

        response = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED)
        return response, template_center, scale

    def _response_candidates(self, response, template_center, grid_x0, grid_y0, scale):
        """Extract a small set of local maxima from the template response map."""
        if response.size == 0:
            return []

        peak_threshold = 0.18
        maxima = cv2.dilate(response, np.ones((9, 9), np.float32))
        ys, xs = np.where((response >= peak_threshold) & (response >= maxima - 1e-6))
        if ys.size == 0:
            flat_idx = np.argpartition(response.ravel(), -8)[-8:]
            ys, xs = np.unravel_index(flat_idx, response.shape)

        scored = []
        center_x, center_y = template_center
        inv_scale = 1.0 / scale
        for y, x in zip(ys.tolist(), xs.tolist()):
            score = float(response[y, x])
            scored.append((
                score,
                int(round(grid_x0 + (x + center_x) * inv_scale)),
                int(round(grid_y0 + (y + center_y) * inv_scale)),
            ))
        scored.sort(reverse=True)
        return scored

    def _ellipse_template(self, radius_x, radius_y):
        """Create a cached binary perimeter template for the spell-radius ellipse."""
        key = (radius_x, radius_y)
        cache = getattr(self, "_ellipse_template_cache", None)
        if cache is None:
            cache = {}
            self._ellipse_template_cache = cache
        if key in cache:
            return cache[key]

        pad = 3
        width = radius_x * 2 + pad * 2 + 1
        height = radius_y * 2 + pad * 2 + 1
        center = (radius_x + pad, radius_y + pad)
        template = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(template, center, (radius_x, radius_y), 0, 0, 360, 255, 2)
        template = cv2.dilate(template, np.ones((3, 3), np.uint8))
        result = (template, center)
        cache[key] = result
        return result

    def _ellipse_sample_offsets(self, radius_x, radius_y):
        """Precompute integer ellipse perimeter offsets so arc scoring is vectorized."""
        key = (radius_x, radius_y)
        cache = getattr(self, "_ellipse_offset_cache", None)
        if cache is None:
            cache = {}
            self._ellipse_offset_cache = cache
        if key in cache:
            return cache[key]

        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        dx = np.rint(np.cos(angles) * radius_x).astype(np.int16)
        dy = np.rint(np.sin(angles) * radius_y).astype(np.int16)
        offsets = np.stack((dx, dy), axis=1)
        cache[key] = offsets
        return offsets

    def _ellipse_arc_score(self, edge_hits, cx, cy, radius_x, radius_y):
        """Measure how much of the expected ellipse perimeter lands on detected edge pixels."""
        offsets = self._ellipse_sample_offsets(radius_x, radius_y)
        xs = cx + offsets[:, 0]
        ys = cy + offsets[:, 1]
        valid = (
            (xs >= 0)
            & (ys >= 0)
            & (xs < edge_hits.shape[1])
            & (ys < edge_hits.shape[0])
        )
        if not np.any(valid):
            return 0.0
        hits = edge_hits[ys[valid], xs[valid]] > 0
        return float(np.count_nonzero(hits) / offsets.shape[0])

    def _purple_release_score(self, purple_mask, arena_px, candidate):
        """Score purple blobs above the ellipse where the release elixir marker appears."""
        arena_x = int(round(arena_px[0]))
        arena_y = int(round(arena_px[1]))
        cx = float(candidate.center_x - arena_x)
        cy = float(candidate.center_y - arena_y)
        radius_x = float(candidate.radius_x_px or candidate.radius_px or 0.0)
        radius_y = float(candidate.radius_y_px or candidate.radius_px or 0.0)

        x0 = max(0, int(round(cx - radius_x * 0.58)))
        x1 = min(purple_mask.shape[1], int(round(cx + radius_x * 0.58)))
        y0 = max(0, int(round(cy - radius_y * 1.35)))
        y1 = min(purple_mask.shape[0], int(round(cy - radius_y * 0.02)))
        if x1 <= x0 or y1 <= y0:
            return 0.0

        roi = purple_mask[y0:y1, x0:x1]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        best = 0.0
        target_x = (x1 - x0) * 0.5
        target_y = (y1 - y0) * 0.58
        norm_x = max(1.0, (x1 - x0) * 0.5)
        norm_y = max(1.0, (y1 - y0) * 0.8)

        for contour in contours:
            area = cv2.contourArea(contour)
            if not 12 <= area <= 2500:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h < 8 or w < 5:
                continue

            aspect = w / max(1, h)
            if not 0.2 <= aspect <= 1.8:
                continue

            contour_cx = x + w / 2.0
            contour_cy = y + h / 2.0
            distance = np.hypot(
                (contour_cx - target_x) / norm_x,
                (contour_cy - target_y) / norm_y,
            )
            proximity = max(0.0, 1.0 - distance)
            area_score = min(1.0, area / 140.0)
            upper_score = max(0.0, 1.0 - contour_cy / max(1.0, roi.shape[0]))
            best = max(best, 0.50 * area_score + 0.35 * proximity + 0.15 * upper_score)

        return best

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
                f"{idx} {candidate.confidence:.2f} a{candidate.arc_score:.2f}",
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
