"""Autonomous, fail-closed control of the two-phone physical lab.

The physical-lab runner deliberately keeps game UI policy outside the
experiment schema.  This module is the thin, replaceable UI policy layer:
it uses the existing ADB controller, calibrated logical coordinates, reviewed
screen-state detectors, and the existing artifact/run records.

The controller is intentionally conservative.  Card replacement is admitted
only after screenshot matching verifies the complete fixed deck.  A connected
run requires reviewed calibration and lifecycle manifests.  If a UI element
cannot be located or a lifecycle state is ambiguous, the run is rejected
without force-stopping the game or deleting a recording.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .artifacts import hash_file
from .calibration import CalibrationArtifact, CalibrationError
from .comparison import compare_observation_to_replay
from .devices import (
    ActionReceipt,
    AdbPhoneController,
    AdbScreenCapture,
    CaptureManifest,
    DeviceInfo,
    Frame,
    monotonic_time_us,
    sha256_bytes,
)
from .lifecycle import (
    LIFECYCLE_PATH,
    LifecycleReport,
    LifecycleState,
    LifecycleTransition,
)
from .replay import run_simulator_replay
from .runner import ActionLogEntry, PhysicalRunResult, write_run_artifacts
from .schema import (
    DeviceSpec,
    EvidenceStatus,
    ExperimentSpec,
    PhysicalAction,
    PhysicalLabError,
    canonical_hash,
)
from .screen_state import TemplateLifecycleDetector
from .sync import SynchronizationResult, estimate_clock_alignment, markers_from_captures


FIXED_HOG_CYCLE_DECK: tuple[str, ...] = (
    "hog-rider",
    "cannon",
    "musketeer",
    "skeletons",
    "ice-golem",
    "ice-spirit",
    "fireball",
    "log",
)

# The repository's card-art assets use the-log while the ruleset uses log.
CARD_ASSET_NAMES: Mapping[str, str] = {
    "log": "the-log",
}


class AutomationError(PhysicalLabError):
    """Raised when a physical UI action cannot be verified safely."""


@dataclass(frozen=True, slots=True)
class UiProfile:
    """Normalized coordinates for one known portrait device layout."""

    device_label: str
    screen_width_px: int
    screen_height_px: int
    deck_card_y_norm: tuple[float, float]
    collection_top_norm: float
    remove_button_y_norm: float
    online_row_y_norm: float = 0.45
    challenge_menu_y_norm: float = 0.56
    standard_challenge_y_norm: float = 0.35
    accept_challenge_y_norm: float = 0.42

    @classmethod
    def for_device(cls, device_label: str, info: DeviceInfo) -> "UiProfile":
        if info.screen_width_px is None or info.screen_height_px is None:
            raise AutomationError(f"device {device_label} did not report screen dimensions")
        # These are the two reviewed lab handset layouts observed during the
        # first physical setup.  A new resolution must receive an explicit
        # profile instead of silently inheriting unsafe tap coordinates.
        if info.screen_width_px != 1080 or info.screen_height_px not in {2280, 2400}:
            raise AutomationError(
                f"unsupported physical-lab display for {device_label}: "
                f"{info.screen_width_px}x{info.screen_height_px}; "
                "add a reviewed UiProfile before running automation"
            )
        if info.screen_height_px >= 2350:
            card_y = (0.275, 0.425)
            collection_top = 0.59
            remove_y = 0.42
        else:
            # The SM-G970F deck view includes the "regular deck" header.
            card_y = (0.355, 0.535)
            collection_top = 0.66
            remove_y = 0.49
        return cls(
            device_label=device_label,
            screen_width_px=info.screen_width_px,
            screen_height_px=info.screen_height_px,
            deck_card_y_norm=card_y,
            collection_top_norm=collection_top,
            remove_button_y_norm=remove_y,
        )

    def point(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        if not 0 <= x_norm <= 1 or not 0 <= y_norm <= 1:
            raise AutomationError("normalized UI coordinate is outside the screen")
        return (
            int(round(x_norm * self.screen_width_px)),
            int(round(y_norm * self.screen_height_px)),
        )

    def deck_card_centers(self) -> tuple[tuple[int, int], ...]:
        x_norms = (0.12, 0.345, 0.57, 0.795)
        result: list[tuple[int, int]] = []
        for y_norm in self.deck_card_y_norm:
            result.extend(self.point(x_norm, y_norm) for x_norm in x_norms)
        return tuple(result)

    def collection_tab(self) -> tuple[int, int]:
        return self.point(0.28, 0.94)

    def deck_editor_tab(self) -> tuple[int, int]:
        """Open the deck editor from the top Decks/Collection switcher."""

        return self.point(0.28, 0.12)

    def battle_tab(self) -> tuple[int, int]:
        return self.point(0.55, 0.94)

    def social_tab(self) -> tuple[int, int]:
        return self.point(0.74, 0.94)

    def ok_button(self) -> tuple[int, int]:
        return self.point(0.50, 0.955)

    def online_row(self) -> tuple[int, int]:
        return self.point(0.50, self.online_row_y_norm)

    def challenge_menu(self) -> tuple[int, int]:
        return self.point(0.20, self.challenge_menu_y_norm)

    def standard_challenge(self) -> tuple[int, int]:
        return self.point(0.50, self.standard_challenge_y_norm)

    def accept_challenge(self) -> tuple[int, int]:
        return self.point(0.55, self.accept_challenge_y_norm)


@dataclass(frozen=True, slots=True)
class CardMatch:
    card_id: str
    score: float
    center: tuple[int, int]


class CardVision:
    """Small OpenCV matcher for card/deck/hand UI regions.

    The matcher only answers a bounded UI question.  It never extracts game
    entities or promotes observations.  Templates are repository assets and
    are loaded lazily so importing the physical lab remains lightweight.
    """

    def __init__(
        self,
        template_root: str | Path,
        *,
        threshold: float = 0.62,
        search_scales: Sequence[float | tuple[float, float]] = (
            (0.72, 0.78),
            (0.80, 0.86),
            (0.86, 0.94),
            (0.92, 1.02),
            (1.00, 1.10),
        ),
    ) -> None:
        self.template_root = Path(template_root).resolve()
        self.threshold = float(threshold)
        if not 0 < self.threshold <= 1:
            raise AutomationError("card matcher threshold must be between zero and one")
        normalized_scales: list[tuple[float, float]] = []
        for value in search_scales:
            if isinstance(value, (tuple, list)):
                if len(value) != 2:
                    raise AutomationError("card matcher scale pairs need x and y values")
                x_scale, y_scale = float(value[0]), float(value[1])
            else:
                x_scale = y_scale = float(value)
            if x_scale <= 0 or y_scale <= 0:
                raise AutomationError("card matcher needs positive search scales")
            normalized_scales.append((x_scale, y_scale))
        self.search_scales = tuple(normalized_scales)
        if not self.search_scales:
            raise AutomationError("card matcher needs positive search scales")
        self._templates: dict[str, Any] = {}

    @staticmethod
    def _cv() -> tuple[Any, Any]:
        try:
            import cv2
            import numpy as np
        except ImportError as error:  # pragma: no cover - environment-specific
            raise AutomationError("autonomous card setup requires OpenCV and NumPy") from error
        return cv2, np

    def _template(self, card_id: str) -> Any:
        cached = self._templates.get(card_id)
        if cached is not None:
            return cached
        cv2, _np = self._cv()
        asset_name = CARD_ASSET_NAMES.get(card_id, card_id)
        path = self.template_root / f"{asset_name}.png"
        if not path.is_file():
            raise AutomationError(f"card template is missing: {path}")
        # Keep color here.  Grayscale card art produces many accidental
        # matches on the same bright/dark silhouette (especially on empty
        # slots and the collection background); the UI retains useful color
        # information even when card levels and labels differ.
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise AutomationError(f"card template cannot be decoded: {path}")
        # Ignore the outer asset frame and the level/name strip.  The central
        # illustration is stable across the two observed account card levels.
        height, width = image.shape[:2]
        image = image[int(height * 0.08) : int(height * 0.82), int(width * 0.10) : int(width * 0.90)]
        self._templates[card_id] = image
        return image

    @staticmethod
    def _image(frame: Frame) -> Any:
        cv2, np = CardVision._cv()
        if frame.payload is None:
            raise AutomationError("cannot inspect a frame without image payload")
        image = cv2.imdecode(np.frombuffer(frame.payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise AutomationError("cannot decode phone screenshot")
        return image

    def _best_in_bounds(
        self,
        image: Any,
        card_id: str,
        bounds: tuple[int, int, int, int],
    ) -> CardMatch | None:
        cv2, _np = self._cv()
        x0, y0, x1, y1 = bounds
        height, width = image.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        region = image[y0:y1, x0:x1]
        if region.size == 0:
            return None
        template = self._template(card_id)
        best: CardMatch | None = None
        for x_scale, y_scale in self.search_scales:
            template_width = max(8, int(round(template.shape[1] * x_scale)))
            template_height = max(8, int(round(template.shape[0] * y_scale)))
            if template_width >= region.shape[1] or template_height >= region.shape[0]:
                continue
            resized = cv2.resize(template, (template_width, template_height), interpolation=cv2.INTER_AREA)
            scores = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
            _min_value, max_value, _min_location, max_location = cv2.minMaxLoc(scores)
            score = float(max_value)
            center = (
                x0 + max_location[0] + template_width // 2,
                y0 + max_location[1] + template_height // 2,
            )
            candidate = CardMatch(card_id, score, center)
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def match_slot(self, frame: Frame, profile: UiProfile, slot: int) -> CardMatch | None:
        if not 0 <= slot < 8:
            raise AutomationError(f"deck slot out of range: {slot}")
        image = self._image(frame)
        center_x, center_y = profile.deck_card_centers()[slot]
        width = int(profile.screen_width_px * 0.22)
        height = int(profile.screen_height_px * 0.19)
        bounds = (center_x - width // 2, center_y - height // 2, center_x + width // 2, center_y + height // 2)
        matches = [self._best_in_bounds(image, card_id, bounds) for card_id in FIXED_HOG_CYCLE_DECK]
        matches = [match for match in matches if match is not None]
        if not matches:
            return None
        best = max(matches, key=lambda match: match.score)
        return best if best.score >= self.threshold else None

    def deck_matches(self, frame: Frame, profile: UiProfile) -> tuple[CardMatch | None, ...]:
        return tuple(self.match_slot(frame, profile, slot) for slot in range(8))

    def find_collection_card(
        self,
        frame: Frame,
        profile: UiProfile,
        card_id: str,
        *,
        scrolled: bool = False,
    ) -> CardMatch | None:
        image = self._image(frame)
        # Before the first swipe the deck occupies the upper part of the
        # screen, so keep the search in the collection band.  Once the page
        # scrolls, collection cards can move all the way to the top; using the
        # initial lower-band ROI after that point silently misses valid cards.
        top_norm = 0.04 if scrolled else profile.collection_top_norm
        bounds = (0, int(profile.screen_height_px * top_norm), profile.screen_width_px, int(profile.screen_height_px * 0.94))
        return self._best_in_bounds(image, card_id, bounds)

    def find_hand_card(self, frame: Frame, profile: UiProfile, card_id: str) -> CardMatch | None:
        image = self._image(frame)
        bounds = (
            0,
            int(profile.screen_height_px * 0.79),
            profile.screen_width_px,
            profile.screen_height_px,
        )
        return self._best_in_bounds(image, card_id, bounds)

    def find_card_near(
        self,
        frame: Frame,
        card_id: str,
        anchor: tuple[int, int],
        *,
        x_radius_norm: float = 0.14,
        y_radius_norm: float = 0.18,
    ) -> CardMatch | None:
        """Match a card near a previously tapped collection-card anchor."""

        image = self._image(frame)
        height, width = image.shape[:2]
        x, y = anchor
        bounds = (
            int(x - width * x_radius_norm),
            int(y - height * y_radius_norm),
            int(x + width * x_radius_norm),
            int(y + height * y_radius_norm),
        )
        return self._best_in_bounds(image, card_id, bounds)


class AutonomousPhone:
    """ADB UI primitives with screenshot verification and no force-stop."""

    def __init__(self, controller: AdbPhoneController, profile: UiProfile, vision: CardVision) -> None:
        self.controller = controller
        self.profile = profile
        self.vision = vision

    def screenshot(self) -> Frame:
        return self.controller.screenshot()

    def tap(self, point: tuple[int, int]) -> ActionReceipt:
        return self.controller.tap_screen(*point)

    def tap_norm(self, x_norm: float, y_norm: float) -> ActionReceipt:
        return self.tap(self.profile.point(x_norm, y_norm))

    def clear_vendor_overlay(self) -> None:
        """Stop only ASUS Game Genie, which can steal the touchscreen.

        Some ROG firmware exposes a touch-lock overlay above the game after
        long automated gestures.  ``am force-stop`` targets the vendor
        overlay package, never the Clash Royale process or a match.  On
        non-ASUS devices the package is absent and the command is harmless.
        """

        self.controller._run("shell", "am", "force-stop", "com.asus.gamewidget")
        time.sleep(0.2)

    def swipe_up(self) -> None:
        width = self.profile.screen_width_px
        height = self.profile.screen_height_px
        self.controller._run(
            "shell",
            "input",
            "swipe",
            str(width // 2),
            str(int(height * 0.88)),
            str(width // 2),
            str(int(height * 0.60)),
            "450",
        )

    def swipe_down(self) -> None:
        width = self.profile.screen_width_px
        height = self.profile.screen_height_px
        self.controller._run(
            "shell",
            "input",
            "swipe",
            str(width // 2),
            str(int(height * 0.28)),
            str(width // 2),
            str(int(height * 0.84)),
            "450",
        )

    def _scroll_signature(self, frame: Frame) -> Any:
        cv2, np = CardVision._cv()
        image = CardVision._image(frame)
        height = image.shape[0]
        # Ignore the fixed resource bar and the animated lower character art;
        # retain the deck/collection content that actually moves while
        # scrolling.  A tolerant signature avoids treating a card shimmer as
        # evidence that the page is still moving.
        crop = image[int(height * 0.24) : int(height * 0.75)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        reduced = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        return reduced.astype(np.int16)

    def _scroll_to_top(self, *, max_swipes: int = 12) -> None:
        """Return the deck/collection page to its deterministic top position."""

        previous_signature: Any | None = None
        for _attempt in range(max_swipes + 1):
            signature = self._scroll_signature(self.screenshot())
            if previous_signature is not None:
                cv2, np = CardVision._cv()
                del cv2
                if float(np.mean(np.abs(signature - previous_signature))) < 3.0:
                    return
            previous_signature = signature
            if _attempt < max_swipes:
                self.swipe_down()
                time.sleep(0.35)
        raise AutomationError(
            f"{self.profile.device_label} collection did not settle at the deck top"
        )

    def record(self, capture: AdbScreenCapture | None = None) -> Frame:
        frame = self.screenshot()
        if capture is not None:
            capture.record_frame(frame)
        return frame

    def wait_for(
        self,
        predicate: Callable[[Frame], bool],
        *,
        timeout_s: float,
        poll_s: float = 0.25,
        description: str,
    ) -> Frame:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                frame = self.screenshot()
                if predicate(frame):
                    return frame
            except (AutomationError, PhysicalLabError) as error:
                last_error = error
            time.sleep(poll_s)
        detail = f": {last_error}" if last_error is not None else ""
        raise AutomationError(f"timed out waiting for {self.profile.device_label} {description}{detail}")

    def _slot_bounds(self, slot: int) -> tuple[int, int, int, int]:
        center_x, center_y = self.profile.deck_card_centers()[slot]
        width = int(self.profile.screen_width_px * 0.22)
        height = int(self.profile.screen_height_px * 0.19)
        return (
            center_x - width // 2,
            center_y - height // 2,
            center_x + width // 2,
            center_y + height // 2,
        )

    def _slot_looks_empty(self, frame: Frame, slot: int) -> bool:
        """Check the empty blue slot placeholder independently of card identity."""

        cv2, np = CardVision._cv()
        image = CardVision._image(frame)
        x0, y0, x1, y1 = self._slot_bounds(slot)
        height, width = image.shape[:2]
        region = image[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
        if region.size == 0:
            return False
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        bright_fraction = float(np.mean(gray > 100))
        # Empty placeholders are dark blue with a small purple plus; every
        # occupied card seen in the reviewed layouts has a much larger bright
        # art/level-strip fraction.
        return bright_fraction < 0.20

    def _looks_like_deck_editor(self, frame: Frame) -> bool:
        """Verify the top Decks tab is selected before inspecting deck slots.

        The collection page and deck editor share the bottom navigation and
        much of their background. The selected top tab is the stable,
        reviewed distinction on both lab handsets: its blue panel is brighter
        than the unselected sibling. Requiring that distinction prevents a
        collection card from being mistaken for a deck slot.
        """

        cv2, np = CardVision._cv()
        image = CardVision._image(frame)
        height, width = image.shape[:2]
        left = image[
            int(height * 0.09) : int(height * 0.17),
            int(width * 0.08) : int(width * 0.43),
        ]
        right = image[
            int(height * 0.09) : int(height * 0.17),
            int(width * 0.50) : int(width * 0.92),
        ]
        if left.size == 0 or right.size == 0:
            return False
        # Compare luminance rather than a fixed RGB value so the same rule
        # remains valid across the two reviewed display resolutions.
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        return float(np.mean(left_gray) - np.mean(right_gray)) >= 18.0

    def configure_fixed_deck(self, *, max_swipes: int = 36) -> tuple[str, ...]:
        """Replace the active deck until it is exactly the fixed Hog cycle."""

        self.clear_vendor_overlay()
        self.tap(self.profile.deck_editor_tab())
        self.wait_for(
            self._looks_like_deck_editor,
            timeout_s=3.0,
            description="deck editor to become selected",
        )
        # A rejected or manually interrupted preparation may leave a card
        # action panel open.  Dismiss that menu only; this is not a game
        # cancellation and never sends a force-stop or match interruption.
        panel_frame = self.screenshot()
        if (
            self._find_red_button(panel_frame, self.profile) is not None
            or self._find_yellow_button(panel_frame, self.profile) is not None
        ):
            self.controller.press_back()
            time.sleep(0.35)
        # Work slot-by-slot.  A global "missing card" calculation is unsafe
        # while the deck UI is changing because an unreadable card could be
        # mistaken for a missing card and create duplicates.  The desired
        # card for each position is known, so each position is independently
        # verified, replaced, and verified again.
        for slot, replacement in enumerate(FIXED_HOG_CYCLE_DECK):
            self._scroll_to_top()
            frame = self.screenshot()
            slot_match = self.vision.match_slot(frame, self.profile, slot)
            if slot_match is not None and slot_match.card_id == replacement:
                continue

            # The game opens the removal panel only after the card itself is
            # selected.  Empty slots have no removal panel and go directly to
            # the collection; this is why the panel is located after the tap.
            self.tap(self.profile.deck_card_centers()[slot])
            time.sleep(0.25)
            remove = self._find_red_button(self.screenshot(), self.profile)
            if remove is not None:
                self.tap(remove)
                time.sleep(0.35)
                self.wait_for(
                    lambda candidate: self._slot_looks_empty(candidate, slot),
                    timeout_s=3.0,
                    description=f"slot {slot} to become empty after removal",
                )
            elif not self._slot_looks_empty(self.screenshot(), slot):
                raise AutomationError(
                    f"{self.profile.device_label} slot {slot} is occupied but has no verified Remove panel"
                )

            self._scroll_to_top()
            card_point = self._find_and_tap_collection_card(replacement, max_swipes=max_swipes)
            if card_point is None:
                raise AutomationError(
                    f"{self.profile.device_label} could not locate collection card {replacement}"
                )
            self._scroll_to_top()
            self.wait_for(
                lambda candidate: (
                    (match := self.vision.match_slot(candidate, self.profile, slot)) is not None
                    and match.card_id == replacement
                ),
                timeout_s=3.0,
                description=f"deck slot {slot} to become {replacement}",
            )

        self._scroll_to_top()
        final = self.vision.deck_matches(self.screenshot(), self.profile)
        final_ids = tuple(match.card_id if match is not None else "unknown" for match in final)
        if final_ids != FIXED_HOG_CYCLE_DECK:
            raise AutomationError(
                f"{self.profile.device_label} fixed deck verification failed: {final_ids!r}"
            )
        self.tap(self.profile.ok_button())
        time.sleep(0.5)
        return final_ids

    def _find_and_tap_collection_card(self, card_id: str, *, max_swipes: int) -> tuple[int, int] | None:
        seen_hashes: set[str] = set()
        last_rejection: str | None = None
        for _attempt in range(max_swipes + 1):
            frame = self.screenshot()
            payload_hash = frame.payload_hash or ""
            if payload_hash in seen_hashes:
                raise AutomationError(
                    f"{self.profile.device_label} collection did not move while searching for {card_id}"
                )
            seen_hashes.add(payload_hash)
            match = self.vision.find_collection_card(
                frame,
                self.profile,
                card_id,
                scrolled=_attempt > 0,
            )
            if match is not None and match.score >= self.vision.threshold:
                self.tap(match.center)
                time.sleep(0.25)
                selected = self.vision.find_card_near(
                    self.screenshot(),
                    card_id,
                    match.center,
                )
                if selected is None or selected.score < 0.52:
                    last_rejection = (
                        f"{self.profile.device_label} collection candidate {card_id} "
                        "did not remain identifiable after selection"
                    )
                    self.controller.press_back()
                    time.sleep(0.25)
                    if _attempt < max_swipes:
                        self.swipe_up()
                        time.sleep(0.35)
                        continue
                    break
                use = self._find_yellow_button(self.screenshot(), self.profile)
                if use is None:
                    last_rejection = (
                        f"{self.profile.device_label} selected {card_id} without a verified Use button"
                    )
                    self.controller.press_back()
                    time.sleep(0.25)
                    if _attempt < max_swipes:
                        self.swipe_up()
                        time.sleep(0.35)
                        continue
                    break
                self.tap(use)
                time.sleep(0.35)
                return match.center
            if _attempt < max_swipes:
                self.swipe_up()
                time.sleep(0.35)
        if last_rejection is not None:
            raise AutomationError(last_rejection)
        return None

    @staticmethod
    def _find_red_button(frame: Frame, profile: UiProfile) -> tuple[int, int] | None:
        cv2, np = CardVision._cv()
        image = CardVision._image(frame)
        # The remove button is a large red rectangle in the upper deck area.
        hsv = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame.payload or b"", dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2HSV)
        mask_a = cv2.inRange(hsv, np.array([0, 90, 80]), np.array([12, 255, 255]))
        mask_b = cv2.inRange(hsv, np.array([165, 90, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask_a, mask_b)
        height, width = image.shape[:2]
        mask[: int(height * 0.22), :] = 0
        mask[int(height * 0.62) :, :] = 0
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int, int]] = []
        expected_y = profile.remove_button_y_norm * height
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / h if h else 0.0
            center_y = y + h / 2.0
            if (
                area >= int(width * height * 0.004)
                and x <= int(width * 0.40)
                and w >= int(width * 0.15)
                and h >= int(height * 0.035)
                and h <= int(height * 0.09)
                and 1.35 <= aspect <= 3.50
                and abs(center_y - expected_y) <= height * 0.12
            ):
                candidates.append((area, x, y, w, h))
        if not candidates:
            return None
        _area, x, y, w, h = max(candidates)
        return x + w // 2, y + h // 2

    @staticmethod
    def _find_yellow_button(frame: Frame, profile: UiProfile) -> tuple[int, int] | None:
        """Find the localized yellow ``Use``/``Verwenden`` card action."""

        cv2, np = CardVision._cv()
        image = CardVision._image(frame)
        if frame.payload is None:
            return None
        color = cv2.imdecode(np.frombuffer(frame.payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if color is None:
            return None
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        # The game renders this as yellow on one handset and orange-yellow on
        # the other; include both hues while keeping the button geometry
        # restrictive.
        mask = cv2.inRange(hsv, np.array([5, 70, 80]), np.array([45, 255, 255]))
        height, width = image.shape[:2]
        mask[: int(height * 0.20), :] = 0
        mask[int(height * 0.92) :, :] = 0
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / h if h else 0.0
            center_y = y + h / 2.0
            if (
                area >= int(width * height * 0.004)
                and x <= int(width * 0.85)
                and w >= int(width * 0.15)
                and h >= int(height * 0.035)
                and h <= int(height * 0.065)
                and 1.80 <= aspect <= 3.50
            ):
                candidates.append((area, x, y, w, h))
        if not candidates:
            return None
        _area, x, y, w, h = max(candidates)
        return x + w // 2, y + h // 2

    def send_friendly_challenge(self) -> None:
        self.tap(self.profile.social_tab())
        time.sleep(0.8)
        self.tap(self.profile.online_row())
        time.sleep(0.35)
        self.tap(self.profile.challenge_menu())
        time.sleep(0.6)
        self.tap(self.profile.standard_challenge())

    def accept_friendly_challenge(self) -> None:
        self.tap(self.profile.social_tab())
        time.sleep(0.8)
        self.tap(self.profile.accept_challenge())

    def return_to_lobby(self) -> None:
        """Navigate out of menus/results without interrupting an active match."""

        self.tap(self.profile.battle_tab())
        time.sleep(0.7)

    def dismiss_result(self) -> None:
        self.tap(self.profile.ok_button())
        time.sleep(0.7)

    def select_and_place(
        self,
        card_id: str,
        *,
        calibration: CalibrationArtifact,
        arena_cell: tuple[int, int],
        capture: AdbScreenCapture | None = None,
    ) -> tuple[int, ActionReceipt, ActionReceipt]:
        frame = self.record(capture)
        match = self.vision.find_hand_card(frame, self.profile, card_id)
        if match is None or match.score < self.vision.threshold:
            raise AutomationError(
                f"{self.profile.device_label} could not identify {card_id} in the current hand"
            )
        slot_width = self.profile.screen_width_px / 4.0
        slot = min(3, max(0, int(match.center[0] / slot_width)))
        selected = self.controller.tap_screen(*self.profile.point((slot + 0.5) / 4.0, 0.875))
        if not selected.accepted:
            raise AutomationError(f"{self.profile.device_label} rejected card selection for {card_id}")
        placed = self.controller.tap_screen(*calibration.cell_to_pixel(arena_cell))
        if not placed.accepted:
            raise AutomationError(f"{self.profile.device_label} rejected placement for {card_id}")
        self.record(capture)
        return slot, selected, placed


@dataclass(frozen=True, slots=True)
class AutonomousSessionConfig:
    repository_root: Path
    raw_media_root: Path
    run_id: str
    cannon_delay_us: int = 17_000
    max_collection_swipes: int = 36
    lifecycle_timeout_s: float = 60.0
    result_timeout_s: float = 330.0


class AutonomousPhysicalLab:
    """Run one complete physical workflow once prerequisites are reviewed."""

    def __init__(
        self,
        *,
        spec: ExperimentSpec,
        controller_a: AdbPhoneController,
        controller_b: AdbPhoneController,
        calibration_a: CalibrationArtifact,
        calibration_b: CalibrationArtifact,
        lifecycle_manifest_a: str | Path,
        lifecycle_manifest_b: str | Path,
        config: AutonomousSessionConfig,
        template_root: str | Path,
    ) -> None:
        self.spec = spec
        self._validate_probe_spec()
        self.controllers = {"A": controller_a, "B": controller_b}
        self.calibrations = {"A": calibration_a, "B": calibration_b}
        self.config = config
        self.infos = self._preflight_devices()
        self.profiles = {
            side: UiProfile.for_device(side, self.infos[side]) for side in ("A", "B")
        }
        vision = CardVision(template_root)
        self.phones = {
            side: AutonomousPhone(self.controllers[side], self.profiles[side], vision)
            for side in ("A", "B")
        }
        self.detectors = {
            "A": TemplateLifecycleDetector(
                self.controllers["A"].screenshot,
                lifecycle_manifest_a,
                expected_device_id="A",
            ),
            "B": TemplateLifecycleDetector(
                self.controllers["B"].screenshot,
                lifecycle_manifest_b,
                expected_device_id="B",
            ),
        }

    def _validate_probe_spec(self) -> None:
        expected = (
            ("A", "hog-rider", (3, 20)),
            ("B", "cannon", (8, 13)),
        )
        actual = tuple(
            (action.side, action.card_id, action.arena_cell)
            for action in self.spec.actions
        )
        if actual != expected or len(self.spec.actions) != 2:
            raise AutomationError(
                "autonomous controller only admits the reviewed Hog/Cannon probe "
                f"with actions {expected!r}; received {actual!r}"
            )

    def _preflight_devices(self) -> dict[str, DeviceInfo]:
        infos: dict[str, DeviceInfo] = {}
        for side in ("A", "B"):
            info = self.controllers[side].device_info()
            if not info.connected:
                raise AutomationError(f"device {side} is not connected")
            if info.serial_hash != self.spec.devices[side].serial_hash:
                raise AutomationError(f"device {side} does not match the experiment identity")
            calibration_hash = self.calibrations[side].device_serial_hash
            if calibration_hash != info.serial_hash:
                raise AutomationError(f"calibration for device {side} does not match its identity")
            infos[side] = info
        return infos

    def prepare(self) -> dict[str, object]:
        """Configure both active decks and return to the lobby.

        This method is safe to run before a capture.  It never starts a match
        and never interrupts one; callers should only invoke it from a lobby
        or deck screen.
        """

        decks = {
            side: self.phones[side].configure_fixed_deck(
                max_swipes=self.config.max_collection_swipes
            )
            for side in ("A", "B")
        }
        for side in ("A", "B"):
            self.phones[side].return_to_lobby()
        return {"status": "prepared", "decks": decks}

    def _observe_pair(self) -> tuple[LifecycleState | None, dict[str, LifecycleState]]:
        states = {side: self.detectors[side].detect() for side in ("A", "B")}
        unique = set(states.values())
        return (next(iter(unique)) if len(unique) == 1 else None), states

    def _wait_pair(
        self,
        target: LifecycleState,
        *,
        observations: list[Mapping[str, str]],
        timeout_s: float | None = None,
    ) -> dict[str, LifecycleState]:
        deadline = time.monotonic() + (timeout_s or self.config.lifecycle_timeout_s)
        last_states: dict[str, LifecycleState] = {}
        while time.monotonic() < deadline:
            observed, states = self._observe_pair()
            last_states = states
            observations.append({side: states[side].value for side in ("A", "B")})
            if observed is target:
                return states
            time.sleep(0.25)
        detail = ", ".join(f"{side}={state.value}" for side, state in sorted(last_states.items()))
        raise AutomationError(f"timed out waiting for both phones to show {target.value} ({detail})")

    def run(self) -> PhysicalRunResult:
        capture_root = self.config.raw_media_root / "physical_lab" / self.config.run_id / "raw"
        captures = {
            side: AdbScreenCapture(self.controllers[side], capture_root / f"{side}.mp4")
            for side in ("A", "B")
        }
        observations: list[Mapping[str, str]] = []
        transitions: list[LifecycleTransition] = []
        actions: list[ActionLogEntry] = []
        action_times: dict[str, int] = {}
        rejection_reasons: list[str] = []
        started = monotonic_time_us()
        lifecycle: LifecycleReport | None = None
        synchronization: SynchronizationResult | None = None
        capture_manifests: dict[str, CaptureManifest] = {}
        current = LifecycleState.RECOVERY
        try:
            for capture in captures.values():
                capture.start()
            for side in ("A", "B"):
                self.phones[side].record(captures[side])

            self.phones["A"].return_to_lobby()
            self.phones["B"].return_to_lobby()
            states = self._wait_pair(LifecycleState.LOBBY, observations=observations)
            transitions.append(self._transition(current, LifecycleState.LOBBY, states))
            current = LifecycleState.LOBBY

            self.phones["B"].send_friendly_challenge()
            states = self._wait_pair(LifecycleState.CHALLENGE_SENT, observations=observations)
            transitions.append(self._transition(current, LifecycleState.CHALLENGE_SENT, states))
            current = LifecycleState.CHALLENGE_SENT

            self.phones["A"].accept_friendly_challenge()
            states = self._wait_pair(LifecycleState.CHALLENGE_ACCEPTED, observations=observations)
            transitions.append(self._transition(current, LifecycleState.CHALLENGE_ACCEPTED, states))
            current = LifecycleState.CHALLENGE_ACCEPTED

            states = self._wait_pair(LifecycleState.LOADING, observations=observations)
            transitions.append(self._transition(current, LifecycleState.LOADING, states))
            current = LifecycleState.LOADING
            battle_started = monotonic_time_us()
            states = self._wait_pair(LifecycleState.BATTLE, observations=observations)
            transitions.append(self._transition(current, LifecycleState.BATTLE, states))
            current = LifecycleState.BATTLE

            slot, selected, placed = self.phones["A"].select_and_place(
                "hog-rider",
                calibration=self.calibrations["A"],
                arena_cell=(3, 20),
                capture=captures["A"],
            )
            action_times["deploy-hog"] = monotonic_time_us() - battle_started
            actions.append(
                self._action_entry(
                    self.spec.actions[0],
                    slot=slot,
                    actual_time=action_times["deploy-hog"],
                    selected=selected,
                    placed=placed,
                )
            )

            time.sleep(max(0.0, self.config.cannon_delay_us / 1_000_000.0))
            slot, selected, placed = self.phones["B"].select_and_place(
                "cannon",
                calibration=self.calibrations["B"],
                arena_cell=(8, 13),
                capture=captures["B"],
            )
            action_times["deploy-cannon"] = monotonic_time_us() - battle_started
            actions.append(
                self._action_entry(
                    self.spec.actions[1],
                    slot=slot,
                    actual_time=action_times["deploy-cannon"],
                    selected=selected,
                    placed=placed,
                )
            )

            states = self._wait_pair(
                LifecycleState.RESULT,
                observations=observations,
                timeout_s=self.config.result_timeout_s,
            )
            transitions.append(self._transition(current, LifecycleState.RESULT, states))
            current = LifecycleState.RESULT
            self.phones["A"].dismiss_result()
            self.phones["B"].dismiss_result()
            states = self._wait_pair(LifecycleState.ARCHIVED, observations=observations)
            transitions.append(self._transition(current, LifecycleState.ARCHIVED, states))
            current = LifecycleState.ARCHIVED
            for side in ("A", "B"):
                self.phones[side].return_to_lobby()
            states = self._wait_pair(LifecycleState.RECOVERY, observations=observations)
            transitions.append(self._transition(current, LifecycleState.RECOVERY, states))
            current = LifecycleState.RECOVERY
        except (AutomationError, PhysicalLabError, CalibrationError, OSError) as error:
            rejection_reasons.append(str(error))
        finally:
            for side, capture in captures.items():
                try:
                    capture_manifests[side] = capture.stop()
                except (AutomationError, PhysicalLabError, OSError) as error:
                    rejection_reasons.append(f"capture {side} failed to stop: {error}")

        if capture_manifests:
            synchronization = estimate_clock_alignment(
                markers_from_captures(capture_manifests),
                device_ids=("A", "B"),
                declared_tolerance_us=min(item.timing_tolerance_us for item in self.spec.measurements),
            )
            if not synchronization.accepted:
                rejection_reasons.extend(synchronization.rejection_reasons)
            for side, manifest in capture_manifests.items():
                if not manifest.stream_verified:
                    rejection_reasons.append(f"capture {side} does not contain a verified video stream")
                if not manifest.frames:
                    rejection_reasons.append(f"capture {side} contains no synchronization frames")

        passed = not rejection_reasons and current is LifecycleState.RECOVERY
        lifecycle = LifecycleReport(
            initial_state=LifecycleState.RECOVERY,
            final_state=current,
            passed=passed,
            transitions=tuple(transitions),
            observations=tuple(observations),
            detector_provenance={
                side: self.detectors[side].provenance() for side in ("A", "B")
            },
        )
        replay = None
        if passed:
            try:
                replay = run_simulator_replay(self.spec, action_times=action_times)
            except PhysicalLabError as error:
                rejection_reasons.append(f"simulator replay failed: {error}")
                passed = False
        result = PhysicalRunResult(
            run_id=self.config.run_id,
            spec=self.spec,
            status=EvidenceStatus.CANDIDATE_ONLY if passed else EvidenceStatus.REJECTED,
            started_at_monotonic_us=started,
            finished_at_monotonic_us=max(started, monotonic_time_us()),
            device_info=self.infos,
            lifecycle=lifecycle,
            captures=capture_manifests,
            synchronization=synchronization,
            actions=tuple(actions),
            replay=replay,
            rejection_reasons=tuple(rejection_reasons),
        )
        write_run_artifacts(
            result,
            repository_root=self.config.repository_root,
            retention_manifest=self.config.repository_root / "outputs/simulator/fidelity_media/retention.json",
        )
        return result

    @staticmethod
    def _transition(
        from_state: LifecycleState,
        to_state: LifecycleState,
        device_states: Mapping[str, LifecycleState],
    ) -> LifecycleTransition:
        return LifecycleTransition(
            from_state=from_state,
            to_state=to_state,
            observed_at_monotonic_us=monotonic_time_us(),
            device_states=dict(device_states),
        )

    @staticmethod
    def _action_entry(
        action: PhysicalAction,
        *,
        slot: int,
        actual_time: int,
        selected: ActionReceipt,
        placed: ActionReceipt,
    ) -> ActionLogEntry:
        return ActionLogEntry(
            action_id=action.action_id,
            side=action.side,
            card_id=action.card_id,
            arena_cell=action.arena_cell,
            card_slot=slot,
            requested_trigger=action.trigger.to_dict(),
            actual_match_time_us=actual_time,
            selected_card_receipt=selected,
            placement_receipt=placed,
            accepted=selected.accepted and placed.accepted,
            reason=None,
        )


def bind_spec_to_devices(spec: ExperimentSpec, serial_a: str, serial_b: str) -> ExperimentSpec:
    """Bind raw serials to a spec without retaining the serial strings."""

    if not serial_a or not serial_b:
        raise AutomationError("both physical serials are required")
    devices = {
        "A": DeviceSpec(sha256_bytes(serial_a.encode("utf-8")), "player", "A"),
        "B": DeviceSpec(sha256_bytes(serial_b.encode("utf-8")), "opponent", "B"),
    }
    return replace(spec, devices=devices)


__all__ = [
    "AutonomousPhysicalLab",
    "AutonomousPhone",
    "AutonomousSessionConfig",
    "AutomationError",
    "CARD_ASSET_NAMES",
    "CardMatch",
    "CardVision",
    "FIXED_HOG_CYCLE_DECK",
    "UiProfile",
    "bind_spec_to_devices",
]
