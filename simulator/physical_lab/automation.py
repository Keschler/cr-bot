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

from dataclasses import dataclass, field, replace
import difflib
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .calibration import CalibrationArtifact, CalibrationError
from .devices import (
    ActionReceipt,
    AdbPhoneController,
    CaptureManifest,
    DeviceInfo,
    Frame,
    ScrcpyScreenCapture,
    monotonic_time_us,
    sha256_bytes,
)
from .lifecycle import (
    LifecycleReport,
    LifecycleState,
    LifecycleTransition,
)
from .observation_waiter import HogBridgeObservationWaiter, LiveTrackObservationSource
from .replay import run_simulator_replay
from .runner import (
    ActionLogEntry,
    CaptureFrameRecorder,
    ClockProvenance,
    ObservationWaiter,
    PhysicalRunResult,
    write_run_artifacts,
)
from .schema import (
    DeviceSpec,
    EvidenceStatus,
    ExperimentSpec,
    PhysicalAction,
    PhysicalLabError,
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

# On the Samsung SM-G970F/beyond0 build, placing these cards in human slots
# 1-3 activates a Hero/Evolution variant instead of the regular card. Keep
# this hardware rule explicit so a campaign cannot silently prepare the wrong
# card identity on Phone B. Slots in code are zero-based; the operator-facing
# safe range is human slots 4-8.
SAMSUNG_HERO_MUSKETEER_MODEL_IDS: frozenset[str] = frozenset({"sm-g970f", "beyond0"})
SAMSUNG_SPECIAL_CARD_FIRST_THREE_IDS: frozenset[str] = frozenset(
    {"cannon", "skeletons", "musketeer", "ice-golem", "ice-spirit"}
)
SAMSUNG_REGULAR_MUSKETEER_MIN_SLOT = 3
SAMSUNG_REGULAR_MUSKETEER_HUMAN_SLOT_RANGE = (4, 8)
# The reviewed Samsung collection capture renders the regular Musketeer at a
# lower art correlation because its level/evolution frame differs from the
# shared ASUS asset. Keep the identity-margin gate unchanged and lower only
# this evidenced per-card score floor.
SAMSUNG_COLLECTION_IDENTITY_MIN_SCORES: Mapping[str, float] = {
    "musketeer": 0.50,
}
ASUS_REGULAR_MUSKETEER_MODEL_IDS: frozenset[str] = frozenset({"asus_ai2302"})
ASUS_SPECIAL_CARD_FIRST_THREE_IDS: frozenset[str] = frozenset({"archers", "musketeer"})
ASUS_REGULAR_MUSKETEER_MIN_SLOT = 3
ASUS_REGULAR_MUSKETEER_HUMAN_SLOT_RANGE = (4, 8)
MUSKETEER_COLLECTION_CANDIDATE_MIN_SCORE = 0.55
ASUS_COLLECTION_CANDIDATE_MIN_SCORES: Mapping[str, float] = {
    "musketeer": MUSKETEER_COLLECTION_CANDIDATE_MIN_SCORE,
    "skeletons": 0.42,
}
ASUS_COLLECTION_IDENTITY_MIN_SCORES: Mapping[str, float] = {
    # The reviewed ASUS regular-Musketeer collection capture scored 0.491
    # against the shared art asset with a 0.149 runner-up margin.  Keep the
    # strict margin gate and admit this evidenced score, rather than
    # broadening all ASUS collection identities.
    "musketeer": 0.48,
    "skeletons": 0.42,
}

# Friendly Testspiel's fixed-deck mode consumes the deck in this exact order:
# the first four entries are the opening hand and the last four are the first
# four replacements.  Keep these as named contracts so physical preparation,
# experiment metadata, and simulator initial conditions cannot drift apart.
FIXED_HOG_CYCLE_OPENING_HAND: tuple[str, ...] = FIXED_HOG_CYCLE_DECK[:4]
FIXED_HOG_CYCLE_REPLACEMENT_ORDER: tuple[str, ...] = FIXED_HOG_CYCLE_DECK[4:]
FIXED_DECK_LONG_PRESS_MS = 900

# The repository's card-art assets use the-log while the ruleset uses log.
CARD_ASSET_NAMES: Mapping[str, str] = {
    "log": "the-log",
}

# The two lab accounts are stable test fixtures.  Keep the target identity
# explicit instead of selecting whichever online row happens to be first; the
# community list is reordered by the game whenever presence/order changes.
DEFAULT_FRIENDLY_TARGET_PLAYER_NAME = "KeschlerHD"
PLAYER_NAME_OCR_SCALE = 2.5


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
            # ASUS 1080x2400 deck editor: the second row is centered near
            # y=1180px. The previous 0.425 estimate landed in the gap
            # between the rows and made slot replacement fail closed.
            card_y = (0.275, 0.49)
            collection_top = 0.59
            remove_y = 0.42
        else:
            # The SM-G970F deck view includes the "regular deck" header.  The
            # The top Decks editor includes the deck-switcher/header above
            # the cards. The reviewed 1080x2280 capture places the card-row
            # centers near y=910 and y=1345 in this top-editor state.
            card_y = (0.40, 0.59)
            collection_top = 0.30
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
        # Reviewed ASUS lobby capture (2026-08-23): the Cards/Collection
        # icon is centered near x=250px.  x=.28 lands on the adjacent arrow
        # hitbox and can open the seasonal K.H.A.O.S surface instead.
        return self.point(0.23 if self.screen_height_px >= 2350 else 0.28, 0.94)

    def deck_editor_tab(self) -> tuple[int, int]:
        """Open the deck editor from the top Decks/Collection switcher."""

        return self.point(0.28, 0.12)

    def battle_tab(self) -> tuple[int, int]:
        # The Samsung bottom navigation is narrower around the battle icon;
        # x=.55 is the right-arrow hitbox on that 1080x2280 layout. The ASUS
        # 1080x2400 layout keeps the battle icon centered near x=.50.
        return self.point(0.42 if self.screen_height_px < 2350 else 0.50, 0.94)

    def social_tab(self) -> tuple[int, int]:
        # On B, x=.74 is also inside the right-arrow hitbox adjacent to the
        # Community shield. Use the shield center from the reviewed B frame.
        return self.point(0.67 if self.screen_height_px < 2350 else 0.74, 0.94)

    def ok_button(self) -> tuple[int, int]:
        return self.point(0.50, 0.955)

    def online_row(self) -> tuple[int, int]:
        return self.point(0.50, self.online_row_y_norm)

    def challenge_menu(self) -> tuple[int, int]:
        return self.point(0.20, self.challenge_menu_y_norm)

    def standard_challenge(self) -> tuple[int, int]:
        return self.point(0.50, self.standard_challenge_y_norm)

    def solo_battle(self) -> tuple[int, int]:
        """The Testspiel ``Solokampf`` button, whose long-press opens options."""

        return self.standard_challenge()

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
        collection_threshold: float | None = None,
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
        self.collection_threshold = (
            # Collection cards are selected by a real tap, so a permissive
            # art-only match can mutate the deck with a neighbouring card.
            # Keep this gate close to the reviewed deck threshold; false
            # positives are materially worse than one extra scroll.
            max(0.60, self.threshold - 0.02)
            if collection_threshold is None
            else float(collection_threshold)
        )
        if not 0 < self.collection_threshold <= 1:
            raise AutomationError("collection card matcher threshold must be between zero and one")
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
        self._card_ids: tuple[str, ...] | None = None

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

    def _candidates_in_bounds(
        self,
        image: Any,
        card_id: str,
        bounds: tuple[int, int, int, int],
        *,
        threshold: float,
        limit: int = 8,
    ) -> tuple[CardMatch, ...]:
        """Return distinct high-confidence peaks instead of one global peak."""

        cv2, np = self._cv()
        x0, y0, x1, y1 = bounds
        height, width = image.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        region = image[y0:y1, x0:x1]
        if region.size == 0:
            return ()
        template = self._template(card_id)
        raw: list[CardMatch] = []
        for x_scale, y_scale in self.search_scales:
            template_width = max(8, int(round(template.shape[1] * x_scale)))
            template_height = max(8, int(round(template.shape[0] * y_scale)))
            if template_width >= region.shape[1] or template_height >= region.shape[0]:
                continue
            resized = cv2.resize(template, (template_width, template_height), interpolation=cv2.INTER_AREA)
            scores = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
            peak_mask = (scores >= threshold).astype(np.uint8)
            component_count, labels = cv2.connectedComponents(peak_mask)
            for label in range(1, component_count):
                ys, xs = np.where(labels == label)
                if xs.size == 0:
                    continue
                component_scores = scores[ys, xs]
                best_index = int(np.argmax(component_scores))
                location_x = int(xs[best_index])
                location_y = int(ys[best_index])
                raw.append(
                    CardMatch(
                        card_id=card_id,
                        score=float(component_scores[best_index]),
                        center=(
                            x0 + location_x + template_width // 2,
                            y0 + location_y + template_height // 2,
                        ),
                    )
                )

        # Collapse the same card detected at several template scales while
        # retaining separate cards in adjacent collection cells.
        suppression_radius = max(48, int(min(width, height) * 0.08))
        accepted: list[CardMatch] = []
        for candidate in sorted(raw, key=lambda match: match.score, reverse=True):
            if any(
                (candidate.center[0] - existing.center[0]) ** 2
                + (candidate.center[1] - existing.center[1]) ** 2
                <= suppression_radius**2
                for existing in accepted
            ):
                continue
            accepted.append(candidate)
            if len(accepted) >= limit:
                break
        return tuple(accepted)

    def card_ids(self) -> tuple[str, ...]:
        """Return all card identities represented by the reviewed art root."""

        if self._card_ids is None:
            card_ids = {
                "log" if path.stem == "the-log" else path.stem
                for path in self.template_root.glob("*.png")
            }
            if not card_ids:
                raise AutomationError(f"card template root is empty: {self.template_root}")
            self._card_ids = tuple(sorted(card_ids))
        return self._card_ids

    def match_slot(self, frame: Frame, profile: UiProfile, slot: int) -> CardMatch | None:
        if not 0 <= slot < 8:
            raise AutomationError(f"deck slot out of range: {slot}")
        image = self._image(frame)
        center_x, center_y = profile.deck_card_centers()[slot]
        width = int(profile.screen_width_px * 0.22)
        height = int(profile.screen_height_px * 0.19)
        bounds = (center_x - width // 2, center_y - height // 2, center_x + width // 2, center_y + height // 2)
        matches = [self._best_in_bounds(image, card_id, bounds) for card_id in self.card_ids()]
        ranked = sorted(
            (match for match in matches if match is not None),
            key=lambda match: match.score,
            reverse=True,
        )
        if not ranked:
            return None
        best = ranked[0]
        runner_up_score = ranked[1].score if len(ranked) > 1 else -1.0
        if best.score < self.threshold or best.score - runner_up_score < 0.08:
            return None
        return best

    def rank_slot_identities(
        self,
        frame: Frame,
        profile: UiProfile,
        slot: int,
        *,
        limit: int = 3,
    ) -> tuple[CardMatch, ...]:
        """Rank card identities within one exact deck-slot ROI."""

        if not 0 <= slot < 8:
            raise AutomationError(f"deck slot out of range: {slot}")
        if limit <= 0:
            raise AutomationError("slot identity ranking limit must be positive")
        image = self._image(frame)
        center_x, center_y = profile.deck_card_centers()[slot]
        width = int(profile.screen_width_px * 0.22)
        height = int(profile.screen_height_px * 0.19)
        bounds = (
            center_x - width // 2,
            center_y - height // 2,
            center_x + width // 2,
            center_y + height // 2,
        )
        matches = [self._best_in_bounds(image, card_id, bounds) for card_id in self.card_ids()]
        ranked = sorted(
            (match for match in matches if match is not None),
            key=lambda match: match.score,
            reverse=True,
        )
        return tuple(ranked[:limit])

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
        bounds = self._collection_search_bounds(image, profile, scrolled=scrolled)
        return self._best_in_bounds(image, card_id, bounds)

    @staticmethod
    def _collection_search_top(image: Any, profile: UiProfile, *, scrolled: bool) -> float:
        """Keep collection matching below every possible active-deck card.

        A swipe may be consumed by an open panel or fail to move the page.
        Treating every post-swipe frame as fully scrolled then lets the card
        matcher select a copy in the active deck and open its Remove panel.
        The lower search band remains populated as the collection scrolls on
        both reviewed phones, so retaining its reviewed boundary is safer
        than admitting the ambiguous upper area.
        """

        del image, scrolled
        # On the reviewed 1080x2400 ASUS page, the active deck ends above
        # y=1200 while the first real collection row can be centered at
        # y=1350 after a swipe.  The older y=1416 boundary excluded that
        # evidenced Skeletons card.  Samsung's reviewed boundary is already
        # higher on screen and remains unchanged.
        if profile.screen_height_px >= 2350:
            return min(profile.collection_top_norm, 0.50)
        return profile.collection_top_norm

    @staticmethod
    def _has_editor_switcher_chrome(image: Any) -> bool:
        """Recognize the persistent deck/collection switcher in a screenshot.

        After a swipe, collection cards may occupy the upper part of the
        screen. That area is safe to search only when the active-deck editor
        is no longer visible. Keep this matcher independent from
        ``AutonomousPhone`` so it is usable in offline preparation checks.
        """

        cv2, np = CardVision._cv()
        height = image.shape[0]
        roi = image[int(height * 0.13) : int(height * 0.19), :]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue = (
            (hsv[:, :, 0] >= 90)
            & (hsv[:, :, 0] <= 125)
            & (hsv[:, :, 1] >= 100)
            & (hsv[:, :, 2] >= 90)
        )
        return float(np.mean(blue)) >= 0.35

    def _collection_search_bounds(
        self,
        image: Any,
        profile: UiProfile,
        *,
        scrolled: bool,
    ) -> tuple[int, int, int, int]:
        """Return a safe collection ROI for the current scroll state."""

        height, width = image.shape[:2]
        top = int(
            height * self._collection_search_top(image, profile, scrolled=scrolled)
        )
        if scrolled and not self._has_editor_switcher_chrome(image):
            # Once the editor chrome has scrolled away, the collection can
            # legitimately occupy the upper half of the screen.
            top = 0
        return (0, top, width, int(height * 0.94))

    def find_collection_card_candidates(
        self,
        frame: Frame,
        profile: UiProfile,
        card_id: str,
        *,
        scrolled: bool = False,
        threshold: float | None = None,
        limit: int = 8,
    ) -> tuple[CardMatch, ...]:
        image = self._image(frame)
        bounds = self._collection_search_bounds(image, profile, scrolled=scrolled)
        return self._candidates_in_bounds(
            image,
            card_id,
            bounds,
            threshold=self.collection_threshold if threshold is None else float(threshold),
            limit=limit,
        )

    def find_hand_card(self, frame: Frame, profile: UiProfile, card_id: str) -> CardMatch | None:
        image = self._image(frame)
        bounds = (
            0,
            int(profile.screen_height_px * 0.79),
            profile.screen_width_px,
            profile.screen_height_px,
        )
        return self._best_in_bounds(image, card_id, bounds)

    def find_hand_card_in_slot(
        self,
        frame: Frame,
        profile: UiProfile,
        card_id: str,
        slot: int,
        *,
        hand_px: tuple[float, float, float, float],
    ) -> CardMatch | None:
        """Match a hand card inside one policy-selected slot."""

        if type(slot) is not int or not 0 <= slot < 4:
            raise AutomationError("hand card slot must be from 0 through 3")
        image = self._image(frame)
        hand_x, hand_y, hand_width, hand_height = hand_px
        if hand_width <= 0 or hand_height <= 0:
            raise AutomationError("reviewed hand rectangle must have positive dimensions")
        slot_width = hand_width / 4.0
        bounds = (
            int(round(hand_x + slot * slot_width)),
            int(round(hand_y)),
            int(round(hand_x + (slot + 1) * slot_width)),
            int(round(hand_y + hand_height)),
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

    def rank_card_identities_near(
        self,
        frame: Frame,
        anchor: tuple[int, int],
        *,
        limit: int = 3,
        x_radius_norm: float = 0.14,
        y_radius_norm: float = 0.18,
    ) -> tuple[CardMatch, ...]:
        """Rank the selected card against every reviewed card-art identity."""

        if limit <= 0:
            raise AutomationError("card identity ranking limit must be positive")
        image = self._image(frame)
        height, width = image.shape[:2]
        x, y = anchor
        bounds = (
            int(x - width * x_radius_norm),
            int(y - height * y_radius_norm),
            int(x + width * x_radius_norm),
            int(y + height * y_radius_norm),
        )
        matches = [self._best_in_bounds(image, card_id, bounds) for card_id in self.card_ids()]
        ranked = sorted(
            (match for match in matches if match is not None),
            key=lambda match: match.score,
            reverse=True,
        )
        return tuple(ranked[:limit])


class AutonomousPhone:
    """ADB UI primitives with frame verification and no force-stop."""

    def __init__(
        self,
        controller: AdbPhoneController,
        profile: UiProfile,
        vision: CardVision,
        *,
        device_model: str | None = None,
        action_frame_provider: Callable[[], Frame] | None = None,
    ) -> None:
        self.controller = controller
        self.profile = profile
        self.vision = vision
        self.device_model = None if device_model is None else device_model.strip().lower()
        if action_frame_provider is not None and not callable(action_frame_provider):
            raise AutomationError("action_frame_provider must be callable when supplied")
        self.action_frame_provider = action_frame_provider

    def _validate_device_deck_constraints(self, target: Sequence[str]) -> None:
        is_known_samsung = self.device_model in SAMSUNG_HERO_MUSKETEER_MODEL_IDS
        # The autonomous preparation script constructs the operator-scoped
        # B phone before it has a model object available. The lab's recorded
        # mapping is B=SM-G970F, so retain the guard for that explicit scope.
        is_known_samsung = is_known_samsung or (
            self.device_model is None and self.profile.device_label.upper() == "B"
        )
        if is_known_samsung:
            violations = tuple(
                (slot + 1, card_id)
                for slot, card_id in enumerate(tuple(target)[:SAMSUNG_REGULAR_MUSKETEER_MIN_SLOT])
                if card_id in SAMSUNG_SPECIAL_CARD_FIRST_THREE_IDS
            )
            if violations:
                low, high = SAMSUNG_REGULAR_MUSKETEER_HUMAN_SLOT_RANGE
                details = ", ".join(f"{card} in slot {slot}" for slot, card in violations)
                raise AutomationError(
                    f"{self.profile.device_label} {self.device_model or 'SM-G970F'} requires "
                    f"Cannon, Skeletons, Musketeer, Ice Golem, and Ice Spirit in human "
                    f"deck slots {low}-{high}; unsafe placement: {details}"
                )

        is_known_asus = self.device_model in ASUS_REGULAR_MUSKETEER_MODEL_IDS
        is_known_asus = is_known_asus or (
            self.device_model is None and self.profile.device_label.upper() == "A"
        )
        if not is_known_asus:
            return
        violations = tuple(
            (slot + 1, card_id)
            for slot, card_id in enumerate(tuple(target)[:ASUS_REGULAR_MUSKETEER_MIN_SLOT])
            if card_id in ASUS_SPECIAL_CARD_FIRST_THREE_IDS
        )
        if violations:
            low, high = ASUS_REGULAR_MUSKETEER_HUMAN_SLOT_RANGE
            details = ", ".join(f"{card} in slot {slot}" for slot, card in violations)
            raise AutomationError(
                f"{self.profile.device_label} {self.device_model or 'ASUS_AI2302'} requires "
                f"regular Archers and Musketeer in human deck slots {low}-{high}; "
                f"unsafe placement: {details}"
            )

    def _collection_candidate_threshold(self, card_id: str) -> float:
        """Return a card-specific candidate floor before any real tap."""

        is_known_asus = self.device_model in ASUS_REGULAR_MUSKETEER_MODEL_IDS
        is_known_asus = is_known_asus or (
            self.device_model is None and self.profile.device_label.upper() == "A"
        )
        if is_known_asus and card_id in ASUS_COLLECTION_CANDIDATE_MIN_SCORES:
            return min(self.vision.collection_threshold, ASUS_COLLECTION_CANDIDATE_MIN_SCORES[card_id])
        return self.vision.collection_threshold

    def _minimum_collection_identity_score(self, card_id: str) -> float:
        """Return the evidence floor for a selected collection-card identity."""

        minimum = max(0.60, self.vision.collection_threshold)
        is_known_samsung = self.device_model in SAMSUNG_HERO_MUSKETEER_MODEL_IDS
        is_known_samsung = is_known_samsung or (
            self.device_model is None and self.profile.device_label.upper() == "B"
        )
        if is_known_samsung:
            minimum = max(
                minimum if card_id not in SAMSUNG_COLLECTION_IDENTITY_MIN_SCORES else 0.0,
                SAMSUNG_COLLECTION_IDENTITY_MIN_SCORES.get(card_id, minimum),
            )
        is_known_asus = self.device_model in ASUS_REGULAR_MUSKETEER_MODEL_IDS
        is_known_asus = is_known_asus or (
            self.device_model is None and self.profile.device_label.upper() == "A"
        )
        if is_known_asus and card_id == "musketeer":
            minimum = min(minimum, ASUS_COLLECTION_IDENTITY_MIN_SCORES[card_id])
        if is_known_asus and card_id == "skeletons":
            minimum = min(minimum, ASUS_COLLECTION_IDENTITY_MIN_SCORES[card_id])
        return minimum

    def _find_existing_target_slot(
        self,
        frame: Frame,
        card_id: str,
        *,
        exclude_slot: int,
    ) -> int | None:
        """Find a high-confidence misplaced copy before collection search.

        The game hides cards already present elsewhere in the active deck
        from the collection picker. Resolve that dependency first when a
        target card is being moved between slots. The lower preflight score
        is intentionally paired with a margin gate and is used only to find
        a donor slot, never to accept a final card selection.
        """

        candidates: list[tuple[float, int]] = []
        for slot in range(8):
            if slot == exclude_slot:
                continue
            # The ASUS editor's rightmost card art extends beyond the
            # nominal slot center used for taps.  A slot-sized crop can clip
            # that art and turn an occupied donor into a false empty slot.
            # Use the same bounded neighbourhood matcher used after a
            # collection tap, while retaining an explicit score and margin
            # gate so this wider preflight can never authorize a collection
            # selection by itself.
            ranked = self.vision.rank_card_identities_near(
                frame,
                self.profile.deck_card_centers()[slot],
                limit=2,
                x_radius_norm=0.14,
                y_radius_norm=0.18,
            )
            if not ranked or ranked[0].card_id != card_id or ranked[0].score < 0.40:
                continue
            runner = ranked[1].score if len(ranked) > 1 else -1.0
            if ranked[0].score - runner < 0.05:
                continue
            candidates.append((ranked[0].score, slot))
        if not candidates:
            return None
        return max(candidates)[1]

    def _slot_matches_expected(self, frame: Frame, slot: int, card_id: str) -> bool:
        """Verify a known expected card with a wider, margin-gated slot ROI.

        Evolution/Hero framing can clip the illustration inside the narrow
        generic slot crop.  The wider matcher remains anchored to one exact
        slot and must name the expected card with both a strong score and a
        clear runner-up margin; it cannot authorize an arbitrary identity.
        """

        ranked = self.vision.rank_card_identities_near(
            frame,
            self.profile.deck_card_centers()[slot],
            limit=2,
            x_radius_norm=0.14,
            y_radius_norm=0.18,
        )
        if not ranked or ranked[0].card_id != card_id or ranked[0].score < 0.58:
            return False
        runner = ranked[1].score if len(ranked) > 1 else -1.0
        return ranked[0].score - runner >= 0.10

    def screenshot(self) -> Frame:
        last_error: AutomationError | None = None
        for attempt in range(3):
            frame = self.controller.screenshot()
            try:
                CardVision._image(frame)
                return frame
            except AutomationError as error:
                last_error = error
                if attempt < 2:
                    time.sleep(0.15)
        raise AutomationError(
            f"{self.profile.device_label} returned three undecodable screenshots: {last_error}"
        )

    def tap(self, point: tuple[int, int]) -> ActionReceipt:
        receipt = self.controller.tap_screen(*point)
        if not receipt.accepted:
            raise AutomationError(
                f"{self.profile.device_label} rejected tap at {point}: "
                f"{receipt.reason or 'unknown reason'}"
            )
        return receipt

    def tap_norm(self, x_norm: float, y_norm: float) -> ActionReceipt:
        return self.tap(self.profile.point(x_norm, y_norm))

    @staticmethod
    def _normalize_ocr_name(value: str) -> str:
        """Normalize a rendered player name for OCR comparison."""

        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    @classmethod
    def _ocr_name_score(cls, observed: str, expected: str) -> float:
        """Return a conservative similarity score for one OCR token."""

        observed_normalized = cls._normalize_ocr_name(observed)
        expected_normalized = cls._normalize_ocr_name(expected)
        if not observed_normalized or not expected_normalized:
            return 0.0
        if observed_normalized == expected_normalized:
            return 1.0
        # OCR occasionally confuses adjacent outline letters in the game's
        # heavy display font (for example ``l`` and ``i`` in KeschlerHD). Do
        # not allow short tokens or arbitrary substrings through this gate.
        if len(observed_normalized) < 5 or len(expected_normalized) < 5:
            return 0.0
        if abs(len(observed_normalized) - len(expected_normalized)) > 2:
            return 0.0
        # Require the distinctive leading name prefix. This admits the
        # reviewed leading-glyph OCR artifact (``YKescuierHD``) while
        # rejecting the adjacent clan tag ``pwn_keschler`` even though the
        # two strings have a similar whole-token edit ratio.
        prefix_length = min(4, len(expected_normalized))
        comparisons = (observed_normalized, observed_normalized[1:])
        scores = [
            difflib.SequenceMatcher(
                a=candidate,
                b=expected_normalized,
                autojunk=False,
            ).ratio()
            for candidate in comparisons
            if candidate
        ]
        score = max(scores, default=0.0)
        has_prefix = any(
            candidate.startswith(expected_normalized[:prefix_length])
            for candidate in comparisons
        )
        # The adjacent clan label is sometimes segmented as just
        # ``keschler``. Require the distinctive target suffix too; reviewed
        # target OCR variants retain the final ``HD`` even when leading glyphs
        # are confused.
        suffix_length = min(2, len(expected_normalized))
        has_suffix = any(
            candidate.endswith(expected_normalized[-suffix_length:])
            for candidate in comparisons
        )
        return score if has_prefix and has_suffix and score >= 0.78 else 0.0

    @staticmethod
    def _parse_ocr_tsv(payload: bytes) -> tuple[tuple[str, float, int, int, int, int], ...]:
        """Parse tesseract TSV word records without trusting OCR as truth."""

        records: list[tuple[str, float, int, int, int, int]] = []
        try:
            text = payload.decode("utf-8", errors="replace")
        except AttributeError:
            return ()
        for line in text.splitlines():
            fields = line.split("\t")
            # level,page,block,paragraph,line,word,left,top,width,height,conf,text
            if len(fields) < 12 or not fields[11].strip():
                continue
            try:
                left, top, width, height = (int(fields[index]) for index in (6, 7, 8, 9))
                confidence = float(fields[10])
            except (TypeError, ValueError):
                continue
            records.append((fields[11].strip(), confidence, left, top, width, height))
        return tuple(records)

    def _ocr_region(
        self,
        frame: Frame,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        psm: int,
        scale: float = 1.0,
        enhance: bool = False,
    ) -> tuple[tuple[str, float, int, int, int, int], ...]:
        """Run bounded OCR on an image region and return local word boxes.

        Clash Royale renders the community screen as a custom canvas, so
        Android's accessibility tree contains no player names.  OCR is used
        only for selecting a named online row and is kept local to the
        current screenshot; it never promotes a game observation.
        """

        if frame.payload is None:
            raise AutomationError(
                f"{self.profile.device_label} cannot OCR a frame without image payload"
            )
        screen_width = self.profile.screen_width_px
        screen_height = self.profile.screen_height_px
        x0 = max(0, min(int(x), screen_width - 1))
        y0 = max(0, min(int(y), screen_height - 1))
        crop_width = max(1, min(int(width), screen_width - x0))
        crop_height = max(1, min(int(height), screen_height - y0))
        geometry = f"{crop_width}x{crop_height}+{x0}+{y0}"
        convert_args = ["magick", "png:-", "-crop", geometry]
        if scale != 1.0:
            convert_args.extend(["-resize", f"{scale * 100:g}%"])
        if enhance:
            convert_args.extend(["-colorspace", "Gray", "-contrast-stretch", "0x20"])
        convert_args.append("png:-")
        try:
            converted = subprocess.run(
                convert_args,
                input=frame.payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as error:
            raise AutomationError(
                f"{self.profile.device_label} cannot run ImageMagick for player-name OCR"
            ) from error
        if converted.returncode != 0 or not converted.stdout:
            detail = converted.stderr.decode("utf-8", errors="replace").strip()
            raise AutomationError(
                f"{self.profile.device_label} ImageMagick failed while preparing player-name OCR"
                + (f": {detail}" if detail else "")
            )
        try:
            recognized = subprocess.run(
                ["tesseract", "stdin", "stdout", "--psm", str(psm), "tsv"],
                input=converted.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as error:
            raise AutomationError(
                f"{self.profile.device_label} cannot run tesseract for player-name OCR"
            ) from error
        if recognized.returncode != 0:
            detail = recognized.stderr.decode("utf-8", errors="replace").strip()
            raise AutomationError(
                f"{self.profile.device_label} tesseract failed while reading player-name OCR"
                + (f": {detail}" if detail else "")
            )
        return self._parse_ocr_tsv(recognized.stdout)

    def _find_online_player_row(
        self,
        frame: Frame,
        player_name: str,
    ) -> tuple[int, int]:
        """Locate the named online player, independent of list ordering."""

        expected = self._normalize_ocr_name(player_name)
        if len(expected) < 3:
            raise AutomationError("target online player name must contain at least three letters")

        width = self.profile.screen_width_px
        height = self.profile.screen_height_px
        # The heading and online rows occupy the middle community panel. Scan
        # overlapping row bands because online players can move up/down when
        # presence updates, while excluding the lower ranking list.
        # Keep the crop close to the left/name column. Including the trophy
        # artwork or the next row makes the stylized name font materially
        # harder for tesseract to segment. The values are normalized to the
        # reviewed 1080px-wide phones and rounded so the crop geometry is
        # stable across both screen heights.
        crop_x = int(round(width * 0.093))
        crop_width = int(round(width * 0.741))
        crop_height = int(round(height * 0.0965))
        crop_starts: list[int] = []
        # The reviewed community cards have a stable row pitch of about
        # 150px on both lab phones. Aligning to the card tops is important:
        # the stylized OCR result is much less stable when a crop cuts
        # through the boundary between two rows.
        crop_y = int(round(height * 0.351))
        last_crop_y = int(round(height * 0.70))
        step = max(1, int(round(height * 0.0658)))
        while crop_y <= last_crop_y:
            crop_starts.append(crop_y)
            crop_y += step

        hits: list[tuple[float, float, float, int]] = []

        def collect_hits(
            crop_y: int,
            words: Iterable[tuple[str, float, int, int, int, int]],
            *,
            minimum_confidence: float = 20.0,
        ) -> None:
            for observed, confidence, _left, top, _word_width, word_height in words:
                score = self._ocr_name_score(observed, expected)
                if score <= 0.0 or confidence < minimum_confidence:
                    continue
                text_center_y = crop_y + (top + word_height / 2.0) / PLAYER_NAME_OCR_SCALE
                if not (height * 0.32 <= text_center_y <= height * 0.78):
                    continue
                hits.append((score, confidence, text_center_y, crop_y))

        # Most reviewed frames are settled enough for one bounded community
        # panel OCR pass. This avoids spawning a separate ImageMagick and
        # tesseract process for every row while still keeping the lower
        # ranking region out of the selector's decision window.
        quick_y = int(round(height * 0.31))
        quick_height = min(int(round(height * 0.40)), height - quick_y)
        for psm in (11, 6):
            collect_hits(
                quick_y,
                self._ocr_region(
                    frame,
                    x=crop_x,
                    y=quick_y,
                    width=crop_width,
                    height=quick_height,
                    psm=psm,
                    scale=PLAYER_NAME_OCR_SCALE,
                    enhance=True,
                ),
            )
            if hits:
                break

        if not hits:
            # A settled row-card pass is the second choice when the wider
            # panel segmentation merged the target with a card boundary.
            for crop_y in crop_starts:
                collect_hits(
                    crop_y,
                    self._ocr_region(
                        frame,
                        x=crop_x,
                        y=crop_y,
                        width=crop_width,
                        height=crop_height,
                        psm=6,
                        scale=PLAYER_NAME_OCR_SCALE,
                        enhance=True,
                    ),
                )

        if not hits:
            # During the community-list animation the row card can move by a
            # few dozen pixels and psm 6 may merge the outlined name with the
            # adjacent card boundary. Retry the same bounded rows with sparse
            # offsets and psm 11. Keep this fallback local and finite: OCR
            # failure must never turn into a blind tap on the first row.
            for offset in (-40, -20, 20, 40):
                for crop_y in crop_starts:
                    adjusted_y = max(0, min(height - crop_height, crop_y + offset))
                    collect_hits(
                        adjusted_y,
                        self._ocr_region(
                            frame,
                            x=crop_x,
                            y=adjusted_y,
                            width=crop_width,
                            height=crop_height,
                            psm=11,
                            scale=PLAYER_NAME_OCR_SCALE,
                            enhance=True,
                        ),
                        minimum_confidence=12.0,
                    )

        if not hits:
            raise AutomationError(
                f"{self.profile.device_label} could not locate online player {player_name!r}"
            )

        # Adjacent overlapping scan bands see the same rendered name. Collapse
        # those observations by the actual text y-coordinate before checking
        # for ambiguity. A second distinct hit is treated as unsafe rather
        # than choosing the first visible account.
        hits.sort(key=lambda hit: (hit[2], -hit[0], -hit[1]))
        clusters: list[list[tuple[float, float, float, int]]] = []
        tolerance = height * 0.055
        for hit in hits:
            if not clusters or hit[2] - clusters[-1][-1][2] > tolerance:
                clusters.append([hit])
            else:
                clusters[-1].append(hit)
        if len(clusters) != 1:
            locations = ", ".join(f"{cluster[0][2]:.0f}px" for cluster in clusters)
            raise AutomationError(
                f"{self.profile.device_label} found multiple online rows matching "
                f"{player_name!r} ({locations}); refusing to guess"
            )
        best = max(clusters[0], key=lambda hit: (hit[0], hit[1]))
        # Player names sit slightly above the vertical center of their row;
        # the offset lands in the row body while staying away from the name
        # glyphs and the right-side trophy control.
        row_y = int(round(best[2] + height * 0.015))
        row_y = max(int(height * 0.34), min(int(height * 0.78), row_y))
        return int(round(width * 0.50)), row_y

    def _frame_contains_player_name(self, frame: Frame, player_name: str) -> bool:
        """Verify that the selected-player popup names the intended account."""

        expected = self._normalize_ocr_name(player_name)
        for psm in (3, 6):
            words = self._ocr_region(
                frame,
                x=0,
                y=0,
                width=self.profile.screen_width_px,
                height=self.profile.screen_height_px,
                psm=psm,
                scale=1.0,
                enhance=False,
            )
            if any(self._ocr_name_score(observed, expected) >= 0.84 for observed, *_ in words):
                return True
        return False

    def _find_challenge_menu_button(self, frame: Frame) -> tuple[int, int]:
        """Find ``Testspiel``/``Friendly Battle`` in the selected-player menu."""

        labels = {
            "testspiel",
            "friendly",
            "friendlybattle",
            "freundschaftskampf",
        }
        candidates: list[tuple[float, int, int]] = []
        for psm in (3, 6, 11):
            words = self._ocr_region(
                frame,
                x=0,
                y=0,
                width=self.profile.screen_width_px,
                height=self.profile.screen_height_px,
                psm=psm,
                scale=1.0,
                enhance=False,
            )
            for observed, confidence, left, top, word_width, word_height in words:
                normalized = self._normalize_ocr_name(observed)
                # The reviewed German build occasionally renders ``Testspiel``
                # as ``Jestspiel`` in the outline font. Keep the fuzzy branch
                # constrained to that distinctive prefix and button geometry.
                recognized_testspiel = (
                    normalized in labels
                    or (
                        len(normalized) >= 7
                        and normalized.startswith(("testsp", "jestsp"))
                    )
                )
                if not recognized_testspiel or confidence < 20.0:
                    continue
                center_x = left + word_width // 2
                center_y = top + word_height // 2
                if center_x <= int(self.profile.screen_width_px * 0.60) and int(
                    self.profile.screen_height_px * 0.25
                ) <= center_y <= int(self.profile.screen_height_px * 0.80):
                    candidates.append((confidence, center_x, center_y))
        if not candidates:
            raise AutomationError(
                f"{self.profile.device_label} selected player menu has no verified Testspiel button"
            )
        if len(candidates) > 1:
            # A single button can produce multiple OCR words in localized
            # variants; choose only the strongest one when their centers are
            # in the same button row. Distinct rows remain ambiguous.
            y_values = [candidate[2] for candidate in candidates]
            if max(y_values) - min(y_values) > self.profile.screen_height_px * 0.08:
                raise AutomationError(
                    f"{self.profile.device_label} selected player menu has multiple Testspiel candidates"
                )
        _confidence, x, y = max(candidates, key=lambda candidate: candidate[0])
        return x, y

    @staticmethod
    def _fixed_deck_order_enabled(
        frame: Frame,
        toggle_point: tuple[int, int],
    ) -> bool:
        """Read the reviewed blue active half of the fixed-order switch.

        The German build labels the two states ``Aus`` and ``Ein``. OCR is
        not reliable on this small outlined label, but the active switch half
        is consistently blue. The right-hand blue fraction is high only in
        the ``Ein`` state in the reviewed B layout.
        """

        cv2, np = CardVision._cv()
        image = CardVision._image(frame)
        height, width = image.shape[:2]
        center_x, center_y = toggle_point
        y0 = max(0, center_y - int(height * 0.018))
        y1 = min(height, center_y + int(height * 0.018))
        left_x0 = max(0, center_x - int(width * 0.065))
        left_x1 = min(width, center_x + int(width * 0.037))
        right_x0 = min(width, center_x + int(width * 0.037))
        right_x1 = min(width, center_x + int(width * 0.139))
        if left_x1 <= left_x0 or right_x1 <= right_x0 or y1 <= y0:
            return False
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(
            hsv,
            np.array([85, 80, 80]),
            np.array([130, 255, 255]),
        )
        left_fraction = float(np.mean(blue[y0:y1, left_x0:left_x1] > 0))
        right_fraction = float(np.mean(blue[y0:y1, right_x0:right_x1] > 0))
        return right_fraction >= 0.50 and right_fraction > left_fraction + 0.10

    def long_press(self, point: tuple[int, int], *, duration_ms: int = FIXED_DECK_LONG_PRESS_MS) -> ActionReceipt:
        """Issue a real hold gesture, not a sequence of taps."""

        if type(duration_ms) is not int or duration_ms <= 0:
            raise AutomationError("long-press duration must be a positive integer")
        receipt = self.controller.long_press_screen(*point, duration_ms=duration_ms)
        if not receipt.accepted:
            raise AutomationError(
                f"{self.profile.device_label} rejected long press at {point}: "
                f"{receipt.reason or 'unknown reason'}"
            )
        return receipt

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

    def _wait_for_scroll_settled(
        self,
        *,
        timeout_s: float = 8.0,
        poll_s: float = 0.15,
        stable_samples: int = 1,
    ) -> None:
        """Wait for inertial collection motion to stop before using tap coordinates."""

        if timeout_s <= 0 or poll_s < 0 or stable_samples <= 0:
            raise AutomationError("scroll-settle parameters must be positive")
        previous = self._scroll_signature(self.screenshot())
        # ADB screenshot transport on the reviewed phones can itself exceed
        # two seconds. Start the settle deadline after the baseline capture
        # so transport latency cannot masquerade as persistent UI motion.
        deadline = time.monotonic() + timeout_s
        stable_count = 0
        while time.monotonic() < deadline:
            time.sleep(poll_s)
            current = self._scroll_signature(self.screenshot())
            cv2, np = CardVision._cv()
            del cv2
            difference = float(np.mean(np.abs(current - previous)))
            if difference < 3.0:
                stable_count += 1
                if stable_count >= stable_samples:
                    return
            else:
                stable_count = 0
            previous = current
        raise AutomationError(
            f"{self.profile.device_label} collection remained in motion after a swipe"
        )

    def record(self, capture: CaptureFrameRecorder | None = None) -> Frame:
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
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        bright_fraction = float(np.mean(gray > 100))
        blue_fraction = float(
            np.mean(
                (hsv[:, :, 0] >= 90)
                & (hsv[:, :, 0] <= 125)
                & (hsv[:, :, 1] >= 100)
            )
        )
        # The selected empty placeholder on the reviewed 1080x2280 phone is
        # 20.4% bright and 82.9% saturated blue. Occupied slots in the same
        # screenshot are at least 33.9% bright and at most 54.0% blue. Require
        # both signals so card art cannot be accepted merely for being dark.
        return bright_fraction < 0.27 and blue_fraction > 0.65

    def _editor_tab_luminance_delta(self, frame: Frame) -> float | None:
        """Return Decks-tab luminance minus Collection-tab luminance."""

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
            return None
        # Compare luminance rather than a fixed RGB value so the same rule
        # remains valid across the two reviewed display resolutions.
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        return float(np.mean(left_gray) - np.mean(right_gray))

    def _has_editor_switcher_chrome(self, frame: Frame) -> bool:
        """Require the wide blue deck switcher below the top editor tabs.

        A seasonal K.H.A.O.S modal happened to satisfy the old two-patch
        luminance comparison.  The actual Decks/Collection editor has a
        saturated blue strip spanning most of y=13-19% on both reviewed
        phones, while that modal does not.  Combining the independent chrome
        signal with the tab delta keeps navigation fail-closed.
        """

        cv2, np = CardVision._cv()
        image = CardVision._image(frame)
        height = image.shape[0]
        roi = image[int(height * 0.13) : int(height * 0.19), :]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue = (
            (hsv[:, :, 0] >= 90)
            & (hsv[:, :, 0] <= 125)
            & (hsv[:, :, 1] >= 100)
            & (hsv[:, :, 2] >= 90)
        )
        return float(np.mean(blue)) >= 0.35

    def _looks_like_deck_editor(self, frame: Frame) -> bool:
        """Verify the top Decks tab is selected before inspecting deck slots."""

        delta = self._editor_tab_luminance_delta(frame)
        return (
            delta is not None
            and delta >= 18.0
            and self._has_editor_switcher_chrome(frame)
        )

    def _looks_like_collection_editor_top(self, frame: Frame) -> bool:
        """Verify the top Collection tab is selected and its switcher is visible."""

        delta = self._editor_tab_luminance_delta(frame)
        return (
            delta is not None
            and delta <= -18.0
            and self._has_editor_switcher_chrome(frame)
        )

    def _looks_like_lobby(self, frame: Frame) -> bool:
        """Recognize the main lobby before opening the card editor.

        The game surface is a custom canvas and does not expose reliable
        Android accessibility labels.  Keep this detector deliberately
        narrow: it only admits the large yellow battle control in the lower
        center of the portrait lobby.  Unknown screens remain fail-closed.
        """

        try:
            cv2, np = CardVision._cv()
            image = CardVision._image(frame)
        except AutomationError:
            return False
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        y0, y1 = int(height * 0.58), int(height * 0.88)
        x0, x1 = int(width * 0.12), int(width * 0.88)
        roi = hsv[y0:y1, x0:x1]
        if roi.size == 0:
            return False
        yellow = cv2.inRange(
            roi,
            np.array([12, 90, 120]),
            np.array([42, 255, 255]),
        )
        contours, _hierarchy = cv2.findContours(
            yellow,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
            area = candidate_width * candidate_height
            aspect = candidate_width / candidate_height if candidate_height else 0.0
            center_y = y0 + y + candidate_height / 2.0
            if (
                area >= int(width * height * 0.015)
                and candidate_width >= int(width * 0.25)
                and candidate_height >= int(height * 0.045)
                and 1.35 <= aspect <= 3.8
                and height * 0.62 <= center_y <= height * 0.86
            ):
                return True
        return False

    def _open_deck_editor_from_lobby(self) -> None:
        """Navigate from a positively recognized lobby to the Decks editor."""

        initial = self.screenshot()
        if self._looks_like_deck_editor(initial):
            return
        if not self._looks_like_lobby(initial):
            raise AutomationError(
                f"{self.profile.device_label} preparation requires a verified top Decks screen; "
                "the current screen is neither Decks nor a recognized lobby, so no input was sent"
            )

        self.tap(self.profile.collection_tab())
        self.wait_for(
            lambda candidate: (
                self._looks_like_deck_editor(candidate)
                or self._looks_like_collection_editor_top(candidate)
            ),
            timeout_s=5.0,
            description="card collection editor to become visible",
        )
        frame = self.screenshot()
        if self._looks_like_deck_editor(frame):
            return
        if not self._looks_like_collection_editor_top(frame):
            raise AutomationError(
                f"{self.profile.device_label} collection navigation reached an unverified screen"
            )
        self.tap(self.profile.deck_editor_tab())
        self.wait_for(
            self._looks_like_deck_editor,
            timeout_s=5.0,
            description="Decks editor to become selected",
        )

    @staticmethod
    def _find_editor_ok_button(frame: Frame) -> tuple[int, int] | None:
        """Locate the large blue editor OK button on a scrolled card picker."""

        cv2, np = CardVision._cv()
        image = CardVision._image(frame)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 80, 120]), np.array([115, 255, 255]))
        height, width = image.shape[:2]
        mask[: int(height * 0.78), :] = 0
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / h if h else 0.0
            if (
                area >= int(width * height * 0.004)
                and w >= int(width * 0.16)
                and h >= int(height * 0.03)
                and 1.5 <= aspect <= 3.5
                and int(width * 0.25) <= x <= int(width * 0.55)
            ):
                candidates.append((area, x, y, w, h))
        if not candidates:
            return None
        _area, x, y, w, h = max(candidates)
        return x + w // 2, y + h // 2

    @staticmethod
    def _find_card_upgrade_tutorial_close(frame: Frame) -> tuple[int, int] | None:
        """Find the close control of the reviewed card-upgrade tutorial.

        The German SM-G970F build can show a modal headed ``Verstärke deine
        Karten!`` while a collection card is being selected. It darkens the
        screen and presents a large purple panel with a red close button in
        the upper-right. Treat this as a modal, never as a card-selection
        confirmation; unrelated red badges must not enter recovery.
        """

        cv2, np = CardVision._cv()
        image = CardVision._image(frame)
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_a = cv2.inRange(
            hsv,
            np.array([0, 100, 80]),
            np.array([12, 255, 255]),
        )
        red_b = cv2.inRange(
            hsv,
            np.array([165, 100, 80]),
            np.array([180, 255, 255]),
        )
        red = cv2.bitwise_or(red_a, red_b)
        red[: int(height * 0.08), :] = 0
        red[int(height * 0.36) :, :] = 0
        # The reviewed close control is the red square at roughly (0.91W,
        # 0.18H). Other red/orange tutorial artwork must not be tappable.
        red[:, : int(width * 0.86)] = 0

        purple = cv2.inRange(
            hsv,
            np.array([115, 45, 35]),
            np.array([170, 255, 255]),
        )
        panel = purple[
            int(height * 0.12) : int(height * 0.88),
            int(width * 0.03) : int(width * 0.97),
        ]
        purple_fraction = float(np.mean(panel > 0)) if panel.size else 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        outer = np.concatenate(
            (
                gray[: int(height * 0.12)].ravel(),
                gray[int(height * 0.88) :].ravel(),
                gray[:, : int(width * 0.03)].ravel(),
                gray[:, int(width * 0.97) :].ravel(),
            )
        )
        # The tutorial dims the underlying editor almost to black. A normal
        # scrolled collection page can contain purple card art and a red
        # upgrade button, so the modal's global darkening is a required
        # second signal.
        if (
            purple_fraction < 0.25
            or float(np.mean(outer)) > 50.0
            or float(np.mean(outer < 45)) < 0.70
        ):
            return None

        contours, _hierarchy = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / h if h else 0.0
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            if (
                area >= int(width * height * 0.0008)
                and w >= int(width * 0.025)
                and h >= int(height * 0.018)
                and 0.55 <= aspect <= 2.2
                and width * 0.86 <= center_x <= width * 0.99
                and height * 0.10 <= center_y <= height * 0.27
            ):
                candidates.append((area, x, y, w, h))
        if not candidates:
            return None
        _area, x, y, w, h = max(candidates)
        return x + w // 2, y + h // 2

    def _dismiss_card_upgrade_tutorial(self, frame: Frame) -> bool:
        """Close the known upgrade tutorial and report whether it was shown."""

        close = self._find_card_upgrade_tutorial_close(frame)
        if close is None:
            return False
        self.tap(close)
        time.sleep(0.35)
        if self._find_card_upgrade_tutorial_close(self.screenshot()) is not None:
            raise AutomationError(
                f"{self.profile.device_label} card-upgrade tutorial did not close"
            )
        return True

    def recover_editor_top(self) -> None:
        """Recover only a positively identified deck/card editor surface."""

        self.clear_vendor_overlay()
        frame = self.screenshot()
        if self._looks_like_deck_editor(frame):
            return
        if not self._looks_like_collection_editor_top(frame):
            if self._find_editor_ok_button(frame) is None:
                raise AutomationError(
                    f"{self.profile.device_label} is not a verified scrolled card editor; no input sent"
                )
        self._ensure_deck_editor_top()

    def resume_open_remove_panel(self, slot: int) -> None:
        """Remove one explicitly reviewed selected deck card, then stop."""

        if not 0 <= slot < 8:
            raise AutomationError("resume remove slot must be from 0 through 7")
        frame = self.screenshot()
        if not self._looks_like_deck_editor(frame):
            raise AutomationError("open Remove-panel recovery requires verified top Decks")
        remove = self._find_red_button(frame, self.profile)
        if remove is None:
            raise AutomationError("open Remove-panel recovery found no verified red Remove button")
        self.tap(remove)
        time.sleep(0.35)
        self.wait_for(
            lambda candidate: self._slot_looks_empty(candidate, slot),
            timeout_s=3.0,
            description=f"slot {slot} to become empty during explicit recovery",
        )

    def _ensure_deck_editor_top(self) -> None:
        """Return a known editor workflow to the verified top Decks surface."""

        self._scroll_to_top()
        frame = self.screenshot()
        if self._looks_like_deck_editor(frame):
            return
        if not self._looks_like_collection_editor_top(frame):
            raise AutomationError(
                f"{self.profile.device_label} editor switcher is not visible after scrolling to top"
            )
        self.tap(self.profile.deck_editor_tab())
        self.wait_for(
            self._looks_like_deck_editor,
            timeout_s=3.0,
            description="deck editor to become selected",
        )

    def _remove_deck_slot(self, slot: int) -> None:
        """Remove one occupied deck slot through its verified action panel."""

        if not 0 <= slot < 8:
            raise AutomationError("deck removal slot must be from 0 through 7")
        self.tap(self.profile.deck_card_centers()[slot])
        time.sleep(0.25)
        remove = self._find_red_button(self.screenshot(), self.profile)
        if remove is not None:
            self.tap(remove)
            time.sleep(0.35)
            self.wait_for(
                lambda candidate: self._slot_looks_empty(candidate, slot),
                timeout_s=3.0,
                description=f"deck slot {slot} to become empty after removal",
            )
            return
        if not self._slot_looks_empty(self.screenshot(), slot):
            raise AutomationError(
                f"{self.profile.device_label} slot {slot} is occupied but has no verified Remove panel"
            )

    def configure_fixed_deck(
        self,
        *,
        target_deck: Sequence[str] = FIXED_HOG_CYCLE_DECK,
        max_swipes: int = 36,
    ) -> tuple[str, ...]:
        """Replace the active deck until it is exactly ``target_deck``.

        The default remains the reviewed Hog cycle for backwards
        compatibility.  Campaign cases pass their own validated ordered
        deck so one or more cards can be changed without weakening the
        screenshot identity gates.
        """

        target = tuple(str(card).lower() for card in target_deck)
        if len(target) != 8 or len(set(target)) != 8 or any(not card for card in target):
            raise AutomationError("target fixed deck must contain eight unique non-empty cards")
        self._validate_device_deck_constraints(target)

        self.clear_vendor_overlay()
        self._open_deck_editor_from_lobby()
        initial_frame = self.screenshot()
        if not self._looks_like_deck_editor(initial_frame):
            raise AutomationError(
                f"{self.profile.device_label} preparation requires a verified top Decks screen; "
                "no input was sent to Clash Royale"
            )
        # A rejected or manually interrupted preparation may leave a card
        # action panel open. Reject rather than using Android Back, which can
        # open the app-exit dialog on some reviewed UI surfaces.
        panel_frame = self.screenshot()
        if self._find_red_button(panel_frame, self.profile) is not None:
            raise AutomationError(
                f"{self.profile.device_label} preparation requires the deck editor with no open card panel"
            )
        # Work slot-by-slot.  A global "missing card" calculation is unsafe
        # while the deck UI is changing because an unreadable card could be
        # mistaken for a missing card and create duplicates.  The desired
        # card for each position is known, so each position is independently
        # verified, replaced, and verified again.
        selection_verified_slots: dict[int, str] = {}
        for slot, replacement in enumerate(target):
            self._ensure_deck_editor_top()
            frame = self.screenshot()
            slot_match = self.vision.match_slot(frame, self.profile, slot)
            if (
                slot_match is not None and slot_match.card_id == replacement
            ) or self._slot_matches_expected(frame, slot, replacement):
                continue

            donor_slot = self._find_existing_target_slot(
                frame,
                replacement,
                exclude_slot=slot,
            )
            if donor_slot is not None:
                self._remove_deck_slot(donor_slot)
                self._ensure_deck_editor_top()
                frame = self.screenshot()
                slot_match = self.vision.match_slot(frame, self.profile, slot)
                if (
                    slot_match is not None and slot_match.card_id == replacement
                ) or self._slot_matches_expected(frame, slot, replacement):
                    continue

            # The game opens the removal panel only after the card itself is
            # selected.  Empty slots have no removal panel and go directly to
            # the collection; this is why the panel is located after the tap.
            self._remove_deck_slot(slot)

            self._ensure_deck_editor_top()
            card_point = self._find_and_tap_collection_card(replacement, max_swipes=max_swipes)
            if card_point is None:
                raise AutomationError(
                    f"{self.profile.device_label} could not locate collection card {replacement}"
                )
            self._ensure_deck_editor_top()
            self.wait_for(
                lambda candidate: not self._slot_looks_empty(candidate, slot),
                timeout_s=3.0,
                description=f"deck slot {slot} to become occupied by {replacement}",
            )
            # Some current builds use updated card art while retaining the
            # same card identity.  The collection candidate already passed a
            # strict identity/margin recheck and the game returned to the
            # verified deck editor after the explicit Use/O.K. action; retain
            # that identity when the legacy art matcher cannot recognize the
            # updated slot art.
            selection_verified_slots[slot] = replacement

        self._ensure_deck_editor_top()
        final_frame = self.screenshot()
        final = self.vision.deck_matches(final_frame, self.profile)
        final_ids = tuple(
            selection_verified_slots[slot]
            if slot in selection_verified_slots and not self._slot_looks_empty(final_frame, slot)
            else (
                match.card_id
                if match is not None
                else (
                    target[slot]
                    if self._slot_matches_expected(final_frame, slot, target[slot])
                    else "unknown"
                )
            )
            for slot, match in enumerate(final)
        )
        if final_ids != target:
            raise AutomationError(
                f"{self.profile.device_label} fixed deck verification failed: {final_ids!r}"
            )
        self.tap(self.profile.ok_button())
        time.sleep(0.5)
        return final_ids

    def _find_and_tap_collection_card(self, card_id: str, *, max_swipes: int) -> tuple[int, int] | None:
        seen_hashes: set[str] = set()
        last_rejection: str | None = None
        search_trace: list[str] = []
        for _attempt in range(max_swipes + 1):
            frame = self.screenshot()
            payload_hash = frame.payload_hash or ""
            if payload_hash in seen_hashes:
                raise AutomationError(
                    f"{self.profile.device_label} collection did not move while searching for {card_id}"
                )
            seen_hashes.add(payload_hash)
            matches = self.vision.find_collection_card_candidates(
                frame,
                self.profile,
                card_id,
                scrolled=_attempt > 0,
                threshold=self._collection_candidate_threshold(card_id),
            )
            if not matches:
                search_trace.append(f"page={_attempt}:candidates=0")
            for match in matches:
                pre_identities = self.vision.rank_card_identities_near(
                    frame,
                    match.center,
                    limit=2,
                )
                pre_winner = pre_identities[0] if pre_identities else None
                pre_runner_up = pre_identities[1] if len(pre_identities) > 1 else None
                pre_margin = (
                    pre_winner.score - pre_runner_up.score
                    if pre_winner is not None and pre_runner_up is not None
                    else 1.0
                )
                if (
                    pre_winner is None
                    or pre_winner.card_id != card_id
                    or pre_winner.score < self._minimum_collection_identity_score(card_id)
                    or pre_margin < 0.12
                ):
                    identity_summary = (
                        "none"
                        if pre_winner is None
                        else f"{pre_winner.card_id}:{pre_winner.score:.3f}/margin:{pre_margin:.3f}"
                    )
                    last_rejection = (
                        f"{self.profile.device_label} rejected collection candidate {card_id} "
                        f"at {match.center}: pre-tap identity {identity_summary}"
                    )
                    search_trace.append(
                        f"page={_attempt}:candidate={match.center}:{match.score:.3f}:"
                        f"pre_identity={identity_summary}:rejected"
                    )
                    continue
                self.tap(match.center)
                time.sleep(0.25)
                selected_frame = self.screenshot()
                if self._dismiss_card_upgrade_tutorial(selected_frame):
                    # The modal interrupts the card-selection state. Do not
                    # infer that the underlying candidate was selected; stop
                    # this preparation so the operator can review the deck
                    # rather than risk committing the wrong card.
                    raise AutomationError(
                        f"{self.profile.device_label} card-upgrade tutorial appeared while "
                        f"selecting {card_id}; candidate {match.center} was not accepted"
                    )
                identities = self.vision.rank_card_identities_near(
                    selected_frame,
                    match.center,
                    limit=2,
                )
                winner = identities[0] if identities else None
                runner_up = identities[1] if len(identities) > 1 else None
                identity_margin = (
                    winner.score - runner_up.score
                    if winner is not None and runner_up is not None
                    else 1.0
                )
                post_identity_verified = not (
                    winner is None
                    or winner.card_id != card_id
                    or winner.score < self._minimum_collection_identity_score(card_id)
                    or identity_margin < 0.12
                )
                use = self._find_yellow_button(
                    selected_frame,
                    self.profile,
                    anchor=match.center,
                )
                if use is None:
                    use = self._find_editor_ok_button(selected_frame)
                if not post_identity_verified and use is None:
                    identity_summary = (
                        "none"
                        if winner is None
                        else f"{winner.card_id}:{winner.score:.3f}/margin:{identity_margin:.3f}"
                    )
                    last_rejection = (
                        f"{self.profile.device_label} rejected collection candidate {card_id} "
                        f"at {match.center}: selected identity {identity_summary}"
                    )
                    search_trace.append(
                        f"page={_attempt}:candidate={match.center}:{match.score:.3f}:"
                        f"identity={identity_summary}:rejected"
                    )
                    self._dismiss_collection_selection(match.center)
                    continue
                if not post_identity_verified:
                    search_trace.append(
                        f"page={_attempt}:candidate={match.center}:{match.score:.3f}:"
                        f"pre_identity={pre_winner.card_id}:{pre_winner.score:.3f}/"
                        f"margin:{pre_margin:.3f}:post_layout_shift"
                    )
                # The current German SM-G970F build confirms a selected
                # collection card with a blue ``O.K.`` button instead of the
                # yellow ``Use``/``Verwenden`` control used by the earlier
                # reviewed layout.  Accept either only after the selected
                # card's identity passed the strict pre-tap identity/margin
                # gate. Some full-screen info panels move the artwork away
                # from its original anchor, making a post-tap recheck
                # geometrically impossible even though the action surface is
                # verified.
                if use is None:
                    last_rejection = (
                        f"{self.profile.device_label} selected {card_id} without a verified Use/O.K. button"
                    )
                    search_trace.append(
                        f"page={_attempt}:candidate={match.center}:{match.score:.3f}:"
                        f"identity={winner.card_id}:{winner.score:.3f}:use=missing"
                    )
                    self._dismiss_collection_selection(match.center)
                    continue
                self.tap(use)
                time.sleep(0.35)
                return match.center
            if _attempt < max_swipes:
                self.swipe_up()
                self._wait_for_scroll_settled()
        if last_rejection is not None:
            raise AutomationError(
                f"{last_rejection}; search_trace=[{' | '.join(search_trace)}]"
            )
        return None

    def _dismiss_collection_selection(self, anchor: tuple[int, int]) -> None:
        """Close a selected collection card without invoking Android Back.

        In the reviewed Clash Royale UI, tapping the selected card toggles its
        action panel. Android Back instead opens the app-exit confirmation,
        which is unsafe recovery behavior even while the game is in the
        lobby. If the panel remains after the toggle, reject rather than
        attempting a broader gesture.
        """

        self.tap(anchor)
        # Selection dismissal is animated on the ASUS build.  A single
        # immediate screenshot can still contain the old blue O.K. panel
        # even though the tap has already been accepted, which used to turn
        # a safe recovery into a false rejection.
        time.sleep(0.20)
        try:
            self.wait_for(
                lambda candidate: self._looks_like_deck_editor(candidate)
                and self._find_red_button(candidate, self.profile) is None,
                timeout_s=0.75,
                poll_s=0.15,
                description="collection selection to close after card toggle",
            )
            return
        except AutomationError:
            pass
        frame = self.screenshot()
        # Some builds close the selection immediately and return to the top
        # Decks surface. In that state the generic blue-button detector can
        # mistake the editor chrome for an O.K. button, so prefer the stronger
        # screen-state proof first.
        if self._looks_like_deck_editor(frame) and self._find_red_button(frame, self.profile) is None:
            return
        # On the current German surface tapping the card again does not close
        # the selection panel; the reviewed blue O.K. control does.  Use it
        # only when the screenshot positively identifies that control.
        ok_button = self._find_editor_ok_button(frame)
        if ok_button is not None:
            self.tap(ok_button)
            try:
                self.wait_for(
                    lambda candidate: (
                        self._looks_like_deck_editor(candidate)
                        and self._find_red_button(candidate, self.profile) is None
                    ),
                    timeout_s=2.5,
                    poll_s=0.15,
                    description="deck editor after collection selection dismissal",
                )
            except AutomationError as error:
                raise AutomationError(
                    f"{self.profile.device_label} could not dismiss the selected collection card"
                ) from error
            return
        if self._find_yellow_button(frame, self.profile, anchor=anchor) is not None:
            raise AutomationError(
                f"{self.profile.device_label} could not dismiss the selected collection card"
            )

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
        # A card in the second deck row opens its action panel lower than a
        # first-row card on the current 1080x2280 layout.
        mask[int(height * 0.82) :, :] = 0
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int, int]] = []
        expected_y = profile.remove_button_y_norm * height
        expected_ys = (expected_y, expected_y + height * 0.18)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / h if h else 0.0
            center_y = y + h / 2.0
            if (
                area >= int(width * height * 0.004)
                and w >= int(width * 0.15)
                and h >= int(height * 0.035)
                and h <= int(height * 0.09)
                and 1.35 <= aspect <= 3.50
                and min(abs(center_y - expected) for expected in expected_ys) <= height * 0.12
            ):
                candidates.append((area, x, y, w, h))
        if not candidates:
            return None
        _area, x, y, w, h = max(candidates)
        return x + w // 2, y + h // 2

    @staticmethod
    def _find_yellow_button(
        frame: Frame,
        profile: UiProfile,
        *,
        anchor: tuple[int, int] | None = None,
    ) -> tuple[int, int] | None:
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
            center_x = x + w / 2.0
            if (
                area >= int(width * height * 0.004)
                and x <= int(width * 0.85)
                and w >= int(width * 0.15)
                and h >= int(height * 0.035)
                and h <= int(height * 0.065)
                and 1.80 <= aspect <= 3.50
                and (
                    anchor is None
                    or (
                        abs(center_x - anchor[0]) <= width * 0.20
                        # On the Samsung picker the selected card's action
                        # panel can place Wählen only ~80px below the card
                        # anchor. Keep the relation directional, but do not
                        # require the larger ASUS-style gap.
                        and height * 0.015 <= center_y - anchor[1] <= height * 0.35
                    )
                )
            ):
                candidates.append((area, x, y, w, h))
        if not candidates:
            return None
        _area, x, y, w, h = max(candidates)
        return x + w // 2, y + h // 2

    def open_testspiel_solo(
        self,
        *,
        target_player_name: str | None = None,
        fixed_deck_order: bool = False,
        fixed_deck_toggle_point: tuple[int, int] | None = None,
        test_match_start_point: tuple[int, int] | None = None,
        long_press_ms: int = FIXED_DECK_LONG_PRESS_MS,
        opening_hand: Sequence[str] = FIXED_HOG_CYCLE_OPENING_HAND,
        replacement_order: Sequence[str] = FIXED_HOG_CYCLE_REPLACEMENT_ORDER,
    ) -> dict[str, object]:
        """Open Testspiel/Solokampf and optionally arm fixed deck order.

        A normal tap starts the ordinary challenge immediately.  A long press
        opens the hidden Testspiel options where Clash Royale exposes fixed
        deck order.  The option and final start controls are intentionally
        supplied as reviewed points: the game renders this menu as a custom
        surface and its location varies between builds.  If those points are
        omitted, the method stops after the long press and fails closed rather
        than guessing a toggle or starting the wrong mode.
        """

        opening = tuple(str(card).lower() for card in opening_hand)
        replacements = tuple(str(card).lower() for card in replacement_order)
        if len(opening) != 4 or len(replacements) != 4 or len(set(opening + replacements)) != 8:
            raise AutomationError("fixed-deck opening and replacement order must contain eight unique cards")

        if target_player_name is None:
            # Compatibility path for the low-level offline primitive. The
            # connected coordinator below verifies the community surface
            # before navigating it.
            self.tap(self.profile.social_tab())
            time.sleep(0.8)
        target_receipt: ActionReceipt | None = None
        target_row: tuple[int, int] | None = None
        challenge_menu_point: tuple[int, int] | None = None
        if target_player_name is None:
            # Compatibility path for the low-level offline primitive. The
            # connected coordinator always supplies the named account below.
            self.tap(self.profile.online_row())
            time.sleep(0.35)
            self.tap(self.profile.challenge_menu())
        else:
            # First try the current surface. This makes the operation
            # idempotent when a previous safe recovery already left the
            # community list open, and avoids tapping a navigation arrow just
            # because the community heading was momentarily missed by OCR.
            last_error: AutomationError | None = None
            last_frame: Frame | None = None
            for attempt in range(3):
                last_frame = self.screenshot()
                try:
                    target_row = self._find_online_player_row(last_frame, target_player_name)
                    break
                except AutomationError as error:
                    last_error = error
                    if attempt < 2:
                        # Player presence cards animate independently of the
                        # game clock. A short settled-frame retry avoids
                        # navigating away from an already-open Community list
                        # when one screenshot catches a transition frame.
                        time.sleep(0.25)
            else:
                if last_frame is None or not self._looks_like_lobby(last_frame):
                    raise AutomationError(
                        f"{self.profile.device_label} could not locate online player "
                        f"{target_player_name!r} on the current surface; refusing to "
                        "navigate an unrecognized screen"
                    ) from last_error
                self.tap(self.profile.social_tab())
                time.sleep(0.8)
                for attempt in range(3):
                    last_frame = self.screenshot()
                    try:
                        target_row = self._find_online_player_row(last_frame, target_player_name)
                        break
                    except AutomationError as error:
                        last_error = error
                        if attempt < 2:
                            time.sleep(0.25)
                else:
                    raise AutomationError(
                        f"{self.profile.device_label} could not locate online player "
                        f"{target_player_name!r} after opening Community"
                    ) from last_error
            target_receipt = self.tap(target_row)
            time.sleep(0.35)
            selected = self.screenshot()
            challenge_error: AutomationError | None = None
            for attempt in range(3):
                if not self._frame_contains_player_name(selected, target_player_name):
                    challenge_error = AutomationError(
                        f"{self.profile.device_label} row tap did not open the requested "
                        f"player profile {target_player_name!r}"
                    )
                else:
                    try:
                        challenge_menu_point = self._find_challenge_menu_button(selected)
                        break
                    except AutomationError as error:
                        challenge_error = error
                if attempt < 2:
                    time.sleep(0.25)
                    selected = self.screenshot()
            else:
                raise AutomationError(
                    f"{self.profile.device_label} selected player popup did not expose "
                    f"a verified Testspiel control for {target_player_name!r}"
                ) from challenge_error
            self.tap(challenge_menu_point)
        time.sleep(0.6)
        if not fixed_deck_order:
            receipt = self.tap(self.profile.solo_battle())
            result: dict[str, object] = {
                "mode": "testspiel_solo",
                "fixed_deck_order": False,
                "opening_hand": list(opening),
                "challenge_receipt": receipt.to_dict(),
            }
            if target_player_name is not None:
                result.update(
                    {
                        "target_player_name": target_player_name,
                        "target_row": list(target_row or ()),
                        "target_row_receipt": target_receipt.to_dict() if target_receipt else None,
                        "challenge_menu_point": list(challenge_menu_point or ()),
                    }
                )
            return result

        long_press = self.long_press(self.profile.solo_battle(), duration_ms=long_press_ms)
        if fixed_deck_toggle_point is None:
            raise AutomationError(
                f"{self.profile.device_label} opened Testspiel Solokampf options, but no reviewed "
                "fixed-deck toggle point was supplied"
            )
        # The fixed-order preference persists between friendly matches. A
        # coordinator rerun must not blindly invert an already-enabled
        # switch, or the four-card opening-hand contract would silently be
        # lost. The low-level compatibility path keeps its historical tap;
        # connected named-player runs perform the reviewed visual check.
        already_enabled = False
        if target_player_name is not None:
            already_enabled = self._fixed_deck_order_enabled(
                self.screenshot(),
                fixed_deck_toggle_point,
            )
        toggle: ActionReceipt | None = None
        if not already_enabled:
            toggle = self.tap(fixed_deck_toggle_point)
            if target_player_name is not None:
                time.sleep(0.25)
                if not self._fixed_deck_order_enabled(
                    self.screenshot(),
                    fixed_deck_toggle_point,
                ):
                    raise AutomationError(
                        f"{self.profile.device_label} fixed-deck switch did not verify as enabled"
                    )
        result: dict[str, object] = {
            "mode": "testspiel_solo",
            "fixed_deck_order": True,
            "opening_hand": list(opening),
            "replacement_order": list(replacements),
            "long_press_receipt": long_press.to_dict(),
            "toggle_receipt": toggle.to_dict() if toggle is not None else None,
            "fixed_deck_already_enabled": already_enabled,
        }
        if target_player_name is not None:
            result.update(
                {
                    "target_player_name": target_player_name,
                    "target_row": list(target_row or ()),
                    "target_row_receipt": target_receipt.to_dict() if target_receipt else None,
                    "challenge_menu_point": list(challenge_menu_point or ()),
                }
            )
        if test_match_start_point is None:
            result["started"] = False
            result["state"] = "fixed_deck_options_enabled"
            return result
        start = self.tap(test_match_start_point)
        result["started"] = True
        result["state"] = "testspiel_waiting_for_opponent"
        result["start_receipt"] = start.to_dict()
        return result

    def send_friendly_challenge(
        self,
        *,
        target_player_name: str | None = None,
        fixed_deck_order: bool = False,
        fixed_deck_toggle_point: tuple[int, int] | None = None,
        test_match_start_point: tuple[int, int] | None = None,
        long_press_ms: int = FIXED_DECK_LONG_PRESS_MS,
        opening_hand: Sequence[str] = FIXED_HOG_CYCLE_OPENING_HAND,
        replacement_order: Sequence[str] = FIXED_HOG_CYCLE_REPLACEMENT_ORDER,
    ) -> dict[str, object]:
        """Start a Testspiel challenge, optionally with fixed deck order."""

        return self.open_testspiel_solo(
            target_player_name=target_player_name,
            fixed_deck_order=fixed_deck_order,
            fixed_deck_toggle_point=fixed_deck_toggle_point,
            test_match_start_point=test_match_start_point,
            long_press_ms=long_press_ms,
            opening_hand=opening_hand,
            replacement_order=replacement_order,
        )

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
        expected_slot: int | None = None,
        capture: CaptureFrameRecorder | None = None,
    ) -> tuple[int, ActionReceipt, ActionReceipt]:
        if expected_slot is not None and not 0 <= expected_slot < 4:
            raise AutomationError(
                f"{self.profile.device_label} expected hand slot must be from 0 through 3"
            )
        if (
            calibration.screen_width_px != self.profile.screen_width_px
            or calibration.screen_height_px != self.profile.screen_height_px
        ):
            raise AutomationError(
                f"{self.profile.device_label} calibration dimensions "
                f"{calibration.screen_width_px}x{calibration.screen_height_px} do not match "
                f"the native screen {self.profile.screen_width_px}x{self.profile.screen_height_px}"
            )
        if calibration.hand_slot_count != 4:
            raise AutomationError(
                f"{self.profile.device_label} reviewed battle hand must contain four slots"
            )
        if self.action_frame_provider is None:
            frame = self.record(capture)
        else:
            frame = self.action_frame_provider()
            if capture is not None:
                capture.record_frame(frame)
        find_hand_card_in_slot = getattr(self.vision, "find_hand_card_in_slot", None)
        if expected_slot is not None and callable(find_hand_card_in_slot):
            match = find_hand_card_in_slot(
                frame,
                self.profile,
                card_id,
                expected_slot,
                hand_px=calibration.hand_px,
            )
        else:
            # Keep compatibility with reviewed/custom vision adapters that
            # expose only the original full-hand matcher.
            match = self.vision.find_hand_card(frame, self.profile, card_id)
        if match is None or match.score < self.vision.threshold:
            raise AutomationError(
                f"{self.profile.device_label} could not identify {card_id} in the current hand"
            )
        hand_x, _hand_y, hand_width, _hand_height = calibration.hand_px
        slot = min(
            3,
            max(
                0,
                int((match.center[0] - hand_x) / (hand_width / calibration.hand_slot_count)),
            ),
        )
        if expected_slot is not None and slot != expected_slot:
            raise AutomationError(
                f"{self.profile.device_label} detected {card_id} in hand slot {slot}, "
                f"expected reviewed slot {expected_slot}; no placement was sent"
            )
        selected = self.controller.tap_screen(*calibration.slot_to_pixel(slot))
        if not selected.accepted:
            raise AutomationError(f"{self.profile.device_label} rejected card selection for {card_id}")
        placed = self.controller.tap_screen(*calibration.cell_to_pixel(arena_cell))
        if not placed.accepted:
            raise AutomationError(f"{self.profile.device_label} rejected placement for {card_id}")
        # The live prototype's next loop iteration takes the next source
        # screenshot immediately. Keep this extra post-action frame for
        # evidence capture, but avoid duplicating it in the normal controller
        # path where no recorder is attached.
        if capture is not None:
            if self.action_frame_provider is None:
                self.record(capture)
            else:
                capture.record_frame(self.action_frame_provider())
        return slot, selected, placed


@dataclass(frozen=True, slots=True)
class AutonomousSessionConfig:
    repository_root: Path
    raw_media_root: Path
    run_id: str
    target_player_name: str | None = DEFAULT_FRIENDLY_TARGET_PLAYER_NAME
    max_collection_swipes: int = 36
    lifecycle_timeout_s: float = 60.0
    result_timeout_s: float = 330.0
    fixed_deck_order: bool = False
    fixed_deck_toggle_point: tuple[int, int] | None = None
    test_match_start_point: tuple[int, int] | None = None
    fixed_deck_long_press_ms: int = FIXED_DECK_LONG_PRESS_MS
    capture_time_limit_s: int = 360
    retention_manifest: Path | None = None
    decks: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {"A": FIXED_HOG_CYCLE_DECK, "B": FIXED_HOG_CYCLE_DECK}
    )

    def deck_for_side(self, side: str) -> tuple[str, ...]:
        try:
            deck = tuple(self.decks[side])
        except (KeyError, TypeError) as error:
            raise AutomationError(f"missing configured deck for side {side}") from error
        if len(deck) != 8 or len(set(deck)) != 8:
            raise AutomationError(f"configured deck for side {side} must contain eight unique cards")
        return deck


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
        observation_waiter: ObservationWaiter | None = None,
    ) -> None:
        self.spec = spec
        self._validate_probe_spec()
        self.controllers = {"A": controller_a, "B": controller_b}
        self.calibrations = {"A": calibration_a, "B": calibration_b}
        self.config = config
        self.observation_waiter = observation_waiter
        self.infos = self._preflight_devices()
        self.profiles = {
            side: UiProfile.for_device(side, self.infos[side]) for side in ("A", "B")
        }
        vision = CardVision(template_root)
        self.phones = {
            side: AutonomousPhone(
                self.controllers[side],
                self.profiles[side],
                vision,
                device_model=self.infos[side].model,
            )
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
        if not self.spec.actions:
            raise AutomationError("autonomous controller requires at least one reviewed action")
        decks = self.spec.initial_conditions.decks
        seen_ids: set[str] = set()
        previous_time = -1
        for action in self.spec.actions:
            if action.action_id in seen_ids:
                raise AutomationError(f"autonomous action IDs are not unique: {action.action_id}")
            seen_ids.add(action.action_id)
            deck = decks.get(action.side)
            if deck is None or action.card_id not in deck:
                raise AutomationError(
                    f"autonomous action {action.action_id} card {action.card_id} is absent from side {action.side} deck"
                )
            if action.card_slot is None:
                raise AutomationError(
                    f"autonomous action {action.action_id} requires a reviewed opening-hand card_slot"
                )
            if action.card_slot >= 4 or deck[action.card_slot] != action.card_id:
                raise AutomationError(
                    f"autonomous action {action.action_id} must identify its fixed-deck opening-hand slot"
                )
            if action.trigger.type.value == "match_time_us":
                if action.trigger.value < previous_time:
                    raise AutomationError("autonomous actions must be ordered by non-decreasing match time")
                previous_time = action.trigger.value
            elif action.trigger.event is None:
                raise AutomationError(f"autonomous action {action.action_id} lacks an observation event")

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
                target_deck=self.config.deck_for_side(side),
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

    def _require_initial_lobby(
        self,
        observations: list[Mapping[str, str]],
    ) -> dict[str, LifecycleState]:
        """Reject unknown/non-lobby starts before any Clash Royale input."""

        observed, states = self._observe_pair()
        observations.append({side: states[side].value for side in ("A", "B")})
        if observed is not LifecycleState.LOBBY:
            detail = ", ".join(
                f"{side}={state.value}" for side, state in sorted(states.items())
            )
            raise AutomationError(
                "autonomous run requires both phones to be positively detected in lobby "
                f"before any game input ({detail})"
            )
        return states

    def run(self) -> PhysicalRunResult:
        capture_root = self.config.raw_media_root / "physical_lab" / self.config.run_id / "raw"
        captures = {
            side: ScrcpyScreenCapture(
                self.controllers[side],
                capture_root / f"{side}.mp4",
                time_limit_s=self.config.capture_time_limit_s,
            )
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
        battle_started: int | None = None
        try:
            for capture in captures.values():
                capture.start()
            observation_waiter = self.observation_waiter
            if observation_waiter is None and any(
                action.trigger.type.value == "after_observation" for action in self.spec.actions
            ):
                observation_waiter = HogBridgeObservationWaiter(
                    LiveTrackObservationSource(
                        self.phones["A"].screenshot,
                        recorder=captures["A"],
                    )
                )
            for side in ("A", "B"):
                self.phones[side].record(captures[side])

            states = self._require_initial_lobby(observations)
            transitions.append(self._transition(current, LifecycleState.LOBBY, states))
            current = LifecycleState.LOBBY

            self.phones["B"].send_friendly_challenge(
                target_player_name=self.config.target_player_name,
                fixed_deck_order=self.config.fixed_deck_order,
                fixed_deck_toggle_point=self.config.fixed_deck_toggle_point,
                test_match_start_point=self.config.test_match_start_point,
                long_press_ms=self.config.fixed_deck_long_press_ms,
                opening_hand=self.config.deck_for_side("B")[:4],
                replacement_order=self.config.deck_for_side("B")[4:],
            )
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
            states = self._wait_pair(LifecycleState.BATTLE, observations=observations)
            transitions.append(self._transition(current, LifecycleState.BATTLE, states))
            current = LifecycleState.BATTLE
            # This is a conservative workstation-time anchor taken only after
            # both reviewed lifecycle detectors positively report BATTLE.  It
            # is not an OCR-derived game-clock value, so connected runs remain
            # candidate evidence until a reviewed match-clock mapping is
            # attached during ingest.
            battle_started = monotonic_time_us()

            for action in self.spec.actions:
                trigger = action.trigger
                if trigger.type.value == "match_time_us":
                    remaining_us = trigger.value - (monotonic_time_us() - battle_started)
                    if remaining_us < 0:
                        raise AutomationError(
                            f"action {action.action_id} missed its reviewed match-time boundary"
                        )
                    if remaining_us:
                        time.sleep(remaining_us / 1_000_000)
                else:
                    if observation_waiter is None:
                        raise AutomationError(
                            f"action {action.action_id} requires an observation waiter"
                        )
                    try:
                        observed_boundary_time = observation_waiter(
                            trigger.event or "",
                            trigger.value,
                            self.spec.duration_us,
                        )
                    except (PhysicalLabError, TimeoutError) as error:
                        raise AutomationError(
                            f"after-observation trigger for {action.action_id} failed: {error}"
                        ) from error
                    if type(observed_boundary_time) is not int or observed_boundary_time < 0:
                        raise AutomationError(
                            f"observation waiter returned an invalid boundary for {action.action_id}"
                        )
                slot, selected, placed = self.phones[action.side].select_and_place(
                    action.card_id,
                    calibration=self.calibrations[action.side],
                    arena_cell=action.arena_cell,
                    expected_slot=action.card_slot,
                    capture=captures[action.side],
                )
                actual_time = self._placement_match_time(
                    placed,
                    battle_started_at_monotonic_us=battle_started,
                )
                action_times[action.action_id] = actual_time
                actions.append(
                    self._action_entry(
                        action,
                        slot=slot,
                        actual_time=actual_time,
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
            if current in {
                LifecycleState.CHALLENGE_ACCEPTED,
                LifecycleState.LOADING,
                LifecycleState.BATTLE,
                LifecycleState.RESULT,
                LifecycleState.ARCHIVED,
            }:
                try:
                    current = self._recover_without_cancelling_match(
                        current,
                        observations=observations,
                        transitions=transitions,
                    )
                except (AutomationError, PhysicalLabError, CalibrationError, OSError) as recovery_error:
                    rejection_reasons.append(
                        f"active match recovery failed without cancellation: {recovery_error}"
                    )
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
            clock_provenance=ClockProvenance(
                battle_start_monotonic_us=battle_started,
                capture_start_monotonic_us={
                    side: manifest.started_at_monotonic_us
                    for side, manifest in capture_manifests.items()
                },
            ),
        )
        write_run_artifacts(
            result,
            repository_root=self.config.repository_root,
            retention_manifest=(
                self.config.retention_manifest
                or self.config.repository_root / "outputs/simulator/fidelity_media/retention.json"
            ),
        )
        return result

    @staticmethod
    def _placement_match_time(
        receipt: ActionReceipt,
        *,
        battle_started_at_monotonic_us: int,
    ) -> int:
        """Map an accepted placement receipt onto the provisional battle axis."""

        if not receipt.accepted:
            raise AutomationError("cannot time a rejected placement receipt")
        if receipt.completed_at_monotonic_us < battle_started_at_monotonic_us:
            raise AutomationError("placement receipt predates the reviewed BATTLE boundary")
        return receipt.completed_at_monotonic_us - battle_started_at_monotonic_us

    def _recover_without_cancelling_match(
        self,
        current: LifecycleState,
        *,
        observations: list[Mapping[str, str]],
        transitions: list[LifecycleTransition],
    ) -> LifecycleState:
        """Wait out an accepted/active match, then return through normal UI."""

        if current in {
            LifecycleState.CHALLENGE_ACCEPTED,
            LifecycleState.LOADING,
            LifecycleState.BATTLE,
        }:
            states = self._wait_pair(
                LifecycleState.RESULT,
                observations=observations,
                timeout_s=self.config.result_timeout_s,
            )
            transitions.append(self._transition(current, LifecycleState.RESULT, states))
            current = LifecycleState.RESULT
        if current is LifecycleState.RESULT:
            self.phones["A"].dismiss_result()
            self.phones["B"].dismiss_result()
            states = self._wait_pair(LifecycleState.ARCHIVED, observations=observations)
            transitions.append(self._transition(current, LifecycleState.ARCHIVED, states))
            current = LifecycleState.ARCHIVED
        if current is LifecycleState.ARCHIVED:
            for side in ("A", "B"):
                self.phones[side].return_to_lobby()
            states = self._wait_pair(LifecycleState.RECOVERY, observations=observations)
            transitions.append(self._transition(current, LifecycleState.RECOVERY, states))
            current = LifecycleState.RECOVERY
        return current

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

    if (
        not serial_a
        or not serial_b
        or any(character.isspace() for character in serial_a)
        or any(character.isspace() for character in serial_b)
    ):
        raise AutomationError("both physical serials are required")
    if serial_a == serial_b:
        raise AutomationError("physical sides A and B must use distinct devices")
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
    "DEFAULT_FRIENDLY_TARGET_PLAYER_NAME",
    "CardMatch",
    "CardVision",
    "FIXED_DECK_LONG_PRESS_MS",
    "FIXED_HOG_CYCLE_DECK",
    "FIXED_HOG_CYCLE_OPENING_HAND",
    "FIXED_HOG_CYCLE_REPLACEMENT_ORDER",
    "SAMSUNG_HERO_MUSKETEER_MODEL_IDS",
    "SAMSUNG_SPECIAL_CARD_FIRST_THREE_IDS",
    "SAMSUNG_REGULAR_MUSKETEER_HUMAN_SLOT_RANGE",
    "SAMSUNG_REGULAR_MUSKETEER_MIN_SLOT",
    "SAMSUNG_COLLECTION_IDENTITY_MIN_SCORES",
    "ASUS_REGULAR_MUSKETEER_MODEL_IDS",
    "ASUS_REGULAR_MUSKETEER_MIN_SLOT",
    "ASUS_REGULAR_MUSKETEER_HUMAN_SLOT_RANGE",
    "MUSKETEER_COLLECTION_CANDIDATE_MIN_SCORE",
    "ASUS_COLLECTION_CANDIDATE_MIN_SCORES",
    "ASUS_COLLECTION_IDENTITY_MIN_SCORES",
    "UiProfile",
    "bind_spec_to_devices",
]
