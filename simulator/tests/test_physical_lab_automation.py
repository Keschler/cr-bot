from __future__ import annotations

from pathlib import Path

import pytest

from simulator.physical_lab import CardVision, UiProfile
from simulator.physical_lab.automation import AutonomousPhone, FIXED_HOG_CYCLE_DECK
from simulator.physical_lab.devices import Frame


def _card_frame(*, card_id: str, x: int, y: int, width: int = 250, height: int = 300) -> Frame:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    root = Path("assets/templates/cr-api-assets/cards-gold")
    asset_name = "the-log" if card_id == "log" else card_id
    asset = cv2.imread(str(root / f"{asset_name}.png"), cv2.IMREAD_COLOR)
    assert asset is not None
    asset = cv2.resize(asset, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((2400, 1080, 3), dtype=np.uint8)
    canvas[y : y + height, x : x + width] = asset
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    return Frame(source_device="A", frame_index=0, workstation_monotonic_us=0, payload=encoded.tobytes())


def test_card_vision_accepts_a_high_confidence_deck_card() -> None:
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    center_x, center_y = profile.deck_card_centers()[0]
    frame = _card_frame(card_id="hog-rider", x=center_x - 125, y=center_y - 150)

    match = CardVision("assets/templates/cr-api-assets/cards-gold").match_slot(frame, profile, 0)

    assert match is not None
    assert match.card_id == "hog-rider"
    assert match.score >= 0.62
    selected = CardVision("assets/templates/cr-api-assets/cards-gold").find_card_near(
        frame, "hog-rider", match.center
    )
    assert selected is not None
    assert selected.score >= 0.52


def test_card_vision_finds_a_collection_card_without_grayscale_false_positive() -> None:
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    frame = _card_frame(card_id="cannon", x=100, y=1500)

    match = CardVision("assets/templates/cr-api-assets/cards-gold").find_collection_card(
        frame, profile, "cannon"
    )

    assert match is not None
    assert match.card_id == "cannon"
    assert match.score >= 0.62


def test_card_vision_searches_the_full_collection_after_scrolling() -> None:
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    frame = _card_frame(card_id="hog-rider", x=100, y=300)

    match = CardVision("assets/templates/cr-api-assets/cards-gold").find_collection_card(
        frame, profile, "hog-rider", scrolled=True
    )

    assert match is not None
    assert match.card_id == "hog-rider"
    assert match.score >= 0.62


def test_red_remove_button_requires_the_selected_card_panel() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    canvas = np.zeros((2400, 1080, 3), dtype=np.uint8)
    # BGR red rectangle with the same broad geometry as the reviewed Remove
    # panel; a red card illustration has a taller, portrait-shaped region and
    # must not be mistaken for this control.
    cv2.rectangle(canvas, (18, 1036), (273, 1152), (0, 0, 255), thickness=-1)
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    frame = Frame(source_device="A", frame_index=0, workstation_monotonic_us=0, payload=encoded.tobytes())

    assert AutonomousPhone._find_red_button(frame, profile) == (146, 1094)


def test_yellow_use_button_requires_the_selected_collection_card() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    canvas = np.zeros((2400, 1080, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (280, 1292), (534, 1408), (0, 215, 255), thickness=-1)
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    frame = Frame(source_device="A", frame_index=0, workstation_monotonic_us=0, payload=encoded.tobytes())

    assert AutonomousPhone._find_yellow_button(frame, profile) == (407, 1350)


def test_slot_occupancy_gate_does_not_treat_an_unreadable_card_as_empty() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    vision = CardVision("assets/templates/cr-api-assets/cards-gold")
    phone = AutonomousPhone(object(), profile, vision)
    empty = np.zeros((2400, 1080, 3), dtype=np.uint8)
    card = cv2.imread("assets/templates/cr-api-assets/cards-gold/hog-rider.png", cv2.IMREAD_COLOR)
    assert card is not None
    card = cv2.resize(card, (250, 300), interpolation=cv2.INTER_AREA)
    center_x, center_y = profile.deck_card_centers()[0]
    occupied = empty.copy()
    occupied[center_y - 150 : center_y + 150, center_x - 125 : center_x + 125] = card
    frames: list[Frame] = []
    for image in (empty, occupied):
        success, encoded = cv2.imencode(".png", image)
        assert success
        frames.append(Frame("A", 0, 0, encoded.tobytes()))

    assert phone._slot_looks_empty(frames[0], 0)
    assert not phone._slot_looks_empty(frames[1], 0)


def test_deck_editor_gate_rejects_the_collection_page() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    phone = AutonomousPhone(object(), profile, CardVision("assets/templates/cr-api-assets/cards-gold"))

    def frame(*, decks_selected: bool) -> Frame:
        canvas = np.full((2400, 1080, 3), (28, 64, 102), dtype=np.uint8)
        left_color, right_color = (
            ((34, 107, 162), (28, 64, 102))
            if decks_selected
            else ((28, 64, 102), (34, 107, 162))
        )
        cv2.rectangle(canvas, (86, 216), (464, 408), left_color, thickness=-1)
        cv2.rectangle(canvas, (540, 216), (994, 408), right_color, thickness=-1)
        success, encoded = cv2.imencode(".png", canvas)
        assert success
        return Frame("A", 0, 0, encoded.tobytes())

    assert phone._looks_like_deck_editor(frame(decks_selected=True))
    assert not phone._looks_like_deck_editor(frame(decks_selected=False))


def test_fixed_deck_order_is_explicit() -> None:
    assert FIXED_HOG_CYCLE_DECK == (
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
        "fireball",
        "log",
    )
