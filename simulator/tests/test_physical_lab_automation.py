from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import simulator.physical_lab.automation as automation_module
from simulator.physical_lab import CardVision, LifecycleState, UiProfile
from simulator.physical_lab.automation import (
    AutonomousPhysicalLab,
    AutonomousPhone,
    AutomationError,
    FIXED_HOG_CYCLE_DECK,
)
from simulator.physical_lab.devices import Frame
from simulator.physical_lab.devices import ActionReceipt, FakePhoneController


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


def _editor_header_frame(*, decks_selected: bool) -> Frame:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    canvas = np.full((2400, 1080, 3), (28, 64, 102), dtype=np.uint8)
    left_color, right_color = (
        ((200, 140, 40), (100, 60, 20))
        if decks_selected
        else ((100, 60, 20), (200, 140, 40))
    )
    cv2.rectangle(canvas, (86, 216), (464, 408), left_color, thickness=-1)
    cv2.rectangle(canvas, (540, 216), (994, 408), right_color, thickness=-1)
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    return Frame("A", 0, 0, encoded.tobytes())


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


def test_deck_slot_identity_does_not_bless_a_furnace_as_hog() -> None:
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    center_x, center_y = profile.deck_card_centers()[0]
    frame = _card_frame(card_id="furnace", x=center_x - 125, y=center_y - 150)

    match = CardVision("assets/templates/cr-api-assets/cards-gold").match_slot(frame, profile, 0)

    assert match is not None
    assert match.card_id == "furnace"


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


def test_card_vision_keeps_distinct_candidates_on_the_same_collection_page() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    asset = cv2.imread(
        "assets/templates/cr-api-assets/cards-gold/hog-rider.png",
        cv2.IMREAD_COLOR,
    )
    assert asset is not None
    asset = cv2.resize(asset, (250, 300), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((2400, 1080, 3), dtype=np.uint8)
    canvas[500:800, 80:330] = asset
    canvas[1400:1700, 650:900] = asset
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    frame = Frame("A", 0, 0, encoded.tobytes())

    matches = CardVision(
        "assets/templates/cr-api-assets/cards-gold"
    ).find_collection_card_candidates(frame, profile, "hog-rider", scrolled=True)

    assert len(matches) == 2
    assert all(match.score >= 0.62 for match in matches)


def test_scrolled_collection_search_excludes_visible_active_deck() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.49), 0.59, 0.42)
    asset = cv2.imread(
        "assets/templates/cr-api-assets/cards-gold/hog-rider.png",
        cv2.IMREAD_COLOR,
    )
    assert asset is not None
    asset = cv2.resize(asset, (220, 300), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((2400, 1080, 3), dtype=np.uint8)
    # Visible editor switcher and one Hog copy only in active deck slot 0.
    cv2.rectangle(canvas, (0, 312), (1079, 456), (200, 140, 40), thickness=-1)
    center_x, center_y = profile.deck_card_centers()[0]
    canvas[center_y - 150 : center_y + 150, center_x - 110 : center_x + 110] = asset
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    frame = Frame("A", 0, 0, encoded.tobytes())

    matches = CardVision(
        "assets/templates/cr-api-assets/cards-gold"
    ).find_collection_card_candidates(frame, profile, "hog-rider", scrolled=True)

    assert matches == ()


def test_asus_scrolled_collection_admits_first_row_below_active_deck() -> None:
    profile = UiProfile("A", 1080, 2400, (0.275, 0.49), 0.59, 0.42)

    assert CardVision._collection_search_top(object(), profile, scrolled=True) == 0.50


def test_expected_slot_uses_wide_identity_with_score_and_margin() -> None:
    profile = UiProfile("A", 1080, 2400, (0.275, 0.49), 0.59, 0.42)

    class Vision:
        @staticmethod
        def rank_card_identities_near(*_args: object, **_kwargs: object) -> tuple[object, ...]:
            return (
                SimpleNamespace(card_id="skeletons", score=0.679),
                SimpleNamespace(card_id="wizard", score=0.308),
            )

    phone = AutonomousPhone(object(), profile, Vision())
    frame = Frame("A", 0, 0, b"unused by fake vision")

    assert phone._slot_matches_expected(frame, 2, "skeletons")
    assert not phone._slot_matches_expected(frame, 2, "wizard")


def test_card_identity_ranking_names_a_false_hog_candidate() -> None:
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    frame = _card_frame(card_id="furnace", x=300, y=1200)
    identities = CardVision(
        "assets/templates/cr-api-assets/cards-gold"
    ).rank_card_identities_near(frame, (425, 1350), limit=2)

    assert identities[0].card_id == "furnace"
    assert identities[0].score - identities[1].score >= 0.08


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


def test_red_remove_button_may_follow_a_right_hand_deck_card() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("B", 1080, 2280, (0.355, 0.535), 0.66, 0.49)
    canvas = np.zeros((2280, 1080, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (484, 1068), (714, 1177), (0, 40, 245), thickness=-1)
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    frame = Frame("B", 0, 0, encoded.tobytes())

    assert AutonomousPhone._find_red_button(frame, profile) == (599, 1123)


def test_red_remove_button_supports_a_second_deck_row() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2280, (0.355, 0.535), 0.30, 0.49)
    canvas = np.zeros((2280, 1080, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (18, 1530), (273, 1645), (0, 40, 245), thickness=-1)
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    frame = Frame("A", 0, 0, encoded.tobytes())

    assert AutonomousPhone._find_red_button(frame, profile) == (146, 1588)


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
    assert AutonomousPhone._find_yellow_button(frame, profile, anchor=(407, 1000)) == (407, 1350)


def test_yellow_use_button_anchor_rejects_an_unrelated_card_strip() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    canvas = np.zeros((2400, 1080, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (280, 1843), (534, 1957), (0, 215, 255), thickness=-1)
    cv2.rectangle(canvas, (558, 2034), (779, 2133), (0, 215, 255), thickness=-1)
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    frame = Frame("A", 0, 0, encoded.tobytes())

    assert AutonomousPhone._find_yellow_button(frame, profile, anchor=(407, 1526)) == (407, 1900)


def test_slot_occupancy_gate_does_not_treat_an_unreadable_card_as_empty() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    vision = CardVision("assets/templates/cr-api-assets/cards-gold")
    phone = AutonomousPhone(object(), profile, vision)
    empty = np.full((2400, 1080, 3), (130, 65, 20), dtype=np.uint8)
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


def test_samsung_b_rejects_regular_musketeer_in_the_first_three_slots() -> None:
    profile = UiProfile("B", 1080, 2280, (0.355, 0.535), 0.30, 0.49)
    phone = AutonomousPhone(
        object(),
        profile,
        CardVision("assets/templates/cr-api-assets/cards-gold"),
    )

    with pytest.raises(AutomationError, match="human deck slots 4-8"):
        phone._validate_device_deck_constraints(FIXED_HOG_CYCLE_DECK)

    phone._validate_device_deck_constraints(
        ("hog-rider", "fireball", "log", "cannon", "skeletons", "musketeer", "ice-golem", "ice-spirit")
    )
    assert phone._minimum_collection_identity_score("musketeer") == 0.50
    assert phone._minimum_collection_identity_score("cannon") == 0.60


def test_asus_a_rejects_regular_special_cards_in_the_first_three_slots() -> None:
    profile = UiProfile("A", 1080, 2400, (0.275, 0.49), 0.59, 0.42)
    phone = AutonomousPhone(
        object(),
        profile,
        CardVision("assets/templates/cr-api-assets/cards-gold"),
    )

    with pytest.raises(AutomationError, match="regular Archers and Musketeer"):
        phone._validate_device_deck_constraints(
            ("hog-rider", "cannon", "musketeer", "skeletons", "ice-golem", "ice-spirit", "fireball", "log")
        )

    with pytest.raises(AutomationError, match="archers in slot 1"):
        phone._validate_device_deck_constraints(
            ("archers", "cannon", "skeletons", "musketeer", "ice-golem", "ice-spirit", "fireball", "log")
        )

    phone._validate_device_deck_constraints(
        ("hog-rider", "cannon", "skeletons", "musketeer", "ice-golem", "ice-spirit", "fireball", "log")
    )
    assert phone._collection_candidate_threshold("musketeer") == 0.55


def test_samsung_b_profile_targets_the_reviewed_deck_rows() -> None:
    info = SimpleNamespace(screen_width_px=1080, screen_height_px=2280)

    profile = UiProfile.for_device("B", info)

    assert profile.deck_card_centers()[0] == (130, 912)
    assert profile.deck_card_centers()[4] == (130, 1345)


def test_card_upgrade_tutorial_close_is_detected_only_in_its_modal_geometry() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    canvas = np.full((2280, 1080, 3), (8, 12, 20), dtype=np.uint8)
    cv2.rectangle(canvas, (35, 350), (1045, 2050), (120, 20, 120), thickness=-1)
    cv2.rectangle(canvas, (950, 375), (1025, 450), (0, 0, 255), thickness=-1)
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    tutorial = Frame("B", 0, 0, encoded.tobytes())

    assert AutonomousPhone._find_card_upgrade_tutorial_close(tutorial) == (988, 413)

    normal_canvas = np.full((2280, 1080, 3), (80, 60, 30), dtype=np.uint8)
    success, normal_payload = cv2.imencode(".png", normal_canvas)
    assert success
    normal = Frame("B", 0, 0, normal_payload.tobytes())
    assert AutonomousPhone._find_card_upgrade_tutorial_close(normal) is None


def test_scroll_settle_gate_requires_stable_screenshot_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    phone = AutonomousPhone(object(), profile, CardVision("assets/templates/cr-api-assets/cards-gold"))
    canvas = np.full((2400, 1080, 3), (80, 60, 30), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    frame = Frame("A", 0, 0, encoded.tobytes())
    samples = 0

    def screenshot() -> Frame:
        nonlocal samples
        samples += 1
        return frame

    monkeypatch.setattr(phone, "screenshot", screenshot)
    phone._wait_for_scroll_settled(timeout_s=0.1, poll_s=0.0, stable_samples=2)

    assert samples == 3


def test_deck_editor_gate_rejects_the_collection_page() -> None:
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    phone = AutonomousPhone(object(), profile, CardVision("assets/templates/cr-api-assets/cards-gold"))

    assert phone._looks_like_deck_editor(_editor_header_frame(decks_selected=True))
    assert not phone._looks_like_deck_editor(_editor_header_frame(decks_selected=False))
    assert phone._looks_like_collection_editor_top(_editor_header_frame(decks_selected=False))


def test_deck_editor_gate_rejects_seasonal_modal_false_positive() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    phone = AutonomousPhone(object(), profile, CardVision("assets/templates/cr-api-assets/cards-gold"))
    canvas = np.full((2400, 1080, 3), (35, 20, 65), dtype=np.uint8)
    # Reproduce the misleading left/right luminance difference without the
    # editor's wide saturated-blue deck switcher.
    cv2.rectangle(canvas, (86, 216), (464, 408), (80, 80, 160), thickness=-1)
    cv2.rectangle(canvas, (540, 216), (994, 408), (20, 20, 40), thickness=-1)
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    modal = Frame("A", 0, 0, encoded.tobytes())

    assert phone._editor_tab_luminance_delta(modal) >= 18.0
    assert not phone._looks_like_deck_editor(modal)


def test_asus_collection_navigation_uses_reviewed_cards_icon_center() -> None:
    profile = UiProfile.for_device(
        "A",
        SimpleNamespace(screen_width_px=1080, screen_height_px=2400),
    )

    assert profile.collection_tab() == (248, 2256)


def test_lobby_gate_recognizes_the_large_battle_button_only() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    profile = UiProfile("A", 1080, 2280, (0.355, 0.535), 0.30, 0.49)
    canvas = np.zeros((2280, 1080, 3), dtype=np.uint8)
    # BGR yellow-orange rectangle at the reviewed lower-center lobby action.
    cv2.rectangle(canvas, (310, 1560), (770, 1800), (0, 190, 255), thickness=-1)
    success, encoded = cv2.imencode(".png", canvas)
    assert success
    lobby = Frame("A", 0, 0, encoded.tobytes())
    phone = AutonomousPhone(object(), profile, CardVision("assets/templates/cr-api-assets/cards-gold"))

    assert phone._looks_like_lobby(lobby)
    assert not phone._looks_like_lobby(_editor_header_frame(decks_selected=True))


def test_online_player_ocr_gate_accepts_name_glyph_noise_but_rejects_clan_tag() -> None:
    assert AutonomousPhone._ocr_name_score("YKescuierHD", "KeschlerHD") >= 0.78
    assert AutonomousPhone._ocr_name_score("KescHIERHD", "KeschlerHD") >= 0.78
    assert AutonomousPhone._ocr_name_score("keschler", "KeschlerHD") == 0.0
    assert AutonomousPhone._ocr_name_score("pwn_keschler", "KeschlerHD") == 0.0


def test_preparation_sends_no_game_input_from_an_unverified_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Controller:
        def __init__(self) -> None:
            self.vendor_commands: list[tuple[str, ...]] = []

        def _run(self, *args: str) -> None:
            self.vendor_commands.append(args)

        def screenshot(self) -> Frame:
            return _editor_header_frame(decks_selected=False)

        def tap_screen(self, _x: int, _y: int) -> None:
            pytest.fail("preparation must not tap Clash Royale from an unverified screen")

    controller = Controller()
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    phone = AutonomousPhone(controller, profile, CardVision("assets/templates/cr-api-assets/cards-gold"))
    monkeypatch.setattr(automation_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(automation_module.AutomationError, match="verified top Decks screen"):
        phone.configure_fixed_deck(
            target_deck=(
                "hog-rider",
                "cannon",
                "skeletons",
                "musketeer",
                "ice-golem",
                "ice-spirit",
                "fireball",
                "log",
            ),
            max_swipes=1,
        )

    assert controller.vendor_commands == [
        ("shell", "am", "force-stop", "com.asus.gamewidget")
    ]


def test_connected_run_rejects_a_battle_start_before_game_input() -> None:
    class Detector:
        def detect(self) -> LifecycleState:
            return LifecycleState.BATTLE

    lab = object.__new__(AutonomousPhysicalLab)
    lab.detectors = {"A": Detector(), "B": Detector()}
    observations: list[dict[str, str]] = []

    with pytest.raises(automation_module.AutomationError, match="before any game input"):
        lab._require_initial_lobby(observations)

    assert observations == [{"A": "battle", "B": "battle"}]


def test_placement_time_uses_completed_receipt_not_observation_boundary() -> None:
    receipt = ActionReceipt(
        receipt_id="B-tap-1",
        device_id="B",
        accepted=True,
        requested_at_monotonic_us=1_250_000,
        completed_at_monotonic_us=1_275_000,
    )

    assert AutonomousPhysicalLab._placement_match_time(
        receipt,
        battle_started_at_monotonic_us=1_000_000,
    ) == 275_000


def test_placement_time_rejects_a_receipt_before_battle_boundary() -> None:
    receipt = ActionReceipt(
        receipt_id="B-tap-early",
        device_id="B",
        accepted=True,
        requested_at_monotonic_us=900_000,
        completed_at_monotonic_us=950_000,
    )

    with pytest.raises(automation_module.AutomationError, match="predates"):
        AutonomousPhysicalLab._placement_match_time(
            receipt,
            battle_started_at_monotonic_us=1_000_000,
        )


def test_active_match_recovery_waits_for_result_before_any_navigation() -> None:
    calls: list[str] = []

    class Phone:
        def __init__(self, side: str) -> None:
            self.side = side

        def dismiss_result(self) -> None:
            calls.append(f"dismiss-{self.side}")

        def return_to_lobby(self) -> None:
            calls.append(f"lobby-{self.side}")

    lab = object.__new__(AutonomousPhysicalLab)
    lab.config = SimpleNamespace(result_timeout_s=330.0)
    lab.phones = {"A": Phone("A"), "B": Phone("B")}

    def wait_pair(target, *, observations, timeout_s=None):
        del observations, timeout_s
        calls.append(f"wait-{target.value}")
        return {"A": target, "B": target}

    lab._wait_pair = wait_pair
    transitions = []
    current = lab._recover_without_cancelling_match(
        LifecycleState.BATTLE,
        observations=[],
        transitions=transitions,
    )

    assert current is LifecycleState.RECOVERY
    assert calls == [
        "wait-result",
        "dismiss-A",
        "dismiss-B",
        "wait-archived",
        "lobby-A",
        "lobby-B",
        "wait-recovery",
    ]
    assert [transition.to_state for transition in transitions] == [
        LifecycleState.RESULT,
        LifecycleState.ARCHIVED,
        LifecycleState.RECOVERY,
    ]


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


def test_solo_battle_long_press_is_a_real_hold_and_records_opening_hand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(automation_module.time, "sleep", lambda _seconds: None)
    controller = FakePhoneController("A", serial_label="phone-a")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    phone = AutonomousPhone(
        controller,
        profile,
        CardVision("assets/templates/cr-api-assets/cards-gold"),
    )

    result = phone.open_testspiel_solo(
        fixed_deck_order=True,
        fixed_deck_toggle_point=(700, 1600),
        long_press_ms=900,
    )

    assert controller.long_presses == [(*profile.solo_battle(), 900)]
    assert result["state"] == "fixed_deck_options_enabled"
    assert result["opening_hand"] == ["hog-rider", "cannon", "musketeer", "skeletons"]
    assert result["replacement_order"] == ["ice-golem", "ice-spirit", "fireball", "log"]


def test_fixed_deck_testspiel_start_requires_a_reviewed_start_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(automation_module.time, "sleep", lambda _seconds: None)
    controller = FakePhoneController("A", serial_label="phone-a")
    phone = AutonomousPhone(
        controller,
        UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42),
        CardVision("assets/templates/cr-api-assets/cards-gold"),
    )

    with pytest.raises(automation_module.AutomationError, match="toggle point"):
        phone.open_testspiel_solo(fixed_deck_order=True)

    assert controller.long_presses


def test_card_placement_rejects_a_hand_slot_drift_before_tapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = FakePhoneController("A", serial_label="phone-a")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)

    class Vision:
        threshold = 0.62

        @staticmethod
        def find_hand_card(_frame: Frame, _profile: UiProfile, _card_id: str) -> object:
            return automation_module.CardMatch("hog-rider", 0.99, (810, 2100))

    phone = AutonomousPhone(controller, profile, Vision())
    monkeypatch.setattr(phone, "record", lambda _capture=None: Frame("A", 0, 0, b"frame"))

    with pytest.raises(automation_module.AutomationError, match="expected reviewed slot 0"):
        phone.select_and_place(
            "hog-rider",
            calibration=automation_module.CalibrationArtifact.for_screen(
                device_label="A",
                screen_width_px=1080,
                screen_height_px=2400,
            ),
            arena_cell=(3, 20),
            expected_slot=0,
        )

    assert controller.taps == []


def test_card_placement_uses_the_reviewed_hand_rectangle_for_native_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = FakePhoneController("A", serial_label="phone-a")
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)

    class Vision:
        threshold = 0.62

        @staticmethod
        def find_hand_card(_frame: Frame, _profile: UiProfile, _card_id: str) -> object:
            # The reviewed ASUS hand cards are centered around x=338, 540,
            # 742, 944 rather than at quarter-screen centers.
            return automation_module.CardMatch("hog-rider", 0.99, (338, 2179))

    phone = AutonomousPhone(controller, profile, Vision())
    monkeypatch.setattr(phone, "record", lambda _capture=None: Frame("A", 0, 0, b"frame"))
    calibration = automation_module.CalibrationArtifact.for_screen(
        device_label="A",
        screen_width_px=1080,
        screen_height_px=2400,
        hand_px=(237, 1900, 808, 500),
    )

    slot, _selected, _placed = phone.select_and_place(
        "hog-rider",
        calibration=calibration,
        arena_cell=(3, 20),
        expected_slot=0,
    )

    assert slot == 0
    assert controller.taps[0] == calibration.slot_to_pixel(0)
