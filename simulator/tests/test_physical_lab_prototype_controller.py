from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

from cr_bot.domain.frame_analysis import FrameAnalysisResult
from cr_bot.domain.game_state import Action, Detection, Match
from simulator.observation import PolicyObservationV1
from simulator.observation_v2 import PolicyObservationV2
from simulator.physical_lab.automation import AutonomousPhone, CardMatch, UiProfile
from simulator.physical_lab.calibration import CalibrationArtifact
from simulator.physical_lab.devices import FakePhoneController, Frame
from simulator.physical_lab.prototype_controller import (
    AdbH264FrameSource,
    AdbScreenshotSource,
    CachedAdbPhoneController,
    DecisionRecord,
    LiveDetectionFilter,
    LiveHandStateFilter,
    LivePrototypeRunner,
    LIVE_OWN_DECK_CARD_NAMES,
    LIVE_IGNORED_DETECTOR_LABELS,
    LIVE_OWN_DETECTOR_ALIASES,
    PrototypeControllerError,
    SourceFrame,
    action_to_dict,
    build_arg_parser,
    configure_detector_inference_size,
    format_decision_record,
    observation_to_model_inputs,
    policy_action_from_batch,
    _filter_live_analysis,
)


def _observation() -> PolicyObservationV2:
    legal_play = np.zeros((4, 32, 18), dtype=bool)
    legal_play[0, 20, 9] = True
    v1 = PolicyObservationV1(
        board=np.zeros((21, 32, 18), dtype=np.float32),
        global_vector=np.zeros((768,), dtype=np.float32),
        spatial_masks=np.zeros((4, 32, 18), dtype=bool),
        legal_play=legal_play,
        legal_wait=True,
    )
    return PolicyObservationV2.from_v1(v1)


def test_adb_source_retries_an_empty_screenshot_payload() -> None:
    class Controller:
        def __init__(self) -> None:
            self.frames = iter(
                (
                    SimpleNamespace(
                        payload=b"",
                        frame_index=0,
                        workstation_monotonic_us=1_000_000,
                    ),
                    SimpleNamespace(
                        payload=b"valid",
                        frame_index=1,
                        workstation_monotonic_us=1_200_000,
                    ),
                )
            )

        def screenshot(self):
            return next(self.frames)

    sleeps: list[float] = []

    def decode(frame):
        if not frame.payload:
            raise PrototypeControllerError("empty screenshot")
        return frame.payload

    source = AdbScreenshotSource(
        Controller(),
        decode=decode,
        retry_delay_s=0.1,
        sleep_fn=sleeps.append,
    )
    result = source.next_frame()

    assert result.image == b"valid"
    assert result.frame_index == 1
    assert result.timestamp_s == 1.2
    assert sleeps == [0.1]


def test_cached_adb_controller_reuses_recent_screenshot_connectivity() -> None:
    now = [10.0]
    controller = CachedAdbPhoneController(
        "serial",
        connection_check_interval_s=5.0,
        monotonic_fn=lambda: now[0],
    )
    info_calls = 0

    def device_info():
        nonlocal info_calls
        info_calls += 1
        return SimpleNamespace(connected=True)

    controller.device_info = device_info
    controller._last_successful_screenshot_s = now[0]
    controller._require_connected()
    assert info_calls == 0

    now[0] = 16.0
    controller._require_connected()
    assert info_calls == 1


def test_h264_stream_is_serial_scoped_and_provides_recent_action_frames() -> None:
    width, height = 2, 1
    raw_frame = bytes((10, 20, 30, 40, 50, 60))
    commands: list[list[str]] = []
    alive_marks = 0

    class Process:
        def __init__(self, stdout) -> None:
            self.stdout = stdout
            self.terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self) -> None:
            self.terminated = True

    def popen(command, **_kwargs):
        commands.append(list(command))
        if command[0] == "adb":
            return Process(BytesIO(b"h264-input"))
        return Process(BytesIO(raw_frame))

    def mark_alive() -> None:
        nonlocal alive_marks
        alive_marks += 1

    controller = SimpleNamespace(
        adb_executable="adb",
        serial="R7AIB700D744BX7",
        device_id="LIVE",
        device_info=lambda: SimpleNamespace(
            connected=True,
            screen_width_px=width,
            screen_height_px=height,
        ),
        mark_transport_alive=mark_alive,
    )
    source = AdbH264FrameSource(
        controller,
        popen_factory=popen,
        restart_delay_s=0.05,
        action_frame_max_age_s=1.0,
        action_frame_wait_timeout_s=1.0,
    )

    try:
        source_frame = source.next_frame()
        action_frame = source.frame_for_action()
    finally:
        source.close()

    assert source_frame.image.shape == (height, width, 3)
    assert source_frame.image.reshape(-1).tolist() == list(raw_frame)
    assert action_frame.source_device == "LIVE"
    assert action_frame.frame_index == source_frame.frame_index
    assert action_frame.payload
    assert alive_marks >= 1
    assert commands[0][:3] == ["adb", "-s", "R7AIB700D744BX7"]
    assert commands[0][3] == "exec-out"
    assert commands[1][0] == "ffmpeg"


def test_autonomous_phone_can_verify_a_card_from_a_stream_frame() -> None:
    controller = FakePhoneController("A")
    screenshot_calls = 0

    def unexpected_screenshot():
        nonlocal screenshot_calls
        screenshot_calls += 1
        raise AssertionError("stream-backed action verification should not call ADB screenshot")

    controller.screenshot = unexpected_screenshot
    profile = UiProfile("A", 1080, 2400, (0.275, 0.425), 0.59, 0.42)
    frame = Frame("A", 10, 10, b"stream-frame")

    class Vision:
        threshold = 0.62

        @staticmethod
        def find_hand_card_in_slot(_frame, _profile, _card_id, _slot, *, hand_px):
            assert hand_px == (237.0, 1900.0, 808.0, 500.0)
            return CardMatch("hog-rider", 0.99, (338, 2179))

    phone = AutonomousPhone(
        controller,
        profile,
        Vision(),
        action_frame_provider=lambda: frame,
    )
    calibration = CalibrationArtifact.for_screen(
        device_label="A",
        screen_width_px=1080,
        screen_height_px=2400,
        hand_px=(237.0, 1900.0, 808.0, 500.0),
    )

    slot, _selected, _placed = phone.select_and_place(
        "hog-rider",
        calibration=calibration,
        arena_cell=(3, 20),
        expected_slot=0,
    )

    assert slot == 0
    assert screenshot_calls == 0
    assert len(controller.taps) == 2


def test_detector_speed_adapter_overrides_each_yolo_call_size() -> None:
    calls: list[int] = []

    class Model:
        def predict(self, *_args, **kwargs):
            calls.append(kwargs["imgsz"])
            return []

    detector = SimpleNamespace(models=[Model(), Model()])
    configure_detector_inference_size(detector, 640)

    detector.models[0].predict(object(), imgsz=896)
    detector.models[1].predict(object(), imgsz=896)

    assert calls == [640, 640]
    assert detector._prototype_inference_size == 640


def test_detector_resolution_defaults_to_original_extractor_size() -> None:
    parser = build_arg_parser()

    default_args = parser.parse_args(["--video", "gameplay.mp4"])
    fast_args = parser.parse_args(["--video", "gameplay.mp4", "--yolo-image-size", "640"])
    screenshot_args = parser.parse_args(
        ["--serial", "SERIAL", "--adb-transport", "screenshot"]
    )

    assert default_args.yolo_image_size == 896
    assert fast_args.yolo_image_size == 640
    assert default_args.adb_transport == "stream"
    assert screenshot_args.adb_transport == "screenshot"


def test_observation_boundary_builds_one_step_recurrent_inputs() -> None:
    torch = pytest.importorskip("torch")

    board, global_features, entities, entity_mask, masks = observation_to_model_inputs(
        _observation(), device="cpu"
    )

    assert tuple(board.shape) == (1, 1, 21, 32, 18)
    assert tuple(global_features.shape) == (1, 1, 768)
    assert tuple(entities.shape) == (1, 1, 128, 32)
    assert tuple(entity_mask.shape) == (1, 1, 128)
    assert tuple(masks.mode.shape) == (1, 1, 2)
    assert tuple(masks.card.shape) == (1, 1, 4)
    assert tuple(masks.placement.shape) == (1, 1, 4, 32, 18)
    assert bool(masks.mode[0, 0, 0])
    assert bool(masks.mode[0, 0, 1])
    assert bool(masks.card[0, 0, 0])
    assert not bool(masks.card[0, 0, 1])
    assert torch.equal(
        masks.placement[0, 0, 0],
        torch.from_numpy(_observation().legal_play[0].copy()),
    )


def test_policy_batch_decoding_preserves_column_row_convention() -> None:
    torch = pytest.importorskip("torch")

    wait = SimpleNamespace(
        mode=torch.tensor([[0]], dtype=torch.long),
        card_slot=torch.tensor([[0]], dtype=torch.long),
        placement=torch.tensor([[[0, 0]]], dtype=torch.long),
    )
    play = SimpleNamespace(
        mode=torch.tensor([[1]], dtype=torch.long),
        card_slot=torch.tensor([[2]], dtype=torch.long),
        # ActionBatch stores (row, column); visual actions use (column, row).
        placement=torch.tensor([[[20, 9]]], dtype=torch.long),
    )

    assert action_to_dict(policy_action_from_batch(wait)) == {"kind": "wait"}
    assert policy_action_from_batch(play) == Action(
        kind="Play", card_idx=2, cell=(9, 20)
    )


def test_live_runner_omits_ignored_bomb_objects_before_session() -> None:
    def match(label: str, team: str = "ally") -> Match:
        return Match(
            troop=Detection(
                track_id=1,
                class_name=label,
                team=team,
                confidence=0.9,
                x1=0.0,
                y1=0.0,
                x2=1.0,
                y2=1.0,
                center_x=0.5,
                center_y=0.5,
            ),
            bar=None,
        )

    analysis = FrameAnalysisResult(
        rendered=None,
        elixir={},
        elixir_change=None,
        towers_hp={},
        time=None,
        time_left_s=None,
        total_remaining_s=None,
        overtime=False,
        hand_state={},
        yolo_boxes=None,
        clock_boxes=[],
        emote_boxes=[],
        matches=[
            match("bomb"),
            match("bomber"),
            match("hog"),
            match("skeleton-evolution"),
            match("giant", team="enemy"),
        ],
        arena_px=(0, 0, 1, 1),
        tower_hp_debug_steps={},
        timer_debug_steps={},
    )

    filtered = _filter_live_analysis(analysis)

    assert "bomb" in LIVE_IGNORED_DETECTOR_LABELS
    assert "hog-rider" in LIVE_OWN_DECK_CARD_NAMES
    assert LIVE_OWN_DETECTOR_ALIASES["skeleton-evolution"] == "skeletons"
    assert [item.troop.class_name for item in filtered.matches] == [
        "hog",
        "skeleton-evolution",
        "giant",
    ]
    assert [item.troop.class_name for item in analysis.matches] == [
        "bomb",
        "bomber",
        "hog",
        "skeleton-evolution",
        "giant",
    ]


def test_live_detection_filter_seeds_own_play_and_confirms_enemy_on_second_frame() -> None:
    def match(
        label: str,
        team: str,
        *,
        track_id: int | None,
        center: tuple[float, float],
    ) -> Match:
        center_x, center_y = center
        return Match(
            troop=Detection(
                track_id=track_id,
                class_name=label,
                team=team,
                confidence=0.9,
                x1=center_x - 5.0,
                y1=center_y - 5.0,
                x2=center_x + 5.0,
                y2=center_y + 5.0,
                center_x=center_x,
                center_y=center_y,
            ),
            bar=None,
        )

    def analysis(matches: list[Match]) -> FrameAnalysisResult:
        return FrameAnalysisResult(
            rendered=None,
            elixir={},
            elixir_change=None,
            towers_hp={},
            time=None,
            time_left_s=None,
            total_remaining_s=None,
            overtime=False,
            hand_state={},
            yolo_boxes=None,
            clock_boxes=[],
            emote_boxes=[],
            matches=matches,
            arena_px=(0, 0, 568, 896),
            tower_hp_debug_steps={},
            timer_debug_steps={},
        )

    detection_filter = LiveDetectionFilter()
    detection_filter.notify_own_play(
        card_name="hog-rider",
        cell=(3, 20),
        arena_px=(0, 0, 568, 896),
        timestamp_s=10.0,
    )
    enemy = match("giant", "enemy", track_id=17, center=(300.0, 300.0))

    first = detection_filter.update(analysis([enemy]), timestamp_s=10.25)
    first_classes = [item.troop.class_name for item in first.matches]
    assert first_classes == ["hog-rider"]
    assert first.matches[0].troop.team == "ally"

    second = detection_filter.update(analysis([enemy]), timestamp_s=10.50)
    assert [item.troop.class_name for item in second.matches] == [
        "giant",
        "hog-rider",
    ]
    assert second.matches[0].troop.team == "enemy"
    assert second.matches[1].troop.team == "ally"

    # Once the extractor catches the dispatched troop, the temporary seed is
    # removed and the real detection (including its position/HP) is retained.
    own_detection = match("hog", "ally", track_id=41, center=(300.0, 700.0))
    third = detection_filter.update(
        analysis([own_detection]),
        timestamp_s=10.75,
    )
    assert [item.troop.class_name for item in third.matches] == ["hog"]


def _hand_state(
    *cards: str | None,
    next_card: str | None = "cannon",
) -> dict[str, object]:
    return {
        **{
            f"card_{index}": (card, 99.0) if card is not None else None
            for index, card in enumerate(cards, start=1)
        },
        "next_card": (next_card, 99.0) if next_card is not None else None,
    }


def test_live_hand_filter_warms_up_and_rejects_duplicate_initial_predictions() -> None:
    filter_ = LiveHandStateFilter()

    duplicate = _hand_state("fire-spirit", "knight", "dart-goblin", "fire-spirit")
    for _ in range(2):
        filtered = filter_.update(duplicate)
        assert not filter_.ready
        assert filtered["card_1"] is None

    filtered = filter_.update(duplicate)

    assert not filter_.ready
    assert filtered["card_1"] == ("fire-spirit", 99.0)
    assert filtered["card_4"] is None


def test_live_hand_filter_uses_stable_next_card_for_replacement() -> None:
    filter_ = LiveHandStateFilter()
    before = _hand_state("log", "ice-spirit", "ice-golem", "hog-rider")
    after = _hand_state(
        "log", "cannon", "ice-golem", "hog-rider", next_card="fireball"
    )

    for _ in range(3):
        filter_.update(before)
    assert filter_.ready

    filter_.expect_replacement(1)
    held = filter_.update(before)
    assert filter_.ready
    assert held["card_2"] == ("cannon", 99.0)
    assert held["next_card"] is None

    first = filter_.update(after)
    second = filter_.update(after)
    third = filter_.update(after)
    assert first["card_2"] == ("cannon", 99.0)
    assert second["card_2"] == ("cannon", 99.0)
    assert third["card_2"] == ("cannon", 99.0)
    assert first["next_card"] is None
    assert second["next_card"] is None
    assert third["next_card"] == ("fireball", 99.0)
    assert filter_.ready


def test_runner_dispatches_play_but_never_dispatches_wait() -> None:
    class Source:
        def __init__(self) -> None:
            self.frames = [
                SourceFrame(object(), 0, 1.0),
                SourceFrame(object(), 1, 2.0),
            ]
            self.closed = False

        def next_frame(self):
            return self.frames.pop(0) if self.frames else None

        def close(self) -> None:
            self.closed = True

    class Actor:
        def __init__(self) -> None:
            self.actions = iter(
                (
                    Action(kind="Wait"),
                    Action(kind="Play", card_idx=0, cell=(9, 20)),
                )
            )
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def decide(self, _observation):
            return next(self.actions)

    state = SimpleNamespace(
        hud=SimpleNamespace(
            hand_cards=["hog-rider", "cannon", "musketeer", "skeletons"],
            elixir_self=10.0,
            time_left_s=120.0,
        )
    )
    step = SimpleNamespace(
        in_game=True,
        should_emit=True,
        game_state=state,
        analysis=SimpleNamespace(matches=[]),
    )
    dispatched: list[tuple[object, object]] = []

    runner = LivePrototypeRunner(
        Source(),
        detector=object(),
        actor=Actor(),
        execute=True,
        phone=object(),
        calibration=object(),
        normalize=False,
        poll_interval_s=0.0,
        min_action_interval_s=0.0,
        post_action_delay_s=0.0,
        session=SimpleNamespace(process=lambda *_args, **_kwargs: step),
        process_frame_fn=lambda *_args, **_kwargs: object(),
        normalize_frame_fn=lambda image: image,
        observation_builder=lambda _step: object(),
        dispatch_fn=lambda phone, action, *_args, **_kwargs: dispatched.append(
            (phone, action)
        ),
    )
    records = []
    summary = runner.run(on_record=records.append)

    assert summary.as_dict() == {
        "frames": 2,
        "emitted_frames": 2,
        "waits": 1,
        "proposed_plays": 1,
        "dispatched_plays": 1,
    }
    assert len(dispatched) == 1
    assert dispatched[0][1] == Action(kind="Play", card_idx=0, cell=(9, 20))
    assert [record.result for record in records] == ["wait", "dispatched"]
    assert records[0].visual_state == {
        "hand": ["hog-rider", "cannon", "musketeer", "skeletons"],
        "next_card": None,
        "elixir": 10.0,
        "enemy_elixir_est": None,
        "time_left_s": 120.0,
        "total_remaining_s": None,
        "overtime": False,
        "ally_units": [],
        "enemy_units": [],
        "seen_enemy_cards": [],
        "tower_hp_self": [],
        "tower_hp_enemy": [],
        "detection_count": 0,
        "arena_px": [],
    }


def test_decision_output_combines_frame_action_and_extracted_state() -> None:
    output = format_decision_record(
        DecisionRecord(
            frame_index=1884,
            timestamp_s=7457.318,
            in_game=True,
            emitted=True,
            action={"kind": "wait"},
            result="wait",
            visual_state={
                "hand": ["ice-golem", "cannon", "log", "musketeer"],
                "next_card": "hog-rider",
                "elixir": 8.5,
                "time_left_s": 274.2,
                "overtime": False,
                "ally_units": [
                    {
                        "label": "dark-prince",
                        "track": 644,
                        "center_px": [875.6, 851.2],
                        "estimated_hp": 1234,
                        "confidence": 0.56,
                    }
                ],
                "enemy_units": [
                    {
                        "label": "night-witch",
                        "track": 620,
                        "center_px": [378.4, 1552.2],
                        "estimated_hp": 987,
                        "confidence": 0.73,
                    },
                    {
                        "label": "electro-wizard",
                        "track": 640,
                        "center_px": [302.8, 1375.2],
                        "estimated_hp": 1456,
                        "confidence": 0.95,
                    },
                ],
                "detection_count": 3,
                "tower_hp_self": [1000, 3000, 1200],
                "tower_hp_enemy": [900, 3000, 1100],
                "seen_enemy_cards": [4, 17],
            },
        )
    )

    assert "frame=1884 · t=7457.318s" in output
    assert "hand=[ice-golem · cannon · log · musketeer]" in output
    visual_line = next(line for line in output.splitlines() if "VISUAL" in line)
    assert "dark-prince#644 hp=1234 @875.6,851.2 [0.56]" in visual_line
    assert "night-witch#620 hp=987 @378.4,1552.2 [0.73]" in visual_line
    assert "electro-wizard#640 hp=1456 @302.8,1375.2 [0.95]" in visual_line
    assert "action={'kind': 'wait'}  result=wait" in output


def test_runner_rejects_execution_without_action_sink() -> None:
    with pytest.raises(RuntimeError, match="requires both an AutonomousPhone"):
        LivePrototypeRunner(
            SimpleNamespace(next_frame=lambda: None, close=lambda: None),
            detector=object(),
            actor=object(),
            execute=True,
            normalize=False,
            session=object(),
            process_frame_fn=lambda *_args, **_kwargs: object(),
            normalize_frame_fn=lambda image: image,
        )
