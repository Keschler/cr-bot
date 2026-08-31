from __future__ import annotations

from types import SimpleNamespace
import pytest

from cr_bot.domain.game_state import (
    Action,
    Detection,
    GameState,
    HudState,
    Match,
    PrincessTowerState,
)
from cr_bot.domain.card_metadata import CARD_METADATA
from simulator.observation import ACTION_MASK_SHAPE
from simulator.physical_lab.policy_bridge import (
    PolicyBridgeError,
    dispatch_policy_action,
    observation_from_game_state,
    observation_from_match_step,
    observation_v2_from_game_state,
    placement_command_from_policy_action,
)


def _state(*, elixir: float = 10.0) -> GameState:
    towers = PrincessTowerState(
        own_left_alive=True,
        own_right_alive=True,
        enemy_left_alive=True,
        enemy_right_alive=True,
    )
    return GameState(
        hud=HudState(
            time_left_s=120.0,
            overtime=False,
            elixir_self=elixir,
            hand_cards=["hog-rider", "cannon", "musketeer", "skeletons"],
            next_card="ice-golem",
            tower_hp_self=[3_050, 6_500, 3_050],
            tower_hp_enemy=[3_050, 6_500, 3_050],
            princess_towers=towers,
        ),
        total_remaining_s=120.0,
        own_units=[],
        enemy_units=[],
        seen_enemy_cards=[],
        elixir_enemy_est=5.0,
        own_king_active=False,
        enemy_king_active=False,
        started=True,
    )


def test_visual_state_projects_to_the_pinned_policy_contract() -> None:
    observation = observation_from_game_state(
        _state(),
        arena_px=(0.0, 0.0, 1.0, 1.0),
    )

    assert observation.board.shape == (21, 32, 18)
    assert observation.global_vector.shape == (768,)
    assert observation.spatial_masks.shape == ACTION_MASK_SHAPE
    assert observation.legal_play.shape == ACTION_MASK_SHAPE
    assert observation.legal_wait is True
    # Hog is an own-side ground card in the viewer-local grid.
    assert observation.legal_play[0, 20, 9]
    assert not observation.legal_play[0, 10, 9]


@pytest.mark.parametrize(
    ("detector_name", "metadata_name"),
    (("skeleton", "skeletons"), ("bat", "bats")),
)
def test_detector_unit_labels_are_normalized_for_v1_and_v2_features(
    detector_name: str,
    metadata_name: str,
) -> None:
    state = _state()
    state.enemy_units = [
        Match(
            troop=Detection(
                track_id=7,
                class_name=detector_name,
                team="enemy",
                confidence=0.9,
                x1=0.45,
                y1=0.45,
                x2=0.55,
                y2=0.55,
                center_x=0.5,
                center_y=0.5,
                estimated_hp=130.0,
            ),
            bar=None,
        )
    ]

    observation = observation_from_game_state(
        state,
        arena_px=(0.0, 0.0, 1.0, 1.0),
    )
    v2_observation = observation_v2_from_game_state(
        state,
        arena_px=(0.0, 0.0, 1.0, 1.0),
    )

    assert observation.board.shape == (21, 32, 18)
    assert v2_observation.entity_mask[0]
    # The extractor state is not mutated while the feature view uses the
    # corresponding metadata key.
    assert state.enemy_units[0].troop.class_name == detector_name
    max_card_id = max(int(metadata["id"]) for metadata in CARD_METADATA.values())
    assert v2_observation.entity_tokens[0, 0] == pytest.approx(
        CARD_METADATA[metadata_name]["id"] / max_card_id
    )


def test_unrepresentable_detector_effect_is_omitted_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = _state()
    state.enemy_units = [
        Match(
            troop=Detection(
                track_id=8,
                class_name="bomb",
                team="enemy",
                confidence=0.9,
                x1=0.45,
                y1=0.45,
                x2=0.55,
                y2=0.55,
                center_x=0.5,
                center_y=0.5,
                estimated_hp=None,
            ),
            bar=None,
        )
    ]

    with caplog.at_level("WARNING", logger="simulator.physical_lab.policy_bridge"):
        observation = observation_v2_from_game_state(
            state,
            arena_px=(0.0, 0.0, 1.0, 1.0),
        )

    assert not observation.entity_mask.any()
    assert "unsupported detector label='bomb'" in caplog.text


def test_match_step_helper_skips_non_emitted_frames() -> None:
    assert observation_from_match_step(SimpleNamespace(in_game=False)) is None
    step = SimpleNamespace(
        in_game=True,
        should_emit=True,
        game_state=_state(),
        analysis=SimpleNamespace(arena_px=(0.0, 0.0, 1.0, 1.0)),
    )
    observation = observation_from_match_step(step)
    assert observation is not None
    assert observation.legal_wait is True


def test_match_step_keeps_wait_legal_when_timer_ocr_is_missing() -> None:
    state = _state()
    state.hud.time_left_s = None  # type: ignore[assignment]
    step = SimpleNamespace(
        in_game=True,
        should_emit=True,
        game_state=state,
        analysis=SimpleNamespace(arena_px=(0.0, 0.0, 1.0, 1.0)),
    )

    observation = observation_from_match_step(step)

    assert observation is not None
    assert observation.legal_wait is True


def test_action_resolution_uses_hand_identity_and_rejects_stale_legality() -> None:
    state = _state()
    observation = observation_from_game_state(state, arena_px=(0.0, 0.0, 1.0, 1.0))

    command = placement_command_from_policy_action(
        Action(kind="Play", card_idx=0, cell=(9, 20)),
        state,
        observation=observation,
    )
    assert command is not None
    assert command.card_id == "hog-rider"
    assert command.card_slot == 0
    assert command.arena_cell == (9, 20)
    assert placement_command_from_policy_action(Action(kind="Wait"), state) is None

    with pytest.raises(PolicyBridgeError, match="not legal"):
        placement_command_from_policy_action(
            Action(kind="Play", card_idx=0, cell=(9, 10)),
            state,
            observation=observation,
        )


def test_unaffordable_card_is_not_dispatchable() -> None:
    state = _state(elixir=0.0)
    observation = observation_from_game_state(state, arena_px=(0.0, 0.0, 1.0, 1.0))

    with pytest.raises(PolicyBridgeError, match="not legal"):
        placement_command_from_policy_action(
            Action(kind="Play", card_idx=0, cell=(9, 20)),
            state,
            observation=observation,
        )


def test_dispatch_delegates_identity_and_pixel_mapping_to_phone() -> None:
    state = _state()
    observation = observation_from_game_state(state, arena_px=(0.0, 0.0, 1.0, 1.0))
    calls: list[tuple[str, dict[str, object]]] = []

    class FakePhone:
        def select_and_place(self, card_id: str, **kwargs: object):
            calls.append((card_id, kwargs))
            return (kwargs["expected_slot"], SimpleNamespace(accepted=True), SimpleNamespace(accepted=True))

    receipt = dispatch_policy_action(
        FakePhone(),
        Action(kind="Play", card_idx=0, cell=(9, 20)),
        state,
        calibration=object(),
        observation=observation,
    )

    assert receipt is not None
    assert calls == [
        (
            "hog-rider",
            {
                "calibration": calls[0][1]["calibration"],
                "arena_cell": (9, 20),
                "expected_slot": 0,
                "capture": None,
            },
        )
    ]
