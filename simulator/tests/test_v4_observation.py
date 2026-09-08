from __future__ import annotations

import numpy as np
import pytest

from simulator.observation import BOARD_SHAPE, GLOBAL_VECTOR_SHAPE
from simulator.observation_v2 import ENTITY_TOKEN_SHAPE
from simulator.observation_v3 import (
    BELIEF_CARD_COUNT,
    EVENT_HISTORY_SHAPE,
    HAND_TOKEN_SHAPE,
    OBSERVATION_V3_CONTRACT_HASH,
    PolicyObservationV3,
    calculate_observation_v3_contract_hash,
    empty_event_history,
    empty_hand_tokens,
    observation_v3_contract_manifest,
    uniform_hand_belief,
)
from simulator.public_state_estimator import (
    PublicEvent,
    PublicStateEstimator,
    belief_index,
)


def _v1_board() -> np.ndarray:
    return np.zeros(BOARD_SHAPE, dtype=np.float32)


def _v1_global() -> np.ndarray:
    return np.zeros(GLOBAL_VECTOR_SHAPE, dtype=np.float32)


def _fake_v1() -> object:
    from simulator.observation import PolicyObservationV1

    board = _v1_board()
    glob = _v1_global()
    legal = np.zeros((4, 32, 18), dtype=bool)
    legal[:, 17:, :] = True
    return PolicyObservationV1(
        board=board,
        global_vector=glob,
        legal_play=legal,
        legal_wait=True,
        spatial_masks=np.zeros((4, 32, 18), dtype=bool),
    )


def test_v3_contract_hash_is_stable() -> None:
    assert calculate_observation_v3_contract_hash() == OBSERVATION_V3_CONTRACT_HASH
    manifest = observation_v3_contract_manifest()
    assert manifest["schema_version"] == "public-v3-1"
    assert manifest["tensors"]["hand_tokens"]["shape"] == list(HAND_TOKEN_SHAPE)


def test_from_v2_defaults_are_uninformative() -> None:
    obs = PolicyObservationV3.from_v2(_fake_v1())
    assert obs.entity_tokens.shape == ENTITY_TOKEN_SHAPE
    assert obs.hand_tokens.shape == HAND_TOKEN_SHAPE
    assert float(obs.opp_hand_probs.sum()) == pytest.approx(1.0)
    assert tuple(obs.opp_elixir_interval.tolist()) == (0.0, 10.0)
    assert bool((obs.event_history == 0.0).all())
    mode_mask, card_mask, placement_mask = obs.structured_action_masks()
    assert mode_mask.tolist() == [True, True]
    assert placement_mask.shape == (4, 32, 18)


def test_v3_rejects_overconfident_belief() -> None:
    probs, out = uniform_hand_belief()
    bad = np.array(probs)
    bad[0] = 2.0
    with pytest.raises(ValueError, match="opp_hand_probs"):
        PolicyObservationV3.from_v2(_fake_v1(), opp_hand_probs=bad, opp_out_of_cycle=out)
    with pytest.raises(ValueError, match="opp_elixir_interval"):
        PolicyObservationV3.from_v2(
            _fake_v1(),
            opp_hand_probs=np.array(probs),
            opp_out_of_cycle=out,
            opp_elixir_interval=(7.0, 2.0),
        )


def test_estimator_is_deterministic_and_ordered() -> None:
    def run() -> object:
        est = PublicStateEstimator()
        est.observe(PublicEvent("own_deploy", 3.0, 4.0))
        est.observe_opponent_spend(tick_seconds=5.0, card_key="hog-rider", elixir_cost=4.0)
        return est.snapshot()

    first, second = run(), run()
    assert first.tick_seconds == second.tick_seconds == 5.0
    assert np.array_equal(first.opp_hand_probs, second.opp_hand_probs)
    assert np.array_equal(first.event_history, second.event_history)
    # Spending tightens the upper elixir bound below the starting maximum.
    assert first.opp_elixir_hi < 10.0
    assert first.opp_elixir_lo <= first.opp_elixir_hi


def test_estimator_rejects_time_travel() -> None:
    est = PublicStateEstimator()
    est.observe(PublicEvent("own_deploy", 5.0, 4.0))
    with pytest.raises(ValueError, match="non-decreasing"):
        est.observe(PublicEvent("enemy_deploy", 4.0, 3.0))
    with pytest.raises(ValueError, match="unknown public event"):
        PublicEvent("nuke", 6.0, 0.0)


def test_cycle_rule_out_requires_full_deck_evidence() -> None:
    est = PublicStateEstimator(deck_size=8)
    for index in range(7):
        est.observe_opponent_spend(
            tick_seconds=float(index + 1), card_key=f"card-{index}", elixir_cost=2.0
        )
    assert not bool(est.out_of_cycle().any())
    est.observe_opponent_spend(tick_seconds=8.0, card_key="card-7", elixir_cost=2.0)
    out = est.out_of_cycle()
    assert int(out.sum()) == BELIEF_CARD_COUNT - 8
    assert float(est.hand_belief().sum()) == pytest.approx(1.0)


def test_estimator_exposes_no_strategy_api() -> None:
    est = PublicStateEstimator()
    names = " ".join(dir(est)).lower()
    assert "recommend" not in names
    assert "suggest" not in names
    assert "should_play" not in names
    assert isinstance(belief_index("hog-rider"), int)


def test_build_observation_carries_belief() -> None:
    est = PublicStateEstimator()
    est.observe_opponent_spend(tick_seconds=2.0, card_key="hog-rider", elixir_cost=4.0)
    obs = est.build_observation(_fake_v1(), hand_tokens=empty_hand_tokens())
    assert isinstance(obs, PolicyObservationV3)
    assert obs.event_history.shape == EVENT_HISTORY_SHAPE
    assert float(obs.opp_hand_probs.sum()) == pytest.approx(1.0)
    assert empty_event_history().shape == EVENT_HISTORY_SHAPE
