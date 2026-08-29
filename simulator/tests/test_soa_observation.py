from __future__ import annotations

import numpy as np


def _observation_arrays(observation):
    return (
        observation.board,
        observation.global_vector,
        observation.spatial_masks,
        observation.legal_play,
    )


def test_soa_observation_matches_reference_projection() -> None:
    from simulator.actions import WaitAction
    from simulator.engine import BattleEngine
    from simulator.observation import ObservationMemory, build_policy_observation
    from simulator.ruleset import load_ruleset
    from simulator.soa import ObservationSoA

    ruleset = load_ruleset("v1")
    engine = BattleEngine(ruleset, validate_every_tick=False)
    deck = (
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
        "fireball",
        "log",
    )
    state = engine.new_battle((deck, deck), seed=7, shuffle_decks=False)
    for player in state.players:
        player.elixir_milli = ruleset.match.max_elixir_milli
    engine._spawn_single_at(
        state,
        ruleset.card("hog-rider"),
        owner=0,
        x_mtile=3_500,
        y_mtile=22_000,
        deploy_remaining_us=0,
    )

    soa_state = ObservationSoA()
    for _ in range(4):
        for viewer in (0, 1):
            expected = build_policy_observation(
                state,
                ruleset,
                viewer=viewer,
                memory=ObservationMemory(viewer),
                legal_action_cells_callback=engine.legal_action_cells,
            )
            actual = build_policy_observation(
                state,
                ruleset,
                viewer=viewer,
                memory=ObservationMemory(viewer),
                legal_action_cells_callback=engine.legal_action_cells,
                soa_state=soa_state,
            )
            for expected_array, actual_array in zip(
                _observation_arrays(expected),
                _observation_arrays(actual),
                strict=True,
            ):
                np.testing.assert_array_equal(actual_array, expected_array)
            assert actual.legal_wait == expected.legal_wait
        engine.step(state, (WaitAction(0), WaitAction(1)))


def test_soa_static_legality_matches_engine_for_policy_cards() -> None:
    from simulator.engine import BattleEngine
    from simulator.ruleset import load_ruleset
    from simulator.soa import ObservationSoA

    ruleset = load_ruleset("v1")
    engine = BattleEngine(ruleset, validate_every_tick=False)
    deck = (
        "fireball",
        "log",
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
    )
    state = engine.new_battle((deck, deck), seed=13, shuffle_decks=False)
    for player in state.players:
        player.elixir_milli = ruleset.match.max_elixir_milli

    soa_state = ObservationSoA()
    soa_state.sync(state, lambda card_id: card_id)
    for viewer in (0, 1):
        assert soa_state.legal_action_cells_if_static(
            state, ruleset, viewer
        ) == engine.legal_action_cells(state, viewer)

    engine._spawn_single_at(
        state,
        ruleset.card("cannon"),
        owner=0,
        x_mtile=3_500,
        y_mtile=20_500,
        deploy_remaining_us=0,
    )
    soa_state.sync(state, lambda card_id: card_id)
    assert soa_state.legal_action_cells_if_static(state, ruleset, 0) is None


def test_soa_observation_matches_reference_with_dynamic_building() -> None:
    from simulator.engine import BattleEngine
    from simulator.observation import ObservationMemory, build_policy_observation
    from simulator.ruleset import load_ruleset
    from simulator.soa import ObservationSoA

    ruleset = load_ruleset("v1")
    engine = BattleEngine(ruleset, validate_every_tick=False)
    deck = (
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
        "fireball",
        "log",
    )
    state = engine.new_battle((deck, deck), seed=29, shuffle_decks=False)
    engine._spawn_single_at(
        state,
        ruleset.card("cannon"),
        owner=0,
        x_mtile=3_500,
        y_mtile=20_500,
        deploy_remaining_us=0,
    )

    soa_state = ObservationSoA()
    for viewer in (0, 1):
        expected = build_policy_observation(
            state,
            ruleset,
            viewer=viewer,
            memory=ObservationMemory(viewer),
            legal_action_cells_callback=engine.legal_action_cells,
        )
        actual = build_policy_observation(
            state,
            ruleset,
            viewer=viewer,
            memory=ObservationMemory(viewer),
            legal_action_cells_callback=engine.legal_action_cells,
            soa_state=soa_state,
        )
        for expected_array, actual_array in zip(
            _observation_arrays(expected),
            _observation_arrays(actual),
            strict=True,
        ): 
            np.testing.assert_array_equal(actual_array, expected_array)
        assert actual.legal_wait == expected.legal_wait


def test_soa_v2_entity_rows_match_legacy_projection() -> None:
    from simulator.observation import ObservationMemory
    from simulator.observation_v2_adapter import build_policy_observation_v2
    from simulator.engine import BattleEngine
    from simulator.ruleset import load_ruleset
    from simulator.soa import ObservationSoA

    ruleset = load_ruleset("v1")
    engine = BattleEngine(ruleset, validate_every_tick=False)
    deck = (
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
        "fireball",
        "log",
    )
    state = engine.new_battle((deck, deck), seed=31, shuffle_decks=False)
    engine._spawn_single_at(
        state,
        ruleset.card("hog-rider"),
        owner=0,
        x_mtile=3_500,
        y_mtile=22_000,
        deploy_remaining_us=0,
    )
    engine._spawn_single_at(
        state,
        ruleset.card("cannon"),
        owner=1,
        x_mtile=14_500,
        y_mtile=9_500,
        deploy_remaining_us=0,
    )

    soa_state = ObservationSoA()
    for viewer in (0, 1):
        expected = build_policy_observation_v2(
            state,
            ruleset,
            viewer=viewer,
            memory=ObservationMemory(viewer),
            legal_action_cells_callback=engine.legal_action_cells,
        )
        actual = build_policy_observation_v2(
            state,
            ruleset,
            viewer=viewer,
            memory=ObservationMemory(viewer),
            legal_action_cells_callback=engine.legal_action_cells,
            soa_state=soa_state,
        )
        for expected_array, actual_array in zip(
            (expected.board, expected.global_vector, expected.entity_tokens,
             expected.entity_mask, expected.legal_play),
            (actual.board, actual.global_vector, actual.entity_tokens,
             actual.entity_mask, actual.legal_play),
            strict=True,
        ):
            np.testing.assert_array_equal(actual_array, expected_array)
        assert actual.legal_wait == expected.legal_wait
