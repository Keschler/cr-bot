from __future__ import annotations

import pytest

from simulator.actions import PlayCardAction, WaitAction
from simulator.engine import BattleEngine
from simulator.ruleset import load_fixed_ruleset
from simulator.rl.opponent_pool import (
    ARCHETYPE_NAMES,
    OpponentPool,
    OpponentPoolError,
    RandomLegalController,
    make_opponent_controller,
)
from simulator.rl.evaluation_matrix import OpponentDeckSpec as MatrixDeckSpec


def test_pool_sampling_is_reproducible_and_covers_named_archetypes() -> None:
    ruleset = load_fixed_ruleset()
    first = OpponentPool(ruleset, seed=91)
    second = OpponentPool(ruleset, seed=91)

    first_rows = [first.sample_deck(index).as_dict() for index in range(24)]
    second_rows = [second.sample_deck(index).as_dict() for index in range(24)]

    assert first_rows == second_rows
    assert set(row["archetype"] for row in first_rows) == set(ARCHETYPE_NAMES)
    for row in first_rows:
        cards = row["cards"]
        assert len(cards) == ruleset.match.deck_size
        assert len(set(cards)) == len(cards)
        assert set(cards) <= set(ruleset.interaction_set)


def test_explicit_archetype_decks_and_scenario_are_reproducible() -> None:
    pool = OpponentPool(load_fixed_ruleset(), seed=12)
    for archetype in ARCHETYPE_NAMES:
        deck = pool.sample_deck(3, archetype=archetype)
        assert deck.archetype == archetype
        assert len(deck.cards) == 8

    first = pool.sample(7, archetype="beatdown", strategy="tank-support")
    second = OpponentPool(load_fixed_ruleset(), seed=12).sample(
        7, archetype="beatdown", strategy="tank-support"
    )
    assert first.as_dict() == second.as_dict()
    assert first.strategy == "beatdown"
    assert first.build_controller().__class__.__name__ == "BeatdownTankSupportController"


def test_unique_sampling_covers_curated_variants_and_random_decks() -> None:
    ruleset = load_fixed_ruleset()
    first = OpponentPool(ruleset, seed=91)
    second = OpponentPool(ruleset, seed=91)
    expected_curated_variants = {
        "deterministic-cycle": 1,
        "aggressive-pressure": 3,
        "defensive-cycle": 3,
        "beatdown": 3,
        "air-beatdown": 2,
        "siege-bait": 3,
    }

    for archetype, count in expected_curated_variants.items():
        first_decks = first.sample_decks(count, archetype=archetype, unique=True)
        second_decks = second.sample_decks(count, archetype=archetype, unique=True)

        assert [deck.as_dict() for deck in first_decks] == [
            deck.as_dict() for deck in second_decks
        ]
        assert len({frozenset(deck.cards) for deck in first_decks}) == count
        assert all(
            set(deck.cards) <= set(ruleset.interaction_set) for deck in first_decks
        )

    random_decks = first.sample_decks(
        16, start_index=200, archetype="random-legal", unique=True
    )
    assert len({frozenset(deck.cards) for deck in random_decks}) == 16


def test_unique_sampling_fails_closed_when_archetype_has_too_few_variants() -> None:
    pool = OpponentPool(load_fixed_ruleset(), seed=91)

    with pytest.raises(OpponentPoolError, match="could not sample 2 unique decks"):
        pool.sample_decks(2, archetype="deterministic-cycle", unique=True)


def test_evaluation_variants_preserve_archetype_core_and_are_reproducible() -> None:
    first_pool = OpponentPool(load_fixed_ruleset(), seed=91)
    second_pool = OpponentPool(load_fixed_ruleset(), seed=91)

    first = first_pool.sample_deck(
        100_000,
        archetype="air-beatdown",
        allow_variants=True,
    )
    second = second_pool.sample_deck(
        100_000,
        archetype="air-beatdown",
        allow_variants=True,
    )

    assert first.as_dict() == second.as_dict()
    assert first.source == "curated-variant"
    assert "variant" in first.tags
    assert first.cards[:2] == ("lava-hound", "balloon")
    assert set(first.cards) <= set(load_fixed_ruleset().interaction_set)


def test_unique_pool_decks_are_compatible_with_evaluation_matrix_deck_axis() -> None:
    pool = OpponentPool(load_fixed_ruleset(), seed=17)
    decks = pool.sample_decks(12, archetype=None, unique=True)

    matrix_decks = tuple(
        MatrixDeckSpec(
            deck_id=deck.deck_id,
            cards=deck.cards,
            tags=deck.tags,
            metadata={"archetype": deck.archetype, "source": deck.source},
        )
        for deck in decks
    )

    assert len(matrix_decks) == 12
    assert len({deck.deck_id for deck in matrix_decks}) == len(matrix_decks)
    assert all(len(deck.cards) == 8 for deck in matrix_decks)


def test_controller_factory_accepts_strategy_aliases() -> None:
    assert make_opponent_controller("cycle").__class__.__name__ == "DeterministicCycleController"
    assert make_opponent_controller("aggressive").__class__.__name__ == "AggressivePressureController"
    assert make_opponent_controller("defensive").__class__.__name__ == "DefensiveCycleController"
    assert make_opponent_controller("tank-support").__class__.__name__ == "BeatdownTankSupportController"
    assert make_opponent_controller("siege").__class__.__name__ == "SiegeBaitController"
    assert isinstance(make_opponent_controller("random", seed=4), RandomLegalController)


def test_all_controllers_emit_legal_actions_for_both_players_over_steps() -> None:
    ruleset = load_fixed_ruleset()
    engine = BattleEngine(ruleset=ruleset, validate_every_tick=True)
    pool = OpponentPool(ruleset, seed=4)
    strategy_names = (
        "deterministic-cycle",
        "aggressive-pressure",
        "defensive-cycle",
        "beatdown",
        "siege-bait",
        "random-legal",
    )

    for index, strategy in enumerate(strategy_names):
        deck = pool.sample_deck(index, archetype=strategy if strategy in ARCHETYPE_NAMES else "random-legal")
        state = engine.new_battle((deck.cards, deck.cards), seed=300 + index, shuffle_decks=False)
        controllers = (
            make_opponent_controller(strategy, seed=50 + index),
            make_opponent_controller(strategy, seed=50 + index),
        )
        for _ in range(12):
            if state.terminal:
                break
            actions = tuple(
                controller.choose_action(engine, state, player)
                for player, controller in enumerate(controllers)
            )
            for player, action in enumerate(actions):
                assert action.player == player if isinstance(action, (PlayCardAction, WaitAction)) else False
                assert engine.validate_action(state, action) is None
            engine.step(state, actions)
