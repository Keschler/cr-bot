from __future__ import annotations

from collections import Counter
from dataclasses import replace
import random
from types import SimpleNamespace

import pytest
import numpy as np

from simulator.rl.v4.bootstrap import (
    BootstrapConfig,
    BootstrapOpponentSpec,
    catalog_training_opponent_specs,
    generate_bootstrap_dataset,
    relabel_bootstrap_samples,
)
from simulator.rl.v4.catalog_evaluation import catalog_opponents
from simulator.rl.v4.real_decks import load_real_decks


def test_training_catalog_specs_preserve_native_identity_and_split() -> None:
    specs = catalog_training_opponent_specs()
    train = catalog_opponents()["train"]
    evaluation_decks = tuple(deck for deck in load_real_decks() if deck.split == "eval")
    evaluation_ids = {deck.deck_id for deck in evaluation_decks}

    assert len(specs) == len(train) == 30
    assert {spec.deck_id for spec in specs} == {deck["deck_id"] for deck in train}
    assert not ({spec.deck_id for spec in specs} & evaluation_ids)
    assert not (
        {frozenset(spec.cards) for spec in specs}
        & {frozenset(deck.cards) for deck in evaluation_decks}
    )
    for spec, deck in zip(specs, train):
        assert spec.deck_id == deck["deck_id"]
        assert spec.cards == tuple(deck["deck"])
        assert spec.archetype == deck["archetype"]
        assert spec.strategy == deck["strategy"]
        assert spec.split == "train"
        assert spec.source_url == deck["source_url"]
        assert spec.controller_seed == deck["controller_seed"]


@pytest.fixture
def fake_catalog_collection(monkeypatch):
    import simulator.rl.v4.bootstrap as bootstrap

    environments = []

    class FakeEnv:
        def __init__(self, engine, **kwargs):
            self.engine = engine
            self.steps = 0
            self.state = SimpleNamespace(
                terminal=False, phase="regulation", elapsed_us=0
            )
            environments.append(self)

        def reset_v2(self, **kwargs):
            self.reset_kwargs = kwargs
            return None

        def step_v2(self, actions):
            self.steps += 1
            self.state.elapsed_us = self.steps * 250_000
            self.state.terminal = self.steps >= 20

    controller = SimpleNamespace(choose_action=lambda *args: None)
    monkeypatch.setattr(bootstrap, "load_fixed_ruleset", lambda: object())
    monkeypatch.setattr(bootstrap, "BattleEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(bootstrap, "SimulatorEnv", FakeEnv)
    monkeypatch.setattr(bootstrap, "PublicStateEstimator", lambda: object())
    monkeypatch.setattr(
        bootstrap.OpponentScenario,
        "build_controller",
        lambda self: controller,
    )

    seen_teachers = []

    def capture(
        env,
        estimator,
        *,
        archetype,
        episode,
        decision,
        seed,
        teacher=None,
        action_teacher=None,
        driver_name=None,
    ):
        del estimator, archetype, seed, action_teacher, driver_name
        # One rare PLAY label and a long natural WAIT tail.  The rare sample
        # must survive the bounded reservoir even when the common stratum does
        # not fit in full.
        # Put the rare card stratum at the end so dynamic stratum discovery
        # has to redistribute the earlier WAIT surplus.
        is_play = decision == 19
        target = SimpleNamespace(mode=1 if is_play else 0, card_slot=0)
        state_id = f"rare-hog-{episode}" if is_play else f"wait-{episode}-{decision}"
        sample = SimpleNamespace(
            family="catalog",
            target=target,
            legal_play=np.zeros((4, 2, 2), dtype=bool),
            hand_tokens=np.zeros((4, 16), dtype=np.float32),
            entity_tokens=np.zeros((128, 32), dtype=np.float32),
            entity_mask=np.zeros((128,), dtype=bool),
            legal_wait=True,
            provenance={
                "episode": episode,
                "decision": decision,
                "state_hash": state_id,
                "state_id": state_id,
                "own_elixir": 7.0,
                "retention_stratum": "play:hog-rider" if is_play else "wait:no-play",
            },
            mode_weight=1.0,
            retention_stratum="",
        )
        seen_teachers.append(teacher)
        return sample, None

    monkeypatch.setattr(bootstrap, "_capture_sample", capture)
    return environments, seen_teachers


def _spec(deck_id: str = "train-test") -> BootstrapOpponentSpec:
    return BootstrapOpponentSpec(
        deck_id=deck_id,
        archetype="aggressive-pressure",
        cards=(
            "cannon",
            "fireball",
            "hog-rider",
            "ice-golem",
            "ice-spirit",
            "musketeer",
            "skeletons",
            "zap",
        ),
        strategy="aggressive-pressure",
        controller_seed=314159,
        source_url="https://example.invalid/train-test",
    )


def test_catalog_collection_preserves_controller_identity_and_inverse_strata(
    fake_catalog_collection,
) -> None:
    environments, _ = fake_catalog_collection
    summaries = []
    samples = generate_bootstrap_dataset(
        BootstrapConfig(
            n_states=4,
            seed=61,
            max_decisions=20,
            samples_per_episode=4,
            require_terminal=True,
            opponent_specs=(_spec(),),
        ),
        episode_summaries=summaries,
    )

    assert len(environments) == 1
    assert environments[0].steps == 20
    assert len(samples) == 4
    assert len({sample.provenance["state_id"] for sample in samples}) == len(samples)
    assert any(sample.provenance["retention_stratum"] == "play:hog-rider" for sample in samples)
    assert {sample.provenance["opponent_deck_id"] for sample in samples} == {"train-test"}
    assert all(
        sample.provenance["opponent_deck_cards"] == _spec().cards
        for sample in samples
    )
    assert all(sample.provenance["opponent_strategy"] == "aggressive-pressure" for sample in samples)
    assert all(sample.provenance["opponent_controller_seed"] == 314159 for sample in samples)
    assert summaries[0].opponent_controller_seed == 314159

    counts = Counter(sample.provenance["retention_stratum"] for sample in samples)
    weighted = {
        stratum: sum(
            sample.mode_weight
            for sample in samples
            if sample.provenance["retention_stratum"] == stratum
        )
        for stratum in counts
    }
    # The reservoir keeps at least one rare Hog state.  Inverse inclusion
    # weighting reconstructs the natural 1:19 eligible ratio exactly after the
    # dataset-wide mean-one normalization.
    assert weighted["wait:no-play"] / weighted["play:hog-rider"] == pytest.approx(19.0)
    assert sum(sample.mode_weight for sample in samples) == pytest.approx(len(samples))
    assert all(
        sample.provenance["eligible_stratum_count"]
        >= sample.provenance["retained_stratum_count"]
        for sample in samples
    )


def test_catalog_duplicate_state_id_is_an_input_error(fake_catalog_collection, monkeypatch) -> None:
    import simulator.rl.v4.bootstrap as bootstrap

    def duplicate_capture(
        env,
        estimator,
        *,
        archetype,
        episode,
        decision,
        seed,
        teacher=None,
        action_teacher=None,
        driver_name=None,
    ):
        del env, estimator, archetype, episode, decision, seed, teacher, action_teacher, driver_name
        sample = SimpleNamespace(
            target=SimpleNamespace(mode=0, card_slot=0),
            legal_play=np.zeros((4, 2, 2), dtype=bool),
            legal_wait=True,
            provenance={"state_id": "duplicate", "retention_stratum": "wait:no-play"},
            retention_stratum="",
        )
        return sample, None

    monkeypatch.setattr(bootstrap, "_capture_sample", duplicate_capture)
    with pytest.raises(RuntimeError, match="duplicate state ID"):
        generate_bootstrap_dataset(
            BootstrapConfig(
                n_states=4,
                max_decisions=3,
                samples_per_episode=4,
                opponent_specs=(_spec(),),
            )
        )


def test_catalog_rejects_reserved_eval_deck_before_collection() -> None:
    reserved = next(deck for deck in load_real_decks() if deck.split == "eval")
    spec = BootstrapOpponentSpec.from_real_deck(reserved)
    with pytest.raises(ValueError, match="reserved final evaluation"):
        generate_bootstrap_dataset(
            BootstrapConfig(
                n_states=2,
                max_decisions=2,
                opponent_specs=(spec,),
            )
        )


@pytest.mark.parametrize("overlap", ["id", "cards"])
def test_catalog_rejects_spoofed_reserved_identity_before_environment(
    monkeypatch, overlap: str,
) -> None:
    import simulator.rl.v4.bootstrap as bootstrap

    reserved = next(deck for deck in load_real_decks() if deck.split == "eval")
    spec = (
        replace(_spec(), deck_id=reserved.deck_id)
        if overlap == "id"
        else replace(_spec(), cards=tuple(reversed(reserved.cards)))
    )

    def unexpected_environment(*args, **kwargs):
        pytest.fail("reserved identity was not rejected before environment creation")

    monkeypatch.setattr(bootstrap, "SimulatorEnv", unexpected_environment)
    with pytest.raises(ValueError, match="reserved final evaluation"):
        generate_bootstrap_dataset(BootstrapConfig(n_states=2, opponent_specs=(spec,)))


def test_catalog_fixed_grid_ignores_state_quota_and_records_actual_match_seeds(
    fake_catalog_collection,
) -> None:
    environments, _ = fake_catalog_collection
    first = _spec()
    second = replace(
        _spec("train-second"),
        cards=(*first.cards[:-1], "the-log"),
        controller_seed=None,
    )
    config = BootstrapConfig(
        n_states=1,
        seed=13,
        max_decisions=20,
        samples_per_episode=4,
        opponent_specs=(first, second),
        episodes_per_archetype=3,
        episodes_per_opponent=2,
        require_terminal=True,
    )
    summaries = []
    samples = generate_bootstrap_dataset(config, episode_summaries=summaries)

    assert len(environments) == len(summaries) == 4
    assert len(samples) == 16
    assert [summary.opponent_deck_id for summary in summaries] == [
        first.deck_id, first.deck_id, second.deck_id, second.deck_id,
    ]
    assert len({summary.match_seed for summary in summaries}) == 4
    by_episode = {summary.episode: summary for summary in summaries}
    for env, summary in zip(environments, summaries):
        assert env.steps == 20
        assert env.reset_kwargs["seed"] == summary.match_seed
        assert env.reset_kwargs["decks"][1] == summary.opponent_deck_cards
        assert env.reset_kwargs["shuffle_decks"] is True
        assert summary.opponent_controller_seed == (
            first.controller_seed if summary.opponent_deck_id == first.deck_id
            else summary.match_seed
        )
    for sample in samples:
        summary = by_episode[sample.provenance["episode"]]
        assert sample.provenance["match_seed"] == summary.match_seed
        assert sample.provenance["opponent_controller_seed"] == summary.opponent_controller_seed
        assert sample.mode_weight == pytest.approx(
            sample.provenance["eligible_stratum_count"]
            / sample.provenance["retained_stratum_count"] * len(samples) / 80
        )
    repeated = generate_bootstrap_dataset(config)
    assert [sample.provenance for sample in repeated] == [sample.provenance for sample in samples]
    assert [sample.mode_weight for sample in repeated] == [sample.mode_weight for sample in samples]


def test_catalog_mode_provenance_counts_labels_when_driver_differs(
    fake_catalog_collection, monkeypatch,
) -> None:
    import simulator.rl.v4.bootstrap as bootstrap

    original_capture = bootstrap._capture_sample

    def wait_labels(*args, **kwargs):
        sample, action = original_capture(*args, **kwargs)
        # The driver still contributes a late PLAY stratum, while every
        # supervised label is WAIT.
        sample.target = SimpleNamespace(mode=0, card_slot=0)
        return sample, action

    monkeypatch.setattr(bootstrap, "_capture_sample", wait_labels)
    summaries = []
    samples = generate_bootstrap_dataset(
        BootstrapConfig(
            n_states=4, max_decisions=20, samples_per_episode=4,
            opponent_specs=(_spec(),),
        ),
        action_teacher=lambda **kwargs: None,
        episode_summaries=summaries,
    )
    assert {sample.retention_stratum for sample in samples} == {
        "wait:no-play", "play:hog-rider",
    }
    assert summaries[0].eligible_wait == 20
    assert summaries[0].eligible_play == 0
    assert summaries[0].retained_wait == 4
    assert summaries[0].retained_play == 0
    assert all(sample.provenance["eligible_mode_count"] == 20 for sample in samples)
    assert all(sample.provenance["retained_mode_count"] == 4 for sample in samples)


def test_late_stratum_sampling_has_exact_uniform_inclusion() -> None:
    from simulator.rl.v4.bootstrap import _EpisodeStratifiedReservoir

    class EnumeratedRng:
        def __init__(self, replacement: int, selection: int):
            self.replacement = replacement
            self.selection = selection

        def randrange(self, stop: int) -> int:
            assert stop == 3
            return self.replacement

        def sample(self, population, count: int):
            assert len(population) == 2 and count == 1
            return [population[self.selection]]

    inclusion = Counter()
    # Enumerate every equally likely reservoir replacement and final subset
    # choice.  Adding PLAY late must leave each earlier WAIT probability 1/3.
    for replacement in range(3):
        for selection in range(2):
            reservoir = _EpisodeStratifiedReservoir(
                2, EnumeratedRng(replacement, selection),
            )
            for decision in range(3):
                reservoir.add("wait:no-play", decision)
            reservoir.add("play:hog-rider", "late-hog")
            retained, counts = reservoir.finalize()
            assert counts == {"wait:no-play": 1, "play:hog-rider": 1}
            inclusion.update(retained)
    assert inclusion == {0: 2, 1: 2, 2: 2, "late-hog": 6}


def test_reservoir_retains_all_strata_under_a_hard_64_state_cap() -> None:
    from simulator.rl.v4.bootstrap import _EpisodeStratifiedReservoir

    reservoir = _EpisodeStratifiedReservoir(64, random.Random(31))
    population = {"wait:no-play": 1000, "wait:play-eligible": 100}
    population.update({f"play:card-{card}": card + 1 for card in range(8)})
    for stratum, count in population.items():
        for index in range(count):
            reservoir.add(stratum, (stratum, index))
    retained, counts = reservoir.finalize()
    assert len(retained) == sum(counts.values()) == 64
    assert set(counts) == set(population)
    assert all(1 <= counts[stratum] <= count for stratum, count in population.items())
    assert len(set(retained)) == 64
    assert reservoir.samples() == retained
    assert reservoir.retained_counts() == counts
    with pytest.raises(RuntimeError, match="finalized"):
        reservoir.add("wait:no-play", ("wait:no-play", 1001))


def test_reservoir_rejects_budget_too_small_for_stratum_support() -> None:
    from simulator.rl.v4.bootstrap import _EpisodeStratifiedReservoir

    reservoir = _EpisodeStratifiedReservoir(2, random.Random(31))
    reservoir.add("wait:no-play", 0)
    reservoir.add("wait:play-eligible", 1)
    with pytest.raises(RuntimeError, match="increase samples_per_episode"):
        reservoir.add("play:hog-rider", 2)
    assert len(reservoir.samples()) == 2


def test_relabel_preserves_state_identity_strata_and_inclusion_weights(
    fake_catalog_collection,
) -> None:
    samples = generate_bootstrap_dataset(
        BootstrapConfig(
            n_states=4,
            seed=61,
            max_decisions=20,
            samples_per_episode=4,
            opponent_specs=(_spec(),),
        )
    )
    weights = [sample.mode_weight for sample in samples]
    state_ids = [sample.provenance["state_id"] for sample in samples]
    strata = [sample.retention_stratum for sample in samples]

    def saving_teacher(**kwargs):
        del kwargs
        return SimpleNamespace(mode=0, card_slot=0, row=0, col=0, wait_duration_idx=0)

    relabeled = relabel_bootstrap_samples(
        samples, saving_teacher, teacher_name="spending-teacher"
    )
    assert [sample.mode_weight for sample in relabeled] == weights
    assert [sample.provenance["state_id"] for sample in relabeled] == state_ids
    assert [sample.retention_stratum for sample in relabeled] == strata
    assert all(sample.target.mode == 0 for sample in relabeled)
    assert all(relabel is not original for relabel, original in zip(relabeled, samples))
    assert any(sample.target.mode == 1 for sample in samples)
    for relabel, original in zip(relabeled, samples):
        assert relabel.hand_tokens is original.hand_tokens
        assert relabel.entity_tokens is original.entity_tokens
        assert relabel.legal_play is original.legal_play
        assert relabel.provenance is not original.provenance
        assert "relabel_teacher" not in original.provenance
    assert all(
        sample.provenance["relabel_teacher"] == "spending-teacher"
        for sample in relabeled
    )


def test_driver_name_is_part_of_stratification_provenance(fake_catalog_collection) -> None:
    samples = generate_bootstrap_dataset(
        BootstrapConfig(
            n_states=4,
            seed=61,
            max_decisions=20,
            samples_per_episode=4,
            opponent_specs=(_spec(),),
        ),
        driver_name="current-teacher",
    )
    assert all(
        sample.provenance["teacher_driver"] == "current-teacher"
        for sample in samples
    )
    assert all(
        sample.provenance["retention_stratum"].startswith("current-teacher:")
        for sample in samples
    )


@pytest.mark.parametrize("legacy_repeats", [None, 2])
def test_catalog_forwards_teacher_and_preserves_legacy_repeat_knob(
    fake_catalog_collection, legacy_repeats: int | None,
) -> None:
    environments, seen_teachers = fake_catalog_collection

    def custom_teacher(**kwargs):
        raise AssertionError("the fake capture hook should receive this callable")

    samples = generate_bootstrap_dataset(
        BootstrapConfig(
            n_states=2,
            max_decisions=20,
            samples_per_episode=4,
            opponent_specs=(_spec(),),
            episodes_per_archetype=legacy_repeats,
        ),
        teacher=custom_teacher,
    )
    repeats = legacy_repeats or 1
    assert len(environments) == repeats
    assert len(samples) == 4 * repeats
    assert len(seen_teachers) == 20 * repeats
    assert all(teacher is custom_teacher for teacher in seen_teachers)


def test_default_six_family_config_remains_opt_in_compatible() -> None:
    config = BootstrapConfig(n_states=2, seed=3)
    assert config.opponent_specs is None
    assert config.episodes_per_opponent is None
