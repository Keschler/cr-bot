from __future__ import annotations

import json

import pytest

from rl.curriculum import (
    BCTeacherConfidencePolicy,
    CurriculumConfigurationError,
    CurriculumPhase,
    CurriculumSchedule,
)
from rl.league import (
    DeckConditionedOpponentScope,
    DeckSpec,
    HistoricalCheckpoint,
    LeagueConfig,
    LeagueConfigurationError,
    LeagueOrchestrator,
    LeagueRatingBook,
    LeagueRunState,
    LeagueSampler,
    LeagueSamplingError,
    deterministic_seed,
)


def _scope() -> DeckConditionedOpponentScope:
    return DeckConditionedOpponentScope(
        scope_id="top-200-v1",
        ruleset_id="v1",
        conditioning_mode="card_tokens",
        metadata=(("population", "top-200"), ("source", "curated")),
        decks=(
            DeckSpec(
                "hog-26",
                (
                    "hog_rider",
                    "musketeer",
                    "cannon",
                    "ice_spirit",
                    "skeletons",
                    "ice_golem",
                    "fireball",
                    "the_log",
                ),
                tags=("cycle", "hog"),
            ),
            DeckSpec(
                "pekka-bridge",
                (
                    "pekka",
                    "battle_ram",
                    "bandit",
                    "royal_ghost",
                    "electro_wizard",
                    "poison",
                    "zap",
                    "magic_archer",
                ),
            ),
            DeckSpec(
                "lava",
                (
                    "lava_hound",
                    "balloon",
                    "mega_minion",
                    "skeleton_dragons",
                    "tombstone",
                    "fireball",
                    "arrows",
                    "guards",
                ),
            ),
        ),
    )


def _checkpoints() -> tuple[HistoricalCheckpoint, ...]:
    return (
        HistoricalCheckpoint(
            checkpoint_id="main-step-100",
            agent_id="main",
            role="main_agent",
            step=100,
            artifact="checkpoints/main-100.pt",
            deck_scope_id="top-200-v1",
            deck_ids=("hog-26", "pekka-bridge"),
            metadata=(("lineage", "main"),),
        ),
        HistoricalCheckpoint(
            checkpoint_id="exploit-step-80",
            agent_id="exploit-a",
            role="exploiter",
            step=80,
            artifact="checkpoints/exploit-a-80.pt",
            deck_scope_id="top-200-v1",
            deck_ids=("lava",),
            metadata=(("lineage", "exploiter"),),
        ),
    )


def _league(**overrides) -> LeagueConfig:
    values = {
        "league_id": "hog-league-v1",
        "seed": 17,
        "scope": _scope(),
        "main_agent_id": "main",
        "exploiter_ids": ("exploit-a", "exploit-b"),
        "historical_checkpoints": _checkpoints(),
    }
    values.update(overrides)
    return LeagueConfig(**values)


def test_bc_confidence_policy_is_explicit_and_drops_uncertain_labels() -> None:
    policy = BCTeacherConfidencePolicy()

    assert policy.band(0.95) == "high"
    assert policy.weight(0.95) == pytest.approx(1.0)
    assert policy.band(0.80) == "medium"
    assert policy.weight(0.80) == pytest.approx(0.5)
    assert policy.band(0.55) == "low"
    assert policy.weight(0.55) == pytest.approx(0.25)
    assert policy.band(0.20) == "rejected"
    assert policy.weight(0.20) == 0.0

    with pytest.raises(CurriculumConfigurationError):
        policy.band(1.1)


def test_curriculum_phase_boundaries_and_json_round_trip() -> None:
    schedule = CurriculumSchedule(
        schedule_id="hog-bc-to-rl-v1",
        seed=9,
        phases=(
            CurriculumPhase(
                phase_id="clean-heuristic",
                start_step=0,
                end_step=100,
                teacher_id="heuristic-v1",
                bc_loss_coefficient=1.0,
                description="Only explicitly accepted heuristic labels.",
            ),
            CurriculumPhase(
                phase_id="confidence-mix",
                start_step=100,
                end_step=250,
                teacher_id="high-confidence-replay",
                bc_loss_coefficient=0.25,
            ),
            CurriculumPhase(
                phase_id="rl-dominant",
                start_step=250,
                teacher_id="high-confidence-replay",
                bc_loss_coefficient=0.05,
            ),
        ),
    )

    assert schedule.phase_at(0).phase_id == "clean-heuristic"
    assert schedule.phase_at(99).phase_id == "clean-heuristic"
    assert schedule.phase_at(100).phase_id == "confidence-mix"
    assert schedule.phase_at(250).phase_id == "rl-dominant"

    early = schedule.decide(25, 0.95)
    late = schedule.decide(275, 0.80)
    rejected = schedule.decide(25, 0.20)
    assert early.accepted is True
    assert early.loss_weight == pytest.approx(1.0)
    assert late.loss_weight == pytest.approx(0.025)
    assert rejected.accepted is False
    assert rejected.loss_weight == 0.0

    encoded = schedule.to_json()
    restored = CurriculumSchedule.from_mapping(json.loads(encoded))
    assert restored == schedule
    assert json.loads(encoded)["schema_version"] == 1


def test_curriculum_phases_must_be_contiguous() -> None:
    with pytest.raises(CurriculumConfigurationError, match="contiguous"):
        CurriculumSchedule(
            schedule_id="gap",
            phases=(
                CurriculumPhase("first", 0, 10),
                CurriculumPhase("second", 11, None),
            ),
        )

    with pytest.raises(CurriculumConfigurationError, match="first"):
        CurriculumSchedule(
            schedule_id="late-start",
            phases=(CurriculumPhase("first", 1, None),),
        )


def test_league_config_round_trip_preserves_deck_conditioned_scope() -> None:
    config = _league()
    restored = LeagueConfig.from_mapping(json.loads(config.to_json()))

    assert restored == config
    assert restored.scope.scope_id == "top-200-v1"
    assert restored.scope.ruleset_id == "v1"
    assert restored.scope.conditioning_mode == "card_tokens"
    assert restored.scope.deck_ids == ("hog-26", "pekka-bridge", "lava")
    assert dict(restored.scope.metadata) == {
        "population": "top-200",
        "source": "curated",
    }


def test_league_sampling_is_deterministic_and_records_historical_artifact() -> None:
    config = _league()
    sampler_a = LeagueSampler(config)
    sampler_b = LeagueSampler(LeagueConfig.from_mapping(json.loads(config.to_json())))

    selections_a = [sampler_a.sample("main", index) for index in range(24)]
    selections_b = [sampler_b.sample("main", index) for index in range(24)]
    assert [item.as_dict() for item in selections_a] == [
        item.as_dict() for item in selections_b
    ]
    assert len({item.selection_seed for item in selections_a}) == 24

    historical = [item for item in selections_a if item.source == "historical"]
    assert historical
    for item in historical:
        assert item.checkpoint_id is not None
        assert item.artifact is not None
        checkpoint = next(
            value
            for value in config.historical_checkpoints
            if value.checkpoint_id == item.checkpoint_id
        )
        assert item.deck_id in checkpoint.deck_ids
        assert item.checkpoint_metadata
        assert item.scope_metadata == config.scope.metadata


def test_main_and_exploiter_roles_use_their_declared_source_pools() -> None:
    main_history_only = _league(
        main_agent_historical_probability=1.0,
        main_agent_exploiter_probability=0.0,
    )
    main_selection = LeagueSampler(main_history_only).sample("main", 0)
    assert main_selection.learner_role == "main_agent"
    assert main_selection.source == "historical"

    exploiter_main_only = _league(
        exploiter_main_agent_probability=1.0,
        exploiter_historical_probability=0.0,
    )
    exploiter_selection = LeagueSampler(exploiter_main_only).sample("exploit-a", 0)
    assert exploiter_selection.learner_role == "exploiter"
    assert exploiter_selection.source == "main_agent"
    assert exploiter_selection.opponent_agent_id == "main"
    assert exploiter_selection.checkpoint_id is None


def test_league_state_and_seed_are_replayable_without_global_rng() -> None:
    state = LeagueRunState("hog-league-v1")
    advanced = state.after_match(3)
    assert advanced.next_match_index == 3
    assert LeagueRunState.from_mapping(advanced.as_dict()) == advanced

    first = deterministic_seed(17, "hog-league-v1", "main", 0)
    second = deterministic_seed(17, "hog-league-v1", "main", 1)
    assert first != second
    assert first == deterministic_seed(17, "hog-league-v1", "main", 0)


def test_league_rejects_scope_or_deck_mismatches() -> None:
    with pytest.raises(LeagueConfigurationError, match="different deck scope"):
        _league(
            historical_checkpoints=(
                HistoricalCheckpoint(
                    checkpoint_id="wrong-scope",
                    agent_id="main",
                    role="main_agent",
                    step=1,
                    artifact="checkpoints/wrong.pt",
                    deck_scope_id="other-scope",
                    deck_ids=("hog-26",),
                ),
            )
        )

    with pytest.raises(LeagueSamplingError, match="not available"):
        LeagueSampler(_league()).sample("main", 0, deck_id="not-in-scope")


def test_league_requires_candidates_for_positive_sampling_weights() -> None:
    with pytest.raises(LeagueConfigurationError, match="historical checkpoints"):
        _league(
            historical_checkpoints=(),
            main_agent_historical_probability=1.0,
            main_agent_exploiter_probability=0.0,
            exploiter_main_agent_probability=1.0,
            exploiter_historical_probability=0.0,
        )


def test_orchestrator_records_outcomes_and_separates_frozen_checkpoint_rating() -> None:
    config = _league(
        main_agent_historical_probability=1.0,
        main_agent_exploiter_probability=0.0,
    )
    orchestrator = LeagueOrchestrator(config)
    record = orchestrator.run_one("main", lambda selection: "win")

    assert record.match_index == 0
    assert record.opponent_agent_id == "main"
    assert record.opponent_rating_id == "main@main-step-100"
    assert record.learner_rating_after > record.learner_rating_before
    assert orchestrator.run_state.next_match_index == 1
    assert orchestrator.ratings.rating("main") > 1500.0
    restored = LeagueRatingBook.from_mapping(orchestrator.ratings.as_dict())
    assert restored == orchestrator.ratings
