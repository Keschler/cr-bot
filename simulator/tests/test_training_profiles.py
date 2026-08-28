from __future__ import annotations

import json

import pytest

from simulator.engine import ENGINE_VERSION
from simulator.ruleset import load_ruleset
from simulator.roster import PLAYER_DECK
from simulator.trainer import PPOConfig, PPOTrainer
from simulator.training_profiles import (
    TrainingProfile,
    TrainingProfileError,
    validate_training_profile,
)


def test_smoke_profile_is_explicitly_allowed_but_keeps_ruleset_identity() -> None:
    profile = TrainingProfile(
        profile_id="hog-smoke",
        ruleset_id="v1",
        opponent_decks=(PLAYER_DECK,),
        purpose="smoke",
    )

    result = validate_training_profile(profile)

    assert result["training_ready"] is True
    assert result["readiness_source"] == "explicit_provisional_smoke"
    assert result["ruleset_hash"] == load_ruleset("v1").content_hash


def test_serious_profile_requires_a_matching_ready_report(tmp_path) -> None:
    ruleset = load_ruleset("v1")
    report = {
        "profile_id": "hog-core",
        "ruleset_id": ruleset.ruleset_id,
        "ruleset_hash": ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "summary": {"ready": True},
        "cards": {"hog-rider": {"status": "fidelity_ready"}},
        "mechanics": {"hog_isolated_movement": {"status": "heldout_validated"}},
    }
    report_path = tmp_path / "readiness.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    profile = TrainingProfile(
        profile_id="hog-core",
        ruleset_id="v1",
        opponent_decks=(PLAYER_DECK,),
        required_cards=("hog-rider",),
        required_mechanics=("hog_isolated_movement",),
        readiness_report=report_path,
    )

    result = validate_training_profile(profile)

    assert result["training_ready"] is True
    assert result["readiness_source"] == str(report_path)


def test_serious_profile_rejects_provisional_ruleset_without_report() -> None:
    profile = TrainingProfile(
        profile_id="hog-core",
        ruleset_id="v1",
        required_cards=("hog-rider",),
    )

    with pytest.raises(TrainingProfileError, match="requires a readiness_report"):
        validate_training_profile(profile)


def test_serious_profile_must_declare_a_scoped_requirement(tmp_path) -> None:
    report_path = tmp_path / "ready.json"
    report_path.write_text(
        json.dumps(
            {
                "ruleset_id": load_ruleset("v1").ruleset_id,
                "ruleset_hash": load_ruleset("v1").content_hash,
                "engine_version": ENGINE_VERSION,
                "summary": {"ready": True},
            }
        ),
        encoding="utf-8",
    )
    profile = TrainingProfile(
        profile_id="unscoped",
        ruleset_id="v1",
        readiness_report=report_path,
    )

    with pytest.raises(TrainingProfileError, match="at least one required"):
        validate_training_profile(profile)


def test_profile_rejects_non_fixed_player_deck() -> None:
    profile = TrainingProfile(
        profile_id="bad",
        ruleset_id="v1",
        player_deck=tuple(reversed(PLAYER_DECK)),
        purpose="smoke",
    )

    with pytest.raises(TrainingProfileError, match="fixed Hog-cycle"):
        validate_training_profile(profile)


def test_trainer_accepts_a_ready_scoped_profile_without_global_roster_readiness(tmp_path) -> None:
    ruleset = load_ruleset("v1")
    report_path = tmp_path / "readiness.json"
    report_path.write_text(
        json.dumps(
            {
                "profile_id": "hog-core",
                "ruleset_id": ruleset.ruleset_id,
                "ruleset_hash": ruleset.content_hash,
                "engine_version": ENGINE_VERSION,
                "summary": {"ready": True},
                "cards": {"hog-rider": {"status": "fidelity_ready"}},
            }
        ),
        encoding="utf-8",
    )
    profile = TrainingProfile(
        profile_id="hog-core",
        ruleset_id="v1",
        purpose="training",
        required_cards=("hog-rider",),
        readiness_report=report_path,
    )

    trainer = PPOTrainer(
        PPOConfig(
            training_profile=profile,
            total_steps=1,
            rollout_steps=1,
            num_envs=1,
            checkpoint_every=1,
            eval_every=1,
            eval_episodes=1,
        )
    )

    assert trainer.training_profile_result is not None
    assert trainer.training_profile_result["training_ready"] is True
    assert trainer.checkpoint_metadata()["training_profile_id"] == "hog-core"
