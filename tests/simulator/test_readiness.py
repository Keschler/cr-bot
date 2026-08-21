from __future__ import annotations

import json

import pytest

from simulator.cli import main as simulator_main
from simulator.engine import ENGINE_VERSION, BattleEngine
from simulator.readiness import (
    ReadinessError,
    build_training_readiness_report,
    declared_mechanics_for_ruleset,
)


def _report(engine: BattleEngine, *, split: str, group: str, count: int = 20) -> dict:
    return {
        "schema_version": 1,
        "dataset_split": split,
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "mechanics": {
            "hog_cannon_pull_targeting": {
                "samples": {"count": count, "agreement_count": count},
                "traces": {"count": 0, "agreement_count": 0},
                "evidence": {
                    "group_ids": [group],
                    "source_ids": [f"source-{group}"],
                    "methods": ["offline_vision"],
                },
            }
        },
    }


def test_calibration_never_satisfies_heldout_readiness(tmp_path) -> None:
    engine = BattleEngine()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_report(engine, split="calibration", group="used")), encoding="utf-8")

    report = build_training_readiness_report(
        [path],
        ruleset_id=engine.ruleset.ruleset_id,
        ruleset_hash=engine.ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        requirements={"hog_cannon_targeting": ("hog_cannon_pull_targeting",)},
        minimum_heldout_groups=1,
    )

    assert not report["summary"]["ready"]
    assert report["mechanics"]["hog_cannon_targeting"]["status"] == "calibrated_only"


def test_independent_passing_heldout_evidence_marks_mechanic_validated(tmp_path) -> None:
    engine = BattleEngine()
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps(_report(engine, split="heldout", group="untouched")), encoding="utf-8")

    report = build_training_readiness_report(
        [path],
        ruleset_id=engine.ruleset.ruleset_id,
        ruleset_hash=engine.ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        requirements={"hog_cannon_targeting": ("hog_cannon_pull_targeting",)},
        minimum_heldout_groups=1,
    )

    assert report["summary"]["ready"]
    assert report["mechanics"]["hog_cannon_targeting"]["status"] == "heldout_validated"


def test_cross_split_group_leakage_fails_closed(tmp_path) -> None:
    engine = BattleEngine()
    paths = []
    for split in ("calibration", "heldout"):
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(_report(engine, split=split, group="same-video")), encoding="utf-8")
        paths.append(path)

    report = build_training_readiness_report(
        paths,
        ruleset_id=engine.ruleset.ruleset_id,
        ruleset_hash=engine.ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        requirements={"hog_cannon_targeting": ("hog_cannon_pull_targeting",)},
        minimum_heldout_groups=1,
    )

    assert not report["summary"]["ready"]
    assert report["heldout_leakage_groups"] == ["same-video"]


def test_wrong_ruleset_identity_is_rejected(tmp_path) -> None:
    engine = BattleEngine()
    raw = _report(engine, split="heldout", group="untouched")
    raw["ruleset_hash"] = "sha256:" + "0" * 64
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ReadinessError, match="does not match"):
        build_training_readiness_report(
            [path],
            ruleset_id=engine.ruleset.ruleset_id,
            ruleset_hash=engine.ruleset.content_hash,
            engine_version=ENGINE_VERSION,
        )


def test_single_heldout_group_cannot_pass_independence_gate(tmp_path) -> None:
    engine = BattleEngine()
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps(_report(engine, split="heldout", group="only-one")), encoding="utf-8")

    report = build_training_readiness_report(
        [path],
        ruleset_id=engine.ruleset.ruleset_id,
        ruleset_hash=engine.ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        requirements={"hog_cannon_targeting": ("hog_cannon_pull_targeting",)},
    )

    assert not report["summary"]["ready"]
    assert report["mechanics"]["hog_cannon_targeting"]["status"] == "heldout_failed"


def test_candidate_report_is_visible_but_cannot_satisfy_readiness(tmp_path) -> None:
    engine = BattleEngine()
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "log_motion_candidate_report",
                "ruleset_id": engine.ruleset.ruleset_id,
                "ruleset_hash": engine.ruleset.content_hash,
                "engine_version": ENGINE_VERSION,
                "cache_hash": "sha256:" + "1" * 64,
                "mechanics": {"log_rolling_speed": {"candidate_count": 2}},
            }
        ),
        encoding="utf-8",
    )

    report = build_training_readiness_report(
        [],
        candidate_report_paths=[path],
        ruleset_id=engine.ruleset.ruleset_id,
        ruleset_hash=engine.ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        requirements={"log_rolling_motion": ("log_rolling_speed",)},
    )

    mechanic = report["mechanics"]["log_rolling_motion"]
    assert mechanic["status"] == "candidate_only"
    assert mechanic["candidate_evidence"]["candidate_count"] == 2
    assert mechanic["candidate_evidence"]["can_satisfy_heldout_gate"] is False
    assert not report["summary"]["ready"]


def test_zero_candidate_attempt_is_distinct_from_missing_evidence(tmp_path) -> None:
    engine = BattleEngine()
    path = tmp_path / "rejected.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "cannon_lifetime_candidate_report",
                "ruleset_id": engine.ruleset.ruleset_id,
                "ruleset_hash": engine.ruleset.content_hash,
                "engine_version": ENGINE_VERSION,
                "cache_hash": "sha256:" + "2" * 64,
                "mechanics": {"cannon_lifetime_hp_decay": {"candidate_count": 0}},
            }
        ),
        encoding="utf-8",
    )

    report = build_training_readiness_report(
        [],
        candidate_report_paths=[path],
        ruleset_id=engine.ruleset.ruleset_id,
        ruleset_hash=engine.ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        requirements={"cannon_lifetime_hp_decay": ("cannon_lifetime*",)},
    )

    assert report["mechanics"]["cannon_lifetime_hp_decay"]["status"] == "candidate_rejected"


def test_candidate_cache_hash_cannot_reappear_as_heldout_media(tmp_path) -> None:
    engine = BattleEngine()
    media_hash = "sha256:" + "3" * 64
    heldout = _report(engine, split="heldout", group="renamed-group")
    heldout["case_results"] = [{"media_hash": media_hash}]
    heldout_path = tmp_path / "heldout.json"
    heldout_path.write_text(json.dumps(heldout), encoding="utf-8")
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "log_motion_candidate_report",
                "ruleset_id": engine.ruleset.ruleset_id,
                "ruleset_hash": engine.ruleset.content_hash,
                "engine_version": ENGINE_VERSION,
                "cache_hash": media_hash,
                "mechanics": {"log_rolling_speed": {"candidate_count": 1}},
            }
        ),
        encoding="utf-8",
    )

    report = build_training_readiness_report(
        [heldout_path],
        candidate_report_paths=[candidate_path],
        ruleset_id=engine.ruleset.ruleset_id,
        ruleset_hash=engine.ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        requirements={"hog_cannon_targeting": ("hog_cannon_pull_targeting",)},
        minimum_heldout_groups=1,
    )

    assert report["heldout_leakage_groups"] == []
    assert report["heldout_leakage_media_hashes"] == [media_hash]
    assert not report["summary"]["ready"]


def test_autonomous_interaction_batch_is_candidate_only_and_tracks_all_cache_hashes(tmp_path) -> None:
    engine = BattleEngine()
    media_hash = "sha256:" + "4" * 64
    candidate_path = tmp_path / "autonomous.json"
    candidate_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "autonomous_interaction_candidate_batch",
                "ruleset_id": engine.ruleset.ruleset_id,
                "ruleset_hash": engine.ruleset.content_hash,
                "engine_version": ENGINE_VERSION,
                "cache_hashes": [media_hash],
                "sources": [{"cache_hash": media_hash}],
                "mechanics": {
                    "hog-rider_bridge_path_topology": {"candidate_count": 1}
                },
            }
        ),
        encoding="utf-8",
    )
    report = build_training_readiness_report(
        [],
        candidate_report_paths=[candidate_path],
        ruleset_id=engine.ruleset.ruleset_id,
        ruleset_hash=engine.ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        requirements={"hog_bridge_pathfinding": ("hog-rider_bridge_path_topology",)},
    )
    assert report["mechanics"]["hog_bridge_pathfinding"]["status"] == "candidate_only"
    assert report["mechanics"]["hog_bridge_pathfinding"]["candidate_evidence"]["candidate_count"] == 1


def test_readiness_cli_writes_failure_report_when_no_heldout_evidence(tmp_path) -> None:
    output = tmp_path / "readiness.json"

    exit_code = simulator_main(["readiness", "--json-out", str(output)])

    assert exit_code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["ready"] is False
    assert payload["summary"]["heldout_validated_count"] == 0


def test_ruleset_readiness_matrix_expands_to_every_card_component() -> None:
    ruleset = BattleEngine().ruleset
    requirements = declared_mechanics_for_ruleset(ruleset)
    assert "card:hog-rider:movement" in requirements
    assert "card:fireball:spell_geometry" in requirements
