from __future__ import annotations

import json

import pytest

from simulator.engine import BASE_HOG_CYCLE_DECK, ENGINE_VERSION, BattleEngine
from simulator.cli import main as simulator_main
from simulator.fidelity import DatasetSplit
from simulator.validation import (
    ValidationCorpusError,
    apply_fidelity_gate,
    evaluate_fidelity_corpus,
    load_validation_corpus,
    load_validation_corpus_pinned,
    validation_corpus_from_dict,
)


def _scenario(engine: BattleEngine, *, split: str = "heldout") -> dict[str, object]:
    return {
        "schema_version": 1,
        "scenario_id": f"hog-cannon-{split}",
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "seed": 0,
        "shuffle_decks": False,
        "decks": [list(BASE_HOG_CYCLE_DECK), list(BASE_HOG_CYCLE_DECK)],
        "actions": [
            {
                "tick": 0,
                "action": {
                    "kind": "play",
                    "player": 0,
                    "card_slot": 0,
                    "cell": [3, 23],
                },
            },
            {
                "tick": 30,
                "action": {
                    "kind": "play",
                    "player": 1,
                    "card_slot": 1,
                    "cell": [3, 13],
                },
            },
        ],
        "max_ticks": 160,
        "split": split,
        "tags": ["targeting"],
        "oracle": {"promoted": split != "candidate"},
    }


def _evidence(source_id: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "group_id": source_id,
        "method": "controlled_capture_120fps_v1",
        "confidence": 0.99,
        "notes": "Test fixture standing in for a preassigned observed capture.",
    }


def _heldout_case(engine: BattleEngine) -> dict[str, object]:
    return {
        "case_id": "heldout-hog-cannon",
        "split": "heldout",
        "evidence": _evidence("capture-heldout-001"),
        "scenario": _scenario(engine),
        "measurements": [
            {
                "sample_id": "heldout:pull-tick",
                "mechanic": "hog_cannon_targeting",
                "observed_value": 30,
                "observed_tick": 30,
                "tolerance": {"absolute": 0, "ticks": 0},
                "extractor": {
                    "type": "first_event_tick",
                    "event_kind": "target_changed",
                    "filters": {
                        "card_id": "hog-rider",
                        "target_card_id": "cannon",
                    },
                },
            },
            {
                "sample_id": "heldout:target-card",
                "mechanic": "hog_cannon_targeting",
                "observed_value": "cannon",
                "extractor": {
                    "type": "first_event_field",
                    "event_kind": "target_changed",
                    "field": "target_card_id",
                    "filters": {
                        "card_id": "hog-rider",
                        "target_card_id": "cannon",
                    },
                },
            },
            {
                "sample_id": "heldout:cannon-final-hp",
                "mechanic": "cannon_combat",
                # Synthetic fixture pinned to the current engine outcome. The
                # remaining 12 HP includes placement-started linear lifetime
                # decay and deployment-time Hog target acquisition.
                "observed_value": 12,
                "tolerance": {"absolute": 0},
                "extractor": {
                    "type": "final_entity_hp",
                    "filters": {"owner": 1, "card_id": "cannon"},
                },
            },
        ],
        "traces": [
            {
                "trace_id": "heldout:hog-target-sequence",
                "mechanic": "hog_cannon_targeting",
                "included_event_kinds": ["target_changed"],
                "filters": {"card_id": "hog-rider"},
                "events": [
                    {
                        "tick": 19,
                        "kind": "target_changed",
                        "values": {"target_card_id": "princess-tower"},
                    },
                    {
                        "tick": 30,
                        "kind": "target_changed",
                        "values": {"target_card_id": "cannon"},
                    },
                ],
            }
        ],
    }


def _corpus(engine: BattleEngine) -> dict[str, object]:
    validation_case = {
        "case_id": "validation-intentional-mismatch",
        "split": "validation",
        "evidence": _evidence("capture-validation-001"),
        "scenario": _scenario(engine, split="validation"),
        "measurements": [
            {
                "sample_id": "validation:wrong-count",
                "mechanic": "split_isolation",
                "observed_value": 999,
                "extractor": {
                    "type": "event_count",
                    "event_kind": "card_played",
                },
            }
        ],
        "traces": [],
    }
    return {
        "schema_version": 1,
        "corpus_id": "controlled-hog-cycle-test-v1",
        "engine_version": ENGINE_VERSION,
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "cases": [_heldout_case(engine), validation_case],
    }


def test_corpus_runs_scenarios_and_reports_only_preassigned_heldout_data() -> None:
    engine = BattleEngine()
    corpus = validation_corpus_from_dict(_corpus(engine))

    report, samples, traces = evaluate_fidelity_corpus(
        engine,
        corpus,
        split=DatasetSplit.HELDOUT,
    )

    assert len(samples) == 3
    assert len(traces) == 1
    payload = report.to_dict()
    assert payload["dataset_split"] == "heldout"
    assert payload["corpus_id"] == "controlled-hog-cycle-test-v1"
    assert payload["corpus_hash"] == corpus.content_hash
    assert payload["engine_version"] == ENGINE_VERSION
    assert payload["ruleset_hash"] == engine.ruleset.content_hash
    assert payload["tick_us"] == engine.ruleset.tick_us
    assert corpus.content_hash.startswith("sha256:")
    assert payload["overall"]["samples"]["count"] == 3
    assert payload["overall"]["samples"]["agreement_count"] == 3
    assert payload["overall"]["traces"]["agreement_count"] == 1
    assert payload["overall"]["mechanic_count"] == 2
    assert payload["overall"]["observation_group_count"] == 1
    assert payload["excluded_by_split"] == {
        "validation": {"sample_comparisons": 1, "trace_comparisons": 0}
    }
    assert payload["trace_divergences"] == []
    assert payload["case_results"][0]["case_id"] == "heldout-hog-cannon"
    assert len(payload["case_results"][0]["replay_hash"]) == 64
    interval = payload["mechanics"]["hog_cannon_targeting"]["samples"][
        "agreement_confidence_interval"
    ]
    assert interval["lower"] < 1.0 <= interval["upper"]


def test_missing_simulator_output_is_an_explicit_failed_measurement() -> None:
    engine = BattleEngine()
    raw = _corpus(engine)
    heldout = raw["cases"][0]
    heldout["measurements"] = [
        {
            "sample_id": "heldout:missing",
            "mechanic": "missing_mechanic",
            "observed_value": 1,
            "extractor": {
                "type": "first_event_field",
                "event_kind": "never-emitted",
                "field": "damage",
            },
        }
    ]
    heldout["traces"] = []
    corpus = validation_corpus_from_dict(raw)

    report, samples, _ = evaluate_fidelity_corpus(engine, corpus)

    missing = next(item for item in samples if item.sample_id == "heldout:missing")
    assert not missing.agrees
    assert missing.simulated is None
    assert missing.reason == "missing_simulation"
    assert report.to_dict()["overall"]["samples"]["simulated_count"] == 0


def test_position_samples_report_trajectory_error_at_pinned_ticks() -> None:
    engine = BattleEngine()
    raw = _corpus(engine)
    heldout = raw["cases"][0]
    heldout["measurements"] = [
        {
            "sample_id": "heldout:hog-x-60",
            "mechanic": "hog_movement_x",
            "observed_value": 3_520,
            "observed_tick": 60,
            "tolerance": {"absolute": 25, "ticks": 0},
            "extractor": {
                "type": "entity_x_mtile_at_tick",
                "tick": 60,
                "filters": {"owner": 0, "card_id": "hog-rider"},
            },
        },
        {
            "sample_id": "heldout:hog-y-60",
            "mechanic": "hog_movement_y",
            "observed_value": 18_500,
            "observed_tick": 60,
            "tolerance": {"absolute": 100, "ticks": 0},
            "extractor": {
                "type": "entity_y_mtile_at_tick",
                "tick": 60,
                "filters": {"owner": 0, "card_id": "hog-rider"},
            },
        },
    ]
    heldout["traces"] = []

    report, samples, _ = evaluate_fidelity_corpus(
        engine,
        validation_corpus_from_dict(raw),
    )

    heldout_samples = [
        sample for sample in samples if sample.observed.split is DatasetSplit.HELDOUT
    ]
    assert [sample.simulated.value for sample in heldout_samples if sample.simulated] == [3_500, 18_580]
    assert all(sample.agrees for sample in heldout_samples)
    assert report.to_dict()["mechanics"]["hog_movement_y"]["samples"]["mae"] == 80


def test_displacement_speed_extractor_uses_two_pinned_snapshots() -> None:
    engine = BattleEngine()
    raw = _corpus(engine)
    heldout = raw["cases"][0]
    heldout["measurements"] = [
        {
            "sample_id": "heldout:hog-speed-20-60",
            "mechanic": "hog_displacement_speed",
            "observed_value": 2_400,
            "observed_tick": 60,
            "tolerance": {"absolute": 250, "ticks": 0},
            "extractor": {
                "type": "entity_displacement_speed_mtile_per_s",
                "start_tick": 20,
                "end_tick": 60,
                "filters": {"owner": 0, "card_id": "hog-rider"},
            },
        }
    ]
    heldout["traces"] = []

    _, samples, _ = evaluate_fidelity_corpus(
        engine,
        validation_corpus_from_dict(raw),
    )

    sample = next(item for item in samples if item.sample_id == "heldout:hog-speed-20-60")
    assert sample.simulated is not None
    assert sample.simulated.tick == 60
    assert sample.simulated.value == 2_400
    assert sample.agrees


def test_card_move_speed_extractor_does_not_depend_on_an_inferred_target() -> None:
    engine = BattleEngine()
    raw = _corpus(engine)
    heldout = raw["cases"][0]
    heldout["measurements"] = [
        {
            "sample_id": "heldout:hog-base-speed",
            "mechanic": "hog_isolated_movement_speed",
            "observed_value": 2_400,
            "observed_tick": 20,
            "tolerance": {"absolute": 120, "ticks": 0},
            "extractor": {
                "type": "card_move_speed_mtile_per_s",
                "filters": {"owner": 0, "card_id": "hog-rider"},
            },
        }
    ]
    heldout["traces"] = []

    _, samples, _ = evaluate_fidelity_corpus(
        engine,
        validation_corpus_from_dict(raw),
    )

    sample = next(item for item in samples if item.sample_id == "heldout:hog-base-speed")
    assert sample.simulated is not None
    assert sample.simulated.value == 2_400
    assert sample.simulated.tick == 20
    assert sample.agrees


def test_file_corpus_supports_safe_relative_scenarios_and_atomic_report(tmp_path) -> None:
    engine = BattleEngine()
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    scenario_path = scenario_dir / "pull.json"
    scenario_path.write_text(json.dumps(_scenario(engine)), encoding="utf-8")
    heldout = _heldout_case(engine)
    heldout.pop("scenario")
    heldout["scenario_path"] = "scenarios/pull.json"
    raw = {
        "schema_version": 1,
        "corpus_id": "file-corpus",
        "engine_version": ENGINE_VERSION,
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "cases": [heldout],
    }
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_validation_corpus(corpus_path)
    sealed_hash = loaded.content_hash
    assert load_validation_corpus_pinned(
        corpus_path,
        expected_hash=sealed_hash,
    ).content_hash == sealed_hash
    with pytest.raises(ValidationCorpusError, match="expected pin"):
        load_validation_corpus_pinned(
            corpus_path,
            expected_hash="sha256:" + "0" * 64,
        )
    scenario_path.write_text("{}", encoding="utf-8")
    report, _, _ = evaluate_fidelity_corpus(engine, loaded)
    report_path = tmp_path / "report.json"
    report.write_json(report_path)

    assert loaded.base_dir == tmp_path.resolve()
    assert loaded.cases[0].scenario is not None
    assert loaded.cases[0].scenario_path is None
    assert report.corpus_hash == sealed_hash
    assert json.loads(report_path.read_text(encoding="utf-8")) == report.to_dict()
    assert not (tmp_path / ".report.json.tmp").exists()


@pytest.mark.parametrize(
    "unsafe_path",
    ("../scenario.json", "/tmp/scenario.json", "https://example/scenario.json", "a\\b.json"),
)
def test_corpus_rejects_scenario_path_escape(unsafe_path: str) -> None:
    engine = BattleEngine()
    case = _heldout_case(engine)
    case.pop("scenario")
    case["scenario_path"] = unsafe_path
    raw = {
        "schema_version": 1,
        "corpus_id": "unsafe",
        "engine_version": ENGINE_VERSION,
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "cases": [case],
    }

    with pytest.raises(ValidationCorpusError, match="path|relative|traverse"):
        validation_corpus_from_dict(raw)


def test_corpus_rejects_ruleset_and_split_mismatch() -> None:
    engine = BattleEngine()
    bad_hash = _corpus(engine)
    bad_hash["ruleset_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationCorpusError, match="hash"):
        validation_corpus_from_dict(bad_hash)

    bad_split = _corpus(engine)
    bad_split["cases"][0]["scenario"]["split"] = "validation"
    with pytest.raises(ValidationCorpusError, match="split"):
        validation_corpus_from_dict(bad_split)

    bad_engine = _corpus(engine)
    bad_engine["engine_version"] = "reference-future"
    with pytest.raises(ValidationCorpusError, match="engine version"):
        validation_corpus_from_dict(bad_engine)


def test_evidence_group_cannot_cross_validation_and_heldout_splits() -> None:
    engine = BattleEngine()
    raw = _corpus(engine)
    raw["cases"][1]["evidence"]["group_id"] = raw["cases"][0]["evidence"]["group_id"]

    with pytest.raises(ValidationCorpusError, match="occurs in both"):
        validation_corpus_from_dict(raw)


def test_fidelity_gate_rejects_empty_or_underperforming_required_mechanics() -> None:
    engine = BattleEngine()
    report, _, _ = evaluate_fidelity_corpus(
        engine,
        validation_corpus_from_dict(_corpus(engine)),
    )

    passing = apply_fidelity_gate(
        report,
        min_observations=4,
        min_agreement_rate=1.0,
        required_mechanics=("hog_cannon_targeting", "cannon_combat"),
    )
    assert passing.gate is not None and passing.gate["passed"] is True

    failing = apply_fidelity_gate(
        report,
        min_observations=5,
        min_agreement_rate=1.0,
        required_mechanics=("hog_movement",),
    )
    assert failing.gate is not None and failing.gate["passed"] is False
    assert len(failing.gate["failures"]) == 2


def test_fidelity_cli_returns_nonzero_for_empty_heldout_split(tmp_path) -> None:
    engine = BattleEngine()
    corpus_path = tmp_path / "empty.json"
    report_path = tmp_path / "report.json"
    corpus_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus_id": "empty-heldout",
                "engine_version": ENGINE_VERSION,
                "ruleset_id": engine.ruleset.ruleset_id,
                "ruleset_hash": engine.ruleset.content_hash,
                "cases": [],
            }
        ),
        encoding="utf-8",
    )

    exit_code = simulator_main(
        ["fidelity", str(corpus_path), "--json-out", str(report_path)]
    )

    assert exit_code == 2
    assert json.loads(report_path.read_text(encoding="utf-8"))["gate"]["passed"] is False


def test_ambiguous_final_entity_selector_fails_instead_of_choosing_one() -> None:
    engine = BattleEngine()
    scenario = _scenario(engine)
    scenario["actions"] = [
        {
            "tick": 0,
            "action": {
                "kind": "play",
                "player": 0,
                "card_slot": 3,
                "cell": [8, 20],
            },
        }
    ]
    case = _heldout_case(engine)
    case["scenario"] = scenario
    case["measurements"] = [
        {
            "sample_id": "heldout:ambiguous-skeleton",
            "mechanic": "skeleton_hp",
            "observed_value": 81,
            "extractor": {
                "type": "final_entity_hp",
                "filters": {"owner": 0, "card_id": "skeletons"},
            },
        }
    ]
    case["traces"] = []
    raw = {
        "schema_version": 1,
        "corpus_id": "ambiguous",
        "engine_version": ENGINE_VERSION,
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "cases": [case],
    }

    with pytest.raises(ValidationCorpusError, match="matched 3 final entities"):
        evaluate_fidelity_corpus(engine, validation_corpus_from_dict(raw))
