from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "codex_annotation"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    packages = run_dir / "work_packages"
    outputs = run_dir / "worker_outputs"
    run_id = "fixture-run"
    onset_id = "enemy-unit-000100-b000001"
    _write(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "segment": {
                "start_frame": 0,
                "end_frame_exclusive": 400,
            },
        },
    )
    _write(
        run_dir / "enemy_onsets.json",
        {
            "run_id": run_id,
            "stage": "enemy_onsets",
            "onsets": [
                {
                    "onset_id": onset_id,
                    "event_frame_index": 100,
                    "kind": "unit_or_building",
                    "absence_confirmed": True,
                    "persistence_confirmed": True,
                }
            ],
        },
    )
    overlap_package = {
        "run_id": run_id,
        "fps": 10.0,
        "target_range": [0, 400],
        "decision_schema_version": 2,
        "candidates": [
            {
                "onset_id": onset_id,
                "event_frame_index": 100,
                "sampled_frame_indices": [95, 100, 104, 109],
                "focus_artifact": "reviews/focus.jpg",
            }
        ],
    }
    _write(packages / "identity-overlap-000000-000400.json", overlap_package)
    _write(
        outputs / "identity-overlap-000000-000400.json",
        {
            "run_id": run_id,
            "stage": "enemy_overlap_adjudication_chunk",
            "target_range": [0, 400],
            "decisions": [
                {
                    "onset_id": onset_id,
                    "overlap_event_exists": True,
                    "event_frame_index": 100,
                    "evidence": {
                        "secondary_absent_before": True,
                        "secondary_appears_at_marker": True,
                        "secondary_persists_or_resolves_after": True,
                        "direct_new_actor": True,
                    },
                    "side": "unresolved",
                    "reason": "direct actor",
                }
            ],
        },
    )
    side_package = {
        "run_id": run_id,
        "fps": 10.0,
        "target_range": [0, 400],
        "decision_schema_version": 2,
        "candidates": [
            {
                "onset_id": onset_id,
                "approximate_frame_index": 100,
                "event_frame_index": 100,
                "full_arena_artifact": "reviews/arena.jpg",
                "existence_evidence": {
                    "secondary_absent_before": True,
                    "secondary_appears_at_marker": True,
                    "secondary_persists_or_resolves_after": True,
                    "direct_new_actor": True,
                },
                "existence_reason": "direct actor",
                "source_onset_ids": [onset_id],
            }
        ],
    }
    _write(packages / "identity-side-000000-000400.json", side_package)
    _write(
        outputs / "identity-side-000000-000400.json",
        {
            "run_id": run_id,
            "stage": "enemy_side_check_chunk",
            "target_range": [0, 400],
            "model": "gpt-5.6-terra",
            "decisions": [
                {
                    "onset_id": onset_id,
                    "side": "own",
                    "direct": True,
                    "team_indicator": "blue",
                    "origin": "lower",
                    "motion": "upward",
                    "reason": "direct blue actor",
                }
            ],
        },
    )
    return run_dir


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_prepare_and_merge_simultaneous_enemy_recovery(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    _run(
        "prepare_simultaneous_enemy_recovery_packages.py",
        "--run-dir",
        str(run_dir),
    )
    package_path = (
        run_dir
        / "work_packages"
        / "identity-simultaneous-recovery-000000-000400.json"
    )
    package = json.loads(package_path.read_text())
    assert package["task"] == "additional_simultaneous_enemy_actor"
    assert package["candidates"] == [
        {
            "onset_id": "enemy-unit-000100-b000001",
            "event_frame_index": 100,
            "sampled_frame_indices": [95, 100, 104, 109],
            "full_arena_artifact": "reviews/arena.jpg",
            "focus_artifact": "reviews/focus.jpg",
            "known_own_actor": {
                "side": "own",
                "direct": True,
                "team_indicator": "blue",
                "origin": "lower",
                "motion": "upward",
                "reason": "direct blue actor",
            },
            "existence_reason": "direct actor",
        }
    ]

    _write(
        run_dir / "worker_outputs" / package_path.name,
        {
            "run_id": "fixture-run",
            "stage": "enemy_overlap_adjudication_chunk",
            "target_range": [0, 400],
            "annotation_session_id": "recovery-session",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "medium",
            "decisions": [
                {
                    "onset_id": "enemy-unit-000100-b000001",
                    "overlap_event_exists": True,
                    "event_frame_index": 104,
                    "evidence": {
                        "secondary_absent_before": True,
                        "secondary_appears_at_marker": True,
                        "secondary_persists_or_resolves_after": True,
                        "direct_new_actor": True,
                    },
                    "side": "unresolved",
                    "reason": "a separate red actor resolves behind the blue one",
                }
            ],
        },
    )
    _run(
        "merge_simultaneous_enemy_recovery_chunks.py",
        "--run-dir",
        str(run_dir),
    )
    result = json.loads(
        (
            run_dir
            / "recovery_outputs"
            / "enemy_simultaneous_recoveries.json"
        ).read_text()
    )
    assert result["stage"] == "enemy_simultaneous_recoveries"
    assert result["recoveries"][0]["event_frame_index"] == 104
    assert result["recoveries"][0]["side"] == "enemy"
    assert result["recoveries"][0]["verification_artifacts"] == [
        "reviews/arena.jpg",
        "reviews/focus.jpg",
    ]
    _run(
        "merge_enemy_unit_gate_chunks.py",
        "--run-dir",
        str(run_dir),
    )
    identities = json.loads(
        (run_dir / "enemy_identities.json").read_text()
    )
    recovered = [
        row
        for row in identities["decisions"]
        if row["onset_id"].startswith("enemy-simultaneous-unit-")
    ]
    assert len(recovered) == 1
    assert recovered[0]["event_frame_index"] == 104
    assert recovered[0]["side"] == "enemy"
    # Resume-safe deterministic stages may execute again after a worker/budget
    # pause. Rebuilding the same stable recovery must not duplicate or reject
    # it.
    _run(
        "merge_enemy_unit_gate_chunks.py",
        "--run-dir",
        str(run_dir),
    )
    identities = json.loads(
        (run_dir / "enemy_identities.json").read_text()
    )
    assert sum(
        row["onset_id"].startswith("enemy-simultaneous-unit-")
        for row in identities["decisions"]
    ) == 1


def test_simultaneous_recovery_deduplicates_known_enemy_frame() -> None:
    path = SCRIPTS / "merge_enemy_unit_gate_chunks.py"
    spec = importlib.util.spec_from_file_location("enemy_gate_merge", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recovery = {"event_frame_index": 104, "onset_id": "recovered"}
    kept, duplicates = module._novel_simultaneous_recoveries(
        [recovery],
        accepted_enemy_frames=[100],
    )
    assert kept == []
    assert duplicates == [recovery]


def test_unresolved_side_is_targeted_and_overlaid(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    base_path = (
        run_dir
        / "worker_outputs"
        / "identity-side-000000-000400.json"
    )
    base = json.loads(base_path.read_text())
    base["decisions"][0].update(
        {
            "side": "unresolved",
            "direct": False,
            "team_indicator": None,
            "origin": None,
            "motion": None,
            "reason": "occluded in the cheap pass",
        }
    )
    _write(base_path, base)
    _run(
        "prepare_enemy_side_escalation_packages.py",
        "--run-dir",
        str(run_dir),
    )
    package_path = (
        run_dir
        / "work_packages"
        / "identity-side-escalation-000000-000400.json"
    )
    package = json.loads(package_path.read_text())
    assert len(package["candidates"]) == 1
    _write(
        run_dir / "worker_outputs" / package_path.name,
        {
            "run_id": "fixture-run",
            "stage": "enemy_side_check_chunk",
            "target_range": [0, 400],
            "annotation_session_id": "sol-escalation",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "medium",
            "decisions": [
                {
                    "onset_id": "enemy-unit-000100-b000001",
                    "side": "enemy",
                    "direct": True,
                    "team_indicator": "red",
                    "origin": "upper",
                    "motion": "downward",
                    "reason": "red actor resolves in the focused review",
                }
            ],
        },
    )
    _run(
        "merge_enemy_side_escalation_chunks.py",
        "--run-dir",
        str(run_dir),
    )
    _run(
        "merge_enemy_unit_gate_chunks.py",
        "--run-dir",
        str(run_dir),
    )
    identities = json.loads(
        (run_dir / "enemy_identities.json").read_text()
    )
    event = next(
        row
        for row in identities["decisions"]
        if row["onset_id"] == "enemy-unit-000100-b000001"
    )
    assert event["event_exists"] is True
    assert event["side"] == "enemy"


def test_prepare_rejects_non_terra_canonical_side_output(
    tmp_path: Path,
) -> None:
    run_dir = _fixture_run(tmp_path)
    path = (
        run_dir
        / "worker_outputs"
        / "identity-side-000000-000400.json"
    )
    value = json.loads(path.read_text())
    value["model"] = "gpt-5.6-sol"
    _write(path, value)
    completed = subprocess.run(
        [
            sys.executable,
            str(
                SCRIPTS
                / "prepare_simultaneous_enemy_recovery_packages.py"
            ),
            "--run-dir",
            str(run_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "canonical side output is not Terra" in completed.stderr
