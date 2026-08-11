from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.codex_annotation.merge_own_slot_interval_chunks import (
    deduplicate_exact_own_events,
    select_confirmation_frame,
    select_elixir_onset,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _decision(
    interval_id: str,
    artifact: str,
    *,
    decision: str,
    card: str | None = None,
    event_frame: int | None = None,
    confirmation_frame: int | None = None,
) -> dict[str, object]:
    return {
        "interval_id": interval_id,
        "decision": decision,
        "card": card,
        "event_frame_index": event_frame,
        "confirmation_frame_index": confirmation_frame,
        "artifact": artifact,
        "reason": "direct visible evidence",
    }


def test_select_elixir_onset_uses_card_cost_across_split_digit_drops() -> None:
    interval = {
        "empty_range": [20, 50],
        "elixir_drop_transitions": [
            {"frame_index": 28, "before": 9, "after": 8, "drop": 1},
            {"frame_index": 31, "before": 8, "after": 4, "drop": 4},
            {"frame_index": 50, "before": 5, "after": 3, "drop": 2},
            {"frame_index": 53, "before": 3, "after": 2, "drop": 1},
        ],
    }
    assert select_elixir_onset(
        interval, card="ice-spirit", fallback_event_frame=22
    ) == 28
    assert select_elixir_onset(
        interval, card="hog-rider", fallback_event_frame=30
    ) == 31

    cannon = {
        "empty_range": [49, 63],
        "elixir_drop_transitions": interval["elixir_drop_transitions"][2:],
    }
    assert select_elixir_onset(
        cannon, card="cannon", fallback_event_frame=49
    ) == 50


def test_exact_own_event_deduplication_is_deterministic() -> None:
    common = {
        "candidate_id": "own:000100",
        "card": "fireball",
        "event_frame_index": 101,
        "verification_artifacts": ["reviews/later.jpg"],
        "confirmation_frame_index": 108,
        "confirmation_artifacts": ["reviews/later-confirm.jpg"],
    }
    earlier_confirmation = {
        **common,
        "candidate_id": "own:000099",
        "verification_artifacts": ["reviews/earlier.jpg"],
        "confirmation_frame_index": 106,
        "confirmation_artifacts": ["reviews/earlier-confirm.jpg"],
    }
    distinct = {
        **common,
        "candidate_id": "own:000150",
        "event_frame_index": 151,
    }

    merged, count = deduplicate_exact_own_events(
        [common, distinct, earlier_confirmation]
    )

    assert count == 1
    assert len(merged) == 2
    selected = next(row for row in merged if row["event_frame_index"] == 101)
    assert selected["confirmation_frame_index"] == 106
    assert selected["candidate_id"] == "own:000099"


def test_confirmation_frame_extends_beyond_interval_sheet_when_needed() -> None:
    interval = {
        "interval_id": "own-slot:3:002002-002015",
        "sampled_frame_indices": list(range(2000, 2022)),
    }

    assert select_confirmation_frame(
        interval,
        event_frame=2018,
        segment_end=2918,
    ) == 2023


def test_merge_own_slot_intervals_builds_semantics_and_dedupes_rejection(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_id = "run"
    _write(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "segment": {"start_frame": 0, "end_frame_exclusive": 100},
            "candidate_discovery": {
                "own_candidates": [
                    {
                        "candidate_id": "own:000005",
                        "approximate_frame_index": 5,
                    },
                    {
                        "candidate_id": "own:000060",
                        "approximate_frame_index": 60,
                    },
                ]
            },
        },
    )
    package = {
        "run_id": run_id,
        "stage": "own_slot_intervals_package",
        "target_range": [0, 100],
        "intervals": [
            {
                    "interval_id": "own-slot:1:000010-000015",
                    "candidate_id": "own:000005",
                    "empty_range": [10, 15],
                "sampled_frame_indices": list(range(8, 22)),
                "artifact": "reviews/drag.jpg",
                "return_evidence": {
                    "outcome_constraint": "canceled",
                },
            },
            {
                    "interval_id": "own-slot:1:000030-000040",
                    "candidate_id": "own:000005",
                    "empty_range": [30, 40],
                "sampled_frame_indices": list(range(28, 47)),
                "artifact": "reviews/release.jpg",
                "return_evidence": {
                    "outcome_constraint": "released",
                },
            },
            {
                    "interval_id": "own-slot:2:000060-000070",
                    "candidate_id": "own:000060",
                    "empty_range": [60, 70],
                "sampled_frame_indices": list(range(58, 77)),
                "artifact": "reviews/canceled.jpg",
                "return_evidence": {
                    "outcome_constraint": "canceled",
                },
            },
        ],
    }
    output = {
        "run_id": run_id,
        "stage": "own_slot_intervals_chunk",
        "target_range": [0, 100],
        "annotation_session_id": "worker-1",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "low",
        "decisions": [
            _decision(
                "own-slot:1:000010-000015",
                "reviews/drag.jpg",
                decision="canceled",
            ),
            _decision(
                "own-slot:1:000030-000040",
                "reviews/release.jpg",
                decision="released",
                card="hog-rider",
                event_frame=31,
                confirmation_frame=38,
            ),
            _decision(
                "own-slot:2:000060-000070",
                "reviews/canceled.jpg",
                decision="canceled",
            ),
        ],
    }
    name = "own-slot-000000-000100.json"
    _write(run_dir / "work_packages" / name, package)
    _write(run_dir / "worker_outputs" / name, output)

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "merge_own_slot_interval_chunks.py"
    )
    subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir)],
        check=True,
    )

    merged = json.loads((run_dir / "own_semantics.json").read_text())
    assert [
        (row["card"], row["event_frame_index"])
        for row in merged["events"]
    ] == [("hog-rider", 31)]
    assert merged["events"][0]["candidate_id"].startswith(
        "completeness:own:slot-"
    )
    assert merged["events"][0]["confirmation_frame_index"] == 36
    assert merged["events"][0]["own_confirmation"]["release_confirmed"] is True
    assert merged["rejected_candidates"] == [
        {
            "candidate_id": "own:000060",
            "reason": "own_slot_canceled: direct visible evidence",
        }
    ]


def test_merge_own_slot_intervals_fails_closed_without_output(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write(run_dir / "manifest.json", {"run_id": "run"})
    _write(
        run_dir / "work_packages" / "own-slot-000000-000100.json",
        {
            "run_id": "run",
            "stage": "own_slot_intervals_package",
            "target_range": [0, 100],
            "intervals": [],
        },
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "merge_own_slot_interval_chunks.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "missing own-slot worker output" in completed.stderr
