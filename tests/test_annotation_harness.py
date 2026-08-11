from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cr_bot.annotation_harness import (
    ANNOTATION_TYPE,
    LABEL_MARGIN_PX,
    PROMPT_VERSION,
    build_scan_windows,
    cluster_own_signals,
    finalize_annotation,
    prepare_annotation_run,
    render_review_sheet,
    sha256_file,
    validate_decisions,
)
from cr_bot.annotation_stages import (
    WORKFLOW_VERSION,
    assemble_staged_decisions,
    checkpoint_stage,
)


def _write_test_video(path: Path, *, fps: float = 10.0, frames: int = 4) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (108, 240),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video writer is unavailable")
    try:
        for index in range(frames):
            image = np.full((240, 108, 3), (index * 40) % 256, dtype=np.uint8)
            cv2.rectangle(image, (10 + index, 20), (40 + index, 80), (0, 0, 255), -1)
            writer.write(image)
    finally:
        writer.release()


def test_prepare_run_preserves_source_indices_and_renders_review(tmp_path: Path):
    video = tmp_path / "source.avi"
    _write_test_video(video)
    run_dir = tmp_path / "run"

    manifest = prepare_annotation_run(
        video_path=video,
        output_dir=run_dir,
        start_time_s=0.1,
        end_time_s=0.3,
        own_change_threshold=0.0,
    )

    assert manifest["annotation_type"] == ANNOTATION_TYPE
    assert manifest["workflow_version"] == WORKFLOW_VERSION
    assert manifest["segment"]["start_frame"] == 1
    assert manifest["segment"]["end_frame_exclusive"] == 3
    assert [row["source_frame_index"] for row in manifest["frames"]] == [1, 2]
    assert [row["video_time_s"] for row in manifest["frames"]] == [0.1, 0.2]
    prepared = cv2.imread(str(run_dir / manifest["frames"][0]["path"]))
    assert prepared.shape[:2] == (2400 + LABEL_MARGIN_PX, 1080)

    review = render_review_sheet(
        run_dir=run_dir,
        output_path=run_dir / "review.jpg",
        start_frame=1,
        end_frame=3,
        purpose="arena",
        columns=2,
        tile_width=200,
    )
    assert review.is_file()
    assert cv2.imread(str(review)).shape[1] == 400
    assert (run_dir / "verification.json").is_file()
    assert (run_dir / "localization.json").is_file()
    assert (run_dir / "completeness.json").is_file()
    assert (run_dir / "checkpoints.json").is_file()


def test_enemy_scan_windows_are_exhaustive_and_shifted():
    windows = build_scan_windows(
        start_frame=0,
        end_frame=40,
        fps=10.0,
        window_seconds=2.0,
        overlap_seconds=1.0,
    )

    assert windows[0]["inspection_start_frame"] == 0
    assert windows[0]["inspection_end_frame_exclusive"] == 20
    assert any(
        row["inspection_start_frame"] == 5
        and row["sources"] == ["exhaustive_enemy_scan_pass_2"]
        for row in windows
    )
    covered = {
        frame
        for row in windows
        for frame in range(
            row["inspection_start_frame"], row["inspection_end_frame_exclusive"]
        )
    }
    assert covered == set(range(40))


def test_v5_enemy_scan_uses_compact_boundary_pass():
    windows = build_scan_windows(
        start_frame=0,
        end_frame=40,
        fps=10.0,
        window_seconds=1.0,
        overlap_seconds=0.0,
    )

    primary = [
        row for row in windows if row["candidate_id"].startswith("enemy-scan:")
    ]
    boundary = [
        row
        for row in windows
        if row["candidate_id"].startswith("enemy-boundary:")
    ]
    assert [
        (row["inspection_start_frame"], row["inspection_end_frame_exclusive"])
        for row in primary
    ] == [(0, 10), (10, 20), (20, 30), (30, 40)]
    assert [
        (row["inspection_start_frame"], row["inspection_end_frame_exclusive"])
        for row in boundary
    ] == [(8, 12), (18, 22), (28, 32)]


def test_v7_splits_long_continuous_own_interaction_into_reviewable_candidates():
    signals = [
        {"source_frame_index": frame, "change_score": score}
        for frame, score in [
            (3, 20.0),
            (5, 22.0),
            (10, 11.0),
            (17, 12.0),
            (18, 21.0),
            (22, 12.0),
            (28, 10.0),
            (30, 11.0),
        ]
    ]

    candidates = cluster_own_signals(
        signals,
        fps=10.0,
        max_gap_frames=7,
        segment_start_frame=0,
        segment_end_frame=100,
    )

    assert [row["candidate_id"] for row in candidates] == [
        "own:000005",
        "own:000018",
        "own:000030",
    ]
    assert candidates[-1]["inspection_start_frame"] <= 29
    assert candidates[-1]["inspection_end_frame_exclusive"] > 31


def test_v7_rejects_unreadable_review_sheet(tmp_path: Path):
    video = tmp_path / "source.avi"
    _write_test_video(video, frames=22)
    run_dir = tmp_path / "run"
    prepare_annotation_run(video_path=video, output_dir=run_dir)

    with pytest.raises(ValueError, match="maximum is 20"):
        render_review_sheet(
            run_dir=run_dir,
            output_path=run_dir / "too-many.jpg",
            start_frame=0,
            end_frame=21,
            purpose="arena",
        )


def test_v7_enemy_unit_requires_later_identity_evidence(tmp_path: Path):
    video = tmp_path / "source.avi"
    _write_test_video(video, frames=25)
    run_dir = tmp_path / "run"
    prepare_annotation_run(video_path=video, output_dir=run_dir)
    event_id = "event-enemy-000001-mega-knight"
    identity_scope = "identity-enemy-000001"
    onset_a = render_review_sheet(
        run_dir=run_dir,
        output_path=run_dir / "reviews/onset-a.jpg",
        start_frame=0,
        end_frame=10,
        purpose="arena",
    )
    onset_b = render_review_sheet(
        run_dir=run_dir,
        output_path=run_dir / "reviews/onset-b.jpg",
        start_frame=10,
        end_frame=12,
        purpose="arena",
    )
    identity = render_review_sheet(
        run_dir=run_dir,
        output_path=run_dir / "reviews/identity.jpg",
        start_frame=4,
        end_frame=8,
        event_id=identity_scope,
        purpose="identity",
        focus_cell=(2, 8),
    )
    identity_later = render_review_sheet(
        run_dir=run_dir,
        output_path=run_dir / "reviews/identity-later.jpg",
        start_frame=8,
        end_frame=12,
        event_id=identity_scope,
        purpose="identity",
        focus_cell=(2, 8),
    )
    verification = json.loads((run_dir / "verification.json").read_text())
    candidate_id = next(
        row["candidate_id"]
        for row in json.loads((run_dir / "manifest.json").read_text())[
            "candidate_discovery"
        ]["enemy_scan_windows"]
        if row["candidate_id"].startswith("enemy-scan:")
    )
    verification.update(
        {
            "annotation_session_id": "sol-primary-test",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "medium",
        }
    )
    verification["events"] = [
        {
            "event_id": event_id,
            "candidate_id": candidate_id,
            "side": "enemy",
            "card": "mega-knight",
            "event_frame_index": 1,
            "evidence": {
                "elixir_drop": None,
                "hand_transition": None,
                "deployment_onset": True,
                "first_visible_object": True,
                "side_direction": True,
                "impact_sequence": None,
            },
            "ambiguity": "none",
            "verification_artifacts": [
                str(onset_a.relative_to(run_dir)),
                str(onset_b.relative_to(run_dir)),
            ],
            "confirmation_frame_index": None,
            "confirmation_artifacts": [],
            "own_confirmation": None,
            "identity_frame_index": None,
            "identity_artifacts": [],
        }
    ]
    (run_dir / "verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity_frame_index"):
        checkpoint_stage(run_dir, "verification")

    verification["events"][0]["identity_frame_index"] = 4
    verification["events"][0]["identity_artifacts"] = [
        str(identity.relative_to(run_dir)),
        str(identity_later.relative_to(run_dir)),
    ]
    (run_dir / "verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="at least 6"):
        checkpoint_stage(run_dir, "verification")

    verification["events"][0]["identity_frame_index"] = 6
    (run_dir / "verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    checkpoint = checkpoint_stage(run_dir, "verification")
    assert checkpoint["event_count"] == 1

    boundary_candidate_id = next(
        row["candidate_id"]
        for row in json.loads((run_dir / "manifest.json").read_text())[
            "candidate_discovery"
        ]["enemy_scan_windows"]
        if row["candidate_id"].startswith("enemy-boundary:")
    )
    verification["events"][0]["candidate_id"] = boundary_candidate_id
    (run_dir / "verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="outside candidate"):
        checkpoint_stage(run_dir, "verification")


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    manifest = {
        "run_id": "test-run",
        "annotation_type": ANNOTATION_TYPE,
        "prompt_version": PROMPT_VERSION,
        "video": str(video),
        "video_sha256": sha256_file(video),
        "fps": 10.0,
        "segment": {
            "start_time_s": 0.0,
            "end_time_s": 10.0,
            "start_frame": 0,
            "end_frame_exclusive": 100,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, manifest


def _decisions() -> dict:
    return {
        "run_id": "test-run",
        "events": [
            {
                "candidate_id": "own:000010",
                "side": "own",
                "card": "hog-rider",
                "event_frame_index": 10,
                "location_frame_index": 11,
                "location_rule": "spawn_center",
                "cell": [3, 17],
                "evidence": {
                    "elixir_drop": True,
                    "hand_transition": True,
                    "deployment_onset": True,
                    "first_visible_object": True,
                    "side_direction": None,
                    "impact_sequence": None,
                },
                "ambiguity": "none",
                "review_artifacts": ["review-own-000010.jpg"],
            }
        ],
        "rejected_candidates": [],
        "completeness_sweeps": [
            {"side": "own", "completed": True, "notes": ""},
            {"side": "enemy", "completed": True, "notes": ""},
        ],
        "adjudications": [],
    }


def test_finalize_validates_writes_and_locks_without_overwrite(tmp_path: Path):
    run_dir, _ = _manifest(tmp_path)
    (run_dir / "review-own-000010.jpg").write_bytes(b"review image")
    decisions_path = run_dir / "decisions.json"
    decisions_path.write_text(json.dumps(_decisions()), encoding="utf-8")
    final_path = tmp_path / "ground_truth.json"
    audit_path = tmp_path / "ground_truth.audit.json"

    final, audit, lock = finalize_annotation(
        run_dir=run_dir,
        decisions_path=decisions_path,
        output_path=final_path,
        audit_output_path=audit_path,
    )

    compact = json.loads(final.read_text(encoding="utf-8"))
    audit_data = json.loads(audit.read_text(encoding="utf-8"))
    lock_data = json.loads(lock.read_text(encoding="utf-8"))
    assert compact["events"] == [
        {
            "side": "own",
            "card": "hog-rider",
            "frame_index": 10,
            "cell": [3, 17],
        }
    ]
    assert audit_data["locked"] is True
    assert audit_data["final_sha256"] == sha256_file(final)
    assert audit_data["accepted_events"][0]["review_artifacts"] == [
        {
            "path": str((run_dir / "review-own-000010.jpg").resolve()),
            "sha256": sha256_file(run_dir / "review-own-000010.jpg"),
        }
    ]
    assert lock_data["audit_sha256"] == sha256_file(audit)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize_annotation(
            run_dir=run_dir,
            decisions_path=decisions_path,
            output_path=final_path,
            audit_output_path=audit_path,
        )


def test_validation_requires_sweeps_evidence_and_valid_cell(tmp_path: Path):
    _, manifest = _manifest(tmp_path)
    decisions = _decisions()
    decisions["completeness_sweeps"][1]["completed"] = False
    with pytest.raises(ValueError, match="own and enemy"):
        validate_decisions(manifest, decisions)

    decisions = _decisions()
    decisions["events"][0]["cell"] = [18, 0]
    with pytest.raises(ValueError, match="invalid.*cell"):
        validate_decisions(manifest, decisions)

    decisions = _decisions()
    decisions["events"][0]["evidence"]["elixir_drop"] = False
    with pytest.raises(ValueError, match="own event requires"):
        validate_decisions(manifest, decisions)


def test_v7_enforces_release_review_verification_localization_completeness_order(tmp_path: Path):
    video = tmp_path / "source.avi"
    _write_test_video(video, frames=8)
    run_dir = tmp_path / "staged"
    manifest = prepare_annotation_run(
        video_path=video,
        output_dir=run_dir,
        start_time_s=0.0,
        end_time_s=0.8,
        own_change_threshold=999.0,
    )
    verification_review = render_review_sheet(
        run_dir=run_dir,
        output_path=run_dir / "reviews/verify.jpg",
        start_frame=0,
        end_frame=4,
        purpose="own_context",
    )
    confirmation_review = render_review_sheet(
        run_dir=run_dir,
        output_path=run_dir / "reviews/confirm.jpg",
        start_frame=5,
        end_frame=8,
        event_id="release-completeness-own-000001",
        purpose="own_confirmation",
    )
    verification = json.loads((run_dir / "verification.json").read_text())
    verification.update(
        {
            "annotation_session_id": "terra-primary-test",
            "model": "terra",
            "reasoning_effort": "medium",
        }
    )
    verification["events"] = [
        {
            "event_id": "event-own-000001-hog-rider",
            "candidate_id": "completeness:own:000001",
            "side": "own",
            "card": "hog-rider",
            "event_frame_index": 1,
            "evidence": {
                "elixir_drop": True,
                "hand_transition": True,
                "deployment_onset": True,
                "first_visible_object": True,
                "side_direction": None,
                "impact_sequence": None,
            },
            "ambiguity": "none",
            "verification_artifacts": [
                str(verification_review.relative_to(run_dir))
            ],
            "confirmation_frame_index": 6,
            "confirmation_artifacts": [
                str(confirmation_review.relative_to(run_dir))
            ],
            "own_confirmation": {
                "release_confirmed": True,
                "elixir_spend_persisted": True,
                "hand_cycle_completed": True,
                "post_release_effect": True,
            },
            "identity_frame_index": None,
            "identity_artifacts": [],
        }
    ]
    (run_dir / "verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="checkpoint verification"):
        render_review_sheet(
            run_dir=run_dir,
            output_path=run_dir / "reviews/too-early.jpg",
            start_frame=1,
            end_frame=2,
            event_id="event-own-000001-hog-rider",
            purpose="grid",
            grid_center=(3, 17),
        )

    canceled_drag = json.loads(json.dumps(verification))
    canceled_drag["events"][0]["own_confirmation"]["release_confirmed"] = False
    (run_dir / "verification.json").write_text(
        json.dumps(canceled_drag), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="own confirmation requires"):
        checkpoint_stage(run_dir, "verification")

    (run_dir / "verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="release_review checkpoint"):
        checkpoint_stage(run_dir, "verification")

    release_review = json.loads((run_dir / "release_review.json").read_text())
    release_review.update(
        {
            "annotation_session_id": "sol-release-review-test",
            "model": "sol",
            "reasoning_effort": "medium",
            "reviews": [
                {
                    "event_id": "event-own-000001-hog-rider",
                    "decision": "released",
                    "confirmation_frame_index": 6,
                    "confirmation_artifacts": [
                        str(confirmation_review.relative_to(run_dir))
                    ],
                    "checks": verification["events"][0]["own_confirmation"],
                }
            ],
        }
    )
    (run_dir / "release_review.json").write_text(
        json.dumps(release_review), encoding="utf-8"
    )
    release_review["annotation_session_id"] = verification["annotation_session_id"]
    (run_dir / "release_review.json").write_text(
        json.dumps(release_review), encoding="utf-8"
    )
    checkpoint_stage(run_dir, "release_review")
    with pytest.raises(ValueError, match="fresh annotation_session_id"):
        checkpoint_stage(run_dir, "verification")

    release_review["annotation_session_id"] = "sol-release-review-test"
    (run_dir / "release_review.json").write_text(
        json.dumps(release_review), encoding="utf-8"
    )
    checkpoint_stage(run_dir, "release_review")
    checkpoint_stage(run_dir, "verification")
    macro = render_review_sheet(
        run_dir=run_dir,
        output_path=run_dir / "reviews/macro.jpg",
        start_frame=1,
        end_frame=2,
        event_id="event-own-000001-hog-rider",
        purpose="macro",
    )
    grid = render_review_sheet(
        run_dir=run_dir,
        output_path=run_dir / "reviews/grid.jpg",
        start_frame=1,
        end_frame=2,
        event_id="event-own-000001-hog-rider",
        purpose="grid",
        grid_center=(3, 17),
    )
    localization = json.loads((run_dir / "localization.json").read_text())
    localization.update(
        {
            "annotation_session_id": "terra-primary-test",
            "model": "terra",
            "reasoning_effort": "medium",
        }
    )
    localization["locations"] = [
        {
            "event_id": "event-own-000001-hog-rider",
            "location_frame_index": 1,
            "location_rule": "spawn_center",
            "cell": [3, 17],
            "ambiguity": "none",
            "unavailable_reason": None,
            "macro_review_artifacts": [str(macro.relative_to(run_dir))],
            "grid_review_artifacts": [str(grid.relative_to(run_dir))],
            "adjudication_artifacts": [],
        }
    ]
    illegal = json.loads(json.dumps(localization))
    illegal["locations"][0]["cell"] = [1, 15]
    (run_dir / "localization.json").write_text(
        json.dumps(illegal), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not a legal own deployment"):
        checkpoint_stage(run_dir, "localization")

    unavailable = json.loads(json.dumps(localization))
    unavailable["locations"][0].update(
        {
            "location_rule": "unavailable",
            "cell": None,
            "ambiguity": "unscorable",
            "unavailable_reason": "marker not visible",
            "adjudication_artifacts": [],
        }
    )
    (run_dir / "localization.json").write_text(
        json.dumps(unavailable), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="artifact path list required"):
        checkpoint_stage(run_dir, "localization")

    (run_dir / "localization.json").write_text(
        json.dumps(localization), encoding="utf-8"
    )
    checkpoint_stage(run_dir, "localization")

    own_sweep = render_review_sheet(
        run_dir=run_dir,
        output_path=run_dir / "reviews/complete-own.jpg",
        start_frame=0,
        end_frame=8,
        purpose="own_context",
    )
    enemy_sweep = render_review_sheet(
        run_dir=run_dir,
        output_path=run_dir / "reviews/complete-enemy.jpg",
        start_frame=0,
        end_frame=8,
        purpose="arena",
    )
    completeness = json.loads((run_dir / "completeness.json").read_text())
    completeness.update(
        {
            "annotation_session_id": "terra-completeness-fresh-test",
            "model": "terra",
            "reasoning_effort": "medium",
        }
    )
    for sweep, artifact in zip(
        completeness["sweeps"], (own_sweep, enemy_sweep), strict=True
    ):
        sweep["reviewed_ranges"] = [[0, 8]]
        sweep["review_artifacts"] = [str(artifact.relative_to(run_dir))]
        sweep["completed"] = True
    (run_dir / "completeness.json").write_text(
        json.dumps(completeness), encoding="utf-8"
    )
    same_session = json.loads(json.dumps(completeness))
    same_session["annotation_session_id"] = "terra-primary-test"
    (run_dir / "completeness.json").write_text(
        json.dumps(same_session), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fresh annotation_session_id"):
        checkpoint_stage(run_dir, "completeness")

    (run_dir / "completeness.json").write_text(
        json.dumps(completeness), encoding="utf-8"
    )
    checkpoint_stage(run_dir, "completeness")

    decisions, checkpoints = assemble_staged_decisions(run_dir, manifest)
    assert decisions["events"][0]["cell"] == [3, 17]
    assert all(checkpoints[stage] for stage in (
        "verification",
        "localization",
        "completeness",
    ))
    final, audit, lock = finalize_annotation(
        run_dir=run_dir,
        decisions_path=run_dir / "unused-staged-decisions.json",
        output_path=tmp_path / "staged-ground-truth.json",
        audit_output_path=tmp_path / "staged-ground-truth.audit.json",
    )
    assert final.is_file()
    assert audit.is_file()
    assert lock.is_file()


def test_v5_rejects_stale_checkpoint_and_casual_unavailable_location(
    tmp_path: Path,
):
    video = tmp_path / "source.avi"
    _write_test_video(video)
    run_dir = tmp_path / "staged"
    prepare_annotation_run(
        video_path=video,
        output_dir=run_dir,
        start_time_s=0.0,
        end_time_s=0.2,
    )
    verification = json.loads((run_dir / "verification.json").read_text())
    verification.update(
        {
            "annotation_session_id": "terra-primary-test",
            "model": "terra",
            "reasoning_effort": "medium",
        }
    )
    (run_dir / "verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    checkpoint_stage(run_dir, "verification")
    verification["instructions"] += " changed"
    (run_dir / "verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stale"):
        assemble_staged_decisions(
            run_dir, json.loads((run_dir / "manifest.json").read_text())
        )
