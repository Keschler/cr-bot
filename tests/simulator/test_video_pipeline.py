from __future__ import annotations

from simulator.video_pipeline import (
    PRE_EVOLUTION_CUTOFF,
    SOURCE_CHANNEL_URL,
    VideoPipelineError,
    assign_video_split,
    build_action_window_extractor_jobs,
    build_action_window_manifest,
    batch_replay_cache_track_manifest,
    build_extractor_jobs,
    discover_source_manifest,
    filter_source_manifest,
    _inspect_replay_cache_completeness,
    mine_clean_tracks,
    merge_hud_track_manifests,
    merge_track_manifests,
    retention_records,
    run_extractor_jobs,
    select_hud_variant,
    validate_source_entry,
    video_truth_to_observation_manifest,
)
from simulator.cli import main as simulator_main
import gzip
import json
import pickle
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np


VIDEO_ID = "abcdefghijk"


def _source(**extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "channel_url": SOURCE_CHANNEL_URL,
        "video_id": VIDEO_ID,
        "upload_date": "2023-06-18",
        "raw_path": "raw/abcdefghijk.mp4",
        "media_sha256": "sha256:" + "a" * 64,
    }
    row.update(extra)
    return row


def test_source_boundary_is_exact_and_fail_closed() -> None:
    assert validate_source_entry(_source())["upload_date"] == "2023-06-18"
    assert PRE_EVOLUTION_CUTOFF.isoformat() == "2023-06-19"
    for extra in (
        {"upload_date": "2023-06-19"},
        {"channel_url": "https://www.youtube.com/@other/videos"},
        {"upload_date": None},
    ):
        try:
            validate_source_entry(_source(**extra))
        except VideoPipelineError:
            pass
        else:
            raise AssertionError("out-of-scope source was accepted")


def test_source_filter_retains_rejection_reasons_and_deduplicates() -> None:
    result = filter_source_manifest(
        [_source(), _source(video_id="zyxwvutsrqp"), _source(upload_date="2024-01-01")]
    )
    assert [row["video_id"] for row in result["accepted"]] == [VIDEO_ID, "zyxwvutsrqp"]
    assert len(result["rejected"]) == 1
    assert "pre-evolution" in result["rejected"][0]["reason"]


def test_hud_selection_requires_one_profile() -> None:
    assert select_hud_variant(hand_y=2_020, elixir_y=2_310) == "standard"
    assert select_hud_variant(hand_y=1_960, elixir_y=2_250) == "alternative"
    assert select_hud_variant(hand_y=1_990, elixir_y=2_280) is None


def test_clean_track_mining_is_group_split_and_discards_ambiguous_tracks() -> None:
    source = _source(
        tracks=[
            {
                "track_id": "hog-1",
                "card_id": "hog-rider",
                "confidence": 0.995,
                "hud_variant": "standard",
                "samples": [
                    {
                        "frame_idx": index,
                        "x_mtile": 3_000 + index * 20,
                        "y_mtile": 20_000 - index * 20,
                        "confidence": 0.99,
                    }
                    for index in range(20)
                ],
            },
            {
                "track_id": "ambiguous",
                "card_id": "hog-rider",
                "confidence": 0.50,
                "hud_variant": "standard",
                "samples": [],
            },
        ]
    )
    result = mine_clean_tracks(
        filter_source_manifest([source]),
        confidence_threshold=0.98,
        minimum_track_frames=20,
    )
    assert result["summary"]["accepted_case_count"] == 1
    assert result["summary"]["discarded_track_count"] == 1
    case = result["cases"][0]
    assert case["split"] == assign_video_split(VIDEO_ID)
    assert result["split_salt"] == "simulator-v1-video-split"
    assert case["evidence"]["source_id"] == f"yersoncz:{VIDEO_ID}"


def test_clean_track_mining_supports_reproducible_custom_heldout_salt() -> None:
    source = _source(
        tracks=[
            {
                "track_id": "hog-1",
                "card_id": "hog-rider",
                "confidence": 0.995,
                "hud_variant": "standard",
                "samples": [
                    {
                        "frame_idx": index,
                        "x_mtile": 3_000 + index * 20,
                        "y_mtile": 20_000 - index * 20,
                        "confidence": 0.99,
                    }
                    for index in range(20)
                ],
            }
        ]
    )
    salt = "custom-heldout-salt"
    result = mine_clean_tracks(
        filter_source_manifest([source]),
        confidence_threshold=0.98,
        minimum_track_frames=20,
        split_salt=salt,
    )
    assert result["split_salt"] == salt
    assert result["cases"][0]["split"] == assign_video_split(VIDEO_ID, salt=salt)


def test_clean_track_mining_rejects_static_detector_linger() -> None:
    source = _source(
        tracks=[
            {
                "track_id": "linger",
                "card_id": "hog-rider",
                "confidence": 0.995,
                "hud_variant": "standard",
                "samples": [
                    {
                        "frame_idx": index * 3,
                        "x_mtile": 8_000 + (index % 2),
                        "y_mtile": 16_000,
                        "confidence": 0.99,
                    }
                    for index in range(20)
                ],
            }
        ]
    )
    result = mine_clean_tracks(filter_source_manifest([source]))
    assert result["summary"]["accepted_case_count"] == 0
    assert result["discarded"][0]["reason"].startswith("track displacement")


def test_clean_track_mining_rejects_detector_teleport_without_card_speed_leakage() -> None:
    source = _source(
        tracks=[
            {
                "track_id": "teleport",
                "card_id": "hog-rider",
                "confidence": 0.995,
                "hud_variant": "standard",
                "samples": [
                    {
                        "frame_idx": index * 3,
                        "x_mtile": 8_000 if index < 9 else 15_000,
                        "y_mtile": 16_000,
                        "confidence": 0.99,
                    }
                    for index in range(20)
                ],
            }
        ]
    )
    result = mine_clean_tracks(filter_source_manifest([source]))
    assert result["summary"]["accepted_case_count"] == 0
    assert "implausible motion step" in result["discarded"][0]["reason"]


def test_clean_track_mining_rejects_irregular_path_without_using_card_stats() -> None:
    source = _source(
        tracks=[
            {
                "track_id": "zig-zag",
                "card_id": "hog-rider",
                "confidence": 0.995,
                "hud_variant": "standard",
                "samples": [
                    {
                        "frame_idx": index * 3,
                        "x_mtile": 8_000 + (index % 2) * 2_000,
                        "y_mtile": 16_000 - index * 400,
                        "confidence": 0.99,
                    }
                    for index in range(20)
                ],
            }
        ]
    )
    result = mine_clean_tracks(
        filter_source_manifest([source]),
        maximum_path_to_displacement_ratio=1.2,
        maximum_step_speed_mtile_per_s=100_000,
    )
    assert result["summary"]["accepted_case_count"] == 0
    assert "path is too irregular" in result["discarded"][0]["reason"]


def test_clean_track_mining_rejects_unstable_detector_speed() -> None:
    source = _source(
        tracks=[
            {
                "track_id": "unstable",
                "card_id": "hog-rider",
                "confidence": 0.995,
                "hud_variant": "standard",
                "samples": [
                    {
                        "frame_idx": index * 3,
                        "x_mtile": 8_000
                        + sum(100 if step % 2 == 0 else 2_000 for step in range(index)),
                        "y_mtile": 16_000,
                        "confidence": 0.99,
                    }
                    for index in range(20)
                ],
            }
        ]
    )
    result = mine_clean_tracks(
        filter_source_manifest([source]),
        maximum_speed_iqr_ratio=0.1,
        maximum_step_speed_mtile_per_s=100_000,
    )
    assert result["summary"]["accepted_case_count"] == 0
    assert "step speed is too unstable" in result["discarded"][0]["reason"]


def test_retention_records_are_truth_gated() -> None:
    records = retention_records(
        filter_source_manifest([_source()]),
        truth_manifest_path="outputs/truth.json",
    )
    assert records[0]["truth_extracted"] is True
    assert records[0]["eviction_eligible"] is True
    assert records[0]["path"].endswith("raw/abcdefghijk.mp4")


def test_retention_records_can_be_limited_to_sources_with_truth() -> None:
    source = filter_source_manifest([_source(), _source(video_id="zyxwvutsrqp")])
    records = retention_records(
        source,
        truth_manifest_path="outputs/truth.json",
        truth_manifest={"cases": [{"video_id": VIDEO_ID}]},
    )
    assert [record["video_id"] for record in records] == [VIDEO_ID]


def test_video_truth_cli_writes_truth_and_retention_manifests(tmp_path) -> None:
    source = _source(
        tracks=[
            {
                "track_id": "hog-1",
                "card_id": "hog-rider",
                "confidence": 0.995,
                "hud_variant": "alternative",
                "samples": [
                    {
                        "frame_idx": index,
                        "x_mtile": 3_000 + index * 20,
                        "y_mtile": 20_000 - index * 20,
                    }
                    for index in range(20)
                ],
            }
        ]
    )
    source_path = tmp_path / "source.json"
    truth_path = tmp_path / "truth.json"
    retention_path = tmp_path / "retention.json"
    source_path.write_text(json.dumps([source]), encoding="utf-8")
    assert simulator_main(
        [
            "mine-video-truth",
            str(source_path),
            "--json-out",
            str(truth_path),
            "--retention-out",
            str(retention_path),
        ]
    ) == 0
    assert json.loads(truth_path.read_text(encoding="utf-8"))["summary"]["accepted_case_count"] == 1
    assert json.loads(retention_path.read_text(encoding="utf-8"))["artifacts"][0]["eviction_eligible"]


def test_video_truth_cli_fails_closed_on_empty_truth(tmp_path) -> None:
    source_path = tmp_path / "source.json"
    truth_path = tmp_path / "truth.json"
    retention_path = tmp_path / "retention.json"
    source_path.write_text(json.dumps([_source()]), encoding="utf-8")
    assert simulator_main(
        [
            "mine-video-truth",
            str(source_path),
            "--json-out",
            str(truth_path),
            "--retention-out",
            str(retention_path),
        ]
    ) == 2
    assert json.loads(truth_path.read_text(encoding="utf-8"))["summary"]["truth_ready"] is False
    assert json.loads(retention_path.read_text(encoding="utf-8"))["artifacts"] == []


def test_source_discovery_reuses_cutoff_aware_repository_resolver(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_resolve(source, **kwargs):
        calls["source"] = source
        calls.update(kwargs)
        return [_source()]

    import cr_bot.mining.video_manifest as video_manifest

    monkeypatch.setattr(video_manifest, "resolve_video_manifest", fake_resolve)
    result = discover_source_manifest(max_videos=7, cookies_from_browser="firefox")
    assert result["accepted"][0]["video_id"] == VIDEO_ID
    assert calls["source"] == SOURCE_CHANNEL_URL
    assert calls["before_date"] == "2023-06-19"
    assert calls["max_videos"] == 7
    assert calls["cookies_from_browser"] == "firefox"


def test_extractor_plan_covers_both_hud_variants_and_dry_run(tmp_path) -> None:
    video_path = tmp_path / "abcdefghijk.mp4"
    video_path.write_bytes(b"fixture")
    source = filter_source_manifest(
        [_source(analysis_video_path=str(video_path))]
    )
    plan = build_extractor_jobs(source, output_root=tmp_path / "extractor")
    assert plan["ready_job_count"] == 2
    assert [job["hud_variant"] for job in plan["jobs"]] == ["alternative", "standard"]
    assert all(job["status"] == "ready" for job in plan["jobs"])
    assert any("--alternative-rois" in job["command"] for job in plan["jobs"])
    run = run_extractor_jobs(plan)
    assert run["execute"] is False
    assert run["completed_count"] == 0
    assert run["failed_count"] == 0
    assert run["skipped_count"] == 2


def test_extractor_run_resumes_existing_replay_cache_without_overwriting(tmp_path) -> None:
    video_path = tmp_path / "abcdefghijk.mp4"
    video_path.write_bytes(b"fixture")
    source = filter_source_manifest(
        [_source(analysis_video_path=str(video_path))]
    )
    plan = build_extractor_jobs(source, output_root=tmp_path / "extractor")
    replay_path = Path(plan["jobs"][0]["replay_cache_path"])
    replay_path.parent.mkdir(parents=True)
    replay_path.write_bytes(b"sealed-cache")

    run = run_extractor_jobs(plan, execute=True)

    assert run["completed_count"] == 1
    assert run["skipped_count"] == 1
    existing = next(row for row in run["jobs"] if row["status"] == "already_complete")
    assert existing["returncode"] == 0
    assert existing["replay_cache_bytes"] == len(b"sealed-cache")
    assert replay_path.read_bytes() == b"sealed-cache"


def test_extractor_timeout_is_recorded_and_does_not_block_batch(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "abcdefghijk.mp4"
    video_path.write_bytes(b"fixture")
    source = filter_source_manifest(
        [_source(analysis_video_path=str(video_path))]
    )
    plan = build_extractor_jobs(source, output_root=tmp_path / "extractor")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs.get("timeout"),
            output="partial stdout",
            stderr="stalled extractor",
        )

    import simulator.video_pipeline as video_pipeline

    monkeypatch.setattr(video_pipeline.subprocess, "run", fake_run)
    run = run_extractor_jobs(
        plan,
        execute=True,
        workspace_root=tmp_path,
        retention_manifest_path=tmp_path / "retention.json",
        raw_media_root=tmp_path / "raw",
        reserve_bytes=0,
        job_timeout_s=0.01,
    )

    assert run["failed_count"] == 2
    assert all(row["status"] == "timeout" for row in run["jobs"])
    assert all(row["returncode"] is None for row in run["jobs"])


def test_replay_cache_completeness_rejects_a_valid_but_truncated_prefix(tmp_path) -> None:
    cache_path = tmp_path / "replay-cache.json"
    with gzip.open(cache_path, "wb") as handle:
        pickle.dump({"schema_version": 1}, handle)
        for timestamp in (12.0, 12.1, 12.2):
            pickle.dump(SimpleNamespace(video_time_s=timestamp), handle)

    result = _inspect_replay_cache_completeness(
        cache_path,
        expected_start_s=12.0,
        expected_duration_s=1.0,
        sample_interval_s=0.1,
    )

    assert result is not None
    assert result["recognized"] is True
    assert result["complete"] is False
    assert result["frame_count"] == 3


def test_extractor_plan_uses_per_source_gameplay_window(tmp_path) -> None:
    video_path = tmp_path / "abcdefghijk.mp4"
    video_path.write_bytes(b"fixture")
    source = filter_source_manifest(
        [_source(
            analysis_video_path=str(video_path),
            analysis_start_time_s=12.5,
            analysis_duration_s=4.0,
        )]
    )

    plan = build_extractor_jobs(source, output_root=tmp_path / "extractor")

    for job in plan["jobs"]:
        assert job["source"]["analysis_start_time_s"] == 12.5
        assert job["source"]["analysis_duration_s"] == 4.0
        assert "--video-start-time" in job["command"]
        assert job["command"][job["command"].index("--video-start-time") + 1] == "12.5"
        assert "--video-duration" not in job["command"]
        assert job["command"][job["command"].index("--video-end-time") + 1] == "16.5"


def test_action_window_selector_is_deterministic_and_covers_cards_without_truth_promotion(tmp_path) -> None:
    video_path = tmp_path / "abcdefghijk.mp4"
    video_path.write_bytes(b"fixture")
    source = filter_source_manifest(
        [_source(analysis_video_path=str(video_path), duration_s=60.0)]
    )
    candidate_path = tmp_path / f"{VIDEO_ID}.jsonl"
    rows = [
        {
            "video_id": VIDEO_ID,
            "card": "cannon",
            "video_time_s": 10.0,
            "avg_confidence": 0.97,
            "clock_confirmed": True,
            "frame_index": 300,
            "cell": [8, 12],
            "event_id": "cannon-1",
        },
        {
            "video_id": VIDEO_ID,
            "card": "fireball",
            "video_time_s": 20.0,
            "avg_confidence": 0.96,
            "clock_confirmed": True,
            "frame_index": 600,
            "cell": [8, 15],
            "event_id": "fireball-1",
        },
        {
            "video_id": VIDEO_ID,
            "card": "hog-rider-evolution",
            "video_time_s": 30.0,
            "avg_confidence": 0.99,
            "clock_confirmed": True,
            "frame_index": 900,
            "event_id": "excluded-form",
        },
        {
            "video_id": VIDEO_ID,
            "card": "log",
            "video_time_s": 40.0,
            "avg_confidence": 0.50,
            "clock_confirmed": True,
            "frame_index": 1200,
            "event_id": "low-confidence",
        },
    ]
    candidate_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    first = build_action_window_manifest(
        source,
        tmp_path,
        confidence_threshold=0.85,
        max_windows_per_video=4,
        window_before_s=1.0,
        window_after_s=2.0,
    )
    second = build_action_window_manifest(
        source,
        tmp_path,
        confidence_threshold=0.85,
        max_windows_per_video=4,
        window_before_s=1.0,
        window_after_s=2.0,
    )
    assert first == second
    assert first["summary"]["window_count"] == 2
    assert {row["anchor_card_id"] for row in first["accepted"]} == {"cannon", "fireball"}
    assert all("hog-rider-evolution" not in json.dumps(row) for row in first["accepted"])
    assert first["accepted"][0]["candidate_file_sha256"].startswith("sha256:")

    plan = build_action_window_extractor_jobs(
        first,
        output_root=tmp_path / "action-extractor",
    )
    assert plan["ready_job_count"] == 4
    assert len(plan["jobs"]) == 4
    assert all("--video-start-time" in job["command"] for job in plan["jobs"])
    assert all("--video-end-time" in job["command"] for job in plan["jobs"])


def test_action_window_selector_rejects_unanchored_candidates(tmp_path) -> None:
    video_path = tmp_path / "abcdefghijk.mp4"
    video_path.write_bytes(b"fixture")
    source = filter_source_manifest(
        [_source(analysis_video_path=str(video_path), duration_s=60.0)]
    )
    (tmp_path / f"{VIDEO_ID}.jsonl").write_text(
        json.dumps(
            {
                "video_id": VIDEO_ID,
                "card": "cannon",
                "video_time_s": 10.0,
                "avg_confidence": 0.99,
                "event_id": "unanchored",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = build_action_window_manifest(source, tmp_path)
    assert manifest["accepted"] == []
    assert manifest["summary"]["rejected_candidate_count"] >= 1


def test_action_window_cli_emits_resumable_dual_hud_plan(tmp_path) -> None:
    video_path = tmp_path / "abcdefghijk.mp4"
    video_path.write_bytes(b"fixture")
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps([_source(analysis_video_path=str(video_path), duration_s=60.0)]),
        encoding="utf-8",
    )
    candidate_path = tmp_path / f"{VIDEO_ID}.jsonl"
    candidate_path.write_text(
        json.dumps(
            {
                "video_id": VIDEO_ID,
                "card": "cannon",
                "video_time_s": 10.0,
                "avg_confidence": 0.99,
                "clock_confirmed": True,
                "frame_index": 300,
                "event_id": "cannon-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "action-plan.json"
    assert simulator_main(
        [
            "plan-action-windows",
            str(source_path),
            str(tmp_path),
            "--output-root",
            str(tmp_path / "extractor"),
            "--json-out",
            str(output_path),
        ]
    ) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["windows"]["summary"]["window_count"] == 1
    assert payload["plan"]["ready_job_count"] == 2
    assert payload["run"]["execute"] is False


def test_hud_merge_selects_one_profile_without_double_counting() -> None:
    standard = filter_source_manifest(
        [
            _source(
                hud_variant="standard",
                tracks=[
                    {"track_id": "a", "card_id": "hog-rider", "confidence": 0.91},
                    {"track_id": "b", "card_id": "musketeer", "confidence": 0.92},
                ],
            )
        ]
    )
    alternative = filter_source_manifest(
        [
            _source(
                hud_variant="alternative",
                tracks=[
                    {"track_id": "a", "card_id": "hog-rider", "confidence": 0.99},
                ],
            )
        ]
    )

    merged = merge_hud_track_manifests([standard, alternative])

    assert merged["summary"]["track_count"] == 2
    assert merged["accepted"][0]["hud_variant"] == "standard"
    assert merged["accepted"][0]["hud_selection"] == (
        "track_count_then_confidence_then_standard"
    )
    assert len(merged["accepted"][0]["hud_candidates"]) == 2


def test_hud_merge_records_agreement_without_promoting_duplicate_evidence() -> None:
    def track(*, offset: int = 0, track_id: str = "hog-7") -> dict[str, object]:
        return {
            "track_id": track_id,
            "parent_track_id": "7",
            "card_id": "hog-rider",
            "owner": "ally",
            "confidence": 0.995,
            "samples": [
                {
                    "frame_idx": frame,
                    "x_mtile": 4_000 + frame * 30 + offset,
                    "y_mtile": 20_000 - frame * 20,
                }
                for frame in range(10)
            ],
        }

    standard = filter_source_manifest(
        [_source(hud_variant="standard", tracks=[track()])]
    )
    alternative = filter_source_manifest(
        [_source(hud_variant="alternative", tracks=[track(offset=25)])]
    )
    merged = merge_hud_track_manifests([standard, alternative])
    row = merged["accepted"][0]

    assert row["hud_agreement"]["status"] == "agree"
    assert row["hud_agreement"]["independent_evidence"] is False
    assert row["hud_agreement"]["matched_track_count"] == 1
    assert row["tracks"][0]["hud_agreement"]["status"] == "agree"
    assert row["tracks"][0]["hud_agreement"]["overlap_sample_count"] == 10
    # The merged source still contains one interpretation of the frames.
    assert merged["summary"]["track_count"] == 1


def test_hud_merge_marks_trajectory_disagreement() -> None:
    def track(offset: int) -> dict[str, object]:
        return {
            "track_id": "hog-8",
            "card_id": "hog-rider",
            "owner": "ally",
            "confidence": 0.995,
            "samples": [
                {
                    "frame_idx": frame,
                    "x_mtile": 4_000 + frame * 30 + offset,
                    "y_mtile": 20_000 - frame * 20,
                }
                for frame in range(10)
            ],
        }

    merged = merge_hud_track_manifests(
        [
            filter_source_manifest([_source(hud_variant="standard", tracks=[track(0)])]),
            filter_source_manifest(
                [_source(hud_variant="alternative", tracks=[track(1_000)])]
            ),
        ],
        agreement_tolerance_mtile=250,
    )
    row = merged["accepted"][0]
    assert row["hud_agreement"]["status"] == "disagree"
    assert row["tracks"][0]["hud_agreement"]["status"] == "disagree"


def test_repeated_extractor_roots_select_best_candidate_deterministically() -> None:
    def source(track_count: int, confidence: float, root: str) -> dict[str, object]:
        tracks = [
            {
                "track_id": f"hog-{index}",
                "card_id": "hog-rider",
                "owner": "ally",
                "confidence": confidence,
                "samples": [
                    {
                        "frame_idx": frame,
                        "x_mtile": 4_000 + frame * 30,
                        "y_mtile": 20_000 - frame * 20,
                    }
                    for frame in range(10)
                ],
            }
            for index in range(track_count)
        ]
        return {
            **_source(hud_variant="standard", tracks=tracks),
            "replay_cache_path": f"{root}/replay-cache.json",
            "replay_cache_sha256": "sha256:" + "b" * 64,
        }

    first = filter_source_manifest([source(1, 0.99, "first")])
    second = filter_source_manifest([source(2, 0.90, "second")])
    merged = merge_track_manifests([first, second], hud_variant="standard")
    assert merged["summary"]["candidate_source_count"] == 2
    assert merged["summary"]["track_count"] == 2
    row = merged["accepted"][0]
    assert row["replay_cache_path"] == "second/replay-cache.json"
    assert len(row["extractor_root_candidates"]) == 2


def test_extract_video_cli_emits_dry_run_job_plan(tmp_path) -> None:
    source_path = tmp_path / "source.json"
    output_path = tmp_path / "extractor.json"
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fixture")
    source_path.write_text(
        json.dumps([_source(analysis_video_path=str(video_path))]),
        encoding="utf-8",
    )
    assert simulator_main(
        [
            "extract-video",
            str(source_path),
            "--output-root",
            str(tmp_path / "jobs"),
            "--json-out",
            str(output_path),
        ]
    ) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["plan"]["ready_job_count"] == 2
    assert payload["run"]["execute"] is False


def test_replay_cache_adapter_emits_mapped_tracks(tmp_path) -> None:
    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.domain.game_state import Detection, Match
    from cr_bot.replay import ReplayCacheWriter
    from simulator.video_pipeline import replay_cache_track_manifest

    cache_path = tmp_path / "cache.json"
    with ReplayCacheWriter(cache_path) as writer:
        for frame_idx in range(5):
            x = 400 + frame_idx * 20
            detection = Detection(
                track_id=3,
                class_name="hog-rider",
                team="ally",
                confidence=0.995,
                x1=x - 20,
                y1=700,
                x2=x + 20,
                y2=740,
                center_x=x,
                center_y=720,
                estimated_hp=1.0,
            )
            stationary_building = Detection(
                track_id=4,
                class_name="cannon",
                team="enemy",
                confidence=0.995,
                x1=500,
                y1=400,
                x2=540,
                y2=440,
                center_x=520,
                center_y=420,
                estimated_hp=1.0,
            )
            analysis = FrameAnalysisResult(
                rendered=None,
                elixir={},
                elixir_change=None,
                towers_hp={},
                time="2:59",
                time_left_s=179,
                total_remaining_s=299,
                overtime=False,
                hand_state={},
                yolo_boxes=None,
                clock_boxes=[],
                emote_boxes=[],
                matches=[
                    Match(troop=detection, bar=None),
                    Match(troop=stationary_building, bar=None),
                ],
                arena_px=(0, 0, 1000, 1000),
                tower_hp_debug_steps={},
                timer_debug_steps={},
            )
            writer.write(
                frame_idx=frame_idx,
                video_time_s=frame_idx * 0.1,
                analysis=analysis,
                frame=np.zeros((20, 20, 3), dtype=np.uint8),
            )
    manifest = replay_cache_track_manifest(
        _source(),
        cache_path,
        hud_variant="standard",
        confidence_threshold=0.98,
        minimum_track_frames=5,
        isolation_radius_mtile=100,
    )
    tracks = manifest["accepted"][0]["tracks"]
    assert len(tracks) == 1
    assert tracks[0]["card_id"] == "hog-rider"
    assert all(track["card_id"] != "cannon" for track in tracks)
    assert tracks[0]["samples"][0]["x_mtile"] < tracks[0]["samples"][-1]["x_mtile"]
    assert manifest["accepted"][0]["replay_cache_sha256"].startswith("sha256:")


def test_video_truth_compiles_to_fidelity_manifest() -> None:
    source = _source(
        tracks=[
            {
                "track_id": "hog-1",
                "card_id": "hog-rider",
                "owner": "ally",
                "confidence": 0.995,
                "hud_variant": "standard",
                "hud_agreement": {
                    "status": "agree",
                    "independent_evidence": False,
                    "matched_track_id": "hog-1-alt",
                    "overlap_sample_count": 20,
                    "position_mae_mtile": 0.0,
                    "position_max_error_mtile": 0.0,
                    "tolerance_mtile": 250,
                    "variants": ["standard", "alternative"],
                },
                "samples": [
                    {
                        "frame_idx": index,
                        "x_mtile": 3_000 + index * 20,
                        "y_mtile": 20_000 - index * 20,
                        "video_time_s": index / 10,
                    }
                    for index in range(20)
                ],
            }
        ]
    )
    truth = mine_clean_tracks(
        filter_source_manifest([source]),
        confidence_threshold=0.98,
        minimum_track_frames=20,
    )
    assert truth["cases"][0]["evidence"]["hud_agreement"]["status"] == "agree"
    mining_manifest = video_truth_to_observation_manifest(truth)
    assert len(mining_manifest["clips"]) == 1
    assert mining_manifest["clips"][0]["initial"]["entities"][0]["owner"] == 0
    assert mining_manifest["clips"][0]["tracks"][0]["samples"] == []
    assert mining_manifest["clips"][0]["tracks"][0]["displacement_speed"][
        "compare_to_card_base_speed"
    ] is True


def test_video_truth_speed_estimator_can_measure_curved_track_path() -> None:
    source = _source(
        tracks=[
            {
                "track_id": "curved-hog",
                "card_id": "hog-rider",
                "confidence": 0.995,
                "hud_variant": "standard",
                "samples": [
                    {
                        "frame_idx": frame,
                        "x_mtile": x,
                        "y_mtile": y,
                        "video_time_s": frame / 10,
                        "confidence": 0.99,
                    }
                    for frame, (x, y) in enumerate(
                        ((0, 0), (1_000, 0), (1_000, 1_000), (2_000, 1_000))
                    )
                ],
            }
        ]
    )
    truth = mine_clean_tracks(
        filter_source_manifest([source]),
        confidence_threshold=0.98,
        minimum_track_frames=4,
        minimum_displacement_mtile=500,
        minimum_elapsed_s=0.2,
        maximum_step_speed_mtile_per_s=20_000,
    )
    endpoint = video_truth_to_observation_manifest(truth, speed_estimator="endpoint")
    path = video_truth_to_observation_manifest(truth, speed_estimator="path_length")
    endpoint_speed = endpoint["clips"][0]["tracks"][0]["displacement_speed"]
    path_speed = path["clips"][0]["tracks"][0]["displacement_speed"]
    assert endpoint_speed["speed_estimator"] == "endpoint"
    assert path_speed["speed_estimator"] == "path_length"
    assert endpoint["clips"][0]["method"].endswith(":endpoint")
    assert path["clips"][0]["method"].endswith(":path_length")
    assert path_speed["observed_mtile_per_s"] > endpoint_speed["observed_mtile_per_s"]


def test_video_truth_speed_estimator_rejects_unknown_mode() -> None:
    truth = {
        "cases": [
            {
                "case_id": "case",
                "video_id": VIDEO_ID,
                "card_id": "hog-rider",
                "track_id": "hog",
                "evidence": {"media_sha256": "sha256:" + "b" * 64},
                "samples": [
                    {"frame_idx": 0, "x_mtile": 0, "y_mtile": 0, "confidence": 1.0},
                    {"frame_idx": 1, "x_mtile": 1_000, "y_mtile": 0, "confidence": 1.0},
                ],
            }
        ]
    }
    try:
        video_truth_to_observation_manifest(truth, speed_estimator="unknown")
    except VideoPipelineError as error:
        assert "speed_estimator" in str(error)
    else:
        raise AssertionError("unknown speed estimator should fail closed")


def test_batch_replay_miner_records_missing_jobs_without_fabricating_truth(tmp_path) -> None:
    result = batch_replay_cache_track_manifest(
        filter_source_manifest([_source()]),
        tmp_path / "extractor",
        hud_variant="standard",
        minimum_track_frames=5,
    )
    assert result["summary"]["accepted_source_count"] == 0
    assert result["summary"]["rejected_source_count"] == 1
    assert "missing replay cache" in result["rejected"][0]["reason"]


def test_batch_replay_miner_discovers_each_action_window_without_collapsing_video(
    tmp_path, monkeypatch
) -> None:
    """Window caches are separate evidence units even when video_id is shared."""

    root = tmp_path / "extractor"
    paths = [
        root / VIDEO_ID / "standard" / "replay-cache.json",
        root / f"{VIDEO_ID}:action-window:000" / "standard" / "replay-cache.json",
        root / f"{VIDEO_ID}:action-window:001" / "standard" / "replay-cache.json",
    ]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"sealed-fixture")

    def fake_track_manifest(source, cache_path, **kwargs):
        return filter_source_manifest(
            [
                {
                    **source,
                    "hud_variant": kwargs["hud_variant"],
                    "replay_cache_path": str(cache_path),
                    "tracks": [],
                }
            ]
        )

    monkeypatch.setattr(
        "simulator.video_pipeline.replay_cache_track_manifest", fake_track_manifest
    )
    result = batch_replay_cache_track_manifest(
        filter_source_manifest([_source()]),
        root,
        hud_variant="standard",
    )

    assert result["summary"]["discovered_cache_job_count"] == 3
    assert result["summary"]["accepted_source_count"] == 3
    assert [row.get("source_group_id") for row in result["accepted"]] == [
        VIDEO_ID,
        f"{VIDEO_ID}:action-window:000",
        f"{VIDEO_ID}:action-window:001",
    ]
