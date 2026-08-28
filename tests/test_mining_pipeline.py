from __future__ import annotations

import json
from pathlib import Path

from scipy.io import wavfile

from cr_bot.audio.features import AudioFeatureConfig
from cr_bot.audio.manifest_dataset import ManifestAudioDataset
from cr_bot.mining import DEFAULT_CHANNEL_URL, phase_metadata
from cr_bot.mining.coverage import compute_coverage, split_spell_coverage
from cr_bot.mining.dataset_split import assert_no_video_leakage, split_manifest_rows
from cr_bot.mining.enemy_event_export import export_enemy_candidate_rows
from cr_bot.mining.video_manifest import (
    _manifest_fetch_cmd,
    _normalize_manifest_entry,
    filter_entries_before_date,
    normalize_source,
)


def test_default_source_uses_yerson_channel():
    assert normalize_source(None) == f"{DEFAULT_CHANNEL_URL}/videos"


def test_manifest_entry_prefers_real_video_id_over_channel_or_playlist_id():
    row = _normalize_manifest_entry(
        {
            "id": "UCPKaY_SMlcT2cTyscSgpofA",
            "url": "https://www.youtube.com/watch?v=tvA-OvUUHmw",
            "title": "sample",
        },
        source_type="channel",
    )

    assert row["video_id"] == "tvA-OvUUHmw"


def test_manifest_filters_to_before_january_12_2025():
    rows = [
        {"video_id": "old", "upload_date": "20250111"},
        {"video_id": "cutoff", "upload_date": "20250112"},
        {"video_id": "new", "upload_date": "20250301"},
        {"video_id": "unknown", "upload_date": None},
    ]

    filtered = filter_entries_before_date(rows, "01/12/2025")

    assert [row["video_id"] for row in filtered] == ["old"]


def test_manifest_preflight_command_uses_fast_fail_and_single_playlist_item():
    cmd = _manifest_fetch_cmd(
        source_type="channel",
        before_date="01/12/2025",
        cookies_from_browser="firefox",
        playlist_items="1",
        fast_fail=True,
    )

    assert "--playlist-items" in cmd
    assert "1" in cmd
    assert "--retries" in cmd
    assert "--extractor-retries" in cmd
    assert "--cookies-from-browser" in cmd


def test_phase_metadata_covers_double_and_overtime_variants():
    assert phase_metadata(time_left_s=90.0, total_remaining_s=210.0, overtime=False)["match_phase"] == "normal"
    assert phase_metadata(time_left_s=45.0, total_remaining_s=165.0, overtime=False)["match_phase"] == "double"
    assert phase_metadata(time_left_s=90.0, total_remaining_s=90.0, overtime=True)["match_phase"] == "overtime_double"
    assert phase_metadata(time_left_s=45.0, total_remaining_s=45.0, overtime=True)["match_phase"] == "overtime_triple"


def test_enemy_candidate_export_attaches_required_metadata():
    rows = export_enemy_candidate_rows(
        [
            {
                "event_id": "abc_000001_fireball",
                "card": "fireball",
                "video_time_s": 184.2,
                "time_left_s": 115.8,
                "total_remaining_s": 115.8,
                "track_id": 7,
                "cell": [4, 9],
                "clock_confirmed": True,
                "frame_confirmed": True,
                "avg_confidence": 0.91,
                "team_ratio": 0.9,
                "best_class": "fireball",
                "class_votes": {"fireball": 3},
                "overtime": True,
            }
        ],
        video_id="abc",
    )

    row = rows[0]
    assert row["is_spell"] is True
    assert row["overtime"] is True
    assert row["double_elixir"] is True
    assert row["triple_elixir"] is False
    assert row["match_phase"] == "overtime_double"
    assert row["quality_tier"] == "gold"


def test_split_manifest_rows_prevents_video_leakage():
    rows = [
        {"event_id": "a1", "video_id": "v1", "span_id": "v1s1", "card": "knight"},
        {"event_id": "a2", "video_id": "v1", "span_id": "v1s1", "card": "archers"},
        {"event_id": "b1", "video_id": "v2", "span_id": "v2s1", "card": "knight"},
        {"event_id": "c1", "video_id": "v3", "span_id": "v3s1", "card": "archers"},
    ]

    splits = split_manifest_rows(rows, val_fraction=0.25, test_fraction=0.25, seed=0)

    assert_no_video_leakage(splits)
    assert sum(len(items) for items in splits.values()) == len(rows)


def test_coverage_reports_spells_separately():
    rows = [
        {"card": "fireball", "quality_tier": "gold", "match_phase": "double", "split": "val"},
        {"card": "fireball", "quality_tier": "silver", "match_phase": "overtime_double", "split": "train"},
        {"card": "knight", "quality_tier": "bronze", "match_phase": "normal", "split": "train"},
    ]

    coverage = compute_coverage(rows)
    spells = split_spell_coverage(coverage)

    assert coverage["fireball"]["gold_count"] == 1
    assert coverage["fireball"]["silver_gold_count"] == 2
    assert spells["moving"]["fireball"]["observed_count"] == 2


def test_manifest_audio_dataset_reads_real_manifest_rows(tmp_path: Path):
    config = AudioFeatureConfig()
    wav_path = tmp_path / "sample.wav"
    wavfile.write(wav_path, config.sample_rate, [0] * config.num_samples)
    manifest_path = tmp_path / "train.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "card": "knight",
                "wav_path": str(wav_path),
                "quality_tier": "silver",
                "match_phase": "double",
                "weight": 0.7,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = ManifestAudioDataset(manifest_path, ["no_event", "knight"], config)
    features, label, weight = dataset[0]

    assert tuple(features.shape[:1]) == (1,)
    assert label == 1
    assert weight == 0.7
