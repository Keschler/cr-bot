from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from simulator.actions import PlayCardAction
from simulator.engine import ENGINE_VERSION, BattleEngine
from simulator.mining import (
    MiningManifestError,
    _pull_frame_is_contaminated,
    _labeled_frame_video_time,
    _validate_replay_source_level,
    assigned_split,
    compile_observation_manifest,
    compile_replay_cache_hog_cannon_pulls,
    compile_replay_cache_movement,
    corpus_to_dict,
    discover_replay_cache_interactions,
    discover_replay_cache_interactions_batch,
    discover_replay_cache_cannon_lifetimes,
    discover_replay_cache_fireball_flights,
    discover_replay_cache_log_motion,
    discover_replay_cache_tower_damage,
    merge_replay_interaction_reports,
    _movement_mechanic,
    _movement_speed_ratio_permille,
)
from simulator.validation import evaluate_fidelity_corpus, validation_corpus_from_dict


def _clean_clip(engine: BattleEngine, *, clip_id: str, confidence: float) -> dict[str, object]:
    state = engine.new_battle(seed=5, shuffle_decks=False)
    engine.step(state, (PlayCardAction(0, 0, (3, 17)),))
    for _ in range(19):
        engine.step(state)
    hog = next(
        entity
        for entity in state.entities.values()
        if entity.owner == 0 and entity.card_id == "hog-rider"
    )
    return {
        "clip_id": clip_id,
        "group_id": f"match:{clip_id}",
        "split": "heldout",
        "media_hash": "sha256:" + "a" * 64,
        "frame_start": 1200,
        "frame_end": 1210,
        "method": "offline_detector_ensemble_homography_v1",
        "confidence": confidence,
        "initial_state": state.to_primitive(include_events=False),
        "tracks": [
            {
                "track_id": "hog-1",
                "mechanic": "hog_movement",
                "confidence": confidence,
                "selector": {"uid": hog.uid},
                "samples": [
                    {
                        "tick": state.tick,
                        "x_mtile": hog.x_mtile,
                        "y_mtile": hog.y_mtile,
                        "hp": hog.hp,
                        "alive": True,
                        "confidence": confidence,
                    }
                ],
            }
        ],
    }


def test_high_confidence_tracks_compile_to_runnable_sealed_corpus() -> None:
    engine = BattleEngine()
    manifest = {
        "schema_version": 1,
        "corpus_id": "auto-hog-tracks-v1",
        "confidence_threshold": 0.98,
        "position_tolerance_mtile": 200,
        "hp_tolerance": 0,
        "clips": [
            _clean_clip(engine, clip_id="clean", confidence=0.995),
            _clean_clip(engine, clip_id="ambiguous", confidence=0.7),
        ],
    }

    result = compile_observation_manifest(manifest, engine=engine)

    assert [case.case_id for case in result.corpus.cases] == ["clean"]
    assert [(item.clip_id, item.reason.split(":")[0]) for item in result.discarded] == [
        ("ambiguous", "confidence_below_threshold")
    ]
    assert len(result.corpus.cases[0].measurements) == 4
    report, comparisons, _ = evaluate_fidelity_corpus(engine, result.corpus)
    assert len(comparisons) == 4
    assert all(item.agrees for item in comparisons)
    assert report.to_dict()["overall"]["samples"]["agreement_rate"] == 1.0

    encoded = corpus_to_dict(result.corpus)
    reloaded = validation_corpus_from_dict(json.loads(json.dumps(encoded)))
    assert reloaded.content_hash == result.corpus.content_hash


def test_group_split_assignment_is_stable_and_group_level() -> None:
    assert assigned_split("match-42") == assigned_split("match-42")
    assert assigned_split("match-42", salt="other") in {
        "calibration",
        "validation",
        "heldout",
    }


def test_label_frame_time_uses_declared_fps_not_native_cache_indices() -> None:
    native_frame_times = {10_120: 80.0, 10_373: 82.106}

    assert _labeled_frame_video_time(
        806,
        label_fps=10.0,
        replay_frame_times=native_frame_times,
    ) == pytest.approx(80.6)
    assert _labeled_frame_video_time(
        10_120,
        label_fps=None,
        replay_frame_times=native_frame_times,
    ) == 80.0


def test_observed_initial_entities_are_deterministically_materialized() -> None:
    engine = BattleEngine()
    manifest = {
        "schema_version": 1,
        "corpus_id": "observed-snapshot-v1",
        "confidence_threshold": 0.98,
        "clips": [
            {
                "clip_id": "isolated-hog",
                "group_id": "video-7",
                "split": "heldout",
                "media_hash": "sha256:" + "b" * 64,
                "frame_start": 500,
                "frame_end": 510,
                "method": "detector_a_b_tracking_homography_v1",
                "confidence": 0.995,
                "seed": 7,
                "initial": {
                    "tick": 20,
                    "elapsed_us": 1_000_000,
                    "phase": "regulation",
                    "entities": [
                        {
                            "track_id": "hog-visible-1",
                            "card_id": "hog-rider",
                            "owner": 0,
                            "x_mtile": 3_500,
                            "y_mtile": 20_500,
                            "hp": 1_697,
                            "confidence": 0.995,
                        }
                    ],
                    "towers": [],
                },
                "tracks": [
                    {
                        "track_id": "hog-visible-1",
                        "mechanic": "hog_movement",
                        "confidence": 0.995,
                        "samples": [
                            {
                                "tick": 20,
                                "x_mtile": 3_500,
                                "y_mtile": 20_500,
                                "confidence": 0.995,
                            }
                        ],
                    }
                ],
            }
        ],
    }

    result = compile_observation_manifest(manifest, engine=engine)
    scenario = result.corpus.cases[0].scenario
    assert scenario is not None and scenario.initial_state is not None
    report, comparisons, _ = evaluate_fidelity_corpus(engine, result.corpus)
    assert len(comparisons) == 2 and all(item.agrees for item in comparisons)
    assert report.to_dict()["overall"]["observation_group_count"] == 1


def test_invalid_vision_initial_state_is_discarded_without_losing_clean_clips() -> None:
    engine = BattleEngine()

    def clip(clip_id: str, x: int, y: int) -> dict[str, object]:
        return {
            "clip_id": clip_id,
            "group_id": f"group:{clip_id}",
            "split": "heldout",
            "media_hash": "sha256:" + "c" * 64,
            "frame_start": 1,
            "frame_end": 2,
            "method": "automatic_homography_v1",
            "confidence": 0.995,
            "seed": 1,
            "initial": {
                "tick": 10,
                "elapsed_us": 500_000,
                "phase": "regulation",
                "entities": [
                    {
                        "track_id": f"track:{clip_id}",
                        "card_id": "hog-rider",
                        "owner": 0,
                        "x_mtile": x,
                        "y_mtile": y,
                        "hp": 1_697,
                        "confidence": 0.995,
                    }
                ],
                "towers": [],
            },
            "tracks": [
                {
                    "track_id": f"track:{clip_id}",
                    "mechanic": "hog_movement",
                    "confidence": 0.995,
                    "samples": [
                        {
                            "tick": 10,
                            "x_mtile": x,
                            "y_mtile": y,
                            "confidence": 0.995,
                        }
                    ],
                }
            ],
        }

    result = compile_observation_manifest(
        {
            "schema_version": 1,
            "corpus_id": "invalid-position-filter-v1",
            "confidence_threshold": 0.98,
            "clips": [
                clip("overlaps-king", 9_000, 28_500),
                clip("clean", 3_500, 20_500),
            ],
        },
        engine=engine,
    )

    assert [case.case_id for case in result.corpus.cases] == ["clean"]
    assert len(result.discarded) == 1
    assert result.discarded[0].clip_id == "overlaps-king"
    assert result.discarded[0].reason.startswith(
        "invalid_initial_state:active troop overlaps a living structure"
    )


def test_replay_cache_is_mined_without_manual_track_manifest(tmp_path) -> None:
    import numpy as np

    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.domain.game_state import Detection, Match
    from cr_bot.features.action_space import ACTION_GRID
    from cr_bot.replay.cache import ReplayCacheWriter

    cache_path = tmp_path / "movement.pkl.gz"
    arena_px = (0, 0, 100, 100)
    with ReplayCacheWriter(cache_path) as writer:
        for frame_idx in range(3):
            norm_x, norm_y = ACTION_GRID.cell_to_norm_center(3, 20 - frame_idx)
            detection = Detection(
                track_id=77,
                class_name="hog",
                team="ally",
                confidence=0.995,
                x1=norm_x * 100,
                y1=norm_y * 100,
                x2=norm_x * 100,
                y2=norm_y * 100,
                center_x=norm_x * 100,
                center_y=norm_y * 100,
                estimated_hp=1.0,
            )
            analysis = FrameAnalysisResult(
                rendered=None,
                elixir={"estimated_value": 5, "displayed_digit": 0},
                elixir_change=None,
                towers_hp={"enemy_support_left": 3_052},
                time=None,
                time_left_s=170.0 - frame_idx * 0.1,
                total_remaining_s=290.0 - frame_idx * 0.1,
                overtime=False,
                hand_state={},
                yolo_boxes=None,
                clock_boxes=[],
                emote_boxes=[],
                matches=[Match(troop=detection, bar=None)],
                arena_px=arena_px,
                tower_hp_debug_steps={},
                timer_debug_steps={},
            )
            writer.write(
                frame_idx=frame_idx,
                video_time_s=frame_idx * 0.1,
                analysis=analysis,
                frame=np.zeros((10, 10, 3), dtype=np.uint8),
            )

    result = compile_replay_cache_movement(
        cache_path,
        corpus_id="replay-movement-v1",
        group_id="video-001",
        source_level=11,
        evidence_split="calibration",
        minimum_track_frames=3,
        maximum_speed_ratio_permille=5_000,
    )

    assert len(result.corpus.cases) == 1
    assert result.corpus.cases[0].split.value == "calibration"
    case = result.corpus.cases[0]
    assert case.evidence.media_hash is not None
    assert case.evidence.frame_start == 0 and case.evidence.frame_end == 2
    # Ordinary isolated tracks establish level-invariant displacement speed,
    # but do not pretend an off-screen target is known well enough to compare
    # absolute x/y. Bridge tracks retain coordinates because bridge occupancy
    # itself is an observable topological constraint.
    assert len(case.measurements) == 1
    speed = next(
        item
        for item in case.measurements
        if item.extractor.extractor_type == "card_move_speed_mtile_per_s"
    )
    assert speed.observed.observed_value == 10_000
    assert speed.extractor.extractor_type == "card_move_speed_mtile_per_s"
    assert speed.extractor.start_tick is None and speed.extractor.end_tick is None
    assert case.scenario is not None and case.scenario.initial_state is not None
    initial_hog = next(
        entity
        for entity in case.scenario.initial_state["entities"]
        if entity["card_id"] == "hog-rider"
    )
    assert initial_hog["deploy_remaining_us"] == 0

    with pytest.raises(MiningManifestError, match="does not match ruleset level"):
        compile_replay_cache_movement(
            cache_path,
            corpus_id="wrong-level",
            group_id="video-001",
            source_level=15,
            minimum_track_frames=3,
        )

    cross_level = compile_replay_cache_movement(
        cache_path,
        corpus_id="level-invariant-movement",
        group_id="video-001-level15",
        source_level=15,
        evidence_split="heldout",
        minimum_track_frames=3,
        maximum_speed_ratio_permille=5_000,
        level_invariant_current_ruleset=True,
        expected_support_tower_hp=3_052,
        use_expected_speed_gate=False,
    )
    assert len(cross_level.corpus.cases) == 1
    assert cross_level.corpus.cases[0].split.value == "heldout"
    assert "level_invariant_movement" in cross_level.corpus.cases[0].evidence.method

    with pytest.raises(MiningManifestError, match="requires expected_support_tower_hp"):
        compile_replay_cache_movement(
            cache_path,
            corpus_id="unconfirmed-cross-level",
            group_id="video-001-level15",
            source_level=15,
            minimum_track_frames=3,
            level_invariant_current_ruleset=True,
            use_expected_speed_gate=False,
        )

    with pytest.raises(MiningManifestError, match="cannot use the expected-speed"):
        compile_replay_cache_movement(
            cache_path,
            corpus_id="leaky-heldout",
            group_id="video-001",
            source_level=11,
            evidence_split="heldout",
            minimum_track_frames=3,
        )


def test_movement_mechanic_classifies_bridge_trajectory_without_manual_label() -> None:
    engine = BattleEngine()
    assert _movement_mechanic(
        "hog-rider",
        [
            {"x_mtile": 3_500, "y_mtile": 17_500},
            {"x_mtile": 3_500, "y_mtile": 16_000},
            {"x_mtile": 3_500, "y_mtile": 14_500},
        ],
        engine.ruleset,
    ) == "hog-rider_isolated_bridge_path"
    assert _movement_mechanic(
        "musketeer",
        [
            {"x_mtile": 9_000, "y_mtile": 20_000},
            {"x_mtile": 9_000, "y_mtile": 19_000},
        ],
        engine.ruleset,
    ) == "musketeer_isolated_movement"


def test_movement_speed_gate_distinguishes_base_motion_from_status_or_track_jump() -> None:
    assert _movement_speed_ratio_permille(700, 1_000_000, 900) == 777
    assert _movement_speed_ratio_permille(180, 1_000_000, 900) == 200
    assert _movement_speed_ratio_permille(2_700, 1_000_000, 900) == 3_000


def test_tower_damage_discovery_requires_stable_exact_attributed_plateaus(tmp_path) -> None:
    import numpy as np

    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.domain.game_state import Detection, Match
    from cr_bot.features.action_space import ACTION_GRID
    from cr_bot.replay.cache import ReplayCacheWriter

    cache_path = tmp_path / "tower-damage.pkl.gz"
    values = [3_052] * 4 + [2_735] * 4 + [2_418] * 4
    with ReplayCacheWriter(cache_path) as writer:
        for frame_idx, tower_hp in enumerate(values):
            detection = Detection(
                track_id=77,
                class_name="hog",
                team="ally",
                confidence=0.995,
                x1=10,
                y1=10,
                x2=20,
                y2=20,
                center_x=15,
                center_y=15,
                estimated_hp=1.0,
            )
            analysis = FrameAnalysisResult(
                rendered=None,
                elixir={"estimated_value": 5, "displayed_digit": 0},
                elixir_change=None,
                towers_hp={
                    "enemy_support_left": tower_hp,
                    "enemy_support_right": 3_052,
                },
                time=None,
                time_left_s=170.0,
                total_remaining_s=290.0 - frame_idx * 0.4,
                overtime=False,
                hand_state={},
                yolo_boxes=None,
                clock_boxes=[],
                emote_boxes=[],
                matches=[Match(troop=detection, bar=None)],
                arena_px=(0, 0, 100, 100),
                tower_hp_debug_steps={},
                timer_debug_steps={},
            )
            writer.write(
                frame_idx=frame_idx,
                video_time_s=frame_idx * 0.4,
                analysis=analysis,
                frame=np.zeros((1, 1, 3), dtype=np.uint8),
            )

    report = discover_replay_cache_tower_damage(
        cache_path,
        source_level=11,
        confidence_threshold=0.9,
        minimum_plateau_frames=3,
    )

    assert [row["damage"] for row in report["candidates"]] == [317, 317]
    assert {row["card_id"] for row in report["candidates"]} == {"hog-rider"}
    assert {tuple(row["attacker_track_ids"]) for row in report["candidates"]} == {
        (77,)
    }
    assert report["intervals"] == [
        {
            "card_id": "hog-rider",
            "tower": "enemy_support_left",
            "first_frame_idx": 4,
            "second_frame_idx": 8,
            "observed_interval_ms": 1_600,
            "declared_interval_ms": 1_600,
            "absolute_error_ms": 0,
        }
    ]
    assert report["mechanics"] == {
        "hog-rider_tower_damage": {
            "candidate_count": 2,
            "declared_damage": 317,
            "observed_damage_values": [317],
            "exact_agreement_count": 2,
        },
        "hog-rider_tower_repeat_interval": {
            "candidate_count": 1,
            "declared_interval_ms": 1_600,
            "observed_interval_ms": [1_600],
            "mae_ms": 0.0,
            "p95_absolute_error_ms": 0,
        },
    }


def test_tower_damage_discovery_rejects_version_mismatched_delta(tmp_path) -> None:
    import numpy as np

    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.domain.game_state import Detection, Match
    from cr_bot.replay.cache import ReplayCacheWriter

    cache_path = tmp_path / "old-hog-damage.pkl.gz"
    for_write = [3_052] * 3 + [2_734] * 3
    with ReplayCacheWriter(cache_path) as writer:
        for frame_idx, tower_hp in enumerate(for_write):
            detection = Detection(
                77, "hog", "ally", 0.995, 10, 10, 20, 20, 15, 15, 1.0
            )
            analysis = FrameAnalysisResult(
                None,
                {},
                None,
                {"enemy_support_left": tower_hp, "enemy_support_right": 3_052},
                None,
                170.0,
                290.0 - frame_idx * 0.1,
                False,
                {},
                None,
                [],
                [],
                [Match(detection, None)],
                (0, 0, 100, 100),
                {},
                {},
            )
            writer.write(
                frame_idx=frame_idx,
                video_time_s=frame_idx * 0.1,
                analysis=analysis,
                frame=np.zeros((1, 1, 3), dtype=np.uint8),
            )

    report = discover_replay_cache_tower_damage(cache_path, source_level=11)

    assert report["candidates"] == []
    assert report["rejected"][0]["damage"] == 318
    assert report["rejected"][0]["reason"] == "damage_delta_not_declared"


def test_log_motion_discovery_ignores_static_deployment_overlay(tmp_path) -> None:
    import numpy as np

    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.domain.game_state import Detection, Match
    from cr_bot.features.action_space import ACTION_GRID
    from cr_bot.replay.cache import ReplayCacheWriter

    cache_path = tmp_path / "log-motion.pkl.gz"
    truth_path = tmp_path / "actions.json"
    truth_path.write_text(
        json.dumps(
            {
                "fps": 10.0,
                "events": [
                    {"side": "own", "card": "the-log", "frame_index": 0}
                ],
            }
        ),
        encoding="utf-8",
    )
    with ReplayCacheWriter(cache_path) as writer:
        for frame_idx in range(9):
            # Two detector frames remain static. The physical rolling segment
            # then advances 400 milli-tiles every 100 ms (4,000 mtile/s).
            y_mtile = 22_400 - max(0, frame_idx - 2) * 400
            y_px = (ACTION_GRID.y0 + y_mtile / 32_000 * ACTION_GRID.height) * 100
            detection = Detection(
                track_id=81,
                class_name="the-log",
                team="ally",
                confidence=0.95,
                x1=50,
                y1=y_px,
                x2=50,
                y2=y_px,
                center_x=50,
                center_y=y_px,
                estimated_hp=1.0,
            )
            analysis = FrameAnalysisResult(
                rendered=None,
                elixir={},
                elixir_change=None,
                towers_hp={"enemy_support_left": 3_052},
                time=None,
                time_left_s=170.0,
                total_remaining_s=290.0 - frame_idx * 0.1,
                overtime=False,
                hand_state={},
                yolo_boxes=None,
                clock_boxes=[],
                emote_boxes=[],
                matches=[Match(detection, None)],
                arena_px=(0, 0, 100, 100),
                tower_hp_debug_steps={},
                timer_debug_steps={},
            )
            writer.write(
                frame_idx=frame_idx,
                video_time_s=frame_idx * 0.1,
                analysis=analysis,
                frame=np.zeros((1, 1, 3), dtype=np.uint8),
            )

    report = discover_replay_cache_log_motion(
        cache_path,
        ground_truth_path=truth_path,
        source_level=11,
        minimum_moving_steps=5,
    )

    assert report["rejected"] == []
    candidate = report["candidates"][0]
    assert candidate["detector_onset_video_time_ms"] == 0
    assert candidate["selected_segment_start_video_time_ms"] == 200
    assert candidate["moving_steps"] == 6
    assert candidate["observed_speed_mtile_per_s"] == 4_000
    assert candidate["relative_error_permille"] == 0
    assert report["mechanics"]["log_rolling_speed"]["mean_observed_speed_mtile_per_s"] == 4_000


def test_fireball_flight_discovery_requires_localized_compact_impact(tmp_path) -> None:
    import numpy as np

    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.domain.game_state import Detection, Match
    from cr_bot.features.action_space import ACTION_GRID
    from cr_bot.replay.cache import ReplayCacheWriter

    cache_path = tmp_path / "fireball-flight.pkl.gz"
    truth_path = tmp_path / "actions.json"
    truth_path.write_text(
        json.dumps(
            {
                "fps": 10.0,
                "events": [
                    {
                        "side": "enemy",
                        "card": "fireball",
                        "frame_index": 0,
                        "cell": [12, 28],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def detection(track_id: int, x_mtile: int, y_mtile: int) -> Detection:
        x_px = (ACTION_GRID.x0 + x_mtile / 18_000 * ACTION_GRID.width) * 100
        y_px = (ACTION_GRID.y0 + y_mtile / 32_000 * ACTION_GRID.height) * 100
        return Detection(
            track_id=track_id,
            class_name="fireball",
            team="enemy",
            confidence=0.95,
            x1=x_px,
            y1=y_px,
            x2=x_px,
            y2=y_px,
            center_x=x_px,
            center_y=y_px,
            estimated_hp=1.0,
        )

    target_x, target_y = 12_500, 28_500
    with ReplayCacheWriter(cache_path) as writer:
        for frame_idx in range(10):
            matches = []
            if 1 <= frame_idx <= 7:
                matches.append(
                    Match(detection(10, 9_000 + frame_idx * 500, 2_000 + frame_idx * 3_000), None)
                )
            if frame_idx >= 8:
                matches.append(Match(detection(11, target_x, target_y - (frame_idx - 8) * 100), None))
            analysis = FrameAnalysisResult(
                rendered=None,
                elixir={},
                elixir_change=None,
                towers_hp={"enemy_support_left": 3_052},
                time=None,
                time_left_s=170.0,
                total_remaining_s=290.0 - frame_idx * 0.1,
                overtime=False,
                hand_state={},
                yolo_boxes=None,
                clock_boxes=[],
                emote_boxes=[],
                matches=matches,
                arena_px=(0, 0, 100, 100),
                tower_hp_debug_steps={},
                timer_debug_steps={},
            )
            writer.write(
                frame_idx=frame_idx,
                video_time_s=frame_idx * 0.1,
                analysis=analysis,
                frame=np.zeros((1, 1, 3), dtype=np.uint8),
            )

    report = discover_replay_cache_fireball_flights(
        cache_path,
        ground_truth_path=truth_path,
        source_level=11,
        minimum_flight_samples=6,
    )

    assert report["rejected"] == []
    candidate = report["candidates"][0]
    assert (candidate["flight_track_id"], candidate["impact_track_id"]) == (10, 11)
    assert candidate["impact_target_error_mtile"] == 0
    assert candidate["observed_action_to_impact_ms"] == 800
    assert report["mechanics"]["fireball_action_to_impact"]["candidate_count"] == 1


def test_cannon_lifetime_discovery_scores_action_anchored_clean_expiry(tmp_path) -> None:
    import numpy as np

    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.domain.game_state import Detection, Match
    from cr_bot.features.action_space import ACTION_GRID
    from cr_bot.replay.cache import ReplayCacheWriter

    cache_path = tmp_path / "cannon-lifetime.pkl.gz"
    truth_path = tmp_path / "actions.json"
    truth_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "side": "own",
                        "card": "cannon",
                        "frame_index": 0,
                        "cell": [8, 20],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    arena_px = (0, 0, 100, 100)
    norm_x, norm_y = ACTION_GRID.cell_to_norm_center(8, 20)
    with ReplayCacheWriter(cache_path) as writer:
        for frame_idx in range(33):
            matches = []
            if 1 <= frame_idx <= 30:
                # Post-deploy lifetime hypothesis: full HP at t=1 s, then a
                # linear 30-second drain. The final two frames prove absence.
                hp_ratio = max(0.0, 1.0 - (frame_idx - 1) / 30.0)
                detection = Detection(
                    track_id=88,
                    class_name="cannon",
                    team="ally",
                    confidence=0.995,
                    x1=norm_x * 100,
                    y1=norm_y * 100,
                    x2=norm_x * 100,
                    y2=norm_y * 100,
                    center_x=norm_x * 100,
                    center_y=norm_y * 100,
                    estimated_hp=hp_ratio,
                )
                matches.append(Match(troop=detection, bar=None))
            analysis = FrameAnalysisResult(
                rendered=None,
                elixir={"estimated_value": 5, "displayed_digit": 0},
                elixir_change=None,
                towers_hp={"enemy_support_left": 3_052},
                time=None,
                time_left_s=170.0,
                total_remaining_s=290.0 - frame_idx,
                overtime=False,
                hand_state={},
                yolo_boxes=None,
                clock_boxes=[],
                emote_boxes=[],
                matches=matches,
                arena_px=arena_px,
                tower_hp_debug_steps={},
                timer_debug_steps={},
            )
            writer.write(
                frame_idx=frame_idx,
                video_time_s=float(frame_idx),
                analysis=analysis,
                frame=np.zeros((1, 1, 3), dtype=np.uint8),
            )

    report = discover_replay_cache_cannon_lifetimes(
        cache_path,
        ground_truth_path=truth_path,
        source_level=11,
        maximum_track_gap_s=1.1,
    )

    assert report["kind"] == "cannon_lifetime_candidate_report"
    assert report["rejected"] == []
    assert len(report["candidates"]) == 1
    candidate = report["candidates"][0]
    assert candidate["best_lifetime_start_hypothesis"] == "post_deploy"
    assert candidate["observed_duration_ms"] == 31_000
    assert candidate["curve_mae_permille"]["post_deploy"] <= 1

    inferred = discover_replay_cache_cannon_lifetimes(
        cache_path,
        source_level=11,
        maximum_track_gap_s=1.1,
    )
    assert inferred["automation"]["action_inference"] == "inferred_track_onset"
    assert inferred["ground_truth_hash"] is None
    assert inferred["candidates"][0]["action_source"] == "inferred_track_onset"


def test_movement_miner_allows_far_units_but_rejects_local_interference(tmp_path) -> None:
    import numpy as np

    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.domain.game_state import Detection, Match
    from cr_bot.features.action_space import ACTION_GRID
    from cr_bot.replay.cache import ReplayCacheWriter

    cache_path = tmp_path / "local-isolation.pkl.gz"
    arena_px = (0, 0, 100, 100)

    def detection(track_id, class_name, team, cell, confidence=0.995):
        norm_x, norm_y = ACTION_GRID.cell_to_norm_center(*cell)
        return Detection(
            track_id=track_id,
            class_name=class_name,
            team=team,
            confidence=confidence,
            x1=norm_x * 100,
            y1=norm_y * 100,
            x2=norm_x * 100,
            y2=norm_y * 100,
            center_x=norm_x * 100,
            center_y=norm_y * 100,
            estimated_hp=1.0,
        )

    with ReplayCacheWriter(cache_path) as writer:
        for frame_idx in range(4):
            matches = [
                Match(detection(77, "hog", "ally", (3, 22 - frame_idx)), None),
                # A busy opposite lane must not discard an otherwise isolated run.
                Match(detection(88, "knight", "enemy", (14, 7)), None),
            ]
            if frame_idx == 3:
                # Even an untracked, lower-confidence nearby unit terminates
                # the run. Stable IDs and candidate-grade confidence are
                # requirements for truth candidates, not for contaminants.
                matches.append(
                    Match(
                        detection(None, "skeleton", "enemy", (4, 19), confidence=0.30),
                        None,
                    )
                )
            analysis = FrameAnalysisResult(
                rendered=None,
                elixir={},
                elixir_change=None,
                towers_hp={"enemy_support_left": 3_052},
                time=None,
                time_left_s=170.0,
                total_remaining_s=290.0 - frame_idx * 0.1,
                overtime=False,
                hand_state={},
                yolo_boxes=None,
                clock_boxes=[],
                emote_boxes=[],
                matches=matches,
                arena_px=arena_px,
                tower_hp_debug_steps={},
                timer_debug_steps={},
            )
            writer.write(
                frame_idx=frame_idx,
                video_time_s=frame_idx * 0.1,
                analysis=analysis,
                frame=np.zeros((10, 10, 3), dtype=np.uint8),
            )

    result = compile_replay_cache_movement(
        cache_path,
        corpus_id="local-isolation-v2",
        group_id="video-local",
        source_level=11,
        evidence_split="calibration",
        confidence_threshold=0.98,
        minimum_track_frames=3,
        minimum_displacement_mtile=500,
        isolation_radius_mtile=3_500,
        maximum_speed_ratio_permille=5_000,
    )

    assert len(result.corpus.cases) == 1
    case = result.corpus.cases[0]
    assert case.evidence.frame_start == 0
    assert case.evidence.frame_end == 2
    assert "locally_isolated" in case.evidence.method


def test_replay_level_gate_rejects_support_hp_above_pinned_maximum() -> None:
    engine = BattleEngine()

    assert not _validate_replay_source_level(
        SimpleNamespace(towers_hp={"enemy_support_left": 2_900}),
        source_level=11,
        engine=engine,
    )

    # The vision stack's explicit unreadable-tower fallback is missing data,
    # not proof that the source match used that level.
    assert not _validate_replay_source_level(
        SimpleNamespace(towers_hp={"enemy_support_left": 4_424}),
        source_level=11,
        engine=engine,
    )

    with pytest.raises(MiningManifestError, match="conflicts with Level 11"):
        _validate_replay_source_level(
            SimpleNamespace(towers_hp={"enemy_support_left": 4_000}),
            source_level=11,
            engine=engine,
        )

    # King values are intentionally ignored: compact caches substitute the
    # live-capture default for inactive kings, so they are not level evidence.
    assert _validate_replay_source_level(
        SimpleNamespace(
            towers_hp={"enemy_support_left": 3_052, "enemy_king": 7_032}
        ),
        source_level=11,
        engine=engine,
    )

    assert _validate_replay_source_level(
        SimpleNamespace(towers_hp={"enemy_support_left": 4_424}),
        source_level=15,
        engine=engine,
        expected_support_tower_hp=4_424,
    )

    with pytest.raises(MiningManifestError, match="explicit expected support-tower HP"):
        _validate_replay_source_level(
            SimpleNamespace(towers_hp={"enemy_support_left": 4_424}),
            source_level=15,
            engine=engine,
        )


def test_pull_contamination_checks_both_hog_and_cannon_neighborhoods() -> None:
    hog = ("hog-rider", 1, 27)
    cannon = ("cannon", 0, 35)
    units = [
        {"key": hog, "x_mtile": 5_500, "y_mtile": 18_000},
        {"key": cannon, "x_mtile": 9_500, "y_mtile": 20_000},
        {
            "key": ("skeletons", 0, 49),
            "x_mtile": 7_000,
            "y_mtile": 20_300,
        },
    ]

    assert _pull_frame_is_contaminated(
        units,
        hog_key=hog,
        cannon_key=cannon,
        hog_position=(5_500, 18_000),
        cannon_position=(9_500, 20_000),
        radius_mtile=3_500,
    )
    assert not _pull_frame_is_contaminated(
        units[:2],
        hog_key=hog,
        cannon_key=cannon,
        hog_position=(5_500, 18_000),
        cannon_position=(9_500, 20_000),
        radius_mtile=3_500,
    )


@pytest.mark.parametrize("contaminated", (False, True))
def test_action_anchored_hog_cannon_pull_is_mined_with_target_trace(
    tmp_path,
    contaminated: bool,
) -> None:
    import numpy as np

    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.domain.game_state import Detection, Match
    from cr_bot.features.action_space import ACTION_GRID
    from cr_bot.replay.cache import ReplayCacheWriter

    cache_path = tmp_path / "pull.pkl.gz"
    truth_path = tmp_path / "pull.json"
    truth_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "side": "own",
                        "card": "cannon",
                        "frame_index": 0,
                        "cell": [9, 20],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    arena_px = (0, 0, 100, 100)
    with ReplayCacheWriter(cache_path) as writer:
        for frame_idx in range(1, 7):
            cannon_x, cannon_y = ACTION_GRID.cell_to_norm_center(9, 20)
            hog_x, hog_y = ACTION_GRID.cell_to_norm_center(3 + frame_idx, 16 + frame_idx // 2)

            def detection(track_id, class_name, team, x, y):
                return Detection(
                    track_id=track_id,
                    class_name=class_name,
                    team=team,
                    confidence=0.995,
                    x1=x * 100,
                    y1=y * 100,
                    x2=x * 100,
                    y2=y * 100,
                    center_x=x * 100,
                    center_y=y * 100,
                    estimated_hp=1.0,
                )

            matches = [
                Match(detection(88, "cannon", "ally", cannon_x, cannon_y), None),
                Match(detection(77, "hog", "enemy", hog_x, hog_y), None),
            ]
            if contaminated:
                matches.append(
                    Match(
                        detection(99, "skeleton", "ally", cannon_x, cannon_y),
                        None,
                    )
                )
            analysis = FrameAnalysisResult(
                rendered=None,
                elixir={"estimated_value": 5, "displayed_digit": 0},
                elixir_change=None,
                towers_hp={"enemy_support_left": 3_052},
                time=None,
                time_left_s=170.0,
                total_remaining_s=290.0,
                overtime=False,
                hand_state={},
                yolo_boxes=None,
                clock_boxes=[],
                emote_boxes=[],
                matches=matches,
                arena_px=arena_px,
                tower_hp_debug_steps={},
                timer_debug_steps={},
            )
            writer.write(
                frame_idx=frame_idx,
                video_time_s=frame_idx * 0.1,
                analysis=analysis,
                frame=np.zeros((10, 10, 3), dtype=np.uint8),
            )

    result = compile_replay_cache_hog_cannon_pulls(
        cache_path,
        ground_truth_path=truth_path,
        corpus_id="pulls-v1",
        group_id="match-1",
        source_level=11,
        minimum_track_frames=5,
    )

    assert len(result.corpus.cases) == 1
    case = result.corpus.cases[0]
    assert case.traces[0].observed.mechanic == "hog_cannon_pull_targeting"
    assert ":localized_action_cell:" in case.evidence.method
    assert ("targeting_only_contaminated_path" in case.evidence.method) is contaminated
    if contaminated:
        assert {item.observed.mechanic for item in case.measurements} == {
            "hog_cannon_pull_initial_state_x_mtile",
            "hog_cannon_pull_initial_state_y_mtile",
        }
    assert corpus_to_dict(result.corpus)["cases"][0]["traces"][0]["events"][0][
        "kind"
    ] == "target_changed"


def test_action_free_interaction_miner_discovers_bridge_paths(tmp_path) -> None:
    import numpy as np

    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.domain.game_state import Detection, Match
    from cr_bot.features.action_space import ACTION_GRID
    from cr_bot.replay.cache import ReplayCacheWriter

    cache_path = tmp_path / "bridge-path.pkl.gz"
    arena_px = (0, 0, 100, 100)

    def detection(track_id: int, x_mtile: int, y_mtile: int) -> Detection:
        norm_x = ACTION_GRID.x0 + x_mtile / 18_000 * ACTION_GRID.width
        norm_y = ACTION_GRID.y0 + y_mtile / 32_000 * ACTION_GRID.height
        return Detection(
            track_id=track_id,
            class_name="hog",
            team="ally",
            confidence=0.995,
            x1=norm_x * 100,
            y1=norm_y * 100,
            x2=norm_x * 100,
            y2=norm_y * 100,
            center_x=norm_x * 100,
            center_y=norm_y * 100,
            estimated_hp=1.0,
        )

    with ReplayCacheWriter(cache_path) as writer:
        for frame_idx in range(1, 11):
            y_mtile = 19_000 - (frame_idx - 1) * 600
            analysis = FrameAnalysisResult(
                rendered=None,
                elixir={},
                elixir_change=None,
                towers_hp={"enemy_support_left": 3_052},
                time=None,
                time_left_s=170.0,
                total_remaining_s=300.0 - frame_idx * 0.1,
                overtime=False,
                hand_state={},
                yolo_boxes=None,
                clock_boxes=[],
                emote_boxes=[],
                matches=[Match(detection(77, 3_500, y_mtile), None)],
                arena_px=arena_px,
                tower_hp_debug_steps={},
                timer_debug_steps={},
            )
            writer.write(
                frame_idx=frame_idx,
                video_time_s=frame_idx * 0.1,
                analysis=analysis,
                frame=np.zeros((1, 1, 3), dtype=np.uint8),
            )

    report = discover_replay_cache_interactions(
        cache_path,
        source_level=11,
        minimum_track_frames=5,
        minimum_bridge_displacement_mtile=1_000,
    )

    bridge = [
        row
        for row in report["candidates"]
        if row["mechanic"] == "hog-rider_bridge_path_topology"
    ]
    assert len(bridge) == 1
    assert bridge[0]["bridge_index"] == 0
    assert bridge[0]["direction"] == "toward_enemy"
    assert bridge[0]["truth_promoted"] is False
    assert report["mechanics"]["hog-rider_bridge_path_topology"]["candidate_count"] == 1


def test_action_free_interaction_miner_discovers_hog_cannon_approach(tmp_path) -> None:
    import numpy as np

    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.domain.game_state import Detection, Match
    from cr_bot.features.action_space import ACTION_GRID
    from cr_bot.replay.cache import ReplayCacheWriter

    cache_path = tmp_path / "hog-cannon-approach.pkl.gz"
    arena_px = (0, 0, 100, 100)

    def detection(track_id: int, class_name: str, team: str, x_mtile: int, y_mtile: int) -> Detection:
        norm_x = ACTION_GRID.x0 + x_mtile / 18_000 * ACTION_GRID.width
        norm_y = ACTION_GRID.y0 + y_mtile / 32_000 * ACTION_GRID.height
        return Detection(
            track_id=track_id,
            class_name=class_name,
            team=team,
            confidence=0.995,
            x1=norm_x * 100,
            y1=norm_y * 100,
            x2=norm_x * 100,
            y2=norm_y * 100,
            center_x=norm_x * 100,
            center_y=norm_y * 100,
            estimated_hp=1.0,
        )

    with ReplayCacheWriter(cache_path) as writer:
        for frame_idx in range(1, 7):
            matches = [
                Match(detection(88, "cannon", "ally", 8_500, 16_000), None),
                Match(
                    detection(
                        77,
                        "hog",
                        "enemy",
                        8_500,
                        20_000 - (frame_idx - 1) * 1_000,
                    ),
                    None,
                ),
            ]
            analysis = FrameAnalysisResult(
                rendered=None,
                elixir={},
                elixir_change=None,
                towers_hp={"enemy_support_left": 3_052},
                time=None,
                time_left_s=170.0,
                total_remaining_s=300.0 - frame_idx * 0.1,
                overtime=False,
                hand_state={},
                yolo_boxes=None,
                clock_boxes=[],
                emote_boxes=[],
                matches=matches,
                arena_px=arena_px,
                tower_hp_debug_steps={},
                timer_debug_steps={},
            )
            writer.write(
                frame_idx=frame_idx,
                video_time_s=frame_idx * 0.1,
                analysis=analysis,
                frame=np.zeros((1, 1, 3), dtype=np.uint8),
            )

    report = discover_replay_cache_interactions(
        cache_path,
        source_level=11,
        minimum_track_frames=5,
    )
    pulls = [
        row
        for row in report["candidates"]
        if row["mechanic"] == "hog_cannon_targeting_candidate"
    ]
    assert len(pulls) == 1
    assert pulls[0]["target_hypothesis"] == "cannon"
    assert pulls[0]["classification"] == "approach_to_only_local_building"
    assert pulls[0]["truth_promoted"] is False


def test_action_free_interaction_batch_retains_bad_sources(tmp_path) -> None:
    report = discover_replay_cache_interactions_batch(
        [tmp_path / "missing-cache.json"],
        source_level=11,
    )
    assert report["source_count"] == 0
    assert report["failed_source_count"] == 1
    assert report["candidate_count"] == 0
    assert report["source_failures"][0]["reason"].startswith("replay cache is not a file")


def test_action_free_batch_inherits_level_proof_only_for_same_video_windows(tmp_path) -> None:
    import numpy as np

    from cr_bot.domain.frame_analysis import FrameAnalysisResult
    from cr_bot.replay.cache import ReplayCacheWriter

    proof = tmp_path / "video-1" / "standard" / "replay-cache.json"
    window = tmp_path / "video-1:action-window:000" / "standard" / "replay-cache.json"

    def write_cache(path, towers_hp):
        path.parent.mkdir(parents=True, exist_ok=True)
        with ReplayCacheWriter(path) as writer:
            writer.write(
                frame_idx=1,
                video_time_s=0.1,
                analysis=FrameAnalysisResult(
                    rendered=None,
                    elixir={},
                    elixir_change=None,
                    towers_hp=towers_hp,
                    time=None,
                    time_left_s=299.9,
                    total_remaining_s=299.9,
                    overtime=False,
                    hand_state={},
                    yolo_boxes=None,
                    clock_boxes=[],
                    emote_boxes=[],
                    matches=[],
                    arena_px=(0, 0, 100, 100),
                    tower_hp_debug_steps={},
                    timer_debug_steps={},
                ),
                frame=np.zeros((1, 1, 3), dtype=np.uint8),
            )

    write_cache(proof, {"enemy_support_left": 3_052})
    write_cache(window, {})
    report = discover_replay_cache_interactions_batch(
        [window],
        source_level=11,
        level_proof_paths=[proof],
    )
    assert report["source_count"] == 1
    assert report["failed_source_count"] == 0
    assert report["level_proof_failures"] == []
    assert report["level_proof_sources"][0]["source_video_key"] == "video-1"
    assert report["sources"][0]["cache_path"].endswith(
        "video-1:action-window:000/standard/replay-cache.json"
    )

    # A valid proof from another recording must never be reused for this
    # window: source level is a property of the recording, not of the HUD
    # profile or a directory-wide batch.
    other_proof = tmp_path / "video-2" / "standard" / "replay-cache.json"
    write_cache(other_proof, {"enemy_support_left": 3_052})
    mismatched = discover_replay_cache_interactions_batch(
        [window],
        source_level=11,
        level_proof_paths=[other_proof],
    )
    assert mismatched["source_count"] == 0
    assert mismatched["failed_source_count"] == 1
    assert "no exact full support-tower HP" in mismatched["source_failures"][0]["reason"]


def test_dual_hud_interaction_merge_pairs_only_agreeing_candidates(tmp_path) -> None:
    engine = BattleEngine()

    def report(path: str, cache_hash: str, *, x: int) -> object:
        payload = {
            "schema_version": 1,
            "kind": "autonomous_interaction_candidate_batch",
            "ruleset_id": engine.ruleset.ruleset_id,
            "ruleset_hash": engine.ruleset.content_hash,
            "engine_version": ENGINE_VERSION,
            "sources": [{"cache_path": path, "cache_hash": cache_hash}],
            "cache_hashes": [cache_hash],
            "candidates": [
                {
                    "cache_path": path,
                    "cache_hash": cache_hash,
                    "mechanic": "hog-rider_bridge_path_topology",
                    "card_id": "hog-rider",
                    "owner": 0,
                    "track_id": 11,
                    "video_time_ms": 1_000,
                    "x_mtile": x,
                    "y_mtile": 17_000,
                    "minimum_confidence": 0.99,
                    "truth_promoted": False,
                }
            ],
        }
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    standard = report(
        "video-42/standard/replay-cache.json",
        "sha256:" + "a" * 64,
        x=3_500,
    )
    alternative = report(
        "video-42/alternative/replay-cache.json",
        "sha256:" + "b" * 64,
        x=3_550,
    )

    merged = merge_replay_interaction_reports([standard, alternative])

    assert merged["candidate_count"] == 1
    assert merged["rejected_count"] == 0
    assert merged["hud_groups"] == [
        {
            "source_group": "video-42",
            "hud_variants": ["alternative", "standard"],
            "paired_count": 1,
            "standard_candidate_count": 1,
            "alternative_candidate_count": 1,
            "both_hud_present": True,
        }
    ]
    candidate = merged["candidates"][0]
    assert candidate["truth_promoted"] is False
    assert candidate["position_delta_mtile"] == 50
    assert candidate["hud_variants"] == ["standard", "alternative"]


def test_dual_hud_interaction_merge_retains_unpaired_rows_and_can_fail_closed(tmp_path) -> None:
    engine = BattleEngine()
    payload = {
        "schema_version": 1,
        "kind": "autonomous_interaction_candidate_report",
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "cache_path": str(tmp_path / "only-standard" / "standard" / "replay-cache.json"),
        "cache_hash": "sha256:" + "c" * 64,
        "candidates": [
            {
                "mechanic": "track_onset_action_candidate",
                "card_id": "hog-rider",
                "owner": 0,
                "video_time_ms": 100,
                "minimum_confidence": 0.9,
                "truth_promoted": False,
            }
        ],
    }
    path = tmp_path / "standard.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    merged = merge_replay_interaction_reports([path])
    assert merged["candidate_count"] == 0
    assert merged["rejected_count"] == 1
    assert merged["rejected"][0]["reason"] == "missing_agreeing_alternative_hud"

    required = merge_replay_interaction_reports([path], require_both_hud=True)
    assert required["failed_source_count"] == 1
    assert required["source_failures"][0]["reason"] == "no source contains both HUD variants"


def test_dual_hud_interaction_merge_preserves_upstream_source_failures(tmp_path) -> None:
    engine = BattleEngine()
    payload = {
        "schema_version": 1,
        "kind": "autonomous_interaction_candidate_batch",
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "source_failures": [{"cache_path": "bad-cache", "reason": "wrong level"}],
        "sources": [],
        "candidates": [],
    }
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    merged = merge_replay_interaction_reports([path])

    assert merged["failed_source_count"] == 1
    assert merged["source_failures"] == [
        {
            "report_path": str(path.resolve()),
            "cache_path": "bad-cache",
            "reason": "wrong level",
        }
    ]
