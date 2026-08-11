from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from cr_bot.annotation_pipeline import (
    MODEL_COST_MULTIPLIERS,
    MODEL_PROFILES,
    accumulated_weighted_tokens,
    completed_job_matches,
    job_fingerprint,
    load_state,
    normalize_enemy_unit_decision_roles,
    normalize_enemy_spell_decision_artifacts,
    normalize_enemy_spell_confirmation_artifacts,
    sha256_file,
    validate_enemy_existence_decisions,
    validate_enemy_identity_decisions,
    validate_enemy_side_check_decisions,
    validate_enemy_spell_confirmation_decisions,
    validate_enemy_unit_scan_decisions,
    validate_enemy_unit_decisions,
    validate_own_adjudication_decisions,
    validate_own_semantic_decisions,
    validate_own_release_review_decisions,
)
from cr_bot.own_localization import validate_own_localization_decisions
from cr_bot.own_localization_cascade import (
    best_v2_cluster,
    route_after_primary,
    route_after_tiebreak,
    route_v2_after_terra_verify,
    route_v2_initial,
    select_medoid,
    select_v2_consensus,
)
from cr_bot.annotation_stages import _validate_candidate_frame
from scripts.codex_annotation.run_model_worker import (
    model_cost_multiplier,
    package_image_paths,
    recover_json_object,
)
from scripts.codex_annotation.run_label_independent_own_localization_v2 import (
    ROLE_PROMPTS as LOCALIZATION_V2_ROLE_PROMPTS,
    SPECIALIZED_PROMPTS as LOCALIZATION_V2_SPECIALIZED_PROMPTS,
    _specialized_prompt as localization_v2_specialized_prompt,
)
from scripts.codex_annotation.merge_semantic_workers import (
    _enemy_event,
    _ensure_candidate_support,
    _ensure_event_context,
)
from scripts.codex_annotation.prepare_own_adjudication_packages import (
    _cluster_rows,
)
from scripts.codex_annotation.materialize_own_proposal_union import (
    select_proposal_event,
)
from scripts.codex_annotation.render_own_discovery import _candidate_groups
from scripts.codex_annotation.render_enemy_identity_targets import (
    _select_dominant_track,
    _target_center,
)
from scripts.codex_annotation.prepare_enemy_onset_deck_package import (
    build_onset_deck_package,
)
from scripts.codex_annotation.prepare_own_localization_packages import (
    _frame_indices,
    _grid_frame_groups,
    _own_events,
    _rule_options,
)
from scripts.codex_annotation.evaluate_own_localization import evaluate_locations


def test_own_localization_validator_requires_exact_blind_package_coverage():
    target = {
        "event_id": "event-own-000028-ice-spirit",
        "review_frame_indices": [23, 25, 27, 28, 29, 31, 33, 36],
        "location_rule_options": ["spawn_center", "deployment_center"],
        "macro_review_artifacts": ["localization/reviews/ice-spirit-macro.jpg"],
        "grid_review_artifacts": ["localization/reviews/ice-spirit-grid.jpg"],
    }
    package = {"targets": [target]}
    decision = {
        "event_id": target["event_id"],
        "location_frame_index": 28,
        "location_rule": "spawn_center",
        "cell": [2, 17],
        "macro_review_artifacts": target["macro_review_artifacts"],
        "grid_review_artifacts": target["grid_review_artifacts"],
        "confidence": "direct",
        "reason": "new deployment center is visible inside the labeled cell",
    }
    document = {"stage": "own_localization_chunk", "decisions": [decision]}

    assert validate_own_localization_decisions(document, package) == [decision]

    missing = {"stage": "own_localization_chunk", "decisions": []}
    with pytest.raises(ValueError, match="cover every localization target"):
        validate_own_localization_decisions(missing, package)

    off_sheet = json.loads(json.dumps(document))
    off_sheet["decisions"][0]["location_frame_index"] = 30
    with pytest.raises(ValueError, match="not present in the evidence"):
        validate_own_localization_decisions(off_sheet, package)

    bad_cell = json.loads(json.dumps(document))
    bad_cell["decisions"][0]["cell"] = [18, 17]
    with pytest.raises(ValueError, match="invalid.*cell"):
        validate_own_localization_decisions(bad_cell, package)


def test_own_localization_evidence_window_and_card_rules_are_generic():
    macro, grid = _frame_indices(50, 0, 100)
    assert macro == list(range(41, 69))
    assert grid == [
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
        53, 54, 55, 56, 57, 58, 60, 62, 65, 68,
    ]
    assert _grid_frame_groups(50, grid) == (
        [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
        [54, 55, 56, 57, 58, 60, 62, 65, 68],
    )
    assert _rule_options("ice-spirit") == ["spawn_center", "deployment_center"]
    assert _rule_options("cannon") == ["deployment_center", "spawn_center"]
    assert _rule_options("fireball") == ["target_center", "impact_center"]
    assert _rule_options("log") == ["initial_rolling_object_center"]


def test_own_localization_filters_combined_sealed_semantics_by_side():
    own = {"side": "own", "card": "ice-spirit", "event_frame_index": 10}
    enemy = {"side": "enemy", "card": "cannon", "event_frame_index": 12}

    assert _own_events({"events": [enemy, own]}) == [own]


def test_own_location_score_uses_independent_coordinate_tolerance():
    target = {
        "event_id": "event-own-000028-ice-spirit",
        "card": "ice-spirit",
        "event_frame_index": 28,
        "review_frame_indices": [28],
        "location_rule_options": ["spawn_center"],
        "macro_review_artifacts": ["macro.jpg"],
        "grid_review_artifacts": ["grid.jpg"],
    }
    package = {"targets": [target]}
    decision = {
        "event_id": target["event_id"],
        "location_frame_index": 28,
        "location_rule": "spawn_center",
        "cell": [3, 18],
        "macro_review_artifacts": target["macro_review_artifacts"],
        "grid_review_artifacts": target["grid_review_artifacts"],
        "confidence": "direct",
        "reason": "visible deployment",
    }
    prediction = {"stage": "own_localization_chunk", "decisions": [decision]}
    truth = [
        {"side": "own", "card": "ice-spirit", "frame_index": 29, "cell": [2, 17]}
    ]
    report = evaluate_locations(
        truth, package, prediction, frame_tolerance=5, cell_tolerance=1
    )
    assert report["correct"] == 1
    decision["cell"] = [4, 18]
    report = evaluate_locations(
        truth, package, prediction, frame_tolerance=5, cell_tolerance=1
    )
    assert report["correct"] == 0


def test_label_independent_localization_routes_only_on_blind_agreement():
    def decision(cell, confidence="direct"):
        return {"cell": cell, "confidence": confidence}

    agreeing = [("marker", decision([8, 20])), ("badge", decision([9, 21]))]
    assert route_after_primary(agreeing, "cannon") == "accept"

    disagreeing = [("marker", decision([6, 18])), ("badge", decision([9, 23]))]
    assert route_after_primary(disagreeing, "skeletons") == "luna_tiebreak"
    disagreeing.append(("tiebreak", decision([8, 22])))
    assert route_after_tiebreak(disagreeing, "skeletons") == "accept"
    assert select_medoid(disagreeing, "skeletons")[0] == "tiebreak"


def test_label_independent_localization_rejects_illegal_own_troop_consensus():
    attempts = [
        ("marker", {"cell": [8, 10], "confidence": "direct"}),
        ("badge", {"cell": [8, 11], "confidence": "direct"}),
    ]
    assert route_after_primary(attempts, "musketeer") == "luna_tiebreak"
    assert select_medoid(attempts, "musketeer") is None


def test_v2_localization_requires_cross_model_three_vote_acceptance():
    def row(cell, confidence="direct"):
        return {"cell": cell, "confidence": confidence}

    attempts = [
        ("luna_marker", row([8, 20])),
        ("luna_temporal", row([8, 21])),
        ("luna_specialized", row([9, 20])),
        ("terra_residual", row([9, 21])),
    ]
    assert route_v2_initial(attempts, "cannon") == "accept"

    correlated_luna_only = [
        ("luna_marker", row([8, 20])),
        ("luna_temporal", row([8, 20])),
        ("luna_specialized", row([8, 20])),
        ("terra_residual", row([12, 24])),
    ]
    assert route_v2_initial(correlated_luna_only, "cannon") == "terra_verify"
    correlated_luna_only.append(("terra_verify", row([12, 23])))
    assert (
        route_v2_after_terra_verify(correlated_luna_only, "cannon")
        == "sol_specialized"
    )

    one_persistently_invalid_role = [
        ("luna_marker", row([8, 20])),
        ("luna_temporal", row([8, 21])),
        ("terra_residual", row([9, 20])),
    ]
    assert route_v2_initial(one_persistently_invalid_role, "cannon") == "accept"


def test_v2_selection_prefers_cross_family_cluster_and_excludes_dissenters():
    def row(cell):
        return {"cell": cell, "confidence": "direct"}

    attempts = [
        ("luna_marker", row([4, 20])),
        ("luna_temporal", row([4, 20])),
        ("luna_specialized", row([4, 20])),
        ("terra_residual", row([10, 24])),
        ("terra_verify", row([10, 25])),
        ("sol_specialized", row([11, 24])),
    ]
    cluster = best_v2_cluster(attempts, "hog-rider", minimum_size=2)
    assert cluster == [3, 4, 5]
    selected = select_v2_consensus(attempts, "hog-rider")
    assert selected is not None
    assert selected[0][0] == "sol_specialized"
    assert selected[0][1]["cell"] == [11, 24]
    assert selected[1] == [3, 4, 5]


def test_v2_localization_prompt_routing_is_global_and_templates_format():
    assert localization_v2_specialized_prompt("log").name == (
        "own_rolling_spell_localization_v2.txt"
    )
    assert localization_v2_specialized_prompt("fireball").name == (
        "own_targeted_spell_localization_v2.txt"
    )
    assert localization_v2_specialized_prompt("skeletons").name == (
        "own_unit_building_localization_v2.txt"
    )
    for path in {
        *LOCALIZATION_V2_ROLE_PROMPTS.values(),
        *LOCALIZATION_V2_SPECIALIZED_PROMPTS.values(),
    }:
        rendered = path.read_text(encoding="utf-8").format(
            RUN_DIR="/blind/run",
            PACKAGE_FILE="package.json",
            OUTPUT_FILE="worker-output.json",
            SESSION_ID="blind-session",
            MODEL="gpt-5.6-luna",
            REASONING_EFFORT="low",
            EXAMPLE_FRAME="480",
        )
        assert "package.json" in rendered
        assert "worker-output.json" in rendered


def test_default_profile_is_terra_only_and_current_model_weights_are_normalized():
    assert MODEL_COST_MULTIPLIERS == {
        "gpt-5.6-sol": 2.5,
        "gpt-5.6-terra": 1.0,
        "gpt-5.6-luna": 0.1,
    }
    efficient = MODEL_PROFILES["terra-efficient"]
    assert {spec.model for spec in efficient.values()} == {"gpt-5.6-terra"}
    assert {spec.cost_multiplier for spec in efficient.values()} == {1}
    assert efficient["own_completeness"].reasoning_effort == "low"
    assert efficient["own_adjudication"].reasoning_effort == "low"
    assert efficient["own_release_review"].reasoning_effort == "low"

    experimental = MODEL_PROFILES["sol-experimental"]
    assert {spec.model for spec in experimental.values()} == {"gpt-5.6-sol"}
    assert {spec.cost_multiplier for spec in experimental.values()} == {2.5}
    hybrid = MODEL_PROFILES["hybrid-accuracy"]
    assert hybrid["enemy_existence"].model == "gpt-5.6-sol"
    assert hybrid["enemy_existence"].cost_multiplier == 2.5
    assert hybrid["enemy_side_check"].model == "gpt-5.6-terra"
    assert hybrid["enemy_side_check"].reasoning_effort == "low"
    assert hybrid["enemy_side_escalation"].model == "gpt-5.6-luna"
    assert hybrid["own_slot_primary"].model == "gpt-5.6-luna"
    assert hybrid["own_slot_primary"].reasoning_effort == "low"
    assert hybrid["own_slot_primary"].cost_multiplier == 0.1
    assert hybrid["enemy_spell_recovery"].model == "gpt-5.6-terra"
    assert hybrid["enemy_spell_recovery"].reasoning_effort == "low"
    assert {
        spec.model
        for name, spec in hybrid.items()
        if name
        not in {
            "enemy_existence",
            "enemy_side_escalation",
            "own_slot_primary",
        }
    } == {"gpt-5.6-terra"}
    assert model_cost_multiplier("gpt-5.6-terra") == 1
    assert model_cost_multiplier("gpt-5.6-sol") == 2.5
    assert model_cost_multiplier("gpt-5.6-luna") == 0.1
    luna = MODEL_PROFILES["luna-experimental"]
    assert {spec.model for spec in luna.values()} == {"gpt-5.6-luna"}
    assert {spec.cost_multiplier for spec in luna.values()} == {0.1}
    try:
        model_cost_multiplier("unpriced-model")
    except ValueError as error:
        assert "unknown model pricing" in str(error)
    else:
        raise AssertionError("unknown models must fail closed")


def test_independent_own_release_review_is_exact_and_fails_closed():
    package = {
        "reviews": [
            {
                "event_id": "event-own-000010-hog-rider",
                "confirmation_frame_index": 15,
                "confirmation_artifacts": ["reviews/release-opaque.jpg"],
            }
        ]
    }
    valid = {
        "decisions": [
            {
                "event_id": "event-own-000010-hog-rider",
                "decision": "released",
                "confirmation_frame_index": 15,
                "confirmation_artifacts": ["reviews/release-opaque.jpg"],
                "checks": {
                    "release_confirmed": True,
                    "elixir_spend_persisted": True,
                    "hand_cycle_completed": None,
                    "post_release_effect": True,
                },
                "reason": None,
            }
        ]
    }
    validate_own_release_review_decisions(valid, package)

    canceled = json.loads(json.dumps(valid))
    canceled["decisions"][0].update(
        {
            "decision": "canceled",
            "reason": "card remains in hand and elixir returns",
        }
    )
    canceled["decisions"][0]["checks"]["release_confirmed"] = False
    canceled["decisions"][0]["checks"]["elixir_spend_persisted"] = False
    canceled["decisions"][0]["checks"]["post_release_effect"] = False
    validate_own_release_review_decisions(canceled, package)

    guessed = json.loads(json.dumps(valid))
    guessed["decisions"][0]["checks"]["post_release_effect"] = False
    try:
        validate_own_release_review_decisions(guessed, package)
    except ValueError as error:
        assert "released requires" in str(error)
    else:
        raise AssertionError("unsupported own release was accepted")


def test_release_review_workers_filter_canceled_event_before_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "work_packages").mkdir(parents=True)
    (run_dir / "worker_outputs").mkdir()
    checks = {
        "release_confirmed": True,
        "elixir_spend_persisted": True,
        "hand_cycle_completed": True,
        "post_release_effect": True,
    }

    def event(event_id: str, candidate_id: str, frame: int) -> dict:
        return {
            "event_id": event_id,
            "candidate_id": candidate_id,
            "side": "own",
            "card": "hog-rider",
            "event_frame_index": frame,
            "confirmation_frame_index": frame + 5,
            "confirmation_artifacts": [f"reviews/release-{frame}.jpg"],
            "own_confirmation": dict(checks),
        }

    released = event("event-own-000010-hog-rider", "own:000005", 10)
    canceled = event("event-own-000030-hog-rider", "own:000025", 30)
    (run_dir / "verification.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "annotation_session_id": "verification-session",
                "events": [released, canceled],
                "rejected_candidates": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "own_semantics.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "worker_provenance": [
                    {"annotation_session_id": "adjudication-session"}
                ],
            }
        ),
        encoding="utf-8",
    )
    package = {
        "run_id": "run",
        "target_range": [0, 100],
        "reviews": [
            {
                "event_id": row["event_id"],
                "candidate_id": row["candidate_id"],
                "confirmation_frame_index": row["confirmation_frame_index"],
                "confirmation_artifacts": row["confirmation_artifacts"],
            }
            for row in (released, canceled)
        ],
    }
    package_path = run_dir / "work_packages/own-release-000000-000100.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    decisions = {
        "run_id": "run",
        "stage": "own_release_review_chunk",
        "target_range": [0, 100],
        "annotation_session_id": "release-worker-session",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "low",
        "decisions": [
            {
                "event_id": released["event_id"],
                "decision": "released",
                "confirmation_frame_index": 15,
                "confirmation_artifacts": ["reviews/release-10.jpg"],
                "checks": checks,
                "reason": None,
            },
            {
                "event_id": canceled["event_id"],
                "decision": "canceled",
                "confirmation_frame_index": 35,
                "confirmation_artifacts": ["reviews/release-30.jpg"],
                "checks": {
                    "release_confirmed": False,
                    "elixir_spend_persisted": False,
                    "hand_cycle_completed": False,
                    "post_release_effect": False,
                },
                "reason": "card returned to hand without a persistent spend",
            },
        ],
    }
    (run_dir / "worker_outputs" / package_path.name).write_text(
        json.dumps(decisions), encoding="utf-8"
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/codex_annotation/build_release_review_from_own_adjudication.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--session-id",
            "release-aggregate-session",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    verification = json.loads((run_dir / "verification.json").read_text())
    assert [row["event_id"] for row in verification["events"]] == [
        released["event_id"]
    ]
    assert verification["rejected_candidates"] == [
        {
            "candidate_id": "own:000025",
            "reason": (
                "independent_release_canceled: card returned to hand without "
                "a persistent spend"
            ),
        }
    ]
    review = json.loads((run_dir / "release_review.json").read_text())
    assert review["annotation_session_id"] == "release-aggregate-session"
    assert review["worker_provenance"][0]["annotation_session_id"] == (
        "release-worker-session"
    )
    assert [row["event_id"] for row in review["reviews"]] == [
        released["event_id"]
    ]


def test_worker_resolves_only_package_images_inside_run_dir(tmp_path: Path):
    run_dir = tmp_path / "run"
    reviews = run_dir / "reviews"
    reviews.mkdir(parents=True)
    first = reviews / "first.jpg"
    second = reviews / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    package = run_dir / "package.json"
    package.write_text(
        json.dumps(
            {
                "artifact": "reviews/first.jpg",
                "nested": [
                    {"artifact": "reviews/second.png"},
                    "reviews/first.jpg",
                    "not-an-image.json",
                ],
            }
        ),
        encoding="utf-8",
    )
    assert package_image_paths(package, run_dir) == [
        first.resolve(),
        second.resolve(),
    ]

    package.write_text(
        json.dumps(
            {
                "artifact": "reviews/first.jpg",
                "attached_images": ["reviews/second.png"],
            }
        ),
        encoding="utf-8",
    )
    assert package_image_paths(package, run_dir) == [second.resolve()]

    package.write_text(
        json.dumps({"artifact": "../outside.jpg"}),
        encoding="utf-8",
    )
    try:
        package_image_paths(package, run_dir)
    except ValueError as error:
        assert "escapes run directory" in str(error)
    else:
        raise AssertionError("package image outside run directory accepted")


def test_enemy_identity_crop_tracks_marker_nearest_verified_onset():
    burst = {
        "track_ids": [10, 11, 12],
        "track_start_frames": [80, 85, 88],
        "first_bboxes": [
            [10, 20, 10, 10],
            [100, 200, 20, 20],
            [300, 400, 30, 30],
        ],
    }
    tracks = {
        10: {90: (20, 30, 10, 10)},
        11: {90: (120, 220, 20, 20)},
        12: {90: (320, 420, 30, 30)},
    }
    center_x, center_y, boxes = _target_center(
        burst=burst,
        tracks=tracks,
        frame_index=90,
        onset_frame=85,
    )
    assert boxes == [(120, 220, 20, 20)]
    assert (center_x, center_y) == (130, 230)


def test_enemy_identity_v2_selects_one_eligible_moving_track():
    burst = {
        "burst_id": "enemy-marker-burst:000001",
        "start_frame": 100,
        "track_ids": [10, 11, 12],
        "track_start_frames": [100, 100, 106],
        "first_bboxes": [
            [100, 100, 10, 10],
            [120, 100, 10, 10],
            [140, 100, 10, 10],
        ],
    }
    tracks = {
        # A same-onset stationary component.
        10: {
            100: (100, 100, 10, 10),
            101: (100, 100, 10, 10),
            102: (100, 100, 10, 10),
        },
        # A same-onset moving/persistent component should be the sole focus.
        11: {
            100: (120, 100, 10, 10),
            101: (124, 106, 10, 10),
            102: (130, 112, 10, 10),
            103: (136, 118, 10, 10),
            104: (142, 124, 10, 10),
        },
        # A very persistent old track starts outside the burst/onset window.
        12: {
            frame: (140 + frame, 100, 10, 10)
            for frame in range(106, 130)
        },
    }
    selected, anchor = _select_dominant_track(
        burst=burst,
        tracks=tracks,
        onset_frame=100,
        onset_tolerance=2,
    )
    assert selected.track_id == 11
    assert selected.eligible is True
    assert selected.observation_count == 5
    assert anchor == (120, 100, 10, 10)


def test_enemy_identity_v2_falls_back_with_ineligible_metadata():
    burst = {
        "burst_id": "enemy-marker-burst:000002",
        "start_frame": 100,
        "track_ids": [21],
        "track_start_frames": [110],
        "first_bboxes": [[20, 30, 10, 10]],
    }
    selected, anchor = _select_dominant_track(
        burst=burst,
        tracks={21: {110: (20, 30, 10, 10)}},
        onset_frame=100,
        onset_tolerance=2,
    )
    assert selected.track_id == 21
    assert selected.eligible is False
    assert anchor == (20, 30, 10, 10)


def test_worker_can_recover_json_returned_as_final_message():
    assert recover_json_object('{"stage":"own_semantics_chunk"}') == {
        "stage": "own_semantics_chunk"
    }
    assert recover_json_object(
        "Completed.\n```json\n{\"events\": []}\n```"
    ) == {"events": []}
    assert recover_json_object("I could not complete the task.") is None


def test_own_discovery_groups_overlaps_but_caps_timeline_span():
    candidates = [
        {"candidate_id": "own:000005", "approximate_frame_index": 5},
        {"candidate_id": "own:000018", "approximate_frame_index": 18},
        {"candidate_id": "own:000030", "approximate_frame_index": 30},
        {"candidate_id": "own:000079", "approximate_frame_index": 79},
    ]
    groups = _candidate_groups(
        candidates,
        segment_start=0,
        segment_end=100,
        before_frames=16,
        after_frames=20,
        max_group_span_frames=60,
    )
    assert [[row["candidate_id"] for row in group[2]] for group in groups] == [
        ["own:000005", "own:000018", "own:000030"],
        ["own:000079"],
    ]


def test_worker_prompts_do_not_forbid_the_tool_needed_to_write_output():
    prompt_dir = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "prompts"
    )
    for name in (
        "own_semantics_chunk.txt",
        "own_completeness_chunk.txt",
        "own_adjudication_chunk.txt",
        "enemy_spells_chunk.txt",
        "enemy_spell_confirmation_chunk.txt",
    ):
        prompt = (prompt_dir / name).read_text(encoding="utf-8").lower()
        normalized = " ".join(prompt.split())
        assert "do not call tools" not in prompt
        assert "use filesystem tools only" in normalized
        assert "required output json" in normalized


def _spell_confirmation_checks(
    *,
    resolved_after: bool | None = True,
    boundary_truncated: bool | None = False,
) -> dict[str, bool | None]:
    return {
        "absent_before": True,
        "coherent_sequence": True,
        "independent_spell_object_or_resolution": True,
        "not_unit_attack_or_ability": True,
        "not_targeting_overlay_or_floating_label": True,
        "enemy_direction_or_origin": True,
        "resolved_after": resolved_after,
        "boundary_truncated": boundary_truncated,
    }


def test_enemy_spell_confirmation_is_exact_side_aware_and_boundary_safe():
    review = {
        "review_id": "spell-review:enemy-000020",
        "proposal_frame_index": 20,
        "sampled_frame_indices": list(range(8, 51)),
        "confirmation_artifacts": [
            "reviews/spell-before.jpg",
            "reviews/spell-after.jpg",
        ],
        "segment_end_sentinel": False,
    }
    package = {
        "segment": {"start_frame": 0, "end_frame_exclusive": 100},
        "reviews": [review],
    }
    valid_row = {
        "review_id": review["review_id"],
        "decision": "confirmed",
        "event_frame_index": 35,
        "effect_class": "rolling_object",
        "confirmation_artifacts": review["confirmation_artifacts"],
        "checks": _spell_confirmation_checks(),
        "reason": "new upper-origin rolling object moves downward",
    }
    validate_enemy_spell_confirmation_decisions(
        {"decisions": [valid_row]}, package
    )

    own_spell = json.loads(json.dumps(valid_row))
    own_spell["checks"]["enemy_direction_or_origin"] = False
    try:
        validate_enemy_spell_confirmation_decisions(
            {"decisions": [own_spell]}, package
        )
    except ValueError as error:
        assert "not directly confirmed" in str(error)
    else:
        raise AssertionError("own-direction spell passed the enemy gate")

    attack = json.loads(json.dumps(valid_row))
    attack["checks"]["not_unit_attack_or_ability"] = False
    try:
        validate_enemy_spell_confirmation_decisions(
            {"decisions": [attack]}, package
        )
    except ValueError as error:
        assert "not directly confirmed" in str(error)
    else:
        raise AssertionError("unit attack passed the enemy-spell gate")

    rejected = json.loads(json.dumps(valid_row))
    rejected.update(
        {
            "decision": "rejected",
            "event_frame_index": None,
            "effect_class": "unresolved",
        }
    )
    validate_enemy_spell_confirmation_decisions(
        {"decisions": [rejected]}, package
    )

    sentinel = {
        **review,
        "review_id": "spell-review:segment-end-000100",
        "proposal_frame_index": 99,
        "sampled_frame_indices": list(range(87, 100)),
        "segment_end_sentinel": True,
    }
    boundary_row = {
        **valid_row,
        "review_id": sentinel["review_id"],
        "event_frame_index": 99,
        "effect_class": "directional_projectile",
        "checks": _spell_confirmation_checks(
            resolved_after=None, boundary_truncated=True
        ),
    }
    validate_enemy_spell_confirmation_decisions(
        {"decisions": [boundary_row]},
        {**package, "reviews": [sentinel]},
    )
    boundary_row["event_frame_index"] = 97
    try:
        validate_enemy_spell_confirmation_decisions(
            {"decisions": [boundary_row]},
            {**package, "reviews": [sentinel]},
        )
    except ValueError as error:
        assert "lacks forward resolution" in str(error)
    else:
        raise AssertionError("non-final unresolved sequence passed boundary rule")


def test_enemy_spell_confirmation_merge_filters_and_corrects_before_identity(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    (run_dir / "work_packages").mkdir(parents=True)
    (run_dir / "worker_outputs").mkdir()
    manifest = {
        "run_id": "run",
        "segment": {"start_frame": 0, "end_frame_exclusive": 100},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    unit = {
        "onset_id": "enemy-unit-000010",
        "event_frame_index": 10,
        "kind": "unit_or_building",
    }
    spell = {
        "onset_id": "enemy-000020",
        "candidate_id": "enemy-scan:000020-000040:p1",
        "event_frame_index": 20,
        "kind": "spell",
    }
    false_spell = {
        "onset_id": "enemy-000060",
        "candidate_id": "enemy-scan:000060-000080:p1",
        "event_frame_index": 60,
        "kind": "spell",
    }
    (run_dir / "enemy_onsets.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "stage": "enemy_onsets",
                "onsets": [unit, spell, false_spell],
            }
        ),
        encoding="utf-8",
    )
    reviews = [
        {
            "review_id": "spell-review:enemy-000020",
            "source_onset_id": "enemy-000020",
            "source_candidate_id": spell["candidate_id"],
            "proposal_frame_index": 20,
            "sampled_frame_indices": list(range(8, 51)),
            "confirmation_artifacts": [
                "reviews/first-before.jpg",
                "reviews/first-after.jpg",
            ],
            "segment_end_sentinel": False,
        },
        {
            "review_id": "spell-review:enemy-000060",
            "source_onset_id": "enemy-000060",
            "source_candidate_id": false_spell["candidate_id"],
            "proposal_frame_index": 60,
            "sampled_frame_indices": list(range(48, 91)),
            "confirmation_artifacts": [
                "reviews/second-before.jpg",
                "reviews/second-after.jpg",
            ],
            "segment_end_sentinel": False,
        },
    ]
    package = {
        "run_id": "run",
        "stage": "enemy_spell_confirmation_package",
        "target_range": [0, 100],
        "segment": manifest["segment"],
        "reviews": reviews,
    }
    package_path = (
        run_dir
        / "work_packages"
        / "enemy-spell-confirmation-000000-000100.json"
    )
    package_path.write_text(json.dumps(package), encoding="utf-8")
    output = {
        "run_id": "run",
        "stage": "enemy_spell_confirmation_chunk",
        "target_range": [0, 100],
        "annotation_session_id": "independent-spell-gate",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "low",
        "decisions": [
            {
                "review_id": reviews[0]["review_id"],
                "decision": "confirmed",
                "event_frame_index": 35,
                "effect_class": "rolling_object",
                "confirmation_artifacts": reviews[0][
                    "confirmation_artifacts"
                ],
                "checks": _spell_confirmation_checks(),
                "reason": "later distinct enemy rolling object",
            },
            {
                "review_id": reviews[1]["review_id"],
                "decision": "rejected",
                "event_frame_index": None,
                "effect_class": "unresolved",
                "confirmation_artifacts": reviews[1][
                    "confirmation_artifacts"
                ],
                "checks": {
                    **_spell_confirmation_checks(),
                    "enemy_direction_or_origin": False,
                },
                "reason": "own floating label and upward motion",
            },
        ],
    }
    (
        run_dir
        / "worker_outputs"
        / "enemy-spell-confirmation-000000-000100.json"
    ).write_text(json.dumps(output), encoding="utf-8")
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "merge_enemy_spell_confirmation_chunks.py"
    )
    subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    merged = json.loads((run_dir / "enemy_onsets.json").read_text())
    assert [row["kind"] for row in merged["onsets"]] == [
        "unit_or_building",
        "spell",
    ]
    retained = merged["onsets"][1]
    assert retained["onset_id"] == "enemy-000020"
    assert retained["event_frame_index"] == 35
    assert retained["verification_artifacts"] == reviews[0][
        "confirmation_artifacts"
    ]
    assert all(
        row["onset_id"] != false_spell["onset_id"]
        for row in merged["onsets"]
    )
    audit = json.loads(
        (run_dir / "enemy_spell_confirmation.json").read_text()
    )
    assert [row["retained"] for row in audit["decisions"]] == [True, False]


def test_enemy_spell_reconciliation_uses_independent_confounds_and_boundary(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    (run_dir / "work_packages").mkdir(parents=True)
    (run_dir / "worker_outputs").mkdir()
    segment = {"start_frame": 0, "end_frame_exclusive": 100}
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run", "segment": segment}),
        encoding="utf-8",
    )
    (run_dir / "own_semantics.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "events": [
                    {
                        "side": "own",
                        "card": "log",
                        "event_frame_index": 22,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def onset(onset_id: str, frame: int, kind: str = "spell") -> dict:
        return {
            "onset_id": onset_id,
            "candidate_id": f"candidate:{frame}",
            "event_frame_index": frame,
            "kind": kind,
            "sampled_frame_indices": [frame],
            "absence_confirmed": True,
            "persistence_confirmed": True,
            "evidence": {
                "elixir_drop": None,
                "hand_transition": None,
                "deployment_onset": kind == "unit_or_building",
                "first_visible_object": True,
                "side_direction": None,
                "impact_sequence": kind == "spell",
            },
            "verification_artifacts": [f"reviews/{frame}.jpg"],
            "identity_artifacts": [],
        }

    raw = [
        onset("enemy-000010", 10),
        onset("enemy-000020", 20),
        onset("enemy-000040", 40),
        onset("enemy-unit-000040", 40, "unit_or_building"),
        onset("enemy-000090", 90),
    ]
    (run_dir / "enemy_onsets.json").write_text(
        json.dumps({"run_id": "run", "onsets": raw}),
        encoding="utf-8",
    )
    identities = [
        {
            "onset_id": row["onset_id"],
            "event_exists": True,
            "event_frame_index": row["event_frame_index"],
            "side": "enemy",
        }
        for row in raw
    ]
    (run_dir / "enemy_identities.json").write_text(
        json.dumps({"run_id": "run", "decisions": identities}),
        encoding="utf-8",
    )
    (run_dir / "enemy_spell_reconciliation_candidates.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "classifications": [
                    {
                        "onset_id": "enemy-000010",
                        "event_frame_index": 10,
                        "classification": "unconfounded_broad_spell",
                    },
                    {
                        "onset_id": "enemy-000020",
                        "event_frame_index": 20,
                        "classification": "own_spell_confound",
                    },
                    {
                        "onset_id": "enemy-000040",
                        "event_frame_index": 40,
                        "classification": "unit_confound",
                    },
                    {
                        "onset_id": "enemy-000090",
                        "event_frame_index": 90,
                        "classification": "unconfounded_broad_spell",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    recovery_review = {
        "review_id": "spell-recovery:enemy-000020",
        "source_onset_id": "enemy-000020",
        "source_candidate_id": "candidate:20",
        "proposal_frame_index": 20,
        "segment_end_sentinel": False,
        "inspection_range": [24, 43],
        "sampled_frame_indices": list(range(24, 43)),
        "confirmation_artifacts": ["reviews/recovery.jpg"],
    }
    boundary_review = {
        "review_id": "spell-review:segment-end-000100",
        "source_onset_id": None,
        "source_candidate_id": None,
        "proposal_frame_index": 99,
        "segment_end_sentinel": True,
        "inspection_range": [87, 100],
        "sampled_frame_indices": list(range(87, 100)),
        "confirmation_artifacts": ["reviews/boundary.jpg"],
    }

    def write_family(
        name: str,
        review: dict,
        *,
        frame: int,
        boundary: bool,
    ) -> None:
        package = {
            "run_id": "run",
            "stage": "enemy_spell_confirmation_package",
            "target_range": [0, 100],
            "segment": segment,
            "reviews": [review],
        }
        (run_dir / "work_packages" / name).write_text(
            json.dumps(package), encoding="utf-8"
        )
        checks = _spell_confirmation_checks()
        checks["boundary_truncated"] = boundary
        checks["resolved_after"] = None if boundary else True
        output = {
            "run_id": "run",
            "stage": "enemy_spell_confirmation_chunk",
            "target_range": [0, 100],
            "annotation_session_id": f"session-{name}",
            "model": "gpt-5.6-terra",
            "reasoning_effort": "low",
            "decisions": [
                {
                    "review_id": review["review_id"],
                    "decision": "confirmed",
                    "event_frame_index": frame,
                    "effect_class": (
                        "directional_projectile"
                        if boundary
                        else "rolling_object"
                    ),
                    "confirmation_artifacts": review[
                        "confirmation_artifacts"
                    ],
                    "checks": checks,
                    "reason": "direct independent physical spell evidence",
                }
            ],
        }
        (run_dir / "worker_outputs" / name).write_text(
            json.dumps(output), encoding="utf-8"
        )

    write_family(
        "enemy-spell-recovery-left-000000-000100.json",
        recovery_review,
        frame=30,
        boundary=False,
    )
    write_family(
        "enemy-spell-boundary-000000-000100.json",
        boundary_review,
        frame=98,
        boundary=True,
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "merge_enemy_spell_reconciliation.py"
    )
    subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    merged = json.loads((run_dir / "enemy_onsets.json").read_text())
    assert [
        (row["kind"], row["event_frame_index"])
        for row in merged["onsets"]
    ] == [
        ("spell", 10),
        ("spell", 30),
        ("unit_or_building", 40),
        ("spell", 90),
        ("spell", 98),
    ]
    audit = json.loads(
        (run_dir / "enemy_spell_reconciliation.json").read_text()
    )
    assert [
        (row["classification"], row["retained"])
        for row in audit["classifications"]
    ] == [
        ("unconfounded_broad_spell", True),
        ("own_spell_confound", True),
        ("unit_confound", False),
        ("unconfounded_broad_spell", True),
    ]


def test_enemy_unit_decisions_fail_closed_and_cover_every_burst():
    package = {
        "bursts": [
            {
                "burst_id": "enemy-marker-burst:000001",
                "package_role": "owned_burst",
            },
            {
                "burst_id": "enemy-marker-burst:000002",
                "package_role": "context_burst",
            },
        ]
    }
    valid = {
        "burst_decisions": [
            {
                "burst_id": "enemy-marker-burst:000001",
                "decision": "accepted",
                "accepted_onset_ids": [],
            },
            {
                "burst_id": "enemy-marker-burst:000002",
                "decision": "context_only",
                "accepted_onset_ids": [],
            },
        ],
        "onsets": [],
    }
    validate_enemy_unit_decisions(valid, package)

    clerical = {
        **valid,
        "burst_decisions": [
            {**valid["burst_decisions"][0], "decision": "context_only"},
            {**valid["burst_decisions"][1], "decision": "accepted"},
        ],
    }
    assert normalize_enemy_unit_decision_roles(clerical, package)
    assert [
        row["decision"] for row in clerical["burst_decisions"]
    ] == ["rejected", "context_only"]
    validate_enemy_unit_decisions(clerical, package)

    for invalid in (
        {**valid, "burst_decisions": valid["burst_decisions"][:1]},
        {
            **valid,
            "burst_decisions": [
                valid["burst_decisions"][0],
                {
                    **valid["burst_decisions"][1],
                    "decision": "rejected",
                },
            ],
        },
        {**valid, "onsets": [{"event_frame_index": 1}]},
    ):
        try:
            validate_enemy_unit_decisions(invalid, package)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid enemy-unit decision output accepted")


def test_enemy_unit_scan_requires_direct_enemy_sequence():
    artifact = "reviews/verify-enemy-scan-000100-000120-p1.jpg"
    package = {
        "owned_event_range": [100, 200],
        "primary_windows": [
            {
                "candidate_id": "enemy-scan:000100-000120:p1",
                "inspection_start_frame": 100,
                "inspection_end_frame_exclusive": 120,
                "verification_artifact": artifact,
            }
        ],
        "boundary_windows": [],
    }
    valid = {
        "onsets": [
            {
                "onset_id": "enemy-scan-unit-000110",
                "candidate_id": "enemy-scan:000100-000120:p1",
                "event_frame_index": 110,
                "kind": "unit_or_building",
                "side": "enemy",
                "evidence": {
                    "absent_before": True,
                    "independent_after": True,
                    "persistent_or_resolved_after": True,
                    "direct_enemy_side": True,
                },
                "verification_artifacts": [artifact],
                "reason": "new red body appears and persists",
            }
        ]
    }
    validate_enemy_unit_scan_decisions(valid, package)
    validate_enemy_unit_scan_decisions({"onsets": []}, package)

    invalid = json.loads(json.dumps(valid))
    invalid["onsets"][0]["evidence"]["direct_enemy_side"] = False
    try:
        validate_enemy_unit_scan_decisions(invalid, package)
    except ValueError:
        pass
    else:
        raise AssertionError("unit scan accepted non-enemy evidence")


def test_own_validator_accepts_compensated_one_cost_and_rejects_held_spell():
    artifact = "reviews/discover-own-000010.jpg"
    package = {
        "candidates": [
            {
                "candidate_id": "own:000010",
                "discovery_artifact": artifact,
                "discovery_frame_indices": [8, 10, 12, 16, 20],
            }
        ]
    }
    event = {
        "candidate_id": "own:000010",
        "card": "skeletons",
        "event_frame_index": 10,
        "evidence": {
            "elixir_drop": True,
            "hand_transition": True,
            "deployment_onset": True,
            "first_visible_object": None,
            "side_direction": None,
            "impact_sequence": None,
        },
        "transition_observation": {
            "elixir_before": 5,
            "elixir_after": 5,
            "observed_elixir_delta": 0,
            "total_released_cost": 1,
            "regeneration_compensated": True,
            "occupied_slots_before": [0, 1, 2, 3],
            "cooldown_slots_after": [2],
        },
        "spell_release": None,
        "verification_artifacts": [artifact],
        "confirmation_frame_index": 16,
        "confirmation_artifacts": [artifact],
        "own_confirmation": {
            "release_confirmed": True,
            "elixir_spend_persisted": True,
            "hand_cycle_completed": True,
            "post_release_effect": True,
        },
    }
    document = {
        "events": [event],
        "rejected_candidates": [],
        "pending_at_end": [],
    }
    validate_own_semantic_decisions(
        document, package, require_candidate_coverage=True
    )

    adjudication_package = {
        "proposals": [
            {
                "proposal_id": "own-proposal:000010:skeletons",
                "candidate_ids": ["own:000010"],
                "card": "skeletons",
                "candidate_evidence": package["candidates"],
            }
        ]
    }
    adjudication_document = {
        "events": [
            {
                "proposal_id": "own-proposal:000010:skeletons",
                **event,
            }
        ],
        "rejected_proposals": [],
    }
    validate_own_adjudication_decisions(
        adjudication_document, adjudication_package
    )

    compensated_without_a_visible_digit_drop = json.loads(
        json.dumps(document)
    )
    compensated_without_a_visible_digit_drop["events"][0]["evidence"][
        "elixir_drop"
    ] = False
    validate_own_semantic_decisions(
        compensated_without_a_visible_digit_drop,
        package,
        require_candidate_coverage=True,
    )

    for field, extra_key in (
        ("evidence", "unsupported_evidence"),
        ("own_confirmation", "unsupported_confirmation"),
    ):
        invalid = json.loads(json.dumps(document))
        invalid["events"][0][field][extra_key] = None
        try:
            validate_own_semantic_decisions(
                invalid, package, require_candidate_coverage=True
            )
        except ValueError as error:
            assert "release persistence is not confirmed" in str(error)
        else:
            raise AssertionError(f"extra {field} key accepted")

        invalid_adjudication = json.loads(json.dumps(adjudication_document))
        invalid_adjudication["events"][0][field][extra_key] = None
        try:
            validate_own_adjudication_decisions(
                invalid_adjudication, adjudication_package
            )
        except ValueError as error:
            assert "release persistence is not confirmed" in str(error)
        else:
            raise AssertionError(
                f"adjudication accepted extra {field} key"
            )

    for field, required_key in (
        ("evidence", "impact_sequence"),
        ("own_confirmation", "post_release_effect"),
    ):
        invalid = json.loads(json.dumps(document))
        del invalid["events"][0][field][required_key]
        try:
            validate_own_semantic_decisions(
                invalid, package, require_candidate_coverage=True
            )
        except ValueError as error:
            assert "release persistence is not confirmed" in str(error)
        else:
            raise AssertionError(f"missing {field} key accepted")

    held_spell = json.loads(json.dumps(event))
    held_spell["card"] = "fireball"
    held_spell["transition_observation"].update(
        {
            "elixir_before": 8,
            "elixir_after": 4,
            "observed_elixir_delta": 4,
            "total_released_cost": 4,
            "regeneration_compensated": False,
        }
    )
    held_spell["spell_release"] = {
        "targeting_overlay_cleared": False,
        "projectile_or_impact_visible": False,
    }
    try:
        validate_own_semantic_decisions(
            {
                "events": [held_spell],
                "rejected_candidates": [],
                "pending_at_end": [],
            },
            package,
            require_candidate_coverage=True,
        )
    except ValueError as error:
        assert "spell release is not directly confirmed" in str(error)
    else:
        raise AssertionError("held spell preview accepted as a release")


def test_own_merge_assigns_corrected_boundary_event_to_next_chunk(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    packages = run_dir / "work_packages"
    outputs = run_dir / "worker_outputs"
    packages.mkdir(parents=True)
    outputs.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run"}), encoding="utf-8"
    )
    candidate = {
        "candidate_id": "own:000045",
        "discovery_artifact": "reviews/discover-own-000045.jpg",
    }
    event = {
        "candidate_id": "own:000045",
        "card": "ice-golem",
        "event_frame_index": 50,
    }
    for start, end, role in (
        (0, 50, "owned_peak"),
        (50, 100, "context_peak"),
    ):
        name = f"own-{start:06d}-{end:06d}.json"
        (packages / name).write_text(
            json.dumps(
                {
                    "run_id": "run",
                    "target_range": [start, end],
                    "candidates": [{**candidate, "package_role": role}],
                }
            ),
            encoding="utf-8",
        )
        (outputs / name).write_text(
            json.dumps(
                {
                    "run_id": "run",
                    "stage": "own_semantics_chunk",
                    "target_range": [start, end],
                    "events": [event],
                    "rejected_candidates": [],
                }
            ),
            encoding="utf-8",
        )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "merge_semantic_chunks.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--side",
            "own",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    merged = json.loads((run_dir / "own_semantics.json").read_text())
    assert merged["events"] == [event]


def test_own_adjudication_clustering_preserves_later_distinct_candidate():
    rows = [
        {
            "candidate_id": candidate_id,
            "card": "skeletons",
            "event_frame_index": frame,
        }
        for candidate_id, frame in (
            ("own:000000", 0),
            ("own:000000", 15),
            ("own:000030", 30),
        )
    ]
    clusters = _cluster_rows(rows, max_span_frames=20)
    assert [
        [row["event_frame_index"] for row in cluster]
        for cluster in clusters
    ] == [[0, 15], [30]]

    held_then_released = [
        {
            "candidate_id": candidate_id,
            "card": "hog-rider",
            "event_frame_index": frame,
        }
        for candidate_id, frame in (
            ("own:000005", 16),
            ("own:000030", 30),
        )
    ]
    assert [
        [row["event_frame_index"] for row in cluster]
        for cluster in _cluster_rows(held_then_released)
    ] == [[16], [30]]


def test_deterministic_own_union_keeps_each_proposal_and_latest_complete_row():
    def proposal(proposal_id: str, frames: tuple[int, ...]):
        rows = [
            {
                "candidate_id": f"own:{frame:06d}",
                "card": "hog-rider",
                "event_frame_index": frame,
                "complete_schema_marker": {"source_frame": frame},
            }
            for frame in frames
        ]
        return {
            "proposal_id": proposal_id,
            "card": "hog-rider",
            "proposed_frames": list(frames),
            "candidate_ids": [row["candidate_id"] for row in rows],
            "candidate_rows": rows,
        }

    held_preview = proposal("own-proposal-0000", (16,))
    later_release = proposal("own-proposal-0001", (29, 30))
    selected = [
        select_proposal_event(held_preview),
        select_proposal_event(later_release),
    ]
    assert [row["event_frame_index"] for row in selected] == [16, 30]
    assert selected[1]["complete_schema_marker"] == {"source_frame": 30}

    # Selection returns a copy; materialization must not mutate its package.
    selected[1]["complete_schema_marker"]["source_frame"] = -1
    assert later_release["candidate_rows"][1]["complete_schema_marker"] == {
        "source_frame": 30
    }


def test_recommended_pipeline_uses_deterministic_own_slot_gate():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/codex_annotation/run_annotation_pipeline.py"
    ).read_text(encoding="utf-8")
    assert 'family="own-adjudication"' not in source
    assert 'prepare_own_slot_interval_packages.py' in source
    assert 'merge_own_slot_interval_chunks.py' in source
    assert 'build_release_review_from_own_slots.py' in source
    assert 'materialize_own_proposal_union.py' not in source
    assert '"--one-target-per-package",' not in source


def test_semantic_merge_adds_previous_scan_at_exact_window_boundary(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    previous = run_dir / "reviews" / "scan-000320-000340.jpg"
    current = run_dir / "reviews" / "scan-000340-000360.jpg"
    identity = run_dir / "reviews" / "focus-000340-000360.jpg"
    reviews = [
        {
            "path": str(previous),
            "purpose": "arena",
            "start_frame": 320,
            "end_frame_exclusive": 340,
        },
        {
            "path": str(current),
            "purpose": "arena",
            "start_frame": 340,
            "end_frame_exclusive": 360,
        },
        {
            "path": str(identity),
            "purpose": "identity",
            "start_frame": 340,
            "end_frame_exclusive": 360,
        },
    ]
    event = {
        "event_id": "event-enemy-000340-giant-snowball",
        "side": "enemy",
        "event_frame_index": 340,
        "verification_artifacts": [
            "reviews/scan-000340-000360.jpg",
            "reviews/focus-000340-000360.jpg",
        ],
    }

    completed = _ensure_event_context(
        event,
        run_dir=run_dir,
        reviews=reviews,
        segment_start=0,
        segment_end=400,
    )

    assert completed["verification_artifacts"] == [
        "reviews/scan-000340-000360.jpg",
        "reviews/scan-000320-000340.jpg",
    ]


def test_semantic_merge_constructs_enemy_spell_event() -> None:
    onset = {
        "onset_id": "enemy-000340",
        "candidate_id": "enemy-scan:000340-000360:p1",
        "event_frame_index": 340,
        "kind": "spell",
        "verification_artifacts": ["reviews/scan.jpg"],
        "evidence": {
            "elixir_drop": None,
            "hand_transition": None,
            "deployment_onset": None,
            "first_visible_object": None,
            "impact_sequence": True,
        },
    }
    identity = {"card": "giant-snowball", "side_evidence": {"direct": True}}

    event = _enemy_event(onset, identity)

    assert event["event_id"] == "event-enemy-000340-giant-snowball"
    assert event["side"] == "enemy"


def test_semantic_merge_rebinds_corrected_onset_to_supporting_scan() -> None:
    manifest = {
        "candidate_discovery": {
            "enemy_scan_windows": [
                {
                    "candidate_id": "enemy-scan:000460-000480:p1",
                    "inspection_start_frame": 460,
                    "inspection_end_frame_exclusive": 480,
                },
                {
                    "candidate_id": "enemy-scan:000480-000500:p1",
                    "inspection_start_frame": 480,
                    "inspection_end_frame_exclusive": 500,
                },
            ]
        }
    }
    event = {
        "event_id": "event-enemy-000487-cannon",
        "side": "enemy",
        "event_frame_index": 487,
        "candidate_id": "enemy-scan:000460-000480:p1",
    }

    rebound = _ensure_candidate_support(event, manifest)

    assert rebound["candidate_id"] == "enemy-scan:000480-000500:p1"


def test_overlap_package_equal_frame_order_has_onset_id_tiebreak() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/codex_annotation/prepare_enemy_overlap_adjudication_packages.py"
    ).read_text(encoding="utf-8")

    assert 'all_onsets[value]["event_frame_index"],\n                value,' in source


def test_enemy_identity_reviews_scope_to_corrected_event_frame() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/codex_annotation/render_enemy_identity_targets.py"
    ).read_text(encoding="utf-8")

    assert source.count("f\"{int(target['event_frame_index']):06d}\"") == 3
    assert 'event_id=target["onset_id"]' not in source


def test_enemy_spell_artifact_normalizer_uses_candidate_owned_sheet() -> None:
    package = {
        "primary_windows": [
            {
                "candidate_id": "enemy-scan:001000-001020:p1",
                "verification_artifact": "reviews/scan-1000-1020.jpg",
            }
        ]
    }
    document = {
        "onsets": [
            {
                "candidate_id": "enemy-scan:001000-001020:p1",
                "verification_artifacts": ["scan 1000.jpg"],
            }
        ]
    }

    assert normalize_enemy_spell_decision_artifacts(document, package)
    assert document["onsets"][0]["verification_artifacts"] == [
        "reviews/scan-1000-1020.jpg"
    ]


def test_enemy_spell_confirmation_normalizer_uses_review_owned_sheets() -> None:
    package = {
        "reviews": [
            {
                "review_id": "spell-recovery:enemy-000278:left",
                "confirmation_artifacts": [
                    "reviews/recovery-278-a.jpg",
                    "reviews/recovery-278-b.jpg",
                ],
            }
        ]
    }
    document = {
        "decisions": [
            {
                "review_id": "spell-recovery:enemy-000278:left",
                "confirmation_artifacts": ["recovery 278.jpg"],
            }
        ]
    }

    assert normalize_enemy_spell_confirmation_artifacts(document, package)
    assert document["decisions"][0]["confirmation_artifacts"] == [
        "reviews/recovery-278-a.jpg",
        "reviews/recovery-278-b.jpg",
    ]


def test_own_candidate_support_is_bounded_but_completeness_is_independent() -> None:
    manifest = {
        "fps": 10.0,
        "segment": {"start_frame": 0, "end_frame_exclusive": 1000},
        "candidate_discovery": {
            "own_candidates": [
                {
                    "candidate_id": "own:000666",
                    "approximate_frame_index": 666,
                }
            ],
            "enemy_scan_windows": [],
        },
    }

    _validate_candidate_frame(manifest, 0, "own:000666", 646)
    _validate_candidate_frame(manifest, 1, "own:000666", 686)
    _validate_candidate_frame(
        manifest,
        2,
        "completeness:own:slot-1-000600-000734",
        734,
    )


def test_rejected_own_proposal_expands_to_candidate_rejections(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    packages = run_dir / "work_packages"
    outputs = run_dir / "worker_outputs"
    packages.mkdir(parents=True)
    outputs.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run"}), encoding="utf-8"
    )
    name = "own-adjudicate-000000-000100.json"
    (packages / name).write_text(
        json.dumps(
            {
                "run_id": "run",
                "target_range": [0, 100],
                "proposals": [
                    {
                        "proposal_id": "own-proposal-0000",
                        "candidate_ids": ["own:000010", "own:000012"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (outputs / name).write_text(
        json.dumps(
            {
                "run_id": "run",
                "stage": "own_adjudication_chunk",
                "target_range": [0, 100],
                "events": [],
                "rejected_proposals": [
                    {
                        "proposal_id": "own-proposal-0000",
                        "reason": "card returned to hand",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "merge_own_adjudication_chunks.py"
    )
    subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    merged = json.loads((run_dir / "own_semantics.json").read_text())
    assert merged["rejected_candidates"] == [
        {
            "candidate_id": "own:000010",
            "proposal_id": "own-proposal-0000",
            "reason": "card returned to hand",
        },
        {
            "candidate_id": "own:000012",
            "proposal_id": "own-proposal-0000",
            "reason": "card returned to hand",
        },
    ]


def _write_pending_merge_chunk(
    run_dir: Path,
    *,
    start: int,
    end: int,
    role: str,
    pending: bool = False,
    rejected: bool = False,
) -> None:
    candidate = {
        "candidate_id": "own:000045",
        "package_role": role,
    }
    name = f"own-{start:06d}-{end:06d}.json"
    (run_dir / "work_packages" / name).write_text(
        json.dumps(
            {
                "run_id": "run",
                "target_range": [start, end],
                "candidates": [candidate],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "worker_outputs" / name).write_text(
        json.dumps(
            {
                "run_id": "run",
                "stage": "own_semantics_chunk",
                "target_range": [start, end],
                "events": [],
                "rejected_candidates": (
                    [
                        {
                            "candidate_id": "own:000045",
                            "reason": "card returned to hand",
                        }
                    ]
                    if rejected
                    else []
                ),
                "pending_at_end": (
                    [
                        {
                            "candidate_id": "own:000045",
                            "card": "hog-rider",
                            "reason": "held across boundary",
                        }
                    ]
                    if pending
                    else []
                ),
            }
        ),
        encoding="utf-8",
    )


def test_pending_own_candidate_requires_and_records_later_resolution(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    (run_dir / "work_packages").mkdir(parents=True)
    (run_dir / "worker_outputs").mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run"}), encoding="utf-8"
    )
    _write_pending_merge_chunk(
        run_dir, start=0, end=50, role="owned_peak", pending=True
    )
    _write_pending_merge_chunk(
        run_dir, start=50, end=100, role="context_peak", rejected=True
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "merge_semantic_chunks.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--side",
            "own",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    merged = json.loads((run_dir / "own_semantics.json").read_text())
    assert merged["rejected_candidates"] == [
        {
            "candidate_id": "own:000045",
            "reason": "card returned to hand",
        }
    ]
    assert merged["pending_resolutions"] == [
        {
            "candidate_id": "own:000045",
            "pending_target_range": [0, 50],
            "resolved_target_range": [50, 100],
            "decision": "rejected",
        }
    ]


def test_pending_own_candidate_without_continuation_fails_merge(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    (run_dir / "work_packages").mkdir(parents=True)
    (run_dir / "worker_outputs").mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run"}), encoding="utf-8"
    )
    _write_pending_merge_chunk(
        run_dir, start=0, end=50, role="owned_peak", pending=True
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "merge_semantic_chunks.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--side",
            "own",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "has no following overlapping chunk" in result.stderr


def test_enemy_identity_gate_requires_existence_and_delayed_evidence():
    package = {
        "onsets": [
            {
                "onset_id": "enemy-unit-000010-b000001",
                "kind": "unit_or_building",
            },
            {"onset_id": "enemy-spell-000020", "kind": "spell"},
        ]
    }
    document = {
        "decisions": [
            {
                "onset_id": "enemy-unit-000010-b000001",
                "event_exists": True,
                "existence_evidence": {
                    "absent_before": True,
                    "independent_after": True,
                    "persistent_after": True,
                },
                "side": "enemy",
                "card": None,
                "side_evidence": {"direct": True},
                "identity_frame_index": None,
                "identity_artifacts": [],
            },
            {
                "onset_id": "enemy-spell-000020",
                "event_exists": False,
                "existence_evidence": {
                    "absent_before": False,
                    "independent_after": False,
                    "persistent_after": False,
                },
                "side": "unresolved",
                "card": None,
                "side_evidence": {"direct": False},
                "identity_frame_index": None,
                "identity_artifacts": [],
            },
        ]
    }
    validate_enemy_identity_decisions(document, package)
    document["decisions"][0]["identity_artifacts"] = ["unexpected.jpg"]
    try:
        validate_enemy_identity_decisions(document, package)
    except ValueError as error:
        assert "must not create identity evidence" in str(error)
    else:
        raise AssertionError("side gate identity evidence accepted")


def test_enemy_existence_requires_exact_coverage_and_a_labeled_frame():
    package = {
        "decision_schema_version": 2,
        "candidates": [
            {
                "onset_id": "enemy-unit-000010-b000001",
                "event_frame_index": 10,
                "sampled_frame_indices": [8, 10, 12, 17],
            },
            {
                "onset_id": "enemy-unit-000030-b000002",
                "event_frame_index": 30,
                "sampled_frame_indices": [28, 30, 32],
            },
        ],
    }
    valid = {
        "decisions": [
            {
                "onset_id": "enemy-unit-000010-b000001",
                "overlap_event_exists": True,
                "event_frame_index": 12,
                "evidence": {
                    "secondary_absent_before": True,
                    "secondary_appears_at_marker": True,
                    "secondary_persists_or_resolves_after": True,
                    "direct_new_actor": True,
                },
                "side": "unresolved",
            },
            {
                "onset_id": "enemy-unit-000030-b000002",
                "overlap_event_exists": False,
                "event_frame_index": None,
                "evidence": {},
                "side": "unresolved",
            },
        ]
    }
    validate_enemy_existence_decisions(valid, package)

    invalid_frame = json.loads(json.dumps(valid))
    invalid_frame["decisions"][0]["event_frame_index"] = 11
    try:
        validate_enemy_existence_decisions(invalid_frame, package)
    except ValueError as error:
        assert "labeled sampled frame" in str(error)
    else:
        raise AssertionError("unlabeled corrected frame accepted")

    missing = {"decisions": valid["decisions"][:1]}
    try:
        validate_enemy_existence_decisions(missing, package)
    except ValueError as error:
        assert "cover package candidates exactly" in str(error)
    else:
        raise AssertionError("incomplete existence output accepted")


def test_enemy_side_check_requires_direct_evidence_or_unresolved():
    package = {
        "decision_schema_version": 2,
        "candidates": [
            {"onset_id": "enemy-unit-000010-b000001"},
            {"onset_id": "enemy-unit-000020-b000002"},
        ]
    }
    valid = {
        "decisions": [
            {
                "onset_id": "enemy-unit-000010-b000001",
                "side": "enemy",
                "direct": True,
                "team_indicator": "red",
                "origin": "upper",
                "motion": None,
            },
            {
                "onset_id": "enemy-unit-000020-b000002",
                "side": "unresolved",
                "direct": False,
                "team_indicator": None,
                "origin": None,
                "motion": None,
            },
        ]
    }
    validate_enemy_side_check_decisions(valid, package)
    valid["decisions"][0]["direct"] = False
    try:
        validate_enemy_side_check_decisions(valid, package)
    except ValueError as error:
        assert "lacks direct evidence" in str(error)
    else:
        raise AssertionError("inferred side accepted")
    invalid_enemy = json.loads(json.dumps(valid))
    invalid_enemy["decisions"][0].update(
        {"direct": True, "team_indicator": None}
    )
    try:
        validate_enemy_side_check_decisions(invalid_enemy, package)
    except ValueError as error:
        assert "red plus origin/direction" in str(error)
    else:
        raise AssertionError("enemy side without red evidence accepted")


def test_resume_fingerprint_keeps_template_and_rendered_prompt_hashes_separate(
    tmp_path: Path,
):
    package = tmp_path / "package.json"
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "output.json"
    package.write_text('{"run_id":"r"}', encoding="utf-8")
    prompt.write_text("package={PACKAGE_FILE}", encoding="utf-8")
    output.write_text('{"stage":"done"}', encoding="utf-8")
    spec = MODEL_PROFILES["terra-efficient"]["own_primary"]
    fingerprint = job_fingerprint(
        package=package,
        prompt=prompt,
        model_spec=spec,
    )
    assert "prompt_template_sha256" in fingerprint
    assert "prompt_sha256" not in fingerprint
    assert fingerprint["evidence_sha256"] == {}

    row = {
        "status": "succeeded",
        **fingerprint,
        "prompt_sha256": "rendered-prompt-hash",
        "output_sha256": sha256_file(output),
    }
    assert completed_job_matches(
        row,
        fingerprint=fingerprint,
        output=output,
    )


def test_resume_fingerprint_changes_when_review_pixels_change(tmp_path: Path):
    run_dir = tmp_path / "run"
    packages = run_dir / "work_packages"
    reviews = run_dir / "reviews"
    packages.mkdir(parents=True)
    reviews.mkdir()
    image = reviews / "evidence.jpg"
    image.write_bytes(b"first pixels")
    package = packages / "own-000000-000100.json"
    package.write_text(
        json.dumps({"artifact": "reviews/evidence.jpg"}),
        encoding="utf-8",
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("inspect", encoding="utf-8")
    spec = MODEL_PROFILES["terra-efficient"]["own_primary"]
    first = job_fingerprint(
        package=package,
        prompt=prompt,
        model_spec=spec,
    )
    image.write_bytes(b"changed pixels")
    second = job_fingerprint(
        package=package,
        prompt=prompt,
        model_spec=spec,
    )
    assert first["package_sha256"] == second["package_sha256"]
    assert first["evidence_sha256"] != second["evidence_sha256"]


def test_side_packages_preserve_candidates_until_side_is_known(tmp_path: Path):
    run_dir = tmp_path / "run"
    packages = run_dir / "work_packages"
    outputs = run_dir / "worker_outputs"
    packages.mkdir(parents=True)
    outputs.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "segment": {
                    "start_frame": 0,
                    "end_frame_exclusive": 800,
                },
            }
        ),
        encoding="utf-8",
    )
    onsets = [
        {
            "onset_id": "enemy-unit-000021-b000000",
            "event_frame_index": 21,
            "verification_artifacts": ["reviews/a.jpg", "reviews/af.jpg"],
        },
        {
            "onset_id": "enemy-unit-000036-b000001",
            "event_frame_index": 36,
            "verification_artifacts": ["reviews/b.jpg", "reviews/bf.jpg"],
        },
        {
            "onset_id": "enemy-unit-000100-b000002",
            "event_frame_index": 100,
            "verification_artifacts": ["reviews/c.jpg", "reviews/cf.jpg"],
        },
    ]
    (run_dir / "enemy_onsets.json").write_text(
        json.dumps({"onsets": onsets}),
        encoding="utf-8",
    )
    for start, end in ((0, 400), (400, 800)):
        (packages / f"identity-{start:06d}-{end:06d}.json").write_text(
            json.dumps(
                {
                    "run_id": "run",
                    "fps": 10.0,
                    "target_range": [start, end],
                    "onsets": [],
                    "own_release_frames": [],
                    "rejected_own_drags": [],
                }
            ),
            encoding="utf-8",
        )
    evidence = {
        "secondary_absent_before": True,
        "secondary_appears_at_marker": True,
        "secondary_persists_or_resolves_after": True,
        "direct_new_actor": True,
    }
    overlap_candidates = [
        {
            "onset_id": row["onset_id"],
            "event_frame_index": row["event_frame_index"],
            "sampled_frame_indices": [
                row["event_frame_index"],
                corrected,
            ],
        }
        for row, corrected in zip(onsets, (37, 36, 100), strict=True)
    ]
    for start, end, candidates in (
        (0, 400, overlap_candidates),
        (400, 800, []),
    ):
        (
            packages
            / f"identity-overlap-{start:06d}-{end:06d}.json"
        ).write_text(
            json.dumps(
                {
                    "run_id": "run",
                    "target_range": [start, end],
                    "decision_schema_version": 2,
                    "candidates": candidates,
                }
            ),
            encoding="utf-8",
        )
    (outputs / "identity-overlap-000000-000400.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "stage": "enemy_overlap_adjudication_chunk",
                "target_range": [0, 400],
                "decisions": [
                    {
                        "onset_id": onsets[0]["onset_id"],
                        "overlap_event_exists": True,
                            "event_frame_index": 37,
                            "evidence": evidence,
                            "side": "unresolved",
                        },
                    {
                        "onset_id": onsets[1]["onset_id"],
                        "overlap_event_exists": True,
                            "event_frame_index": 36,
                            "evidence": evidence,
                            "side": "unresolved",
                        },
                    {
                        "onset_id": onsets[2]["onset_id"],
                        "overlap_event_exists": True,
                            "event_frame_index": 100,
                            "evidence": evidence,
                            "side": "unresolved",
                        },
                ],
            }
        ),
        encoding="utf-8",
    )
    (outputs / "identity-overlap-000400-000800.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "stage": "enemy_overlap_adjudication_chunk",
                "target_range": [400, 800],
                "decisions": [],
            }
        ),
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "prepare_enemy_side_check_packages.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--chunk-frames",
            "400",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    side = json.loads(
        (packages / "identity-side-000000-000400.json").read_text()
    )
    assert [row["event_frame_index"] for row in side["candidates"]] == [
        37,
        36,
        100,
    ]
    assert [row["source_onset_ids"] for row in side["candidates"]] == [
        [row["onset_id"]] for row in onsets
    ]


def test_enemy_unit_gate_mini_dag_deduplicates_after_side_and_keeps_spell(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    packages = run_dir / "work_packages"
    outputs = run_dir / "worker_outputs"
    packages.mkdir(parents=True)
    outputs.mkdir()
    manifest = {
        "run_id": "run",
        "segment": {"start_frame": 0, "end_frame_exclusive": 100},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    def unit(onset_id: str, frame: int) -> dict:
        return {
            "onset_id": onset_id,
            "candidate_id": f"candidate:{frame}",
            "event_frame_index": frame,
            "kind": "unit_or_building",
            "absence_confirmed": None,
            "persistence_confirmed": True,
            "evidence": {
                "elixir_drop": None,
                "hand_transition": None,
                "deployment_onset": True,
                "first_visible_object": True,
                "side_direction": None,
                "impact_sequence": None,
            },
            "verification_artifacts": [
                f"reviews/{onset_id}.jpg",
                f"reviews/{onset_id}-focus.jpg",
            ],
            "identity_artifacts": [],
        }

    units = [
        unit("enemy-unit-000010-b000001", 10),
        unit("enemy-unit-000015-b000002", 15),
        unit("enemy-unit-000030-b000003", 30),
        unit("enemy-unit-000050-b000004", 50),
    ]
    spell = {
        "onset_id": "enemy-spell-000070",
        "candidate_id": "enemy-scan:000060-000080:p1",
        "event_frame_index": 70,
        "kind": "spell",
        "absence_confirmed": True,
        "persistence_confirmed": True,
        "evidence": {
            "elixir_drop": None,
            "hand_transition": None,
            "deployment_onset": True,
            "first_visible_object": True,
            "side_direction": True,
            "impact_sequence": True,
        },
        "verification_artifacts": ["reviews/spell.jpg"],
        "identity_artifacts": [],
    }
    (run_dir / "enemy_onsets.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "stage": "enemy_onsets",
                "onsets": [*units, spell],
            }
        ),
        encoding="utf-8",
    )
    sampled = {
        units[0]["onset_id"]: [10, 20],
        units[1]["onset_id"]: [15, 21],
        units[2]["onset_id"]: [30],
        units[3]["onset_id"]: [50],
    }
    overlap_package = {
        "run_id": "run",
        "target_range": [0, 100],
        "decision_schema_version": 2,
        "candidates": [
            {
                "onset_id": row["onset_id"],
                "event_frame_index": row["event_frame_index"],
                "sampled_frame_indices": sampled[row["onset_id"]],
            }
            for row in units
        ],
    }
    (
        packages / "identity-overlap-000000-000100.json"
    ).write_text(json.dumps(overlap_package), encoding="utf-8")
    evidence = {
        "secondary_absent_before": True,
        "secondary_appears_at_marker": True,
        "secondary_persists_or_resolves_after": True,
        "direct_new_actor": True,
    }
    existence = {
        "run_id": "run",
        "stage": "enemy_overlap_adjudication_chunk",
        "target_range": [0, 100],
        "annotation_session_id": "existence",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "decisions": [
            {
                "onset_id": units[0]["onset_id"],
                "overlap_event_exists": True,
                "event_frame_index": 20,
                "evidence": evidence,
                "side": "unresolved",
            },
            {
                "onset_id": units[1]["onset_id"],
                "overlap_event_exists": True,
                "event_frame_index": 21,
                "evidence": evidence,
                "side": "unresolved",
            },
            {
                "onset_id": units[2]["onset_id"],
                "overlap_event_exists": True,
                "event_frame_index": 30,
                "evidence": evidence,
                "side": "unresolved",
            },
            {
                "onset_id": units[3]["onset_id"],
                "overlap_event_exists": False,
                "event_frame_index": None,
                "evidence": {},
                "side": "unresolved",
            },
        ],
    }
    (
        outputs / "identity-overlap-000000-000100.json"
    ).write_text(json.dumps(existence), encoding="utf-8")
    (packages / "identity-000000-000100.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "fps": 10.0,
                "target_range": [0, 100],
                "onsets": [*units, spell],
                "own_release_frames": [],
                "rejected_own_drags": [],
            }
        ),
        encoding="utf-8",
    )
    prepare_side = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "prepare_enemy_side_check_packages.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(prepare_side),
            "--run-dir",
            str(run_dir),
            "--chunk-frames",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    side_package = json.loads(
        (packages / "identity-side-000000-000100.json").read_text()
    )
    assert len(side_package["candidates"]) == 3
    (outputs / "identity-side-000000-000100.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "stage": "enemy_side_check_chunk",
                "target_range": [0, 100],
                "annotation_session_id": "side",
                "model": "gpt-5.6-terra",
                "reasoning_effort": "low",
                "decisions": [
                    {
                        "onset_id": units[0]["onset_id"],
                        "side": "enemy",
                        "direct": True,
                        "team_indicator": "red",
                        "origin": "upper",
                        "motion": None,
                    },
                    {
                        "onset_id": units[1]["onset_id"],
                        "side": "enemy",
                        "direct": True,
                        "team_indicator": "red",
                        "origin": "upper",
                        "motion": None,
                    },
                    {
                        "onset_id": units[2]["onset_id"],
                        "side": "own",
                        "direct": True,
                        "team_indicator": "blue",
                        "origin": "lower",
                        "motion": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    merge = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "merge_enemy_unit_gate_chunks.py"
    )
    subprocess.run(
        [sys.executable, str(merge), "--run-dir", str(run_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    merged = json.loads((run_dir / "enemy_identities.json").read_text())
    by_id = {row["onset_id"]: row for row in merged["decisions"]}
    assert by_id[units[0]["onset_id"]]["event_exists"] is False
    assert by_id[units[1]["onset_id"]]["event_exists"] is True
    assert by_id[units[1]["onset_id"]]["event_frame_index"] == 21
    assert by_id[units[2]["onset_id"]]["side"] == "own"
    assert by_id[units[3]["onset_id"]]["event_exists"] is False
    assert by_id[spell["onset_id"]]["event_exists"] is True
    assert by_id[spell["onset_id"]]["side"] == "enemy"
    updated_onsets = json.loads(
        (run_dir / "enemy_onsets.json").read_text()
    )["onsets"]
    assert {
        row["onset_id"]: row["event_frame_index"] for row in updated_onsets
    }[units[1]["onset_id"]] == 21


def test_state_rejects_profile_switch_and_counts_only_successes(tmp_path: Path):
    path = tmp_path / "pipeline_state.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "run",
                "profile": "terra-efficient",
                "jobs": {
                    "a": {"status": "succeeded", "weighted_tokens": 100},
                    "b": {"status": "failed", "weighted_tokens": 500},
                    "c": {"status": "succeeded", "weighted_tokens": None},
                },
            }
        ),
        encoding="utf-8",
    )
    state = load_state(path, run_id="run", profile="terra-efficient")
    assert accumulated_weighted_tokens(state) == 100

    try:
        load_state(path, run_id="run", profile="sol-experimental")
    except ValueError as error:
        assert "profile changed" in str(error)
    else:
        raise AssertionError("profile switch should be rejected")


def test_enemy_card_merge_fails_closed_then_accepts_direct_clear_identity(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    packages = run_dir / "work_packages"
    outputs = run_dir / "worker_outputs"
    packages.mkdir(parents=True)
    outputs.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run"}),
        encoding="utf-8",
    )
    target = {
        "onset_id": "enemy-000010",
        "event_frame_index": 10,
        "kind": "unit_or_building",
        "identity_frame_index": 18,
        "identity_artifacts": ["reviews/a.jpg", "reviews/b.jpg"],
    }
    (run_dir / "enemy_identity_targets.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "stage": "enemy_identity_targets",
                "targets": [target],
            }
        ),
        encoding="utf-8",
    )
    package = {
        "run_id": "run",
        "target_range": [0, 20],
        "targets": [target],
    }
    package_path = packages / "cards-000000-000020.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    output_path = outputs / package_path.name

    def write_card(card):
        output_path.write_text(
            json.dumps(
                {
                    "run_id": "run",
                    "stage": "enemy_cards_chunk",
                    "target_range": [0, 20],
                    "annotation_session_id": "test",
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "medium",
                    "cards": [card],
                }
            ),
            encoding="utf-8",
        )

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "codex_annotation"
        / "merge_enemy_card_chunks.py"
    )
    write_card(
        {
            "onset_id": "enemy-000010",
            "card": None,
            "confidence": "unresolved",
            "visibility": "obscured",
            "identity_frame_index": 18,
            "identity_artifacts": ["reviews/a.jpg", "reviews/b.jpg"],
        }
    )
    unresolved = subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert unresolved.returncode != 0
    assert "remains unresolved" in unresolved.stderr

    write_card(
        {
            "onset_id": "enemy-000010",
            "card": "mega-knight",
            "confidence": "direct",
            "visibility": "clear",
            "identity_frame_index": 18,
            "identity_artifacts": ["reviews/a.jpg", "reviews/b.jpg"],
        }
    )
    subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir)],
        text=True,
        capture_output=True,
        check=True,
    )
    merged = json.loads((run_dir / "enemy_cards.json").read_text())
    assert [row["card"] for row in merged["cards"]] == ["mega-knight"]


def test_neighbor_identity_sheet_is_a_valid_single_delayed_artifact():
    target = {
        "onset_id": "enemy-000010",
        "event_frame_index": 10,
        "kind": "unit_or_building",
        "identity_frame_index": 18,
        "identity_artifacts": ["reviews/v3-neighbors/identity.jpg"],
        "identity_render_options": {"mode": "neighbor_candidates"},
    }
    package = {"targets": [target]}
    document = {
        "cards": [
            {
                "onset_id": "enemy-000010",
                "card": "ice-spirit",
                "confidence": "direct",
                "visibility": "clear",
                "identity_frame_index": 18,
                "identity_artifacts": ["reviews/v3-neighbors/identity.jpg"],
            }
        ]
    }
    from cr_bot.annotation_pipeline import validate_enemy_card_decisions

    validate_enemy_card_decisions(document, package)

    document["cards"][0]["card"] = "evo-knight"
    validate_enemy_card_decisions(document, package)


def test_onset_deck_package_attaches_full_onset_and_identity_but_omits_focus():
    source = {
        "run_id": "run-1",
        "targets": [
            {
                "onset_id": "enemy-unit-10",
                "kind": "unit_or_building",
                "verification_artifacts": [
                    "reviews/onset.jpg",
                    "reviews/focus.jpg",
                ],
                "identity_artifacts": ["reviews/neighbor.jpg"],
            },
            {
                "onset_id": "enemy-spell-20",
                "kind": "spell",
                "verification_artifacts": [
                    "reviews/spell-a.jpg",
                    "reviews/spell-b.jpg",
                ],
                "identity_artifacts": [],
            },
        ],
    }

    package = build_onset_deck_package(source, [0, 100])

    assert package["target_range"] == [0, 100]
    assert package["deck_constraint"] == {"maximum_base_card_slots": 8}
    assert package["attached_images"] == [
        "reviews/onset.jpg",
        "reviews/neighbor.jpg",
        "reviews/spell-a.jpg",
        "reviews/spell-b.jpg",
    ]
