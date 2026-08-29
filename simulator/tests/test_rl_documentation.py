from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rl_documentation_matches_matrix_schema_and_evidence_axes() -> None:
    from rl.evaluation_matrix import (
        EVALUATION_MATRIX_KIND,
        EVALUATION_MATRIX_SCHEMA_VERSION,
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")

    assert (
        f"The matrix report is schema version {EVALUATION_MATRIX_SCHEMA_VERSION}"
        in readme
    )
    assert f"`{EVALUATION_MATRIX_KIND}`" in readme

    assert "`actor_controls_actions` is `true` only for the neural actor" in readme
    assert "`actor_controls_actions=true`" in readme
    assert "The retained six-deck held-out smoke audit" in readme
    assert "self-play matrix uses the fixed prototype player" in readme
    assert "`held_out=false`" in readme
    assert "`held_out_audit.disjointness_verified=true`" in readme
    assert "`deterministic-cycle` archetype is deliberately absent" in readme
    assert "`6 × 6 × 4 = 144`" in readme
    assert "`--training-report`" in readme
    assert "`--no-match-results`" in readme
    assert "`--segment-offset 16`" in readme
    assert "`--direct-public-slot-card-features`" in readme

    assert "`target_play_trace` contains target `PLAY` attempts only" in readme
    assert "`troop_positions_end`" in readme
    assert "`tower_hp_before`, `tower_hp_after`, `tower_hp_end`" in readme
    assert "prototype `--trace-out` contains every decision" in goal
    assert "`troop_positions_end` and `tower_hp_end` are only terminal/cap-time" in goal


def test_rl_documentation_labels_generated_actor_audit_artifacts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")

    from simulator.engine import ENGINE_VERSION

    for document in (readme, goal):
        assert f"engine `{ENGINE_VERSION}`" in document

    assert "generated checkpoint and reports are local artifacts" in readme
    assert "previously recorded generalized actor and reports are generated local" in goal
    assert "The latest retained generalized actor is" not in goal
    assert "generalized-strategic-context-v1.pt" not in readme
    assert "generalized-strategic-context-v1.pt" not in goal
    assert "generalized-strategic-direct-v1.pt" not in readme
    assert "generalized-strategic-direct-v1.pt" not in goal
    assert "generalized-action-focused.pt" not in readme

    assert "8 wins, 0 losses, 0 draws, 0 truncated" in readme
    assert "1 win, 5 losses, 0 draws, 0 truncated" in readme
    assert "1 win, 35 losses, 0 draws, 0 truncated" in readme
    assert "do not establish the mission's any-deck" in goal
    assert "finite `all_wins=true` result is not a universal-win claim" in goal
