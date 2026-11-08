"""Outcome browsing must use completed results and filter before pagination."""
import json
import os
from pathlib import Path

import pytest

from sim_viewer.library import ArtifactLibrary, ViewerPaths


def recording(root, name, *, winner=0, terminal=True, reason="regulation_crowns",
              ended=True, declared=None, source="evaluation", modified=100, partial=False):
    metadata = {}
    if declared:
        metadata = {"evaluation": {"result": declared},
                    "native_record": {"terminated": terminal, "truncated": False, "cap_reached": False}}
    rows = [{"type": "header", "schema_version": 1, "source": source, "metadata": metadata},
            {"type": "frame", "index": 0, "terminal": terminal}]
    if ended:
        rows.append({"type": "end", "reason": reason, "winner": winner})
    path = root / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + ('{"type":"end"' if partial else ''))
    os.utime(path, (modified, modified))
    return path


def library(root):
    return ArtifactLibrary(ViewerPaths((root,), root / "viewer"))


def test_completed_results_use_winner_not_filename_and_censored_stays_unknown(tmp_path):
    recording(tmp_path, "win_in_filename_but_loss", winner=1, declared="loss")
    recording(tmp_path, "actual_win", winner=0, declared="win")
    recording(tmp_path, "terminal_draw", winner=None, declared="draw")
    recording(tmp_path, "unfinished_win", ended=False, declared="win")
    recording(tmp_path, "canceled_win", reason="stopped", declared="win")
    recording(tmp_path, "capped", winner=None, terminal=False, reason="episode_cap")
    recording(tmp_path, "unproven", winner=None, terminal=False)
    recording(tmp_path, "contradiction", winner=1, declared="win")
    recording(tmp_path, "partial_footer", ended=False, partial=True, declared="win")
    listing = library(tmp_path).listing(limit=100)
    by_name = {Path(row["path"]).stem: row for row in listing["matches"]}
    assert by_name["win_in_filename_but_loss"]["outcome"] == "loss"
    assert by_name["actual_win"]["outcome"] == "win"
    assert by_name["terminal_draw"]["outcome"] == "draw"
    for name in ("unfinished_win", "canceled_win", "capped", "unproven", "contradiction", "partial_footer"):
        assert by_name[name]["outcome"] is None
    assert by_name["partial_footer"]["unfinished"]
    assert listing["outcome_counts"] == {"win": 1, "loss": 1, "draw": 1, "unknown": 6}


def test_result_is_available_when_last_terminal_frame_exceeds_tail_window(tmp_path):
    path = recording(tmp_path, "large_win", declared="win")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["large_snapshot"] = "x" * 30_000
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    row = library(tmp_path).listing()["matches"][0]
    assert row["outcome"] == "win" and not row["unfinished"]


def test_filters_compose_and_run_before_limit_with_counts_for_current_source(tmp_path):
    recording(tmp_path, "xbow_old_win", modified=10)
    recording(tmp_path, "xbow_loss", winner=1, modified=20)
    recording(tmp_path, "xbow_exhibition_win", source="exhibition", modified=30)
    recording(tmp_path, "bait_win", modified=40)
    viewer = library(tmp_path)
    result = viewer.listing("xbow", limit=1, source="evaluation", outcome="win")
    assert len(result["matches"]) == result["match_count"] == 1
    assert Path(result["matches"][0]["path"]).stem == "xbow_old_win"
    assert result["outcome_counts"] == {"win": 1, "loss": 1, "draw": 0, "unknown": 0}


def test_sort_groups_results_and_uses_newest_order_within_group(tmp_path):
    recording(tmp_path, "win_old", modified=10)
    recording(tmp_path, "unknown", ended=False, modified=50)
    recording(tmp_path, "win_new", modified=40)
    recording(tmp_path, "loss", winner=1, modified=30)
    recording(tmp_path, "draw", winner=None, modified=20)
    viewer = library(tmp_path)
    names = lambda order: [Path(row["path"]).stem for row in viewer.listing(sort=order)["matches"]]
    assert names("wins-first") == ["win_new", "win_old", "loss", "draw", "unknown"]
    assert names("losses-first") == ["loss", "win_new", "win_old", "draw", "unknown"]
    assert names("newest") == ["unknown", "win_new", "loss", "draw", "win_old"]
    assert names("oldest") == ["win_old", "draw", "loss", "win_new", "unknown"]
    assert viewer.listing(outcome="unknown")["match_count"] == 1


@pytest.mark.parametrize("options", [{"outcome": "typo"}, {"sort": "typo"}])
def test_unsupported_filters_are_rejected(tmp_path, options):
    with pytest.raises(ValueError, match="Unsupported"):
        library(tmp_path).listing(**options)


def test_api_handler_exposes_source_and_outcome_filters_without_model_load(tmp_path):
    from sim_viewer.app import create_app
    recording(tmp_path, "xbow_win", modified=10)
    recording(tmp_path, "xbow_loss", winner=1, modified=20)
    app = create_app(paths=ViewerPaths((tmp_path,), tmp_path / "viewer"))
    try:
        handler = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/api/library")
        result = handler(query="xbow", limit=1, source="evaluation", outcome="win", sort="losses-first")
        assert result["match_count"] == 1 and result["matches"][0]["outcome"] == "win"
    finally:
        app.state.matches.close()
