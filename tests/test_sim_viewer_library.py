"""Local viewer discovery and metric semantics without loading any models."""

import json
from pathlib import Path

import pytest

from sim_viewer.library import ArtifactLibrary, ViewerPaths, normalise_series


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_library_excludes_dataset_tensors_and_keeps_live_partial_recording(tmp_path):
    root = tmp_path / "v4"
    root.mkdir()
    (root / "actor.pt").touch()
    (root / "frontier_train.pt").touch()
    (root / "critic.pt").touch()
    write_json(root / "train_match.json", {"decisions": []})
    recording = root / "sample.jsonl"
    recording.write_text(json.dumps({"type": "header", "schema_version": 1, "source": "training"}) + '\n{"type":"frame",')
    paths = ViewerPaths((root,), root / "viewer")
    library = ArtifactLibrary(paths)
    listing = library.listing()
    assert {row["source"] for row in listing["matches"]} == {"frontier", "training"}
    assert next(row for row in listing["matches"] if row["source"] == "training")["active"]
    assert [Path(row["path"]).name for row in library.checkpoints()] == ["actor.pt"]


def test_path_boundaries_resolve_symlinks_before_access(tmp_path):
    root = tmp_path / "v4"
    root.mkdir()
    secret = tmp_path / "outside.json"
    secret.write_text("{}")
    (root / "linked.json").symlink_to(secret)
    paths = ViewerPaths((root,), root / "viewer")
    for value in (secret, root / ".." / "outside.json", root / "linked.json"):
        with pytest.raises(ValueError, match="configured"):
            paths.resolve(value, suffixes=(".json",))
    assert not ArtifactLibrary(paths).listing()["matches"]


def test_censored_evaluation_never_becomes_zero_percent_win_rate():
    raw = {"points": [
        {"update": 0, "evaluation": {"full_match": {"candidate": {"completed": 0, "wins": 0, "truncated": 3, "score_mean_completed": 0}}}},
        {"update": 2, "evaluation": {"full_match": {"candidate": {"completed": 4, "wins": 3, "truncated": 1, "score_mean_completed": .875}}},
         "updates_since_previous_point": {"1": {"update": {"policy_loss": -.2, "entropy": 1.5, "returns_mean": .4}}}},
    ]}
    series = normalise_series(raw)
    assert "win_rate" not in series[0]
    assert series[0]["truncated"] == 3
    assert series[1] == {"step": 1.0, "policy_loss": -.2, "entropy": 1.5, "reward": .4}
    assert series[2]["win_rate"] == .75
    assert series[2]["score"] == .875


def test_live_progress_includes_updates_without_finished_evaluation(tmp_path):
    root = tmp_path / "v4"
    write_json(root / "run" / "progress.json", {"source": "training", "status": "running", "phase": "update", "updates": {"3": {"update": {"policy_loss": .8, "value_loss": .5, "approx_kl": .02}}}, "points": []})
    library = ArtifactLibrary(ViewerPaths((root,), root / "viewer"))
    run = library.run(library.listing()["runs"][0]["id"])
    assert run["series"] == [{"step": 3., "policy_loss": .8, "value_loss": .5, "kl": .02}]
    assert run["summary"]["state"] == "running"
    assert "win_rate" not in run["summary"]


def test_distillation_arms_are_separately_comparable_and_manifest_history_loads(tmp_path):
    root = tmp_path / "v4"
    for method in ("soft-kl", "hard-replay"):
        write_json(root / "experiment" / f"context16_{method}_seed0.json", {"history": [{"epoch": 0, "loss": 2.3}, {"epoch": 1, "loss": 1.2}]})
    write_json(root / "bootstrap" / "manifest.json", {"history": [{"step": 0, "loss": 3.}]})
    library = ArtifactLibrary(ViewerPaths((root,), root / "viewer"))
    runs = library.listing()["runs"]
    assert len(runs) == 3
    for row in runs:
        assert library.run(row["id"])["series"]


def test_live_distillation_arm_identity_survives_sequential_methods(tmp_path):
    root = tmp_path / "v4"
    run = root / "experiment"
    first = "context16_soft-kl_seed0"
    second = "context16_hard-replay_seed0"
    for label in (first, second):
        write_json(run / f"{label}.json", {"history": [{"epoch": 0, "loss": 2.}]})
    progress = run / "viewer" / "progress.json"
    write_json(progress, {"source": "distillation", "label": first, "output_dir": str(run), "history": [{"epoch": 1, "loss": 1.}]})
    library = ArtifactLibrary(ViewerPaths((root,), root / "viewer"))
    before = {row["name"]: row for row in library.listing()["runs"]}
    first_id = before[f"experiment/{first}"]["id"]
    assert len(before) == 2
    assert library.run(first_id)["series"][-1]["loss"] == 1.
    write_json(progress, {"source": "distillation", "label": second, "output_dir": str(run), "history": [{"epoch": 2, "loss": .3}]})
    library.refresh(force=True)
    assert library.run(first_id)["name"] == f"experiment/{first}"
    assert library.run(first_id)["series"][-1]["loss"] == 2.


def test_checkpoint_metrics_match_recorded_update_and_distillation_epoch(tmp_path):
    root = tmp_path / "v4"
    write_json(root / "ppo" / "learning_curve.json", {"points": [{"update": 2, "evaluation": {"full_match": {"candidate": {"completed": 4, "wins": 3}}}}]})
    (root / "ppo" / "checkpoint_update_000002.pt").touch()
    write_json(root / "distill" / "context16_soft-kl_seed0.json", {"method": "soft-kl", "history": [{"epoch": 0, "loss": 2.}, {"epoch": 1, "loss": 1.}]})
    actors = root / "distill" / "actors"
    actors.mkdir()
    for epoch in (0, 1, 2):
        (actors / f"context16_soft-kl_seed0_epoch_{epoch}.pt").touch()
    (actors / "context16_hard-replay_seed0_epoch_1.pt").touch()
    unrelated = root / "ppo" / "other-run"
    unrelated.mkdir()
    (unrelated / "checkpoint_update_000002.pt").touch()
    library = ArtifactLibrary(ViewerPaths((root,), root / "viewer"))
    runs = [library.run(row["id"]) for row in library.listing()["runs"]]
    ppo = next(run for run in runs if run["name"] == "ppo")
    assert len(ppo["checkpoints"]) == 1
    assert ppo["checkpoints"][0]["metrics"]["win_rate"] == .75
    distill = next(run for run in runs if run["name"].startswith("distill/"))
    assert len(distill["checkpoints"]) == 3
    metrics = {cp["step"]: cp.get("metrics") for cp in distill["checkpoints"]}
    assert metrics[0] is None
    assert metrics[1] == {"loss": 2.}
    assert metrics[2] == {"loss": 1.}
