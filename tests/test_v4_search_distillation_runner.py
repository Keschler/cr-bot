"""Regression coverage for cache reuse and completed-run identity."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_v4_search_distillation.py"
spec = importlib.util.spec_from_file_location("search_distillation_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def run_setup(tmp_path, monkeypatch):
    roots = [tmp_path / name for name in ("train", "validation")]
    for index, root in enumerate(roots):
        root.mkdir()
        torch.save([{"entry_id": f"train:hog#{index}:t2",
                     "provenance": {"state_hash": str(index), "match_seed": index}}],
                   root / "frontier_train.pt")
        (root / f"train_hog_{index}_match.json").write_text(json.dumps({
            "config": {"archetype": "hog", "match_index": index, "match_seed": index},
        }))
        (root / f"train_hog_{index}_t2_verdict.json").write_text("{}")
    checkpoint = tmp_path / "anchor.pt"
    checkpoint.write_bytes(b"fixture")
    calls = {"prepare": 0, "train": 0}

    def prepare(*args, **kwargs):
        calls["prepare"] += 1
        return [SimpleNamespace(is_correction=True) for _ in range(2)]

    def train(*args, method, seed, checkpoint_path, **kwargs):
        calls["train"] += 1
        actor_export_dir = Path(kwargs["actor_export_dir"])
        actor_export_prefix = str(kwargs["actor_export_prefix"])
        state = {"weight": torch.tensor([1.0])}
        actor_checkpoints = []
        for epoch in runner.ACTOR_EXPORT_EPOCHS:
            actor_path = actor_export_dir / f"{actor_export_prefix}_epoch{epoch}.pt"
            runner.atomic_torch_save(state, actor_path)
            actor_checkpoints.append({
                "epoch": epoch,
                "path": str(actor_path),
                "sha256": runner.file_sha256(actor_path),
                "format": "plain_state_dict",
            })
        report = {
            "status": "complete", "method": method, "seed": seed,
            "epochs": 20, "actor_export_epochs": list(runner.ACTOR_EXPORT_EPOCHS),
            "actor_checkpoints": actor_checkpoints,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"report": report, "state_dict": state}, checkpoint_path)
        return report

    monkeypatch.setattr(runner, "prepare_records", prepare)
    monkeypatch.setattr(runner, "train_one", train)
    monkeypatch.setattr(runner, "load_frozen_anchor", lambda *a, **kw: object())
    args = ["--dataset-root", str(roots[0]), "--validation-root", str(roots[1]),
            "--checkpoint", str(checkpoint), "--out", str(tmp_path / "out"),
            "--contexts", "0", "--context", "0", "--methods", "hard-none",
            "--seeds", "0", "--device", "cpu"]
    return args, roots, calls


def test_unchanged_run_reuses_cache_and_completed_training(run_setup):
    args, _, calls = run_setup
    assert runner.main(args) == 0
    assert runner.main(args + ["--reuse-cache"]) == 0
    assert calls == {"prepare": 2, "train": 1}
    actor = Path(args[args.index("--out") + 1]) / "actors/context0_hard-none_seed0.pt"
    assert torch.equal(torch.load(actor, weights_only=True)["weight"], torch.tensor([1.0]))


def test_exported_actor_loads_in_shared_v4_evaluator(tmp_path):
    from simulator.rl.v4.frontier import frontier_model_config
    from simulator.rl.v4.model_v4 import RecurrentV4Policy
    from simulator.rl.v4.v4_ppo import load_frozen_anchor

    wrapped = tmp_path / "training.pt"
    exported = tmp_path / "actor.pt"
    torch.save({"state_dict": RecurrentV4Policy(frontier_model_config()).state_dict()}, wrapped)
    digest = runner._export_actor_checkpoint(wrapped, exported)
    assert digest == runner.file_sha256(exported)
    assert load_frozen_anchor(str(exported), frontier_model_config(), device="cpu")


@pytest.mark.parametrize("change", ["correction_limit", "retention_limit", "match", "verdict"])
def test_changed_inputs_reject_cache_and_completed_run(run_setup, change):
    args, roots, calls = run_setup
    assert runner.main(args) == 0
    if change == "correction_limit":
        args += ["--max-corrections", "1"]
    elif change == "retention_limit":
        args += ["--max-retention", "1"]
    else:
        path = next(roots[0].glob(f"*_{change}.json"))
        payload = json.loads(path.read_text())
        payload["changed"] = True
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="prefix cache settings"):
        runner.main(args + ["--reuse-cache"])
    # Automatic cache rebuild must still refuse to skip/resume an old run.
    with pytest.raises(ValueError, match="different data, code, or optimizer settings"):
        runner.main(args)
    assert calls["train"] == 1


def test_failed_method_does_not_prevent_remaining_runs(run_setup, monkeypatch):
    args, _, calls = run_setup
    original = runner.train_one

    def train(*values, method, **kwargs):
        if method == "hard-none":
            raise RuntimeError("fixture failure")
        return original(*values, method=method, **kwargs)

    monkeypatch.setattr(runner, "train_one", train)
    assert runner.main(args + ["--methods", "hard-none,soft-kl"]) == 1
    out = Path(args[args.index("--out") + 1])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["manifest"]["status"] == "failed"
    assert [r["method"] for r in summary["reports"]] == ["soft-kl"]
    failed = json.loads((out / "runs/context0_hard-none_seed0.json").read_text())
    assert failed["status"] == "failed"
    assert "fixture failure" in failed["error"]
    assert calls["train"] == 1


def test_partial_method_stops_remaining_runs(run_setup, monkeypatch):
    args, _, calls = run_setup
    original = runner.train_one

    def train(*values, **kwargs):
        report = original(*values, **kwargs)
        report["status"] = "partial"
        return report

    monkeypatch.setattr(runner, "train_one", train)
    assert runner.main(args + ["--methods", "hard-none,soft-kl"]) == 0
    out = Path(args[args.index("--out") + 1])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["manifest"]["status"] == "partial"
    assert calls["train"] == 1


def test_changed_implementation_rejects_completed_run(run_setup, monkeypatch):
    args, _, calls = run_setup
    assert runner.main(args) == 0
    identity = runner._implementation_identity()
    assert "simulator/rl/v4/search_distillation/preparation.py" in identity
    assert "simulator/rl/v4/search_distillation/training.py" in identity
    assert "simulator/engine/combat.py" in identity
    assert "simulator/public_state_estimator.py" in identity
    assert "simulator/rl/v4/model_v4.py" in identity
    assert "simulator/rl/v4/frontier_eval.py" in identity
    assert any(name.startswith("simulator/rulesets/") for name in identity)
    monkeypatch.setattr(runner, "_implementation_identity", lambda: {**identity, "training.py": "changed"})
    with pytest.raises(ValueError, match="different data, code, or optimizer settings"):
        runner.main(args)
    assert calls["train"] == 1


def test_retention_only_source_overlap_is_rejected(run_setup):
    args, roots, calls = run_setup
    for index, root in enumerate(roots):
        (root / f"train_extra_{index}_match.json").write_text(json.dumps({
            "config": {"archetype": "extra", "match_index": index, "match_seed": 999},
        }))
    with pytest.raises(ValueError, match="source matches share match_seed"):
        runner.main(args)
    assert calls == {"prepare": 0, "train": 0}


def test_retention_source_missing_provenance_is_rejected(run_setup):
    args, roots, calls = run_setup
    (roots[0] / "train_extra_0_match.json").write_text("{}")
    with pytest.raises(ValueError, match="source match requires valid config provenance"):
        runner.main(args)
    assert calls == {"prepare": 0, "train": 0}
