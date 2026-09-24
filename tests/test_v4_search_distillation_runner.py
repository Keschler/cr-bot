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
        (root / f"train_hog_{index}_match.json").write_text("{}")
        (root / f"train_hog_{index}_t2_verdict.json").write_text("{}")
    checkpoint = tmp_path / "anchor.pt"
    checkpoint.write_bytes(b"fixture")
    calls = {"prepare": 0, "train": 0}

    def prepare(*args, **kwargs):
        calls["prepare"] += 1
        return [SimpleNamespace(is_correction=True) for _ in range(2)]

    def train(*args, method, seed, checkpoint_path, **kwargs):
        calls["train"] += 1
        report = {"status": "complete", "method": method, "seed": seed}
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"report": report}, checkpoint_path)
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
        path.write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError, match="prefix cache settings"):
        runner.main(args + ["--reuse-cache"])
    # Automatic cache rebuild must still refuse to skip/resume an old run.
    with pytest.raises(ValueError, match="different data, code, or optimizer settings"):
        runner.main(args)
    assert calls["train"] == 1
