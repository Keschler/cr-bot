"""Runner metadata coverage for scheduled distillation actor artifacts."""

import importlib.util
import json
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_v4_search_distillation.py"
spec = importlib.util.spec_from_file_location("v4_actor_manifest_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def test_run_manifest_records_scheduled_actor_exports(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out"
    run_spec = {
        "label": "context0_soft-kl_seed0",
        "method": "soft-kl",
        "seed": 0,
        "context": 0,
        "epochs": 20,
        "batch_size": 1,
        "learning_rate": 1e-5,
        "retention_weight": 1.0,
        "temperature": 0.25,
    }
    state = {"weight": torch.tensor([1.0])}

    def train(*_args, checkpoint_path, actor_export_dir, actor_export_prefix, **_kwargs):
        assert actor_export_dir == out / "actors"
        assert actor_export_prefix == run_spec["label"]
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        actor_export_dir.mkdir(parents=True, exist_ok=True)
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
            "status": "complete",
            "method": run_spec["method"],
            "seed": run_spec["seed"],
            "epochs": run_spec["epochs"],
            "actor_export_epochs": list(runner.ACTOR_EXPORT_EPOCHS),
            "actor_checkpoints": actor_checkpoints,
        }
        torch.save({"report": report, "state_dict": state}, checkpoint_path)
        return report

    monkeypatch.setattr(runner, "train_one", train)
    report = runner._run_experiment(
        run_spec,
        out=out,
        anchor=object(),
        train_records=[],
        validation_records=[],
        device=torch.device("cpu"),
        effective_deadline=None,
    )

    assert [row["epoch"] for row in report["actor_checkpoints"]] == list(
        runner.ACTOR_EXPORT_EPOCHS
    )
    manifest = json.loads(
        (out / "runs" / "context0_soft-kl_seed0.json").read_text()
    )
    assert manifest["actor_export_epochs"] == list(runner.ACTOR_EXPORT_EPOCHS)
    assert manifest["actor_checkpoints"] == report["actor_checkpoints"]
    assert manifest["actor_sha256"] == runner.file_sha256(
        out / "actors" / "context0_soft-kl_seed0.pt"
    )
    assert runner._actor_exports_valid(report, expected_epochs=run_spec["epochs"])


def test_completed_run_requires_exports_through_configured_epoch(tmp_path: Path) -> None:
    assert not runner._actor_exports_valid({}, expected_epochs=20)

    actor_path = tmp_path / "actor_epoch0.pt"
    runner.atomic_torch_save({"weight": torch.tensor([1.0])}, actor_path)
    report = {
        "epochs": 0,
        "actor_checkpoints": [{
            "epoch": 0,
            "path": str(actor_path),
            "sha256": runner.file_sha256(actor_path),
            "format": "plain_state_dict",
        }],
    }
    assert not runner._actor_exports_valid(report, expected_epochs=20)
