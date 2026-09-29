"""Regression coverage for scheduled plain distillation actor exports."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib

import torch

from simulator.rl.v4.search_distillation.training import (
    ACTOR_EXPORT_EPOCHS,
    train_one,
)


def test_train_one_exports_scheduled_plain_actors_with_hashes(tmp_path: Path) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(11)
        anchor = torch.nn.Linear(1, 1, bias=False)
        initial = {name: value.detach().clone() for name, value in anchor.state_dict().items()}
        records = [SimpleNamespace(is_correction=True)]
        export_dir = tmp_path / "actors"

        def loss(student, *_args, **_kwargs):
            return student.weight.square().sum(), {}

        with patch(
            "simulator.rl.v4.search_distillation.training._stack_records",
            return_value={},
        ), patch(
            "simulator.rl.v4.search_distillation.training.compute_batch_loss",
            side_effect=loss,
        ), patch(
            "simulator.rl.v4.search_distillation.training.evaluate_records",
            return_value={"n": 0},
        ):
            report = train_one(
                anchor,
                records,
                [],
                method="hard-none",
                seed=3,
                context=0,
                epochs=20,
                batch_size=1,
                actor_export_dir=export_dir,
                actor_export_prefix="context0_hard-none_seed3",
            )

        assert report["actor_export_epochs"] == list(ACTOR_EXPORT_EPOCHS)
        exports = {int(row["epoch"]): row for row in report["actor_checkpoints"]}
        assert sorted(exports) == list(ACTOR_EXPORT_EPOCHS)
        for epoch in ACTOR_EXPORT_EPOCHS:
            row = exports[epoch]
            path = Path(row["path"])
            assert path.is_file()
            payload = torch.load(path, map_location="cpu", weights_only=True)
            assert isinstance(payload, dict)
            assert all(isinstance(value, torch.Tensor) for value in payload.values())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        epoch_zero = torch.load(
            exports[0]["path"], map_location="cpu", weights_only=True
        )
        for name, value in initial.items():
            torch.testing.assert_close(epoch_zero[name], value, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
