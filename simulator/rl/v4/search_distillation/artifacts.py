"""Deadlines, atomic checkpoints, and artifact provenance."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import hashlib
import os
import subprocess
import time

import torch

from ..model_v4 import RecurrentV4Policy


class DistillationDeadline(RuntimeError):
    """Raised internally when a run reaches its externally supplied deadline."""


def _deadline_reached(deadline_epoch: float | None) -> bool:
    return deadline_epoch is not None and time.time() >= float(deadline_epoch)


def _check_deadline(deadline_epoch: float | None) -> None:
    if _deadline_reached(deadline_epoch):
        raise DistillationDeadline(f"distillation deadline reached at {time.time():.3f}")


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Write a checkpoint atomically so a deadline cannot leave a bad file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_training_state(
    *, checkpoint_path: Path, student: RecurrentV4Policy, optimizer: torch.optim.Optimizer,
    report: Mapping[str, Any], epoch_completed: int,
) -> None:
    atomic_torch_save({
        "state_dict": student.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "report": dict(report),
        "epoch_completed": int(epoch_completed),
    }, checkpoint_path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

