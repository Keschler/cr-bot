"""Train and resume deterministic search-distillation ablations."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import copy
import math

import numpy as np
import torch

from ..model_v4 import RecurrentV4Policy
from .artifacts import (
    DistillationDeadline,
    _check_deadline,
    _save_training_state,
    atomic_torch_save,
    file_sha256,
)
from .batching import _stack_records
from .evaluation import evaluate_records
from .losses import compute_batch_loss
from .records import CachedPrefix, DEFAULT_TEMPERATURE


# These are the predeclared development points for the comparison.  Epoch zero
# is the untouched anchor actor; later points are exported immediately after
# their corresponding epoch has completed.
ACTOR_EXPORT_EPOCHS: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 20)


def _plain_state_dict(policy: Any) -> dict[str, torch.Tensor]:
    """Copy a policy state dict to CPU in the plain-actor format."""

    state = policy.state_dict()
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError("actor state dict must contain tensors")
    return {
        str(name): value.detach().cpu().clone()
        for name, value in state.items()
    }


def _export_plain_actor(policy: Any, path: Path) -> str:
    """Write or verify one plain actor checkpoint and return its hash."""

    state = _plain_state_dict(policy)
    if path.is_file():
        try:
            existing = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"cannot read actor export {path}") from exc
        if not isinstance(existing, Mapping) or existing.keys() != state.keys() or any(
            not isinstance(existing[key], torch.Tensor)
            or not torch.equal(existing[key], value)
            for key, value in state.items()
        ):
            raise ValueError(f"actor export differs from current epoch state: {path}")
    else:
        atomic_torch_save(state, path)
    return file_sha256(path)


def _validate_actor_export(record: Mapping[str, Any]) -> None:
    """Verify one recorded export before resuming from its training state."""

    try:
        path = Path(str(record["path"]))
        expected = str(record["sha256"])
        format_name = str(record.get("format", "plain_state_dict"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid actor checkpoint metadata") from exc
    if format_name != "plain_state_dict":
        raise RuntimeError(f"actor export has unsupported format: {path}")
    if not path.is_file() or file_sha256(path) != expected:
        raise RuntimeError(f"actor export hash mismatch: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"cannot read actor export {path}") from exc
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise RuntimeError(f"actor export is not a plain state dict: {path}")


def _training_report(
    *, method: str, seed: int, context: int, epochs: int, batch_size: int,
    learning_rate: float, history: Sequence[Mapping[str, Any]],
    optimizer_steps: int, updates_per_epoch: int,
    correction_examples: int, retention_examples: int,
    status: str, metrics: Mapping[str, Any] | None = None,
    resumed_from_epoch: int = 0,
) -> dict[str, Any]:
    total_examples = correction_examples + retention_examples
    report: dict[str, Any] = {
        "status": status,
        "method": method, "seed": int(seed), "context": int(context),
        "epochs": int(epochs), "batch_size": int(batch_size),
        "learning_rate": float(learning_rate), "history": list(history),
        "optimizer_steps": int(optimizer_steps),
        "updates_per_epoch": int(updates_per_epoch),
        "correction_examples_seen": int(correction_examples),
        "retention_examples_seen": int(retention_examples),
        "correction_exposure": float(correction_examples / max(1, total_examples)),
        "resumed_from_epoch": int(resumed_from_epoch),
    }
    if metrics is not None:
        report["validation"] = dict(metrics)
    return report


def train_one(
    anchor: RecurrentV4Policy,
    train_records: Sequence[CachedPrefix],
    validation_records: Sequence[CachedPrefix],
    *,
    method: str,
    seed: int,
    context: int,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    retention_weight: float = 1.0,
    temperature: float = DEFAULT_TEMPERATURE,
    device: torch.device | str = "cpu",
    checkpoint_path: Path | None = None,
    actor_export_dir: Path | None = None,
    actor_export_prefix: str | None = None,
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    """Train one deterministic method/seed and return report metadata.

    Every method receives the same number of optimizer steps per epoch.  The
    hard-none ablation cycles correction records with replacement when its
    pool is smaller; the report records the resulting correction exposure so
    retention gains are not interpreted without accounting for extra updates.
    A partial checkpoint is committed at each completed epoch and is resumed
    automatically when the same run is invoked again.
    """

    device = torch.device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    student = copy.deepcopy(anchor).to(device)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    student.train()
    anchor_device = copy.deepcopy(anchor).to(device)
    anchor_device.eval()
    for parameter in anchor_device.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(learning_rate))
    active = list(train_records) if method != "hard-none" else [r for r in train_records if r.is_correction]
    if not active:
        raise ValueError(f"no active training records for {method}")
    if int(epochs) <= 0 or int(batch_size) <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if actor_export_dir is not None:
        actor_export_dir = Path(actor_export_dir)
        actor_export_prefix = str(
            actor_export_prefix
            if actor_export_prefix is not None
            else (checkpoint_path.stem if checkpoint_path is not None else "actor")
        )
        if not actor_export_prefix:
            raise ValueError("actor_export_prefix must be non-empty")

    # Equal optimizer budget: all methods use the full train-record cardinality
    # to determine steps, while method-specific pools determine labels.
    updates_per_epoch = max(1, int(math.ceil(len(train_records) / int(batch_size))))
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    correction_examples = 0
    retention_examples = 0
    start_epoch = 0
    resumed_from_epoch = 0
    actor_exports: dict[int, dict[str, Any]] = {}
    if checkpoint_path is not None and checkpoint_path.is_file():
        try:
            saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"cannot read existing distillation checkpoint {checkpoint_path}") from exc
        if not isinstance(saved, Mapping) or not isinstance(saved.get("report"), Mapping):
            raise RuntimeError(f"invalid distillation checkpoint schema: {checkpoint_path}")
        saved_report = saved["report"]
        saved_status = saved_report.get("status")
        if saved_status == "complete":
            raise RuntimeError(f"completed checkpoint already exists; use a new run or resume manifest: {checkpoint_path}")
        if saved_status != "partial":
            raise RuntimeError(f"checkpoint is neither complete nor resumable: {checkpoint_path}")
        if (
            saved_report.get("method") != method
            or int(saved_report.get("seed", -1)) != int(seed)
            or int(saved_report.get("context", -1)) != int(context)
        ):
            raise RuntimeError(f"checkpoint metadata does not match requested run: {checkpoint_path}")
        try:
            student.load_state_dict(saved["state_dict"])
            if saved.get("optimizer_state"):
                optimizer.load_state_dict(saved["optimizer_state"])
            history = [dict(row) for row in saved_report.get("history", [])]
            optimizer_steps = int(saved_report.get("optimizer_steps", 0))
            correction_examples = int(saved_report.get("correction_examples_seen", 0))
            retention_examples = int(saved_report.get("retention_examples_seen", 0))
            start_epoch = int(saved.get("epoch_completed", len(history)))
            raw_exports = saved_report.get("actor_checkpoints", ())
            if not isinstance(raw_exports, (list, tuple)):
                raise RuntimeError(
                    f"invalid actor checkpoint metadata in {checkpoint_path}"
                )
            for raw_export in raw_exports:
                if not isinstance(raw_export, Mapping):
                    raise RuntimeError(
                        f"invalid actor checkpoint metadata in {checkpoint_path}"
                    )
                try:
                    export_epoch = int(raw_export["epoch"])
                    export_path = str(raw_export["path"])
                    export_hash = str(raw_export["sha256"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"invalid actor checkpoint metadata in {checkpoint_path}"
                    ) from exc
                if export_epoch in actor_exports:
                    raise RuntimeError(
                        f"duplicate actor checkpoint epoch {export_epoch} in {checkpoint_path}"
                    )
                actor_exports[export_epoch] = {
                    "epoch": export_epoch,
                    "path": export_path,
                    "sha256": export_hash,
                    "format": str(raw_export.get("format", "plain_state_dict")),
                }
                _validate_actor_export(actor_exports[export_epoch])
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith(
                "invalid actor checkpoint metadata"
            ):
                raise
            raise RuntimeError(f"invalid resumable distillation checkpoint: {checkpoint_path}") from exc
        resumed_from_epoch = start_epoch
        print(f"[search-distill] resumed method={method} seed={seed} context={context} epoch={start_epoch}", flush=True)

    def export_actor(epoch: int, policy: Any) -> None:
        if actor_export_dir is None or epoch not in ACTOR_EXPORT_EPOCHS:
            return
        path = actor_export_dir / f"{actor_export_prefix}_epoch{epoch}.pt"
        digest = _export_plain_actor(policy, path)
        actor_exports[epoch] = {
            "epoch": int(epoch),
            "path": str(path),
            "sha256": digest,
            "format": "plain_state_dict",
        }

    def make_report(status: str, metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = _training_report(
            method=method, seed=seed, context=context, epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate, history=history,
            optimizer_steps=optimizer_steps, updates_per_epoch=updates_per_epoch,
            correction_examples=correction_examples, retention_examples=retention_examples,
            status=status, metrics=metrics, resumed_from_epoch=resumed_from_epoch,
        )
        payload["parameters"] = int(sum(p.numel() for p in student.parameters() if p.requires_grad))
        if checkpoint_path is not None:
            payload["checkpoint"] = str(checkpoint_path)
        payload["actor_export_epochs"] = list(ACTOR_EXPORT_EPOCHS)
        payload["actor_checkpoints"] = [
            dict(actor_exports[epoch]) for epoch in sorted(actor_exports)
        ]
        return payload

    # Export the exact untouched anchor before any optimizer step.  On resume,
    # this also verifies that an existing epoch-zero artifact has not drifted.
    export_actor(0, anchor)
    if start_epoch in ACTOR_EXPORT_EPOCHS and start_epoch > 0:
        # A partial checkpoint is written after each completed epoch, so the
        # in-memory student is the state for this boundary when resuming.
        export_actor(start_epoch, student)
    if actor_export_dir is not None:
        missing_exports = [
            epoch for epoch in ACTOR_EXPORT_EPOCHS
            if epoch <= start_epoch and epoch not in actor_exports
        ]
        if missing_exports:
            raise RuntimeError(
                f"resumable checkpoint lacks actor exports for epochs {missing_exports}"
            )

    # A deadline inside epoch zero must resume from the untouched initial
    # state. Save that boundary before the first optimizer step.
    if checkpoint_path is not None and not checkpoint_path.is_file():
        _save_training_state(
            checkpoint_path=checkpoint_path, student=student, optimizer=optimizer,
            report=make_report("partial"), epoch_completed=0,
        )

    try:
        for epoch in range(start_epoch, int(epochs)):
            _check_deadline(deadline_epoch)
            rng = np.random.default_rng(int(seed) + epoch * 1009)
            order: list[int] = []
            while len(order) < updates_per_epoch * int(batch_size):
                order.extend(int(i) for i in rng.permutation(len(active)).tolist())
            order = order[:updates_per_epoch * int(batch_size)]
            totals: list[float] = []
            epoch_corrections = 0
            epoch_retention = 0
            for start in range(0, len(order), int(batch_size)):
                _check_deadline(deadline_epoch)
                group = [active[int(i)] for i in order[start:start + int(batch_size)]]
                epoch_corrections += sum(int(record.is_correction) for record in group)
                epoch_retention += len(group) - sum(int(record.is_correction) for record in group)
                batch = _stack_records(group, context=context, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss, details = compute_batch_loss(
                    student, anchor_device, batch, group, method=method,
                    temperature=temperature, retention_weight=retention_weight,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {method} loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                _check_deadline(deadline_epoch)
                optimizer.step()
                optimizer_steps += 1
                totals.append(float(loss.detach().cpu()))
            correction_examples += epoch_corrections
            retention_examples += epoch_retention
            history.append({
                "epoch": float(epoch), "loss": float(np.mean(totals)),
                "optimizer_steps": float(updates_per_epoch),
                "correction_examples": float(epoch_corrections),
                "retention_examples": float(epoch_retention),
            })
            export_actor(epoch + 1, student)
            if checkpoint_path is not None:
                partial = make_report("partial")
                _save_training_state(
                    checkpoint_path=checkpoint_path, student=student, optimizer=optimizer,
                    report=partial, epoch_completed=epoch + 1,
                )
            print(f"[search-distill] method={method} seed={seed} epoch={epoch + 1}/{epochs} loss={history[-1]['loss']:.6f} steps={updates_per_epoch}", flush=True)
        metrics = evaluate_records(
            student, anchor_device, validation_records, context=context,
            batch_size=batch_size, device=device, deadline_epoch=deadline_epoch,
        )
    except DistillationDeadline as exc:
        partial = make_report("partial")
        partial["deadline_epoch"] = float(deadline_epoch) if deadline_epoch is not None else None
        partial["deadline_reason"] = str(exc)
        # The latest completed epoch, or the untouched initial state, remains
        # on disk. Discard any partial-epoch updates when resuming.
        return partial

    payload = make_report("complete", metrics)
    if checkpoint_path is not None:
        _save_training_state(
            checkpoint_path=checkpoint_path, student=student, optimizer=optimizer,
            report=payload, epoch_completed=int(epochs),
        )
    return payload
