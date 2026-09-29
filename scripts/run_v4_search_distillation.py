"""Run the V4 search-policy / recurrent-context ablation.

The input directory is a frontier-generation output containing
``frontier_train.pt``, ``frontier_sealed.pt``, source ``*_match.json`` files,
and source ``*_verdict.json`` files.  Prefixes are replayed once per source
match and cached before any model training starts.

Example smoke run::

    PYTHONPATH=.:src outputs/venv/bin/python -u scripts/run_v4_search_distillation.py \
      --dataset-root outputs/v4/runpod_frontier_dataset_20260921 \
      --validation-root /path/to/disjoint-confirmed-validation \
      --checkpoint outputs/v4/bootstrap_seed0/actor.pt \
      --out outputs/v4/search_distillation_smoke \
      --context 32 --methods hard-none,soft-kl --seeds 0 --epochs 1 \
      --batch-size 4 --max-retention 16
"""

from __future__ import annotations

import argparse
import json
from itertools import product
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from simulator.rl.v4.frontier import frontier_model_config
from simulator.rl.v4.search_distillation.artifacts import atomic_torch_save
from simulator.rl.v4.search_distillation import (
    DEFAULT_CONTEXT,
    DEFAULT_MIN_LCB,
    SEARCH_DISTILLATION_VERSION,
    file_sha256,
    git_revision,
    load_frontier_entries,
    prepare_records,
    train_one,
)
from simulator.rl.v4.search_distillation.preparation import parse_entry_id, match_json_path
from simulator.rl.v4.search_distillation.training import ACTOR_EXPORT_EPOCHS
from simulator.rl.v4.v4_ppo import load_frozen_anchor


METHODS = ("hard-none", "hard-replay", "soft-replay", "soft-kl")


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _csv(value: str, *, allowed: tuple[str, ...] | None = None) -> list[str]:
    result = [part.strip() for part in str(value).split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("must contain at least one value")
    if allowed is not None:
        invalid = [part for part in result if part not in allowed]
        if invalid:
            raise argparse.ArgumentTypeError(f"invalid values {invalid}; allowed={allowed}")
    return result


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--validation-root", type=Path, required=True,
        help="disjoint development pool; its frontier_train.pt is used for validation",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=_nonnegative, default=DEFAULT_CONTEXT,
                        help="maximum cached burn-in context (default: 32)")
    parser.add_argument("--contexts", default="0,32",
                        help="context lengths to train/evaluate (default: 0,32)")
    parser.add_argument("--methods", default=",".join(METHODS),
                        help="comma-separated ablations")
    parser.add_argument("--seeds", default="0,1,2",
                        help="comma-separated random seeds")
    parser.add_argument("--epochs", type=_positive, default=20)
    parser.add_argument("--batch-size", type=_positive, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--min-lcb", type=float, default=DEFAULT_MIN_LCB)
    parser.add_argument("--max-corrections", type=_nonnegative, default=0,
                        help="limit corrections per split; 0 keeps all")
    parser.add_argument("--max-retention", type=_nonnegative, default=3000,
                        help="ordinary replay states per split")
    parser.add_argument("--validation-retention", type=_nonnegative, default=750)
    parser.add_argument("--min-train-corrections", type=_nonnegative, default=1,
                        help="minimum independently confirmed training corrections")
    parser.add_argument("--min-validation-corrections", type=_nonnegative, default=1,
                        help="minimum independently confirmed validation corrections")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument(
        "--allow-discovery-confidence", action="store_true",
        help="allow legacy verdicts without independent confirmation replicas (exploratory only)",
    )
    parser.add_argument(
        "--outcome-replicas", type=_nonnegative, default=4,
        help="paired outcome replicas per selected validation action (default: 4)",
    )
    parser.add_argument(
        "--outcome-horizon", type=_positive, default=128,
        help="paired outcome continuation horizon (default: 128)",
    )
    parser.add_argument(
        "--deadline-epoch", type=float, default=None,
        help="absolute Unix deadline; defaults to V4_DEADLINE_EPOCH when set",
    )
    parser.add_argument(
        "--deadline-reserve-seconds", type=float,
        default=float(os.environ.get("V4_DEADLINE_RESERVE_SECONDS", "120")),
        help="stop new work this many seconds before the absolute deadline",
    )
    return parser


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(value)


def _trim_corrections(records: list[Any], limit: int) -> list[Any]:
    corrections = [record for record in records if record.is_correction]
    retention = [record for record in records if not record.is_correction]
    if limit > 0:
        corrections = corrections[:limit]
    return corrections + retention


def _check_disjoint_entries(train_entries: list[Any], validation_entries: list[Any]) -> None:
    """Keep source matches and states out of both optimizer and validation data."""
    for field in ("state_hash", "match_seed"):
        if any(entry.get("provenance", {}).get(field) is None for entry in [*train_entries, *validation_entries]):
            raise ValueError(f"train/validation entries require provenance.{field}")
        train_values = {entry.get("provenance", {}).get(field) for entry in train_entries}
        validation_values = {entry.get("provenance", {}).get(field) for entry in validation_entries}
        train_values.discard(None)
        validation_values.discard(None)
        overlap = train_values & validation_values
        if overlap:
            raise ValueError(f"train and validation share {field}: {len(overlap)} overlapping values")


def _dataset_identity(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    return {
        "root": str(root),
        "frontier_sha256": file_sha256(root / "frontier_train.pt"),
        "sources": {
            path.name: file_sha256(path)
            for path in sorted(set(root.glob("*_match.json")) | set(root.glob("*_verdict.json")))
        },
    }


def _implementation_identity() -> dict[str, str]:
    """Fingerprint simulation, observation, training and evaluation dependencies.

    Use a conservative source closure: replay depends on engine and ruleset
    behavior as well as distillation itself. Paths are repository-relative so
    identically checked out code has the same identity on another machine.
    """
    paths = set((REPO_ROOT / "simulator").rglob("*.py"))
    paths.update((REPO_ROOT / "simulator" / "rulesets").rglob("*.json"))
    paths.update((REPO_ROOT / "src" / "cr_bot").rglob("*.py"))
    return {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in sorted(paths)
        if (
            "__pycache__" not in path.parts
            and "tests" not in path.parts
            and "outputs" not in path.parts
            and "dist" not in path.parts
        )
    }


def _source_match_seeds(root: Path, entries: list[Any], retention_count: int) -> set[int]:
    """Account for every match eligible for correction or retention replay."""
    keys = [parse_entry_id(str(entry["entry_id"]))[:2] for entry in entries]
    paths = {match_json_path(root, split, match_id) for split, match_id in keys}
    splits = {split for split, _ in keys}
    if retention_count > 0:
        paths.update(
            path for path in root.glob("*_match.json")
            if not splits or path.name.split("_", 1)[0] in splits
        )
    seeds = set()
    for path in sorted(paths):
        try:
            match = json.loads(path.read_text(encoding="utf-8"))
            config = match["config"]
            seed = config["match_seed"]
            if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("match_seed must be an integer")
            if config.get("archetype") is None or config.get("match_index") is None:
                raise ValueError("missing archetype or match_index")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"source match requires valid config provenance: {path}") from exc
        seeds.add(seed)
    return seeds


def _check_disjoint_sources(args: argparse.Namespace, train_entries: list[Any], validation_entries: list[Any]) -> None:
    _check_disjoint_entries(train_entries, validation_entries)
    train = _source_match_seeds(args.dataset_root, train_entries, args.max_retention)
    validation = _source_match_seeds(args.validation_root, validation_entries, args.validation_retention)
    overlap = train & validation
    if overlap:
        raise ValueError(f"train and validation source matches share match_seed: {len(overlap)} overlapping values")


def _cache_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Identity of both the replay inputs and the selected training examples."""
    return {
        "version": SEARCH_DISTILLATION_VERSION,
        "context": int(args.context),
        "min_lcb": float(args.min_lcb),
        "max_corrections": int(args.max_corrections),
        "max_retention": int(args.max_retention),
        "validation_retention": int(args.validation_retention),
        "require_independent": not bool(args.allow_discovery_confidence),
        "train": _dataset_identity(args.dataset_root),
        "validation": _dataset_identity(args.validation_root),
        "implementation_sha256": _implementation_identity(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _deadline_from_args(args: argparse.Namespace) -> float | None:
    value = args.deadline_epoch
    if value is None:
        raw = os.environ.get("V4_DEADLINE_EPOCH")
        value = float(raw) if raw else None
    if value is not None and (not float(value) > 0.0):
        raise ValueError("deadline epoch must be positive")
    if args.deadline_reserve_seconds < 0.0:
        raise ValueError("deadline reserve must be non-negative")
    return float(value) if value is not None else None


def _actor_exports_valid(
    report: Mapping[str, Any], *, expected_epochs: int | None = None,
) -> bool:
    """Check recorded scheduled actors before treating a run as complete."""

    if not isinstance(expected_epochs, int) or isinstance(expected_epochs, bool):
        return False
    if expected_epochs < 0:
        return False
    raw_exports = report.get("actor_checkpoints")
    if not isinstance(raw_exports, (list, tuple)):
        return False
    by_epoch: dict[int, Mapping[str, Any]] = {}
    for raw in raw_exports:
        if not isinstance(raw, Mapping):
            return False
        try:
            epoch = int(raw["epoch"])
            path = Path(str(raw["path"]))
            expected = str(raw["sha256"])
        except (KeyError, TypeError, ValueError):
            return False
        if str(raw.get("format", "plain_state_dict")) != "plain_state_dict":
            return False
        if epoch in by_epoch or not path.is_file():
            return False
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            digest = file_sha256(path)
        except Exception:
            return False
        if (
            not isinstance(payload, Mapping)
            or not payload
            or not all(isinstance(value, torch.Tensor) for value in payload.values())
            or digest != expected
        ):
            return False
        by_epoch[epoch] = raw
    required = {epoch for epoch in ACTOR_EXPORT_EPOCHS if epoch <= expected_epochs}
    return required.issubset(by_epoch)


def _run_is_complete(
    run_manifest: Mapping[str, Any], report_path: Path, checkpoint_path: Path,
    label: str, *, expected_epochs: int | None = None,
) -> bool:
    if run_manifest.get("status") != "complete" or not report_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, ValueError, RuntimeError, EOFError):
        return False
    saved_report = saved.get("report", {}) if isinstance(saved, dict) else {}
    return bool(
        report.get("status") == "complete"
        and report.get("label", label) == label
        and saved_report.get("status") == "complete"
        and saved_report.get("method") == report.get("method")
        and int(saved_report.get("seed", -1)) == int(report.get("seed", -2))
        and _actor_exports_valid(report, expected_epochs=expected_epochs)
    )


def _export_actor_checkpoint(checkpoint_path: Path, actor_path: Path) -> str:
    """Export the plain V4 actor weights used by shared held-out evaluation."""

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = saved.get("state_dict") if isinstance(saved, Mapping) else None
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError(f"distillation checkpoint has no actor state dict: {checkpoint_path}")
    if actor_path.exists():
        existing = torch.load(actor_path, map_location="cpu", weights_only=True)
        if not isinstance(existing, Mapping) or existing.keys() != state.keys() or any(
            not torch.equal(existing[key], value) for key, value in state.items()
        ):
            raise ValueError(f"actor export differs from training checkpoint: {actor_path}")
    else:
        atomic_torch_save(state, actor_path)
    return file_sha256(actor_path)


def _load_or_prepare_records(
    args: argparse.Namespace, data_identity: Mapping[str, Any],
    effective_deadline: float | None,
) -> tuple[list[Any], list[Any]] | None:
    """Reuse matching prefixes or replay disjoint inputs within the budget."""
    dataset_root = args.dataset_root.expanduser().resolve()
    validation_root = args.validation_root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    cache_path = out / "prefix_cache.pt"
    cached = None
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache_matches = cached.get("identity") == data_identity
        if not cache_matches and args.reuse_cache:
            raise ValueError("prefix cache settings do not match this run")
        if cache_matches:
            train_records = list(cached["train_records"])
            validation_records = list(cached["validation_records"])
            print(f"[search-distill] reused {cache_path}", flush=True)
        else:
            cached = None
    if cached is None:
        train_path = dataset_root / "frontier_train.pt"
        validation_path = validation_root / "frontier_train.pt"
        if not train_path.is_file() or not validation_path.is_file():
            raise FileNotFoundError(
                "dataset-root and validation-root each require frontier_train.pt"
            )
        train_entries = load_frontier_entries(train_path)
        validation_entries = load_frontier_entries(validation_path)
        _check_disjoint_sources(args, train_entries, validation_entries)
        print(f"[search-distill] replaying prefixes: train entries={len(train_entries)} validation entries={len(validation_entries)} context={args.context}", flush=True)
        if effective_deadline is not None and time.time() >= effective_deadline:
            _atomic_json(out / "run_state.json", {
                "status": "partial", "reason": "deadline_before_prefix_replay",
                "deadline_epoch": _deadline_from_args(args),
            })
            return None
        train_records = prepare_records(
            dataset_root, train_entries, context=int(args.context),
            min_lcb=float(args.min_lcb), retention_count=int(args.max_retention), seed=101,
            require_independent=not bool(args.allow_discovery_confidence),
            deadline_epoch=effective_deadline,
        )
        validation_records = prepare_records(
            validation_root, validation_entries, context=int(args.context),
            min_lcb=float(args.min_lcb), retention_count=int(args.validation_retention), seed=202,
            require_independent=not bool(args.allow_discovery_confidence),
            deadline_epoch=effective_deadline,
        )
        train_records = _trim_corrections(train_records, int(args.max_corrections))
        validation_records = _trim_corrections(validation_records, int(args.max_corrections))
        atomic_torch_save({
            "identity": data_identity, "train_records": train_records,
            "validation_records": validation_records,
        }, cache_path)
        print(f"[search-distill] prefix cache -> {cache_path}", flush=True)

    return train_records, validation_records


def _run_experiment(
    run_spec: dict[str, Any], *, out: Path, anchor: Any,
    train_records: list[Any], validation_records: list[Any],
    device: torch.device, effective_deadline: float | None,
) -> dict[str, Any] | None:
    """Resume one matching run, recording success, interruption, or failure."""
    label = run_spec["label"]
    method, seed, context = (run_spec[name] for name in ("method", "seed", "context"))
    report_path = out / f"{label}.json"
    checkpoint_path = out / "checkpoints" / f"{label}.pt"
    actor_path = out / "actors" / f"{label}.pt"
    run_manifest_path = out / "runs" / f"{label}.json"
    existing = {}
    if run_manifest_path.is_file():
        try:
            existing = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    if existing and existing.get("spec") != run_spec:
        raise ValueError(
            f"existing {label} uses different data, code, or optimizer settings; "
            "choose a new --out directory"
        )
    if existing.get("spec") == run_spec and _run_is_complete(
        existing, report_path, checkpoint_path, label,
        expected_epochs=int(run_spec["epochs"]),
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["actor_checkpoint"] = str(actor_path)
        report["actor_sha256"] = _export_actor_checkpoint(checkpoint_path, actor_path)
        _atomic_json(report_path, report)
        _atomic_json(run_manifest_path, {
            **existing,
            "status": "complete",
            "actor_export_epochs": list(ACTOR_EXPORT_EPOCHS),
            "actor_checkpoints": report.get("actor_checkpoints", []),
            "actor_checkpoint": str(actor_path),
            "actor_sha256": report["actor_sha256"],
        })
        print(f"[search-distill] skip completed {label}", flush=True)
        return report
    print(f"[search-distill] start {label} device={device}", flush=True)
    _atomic_json(run_manifest_path, {
        "status": "running", "spec": run_spec,
        "started_at": time.time(), "report": str(report_path),
        "checkpoint": str(checkpoint_path),
        "actor_export_epochs": list(ACTOR_EXPORT_EPOCHS),
    })
    try:
        report = train_one(
            anchor, train_records, validation_records,
            method=method, seed=seed, context=context,
            epochs=run_spec["epochs"], batch_size=run_spec["batch_size"],
            learning_rate=run_spec["learning_rate"], retention_weight=run_spec["retention_weight"],
            temperature=run_spec["temperature"], device=device,
            checkpoint_path=checkpoint_path,
            actor_export_dir=out / "actors",
            actor_export_prefix=label,
            deadline_epoch=effective_deadline,
        )
        report["label"] = label
        report_status = str(report.get("status", "complete"))
        if report_status == "complete":
            report["actor_checkpoint"] = str(actor_path)
            report["actor_sha256"] = _export_actor_checkpoint(checkpoint_path, actor_path)
    except Exception as exc:
        _atomic_json(run_manifest_path, {
            "status": "failed", "spec": run_spec,
            "error": f"{type(exc).__name__}: {exc}",
            "finished_at": time.time(), "report": str(report_path),
            "checkpoint": str(checkpoint_path),
        })
        print(f"[search-distill] failed {label}: {type(exc).__name__}: {exc}", flush=True)
        return None
    _atomic_json(report_path, report)
    _atomic_json(run_manifest_path, {
        "status": report_status, "spec": run_spec,
        "finished_at": time.time(), "report": str(report_path),
        "checkpoint": str(checkpoint_path),
        "optimizer_steps": int(report.get("optimizer_steps", 0)),
        "actor_export_epochs": list(ACTOR_EXPORT_EPOCHS),
        "actor_checkpoints": report.get("actor_checkpoints", []),
        "actor_checkpoint": report.get("actor_checkpoint"),
        "actor_sha256": report.get("actor_sha256"),
    })
    return report


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    deadline_epoch = _deadline_from_args(args)
    deadline_reserve = float(args.deadline_reserve_seconds)
    effective_deadline = (
        deadline_epoch - deadline_reserve if deadline_epoch is not None else None
    )

    def budget_available() -> bool:
        return effective_deadline is None or time.time() < effective_deadline

    dataset_root = args.dataset_root.expanduser().resolve()
    validation_root = args.validation_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not validation_root.is_dir():
        raise FileNotFoundError(validation_root)
    if validation_root == dataset_root:
        raise ValueError("--validation-root must be disjoint from --dataset-root")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.learning_rate <= 0.0 or args.retention_weight < 0.0 or args.temperature <= 0.0:
        raise ValueError("learning-rate and temperature must be positive; retention-weight non-negative")
    methods = _csv(args.methods, allowed=METHODS)
    contexts = [int(value) for value in _csv(args.contexts)]
    if any(value < 0 or value > args.context for value in contexts):
        raise ValueError("each --contexts value must be between zero and --context")
    seeds = [int(value) for value in _csv(args.seeds)]
    if any(value < 0 for value in seeds):
        raise ValueError("seeds must be non-negative")
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "prefix_cache.pt"
    started = time.perf_counter()
    data_identity = _cache_identity(args)
    prepared = _load_or_prepare_records(args, data_identity, effective_deadline)
    if prepared is None:
        return 0
    train_records, validation_records = prepared

    # Outcome evaluation is deliberately explicit and configurable.  Cached
    # observations remain reusable while changing this evaluation budget.
    for record in [*train_records, *validation_records]:
        record.outcome_replicas = int(args.outcome_replicas)
        record.outcome_horizon = int(args.outcome_horizon)
    n_train_c = sum(record.is_correction for record in train_records)
    n_train_r = len(train_records) - n_train_c
    n_val_c = sum(record.is_correction for record in validation_records)
    n_val_r = len(validation_records) - n_val_c
    if n_train_c < int(args.min_train_corrections) or n_val_c < int(args.min_validation_corrections):
        raise ValueError(
            f"insufficient independently confirmed corrections: "
            f"train={n_train_c} (need {args.min_train_corrections}), "
            f"validation={n_val_c} (need {args.min_validation_corrections})"
        )
    print(f"[search-distill] records train={len(train_records)} ({n_train_c} corrections/{n_train_r} retention) validation={len(validation_records)} ({n_val_c} corrections/{n_val_r} retention)", flush=True)

    device = _resolve_device(args.device)
    if not budget_available():
        _atomic_json(out / "run_state.json", {
            "status": "partial", "reason": "deadline_before_training",
            "deadline_epoch": deadline_epoch,
        })
        return 0
    anchor = load_frozen_anchor(str(checkpoint), config=frontier_model_config(), device="cpu")
    reports: list[dict[str, Any]] = []
    expected_labels = [
        f"context{context}_{method}_seed{seed}"
        for context in contexts for method in methods for seed in seeds
    ]
    model_hash = file_sha256(checkpoint)
    source_hashes = {
        "train_frontier": file_sha256(dataset_root / "frontier_train.pt"),
        "validation_frontier": file_sha256(validation_root / "frontier_train.pt"),
        "search_distillation": data_identity["implementation_sha256"],
        "runner": file_sha256(Path(__file__).resolve()),
    }

    def write_global(status: str) -> None:
        manifest = {
            "version": SEARCH_DISTILLATION_VERSION,
            "status": status,
            "dataset_root": str(dataset_root),
            "validation_root": str(validation_root),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": model_hash,
            "source_hashes": source_hashes,
            "data_identity": data_identity,
            "git_revision": git_revision(REPO_ROOT),
            "device": str(device),
            "deadline_epoch": deadline_epoch,
            "effective_deadline_epoch": effective_deadline,
            "deadline_reserve_seconds": deadline_reserve,
            "config": {key: value for key, value in vars(args).items() if key not in {"dataset_root", "validation_root", "checkpoint", "out"}},
            "counts": {
                "train": len(train_records), "train_corrections": n_train_c,
                "train_retention": n_train_r, "validation": len(validation_records),
                "validation_corrections": n_val_c, "validation_retention": n_val_r,
            },
            "cache": str(cache_path),
            "reports": [report.get("label") for report in reports],
            "actor_export_epochs": list(ACTOR_EXPORT_EPOCHS),
            "actor_checkpoints": {
                str(report.get("label")): report.get("actor_checkpoints", [])
                for report in reports
                if report.get("label") is not None
            },
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        _atomic_json(out / "manifest.json", manifest)
        _atomic_json(out / "summary.json", {"manifest": manifest, "reports": reports})

    stopped_for_deadline = False
    failures = 0
    for context, method, seed in product(contexts, methods, seeds):
        label = f"context{context}_{method}_seed{seed}"
        if not budget_available():
            stopped_for_deadline = True
            break
        run_spec = {
            "version": SEARCH_DISTILLATION_VERSION, "label": label,
            "method": method, "seed": int(seed), "context": int(context),
            "epochs": int(args.epochs), "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "retention_weight": float(args.retention_weight),
            "temperature": float(args.temperature),
            "outcome_replicas": int(args.outcome_replicas),
            "outcome_horizon": int(args.outcome_horizon),
            "checkpoint_sha256": model_hash,
            "source_hashes": source_hashes,
            "data_identity": data_identity,
            "actor_export_epochs": list(ACTOR_EXPORT_EPOCHS),
        }
        report = _run_experiment(
            run_spec, out=out, anchor=anchor, train_records=train_records,
            validation_records=validation_records, device=device,
            effective_deadline=effective_deadline,
        )
        if report is None:
            failures += 1
        else:
            reports.append(report)
        write_global("running")
        if report is not None and report.get("status", "complete") != "complete":
            stopped_for_deadline = True
            break
    complete = len({report.get("label") for report in reports if report.get("status") == "complete"}) == len(expected_labels)
    status = "complete" if complete else ("partial" if stopped_for_deadline or deadline_epoch is not None else "failed")
    write_global(status)
    print(f"[search-distill] {status.upper()} reports={len(reports)} out={out}", flush=True)
    return 0 if status in {"complete", "partial"} and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
