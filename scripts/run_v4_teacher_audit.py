"""Run the resumable V4 search-teacher reliability audit.

The broad frontier generator stores an eight-replica, H=128 verdict for every
candidate.  This script re-evaluates a deterministic sample at 16 replicas,
then evaluates a smaller, pre-selected subset at H=28/56/112 with 32
replicas.  It deliberately keeps the original selected alternative fixed
when calculating confirmation statistics; selecting the best action and
measuring that same action on the same replicas would be optimistic.

Sources are specified as ``ROOT[:SPLIT]``.  A source without a suffix means
``train``.  The sealed split is therefore opt-in:

    python scripts/run_v4_teacher_audit.py \
        --source outputs/v4/runpod_frontier_dataset_20260921:train \
        --source outputs/v4/runpod_frontier_dataset_20260921:sealed \
        --checkpoint outputs/v4/bootstrap_seed0/actor.pt \
        --out outputs/v4/teacher_audit_20260921 --workers 26

The output is safe to resume.  Each state/horizon is written to its own file
through an atomic rename, and an existing complete result is skipped only
after its identity and evaluation settings have been checked.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import multiprocessing
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

AUDIT_SCHEMA = "teacher-audit-v2"
AUDIT_VERSION = "teacher-audit-v2"
DEFAULT_HORIZONS = (28, 56, 112)
DEFAULT_MAX_BRANCHES = 7
DEFAULT_DELTA_MIN = 0.05
_WORKER_POLICY: Any = None


@dataclass(frozen=True, slots=True)
class SourceSpec:
    root: Path
    split: str
    source_id: str


@dataclass(frozen=True, slots=True)
class AuditTask:
    record: dict[str, Any]
    phase: str
    horizon: int
    n_replicas: int
    max_branches: int
    delta_min: float
    output_path: str


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    tmp.write_bytes(_json_bytes(payload))
    tmp.replace(path)


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def _stable_rank(seed: int, value: str) -> str:
    return sha256(f"{seed}|{value}".encode()).hexdigest()


def parse_source_token(token: str) -> tuple[Path, str]:
    """Parse ``ROOT[:train|sealed|all]`` with train as the safe default."""

    raw = str(token).strip()
    if not raw:
        raise ValueError("--source cannot be empty")
    root_text, separator, suffix = raw.rpartition(":")
    if separator and suffix in {"train", "sealed", "all"} and root_text:
        return Path(root_text).expanduser().resolve(), suffix
    return Path(raw).expanduser().resolve(), "train"


def expand_sources(tokens: Sequence[str]) -> list[SourceSpec]:
    """Expand source tokens and deduplicate exact root/split pairs."""

    expanded: list[tuple[Path, str]] = []
    for token in tokens:
        root, selector = parse_source_token(token)
        splits = ("train", "sealed") if selector == "all" else (selector,)
        for split in splits:
            item = (root, split)
            if item not in expanded:
                expanded.append(item)
    if not expanded:
        raise ValueError("at least one --source is required")
    specs: list[SourceSpec] = []
    for root, split in expanded:
        manifest = root / "manifest.json"
        if manifest.is_file():
            source_id = sha256_file(manifest)[:12]
        else:
            source_id = sha256(str(root).encode()).hexdigest()[:12]
        specs.append(SourceSpec(root=root, split=split, source_id=source_id))
    return specs


def _match_path(spec: SourceSpec, match_id: str) -> Path:
    archetype, index = match_id.rsplit("#", 1)
    return spec.root / f"{spec.split}_{archetype}_{index}_match.json"


def _regret_bin(regret: float, cuts: tuple[float, float]) -> str:
    if regret < cuts[0]:
        return "low"
    if regret < cuts[1]:
        return "mid"
    return "high"


def _cuts(records: Sequence[dict[str, Any]]) -> tuple[float, float]:
    values = sorted(float(r["source_verdict"].get("regret", 0.0)) for r in records)
    if not values:
        return (0.0, 0.0)
    return values[len(values) // 3], values[(2 * len(values)) // 3]


def _record_id(record: dict[str, Any]) -> str:
    return str(record["audit_id"])


def _select_round_robin(
    records: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
    per_match_cap: int,
) -> list[dict[str, Any]]:
    """Select a deterministic stratified sample with a source-match cap."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if count > len(records):
        raise ValueError(f"requested {count} states but only {len(records)} are available")
    cuts = _cuts(records)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("split", "?")),
            str(record.get("archetype", "?")),
            _regret_bin(float(record["source_verdict"].get("regret", 0.0)), cuts),
        )
        groups[key].append(record)
    for group in groups.values():
        group.sort(key=lambda r: _stable_rank(seed, _record_id(r)))

    picked: list[dict[str, Any]] = []
    used: Counter[tuple[str, str, str]] = Counter()
    # First pass is round-robin across split/archetype/regret strata.
    while len(picked) < count:
        progressed = False
        for key in sorted(groups):
            group = groups[key]
            while group:
                candidate = group.pop(0)
                match_key = (
                    str(candidate["source_id"]),
                    str(candidate["split"]),
                    str(candidate["match_id"]),
                )
                if used[match_key] >= per_match_cap:
                    continue
                used[match_key] += 1
                picked.append(dict(candidate, regret_bin=key[2]))
                progressed = True
                break
            if len(picked) >= count:
                break
        if not progressed:
            break
    if len(picked) < count:
        raise ValueError(
            f"source pool has {len(records)} states but only {len(picked)} satisfy "
            f"the per-match cap of {per_match_cap}"
        )
    return picked


def select_audit_records(
    records: Sequence[dict[str, Any]],
    *,
    count: int = 300,
    seed: int = 20261003,
    per_match_cap: int = 5,
) -> list[dict[str, Any]]:
    """Public deterministic first-pass selector."""

    return _select_round_robin(
        records, count=count, seed=seed, per_match_cap=per_match_cap
    )


def select_high_rep_records(
    records: Sequence[dict[str, Any]],
    *,
    count: int = 100,
    seed: int = 20261004,
    per_match_cap: int = 2,
) -> list[dict[str, Any]]:
    """Select important/uncertain states without using new audit outcomes."""

    ranked = [dict(r) for r in records]
    # Filter score is available before the audit and is the intended priority
    # signal.  Stable rank breaks ties without depending on input order.
    ranked.sort(
        key=lambda r: (
            -float(r.get("source_verdict", {}).get("filter", {}).get("score", 0.0)),
            _stable_rank(seed, _record_id(r)),
        )
    )
    cuts = _cuts(ranked)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in ranked:
        key = (
            str(record.get("split", "?")),
            str(record.get("archetype", "?")),
            _regret_bin(float(record["source_verdict"].get("regret", 0.0)), cuts),
        )
        groups[key].append(record)
    for group in groups.values():
        group.sort(
            key=lambda r: (
                -float(r.get("source_verdict", {}).get("filter", {}).get("score", 0.0)),
                _stable_rank(seed, _record_id(r)),
            )
        )
    picked: list[dict[str, Any]] = []
    used: Counter[tuple[str, str, str]] = Counter()
    while len(picked) < count:
        progressed = False
        for key in sorted(groups):
            group = groups[key]
            while group:
                candidate = group.pop(0)
                match_key = (
                    str(candidate["source_id"]),
                    str(candidate["split"]),
                    str(candidate["match_id"]),
                )
                if used[match_key] >= per_match_cap:
                    continue
                used[match_key] += 1
                picked.append(dict(candidate, regret_bin=key[2]))
                progressed = True
                break
            if len(picked) >= count:
                break
        if not progressed:
            raise ValueError(
                f"could not select {count} high-rep states under match cap {per_match_cap}"
            )
    return picked


def load_source_records(specs: Sequence[SourceSpec]) -> list[dict[str, Any]]:
    """Load accepted frontier verdicts and attach their source provenance."""

    records: list[dict[str, Any]] = []
    for spec in specs:
        verdict_path = spec.root / f"{spec.split}_verdicts.json"
        if not verdict_path.is_file():
            raise FileNotFoundError(verdict_path)
        verdicts = json.loads(verdict_path.read_text())
        if not isinstance(verdicts, list):
            raise ValueError(f"{verdict_path} must contain a JSON list")
        for verdict in verdicts:
            if verdict.get("class") != "frontier":
                continue
            match_path = _match_path(spec, str(verdict["match_id"]))
            if not match_path.is_file():
                raise FileNotFoundError(match_path)
            audit_id = (
                f"{spec.source_id}:{spec.split}:{verdict['match_id']}:t{verdict['t']}"
            )
            records.append(
                {
                    "audit_id": audit_id,
                    "source_id": spec.source_id,
                    "source_root": str(spec.root),
                    "split": spec.split,
                    "match_id": verdict["match_id"],
                    "archetype": verdict["archetype"],
                    "t": int(verdict["t"]),
                    "tick": int(verdict["tick"]),
                    "state_hash": verdict["state_hash"],
                    "match_path": str(match_path),
                    "source_verdict": {
                        "class": verdict.get("class"),
                        "regret": verdict.get("regret"),
                        "stability": verdict.get("stability"),
                        "best_source": verdict.get("best_source"),
                        "best_action": verdict.get("best_action"),
                        "replica_indices": verdict.get("replica_indices", []),
                        "best_mean": verdict.get("best_mean"),
                        "sim_mean": verdict.get("sim_mean"),
                        "filter": verdict.get("filter", {}),
                        "tags": verdict.get("tags", {}),
                        "branches": verdict.get("branches", []),
                    },
                }
            )
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["audit_id"] in by_id:
            raise ValueError(f"duplicate audit id: {record['audit_id']}")
        by_id[record["audit_id"]] = record
    return list(by_id.values())


def _action_key(branch: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(branch.get("mode", 0)),
        int(branch.get("slot", 0)),
        int(branch.get("row", 0)),
        int(branch.get("col", 0)),
    )


def _source_best_action(source_verdict: dict[str, Any]) -> dict[str, Any] | None:
    """Return the exact source winner, including duplicate source branches.

    ``best_source`` is provenance and is not an action identity: frontier
    planning may emit multiple ``policy-top`` branches.  New verdicts persist
    ``best_action`` directly; old verdicts are reconstructed by selecting the
    highest-mean branch among the winning source.
    """

    stored = source_verdict.get("best_action")
    if isinstance(stored, dict):
        return {
            "source": str(stored.get("source", source_verdict.get("best_source", ""))),
            "mode": int(stored.get("mode", 0)),
            "slot": int(stored.get("slot", 0)),
            "row": int(stored.get("row", 0)),
            "col": int(stored.get("col", 0)),
        }
    source = source_verdict.get("best_source")
    candidates = [
        branch for branch in source_verdict.get("branches", [])
        if branch.get("source") == source
    ]
    if not candidates:
        return None
    branch = max(candidates, key=lambda value: float(value.get("mean", -float("inf"))))
    return {
        "source": str(branch.get("source", source)),
        "mode": int(branch.get("mode", 0)),
        "slot": int(branch.get("slot", 0)),
        "row": int(branch.get("row", 0)),
        "col": int(branch.get("col", 0)),
    }


def paired_action_metrics(
    verdict: dict[str, Any],
    frozen_action: dict[str, Any] | None,
    *,
    indices: Iterable[int] | None = None,
    delta_min: float = DEFAULT_DELTA_MIN,
) -> dict[str, Any]:
    """Compute paired alternative-minus-sim statistics for one fixed action."""

    if frozen_action is None:
        return {"available": False, "reason": "source best action is missing"}
    target = _action_key(frozen_action)
    sim = next((b for b in verdict.get("branches", []) if b.get("source") == "sim"), None)
    alternative = next((b for b in verdict.get("branches", []) if _action_key(b) == target), None)
    if sim is None or alternative is None:
        return {
            "available": False,
            "reason": "frozen action or sim branch is absent from reevaluation",
            "frozen_action": frozen_action,
        }
    sim_values = [float(v) for v in sim.get("values", [])]
    alt_values = [float(v) for v in alternative.get("values", [])]
    n = min(len(sim_values), len(alt_values))
    selected = list(range(n)) if indices is None else [i for i in indices if 0 <= i < n]
    deltas = [alt_values[i] - sim_values[i] for i in selected]
    if not deltas:
        return {"available": False, "reason": "no selected replicas", "frozen_action": frozen_action}
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    se = sd / math.sqrt(len(deltas))
    lcb = mean - 1.96 * se
    ucb = mean + 1.96 * se
    return {
        "available": True,
        "frozen_action": frozen_action,
        "action_source": alternative.get("source"),
        "replica_indices": selected,
        "n": len(deltas),
        "deltas": [round(v, 6) for v in deltas],
        "mean_delta": round(mean, 6),
        "sd_delta": round(sd, 6),
        "se_delta": round(se, 6),
        "lcb95": round(lcb, 6),
        "ucb95": round(ucb, 6),
        "positive_fraction": round(sum(v > 0.0 for v in deltas) / len(deltas), 6),
        "classification": (
            "strong" if lcb > delta_min else
            "negative" if ucb < 0.0 else
            "probable" if mean > 0.0 else
            "ambiguous"
        ),
    }


def _worker_init(checkpoint: str) -> None:
    global _WORKER_POLICY
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from simulator.rl.v4.frontier import frontier_model_config
    from simulator.rl.v4.v4_ppo import load_frozen_anchor

    _WORKER_POLICY = load_frozen_anchor(
        checkpoint, config=frontier_model_config(), device="cpu"
    )


def _candidate_from_record(record: dict[str, Any], match: dict[str, Any]) -> dict[str, Any]:
    decision = match["decisions"][int(record["t"])]
    if decision.get("state_hash") != record["state_hash"]:
        raise ValueError(
            f"state hash mismatch for {record['audit_id']}: "
            f"{decision.get('state_hash')} != {record['state_hash']}"
        )
    return {
        "match_id": record["match_id"],
        "archetype": record["archetype"],
        "t": int(record["t"]),
        "tick": int(record["tick"]),
        "state_hash": record["state_hash"],
        "tags": decision.get("tags", record["source_verdict"].get("tags", {})),
        "filter": record["source_verdict"].get("filter", {}),
        "learner": decision["learner"],
        "hand": decision.get("hand", []),
        "elixir": decision.get("elixir", 0.0),
    }


def _run_task(task: AuditTask) -> dict[str, Any]:
    global _WORKER_POLICY
    if _WORKER_POLICY is None:
        raise RuntimeError("worker policy was not initialized")
    from simulator.rl.v4.frontier_eval import EvalConfig, evaluate_candidate

    record = task.record
    match = json.loads(Path(record["match_path"]).read_text())
    candidate = _candidate_from_record(record, match)
    replica_seed_base = int(match["config"].get("pool_seed", 0))
    frozen = _source_best_action(record["source_verdict"])
    if frozen is None:
        raise ValueError(f"source verdict has no frozen winning action: {record['audit_id']}")
    # Source frontier verdicts selected their winner from their own replicas
    # (normally indices 0..7).  Start this audit after that source range so
    # every audit sample is independent of the action-selection samples.
    source_indices = tuple(
        int(value) for value in record["source_verdict"].get("replica_indices", ())
    )
    if not source_indices:
        source_branches = record["source_verdict"].get("branches", ())
        source_indices = tuple(
            int(replica.get("replica_index", position))
            for branch in source_branches
            for position, replica in enumerate(branch.get("replicas", ()))
            if isinstance(replica, dict)
        )
    source_replica_count = max(source_indices, default=7) + 1
    audit_indices = tuple(
        range(source_replica_count, source_replica_count + int(task.n_replicas))
    )
    verdict = evaluate_candidate(
        match=match,
        candidate=candidate,
        policy=_WORKER_POLICY,
        eval_cfg=EvalConfig(
            horizon=task.horizon,
            n_replicas=task.n_replicas,
            max_branches=task.max_branches,
            replica_seed_base=replica_seed_base,
            replica_indices=audit_indices,
            frozen_action=frozen,
        ),
        device="cpu",
    )
    metrics = dict(verdict.get("frozen_confirmation", {}))
    if metrics.get("available"):
        metrics["delta_min"] = float(task.delta_min)
        metrics["classification"] = (
            "strong" if float(metrics.get("lcb95", -float("inf"))) > float(task.delta_min)
            else "negative" if float(metrics.get("ucb95", float("inf"))) < 0.0
            else "probable" if float(metrics.get("mean_delta", 0.0)) > 0.0
            else "ambiguous"
        )
    else:
        metrics.setdefault("frozen_action", frozen)
        metrics.setdefault("reason", "independent confirmation unavailable")
    return {
        "schema": AUDIT_SCHEMA,
        "audit_id": record["audit_id"],
        "phase": task.phase,
        "source": {
            key: record[key]
            for key in (
                "source_id", "source_root", "split", "match_id", "archetype",
                "t", "tick", "state_hash", "regret_bin",
            )
        },
        "original": record["source_verdict"],
        "eval_config": {
            "horizon": task.horizon,
            "n_replicas": task.n_replicas,
            "max_branches": task.max_branches,
            "replica_seed_base": replica_seed_base,
        },
        "frozen_confirmation": metrics,
        "verdict": verdict,
        "status": "complete",
    }


def _valid_result(path: Path, task: AuditTask) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    cfg = payload.get("eval_config", {})
    return (
        payload.get("schema") == AUDIT_SCHEMA
        and payload.get("status") == "complete"
        and payload.get("audit_id") == task.record["audit_id"]
        and payload.get("phase") == task.phase
        and payload.get("source", {}).get("state_hash") == task.record["state_hash"]
        and int(cfg.get("horizon", -1)) == task.horizon
        and int(cfg.get("n_replicas", -1)) == task.n_replicas
        and int(cfg.get("max_branches", -1)) == task.max_branches
    )


def _source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        REPO / "simulator/rl/v4/frontier.py",
        REPO / "simulator/rl/v4/frontier_eval.py",
        REPO / "simulator/rl/v4/model_v4.py",
        *sorted((REPO / "simulator/rl/v4/v4_ppo").rglob("*.py")),
        *sorted((REPO / "simulator/rl/v4/frontier_ppo").rglob("*.py")),
        REPO / "simulator/public_state_estimator.py",
    ]
    return {str(path): sha256_file(path) for path in paths if path.is_file()}


def _source_manifest_hashes(specs: Sequence[SourceSpec]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for spec in specs:
        for name in ("manifest.json", f"{spec.split}_verdicts.json"):
            path = spec.root / name
            if path.is_file():
                hashes[f"{spec.source_id}/{name}"] = sha256_file(path)
    return hashes


def _config_for_manifest(args: argparse.Namespace, specs: Sequence[SourceSpec]) -> dict[str, Any]:
    return {
        "audit_version": AUDIT_VERSION,
        "sources": [
            {"root": str(s.root), "split": s.split, "source_id": s.source_id}
            for s in specs
        ],
        "first_count": args.first_count,
        "high_count": args.high_count,
        "selection_seed": args.selection_seed,
        "high_selection_seed": args.high_selection_seed,
        "per_match_cap": args.per_match_cap,
        "high_per_match_cap": args.high_per_match_cap,
        "max_branches": args.max_branches,
        "delta_min": args.delta_min,
        "phase1": {"horizon": 128, "n_replicas": 16},
        "phase2": {"horizons": list(DEFAULT_HORIZONS), "n_replicas": 32},
    }


def _load_or_create_selection(
    *,
    out: Path,
    specs: Sequence[SourceSpec],
    args: argparse.Namespace,
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection_path = out / "selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        if selection.get("schema") != AUDIT_SCHEMA:
            raise ValueError(f"unsupported selection schema in {selection_path}")
        return selection["first_pass"], selection["high_rep"]
    selected = select_audit_records(
        records,
        count=args.first_count,
        seed=args.selection_seed,
        per_match_cap=args.per_match_cap,
    )
    high = select_high_rep_records(
        selected,
        count=args.high_count,
        seed=args.high_selection_seed,
        per_match_cap=args.high_per_match_cap,
    )
    atomic_json(
        selection_path,
        {
            "schema": AUDIT_SCHEMA,
            "manifest_config": manifest["config"],
            "first_pass": selected,
            "high_rep": high,
        },
    )
    return selected, high


def _phase_tasks(
    records: Sequence[dict[str, Any]],
    *,
    phase: str,
    horizon: int,
    n_replicas: int,
    args: argparse.Namespace,
    out: Path,
) -> list[AuditTask]:
    phase_dir = out / phase
    tasks: list[AuditTask] = []
    for record in records:
        stem = _safe_component(record["audit_id"])
        path = phase_dir / f"{stem}.json"
        task = AuditTask(
            record=record,
            phase=phase,
            horizon=horizon,
            n_replicas=n_replicas,
            max_branches=args.max_branches,
            delta_min=args.delta_min,
            output_path=str(path),
        )
        if not _valid_result(path, task):
            tasks.append(task)
    return tasks


def _run_phase(tasks: Sequence[AuditTask], checkpoint: Path, workers: int) -> None:
    if not tasks:
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(str(checkpoint),),
    ) as pool:
        futures = {pool.submit(_run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            payload = future.result()
            atomic_json(Path(task.output_path), payload)
            print(
                f"{task.phase} {task.record['audit_id']} complete",
                flush=True,
            )


def _iter_result_files(out: Path, phase: str) -> Iterable[dict[str, Any]]:
    directory = out / phase
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "complete":
            yield payload


def _mean_or_none(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _phase_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    verdicts = [r.get("verdict", {}) for r in results]
    frozen = [r.get("frozen_confirmation", {}) for r in results]
    means = [float(v["regret"]) for v in verdicts if "regret" in v]
    classes = Counter(str(v.get("class", "?")) for v in verdicts)
    classifications = Counter(
        str(m.get("classification", "unavailable"))
        for m in frozen
        if m.get("available")
    )
    confirmation_means = [
        float(m["mean_delta"]) for m in frozen if m.get("available")
    ]
    return {
        "n_complete": len(results),
        "classes": dict(classes),
        "mean_regret": _mean_or_none(means),
        "median_regret": round(statistics.median(means), 6) if means else None,
        "frozen_confirmation": {
            "n_available": len(confirmation_means),
            "mean_delta": _mean_or_none(confirmation_means),
            "classifications": dict(classifications),
        },
    }


def _horizon_summary(results_by_horizon: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    from simulator.rl.v4.frontier_eval import fixed_action_horizon_gate, horizon_stability

    reports = {
        int(h): [r.get("verdict", {}) for r in values]
        for h, values in results_by_horizon.items()
    }
    # horizon_stability keys by state_hash.  The selected states are unique;
    # if a caller supplies duplicate source roots, use a conservative count in
    # the report and leave the per-state details intact.
    try:
        stability = horizon_stability(reports, reference=112)
    except (KeyError, TypeError, ValueError):
        stability = {"reference": 112, "horizons": {}, "chosen_horizon": 112}
    required = tuple(sorted(int(value) for value in reports if int(value) != 112))
    gate = fixed_action_horizon_gate(
        {int(h): list(values) for h, values in results_by_horizon.items()},
        reference=112,
        required_horizons=required,
    )
    stability["fixed_action_gate"] = gate
    stability["gate_pass"] = bool(gate["pass"])
    return stability


def _render_report(
    *,
    out: Path,
    manifest: dict[str, Any],
    selection: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    summaries: dict[str, Any],
) -> None:
    first, high = selection
    lines = [
        "# V4 Search-Teacher Reliability Audit",
        "",
        f"Schema: `{AUDIT_SCHEMA}`",
        "",
        "This report re-evaluates fixed frontier actions with paired simulator replicas. "
        "The frozen-action confirmation metrics avoid choosing and testing the best "
        "action on the same replicas.",
        "",
        "## Selection",
        "",
        f"- First pass states: {len(first)}",
        f"- High-replication states: {len(high)}",
        f"- Sources: {len(manifest['config']['sources'])}",
        f"- Selection seed: `{manifest['config']['selection_seed']}`",
        "- Existing sealed states are development data once included in this audit; "
        "a fresh sealed set is required for final reporting.",
        "",
        "## Phase summaries",
        "",
    ]
    for phase, summary in summaries.items():
        lines.append(f"### {phase}")
        lines.append("")
        lines.append(f"- Complete: {summary.get('n_complete', 0)}")
        lines.append(f"- Classes: `{json.dumps(summary.get('classes', {}), sort_keys=True)}`")
        lines.append(f"- Mean regret: {summary.get('mean_regret')}")
        lines.append(f"- Median regret: {summary.get('median_regret')}")
        confirmation = summary.get("frozen_confirmation", {})
        lines.append(
            f"- Frozen confirmation: {json.dumps(confirmation, sort_keys=True)}"
        )
        if phase == "horizon_stability":
            lines.append(f"- Horizon stability: `{json.dumps(summary, sort_keys=True)}`")
        lines.append("")
    atomic_json(out / "summary.json", {"schema": AUDIT_SCHEMA, "summaries": summaries})
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Dataset root, optionally suffixed :train, :sealed, or :all; repeatable.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-count", type=int, default=300)
    parser.add_argument("--high-count", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=20261003)
    parser.add_argument("--high-selection-seed", type=int, default=20261004)
    parser.add_argument("--per-match-cap", type=int, default=5)
    parser.add_argument("--high-per-match-cap", type=int, default=2)
    parser.add_argument("--max-branches", type=int, default=DEFAULT_MAX_BRANCHES)
    parser.add_argument("--delta-min", type=float, default=DEFAULT_DELTA_MIN)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Select and write the manifest without evaluating.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.first_count < 1 or args.high_count < 1:
        raise ValueError("first/high counts must be positive")
    if args.high_count > args.first_count:
        raise ValueError("high-count cannot exceed first-count")
    if args.per_match_cap < 1 or args.high_per_match_cap < 1:
        raise ValueError("match caps must be positive")
    if args.max_branches < 2:
        raise ValueError("max-branches must be at least 2")
    if args.workers < 1:
        raise ValueError("workers must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        specs = expand_sources(args.source)
        args.checkpoint = args.checkpoint.expanduser().resolve()
        out = args.out.expanduser().resolve()
        if not args.dry_run and not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        records = load_source_records(specs)
        if len(records) < args.first_count:
            raise ValueError(
                f"selected sources provide {len(records)} frontier states, "
                f"but --first-count requests {args.first_count}; "
                "sealed is opt-in (append :sealed to --source)."
            )
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": AUDIT_SCHEMA,
            "config": _config_for_manifest(args, specs),
            "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint.is_file() else None,
            "source_manifest_hashes": _source_manifest_hashes(specs),
            "source_hashes": _source_hashes(),
            "record_count": len(records),
        }
        manifest_path = out / "manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text())
            if existing != manifest:
                raise ValueError(
                    f"existing manifest differs at {manifest_path}; choose the same settings or a new --out"
                )
        else:
            atomic_json(manifest_path, manifest)
        first, high = _load_or_create_selection(
            out=out, specs=specs, args=args, manifest=manifest, records=records
        )
        print(f"selected first={len(first)} high={len(high)} source_records={len(records)}")
        if args.dry_run:
            return 0
        _run_phase(
            _phase_tasks(
                first,
                phase="phase1_16rep_h128",
                horizon=128,
                n_replicas=16,
                args=args,
                out=out,
            ),
            args.checkpoint,
            args.workers,
        )
        for horizon in DEFAULT_HORIZONS:
            _run_phase(
                _phase_tasks(
                    high,
                    phase=f"phase2_32rep_h{horizon}",
                    horizon=horizon,
                    n_replicas=32,
                    args=args,
                    out=out,
                ),
                args.checkpoint,
                args.workers,
            )

        summaries: dict[str, Any] = {}
        phase1 = list(_iter_result_files(out, "phase1_16rep_h128"))
        summaries["phase1_16rep_h128"] = _phase_summary(phase1)
        by_horizon: dict[int, list[dict[str, Any]]] = {}
        for horizon in DEFAULT_HORIZONS:
            results = list(_iter_result_files(out, f"phase2_32rep_h{horizon}"))
            summaries[f"phase2_32rep_h{horizon}"] = _phase_summary(results)
            by_horizon[horizon] = results
        summaries["horizon_stability"] = _horizon_summary(by_horizon)
        _render_report(
            out=out,
            manifest=manifest,
            selection=(first, high),
            summaries=summaries,
        )
        print(f"COMPLETE {out}")
        return 0
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
