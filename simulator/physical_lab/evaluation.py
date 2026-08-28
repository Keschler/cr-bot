"""Fast re-evaluation of sealed physical cases.

Each physical extraction is immutable input.  This module only reruns the
current deterministic simulator and writes a new aggregate snapshot, so a
simulator change can be scored against every stored case without touching raw
video, replay caches, or normalized observations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..engine import ENGINE_VERSION
from .comparison import ComparisonReport, compare_observation_to_replay
from .observation import ObservationManifest
from .replay import action_match_time_us, run_simulator_replay
from .schema import ExperimentSpec, PhysicalLabError, canonical_hash


EVALUATION_SCHEMA_VERSION = 1


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhysicalLabError(f"cannot load stored physical artifact {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise PhysicalLabError(f"stored physical artifact must be an object: {path}")
    return raw


def _verify_hash(raw: Mapping[str, Any], field: str, *, path: Path) -> None:
    declared = raw.get(field)
    if not isinstance(declared, str):
        raise PhysicalLabError(f"{path} is missing {field}")
    unsigned = dict(raw)
    unsigned.pop(field, None)
    if canonical_hash(unsigned) != declared:
        raise PhysicalLabError(f"{path} has an invalid {field}")


def _resolved_reference(value: object, *, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PhysicalLabError(f"stored case reference {field} is missing")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _action_times(run: Mapping[str, Any]) -> dict[str, int]:
    rows = run.get("actions", [])
    if not isinstance(rows, list):
        raise PhysicalLabError("stored physical run actions must be an array")
    result: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("accepted") is not True:
            continue
        action_id = row.get("action_id")
        value = action_match_time_us(run, row)
        if isinstance(action_id, str) and type(value) is int and value >= 0:
            result[action_id] = value
    return result


def _score(report: ComparisonReport) -> dict[str, object]:
    """Return a bounded score with explicit component provenance."""

    metrics = report.metrics
    components: dict[str, float] = {}

    def number(name: str, default: float | None = None) -> float | None:
        value = metrics.get(name, default)
        return float(value) if type(value) in (int, float) else default

    event = number("event_agreement_rate")
    lifecycle = number("alive_dead_spawn_transform_agreement_rate")
    target = number("target_retarget_agreement_rate")
    victim = number("victim_set_agreement_rate")
    position = number("position_within_tolerance_rate")
    timing_metric = metrics.get("timing_error_us")
    timing_mean = (
        float(timing_metric.get("mean"))
        if isinstance(timing_metric, Mapping) and type(timing_metric.get("mean")) in (int, float)
        else None
    )
    timing_tolerance = number("timing_tolerance_us", 10_000.0) or 10_000.0
    timing = None if timing_mean is None else max(0.0, min(1.0, 1.0 - timing_mean / max(1.0, timing_tolerance)))

    for name, value in (
        ("event_agreement", event),
        ("lifecycle_agreement", lifecycle),
        ("target_agreement", target),
        ("victim_agreement", victim),
        ("position_within_tolerance", position),
        ("timing_within_tolerance", timing),
    ):
        if value is not None:
            components[name] = max(0.0, min(1.0, value))

    weights = {
        "event_agreement": 0.35,
        "position_within_tolerance": 0.25,
        "timing_within_tolerance": 0.10,
        "lifecycle_agreement": 0.15,
        "target_agreement": 0.075,
        "victim_agreement": 0.075,
    }
    denominator = sum(weight for name, weight in weights.items() if name in components)
    score = (
        sum(weights[name] * components[name] for name in components) / denominator
        if denominator
        else 0.0
    )
    return {
        "score": round(max(0.0, min(1.0, score)), 8),
        "components": components,
        "weights": weights,
        "definition": "weighted normalized comparison metrics; no displayed game-clock values",
    }


def evaluate_stored_cases(
    cases_root: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate every ``extracted-case.json`` below ``cases_root``."""

    root = Path(cases_root).resolve()
    if not root.is_dir():
        raise PhysicalLabError(f"stored physical cases root is not a directory: {root}")
    repo = None if repository_root is None else Path(repository_root).resolve()
    case_paths = sorted(root.rglob("extracted-case.json"))
    rows: list[dict[str, object]] = []
    for case_path in case_paths:
        row: dict[str, object] = {"case_path": str(case_path)}
        try:
            case = _load_json(case_path)
            _verify_hash(case, "case_hash", path=case_path)
            if case.get("kind") != "physical_lab_extracted_case":
                raise PhysicalLabError("unsupported extracted-case kind")
            run_path = case_path.parent / "run.json"
            observation_ref = case.get("observation")
            observation_path = _resolved_reference(
                observation_ref.get("path") if isinstance(observation_ref, Mapping) else None,
                base=case_path.parent,
                field="observation.path",
            )
            run = _load_json(run_path)
            _verify_hash(run, "run_hash", path=run_path)
            observation = ObservationManifest.load(observation_path)
            spec = ExperimentSpec.from_dict(run.get("experiment"))
            if run.get("experiment_hash") != spec.experiment_hash():
                raise PhysicalLabError("stored run experiment hash does not match its specification")
            replay = run_simulator_replay(spec, action_times=_action_times(run))
            report = compare_observation_to_replay(observation, replay)
            score = _score(report)
            divergence = report.first_divergence
            row.update(
                {
                    "run_id": observation.run_id,
                    "experiment_hash": spec.experiment_hash(),
                    "status": "evaluated",
                    "observation_status": observation.status.value,
                    "eligible": report.eligible,
                    "score": score["score"],
                    "score_detail": score,
                    "metrics": report.metrics,
                    "divergence": divergence,
                    "divergence_match_time_us": (
                        None
                        if not isinstance(divergence, Mapping)
                        else divergence.get("match_time_us")
                    ),
                    "comparison_hash": report.comparison_hash,
                    "engine_version": ENGINE_VERSION,
                    "repository_root": None if repo is None else str(repo),
                }
            )
        except (OSError, KeyError, TypeError, ValueError, PhysicalLabError) as error:
            row.update({"status": "rejected_artifact", "error": str(error)})
        rows.append(row)

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    scores = [float(row["score"]) for row in evaluated if type(row.get("score")) in (int, float)]
    payload: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "physical_lab_stored_case_evaluation",
        "cases_root": str(root),
        "repository_root": None if repo is None else str(repo),
        "engine_version": ENGINE_VERSION,
        "case_count": len(rows),
        "evaluated_case_count": len(evaluated),
        "rejected_artifact_count": len(rows) - len(evaluated),
        "mean_score": None if not scores else sum(scores) / len(scores),
        "cases": rows,
    }
    payload["evaluation_hash"] = canonical_hash(payload)
    return payload


def write_stored_evaluation(path: str | Path, payload: Mapping[str, object]) -> str:
    destination = Path(path)
    unsigned = dict(payload)
    unsigned.pop("evaluation_hash", None)
    sealed = {**unsigned, "evaluation_hash": canonical_hash(unsigned)}
    encoded = json.dumps(sealed, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise PhysicalLabError(f"stored evaluation output already differs: {destination}")
        return str(sealed["evaluation_hash"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return str(sealed["evaluation_hash"])


__all__ = ["EVALUATION_SCHEMA_VERSION", "evaluate_stored_cases", "write_stored_evaluation"]
