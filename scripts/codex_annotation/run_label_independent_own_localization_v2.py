from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.own_localization import validate_own_localization_decisions
from cr_bot.own_localization_cascade import (
    direct_legal,
    route_v2_after_terra_verify,
    route_v2_initial,
    select_v2_consensus,
)
from scripts.codex_annotation.prepare_own_localization_packages import (
    prepare_packages,
)


PROMPT_DIR = ROOT / "scripts/codex_annotation/prompts"
ROLE_PROMPTS = {
    "luna_marker": PROMPT_DIR / "own_localization_v2_marker.txt",
    "luna_temporal": PROMPT_DIR / "own_localization_v2_temporal.txt",
    "terra_residual": PROMPT_DIR / "own_localization_v2_residual.txt",
    "terra_verify": PROMPT_DIR / "own_localization_v2_coordinate_audit.txt",
}
SPECIALIZED_PROMPTS = {
    "rolling_spell": PROMPT_DIR / "own_rolling_spell_localization_v2.txt",
    "targeted_spell": PROMPT_DIR / "own_targeted_spell_localization_v2.txt",
    "unit_or_building": PROMPT_DIR / "own_unit_building_localization_v2.txt",
}
MODELS = {
    "luna_marker": ("gpt-5.6-luna", "low"),
    "luna_temporal": ("gpt-5.6-luna", "medium"),
    "luna_specialized": ("gpt-5.6-luna", "medium"),
    "terra_residual": ("gpt-5.6-terra", "medium"),
    "terra_verify": ("gpt-5.6-terra", "high"),
    "sol_specialized": ("gpt-5.6-sol", "high"),
}
INITIAL_ROLES = (
    "luna_marker",
    "luna_temporal",
    "luna_specialized",
    "terra_residual",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _specialized_prompt(card: str) -> Path:
    base = card[4:] if card.startswith("evo-") else card
    if base == "log":
        return SPECIALIZED_PROMPTS["rolling_spell"]
    if CARD_METADATA[base]["kind"] == "spell":
        return SPECIALIZED_PROMPTS["targeted_spell"]
    return SPECIALIZED_PROMPTS["unit_or_building"]


def _prompt_for(role: str, card: str) -> Path:
    if role in {"luna_specialized", "sol_specialized"}:
        return _specialized_prompt(card)
    return ROLE_PROMPTS[role]


def _frozen_policy(*, source_path: Path, source: dict[str, Any]) -> dict[str, Any]:
    implementation_paths = {
        "runner": Path(__file__).resolve(),
        "renderer": ROOT / "scripts/codex_annotation/prepare_own_localization_packages.py",
        "validator": ROOT / "src/cr_bot/own_localization.py",
        "cascade": ROOT / "src/cr_bot/own_localization_cascade.py",
        "worker_harness": ROOT / "scripts/codex_annotation/run_model_worker.py",
    }
    prompt_paths = {**ROLE_PROMPTS, **SPECIALIZED_PROMPTS}
    own_count = sum(
        isinstance(row, dict) and row.get("side", "own") == "own"
        for row in source.get("events", [])
    )
    return {
        "version": 2,
        "policy_id": "fixed-global-v2",
        "revision": 2,
        "source_sha256": _sha256(source_path),
        "source_run_id": source.get("run_id"),
        "derived_own_target_count": own_count,
        "reference_access": "prohibited_until_complete_batch_seal",
        "worker_isolation": "one event and one role per work directory",
        "worker_concurrency": 1,
        "structural_retry": "same package/model/prompt at most once after exit 65",
        "persistent_invalid_role": (
            "exclude the role from agreement and selection, record it as invalid, "
            "and continue the pre-frozen Terra/Sol escalation path"
        ),
        "evidence": {
            "version": 2,
            "package_size": 1,
            "macro_offsets": "dense event-9 through event+18",
            "grid_offsets": [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18],
            "grid_split": "onset through event+3; follow-through after event+3",
            "ordinary_scope": "original grid rows 14-31",
            "targeted_spell_scope": "full arena",
        },
        "initial_roles": list(INITIAL_ROLES),
        "routing": {
            "initial_accept": (
                "pairwise Chebyshev-1 clique of at least three direct legal "
                "decisions spanning Luna and Terra"
            ),
            "terra_verify": "run for every target failing initial acceptance",
            "post_terra_accept": (
                "pairwise Chebyshev-1 clique of at least three direct legal "
                "decisions spanning Luna and Terra"
            ),
            "sol_specialized": "run for every target failing post-Terra acceptance",
        },
        "selection": (
            "strongest pairwise Chebyshev-1 direct-legal clique ranked by model-family "
            "diversity, vote count, fixed tier weight, then compactness; choose the "
            "highest-tier medoid only inside that clique"
        ),
        "model_cost_weights": {
            "gpt-5.6-luna": 0.1,
            "gpt-5.6-terra": 1.0,
            "gpt-5.6-sol": 2.5,
        },
        "models": {
            role: {"model": model, "effort": effort}
            for role, (model, effort) in MODELS.items()
        },
        "specialized_prompt_routing": {
            "log": "rolling_spell",
            "other_spell": "targeted_spell",
            "troop_or_building": "unit_or_building",
        },
        "prompt_sha256s": {
            name: _sha256(path) for name, path in sorted(prompt_paths.items())
        },
        "implementation_sha256s": {
            name: _sha256(path) for name, path in sorted(implementation_paths.items())
        },
    }


def _materialize_role_package(
    *, base_package: Path, role: str, cascade_dir: Path
) -> Path:
    destination = cascade_dir / "worker_packages" / role / base_package.parent.name
    package_path = destination / "package.json"
    if destination.exists():
        if not package_path.is_file():
            raise ValueError(f"incomplete role package {destination}")
        source_document = _read(base_package)
        if _read(package_path) != source_document:
            raise ValueError(f"role package changed for {destination}")
        return package_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_package.parent, destination)
    return package_path


def _run_worker(
    *,
    cascade_dir: Path,
    base_package: Path,
    role: str,
    card: str,
    event_frame: int,
) -> tuple[dict[str, Any] | None, Path]:
    package_path = _materialize_role_package(
        base_package=base_package,
        role=role,
        cascade_dir=cascade_dir,
    )
    workdir = package_path.parent
    model, effort = MODELS[role]
    prompt = _prompt_for(role, card)
    stem = base_package.parent.name
    promoted = cascade_dir / "worker_outputs" / role / f"{stem}.json"
    if promoted.is_file():
        document = _read(promoted)
        validate_own_localization_decisions(document, _read(package_path))
        return document, package_path

    for attempt in (1, 2):
        expected = workdir / f"worker-output-{role}-a{attempt}.json"
        result_path = cascade_dir / "results" / role / f"{stem}-a{attempt}.json"
        if result_path.is_file():
            result = _read(result_path)
            if result.get("status") == "invalid_output":
                continue
            raise ValueError(f"incomplete prior worker attempt {result_path}")
        command = [
            sys.executable,
            str(ROOT / "scripts/codex_annotation/run_model_worker.py"),
            "--model", model,
            "--reasoning-effort", effort,
            "--prompt-file", str(prompt),
            "--run-dir", str(workdir),
            "--session-id", f"fixed-global-v2-{role}-{stem}-a{attempt}",
            "--workdir", str(workdir),
            "--log-dir", str(cascade_dir / "logs" / role / stem),
            "--label", f"{role}-{stem}-a{attempt}",
            "--expected-output", str(expected),
            "--expected-stage", "own_localization_chunk",
            "--expected-package", str(package_path),
            "--promote-to", str(promoted),
            "--result-file", str(result_path),
            "--prompt-var", "PACKAGE_FILE=package.json",
            "--prompt-var", f"OUTPUT_FILE={expected.name}",
            "--prompt-var", f"EXAMPLE_FRAME={event_frame}",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode == 0:
            document = _read(promoted)
            validate_own_localization_decisions(document, _read(package_path))
            return document, package_path
        if completed.returncode != 65:
            raise RuntimeError(f"{role} worker failed for {stem}: {completed.returncode}")
    return None, package_path


def _one_row(document: dict[str, Any], event_id: str) -> dict[str, Any]:
    rows = document.get("decisions")
    if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("event_id") != event_id:
        raise ValueError(f"worker must return exactly the isolated event {event_id}")
    return rows[0]


def _canonicalize_selected_artifacts(
    *,
    selected: dict[str, Any],
    canonical_target: dict[str, Any],
    source_package: Path,
    canonical_run_dir: Path,
) -> dict[str, Any]:
    source_target = _read(source_package)["targets"][0]
    normalized = deepcopy(selected)
    for key in ("macro_review_artifacts", "grid_review_artifacts"):
        source_paths = [source_package.parent / value for value in source_target[key]]
        canonical_paths = [canonical_run_dir / value for value in canonical_target[key]]
        if len(source_paths) != len(canonical_paths) or any(
            _sha256(source) != _sha256(canonical)
            for source, canonical in zip(source_paths, canonical_paths)
        ):
            raise ValueError("canonical localization evidence is not byte-identical")
        normalized[key] = canonical_target[key]
    return normalized


def _cost_report(cascade_dir: Path) -> dict[str, Any]:
    result_paths = sorted((cascade_dir / "results").glob("*/*.json"))
    rows = [_read(path) for path in result_paths]
    by_model: dict[str, dict[str, float | int]] = {}
    by_role: dict[str, dict[str, float | int]] = {}
    for path, row in zip(result_paths, rows):
        model = str(row.get("model"))
        role = path.parent.name
        for table, key in ((by_model, model), (by_role, role)):
            entry = table.setdefault(key, {"attempts": 0, "raw_tokens": 0, "weighted_tokens": 0.0})
            entry["attempts"] += 1
            entry["raw_tokens"] += int(row.get("raw_tokens") or 0)
            entry["weighted_tokens"] += float(row.get("weighted_tokens") or 0.0)
    return {
        "attempts_with_result_sidecars": len(rows),
        "raw_tokens": sum(int(row.get("raw_tokens") or 0) for row in rows),
        "weighted_tokens": sum(float(row.get("weighted_tokens") or 0.0) for row in rows),
        "by_model": by_model,
        "by_role": by_role,
        "result_files": [str(path.relative_to(cascade_dir)) for path in result_paths],
    }


def _prepared_manifest(
    *,
    run_dir: Path,
    cascade_dir: Path,
    policy_path: Path,
    index: dict[str, Any],
) -> dict[str, Any]:
    package_paths = [run_dir / value for value in index["packages"]]
    isolated_paths = [run_dir / value for value in index["isolated_packages"]]
    artifacts: set[Path] = set()
    for package_path in package_paths:
        target = _read(package_path)["targets"][0]
        for key in ("macro_review_artifacts", "grid_review_artifacts"):
            artifacts.update(run_dir / value for value in target[key])
    return {
        "version": 2,
        "policy_sha256": _sha256(policy_path),
        "source_sha256": str(_read(policy_path)["source_sha256"]),
        "package_index_sha256": _sha256(cascade_dir / "packages/package_index.json"),
        "target_count": int(index["target_count"]),
        "package_sha256s": {
            str(path.relative_to(run_dir)): _sha256(path) for path in package_paths
        },
        "isolated_package_sha256s": {
            str(path.relative_to(run_dir)): _sha256(path) for path in isolated_paths
        },
        "evidence_sha256s": {
            str(path.relative_to(run_dir)): _sha256(path) for path in sorted(artifacts)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen global-v2 blind own-localization cascade."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze policy and evidence hashes without launching a worker.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source_path = args.source_file.resolve()
    cascade_dir = args.output_dir.resolve()
    source = _read(source_path)
    if source.get("run_id") != _read(run_dir / "manifest.json").get("run_id"):
        raise ValueError("semantic source run_id does not match localization run")
    if (run_dir.parent / "BATCH_EVALUATED.json").exists():
        raise ValueError("the enclosing multi-video batch has already been evaluated")

    source_sha256 = _sha256(source_path)
    registry_path = run_dir / "own_localization_label_independent/EVALUATED_RUNS.json"
    if registry_path.is_file() and any(
        row.get("source_sha256") == source_sha256
        for row in _read(registry_path).get("evaluated", [])
    ):
        raise ValueError("this semantic source has already been evaluated")

    cascade_dir.mkdir(parents=True, exist_ok=True)
    policy_path = cascade_dir / "frozen_policy.json"
    policy = _frozen_policy(source_path=source_path, source=source)
    if policy_path.exists() and _read(policy_path) != policy:
        raise ValueError("frozen v2 policy changed; use a fresh complete batch")
    atomic_write_json(policy_path, policy)

    index_path = cascade_dir / "packages/package_index.json"
    if index_path.is_file():
        index = _read(index_path)
        if (
            index.get("run_id") != source.get("run_id")
            or index.get("evidence_version") != 2
            or index.get("source_file") != str(source_path)
            or index.get("chunk_size") != 1
        ):
            raise ValueError("prepared localization package index changed")
    else:
        index = prepare_packages(
            run_dir=run_dir,
            source_file=source_path,
            output_dir=cascade_dir / "packages",
            chunk_size=1,
        )
    canonical_packages = [run_dir / value for value in index["packages"]]
    isolated_packages = [run_dir / value for value in index["isolated_packages"]]
    canonical_by_id = {
        _read(path)["targets"][0]["event_id"]: _read(path)["targets"][0]
        for path in canonical_packages
    }
    isolated_by_id = {
        _read(path)["targets"][0]["event_id"]: path for path in isolated_packages
    }
    if set(canonical_by_id) != set(isolated_by_id):
        raise ValueError("canonical and isolated package coverage differs")
    prepared = _prepared_manifest(
        run_dir=run_dir,
        cascade_dir=cascade_dir,
        policy_path=policy_path,
        index=index,
    )
    prepared_path = cascade_dir / "PREPARED.json"
    if args.prepare_only:
        if prepared_path.is_file() and _read(prepared_path) != prepared:
            raise ValueError("frozen v2 evidence changed; use a fresh complete batch")
        atomic_write_json(prepared_path, prepared)
        print(json.dumps({"prepared": str(prepared_path), **prepared}, indent=2))
        return
    if not prepared_path.is_file() or _read(prepared_path) != prepared:
        raise ValueError("run --prepare-only to freeze v2 policy and evidence first")
    ordered_ids = sorted(
        canonical_by_id,
        key=lambda event_id: (
            canonical_by_id[event_id]["event_frame_index"], event_id
        ),
    )

    routing = []
    selected_rows = []
    for event_id in ordered_ids:
        target = canonical_by_id[event_id]
        card = target["card"]
        attempts: list[tuple[str, dict[str, Any]]] = []
        source_packages: dict[str, Path] = {}
        invalid_roles: list[str] = []
        for role in INITIAL_ROLES:
            document, role_package = _run_worker(
                cascade_dir=cascade_dir,
                base_package=isolated_by_id[event_id],
                role=role,
                card=card,
                event_frame=int(target["event_frame_index"]),
            )
            if document is None:
                invalid_roles.append(role)
            else:
                attempts.append((role, _one_row(document, event_id)))
                source_packages[role] = role_package

        route = route_v2_initial(attempts, card)
        if route == "terra_verify":
            document, role_package = _run_worker(
                cascade_dir=cascade_dir,
                base_package=isolated_by_id[event_id],
                role="terra_verify",
                card=card,
                event_frame=int(target["event_frame_index"]),
            )
            if document is None:
                invalid_roles.append("terra_verify")
            else:
                attempts.append(("terra_verify", _one_row(document, event_id)))
                source_packages["terra_verify"] = role_package
            route = route_v2_after_terra_verify(attempts, card)
        if route == "sol_specialized":
            document, role_package = _run_worker(
                cascade_dir=cascade_dir,
                base_package=isolated_by_id[event_id],
                role="sol_specialized",
                card=card,
                event_frame=int(target["event_frame_index"]),
            )
            if document is None:
                invalid_roles.append("sol_specialized")
            else:
                attempts.append(("sol_specialized", _one_row(document, event_id)))
                source_packages["sol_specialized"] = role_package

        selection = select_v2_consensus(attempts, card)
        if selection is None:
            raise ValueError(f"no legal localization decision for {event_id}")
        (selected_role, selected), cluster = selection
        selected_rows.append(
            _canonicalize_selected_artifacts(
                selected=selected,
                canonical_target=target,
                source_package=source_packages[selected_role],
                canonical_run_dir=run_dir,
            )
        )
        routing.append(
            {
                "event_id": event_id,
                "card": card,
                "invalid_roles": invalid_roles,
                "attempts": [
                    {
                        "role": role,
                        "cell": row["cell"],
                        "confidence": row["confidence"],
                        "direct_legal": direct_legal(row, card),
                    }
                    for role, row in attempts
                ],
                "agreement_cluster_indices": cluster,
                "agreement_cluster_roles": [attempts[index][0] for index in cluster],
                "selected_role": selected_role,
                "selected_cell": selected["cell"],
            }
        )

    aggregate_package = {
        "run_id": index["run_id"],
        "stage": "own_localization_targets",
        "evidence_version": 2,
        "target_range": [
            min(canonical_by_id[event_id]["event_frame_index"] for event_id in ordered_ids),
            max(canonical_by_id[event_id]["event_frame_index"] for event_id in ordered_ids) + 1,
        ],
        "targets": [canonical_by_id[event_id] for event_id in ordered_ids],
    }
    prediction = {
        "run_id": index["run_id"],
        "stage": "own_localization_chunk",
        "target_range": aggregate_package["target_range"],
        "annotation_session_id": "fixed-global-v2-own-localization",
        "model": "fixed-global-v2-cascade",
        "reasoning_effort": "mixed",
        "decisions": selected_rows,
    }
    validate_own_localization_decisions(prediction, aggregate_package)
    atomic_write_json(cascade_dir / "aggregate_package.json", aggregate_package)
    atomic_write_json(cascade_dir / "routing.json", {"version": 2, "events": routing})
    atomic_write_json(cascade_dir / "sealed_prediction.json", prediction)
    atomic_write_json(cascade_dir / "cost.json", _cost_report(cascade_dir))
    atomic_write_json(
        cascade_dir / "SEALED.json",
        {
            "version": 2,
            "source_sha256": source_sha256,
            "policy_sha256": _sha256(policy_path),
            "package_sha256": _sha256(cascade_dir / "aggregate_package.json"),
            "routing_sha256": _sha256(cascade_dir / "routing.json"),
            "prediction_sha256": _sha256(cascade_dir / "sealed_prediction.json"),
            "cost_sha256": _sha256(cascade_dir / "cost.json"),
            "event_count": len(selected_rows),
        },
    )
    print(
        json.dumps(
            {
                "sealed_prediction": str(cascade_dir / "sealed_prediction.json"),
                "events": len(selected_rows),
                "cost": _read(cascade_dir / "cost.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
