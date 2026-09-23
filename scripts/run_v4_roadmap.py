"""Run the V4 roadmap as a resumable, fail-closed phase pipeline.

The roadmap contains experiments that are already executable and experiments
whose proposed protocol is not implemented in this checkout.  This runner
keeps that distinction explicit.  It can run the broad frontier collection,
record an externally supplied implementation for a gated phase, and run the
final PPO/full-match phase only after every earlier gate is complete.

The default plan intentionally stops at the first unsupported phase.  It does
not turn the existing Stage-2 or PPO smoke scripts into claims that the
search-policy, retention, burn-in, or Expert-Iteration experiments were run.

Examples::

    # Inspect the commands and gates without starting compute.
    outputs/venv/bin/python scripts/run_v4_roadmap.py \
        --run-dir outputs/v4/roadmap_20260921 --dry-run

    # Run or resume the broad collection phase.
    outputs/venv/bin/python scripts/run_v4_roadmap.py \
        --run-dir outputs/v4/roadmap_20260921 --phases broad_generation \
        --workers 26

    # Inspect a previous run.
    outputs/venv/bin/python scripts/run_v4_roadmap.py \
        --run-dir outputs/v4/roadmap_20260921 --status

An unsupported phase can be supplied as an explicit external command with
``--phase-command phase='command ...'``.  The command must create that phase's
declared artifact.  Such a phase is recorded as ``complete_external`` so the
report never presents it as implemented by this repository.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ROADMAP_SCHEMA_VERSION = "v4-roadmap-v1"

PHASE_ORDER: tuple[str, ...] = (
    "broad_generation",
    "teacher_audit",
    "gru_context",
    "supervised_ablation",
    "expert_iteration",
    "architecture",
    "final_evaluation",
)


class RoadmapError(RuntimeError):
    """Raised for a protocol or artifact-gate violation."""


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """Static contract for one roadmap phase."""

    name: str
    description: str
    dependencies: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    supported: bool = True
    consumes_sealed: bool = False
    final_only: bool = False
    implementation_note: str = ""


@dataclass(frozen=True, slots=True)
class RoadmapOptions:
    """Command settings that are persisted in the run state."""

    run_dir: Path
    checkpoint: Path
    python: str = sys.executable
    train_matches_per_archetype: int = 90
    sealed_matches_per_archetype: int = 36
    candidates_per_match: int = 16
    replicas: int = 8
    branches: int = 7
    horizon: int = 128
    max_decisions: int = 1200
    max_entries_per_split: int = 1200
    workers: int = 3
    required_train_entries: int = 1000
    required_sealed_entries: int = 400
    ppo_updates: str = "0,128"
    ppo_eval_matches: int = 80
    ppo_eval_max_decisions: int = 1200
    ppo_choice_replicas: int = 8
    ppo_device: str = "auto"

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["run_dir"] = str(self.run_dir)
        result["checkpoint"] = str(self.checkpoint)
        return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, default=str) + "\n"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical(payload), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: Path, *, base: Path = REPO_ROOT) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _phase_output(run_dir: Path, phase: str) -> Path:
    return run_dir / phase


def phase_specs(options: RoadmapOptions) -> dict[str, PhaseSpec]:
    """Return the explicit phase contracts.

    The unsupported phases remain in the plan so status output shows exactly
    which protocol component blocks the final evaluation.
    """

    broad = _phase_output(options.run_dir, "broad_generation")
    final = _phase_output(options.run_dir, "final_evaluation")
    return {
        "broad_generation": PhaseSpec(
            name="broad_generation",
            description="Mine disjoint train/sealed frontier candidates.",
            artifacts=(
                "broad_generation/manifest.json",
                "broad_generation/summary.json",
                "broad_generation/frontier_train.pt",
                "broad_generation/frontier_sealed.pt",
                "broad_generation/train_verdicts.json",
                "broad_generation/sealed_verdicts.json",
            ),
            implementation_note=(
                "Executable through scripts/run_v4_frontier_generation.py; "
                "this is collection only, not teacher validation."
            ),
        ),
        "teacher_audit": PhaseSpec(
            name="teacher_audit",
            description="Audit paired search labels at 16/32 replicas and multiple horizons.",
            dependencies=("broad_generation",),
            artifacts=("teacher_audit/REPORT.json",),
            supported=False,
            implementation_note=(
                "No checked-in command produces the required multi-horizon report. "
                "frontier_eval.evaluate_candidate is reusable for an external runner."
            ),
        ),
        "gru_context": PhaseSpec(
            name="gru_context",
            description="Compare zero-state and 32/64-step recurrent burn-in.",
            dependencies=("teacher_audit",),
            artifacts=("gru_context/REPORT.json",),
            supported=False,
            implementation_note=(
                "Existing sequence batching is reusable, but frontier entries reset "
                "hidden state and do not store observation prefixes."
            ),
        ),
        "supervised_ablation": PhaseSpec(
            name="supervised_ablation",
            description="Run hard/soft target, replay, and old-policy-KL ablations.",
            dependencies=("gru_context",),
            artifacts=("supervised_ablation/REPORT.json",),
            supported=False,
            implementation_note=(
                "v4_supervised_loss has soft placement only; the four-way target/KL "
                "protocol and retention sampler are not implemented."
            ),
        ),
        "expert_iteration": PhaseSpec(
            name="expert_iteration",
            description="Run three policy-rollout/search/distillation rounds.",
            dependencies=("supervised_ablation",),
            artifacts=("expert_iteration/round_3.json",),
            supported=False,
            implementation_note=(
                "No runner persists current-policy on-policy states, searches them, "
                "and aggregates D0 through D3."
            ),
        ),
        "architecture": PhaseSpec(
            name="architecture",
            description="Compare compact, medium, and large models on identical data.",
            dependencies=("expert_iteration",),
            artifacts=("architecture/REPORT.json",),
            supported=False,
            implementation_note=(
                "ModelConfigV4 is configurable, but no same-dataset architecture "
                "training/three-seed driver exists."
            ),
        ),
        "final_evaluation": PhaseSpec(
            name="final_evaluation",
            description="Run the final PPO curve and paired full-match evaluation.",
            dependencies=("architecture",),
            artifacts=("final_evaluation/ppo_curve/learning_curve.json",),
            consumes_sealed=True,
            final_only=True,
            implementation_note=(
                "Uses the existing controlled PPO curve. It consumes sealed entries "
                "only here and must not be used as evidence for missing earlier phases."
            ),
        ),
    }


def _command_for_phase(options: RoadmapOptions, phase: str) -> list[str] | None:
    """Build the command for the phases implemented in this checkout."""

    python = str(options.python)
    if phase == "broad_generation":
        output = _phase_output(options.run_dir, phase)
        return [
            python,
            str(REPO_ROOT / "scripts" / "run_v4_frontier_generation.py"),
            "--checkpoint",
            str(options.checkpoint),
            "--out",
            str(output),
            "--train-matches-per-archetype",
            str(options.train_matches_per_archetype),
            "--sealed-matches-per-archetype",
            str(options.sealed_matches_per_archetype),
            "--candidates-per-match",
            str(options.candidates_per_match),
            "--replicas",
            str(options.replicas),
            "--branches",
            str(options.branches),
            "--horizon",
            str(options.horizon),
            "--max-decisions",
            str(options.max_decisions),
            "--max-entries-per-split",
            str(options.max_entries_per_split),
            "--workers",
            str(options.workers),
        ]
    if phase == "final_evaluation":
        output = _phase_output(options.run_dir, phase) / "ppo_curve"
        train_entries = _phase_output(options.run_dir, "broad_generation") / "frontier_train.pt"
        sealed_entries = _phase_output(options.run_dir, "broad_generation") / "frontier_sealed.pt"
        return [
            python,
            str(REPO_ROOT / "scripts" / "run_v4_learning_curve.py"),
            "--checkpoint",
            str(options.checkpoint),
            "--train-entries",
            str(train_entries),
            "--sealed-entries",
            str(sealed_entries),
            "--out",
            str(output),
            "--updates",
            str(options.ppo_updates),
            "--eval-matches",
            str(options.ppo_eval_matches),
            "--eval-max-decisions",
            str(options.ppo_eval_max_decisions),
            "--choice-replicas",
            str(options.ppo_choice_replicas),
            "--device",
            str(options.ppo_device),
        ]
    return None


def _parse_phase_commands(values: Sequence[str]) -> dict[str, str]:
    commands: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise RoadmapError(
                f"--phase-command must use PHASE=COMMAND, got {raw!r}"
            )
        name, command = raw.split("=", 1)
        name = name.strip()
        command = command.strip()
        if name not in PHASE_ORDER:
            raise RoadmapError(f"unknown phase in --phase-command: {name!r}")
        if not command:
            raise RoadmapError(f"empty external command for phase {name!r}")
        if name in commands:
            raise RoadmapError(f"duplicate external command for phase {name!r}")
        commands[name] = command
    return commands


def _format_external_command(
    template: str, options: RoadmapOptions, phase: str
) -> list[str]:
    values = {
        "repo": str(REPO_ROOT),
        "run_dir": str(options.run_dir),
        "phase_dir": str(_phase_output(options.run_dir, phase)),
        "checkpoint": str(options.checkpoint),
        "train_entries": str(
            _phase_output(options.run_dir, "broad_generation") / "frontier_train.pt"
        ),
        "sealed_entries": str(
            _phase_output(options.run_dir, "broad_generation") / "frontier_sealed.pt"
        ),
    }
    try:
        rendered = template.format(**values)
    except (KeyError, ValueError) as exc:
        raise RoadmapError(
            f"invalid placeholder in external command for {phase!r}: {exc}"
        ) from exc
    return shlex.split(rendered)


def _artifact_paths(run_dir: Path, spec: PhaseSpec) -> tuple[Path, ...]:
    return tuple(run_dir / relative for relative in spec.artifacts)


def artifacts_complete(run_dir: Path, spec: PhaseSpec) -> bool:
    """Return true only when every declared artifact exists and is non-empty."""

    paths = _artifact_paths(run_dir, spec)
    return bool(paths) and all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _status_satisfies(status: str | None) -> bool:
    return status in {"complete", "complete_external"}


def _sealed_token(path: str) -> bool:
    """Detect an existing sealed dataset path, excluding count/label options."""

    normalized = path.replace("\\", "/")
    return normalized.endswith("frontier_sealed.pt") or "--sealed-entries" in normalized


def validate_sealed_guard(spec: PhaseSpec, command: Sequence[str]) -> None:
    """Reject sealed-data use outside the final phase."""

    command_text = " ".join(str(item) for item in command)
    if spec.consumes_sealed and not spec.final_only:
        raise RoadmapError(f"phase {spec.name!r} is marked as sealed-consuming but is not final-only")
    if any(_sealed_token(token) for token in command):
        if not spec.final_only:
            raise RoadmapError(
                f"refusing to run {spec.name!r}: command consumes sealed data before final_evaluation"
            )
    if spec.final_only and not spec.consumes_sealed and _sealed_token(command_text):
        raise RoadmapError(
            f"phase {spec.name!r} command mentions sealed data but its contract does not allow it"
        )


def _load_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "roadmap_state.json"
    if not path.is_file():
        return {
            "schema_version": ROADMAP_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "phases": {},
        }
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != ROADMAP_SCHEMA_VERSION:
        raise RoadmapError(f"invalid roadmap state file: {path}")
    payload.setdefault("phases", {})
    return payload


def _save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    _write_json_atomic(run_dir / "roadmap_state.json", state)


def _data_counts(run_dir: Path) -> dict[str, int] | None:
    summary = run_dir / "broad_generation" / "summary.json"
    if not summary.is_file():
        return None
    try:
        payload = _read_json(summary)
        counts = payload.get("counts", {})
        return {"train": int(counts.get("train", 0)), "sealed": int(counts.get("sealed", 0))}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _phase_record(
    state: Mapping[str, Any], spec: PhaseSpec, run_dir: Path
) -> dict[str, Any]:
    stored = state.get("phases", {}).get(spec.name, {})
    if not isinstance(stored, Mapping):
        stored = {}
    status = str(stored.get("status", "pending"))
    if status == "running":
        status = "stale_running"
    if status in {"complete", "complete_external"} and not artifacts_complete(run_dir, spec):
        status = "stale_missing_artifacts"
    if status in {"pending", "stale_running", "stale_missing_artifacts"} and artifacts_complete(run_dir, spec):
        status = "complete_artifacts"
    result = dict(stored)
    result["status"] = status
    result["artifacts_present"] = artifacts_complete(run_dir, spec)
    result["artifacts"] = [str(path) for path in _artifact_paths(run_dir, spec)]
    return result


def status_report(
    options: RoadmapOptions,
    *,
    external_commands: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable status report without running anything."""

    state = _load_state(options.run_dir) if options.run_dir.is_dir() else {"phases": {}}
    specs = phase_specs(options)
    rows: dict[str, Any] = {}
    for name in PHASE_ORDER:
        spec = specs[name]
        row = _phase_record(state, spec, options.run_dir)
        row.update(
            {
                "supported_in_repo": bool(spec.supported),
                "external_command_configured": name in (external_commands or {}),
                "dependencies": list(spec.dependencies),
                "consumes_sealed": spec.consumes_sealed,
                "final_only": spec.final_only,
                "implementation_note": spec.implementation_note,
            }
        )
        rows[name] = row
    counts = _data_counts(options.run_dir)
    return {
        "schema_version": ROADMAP_SCHEMA_VERSION,
        "run_dir": str(options.run_dir),
        "checkpoint": str(options.checkpoint),
        "data_counts": counts,
        "required_data_counts": {
            "train": options.required_train_entries,
            "sealed": options.required_sealed_entries,
        },
        "phases": rows,
        "sealed_policy": "sealed data may be consumed only by final_evaluation",
    }


def _parse_phases(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return PHASE_ORDER
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise RoadmapError("--phases cannot be empty")
    unknown = sorted(set(names) - set(PHASE_ORDER))
    if unknown:
        raise RoadmapError(f"unknown roadmap phases: {unknown}")
    order = {name: index for index, name in enumerate(PHASE_ORDER)}
    return tuple(sorted(set(names), key=order.__getitem__))


def _dependency_status(
    state: Mapping[str, Any], specs: Mapping[str, PhaseSpec], phase: str, run_dir: Path
) -> tuple[bool, str]:
    for dependency in specs[phase].dependencies:
        record = _phase_record(state, specs[dependency], run_dir)
        if not _status_satisfies(str(record.get("status"))):
            return False, f"dependency {dependency} is {record.get('status', 'pending')}"
    return True, ""


def _validate_final_data_gate(options: RoadmapOptions, run_dir: Path) -> None:
    counts = _data_counts(run_dir)
    if counts is None:
        raise RoadmapError("final evaluation requires broad_generation/summary.json")
    if counts["train"] < options.required_train_entries:
        raise RoadmapError(
            f"final evaluation requires at least {options.required_train_entries} train entries; "
            f"found {counts['train']}"
        )
    if counts["sealed"] < options.required_sealed_entries:
        raise RoadmapError(
            f"final evaluation requires at least {options.required_sealed_entries} sealed entries; "
            f"found {counts['sealed']}"
        )


def _command_for(
    options: RoadmapOptions,
    phase: str,
    external_commands: Mapping[str, str],
) -> tuple[list[str] | None, bool]:
    if phase in external_commands:
        return _format_external_command(external_commands[phase], options, phase), True
    return _command_for_phase(options, phase), False


def _print_plan(
    options: RoadmapOptions,
    selected: Sequence[str],
    specs: Mapping[str, PhaseSpec],
    external_commands: Mapping[str, str],
) -> None:
    print(f"run_dir: {options.run_dir}")
    for name in selected:
        spec = specs[name]
        command, external = _command_for(options, name, external_commands)
        if command is None:
            state = "UNSUPPORTED"
            detail = spec.implementation_note
        else:
            state = "EXTERNAL" if external else "READY"
            detail = " ".join(shlex.quote(item) for item in command)
        print(f"[{state}] {name}: {spec.description}")
        print(f"  depends: {', '.join(spec.dependencies) or 'none'}")
        print(f"  artifacts: {', '.join(spec.artifacts) or 'none'}")
        print(f"  {detail}")


def run_plan(
    options: RoadmapOptions,
    selected: Sequence[str],
    external_commands: Mapping[str, str],
) -> int:
    """Execute selected phases, resuming from valid artifacts."""

    options.run_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(options.run_dir)
    state["options"] = options.as_dict()
    state["phase_order"] = list(PHASE_ORDER)
    specs = phase_specs(options)
    _save_state(options.run_dir, state)

    for phase in selected:
        spec = specs[phase]
        current = _phase_record(state, spec, options.run_dir)
        if _status_satisfies(str(current.get("status"))):
            print(f"[roadmap] {phase}: already complete ({current['status']})")
            continue

        dependency_ok, dependency_reason = _dependency_status(
            state, specs, phase, options.run_dir
        )
        if not dependency_ok:
            record = dict(state.get("phases", {}).get(phase, {}))
            record.update({"status": "blocked", "message": dependency_reason})
            state.setdefault("phases", {})[phase] = record
            _save_state(options.run_dir, state)
            print(f"[roadmap] {phase}: BLOCKED — {dependency_reason}")
            return 2

        command, external = _command_for(options, phase, external_commands)
        if command is None:
            record = dict(state.get("phases", {}).get(phase, {}))
            record.update(
                {
                    "status": "unsupported",
                    "message": spec.implementation_note,
                    "supported_in_repo": False,
                }
            )
            state.setdefault("phases", {})[phase] = record
            _save_state(options.run_dir, state)
            print(f"[roadmap] {phase}: UNSUPPORTED — {spec.implementation_note}")
            return 2

        validate_sealed_guard(spec, command)
        if spec.final_only:
            _validate_final_data_gate(options, options.run_dir)

        log_path = options.run_dir / "logs" / f"{phase}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        phase_dir = _phase_output(options.run_dir, phase)
        phase_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "status": "running",
            "started_at": _utc_now(),
            "command": command,
            "log": str(log_path),
            "external": external,
            "supported_in_repo": bool(spec.supported),
            "sealed_guard": {
                "consumes_sealed": spec.consumes_sealed,
                "final_only": spec.final_only,
            },
        }
        state.setdefault("phases", {})[phase] = record
        _save_state(options.run_dir, state)
        print(f"[roadmap] {phase}: START")
        print(f"[roadmap] log: {log_path}")
        try:
            env = os.environ.copy()
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                item for item in (str(REPO_ROOT), str(REPO_ROOT / "src"), current_pythonpath) if item
            )
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"[{_utc_now()}] command: {shlex.join(command)}\n")
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        except OSError as exc:
            record.update({"status": "failed", "returncode": None, "error": str(exc)})
            state["phases"][phase] = record
            _save_state(options.run_dir, state)
            print(f"[roadmap] {phase}: FAILED to start — {exc}")
            return 1

        present = artifacts_complete(options.run_dir, spec)
        if completed.returncode != 0:
            record.update(
                {
                    "status": "failed",
                    "returncode": int(completed.returncode),
                    "finished_at": _utc_now(),
                    "artifacts_present": present,
                }
            )
            state["phases"][phase] = record
            _save_state(options.run_dir, state)
            print(f"[roadmap] {phase}: FAILED (exit {completed.returncode})")
            return 1
        if not present:
            record.update(
                {
                    "status": "failed",
                    "returncode": 0,
                    "finished_at": _utc_now(),
                    "artifacts_present": False,
                    "error": "command exited successfully but did not create all declared artifacts",
                }
            )
            state["phases"][phase] = record
            _save_state(options.run_dir, state)
            print(f"[roadmap] {phase}: FAILED — expected artifacts are missing")
            return 1

        record.update(
            {
                "status": "complete_external" if external else "complete",
                "returncode": 0,
                "finished_at": _utc_now(),
                "artifacts_present": True,
                "implementation_note": spec.implementation_note,
            }
        )
        state["phases"][phase] = record
        _save_state(options.run_dir, state)
        print(f"[roadmap] {phase}: COMPLETE")

    print("[roadmap] selected phases complete")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    default_run = REPO_ROOT / "outputs" / "v4" / f"roadmap_{date.today():%Y%m%d}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=REPO_ROOT / "outputs" / "v4" / "stage2_12k_v2_e12_s12.pt",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--phases", default="all", help="all or comma-separated phase names")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--json", action="store_true", help="print status/plan as JSON where supported")
    parser.add_argument(
        "--phase-command", action="append", default=[], metavar="PHASE=COMMAND",
        help="override an unsupported phase with an external command; supports {run_dir}, {phase_dir}, {checkpoint}, {train_entries}, {sealed_entries}",
    )
    parser.add_argument("--train-matches-per-archetype", type=int, default=90)
    parser.add_argument("--sealed-matches-per-archetype", type=int, default=36)
    parser.add_argument("--candidates-per-match", type=int, default=16)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--branches", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument("--max-entries-per-split", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--required-train-entries", type=int, default=1000)
    parser.add_argument("--required-sealed-entries", type=int, default=400)
    parser.add_argument("--ppo-updates", default="0,128")
    parser.add_argument("--ppo-eval-matches", type=int, default=80)
    parser.add_argument("--ppo-eval-max-decisions", type=int, default=1200)
    parser.add_argument("--ppo-choice-replicas", type=int, default=8)
    parser.add_argument("--ppo-device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _options_from_args(args: argparse.Namespace) -> RoadmapOptions:
    integer_names = (
        "train_matches_per_archetype", "sealed_matches_per_archetype", "candidates_per_match",
        "replicas", "branches", "horizon", "max_decisions", "max_entries_per_split",
        "workers", "required_train_entries", "required_sealed_entries", "ppo_eval_matches",
        "ppo_eval_max_decisions", "ppo_choice_replicas",
    )
    for name in integer_names:
        if int(getattr(args, name)) < 1:
            raise RoadmapError(f"--{name.replace('_', '-')} must be positive")
    return RoadmapOptions(
        run_dir=_resolve(args.run_dir),
        checkpoint=_resolve(args.checkpoint),
        python=str(args.python_executable),
        train_matches_per_archetype=int(args.train_matches_per_archetype),
        sealed_matches_per_archetype=int(args.sealed_matches_per_archetype),
        candidates_per_match=int(args.candidates_per_match),
        replicas=int(args.replicas),
        branches=int(args.branches),
        horizon=int(args.horizon),
        max_decisions=int(args.max_decisions),
        max_entries_per_split=int(args.max_entries_per_split),
        workers=int(args.workers),
        required_train_entries=int(args.required_train_entries),
        required_sealed_entries=int(args.required_sealed_entries),
        ppo_updates=str(args.ppo_updates),
        ppo_eval_matches=int(args.ppo_eval_matches),
        ppo_eval_max_decisions=int(args.ppo_eval_max_decisions),
        ppo_choice_replicas=int(args.ppo_choice_replicas),
        ppo_device=str(args.ppo_device),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        options = _options_from_args(args)
        external_commands = _parse_phase_commands(args.phase_command)
        selected = _parse_phases(args.phases)
        specs = phase_specs(options)
        if args.status:
            report = status_report(options, external_commands=external_commands)
            if args.json:
                print(_canonical(report), end="")
            else:
                print(f"run_dir: {report['run_dir']}")
                print(f"data_counts: {report['data_counts']} required={report['required_data_counts']}")
                for name in PHASE_ORDER:
                    row = report["phases"][name]
                    print(f"[{row['status']}] {name} — artifacts={row['artifacts_present']}")
                    if row.get("message"):
                        print(f"  {row['message']}")
            return 0
        if args.dry_run:
            _print_plan(options, selected, specs, external_commands)
            if args.json:
                report = status_report(options, external_commands=external_commands)
                print(_canonical(report), end="")
            return 0
        if not options.checkpoint.is_file():
            raise RoadmapError(f"checkpoint does not exist: {options.checkpoint}")
        return run_plan(options, selected, external_commands)
    except (RoadmapError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
