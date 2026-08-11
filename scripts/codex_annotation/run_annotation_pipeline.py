from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_pipeline import (
    MODEL_PROFILES,
    ModelSpec,
    accumulated_weighted_tokens,
    atomic_write_state,
    completed_job_matches,
    job_fingerprint,
    load_state,
    sha256_file,
)
from cr_bot.annotation_stages import WORKFLOW_VERSION


SCRIPT_DIR = ROOT / "scripts" / "codex_annotation"
PROMPT_DIR = SCRIPT_DIR / "prompts"


def _run_local(args: list[str], *, dry_run: bool) -> None:
    command = [sys.executable, *args]
    if dry_run:
        print(json.dumps({"local": command}))
        return
    subprocess.run(command, cwd=ROOT, check=True)


def _package_paths(run_dir: Path, prefix: str) -> list[Path]:
    return sorted((run_dir / "work_packages").glob(f"{prefix}-??????-??????.json"))


def _run_workers(
    *,
    run_dir: Path,
    state_path: Path,
    state: dict,
    family: str,
    packages: Iterable[Path],
    prompt: Path,
    model_spec: ModelSpec,
    expected_stage: str,
    output_name,
    dry_run: bool,
    max_weighted_tokens: int | None,
    max_new_jobs: int | None,
    run_counter: dict[str, int],
) -> bool:
    output_dir = run_dir / "worker_outputs"
    attempts_dir = run_dir / "worker_attempts"
    results_dir = run_dir / "pipeline_results"
    logs_dir = run_dir / "worker_logs"
    for directory in (output_dir, attempts_dir, results_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for package in packages:
        stable = output_dir / output_name(package)
        job_id = f"{family}:{package.stem}"
        fingerprint = job_fingerprint(
            package=package,
            prompt=prompt,
            model_spec=model_spec,
        )
        if completed_job_matches(
            state["jobs"].get(job_id),
            fingerprint=fingerprint,
            output=stable,
        ):
            print(json.dumps({"job": job_id, "status": "skipped_valid"}))
            continue
        if stable.exists():
            raise ValueError(
                f"{stable} exists without matching resumable state; preserve or "
                "archive it before rerunning this v7 job"
            )
        empty_field = {
            "enemy_overlap_adjudication_chunk": ("candidates", "decisions"),
            "enemy_side_check_chunk": ("candidates", "decisions"),
            "enemy_cards_chunk": ("targets", "cards"),
            "own_slot_intervals_chunk": ("intervals", "decisions"),
            "own_release_review_chunk": ("reviews", "decisions"),
            "enemy_spell_confirmation_chunk": ("reviews", "decisions"),
        }.get(expected_stage)
        if empty_field is not None:
            package_document = json.loads(
                package.read_text(encoding="utf-8")
            )
            input_field, output_field = empty_field
            if package_document.get(input_field) == []:
                document = {
                    "run_id": package_document["run_id"],
                    "stage": expected_stage,
                    "target_range": package_document["target_range"],
                    "annotation_session_id": "deterministic-empty-package",
                    "model": model_spec.model,
                    "reasoning_effort": model_spec.reasoning_effort,
                    output_field: [],
                }
                atomic_write_state(stable, document)
                result = {
                    "status": "succeeded",
                    "label": job_id,
                    "model": model_spec.model,
                    "reasoning_effort": model_spec.reasoning_effort,
                    "session_id": "deterministic-empty-package",
                    "raw_tokens": 0,
                    "cost_multiplier": model_spec.cost_multiplier,
                    "weighted_tokens": 0,
                    "output_sha256": sha256_file(stable),
                }
                state["jobs"][job_id] = {**fingerprint, **result}
                atomic_write_state(state_path, state)
                print(json.dumps({"job": job_id, "status": "empty"}))
                continue
        if (
            max_new_jobs is not None
            and run_counter["new_jobs"] >= max_new_jobs
        ):
            state["status"] = "paused_job_limit"
            atomic_write_state(state_path, state)
            return False
        if (
            max_weighted_tokens is not None
            and accumulated_weighted_tokens(state) >= max_weighted_tokens
        ):
            state["status"] = "paused_budget"
            atomic_write_state(state_path, state)
            return False
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_id = f"{family}-{package.stem}-{timestamp}"
        attempt = attempts_dir / f"{family}-{package.stem}-{timestamp}.json"
        result_path = results_dir / f"{family}-{package.stem}-{timestamp}.json"
        command = [
            sys.executable,
            str(SCRIPT_DIR / "run_model_worker.py"),
            "--model",
            model_spec.model,
            "--reasoning-effort",
            model_spec.reasoning_effort,
            "--prompt-file",
            str(prompt),
            "--run-dir",
            str(run_dir),
            "--session-id",
            session_id,
            "--workdir",
            str(ROOT),
            "--log-dir",
            str(logs_dir),
            "--label",
            job_id.replace(":", "-"),
            "--prompt-var",
            f"PACKAGE_FILE={package}",
            "--prompt-var",
            f"OUTPUT_FILE={attempt}",
            "--expected-output",
            str(attempt),
            "--expected-stage",
            expected_stage,
            "--expected-package",
            str(package),
            "--promote-to",
            str(stable),
            "--result-file",
            str(result_path),
        ]
        if dry_run:
            print(json.dumps({"worker": command}))
            continue
        state["jobs"][job_id] = {
            "status": "running",
            **fingerprint,
            "session_id": session_id,
        }
        atomic_write_state(state_path, state)
        run_counter["new_jobs"] += 1
        completed = subprocess.run(command, cwd=ROOT, check=False)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        state["jobs"][job_id] = {
            **fingerprint,
            **result,
        }
        if stable.is_file():
            state["jobs"][job_id]["output_sha256"] = sha256_file(stable)
        atomic_write_state(state_path, state)
        if completed.returncode == 75 or result["status"] == "paused_quota":
            state["status"] = "paused_quota"
            atomic_write_state(state_path, state)
            return False
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
    return True


def _ensure_evidence_and_packages(
    run_dir: Path,
    *,
    chunk_frames: int,
    dry_run: bool,
) -> None:
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    reviews = run_dir / "reviews"
    enemy_windows = manifest["candidate_discovery"]["enemy_scan_windows"]
    expected_candidate_sheets = [
        reviews / f"verify-{row['candidate_id'].replace(':', '-')}.jpg"
        for row in enemy_windows
    ]
    if not all(path.is_file() for path in expected_candidate_sheets):
        _run_local(
            [
                str(SCRIPT_DIR / "render_annotation_candidates.py"),
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(reviews),
                "--skip-own",
                "--enemy-tile-width",
                "240",
            ],
            dry_run=dry_run,
        )
    marker_path = run_dir / "enemy_marker_candidates.json"
    if not marker_path.is_file():
        _run_local(
            [
                str(SCRIPT_DIR / "discover_enemy_marker_candidates.py"),
                "--run-dir",
                str(run_dir),
            ],
            dry_run=dry_run,
        )
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_ready = all(
            isinstance(row.get("review_artifact"), str)
            and (run_dir / row["review_artifact"]).is_file()
            and isinstance(row.get("focus_review_artifact"), str)
            and (run_dir / row["focus_review_artifact"]).is_file()
            for row in marker["bursts"]
        )
    else:
        marker_ready = False
    if not marker_ready:
        _run_local(
            [
                str(SCRIPT_DIR / "render_enemy_marker_candidates.py"),
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(reviews),
                "--tile-width",
                "360",
            ],
            dry_run=dry_run,
        )
    _run_local(
        [
            str(SCRIPT_DIR / "prepare_semantic_packages.py"),
            "--run-dir",
            str(run_dir),
            "--chunk-frames",
            str(chunk_frames),
            "--skip-own",
        ],
        dry_run=dry_run,
    )
    _run_local(
        [
            str(
                SCRIPT_DIR
                / "prepare_own_slot_interval_packages.py"
            ),
            "--run-dir",
            str(run_dir),
            "--chunk-frames",
            str(chunk_frames),
        ],
        dry_run=dry_run,
    )
    _run_local(
        [
            str(SCRIPT_DIR / "prepare_enemy_marker_packages.py"),
            "--run-dir",
            str(run_dir),
            "--chunk-frames",
            str(chunk_frames),
        ],
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the model-pinned v7 semantic annotation DAG. "
            "This uses local Codex subscriptions, never the OpenAI API."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=sorted(MODEL_PROFILES),
        default="hybrid-accuracy",
    )
    parser.add_argument(
        "--allow-profile-change",
        action="store_true",
        help=(
            "Migrate an existing run to a measured profile; unchanged jobs are "
            "reused by fingerprint and changed stable outputs must be archived."
        ),
    )
    parser.add_argument("--chunk-frames", type=int, default=200)
    parser.add_argument("--max-weighted-tokens", type=int)
    parser.add_argument(
        "--max-new-jobs",
        type=int,
        help="Pause after launching this many previously incomplete workers.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create deterministic evidence and packages, then stop before workers.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("workflow_version") != WORKFLOW_VERSION:
        raise ValueError(
            f"pipeline requires a fresh v{WORKFLOW_VERSION} run"
        )
    profile = MODEL_PROFILES[args.profile]
    if args.max_new_jobs is not None and args.max_new_jobs < 1:
        parser.error("--max-new-jobs must be positive")
    run_counter = {"new_jobs": 0}
    state_path = run_dir / "pipeline_state.json"
    state = load_state(
        state_path,
        run_id=manifest["run_id"],
        profile=args.profile,
        allow_profile_change=args.allow_profile_change,
    )
    previous_chunk_frames = state.get("chunk_frames")
    if (
        previous_chunk_frames is not None
        and previous_chunk_frames != args.chunk_frames
    ):
        raise ValueError(
            "chunk size changed for an existing run; use a fresh run so stale "
            "package partitions and worker outputs cannot be mixed"
        )
    state["chunk_frames"] = args.chunk_frames
    if state.get("status") == "semantic_complete":
        expected = {
            "release_review_sha256": run_dir / "release_review.json",
            "verification_sha256": run_dir / "verification.json",
        }
        if all(
            path.is_file() and state.get(key) == sha256_file(path)
            for key, path in expected.items()
        ):
            print(
                json.dumps(
                    {
                        "status": "semantic_complete",
                        "weighted_tokens": accumulated_weighted_tokens(state),
                        "profile": args.profile,
                        "resume": "nothing_to_do",
                    }
                )
            )
            return
        raise ValueError(
            "semantic_complete state does not match final semantic artifacts"
        )
    if not args.dry_run:
        state["status"] = "running"
        atomic_write_state(state_path, state)

    _ensure_evidence_and_packages(
        run_dir,
        chunk_frames=args.chunk_frames,
        dry_run=args.dry_run,
    )
    if args.prepare_only:
        if not args.dry_run:
            state["status"] = "prepared"
            atomic_write_state(state_path, state)
        print(json.dumps({"status": state["status"], "profile": args.profile}))
        return
    worker_groups = [
        (
            "own-slot-primary",
            "own-slot",
            PROMPT_DIR / "own_slot_intervals_chunk.txt",
            profile["own_slot_primary"],
            "own_slot_intervals_chunk",
            lambda path: path.name,
        ),
        (
            "enemy-spells",
            "enemy-spells",
            PROMPT_DIR / "enemy_spells_chunk.txt",
            profile["enemy_spells"],
            "enemy_spell_onsets_chunk",
            lambda path: path.name,
        ),
    ]
    for family, prefix, prompt, spec, stage, output_name in worker_groups:
        if not _run_workers(
            run_dir=run_dir,
            state_path=state_path,
            state=state,
            family=family,
            packages=_package_paths(run_dir, prefix),
            prompt=prompt,
            model_spec=spec,
            expected_stage=stage,
            output_name=output_name,
            dry_run=args.dry_run,
            max_weighted_tokens=args.max_weighted_tokens,
            max_new_jobs=args.max_new_jobs,
            run_counter=run_counter,
        ):
            print(json.dumps({"status": state["status"], "resume": True}))
            return
    if args.dry_run:
        return
    _run_local(
        [
            str(SCRIPT_DIR / "merge_own_slot_interval_chunks.py"),
            "--run-dir",
            str(run_dir),
        ],
        dry_run=False,
    )
    _run_local(
        [
            str(SCRIPT_DIR / "merge_enemy_onset_chunks.py"),
            "--run-dir",
            str(run_dir),
        ],
        dry_run=False,
    )
    _run_local(
        [
            str(SCRIPT_DIR / "prepare_enemy_decision_packages.py"),
            "--run-dir",
            str(run_dir),
            "--chunk-frames",
            str(args.chunk_frames),
        ],
        dry_run=False,
    )
    _run_local(
        [
            str(
                SCRIPT_DIR
                / "prepare_enemy_overlap_adjudication_packages.py"
            ),
            "--run-dir",
            str(run_dir),
            "--all-candidates",
            "--chunk-frames",
            str(args.chunk_frames),
        ],
        dry_run=False,
    )
    if not _run_workers(
        run_dir=run_dir,
        state_path=state_path,
        state=state,
        family="enemy-existence",
        packages=_package_paths(run_dir, "identity-overlap"),
        prompt=PROMPT_DIR / "enemy_overlap_adjudication_chunk.txt",
        model_spec=profile["enemy_existence"],
        expected_stage="enemy_overlap_adjudication_chunk",
        output_name=lambda path: path.name,
        dry_run=False,
        max_weighted_tokens=args.max_weighted_tokens,
        max_new_jobs=args.max_new_jobs,
        run_counter=run_counter,
    ):
        print(json.dumps({"status": state["status"], "resume": True}))
        return
    _run_local(
        [
            str(SCRIPT_DIR / "prepare_enemy_side_check_packages.py"),
            "--run-dir",
            str(run_dir),
            "--chunk-frames",
            str(args.chunk_frames),
        ],
        dry_run=False,
    )
    if not _run_workers(
        run_dir=run_dir,
        state_path=state_path,
        state=state,
        family="enemy-side-check",
        packages=_package_paths(run_dir, "identity-side"),
        prompt=PROMPT_DIR / "enemy_side_check_chunk.txt",
        model_spec=profile["enemy_side_check"],
        expected_stage="enemy_side_check_chunk",
        output_name=lambda path: path.name,
        dry_run=False,
        max_weighted_tokens=args.max_weighted_tokens,
        max_new_jobs=args.max_new_jobs,
        run_counter=run_counter,
    ):
        print(json.dumps({"status": state["status"], "resume": True}))
        return
    _run_local(
        [
            str(
                SCRIPT_DIR
                / "prepare_enemy_side_escalation_packages.py"
            ),
            "--run-dir",
            str(run_dir),
        ],
        dry_run=False,
    )
    if not _run_workers(
        run_dir=run_dir,
        state_path=state_path,
        state=state,
        family="enemy-side-escalation",
        packages=_package_paths(run_dir, "identity-side-escalation"),
        prompt=PROMPT_DIR / "enemy_side_check_chunk.txt",
        model_spec=profile["enemy_side_escalation"],
        expected_stage="enemy_side_check_chunk",
        output_name=lambda path: path.name,
        dry_run=False,
        max_weighted_tokens=args.max_weighted_tokens,
        max_new_jobs=args.max_new_jobs,
        run_counter=run_counter,
    ):
        print(json.dumps({"status": state["status"], "resume": True}))
        return
    _run_local(
        [
            str(
                SCRIPT_DIR / "merge_enemy_side_escalation_chunks.py"
            ),
            "--run-dir",
            str(run_dir),
        ],
        dry_run=False,
    )
    _run_local(
        [
            str(
                SCRIPT_DIR
                / "prepare_simultaneous_enemy_recovery_packages.py"
            ),
            "--run-dir",
            str(run_dir),
        ],
        dry_run=False,
    )
    if not _run_workers(
        run_dir=run_dir,
        state_path=state_path,
        state=state,
        family="enemy-simultaneous-recovery",
        packages=_package_paths(run_dir, "identity-simultaneous-recovery"),
        prompt=PROMPT_DIR / "enemy_simultaneous_recovery_chunk.txt",
        model_spec=profile["enemy_simultaneous_recovery"],
        expected_stage="enemy_overlap_adjudication_chunk",
        output_name=lambda path: path.name,
        dry_run=False,
        max_weighted_tokens=args.max_weighted_tokens,
        max_new_jobs=args.max_new_jobs,
        run_counter=run_counter,
    ):
        print(json.dumps({"status": state["status"], "resume": True}))
        return
    _run_local(
        [
            str(
                SCRIPT_DIR
                / "merge_simultaneous_enemy_recovery_chunks.py"
            ),
            "--run-dir",
            str(run_dir),
        ],
        dry_run=False,
    )
    _run_local(
        [
            str(SCRIPT_DIR / "merge_enemy_unit_gate_chunks.py"),
            "--run-dir",
            str(run_dir),
        ],
        dry_run=False,
    )
    _run_local(
        [
            str(
                SCRIPT_DIR
                / "prepare_enemy_spell_recovery_packages.py"
            ),
            "--run-dir",
            str(run_dir),
            "--chunk-frames",
            str(args.chunk_frames),
        ],
        dry_run=False,
    )
    for lane in ("left", "right"):
        if not _run_workers(
            run_dir=run_dir,
            state_path=state_path,
            state=state,
            family=f"enemy-spell-recovery-{lane}",
            packages=_package_paths(
                run_dir, f"enemy-spell-recovery-{lane}"
            ),
            prompt=PROMPT_DIR / "enemy_spell_recovery_chunk.txt",
            model_spec=profile["enemy_spell_recovery"],
            expected_stage="enemy_spell_confirmation_chunk",
            output_name=lambda path: path.name,
            dry_run=False,
            max_weighted_tokens=args.max_weighted_tokens,
            max_new_jobs=args.max_new_jobs,
            run_counter=run_counter,
        ):
            print(json.dumps({"status": state["status"], "resume": True}))
            return
    if not _run_workers(
        run_dir=run_dir,
        state_path=state_path,
        state=state,
        family="enemy-spell-boundary",
        packages=_package_paths(run_dir, "enemy-spell-boundary"),
        prompt=PROMPT_DIR / "enemy_spell_boundary_chunk.txt",
        model_spec=profile["enemy_spell_boundary"],
        expected_stage="enemy_spell_confirmation_chunk",
        output_name=lambda path: path.name,
        dry_run=False,
        max_weighted_tokens=args.max_weighted_tokens,
        max_new_jobs=args.max_new_jobs,
        run_counter=run_counter,
    ):
        print(json.dumps({"status": state["status"], "resume": True}))
        return
    _run_local(
        [
            str(SCRIPT_DIR / "merge_enemy_spell_reconciliation.py"),
            "--run-dir",
            str(run_dir),
        ],
        dry_run=False,
    )
    _run_local(
        [
            str(SCRIPT_DIR / "prepare_enemy_identity_targets.py"),
            "--run-dir",
            str(run_dir),
        ],
        dry_run=False,
    )
    _run_local(
        [
            str(SCRIPT_DIR / "render_enemy_identity_targets.py"),
            "--run-dir",
            str(run_dir),
        ],
        dry_run=False,
    )
    _run_local(
        [
            str(SCRIPT_DIR / "render_enemy_identity_roster.py"),
            "--run-dir",
            str(run_dir),
        ],
        dry_run=False,
    )
    _run_local(
        [
            str(SCRIPT_DIR / "prepare_enemy_roster_package.py"),
            "--run-dir",
            str(run_dir),
            "--output-prefix",
            "card-roster",
        ],
        dry_run=False,
    )
    if not _run_workers(
        run_dir=run_dir,
        state_path=state_path,
        state=state,
        family="enemy-card-roster",
        packages=_package_paths(run_dir, "card-roster"),
        prompt=PROMPT_DIR / "enemy_card_roster_assignment.txt",
        model_spec=profile["enemy_card"],
        expected_stage="enemy_cards_chunk",
        output_name=lambda path: path.name,
        dry_run=False,
        max_weighted_tokens=args.max_weighted_tokens,
        max_new_jobs=args.max_new_jobs,
        run_counter=run_counter,
    ):
        print(json.dumps({"status": state["status"], "resume": True}))
        return
    _run_local(
        [
            str(SCRIPT_DIR / "merge_enemy_card_chunks.py"),
            "--run-dir",
            str(run_dir),
            "--package-prefix",
            "card-roster",
        ],
        dry_run=False,
    )
    verification_session = (
        "pipeline-verification-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    model_label = "+".join(
        sorted({spec.model for spec in profile.values()})
    )
    efforts = sorted({spec.reasoning_effort for spec in profile.values()})
    reasoning_label = efforts[0] if len(efforts) == 1 else "mixed"
    _run_local(
        [
            str(SCRIPT_DIR / "merge_semantic_workers.py"),
            "--run-dir",
            str(run_dir),
            "--session-id",
            verification_session,
            "--model-label",
            model_label,
            "--reasoning-effort",
            reasoning_label,
        ],
        dry_run=False,
    )
    release_session = (
        "pipeline-release-review-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    _run_local(
        [
            str(SCRIPT_DIR / "build_release_review_from_own_slots.py"),
            "--run-dir",
            str(run_dir),
            "--session-id",
            release_session,
        ],
        dry_run=False,
    )
    for stage in ("release_review", "verification"):
        _run_local(
            [
                str(SCRIPT_DIR / "checkpoint_annotation_stage.py"),
                "--run-dir",
                str(run_dir),
                "--stage",
                stage,
            ],
            dry_run=False,
        )
    state["status"] = "semantic_complete"
    state["weighted_tokens"] = accumulated_weighted_tokens(state)
    state["release_review_sha256"] = sha256_file(
        run_dir / "release_review.json"
    )
    state["verification_sha256"] = sha256_file(run_dir / "verification.json")
    atomic_write_state(state_path, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "weighted_tokens": state["weighted_tokens"],
                "profile": args.profile,
            }
        )
    )


if __name__ == "__main__":
    main()
