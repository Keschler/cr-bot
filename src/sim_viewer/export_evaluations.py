"""Export hash-verified reproductions of saved native evaluations for playback.

Run as ``python -m sim_viewer.export_evaluations --help``. This adapter lives
outside the native evaluator's sealed source scope and never changes its action
sampling, public observation history, recurrent memory, or opponent identity.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
from typing import Any, Sequence
from uuid import uuid4


EXPORT_SCHEMA = "native-evaluation-viewer-export-1"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _worker_init() -> None:
    # Set these before Torch/NumPy are imported in spawned workers.
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    import torch

    torch.set_num_threads(1)


def _load_sealed_record(path: Path, dev: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    row = _read(path)
    if not isinstance(row, dict) or not all(isinstance(row.get(key), dict) for key in ("job", "native_record", "diagnostics")):
        raise ValueError(f"expected a saved native development match: {path}")
    manifest_path = path.parent.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"saved match needs its original screen manifest: {manifest_path}")
    manifest = _read(manifest_path)
    identity = manifest.get("identity", {})
    if (not isinstance(identity, dict) or dev.digest(identity) != manifest.get("run_hash")
            or row.get("run_hash") != manifest.get("run_hash")):
        raise ValueError(f"saved screen manifest/run identity differs: {path}")
    source = identity.get("source")
    if not isinstance(source, dict) or source != dev.runtime_source_fingerprint():
        raise ValueError(f"native source seal differs from the original evaluation: {path}")
    digests_path = manifest_path.with_name("raw_digests.json")
    if not digests_path.is_file() or _read(digests_path).get(str(path.relative_to(manifest_path.parent))) != _sha(path):
        raise ValueError(f"original raw match digest seal differs: {path}")
    job, native = row["job"], row["native_record"]
    arm = job.get("arm", {})
    if (arm.get("kind") != "actor" or not arm.get("model_config", {}).get("fixed_wait")
            or native.get("deterministic_policy") is not False
            or native.get("decision_interval_us") != 250_000):
        raise ValueError("export requires a native stochastic actor with fixed 250 ms actions")
    if arm not in identity.get("arms", []):
        raise ValueError(f"saved actor differs from the screen manifest: {path}")
    dev.validate_match(row, job)
    if _sha(Path(arm["checkpoint"])) != arm["sha256"]:
        raise ValueError("actor checkpoint hash differs from the original evaluation")
    return row, source


def export_record(record: str | Path, out: str | Path) -> dict[str, Any]:
    """Reproduce one original job and publish snapshots only after exact checks.

    Instrumentation is scoped to this synchronous worker. Use separate spawned
    processes for concurrency; monkeypatching native hooks is not thread safe.
    """
    from simulator.rl.v4 import policy_development as dev
    from simulator.viewer.projection import project_policy, project_state
    from simulator.viewer.recording import JsonlRecorder

    path = Path(record).expanduser().resolve()
    directory = Path(out).expanduser().resolve()
    row, source = _load_sealed_record(path, dev)
    job, expected = row["job"], row["native_record"]
    arm = job["arm"]
    original_sha = _sha(path)
    provenance = dict(
        schema=EXPORT_SCHEMA, original_record_path=str(path), original_record_sha256=original_sha,
        checkpoint_sha256=arm["sha256"], native_source_sha256=source["sha256"],
        run_hash=row["run_hash"], job_sha256=dev.digest(job),
    )
    name = "__".join((path.parent.parent.name, arm["name"], job["cell_id"], expected["result"], original_sha[:12]))
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{name}.jsonl"
    receipt_path = directory / f"{name}.receipt.json"
    if output.exists() or receipt_path.exists():
        if not output.is_file() or not receipt_path.is_file():
            raise ValueError(f"incomplete export exists; choose a new output directory: {output}")
        receipt = _read(receipt_path)
        if (receipt.get("provenance") != provenance or receipt.get("exact_verified") is not True
                or receipt.get("recording_sha256") != _sha(output)):
            raise ValueError(f"existing export provenance or recording hash differs: {output}")
        return {**receipt, "reused": True}

    evaluation = dict(
        deck_id=expected["deck_id"], archetype=expected["archetype"],
        style=expected["opponent"].get("style"), seed=expected["seed"], result=expected["result"],
        checkpoint_sha256=arm["sha256"], original_record_sha256=original_sha, exact_verified=True,
    )
    metadata = dict(
        **evaluation, evaluation=evaluation, original_record_path=str(path),
        checkpoint=arm["checkpoint"], model_config=arm["model_config"],
        policy_seed=expected["policy_seed"], controller_seed=expected["controller_seed"],
        opponent=expected["opponent"], learner_deck=expected["learner_deck"],
        opponent_deck=expected["opponent"]["deck"], decision_interval_us=250_000,
        engine_version=expected["engine_version"], ruleset_hash=expected["ruleset_hash"],
        native_source_sha256=source["sha256"], sampling="stochastic", recurrent_history="carry",
        native_record=expected,
    )
    # The .tmp extension keeps unverified frames out of viewer discovery.
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    actor = dev._actor(arm["checkpoint"], arm["sha256"], dev.canonical(arm["model_config"]))
    original_rollout = actor.rollout_sample
    original_step = dev._TraceEnvironment.step_v2
    recorder = None
    current_policy = None
    trace_env = None
    event_cursor = 0
    index = 0

    def rollout(*args: Any, **kwargs: Any) -> Any:
        nonlocal current_policy
        result = original_rollout(*args, **kwargs)
        masks = args[9] if len(args) > 9 else kwargs["masks"]
        current_policy = project_policy(result[0], masks, result[1])
        return result

    def step(trace: Any, actions: Any) -> Any:
        nonlocal recorder, current_policy, trace_env, event_cursor, index
        if current_policy is None:
            raise RuntimeError("native decision has no captured policy distribution")
        if trace_env is not None and trace_env is not trace:
            raise RuntimeError("native evaluation unexpectedly constructed multiple matches")
        trace_env = trace
        state = trace.env.state
        if recorder is None:
            metadata.update(ruleset_id=state.ruleset_id, tick_us=trace.engine.ruleset.tick_us,
                            decision_interval_ticks=trace.decision_interval_ticks)
            recorder = JsonlRecorder(temporary, match_id=name, source="evaluation", metadata=metadata)
        frame = project_state(state, trace.engine.ruleset, index=index,
                              events=state.events[event_cursor:], actions=actions, policy=current_policy)
        event_cursor = len(state.events)
        current_policy = None
        result = original_step(trace, actions)
        frame["reward"] = float(result.rewards[0])
        recorder.append(frame)
        index += 1
        return result

    actor.rollout_sample = rollout
    dev._TraceEnvironment.step_v2 = step
    try:
        try:
            actual = dev.execute_match(job)
        finally:
            actor.rollout_sample = original_rollout
            dev._TraceEnvironment.step_v2 = original_step
        differences = [key for key in ("native_record", "diagnostics") if actual.get(key) != row[key]]
        if actual.get("job") != job:
            differences.append("job")
        if differences:
            raise ValueError(f"native replay differs from saved evaluation ({', '.join(differences)}): {path}")
        if dev.runtime_source_fingerprint() != source:
            raise ValueError("native source changed during export")
        if _sha(path) != original_sha or _sha(Path(arm["checkpoint"])) != arm["sha256"]:
            raise ValueError("original record or actor changed during export")
        if recorder is None or trace_env is None or index != expected["decisions"] or current_policy is not None:
            raise ValueError("captured frame count differs from original native decisions")
        state = trace_env.env.state
        if state.state_hash() != expected["final_state_hash"]:
            raise ValueError("captured final state differs from original native evaluation")
        recorder.append(project_state(state, trace_env.engine.ruleset, index=index, events=state.events[event_cursor:]))
        recorder.finish(expected["terminal_reason"], winner=expected["winner"])
        receipt = dict(
            provenance=provenance, recording_path=str(output), recording_sha256=_sha(temporary),
            original_record_path=str(path), frame_count=index + 1, evaluation=evaluation,
            exact_verified=True,
        )
        temporary.replace(output)
        try:
            _atomic_json(receipt_path, receipt)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        return {**receipt, "reused": False}
    finally:
        temporary.unlink(missing_ok=True)


def collect_records(records: Sequence[str | Path], screens: Sequence[str | Path]) -> list[Path]:
    paths = [Path(value).expanduser().resolve() for value in records]
    for value in screens:
        directory = Path(value).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"screen arm directory does not exist: {directory}")
        found = sorted(directory.glob("*.json"))
        if not found:
            raise ValueError(f"screen must be an arm directory containing raw match JSON: {directory}")
        paths.extend(found)
    result = sorted(set(paths))
    if not result:
        raise ValueError("provide raw match JSON files or --screen directories")
    if any(not path.is_file() for path in result):
        raise ValueError("an original match JSON file does not exist")
    return result


def export_records(records: Sequence[Path], out: Path, *, workers: int = 3) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    completed = []
    # A failed invocation must not leave a previous manifest claiming success.
    manifest_path = out / "manifest.json"
    _atomic_json(manifest_path, dict(schema=EXPORT_SCHEMA, status="running", records=[str(p) for p in records]))
    try:
        if workers == 1:
            _worker_init()
            for path in records:
                completed.append(export_record(path, out))
                print(f"verified viewer matches {len(completed)}/{len(records)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_worker_init) as pool:
                futures = {pool.submit(export_record, path, out): path for path in records}
                for future in as_completed(futures):
                    completed.append(future.result())
                    print(f"verified viewer matches {len(completed)}/{len(records)}", flush=True)
        manifest = dict(schema=EXPORT_SCHEMA, status="complete", exact_verified=True,
                        matches=sorted(completed, key=lambda item: item["original_record_path"]), count=len(completed))
        _atomic_json(manifest_path, manifest)
        return manifest
    except BaseException as error:
        _atomic_json(manifest_path, dict(schema=EXPORT_SCHEMA, status="failed", error=str(error),
                                       requested_count=len(records), verified_count=len(completed), matches=completed))
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="*", help="Saved raw match JSON files with their original screen manifest")
    parser.add_argument("--screen", action="append", default=[], help="Raw arm directory, e.g. screen_train/control; repeat to combine screens")
    parser.add_argument("--out", required=True, type=Path, help="Viewer JSONL, provenance receipts, and export manifest directory")
    parser.add_argument("--workers", type=int, default=3, help="Spawned native replay workers (default: 3; use 1 for synchronous replay)")
    args = parser.parse_intermixed_args(argv)
    try:
        paths = collect_records(args.records, args.screen)
        manifest = export_records(paths, args.out, workers=args.workers)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Verified {manifest['count']} original evaluations. Manifest: {args.out.resolve() / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
