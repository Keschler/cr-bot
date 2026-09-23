"""Run a resumable V4 command under a persistent wall-clock budget.

Dollar budgets require an explicit provider stop command: terminating Python
does not stop a rented machine's billing. Commands are JSON argv arrays, never
shell strings. The cleanup reserve is excluded from the compute deadline.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def argv_json(value: str) -> list[str]:
    result = json.loads(value)
    if not isinstance(result, list) or not result or any(not isinstance(x, str) or not x for x in result):
        raise argparse.ArgumentTypeError("command must be a nonempty JSON array of strings")
    return result


def budget_seconds(budget: float | None, rate: float | None, runtime: float | None, reserve: float) -> float:
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("reserve must be finite and nonnegative")
    if budget is not None:
        if rate is None or not math.isfinite(rate) or rate <= 0 or not math.isfinite(budget) or budget <= 0:
            raise ValueError("budget and hourly rate must be finite and positive")
        total = budget / rate * 3600
    else:
        if runtime is None or not math.isfinite(runtime) or runtime <= 0:
            raise ValueError("max runtime must be finite and positive")
        total = runtime
    if total <= reserve:
        raise ValueError("cleanup reserve must be smaller than total budget")
    return total


def terminate_group(process: subprocess.Popen, grace: float = 5) -> None:
    # Kill descendants even if the group leader has already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_hook(command: list[str], timeout: float) -> int:
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    finally:
        terminate_group(process)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--budget-usd", type=float)
    group.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--hourly-rate", type=float)
    parser.add_argument("--reserve-seconds", type=float, default=300)
    parser.add_argument("--archive-command-json", type=argv_json)
    parser.add_argument("--stop-command-json", type=argv_json)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply a command after --")
    if args.budget_usd is not None and not args.stop_command_json:
        parser.error("dollar budget requires --stop-command-json to stop provider billing")
    try:
        total = budget_seconds(args.budget_usd, args.hourly_rate, args.max_runtime_seconds, args.reserve_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock = args.state.with_suffix(".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another budget guard owns this state")
    config = {"command": command, "budget_usd": args.budget_usd, "hourly_rate": args.hourly_rate,
              "max_runtime_seconds": args.max_runtime_seconds, "reserve_seconds": args.reserve_seconds,
              "archive_command": args.archive_command_json, "stop_command": args.stop_command_json}
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state["config"] != config:
            parser.error("budget configuration changed; use a new explicitly budgeted state file")
        if state.get("status") == "complete":
            return 0
    else:
        now = time.time()
        state = {"config": config, "started_at": now, "deadline_epoch": now + total - args.reserve_seconds,
                 "budget_end_epoch": now + total}
    state.update(status="running", pid=os.getpid())
    atomic_json(args.state, state)
    env = dict(os.environ, V4_DEADLINE_EPOCH=str(state["deadline_epoch"]))
    process = None
    code = 124
    interrupted = False
    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    try:
        if time.time() < state["deadline_epoch"]:
            process = subprocess.Popen(command, env=env, start_new_session=True)
            while process.poll() is None and time.time() < state["deadline_epoch"] and not interrupted:
                time.sleep(min(.25, max(0, state["deadline_epoch"] - time.time())))
            code = process.poll()
            if code is None:
                code = 130 if interrupted else 124
            terminate_group(process)
        state.update(status="complete" if code == 0 else "budget_exhausted" if code == 124 else "failed",
                     returncode=code, finished_at=time.time())
    except Exception as exc:
        state.update(status="failed", error=str(exc), returncode=1)
        code = 1
    finally:
        if process is not None and process.poll() is None:
            terminate_group(process)
        atomic_json(args.state, state)
        # Save status before a provider hook can power down the machine.
        for name, hook in (("archive", args.archive_command_json), ("stop", args.stop_command_json)):
            if not hook:
                continue
            remaining = max(1, state["budget_end_epoch"] - time.time())
            timeout = min(60, remaining) if name == "stop" else max(1, remaining - 30)
            try:
                result = run_hook(hook, timeout)
                state[name + "_returncode"] = result
                if result:
                    code = 1
            except Exception as exc:
                state[name + "_error"] = str(exc)
                code = 1
            if code and state["status"] == "complete":
                state["status"] = "cleanup_failed"
            atomic_json(args.state, state)
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
