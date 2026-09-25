"""Picklable replay tasks and serial or spawn-based execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from simulator.ruleset import load_fixed_ruleset
from simulator.rl.foundation.opponent_pool import OpponentPool
from simulator.rl.v4.simulator_teacher import TeacherError
from simulator.rl.v4.timing_curriculum import replay_timing_sequences
from .records import _new_sim_env
from .targets import _search_live


@dataclass(frozen=True)
class TimingSearchTask:
    """Replay one timing sequence, searching the requested offsets.

    All fields are picklable; the worker rebuilds an isolated simulator
    instance from ``timing_config`` + ``(family, seq_idx)`` and runs the
    exact same ``_search_live`` call the serial path makes.  ``champs``
    parallels ``offsets`` (``None`` entries allowed); ``cells_list``
    optionally overrides candidate cells per offset (regret union cells).
    """

    family: str
    seq_idx: int
    timing_config: Any
    offsets: tuple[int, ...]
    records: tuple[dict[str, Any], ...]
    champs: tuple[Any, ...]
    horizon: int
    max_branches: int
    seed: int
    cells_list: tuple[Any, ...] = ()
    continuation_mode: str = "wait"
    opponent_strategy: str = "deterministic-cycle"


@dataclass(frozen=True)
class SimSearchTask:
    """Rebuild one sim-natural reset and search it."""

    index: int
    record: dict[str, Any]
    champ: Any
    horizon: int
    max_branches: int
    seed: int
    pool_seed: int
    distill_seed: int
    actor_cells: Any = ()
    sim_family_mix: Any = None
    continuation_mode: str = "wait"
    opponent_strategy: str = "deterministic-cycle"


def _worker_init() -> None:
    """Keep per-process thread pools at one thread per worker.

    Simulator stepping is single-threaded Python plus small-array numpy;
    without this, every worker would inherit (spawn) or share (fork) a
    full-size BLAS/torch thread pool and oversubscribe the machine.
    Environment variables are best-effort for libraries initialized after
    this runs; the torch calls take effect immediately.
    """

    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass


def _run_timing_task(task: TimingSearchTask) -> list[tuple[int, dict[str, Any]]]:
    """Replay one sequence in this process, searching requested offsets.

    Mirrors the serial ``replay_timing_sequences`` semantics exactly: an
    unknown family raises, while per-sequence build errors (including a
    failing hook capture) keep partial results and move on.
    """

    from ..timing_curriculum import TIMING_FAMILIES

    if task.family not in TIMING_FAMILIES:
        raise TeacherError(f"unknown timing family for replay: {task.family!r}")
    wanted = {int(offset) for offset in task.offsets}
    cell_lookup = {
        int(offset): (tuple(c) if c is not None else None)
        for offset, c in zip(task.offsets, task.cells_list)
    } if task.cells_list else {}
    by_offset = {int(offset): (record, champ) for offset, record, champ in zip(task.offsets, task.records, task.champs)}
    found: list[tuple[int, dict[str, Any]]] = []

    def hook(env: Any, record: dict[str, Any], offset: int) -> None:
        if int(offset) not in wanted:
            return
        rec, champ = by_offset[int(offset)]
        search = _search_live(
            env, rec, champ,
            horizon=task.horizon, max_branches=task.max_branches, seed=task.seed,
            actor_cells=cell_lookup.get(int(offset)),
            continuation_mode=task.continuation_mode,
            opponent_strategy=task.opponent_strategy,
        )
        found.append((int(offset), search))

    try:
        replay_timing_sequences(
            task.timing_config, [(task.family, int(task.seq_idx))], live_hook=hook
        )
    except TeacherError:
        pass
    found.sort(key=lambda item: item[0])
    return found


def _run_sim_task(task: SimSearchTask) -> dict[str, Any]:
    """Rebuild one sim reset in this process and search it."""

    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=task.pool_seed)
    mix = dict(task.sim_family_mix) if task.sim_family_mix else None
    env, _, _ = _new_sim_env(ruleset, pool, task.seed, int(task.index), task.distill_seed, mix)
    try:
        return _search_live(
            env, task.record, task.champ,
            horizon=task.horizon, max_branches=task.max_branches, seed=task.seed,
            actor_cells=tuple(task.actor_cells) if task.actor_cells else None,
            continuation_mode=task.continuation_mode,
            opponent_strategy=task.opponent_strategy,
        )
    finally:
        del env


def run_search_tasks(
    timing_tasks: Sequence[TimingSearchTask],
    sim_tasks: Sequence[SimSearchTask],
    *,
    workers: int = 1,
    progress: Any = None,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[int, dict[str, Any]]]:
    """Execute search tasks, joining results in canonical task order.

    Returns ``(timing_results, sim_results)`` where timing keys are
    ``(family, seq_idx, offset)`` and sim keys are reset indices.  With
    ``workers=1`` tasks run inline in order (the reference path); with
    ``workers>1`` a spawn-context process pool distributes them while the
    parent reports ``progress(done, total)`` as results arrive.  Either way
    the joined dictionaries are identical for identical inputs.
    """

    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    timing_tasks = list(timing_tasks)
    sim_tasks = list(sim_tasks)
    total = len(timing_tasks) + len(sim_tasks)
    timing_out: dict[tuple[str, int, int], dict[str, Any]] = {}
    sim_out: dict[int, dict[str, Any]] = {}
    if total == 0:
        return timing_out, sim_out
    done = 0

    def note() -> None:
        if progress is not None:
            progress(done, total)

    if workers == 1:
        for task in timing_tasks:
            for offset, search in _run_timing_task(task):
                timing_out[(task.family, int(task.seq_idx), int(offset))] = search
            done += 1
            note()
        for task in sim_tasks:
            sim_out[int(task.index)] = _run_sim_task(task)
            done += 1
            note()
        return timing_out, sim_out

    import concurrent.futures
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_worker_init
    ) as pool_executor:
        timing_futures = [pool_executor.submit(_run_timing_task, task) for task in timing_tasks]
        sim_futures = [pool_executor.submit(_run_sim_task, task) for task in sim_tasks]
        for task, future in zip(timing_tasks, timing_futures):
            for offset, search in future.result():
                timing_out[(task.family, int(task.seq_idx), int(offset))] = search
            done += 1
            note()
        for task, future in zip(sim_tasks, sim_futures):
            sim_out[int(task.index)] = future.result()
            done += 1
            note()
    return timing_out, sim_out

