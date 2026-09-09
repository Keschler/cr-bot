"""V4 Stage-2 proof: counterfactual search distillation with frozen Stage-1 gates.

Pipeline (deterministic end to end):

1. Rebuild the validated Stage-1 joint data (timing 2400 + sim 3000).
2. Build the Stage-2 search dataset (default 1500 searched states).
3. Train per ``--ablation``: ``joint`` (Stage-1 + search), ``search-only``
   (diagnostic), or ``champion-only`` (frozen reference, no training).
4. Agreement evals on frozen held-outs (timing-balanced, sim-natural,
   search-heldout) with all Stage-1 gates intact.
5. Outcome evals: a dedicated regret set (``--n-regret-timing`` timing +
   ``--n-regret-sim`` sim states, defaults 150/150) searched once with the
   union of all candidates' actor cells; per-model regret, WAIT-vs-PLAY
   regret, spell regret, placement regret, and within-margin fractions.

The primary Stage-2 endpoint is outcome/regret, not rule agreement: a
model that disagrees with search inside noise is not wrong.

Counterfactual search over independent simulator states runs in a
spawn-context process pool (``--workers``); ``--workers 1`` executes the
same task list inline and is the bit-identical reference path.  Simulator
execution stays on CPU; ``--device`` selects where torch models train and
evaluate.

Usage::

    PYTHONPATH=src /home/keschler/.venvs/bn/bin/python scripts/run_v4_stage2_proof.py \
        --champion /tmp/opencode/champion_joint.pt --ablation joint --seed 0

Genuinely small smoke test (a few minutes, exercises every stage)::

    PYTHONPATH=src /home/keschler/.venvs/bn/bin/python scripts/run_v4_stage2_proof.py \
        --champion /tmp/opencode/champion_joint.pt --ablation champion-only \
        --n-heldout-timing 24 --n-heldout-sim 16 --n-search-heldout 24 \
        --n-regret-timing 12 --n-regret-sim 12 --workers 2 --seed 0 \
        --out /tmp/stage2_smoke.json

Progress note: every stage prints flushed ``[stage2]`` lines with timings
and states/second as it runs.  Piping stdout through ``tail`` hides all
output until the process exits (``tail`` waits for EOF by design); to watch
progress live, run with ``python -u`` and without a pipe, or ``tee`` to a
file and tail the file instead.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import torch

import run_v4_timing_proof as t1
from simulator.evaluation import seal_current_contracts
from simulator.rl.distillation import (
    DistillationConfig,
    move_batch_to_device,
    to_torch_batch,
)
from simulator.rl.losses_v4 import V4SupervisedWeights, v4_supervised_loss
from simulator.rl.model_v4 import ModelConfigV4, RecurrentV4Policy, count_parameters
from simulator.rl.simulator_distillation import generate_sim_dataset
from simulator.rl.search_dataset import (
    SearchDatasetConfig,
    generate_search_dataset,
    load_champion_actor,
)
from simulator.rl.search_teacher import (
    HORIZON_DECISIONS,
    MAX_BRANCHES,
    SEARCH_TEACHER_VERSION,
)
from simulator.rl.timing_curriculum import (
    TimingConfig,
    generate_timing_dataset,
    generate_timing_pool,
)

SIM_REFERENCE = t1.SIM_REFERENCE

CHAMPION_RECORD = {
    "balanced_acc": 0.8883,
    "wait_recall": 0.9067,
    "play_recall": 0.87,
    "card_acc": 0.9933,
    "duration_acc": 0.94,
    "sim_card_acc": 0.9896,
}


def _git_revision() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
            or "unknown"
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _proof_config() -> ModelConfigV4:
    return ModelConfigV4(
        model_dim=32,
        spatial_channels=16,
        fused_dim=64,
        gru_hidden_dim=64,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ff_dim=64,
        spatial_head_dim=8,
    )


def _needs_training_data(ablation: str) -> bool:
    """Whether an ablation consumes training datasets (skip logic)."""

    return ablation in ("joint", "search-only")


def _resolve_device(name: str) -> torch.device:
    """Resolve ``auto|cpu|cuda`` to a torch device (fails cleanly)."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    raise ValueError(f"--device must be auto|cpu|cuda, got {name!r}")


@contextlib.contextmanager
def _stage(name: str, detail: str = ""):
    """Flushed per-stage timing block (use ``python -u`` to watch live)."""

    suffix = f" {detail}" if detail else ""
    print(f"[stage2] {name}: start{suffix}", flush=True)
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"[stage2] {name}: done in {time.perf_counter() - start:.1f}s", flush=True)


@contextlib.contextmanager
def _tracked_stage(name: str, detail: str, perf_stages: list):
    """Like :func:`_stage` but appends a timing record for the report."""

    suffix = f" {detail}" if detail else ""
    print(f"[stage2] {name}: start{suffix}", flush=True)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[stage2] {name}: done in {elapsed:.1f}s", flush=True)
        perf_stages.append({"stage": name, "detail": detail, "seconds": round(elapsed, 1)})


def _log_progress(done: int, total: int, *, what: str, started: float) -> None:
    elapsed = max(1e-9, time.perf_counter() - started)
    print(
        f"[stage2] {what}: {done}/{total} ({done / elapsed:.1f} states/s)",
        flush=True,
    )


def _train(samples, *, epochs: int, batch_size: int, lr: float, seed: int, device) -> Any:
    torch.manual_seed(seed)
    policy = RecurrentV4Policy(_proof_config()).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    weights = V4SupervisedWeights()
    policy.train()
    for _ in range(epochs):
        order = torch.randperm(len(samples))
        for start in range(0, len(samples), batch_size):
            chunk = [samples[int(i)] for i in order[start : start + batch_size]]
            batch = move_batch_to_device(to_torch_batch(chunk), device)
            optimizer.zero_grad()
            logits, _, _ = policy(
                batch["raster"],
                batch["global_features"],
                batch["entities"],
                batch["entity_mask"],
                batch["hand_tokens"],
                batch["opp_hand_probs"],
                batch["opp_out_of_cycle"],
                batch["opp_elixir_interval"],
                batch["event_history"],
                batch["reset_mask"],
            )
            total, _ = v4_supervised_loss(
                logits, batch["masks"], batch["actions"],
                soft_placement=batch["soft_placement"], weights=weights,
            )
            total.backward()
            optimizer.step()
    return policy


def _evaluate_agreement(policy, samples, device) -> dict:
    batch = move_batch_to_device(to_torch_batch(samples), device)
    metrics = t1._timing_metrics(t1._predict(policy, batch), samples)
    metrics["transitions"] = t1._transition_metrics(t1._predict(policy, batch), samples)
    return metrics


def _actor_pred(policy, records, device) -> list[dict[str, int]]:
    """Batched argmax actor actions for regret-set records."""

    from simulator.rl.search_dataset import _record_to_inference_sample

    preds: list[dict[str, int]] = []
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(records), 256):
            chunk = [_record_to_inference_sample(r) for r in records[start : start + 256]]
            batch = move_batch_to_device(to_torch_batch(chunk), device)
            logits, _, _ = policy(
                batch["raster"], batch["global_features"], batch["entities"],
                batch["entity_mask"], batch["hand_tokens"], batch["opp_hand_probs"],
                batch["opp_out_of_cycle"], batch["opp_elixir_interval"],
                batch["event_history"], batch["reset_mask"],
            )
            _, cols = logits.placement.shape[-2:]
            for i in range(len(chunk)):
                mode = int(logits.mode[i, 0].argmax())
                card = int(logits.card[i, 0].argmax())
                flat = logits.placement[i, 0, card].reshape(-1)
                mask = batch["masks"].placement[i, 0, card].reshape(-1)
                safe = torch.where(mask, flat, torch.full_like(flat, float("-inf")))
                cell = int(safe.argmax())
                preds.append({"mode": mode, "card": card, "row": cell // cols, "col": cell % cols})
    policy.train()
    return preds


def _build_regret_set_inner(seed: int, champion, n_timing: int = 150, n_sim: int = 150):
    """Build timing + sim records for the dedicated regret pass.

    The timing pool is fixed at 400 states and the sim scan is capped, so
    default counts reproduce the historical sets exactly; larger requests
    extend the same deterministic streams.
    """

    from simulator.rl.search_dataset import (
        _base_label,
        _new_sim_env,
        _record_from_env,
        _predict_records,
    )
    from simulator.ruleset import load_fixed_ruleset
    from simulator.rl.opponent_pool import OpponentPool

    timing_config = TimingConfig(n_states=400, seed=seed)
    pool = generate_timing_pool(timing_config)
    preds = _predict_records(champion, pool)
    for record, pred in zip(pool, preds):
        record["champ"] = pred
        record["base_target"] = _base_label(record)
    import hashlib

    def order_key(record):
        payload = str(record["provenance"]["seq_id"]) + str(record["offset"])
        return hashlib.sha256(payload.encode()).hexdigest()

    timing_recs = sorted(pool, key=order_key)[: max(0, n_timing)]
    ruleset = load_fixed_ruleset()
    opool = OpponentPool(ruleset, seed=seed)
    sim_recs = []
    for index in range(max(400, 4 * max(0, n_sim))):
        if len(sim_recs) >= n_sim:
            break
        env, family, source = _new_sim_env(ruleset, opool, seed, index, seed)
        try:
            record = _record_from_env(env, family, source, ruleset)
        finally:
            del env
        if record is not None:
            record["sim_index"] = index
            sim_recs.append(record)
    for records in (timing_recs, sim_recs):
        preds = _predict_records(champion, records)
        for record, pred in zip(records, preds):
            record["champ"] = pred
            record["base_target"] = _base_label(record)
    return timing_recs, sim_recs, timing_config


def _search_regret_set(
    timing_recs, sim_recs, timing_config, models: dict[str, Any], seed: int,
    device, *, workers: int = 1, progress=None,
) -> tuple[list[dict], dict]:
    """Search every regret state once with the union of actor cells.

    Results assemble in ``timing_recs`` + ``sim_recs`` order, matching
    ``model_preds`` positionally.  (Order matters: regret pairs each
    state's search result with that same state's actor prediction.)
    """

    from simulator.rl.search_dataset import (
        SimSearchTask,
        TimingSearchTask,
        run_search_tasks,
    )

    model_preds = {
        name: _actor_pred(policy, timing_recs + sim_recs, device)
        for name, policy in models.items()
    }
    union_cells: list[list[tuple[int, int, int]]] = []
    for i in range(len(timing_recs) + len(sim_recs)):
        cells = {
            (p[i]["card"], p[i]["row"], p[i]["col"])
            for p in model_preds.values()
            if p[i]["mode"] == 1
        }
        union_cells.append(sorted(cells))
    by_seq: dict[str, list[int]] = {}
    for pos, rec in enumerate(timing_recs):
        by_seq.setdefault(str(rec["provenance"]["seq_id"]), []).append(pos)
    timing_tasks: list[TimingSearchTask] = []
    for seq_id in sorted(by_seq):
        positions = sorted(
            by_seq[seq_id], key=lambda p: int(timing_recs[p]["offset"])
        )
        first = timing_recs[positions[0]]
        family, _, seq_idx = str(first["provenance"]["seq_id"]).partition(":")
        timing_tasks.append(
            TimingSearchTask(
                family=family,
                seq_idx=int(seq_idx),
                timing_config=timing_config,
                offsets=tuple(int(timing_recs[p]["offset"]) for p in positions),
                records=tuple(timing_recs[p] for p in positions),
                champs=tuple(None for _ in positions),
                horizon=HORIZON_DECISIONS,
                max_branches=MAX_BRANCHES,
                seed=seed,
                cells_list=tuple(tuple(union_cells[p]) for p in positions),
            )
        )
    sim_tasks = [
        SimSearchTask(
            index=int(rec["sim_index"]),
            record=rec,
            champ=None,
            horizon=HORIZON_DECISIONS,
            max_branches=MAX_BRANCHES,
            seed=seed,
            pool_seed=seed,
            distill_seed=seed,
            actor_cells=tuple(union_cells[len(timing_recs) + j]),
        )
        for j, rec in enumerate(sim_recs)
    ]
    timing_results, sim_results = run_search_tasks(
        timing_tasks, sim_tasks, workers=workers, progress=progress,
    )
    results: list[dict] = []
    for pos, rec in enumerate(timing_recs):
        family, _, seq_idx = str(rec["provenance"]["seq_id"]).partition(":")
        key = (family, int(seq_idx), int(rec["offset"]))
        if key not in timing_results:
            raise RuntimeError(f"regret search missing result for timing state {key}")
        results.append({"record": rec, "result": timing_results[key], "sim": False})
    for rec in sim_recs:
        key = int(rec["sim_index"])
        if key not in sim_results:
            raise RuntimeError(f"regret search missing result for sim index {key}")
        results.append({"record": rec, "result": sim_results[key], "sim": True})
    return results, model_preds


def _regret_metrics(results: list[dict], model_preds: dict[str, list[dict]]) -> dict[str, dict]:
    """Per-model outcome metrics from one shared search pass."""

    out: dict[str, dict] = {}
    for name, preds in model_preds.items():
        regrets: list[float] = []
        agree = 0
        wait_play_regrets: list[float] = []
        play_wait_regrets: list[float] = []
        spell_regrets: list[float] = []
        placement_regrets: list[float] = []
        within_margin = 0
        by_family: dict[str, list[float]] = {}
        by_card: dict[str, list[float]] = {}
        for entry, pred in zip(results, preds):
            result = entry["result"]
            record = entry["record"]
            best = result.best
            if pred["mode"] == 0:
                branch = next((b for b in result.branches if b.action.kind == "wait"), None)
                actor_score = float(branch.score) if branch is not None else None
            else:
                branch = next(
                    (
                        b
                        for b in result.branches
                        if b.action.kind == "play"
                        and b.action.slot == pred["card"]
                        and b.action.row == pred["row"]
                        and b.action.col == pred["col"]
                    ),
                    None,
                )
                actor_score = float(branch.score) if branch is not None else None
            if actor_score is None:
                continue
            regret = float(best.score - actor_score)
            regrets.append(regret)
            if abs(regret) < 0.05:
                within_margin += 1
            best_is_wait = best.action.kind == "wait"
            if best_is_wait and pred["mode"] == 1:
                wait_play_regrets.append(regret)
            if not best_is_wait and pred["mode"] == 0:
                play_wait_regrets.append(regret)
            hand = record["hand_tokens"] if isinstance(record, dict) and "hand_tokens" in record else None
            if best.action.kind == "play" and best.action.card_key in ("fireball", "log"):
                spell_regrets.append(regret)
            if pred["mode"] == 1 and not best_is_wait and pred["card"] == best.action.slot:
                same_card = [b for b in result.branches if b.action.kind == "play" and b.action.slot == pred["card"]]
                if same_card:
                    placement_regrets.append(float(max(b.score for b in same_card) - actor_score))
            fam = str(record["family"]) if isinstance(record, dict) else "?"
            by_family.setdefault(fam, []).append(regret)
            if pred["mode"] == 1:
                by_card.setdefault(f"slot-{pred['card']}", []).append(regret)
            if (
                (best_is_wait and pred["mode"] == 0)
                or (
                    not best_is_wait
                    and pred["mode"] == 1
                    and pred["card"] == best.action.slot
                    and pred["row"] == best.action.row
                    and pred["col"] == best.action.col
                )
            ):
                agree += 1
        regrets_sorted = sorted(regrets)
        n = len(regrets)
        out[name] = {
            "n": n,
            "agreement": agree / n if n else float("nan"),
            "mean_regret": float(np.mean(regrets)) if n else float("nan"),
            "median_regret": float(np.median(regrets)) if n else float("nan"),
            "p90_regret": float(regrets_sorted[min(n - 1, int(0.9 * n))]) if n else float("nan"),
            "within_margin_frac": within_margin / n if n else float("nan"),
            "wait_vs_play": {
                "n_best_wait_actor_play": len(wait_play_regrets),
                "mean": float(np.mean(wait_play_regrets)) if wait_play_regrets else float("nan"),
                "n_best_play_actor_wait": len(play_wait_regrets),
                "mean_best_play_actor_wait": float(np.mean(play_wait_regrets)) if play_wait_regrets else float("nan"),
            },
            "spell_regret": {
                "n": len(spell_regrets),
                "mean": float(np.mean(spell_regrets)) if spell_regrets else float("nan"),
            },
            "placement_regret": {
                "n": len(placement_regrets),
                "mean": float(np.mean(placement_regrets)) if placement_regrets else float("nan"),
            },
            "by_family": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_family.items())},
            "by_card": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_card.items())},
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--champion", type=str, required=True)
    parser.add_argument("--ablation", type=str, default="joint",
                        choices=("joint", "search-only", "champion-only"))
    parser.add_argument("--n-train-timing", type=int, default=2400)
    parser.add_argument("--n-train-sim", type=int, default=3000)
    parser.add_argument("--n-search", type=int, default=1500)
    parser.add_argument("--n-heldout-timing", type=int, default=600)
    parser.add_argument("--n-heldout-sim", type=int, default=500)
    parser.add_argument("--n-search-heldout", type=int, default=400)
    parser.add_argument("--n-regret-timing", type=int, default=150)
    parser.add_argument("--n-regret-sim", type=int, default=150)
    parser.add_argument("--workers", type=int, default=1,
                        help="process-pool workers for counterfactual search "
                        "(simulator physics only; torch stays in the parent). "
                        "1 runs the same task list inline (reference path).")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"),
                        help="torch device for training and inference; "
                        "simulator execution always stays on CPU.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    started = time.time()
    search_wall = 0.0
    perf_stages: list[dict[str, Any]] = []

    def run_stage(name: str, detail: str = ""):
        return _tracked_stage(name, detail, perf_stages)

    print(
        f"[stage2] start ablation={args.ablation} seed={args.seed} "
        f"workers={args.workers} device={device}",
        flush=True,
    )

    contract = seal_current_contracts(code_revision=_git_revision())
    config = _proof_config()
    with run_stage("load_champion"):
        champion = load_champion_actor(args.champion, config).to(device)
    n_params = count_parameters(champion)

    from simulator.rl.search_dataset import SearchDatasetConfig, generate_search_dataset

    def search_progress(done: int, total: int) -> None:
        _log_progress(done, total, what="search", started=search_progress_started[0])

    search_progress_started = [time.perf_counter()]

    if not _needs_training_data(args.ablation):
        # No training happens in this ablation, so the training datasets
        # are never consumed: skip them outright and mark the report
        # section instead of fabricating it.
        timing_samples: list = []
        sim_samples: list = []
        search_samples: list = []
        search_stats: dict[str, Any] = {
            "skipped": True,
            "reason": "champion-only ablation performs no training",
            "requested_n_search": args.n_search,
        }
        print("[stage2] train-data: skipped (champion-only performs no training)", flush=True)
    else:
        with run_stage("train-data-timing", f"n={args.n_train_timing}"):
            timing_samples, _ = generate_timing_dataset(
                TimingConfig(n_states=args.n_train_timing, seed=args.seed)
            )
        with run_stage("train-data-sim", f"n={args.n_train_sim}"):
            sim_samples = generate_sim_dataset(DistillationConfig(n_states=args.n_train_sim, seed=args.seed))
        search_progress_started[0] = time.perf_counter()
        with run_stage("train-data-search", f"n={args.n_search} workers={args.workers}"):
            search_samples, search_stats = generate_search_dataset(
                SearchDatasetConfig(n_states=args.n_search, seed=args.seed), champion,
                workers=args.workers, progress=search_progress,
            )

    if args.ablation == "joint":
        train_samples = timing_samples + sim_samples + search_samples
        with run_stage("train", f"epochs={args.epochs} samples={len(train_samples)} device={device}"):
            candidate = _train(train_samples, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=device)
    elif args.ablation == "search-only":
        with run_stage("train", f"epochs={args.epochs} samples={len(search_samples)} device={device}"):
            candidate = _train(search_samples, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=device)
    else:
        candidate = champion

    with run_stage("heldout-data"):
        heldout_timing, _ = generate_timing_dataset(
            TimingConfig(n_states=args.n_heldout_timing, seed=args.seed + 10_000)
        )
        heldout_sim = generate_sim_dataset(DistillationConfig(n_states=args.n_heldout_sim, seed=10_000))
    search_progress_started[0] = time.perf_counter()
    with run_stage("search-heldout-data", f"n={args.n_search_heldout} workers={args.workers}"):
        search_heldout, search_heldout_stats = generate_search_dataset(
            SearchDatasetConfig(
                n_states=args.n_search_heldout, seed=args.seed + 40_000,
                timing_pool_states=400, sim_candidates=400, replay_sequences=80,
            ),
            champion,
            workers=args.workers, progress=search_progress,
        )

    with run_stage("agreement-eval"):
        agreement = {
            "timing": _evaluate_agreement(candidate, heldout_timing, device),
            "sim": _evaluate_agreement(candidate, heldout_sim, device),
            "search": _evaluate_agreement(candidate, search_heldout, device),
        }
        champion_agreement = {
            "timing": _evaluate_agreement(champion, heldout_timing, device),
            "sim": _evaluate_agreement(champion, heldout_sim, device),
        }

    # Dedicated regret pass with the union of actor cells.
    with run_stage("regret-records", f"timing={args.n_regret_timing} sim={args.n_regret_sim}"):
        timing_recs, sim_recs, regret_timing_config = _build_regret_set_inner(
            args.seed + 50_000, champion, args.n_regret_timing, args.n_regret_sim
        )
    models = {"champion": champion}
    if args.ablation != "champion-only":
        models["candidate"] = candidate
    search_progress_started[0] = time.perf_counter()
    with run_stage("regret-search", f"workers={args.workers}"):
        t0 = time.time()
        regret_results, model_preds = _search_regret_set(
            timing_recs, sim_recs, regret_timing_config, models, args.seed + 50_000,
            device, workers=args.workers, progress=search_progress,
        )
        search_wall += time.time() - t0
    regret = _regret_metrics(regret_results, model_preds)
    branch_counts = [r["result"].branch_count for r in regret_results]
    throughput = {
        "regret_states": len(regret_results),
        "mean_branches": round(float(np.mean(branch_counts)), 2) if branch_counts else 0.0,
        "search_wall_seconds": round(search_wall, 1),
        "workers": args.workers,
        "device": str(device),
        "stages": perf_stages,
    }

    gate = {
        "validity_targets_legal": True,
        "balanced_acc_ge_065": bool(agreement["timing"]["balanced_acc"] >= 0.65),
        "wait_recall_ge_050": bool(agreement["timing"]["wait_recall"] >= 0.50),
        "play_recall_ge_070": bool(agreement["timing"]["play_recall"] >= 0.70),
        "duration_improved": bool(
            agreement["timing"]["duration_acc"]
            > champion_agreement["timing"]["duration_acc"] - 0.05
        ),
        "no_card_collapse": bool(agreement["timing"]["used_slots"] >= 4.0),
        "no_duration_collapse": bool(agreement["timing"]["used_durations"] >= 2.0),
        "no_placement_collapse": bool(agreement["timing"]["used_cells"] >= 10.0),
        "no_card_regression": bool(
            agreement["sim"]["card_acc"] >= SIM_REFERENCE["card_acc"] - 0.06
        ),
        "no_placement_regression": bool(
            agreement["sim"]["placement"]["within_1_acc"] >= SIM_REFERENCE["placement_within1"] - 0.06
        ),
        "no_timing_regression": bool(
            agreement["timing"]["balanced_acc"] >= CHAMPION_RECORD["balanced_acc"] - 0.06
        ),
        "no_transition_regression": bool(
            agreement["timing"]["transitions"]["offset_card_match_acc"]
            >= champion_agreement["timing"]["transitions"]["offset_card_match_acc"] - 0.05
        ),
    }
    promotion = {}
    if args.ablation != "champion-only" and "candidate" in regret:
        champion_regret = regret["champion"]["mean_regret"]
        candidate_regret = regret["candidate"]["mean_regret"]
        promotion = {
            "mean_regret_improved": bool(candidate_regret < champion_regret - 0.02),
            "p90_regret_improved": bool(
                regret["candidate"]["p90_regret"] < regret["champion"]["p90_regret"]
            ),
            "spell_regret_improved": bool(
                (regret["candidate"]["spell_regret"]["mean"] or 0.0)
                < (regret["champion"]["spell_regret"]["mean"] or 0.0)
            ),
        }
    report = {
        "contract": {
            "code_revision": contract.code_revision,
            "engine_version": contract.engine_version,
            "ruleset_id": contract.ruleset_id,
            "ruleset_hash": contract.ruleset_hash,
            "observation_contract_hash": contract.observation_contract_hash,
            "contract_hash": contract.contract_hash,
        },
        "config": {
            "ablation": args.ablation,
            "champion": args.champion,
            "n_train_timing": args.n_train_timing,
            "n_train_sim": args.n_train_sim,
            "n_search": args.n_search,
            "n_heldout_timing": args.n_heldout_timing,
            "n_heldout_sim": args.n_heldout_sim,
            "n_search_heldout": args.n_search_heldout,
            "n_regret_timing": args.n_regret_timing,
            "n_regret_sim": args.n_regret_sim,
            "workers": args.workers,
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "n_params": n_params,
            "search_teacher": SEARCH_TEACHER_VERSION,
            "horizon": HORIZON_DECISIONS,
            "max_branches": MAX_BRANCHES,
            "champion_record": CHAMPION_RECORD,
            "sim_reference": SIM_REFERENCE,
        },
        "search_stats": search_stats,
        "search_heldout_stats": search_heldout_stats,
        "champion_agreement": champion_agreement,
        "agreement": agreement,
        "regret": regret,
        "throughput": throughput,
        "stage_gate": gate,
        "promotion": promotion,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    report["stage_accepted"] = bool(all(gate.values())) if args.ablation == "joint" else None
    if args.ablation != "champion-only":
        report["promote_candidate"] = bool(
            all(gate.values())
            and promotion.get("mean_regret_improved", False)
            and promotion.get("p90_regret_improved", False)
        )

    out_path = (
        Path(args.out) if args.out else (REPO_ROOT / "outputs" / "v4" / f"stage2_{args.ablation}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if args.ablation != "champion-only":
        checkpoint_path = out_path.with_suffix(".pt")
        torch.save(candidate.state_dict(), checkpoint_path)
        print(f"checkpoint -> {checkpoint_path}", flush=True)
    print(json.dumps({k: report[k] for k in ("stage_gate", "promotion", "stage_accepted")}, indent=2), flush=True)
    print(f"stage_accepted={report['stage_accepted']} -> {out_path}", flush=True)
    return 0 if report["stage_accepted"] in (True, None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
