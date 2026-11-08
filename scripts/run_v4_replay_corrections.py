#!/usr/bin/env python3
"""Collect -> verify replay branches -> fit sparse corrections -> fresh games."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "prepare", "collect", "search", "fit", "evaluate", "report", "status"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actor-json", type=Path, default=ROOT / "outputs/v4/selected_actor.json")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026102900)
    parser.add_argument("--train-budget", type=int, default=36)
    parser.add_argument("--calibration-budget", type=int, default=12)
    parser.add_argument("--max-per-episode", type=int, default=3)
    parser.add_argument("--max-branches", type=int, default=10)
    parser.add_argument("--discovery-replicas", type=int, default=3)
    parser.add_argument("--confirmation-replicas", type=int, default=6)
    parser.add_argument("--discovery-horizon", type=int, default=128, help="Native 250 ms decisions")
    parser.add_argument("--confirmation-horizon", type=int, default=256, help="Native 250 ms decisions")
    parser.add_argument("--min-lcb", type=float, default=.05)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--deck-ids", nargs="*", default=(), help="Explicit diagnostic collection subset; evaluations still cover all training decks")
    parser.add_argument("--resume", action="store_true", help="Continue run using its sealed plan, ignoring preparation options")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == "status":
        result = {"out": str(out)}
        for name in ("progress", "dataset/manifest", "result"):
            path = out / f"{name}.json"
            if path.exists():
                value = json.loads(path.read_text())
                result[name] = (dict(status=value["status"], completed=len(value["completed"]), total=len(value["jobs"]))
                                if name == "dataset/manifest" else value if name == "progress" else dict(status=value["status"]))
        print(json.dumps(result, indent=2))
        return
    from simulator.rl.v4 import replay_corrections as pipeline
    from simulator.rl.v4.correction_training import CorrectionTrainingConfig
    from simulator.rl.v4.policy_development import atomic_json

    if args.stage in ("prepare", "run") and not args.resume:
        config = pipeline.CorrectionSearchConfig(seed=args.seed, train_budget=args.train_budget,
            calibration_budget=args.calibration_budget, max_per_episode=args.max_per_episode,
            max_branches=args.max_branches, discovery_replicas=args.discovery_replicas,
            confirmation_replicas=args.confirmation_replicas, discovery_horizon=args.discovery_horizon,
            confirmation_horizon=args.confirmation_horizon, min_lcb=args.min_lcb)
        training = CorrectionTrainingConfig(epochs=args.epochs, learning_rate=args.learning_rate)
        plan = pipeline.prepare(out=out, actor_json=args.actor_json, config=config,
                                training=asdict(training), workers=args.workers, deck_ids=args.deck_ids)
    else:
        plan = pipeline.load_plan(out)
    if args.stage == "prepare":
        print(f"Prepared {len(plan['jobs'])} complete actor episodes: {out / 'plan.json'}")
        return
    stages = ("collect", "search", "fit", "evaluate", "report") if args.stage == "run" else (args.stage,)
    for stage in stages:
        pipeline.load_plan(out)
        atomic_json(out / "progress.json", dict(stage=stage, status="running", plan_sha256=plan["plan_sha256"]))
        print(f"[{stage}] {out}", flush=True)
        try:
            result = getattr(pipeline, stage)(out, plan)
        except Exception as exc:
            atomic_json(out / "progress.json", dict(stage=stage, status="failed", error=str(exc),
                                                    plan_sha256=plan["plan_sha256"]))
            raise
        print(f"[{stage}] complete", flush=True)
    print(f"Report: {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
