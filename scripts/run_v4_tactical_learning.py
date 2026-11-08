#!/usr/bin/env python3
"""Implement and compare tactical prediction, defense plans and adaptive practice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "teachers", "collect", "fit", "evaluate", "run", "status"), default="run")
    parser.add_argument("--actor-json", type=Path, default=ROOT / "outputs/v4/defense_repair_20261006/recommended_actor.json")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--fit-workers", type=int, default=3)
    parser.add_argument("--arms", nargs="+", choices=("control", "outcomes", "plans", "curriculum"), default=["control", "outcomes", "plans", "curriculum"])
    parser.add_argument("--arm", choices=("control", "outcomes", "plans", "curriculum"), help="run only this predeclared fit")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="one training air deck; diagnostic only, never reserved evaluation")
    parser.add_argument("--deck-ids", nargs="+", default=[], help="explicit smoke-only training subset")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(arguments)
    if args.arm and args.stage != "fit":
        parser.error("--arm is only supported with --stage fit")
    from simulator.rl.v4 import tactical_experiment as experiment
    from simulator.rl.v4.tactical_training import TacticalTrainingConfig
    out = args.out.expanduser().resolve()
    try:
        if args.stage == "status":
            print(json.dumps(experiment.status(out), indent=2))
            return 0
        if args.stage == "prepare" or (args.stage == "run" and not (out / "plan.json").exists()):
            plan = experiment.prepare_experiment(out=out, actor_json=args.actor_json,
                training=TacticalTrainingConfig(epochs=args.epochs), arms=args.arms, workers=args.workers,
                fit_workers=args.fit_workers, deck_ids=args.deck_ids, smoke=args.smoke)
        else:
            plan = experiment.load_plan(out)
            explicit = {value.split("=", 1)[0] for value in arguments if value.startswith("--")}
            conflicts = []
            for flag, requested, stored in (
                ("--epochs", args.epochs, plan["training"]["epochs"]),
                ("--arms", args.arms, plan["arms"]),
                ("--workers", args.workers, plan["workers"]),
                ("--fit-workers", args.fit_workers, plan["fit_workers"]),
                ("--smoke", args.smoke, plan["smoke"]),
                ("--deck-ids", sorted(args.deck_ids), sorted(plan["train_decks"])),
            ):
                if flag in explicit and requested != stored:
                    conflicts.append(flag)
            if "--actor-json" in explicit:
                record = json.loads(args.actor_json.read_text())
                requested = dict(record.get("checkpoint", record))
                requested["path"] = str(Path(requested["path"]).resolve())
                if requested != plan["checkpoint"]:
                    conflicts.append("--actor-json")
            if conflicts:
                parser.error("explicit options conflict with the sealed plan: " + ", ".join(conflicts))
        if args.stage in ("teachers", "run"):
            experiment.teacher_screen(out, plan)
        if args.stage in ("collect", "run"):
            experiment.collect(out, plan, resume=args.resume)
        if args.stage in ("fit", "run"):
            experiment.fit(out, plan, resume=args.resume, arm=args.arm)
        if args.stage in ("evaluate", "run"):
            experiment.evaluate(out, plan)
        print(json.dumps(experiment.status(out), indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
