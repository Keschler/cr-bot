#!/usr/bin/env python3
"""Run paired cycle-teacher or varied-opponent policy development screens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("cycle", "diversity"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--actor-json", type=Path, help="explicit actor path/hash/model_config for diversity screen")
    parser.add_argument("--styles", nargs="+", default=[])
    parser.add_argument("--style-split", choices=("train", "dev"), default="train")
    parser.add_argument("--gate-json", type=Path, help="predeclared cycle gate JSON; absent means diagnostic only")
    args = parser.parse_args(argv)
    from simulator.rl.v4.policy_development import DevelopmentArm, run_development_screen, training_opponents
    try:
        if args.experiment == "cycle":
            if args.actor_json:
                parser.error("cycle compares frozen teacher versions; omit --actor-json")
            arms = [DevelopmentArm("defense", "defense"), DevelopmentArm("cycle", "cycle")]
        else:
            if not args.actor_json or not args.styles:
                parser.error("diversity requires --actor-json and explicit --styles")
            record = json.loads(args.actor_json.read_text())
            checkpoint = record.get("checkpoint", record)
            arms = [DevelopmentArm("baseline", "actor", checkpoint["path"], checkpoint["sha256"], checkpoint["model_config"])]
        gate = json.loads(args.gate_json.read_text()) if args.gate_json else None
        report = run_development_screen(out=args.out, arms=arms,
            opponents=training_opponents(args.styles, style_split=args.style_split),
            seeds=args.seeds, workers=args.workers, gate=gate)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
