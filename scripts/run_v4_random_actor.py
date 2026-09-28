#!/usr/bin/env python3
"""Create a seeded untrained V4 actor for a pure reward-only PPO baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from simulator.rl.v4.frontier import frontier_model_config
from simulator.rl.v4.model_v4 import RecurrentV4Policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    output = args.out.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if output.exists() or manifest_path.exists():
        parser.error("output already exists; choose a new --out path")
    torch.manual_seed(args.seed)
    actor = RecurrentV4Policy(frontier_model_config())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        torch.save(actor.state_dict(), temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "version": "v4-random-actor-v1",
        "seed": args.seed,
        "actor_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "model_config": asdict(frontier_model_config()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"actor": str(output), **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
