#!/usr/bin/env python3
"""Save match-opening entries for reward-only PPO, without frontier search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from simulator.rl.v4.raw_ppo import RAW_PPO_STARTS_VERSION, make_match_start_entries
from simulator.rl.v4.real_decks import catalog_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-seed", type=int, default=20261001)
    parser.add_argument("--matches-per-archetype", type=int, default=20)
    parser.add_argument("--profile", choices=("pinned", "mixed", "real"), default="pinned")
    parser.add_argument(
        "--policy-opponent", type=Path, action="append", default=[],
        help="frozen V4 actor checkpoint for mirror matches; may be repeated",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.out.expanduser().resolve()
    if output.exists() or output.with_suffix(".json").exists():
        parser.error("output already exists; choose a new --out path")
    try:
        entries = make_match_start_entries(
            pool_seed=args.pool_seed,
            matches_per_archetype=args.matches_per_archetype,
            profile=args.profile,
            policy_opponents=args.policy_opponent,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        torch.save(entries, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "version": RAW_PPO_STARTS_VERSION,
        "pool_seed": args.pool_seed,
        "matches_per_archetype": args.matches_per_archetype,
        "profile": args.profile,
        "real_deck_catalog_sha256": (
            catalog_sha256() if args.profile == "real" else None
        ),
        "policy_opponents": [
            {"checkpoint": str(path.expanduser().resolve()),
             "sha256": hashlib.sha256(path.expanduser().resolve().read_bytes()).hexdigest()}
            for path in args.policy_opponent
        ],
        "entry_count": len(entries),
        "entry_sha256": digest,
        "opponent_archetypes": sorted({entry["opponent"]["archetype"] for entry in entries}),
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"entries": str(output), **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
