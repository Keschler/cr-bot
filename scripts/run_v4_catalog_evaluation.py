#!/usr/bin/env python3
"""Preflight or execute a frozen five-policy, 40-deck comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON manifest pinning five named actors, hashes, config and ten seeds")
    parser.add_argument("--out", type=Path, help="fresh output directory (required when playing)")
    parser.add_argument("--preflight-only", action="store_true", help="verify artifacts; play zero games")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if not args.preflight_only and args.out is None:
        parser.error("playing requires --out")
    import torch
    from simulator.rl.v4.catalog_evaluation import preflight_comparison, run_catalog_comparison

    torch.set_num_threads(args.torch_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manifest = json.loads(args.manifest.expanduser().read_text())
    if args.preflight_only:
        report = preflight_comparison(manifest, device=args.device)
    else:
        report = run_catalog_comparison(manifest, output_dir=args.out, device=args.device)
    print(json.dumps(report if args.preflight_only else {
        "status": report["status"], "games_played": report["games_played"],
        "report": str(args.out.resolve() / "comparison.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
