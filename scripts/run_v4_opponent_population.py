#!/usr/bin/env python3
"""Save train-only raw PPO openings from a pinned learned Hog opponent population."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True, help="version-1 pinned population JSON")
    parser.add_argument("--pool-seed", type=int, default=20261007)
    parser.add_argument("--episodes", type=int, required=True, help="at least one opening per training member")
    parser.add_argument("--train-feedback", "--feedback", dest="train_feedback", type=Path,
                        help="JSON list of complete train-only TrainMatchupResult records")
    parser.add_argument("--coverage-fraction", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True, help="new raw-opening .pt file")
    args = parser.parse_args(argv)
    output = args.out.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if output.suffix != ".pt":
        parser.error("--out must end in .pt")
    if output.exists() or manifest_path.exists():
        parser.error("output already exists; choose a new --out path")

    import torch

    from simulator.rl.v4.opponent_population import (
        OpponentPopulation, TrainMatchupResult, make_population_match_start_entries,
    )
    from simulator.rl.v4.raw_ppo import RAW_PPO_STARTS_VERSION

    try:
        population_path = args.population.expanduser().resolve(strict=True)
        population_bytes = population_path.read_bytes()
        population = OpponentPopulation.from_dict(json.loads(population_bytes))
        feedback_path = None
        feedback_bytes = None
        feedback = ()
        if args.train_feedback is not None:
            feedback_path = args.train_feedback.expanduser().resolve(strict=True)
            feedback_bytes = feedback_path.read_bytes()
            records = json.loads(feedback_bytes)
            expected = {field.name for field in fields(TrainMatchupResult)}
            if not isinstance(records, list) or any(
                not isinstance(record, dict) or set(record) != expected for record in records
            ):
                raise ValueError("train feedback must be a JSON list of exact TrainMatchupResult records")
            feedback = tuple(TrainMatchupResult(**record) for record in records)
        entries = make_population_match_start_entries(
            population, pool_seed=args.pool_seed, episodes=args.episodes,
            train_results=feedback, coverage_fraction=args.coverage_fraction,
        )
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(entries, handle)
        # Exclusive creation protects existing artifacts even if another process
        # selected the same destination after argument validation.
        with output.open("xb") as handle:
            handle.write(temporary.read_bytes())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    counts = Counter(entry["opponent"]["member_id"] for entry in entries)
    manifest = {
        "version": "v4-opponent-population-starts-v1", "generator": RAW_PPO_STARTS_VERSION,
        "profile": "learned-population", "split": "train",
        "population": str(population_path),
        "population_file_sha256": hashlib.sha256(population_bytes).hexdigest(),
        "population_sha256": population.fingerprint,
        "population_id": population.population_id,
        "population_schema_version": population.schema_version,
        "pool_seed": args.pool_seed, "episodes": args.episodes,
        "coverage_fraction": args.coverage_fraction,
        "sampling_weights": population.sampling_weights(
            train_results=feedback, coverage_fraction=args.coverage_fraction,
        ),
        "training_feedback": str(feedback_path) if feedback_path is not None else None,
        "training_feedback_file_sha256": (
            hashlib.sha256(feedback_bytes).hexdigest() if feedback_bytes is not None else None
        ),
        "training_feedback_sha256": entries[0]["provenance"]["training_feedback_sha256"],
        "training_feedback_count": len(feedback),
        "member_episode_counts": dict(sorted(counts.items())),
        "entry_count": len(entries), "entry_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "limitations": ["learned Hog 2.6 mirrors only", "no policy fitting or match rollout performed"],
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"entries": str(output), "manifest": str(manifest_path), **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
