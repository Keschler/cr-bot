"""Bounded, continuous reference-backend PPO readiness/campaign runner.

Outputs are new directories, never incumbent paths. This is an experiment
launcher, not a promotion evaluator or a complete training-lineage audit.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path


def build_pool(seed: int, size: int = 12) -> tuple[tuple[str, ...], ...]:
    from .opponent_pool import OpponentPool
    from ..ruleset import load_fixed_ruleset

    pool = OpponentPool(load_fixed_ruleset(), seed=seed)
    decks = tuple(tuple(row.cards) for row in pool.sample_decks(size, unique=True))
    if len(decks) != size or len(set(map(frozenset, decks))) != size:
        raise ValueError("campaign requires the requested number of distinct decks")
    return decks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("updates must be positive")

    from .prototype import _load_prototype_checkpoint, train_prototype

    decks = build_pool(args.seed)
    learner, stored, _ = _load_prototype_checkpoint(args.checkpoint, device="cpu")
    del learner
    config = replace(
        stored, envs=8, horizon=256, updates=args.updates, seed=args.seed,
        target_player=0, mixed_sides=True, env_backend="reference",
        overlap_rollouts=False, device=args.device, allow_provisional=True,
        freeze_mode_path=False, freeze_placement_path=False,
        imitation_only=False, expert_execution_probability=0.0,
        behavior_cloning_coef=0.0, behavior_cloning_factor_coef=0.0,
        placement_kl_coef=0.0, placement_rank_coef=0.0,
        learning_rate=3e-5, update_epochs=2, sequence_minibatch_size=4,
        sequence_length=128, recurrent_burn_in=32,
        gamma=0.9995, gae_lambda=0.995, entropy_coef=0.005,
        advantage_normalization="rollout", belief_coef=0.0,
        collect_belief_targets=False, potential_reward_weight=0.1,
        diagnostic_trace_out=None, max_update_approx_kl=0.3,
        max_update_conditional_play_mean_abs_log_ratio=0.5,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "deck_pool": decks, "seed": args.seed,
        "transitions_requested": args.updates * 8 * 256,
        "purpose": "readiness/pilot only; no held-out strength claim",
        "optimizer": "retain checkpoint optimizer; explicitly unfreeze both action paths",
    }
    (args.output_dir / "inputs.json").write_text(json.dumps(provenance, indent=2))
    report = train_prototype(
        config, checkpoint=args.checkpoint,
        checkpoint_out=args.output_dir / "prototype.pt",
        opponent_decks=decks,
        opponent_strategies=("deterministic-cycle", "random-legal"),
        opponent_action=None, expert_guidance=False,
        basic_scenario_sources=(None,) * 8, resume_learning_rate=3e-5,
        progress_callback=lambda update, transitions: print(
            f"update={update} transitions={transitions}", flush=True),
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({key: report.get(key) for key in
                      ("transitions", "outcomes", "wall_seconds", "matchup_rotation")}))


if __name__ == "__main__":
    main()
