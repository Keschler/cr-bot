#!/usr/bin/env python3
"""Run a reproducible controlled V4 frontier-PPO learning curve.

The command evaluates update zero and each requested update on the same sealed
entries, opponent pool, and fresh match seeds.  It writes one directory of
JSON reports and one plain actor checkpoint per curve point.  Nothing is
promoted automatically.

For a dependency-free protocol check:

    python scripts/run_v4_learning_curve.py --smoke --out /tmp/v4-curve-smoke

For a real development run, pass a supplied V4 actor checkpoint and training
frontier file; sealed entries are optional secondary diagnostics. The
historical simx12 checkpoint is intentionally not invented when absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simulator.rl.v4.learning_curve import (  # noqa: E402
    DEFAULT_EVAL_MATCHES,
    DEFAULT_EVAL_MAX_DECISIONS,
    DEFAULT_EVAL_SEED_START,
    DEFAULT_EPISODE_CAP,
    DEFAULT_ROLLOUT_DECISIONS,
    DEFAULT_ROLLOUT_LANES,
    DEFAULT_UPDATE_SCHEDULE,
    LearningCurveConfig,
    parse_update_schedule,
    run_final_evaluation,
    run_learning_curve,
    smoke_hooks,
)


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def _load_opponents(value: str | None) -> tuple[dict[str, Any], ...] | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    try:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(
            "--opponents-json must be a JSON file or JSON array"
        ) from exc
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise argparse.ArgumentTypeError("opponent JSON must be a list of objects")
    return tuple(dict(item) for item in payload)


def _load_model_config(value: str | None) -> dict[str, Any] | None:
    """Load a JSON object that is passed to ``ModelConfigV4``."""

    if value is None:
        return None
    path = Path(value).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(
            "--model-config must be a JSON file or JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("--model-config must contain a JSON object")
    return dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible V4 controlled frontier-PPO learning curve."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="supplied V4 actor checkpoint used at update zero",
    )
    parser.add_argument(
        "--train-entries",
        type=Path,
        help="frontier entries or raw match openings used only for PPO rollouts",
    )
    parser.add_argument(
        "--training-source", choices=("frontier", "match-start"), default="frontier",
        help="identify and validate the PPO training source (match-start requires --bc-coef 0)",
    )
    parser.add_argument(
        "--sealed-entries",
        type=Path,
        help="optional sealed entries used only for secondary diagnostics; never checkpoint selection",
    )
    parser.add_argument(
        "--allow-scripted-frontier-entries", action="store_true",
        help=(
            "permit non-catalog scripted frontier entries for separate pinned/mixed "
            "experiments; the real-deck comparison keeps catalog validation enabled"
        ),
    )
    parser.add_argument(
        "--out", "--output-dir", dest="output_dir", type=Path,
        default=Path("outputs/v4/learning_curve"),
        help="artifact directory (default: outputs/v4/learning_curve)",
    )
    parser.add_argument(
        "--updates", default=",".join(str(value) for value in DEFAULT_UPDATE_SCHEDULE),
        help="comma-separated update points; update 0 is always included",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed-start", type=int, default=DEFAULT_EVAL_SEED_START)
    parser.add_argument("--eval-matches", type=int, default=DEFAULT_EVAL_MATCHES)
    parser.add_argument(
        "--eval-seeds", type=_parse_int_list,
        help="explicit comma-separated fresh evaluation seeds (overrides start/count)",
    )
    parser.add_argument(
        "--rollout-decisions", "--decisions-per-rollout",
        dest="rollout_decisions", type=int, default=DEFAULT_ROLLOUT_DECISIONS,
        help="frontier decisions collected per update",
    )
    parser.add_argument(
        "--rollout-lanes", "--n-lanes", dest="rollout_lanes", type=int,
        default=DEFAULT_ROLLOUT_LANES,
        help="frontier rollout lanes per update",
    )
    parser.add_argument(
        "--episode-cap", type=int, default=DEFAULT_EPISODE_CAP,
        help="training frontier episode truncation cap",
    )
    parser.add_argument(
        "--eval-max-decisions", type=int, default=DEFAULT_EVAL_MAX_DECISIONS,
        help="full-match evaluation decision cap; capped matches are censored",
    )
    parser.add_argument(
        "--deterministic-policy", action="store_true",
        help="use deterministic masked-policy actions; default uses seeded stochastic sampling",
    )
    parser.add_argument(
        "--reward-mode", choices=("terminal", "tower-delta", "truncation"),
        default="tower-delta",
        help="frontier PPO reward mode (tower-delta is the controlled default)",
    )
    parser.add_argument("--choice-replicas", type=int, default=8)
    parser.add_argument(
        "--adaptive-choice-replicas", action="store_true",
        help="extend volatile sealed-state replicas up to the configured maximum",
    )
    parser.add_argument("--choice-max-replicas", type=int, default=32)
    parser.add_argument("--choice-horizon", type=int, default=128)
    parser.add_argument("--choice-se-k", type=float, default=2.0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bc-coef", type=float, default=0.05)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-config",
        type=_load_model_config,
        help=(
            "JSON file or inline object for ModelConfigV4; omitted selects the "
            "checked-in compact frontier config"
        ),
    )
    parser.add_argument(
        "--compare-checkpoint", action="append", default=[], type=Path,
        help=(
            "evaluate an independently trained compatible checkpoint against the "
            "same held-out protocol"
        ),
    )
    parser.add_argument(
        "--opponents-json",
        help="JSON file or inline JSON array of fixed evaluation opponent specs",
    )
    parser.add_argument(
        "--evaluation-only-checkpoint", "--eval-only-checkpoint",
        dest="evaluation_only_checkpoint", type=Path,
        help="evaluate this checkpoint against the held-out protocol without training",
    )
    parser.add_argument(
        "--evaluation-only", action="store_true",
        help="mark the supplied --checkpoint as an evaluation-only baseline",
    )
    parser.add_argument(
        "--final-evaluation", "--final-eval", action="store_true",
        help=(
            "evaluate frozen actors on reserved evaluation decks without training, "
            "sealed diagnostics, or checkpoint selection"
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="run a lightweight synthetic protocol smoke; output makes no quality claim",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint = args.evaluation_only_checkpoint or args.checkpoint
    evaluation_only = bool(args.evaluation_only or args.evaluation_only_checkpoint is not None)
    if args.smoke and args.evaluation_only_checkpoint is not None:
        parser.error("--smoke cannot be combined with --evaluation-only-checkpoint")
    if args.final_evaluation and args.smoke:
        parser.error("--final-evaluation cannot be combined with --smoke")
    if args.final_evaluation and evaluation_only:
        parser.error("--final-evaluation cannot be combined with --evaluation-only")
    try:
        schedule = parse_update_schedule(args.updates)
        opponents = _load_opponents(args.opponents_json)
        config = LearningCurveConfig(
            checkpoint=checkpoint,
            train_entries=args.train_entries,
            training_source=args.training_source,
            sealed_entries=args.sealed_entries,
            output_dir=args.output_dir,
            update_schedule=schedule,
            seed=args.seed,
            eval_seed_start=args.eval_seed_start,
            eval_matches=args.eval_matches,
            eval_seeds=args.eval_seeds,
            rollout_decisions=args.rollout_decisions,
            rollout_lanes=args.rollout_lanes,
            episode_cap=args.episode_cap,
            eval_max_decisions=args.eval_max_decisions,
            deterministic_policy=args.deterministic_policy,
            choice_replicas=args.choice_replicas,
            choice_horizon=args.choice_horizon,
            choice_adaptive_replicas=args.adaptive_choice_replicas,
            choice_max_replicas=args.choice_max_replicas,
            choice_se_k=args.choice_se_k,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            bc_coef=args.bc_coef,
            ppo_epochs=args.ppo_epochs,
            device=args.device,
            smoke=args.smoke,
            reward_mode=args.reward_mode,
            model_config=args.model_config,
            comparison_checkpoints=tuple(args.compare_checkpoint),
            opponent_specs=opponents,
            evaluation_only=evaluation_only,
            allow_scripted_frontier_entries=args.allow_scripted_frontier_entries,
        )
        if args.final_evaluation:
            report = run_final_evaluation(config)
        else:
            report = run_learning_curve(
                config,
                hooks=smoke_hooks(args.seed) if args.smoke else None,
            )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    points = report.get("points", [])
    print(
        json.dumps(
            {
                "curve": str(
                    (
                        config.output_dir
                        / ("final_evaluation.json" if args.final_evaluation else "learning_curve.json")
                    ).resolve()
                ),
                "final_evaluation_report": (
                    str((config.output_dir / "final_evaluation.json").resolve())
                    if args.final_evaluation else None
                ),
                "points": len(points),
                "updates": [point["update"] for point in points],
                "smoke": config.smoke,
                "evaluation_only": config.evaluation_only,
                "final_evaluation": bool(args.final_evaluation),
                "quality_claim": report["provenance"]["quality_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
