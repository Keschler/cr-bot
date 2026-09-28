#!/usr/bin/env python3
"""Train a compatible V4 actor from deterministic simulator observations.

The bootstrap is intentionally small: simulator states are collected through
the public observation boundary, labelled by the retained rules teacher, and
fit with the masked V4 supervised objective.  The output is a plain actor
state dict accepted by ``load_frozen_anchor`` and a manifest describing the
run.

Smoke check::

    PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_bootstrap.py \
        --smoke --out /tmp/v4-bootstrap-smoke
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a compatible V4 actor from public simulator states."
    )
    parser.add_argument("--states", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-decisions", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/v4/bootstrap"))
    parser.add_argument(
        "--smoke", action="store_true",
        help="run a tiny deterministic collection and one update",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow tracked source changes, recording their exact worktree hash",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal_runtime_contract(*, allow_dirty: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture Git provenance and the immutable simulator experiment contract."""

    from simulator.evaluation.contracts import contract_manifest, seal_current_contracts
    from simulator.rl.foundation.provenance import code_revision

    revision = code_revision()
    if (
        revision.get("source") != "git"
        or not isinstance(revision.get("commit"), str)
        or not isinstance(revision.get("tracked_worktree_hash"), str)
        or type(revision.get("worktree_dirty")) is not bool
    ):
        raise RuntimeError("bootstrap requires Git provenance for the tracked source")
    if revision["worktree_dirty"] and not allow_dirty:
        raise RuntimeError(
            "bootstrap requires a clean tracked worktree; pass --allow-dirty only "
            "when intentionally recording an in-progress source revision"
        )
    state = (
        "dirty:" if revision["worktree_dirty"] else "clean:"
    ) + revision["tracked_worktree_hash"]
    contract = seal_current_contracts(
        code_revision=revision["commit"],
        tracked_worktree_state=state,
    )
    return revision, {
        **contract_manifest(contract),
        "contract_hash": contract.contract_hash,
    }


def _train(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from simulator.rl.v4.bootstrap import (
        BootstrapConfig,
        generate_bootstrap_dataset,
        move_batch_to_device,
        to_torch_batch,
    )
    from simulator.rl.v4.frontier import frontier_model_config
    from simulator.rl.v4.model_v4 import RecurrentV4Policy
    from simulator.rl.v4.supervised import v4_supervised_loss

    start_revision, contract = _seal_runtime_contract(allow_dirty=args.allow_dirty)
    states = 32 if args.smoke else args.states
    epochs = 1 if args.smoke else args.epochs
    batch_size = 8 if args.smoke else args.batch_size
    max_decisions = 32 if args.smoke else args.max_decisions
    if type(states) is not int or states <= 0:
        raise ValueError("--states must be a positive integer")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("--epochs must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if not isinstance(args.lr, (int, float)) or args.lr <= 0.0:
        raise ValueError("--lr must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    samples = generate_bootstrap_dataset(
        BootstrapConfig(n_states=states, seed=args.seed, max_decisions=max_decisions)
    )
    config = frontier_model_config()
    policy = RecurrentV4Policy(config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.lr))
    history: list[dict[str, float]] = []
    policy.train()
    for epoch in range(epochs):
        order = torch.randperm(
            len(samples),
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        totals: list[float] = []
        for start in range(0, len(samples), batch_size):
            selected = [samples[int(index)] for index in order[start:start + batch_size]]
            batch = move_batch_to_device(to_torch_batch(selected), device)
            logits, _, _ = policy(
                batch["raster"], batch["global_features"], batch["entities"],
                batch["entity_mask"], batch["hand_tokens"], batch["opp_hand_probs"],
                batch["opp_out_of_cycle"], batch["opp_elixir_interval"],
                batch["event_history"], batch["reset_mask"],
            )
            loss, detail = v4_supervised_loss(
                logits, batch["masks"], batch["actions"],
                soft_placement=batch["soft_placement"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "loss": sum(totals) / len(totals)})

    end_revision, _ = _seal_runtime_contract(allow_dirty=True)
    if end_revision != start_revision:
        raise RuntimeError(
            "tracked source changed during bootstrap; discard the run and restart "
            "from one sealed revision"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    actor_path = args.out / "actor.pt"
    torch.save(policy.state_dict(), actor_path)
    manifest = {
        "schema_version": "v4-bootstrap-manifest-v2",
        "generator": "v4-bootstrap-sim-v1",
        "seed": args.seed,
        "n_states": len(samples),
        "epochs": epochs,
        "batch_size": batch_size,
        "max_decisions": max_decisions,
        "learning_rate": float(args.lr),
        "device": str(device),
        "model_config": asdict(config),
        "families": dict(Counter(sample.family for sample in samples)),
        "targets": {
            "wait": sum(sample.target.mode == 0 for sample in samples),
            "play": sum(sample.target.mode == 1 for sample in samples),
        },
        "history": history,
        "actor": actor_path.name,
        "actor_sha256": _sha256(actor_path),
        "provenance": {
            "git": start_revision,
            "contract": contract,
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _train(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
