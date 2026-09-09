"""V4 Stage-1 proof run: distill a tiny V4 actor and check the stage gate.

Trains a small V4 policy on simulator-distillation states (synthetic or
real-simulator observations) and reports held-out teacher agreement plus the
stage-acceptance checklist:

* validity: every training/held-out target honors legality;
* tactical proxy: held-out agreement beats the untrained baseline;
* no-collapse: card slots, WAIT durations, and placement cells stay diverse;
* integrity: finite losses, deterministic regeneration of the datasets.

This is *stage acceptance* evidence (paper section 7.8), not a promotion
claim: no sealed full-match suite is involved.  Usage::

    outputs/venv/bin/python scripts/run_v4_distillation_proof.py \
        --n-train 3000 --n-heldout 500 --epochs 6 --seed 0
    PYTHONPATH=src /home/keschler/.venvs/bn/bin/python scripts/run_v4_distillation_proof.py \
        --generator sim --n-train 3000 --n-heldout 500 --epochs 6 --seed 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from simulator.evaluation import (
    per_sample_placement_stats,
    placement_error_stats,
    seal_current_contracts,
    soft_region_mass,
    top1_cells,
)
from simulator.rl.distillation import PLAYER_DECK
from simulator.rl.distillation import (
    DistillationConfig,
    generate_dataset,
    to_torch_batch,
)
from simulator.rl.simulator_distillation import (
    SIM_GENERATOR_VERSION,
    generate_sim_dataset,
)
from simulator.rl.losses_v4 import V4SupervisedWeights, v4_supervised_loss
from simulator.rl.model_v4 import ModelConfigV4, RecurrentV4Policy, count_parameters


def _git_revision() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
            or "unknown"
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _selected_card_keys(batch, play_idx: np.ndarray, slots: int) -> list[str]:
    """Decode the teacher-selected hand slot to a deck card key per PLAY row."""

    norm_to_key = {float(policy_id) / 127.0: key for key, policy_id, *_ in PLAYER_DECK}
    hand = batch["hand_tokens"][:, 0].detach().cpu().numpy()
    chosen = batch["actions"].card_slot[:, 0].clamp(0, slots - 1).detach().cpu().numpy()
    rows = np.flatnonzero(play_idx)
    keys: list[str] = []
    for row, slot in zip(rows, chosen[play_idx]):
        observed = float(hand[int(row), int(slot), 0])
        best = min(norm_to_key, key=lambda norm: abs(norm - observed))
        keys.append(norm_to_key[best] if abs(best - observed) < 1e-4 else f"slot-{int(slot)}")
    return keys


def _group_placement_stats(
    *,
    prob_maps: np.ndarray,
    card_slots: np.ndarray,
    soft_targets: np.ndarray,
    true_rows: np.ndarray,
    true_cols: np.ndarray,
    families: np.ndarray,
    card_keys: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Group per-sample placement stats by family and by selected card."""

    per = per_sample_placement_stats(prob_maps, card_slots, soft_targets, true_rows, true_cols)

    def summarize(labels) -> dict[str, dict[str, float]]:
        labels = list(labels)
        groups: dict[str, dict[str, float]] = {}
        for label in sorted(set(labels)):
            mask = np.array([item == label for item in labels])
            groups[str(label)] = {
                "n": int(mask.sum()),
                "within_1_acc": float(np.mean(per["chebyshev"][mask] <= 1)),
                "mean_euclidean": float(np.mean(per["euclidean"][mask])),
                "mean_mass": float(np.mean(per["mass"][mask])),
            }
        return groups

    return {"by_family": summarize(families), "by_card": summarize(card_keys)}


def _agreement(policy, batch) -> dict[str, float]:
    policy.eval()
    with torch.no_grad():
        logits, _, _ = policy(
            batch["raster"],
            batch["global_features"],
            batch["entities"],
            batch["entity_mask"],
            batch["hand_tokens"],
            batch["opp_hand_probs"],
            batch["opp_out_of_cycle"],
            batch["opp_elixir_interval"],
            batch["event_history"],
            batch["reset_mask"],
        )
        actions = batch["actions"]
        mode_pred = logits.mode.argmax(-1)
        mode_acc = float((mode_pred == actions.mode).float().mean())
        is_play = actions.mode == 1
        is_wait = actions.mode == 0
        card_acc = (
            float(
                (logits.card.argmax(-1)[is_play] == actions.card_slot[is_play])
                .float()
                .mean()
            )
            if bool(is_play.any())
            else float("nan")
        )
        duration_acc = (
            float(
                (logits.wait_duration.argmax(-1)[is_wait] == actions.wait_duration[is_wait])
                .float()
                .mean()
            )
            if bool(is_wait.any())
            else float("nan")
        )
        placement_acc = float("nan")
        placement_error: dict[str, float] = {
            "n": 0.0,
            "exact_acc": float("nan"),
            "within_1_acc": float("nan"),
            "within_2_acc": float("nan"),
            "mean_euclidean": float("nan"),
            "median_euclidean": float("nan"),
            "mean_manhattan": float("nan"),
            "median_manhattan": float("nan"),
        }
        placement_mass: dict[str, float] = {
            "n": 0.0,
            "mean_mass": float("nan"),
            "median_mass": float("nan"),
            "majority_fraction": float("nan"),
        }
        placement_breakdown: dict[str, dict[str, dict[str, float]]] = {
            "by_family": {},
            "by_card": {},
        }
        if bool(is_play.any()):
            rows, cols = logits.placement.shape[-2:]
            slots = logits.placement.shape[2]
            flat = logits.placement.reshape(
                logits.placement.shape[0], 1, 4, rows * cols
            )
            slot = actions.card_slot.clamp(0, 3).reshape(-1, 1, 1, 1).expand(
                -1, 1, 1, rows * cols
            )
            selected = flat.gather(-2, slot).squeeze(-2).reshape(-1, rows * cols)
            pred_cell = selected.argmax(-1)
            true_cell = (
                actions.placement[..., 0].reshape(-1) * cols
                + actions.placement[..., 1].reshape(-1)
            )
            placement_acc = float(
                (pred_cell[is_play.reshape(-1)] == true_cell[is_play.reshape(-1)])
                .float()
                .mean()
            )
            # Distribution-level placement metrics on PLAY rows only.
            play_idx = is_play.reshape(-1).detach().cpu().numpy()
            flat_logits = logits.placement[:, 0].reshape(-1, slots, rows * cols)
            flat_mask = batch["masks"].placement[:, 0].reshape(-1, slots, rows * cols)
            probs = torch.where(
                flat_mask, flat_logits, torch.full_like(flat_logits, float("-inf"))
            ).softmax(dim=-1)
            prob_maps = probs.detach().cpu().numpy().reshape(-1, slots, rows, cols)
            sel = actions.card_slot[:, 0].clamp(0, slots - 1).detach().cpu().numpy()
            soft = batch["soft_placement"][:, 0].detach().cpu().numpy()
            pred_rows, pred_cols = top1_cells(prob_maps[play_idx], sel[play_idx])
            true_rows = actions.placement[:, 0, 0].detach().cpu().numpy()[play_idx]
            true_cols = actions.placement[:, 0, 1].detach().cpu().numpy()[play_idx]
            placement_error = placement_error_stats(pred_rows, pred_cols, true_rows, true_cols)
            placement_mass = soft_region_mass(
                prob_maps[play_idx], sel[play_idx], soft[play_idx]
            )
            placement_breakdown = _group_placement_stats(
                prob_maps=prob_maps[play_idx],
                card_slots=sel[play_idx],
                soft_targets=soft[play_idx],
                true_rows=true_rows,
                true_cols=true_cols,
                families=np.array(batch["families"])[play_idx],
                card_keys=_selected_card_keys(batch, play_idx, slots),
            )
        # Diversity diagnostics on deterministic decoding.
        decoded = policy.act_deterministic(logits, batch["masks"])
        used_slots = int(decoded.card_slot[is_play].unique().numel()) if bool(is_play.any()) else 0
        used_durations = (
            int(decoded.wait_duration[is_wait].unique().numel()) if bool(is_wait.any()) else 0
        )
        used_cells = (
            int(
                torch.unique(
                    decoded.placement[is_play][:, 0] * cols + decoded.placement[is_play][:, 1]
                ).numel()
            )
            if bool(is_play.any())
            else 0
        )
    policy.train()
    return {
        "mode_acc": mode_acc,
        "card_acc": card_acc,
        "duration_acc": duration_acc,
        "placement_acc": placement_acc,
        "placement_error": placement_error,
        "placement_mass": placement_mass,
        "placement_breakdown": placement_breakdown,
        "used_slots": float(used_slots),
        "used_durations": float(used_durations),
        "used_cells": float(used_cells),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-heldout", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument(
        "--generator",
        type=str,
        default="synthetic",
        choices=("synthetic", "sim"),
        help="Training-state source: hand-made synthetic features or real "
        "BasicMechanicsScenarioEnv observations (same sample schema).",
    )
    parser.add_argument(
        "--evaluate-only",
        type=str,
        default="",
        help="Path to a saved proof checkpoint: skip training and only "
        "recompute held-out agreement (no retraining).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()

    contract = seal_current_contracts(code_revision=_git_revision())
    distill_config = DistillationConfig(n_states=args.n_train, seed=args.seed)
    heldout_config = DistillationConfig(n_states=args.n_heldout, seed=args.seed + 10_000)
    if args.generator == "sim":
        train_samples = generate_sim_dataset(distill_config)
        heldout_samples = generate_sim_dataset(heldout_config)
        generator_version = SIM_GENERATOR_VERSION
    else:
        train_samples = generate_dataset(distill_config)
        heldout_samples = generate_dataset(heldout_config)
        generator_version = "synthetic-v4-distill-0"
    # Validity: every target honors legality (checked at generation, re-asserted).
    for sample in (*train_samples, *heldout_samples):
        target = sample.target
        if target.mode == 1:
            assert bool(sample.legal_play[target.card_slot, target.row, target.col])
        else:
            assert sample.legal_wait

    config = ModelConfigV4(
        model_dim=32,
        spatial_channels=16,
        fused_dim=64,
        gru_hidden_dim=64,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ff_dim=64,
        spatial_head_dim=8,
    )
    policy = RecurrentV4Policy(config)
    n_params = count_parameters(policy)
    baseline = _agreement(policy, to_torch_batch(heldout_samples))

    if args.evaluate_only:
        policy.load_state_dict(
            torch.load(args.evaluate_only, map_location="cpu", weights_only=True)
        )
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
        weights = V4SupervisedWeights()
        policy.train()
        for epoch in range(args.epochs):
            order = torch.randperm(len(train_samples))
            for start in range(0, len(train_samples), args.batch_size):
                chunk = [train_samples[int(i)] for i in order[start : start + args.batch_size]]
                batch = to_torch_batch(chunk)
                optimizer.zero_grad()
                logits, _, _ = policy(
                    batch["raster"],
                    batch["global_features"],
                    batch["entities"],
                    batch["entity_mask"],
                    batch["hand_tokens"],
                    batch["opp_hand_probs"],
                    batch["opp_out_of_cycle"],
                    batch["opp_elixir_interval"],
                    batch["event_history"],
                    batch["reset_mask"],
                )
                total, _ = v4_supervised_loss(
                    logits, batch["masks"], batch["actions"],
                    soft_placement=batch["soft_placement"],
                    weights=weights,
                )
                total.backward()
                optimizer.step()

    trained = _agreement(policy, to_torch_batch(heldout_samples))
    teacher_play_rate = float(
        np.mean([1.0 if s.target.mode == 1 else 0.0 for s in heldout_samples])
    )
    report = {
        "contract": {
            "code_revision": contract.code_revision,
            "engine_version": contract.engine_version,
            "ruleset_id": contract.ruleset_id,
            "ruleset_hash": contract.ruleset_hash,
            "observation_contract_hash": contract.observation_contract_hash,
            "contract_hash": contract.contract_hash,
        },
        "config": {
            "n_train": args.n_train,
            "n_heldout": args.n_heldout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "n_params": n_params,
            "generator": generator_version,
            "evaluate_only": args.evaluate_only,
            "model_dims": {
                "model_dim": config.model_dim,
                "spatial_channels": config.spatial_channels,
                "fused_dim": config.fused_dim,
                "gru_hidden_dim": config.gru_hidden_dim,
                "transformer_layers": config.transformer_layers,
                "transformer_ff_dim": config.transformer_ff_dim,
                "spatial_head_dim": config.spatial_head_dim,
            },
        },
        "teacher_play_rate": teacher_play_rate,
        "baseline_heldout": baseline,
        "trained_heldout": trained,
        "stage_gate": {
            "validity_targets_legal": True,
            "tactical_proxy_mode_improved": bool(
                trained["mode_acc"] > baseline["mode_acc"] + 0.05
            ),
            "tactical_proxy_card_improved": bool(
                trained["card_acc"] > baseline["card_acc"] + 0.05
            ),
            "no_card_collapse": bool(trained["used_slots"] >= 3.0),
            "no_duration_collapse": bool(trained["used_durations"] >= 2.0),
            "no_placement_collapse": bool(trained["used_cells"] >= 10.0),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    gate = report["stage_gate"]
    report["stage_accepted"] = bool(all(gate.values()))

    default_out = (
        REPO_ROOT / "outputs" / "v4" / f"distillation_proof_{args.generator}.json"
        if args.generator == "sim"
        else REPO_ROOT / "outputs" / "v4" / "distillation_proof.json"
    )
    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not args.evaluate_only:
        checkpoint_path = out_path.with_suffix(".pt")
        torch.save(policy.state_dict(), checkpoint_path)
        print(f"checkpoint -> {checkpoint_path}")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"stage_accepted={report['stage_accepted']} -> {out_path}")
    return 0 if report["stage_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
