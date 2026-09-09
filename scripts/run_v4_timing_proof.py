"""V4 Stage-1 timing proof: learn WAIT/PLAY timing from real-simulator states.

Trains the same tiny V4 actor used by the distillation proof on the balanced
timing dataset, then reports capability metrics on two held-out sets:

* ``timing-balanced``: quota-balanced WAIT/PLAY capability test;
* ``timing-natural``: unbalanced natural-distribution rollouts.

Plus a regression check: card/placement agreement on the original
sim-natural held-out states must not regress materially.

Raw mode accuracy is NOT the gate (a 96%-PLAY model scores 0.96 while
learning no timing).  The predeclared gates use balanced accuracy, WAIT and
PLAY recall, duration accuracy, per-card recall, collapse diagnostics, and
the sim-natural regression margin.

Training-loss note: the shared ``v4_supervised_loss`` reports NaN in its
card term on strategic WAIT rows (forward-only ``+inf * is_play=0``
artifact; gradients on those rows are exactly zero, so training is
unaffected).  The loss module is frozen for this task, so this script gates
on behavioral agreement only, never on loss values.

Usage::

    PYTHONPATH=src /home/keschler/.venvs/bn/bin/python scripts/run_v4_timing_proof.py \
        --n-train 2400 --n-heldout 600 --n-natural 600 --epochs 6 --seed 0
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

from simulator.evaluation import seal_current_contracts
from simulator.rl.distillation import (
    PLAYER_DECK,
    DistillationConfig,
    to_torch_batch,
)
from simulator.rl.losses_v4 import V4SupervisedWeights, v4_supervised_loss
from simulator.rl.model_v4 import ModelConfigV4, RecurrentV4Policy, count_parameters
from simulator.rl.simulator_distillation import generate_sim_dataset
from simulator.rl.timing_curriculum import (
    TIMING_FAMILIES,
    TIMING_GENERATOR_VERSION,
    TimingConfig,
    generate_timing_dataset,
    generate_timing_natural,
)
from simulator.rl.timing_teacher import TIMING_TEACHER_VERSION

# Reference card/placement agreement from the sim-natural proof
# (run_v4_distillation_proof.py --generator sim, seed 0, 2026-09-08).
SIM_REFERENCE = {
    "card_acc": 0.983,
    "placement_within1": 0.979,
    "source": "sim proof full, seed 0",
}

_NORM_TO_KEY = {float(policy_id) / 127.0: key for key, policy_id, *_ in PLAYER_DECK}


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


def _card_key(hand_row: np.ndarray, slot: int) -> str:
    observed = float(hand_row[int(slot), 0])
    best = min(_NORM_TO_KEY, key=lambda norm: abs(norm - observed))
    return _NORM_TO_KEY[best] if abs(best - observed) < 1e-4 else f"slot-{int(slot)}"


def _predict(policy, batch) -> dict[str, np.ndarray]:
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
        mode_pred = logits.mode.argmax(-1)[:, 0].detach().cpu().numpy()
        mode_true = actions.mode[:, 0].detach().cpu().numpy()
        card_pred = logits.card.argmax(-1)[:, 0].detach().cpu().numpy()
        card_true = actions.card_slot[:, 0].detach().cpu().numpy()
        dur_pred = logits.wait_duration.argmax(-1)[:, 0].detach().cpu().numpy()
        dur_true = actions.wait_duration[:, 0].detach().cpu().numpy()
        rows, cols = logits.placement.shape[-2:]
        slots = logits.placement.shape[2]
        flat = logits.placement[:, 0].reshape(-1, slots, rows * cols)
        flat_mask = batch["masks"].placement[:, 0].reshape(-1, slots, rows * cols)
        probs = (
            torch.where(flat_mask, flat, torch.full_like(flat, float("-inf")))
            .softmax(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        sel = np.clip(card_true, 0, slots - 1)
        pred_cell = probs[np.arange(len(probs)), sel].reshape(-1, rows, cols)
        pred_rows, pred_cols = np.unravel_index(
            pred_cell.reshape(len(pred_cell), -1).argmax(axis=1), (rows, cols)
        )
        true_rows = actions.placement[:, 0, 0].detach().cpu().numpy()
        true_cols = actions.placement[:, 0, 1].detach().cpu().numpy()
        soft = batch["soft_placement"][:, 0].detach().cpu().numpy()
        mass = (pred_cell * (soft > 0)).sum(axis=(1, 2))
        hand = batch["hand_tokens"][:, 0].detach().cpu().numpy()
    policy.train()
    return {
        "mode_pred": mode_pred,
        "mode_true": mode_true,
        "card_pred": card_pred,
        "card_true": card_true,
        "dur_pred": dur_pred,
        "dur_true": dur_true,
        "pred_rows": np.asarray(pred_rows),
        "pred_cols": np.asarray(pred_cols),
        "true_rows": np.asarray(true_rows),
        "true_cols": np.asarray(true_cols),
        "mass": np.asarray(mass),
        "hand": hand,
    }


def _timing_metrics(pred: dict[str, np.ndarray], samples) -> dict:
    mode_true = pred["mode_true"]
    mode_pred = pred["mode_pred"]
    is_wait = mode_true == 0
    is_play = mode_true == 1
    wait_recall = float((mode_pred[is_wait] == 0).mean()) if is_wait.any() else float("nan")
    play_recall = float((mode_pred[is_play] == 1).mean()) if is_play.any() else float("nan")
    balanced_acc = float(np.nanmean([wait_recall, play_recall]))
    families = np.array([s.family for s in samples])
    per_family = {}
    for family in sorted(set(families.tolist())):
        mask = families == family
        per_family[str(family)] = {
            "n": int(mask.sum()),
            "mode_acc": float((mode_pred[mask] == mode_true[mask]).mean()),
            "wait_share": float(is_wait[mask].mean()),
        }
    dur_true = pred["dur_true"]
    dur_pred = pred["dur_pred"]
    dur_acc = (
        float((dur_pred[is_wait] == dur_true[is_wait]).mean()) if is_wait.any() else float("nan")
    )
    per_duration = {}
    for duration in range(4):
        mask = is_wait & (dur_true == duration)
        per_duration[str(duration)] = {
            "n": int(mask.sum()),
            "acc": float((dur_pred[mask] == dur_true[mask]).mean()) if mask.any() else float("nan"),
        }
    card_keys = np.array(
        [_card_key(pred["hand"][i], pred["card_true"][i]) if is_play[i] else "WAIT" for i in range(len(samples))]
    )
    card_pred_keys = np.array(
        [_card_key(pred["hand"][i], pred["card_pred"][i]) if is_play[i] else "WAIT" for i in range(len(samples))]
    )
    per_card = {}
    for key in sorted(set(card_keys[is_play].tolist())):
        mask = is_play & (card_keys == key)
        per_card[str(key)] = {
            "n": int(mask.sum()),
            "recall": float((card_pred_keys[mask] == key).mean()),
        }
    card_acc = (
        float((card_pred_keys[is_play] == card_keys[is_play]).mean()) if is_play.any() else float("nan")
    )
    cheb = np.maximum(
        np.abs(pred["pred_rows"] - pred["true_rows"]),
        np.abs(pred["pred_cols"] - pred["true_cols"]),
    )
    eucl = np.sqrt(
        (pred["pred_rows"] - pred["true_rows"]) ** 2
        + (pred["pred_cols"] - pred["true_cols"]) ** 2
    )
    placement = {}
    if is_play.any():
        placement = {
            "n": int(is_play.sum()),
            "exact_acc": float((cheb[is_play] == 0).mean()),
            "within_1_acc": float((cheb[is_play] <= 1).mean()),
            "mean_euclidean": float(eucl[is_play].mean()),
            "mean_mass": float(pred["mass"][is_play].mean()),
        }
    decoded_slots = set(int(s) for s in pred["card_pred"][is_play].tolist()) if is_play.any() else set()
    decoded_durs = set(int(s) for s in pred["dur_pred"][is_wait].tolist()) if is_wait.any() else set()
    decoded_cells = (
        set(
            (int(r), int(c))
            for r, c in zip(
                pred["pred_rows"][is_play].tolist(), pred["pred_cols"][is_play].tolist()
            )
        )
        if is_play.any()
        else set()
    )
    return {
        "n": len(samples),
        "mode_acc": float((mode_pred == mode_true).mean()),
        "wait_recall": wait_recall,
        "play_recall": play_recall,
        "balanced_acc": balanced_acc,
        "confusion": {
            "true_wait_pred_wait": int(((mode_true == 0) & (mode_pred == 0)).sum()),
            "true_wait_pred_play": int(((mode_true == 0) & (mode_pred == 1)).sum()),
            "true_play_pred_wait": int(((mode_true == 1) & (mode_pred == 0)).sum()),
            "true_play_pred_play": int(((mode_true == 1) & (mode_pred == 1)).sum()),
        },
        "per_family": per_family,
        "duration_acc": dur_acc,
        "per_duration": per_duration,
        "card_acc": card_acc,
        "per_card": per_card,
        "placement": placement,
        "used_slots": float(len(decoded_slots)),
        "used_durations": float(len(decoded_durs)),
        "used_cells": float(len(decoded_cells)),
    }


def _transition_metrics(pred: dict[str, np.ndarray], samples) -> dict:
    """Score WAIT->PLAY switch-offset prediction on single-switch sequences."""

    by_seq: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        by_seq.setdefault(str(sample.provenance["seq_id"]), []).append(index)
    n_single = 0
    offset_match = 0
    offset_card_match = 0
    for seq_id, indices in by_seq.items():
        indices.sort(key=lambda i: int(samples[i].provenance["sequence_offset"]))
        teacher = [int(pred["mode_true"][i]) for i in indices]
        # Exactly one WAIT->PLAY switch with at least one WAIT.
        switches = [k for k in range(len(teacher) - 1) if teacher[k] == 0 and teacher[k + 1] == 1]
        if len(switches) != 1 or 0 not in teacher:
            continue
        n_single += 1
        switch_at = switches[0] + 1  # offset of the first teacher PLAY
        model = [int(pred["mode_pred"][i]) for i in indices]
        try:
            model_switch = next(k for k, m in enumerate(model) if m == 1 and any(v == 0 for v in model[:k]))
        except StopIteration:
            continue
        if model_switch == switch_at:
            offset_match += 1
            teach_slot = int(pred["card_true"][indices[switch_at]])
            if int(pred["card_pred"][indices[switch_at]]) == teach_slot:
                offset_card_match += 1
    return {
        "n_single_switch": n_single,
        "offset_match": offset_match,
        "offset_match_acc": (offset_match / n_single) if n_single else float("nan"),
        "offset_card_match": offset_card_match,
        "offset_card_match_acc": (offset_card_match / n_single) if n_single else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=2400)
    parser.add_argument("--n-heldout", type=int, default=600)
    parser.add_argument("--n-natural", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument(
        "--sim-mix",
        type=int,
        default=0,
        help="Append this many sim-natural states (same generator/seed convention "
        "as the regression set, but seed+30000) to the TRAIN set only. "
        "Exploratory joint recipe: timing skill + sim-natural card coverage.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()

    contract = seal_current_contracts(code_revision=_git_revision())
    train_samples, train_stats = generate_timing_dataset(
        TimingConfig(n_states=args.n_train, seed=args.seed)
    )
    sim_mix_n = 0
    if args.sim_mix > 0:
        mix_samples = generate_sim_dataset(
            DistillationConfig(n_states=args.sim_mix, seed=args.seed + 30_000)
        )
        train_samples = train_samples + mix_samples
        sim_mix_n = len(mix_samples)
    heldout_samples, heldout_stats = generate_timing_dataset(
        TimingConfig(n_states=args.n_heldout, seed=args.seed + 10_000)
    )
    natural_samples, natural_stats = generate_timing_natural(
        TimingConfig(n_states=args.n_natural, seed=args.seed + 20_000, pool_factor=2)
    )
    # Validity: every teacher target honors legality.
    for sample in (*train_samples, *heldout_samples, *natural_samples):
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
    heldout_batch = to_torch_batch(heldout_samples)
    natural_batch = to_torch_batch(natural_samples)
    baseline_heldout = _timing_metrics(_predict(policy, heldout_batch), heldout_samples)
    baseline_heldout["transitions"] = _transition_metrics(
        _predict(policy, heldout_batch), heldout_samples
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    weights = V4SupervisedWeights()
    policy.train()
    for _ in range(args.epochs):
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

    trained_heldout = _timing_metrics(_predict(policy, heldout_batch), heldout_samples)
    trained_heldout["transitions"] = _transition_metrics(
        _predict(policy, heldout_batch), heldout_samples
    )
    trained_natural = _timing_metrics(_predict(policy, natural_batch), natural_samples)
    trained_natural["transitions"] = _transition_metrics(
        _predict(policy, natural_batch), natural_samples
    )

    # Regression: card/placement on the original sim-natural held-out states
    # (identical states to the sim proof: same config, same seed).
    sim_samples = generate_sim_dataset(DistillationConfig(n_states=500, seed=10_000))
    sim_metrics = _timing_metrics(_predict(policy, to_torch_batch(sim_samples)), sim_samples)

    per_card_gate_ok = True
    for key, row in trained_heldout["per_card"].items():
        if row["n"] >= 30 and not (row["recall"] >= 0.35):
            per_card_gate_ok = False
    gate = {
        "validity_targets_legal": True,
        "balanced_acc_ge_065": bool(trained_heldout["balanced_acc"] >= 0.65),
        "wait_recall_ge_050": bool(trained_heldout["wait_recall"] >= 0.50),
        "play_recall_ge_070": bool(trained_heldout["play_recall"] >= 0.70),
        "duration_improved": bool(
            trained_heldout["duration_acc"] > baseline_heldout["duration_acc"] + 0.05
        ),
        "card_improved": bool(
            trained_heldout["card_acc"] > baseline_heldout["card_acc"] + 0.05
        ),
        "no_card_collapse": bool(trained_heldout["used_slots"] >= 4.0),
        "no_duration_collapse": bool(trained_heldout["used_durations"] >= 2.0),
        "no_placement_collapse": bool(trained_heldout["used_cells"] >= 10.0),
        "per_card_recall": bool(per_card_gate_ok),
        "no_card_regression": bool(
            sim_metrics["card_acc"] >= SIM_REFERENCE["card_acc"] - 0.06
        ),
        "no_placement_regression": bool(
            sim_metrics["placement"]["within_1_acc"]
            >= SIM_REFERENCE["placement_within1"] - 0.06
        ),
    }
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
            "n_natural": args.n_natural,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "n_params": n_params,
            "generator": TIMING_GENERATOR_VERSION,
            "teacher": TIMING_TEACHER_VERSION,
            "sim_reference": SIM_REFERENCE,
            "sim_mix_train": sim_mix_n,
        },
        "train_stats": train_stats,
        "heldout_stats": heldout_stats,
        "natural_stats": natural_stats,
        "baseline_heldout": baseline_heldout,
        "trained_heldout": trained_heldout,
        "trained_natural": trained_natural,
        "sim_regression": sim_metrics,
        "stage_gate": gate,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    report["stage_accepted"] = bool(all(gate.values()))

    out_path = (
        Path(args.out) if args.out else (REPO_ROOT / "outputs" / "v4" / "timing_proof.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    checkpoint_path = out_path.with_suffix(".pt")
    torch.save(policy.state_dict(), checkpoint_path)
    print(f"checkpoint -> {checkpoint_path}")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"stage_accepted={report['stage_accepted']} -> {out_path}")
    return 0 if report["stage_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
