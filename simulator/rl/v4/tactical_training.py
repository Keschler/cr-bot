"""Controlled, full-history tactical-learning ablations for the V4 actor.

Every arm starts from the same actor, visits every training episode at each
epoch, carries its own recurrent state, and anchors every inactive valid row.
The optional outcome head is a training auxiliary; exported actors keep the
existing inference/checkpoint contract. No PPO or privileged critic is used.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np


TRAINING_VERSION = "v4-tactical-learning-1"
ARMS = ("control", "outcomes", "plans", "curriculum")
INPUT_KEYS = ("raster", "global_features", "entities", "entity_mask", "hand_tokens",
              "opp_hand_probs", "opp_out_of_cycle", "opp_elixir_interval", "event_history")


@dataclass(frozen=True, slots=True)
class TacticalTrainingConfig:
    epochs: int = 4
    batch_episodes: int = 2
    chunk_length: int = 32
    learning_rate: float = 5e-5
    retention_weight: float = 1.0
    outcome_weight: float = 0.1
    gradient_clip: float = 1.0
    seed: int = 2026101901

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_episodes", "chunk_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "retention_weight", "outcome_weight", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _torch() -> Any:
    import torch
    return torch


def _atomic_save(path: Path, payload: Any) -> None:
    import os
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def differentiable_unroll(policy: Any, inputs: Mapping[str, Any], hidden: Any, reset: Any) -> tuple[Any, Any, Any]:
    """Unlike actor.forward_step, this explicitly retains training gradients."""
    zt, spatial, context, queries = policy.encoder(*(inputs[key] for key in INPUT_KEYS))
    ht, next_hidden = policy.recurrent.forward_with_hidden(zt, reset, hidden=hidden)
    return policy.heads(zt, ht, context, queries, spatial), ht, next_hidden


def _safe_log_probs(logits: Any, legal: Any) -> tuple[Any, Any]:
    torch = _torch()
    if logits.shape != legal.shape or legal.dtype != torch.bool:
        raise ValueError("logits and boolean legality mask must have identical shapes")
    # Empty conditional distributions (unaffordable cards/cells) are zero,
    # never softmax(-inf,...,-inf). Illegal entries have zero log contribution.
    safe = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
    logp = torch.log_softmax(safe, dim=-1)
    return torch.where(legal, logp.exp(), 0.0), torch.where(legal, logp, 0.0)


def full_action_kl(anchor_logits: Any, student_logits: Any, masks: Any) -> Any:
    """Exact KL(p_anchor || p_student) at EVERY chronological timestep.

    Card and cell distributions are conditional on PLAY and card respectively.
    Fixed WAIT has no duration factor. All unsupported slots contribute zero.
    """
    def categorical(old: Any, new: Any, legal: Any) -> tuple[Any, Any]:
        p, logp = _safe_log_probs(old, legal)
        _, logq = _safe_log_probs(new, legal)
        return (p * (logp - logq)).sum(-1), p
    mode, p_mode = categorical(anchor_logits.mode, student_logits.mode, masks.mode)
    card, p_card = categorical(anchor_logits.card, student_logits.card, masks.card)
    old = anchor_logits.placement.flatten(-2)
    new = student_logits.placement.flatten(-2)
    place, _ = categorical(old, new, masks.placement.flatten(-2))
    return mode + p_mode[..., 1] * (card + (p_card * place).sum(-1))


def scenario_keys(episode: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    """Public tactical contexts, never outcomes or calibration/test evidence."""
    air = np.asarray(episode["air_threat"], bool)
    xbow = np.asarray(episode["xbow_present"], bool)
    elixir = np.asarray(episode["global_features"])[:, 0] * 10
    # Named public card IDs are part of the fixed 2.6 input contract.
    musketeer = np.any(np.rint(np.asarray(episode["hand_tokens"])[..., 0] * 127) == 72, axis=1)
    family = str(metadata["family"])
    return [f"{family}|air={int(a)}|xbow={int(x)}|elixir={min(2, int(e // 4))}|musk={int(m)}"
            for a, x, e, m in zip(air, xbow, elixir, musketeer, strict=True)]


def adaptive_weights(previous: Mapping[str, Mapping[str, float]],
                     before_previous: Mapping[str, Mapping[str, float]],
                     counts: Mapping[str, int]) -> dict[str, float]:
    """Bounded train-only emphasis on learnable, improving tactical contexts.

    All episodes and all inactive retention rows remain in the workload. This
    is an adaptive supervision curriculum, not off-policy PPO replay or an
    arbitrary reset into the middle of a recurrent sequence. Raw priorities
    are in [.5,2]; population mean-one normalization bounds weights in [.25,4].
    """
    raw = {}
    for key in counts:
        entry = previous.get(key)
        if not entry or entry.get("count", 0) < 4:
            raw[key] = 1.0
            continue
        loss = float(entry["nll_sum"]) / float(entry["count"])
        probability = math.exp(-min(40.0, max(0.0, loss)))
        # Intermediate teacher agreement leaves room to learn; hopeless
        # outliers cannot consume the complete budget.
        learnable = 4 * probability * (1 - probability)
        older = before_previous.get(key)
        progress = 0.0 if not older else max(0.0, float(older["nll_sum"]) / max(1.0, float(older["count"])) - loss)
        raw[key] = min(2.0, 0.5 + learnable + min(0.5, progress))
    denominator = sum(counts[key] * value for key, value in raw.items())
    normalization = sum(counts.values()) / denominator if denominator else 1.0
    return {key: value * normalization for key, value in raw.items()}


def chunk_batch(episodes: Sequence[Mapping[str, Any]], metadata: Sequence[Mapping[str, Any]],
                *, start: int, length: int, arm: str, weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Pad ended lanes only; prelude and padding never receive supervision."""
    torch = _torch()
    from .model_v4 import V4ActionBatch, masks_from_legal_play
    if arm not in ARMS or not episodes or start < 0 or length <= 0 or len(episodes) != len(metadata):
        raise ValueError("invalid tactical chunk")
    width = min(length, max(len(episode["label_valid"]) for episode in episodes) - start)
    if width <= 0:
        raise ValueError("chunk starts after every episode has ended")
    keys = (*INPUT_KEYS, "legal_play", "legal_wait", "reset_mask", "label_valid", "active",
            "outcome_values", "outcome_mask", "executed_actions",
            "targets_plans" if arm == "plans" else "targets_cycle")
    result: dict[str, Any] = {}
    real = np.zeros((len(episodes), width), bool)
    for key in keys:
        prototype = np.asarray(episodes[0][key])
        value = np.zeros((len(episodes), width, *prototype.shape[1:]), dtype=prototype.dtype)
        for lane, episode in enumerate(episodes):
            actual = min(width, max(0, len(episode["label_valid"]) - start))
            if actual:
                value[lane, :actual] = episode[key][start:start + actual]
                real[lane, :actual] = True
        result[key] = torch.as_tensor(value)
    result["real"] = torch.as_tensor(real)
    result["valid"] = result["label_valid"] & result["real"]
    result["active"] &= result["valid"]
    result["inactive"] = result["valid"] & ~result["active"]
    result["masks"] = masks_from_legal_play(result["legal_play"], result["legal_wait"] | ~result["valid"])
    target = result["targets_plans" if arm == "plans" else "targets_cycle"].long().clamp_min(0)
    if bool((target[..., 4][result["valid"]] != 0).any()):
        raise ValueError("tactical training requires fixed WAIT duration index zero")
    result["actions"] = V4ActionBatch(target[..., 0], target[..., 1], target[..., 2:4], target[..., 4])
    action = result["executed_actions"].long().clamp_min(0)
    result["factual_actions"] = V4ActionBatch(action[..., 0], action[..., 1], action[..., 2:4], action[..., 4])
    row_weights = np.ones(real.shape, np.float32)
    scenarios = []
    for lane, (episode, meta) in enumerate(zip(episodes, metadata, strict=True)):
        row_keys = scenario_keys(episode, meta)[start:start + width]
        scenarios.append(row_keys)
        for step, key in enumerate(row_keys):
            row_weights[lane, step] = (weights or {}).get(key, 1.0)
    result["scenario_keys"] = scenarios
    result["tactical_weights"] = torch.as_tensor(row_weights)
    return result


def tactical_batch_loss(policy: Any, logits: Any, anchor_logits: Any, recurrent: Any,
                        batch: Mapping[str, Any], *, config: TacticalTrainingConfig,
                        outcome_head: Any = None) -> tuple[Any, dict[str, Any], Any]:
    """Separate active NLL, inactive full-distribution KL and factual aux loss."""
    torch = _torch()
    nll = -policy.log_prob(logits, batch["masks"], batch["actions"])
    active, inactive = batch["active"], batch["inactive"]
    zero = logits.mode.reshape(-1)[:0].sum()
    # Weights are normalized once over the entire fixed active population,
    # rather than cancelled by division by their own per-chunk sum.
    teacher = (nll[active] * batch["tactical_weights"][active]).mean() if bool(active.any()) else zero
    kl = full_action_kl(anchor_logits, logits, batch["masks"])
    retention = kl[inactive].mean() if bool(inactive.any()) else zero
    loss = teacher + config.retention_weight * retention
    details = dict(active_rows=int(active.sum()), inactive_rows=int(inactive.sum()),
                   real_rows=int(batch["real"].sum()), valid_rows=int(batch["valid"].sum()),
                   teacher_nll_sum=float(nll[active].detach().sum()),
                   retention_kl_sum=float(kl[inactive].detach().sum()),
                   teacher_loss=float(teacher.detach()), retention_loss=float(retention.detach()),
                   outcome_targets=0, outcome_loss=0.0)
    if outcome_head is not None:
        from .tactical_outcomes import encode_tactical_actions, tactical_outcome_loss
        prediction = outcome_head(recurrent, encode_tactical_actions(batch["factual_actions"], config=outcome_head.config))
        mask = batch["outcome_mask"] & active[..., None, None]
        auxiliary, metrics = tactical_outcome_loss(prediction, batch["outcome_values"], mask)
        loss = loss + config.outcome_weight * auxiliary
        details.update(outcome_targets=metrics["valid_targets"], outcome_loss=float(auxiliary.detach()))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite tactical objective")
    return loss, details, nll.detach()


def _rows(manifest: Mapping[str, Any], split: str) -> list[Mapping[str, Any]]:
    completed = manifest["completed"]
    # Process completion order must never choose the optimization order.
    return [completed[job["job_id"]] for job in manifest["jobs"] if job["split"] == split]


def _load_group(dataset: Path, rows: Sequence[Mapping[str, Any]], identity: str) -> tuple[list[Any], list[Any]]:
    from .tactical_data import load_episode
    loaded = [load_episode(dataset / row["path"], expected_identity=identity,
                           expected_sha=row["sha256"]) for row in rows]
    return [pair[0] for pair in loaded], [pair[1] for pair in loaded]


def _scenario_counts(dataset: Path, rows: Sequence[Mapping[str, Any]], identity: str) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        episodes, metas = _load_group(dataset, [row], identity)
        episode = episodes[0]
        counts.update(key for key, active in zip(scenario_keys(episode, metas[0]),
                      np.asarray(episode["active"]) & np.asarray(episode["label_valid"]), strict=True) if active)
    return counts


def _accumulate(stats: dict[str, dict[str, float]], nll: Any, batch: Mapping[str, Any]) -> None:
    values = nll.cpu().numpy()
    active = batch["active"].cpu().numpy()
    for lane, keys in enumerate(batch["scenario_keys"]):
        for step, key in enumerate(keys):
            if active[lane, step]:
                entry = stats.setdefault(key, dict(nll_sum=0.0, count=0))
                entry["nll_sum"] += float(values[lane, step])
                entry["count"] += 1


def calibrate(policy: Any, anchor: Any, outcome_head: Any, *, dataset: Path,
              rows: Sequence[Mapping[str, Any]], identity: str, arm: str,
              config: TacticalTrainingConfig) -> dict[str, Any]:
    """Read-only holdout diagnostics; never select an epoch or change weights."""
    torch = _torch()
    total: Counter = Counter()
    with torch.no_grad():
        for offset in range(0, len(rows), config.batch_episodes):
            episodes, metas = _load_group(dataset, rows[offset:offset + config.batch_episodes], identity)
            student_hidden = policy.initial_hidden(len(episodes), device="cpu")
            anchor_hidden = anchor.initial_hidden(len(episodes), device="cpu")
            for start in range(0, max(len(episode["label_valid"]) for episode in episodes), config.chunk_length):
                batch = chunk_batch(episodes, metas, start=start, length=config.chunk_length, arm=arm)
                logits, ht, student_hidden = differentiable_unroll(policy, batch, student_hidden, batch["reset_mask"])
                old, _, anchor_hidden = differentiable_unroll(anchor, batch, anchor_hidden, batch["reset_mask"])
                _, details, _ = tactical_batch_loss(policy, logits, old, ht, batch, config=config, outcome_head=outcome_head)
                for key in ("active_rows", "inactive_rows", "real_rows", "teacher_nll_sum", "retention_kl_sum", "outcome_targets"):
                    total[key] += details[key]
                total["outcome_loss_sum"] += details["outcome_loss"] * details["outcome_targets"]
    return dict(total, teacher_nll=total["teacher_nll_sum"] / max(1, total["active_rows"]),
                inactive_kl=total["retention_kl_sum"] / max(1, total["inactive_rows"]),
                outcome_loss=total["outcome_loss_sum"] / max(1, total["outcome_targets"]),
                used_for_selection=False)


def train_tactical_arm(*, dataset: Path, out: Path, arm: str, checkpoint: Mapping[str, Any],
                       config: TacticalTrainingConfig = TacticalTrainingConfig(), resume: bool = False) -> dict[str, Any]:
    """Fit one fixed-dose ablation, exporting only the final predeclared actor."""
    import json
    from .policy_development import atomic_json, digest, file_sha, runtime_source_fingerprint
    from .model_v4 import ModelConfigV4
    from .v4_ppo.runtime import configure_v4_runtime, load_frozen_anchor
    from .tactical_outcomes import TacticalOutcomeConfig, build_tactical_outcome_head
    from .tactical_data import validate_manifest
    torch = _torch()
    torch.set_num_threads(1)
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, checkpoint_record=checkpoint)
    identity = manifest["identity_sha256"]
    rows, calibration_rows = _rows(manifest, "train"), _rows(manifest, "calibration")
    if not rows or not calibration_rows:
        raise ValueError("training and disjoint calibration episodes are both required")
    jobs = manifest["jobs"]
    origins = [str(job.get("seed", job.get("match_seed"))) for job in jobs]
    if len(set(origins)) != len(origins):
        raise ValueError("collection match origins overlap")
    source = runtime_source_fingerprint()
    if manifest["identity"]["caller"].get("source_sha256") != source["sha256"]:
        raise ValueError("training source differs from the collected public trajectories")
    protocol = dict(version=TRAINING_VERSION, arm=arm, config=asdict(config), checkpoint=dict(checkpoint),
                    dataset=str(dataset), dataset_manifest_sha256=file_sha(manifest_path), source=source,
                    inference="native-stochastic-fixed250ms-carry", final_epoch=config.epochs,
                    curriculum="train-only bounded public-scenario supervision weights; every episode retained")
    run_hash = digest(protocol)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        if not resume or json.loads((out / "protocol.json").read_text()) != dict(protocol=protocol, run_hash=run_hash):
            raise ValueError("existing fit requires explicit resume under the identical protocol")
    elif any(out.iterdir()):
        raise ValueError("fit output directory is not empty")
    atomic_json(out / "protocol.json", dict(protocol=protocol, run_hash=run_hash))
    if (out / "result.json").exists():
        result = json.loads((out / "result.json").read_text())
        if result["run_hash"] != run_hash or file_sha(result["checkpoint"]["path"]) != result["checkpoint"]["sha256"]:
            raise ValueError("completed actor export changed")
        return result
    torch.manual_seed(config.seed)
    anchor = load_frozen_anchor(checkpoint["path"], config=ModelConfigV4(**checkpoint["model_config"]),
                                expected_sha256=checkpoint["sha256"], device="cpu")
    if float(anchor.config.dropout) != 0.0:
        raise ValueError("controlled tactical ablations require dropout=0 for exact matched workloads")
    configure_v4_runtime(anchor)
    student = copy.deepcopy(anchor)
    student.train()
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    outcome = (build_tactical_outcome_head(TacticalOutcomeConfig(recurrent_dim=student.config.gru_hidden_dim,
                 hidden_dim=student.config.gru_hidden_dim)) if arm == "outcomes" else None)
    torch.manual_seed(config.seed)
    parameters = list(student.parameters()) + ([] if outcome is None else list(outcome.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    initial_recurrent = {name: value.detach().clone() for name, value in student.recurrent.named_parameters() if "weight_hh" in name}
    history: list[dict[str, Any]] = []
    prior: dict[str, Any] = {}
    earlier: dict[str, Any] = {}
    start_epoch = 0
    state_path = out / "training_state.pt"
    if state_path.exists():
        if not resume:
            raise ValueError("partial training requires --resume")
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved["run_hash"] != run_hash:
            raise ValueError("partial fit protocol differs")
        student.load_state_dict(saved["actor"])
        optimizer.load_state_dict(saved["optimizer"])
        if outcome is not None:
            outcome.load_state_dict(saved["outcome"])
        history, prior, earlier = saved["history"], saved["prior"], saved["earlier"]
        start_epoch = int(saved["epoch"])
    counts = _scenario_counts(dataset, rows, identity)
    if not counts:
        raise ValueError("corpus has no public air/X-Bow supervision contexts")
    started = time.monotonic()
    expected_frame_visits = sum(int(row["decisions"]) for row in rows)
    for epoch in range(start_epoch, config.epochs):
        weights = adaptive_weights(prior, earlier, counts) if arm == "curriculum" else {}
        order = np.random.default_rng(config.seed + epoch).permutation(len(rows))
        expected_steps = sum(math.ceil(max(int(rows[int(index)]["decisions"]) for index in order[offset:offset + config.batch_episodes])
                             / config.chunk_length) for offset in range(0, len(order), config.batch_episodes))
        totals: Counter = Counter()
        stats: dict[str, dict[str, float]] = {}
        maximum_recurrent_gradient = 0.0
        epoch_started = time.monotonic()
        for group_start in range(0, len(order), config.batch_episodes):
            group = [rows[int(index)] for index in order[group_start:group_start + config.batch_episodes]]
            episodes, metas = _load_group(dataset, group, identity)
            hidden = student.initial_hidden(len(episodes), device="cpu")
            anchor_hidden = anchor.initial_hidden(len(episodes), device="cpu")
            for start in range(0, max(len(episode["label_valid"]) for episode in episodes), config.chunk_length):
                batch = chunk_batch(episodes, metas, start=start, length=config.chunk_length, arm=arm, weights=weights)
                optimizer.zero_grad(set_to_none=True)
                logits, ht, hidden = differentiable_unroll(student, batch, hidden, batch["reset_mask"])
                with torch.no_grad():
                    old, _, anchor_hidden = differentiable_unroll(anchor, batch, anchor_hidden, batch["reset_mask"])
                loss, details, nll = tactical_batch_loss(student, logits, old, ht, batch, config=config, outcome_head=outcome)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=True)
                recurrent_gradient = math.sqrt(sum(float(p.grad.square().sum()) for name, p in student.recurrent.named_parameters()
                                                    if "weight_hh" in name and p.grad is not None))
                maximum_recurrent_gradient = max(maximum_recurrent_gradient, recurrent_gradient)
                optimizer.step()
                hidden, anchor_hidden = hidden.detach(), anchor_hidden.detach()
                _accumulate(stats, nll, batch)
                for key in ("active_rows", "inactive_rows", "real_rows", "valid_rows", "teacher_nll_sum", "retention_kl_sum", "outcome_targets"):
                    totals[key] += details[key]
                totals["outcome_loss_sum"] += details["outcome_loss"] * details["outcome_targets"]
                totals["optimizer_steps"] += 1
                totals["maximum_gradient_norm"] = max(totals["maximum_gradient_norm"], float(gradient))
            if group_start % (config.batch_episodes * 5) == 0:
                atomic_json(out / "progress.json", dict(status="training", arm=arm, epoch=epoch + 1,
                    completed_episode_groups=(group_start // config.batch_episodes) + 1,
                    total_episode_groups=math.ceil(len(rows) / config.batch_episodes),
                    run_hash=run_hash, real_rows=int(totals["real_rows"]),
                    elapsed_seconds=time.monotonic() - started))
        if sum(int(value["count"]) for value in stats.values()) != sum(counts.values()):
            raise RuntimeError("training lost or repeated tactical rows")
        if totals["real_rows"] != expected_frame_visits or totals["optimizer_steps"] != expected_steps:
            raise RuntimeError("training frame/update dose differs from the declared complete-episode schedule")
        entry = dict(epoch=epoch + 1, **totals, wall_seconds=time.monotonic() - epoch_started,
                     teacher_nll=totals["teacher_nll_sum"] / max(1, totals["active_rows"]),
                     inactive_kl=totals["retention_kl_sum"] / max(1, totals["inactive_rows"]),
                     outcome_loss=totals["outcome_loss_sum"] / max(1, totals["outcome_targets"]),
                     recurrent_gradient_max=maximum_recurrent_gradient, curriculum_weights=weights)
        history.append(entry)
        earlier, prior = prior, stats
        _atomic_save(state_path, dict(run_hash=run_hash, actor=student.state_dict(),
                     optimizer=optimizer.state_dict(), outcome=None if outcome is None else outcome.state_dict(),
                     epoch=epoch + 1, history=history, prior=prior, earlier=earlier))
        atomic_json(out / "history.json", history)
        print(f"{arm} epoch {epoch + 1}/{config.epochs}: {int(totals['real_rows'])} rows, NLL {entry['teacher_nll']:.4f}, KL {entry['inactive_kl']:.4f}", flush=True)
    student.eval()
    calibration = calibrate(student, anchor, outcome, dataset=dataset, rows=calibration_rows,
                            identity=identity, arm=arm, config=config)
    if runtime_source_fingerprint() != source or file_sha(manifest_path) != protocol["dataset_manifest_sha256"]:
        raise RuntimeError("source or corpus manifest changed during fit")
    actor_path = out / f"actor_epoch_{config.epochs:03d}.pt"
    _atomic_save(actor_path, {name: value.detach().cpu().clone() for name, value in student.state_dict().items()})
    if outcome is not None:
        _atomic_save(out / "outcome_head.pt", dict(config=asdict(outcome.config), state_dict=outcome.state_dict()))
    recurrent_change = math.sqrt(sum(float((value.detach() - initial_recurrent[name]).square().sum())
                                    for name, value in student.recurrent.named_parameters() if name in initial_recurrent))
    result = dict(status="complete", arm=arm, run_hash=run_hash, final_epoch=config.epochs,
        checkpoint=dict(path=str(actor_path), sha256=file_sha(actor_path), model_config=asdict(student.config)),
        history=history, calibration=calibration, recurrent_weight_change=recurrent_change,
        actor_parameter_updates=any(not torch.equal(value, anchor.state_dict()[name]) for name, value in student.state_dict().items()),
        selected_actor_changed=False, wall_seconds_this_invocation=time.monotonic() - started,
        training_episodes=len(rows), calibration_episodes=len(calibration_rows),
        total_training_frame_visits=sum(int(row["real_rows"]) for row in history),
        total_optimizer_steps=sum(int(row["optimizer_steps"]) for row in history),
        limitation="Auxiliary loss/calibration is not evidence of stronger play. Only fresh complete matches can establish gains.")
    atomic_json(out / "result.json", result)
    atomic_json(out / "progress.json", dict(status="complete", arm=arm, run_hash=run_hash, checkpoint=result["checkpoint"]))
    return result
