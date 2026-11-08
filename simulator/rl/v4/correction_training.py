"""Sparse, independently certified corrections on complete actor histories.

Recorded observations and factual actions remain immutable.  Only accepted
alternative action targets receive NLL; every other valid observation retains
the frozen actor's complete legal action distribution.  Calibration episodes
receive no gradients and never choose an epoch or promote an actor.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import copy
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .policy_development import atomic_json, canonical, digest, file_sha, runtime_source_fingerprint
from .correction_exposure import CorrectionGateConfig, correction_learning_gate, exposure_schedule
from .tactical_training import (
    TacticalTrainingConfig, _atomic_save, _load_group, _rows, chunk_batch,
    differentiable_unroll, tactical_batch_loss, full_action_kl, _safe_log_probs,
)


CORRECTIONS_SCHEMA = "v4-replay-corrections-v1"
TRAINING_VERSION = "v4-replay-correction-training-2"


@dataclass(frozen=True, slots=True)
class CorrectionTrainingConfig:
    epochs: int = 2
    batch_episodes: int = 2
    chunk_length: int = 32
    learning_rate: float = 1e-5
    retention_weight: float = 1.0
    correction_weight: float = 1.0
    correction_exposures: int = 0
    gradient_clip: float = 1.0
    seed: int = 2026102901

    def __post_init__(self) -> None:
        # Reuse the existing finite-budget validation without enabling an
        # auxiliary head or adding another inference/training implementation.
        self.tactical_config()
        if not math.isfinite(self.correction_weight) or self.correction_weight <= 0:
            raise ValueError("correction_weight must be finite and positive")
        if type(self.correction_exposures) is not int or self.correction_exposures < 0:
            raise ValueError("correction_exposures must be a nonnegative integer")

    def tactical_config(self) -> TacticalTrainingConfig:
        return TacticalTrainingConfig(**{key: value for key, value in asdict(self).items()
                                         if key not in ("correction_weight", "correction_exposures")})


def correction_episode(episode: Mapping[str, Any], targets: Mapping[int, Sequence[int]]) -> dict[str, Any]:
    """Overlay labels without modifying any public input or factual action."""
    from .tactical_data import validate_target

    result = dict(episode)
    active = np.zeros(len(episode["label_valid"]), dtype=bool)
    labels = np.array(episode["targets_cycle"], copy=True)
    for step, target in targets.items():
        if type(step) is not int or not 0 <= step < len(active) or not bool(episode["label_valid"][step]):
            raise ValueError("correction must address a valid real episode decision")
        value = np.asarray(target)
        if value.shape != (5,) or value.dtype.kind not in "iu":
            raise ValueError("correction target must be five integer action fields")
        validate_target(value, episode["legal_play"][step], bool(episode["legal_wait"][step]))
        active[step] = True
        labels[step] = value
    result["active"], result["targets_cycle"] = active, labels
    return result


def _replica_provenance(verdict: Mapping[str, Any], best_index: int) -> None:
    """Require actual paired seed evidence, beyond declared replica numbers."""
    branches = verdict["branches"]

    def evidence(branch: Mapping[str, Any], phase: str) -> dict[int, int]:
        indices = verdict[f"{phase}_replica_indices"]
        replicas = (branch.get("discovery_replicas", branch.get("replicas")) if phase == "discovery"
                    else branch.get("confirmation_replicas"))
        values = (branch.get("discovery_values", branch.get("values")) if phase == "discovery"
                  else branch.get("confirmation_values"))
        if (not isinstance(replicas, list) or not isinstance(values, list)
                or len(replicas) != len(indices) or len(values) != len(indices)):
            raise ValueError("accepted correction lacks complete actual replica seed/index evidence")
        seeds: dict[int, int] = {}
        for index, value, replica in zip(indices, values, replicas, strict=True):
            seed = replica.get("replica_seed") if isinstance(replica, Mapping) else None
            if (type(index) is not int or type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF
                    or replica.get("replica_index") != index or type(replica.get("replica_index")) is not int
                    or index in seeds or seed in seeds.values()
                    or not math.isfinite(float(value)) or replica.get("value") != value):
                raise ValueError("accepted correction actual replica seed/index/value evidence differs")
            seeds[index] = seed
        return seeds

    discovery = [evidence(branch, "discovery") for branch in branches]
    if any(seeds != discovery[0] for seeds in discovery[1:]):
        raise ValueError("discovery branches do not use the same paired replica seeds")
    sim = next(branch for branch in branches if branch.get("source") == "sim")
    confirmed_sim = evidence(sim, "confirmation")
    confirmed_best = evidence(branches[best_index], "confirmation")
    if confirmed_sim != confirmed_best or set(discovery[0].values()) & set(confirmed_sim.values()):
        raise ValueError("confirmation must use paired actual seeds disjoint from discovery")


def _verified_target(record: Mapping[str, Any], episode: Mapping[str, Any],
                     metadata: Mapping[str, Any], origin: Mapping[str, Any], *, min_lcb: float) -> list[int] | None:
    from .search_distillation.targets import action_from_record, validated_correction

    if type(record.get("accepted")) is not bool:
        raise ValueError("correction acceptance must be an explicit boolean")
    step = record.get("t")
    if type(step) is not int or not 0 <= step < int(metadata["decisions"]):
        raise ValueError("correction decision is outside its full source episode")
    state_hash = str(episode["state_hash"][step])
    if (record.get("job_id") != metadata["job_id"]
            or record.get("source_episode_sha256") != origin["sha256"]
            or record.get("state_hash") != state_hash):
        raise ValueError("correction source episode/decision identity differs")
    verdict = record.get("verdict")
    if not isinstance(verdict, Mapping) or verdict.get("state_hash") != state_hash:
        raise ValueError("correction verdict belongs to another source state")
    sim = [branch for branch in verdict.get("branches", ()) if branch.get("source") == "sim"]
    factual = tuple(int(value) for value in episode["executed_actions"][step][:4])
    if len(sim) != 1 or action_from_record(sim[0]).key() != factual:
        raise ValueError("correction comparison does not use the factual recorded action")
    if not record["accepted"]:
        return None
    accepted, action, stats = validated_correction(verdict, min_lcb=min_lcb, require_independent=True)
    if (not accepted or action is None or not math.isfinite(float(stats["lcb95"]))):
        raise ValueError("accepted correction lacks independently certified advantage")
    _replica_provenance(verdict, int(stats["best_index"]))
    target = [*action.key(), 0]
    if canonical(record.get("target")) != canonical(target):
        raise ValueError("accepted target differs from its frozen validated action")
    # This also checks prelude exclusion, canonical WAIT and fresh legality.
    correction_episode(episode, {step: target})
    return target


def validate_corrections(*, dataset: Path, checkpoint: Mapping[str, Any],
                         corrections_manifest: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[int, list[int]]]]:
    """Bind independently recomputed labels to exact, disjoint actor episodes."""
    from .tactical_data import load_episode, validate_manifest

    dataset, corrections_manifest = Path(dataset).resolve(), Path(corrections_manifest).resolve()
    dataset_path = dataset / "manifest.json"
    collection = json.loads(dataset_path.read_text())
    validate_manifest(collection, checkpoint_record=checkpoint)
    manifest = json.loads(corrections_manifest.read_text())
    source = runtime_source_fingerprint()
    if (manifest.get("schema") != CORRECTIONS_SCHEMA
            or canonical(manifest.get("checkpoint")) != canonical(dict(checkpoint))
            or manifest.get("dataset_manifest_sha256") != file_sha(dataset_path)
            or manifest.get("source_sha256") != source["sha256"]
            or collection["identity"]["caller"].get("source_sha256") != source["sha256"]):
        raise ValueError("correction manifest checkpoint/dataset/source binding differs")
    sequences = manifest.get("sequences", [])
    if not isinstance(sequences, list):
        raise ValueError("correction sequences must be a list")
    if sequences:
        raise ValueError("sequence training is pending integration; nonempty sequence manifests are unsupported")
    min_lcb = manifest.get("min_lcb")
    if type(min_lcb) not in (int, float) or not math.isfinite(min_lcb) or min_lcb < 0:
        raise ValueError("correction confidence margin must be finite and nonnegative")
    jobs = collection["jobs"]
    if any(job["driver"] != "actor" for job in jobs):
        raise ValueError("correction learning requires actor-driven source histories")
    if not _rows(collection, "train") or not _rows(collection, "calibration"):
        raise ValueError("complete training and held-out calibration episodes are required")
    origins = [job["seed"] for job in jobs]
    if len(origins) != len(set(origins)):
        raise ValueError("training and held-out episode origins overlap")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("correction manifest must declare evaluated records")
    by_job: dict[str, list[Mapping[str, Any]]] = {job["job_id"]: [] for job in jobs}
    seen: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, Mapping) or record.get("job_id") not in by_job or type(record.get("t")) is not int:
            raise ValueError("correction record has an unknown episode or decision")
        key = (record["job_id"], record["t"])
        if key in seen:
            raise ValueError("duplicate correction decision")
        seen.add(key)
        by_job[key[0]].append(record)
    targets: dict[str, dict[int, list[int]]] = {}
    accepted_train = 0
    for job in jobs:
        origin = collection["completed"][job["job_id"]]
        episode, metadata = load_episode(dataset / origin["path"], expected_identity=collection["identity_sha256"],
                                         expected_sha=origin["sha256"])
        if (canonical(metadata) != canonical({key: origin.get(key) for key in metadata})
                or metadata.get("job_id") != job["job_id"]
                or metadata.get("split") != job["split"] or metadata.get("seed") != job["seed"]):
            raise ValueError("episode metadata differs from its declared collection record")
        labels = {}
        for record in by_job[job["job_id"]]:
            target = _verified_target(record, episode, metadata, origin, min_lcb=float(min_lcb))
            if target is not None:
                labels[record["t"]] = target
        targets[job["job_id"]] = labels
        if job["split"] == "train":
            accepted_train += len(labels)
    if not accepted_train:
        raise ValueError("no independently accepted training corrections")
    return collection, manifest, targets


def _group(dataset: Path, rows: Sequence[Mapping[str, Any]], identity: str,
           targets: Mapping[str, Mapping[int, Sequence[int]]]) -> tuple[list[Any], list[Any]]:
    episodes, metadata = _load_group(dataset, rows, identity)
    return [correction_episode(episode, targets[meta["job_id"]])
            for episode, meta in zip(episodes, metadata, strict=True)], metadata


def _sequence_sources(collection: Mapping[str, Any], manifest: Mapping[str, Any], *, dataset: Path,
                      corrections_manifest: Path) -> tuple[dict[str, tuple[Any, Any]], list[dict[str, Any]]]:
    """Bind certified branch histories to their exact factual episode prefix."""
    from .tactical_data import PUBLIC_KEYS, load_episode

    records = manifest.get("sequences", [])
    if not isinstance(records, list):
        raise ValueError("correction sequences must be a list")
    sources, receipts = {}, []
    if not records:
        return sources, receipts
    from .correction_sequences import load_sequence_capture, validate_sequence_certificate

    jobs = {job["job_id"]: job for job in collection["jobs"]}
    parent = corrections_manifest.parent.resolve()
    seen_origins = set()
    for record in records:
        if not isinstance(record, Mapping) or record.get("accepted") is not True:
            raise ValueError("sequence training requires explicit independent acceptance")
        job_id, step = record.get("origin_job_id"), record.get("t")
        if job_id not in jobs or type(step) is not int:
            raise ValueError("sequence has an unknown factual origin")
        origin = collection["completed"][job_id]
        if record.get("source_episode_sha256") != origin["sha256"]:
            raise ValueError("sequence source episode binding differs")
        relative = Path(record.get("path", ""))
        path = (parent / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(parent) or not relative.parts:
            raise ValueError("sequence path must stay within the correction manifest directory")
        arrays, metadata = load_sequence_capture(path, expected_sha=record.get("sha256"))
        factual, factual_meta = load_episode(dataset / origin["path"], expected_identity=collection["identity_sha256"],
                                             expected_sha=origin["sha256"])
        if not 0 <= step < len(factual["label_valid"]):
            raise ValueError("sequence origin decision is outside its factual episode")
        state_hash = str(factual["state_hash"][step])
        expected = dict(origin_job_id=job_id, source_episode_sha256=origin["sha256"], origin_t=step,
                        origin_state_hash=state_hash, prefix_length=step, split=jobs[job_id]["split"],
                        family=factual_meta["family"])
        if any(metadata.get(key) != value for key, value in expected.items()) or metadata.get("plan_completed") is not True:
            raise ValueError("sequence origin, split, prefix or completion binding differs")
        certificate = record.get("certificate")
        if not isinstance(certificate, Mapping):
            raise ValueError("sequence lacks its independent whole-plan certificate")
        validate_sequence_certificate(certificate, min_lcb=float(manifest["min_lcb"]),
                                      expected_state_hash=state_hash, expected_plan_id=metadata["plan_id"])
        if certificate.get("t") != step or canonical(certificate.get("plan_identity")) != canonical(metadata.get("plan_identity")):
            raise ValueError("sequence plan identity or origin differs from its certificate")
        capture_seed = metadata.get("capture_replica_seed")
        replica_seeds = {replica.get("replica_seed") for arm in certificate["arms"].values()
                         for phase in ("discovery_replicas", "confirmation_replicas") for replica in arm.get(phase, [])}
        if type(capture_seed) is not int or capture_seed in replica_seeds:
            raise ValueError("sequence capture seed must be independent of discovery and confirmation")
        for key in (*PUBLIC_KEYS, "reset_mask", "executed_actions", "state_hash", "elapsed_us"):
            if not np.array_equal(arrays[key][:step], factual[key][:step]):
                raise ValueError(f"sequence factual prefix differs: {key}")
        for key in (*PUBLIC_KEYS, "reset_mask", "state_hash", "elapsed_us"):
            if not np.array_equal(arrays[key][step], factual[key][step]):
                raise ValueError(f"sequence origin public input differs: {key}")
        if bool(np.asarray(arrays["active"][:step]).any()) or not bool(np.asarray(arrays["active"][step:]).any()):
            raise ValueError("sequence supervision must address executed branch rows only")
        identity = (job_id, step, metadata["plan_id"])
        source_id = f"sequence:{metadata['job_id']}"
        if identity in seen_origins or source_id in sources:
            raise ValueError("duplicate certified sequence source")
        seen_origins.add(identity)
        sources[source_id] = arrays, dict(metadata, job_id=source_id)
        receipts.append(dict(source_id=source_id, path=str(path), sha256=file_sha(path),
                             metadata=metadata, certificate_sha256=digest(certificate),
                             active_targets={str(int(index)): np.asarray(arrays["targets_cycle"][index]).tolist()
                                             for index in np.flatnonzero(arrays["active"])}))
    return sources, receipts


def _factor_record(logits: Any, masks: Any, target: Sequence[int], lane: int, step: int) -> dict[str, Any]:
    """Keep mode, conditional card and conditional cell evidence distinct."""
    import torch

    mode, slot, row, col, _ = map(int, target)
    p_mode, log_mode = _safe_log_probs(logits.mode[lane, step], masks.mode[lane, step])
    log_joint = float(log_mode[mode])
    result = dict(play_probability=float(p_mode[1]), target_mode_probability=float(p_mode[mode]),
                  card_probability_given_play=None, cell_probability_given_card=None,
                  nearby_cell_probability_given_card=None)
    if mode == 1:
        p_card, log_card = _safe_log_probs(logits.card[lane, step], masks.card[lane, step])
        place_logits, place_mask = logits.placement[lane, step, slot], masks.placement[lane, step, slot]
        p_cell, log_cell = _safe_log_probs(place_logits.flatten(), place_mask.flatten())
        p_cell, log_cell = p_cell.reshape(place_logits.shape), log_cell.reshape(place_logits.shape)
        # A fixed Chebyshev-radius-one neighborhood is diagnostic only; the
        # exact independently certified cell remains the training target.
        neighborhood = p_cell[max(0, row - 1):row + 2, max(0, col - 1):col + 2].sum()
        result.update(card_probability_given_play=float(p_card[slot]), cell_probability_given_card=float(p_cell[row, col]),
                      nearby_cell_probability_given_card=float(neighborhood))
        log_joint += float(log_card[slot]) + float(log_cell[row, col])
    result.update(nll=-log_joint, joint_probability=math.exp(log_joint))
    if not all(value is None or math.isfinite(value) for value in result.values()):
        raise FloatingPointError("nonfinite correction factor diagnostic")
    return result


def _diagnostics(policy: Any, anchor: Any, *, dataset: Path, rows: Sequence[Mapping[str, Any]],
                 identity: str, targets: Mapping[str, Mapping[int, Sequence[int]]],
                 config: CorrectionTrainingConfig,
                 sequence_sources: Mapping[str, tuple[Any, Any]] | None = None) -> dict[str, Any]:
    import torch

    totals: Counter = Counter()
    factors, retention_values = [], []
    groups = []
    # Load one group at a time; never materialize the full factual corpus.
    for offset in range(0, len(rows), config.batch_episodes):
        groups.append((rows[offset:offset + config.batch_episodes], None))
    for source in (sequence_sources or {}).values():
        groups.append((None, source))
    with torch.no_grad():
        for factual_rows, source in groups:
            episodes, metadata = (_group(dataset, factual_rows, identity, targets) if source is None
                                  else ([source[0]], [source[1]]))
            hidden = policy.initial_hidden(len(episodes), device="cpu")
            old_hidden = anchor.initial_hidden(len(episodes), device="cpu")
            for start in range(0, max(len(episode["label_valid"]) for episode in episodes), config.chunk_length):
                batch = chunk_batch(episodes, metadata, start=start, length=config.chunk_length, arm="control")
                logits, recurrent, hidden = differentiable_unroll(policy, batch, hidden, batch["reset_mask"])
                old, _, old_hidden = differentiable_unroll(anchor, batch, old_hidden, batch["reset_mask"])
                nll = -policy.log_prob(logits, batch["masks"], batch["actions"])
                totals["active_rows"] += int(batch["active"].sum())
                totals["teacher_nll_sum"] += float(nll[batch["active"]].sum())
                if source is None:
                    for key in ("inactive_rows", "real_rows", "valid_rows"):
                        totals[key] += int(batch[key.removesuffix("_rows") if key != "inactive_rows" else "inactive"].sum())
                    kl = full_action_kl(old, logits, batch["masks"])[batch["inactive"]]
                    retention_values.extend(kl.tolist())
                    totals["retention_kl_sum"] += float(kl.sum())
                else:
                    totals["sequence_real_rows"] += int(batch["real"].sum())
                for lane, local_step in batch["active"].nonzero().tolist():
                    target = episodes[lane]["targets_cycle"][start + local_step].tolist()
                    factors.append(dict(job_id=metadata[lane]["job_id"], t=start + local_step,
                        origin_job_id=metadata[lane].get("origin_job_id", metadata[lane]["job_id"]),
                        source_kind="factual" if source is None else "sequence", target=target,
                        state_hash=str(episodes[lane]["state_hash"][start + local_step]),
                        before=_factor_record(old, batch["masks"], target, lane, local_step),
                        after=_factor_record(logits, batch["masks"], target, lane, local_step)))
    return dict(totals, correction_rows=totals["active_rows"],
                correction_nll=totals["teacher_nll_sum"] / totals["active_rows"] if totals["active_rows"] else None,
                retention_kl=totals["retention_kl_sum"] / max(1, totals["inactive_rows"]),
                retention_kl_max=max(retention_values, default=None),
                retention_kl_p95=float(np.percentile(retention_values, 95)) if retention_values else None,
                factors=factors, neighborhood="legal Chebyshev-radius-one cells around the exact target",
                used_for_selection=False, frozen_full_history_replay=True)


def _epoch_steps(rows: Sequence[Mapping[str, Any]], config: CorrectionTrainingConfig, epoch: int) -> int:
    order = np.random.default_rng(config.seed + epoch).permutation(len(rows))
    return sum(math.ceil(max(int(rows[int(index)]["decisions"]) for index in order[offset:offset + config.batch_episodes])
                         / config.chunk_length) for offset in range(0, len(order), config.batch_episodes))


def _population(dataset: Path, rows: Sequence[Mapping[str, Any]], identity: str,
                targets: Mapping[str, Mapping[int, Sequence[int]]]) -> dict[str, int]:
    counts = dict(real_rows=0, valid_rows=0, active_rows=0, inactive_rows=0)
    for row in rows:
        episodes, _ = _group(dataset, [row], identity, targets)
        episode = episodes[0]
        valid, active = np.asarray(episode["label_valid"]), np.asarray(episode["active"])
        counts["real_rows"] += len(valid)
        counts["valid_rows"] += int(valid.sum())
        counts["active_rows"] += int((valid & active).sum())
        counts["inactive_rows"] += int((valid & ~active).sum())
    return counts


def _epoch_workload(rows: Sequence[Mapping[str, Any]], targets: Mapping[str, Mapping[int, Sequence[int]]],
                    sequences: Mapping[str, tuple[Any, Any]], population: Mapping[str, int],
                    config: CorrectionTrainingConfig, epoch: int) -> tuple[Any, dict[str, int]]:
    factual = {row["job_id"]: sorted(targets[row["job_id"]]) for row in rows}
    branch = {job: np.flatnonzero(episode["active"]).tolist() for job, (episode, _) in sequences.items()}
    steps = _epoch_steps(rows, config, epoch)
    schedule = exposure_schedule(factual, branch, exposures=config.correction_exposures,
                                 optimizer_steps=steps, seed=config.seed + epoch)
    order = np.random.default_rng(config.seed + epoch).permutation(len(rows))
    ordinary_active = []
    for offset in range(0, len(order), config.batch_episodes):
        group = [rows[int(index)] for index in order[offset:offset + config.batch_episodes]]
        for start in range(0, max(int(row["decisions"]) for row in group), config.chunk_length):
            ordinary_active.append(any(start <= step < start + config.chunk_length for row in group
                                       for step in factual[row["job_id"]]))
    focused = [target for update in schedule for target in update]
    correction_steps = sum(bool(active or extra) for active, extra in zip(ordinary_active, schedule, strict=True))
    return schedule, dict(population, optimizer_steps=steps, correction_optimizer_steps=correction_steps,
        retention_only_optimizer_steps=steps - correction_steps, focused_windows=len(focused),
        focused_target_presentations=len(focused), correction_presentations=population["active_rows"] + len(focused),
        focused_gradient_frame_visits=sum(min(config.chunk_length, step + 1) for _, step in focused),
        focused_burnin_frame_visits=sum(max(0, step + 1 - config.chunk_length) for _, step in focused))


def _focused_loss(policy: Any, episode: Mapping[str, Any], metadata: Mapping[str, Any], *,
                  step: int, config: CorrectionTrainingConfig) -> Any:
    """Rebuild the current student's entire prefix before a targeted window."""
    import torch

    if not bool(episode["active"][step]):
        raise ValueError("focused exposure must address an independently certified target")
    focused = dict(episode)
    focused["active"] = np.zeros_like(episode["active"], dtype=bool)
    focused["active"][step] = True
    start = max(0, step + 1 - config.chunk_length)
    hidden = policy.initial_hidden(1, device="cpu")
    with torch.no_grad():
        for prefix in range(0, start, config.chunk_length):
            batch = chunk_batch([focused], [metadata], start=prefix,
                                length=min(config.chunk_length, start - prefix), arm="control")
            _, _, hidden = differentiable_unroll(policy, batch, hidden, batch["reset_mask"])
    batch = chunk_batch([focused], [metadata], start=start, length=step + 1 - start, arm="control")
    logits, _, _ = differentiable_unroll(policy, batch, hidden.detach(), batch["reset_mask"])
    return -policy.log_prob(logits, batch["masks"], batch["actions"])[batch["active"]].mean()


def _validate_completed_result(result: Mapping[str, Any], *, out: Path, run_hash: str,
                               checkpoint: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                               targets: Mapping[str, Mapping[int, Sequence[int]]],
                               config: CorrectionTrainingConfig, workloads: Sequence[Mapping[str, int]]) -> None:
    import torch

    exported = result.get("checkpoint", {})
    expected_actor = out / f"actor_epoch_{config.epochs:03d}.pt"
    state_path = out / "training_state.pt"
    frames = sum(int(row["decisions"]) for row in rows)
    corrections = sum(len(targets[row["job_id"]]) for row in rows)
    steps = [workload["optimizer_steps"] for workload in workloads]
    if (result.get("run_hash") != run_hash or result.get("status") != "complete"
            or result.get("final_epoch") != config.epochs or result.get("selected_actor_changed") is not False
            or Path(exported.get("path", "")).resolve() != expected_actor
            or exported.get("model_config") != dict(checkpoint["model_config"], fixed_wait=True)
            or not expected_actor.is_file() or file_sha(expected_actor) != exported.get("sha256")
            or not state_path.is_file()):
        raise ValueError("completed correction actor export changed")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    history = state.get("history", [])
    if (state.get("run_hash") != run_hash or state.get("epoch") != config.epochs
            or len(history) != config.epochs or canonical(history) != canonical(result.get("history"))
            or any(entry.get("epoch") != epoch + 1
                   or any(entry.get(key) != value for key, value in workloads[epoch].items())
                   for epoch, entry in enumerate(history))
            or result.get("total_training_frame_visits") != frames * config.epochs
            or result.get("total_optimizer_steps") != sum(steps)
            or result.get("accepted_training_corrections") != corrections
            or result.get("training_episodes") != len(rows)):
        raise ValueError("completed correction fit state or declared frame/update dose differs")
    for key in ("diagnostics_before", "diagnostics_after", "learning_gate"):
        if not isinstance(result.get(key), Mapping) or canonical(result[key]) != canonical(state.get(key)):
            raise ValueError("completed correction diagnostics differ from sealed training state")
    if (canonical(result.get("calibration")) != canonical(result["diagnostics_after"].get("calibration"))
            or canonical(result["learning_gate"]) != canonical(correction_learning_gate(
                result["diagnostics_after"]["train"], result["diagnostics_after"]["calibration"]))):
        raise ValueError("completed correction calibration or learning gate differs")
    for key in ("correction_presentations", "focused_windows", "focused_target_presentations",
                "focused_gradient_frame_visits", "focused_burnin_frame_visits", "correction_optimizer_steps",
                "retention_only_optimizer_steps"):
        if result.get(f"total_{key}") != sum(workload[key] for workload in workloads):
            raise ValueError("completed correction focused dose differs")
    actor = torch.load(expected_actor, map_location="cpu", weights_only=True)
    trained = state.get("actor", {})
    if (not isinstance(actor, Mapping) or set(actor) != set(trained)
            or any(not isinstance(value, torch.Tensor) or not isinstance(trained[name], torch.Tensor)
                   or value.dtype != trained[name].dtype or value.shape != trained[name].shape
                   or not torch.equal(value, trained[name]) for name, value in actor.items())):
        raise ValueError("exported correction actor differs from its final trained state")


def train_corrections(*, dataset: Path, out: Path, checkpoint: Mapping[str, Any], corrections_manifest: Path,
                      config: CorrectionTrainingConfig = CorrectionTrainingConfig(), resume: bool = False) -> dict[str, Any]:
    """Fit the predeclared final epoch; never alter the selected actor receipt."""
    import torch
    from .model_v4 import ModelConfigV4
    from .v4_ppo.runtime import configure_v4_runtime, load_frozen_anchor

    torch.set_num_threads(1)
    dataset, out, corrections_manifest = Path(dataset).resolve(), Path(out).resolve(), Path(corrections_manifest).resolve()
    collection, corrections, targets = validate_corrections(dataset=dataset, checkpoint=checkpoint,
                                                           corrections_manifest=corrections_manifest)
    source = runtime_source_fingerprint()
    rows, held_out = _rows(collection, "train"), _rows(collection, "calibration")
    identity = collection["identity_sha256"]
    population = _population(dataset, rows, identity, targets)
    rows_by_job = {row["job_id"]: row for row in rows}
    epoch_plans = [_epoch_workload(rows, targets, {}, population, config, epoch)
                   for epoch in range(config.epochs)]
    workloads = [workload for _, workload in epoch_plans]
    gate_config = CorrectionGateConfig()
    protocol = dict(version=TRAINING_VERSION, config=asdict(config), checkpoint=dict(checkpoint), source=source,
                    dataset=str(dataset), dataset_manifest_sha256=file_sha(dataset / "manifest.json"),
                    corrections_manifest=str(corrections_manifest), corrections_manifest_sha256=file_sha(corrections_manifest),
                    inference="native-stochastic-fixed250ms-carry", final_epoch=config.epochs,
                    objective="weighted verified correction NLL plus summed focused target NLL; full legal anchor KL on every other valid row",
                    focused_history="current student full-prefix burn-in; final chunk differentiated; no extra retention or optimizer steps",
                    workloads=workloads, learning_gate=asdict(gate_config),
                    calibration="disjoint complete episodes; no gradients or epoch selection")
    run_hash = digest(protocol)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        if not resume or json.loads((out / "protocol.json").read_text()) != dict(protocol=protocol, run_hash=run_hash):
            raise ValueError("existing correction fit requires identical explicit resume")
    elif any(out.iterdir()):
        raise ValueError("new correction fit output must be empty")
    atomic_json(out / "protocol.json", dict(protocol=protocol, run_hash=run_hash))
    if (out / "result.json").exists():
        result = json.loads((out / "result.json").read_text())
        _validate_completed_result(result, out=out, run_hash=run_hash, checkpoint=checkpoint,
                                   rows=rows, targets=targets, config=config, workloads=workloads)
        return result
    torch.manual_seed(config.seed)
    anchor = load_frozen_anchor(checkpoint["path"], config=ModelConfigV4(**checkpoint["model_config"]),
                                expected_sha256=checkpoint["sha256"], device="cpu")
    if float(anchor.config.dropout) != 0.0:
        raise ValueError("correction fitting requires dropout=0 for deterministic carried histories")
    configure_v4_runtime(anchor)
    student = copy.deepcopy(anchor)
    student.train()
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    initial_recurrent = {name: value.detach().clone() for name, value in student.recurrent.named_parameters() if "weight_hh" in name}
    diagnostics_before = {split: _diagnostics(anchor, anchor, dataset=dataset, rows=split_rows,
        identity=identity, targets=targets, config=config)
        for split, split_rows in (("train", rows), ("calibration", held_out))}
    history: list[dict[str, Any]] = []
    start_epoch = 0
    state_path = out / "training_state.pt"
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        saved_epoch = saved.get("epoch")
        saved_history = saved.get("history", [])
        if (not resume or saved.get("run_hash") != run_hash
                or type(saved_epoch) is not int or not 0 <= saved_epoch <= config.epochs
                or len(saved_history) != saved_epoch
                or canonical(saved.get("diagnostics_before")) != canonical(diagnostics_before)
                or any(entry.get("epoch") != epoch + 1
                       or any(entry.get(key) != value for key, value in workloads[epoch].items())
                       for epoch, entry in enumerate(saved_history))):
            raise ValueError("partial correction fit requires identical explicit resume")
        student.load_state_dict(saved["actor"])
        optimizer.load_state_dict(saved["optimizer"])
        history, start_epoch = saved_history, saved_epoch
    expected_corrections = sum(len(targets[row["job_id"]]) for row in rows)
    started = time.monotonic()
    for epoch in range(start_epoch, config.epochs):
        order = np.random.default_rng(config.seed + epoch).permutation(len(rows))
        schedule, workload = epoch_plans[epoch]
        totals: Counter = Counter()
        maximum_recurrent_gradient = 0.0
        for offset in range(0, len(order), config.batch_episodes):
            episodes, metadata = _group(dataset, [rows[int(index)] for index in order[offset:offset + config.batch_episodes]], identity, targets)
            hidden = student.initial_hidden(len(episodes), device="cpu")
            old_hidden = anchor.initial_hidden(len(episodes), device="cpu")
            for start in range(0, max(len(episode["label_valid"]) for episode in episodes), config.chunk_length):
                batch = chunk_batch(episodes, metadata, start=start, length=config.chunk_length, arm="control")
                optimizer.zero_grad(set_to_none=True)
                logits, recurrent, hidden = differentiable_unroll(student, batch, hidden, batch["reset_mask"])
                with torch.no_grad():
                    old, _, old_hidden = differentiable_unroll(anchor, batch, old_hidden, batch["reset_mask"])
                batch["tactical_weights"] *= config.correction_weight
                loss, details, _ = tactical_batch_loss(student, logits, old, recurrent, batch, config=config.tactical_config())
                loss.backward()
                focused = schedule[totals["optimizer_steps"]]
                for source_id, step in focused:
                    # Only one focused source is loaded at a time. Rebuild all
                    # preceding observations using this update's current actor.
                    focused_episodes, focused_metadata = _group(dataset,
                        [rows_by_job[source_id]], identity, targets)
                    focused_loss = _focused_loss(student, focused_episodes[0], focused_metadata[0],
                                                 step=step, config=config)
                    (config.correction_weight * focused_loss).backward()
                    totals["focused_nll_sum"] += float(focused_loss.detach())
                    totals["focused_windows"] += 1
                    totals["focused_target_presentations"] += 1
                    totals["focused_gradient_frame_visits"] += min(config.chunk_length, step + 1)
                    totals["focused_burnin_frame_visits"] += max(0, step + 1 - config.chunk_length)
                totals["correction_optimizer_steps" if details["active_rows"] or focused
                       else "retention_only_optimizer_steps"] += 1
                gradient = torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip, error_if_nonfinite=True)
                maximum_recurrent_gradient = max(maximum_recurrent_gradient, math.sqrt(sum(float(parameter.grad.square().sum())
                    for name, parameter in student.recurrent.named_parameters() if "weight_hh" in name and parameter.grad is not None)))
                optimizer.step()
                hidden, old_hidden = hidden.detach(), old_hidden.detach()
                for key in ("active_rows", "inactive_rows", "real_rows", "valid_rows", "teacher_nll_sum", "retention_kl_sum"):
                    totals[key] += details[key]
                totals["optimizer_steps"] += 1
                totals["maximum_gradient_norm"] = max(totals["maximum_gradient_norm"], float(gradient))
            atomic_json(out / "progress.json", dict(status="training", epoch=epoch + 1, run_hash=run_hash,
                completed_episode_groups=offset // config.batch_episodes + 1,
                total_episode_groups=math.ceil(len(rows) / config.batch_episodes), real_rows=int(totals["real_rows"])))
        totals["correction_presentations"] = totals["active_rows"] + totals["focused_target_presentations"]
        # Counter omits untouched zero fields; write every declared dose field.
        for key in workload:
            totals[key] += 0
        if any(totals[key] != value for key, value in workload.items()):
            raise RuntimeError("correction fit lost/repeated chronological rows, updates or accepted targets")
        entry = dict(epoch=epoch + 1, **totals, correction_nll=totals["teacher_nll_sum"] / expected_corrections,
                     retention_kl=totals["retention_kl_sum"] / max(1, totals["inactive_rows"]),
                     recurrent_gradient_max=maximum_recurrent_gradient)
        history.append(entry)
        _atomic_save(state_path, dict(run_hash=run_hash, actor=student.state_dict(), optimizer=optimizer.state_dict(),
                                     epoch=epoch + 1, history=history, diagnostics_before=diagnostics_before))
        atomic_json(out / "history.json", history)
        print(f"corrections epoch {epoch + 1}/{config.epochs}: {int(totals['real_rows'])} rows, "
              f"{expected_corrections} corrections, NLL {entry['correction_nll']:.4f}, KL {entry['retention_kl']:.4f}", flush=True)
    student.eval()
    diagnostics_after = {split: _diagnostics(student, anchor, dataset=dataset, rows=split_rows,
        identity=identity, targets=targets, config=config)
        for split, split_rows in (("train", rows), ("calibration", held_out))}
    calibration = diagnostics_after["calibration"]
    learning_gate = correction_learning_gate(diagnostics_after["train"], calibration, config=gate_config)
    if (runtime_source_fingerprint() != source
            or file_sha(dataset / "manifest.json") != protocol["dataset_manifest_sha256"]
            or file_sha(corrections_manifest) != protocol["corrections_manifest_sha256"]
            or file_sha(checkpoint["path"]) != checkpoint["sha256"]):
        raise RuntimeError("correction source, checkpoint, dataset or labels changed during fitting")
    actor_path = out / f"actor_epoch_{config.epochs:03d}.pt"
    _atomic_save(actor_path, {name: value.detach().cpu().clone() for name, value in student.state_dict().items()})
    recurrent_change = math.sqrt(sum(float((value.detach() - initial_recurrent[name]).square().sum())
                                    for name, value in student.recurrent.named_parameters() if name in initial_recurrent))
    result = dict(status="complete", run_hash=run_hash, final_epoch=config.epochs,
        checkpoint=dict(path=str(actor_path), sha256=file_sha(actor_path), model_config=asdict(student.config)),
        history=history, calibration=calibration, diagnostics_before=diagnostics_before,
        diagnostics_after=diagnostics_after, learning_gate=learning_gate,
        training_episodes=len(rows), calibration_episodes=len(held_out),
        accepted_training_corrections=expected_corrections,
        accepted_calibration_corrections=sum(len(targets[row["job_id"]]) for row in held_out),
        evaluated_records=len(corrections["records"]), rejected_records=sum(not row["accepted"] for row in corrections["records"]),
        recurrent_weight_change=recurrent_change,
        actor_parameter_updates=any(not torch.equal(value, anchor.state_dict()[name]) for name, value in student.state_dict().items()),
        total_training_frame_visits=sum(int(entry["real_rows"]) for entry in history),
        total_optimizer_steps=sum(int(entry["optimizer_steps"]) for entry in history),
        **{f"total_{key}": sum(int(entry[key]) for entry in history) for key in (
            "correction_presentations", "focused_windows", "focused_target_presentations",
            "focused_gradient_frame_visits", "focused_burnin_frame_visits", "correction_optimizer_steps",
            "retention_only_optimizer_steps")},
        selected_actor_changed=False, wall_seconds_this_invocation=time.monotonic() - started,
        limitation="Correction NLL and held-out KL do not establish stronger play; fresh complete paired matches are required.")
    _atomic_save(state_path, dict(run_hash=run_hash, actor=student.state_dict(), optimizer=optimizer.state_dict(),
        epoch=config.epochs, history=history, diagnostics_before=diagnostics_before,
        diagnostics_after=diagnostics_after, learning_gate=learning_gate))
    atomic_json(out / "result.json", result)
    atomic_json(out / "progress.json", dict(status="complete", run_hash=run_hash, checkpoint=result["checkpoint"]))
    return result
