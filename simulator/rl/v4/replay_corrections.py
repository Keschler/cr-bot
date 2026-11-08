"""Validated single-decision repairs on complete native actor trajectories.

Search can inspect simulator consequences, but proposals and actor inputs use
only public observations. A frozen discovery winner is confirmed on disjoint
policy-sampling replicas at a longer horizon. These are local corrections,
not independent match origins or evidence of a stronger deployed actor.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .policy_development import atomic_json, canonical, digest, file_sha, runtime_source_fingerprint
from .tactical_data import INPUT_KEYS, PUBLIC_KEYS

VERSION = "v4-replay-corrections-v1"
ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class CorrectionSearchConfig:
    discovery_horizon: int = 128
    confirmation_horizon: int = 256
    discovery_replicas: int = 3
    confirmation_replicas: int = 6
    max_branches: int = 10
    max_per_episode: int = 3
    train_budget: int = 36
    calibration_budget: int = 12
    min_spacing: int = 24
    upstream_lookback: int = 48
    min_lcb: float = .05
    seed: int = 2026102900

    def __post_init__(self) -> None:
        for name in ("discovery_horizon", "confirmation_horizon", "discovery_replicas",
                     "confirmation_replicas", "max_branches", "max_per_episode", "train_budget",
                     "calibration_budget", "min_spacing", "upstream_lookback"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.discovery_replicas < 2 or self.confirmation_replicas < 2 or self.max_branches < 3:
            raise ValueError("search needs at least two replicas per phase and three branches")
        if self.confirmation_horizon < self.discovery_horizon or self.confirmation_horizon > 1440:
            raise ValueError("confirmation must be at least as long as discovery and at most a match")
        if not math.isfinite(self.min_lcb) or self.min_lcb < 0:
            raise ValueError("min_lcb must be finite and nonnegative")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _action(value: Sequence[int]) -> dict[str, int]:
    mode, slot, row, col, duration = (int(x) for x in value)
    if duration != 0:
        raise ValueError("correction search requires fixed native WAIT")
    return dict(mode=mode, slot=slot, row=row, col=col)


def episode_match(arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt exact native action records without losing opponent style identity."""
    from simulator.roster import PLAYER_DECK
    from .tactical_data import CollectionJob

    job = CollectionJob(**{key: metadata[key] for key in CollectionJob.__dataclass_fields__})
    if job.driver != "actor" or metadata["native_actor_memory"] != "carry":
        raise ValueError("corrections require native actor trajectories with carried memory")
    opponent = dict(job.opponent)
    if opponent.get("kind") == "v4-checkpoint" or "checkpoint" in opponent:
        raise ValueError("midmatch learned opponents require restoring their own recurrent history")
    return dict(config=dict(match_seed=job.seed, learner_deck=list(PLAYER_DECK),
                            opponent_deck=list(opponent["cards"]), strategy=opponent["strategy"],
                            controller_seed=opponent["controller_seed"], opponent_spec=opponent),
                decisions=[dict(t=t, state_hash=str(state_hash),
                                learner=_action(arrays["executed_actions"][t]),
                                opponent=_action(arrays["opponent_actions"][t]))
                           for t, state_hash in enumerate(arrays["state_hash"])])


def replay_episode_states(arrays: Mapping[str, Any], metadata: Mapping[str, Any],
                          offsets: Sequence[int], policy: Any) -> dict[int, tuple[Any, Any]]:
    """Verify every prefix state AND public input; carry every real GRU step."""
    import torch
    from simulator.public_state_estimator import PublicStateEstimator
    from .frontier_eval import _fresh_env, _recorded_action
    from .v4_ppo.runtime import batch_v4_inputs, configure_v4_runtime, observe_v4_inputs

    requested = set(offsets)
    if not requested:
        return {}
    if any(type(t) is not int or not 0 <= t < len(arrays["state_hash"]) for t in requested):
        raise ValueError("candidate outside the complete episode")
    match = episode_match(arrays, metadata)
    env = _fresh_env(match)
    estimator = PublicStateEstimator()
    configure_v4_runtime(policy)
    policy.eval()
    hidden = policy.initial_hidden(1, device="cpu")
    result = {}
    with torch.no_grad():
        for t in range(max(requested) + 1):
            if env.state.state_hash() != str(arrays["state_hash"][t]):
                raise ValueError(f"exact prefix state diverged at decision {t}")
            _, inputs, _, _ = observe_v4_inputs(env, estimator)
            for name in PUBLIC_KEYS:
                if not np.array_equal(np.asarray(inputs[name]), arrays[name][t]):
                    raise ValueError(f"public prefix input {name} diverged at decision {t}")
            _, _, hidden = policy.forward_step(batch_v4_inputs(inputs, "cpu"), hidden=hidden,
                reset_mask=torch.tensor([[t == 0]], dtype=torch.bool))
            if t in requested:
                env.observe_v2_for_viewer(1)
                result[t] = (env.fork(), hidden.detach().clone())
            if t < max(requested):
                rec = match["decisions"][t]
                env.step_v2_for_viewer((_recorded_action(rec["learner"], 0),
                                        _recorded_action(rec["opponent"], 1)), viewer=0)
    return result


def select_episode_candidates(arrays: Mapping[str, Any], metadata: Mapping[str, Any],
                              config: CorrectionSearchConfig) -> list[dict[str, Any]]:
    """Mine public contexts and earlier spends, never treating flags as errors."""
    from .correction_proposals import candidate_categories

    length = len(arrays["label_valid"])
    tags: dict[int, set[str]] = {}
    for t in range(length):
        if not bool(arrays["label_valid"][t]):
            continue
        inputs = {key: arrays[key][t] for key in PUBLIC_KEYS}
        found = set(candidate_categories(inputs, _action(arrays["executed_actions"][t]),
                                         own_elixir=float(arrays["own_elixir"][t])))
        if found:
            tags[t] = found
    # Reserve the first available response to each publicly distinguishable
    # stationary siege position. No target IDs or authoritative lock states
    # enter this detector. It is an interception opportunity, not a guarantee
    # that connection can be prevented.
    from .simulator_teacher import _entity_card
    seen_siege: set[tuple[str, float, float]] = set()
    onsets: list[tuple[int, str]] = []
    pending_siege: list[int] = []
    for t in range(length):
        current_siege = set()
        for token in arrays["entities"][t][np.asarray(arrays["entity_mask"][t], bool)]:
            if token[1] <= .5 or token[4] <= 0 or token[9] <= .5 or token[10] <= .5:
                continue
            name = _entity_card(token).get("name")
            if name in ("x-bow", "mortar"):
                key = (name, round(float(token[2]), 4), round(float(token[3]), 4))
                current_siege.add(key)
                if key not in seen_siege:
                    seen_siege.add(key)
                    pending_siege.append(t)
                    onsets.append((t, "siege"))
        seen_siege = current_siege
        if "air" in tags.get(t, ()) and (t == 0 or "air" not in tags.get(t - 1, ())):
            onsets.append((t, "air"))
        if pending_siege and bool(arrays["label_valid"][t]):
            hand = arrays["hand_tokens"][t]
            defensive = np.isin(np.rint(hand[:, 0] * 127).astype(int), (114, 51, 72))
            affordable = np.asarray(arrays["legal_play"][t]).reshape(4, -1).any(axis=1)
            if np.any(defensive & affordable):
                if any(t - first <= 24 for first in pending_siege):
                    tags.setdefault(t, set()).add("interception-window")
                pending_siege.clear()
            else:
                pending_siege = [first for first in pending_siege if t - first <= 24]
    # Future public observations select histories for review, never policy
    # inputs. Include real pre-threat spends, not every later ongoing-threat
    # spend merely because another threat frame occurs in the next 12s.
    local = {t: set(value) for t, value in tags.items()}
    for t, category in onsets:
        for earlier in range(max(0, t - config.upstream_lookback), t):
            if (bool(arrays["label_valid"][earlier])
                    and int(arrays["executed_actions"][earlier][0]) == 1
                    and category not in local.get(earlier, ())):
                tags.setdefault(earlier, set()).update(("upstream-spend", "pre-threat-spend"))
    # Also revisit spending before the first unaffordable in-hand air answer.
    # This is a mining flag; branch validation decides if reserving was useful.
    starving = False
    for t in range(length):
        hand = arrays["hand_tokens"][t]
        answer = np.isin(np.rint(hand[:, 0] * 127).astype(int), (72, 28)).any()
        now = "air" in local.get(t, ()) and answer and float(arrays["own_elixir"][t]) < 4
        if now and not starving:
            for earlier in range(max(0, t - config.upstream_lookback), t):
                action = arrays["executed_actions"][earlier]
                if bool(arrays["label_valid"][earlier]) and int(action[0]) == 1:
                    cost = float(arrays["hand_tokens"][earlier, int(action[1]), 1]) * 10
                    if cost >= 2 and float(arrays["own_elixir"][earlier]) >= 4:
                        tags.setdefault(earlier, set()).update(("upstream-spend", "pre-starvation-spend"))
        starving = bool(now)
    ranked = []
    for t, value in tags.items():
        damage = float(np.asarray(arrays["outcome_values"])[t, -1, 0])
        active_priority = int("siege" in value) + int("air" in value)
        spend = int(arrays["executed_actions"][t][0]) == 1
        # Priority is a review budget, not a correctness or reward label.
        priority = damage * 100 + active_priority * .1 + int(spend) * .01
        ranked.append(dict(job_id=metadata["job_id"], t=t, state_hash=str(arrays["state_hash"][t]),
                           tags=sorted(value), priority=priority, review_damage_next12s=damage))
    ranked.sort(key=lambda row: (-row["priority"], row["t"]))
    chosen: list[dict[str, Any]] = []
    # Cover different behavioral contexts before filling the remaining quota.
    for category in ("interception-window", "pre-starvation-spend", "pre-threat-spend", "air", "ranged-ground", "siege", "upstream-spend"):
        for row in ranked:
            if category in row["tags"] and all(abs(row["t"] - old["t"]) >= config.min_spacing for old in chosen):
                chosen.append(row)
                break
        if len(chosen) >= config.max_per_episode:
            return chosen
    for row in ranked:
        if all(abs(row["t"] - old["t"]) >= config.min_spacing for old in chosen):
            chosen.append(row)
        if len(chosen) >= config.max_per_episode:
            break
    return chosen


def _finite(value: Any) -> Any:
    """JSON reports preserve unresolved uncertainty as null, never fake zero."""
    if isinstance(value, Mapping):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_correction(*, env: Any, hidden: Any, policy: Any, candidate: Mapping[str, Any],
                        branches: Sequence[Any], opponent_spec: Mapping[str, Any],
                        config: CorrectionSearchConfig) -> dict[str, Any]:
    """Freeze discovery selection before disjoint, longer-horizon confirmation."""
    from .frontier import _stable_seed
    from .frontier_eval import action_identity, action_key, run_branch
    from .search_distillation.targets import validated_correction

    if env.state.state_hash() != candidate["state_hash"]:
        raise ValueError("prepared state differs from the candidate origin")
    if not branches or branches[0].source != "sim" or len({action_key(b) for b in branches}) != len(branches):
        raise ValueError("branches require exact sim first and distinct action identities")
    if opponent_spec.get("kind") == "v4-checkpoint" or "checkpoint" in opponent_spec:
        raise ValueError("unrestored learned opponents are excluded")
    discovery_ids = list(range(config.discovery_replicas))
    confirmation_ids = list(range(config.discovery_replicas, config.discovery_replicas + config.confirmation_replicas))
    seeds = {_index: _stable_seed(VERSION, config.seed, candidate["state_hash"], _index) & 0xFFFFFFFF
             for _index in discovery_ids + confirmation_ids}
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("discovery and confirmation sampling seeds collided")
    reports = [{**action_identity(branch), "values": [], "replicas": [],
                "discovery_replica_indices": discovery_ids} for branch in branches]

    def run(branch: Any, index: int, horizon: int) -> dict[str, Any]:
        seed = seeds[index]
        value = run_branch(env=env, hidden=hidden, branch=branch, opponent_spec=dict(opponent_spec),
                           replica_seed=seed, horizon=horizon, policy=policy, device="cpu")
        if not math.isfinite(float(value["value"])) or env.state.state_hash() != candidate["state_hash"]:
            raise ValueError("branch produced nonfinite value or mutated its origin")
        return dict(replica_index=index, replica_seed=seed, **value)

    for index in discovery_ids:
        for branch, report in zip(branches, reports, strict=True):
            result = run(branch, index, config.discovery_horizon)
            report["values"].append(float(result["value"]))
            report["replicas"].append(result)
    for report in reports:
        report["mean"] = float(np.mean(report["values"]))
    best_index = max(range(len(reports)), key=lambda i: reports[i]["mean"])
    verdict = dict(version=VERSION, state_hash=candidate["state_hash"], t=candidate["t"],
                   branches=reports, discovery_replica_indices=discovery_ids,
                   discovery_horizon=config.discovery_horizon, confirmation_horizon=config.confirmation_horizon,
                   opponent_specs=[dict(opponent_spec)], frozen_action=action_identity(branches[best_index]))
    if best_index != 0:
        verdict["confirmation_replica_indices"] = confirmation_ids
        for branch_index in (0, best_index):
            report = reports[branch_index]
            results = [run(branches[branch_index], index, config.confirmation_horizon) for index in confirmation_ids]
            report.update(confirmation_values=[float(result["value"]) for result in results],
                          confirmation_replica_indices=confirmation_ids, confirmation_replicas=results)
    accepted, target, statistics = validated_correction(verdict, min_lcb=config.min_lcb, require_independent=True)
    return dict(accepted=accepted, target=None if target is None else [*target.key(), 0],
                verdict=verdict, statistics=_finite(statistics),
                rejection=None if accepted else ("discovery_kept_recorded_action" if best_index == 0 else "confirmation_below_lcb"))


def prepare(*, out: Path, actor_json: Path, config: CorrectionSearchConfig,
            training: Mapping[str, Any], workers: int = 3, deck_ids: Sequence[str] = ()) -> dict[str, Any]:
    from .bootstrap import catalog_training_opponent_specs
    from .tactical_data import build_collection_jobs
    from .real_decks import catalog_sha256

    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be in 1..8")
    receipt = json.loads(Path(actor_json).read_text())
    checkpoint = dict(receipt.get("checkpoint", receipt))
    checkpoint["path"] = str(Path(checkpoint["path"]).resolve())
    if file_sha(checkpoint["path"]) != checkpoint["sha256"] or not checkpoint.get("model_config", {}).get("fixed_wait"):
        raise ValueError("a hashed fixed-WAIT actor with full model configuration is required")
    specs = list(catalog_training_opponent_specs())
    if deck_ids:
        if len(set(deck_ids)) != len(deck_ids) or not set(deck_ids) <= {s.deck_id for s in specs}:
            raise ValueError("deck subset must contain unique training-catalog identities")
        specs = [spec for spec in specs if spec.deck_id in deck_ids]
    jobs = build_collection_jobs(specs, config.seed, drivers=("actor",))
    source = runtime_source_fingerprint()
    plan = dict(version=VERSION, checkpoint=checkpoint, source=source, config=asdict(config),
                training=dict(training), workers=workers, jobs=[asdict(job) for job in jobs],
                catalog_sha256=catalog_sha256(), subset=bool(deck_ids),
                screen_train_seeds=[config.seed + 101], screen_dev_seeds=[config.seed + 102],
                confirmation_seeds=[config.seed + 201, config.seed + 202],
                source_actor_selection_receipt_sha256=file_sha(actor_json),
                inference="native-stochastic-fixed250ms-carry", selected_actor_changed=False)
    plan["plan_sha256"] = digest(plan)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plan.json"
    if path.exists():
        if canonical(json.loads(path.read_text())) != canonical(plan):
            raise ValueError("existing correction plan differs; use a new output directory")
    elif any(out.iterdir()):
        raise ValueError("new correction output must be empty")
    atomic_json(path, plan)
    return plan


def load_plan(out: Path) -> dict[str, Any]:
    plan = json.loads((Path(out) / "plan.json").read_text())
    if digest({key: value for key, value in plan.items() if key != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("correction plan seal differs")
    if runtime_source_fingerprint() != plan["source"]:
        raise ValueError("correction runtime source changed; use a new output directory")
    if file_sha(plan["checkpoint"]["path"]) != plan["checkpoint"]["sha256"]:
        raise ValueError("correction origin actor changed")
    return plan


def collect(out: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    from .tactical_data import collect_dataset
    identity = dict(plan_sha256=plan["plan_sha256"], source_sha256=plan["source"]["sha256"],
                    source_files={str(ROOT / key): value for key, value in plan["source"]["files"].items()})
    path = Path(out) / "dataset"
    return collect_dataset(plan["jobs"], plan["checkpoint"], path, identity,
                           workers=plan["workers"], resume=(path / "manifest.json").exists())


def _dataset(out: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    from .tactical_data import validate_manifest
    manifest = json.loads((out / "dataset/manifest.json").read_text())
    validate_manifest(manifest, checkpoint_record=plan["checkpoint"])
    if (manifest["identity"]["caller"].get("plan_sha256") != plan["plan_sha256"]
            or canonical(manifest["jobs"]) != canonical(plan["jobs"])):
        raise ValueError("correction corpus belongs to another declared job grid")
    return manifest


def _search_job(out_string: str, job_id: str, selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from .tactical_data import _torch_runtime, load_episode
    from .v4_ppo.runtime import load_frozen_anchor
    from .model_v4 import ModelConfigV4
    from .correction_proposals import propose_correction_branches

    _torch_runtime()
    out = Path(out_string)
    plan = load_plan(out)
    manifest = _dataset(out, plan)
    source_row = manifest["completed"][job_id]
    arrays, meta = load_episode(out / "dataset" / source_row["path"],
                               expected_identity=manifest["identity_sha256"], expected_sha=source_row["sha256"])
    checkpoint = plan["checkpoint"]
    policy = load_frozen_anchor(checkpoint["path"], config=ModelConfigV4(**checkpoint["model_config"]),
                                expected_sha256=checkpoint["sha256"], device="cpu")
    states = replay_episode_states(arrays, meta, [int(row["t"]) for row in selected], policy)
    config = CorrectionSearchConfig(**plan["config"])
    records = []
    for candidate in selected:
        t = int(candidate["t"])
        inputs = {key: arrays[key][t] for key in PUBLIC_KEYS}
        branches = propose_correction_branches(inputs, _action(arrays["executed_actions"][t]),
            own_elixir=float(arrays["own_elixir"][t]), max_branches=config.max_branches)
        started = time.monotonic()
        env, hidden = states[t]
        result = evaluate_correction(env=env, hidden=hidden, policy=policy, candidate=candidate,
            branches=branches, opponent_spec=meta["opponent"], config=config)
        records.append(dict(candidate, **result, source_episode_sha256=source_row["sha256"],
                            split=meta["split"], wall_seconds=time.monotonic() - started))
    result = dict(plan_sha256=plan["plan_sha256"], dataset_manifest_sha256=file_sha(out / "dataset/manifest.json"),
                  source_episode_sha256=source_row["sha256"], candidates=list(selected), records=records)
    load_plan(out)
    atomic_json(out / "search" / f"{job_id}.json", result)
    return result


def search(out: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    from .tactical_data import load_episode

    out = Path(out).resolve()
    manifest = _dataset(out, plan)
    config = CorrectionSearchConfig(**plan["config"])
    pools: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for job in plan["jobs"]:
        row = manifest["completed"][job["job_id"]]
        arrays, meta = load_episode(out / "dataset" / row["path"],
                                   expected_identity=manifest["identity_sha256"], expected_sha=row["sha256"])
        candidates = select_episode_candidates(arrays, meta, config)
        if candidates:
            pools[job["split"]].append(candidates)
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    from .tactical_data import _stable_seed
    job_by_id = {job["job_id"]: job for job in plan["jobs"]}
    for split, budget in (("train", config.train_budget), ("calibration", config.calibration_budget)):
        remaining = [row for group in pools[split] for row in group]
        counts: Counter = Counter()
        families: Counter = Counter()
        categories: Counter = Counter()
        def primary(row: Mapping[str, Any]) -> str:
            return next((tag for tag in ("interception-window", "pre-starvation-spend", "pre-threat-spend", "air", "ranged-ground", "siege")
                         if tag in row["tags"]), "other")
        while remaining and sum(counts.values()) < budget:
            # Balance repair kinds, then family and origin. Seeded tie-breaks
            # avoid granting every extra sample to the first catalog decks.
            row = min(remaining, key=lambda r: (categories[primary(r)],
                families[job_by_id[r["job_id"]]["opponent"]["archetype"]], counts[r["job_id"]],
                _stable_seed(config.seed, split, r["job_id"], r["t"])))
            remaining.remove(row)
            selected[row["job_id"]].append(row)
            counts[row["job_id"]] += 1
            families[job_by_id[row["job_id"]]["opponent"]["archetype"]] += 1
            categories[primary(row)] += 1
    selection = dict(plan_sha256=plan["plan_sha256"], candidates=dict(selected),
                     dataset_manifest_sha256=file_sha(out / "dataset/manifest.json"))
    selection_path = out / "search_selection.json"
    if selection_path.exists() and canonical(json.loads(selection_path.read_text())) != canonical(selection):
        raise ValueError("search candidate selection changed on resume")
    atomic_json(selection_path, selection)
    records, pending = [], []
    for job_id, candidates in selected.items():
        path = out / "search" / f"{job_id}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved["plan_sha256"] != plan["plan_sha256"] or saved["candidates"] != candidates
                    or saved["dataset_manifest_sha256"] != selection["dataset_manifest_sha256"]
                    or saved["source_episode_sha256"] != manifest["completed"][job_id]["sha256"]):
                raise ValueError("cached correction search origin differs")
            if (len(saved["records"]) != len(candidates)
                    or {(row["job_id"], row["t"], row["state_hash"]) for row in saved["records"]}
                       != {(row["job_id"], row["t"], row["state_hash"]) for row in candidates}
                    or any(row.get("split") != job_by_id[job_id]["split"]
                           or row.get("source_episode_sha256") != saved["source_episode_sha256"] for row in saved["records"])):
                raise ValueError("cached correction search record coverage differs")
            records.extend(saved["records"])
        else:
            pending.append((job_id, candidates))

    def accept(result: Mapping[str, Any]) -> None:
        records.extend(result["records"])
        atomic_json(out / "progress.json", dict(stage="search", completed=len(records),
                    total=sum(map(len, selected.values())), accepted=sum(bool(row["accepted"]) for row in records)))
        print(f"correction search: {len(records)}/{sum(map(len, selected.values()))} states; "
              f"{sum(bool(row['accepted']) for row in records)} confirmed", flush=True)

    if plan["workers"] == 1:
        for job_id, candidates in pending:
            accept(_search_job(str(out), job_id, candidates))
    elif pending:
        with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_search_job, str(out), job_id, candidates) for job_id, candidates in pending]
            for future in as_completed(futures):
                accept(future.result())
    records.sort(key=lambda row: (row["job_id"], row["t"]))
    accepted = [row for row in records if row["accepted"]]
    summary = dict(evaluated=len(records), accepted=len(accepted),
                   accepted_train=sum(row["split"] == "train" for row in accepted),
                   accepted_calibration=sum(row["split"] == "calibration" for row in accepted),
                   reviewed_decks=dict(Counter(job_by_id[row["job_id"]]["opponent"]["deck_id"] for row in records)),
                   reviewed_tags=dict(Counter(tag for row in records for tag in row["tags"])),
                   accepted_tags=dict(Counter(tag for row in accepted for tag in row["tags"])))
    result = dict(schema=VERSION, checkpoint=plan["checkpoint"], plan_sha256=plan["plan_sha256"],
                  dataset_manifest_sha256=selection["dataset_manifest_sha256"], source_sha256=plan["source"]["sha256"],
                  min_lcb=config.min_lcb, config=asdict(config), records=records, summary=summary)
    load_plan(out)
    atomic_json(out / "corrections.json", result)
    return result


def _corrections(out: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads((out / "corrections.json").read_text())
    manifest = _dataset(out, plan)
    if (value.get("schema") != VERSION or value.get("plan_sha256") != plan["plan_sha256"]
            or canonical(value.get("config")) != canonical(plan["config"])
            or value.get("min_lcb") != plan["config"]["min_lcb"]
            or value.get("dataset_manifest_sha256") != file_sha(out / "dataset/manifest.json")
            or value.get("source_sha256") != plan["source"]["sha256"]
            or canonical(value.get("checkpoint")) != canonical(plan["checkpoint"])):
        raise ValueError("corrections differ from the sealed search plan")
    selection = json.loads((out / "search_selection.json").read_text())
    expected = {(row["job_id"], row["t"], row["state_hash"])
                for group in selection["candidates"].values() for row in group}
    records = value["records"]
    if (len(records) != len(expected)
            or {(row["job_id"], row["t"], row["state_hash"]) for row in records} != expected):
        raise ValueError("correction manifest does not cover the declared search states")
    for row in records:
        origin = manifest["completed"][row["job_id"]]
        if row.get("split") != origin["split"] or row.get("source_episode_sha256") != origin["sha256"]:
            raise ValueError("correction split/origin differs from its source episode")
    if value["summary"]["accepted_train"] != sum(row["accepted"] and row["split"] == "train" for row in records):
        raise ValueError("correction training summary differs from its records")
    return value


def fit(out: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    from .correction_training import CorrectionTrainingConfig, train_corrections

    corrections = _corrections(out, plan)
    if not corrections["summary"]["accepted_train"]:
        result = dict(status="no_validated_training_corrections", selected_actor_changed=False)
        atomic_json(out / "fit_skipped.json", result)
        return result
    return train_corrections(dataset=out / "dataset", out=out / "fit", checkpoint=plan["checkpoint"],
        corrections_manifest=out / "corrections.json", config=CorrectionTrainingConfig(**plan["training"]),
        resume=(out / "fit/protocol.json").exists())


def evaluate(out: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Fresh complete games across ALL training families, then new origins.

    Screen selection never consults reserved decks. Confirmation runs only for
    a nonregressing screen winner; this runner never promotes an actor.
    """
    from .policy_development import DevelopmentArm, run_development_screen, training_opponents

    fitted = fit(out, plan)  # Revalidate protocol, evidence and exported actor.
    if fitted.get("status") != "complete":
        result = dict(status="no_candidate", selected_actor_changed=False)
        atomic_json(out / "evaluation.json", result)
        return result
    candidate = fitted["checkpoint"]
    arms = [DevelopmentArm("control", "actor", **dict(checkpoint=plan["checkpoint"]["path"],
                sha256=plan["checkpoint"]["sha256"], model_config=plan["checkpoint"]["model_config"])),
            DevelopmentArm("corrected", "actor", checkpoint=candidate["path"], sha256=candidate["sha256"],
                           model_config=candidate["model_config"])]
    train = run_development_screen(out=out / "screen_train", arms=arms, opponents=training_opponents(),
        seeds=plan["screen_train_seeds"], workers=plan["workers"])
    dev = run_development_screen(out=out / "screen_dev", arms=arms,
        opponents=training_opponents(("adaptive", "split-pressure"), style_split="dev"),
        seeds=plan["screen_dev_seeds"], workers=plan["workers"])
    comparisons = [train["comparisons"]["corrected"], dev["comparisons"]["corrected"]]
    family_net: Counter = Counter()
    for comparison in comparisons:
        family_net.update(comparison["family_net"])
    net = sum(c["net_wins"] for c in comparisons)
    qualifies = net > 0 and all(value >= 0 for value in family_net.values()) and comparisons[1]["net_wins"] >= 0
    confirmation = None
    if qualifies:
        confirmation = run_development_screen(out=out / "confirmation", arms=arms,
            opponents=training_opponents(), seeds=plan["confirmation_seeds"], workers=plan["workers"])
    result = dict(status="complete", train=train, development=dev, screen_net_wins=net,
                  screen_family_net=dict(family_net), eligible_for_confirmation=qualifies,
                  confirmation=confirmation, selected_actor_changed=False, reserved_evaluation_run=False,
                  note="Local screen; confirmation is separate. No automatic promotion or reserved-deck selection.")
    atomic_json(out / "evaluation.json", result)
    return result


def report(out: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(schema=VERSION, plan_sha256=plan["plan_sha256"], status="partial", selected_actor_changed=False)
    for name in ("corrections", "fit/result", "evaluation"):
        path = out / f"{name}.json"
        if path.exists():
            value = json.loads(path.read_text())
            result[name] = value.get("summary", value) if name == "corrections" else value
    atomic_json(out / "result.json", result)
    lines = ["# Replay correction experiment", "", f"Origin actor: `{plan['checkpoint']['sha256']}`.", "",
             "Proposals use public inputs. Exact state and input prefixes are verified before branching.",
             "Discovery chooses an action; disjoint replicas confirm that same action at the longer horizon.", ""]
    if "corrections" in result:
        lines += [f"Search: **{result['corrections']['accepted']} / {result['corrections']['evaluated']}** "
                  f"confirmed corrections; {result['corrections']['accepted_train']} train and "
                  f"{result['corrections']['accepted_calibration']} calibration.", ""]
    evaluation = result.get("evaluation", {})
    if evaluation.get("status") in ("complete", "no_candidate") and "corrections" in result:
        result["status"] = "complete" if evaluation["status"] == "complete" else "no_validated_training_corrections"
    if evaluation.get("status") == "complete":
        lines += [f"Fresh screen net wins: **{evaluation['screen_net_wins']:+g}**.", "",
                  "| Family | Net wins |", "|---|---:|"]
        lines += [f"| {family} | {net:+g} |" for family, net in sorted(evaluation["screen_family_net"].items())]
        lines += ["", f"Independent confirmation eligible: **{evaluation['eligible_for_confirmation']}**.", ""]
    lines += ["Selected policy unchanged. No reserved-deck evaluation or promotion.", "",
              "Branch replicas share a match origin and do not establish broad strength. These targets",
              "certify changing one decision followed by the frozen actor; they do not credit an initial",
              "WAIT with the benefit of a scripted multi-action saving plan.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    atomic_json(out / "result.json", result)
    atomic_json(out / "progress.json", dict(stage=result["status"], result=str(out / "result.json")))
    return result
