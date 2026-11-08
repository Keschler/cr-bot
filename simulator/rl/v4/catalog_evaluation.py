"""Frozen five-policy comparison on the complete, split-aware deck catalog."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .learning_curve import (
    _coerce_model_config, _default_final_opponents, _default_opponents,
    _runtime_provenance, _validate_match_records, dirty_source_fingerprint, paired_match_statistics,
    sha256_file,
)
from .real_decks import catalog_sha256, load_real_decks
from simulator.roster import PLAYER_DECK


POLICY_NAMES = (
    "Random initialization",
    "Random initialization + PPO",
    "Rules-teacher bootstrap",
    "Bootstrap + PPO",
    "Bootstrap + search distillation",
)
COMPARISON_SCHEMA = "v4-catalog-comparison-1"


def catalog_opponents() -> dict[str, tuple[dict[str, Any], ...]]:
    pools = {"train": _default_opponents(), "eval": _default_final_opponents()}
    for split, expected in (("train", 30), ("eval", 10)):
        pools[split] = tuple(dict(spec, real_deck_split=split) for spec in pools[split])
        if len(pools[split]) != expected:
            raise ValueError(f"comparison requires {expected} {split} decks")
    return pools


def _load_actor(spec: Mapping[str, Any], model_config: Any, device: str) -> Any:
    import torch
    from .v4_ppo.runtime import configure_v4_runtime, load_frozen_anchor

    # Constructing a model before loading weights also consumes random values.
    with torch.random.fork_rng(devices=[]):
        actor = load_frozen_anchor(
            str(Path(spec["checkpoint"]).expanduser().resolve()), model_config,
            expected_sha256=spec["sha256"], device=device,
        )
    configure_v4_runtime(actor)
    return actor


def _parameter_digest(actor: Any) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(actor.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def preflight_comparison(manifest: Mapping[str, Any], *, device: str = "cpu") -> dict[str, Any]:
    """Verify the exact experiment and load every pinned actor without playing."""
    import torch
    from simulator.engine import ENGINE_VERSION
    from simulator.observation_v3 import OBSERVATION_V3_CONTRACT_HASH
    from simulator.ruleset import load_fixed_ruleset

    if manifest.get("schema") != COMPARISON_SCHEMA:
        raise ValueError(f"manifest schema must be {COMPARISON_SCHEMA}")
    seeds = manifest.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) != 10
            or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 10):
        raise ValueError("comparison requires ten unique integer match seeds")
    if manifest.get("max_decisions") != 1440:
        raise ValueError("comparison requires a 1,440-decision cap")
    if manifest.get("deterministic_policy") is not False:
        raise ValueError("comparison requires seeded stochastic policy actions")
    if not isinstance(manifest.get("model_config"), Mapping):
        raise ValueError("comparison requires the explicit model configuration")
    model = _coerce_model_config(manifest["model_config"])
    policies = manifest.get("policies")
    if (not isinstance(policies, list) or len(policies) != len(POLICY_NAMES)
            or any(not isinstance(spec, Mapping) for spec in policies)
            or {spec.get("name") for spec in policies} != set(POLICY_NAMES)):
        raise ValueError("comparison requires each of the five named policies exactly once")
    decks = load_real_decks()
    if manifest.get("catalog_sha256") != catalog_sha256():
        raise ValueError("real-deck catalog hash mismatch")
    pools = catalog_opponents()
    actors = []
    parameters = set()
    paths = set()
    for spec in policies:
        path = Path(spec["checkpoint"]).expanduser().resolve()
        if path in paths:
            raise ValueError("comparison lists a checkpoint path more than once")
        paths.add(path)
        expected = spec.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
            raise ValueError(f"checkpoint hash mismatch for {spec['name']}")
        actor = _load_actor(spec, model, device)
        if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
            raise ValueError(f"non-finite actor parameters for {spec['name']}")
        digest = _parameter_digest(actor)
        if digest in parameters:
            raise ValueError("two named policies contain identical actor weights")
        parameters.add(digest)
        actors.append({"name": spec["name"], "checkpoint": str(path),
                       "sha256": expected, "parameter_sha256": digest})
    return {
        "schema": COMPARISON_SCHEMA, "status": "preflight_passed", "games_played": 0,
        "actors": actors, "model_config": asdict(model),
        "runtime_fixed_wait": True, "deterministic_policy": False,
        "device": device, "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
        "learner_deck": list(PLAYER_DECK), "learner_player": 0,
        "decision_interval_us": 250_000, "engine_version": ENGINE_VERSION,
        "ruleset_hash": load_fixed_ruleset().content_hash,
        "observation_contract_hash": OBSERVATION_V3_CONTRACT_HASH,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "seeds": list(seeds), "max_decisions": 1440, "workers": 1,
        "catalog_sha256": catalog_sha256(),
        "deck_ids": {split: [d.deck_id for d in decks if d.split == split] for split in pools},
        "opponents": {split: list(specs) for split, specs in pools.items()},
        "games_per_policy": {split: len(specs) * len(seeds) for split, specs in pools.items()},
        "expected_total_games": len(policies) * len(decks) * len(seeds),
    }


def _outcome_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {result: sum(row["result"] == result for row in records)
              for result in ("win", "loss", "draw", "truncated")}
    completed = len(records) - counts["truncated"]
    return {
        "games": len(records), **counts, "completed": completed,
        "score_mean_completed": ((counts["win"] + 0.5 * counts["draw"]) / completed
                                 if completed else None),
        "invalid_actions": [sum(row["invalid_actions"][player] for row in records) for player in (0, 1)],
    }


def _validate_conditions(records: Sequence[Mapping[str, Any]], opponents: Sequence[Mapping[str, Any]]) -> None:
    expected = {spec["deck_id"]: spec for spec in opponents}
    for row in records:
        spec = expected[row["deck_id"]]
        identity = row.get("opponent", {})
        if (
            row.get("controller_seed") != spec["controller_seed"]
            or row.get("strategy") != spec["strategy"]
            or identity.get("deck") != list(spec["deck"])
            or identity.get("controller_seed") != spec["controller_seed"]
            or row.get("learner_deck") != list(PLAYER_DECK)
            or row.get("learner_player") != 0
            or row.get("deterministic_policy") is not False
            or row.get("max_decisions") != 1440
        ):
            raise ValueError("match record does not describe its declared comparison conditions")


def run_catalog_comparison(
    manifest: Mapping[str, Any], *, output_dir: Path, device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate every actor once per cell; keep train/eval outcomes separate."""
    from .frontier_ppo.evaluation import full_match_record

    preflight = preflight_comparison(manifest, device=device)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("comparison requires a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _coerce_model_config(manifest["model_config"])
    pools = catalog_opponents()
    report = dict(preflight, status="running", provenance=_runtime_provenance("catalog-comparison-1"),
                  actors=[], pairwise={"train": [], "eval": []})
    report["provenance"]["comparison_runner_sha256"] = sha256_file(
        Path(__file__).resolve().parents[3] / "scripts/run_v4_catalog_evaluation.py"
    )
    root = Path(__file__).resolve().parents[3]
    # Include physics and input adapters, which the narrower PPO runtime
    # fingerprint does not cover. Never fingerprint generated builds/media.
    source_paths = (
        *sorted((root / "simulator").glob("*.py")),
        root / "simulator/engine", root / "simulator/rl",
        *sorted((root / "simulator/rulesets").glob("*.json")),
        root / "src/cr_bot/domain", root / "scripts/run_v4_catalog_evaluation.py",
    )
    report["provenance"]["evaluation_source"] = dirty_source_fingerprint(source_paths)
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    all_records = []
    for index, spec in enumerate(manifest["policies"]):
        policy = _load_actor(spec, model, device)
        split_records = {}
        for split, opponents in pools.items():
            rows = full_match_record(
                policy=policy, opponent_specs=opponents, seeds=manifest["seeds"],
                device=device, deterministic=False, max_decisions=1440,
            )
            raw_path = output_dir / f"actor_{index}_{split}.jsonl"
            raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            _validate_match_records(rows, specs=opponents, seeds=manifest["seeds"],
                                    label=spec["name"], final=split == "eval")
            _validate_conditions(rows, opponents)
            split_records[split] = rows
        all_records.append(split_records)
        report["actors"].append({
            **preflight["actors"][index],
            "outcomes": {split: _outcome_summary(rows) for split, rows in split_records.items()},
            "per_deck": {
                split: {spec["deck_id"]: _outcome_summary([row for row in rows if row["deck_id"] == spec["deck_id"]])
                        for spec in pools[split]}
                for split, rows in split_records.items()
            },
            "raw_records": {split: f"actor_{index}_{split}.jsonl" for split in pools},
        })
        (output_dir / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    for split in pools:
        for baseline in range(len(all_records)):
            for candidate in range(baseline + 1, len(all_records)):
                left, right = all_records[baseline][split], all_records[candidate][split]
                # Match by explicit identity even if a backend returns its
                # cells in a different order.
                key = lambda row: (row["deck_id"], row["seed"])
                left, right = sorted(left, key=key), sorted(right, key=key)
                # Keep fully censored outcomes reviewable instead of losing
                # raw records to the statistics helper's zero-pair exception.
                if not any(a["terminated"] and b["terminated"] for a, b in zip(left, right)):
                    stats = {"n_paired": 0, "n_censored": len(left), "paired_inference_valid": False,
                             "paired_inference_reason": "all matched outcomes are censored"}
                else:
                    stats = paired_match_statistics(left, right, max_decisions=1440)
                report["pairwise"][split].append({"baseline_actor": baseline, "candidate_actor": candidate,
                                                   "statistics": stats})
    report["status"] = "complete"
    report["games_played"] = sum(len(rows) for records in all_records for rows in records.values())
    report["invalid_action_count"] = sum(sum(row["invalid_actions"]) for records in all_records for rows in records.values() for row in rows)
    final_source = dirty_source_fingerprint(source_paths)
    report["source_stable"] = final_source["sha256"] == report["provenance"]["evaluation_source"]["sha256"]
    report["provenance"]["evaluation_source_end_sha256"] = final_source["sha256"]
    report["technical_validity"] = report["invalid_action_count"] == 0 and report["source_stable"]
    report["winner_claims_permitted"] = {
        split: report["technical_validity"] and all(
            pair["statistics"]["paired_inference_valid"] for pair in report["pairwise"][split]
        ) for split in pools
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
