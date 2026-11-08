"""Resumable, separately controlled tactical-learning experiments.

The reserved catalog remains inaccessible until frozen actors pass a fresh
development screen and independent confirmation. Final scores never select
an epoch or modify the production actor.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .policy_development import (
    DevelopmentArm, atomic_json, digest, file_sha, paired_summary,
    run_development_screen, runtime_source_fingerprint, training_opponents,
)
from .tactical_training import ARMS, TacticalTrainingConfig


EXPERIMENT_VERSION = "v4-tactical-ablation-1"
ROOT = Path(__file__).resolve().parents[3]


def prepare_experiment(*, out: Path, actor_json: Path, training: TacticalTrainingConfig,
                       arms: Sequence[str] = ARMS, workers: int = 3,
                       fit_workers: int = 3, deck_ids: Sequence[str] = (),
                       smoke: bool = False) -> dict[str, Any]:
    from .bootstrap import catalog_training_opponent_specs
    from .tactical_data import build_collection_jobs
    from .real_decks import catalog_sha256
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("unique tactical arms are required")
    if "control" not in arms:
        raise ValueError("all ablations require a matched control fit")
    if not 1 <= workers <= 16 or not 1 <= fit_workers <= 4:
        raise ValueError("worker bounds are 1..16 collection/evaluation and 1..4 fits")
    if deck_ids and not smoke:
        raise ValueError("a deck subset is only allowed for a diagnostic smoke")
    receipt = json.loads(Path(actor_json).read_text())
    checkpoint = dict(receipt.get("checkpoint", receipt))
    checkpoint["path"] = str(Path(checkpoint["path"]).resolve())
    if file_sha(checkpoint["path"]) != checkpoint["sha256"] or not checkpoint.get("model_config"):
        raise ValueError("a hashed actor and complete model configuration are required")
    specs = list(catalog_training_opponent_specs())
    if deck_ids:
        requested = set(deck_ids)
        if len(requested) != len(deck_ids) or not requested <= {spec.deck_id for spec in specs}:
            raise ValueError("smoke decks must be unique training identities")
        specs = [spec for spec in specs if spec.deck_id in requested]
    if smoke and not deck_ids:
        specs = [spec for spec in specs if spec.deck_id == "lavaloon-lightning"]
    jobs = build_collection_jobs(specs, 2026101900)
    source = runtime_source_fingerprint()
    plan = dict(version=EXPERIMENT_VERSION, checkpoint=checkpoint, training=asdict(training), arms=list(arms),
        workers=workers, fit_workers=fit_workers, smoke=smoke, train_decks=[spec.deck_id for spec in specs],
        catalog_sha256=catalog_sha256(), jobs=[asdict(job) for job in jobs], source=source,
        teacher_seeds=[2026101911, 2026101912], screen_train_seeds=[2026101921, 2026101922],
        screen_dev_seeds=[2026101923], confirmation_seeds=[2026101931, 2026101932, 2026101933, 2026101934, 2026101935],
        gate=dict(minimum_net_wins=3, minimum_lavaloon_net_wins=1,
                  reject_any_family_delta_at_or_below=-3, minimum_bait_net_wins=0,
                  minimum_xbow_net_wins=0, minimum_dev_net_wins=0),
        teacher_gate=dict(minimum_net_wins=1, minimum_lavaloon_net_wins=0,
                          reject_any_family_delta_at_or_below=-3,
                          minimum_bait_net_wins=0, minimum_xbow_net_wins=0),
        fixed_wait=True, decision_interval_us=250000, reserved_policy="only independently qualified frozen actors; scores are final",
        outcome_scope="factual public tower-damage at2/6/12s on active rows; survival masked without stable public IDs",
        curriculum_scope="train-only scenario weights from prior training epochs; all episodes and inactive KL rows retained",
        shared_data="90 train and30 disjoint calibration episodes at full scale; same corpus and frame/update dose for all fitted arms")
    plan["plan_sha256"] = digest(plan)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        if json.loads((out / "plan.json").read_text()) != plan:
            raise ValueError("experiment plan/source differs; use a new output directory")
    elif any(out.iterdir()):
        raise ValueError("new experiment output must be empty")
    atomic_json(out / "plan.json", plan)
    atomic_json(out / "source_seal.json", source)
    return plan


def load_plan(out: Path) -> dict[str, Any]:
    plan = json.loads((out / "plan.json").read_text())
    if digest({key: value for key, value in plan.items() if key != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("experiment plan seal differs")
    if runtime_source_fingerprint() != plan["source"]:
        raise ValueError("runtime source changed since preparation; use a new experiment")
    if file_sha(plan["checkpoint"]["path"]) != plan["checkpoint"]["sha256"]:
        raise ValueError("selected input actor changed")
    return plan


def _opponents(plan: Mapping[str, Any], styles: Sequence[str] = (), *, split: str = "train") -> list[dict[str, Any]]:
    return [spec for spec in training_opponents(styles, style_split=split) if spec["deck_id"] in plan["train_decks"]]


def qualification(comparison: Mapping[str, Any], gate: Mapping[str, Any],
                  *, development: Mapping[str, Any] | None = None) -> dict[str, Any]:
    families = comparison["family_net"]
    failures = []
    if comparison["net_wins"] < gate["minimum_net_wins"]:
        failures.append("overall_gain")
    if families.get("lavaloon", 0) < gate["minimum_lavaloon_net_wins"]:
        failures.append("lavaloon_gain")
    for family in ("bait", "xbow"):
        if families.get(family, 0) < gate[f"minimum_{family}_net_wins"]:
            failures.append(f"{family}_retention")
    for family, delta in families.items():
        if delta <= gate["reject_any_family_delta_at_or_below"]:
            failures.append(f"family_regression:{family}")
    if development is not None and development["net_wins"] < gate.get("minimum_dev_net_wins", 0):
        failures.append("held_out_style_regression")
    return dict(eligible=not failures, failures=failures, gate=dict(gate))


def _combine(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from collections import Counter
    families: Counter = Counter()
    result = dict(pairs=0, net_wins=0.0, favorable=0, unfavorable=0)
    for entry in comparisons:
        for key in result:
            result[key] += entry[key]
        families.update(entry["family_net"])
    return dict(result, family_net=dict(families), mean_score_delta=result["net_wins"] / max(1, result["pairs"]))


def teacher_screen(out: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    report = run_development_screen(out=out / "teacher_screen", arms=[DevelopmentArm("cycle", "cycle"), DevelopmentArm("plans", "plans")],
              opponents=_opponents(plan), seeds=plan["teacher_seeds"], workers=plan["workers"])
    gate = qualification(report["comparisons"]["plans"], plan["teacher_gate"])
    result = dict(plan_sha256=plan["plan_sha256"], report=report, qualification=gate,
                  note="A teacher gate does not establish student improvement; smoke gates cannot qualify a final actor.")
    atomic_json(out / "teacher_qualification.json", result)
    return result


def validate_collection(out: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a complete self-consistent corpus to THIS experiment's job grid."""
    from .tactical_data import CollectionConfig, canonical, validate_manifest
    manifest = json.loads((out / "dataset/manifest.json").read_text())
    validate_manifest(manifest, checkpoint_record=plan["checkpoint"])
    identity = manifest["identity"]
    caller = identity["caller"]
    if (caller.get("plan_sha256") != plan["plan_sha256"]
            or caller.get("source_sha256") != plan["source"]["sha256"]
            or canonical(manifest["jobs"]) != canonical(plan["jobs"])
            or canonical(identity["config"]) != canonical(asdict(CollectionConfig()))):
        raise ValueError("corpus belongs to a different experiment plan/grid/configuration")
    for record in manifest["completed"].values():
        if file_sha(out / "dataset" / record["path"]) != record["sha256"]:
            raise ValueError("collected episode bytes changed")
    return manifest


def expected_dose(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[int, int]:
    import math
    import numpy as np
    # Match the declared job order, independent of worker completion timing.
    rows = [manifest["completed"][job["job_id"]] for job in manifest["jobs"] if job["split"] == "train"]
    frames = sum(int(row["decisions"]) for row in rows) * config["epochs"]
    updates = 0
    for epoch in range(config["epochs"]):
        order = np.random.default_rng(config["seed"] + epoch).permutation(len(rows))
        for offset in range(0, len(rows), config["batch_episodes"]):
            longest = max(int(rows[int(index)]["decisions"]) for index in order[offset:offset + config["batch_episodes"]])
            updates += math.ceil(longest / config["chunk_length"])
    return frames, updates


def validate_fits(out: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Reject stale/misassociated training reports before any policy screen."""
    from .policy_development import canonical
    from .tactical_training import TRAINING_VERSION
    manifest = validate_collection(out, plan)
    summary = json.loads((out / "fits.json").read_text())
    if summary.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("fit summary belongs to another experiment plan")
    teacher = teacher_screen(out, plan)  # Cached raw records/seals are revalidated.
    expected_arms = set(plan["arms"])
    skipped = "plans" in expected_arms and not plan["smoke"] and not teacher["qualification"]["eligible"]
    if skipped:
        expected_arms.remove("plans")
    if (set(summary["results"]) != expected_arms
            or set(summary["skipped"]) != ({"plans"} if skipped else set())):
        raise ValueError("fit summary does not match the independently gated arm grid")
    frames, updates = expected_dose(manifest, plan["training"])
    for arm, result in summary["results"].items():
        folder = out / f"fit_{arm}"
        protocol_path, result_path = folder / "protocol.json", folder / "result.json"
        if (summary["artifact_seals"][arm]["protocol_sha256"] != file_sha(protocol_path)
                or summary["artifact_seals"][arm]["result_sha256"] != file_sha(result_path)
                or canonical(result) != canonical(json.loads(result_path.read_text()))):
            raise ValueError("fit protocol/result seal differs")
        saved = json.loads(protocol_path.read_text())
        protocol = saved["protocol"]
        expected = dict(version=TRAINING_VERSION, arm=arm, config=plan["training"], checkpoint=plan["checkpoint"],
            dataset=str((out / "dataset").resolve()), dataset_manifest_sha256=file_sha(out / "dataset/manifest.json"),
            source=plan["source"], inference="native-stochastic-fixed250ms-carry", final_epoch=plan["training"]["epochs"],
            curriculum="train-only bounded public-scenario supervision weights; every episode retained")
        actor_path = (folder / f"actor_epoch_{plan['training']['epochs']:03d}.pt").resolve()
        if (canonical(protocol) != canonical(expected) or digest(protocol) != saved["run_hash"]
                or result["run_hash"] != saved["run_hash"] or result["status"] != "complete"
                or result["arm"] != arm or result["final_epoch"] != plan["training"]["epochs"]
                or Path(result["checkpoint"]["path"]).resolve() != actor_path
                or result["checkpoint"]["sha256"] != file_sha(actor_path)
                or result["checkpoint"].get("model_config") != dict(plan["checkpoint"]["model_config"], fixed_wait=True)
                or result["total_training_frame_visits"] != frames
                or result["total_optimizer_steps"] != updates):
            raise ValueError("candidate protocol/actor/dose differs from its predeclared experiment")
        head = folder / "outcome_head.pt"
        if arm == "outcomes" and summary["artifact_seals"][arm].get("outcome_head_sha256") != file_sha(head):
            raise ValueError("consequence head export seal differs")
    return summary


def collect(out: Path, plan: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    from .tactical_data import collect_dataset
    identity = dict(plan_sha256=plan["plan_sha256"], source_sha256=plan["source"]["sha256"],
                    source_files={str(ROOT / key): value for key, value in plan["source"]["files"].items()},
                    model_config=plan["checkpoint"]["model_config"])
    return collect_dataset(plan["jobs"], plan["checkpoint"], out / "dataset", identity,
                           workers=plan["workers"], resume=resume)


def _fit_job(out: str, arm: str, checkpoint: Mapping[str, Any], config: Mapping[str, Any], resume: bool) -> dict[str, Any]:
    import os
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    from .tactical_training import train_tactical_arm
    path = Path(out)
    return train_tactical_arm(dataset=path / "dataset", out=path / f"fit_{arm}", arm=arm,
                              checkpoint=checkpoint, config=TacticalTrainingConfig(**config), resume=resume)


def fit(out: Path, plan: Mapping[str, Any], *, resume: bool, arm: str | None = None) -> dict[str, Any]:
    arms = list(plan["arms"]) if arm is None else [arm]
    if any(name not in plan["arms"] for name in arms):
        raise ValueError("fit arm is outside the predeclared experiment")
    validate_collection(out, plan)
    teacher = teacher_screen(out, plan)
    skipped = {}
    if "plans" in arms and not teacher["qualification"]["eligible"] and not plan["smoke"]:
        arms.remove("plans")
        skipped["plans"] = "Coordinated teacher failed its independent complete-match gate. No plan student fitted."
    results = {}
    if len(arms) == 1:
        name = arms[0]
        results[name] = _fit_job(str(out), name, plan["checkpoint"], plan["training"], resume)
    elif arms:
        import multiprocessing
        with ProcessPoolExecutor(max_workers=min(plan["fit_workers"], len(arms)), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_fit_job, str(out), name, plan["checkpoint"], plan["training"], resume): name for name in arms}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    if arm is None:
        doses = {(result["total_training_frame_visits"], result["total_optimizer_steps"]) for result in results.values()}
        if len(doses) > 1:
            raise RuntimeError("ablation arms received different frame/update budgets")
        seals = {name: dict(protocol_sha256=file_sha(out / f"fit_{name}/protocol.json"),
                           result_sha256=file_sha(out / f"fit_{name}/result.json")) for name in results}
        if "outcomes" in results:
            seals["outcomes"]["outcome_head_sha256"] = file_sha(out / "fit_outcomes/outcome_head.pt")
        atomic_json(out / "fits.json", dict(plan_sha256=plan["plan_sha256"], results=results,
                    skipped=skipped, artifact_seals=seals, matched_dose=True))
    return dict(results=results, skipped=skipped)


def _actor_arm(name: str, checkpoint: Mapping[str, Any]) -> DevelopmentArm:
    return DevelopmentArm(name, "actor", checkpoint["path"], checkpoint["sha256"], checkpoint["model_config"])


def _rows(out: Path, arm: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((out / arm).glob("*.json"))]


def _control_comparisons(out: Path, names: Sequence[str]) -> dict[str, Any]:
    if "control" not in names:
        return {}
    control = _rows(out, "control")
    return {name: paired_summary(control, _rows(out, name)) for name in names if name != "control"}


def evaluate(out: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    fits = validate_fits(out, plan)
    candidates = fits["results"]
    if not candidates:
        raise ValueError("no trained candidates to evaluate")
    arms = [_actor_arm("baseline", plan["checkpoint"])] + [_actor_arm(name, candidates[name]["checkpoint"]) for name in plan["arms"] if name in candidates]
    train = run_development_screen(out=out / "screen_train", arms=arms, opponents=_opponents(plan),
                seeds=plan["screen_train_seeds"], workers=plan["workers"])
    development = run_development_screen(out=out / "screen_dev", arms=arms,
                opponents=_opponents(plan, ("adaptive", "split-pressure"), split="dev"),
                seeds=plan["screen_dev_seeds"], workers=plan["workers"])
    combined = {name: _combine((train["comparisons"][name], development["comparisons"][name])) for name in candidates}
    gates = {name: qualification(combined[name], plan["gate"], development=development["comparisons"][name]) for name in candidates}
    result = dict(status="smoke_complete" if plan["smoke"] else "screen_complete", plan_sha256=plan["plan_sha256"],
        train=train, development=development, combined=combined, qualification=gates,
        relative_to_control=dict(train=_control_comparisons(out / "screen_train", list(candidates)),
                                 development=_control_comparisons(out / "screen_dev", list(candidates))),
        fits_skipped=fits["skipped"], selected_actor_changed=False,
        source_sha256=plan["source"]["sha256"])
    atomic_json(out / "screen_results.json", result)
    eligible = [name for name in candidates if gates[name]["eligible"]]
    if not plan["smoke"] and eligible:
        confirmation_arms = [arms[0]] + [_actor_arm(name, candidates[name]["checkpoint"]) for name in eligible]
        confirmation = run_development_screen(out=out / "confirmation", arms=confirmation_arms,
                opponents=_opponents(plan), seeds=plan["confirmation_seeds"], workers=plan["workers"])
        result["confirmation"] = confirmation
        result["confirmation_qualification"] = {name: qualification(confirmation["comparisons"][name], plan["gate"]) for name in eligible}
        eligible = [name for name in eligible if result["confirmation_qualification"][name]["eligible"]]
        # The same five seeds now complete the 30+10 catalog for independently
        # qualified frozen actors. Reserved scores never alter fitting/gates.
        if eligible:
            from .catalog_evaluation import catalog_opponents
            final_arms = [arms[0]] + [_actor_arm(name, candidates[name]["checkpoint"]) for name in eligible]
            result["reserved"] = run_development_screen(out=out / "reserved", arms=final_arms,
                opponents=catalog_opponents()["eval"], seeds=plan["confirmation_seeds"], workers=plan["workers"], catalog_split="eval")
            result["catalog_40"] = {name: _combine((confirmation["comparisons"][name], result["reserved"]["comparisons"][name])) for name in eligible}
    result["final_catalog_actors"] = eligible if not plan["smoke"] else []
    result["reserved_evaluation_run"] = "reserved" in result
    result["status"] = "smoke_complete" if plan["smoke"] else "complete"
    atomic_json(out / "results.json", result)
    write_results_markdown(out, plan, result)
    atomic_json(out / "completion_receipt.json", dict(status=result["status"], plan_sha256=plan["plan_sha256"],
                results_sha256=file_sha(out / "results.json"), report=str(out / "RESULTS.md"),
                reserved_evaluation_run=result["reserved_evaluation_run"], selected_actor_changed=False))
    return result


def write_results_markdown(out: Path, plan: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    lines = ["# Tactical-learning ablation results", "", f"Status: **{result['status']}**. Selected actor unchanged.", "",
             "All arms use the same public episodes, fixed training dose, native 250 ms actions and carried recurrent history.",
             "Training data and calibration exclude reserved decks and development styles. Calibration never selects an epoch.", "",
             "| Actor | Fresh fixed-controller wins | Held-out-style wins | Net wins versus baseline | Gate |",
             "|---|---:|---:|---:|---|"]
    names = ["baseline", *result["combined"]]
    for name in names:
        train = result["train"]["arms"][name]
        dev = result["development"]["arms"][name]
        delta = "—" if name == "baseline" else str(result["combined"][name]["net_wins"])
        gate = "reference" if name == "baseline" else ("pass" if result["qualification"][name]["eligible"] else ", ".join(result["qualification"][name]["failures"]))
        lines.append(f"| {name} | {train['wins']}/{train['games']} | {dev['wins']}/{dev['games']} | {delta} | {gate} |")
    lines += ["", "Per-family results and comparisons against the trained control are in `results.json`.", ""]
    for name, reason in result["fits_skipped"].items():
        lines.append(f"- {name}: {reason}")
    lines += ["", f"Reserved-deck evaluation run: **{result['reserved_evaluation_run']}**.", "",
              "Consequence supervision in this pilot covers factual tower damage at 2/6/12 seconds. Survival heads are implemented but masked because this public tensor contract carries no stable track identities.",
              "Adaptive practice changes bounded train-only tactical supervision weights, while retaining every episode and every inactive retention row. It does not reset arbitrary mid-match snapshots.",
              "Teacher wins, auxiliary loss and calibration do not establish human-level play. These are bounded simulator experiments against declared scripted opponents.", "",
              f"Source SHA256: `{plan['source']['sha256']}`.", f"Plan SHA256: `{plan['plan_sha256']}`.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def status(out: Path) -> dict[str, Any]:
    result: dict[str, Any] = dict(out=str(out), complete=(out / "completion_receipt.json").exists())
    if (out / "dataset/manifest.json").exists():
        collection = json.loads((out / "dataset/manifest.json").read_text())
        result["collection"] = dict(status=collection["status"], completed=len(collection["completed"]), declared=len(collection["jobs"]))
    result["fits"] = {name: json.loads(path.read_text()) for name in ARMS if (path := out / f"fit_{name}/progress.json").exists()}
    if (out / "completion_receipt.json").exists():
        result["receipt"] = json.loads((out / "completion_receipt.json").read_text())
    return result
