"""Fail-closed training-readiness reporting for the declared simulator scope.

Fidelity reports are intentionally treated as evidence, not as configuration.
In particular, calibration observations can show that a mechanic has been
measured, but they can never satisfy a held-out requirement.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


READINESS_SCHEMA_VERSION = 1

# These are decision-relevant mechanics for the declared base Hog-cycle scope.
# Patterns name the measurements emitted by validation/mining rather than card
# constants, so adding a constant to the ruleset cannot accidentally mark a
# mechanic as validated against the real game.
DECLARED_MECHANICS: Mapping[str, tuple[str, ...]] = {
    "hog_isolated_movement": ("hog-rider_isolated_movement_*",),
    "hog_bridge_pathfinding": (
        "hog-rider_isolated_bridge_path_*",
        "hog-rider_bridge_path_topology",
    ),
    "musketeer_isolated_movement": ("musketeer_isolated_movement_*",),
    "musketeer_bridge_pathfinding": (
        "musketeer_isolated_bridge_path_*",
        "musketeer_bridge_path_topology",
    ),
    "skeletons_isolated_movement": ("skeletons_isolated_movement_*",),
    "skeletons_bridge_pathfinding": (
        "skeletons_isolated_bridge_path_*",
        "skeletons_bridge_path_topology",
    ),
    "ice_golem_isolated_movement": ("ice-golem_isolated_movement_*",),
    "ice_golem_bridge_pathfinding": (
        "ice-golem_isolated_bridge_path_*",
        "ice-golem_bridge_path_topology",
    ),
    "ice_spirit_isolated_movement": ("ice-spirit_isolated_movement_*",),
    "ice_spirit_bridge_pathfinding": ("ice-spirit_isolated_bridge_path_*",),
    "hog_cannon_targeting": (
        "hog_cannon_pull_targeting",
        "hog_cannon_targeting_candidate",
    ),
    "hog_cannon_pull_trajectory": ("hog_cannon_pull_trajectory*",),
    "cannon_lifetime_hp_decay": ("cannon_lifetime*", "cannon_hp_decay*"),
    "cannon_attack_timing": ("cannon_attack*", "cannon_projectile*"),
    "cannon_attack_damage": ("cannon_tower_damage", "cannon_attack_damage*"),
    "hog_attack_timing": (
        "hog-rider_attack*",
        "hog_attack*",
        "hog-rider_tower_repeat_interval",
    ),
    "hog_attack_damage": ("hog-rider_tower_damage", "hog-rider_attack_damage*"),
    "musketeer_attack_projectile_timing": (
        "musketeer_attack*",
        "musketeer_projectile*",
        "musketeer_tower_repeat_interval",
    ),
    "musketeer_attack_damage": ("musketeer_tower_damage", "musketeer_attack_damage*"),
    "skeletons_attack_timing": (
        "skeletons_attack*",
        "skeletons_tower_repeat_interval",
    ),
    "skeletons_attack_damage": ("skeletons_tower_damage", "skeletons_attack_damage*"),
    "ice_golem_attack_timing": (
        "ice-golem_attack*",
        "ice-golem_tower_repeat_interval",
    ),
    "ice_golem_attack_damage": ("ice-golem_tower_damage", "ice-golem_attack_damage*"),
    "tower_attack_activation_timing": ("tower_attack*", "tower_activation*"),
    "ice_spirit_connection_status": ("ice-spirit_connection*", "ice_spirit_connection*"),
    "ice_spirit_impact_damage": ("ice-spirit_tower_damage", "ice-spirit_impact_damage*"),
    "ice_golem_death_slow": ("ice-golem_death*", "ice_golem_death*"),
    "fireball_flight_timing": ("fireball_flight*", "fireball_action_to_impact"),
    "fireball_impact_localization": ("fireball_impact*",),
    "fireball_damage_geometry": ("fireball_damage*", "fireball_victim*"),
    "fireball_crown_damage": ("fireball_tower_damage", "fireball_crown_damage*"),
    "log_rolling_motion": ("log_motion*", "log_rolling_speed*"),
    "log_collision_victims": ("log_collision*", "log_victim*"),
    "log_damage": ("log_tower_damage", "log_damage*"),
    "unit_collision_congestion": ("unit_collision*", "bridge_congestion*"),
}


def declared_mechanics_for_ruleset(ruleset: object) -> dict[str, tuple[str, ...]]:
    """Expand the readiness matrix to every card in the loaded ruleset.

    The base decision-critical requirements remain mandatory.  Each card also
    gets one requirement per executable component; no aggregate percentage can
    hide a rare opponent card or a newly added status/spawn branch.
    """

    from .scenario_factory import card_mechanics

    result = dict(DECLARED_MECHANICS)
    for card_id in getattr(ruleset, "interaction_set"):
        for mechanic in card_mechanics(ruleset, card_id):
            result[f"card:{card_id}:{mechanic}"] = (f"{card_id}_{mechanic}*",)
    return result


class ReadinessError(ValueError):
    """Raised when evidence reports cannot be safely combined."""


def _load_report(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReadinessError(f"cannot load fidelity report {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ReadinessError(f"fidelity report {path} must contain a JSON object")
    required = ("dataset_split", "ruleset_id", "ruleset_hash", "engine_version", "mechanics")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ReadinessError(f"fidelity report {path} is missing: {', '.join(missing)}")
    if raw["dataset_split"] not in {"calibration", "validation", "regression", "heldout"}:
        raise ReadinessError(f"fidelity report {path} has invalid dataset_split")
    if not isinstance(raw["mechanics"], dict):
        raise ReadinessError(f"fidelity report {path} mechanics must be an object")
    raw["_path"] = str(path)
    return raw


def _load_candidate_report(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReadinessError(f"cannot load candidate report {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ReadinessError(f"candidate report {path} must contain a JSON object")
    allowed = {
        "cannon_lifetime_candidate_report",
        "tower_damage_candidate_report",
        "log_motion_candidate_report",
        "fireball_flight_candidate_report",
        "autonomous_interaction_candidate_report",
        "autonomous_interaction_candidate_batch",
        "autonomous_interaction_dual_hud_report",
    }
    if raw.get("kind") not in allowed:
        raise ReadinessError(f"candidate report {path} has unsupported kind {raw.get('kind')!r}")
    required = ("ruleset_id", "ruleset_hash", "engine_version")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ReadinessError(f"candidate report {path} is missing: {', '.join(missing)}")
    mechanics = raw.get("mechanics", {})
    if not isinstance(mechanics, dict):
        raise ReadinessError(f"candidate report {path} mechanics must be an object")
    raw["_path"] = str(path)
    return raw


def _metric_counts(metric: Mapping[str, Any], *, path: str, mechanic: str) -> tuple[int, int]:
    try:
        samples = metric["samples"]
        traces = metric["traces"]
        count = int(samples["count"]) + int(traces["count"])
        agreements = int(samples["agreement_count"]) + int(traces["agreement_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReadinessError(f"invalid metrics for {mechanic!r} in {path}") from error
    if count < 0 or agreements < 0 or agreements > count:
        raise ReadinessError(f"impossible metrics for {mechanic!r} in {path}")
    return count, agreements


def build_training_readiness_report(
    report_paths: Iterable[str | Path],
    *,
    candidate_report_paths: Iterable[str | Path] = (),
    ruleset_id: str,
    ruleset_hash: str,
    engine_version: str,
    minimum_heldout_observations: int = 20,
    minimum_heldout_agreement_rate: float = 0.98,
    minimum_heldout_groups: int = 2,
    requirements: Mapping[str, tuple[str, ...]] = DECLARED_MECHANICS,
) -> dict[str, Any]:
    """Combine pre-split fidelity reports without allowing evidence leakage."""

    if minimum_heldout_observations < 1:
        raise ReadinessError("minimum_heldout_observations must be positive")
    if not 0.0 <= minimum_heldout_agreement_rate <= 1.0:
        raise ReadinessError("minimum_heldout_agreement_rate must be between zero and one")
    if minimum_heldout_groups < 1:
        raise ReadinessError("minimum_heldout_groups must be positive")
    paths = [Path(path) for path in report_paths]
    candidate_paths = [Path(path) for path in candidate_report_paths]
    canonical_paths = [str(path.resolve()) for path in paths + candidate_paths]
    if len(set(canonical_paths)) != len(canonical_paths):
        raise ReadinessError("the same fidelity report was supplied more than once")
    reports = [_load_report(path) for path in paths]
    candidate_reports = [_load_candidate_report(path) for path in candidate_paths]
    for report in reports + candidate_reports:
        identity = (report["ruleset_id"], report["ruleset_hash"], report["engine_version"])
        expected = (ruleset_id, ruleset_hash, engine_version)
        if identity != expected:
            raise ReadinessError(
                f"fidelity report {report['_path']} identity {identity!r} does not match {expected!r}"
            )

    # A capture group used for calibration/validation is no longer untouched.
    # Reject the complete readiness result if another report calls it held-out.
    group_roles: dict[str, set[str]] = {}
    for report in reports:
        split = str(report["dataset_split"])
        for metric in report["mechanics"].values():
            evidence = metric.get("evidence", {})
            for group_id in evidence.get("group_ids", []):
                group_roles.setdefault(str(group_id), set()).add(split)
    leaked_groups = sorted(
        group for group, roles in group_roles.items() if "heldout" in roles and len(roles) > 1
    )
    heldout_media_hashes = {
        str(case["media_hash"])
        for report in reports
        if report["dataset_split"] == "heldout"
        for case in report.get("case_results", [])
        if isinstance(case, dict) and case.get("media_hash")
    }
    candidate_media_hashes: set[str] = set()
    for report in candidate_reports:
        if report.get("cache_hash"):
            candidate_media_hashes.add(str(report["cache_hash"]))
        for cache_hash in report.get("cache_hashes", []):
            if isinstance(cache_hash, str):
                candidate_media_hashes.add(cache_hash)
        for source in report.get("sources", []):
            if isinstance(source, Mapping) and source.get("cache_hash"):
                candidate_media_hashes.add(str(source["cache_hash"]))
    leaked_media_hashes = sorted(heldout_media_hashes & candidate_media_hashes)

    mechanics: dict[str, dict[str, Any]] = {}
    for requirement, patterns in sorted(requirements.items()):
        by_split: dict[str, dict[str, Any]] = {}
        for split in ("calibration", "validation", "regression", "heldout"):
            count = agreements = 0
            matched: set[str] = set()
            groups: set[str] = set()
            sources: set[str] = set()
            for report in reports:
                if report["dataset_split"] != split:
                    continue
                for mechanic_name, metric in report["mechanics"].items():
                    if not any(fnmatchcase(mechanic_name, pattern) for pattern in patterns):
                        continue
                    metric_count, metric_agreements = _metric_counts(
                        metric, path=report["_path"], mechanic=mechanic_name
                    )
                    count += metric_count
                    agreements += metric_agreements
                    matched.add(mechanic_name)
                    evidence = metric.get("evidence", {})
                    groups.update(str(item) for item in evidence.get("group_ids", []))
                    sources.update(str(item) for item in evidence.get("source_ids", []))
            by_split[split] = {
                "observation_count": count,
                "agreement_count": agreements,
                "agreement_rate": agreements / count if count else None,
                "matched_measurements": sorted(matched),
                "group_ids": sorted(groups),
                "source_ids": sorted(sources),
            }
        candidate_count = 0
        candidate_measurements: set[str] = set()
        candidate_sources: set[str] = set()
        candidate_kinds: set[str] = set()
        for report in candidate_reports:
            for mechanic_name, metric in report.get("mechanics", {}).items():
                if not any(fnmatchcase(mechanic_name, pattern) for pattern in patterns):
                    continue
                if not isinstance(metric, dict):
                    raise ReadinessError(
                        f"invalid candidate metric {mechanic_name!r} in {report['_path']}"
                    )
                count = metric.get("candidate_count", 0)
                if type(count) is not int or count < 0:
                    raise ReadinessError(
                        f"invalid candidate_count for {mechanic_name!r} in {report['_path']}"
                    )
                candidate_count += count
                candidate_measurements.add(mechanic_name)
                candidate_sources.add(str(report.get("cache_hash") or report["_path"]))
                candidate_kinds.add(str(report["kind"]))
        heldout = by_split["heldout"]
        if heldout["observation_count"] == 0:
            if by_split["calibration"]["observation_count"]:
                status = "calibrated_only"
            elif candidate_count:
                status = "candidate_only"
            elif candidate_measurements:
                status = "candidate_rejected"
            else:
                status = "missing"
        elif (
            heldout["observation_count"] < minimum_heldout_observations
            or len(heldout["group_ids"]) < minimum_heldout_groups
            or heldout["agreement_rate"] < minimum_heldout_agreement_rate
        ):
            status = "heldout_failed"
        else:
            status = "heldout_validated"
        mechanics[requirement] = {
            "status": status,
            "measurement_patterns": list(patterns),
            "evidence": by_split,
            "candidate_evidence": {
                "candidate_count": candidate_count,
                "matched_measurements": sorted(candidate_measurements),
                "source_ids": sorted(candidate_sources),
                "report_kinds": sorted(candidate_kinds),
                "can_satisfy_heldout_gate": False,
            },
        }

    failed = [name for name, row in mechanics.items() if row["status"] != "heldout_validated"]
    failures = []
    if leaked_groups:
        failures.append("held-out evidence groups also appear in another split: " + ", ".join(leaked_groups))
    if leaked_media_hashes:
        failures.append(
            "held-out media was previously used by a candidate discovery report: "
            + ", ".join(leaked_media_hashes)
        )
    if failed:
        failures.append("mechanics without passing held-out evidence: " + ", ".join(failed))
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "kind": "simulator_training_readiness",
        "ruleset_id": ruleset_id,
        "ruleset_hash": ruleset_hash,
        "engine_version": engine_version,
        "thresholds": {
            "minimum_heldout_observations_per_mechanic": minimum_heldout_observations,
            "minimum_heldout_agreement_rate": minimum_heldout_agreement_rate,
            "minimum_independent_heldout_groups_per_mechanic": minimum_heldout_groups,
        },
        "input_reports": [report["_path"] for report in reports],
        "input_candidate_reports": [report["_path"] for report in candidate_reports],
        "heldout_leakage_groups": leaked_groups,
        "heldout_leakage_media_hashes": leaked_media_hashes,
        "mechanics": mechanics,
        "summary": {
            "ready": not failures,
            "declared_mechanic_count": len(mechanics),
            "heldout_validated_count": len(mechanics) - len(failed),
            "failures": failures,
        },
    }


__all__ = [
    "DECLARED_MECHANICS",
    "READINESS_SCHEMA_VERSION",
    "ReadinessError",
    "build_training_readiness_report",
    "declared_mechanics_for_ruleset",
]
