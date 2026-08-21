"""Differential comparison between normalized observations and simulator replay."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import quantiles
from typing import Any, Iterable, Mapping, Sequence

from .observation import EntityObservation, NormalizedEvent, ObservationManifest
from .replay import SimulatorReplay
from .schema import PhysicalLabError, canonical_hash


@dataclass(frozen=True, slots=True)
class NumericMetric:
    count: int
    mean: float | None
    p95: float | None
    maximum: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "mean": self.mean,
            "p95": self.p95,
            "maximum": self.maximum,
        }


def _metric(values: Iterable[float]) -> NumericMetric:
    values = tuple(float(value) for value in values)
    if not values:
        return NumericMetric(0, None, None, None)
    ordered = sorted(values)
    if len(ordered) == 1:
        p95 = ordered[0]
    else:
        rank = 0.95 * (len(ordered) - 1)
        low = math.floor(rank)
        high = math.ceil(rank)
        fraction = rank - low
        p95 = ordered[low] + (ordered[high] - ordered[low]) * fraction
    return NumericMetric(len(values), sum(values) / len(values), p95, max(values))


def _series(entity: EntityObservation) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            sample.match_time_us if sample.match_time_us is not None else sample.video_time_us,
            sample.x_mtile,
            sample.y_mtile,
        )
        for sample in entity.samples
    )


def _nearest_samples(
    real: Sequence[tuple[int, int, int]],
    simulated: Sequence[tuple[int, int, int]],
) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]:
    if not real or not simulated:
        return ()
    result = []
    for item in real:
        nearest = min(simulated, key=lambda candidate: (abs(candidate[0] - item[0]), candidate[0]))
        result.append((item, nearest))
    return tuple(result)


def _path_length(series: Sequence[tuple[int, int, int]]) -> float:
    return sum(
        math.hypot(float(current[1] - previous[1]), float(current[2] - previous[2]))
        for previous, current in zip(series, series[1:])
    )


def _simulated_entity_series(replay: SimulatorReplay, selector: tuple[str, str, str | None, str | None]) -> tuple[tuple[int, int, int], ...]:
    card_id, owner_label, role, _source_card_id = selector
    owner = 0 if owner_label == "A" else 1
    result: list[tuple[int, int, int]] = []
    for tick, state in sorted(replay.snapshots.items()):
        candidates = [
            entity
            for entity in state.entities.values()
            if entity.card_id == card_id
            and entity.owner == owner
            and (role is None or entity.role == role)
        ]
        if candidates:
            entity = sorted(candidates, key=lambda item: item.uid)[0]
            result.append((tick * _scenario_tick_us(replay), entity.x_mtile, entity.y_mtile))
    return tuple(result)


# Kept as a tiny function to make the replay object backwards-compatible with
# callers that construct a light fake in tests.
def _scenario_tick_us(replay: SimulatorReplay) -> int:
    try:
        return int(replay.scenario_tick_us)  # type: ignore[attr-defined]
    except AttributeError:
        from ..ruleset import load_ruleset

        return load_ruleset(replay.scenario.ruleset_id).tick_us


def _stable_sim_event(event: Any) -> dict[str, Any]:
    data = dict(event.data)
    return {
        "kind": event.kind,
        "card_id": data.get("card_id") or data.get("source_card_id"),
        "owner": data.get("owner", data.get("player")),
        "source_card_id": data.get("source_card_id"),
        "target_role": data.get("target_role") or data.get("target_tower_role"),
        "target_card_id": data.get("target_card_id"),
        "tick": event.tick,
        "values": {
            key: value
            for key, value in data.items()
            if key not in {"uid", "target_uid", "source_uid", "sequence"}
        },
    }


def _stable_real_event(event: NormalizedEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "card_id": event.card_id,
        "owner": None if event.owner is None else (0 if event.owner == "A" else 1),
        "source_card_id": event.source_card_id,
        "target_role": event.target_role,
        "target_card_id": event.target_card_id or event.values.get("target_card_id"),
        "video_time_us": event.video_time_us,
        "match_time_us": event.match_time_us,
        "values": dict(event.values),
        "confidence": event.confidence,
    }


def _event_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("kind"),
        event.get("card_id"),
        event.get("owner"),
        event.get("source_card_id"),
        event.get("target_role"),
        event.get("target_card_id"),
    )


def _subsystem(kind: str) -> str:
    lower = kind.casefold()
    if "target" in lower or "acqui" in lower or "retarget" in lower:
        return "targeting"
    if "projectile" in lower or "flight" in lower:
        return "projectile"
    if "damage" in lower or "tower" in lower or "hp" in lower:
        return "damage"
    if "spawn" in lower or "death" in lower or "transform" in lower:
        return "lifecycle"
    if "status" in lower or "slow" in lower or "freeze" in lower:
        return "status"
    return "timing"


def _follow_up(divergence: Mapping[str, Any] | None) -> dict[str, object] | None:
    if divergence is None:
        return None
    parameter_by_subsystem = {
        "targeting": "building_acquisition_radius_mtile",
        "projectile": "projectile_speed_mtile_per_s",
        "damage": "damage_or_crown_damage",
        "lifecycle": "spawn_delay_us",
        "status": "status_duration_us",
        "timing": "first_hit_delay_us",
    }


def _minimized_scenario(
    replay: SimulatorReplay,
    divergence: Mapping[str, Any] | None,
    *,
    tick_us: int,
) -> dict[str, object] | None:
    if divergence is None:
        return None
    boundary = max(0, int(divergence.get("match_time_us", 0)) // max(1, tick_us))
    scenario = replay.scenario.to_dict()
    actions = [
        action
        for action, scheduled in zip(scenario.get("actions", []), replay.scenario.actions)
        if scheduled.tick <= boundary
    ]
    scenario["actions"] = actions
    scenario["max_ticks"] = max(1, boundary + 1)
    scenario["oracle"] = {
        "promoted": False,
        "source": "physical_lab_first_divergence_minimization",
        "parent_scenario_id": replay.scenario.scenario_id,
    }
    return scenario
    parameter = parameter_by_subsystem.get(str(divergence.get("subsystem")), "mechanic_boundary")
    return {
        "parameter": parameter,
        "strategy": "local_sweep",
        "offsets": [-0.50, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50],
        "reason": "first decision-relevant divergence; keep held-out evidence sealed",
    }


def _first_divergence(
    real_events: Sequence[NormalizedEvent],
    simulated_events: Sequence[Mapping[str, Any]],
    *,
    tick_us: int,
) -> dict[str, object] | None:
    real_rows = [_stable_real_event(event) for event in sorted(real_events, key=lambda item: item.video_time_us)]
    sim_rows = sorted(simulated_events, key=lambda item: (int(item.get("tick", 0)), str(item.get("kind", ""))))
    real_keys = [_event_key(row) for row in real_rows]
    sim_keys = [_event_key(row) for row in sim_rows]
    common = min(len(real_keys), len(sim_keys))
    index = next((index for index in range(common) if real_keys[index] != sim_keys[index]), common)
    if index >= len(real_keys) and index >= len(sim_keys):
        return None
    real = real_rows[index] if index < len(real_rows) else None
    simulated = sim_rows[index] if index < len(sim_rows) else None
    real_time = (real or {}).get("video_time_us", 0)
    if real is not None and real.get("match_time_us") is not None:
        match_time = real["match_time_us"]
    elif simulated is not None:
        match_time = int(simulated.get("tick", 0)) * tick_us
    else:
        match_time = real_time
    kind = str((real or simulated or {}).get("kind", "unknown"))
    return {
        "video_time_us": int(real_time),
        "match_time_us": int(match_time),
        "real": real,
        "simulator": simulated,
        "subsystem": _subsystem(kind),
        "confidence": float((real or {}).get("confidence", 0.0)),
        "event_index": index,
    }


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    run_id: str
    experiment_hash: str
    eligible: bool
    rejection_reasons: tuple[str, ...]
    metrics: Mapping[str, Any]
    first_divergence: Mapping[str, Any] | None
    follow_up: Mapping[str, Any] | None
    simulator_scenario: Mapping[str, Any]
    minimized_scenario: Mapping[str, Any] | None
    comparison_hash: str

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": "physical_lab_divergence_report",
            "run_id": self.run_id,
            "experiment_hash": self.experiment_hash,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
            "metrics": self.metrics,
            "first_divergence": self.first_divergence,
            "follow_up": self.follow_up,
            "simulator_scenario": self.simulator_scenario,
            "minimized_scenario": self.minimized_scenario,
        }
        result["comparison_hash"] = self.comparison_hash
        return result


def compare_observation_to_replay(
    observation: ObservationManifest,
    replay: SimulatorReplay,
    *,
    position_tolerance_mtile: int = 200,
    timing_tolerance_us: int = 10_000,
) -> ComparisonReport:
    """Compare using stable card/owner/role selectors, never detector UIDs."""

    if position_tolerance_mtile < 0 or timing_tolerance_us < 0:
        raise PhysicalLabError("comparison tolerances must be non-negative")
    reasons: list[str] = []
    if observation.status.value == "rejected":
        reasons.append("observation manifest is rejected")
    if observation.experiment_hash != replay.experiment_hash:
        reasons.append("observation and simulator replay use different experiment specifications")
    tick_us = _scenario_tick_us(replay)
    position_errors: list[float] = []
    path_real: list[float] = []
    path_sim: list[float] = []
    velocity_errors: list[float] = []
    per_entity: list[dict[str, object]] = []
    for entity in observation.entities:
        real_series = _series(entity)
        simulated = _simulated_entity_series(replay, entity.selector())
        pairs = _nearest_samples(real_series, simulated)
        errors = [math.hypot(real[1] - sim[1], real[2] - sim[2]) for real, sim in pairs]
        position_errors.extend(errors)
        real_path = _path_length(real_series)
        sim_path = _path_length(simulated)
        path_real.append(real_path)
        path_sim.append(sim_path)
        if len(real_series) >= 2 and len(simulated) >= 2:
            real_duration = max(1, real_series[-1][0] - real_series[0][0])
            sim_duration = max(1, simulated[-1][0] - simulated[0][0])
            velocity_errors.append(abs(real_path * 1_000_000 / real_duration - sim_path * 1_000_000 / sim_duration))
        per_entity.append(
            {
                "selector": {
                    "card_id": entity.card_id,
                    "owner": entity.owner,
                    "role": entity.role,
                    "source_card_id": entity.source_card_id,
                },
                "real_sample_count": len(real_series),
                "simulated_sample_count": len(simulated),
                "position_error": _metric(errors).to_dict(),
                "real_path_length_mtile": real_path,
                "simulated_path_length_mtile": sim_path,
            }
        )

    simulated_events = [_stable_sim_event(event) for event in replay.final_state.events]
    real_event_keys = [_event_key(_stable_real_event(event)) for event in observation.events]
    sim_event_keys = [_event_key(event) for event in simulated_events]
    event_common = sum(1 for key in real_event_keys if key in sim_event_keys)
    event_denominator = max(len(real_event_keys), len(sim_event_keys), 1)
    event_agreement = event_common / event_denominator
    target_real = [
        key
        for event, key in zip(observation.events, real_event_keys)
        if "target" in event.kind.casefold() or "retarget" in event.kind.casefold()
    ]
    target_sim = [
        key
        for event, key in zip(simulated_events, sim_event_keys)
        if "target" in str(event.get("kind", "")).casefold()
        or "retarget" in str(event.get("kind", "")).casefold()
    ]
    target_common = sum(1 for key in target_real if key in target_sim)
    target_agreement = target_common / max(len(target_real), len(target_sim), 1)
    victim_real = [
        key
        for event, key in zip(observation.events, real_event_keys)
        if any(token in event.kind.casefold() for token in ("victim", "damage", "hit"))
    ]
    victim_sim = [
        key
        for event, key in zip(simulated_events, sim_event_keys)
        if any(token in str(event.get("kind", "")).casefold() for token in ("victim", "damage", "hit"))
    ]
    victim_agreement = sum(1 for key in victim_real if key in victim_sim) / max(
        len(victim_real), len(victim_sim), 1
    )
    real_crown_damage = sum(
        float(event.values.get("crown_damage", event.values.get("tower_damage", 0)) or 0)
        for event in observation.events
    )
    sim_crown_damage = sum(
        float(event.get("values", {}).get("crown_damage", event.get("values", {}).get("tower_damage", 0)) or 0)
        for event in simulated_events
    )
    lifecycle_real = [
        key
        for event, key in zip(observation.events, real_event_keys)
        if any(token in event.kind.casefold() for token in ("spawn", "death", "transform", "alive"))
    ]
    lifecycle_sim = [
        key
        for event, key in zip(simulated_events, sim_event_keys)
        if any(token in str(event.get("kind", "")).casefold() for token in ("spawn", "death", "transform", "alive"))
    ]
    lifecycle_agreement = sum(1 for key in lifecycle_real if key in lifecycle_sim) / max(
        len(lifecycle_real), len(lifecycle_sim), 1
    )
    timing_errors: list[float] = []
    for event, real_key in zip(observation.events, real_event_keys):
        candidates = [
            simulated
            for simulated in simulated_events
            if _event_key(simulated) == real_key
        ]
        if candidates:
            real_time = event.match_time_us if event.match_time_us is not None else event.video_time_us
            sim_time = min(candidates, key=lambda item: abs(int(item.get("tick", 0)) * tick_us - real_time))
            timing_errors.append(abs(int(sim_time.get("tick", 0)) * tick_us - real_time))
    first_divergence = _first_divergence(observation.events, simulated_events, tick_us=tick_us)
    follow_up = _follow_up(first_divergence)
    simulator_scenario = replay.scenario.to_dict()
    minimized_scenario = _minimized_scenario(replay, first_divergence, tick_us=tick_us)
    metrics = {
        "position_mae_mtile": _metric(position_errors).to_dict(),
        "position_tolerance_mtile": position_tolerance_mtile,
        "position_within_tolerance_rate": (
            sum(error <= position_tolerance_mtile for error in position_errors) / len(position_errors)
            if position_errors
            else None
        ),
        "velocity_error_mtile_per_s": _metric(velocity_errors).to_dict(),
        "timing_error_us": _metric(timing_errors).to_dict(),
        "timing_tolerance_us": timing_tolerance_us,
        "path_length_real_mtile": _metric(path_real).to_dict(),
        "path_length_simulator_mtile": _metric(path_sim).to_dict(),
        "event_agreement_rate": event_agreement,
        "event_count_real": len(real_event_keys),
        "event_count_simulator": len(sim_event_keys),
        "target_retarget_agreement_rate": target_agreement,
        "victim_set_agreement_rate": victim_agreement,
        "hp_damage_error": {
            "real_crown_damage": real_crown_damage,
            "simulator_crown_damage": sim_crown_damage,
            "absolute_error": abs(real_crown_damage - sim_crown_damage),
        },
        "alive_dead_spawn_transform_agreement_rate": lifecycle_agreement,
        "tower_hit_count_real": sum(1 for event in observation.events if "tower" in event.kind and "damage" in event.kind),
        "tower_hit_count_simulator": sum(
            1
            for event in replay.final_state.events
            if "tower" in event.kind and "damage" in event.kind
        ),
        "per_entity": per_entity,
    }
    if first_divergence is not None and float(first_divergence.get("confidence", 0.0)) < 0.5:
        reasons.append("first divergence is based on low-confidence observed evidence")
    eligible = not reasons
    payload = {
        "run_id": observation.run_id,
        "experiment_hash": observation.experiment_hash,
        "eligible": eligible,
        "rejection_reasons": reasons,
        "metrics": metrics,
        "first_divergence": first_divergence,
        "follow_up": follow_up,
        "simulator_scenario": simulator_scenario,
        "minimized_scenario": minimized_scenario,
    }
    return ComparisonReport(
        run_id=observation.run_id,
        experiment_hash=observation.experiment_hash,
        eligible=eligible,
        rejection_reasons=tuple(reasons),
        metrics=metrics,
        first_divergence=first_divergence,
        follow_up=follow_up,
        simulator_scenario=simulator_scenario,
        minimized_scenario=minimized_scenario,
        comparison_hash=canonical_hash(payload),
    )


__all__ = [
    "ComparisonReport",
    "NumericMetric",
    "compare_observation_to_replay",
]
