"""Differential comparison between normalized observations and simulator replay."""

from __future__ import annotations

from collections import Counter, defaultdict
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


def _stable_sim_event(event: Any, replay: SimulatorReplay | None = None) -> dict[str, Any]:
    data = dict(event.data)
    target = None
    if replay is not None and type(data.get("target_uid")) is int:
        target = replay.final_state.entities.get(int(data["target_uid"]))
    if event.kind == "damage_applied" and target is not None and target.kind == "tower":
        damage = data.get("damage")
        return {
            "kind": "tower_damage_observed",
            "card_id": None,
            "owner": target.owner,
            "source_card_id": data.get("source_card_id"),
            "target_role": None,
            "target_card_id": target.card_id,
            "tick": event.tick,
            "values": {
                "damage": damage,
                "tower_damage": damage,
                "hp_after": data.get("hp_after"),
            },
        }
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
        "certainty": event.certainty.value,
    }


_REAL_SEMANTIC_KIND = {
    "unit_spawn_observed": "entity_spawned",
    "unit_disappearance_observed": "entity_died",
    "unit_transform_observed": "entity_transformed",
    "own_card_play_observed": "card_played",
    "enemy_card_play_observed": "card_played",
}
_SIM_SEMANTIC_KIND = {
    "entity_created": "entity_spawned",
    "entity_spawned": "entity_spawned",
    "entity_died": "entity_died",
    "entity_transformed": "entity_transformed",
    "card_played": "card_played",
}
_SIM_BOOKKEEPING_KINDS = frozenset({"match_started", "entity_deployed"})
_COMPARISON_IGNORED_FAMILIES = frozenset({"card_played"})


def _semantic_kind(kind: str, *, real: bool) -> str | None:
    """Map extractor and engine vocabulary onto comparable event families.

    Card-play and deployment rows are control/bookkeeping evidence rather than
    mechanics.  They remain available for diagnostics and action timing, but
    must not become a false first divergence merely because the two systems
    observe them at different points in the input/render pipeline. They remain
    present in the sealed observation and action-log provenance.
    """

    if not real and kind in _SIM_BOOKKEEPING_KINDS:
        return None
    semantic = (_REAL_SEMANTIC_KIND if real else _SIM_SEMANTIC_KIND).get(kind, kind)
    if semantic in _COMPARISON_IGNORED_FAMILIES:
        return None
    return semantic


def _semantic_event_key(event: Mapping[str, Any], *, real: bool) -> tuple[Any, ...] | None:
    if real and event.get("certainty") == "tentative":
        return None
    semantic = _semantic_kind(str(event.get("kind", "")), real=real)
    if semantic is None:
        return None
    return (
        semantic,
        event.get("card_id"),
        event.get("owner"),
        event.get("source_card_id"),
        event.get("target_role"),
        event.get("target_card_id"),
    )


def _event_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("kind"),
        event.get("card_id"),
        event.get("owner"),
        event.get("source_card_id"),
        event.get("target_role"),
        event.get("target_card_id"),
    )


def _subsystem(kind: str) -> str | None:
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
    if any(
        token in lower
        for token in (
            "attack",
            "cooldown",
            "deploy",
            "duration",
            "expire",
            "interval",
            "lifetime",
            "movement",
            "timing",
        )
    ):
        return "timing"
    # An unclassified event is not evidence for a generic timing parameter.
    # Keeping this boundary explicit prevents an unsupported mechanic from
    # silently turning into a parameter sweep.
    return None


_FOLLOW_UP_PARAMETER_BY_SUBSYSTEM = {
    "targeting": "building_acquisition_radius_mtile",
    "projectile": "projectile_speed_mtile_per_s",
    "damage": "damage_or_crown_damage",
    "lifecycle": "spawn_delay_us",
    "status": "status_duration_us",
    "timing": "first_hit_delay_us",
}
_FOLLOW_UP_OFFSETS = (-0.50, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50)
_FOLLOW_UP_MIN_CONFIDENCE = 0.5


def _verified_divergence(divergence: Mapping[str, Any]) -> bool:
    confidence = divergence.get("confidence")
    return (
        type(confidence) in (int, float)
        and math.isfinite(float(confidence))
        and float(confidence) >= _FOLLOW_UP_MIN_CONFIDENCE
    )


def _follow_up(divergence: Mapping[str, Any] | None) -> dict[str, object] | None:
    if divergence is None:
        return None
    if not _verified_divergence(divergence):
        return None
    subsystem = divergence.get("subsystem")
    if not isinstance(subsystem, str):
        return None
    parameter = _FOLLOW_UP_PARAMETER_BY_SUBSYSTEM.get(subsystem)
    if parameter is None:
        return None
    return {
        "parameter": parameter,
        "strategy": "local_sweep",
        "offsets": list(_FOLLOW_UP_OFFSETS),
        "reason": "first decision-relevant divergence; keep held-out evidence sealed",
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


def _first_divergence(
    real_events: Sequence[NormalizedEvent],
    simulated_events: Sequence[Mapping[str, Any]],
    *,
    tick_us: int,
) -> dict[str, object] | None:
    real_rows = []
    for event in sorted(
        real_events,
        key=lambda item: (
            item.match_time_us if item.match_time_us is not None else item.video_time_us,
            item.video_time_us,
        ),
    ):
        row = _stable_real_event(event)
        key = _semantic_event_key(row, real=True)
        if key is not None:
            real_rows.append((key, row))
    sim_rows = []
    for row in sorted(
        simulated_events,
        key=lambda item: (int(item.get("tick", 0)), str(item.get("kind", ""))),
    ):
        key = _semantic_event_key(row, real=False)
        if key is not None:
            sim_rows.append((key, row))
    # The simulator emits lower-level rows (target acquisition, attacks,
    # projectiles, and damage) that this extractor does not currently observe.
    # Keep real-only families as divergences, but do not let simulator-only
    # families shift the sequence index.
    real_families = {key[0] for key, _row in real_rows}
    sim_rows = [item for item in sim_rows if item[0][0] in real_families]
    real_keys = [key for key, _row in real_rows]
    sim_keys = [key for key, _row in sim_rows]
    common = min(len(real_keys), len(sim_keys))
    index = next((index for index in range(common) if real_keys[index] != sim_keys[index]), common)
    if index >= len(real_keys) and index >= len(sim_keys):
        return None
    real = real_rows[index][1] if index < len(real_rows) else None
    simulated = sim_rows[index][1] if index < len(sim_rows) else None
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

    simulated_events = [_stable_sim_event(event, replay) for event in replay.final_state.events]
    real_rows = [(_stable_real_event(event), event) for event in observation.events]
    real_scorable_rows = [
        (row, event)
        for row, event in real_rows
        if event.certainty.value != "tentative"
    ]
    real_comparable = [
        (event, row, key)
        for row, event in real_scorable_rows
        if (key := _semantic_event_key(row, real=True)) is not None
    ]
    sim_comparable = [
        (row, key)
        for row in simulated_events
        if (key := _semantic_event_key(row, real=False)) is not None
    ]
    real_families = {key[0] for _event, _row, key in real_comparable}
    sim_comparable = [item for item in sim_comparable if item[1][0] in real_families]
    real_event_keys = [key for _event, _row, key in real_comparable]
    sim_event_keys = [key for _row, key in sim_comparable]
    event_common = sum((Counter(real_event_keys) & Counter(sim_event_keys)).values())
    event_denominator = max(len(real_event_keys), len(sim_event_keys), 1)
    event_agreement = event_common / event_denominator
    real_all_keys = [_event_key(row) for row, _event in real_scorable_rows]
    sim_all_keys = [_event_key(row) for row in simulated_events]
    target_real = [
        key
        for (_row, event), key in zip(real_scorable_rows, real_all_keys)
        if "target" in event.kind.casefold() or "retarget" in event.kind.casefold()
    ]
    target_sim = [
        key
        for event, key in zip(simulated_events, sim_all_keys)
        if "target" in str(event.get("kind", "")).casefold()
        or "retarget" in str(event.get("kind", "")).casefold()
    ]
    target_common = sum((Counter(target_real) & Counter(target_sim)).values())
    target_agreement = target_common / max(len(target_real), len(target_sim), 1)
    victim_real = [
        key
        for (_row, event), key in zip(real_scorable_rows, real_all_keys)
        if any(token in event.kind.casefold() for token in ("victim", "damage", "hit"))
    ]
    victim_sim = [
        key
        for event, key in zip(simulated_events, sim_all_keys)
        if any(token in str(event.get("kind", "")).casefold() for token in ("victim", "damage", "hit"))
    ]
    victim_agreement = sum((Counter(victim_real) & Counter(victim_sim)).values()) / max(
        len(victim_real), len(victim_sim), 1
    )
    real_crown_damage = sum(
        float(event.values.get("crown_damage", event.values.get("tower_damage", 0)) or 0)
        for _row, event in real_scorable_rows
    )
    sim_crown_damage = sum(
        float(event.get("values", {}).get("crown_damage", event.get("values", {}).get("tower_damage", 0)) or 0)
        for event in simulated_events
    )
    lifecycle_real = [
        key
        for event, _row, key in real_comparable
        if _semantic_kind(event.kind, real=True) in {
            "entity_spawned",
            "entity_died",
            "entity_transformed",
        }
    ]
    lifecycle_sim = [
        key
        for event, key in sim_comparable
        if _semantic_kind(str(event.get("kind", "")), real=False) in {
            "entity_spawned",
            "entity_died",
            "entity_transformed",
        }
    ]
    lifecycle_agreement = sum((Counter(lifecycle_real) & Counter(lifecycle_sim)).values()) / max(
        len(lifecycle_real), len(lifecycle_sim), 1
    )
    timing_errors: list[float] = []
    real_timed: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    sim_timed: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for event, _row, key in real_comparable:
        real_timed[key].append(
            event.match_time_us if event.match_time_us is not None else event.video_time_us
        )
    for row, key in sim_comparable:
        sim_timed[key].append(int(row.get("tick", 0)) * tick_us)
    # Pair each occurrence at most once.  Reusing one simulator spawn for
    # every detector flicker made the old mean timing error meaningless.
    for key, real_times in real_timed.items():
        simulated_times = sim_timed.get(key, [])
        remaining = list(simulated_times)
        for real_time in real_times:
            if not remaining:
                break
            index = min(range(len(remaining)), key=lambda item: abs(remaining[item] - real_time))
            timing_errors.append(abs(remaining.pop(index) - real_time))
    first_divergence = _first_divergence(observation.events, simulated_events, tick_us=tick_us)
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
        "raw_event_count_real": len(observation.events),
        "raw_event_count_simulator": len(simulated_events),
        "target_retarget_agreement_rate": target_agreement,
        "victim_set_agreement_rate": victim_agreement,
        "hp_damage_error": {
            "real_crown_damage": real_crown_damage,
            "simulator_crown_damage": sim_crown_damage,
            "absolute_error": abs(real_crown_damage - sim_crown_damage),
        },
        "alive_dead_spawn_transform_agreement_rate": lifecycle_agreement,
        "tower_hit_count_real": sum(
            1
            for _row, event in real_scorable_rows
            if "tower" in event.kind and "damage" in event.kind
        ),
        "tower_hit_count_simulator": sum(
            1
            for event in simulated_events
            if "tower" in str(event.get("kind", ""))
            and "damage" in str(event.get("kind", ""))
        ),
        "per_entity": per_entity,
    }
    if first_divergence is not None:
        if not _verified_divergence(first_divergence):
            reasons.append("first divergence is based on low-confidence observed evidence")
        elif _follow_up(first_divergence) is None:
            reasons.append("first divergence subsystem is unsupported for bounded follow-up experiments")
    # Follow-up work is an output of an eligible comparison only.  In
    # particular, rejected observations and experiment-hash mismatches must
    # not produce actionable parameter sweeps.
    follow_up = _follow_up(first_divergence) if not reasons else None
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
