"""Compile high-confidence offline vision tracks into sealed fidelity corpora.

The miner is intentionally model-agnostic: expensive detector ensembles,
homography, OCR, and tracking run elsewhere in the repository and emit one
small JSON manifest.  This module performs the reproducible part—confidence
gating, group-disjoint split assignment, scenario construction, trajectory
measurement generation, and strict corpus validation—without network access
or human promotion of uncertain samples.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .engine import BASE_HOG_CYCLE_DECK, ENGINE_VERSION, BattleEngine
from .fixed import distance_mtile
from .geometry import cell_center_mtile
from .ruleset import Ruleset
from .scenario import Scenario, scenario_from_dict
from .state import BattleState, EntityState, battle_state_from_primitive
from .validation import ValidationCorpus, validation_corpus_from_dict


MINING_SCHEMA_VERSION = 1
_SPLITS = ("calibration", "validation", "heldout")


class MiningManifestError(ValueError):
    """Raised when an offline observation manifest is structurally invalid."""


@dataclass(frozen=True, slots=True)
class DiscardedClip:
    clip_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class MiningResult:
    corpus: ValidationCorpus
    discarded: tuple[DiscardedClip, ...]

    def summary(self) -> dict[str, object]:
        split_counts = {split: 0 for split in _SPLITS}
        for case in self.corpus.cases:
            split_counts[case.split.value] += 1
        return {
            "schema_version": 1,
            "corpus_id": self.corpus.corpus_id,
            "corpus_hash": self.corpus.content_hash,
            "accepted_cases": len(self.corpus.cases),
            "discarded_cases": len(self.discarded),
            "split_counts": split_counts,
            "discarded": [
                {"clip_id": item.clip_id, "reason": item.reason}
                for item in self.discarded
            ],
        }


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MiningManifestError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise MiningManifestError(f"{name} must be an array")
    return value


def _name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MiningManifestError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MiningManifestError(f"{name} must be an integer >= {minimum}")
    return value


def _confidence(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise MiningManifestError(f"{name} must be finite and between 0 and 1")
    return float(value)


def _label_fps(truth: Mapping[str, object]) -> float | None:
    """Return an explicit annotation cadence, rejecting malformed metadata."""

    value = truth.get("fps")
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise MiningManifestError("ground-truth fps must be finite and positive")
    return float(value)


def _labeled_frame_video_time(
    frame_idx: int,
    *,
    label_fps: float | None,
    replay_frame_times: Mapping[int, float],
) -> float | None:
    """Map a label-frame index to video time without assuming cache cadence.

    Curated action files normally declare their annotation FPS. Lightweight
    synthetic fixtures may omit it and use replay-cache frame indices; those
    are interpolated on the cache's own frame/time axis.
    """

    if label_fps is not None:
        return frame_idx / label_fps
    exact = replay_frame_times.get(frame_idx)
    if exact is not None:
        return float(exact)
    ordered = sorted(replay_frame_times.items())
    if len(ordered) < 2:
        return None
    first_idx, first_time = ordered[0]
    last_idx, last_time = ordered[-1]
    if last_idx == first_idx:
        return None
    seconds_per_index = (last_time - first_time) / (last_idx - first_idx)
    return first_time + (frame_idx - first_idx) * seconds_per_index


def assigned_split(group_id: str, *, salt: str = "hog-cycle-sim-v1") -> str:
    """Assign whole capture groups before evaluation using a stable hash."""

    digest = hashlib.sha256(f"{salt}:{group_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "calibration"
    if bucket < 85:
        return "validation"
    return "heldout"


def _canonical_media_hash(value: object, name: str) -> str:
    raw = _name(value, name)
    if not raw.startswith("sha256:") or len(raw) != 71:
        raise MiningManifestError(f"{name} must be sha256:<64 lowercase hex characters>")
    try:
        int(raw[7:], 16)
    except ValueError as error:
        raise MiningManifestError(f"{name} has invalid hex") from error
    if raw != raw.lower():
        raise MiningManifestError(f"{name} must use lowercase hex")
    return raw


def _state_from_clip(
    clip: Mapping[str, Any],
    engine: BattleEngine,
) -> tuple[BattleState, dict[str, int]]:
    if "initial_state" in clip:
        state = battle_state_from_primitive(_object(clip["initial_state"], "clip.initial_state"))
        engine.validate_state(state)
        return state, {}

    initial = _object(clip.get("initial"), "clip.initial")
    seed = _integer(clip.get("seed", 0), "clip.seed")
    decks_raw = clip.get("decks")
    decks = (
        (BASE_HOG_CYCLE_DECK, BASE_HOG_CYCLE_DECK)
        if decks_raw is None
        else tuple(tuple(_array(deck, "clip.decks[]")) for deck in _array(decks_raw, "clip.decks"))
    )
    if len(decks) != 2:
        raise MiningManifestError("clip.decks must contain two decks")
    state = engine.new_battle(decks, seed=seed, shuffle_decks=False)  # type: ignore[arg-type]
    state.tick = _integer(initial["tick"], "clip.initial.tick")
    state.elapsed_us = _integer(initial["elapsed_us"], "clip.initial.elapsed_us")
    state.phase = _name(initial.get("phase", "regulation"), "clip.initial.phase")
    state.events.clear()
    state.event_sequence = 0

    tower_rows = _array(initial.get("towers", []), "clip.initial.towers")
    for index, raw_tower in enumerate(tower_rows):
        tower = _object(raw_tower, f"clip.initial.towers[{index}]")
        owner = _integer(tower["owner"], "tower.owner")
        role = _name(tower["role"], "tower.role")
        match = next(
            (item for item in state.entities.values() if item.kind == "tower" and item.owner == owner and item.role == role),
            None,
        )
        if match is None:
            raise MiningManifestError(f"unknown tower owner/role: {owner}/{role}")
        match.hp = _integer(tower["hp"], "tower.hp")
        match.alive = match.hp > 0

    state.entities = {
        uid: entity for uid, entity in state.entities.items() if entity.kind == "tower"
    }
    track_uids: dict[str, int] = {}
    entity_rows = _array(initial.get("entities", []), "clip.initial.entities")
    for index, raw_entity in enumerate(
        sorted(entity_rows, key=lambda row: str(_object(row, "entity").get("track_id", "")))
    ):
        row = _object(raw_entity, f"clip.initial.entities[{index}]")
        track_id = _name(row["track_id"], "entity.track_id")
        if track_id in track_uids:
            raise MiningManifestError(f"duplicate initial track_id {track_id!r}")
        card_id = engine.ruleset.resolve_card_id(_name(row["card_id"], "entity.card_id"))
        definition = engine.ruleset.card(card_id)
        uid = state.next_uid
        state.next_uid += 1
        hp = _integer(row.get("hp", definition.hitpoints), "entity.hp", minimum=1)
        entity = EntityState(
            uid=uid,
            card_id=card_id,
            owner=_integer(row["owner"], "entity.owner"),
            kind=definition.kind,
            x_mtile=_integer(row["x_mtile"], "entity.x_mtile"),
            y_mtile=_integer(row["y_mtile"], "entity.y_mtile"),
            hp=hp,
            max_hp=int(definition.hitpoints or hp),
            spawn_tick=_integer(row.get("spawn_tick", state.tick), "entity.spawn_tick"),
            deploy_remaining_us=_integer(
                row.get("deploy_remaining_us", 0),
                "entity.deploy_remaining_us",
                minimum=0,
            ),
            lifetime_remaining_us=definition.lifetime_us,
        )
        state.entities[uid] = entity
        if definition.kind == "building":
            state.navigation_revision += 1
        track_uids[track_id] = uid
    engine.validate_state(state)
    return state, track_uids


def _clip_confidences(clip: Mapping[str, Any]) -> tuple[float, ...]:
    values: list[float] = [_confidence(clip["confidence"], "clip.confidence")]
    initial = clip.get("initial")
    if isinstance(initial, dict):
        for section in ("entities", "towers"):
            for index, item in enumerate(initial.get(section, [])):
                row = _object(item, f"clip.initial.{section}[{index}]")
                values.append(_confidence(row.get("confidence", clip["confidence"]), "confidence"))
    for track_index, raw_track in enumerate(clip.get("tracks", [])):
        track = _object(raw_track, f"clip.tracks[{track_index}]")
        values.append(_confidence(track.get("confidence", clip["confidence"]), "track.confidence"))
        for sample_index, raw_sample in enumerate(track.get("samples", [])):
            sample = _object(raw_sample, f"track.samples[{sample_index}]")
            values.append(_confidence(sample.get("confidence", track.get("confidence", clip["confidence"])), "sample.confidence"))
    return tuple(values)


def _measurements(
    clip: Mapping[str, Any],
    track_uids: Mapping[str, int],
    *,
    default_position_tolerance: int,
    default_hp_tolerance: int,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    clip_id = _name(clip["clip_id"], "clip.clip_id")
    for track_index, raw_track in enumerate(_array(clip.get("tracks", []), "clip.tracks")):
        track = _object(raw_track, f"clip.tracks[{track_index}]")
        track_id = _name(track["track_id"], "track.track_id")
        selector_raw = track.get("selector")
        if selector_raw is None:
            if track_id not in track_uids:
                raise MiningManifestError(
                    f"track {track_id!r} needs selector when initial_state is supplied"
                )
            selector: dict[str, object] = {"uid": track_uids[track_id]}
        else:
            selector = dict(_object(selector_raw, "track.selector"))
        mechanic = _name(track.get("mechanic", "movement"), "track.mechanic")
        samples = _array(track.get("samples", []), "track.samples")
        for sample_index, raw_sample in enumerate(samples):
            sample = _object(raw_sample, f"track.samples[{sample_index}]")
            tick = _integer(sample["tick"], "sample.tick")
            for field_name, extractor_name, tolerance in (
                ("x_mtile", "entity_x_mtile_at_tick", default_position_tolerance),
                ("y_mtile", "entity_y_mtile_at_tick", default_position_tolerance),
                ("hp", "entity_hp_at_tick", default_hp_tolerance),
                ("alive", "entity_alive_at_tick", 0),
            ):
                if field_name not in sample:
                    continue
                value = sample[field_name]
                if field_name == "alive":
                    if type(value) is not bool:
                        raise MiningManifestError("sample.alive must be boolean")
                else:
                    _integer(value, f"sample.{field_name}")
                result.append(
                    {
                        "sample_id": f"{clip_id}:{track_id}:{tick}:{field_name}",
                        "mechanic": f"{mechanic}_{field_name}",
                        "observed_value": value,
                        "observed_tick": tick,
                        "tolerance": {"absolute": tolerance, "relative": 0.0, "ticks": 0},
                        "extractor": {
                            "type": extractor_name,
                            "tick": tick,
                            "filters": selector,
                        },
                    }
                )
        speed = track.get("displacement_speed")
        if speed is not None:
            speed_row = _object(speed, "track.displacement_speed")
            start_tick = _integer(speed_row["start_tick"], "displacement_speed.start_tick")
            end_tick = _integer(speed_row["end_tick"], "displacement_speed.end_tick")
            if end_tick <= start_tick:
                raise MiningManifestError(
                    "displacement_speed.end_tick must be after start_tick"
                )
            value = _integer(
                speed_row["observed_mtile_per_s"],
                "displacement_speed.observed_mtile_per_s",
                minimum=1,
            )
            tolerance = _integer(
                speed_row.get("tolerance_mtile_per_s", 120),
                "displacement_speed.tolerance_mtile_per_s",
            )
            compare_to_card_base_speed = bool(
                speed_row.get("compare_to_card_base_speed", False)
            )
            extractor = (
                {
                    "type": "card_move_speed_mtile_per_s",
                    "filters": selector,
                }
                if compare_to_card_base_speed
                else {
                    "type": "entity_displacement_speed_mtile_per_s",
                    "start_tick": start_tick,
                    "end_tick": end_tick,
                    "filters": selector,
                }
            )
            result.append(
                {
                    "sample_id": f"{clip_id}:{track_id}:{start_tick}:{end_tick}:speed",
                    "mechanic": f"{mechanic}_displacement_speed_mtile_per_s",
                    "observed_value": value,
                    "observed_tick": end_tick,
                    "tolerance": {"absolute": tolerance, "relative": 0.0, "ticks": 0},
                    "extractor": extractor,
                }
            )
    return result


def compile_observation_manifest(
    raw: Mapping[str, Any],
    *,
    engine: BattleEngine | None = None,
    confidence_threshold: float | None = None,
) -> MiningResult:
    """Compile detector/tracker output, discarding ambiguity automatically."""

    engine = engine or BattleEngine()
    row = _object(dict(raw), "manifest")
    if _integer(row.get("schema_version"), "manifest.schema_version", minimum=1) != MINING_SCHEMA_VERSION:
        raise MiningManifestError("unsupported mining manifest schema")
    corpus_id = _name(row["corpus_id"], "manifest.corpus_id")
    salt = _name(row.get("split_salt", "hog-cycle-sim-v1"), "manifest.split_salt")
    threshold = _confidence(
        row.get("confidence_threshold", 0.98)
        if confidence_threshold is None
        else confidence_threshold,
        "manifest.confidence_threshold",
    )
    position_tolerance = _integer(
        row.get("position_tolerance_mtile", 200),
        "manifest.position_tolerance_mtile",
    )
    hp_tolerance = _integer(row.get("hp_tolerance", 0), "manifest.hp_tolerance")
    cases: list[dict[str, object]] = []
    discarded: list[DiscardedClip] = []
    for index, raw_clip in enumerate(_array(row["clips"], "manifest.clips")):
        clip = _object(raw_clip, f"manifest.clips[{index}]")
        clip_id = _name(clip.get("clip_id"), f"manifest.clips[{index}].clip_id")
        if clip.get("occluded") is True:
            discarded.append(DiscardedClip(clip_id, "occluded"))
            continue
        confidences = _clip_confidences(clip)
        if min(confidences) < threshold:
            discarded.append(
                DiscardedClip(clip_id, f"confidence_below_threshold:{min(confidences):.6f}")
            )
            continue
        group_id = _name(clip["group_id"], "clip.group_id")
        split = clip.get("split") or assigned_split(group_id, salt=salt)
        if split not in _SPLITS:
            raise MiningManifestError(f"clip.split must be one of {_SPLITS}")
        media_hash = _canonical_media_hash(clip["media_hash"], "clip.media_hash")
        supplied_scenario: Scenario | None = None
        if "scenario" in clip:
            try:
                supplied_scenario = scenario_from_dict(
                    _object(clip["scenario"], "clip.scenario")
                )
            except (KeyError, TypeError, ValueError) as error:
                raise MiningManifestError(f"invalid clip.scenario: {error}") from error
            if (
                supplied_scenario.engine_version != ENGINE_VERSION
                or supplied_scenario.ruleset_id != engine.ruleset.ruleset_id
                or supplied_scenario.ruleset_hash != engine.ruleset.content_hash
            ):
                raise MiningManifestError("clip.scenario does not match engine/ruleset contract")
            state = (
                engine.new_battle(
                    supplied_scenario.decks,
                    seed=supplied_scenario.seed,
                    shuffle_decks=supplied_scenario.shuffle_decks,
                )
                if supplied_scenario.initial_state is None
                else battle_state_from_primitive(
                    supplied_scenario.to_dict()["initial_state"]
                )
            )
            track_uids: dict[str, int] = {}
        else:
            try:
                state, track_uids = _state_from_clip(clip, engine)
            except ValueError as error:
                # Vision positions can fall inside a conservative simulated
                # tower/building radius because of occlusion or homography
                # noise. Such a clip is not a valid initial condition, but it
                # must not prevent independent clean clips from being mined.
                discarded.append(
                    DiscardedClip(clip_id, f"invalid_initial_state:{error}")
                )
                continue
            if clip.get("actions"):
                raise MiningManifestError(
                    "observed snapshots with actions require clip.scenario with a fully "
                    "reconstructed canonical action schedule"
                )
        measurements = _measurements(
            clip,
            track_uids,
            default_position_tolerance=position_tolerance,
            default_hp_tolerance=hp_tolerance,
        )
        if not measurements:
            discarded.append(DiscardedClip(clip_id, "no_supported_measurements"))
            continue
        terminal_ticks: list[int] = []
        for sample in measurements:
            extractor_row = sample["extractor"]
            assert isinstance(extractor_row, dict)
            terminal = extractor_row.get("tick")
            if terminal is None:
                terminal = extractor_row.get("end_tick")
            if terminal is None:
                # Timeless ruleset extractors are associated with the
                # observation endpoint for scenario bounding and reporting.
                terminal = sample["observed_tick"]
            terminal_ticks.append(
                _integer(terminal, "extractor terminal tick")
            )
        end_tick = max(terminal_ticks)
        if end_tick <= state.tick:
            end_tick = state.tick + 1
        if supplied_scenario is None:
            scenario = Scenario(
                scenario_id=f"auto:{clip_id}",
                ruleset_id=engine.ruleset.ruleset_id,
                ruleset_hash=engine.ruleset.content_hash,
                engine_version=ENGINE_VERSION,
                seed=state.seed,
                decks=tuple(player.deck for player in state.players),
                initial_state=state.to_primitive(include_events=False),
                max_ticks=end_tick,
                split=str(split),
                tags=("auto-mined", "trajectory"),
                oracle={"promoted": False, "automated": True},
            )
        else:
            if supplied_scenario.max_ticks is not None and end_tick > supplied_scenario.max_ticks:
                raise MiningManifestError("track sample occurs after clip.scenario max_ticks")
            scenario = replace(
                supplied_scenario,
                scenario_id=f"auto:{clip_id}",
                split=str(split),
                tags=tuple(sorted(set(supplied_scenario.tags) | {"auto-mined"})),
                oracle={"promoted": False, "automated": True},
            )
        frame_start = _integer(clip["frame_start"], "clip.frame_start")
        frame_end = _integer(clip["frame_end"], "clip.frame_end")
        if frame_end < frame_start:
            raise MiningManifestError("clip.frame_end must not precede frame_start")
        cases.append(
            {
                "case_id": clip_id,
                "split": split,
                "evidence": {
                    "source_id": clip_id,
                    "group_id": group_id,
                    "method": _name(clip["method"], "clip.method"),
                    "confidence": min(confidences),
                    "media_hash": media_hash,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "notes": "Automatically confidence-gated; no uncertain samples promoted.",
                },
                "scenario": scenario.to_dict(),
                "measurements": measurements,
                "traces": [],
            }
        )
    if not cases:
        reason_counts: dict[str, int] = {}
        for item in discarded:
            reason_counts[item.reason] = reason_counts.get(item.reason, 0) + 1
        summary = ", ".join(
            f"{reason} ({count})"
            for reason, count in sorted(reason_counts.items())
        )
        raise MiningManifestError(
            "all clips were discarded; refusing to create an empty corpus"
            + (f": {summary}" if summary else "")
        )
    corpus = validation_corpus_from_dict(
        {
            "schema_version": 1,
            "corpus_id": corpus_id,
            "engine_version": ENGINE_VERSION,
            "ruleset_id": engine.ruleset.ruleset_id,
            "ruleset_hash": engine.ruleset.content_hash,
            "cases": cases,
        }
    )
    # Seal the same expanded representation that ``corpus_to_dict`` writes so
    # an out-of-band hash from ``result.summary()`` pins the emitted file.
    corpus = validation_corpus_from_dict(corpus_to_dict(corpus))
    return MiningResult(corpus=corpus, discarded=tuple(discarded))


def load_observation_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MiningManifestError(f"cannot load mining manifest {source}: {error}") from error
    return _object(raw, "manifest")


def corpus_to_dict(corpus: ValidationCorpus) -> dict[str, object]:
    """Return the canonical expanded JSON form accepted by the corpus loader."""

    cases: list[dict[str, object]] = []
    for case in corpus.cases:
        assert case.scenario is not None
        measurements: list[dict[str, object]] = []
        for item in case.measurements:
            observed = item.observed
            extractor = item.extractor
            measurements.append(
                {
                    "sample_id": observed.sample_id,
                    "mechanic": observed.mechanic,
                    "observed_value": observed.observed_value,
                    "observed_tick": observed.observed_tick,
                    "tolerance": {
                        "absolute": observed.tolerance.absolute,
                        "relative": observed.tolerance.relative,
                        "ticks": observed.tolerance.ticks,
                    },
                    "extractor": {
                        "type": extractor.extractor_type,
                        **(
                            {"event_kind": extractor.event_kind}
                            if extractor.event_kind is not None
                            else {}
                        ),
                        **(
                            {"field": extractor.field_name}
                            if extractor.field_name is not None
                            else {}
                        ),
                        **(
                            {"tick": extractor.tick}
                            if extractor.tick is not None
                            else {}
                        ),
                        **(
                            {
                                "start_tick": extractor.start_tick,
                                "end_tick": extractor.end_tick,
                            }
                            if extractor.start_tick is not None
                            else {}
                        ),
                        "filters": dict(extractor.filters),
                    },
                }
            )
        evidence = case.evidence
        traces: list[dict[str, object]] = []
        for trace_spec in case.traces:
            observed_trace = trace_spec.observed
            traces.append(
                {
                    "trace_id": observed_trace.trace_id,
                    "mechanic": observed_trace.mechanic,
                    "included_event_kinds": (
                        sorted(observed_trace.included_event_kinds)
                        if observed_trace.included_event_kinds is not None
                        else None
                    ),
                    "filters": dict(trace_spec.filters),
                    "events": [
                        {
                            "tick": event.tick,
                            "kind": event.kind,
                            "values": dict(event.values),
                            "tick_tolerance": event.tick_tolerance,
                            "value_tolerances": {
                                key: {
                                    "absolute": tolerance.absolute,
                                    "relative": tolerance.relative,
                                    "ticks": tolerance.ticks,
                                }
                                for key, tolerance in event.value_tolerances.items()
                            },
                        }
                        for event in observed_trace.events
                    ],
                }
            )
        cases.append(
            {
                "case_id": case.case_id,
                "split": case.split.value,
                "evidence": {
                    "source_id": evidence.source_id,
                    "group_id": evidence.group_id,
                    "method": evidence.method,
                    "confidence": evidence.confidence,
                    "notes": evidence.notes,
                    "media_hash": evidence.media_hash,
                    "frame_start": evidence.frame_start,
                    "frame_end": evidence.frame_end,
                },
                "scenario": case.scenario.to_dict(),
                "measurements": measurements,
                "traces": traces,
            }
        )
    return {
        "schema_version": corpus.schema_version,
        "corpus_id": corpus.corpus_id,
        "engine_version": corpus.engine_version,
        "ruleset_id": corpus.ruleset_id,
        "ruleset_hash": corpus.ruleset_hash,
        "cases": cases,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _detection_world_position(
    detection: object,
    arena_px: tuple[int, int, int, int],
    *,
    ground_anchor: bool = False,
    center_margin_mtile: int = 0,
) -> tuple[int, int]:
    from cr_bot.features.action_space import ACTION_GRID

    ax, ay, aw, ah = arena_px
    if aw <= 0 or ah <= 0:
        raise MiningManifestError("replay arena dimensions must be positive")
    norm_x = (float(detection.center_x) - ax) / aw
    anchor_y = detection.y2 if ground_anchor else detection.center_y
    norm_y = (float(anchor_y) - ay) / ah
    x = round((norm_x - ACTION_GRID.x0) / ACTION_GRID.width * 18_000)
    y = round((norm_y - ACTION_GRID.y0) / ACTION_GRID.height * 32_000)
    if center_margin_mtile < 0 or center_margin_mtile >= 9_000:
        raise MiningManifestError("center margin must fit inside the policy arena")
    return (
        min(17_999 - center_margin_mtile, max(center_margin_mtile, x)),
        min(31_999 - center_margin_mtile, max(center_margin_mtile, y)),
    )


def _observed_hp(detection: object, maximum: int) -> int:
    observed = detection.estimated_hp
    if observed is None:
        return maximum
    value = float(observed)
    if not math.isfinite(value) or value < 0:
        raise MiningManifestError("detector HP estimate must be finite and non-negative")
    if value <= 1.0:
        hp = round(value * maximum)
    else:
        from cr_bot.domain.troop_hp_level16 import get_unit_hp_level16

        detector_maximum = get_unit_hp_level16(str(detection.class_name))
        hp = (
            round(value / detector_maximum * maximum)
            if detector_maximum is not None and detector_maximum > 0
            else round(value)
        )
    return min(maximum, max(1, hp))


def _validate_replay_source_level(
    analysis: object,
    *,
    source_level: int,
    engine: BattleEngine,
    expected_support_tower_hp: int | None = None,
) -> bool:
    """Fail closed when replay evidence conflicts with the pinned level.

    Compact replay caches do not retain the level-badge detections needed to
    infer a match level reliably. The vision pipeline initializes unreadable
    towers with configured live-capture maxima, so King HP is never admissible
    and the explicit support-tower fallback sentinel is ignored. The caller
    must declare the source level. Any other support HP above the pinned
    Level-11 maximum proves the source (or OCR) unsuitable, while an exact
    full-HP reading positively confirms the declared level.
    """

    if type(source_level) is not int or source_level < 1:
        raise MiningManifestError("source_level must be a positive integer")
    if expected_support_tower_hp is not None and (
        type(expected_support_tower_hp) is not int or expected_support_tower_hp < 1
    ):
        raise MiningManifestError("expected_support_tower_hp must be a positive integer")
    if source_level != engine.ruleset.level and expected_support_tower_hp is None:
        raise MiningManifestError(
            f"replay source level {source_level} does not match ruleset level "
            f"{engine.ruleset.level}; an explicit expected support-tower HP is "
            "required for level-invariant evidence"
        )
    towers_hp = getattr(analysis, "towers_hp", None)
    if not isinstance(towers_hp, Mapping):
        return False
    maximum = (
        int(expected_support_tower_hp)
        if expected_support_tower_hp is not None
        else int(engine.ruleset.tower("princess-tower").hitpoints)
    )
    from cr_bot.domain.constants import PRINCESS_TOWER_HP as vision_support_fallback

    confirmed = False
    for name in (
        "own_support_left",
        "own_support_right",
        "enemy_support_left",
        "enemy_support_right",
    ):
        value = towers_hp.get(name)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            continue
        if float(value) == float(vision_support_fallback) and vision_support_fallback != maximum:
            continue
        if float(value) > maximum:
            raise MiningManifestError(
                f"replay support-tower HP conflicts with Level {source_level}: "
                f"{name}={value} > {maximum}"
            )
        if float(value) == maximum:
            confirmed = True
    return confirmed


def _pull_frame_is_contaminated(
    units: list[dict[str, object]],
    *,
    hog_key: tuple[str, int, int],
    cannon_key: tuple[str, int, int],
    hog_position: tuple[int, int],
    cannon_position: tuple[int, int],
    radius_mtile: int,
) -> bool:
    """Return whether a third unit can affect a mined pull observation."""

    for unit in units:
        if unit["key"] in {hog_key, cannon_key}:
            continue
        unit_position = (int(unit["x_mtile"]), int(unit["y_mtile"]))
        if (
            distance_mtile(*hog_position, *unit_position) < radius_mtile
            or distance_mtile(*cannon_position, *unit_position) < radius_mtile
        ):
            return True
    return False


def _movement_mechanic(
    card_id: str,
    samples: list[dict[str, object]],
    ruleset: Ruleset,
) -> str:
    """Classify isolated trajectories without requiring manual clip labels."""

    river_min = ruleset.arena.river_y_min_mtile
    river_max = ruleset.arena.river_y_max_mtile
    crosses_bridge = any(
        river_min <= int(sample["y_mtile"]) <= river_max
        and any(
            bridge_min <= int(sample["x_mtile"]) <= bridge_max
            for bridge_min, bridge_max in ruleset.arena.bridge_x_ranges_mtile
        )
        for sample in samples
    )
    suffix = "isolated_bridge_path" if crosses_bridge else "isolated_movement"
    return f"{card_id}_{suffix}"


def _movement_speed_ratio_permille(
    displacement_mtile: int,
    duration_us: int,
    expected_speed_mtile_per_s: int,
) -> int:
    if duration_us <= 0 or expected_speed_mtile_per_s <= 0:
        raise ValueError("movement speed ratio requires positive duration and speed")
    observed_speed = displacement_mtile * 1_000_000 // duration_us
    return observed_speed * 1_000 // expected_speed_mtile_per_s


def _trajectory_has_consistent_motion(
    samples: list[dict[str, object]],
    *,
    tick_us: int,
) -> bool:
    """Select continuous motion without consulting the simulator card speed."""

    if len(samples) < 3:
        return False
    # Native captures can contribute multiple noisy detections per 50 ms
    # physics tick. Evaluate motion over >=100 ms baselines so one-pixel box
    # jitter does not look like a reversal or a 4x speed spike.
    smoothed = [samples[0]]
    for sample in samples[1:-1]:
        if int(sample["tick"]) - int(smoothed[-1]["tick"]) >= 2:
            smoothed.append(sample)
    if samples[-1] is not smoothed[-1]:
        smoothed.append(samples[-1])
    if len(smoothed) < 3:
        return False
    segments: list[tuple[int, int, int]] = []
    path_length = 0
    for left, right in zip(smoothed, smoothed[1:]):
        dt_us = (int(right["tick"]) - int(left["tick"])) * tick_us
        if dt_us <= 0:
            return False
        dx = int(right["x_mtile"]) - int(left["x_mtile"])
        dy = int(right["y_mtile"]) - int(left["y_mtile"])
        distance = distance_mtile(0, 0, dx, dy)
        # A pause is attack/status/collision evidence, not isolated base motion.
        if distance < 20:
            return False
        speed = (distance * 1_000_000 + dt_us // 2) // dt_us
        segments.append((dx, dy, speed))
        path_length += distance
    ordered_speeds = sorted(speed for _, _, speed in segments)
    median_speed = ordered_speeds[len(ordered_speeds) // 2]
    if median_speed <= 0:
        return False
    if any(
        speed * 3 < median_speed or speed > median_speed * 3
        for _, _, speed in segments
    ):
        return False
    # Detector ID swaps and combat turns can maintain plausible scalar speed;
    # reject reversals and strongly looping paths independently of card stats.
    if any(
        (left_dx * right_dx + left_dy * right_dy) * 2
        < -distance_mtile(0, 0, left_dx, left_dy)
        * distance_mtile(0, 0, right_dx, right_dy)
        for (left_dx, left_dy, _), (right_dx, right_dy, _) in zip(
            segments, segments[1:]
        )
    ):
        return False
    endpoint = distance_mtile(
        int(smoothed[0]["x_mtile"]),
        int(smoothed[0]["y_mtile"]),
        int(smoothed[-1]["x_mtile"]),
        int(smoothed[-1]["y_mtile"]),
    )
    return endpoint * 1_000 >= path_length * 700


def _replay_cache_interaction_rows(
    cache_path: Path,
    *,
    engine: BattleEngine,
    source_level: int,
    expected_support_tower_hp: int | None,
    confidence_threshold: float,
    contamination_confidence_threshold: float,
) -> tuple[
    dict[tuple[str, int, int], list[dict[str, object]]],
    dict[int, list[dict[str, object]]],
    int,
    int,
    float,
    bool,
]:
    """Read one sealed replay cache into deterministic interaction rows.

    The extractor is deliberately card-stat agnostic.  It only uses detector
    identity, homography, track IDs, and support-tower HP for the source-level
    check.  Card definitions are used to select a coordinate anchor and to
    classify a row; they never decide whether a candidate is accepted.
    """

    from collections import defaultdict

    from cr_bot.replay.cache import ReplayCacheReader
    from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

    tracks: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    frame_units: dict[int, list[dict[str, object]]] = defaultdict(list)
    first_frame: int | None = None
    last_frame: int | None = None
    first_video_time: float | None = None
    last_video_time: float | None = None
    level_confirmed = False

    for replay_frame in ReplayCacheReader(cache_path):
        frame_idx = int(replay_frame.frame_idx)
        video_time_s = float(replay_frame.video_time_s)
        first_frame = frame_idx if first_frame is None else min(first_frame, frame_idx)
        last_frame = frame_idx if last_frame is None else max(last_frame, frame_idx)
        first_video_time = (
            video_time_s if first_video_time is None else min(first_video_time, video_time_s)
        )
        last_video_time = (
            video_time_s if last_video_time is None else max(last_video_time, video_time_s)
        )
        analysis = replay_frame.analysis
        level_confirmed = _validate_replay_source_level(
            analysis,
            source_level=source_level,
            engine=engine,
            expected_support_tower_hp=expected_support_tower_hp,
        ) or level_confirmed
        visible: list[dict[str, object]] = []
        for match in analysis.matches:
            detection = match.troop
            card_id = DIRECT_UNIT_TO_CARD.get(detection.class_name)
            if card_id not in engine.ruleset.cards or detection.team not in {"ally", "enemy"}:
                continue
            confidence = float(detection.confidence)
            if confidence < contamination_confidence_threshold:
                continue
            definition = engine.ruleset.card(card_id)
            x_mtile, y_mtile = _detection_world_position(
                detection,
                analysis.arena_px,
                ground_anchor=definition.kind in {"troop", "building"},
                center_margin_mtile=int(definition.collision_radius_mtile or 0),
            )
            owner = 0 if detection.team == "ally" else 1
            track_id = detection.track_id
            key = (
                card_id,
                owner,
                int(track_id),
            ) if track_id is not None else None
            row = {
                "frame_idx": frame_idx,
                "video_time_s": video_time_s,
                "tick": max(
                    0,
                    round(
                        (300.0 - float(analysis.total_remaining_s or 300.0))
                        * 1_000_000
                        / engine.ruleset.tick_us
                    ),
                ),
                "x_mtile": x_mtile,
                "y_mtile": y_mtile,
                "hp": _observed_hp(detection, int(definition.hitpoints or 1)),
                "confidence": confidence,
                "card_id": card_id,
                "owner": owner,
                "track_id": int(track_id) if track_id is not None else None,
                "kind": definition.kind,
                "movement_layer": str(definition.mechanics.get("movement_layer") or "ground"),
            }
            visible.append(row)
            if key is not None and confidence >= confidence_threshold:
                tracks[key].append(row)
        frame_units[frame_idx] = visible

    if first_frame is None or last_frame is None:
        raise MiningManifestError("replay cache contains no frames")
    assert first_video_time is not None and last_video_time is not None
    return tracks, frame_units, first_frame, last_frame, last_video_time, level_confirmed


def _split_interaction_track(
    rows: list[dict[str, object]],
    *,
    maximum_track_gap_s: float,
) -> list[list[dict[str, object]]]:
    """Split a detector track at time gaps without repairing an occlusion."""

    ordered = sorted(rows, key=lambda row: (int(row["frame_idx"]), float(row["video_time_s"])))
    result: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for row in ordered:
        if current and float(row["video_time_s"]) - float(current[-1]["video_time_s"]) > maximum_track_gap_s:
            result.append(current)
            current = []
        current.append(row)
    if current:
        result.append(current)
    return result


def _bridge_index_for_position(
    ruleset: Ruleset,
    x_mtile: int,
) -> int | None:
    for index, (left, right) in enumerate(ruleset.arena.bridge_x_ranges_mtile):
        if left <= x_mtile <= right:
            return index
    return None


def _interaction_report_mechanics(
    candidates: list[dict[str, object]],
    rejected: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in candidates:
        mechanic = str(row.get("mechanic", "unknown"))
        counts.setdefault(mechanic, {"candidate_count": 0, "rejected_count": 0})
        counts[mechanic]["candidate_count"] += 1
    for row in rejected:
        mechanic = str(row.get("mechanic", "unknown"))
        counts.setdefault(mechanic, {"candidate_count": 0, "rejected_count": 0})
        counts[mechanic]["rejected_count"] += 1
    return {key: counts[key] for key in sorted(counts)}


def discover_replay_cache_interactions(
    cache_path: str | Path,
    *,
    source_level: int,
    engine: BattleEngine | None = None,
    level_invariant_current_ruleset: bool = False,
    expected_support_tower_hp: int | None = None,
    confidence_threshold: float = 0.90,
    contamination_confidence_threshold: float = 0.25,
    minimum_track_frames: int = 5,
    maximum_track_gap_s: float = 0.35,
    isolation_radius_mtile: int = 3_500,
    minimum_bridge_displacement_mtile: int = 1_000,
    minimum_pull_approach_mtile: int = 1_000,
    cannon_duration_tolerance_s: float = 3.0,
    minimum_absent_tail_s: float = 0.5,
    level_proof_verified: bool = False,
) -> dict[str, object]:
    """Mine action-free, high-confidence interaction candidates.

    This is an evidence discovery pass, not a truth writer.  It infers card
    action *candidates* from detector track onsets and then looks for three
    mechanically useful signatures:

    * a ground troop crossing a legal bridge without a detector-contaminated
      local neighborhood;
    * an undamaged Cannon whose observed HP follows a lifetime-decay curve and
      then disappears; and
    * an enemy Hog approaching the only locally visible Cannon.

    No action cell, expected trajectory, or simulator outcome is fabricated.
    Ambiguous windows are retained in ``rejected`` with a reason so a later
    in-game capture or a human audit can target only the missing evidence.
    """

    threshold = _confidence(confidence_threshold, "confidence_threshold")
    contamination_threshold = _confidence(
        contamination_confidence_threshold,
        "contamination_confidence_threshold",
    )
    if contamination_threshold >= threshold:
        raise MiningManifestError(
            "contamination confidence threshold must be below candidate threshold"
        )
    for value, name in (
        (minimum_track_frames, "minimum_track_frames"),
        (minimum_bridge_displacement_mtile, "minimum_bridge_displacement_mtile"),
        (minimum_pull_approach_mtile, "minimum_pull_approach_mtile"),
        (isolation_radius_mtile, "isolation_radius_mtile"),
    ):
        if type(value) is not int or value < 1:
            raise MiningManifestError(f"{name} must be a positive integer")
    for value, name in (
        (maximum_track_gap_s, "maximum_track_gap_s"),
        (cannon_duration_tolerance_s, "cannon_duration_tolerance_s"),
        (minimum_absent_tail_s, "minimum_absent_tail_s"),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise MiningManifestError(f"{name} must be finite and positive")
    engine = engine or BattleEngine()
    if type(level_proof_verified) is not bool:
        raise MiningManifestError("level_proof_verified must be boolean")
    if type(source_level) is not int or source_level != engine.ruleset.level:
        if not level_invariant_current_ruleset:
            raise MiningManifestError(
                f"replay source level {source_level!r} does not match ruleset level "
                f"{engine.ruleset.level}"
            )
        if expected_support_tower_hp is None:
            raise MiningManifestError(
                "cross-level interaction evidence requires expected_support_tower_hp"
            )
    elif expected_support_tower_hp is not None:
        raise MiningManifestError(
            "expected_support_tower_hp is only valid for cross-level evidence"
        )
    if type(level_invariant_current_ruleset) is not bool:
        raise MiningManifestError("level_invariant_current_ruleset must be boolean")
    source = Path(cache_path).resolve()
    if not source.is_file():
        raise MiningManifestError(f"replay cache is not a file: {source}")

    tracks, frame_units, first_frame, last_frame, cache_end_video_time, level_confirmed = _replay_cache_interaction_rows(
        source,
        engine=engine,
        source_level=source_level,
        expected_support_tower_hp=expected_support_tower_hp,
        confidence_threshold=threshold,
        contamination_confidence_threshold=contamination_threshold,
    )
    if not level_confirmed and not level_proof_verified:
        raise MiningManifestError(
            f"replay cache has no exact full support-tower HP confirming Level {source_level}"
        )

    candidates: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    # Track-onset candidates are useful to downstream action-window miners but
    # are explicitly not promoted to action truth.  A cache-boundary onset is
    # retained only as a rejection because the original deployment may be
    # outside the analyzed window.
    for key, rows in sorted(tracks.items()):
        first = rows[0]
        action = {
            "mechanic": "track_onset_action_candidate",
            "card_id": key[0],
            "owner": key[1],
            "track_id": key[2],
            "frame_idx": int(first["frame_idx"]),
            "video_time_ms": round(float(first["video_time_s"]) * 1_000),
            "x_mtile": int(first["x_mtile"]),
            "y_mtile": int(first["y_mtile"]),
            "minimum_confidence": min(float(row["confidence"]) for row in rows),
            "truth_promoted": False,
            "method": "replay_cache_track_onset_no_manual_action_label_v1",
        }
        if int(first["frame_idx"]) <= first_frame:
            rejected.append({**action, "reason": "track_begins_at_cache_boundary"})
        else:
            candidates.append(action)

    def local_isolated(
        row: dict[str, object],
        *,
        owner: int | None = None,
        exclude: tuple[str, int, int] | None = None,
    ) -> bool:
        frame_rows = frame_units.get(int(row["frame_idx"]), [])
        for other in frame_rows:
            other_key = (
                str(other["card_id"]),
                int(other["owner"]),
                int(other["track_id"]),
            ) if other["track_id"] is not None else None
            if exclude is not None and other_key == exclude:
                continue
            if owner is not None and int(other["owner"]) != owner:
                continue
            if distance_mtile(
                int(row["x_mtile"]),
                int(row["y_mtile"]),
                int(other["x_mtile"]),
                int(other["y_mtile"]),
            ) < isolation_radius_mtile:
                return False
        return True

    # Bridge topology candidates.  The detector must show both banks and every
    # river-band sample must lie in one declared bridge range; otherwise the
    # apparent crossing is likely a homography or ID-switch artifact.
    river_min = engine.ruleset.arena.river_y_min_mtile
    river_max = engine.ruleset.arena.river_y_max_mtile
    bridge_margin = 250
    for key, rows in sorted(tracks.items()):
        definition = engine.ruleset.card(key[0])
        if definition.kind != "troop" or str(definition.mechanics.get("movement_layer") or "ground") != "ground":
            continue
        for run_index, run in enumerate(
            _split_interaction_track(rows, maximum_track_gap_s=maximum_track_gap_s)
        ):
            base = {
                "mechanic": f"{key[0]}_bridge_path_topology",
                "card_id": key[0],
                "owner": key[1],
                "track_id": key[2],
                "run_index": run_index,
                "frame_start": int(run[0]["frame_idx"]),
                "frame_end": int(run[-1]["frame_idx"]),
                "minimum_confidence": min(float(row["confidence"]) for row in run),
            }
            if len(run) < minimum_track_frames:
                rejected.append({**base, "reason": "insufficient_track_frames"})
                continue
            y_values = [int(row["y_mtile"]) for row in run]
            if min(y_values) > river_min - bridge_margin or max(y_values) < river_max + bridge_margin:
                continue
            bridge_rows = [
                row
                for row in run
                if river_min <= int(row["y_mtile"]) <= river_max
            ]
            bridge_indices = [
                _bridge_index_for_position(engine.ruleset, int(row["x_mtile"]))
                for row in bridge_rows
            ]
            if not bridge_rows or any(index is None for index in bridge_indices):
                rejected.append({**base, "reason": "river_sample_outside_declared_bridge"})
                continue
            counts: dict[int, int] = {}
            for index in bridge_indices:
                assert index is not None
                counts[index] = counts.get(index, 0) + 1
            bridge_index = min(
                counts,
                key=lambda index: (-counts[index], index),
            )
            if any(index != bridge_index for index in bridge_indices):
                rejected.append({**base, "reason": "bridge_identity_ambiguous"})
                continue
            displacement = distance_mtile(
                int(run[0]["x_mtile"]),
                int(run[0]["y_mtile"]),
                int(run[-1]["x_mtile"]),
                int(run[-1]["y_mtile"]),
            )
            if displacement < minimum_bridge_displacement_mtile:
                rejected.append({**base, "reason": "bridge_displacement_too_small"})
                continue
            if not _trajectory_has_consistent_motion(run, tick_us=engine.ruleset.tick_us):
                rejected.append({**base, "reason": "kinematic_motion_quality_failed"})
                continue
            if any(
                not local_isolated(row, exclude=key)
                for row in run
            ):
                rejected.append({**base, "reason": "nearby_detector_contamination"})
                continue
            direction_delta = int(run[-1]["y_mtile"]) - int(run[0]["y_mtile"])
            direction = "toward_enemy" if (
                (key[1] == 0 and direction_delta < 0)
                or (key[1] == 1 and direction_delta > 0)
            ) else "toward_own_side"
            candidates.append(
                {
                    **base,
                    "bridge_index": bridge_index,
                    "direction": direction,
                    "displacement_mtile": displacement,
                    "river_sample_count": len(bridge_rows),
                    "samples": [
                        {
                            "frame_idx": int(row["frame_idx"]),
                            "video_time_ms": round(float(row["video_time_s"]) * 1_000),
                            "x_mtile": int(row["x_mtile"]),
                            "y_mtile": int(row["y_mtile"]),
                        }
                        for row in run
                    ],
                    "truth_promoted": False,
                    "method": "replay_cache_ground_track_bridge_topology_v1",
                }
            )

    # Cannon lifetime/decay candidates.  The action timestamp is deliberately
    # the detector onset hypothesis and is labelled as such in the report.
    cannon = engine.ruleset.card("cannon")
    cannon_max_hp = int(cannon.hitpoints or 1)
    cannon_lifetime_s = int(cannon.lifetime_us or 0) / 1_000_000
    for key, rows in sorted(tracks.items()):
        if key[0] != "cannon":
            continue
        base = {
            "mechanic": "cannon_lifetime_hp_decay",
            "card_id": "cannon",
            "owner": key[1],
            "track_id": key[2],
            "frame_start": int(rows[0]["frame_idx"]),
            "frame_end": int(rows[-1]["frame_idx"]),
            "minimum_confidence": min(float(row["confidence"]) for row in rows),
            "action_time_hypothesis": "detector_track_onset",
        }
        reasons: list[str] = []
        if len(rows) < minimum_track_frames:
            reasons.append("insufficient_track_frames")
        if level_invariant_current_ruleset:
            # HP/lifetime values are not assumed invariant across levels. The
            # same cache may still produce useful topology/action candidates,
            # but its Cannon decay row cannot be compared with Level-11 data.
            reasons.append("cross_level_hp_decay_not_comparable")
        if int(rows[0]["hp"]) * 100 < cannon_max_hp * 95:
            reasons.append("initial_hp_not_full")
        gaps = [
            float(right["video_time_s"]) - float(left["video_time_s"])
            for left, right in zip(rows, rows[1:])
        ]
        if gaps and max(gaps) > maximum_track_gap_s:
            reasons.append("detector_gap")
        if any(not local_isolated(row, owner=1 - key[1], exclude=key) for row in rows):
            reasons.append("enemy_near_cannon")
        sample_interval = sorted(gaps)[len(gaps) // 2] if gaps else maximum_track_gap_s
        observed_duration = float(rows[-1]["video_time_s"]) + sample_interval - float(rows[0]["video_time_s"])
        if abs(observed_duration - cannon_lifetime_s) > cannon_duration_tolerance_s:
            reasons.append("expiry_duration_outside_gate")
        if float(last_frame) < float(rows[-1]["frame_idx"]) or float(last_frame - rows[-1]["frame_idx"]) <= 0:
            reasons.append("no_absence_tail")
        else:
            last_time = float(rows[-1]["video_time_s"])
            # ``frame_idx`` gaps can be large in native replay caches; use the
            # final cache timestamp for the actual absence-tail gate, including
            # frames with no detector matches.
            if cache_end_video_time - last_time < minimum_absent_tail_s:
                reasons.append("no_absence_tail")
        curve_errors: dict[str, int] = {}
        deploy_s = cannon.deploy_time_us / 1_000_000
        for hypothesis, offset in (("placement", 0.0), ("post_deploy", deploy_s)):
            residuals = []
            for row in rows:
                age = max(0.0, float(row["video_time_s"]) - float(rows[0]["video_time_s"]) - offset)
                expected = max(0, round(cannon_max_hp * (1.0 - age / cannon_lifetime_s)))
                residuals.append(abs(int(row["hp"]) - expected) * 1_000 // cannon_max_hp)
            curve_errors[hypothesis] = sum(residuals) // len(residuals)
        best_hypothesis = min(curve_errors, key=curve_errors.get)
        if curve_errors[best_hypothesis] > 150:
            reasons.append("hp_curve_mismatch")
        record = {
            **base,
            "observed_duration_ms": round(observed_duration * 1_000),
            "curve_mae_permille": curve_errors,
            "best_lifetime_start_hypothesis": best_hypothesis,
            "truth_promoted": False,
        }
        if reasons:
            rejected.append({**record, "reason": reasons[0], "rejection_reasons": reasons})
        else:
            candidates.append(
                {
                    **record,
                    "method": "replay_cache_cannon_onset_hp_curve_v1",
                }
            )

    # Hog→Cannon targeting candidates use only observed approach geometry and
    # the local building set.  They do not assert that every approach was a
    # game pull; the report names the observation precisely so a later gold
    # action/path oracle can promote or reject it.
    cannon_tracks = [item for item in tracks if item[0] == "cannon"]
    for cannon_key in sorted(cannon_tracks):
        cannon_rows = tracks[cannon_key]
        if len(cannon_rows) < minimum_track_frames:
            continue
        cannon_position = (
            int(cannon_rows[min(2, len(cannon_rows) - 1)]["x_mtile"]),
            int(cannon_rows[min(2, len(cannon_rows) - 1)]["y_mtile"]),
        )
        cannon_frames = {int(row["frame_idx"]): row for row in cannon_rows}
        for hog_key, hog_rows in sorted(tracks.items()):
            if hog_key[0] != "hog-rider" or hog_key[1] == cannon_key[1]:
                continue
            common = [row for row in hog_rows if int(row["frame_idx"]) in cannon_frames]
            for run_index, run in enumerate(
                _split_interaction_track(common, maximum_track_gap_s=maximum_track_gap_s)
            ):
                base = {
                    "mechanic": "hog_cannon_targeting_candidate",
                    "hog_track_id": hog_key[2],
                    "hog_owner": hog_key[1],
                    "cannon_track_id": cannon_key[2],
                    "cannon_owner": cannon_key[1],
                    "run_index": run_index,
                    "frame_start": int(run[0]["frame_idx"]) if run else None,
                    "frame_end": int(run[-1]["frame_idx"]) if run else None,
                }
                if len(run) < minimum_track_frames:
                    rejected.append({**base, "reason": "insufficient_overlapping_hog_track"})
                    continue
                other_building_seen = False
                for row in run:
                    for visible in frame_units.get(int(row["frame_idx"]), []):
                        if visible["card_id"] == "cannon" and int(visible["track_id"] or -1) == cannon_key[2]:
                            continue
                        if visible["kind"] != "building" or int(visible["owner"]) != cannon_key[1]:
                            continue
                        if distance_mtile(
                            int(row["x_mtile"]), int(row["y_mtile"]),
                            int(visible["x_mtile"]), int(visible["y_mtile"]),
                        ) <= isolation_radius_mtile:
                            other_building_seen = True
                            break
                    if other_building_seen:
                        break
                if other_building_seen:
                    rejected.append({**base, "reason": "multiple_local_target_buildings"})
                    continue
                distances = [
                    distance_mtile(int(row["x_mtile"]), int(row["y_mtile"]), *cannon_position)
                    for row in run
                ]
                approach = distances[0] - min(distances)
                contact_gate = (
                    int(engine.ruleset.card("hog-rider").collision_radius_mtile or 0)
                    + int(cannon.collision_radius_mtile or 0)
                    + int(engine.ruleset.card("hog-rider").range_mtile or 0)
                    + 500
                )
                if approach < minimum_pull_approach_mtile or min(distances) > contact_gate:
                    rejected.append(
                        {
                            **base,
                            "reason": "no_conclusive_approach_to_cannon",
                            "approach_mtile": approach,
                            "minimum_distance_mtile": min(distances),
                        }
                    )
                    continue
                candidates.append(
                    {
                        **base,
                        "cannon_x_mtile": cannon_position[0],
                        "cannon_y_mtile": cannon_position[1],
                        "approach_mtile": approach,
                        "minimum_distance_mtile": min(distances),
                        "target_hypothesis": "cannon",
                        "classification": "approach_to_only_local_building",
                        "minimum_confidence": min(
                            [float(row["confidence"]) for row in run]
                            + [float(row["confidence"]) for row in cannon_rows]
                        ),
                        "truth_promoted": False,
                        "method": "replay_cache_hog_cannon_action_free_approach_v1",
                    }
                )

    candidates.sort(
        key=lambda row: (
            str(row.get("mechanic", "")),
            str(row.get("card_id", row.get("hog_track_id", ""))),
            int(row.get("frame_start", row.get("frame_idx", 0)) or 0),
            int(row.get("track_id", row.get("hog_track_id", 0)) or 0),
        )
    )
    rejected.sort(
        key=lambda row: (
            str(row.get("mechanic", "")),
            str(row.get("reason", "")),
            int(row.get("frame_start", row.get("frame_idx", 0)) or 0),
        )
    )
    cache_hash = _file_sha256(source)
    return {
        "schema_version": 1,
        "kind": "autonomous_interaction_candidate_report",
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "source_level": source_level,
        "level_invariant_current_ruleset": level_invariant_current_ruleset,
        "expected_support_tower_hp": expected_support_tower_hp,
        "level_proof_verified": level_proof_verified,
        "cache_hash": cache_hash,
        "cache_path": str(source),
        "automation": {
            "action_inference": "track_onset_only",
            "truth_promotion": False,
            "simulator_independent_selection": True,
            "manual_labels_required": False,
        },
        "mechanics": _interaction_report_mechanics(candidates, rejected),
        "candidates": candidates,
        "rejected": rejected,
    }


def discover_replay_cache_interactions_batch(
    cache_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    source_level: int,
    engine: BattleEngine | None = None,
    level_proof_paths: list[str | Path] | tuple[str | Path, ...] = (),
    **kwargs: object,
) -> dict[str, object]:
    """Run the action-free interaction miner over many sealed caches.

    A bad or incomplete cache is retained as a source failure rather than
    aborting the rest of a high-scale run.  The aggregate is deterministic in
    canonical path order and carries every per-cache hash for provenance.
    """

    engine = engine or BattleEngine()
    reports: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    level_proof_failures: list[dict[str, str]] = []

    def source_video_key(path: str | Path) -> str:
        """Return the stable source-video key shared by full/window caches."""

        parent = Path(path).resolve().parent.parent.name
        return parent.split(":action-window:", 1)[0]

    proof_keys: set[str] = set()
    proof_records: list[dict[str, str]] = []
    for raw_proof in sorted({str(Path(path).resolve()) for path in level_proof_paths}):
        proof_path = Path(raw_proof)
        try:
            if not proof_path.is_file():
                raise MiningManifestError(f"replay cache is not a file: {proof_path}")
            _, _, _, _, _, level_confirmed = _replay_cache_interaction_rows(
                proof_path,
                engine=engine,
                source_level=source_level,
                expected_support_tower_hp=kwargs.get("expected_support_tower_hp"),
                confidence_threshold=1.0,
                contamination_confidence_threshold=0.0,
            )
            if not level_confirmed:
                raise MiningManifestError(
                    f"level proof cache has no exact full support-tower HP confirming Level {source_level}"
                )
            proof_key = source_video_key(proof_path)
            proof_keys.add(proof_key)
            proof_records.append(
                {
                    "cache_path": raw_proof,
                    "cache_hash": _file_sha256(proof_path),
                    "source_video_key": proof_key,
                }
            )
        except (OSError, RuntimeError, ValueError) as error:
            level_proof_failures.append({"cache_path": raw_proof, "reason": str(error)})

    for raw_path in sorted({str(Path(path).resolve()) for path in cache_paths}):
        try:
            reports.append(
                discover_replay_cache_interactions(
                    raw_path,
                    source_level=source_level,
                    engine=engine,
                    level_proof_verified=source_video_key(raw_path) in proof_keys,
                    **kwargs,
                )
            )
        except (OSError, RuntimeError, ValueError) as error:
            failures.append({"cache_path": raw_path, "reason": str(error)})
    candidates: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    mechanics: dict[str, dict[str, int]] = {}
    for report in reports:
        cache_path = str(report["cache_path"])
        cache_hash = str(report["cache_hash"])
        for row in report["candidates"]:  # type: ignore[union-attr]
            candidates.append({**row, "cache_path": cache_path, "cache_hash": cache_hash})
        for row in report["rejected"]:  # type: ignore[union-attr]
            rejected.append({**row, "cache_path": cache_path, "cache_hash": cache_hash})
        for mechanic, counts in report["mechanics"].items():  # type: ignore[union-attr]
            destination = mechanics.setdefault(
                str(mechanic), {"candidate_count": 0, "rejected_count": 0}
            )
            destination["candidate_count"] += int(counts["candidate_count"])
            destination["rejected_count"] += int(counts["rejected_count"])
    candidates.sort(
        key=lambda row: (
            str(row.get("cache_path", "")),
            str(row.get("mechanic", "")),
            int(row.get("frame_start", row.get("frame_idx", 0)) or 0),
        )
    )
    rejected.sort(
        key=lambda row: (
            str(row.get("cache_path", "")),
            str(row.get("mechanic", "")),
            int(row.get("frame_start", row.get("frame_idx", 0)) or 0),
            str(row.get("reason", "")),
        )
    )
    return {
        "schema_version": 1,
        "kind": "autonomous_interaction_candidate_batch",
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "source_level": source_level,
        "level_proof_sources": proof_records,
        "level_proof_failures": level_proof_failures,
        "automation": {
            "action_inference": "track_onset_only",
            "truth_promotion": False,
            "simulator_independent_selection": True,
            "manual_labels_required": False,
            "level_proof_inheritance": (
                "same-source-video-key-only" if proof_records else "none"
            ),
        },
        "source_count": len(reports),
        "failed_source_count": len(failures),
        "sources": [
            {
                "cache_path": str(report["cache_path"]),
                "cache_hash": str(report["cache_hash"]),
                "candidate_count": len(report["candidates"]),  # type: ignore[arg-type]
                "rejected_count": len(report["rejected"]),  # type: ignore[arg-type]
            }
            for report in reports
        ],
        "cache_hashes": sorted(
            str(report["cache_hash"])
            for report in reports
            if report.get("cache_hash")
        ),
        "source_failures": failures,
        "mechanics": {key: mechanics[key] for key in sorted(mechanics)},
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "candidates": candidates,
        "rejected": rejected,
    }


def _interaction_source_group(cache_path: object, cache_hash: object) -> tuple[str, str]:
    """Return a stable source group and HUD label for an interaction cache.

    Extractor jobs put the HUD profile directly below the video/window ID:
    ``.../<video-id>/standard/replay-cache.json``.  The group deliberately
    ignores the extractor root, allowing resumed jobs to reconcile.  Unknown
    layouts remain pairable only when the caller supplied an explicit cache
    path/hash; they are never silently assigned to a HUD profile.
    """

    path_value = str(cache_path or "")
    if path_value:
        path = Path(path_value)
        variant = path.parent.name
        if variant in {"standard", "alternative"}:
            return path.parent.parent.name, variant
        return path.stem, "unknown"
    return str(cache_hash or "unknown-source"), "unknown"


def _interaction_candidate_anchor(row: Mapping[str, object]) -> tuple[int | None, int | None, int | None]:
    """Extract an onset/time/position anchor without consulting simulation."""

    time_ms: int | None = None
    for key in ("video_time_ms", "action_video_time_ms", "frame_start"):
        value = row.get(key)
        if type(value) is int:
            time_ms = value
            break
    x_value = row.get("x_mtile", row.get("cannon_x_mtile"))
    y_value = row.get("y_mtile", row.get("cannon_y_mtile"))
    x = int(x_value) if type(x_value) is int else None
    y = int(y_value) if type(y_value) is int else None
    if x is None or y is None:
        samples = row.get("samples")
        if isinstance(samples, list) and samples and isinstance(samples[0], Mapping):
            sample = samples[0]
            x = int(sample["x_mtile"]) if type(sample.get("x_mtile")) is int else None
            y = int(sample["y_mtile"]) if type(sample.get("y_mtile")) is int else None
            if time_ms is None and type(sample.get("video_time_ms")) is int:
                time_ms = int(sample["video_time_ms"])
    return time_ms, x, y


def _interaction_candidate_owner(row: Mapping[str, object]) -> int | None:
    value = row.get("owner") if "owner" in row else row.get("hog_owner")
    return int(value) if type(value) is int else None


def merge_replay_interaction_reports(
    report_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    engine: BattleEngine | None = None,
    onset_tolerance_ms: int = 250,
    position_tolerance_mtile: int = 1_500,
    require_both_hud: bool = False,
) -> dict[str, object]:
    """Reconcile action-free interaction candidates from both HUD profiles.

    Standard and alternative HUD caches are two renderings of the same source
    video, not independent truth.  A pair is therefore emitted only when the
    source group, mechanic/card/owner, onset, and (when available) position
    agree within conservative tolerances.  Unpaired rows stay in ``rejected``
    and every output row remains ``truth_promoted: false``.
    """

    if type(onset_tolerance_ms) is not int or onset_tolerance_ms < 0:
        raise MiningManifestError("onset_tolerance_ms must be a non-negative integer")
    if type(position_tolerance_mtile) is not int or position_tolerance_mtile < 0:
        raise MiningManifestError("position_tolerance_mtile must be a non-negative integer")
    if type(require_both_hud) is not bool:
        raise MiningManifestError("require_both_hud must be boolean")
    engine = engine or BattleEngine()
    canonical_paths = sorted({str(Path(path).resolve()) for path in report_paths})
    if not canonical_paths:
        raise MiningManifestError("at least one interaction report is required")

    allowed_kinds = {
        "autonomous_interaction_candidate_report",
        "autonomous_interaction_candidate_batch",
    }
    expected_identity = (engine.ruleset.ruleset_id, engine.ruleset.content_hash, ENGINE_VERSION)
    rows_by_group: dict[str, list[dict[str, object]]] = {}
    cache_hashes: set[str] = set()
    source_rows: list[dict[str, object]] = []
    source_failures: list[dict[str, str]] = []
    source_levels: set[int] = set()
    for report_path in canonical_paths:
        path = Path(report_path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            source_failures.append({"report_path": report_path, "reason": str(error)})
            continue
        if not isinstance(raw, dict):
            source_failures.append({"report_path": report_path, "reason": "report must be an object"})
            continue
        if raw.get("kind") not in allowed_kinds:
            source_failures.append({
                "report_path": report_path,
                "reason": f"unsupported report kind {raw.get('kind')!r}",
            })
            continue
        identity = (raw.get("ruleset_id"), raw.get("ruleset_hash"), raw.get("engine_version"))
        if identity != expected_identity:
            source_failures.append({
                "report_path": report_path,
                "reason": f"report identity {identity!r} does not match {expected_identity!r}",
            })
            continue
        if type(raw.get("source_level")) is int:
            source_levels.add(int(raw["source_level"]))
        inherited_failures = raw.get("source_failures", [])
        if isinstance(inherited_failures, list):
            for failure in inherited_failures:
                if isinstance(failure, Mapping):
                    source_failures.append({
                        "report_path": report_path,
                        "cache_path": str(failure.get("cache_path") or ""),
                        "reason": str(failure.get("reason") or "upstream source failure"),
                    })
        candidates = raw.get("candidates", [])
        if not isinstance(candidates, list):
            source_failures.append({"report_path": report_path, "reason": "candidates must be an array"})
            continue
        report_sources = raw.get("sources")
        if isinstance(report_sources, list) and report_sources:
            source_specs = [item for item in report_sources if isinstance(item, Mapping)]
        else:
            source_specs = [raw]
        source_by_hash = {
            str(item.get("cache_hash")): item
            for item in source_specs
            if item.get("cache_hash")
        }
        for item in source_specs:
            cache_hash = str(item.get("cache_hash") or "")
            cache_path = str(item.get("cache_path") or "")
            source_group, hud_variant = _interaction_source_group(cache_path, cache_hash)
            if cache_hash:
                cache_hashes.add(cache_hash)
            source_rows.append({
                "source_group": source_group,
                "hud_variant": hud_variant,
                "cache_path": cache_path or None,
                "cache_hash": cache_hash or None,
                "report_path": report_path,
            })
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                source_failures.append({
                    "report_path": report_path,
                    "reason": f"candidate {index} must be an object",
                })
                continue
            cache_hash = str(candidate.get("cache_hash") or raw.get("cache_hash") or "")
            source_spec = source_by_hash.get(cache_hash, {})
            cache_path = str(candidate.get("cache_path") or source_spec.get("cache_path") or raw.get("cache_path") or "")
            source_group, hud_variant = _interaction_source_group(cache_path, cache_hash)
            if cache_hash:
                cache_hashes.add(cache_hash)
            enriched = {
                **dict(candidate),
                "source_group": source_group,
                "hud_variant": hud_variant,
                "cache_path": cache_path or None,
                "cache_hash": cache_hash or None,
                "report_path": report_path,
            }
            rows_by_group.setdefault(source_group, []).append(enriched)

    paired: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    hud_groups: list[dict[str, object]] = []
    for source_group in sorted(rows_by_group):
        rows = rows_by_group[source_group]
        variants = sorted({str(row["hud_variant"]) for row in rows})
        standard = [row for row in rows if row["hud_variant"] == "standard"]
        alternative = [row for row in rows if row["hud_variant"] == "alternative"]
        used_alternative: set[int] = set()
        group_pairs = 0
        for left in sorted(standard, key=lambda row: (str(row.get("mechanic", "")), _interaction_candidate_anchor(row)[0] or -1, int(row.get("track_id", 0) or 0))):
            left_time, left_x, left_y = _interaction_candidate_anchor(left)
            options: list[tuple[int, int, int, dict[str, object]]] = []
            for index, right in enumerate(alternative):
                if index in used_alternative:
                    continue
                if str(left.get("mechanic")) != str(right.get("mechanic")):
                    continue
                if str(left.get("card_id", "")) != str(right.get("card_id", "")):
                    continue
                if _interaction_candidate_owner(left) != _interaction_candidate_owner(right):
                    continue
                right_time, right_x, right_y = _interaction_candidate_anchor(right)
                time_delta = abs(left_time - right_time) if left_time is not None and right_time is not None else 0
                if time_delta > onset_tolerance_ms:
                    continue
                position_delta = 0
                if left_x is not None and left_y is not None and right_x is not None and right_y is not None:
                    position_delta = distance_mtile(left_x, left_y, right_x, right_y)
                    if position_delta > position_tolerance_mtile:
                        continue
                options.append((time_delta, position_delta, index, right))
            if not options:
                rejected.append({**left, "reason": "missing_agreeing_alternative_hud"})
                continue
            time_delta, position_delta, right_index, right = min(options, key=lambda item: (item[0], item[1], item[2]))
            used_alternative.add(right_index)
            group_pairs += 1
            paired.append({
                "mechanic": left.get("mechanic"),
                "card_id": left.get("card_id"),
                "owner": _interaction_candidate_owner(left),
                "source_group": source_group,
                "hud_variants": ["standard", "alternative"],
                "observations": [
                    {key: value for key, value in left.items() if key not in {"source_group", "hud_variant", "report_path"}},
                    {key: value for key, value in right.items() if key not in {"source_group", "hud_variant", "report_path"}},
                ],
                "time_delta_ms": time_delta,
                "position_delta_mtile": position_delta,
                "minimum_confidence": min(float(left.get("minimum_confidence", 0.0)), float(right.get("minimum_confidence", 0.0))),
                "truth_promoted": False,
                "method": "dual_hud_candidate_agreement_v1",
            })
        for index, right in enumerate(alternative):
            if index not in used_alternative:
                rejected.append({**right, "reason": "missing_agreeing_standard_hud"})
        hud_groups.append({
            "source_group": source_group,
            "hud_variants": variants,
            "paired_count": group_pairs,
            "standard_candidate_count": len(standard),
            "alternative_candidate_count": len(alternative),
            "both_hud_present": bool(standard and alternative),
        })

    paired.sort(key=lambda row: (str(row.get("source_group", "")), str(row.get("mechanic", "")), int(row.get("time_delta_ms", 0))))
    rejected.sort(key=lambda row: (str(row.get("source_group", "")), str(row.get("mechanic", "")), str(row.get("reason", ""))))
    if len(source_levels) > 1:
        source_failures.append({
            "report_path": "<aggregate>",
            "reason": "source-level mismatch across interaction reports: "
            + ", ".join(str(level) for level in sorted(source_levels)),
        })
    if require_both_hud and not any(bool(row["both_hud_present"]) for row in hud_groups):
        source_failures.append({"report_path": "<aggregate>", "reason": "no source contains both HUD variants"})
    mechanics = _interaction_report_mechanics(paired, rejected)
    return {
        "schema_version": 1,
        "kind": "autonomous_interaction_dual_hud_report",
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "automation": {
            "hud_reconciliation": "standard_alternative_same_source_v1",
            "truth_promotion": False,
            "independent_truth": False,
            "manual_labels_required": False,
        },
        "onset_tolerance_ms": onset_tolerance_ms,
        "position_tolerance_mtile": position_tolerance_mtile,
        "source_levels": sorted(source_levels),
        "source_count": len(source_rows),
        "failed_source_count": len(source_failures),
        "sources": sorted(source_rows, key=lambda row: (str(row["source_group"]), str(row["hud_variant"]), str(row.get("cache_hash") or ""))),
        "cache_hashes": sorted(cache_hashes),
        "hud_groups": hud_groups,
        "mechanics": mechanics,
        "candidate_count": len(paired),
        "rejected_count": len(rejected),
        "unpaired_count": len(rejected),
        "source_failures": source_failures,
        "candidates": paired,
        "rejected": rejected,
    }


def compile_replay_cache_movement(
    cache_path: str | Path,
    *,
    corpus_id: str,
    group_id: str,
    source_level: int,
    engine: BattleEngine | None = None,
    confidence_threshold: float = 0.98,
    minimum_track_frames: int = 20,
    minimum_displacement_mtile: int = 750,
    isolation_radius_mtile: int = 3_500,
    contamination_confidence_threshold: float = 0.25,
    minimum_speed_ratio_permille: int = 500,
    maximum_speed_ratio_permille: int = 1_500,
    split_salt: str = "hog-cycle-sim-v1",
    evidence_split: str | None = None,
    level_invariant_current_ruleset: bool = False,
    expected_support_tower_hp: int | None = None,
    use_expected_speed_gate: bool = True,
) -> MiningResult:
    """Mine isolated, unoccluded movement runs from an existing replay cache.

    Each supported high-confidence troop is considered independently. A run
    is retained only while no other detected unit is within the configured
    local isolation radius. This permits clean lane movement elsewhere in a
    busy match without pretending the entire arena must be empty. Ambiguous
    local frames terminate a run instead of being sent to a person.
    """

    if type(minimum_track_frames) is not int or minimum_track_frames < 2:
        raise MiningManifestError("minimum_track_frames must be at least 2")
    if type(minimum_displacement_mtile) is not int or minimum_displacement_mtile < 1:
        raise MiningManifestError("minimum_displacement_mtile must be positive")
    if type(isolation_radius_mtile) is not int or isolation_radius_mtile < 1:
        raise MiningManifestError("isolation_radius_mtile must be positive")
    if (
        type(minimum_speed_ratio_permille) is not int
        or type(maximum_speed_ratio_permille) is not int
        or minimum_speed_ratio_permille < 1
        or maximum_speed_ratio_permille <= minimum_speed_ratio_permille
    ):
        raise MiningManifestError(
            "movement speed-ratio gate must be positive and strictly increasing"
        )
    threshold = _confidence(confidence_threshold, "confidence_threshold")
    contamination_threshold = _confidence(
        contamination_confidence_threshold,
        "contamination_confidence_threshold",
    )
    if contamination_threshold >= threshold:
        raise MiningManifestError(
            "contamination confidence threshold must be below the candidate threshold"
        )
    if evidence_split is not None and evidence_split not in _SPLITS:
        raise MiningManifestError(f"evidence_split must be one of {_SPLITS}")
    if evidence_split in {"validation", "heldout"} and use_expected_speed_gate:
        raise MiningManifestError(
            "validation/heldout movement cannot use the expected-speed selection gate; "
            "select the kinematic-only gate"
        )
    engine = engine or BattleEngine()
    if type(source_level) is not int or source_level < 1:
        raise MiningManifestError("source_level must be a positive integer")
    cross_level = source_level != engine.ruleset.level
    if cross_level and not level_invariant_current_ruleset:
        raise MiningManifestError(
            f"replay source level {source_level!r} does not match ruleset level "
            f"{engine.ruleset.level}"
        )
    if cross_level and expected_support_tower_hp is None:
        raise MiningManifestError(
            "cross-level movement evidence requires expected_support_tower_hp"
        )
    if level_invariant_current_ruleset and type(level_invariant_current_ruleset) is not bool:
        raise MiningManifestError("level_invariant_current_ruleset must be boolean")
    source = Path(cache_path).resolve()
    if not source.is_file():
        raise MiningManifestError(f"replay cache is not a file: {source}")

    from cr_bot.replay.cache import ReplayCacheReader
    from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

    runs: list[list[dict[str, object]]] = []
    active_runs: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    level_confirmed = False

    def finish_run(key: tuple[int, str, str]) -> None:
        current = active_runs.pop(key, [])
        if len(current) >= minimum_track_frames:
            runs.append(current)

    for replay_frame in ReplayCacheReader(source):
        analysis = replay_frame.analysis
        level_confirmed = _validate_replay_source_level(
            analysis,
            source_level=source_level,
            engine=engine,
            expected_support_tower_hp=expected_support_tower_hp,
        ) or level_confirmed
        visible_units: list[tuple[object, str | None, int, int]] = []
        candidates: list[tuple[object, str, int, int]] = []
        for match in analysis.matches:
            detection = match.troop
            card_id = DIRECT_UNIT_TO_CARD.get(detection.class_name)
            if detection.team not in {"ally", "enemy"}:
                continue
            detection_confidence = float(detection.confidence)
            if detection_confidence < contamination_threshold:
                continue
            definition = (
                engine.ruleset.card(card_id)
                if card_id in engine.ruleset.interaction_set
                else None
            )
            x_mtile, y_mtile = _detection_world_position(
                detection,
                analysis.arena_px,
                ground_anchor=definition is None or definition.kind != "building",
                center_margin_mtile=(
                    int(definition.collision_radius_mtile or 0)
                    if definition is not None
                    else 0
                ),
            )
            visible_units.append((detection, card_id, x_mtile, y_mtile))
            if (
                detection.track_id is not None
                and definition is not None
                and definition.kind == "troop"
                and detection_confidence >= threshold
            ):
                candidates.append((detection, card_id, x_mtile, y_mtile))

        accepted_keys: set[tuple[int, str, str]] = set()
        if analysis.total_remaining_s is None:
            for key in tuple(active_runs):
                finish_run(key)
            continue
        elapsed_us = round((300.0 - float(analysis.total_remaining_s)) * 1_000_000)
        if elapsed_us < 0:
            for key in tuple(active_runs):
                finish_run(key)
            continue
        tick = max(0, round(elapsed_us / engine.ruleset.tick_us))
        for detection, card_id, x_mtile, y_mtile in candidates:
            key = (int(detection.track_id), card_id, str(detection.team))
            definition = engine.ruleset.card(card_id)
            effective_isolation = max(
                isolation_radius_mtile,
                int(definition.sight_range_mtile or 0),
            )
            contaminated = any(
                other is not detection
                and distance_mtile(x_mtile, y_mtile, other_x, other_y)
                < effective_isolation
                for other, _, other_x, other_y in visible_units
            )
            if contaminated:
                finish_run(key)
                continue
            current = active_runs.get(key)
            if current and (
                float(replay_frame.video_time_s) <= float(current[-1]["video_time_s"])
                or float(replay_frame.video_time_s) - float(current[-1]["video_time_s"]) > 0.25
            ):
                finish_run(key)
                current = None
            if current is None:
                current = []
                active_runs[key] = current
            maximum_hp = int(definition.hitpoints or 1)
            current.append(
                {
                    "frame_idx": int(replay_frame.frame_idx),
                    "video_time_s": float(replay_frame.video_time_s),
                    "tick": tick,
                    "elapsed_us": tick * engine.ruleset.tick_us,
                    "x_mtile": x_mtile,
                    "y_mtile": y_mtile,
                    "hp": _observed_hp(detection, maximum_hp),
                    "confidence": float(detection.confidence),
                    "track_id": str(detection.track_id),
                    "card_id": card_id,
                    "owner": 0 if detection.team == "ally" else 1,
                    "overtime": bool(analysis.overtime),
                }
            )
            accepted_keys.add(key)
        for key in tuple(active_runs):
            if key not in accepted_keys:
                finish_run(key)
    for key in tuple(active_runs):
        finish_run(key)
    if not level_confirmed:
        raise MiningManifestError(
            f"replay cache has no exact full support-tower HP confirming Level "
            f"{source_level}"
        )

    clips: list[dict[str, object]] = []
    cache_hash = _file_sha256(source)
    for index, run in enumerate(runs):
        # The recognized match clock is commonly quantized to whole seconds.
        # Preserve its first absolute tick only as a state anchor, then use the
        # replay's monotonic video timestamps for sub-second trajectory time.
        first_tick = int(run[0]["tick"])
        first_video_time = float(run[0]["video_time_s"])
        for sample in run:
            relative_us = round(
                (float(sample["video_time_s"]) - first_video_time) * 1_000_000
            )
            sample["tick"] = first_tick + round(relative_us / engine.ruleset.tick_us)
            sample["elapsed_us"] = int(sample["tick"]) * engine.ruleset.tick_us
        # Several video frames can map to one physics tick. Keep the highest
        # confidence observation, breaking ties by earliest frame.
        by_tick: dict[int, dict[str, object]] = {}
        for sample in run:
            tick = int(sample["tick"])
            prior = by_tick.get(tick)
            if prior is None or float(sample["confidence"]) > float(prior["confidence"]):
                by_tick[tick] = sample
        samples = [by_tick[tick] for tick in sorted(by_tick)]
        if len(samples) < 2:
            continue
        displacement = max(
            distance_mtile(
                int(samples[0]["x_mtile"]),
                int(samples[0]["y_mtile"]),
                int(item["x_mtile"]),
                int(item["y_mtile"]),
            )
            for item in samples[1:]
        )
        if displacement < minimum_displacement_mtile:
            continue
        first = samples[0]
        definition = engine.ruleset.card(str(first["card_id"]))
        duration_us = (
            int(samples[-1]["tick"]) - int(samples[0]["tick"])
        ) * engine.ruleset.tick_us
        expected_speed = int(definition.move_speed_mtile_per_s or 0)
        if duration_us <= 0 or expected_speed <= 0:
            continue
        speed_ratio_permille = _movement_speed_ratio_permille(
            displacement,
            duration_us,
            expected_speed,
        )
        if use_expected_speed_gate and not (
            minimum_speed_ratio_permille <= speed_ratio_permille <= maximum_speed_ratio_permille
        ):
            # Very slow runs are usually attacks, collision, or an unobserved
            # status spell; very fast runs are detector/track discontinuities.
            # Neither is an isolated base-movement oracle.
            continue
        if not use_expected_speed_gate and not _trajectory_has_consistent_motion(
            samples,
            tick_us=engine.ruleset.tick_us,
        ):
            continue
        movement_mechanic = _movement_mechanic(
            str(first["card_id"]),
            samples,
            engine.ruleset,
        )
        # A stable local track proves displacement speed without proving what
        # the unit was targeting outside the detector's visible neighborhood.
        # Replaying x/y against our automatically chosen Princess Tower would
        # turn an unseen defender into a simulator pathfinding failure. Bridge
        # occupancy is the exception: it supplies an observable topological
        # constraint independent of the eventual target, so retain its
        # trajectory samples for bridge-path fidelity.
        trajectory_samples = (
            samples if movement_mechanic.endswith("_isolated_bridge_path") else []
        )
        endpoint_displacement = distance_mtile(
            int(samples[0]["x_mtile"]),
            int(samples[0]["y_mtile"]),
            int(samples[-1]["x_mtile"]),
            int(samples[-1]["y_mtile"]),
        )
        clip_id = f"{source.stem}:movement:{index:06d}"
        clips.append(
            {
                "clip_id": clip_id,
                "group_id": group_id,
                **({"split": evidence_split} if evidence_split is not None else {}),
                "media_hash": cache_hash,
                "frame_start": int(run[0]["frame_idx"]),
                "frame_end": int(run[-1]["frame_idx"]),
                "method": (
                    "replay_cache_locally_isolated_track_homography_v2:"
                    f"declared_level_{source_level}_support_hp_conflict_checked:"
                    + (
                        "explicit_current_ruleset_level_invariant_movement:"
                        if cross_level
                        else ""
                    )
                    + "base_speed_consistency_gate:"
                    + (
                        "expected_speed_selection_calibration_only:"
                        if use_expected_speed_gate
                        else "target_independent_kinematic_selection:"
                    )
                    + f"confidence_{round(threshold * 1_000)}permille:"
                    f"contamination_confidence_"
                    f"{round(contamination_threshold * 1_000)}permille:"
                    f"minimum_track_frames_{minimum_track_frames}:"
                    f"minimum_displacement_{minimum_displacement_mtile}mtile:"
                    f"isolation_{isolation_radius_mtile}mtile:"
                    f"speed_ratio_{minimum_speed_ratio_permille}_to_"
                    f"{maximum_speed_ratio_permille}permille:"
                    "maximum_gap_250ms"
                ),
                "confidence": min(float(item["confidence"]) for item in run),
                "seed": int.from_bytes(
                    hashlib.sha256(clip_id.encode("utf-8")).digest()[:8], "big"
                ),
                "initial": {
                    "tick": int(first["tick"]),
                    "elapsed_us": int(first["elapsed_us"]),
                    "phase": "overtime" if first["overtime"] else "regulation",
                    "towers": [],
                    "entities": [
                        {
                            "track_id": str(first["track_id"]),
                            "card_id": str(first["card_id"]),
                            "owner": int(first["owner"]),
                            "x_mtile": int(first["x_mtile"]),
                            "y_mtile": int(first["y_mtile"]),
                            "hp": int(first["hp"]),
                            # The clip begins at an already observed trajectory
                            # point, not at the original card-play timestamp.
                            "deploy_remaining_us": 0,
                            "confidence": float(first["confidence"]),
                        }
                    ],
                },
                "tracks": [
                    {
                        "track_id": str(first["track_id"]),
                        "mechanic": movement_mechanic,
                        "confidence": min(float(item["confidence"]) for item in run),
                        "displacement_speed": {
                            "start_tick": int(samples[0]["tick"]),
                            "end_tick": int(samples[-1]["tick"]),
                            "observed_mtile_per_s": (
                                endpoint_displacement * 1_000_000 + duration_us // 2
                            ) // duration_us,
                            "tolerance_mtile_per_s": max(120, expected_speed // 10),
                            "compare_to_card_base_speed": True,
                        },
                        "samples": [
                            {
                                "tick": int(item["tick"]),
                                "x_mtile": int(item["x_mtile"]),
                                "y_mtile": int(item["y_mtile"]),
                                "confidence": float(item["confidence"]),
                            }
                            for item in trajectory_samples
                        ],
                    }
                ],
            }
        )
    if not clips:
        raise MiningManifestError("no isolated high-confidence replay tracks met the gate")
    return compile_observation_manifest(
        {
            "schema_version": MINING_SCHEMA_VERSION,
            "corpus_id": corpus_id,
            "split_salt": split_salt,
            "confidence_threshold": threshold,
            "clips": clips,
        },
        engine=engine,
    )


def discover_replay_cache_cannon_lifetimes(
    cache_path: str | Path,
    *,
    ground_truth_path: str | Path | None = None,
    source_level: int,
    engine: BattleEngine | None = None,
    confidence_threshold: float = 0.90,
    contamination_radius_mtile: int = 10_000,
    maximum_track_gap_s: float = 0.25,
    minimum_absent_tail_s: float = 0.5,
    duration_tolerance_s: float = 3.0,
    maximum_curve_mae_permille: int = 150,
) -> dict[str, object]:
    """Discover undamaged Cannon expiry candidates without promoting truth.

    Tracks must be action-anchored, continuous, Level-11-confirmed, locally
    free of enemy detections, followed by an observed absence tail, and close
    to one of the two explicit lifetime-start hypotheses. Without an action
    file, a detector track onset is used only as a conservative action
    hypothesis; cache-boundary onsets are rejected and the inferred source is
    recorded. The report retains both curve errors so controlled footage can
    resolve whether decay begins at placement or after deployment. An empty
    candidate list is a valid and preferable result when footage is ambiguous.
    """

    threshold = _confidence(confidence_threshold, "confidence_threshold")
    if type(contamination_radius_mtile) is not int or contamination_radius_mtile < 1:
        raise MiningManifestError("contamination_radius_mtile must be positive")
    for value, name in (
        (maximum_track_gap_s, "maximum_track_gap_s"),
        (minimum_absent_tail_s, "minimum_absent_tail_s"),
        (duration_tolerance_s, "duration_tolerance_s"),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise MiningManifestError(f"{name} must be finite and positive")
    if type(maximum_curve_mae_permille) is not int or not (0 < maximum_curve_mae_permille < 1_000):
        raise MiningManifestError("maximum_curve_mae_permille must be between 1 and 999")
    engine = engine or BattleEngine()
    if type(source_level) is not int or source_level != engine.ruleset.level:
        raise MiningManifestError(
            f"replay source level {source_level!r} does not match ruleset level "
            f"{engine.ruleset.level}"
        )
    source = Path(cache_path).resolve()
    if not source.is_file():
        raise MiningManifestError("replay cache must be a file")
    truth_source = Path(ground_truth_path).resolve() if ground_truth_path is not None else None
    label_fps: float | None = None
    actions: list[dict[str, object]] = []
    action_source = "inferred_track_onset"
    if truth_source is not None:
        if not truth_source.is_file():
            raise MiningManifestError("ground-truth action file must be a file")
        try:
            truth = json.loads(truth_source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MiningManifestError(f"cannot read ground-truth actions: {error}") from error
        events = truth.get("events")
        if not isinstance(events, list):
            raise MiningManifestError("ground-truth file must contain an events array")
        label_fps = _label_fps(truth)
        actions = [
            {
                "event_index": index,
                "owner": 0 if event.get("side") == "own" else 1,
                "frame_idx": _integer(event.get("frame_index"), "event.frame_index"),
            }
            for index, event in enumerate(events)
            if isinstance(event, dict)
            and event.get("card") == "cannon"
            and event.get("side") in {"own", "enemy"}
        ]
        if not actions:
            raise MiningManifestError("ground-truth file contains no Cannon actions")
        action_source = "curated_action_file"

    from collections import defaultdict
    from cr_bot.replay.cache import ReplayCacheReader
    from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

    tracks: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    frame_times: dict[int, float] = {}
    level_confirmed = False
    final_video_time = 0.0
    cannon = engine.ruleset.card("cannon")
    maximum_hp = int(cannon.hitpoints or 1)
    for replay_frame in ReplayCacheReader(source):
        analysis = replay_frame.analysis
        frame_idx = int(replay_frame.frame_idx)
        video_time = float(replay_frame.video_time_s)
        frame_times[frame_idx] = video_time
        final_video_time = max(final_video_time, video_time)
        level_confirmed = _validate_replay_source_level(
            analysis,
            source_level=source_level,
            engine=engine,
        ) or level_confirmed
        visible: list[tuple[object, str | None, int, int, int]] = []
        for match in analysis.matches:
            detection = match.troop
            card_id = DIRECT_UNIT_TO_CARD.get(detection.class_name)
            if (
                detection.team not in {"ally", "enemy"}
                or detection.track_id is None
                or float(detection.confidence) < 0.5
            ):
                continue
            definition = engine.ruleset.cards.get(card_id) if card_id else None
            x_mtile, y_mtile = _detection_world_position(
                detection,
                analysis.arena_px,
                ground_anchor=definition is None or definition.kind != "building",
                center_margin_mtile=int(definition.collision_radius_mtile or 0) if definition else 0,
            )
            visible.append(
                (
                    detection,
                    card_id,
                    0 if detection.team == "ally" else 1,
                    x_mtile,
                    y_mtile,
                )
            )
        for detection, card_id, owner, x_mtile, y_mtile in visible:
            if card_id != "cannon" or float(detection.confidence) < threshold:
                continue
            contaminated = any(
                other_owner != owner
                and distance_mtile(x_mtile, y_mtile, other_x, other_y)
                < contamination_radius_mtile
                for other, _, other_owner, other_x, other_y in visible
                if other is not detection
            )
            tracks[(owner, int(detection.track_id))].append(
                {
                    "frame_idx": frame_idx,
                    "video_time_s": video_time,
                    "hp": _observed_hp(detection, maximum_hp),
                    "confidence": float(detection.confidence),
                    "contaminated": contaminated,
                }
            )
    if not level_confirmed:
        raise MiningManifestError(
            f"replay cache has no exact full support-tower HP confirming Level {source_level}"
        )

    candidates: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    lifetime_s = int(cannon.lifetime_us or 0) / 1_000_000
    deploy_s = cannon.deploy_time_us / 1_000_000
    cache_first_frame = min(frame_times) if frame_times else None
    for (owner, track_id), rows in sorted(tracks.items()):
        onset_time = float(rows[0]["video_time_s"])
        if actions:
            timed_actions = [
                (
                    action,
                    _labeled_frame_video_time(
                        int(action["frame_idx"]),
                        label_fps=label_fps,
                        replay_frame_times=frame_times,
                    ),
                )
                for action in actions
                if action["owner"] == owner
            ]
            matching = [
                (action, action_time)
                for action, action_time in timed_actions
                if action_time is not None and 0 <= onset_time - action_time <= 2.0
            ]
        else:
            matching = [
                (
                    {
                        "event_index": None,
                        "owner": owner,
                        "frame_idx": int(rows[0]["frame_idx"]),
                        "inferred": True,
                    },
                    onset_time,
                )
            ]
        key = f"cannon:{owner}:{track_id}"
        if not matching:
            rejected.append({"track": key, "reason": "no_nearby_action_anchor"})
            continue
        action, action_time = min(
            matching,
            key=lambda item: onset_time - float(item[1]),
        )
        action_time = float(action_time)
        gaps = [
            float(right["video_time_s"]) - float(left["video_time_s"])
            for left, right in zip(rows, rows[1:])
        ]
        reasons: list[str] = []
        if not actions and cache_first_frame is not None and int(rows[0]["frame_idx"]) <= cache_first_frame:
            reasons.append("track_begins_at_cache_boundary")
        if gaps and max(gaps) > maximum_track_gap_s:
            reasons.append("detector_gap")
        if any(bool(row["contaminated"]) for row in rows):
            reasons.append("enemy_near_cannon")
        if final_video_time - float(rows[-1]["video_time_s"]) < minimum_absent_tail_s:
            reasons.append("no_absence_tail")
        sample_interval = sorted(gaps)[len(gaps) // 2] if gaps else maximum_track_gap_s
        observed_duration = float(rows[-1]["video_time_s"]) + sample_interval - action_time
        if abs(observed_duration - lifetime_s) > duration_tolerance_s:
            reasons.append("expiry_duration_outside_gate")

        curve_errors: dict[str, int] = {}
        for hypothesis, offset in (("placement", 0.0), ("post_deploy", deploy_s)):
            residuals = []
            for row in rows:
                age = max(0.0, float(row["video_time_s"]) - action_time - offset)
                expected = max(0, round(maximum_hp * (1.0 - age / lifetime_s)))
                residuals.append(abs(int(row["hp"]) - expected) * 1_000 // maximum_hp)
            curve_errors[hypothesis] = sum(residuals) // len(residuals)
        best_hypothesis = min(curve_errors, key=curve_errors.get)  # type: ignore[arg-type]
        if curve_errors[best_hypothesis] > maximum_curve_mae_permille:
            reasons.append("hp_curve_mismatch")
        record = {
            "track": key,
            "owner": owner,
            "track_id": track_id,
            "action_event_index": action["event_index"],
            "action_video_time_ms": round(action_time * 1_000),
            "action_source": action_source,
            "frame_start": int(rows[0]["frame_idx"]),
            "frame_end": int(rows[-1]["frame_idx"]),
            "observed_duration_ms": round(observed_duration * 1_000),
            "curve_mae_permille": curve_errors,
            "best_lifetime_start_hypothesis": best_hypothesis,
            "minimum_confidence": min(float(row["confidence"]) for row in rows),
            "rejection_reasons": reasons,
        }
        if not reasons:
            candidates.append(record)
        else:
            rejected.append({**record, "reason": reasons[0]})
    return {
        "schema_version": 1,
        "kind": "cannon_lifetime_candidate_report",
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "source_level": source_level,
        "cache_hash": _file_sha256(source),
        "ground_truth_hash": _file_sha256(truth_source) if truth_source is not None else None,
        "ground_truth_fps": label_fps,
        "automation": {
            "action_inference": action_source,
            "truth_promotion": False,
            "manual_labels_required": truth_source is not None,
        },
        "mechanics": {
            "cannon_lifetime_hp_decay": {
                "candidate_count": len(candidates),
            }
        },
        "candidates": candidates,
        "rejected": rejected,
    }


def discover_replay_cache_tower_damage(
    cache_path: str | Path,
    *,
    source_level: int,
    engine: BattleEngine | None = None,
    confidence_threshold: float = 0.80,
    minimum_plateau_frames: int = 3,
    attribution_window_frames: int = 5,
) -> dict[str, object]:
    """Discover exact supported-card damage from stable Princess HP plateaus.

    This is intentionally a candidate report. A transition is retained only
    when two adjacent OCR plateaus are stable, their exact delta has one
    declared card signature, and that card is detected for the attacking side
    near the transition. Version-mismatched or ambiguous footage is rejected
    instead of silently rewriting the ruleset.
    """

    threshold = _confidence(confidence_threshold, "confidence_threshold")
    if type(minimum_plateau_frames) is not int or minimum_plateau_frames < 2:
        raise MiningManifestError("minimum_plateau_frames must be at least 2")
    if type(attribution_window_frames) is not int or attribution_window_frames < 0:
        raise MiningManifestError("attribution_window_frames must be non-negative")
    engine = engine or BattleEngine()
    if type(source_level) is not int or source_level != engine.ruleset.level:
        raise MiningManifestError(
            f"replay source level {source_level!r} does not match ruleset level "
            f"{engine.ruleset.level}"
        )
    source = Path(cache_path).resolve()
    if not source.is_file():
        raise MiningManifestError(f"replay cache is not a file: {source}")

    from cr_bot.replay.cache import ReplayCacheReader
    from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

    frames: list[dict[str, object]] = []
    level_confirmed = False
    for replay_frame in ReplayCacheReader(source):
        analysis = replay_frame.analysis
        level_confirmed = _validate_replay_source_level(
            analysis,
            source_level=source_level,
            engine=engine,
        ) or level_confirmed
        visible: dict[str, list[dict[str, object]]] = {"ally": [], "enemy": []}
        for match in analysis.matches:
            detection = match.troop
            card_id = DIRECT_UNIT_TO_CARD.get(detection.class_name)
            if (
                card_id in engine.ruleset.interaction_set
                and detection.team in visible
                and float(detection.confidence) >= threshold
            ):
                definition = engine.ruleset.card(str(card_id))
                x_mtile, y_mtile = _detection_world_position(
                    detection,
                    analysis.arena_px,
                    ground_anchor=definition.kind != "building",
                    center_margin_mtile=int(definition.collision_radius_mtile or 0),
                )
                visible[str(detection.team)].append(
                    {
                        "card_id": str(card_id),
                        "track_id": (
                            int(detection.track_id)
                            if detection.track_id is not None
                            else None
                        ),
                        "x_mtile": x_mtile,
                        "y_mtile": y_mtile,
                    }
                )
        frames.append(
            {
                "frame_idx": int(replay_frame.frame_idx),
                "video_time_s": float(replay_frame.video_time_s),
                "towers_hp": dict(analysis.towers_hp),
                "visible": visible,
            }
        )
    if not level_confirmed:
        raise MiningManifestError(
            f"replay cache has no exact full support-tower HP confirming Level "
            f"{source_level}"
        )

    signatures: dict[int, list[str]] = {}
    for card_id in engine.ruleset.interaction_set:
        definition = engine.ruleset.card(card_id)
        damage = (
            definition.crown_tower_damage
            if definition.kind == "spell"
            else definition.damage
        )
        if damage is not None and damage > 0:
            signatures.setdefault(int(damage), []).append(card_id)

    candidates: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    transition_sequence = 0
    tower_lookup = {
        "enemy_support_left": (1, "right"),
        "enemy_support_right": (1, "left"),
        "own_support_left": (0, "left"),
        "own_support_right": (0, "right"),
    }
    reference_state = engine.new_battle(seed=0, shuffle_decks=False)
    tower_positions = {
        key: (tower.x_mtile, tower.y_mtile)
        for key, (owner, role) in tower_lookup.items()
        for tower in reference_state.entities.values()
        if tower.kind == "tower" and tower.owner == owner and tower.role == role
    }
    for tower_key in (
        "enemy_support_left",
        "enemy_support_right",
        "own_support_left",
        "own_support_right",
    ):
        runs: list[tuple[int, int, int]] = []
        current_value: int | None = None
        start = 0
        for index, frame in enumerate(frames):
            raw_hp = frame["towers_hp"].get(tower_key)  # type: ignore[union-attr]
            value = raw_hp if type(raw_hp) is int else None
            if value == current_value:
                continue
            if current_value is not None:
                runs.append((current_value, start, index - 1))
            current_value = value
            start = index
        if current_value is not None and frames:
            runs.append((current_value, start, len(frames) - 1))

        attacker_team = "ally" if tower_key.startswith("enemy_") else "enemy"
        for before, after in zip(runs, runs[1:]):
            old_hp, old_start, old_end = before
            new_hp, new_start, new_end = after
            old_count = old_end - old_start + 1
            new_count = new_end - new_start + 1
            if old_count < minimum_plateau_frames or new_count < minimum_plateau_frames:
                continue
            delta = old_hp - new_hp
            if delta <= 0:
                continue
            transition_sequence += 1
            low = max(0, new_start - attribution_window_frames)
            high = min(len(frames), new_start + attribution_window_frames + 1)
            tower_x, tower_y = tower_positions[tower_key]
            nearby_detections = [
                detection
                for frame in frames[low:high]
                for detection in frame["visible"][attacker_team]  # type: ignore[index]
                if distance_mtile(
                    tower_x,
                    tower_y,
                    int(detection["x_mtile"]),
                    int(detection["y_mtile"]),
                )
                <= int(
                    engine.ruleset.card(str(detection["card_id"])).sight_range_mtile
                    or 0
                )
                + 2_000
            ]
            visible_cards = sorted(
                {
                    str(detection["card_id"])
                    for detection in nearby_detections
                }
            )
            matching = sorted(set(signatures.get(delta, ())) & set(visible_cards))
            record = {
                "transition_sequence": transition_sequence,
                "tower": tower_key,
                "frame_idx": int(frames[new_start]["frame_idx"]),
                "video_time_ms": round(float(frames[new_start]["video_time_s"]) * 1_000),
                "old_hp": old_hp,
                "new_hp": new_hp,
                "damage": delta,
                "old_plateau_frames": old_count,
                "new_plateau_frames": new_count,
                "visible_supported_cards": visible_cards,
            }
            if len(matching) == 1:
                card_id = matching[0]
                attacker_track_ids = sorted(
                    {
                        int(detection["track_id"])
                        for detection in nearby_detections
                        if detection["card_id"] == card_id
                        and detection["track_id"] is not None
                    }
                )
                candidates.append(
                    {
                        **record,
                        "card_id": card_id,
                        "attacker_track_ids": attacker_track_ids,
                    }
                )
            else:
                reason = (
                    "damage_delta_not_declared"
                    if delta not in signatures
                    else "attacker_not_uniquely_attributed"
                )
                rejected.append({**record, "reason": reason, "matching_cards": matching})

    intervals: list[dict[str, object]] = []
    for left, right in zip(candidates, candidates[1:]):
        if (
            left["tower"] != right["tower"]
            or left["card_id"] != right["card_id"]
            or int(right["transition_sequence"]) != int(left["transition_sequence"]) + 1
            or not (
                set(left["attacker_track_ids"])
                & set(right["attacker_track_ids"])
            )
        ):
            continue
        card_id = str(left["card_id"])
        definition = engine.ruleset.card(card_id)
        # Death damage and suicide impacts can share the direct-attack damage
        # signature but are not repeat attacks from one deployment.
        if definition.kind == "spell" or definition.mechanics.get("death") or bool(
            definition.mechanics.get("suicide_on_attack")
        ):
            continue
        expected = int(definition.attack_interval_us or 0) // 1_000
        observed = int(right["video_time_ms"]) - int(left["video_time_ms"])
        if expected <= 0 or observed > expected * 3:
            continue
        intervals.append(
            {
                "card_id": card_id,
                "tower": left["tower"],
                "first_frame_idx": left["frame_idx"],
                "second_frame_idx": right["frame_idx"],
                "observed_interval_ms": observed,
                "declared_interval_ms": expected,
                "absolute_error_ms": abs(observed - expected),
            }
        )
    mechanics: dict[str, dict[str, object]] = {}
    for card_id in sorted({str(row["card_id"]) for row in candidates}):
        rows = [row for row in candidates if row["card_id"] == card_id]
        declared_damage = int(engine.ruleset.card(card_id).damage or 0)
        mechanics[f"{card_id}_tower_damage"] = {
            "candidate_count": len(rows),
            "declared_damage": declared_damage,
            "observed_damage_values": sorted({int(row["damage"]) for row in rows}),
            "exact_agreement_count": sum(
                int(row["damage"]) == declared_damage for row in rows
            ),
        }
    for card_id in sorted({str(row["card_id"]) for row in intervals}):
        rows = [row for row in intervals if row["card_id"] == card_id]
        errors = sorted(int(row["absolute_error_ms"]) for row in rows)
        rank = max(1, math.ceil(0.95 * len(errors)))
        mechanics[f"{card_id}_tower_repeat_interval"] = {
            "candidate_count": len(rows),
            "declared_interval_ms": int(rows[0]["declared_interval_ms"]),
            "observed_interval_ms": [
                int(row["observed_interval_ms"]) for row in rows
            ],
            "mae_ms": sum(errors) / len(errors),
            "p95_absolute_error_ms": errors[rank - 1],
        }
    return {
        "schema_version": 1,
        "kind": "tower_damage_candidate_report",
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "source_level": source_level,
        "cache_hash": _file_sha256(source),
        "confidence_threshold": threshold,
        "minimum_plateau_frames": minimum_plateau_frames,
        "mechanics": mechanics,
        "candidates": candidates,
        "intervals": intervals,
        "rejected": rejected,
    }


def discover_replay_cache_log_motion(
    cache_path: str | Path,
    *,
    ground_truth_path: str | Path,
    source_level: int,
    engine: BattleEngine | None = None,
    confidence_threshold: float = 0.75,
    maximum_action_delay_s: float = 1.0,
    maximum_track_gap_s: float = 0.25,
    minimum_moving_steps: int = 5,
    minimum_step_speed_mtile_per_s: int = 1_000,
) -> dict[str, object]:
    """Discover action-anchored, monotonic Log rolling-speed candidates.

    The detector can keep the deployment artwork under one track before the
    physical Log begins rolling.  Consequently detector onset is never used
    as motion onset: only the longest consecutive run of direction-consistent
    steps is measured.  This is a candidate report rather than an automatic
    ruleset rewrite.
    """

    threshold = _confidence(confidence_threshold, "confidence_threshold")
    if type(maximum_action_delay_s) not in (int, float) or not math.isfinite(maximum_action_delay_s) or maximum_action_delay_s < 0:
        raise MiningManifestError("maximum_action_delay_s must be finite and non-negative")
    if type(maximum_track_gap_s) not in (int, float) or not math.isfinite(maximum_track_gap_s) or maximum_track_gap_s <= 0:
        raise MiningManifestError("maximum_track_gap_s must be finite and positive")
    if type(minimum_moving_steps) is not int or minimum_moving_steps < 2:
        raise MiningManifestError("minimum_moving_steps must be at least 2")
    if type(minimum_step_speed_mtile_per_s) is not int or minimum_step_speed_mtile_per_s <= 0:
        raise MiningManifestError("minimum_step_speed_mtile_per_s must be positive")
    engine = engine or BattleEngine()
    if type(source_level) is not int or source_level != engine.ruleset.level:
        raise MiningManifestError(
            f"replay source level {source_level!r} does not match ruleset level {engine.ruleset.level}"
        )
    source = Path(cache_path).resolve()
    truth_source = Path(ground_truth_path).resolve()
    if not source.is_file() or not truth_source.is_file():
        raise MiningManifestError("cache and ground-truth action file must exist")
    try:
        truth = json.loads(truth_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MiningManifestError(f"cannot read ground-truth actions: {error}") from error
    events = truth.get("events")
    if not isinstance(events, list):
        raise MiningManifestError("ground-truth file must contain an events array")
    label_fps = _label_fps(truth)

    from collections import defaultdict
    from cr_bot.replay.cache import ReplayCacheReader
    from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

    tracks: dict[int, list[dict[str, object]]] = defaultdict(list)
    frame_times: dict[int, float] = {}
    level_confirmed = False
    for replay_frame in ReplayCacheReader(source):
        frame_times[int(replay_frame.frame_idx)] = float(replay_frame.video_time_s)
        level_confirmed = _validate_replay_source_level(
            replay_frame.analysis, source_level=source_level, engine=engine
        ) or level_confirmed
        for match in replay_frame.analysis.matches:
            detection = match.troop
            if (
                DIRECT_UNIT_TO_CARD.get(detection.class_name) != "log"
                or detection.track_id is None
                or float(detection.confidence) < threshold
            ):
                continue
            x_mtile, y_mtile = _detection_world_position(
                detection,
                replay_frame.analysis.arena_px,
                ground_anchor=True,
                center_margin_mtile=0,
            )
            tracks[int(detection.track_id)].append(
                {
                    "frame_idx": int(replay_frame.frame_idx),
                    "video_time_s": float(replay_frame.video_time_s),
                    "x_mtile": x_mtile,
                    "y_mtile": y_mtile,
                    "confidence": float(detection.confidence),
                }
            )
    if not level_confirmed:
        raise MiningManifestError(
            f"replay cache has no exact full support-tower HP confirming Level {source_level}"
        )

    actions: list[dict[str, object]] = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        try:
            card_id = engine.ruleset.resolve_card_id(str(event.get("card", "")))
        except (KeyError, ValueError):
            continue
        side = event.get("side")
        if card_id != "log" or side not in {"own", "enemy"}:
            continue
        frame_idx = _integer(event.get("frame_index"), "event.frame_index")
        action_time = _labeled_frame_video_time(
            frame_idx, label_fps=label_fps, replay_frame_times=frame_times
        )
        if action_time is not None:
            actions.append(
                {
                    "event_index": event_index,
                    "owner": 0 if side == "own" else 1,
                    "frame_idx": frame_idx,
                    "video_time_s": action_time,
                }
            )
    if not actions:
        raise MiningManifestError("ground-truth file contains no time-resolvable Log actions")

    declared_speed = int(engine.ruleset.card("log").projectile.speed_mtile_per_s)  # type: ignore[union-attr]
    candidates: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    used_tracks: set[int] = set()
    for action in actions:
        action_time = float(action["video_time_s"])
        eligible = [
            (rows[0]["video_time_s"] - action_time, track_id, rows)
            for track_id, rows in tracks.items()
            if track_id not in used_tracks
            and 0 <= float(rows[0]["video_time_s"]) - action_time <= maximum_action_delay_s
        ]
        if not eligible:
            rejected.append({**action, "reason": "no_action_anchored_log_track"})
            continue
        _, track_id, rows = min(eligible, key=lambda item: (item[0], item[1]))
        used_tracks.add(track_id)
        direction = -1 if int(action["owner"]) == 0 else 1
        runs: list[list[tuple[dict[str, object], dict[str, object], int]]] = []
        current: list[tuple[dict[str, object], dict[str, object], int]] = []
        for left, right in zip(rows, rows[1:]):
            dt = float(right["video_time_s"]) - float(left["video_time_s"])
            signed_dy = direction * (int(right["y_mtile"]) - int(left["y_mtile"]))
            speed = round(signed_dy / dt) if dt > 0 else 0
            if 0 < dt <= maximum_track_gap_s and speed >= minimum_step_speed_mtile_per_s:
                current.append((left, right, speed))
            else:
                if current:
                    runs.append(current)
                current = []
        if current:
            runs.append(current)
        moving = max(runs, key=len, default=[])
        base = {
            "action_event_index": action["event_index"],
            "action_video_time_ms": round(action_time * 1_000),
            "owner": action["owner"],
            "track_id": track_id,
            "detector_onset_video_time_ms": round(float(rows[0]["video_time_s"]) * 1_000),
        }
        if len(moving) < minimum_moving_steps:
            rejected.append({**base, "reason": "insufficient_monotonic_motion", "moving_steps": len(moving)})
            continue
        start = moving[0][0]
        end = moving[-1][1]
        duration_s = float(end["video_time_s"]) - float(start["video_time_s"])
        displacement = direction * (int(end["y_mtile"]) - int(start["y_mtile"]))
        observed_speed = round(displacement / duration_s)
        candidates.append(
            {
                **base,
                "selected_segment_start_video_time_ms": round(float(start["video_time_s"]) * 1_000),
                "selected_segment_end_video_time_ms": round(float(end["video_time_s"]) * 1_000),
                "selected_segment_start_after_action_ms": round((float(start["video_time_s"]) - action_time) * 1_000),
                "moving_steps": len(moving),
                "observed_displacement_mtile": displacement,
                "observed_duration_ms": round(duration_s * 1_000),
                "observed_speed_mtile_per_s": observed_speed,
                "declared_speed_mtile_per_s": declared_speed,
                "absolute_error_mtile_per_s": abs(observed_speed - declared_speed),
                "relative_error_permille": round(abs(observed_speed - declared_speed) * 1_000 / declared_speed),
                "minimum_confidence": min(float(row["confidence"]) for pair in moving for row in pair[:2]),
                "frame_start": start["frame_idx"],
                "frame_end": end["frame_idx"],
            }
        )
        lateral_displacement = abs(int(end["x_mtile"]) - int(start["x_mtile"]))
        candidates[-1]["lateral_displacement_mtile"] = lateral_displacement
        candidates[-1]["lateral_to_forward_permille"] = round(
            lateral_displacement * 1_000 / max(1, displacement)
        )
        if lateral_displacement * 4 > displacement:
            rejected.append({**candidates.pop(), "reason": "excessive_lateral_drift"})
    speed_rows = [int(row["observed_speed_mtile_per_s"]) for row in candidates]
    errors = [abs(value - declared_speed) for value in speed_rows]
    mechanic = {
        "candidate_count": len(speed_rows),
        "declared_speed_mtile_per_s": declared_speed,
        "observed_speed_mtile_per_s": speed_rows,
        "mean_observed_speed_mtile_per_s": (
            round(sum(speed_rows) / len(speed_rows)) if speed_rows else None
        ),
        "mae_mtile_per_s": (
            sum(errors) / len(errors) if errors else None
        ),
    }
    return {
        "schema_version": 1,
        "kind": "log_motion_candidate_report",
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "source_level": source_level,
        "cache_hash": _file_sha256(source),
        "ground_truth_hash": _file_sha256(truth_source),
        "ground_truth_fps": label_fps,
        "declared_speed_mtile_per_s": declared_speed,
        "mechanics": {"log_rolling_speed": mechanic},
        "candidates": candidates,
        "rejected": rejected,
    }


def discover_replay_cache_fireball_flights(
    cache_path: str | Path,
    *,
    ground_truth_path: str | Path,
    source_level: int,
    engine: BattleEngine | None = None,
    confidence_threshold: float = 0.75,
    maximum_track_onset_delay_s: float = 1.0,
    maximum_impact_gap_s: float = 0.7,
    target_tolerance_mtile: int = 2_000,
    minimum_flight_samples: int = 6,
) -> dict[str, object]:
    """Discover Fireball action-to-impact timing from moving/explosion tracks.

    A candidate needs a localized labeled cast, a direction-consistent moving
    Fireball track, and a later compact Fireball-effect track centered near
    the labeled target. The first compact-effect frame is the observed impact;
    no airborne screen point is projected onto the ground plane for speed.
    """

    threshold = _confidence(confidence_threshold, "confidence_threshold")
    if type(maximum_track_onset_delay_s) not in (int, float) or not math.isfinite(maximum_track_onset_delay_s) or maximum_track_onset_delay_s < 0:
        raise MiningManifestError("maximum_track_onset_delay_s must be finite and non-negative")
    if type(maximum_impact_gap_s) not in (int, float) or not math.isfinite(maximum_impact_gap_s) or maximum_impact_gap_s <= 0:
        raise MiningManifestError("maximum_impact_gap_s must be finite and positive")
    if type(target_tolerance_mtile) is not int or target_tolerance_mtile <= 0:
        raise MiningManifestError("target_tolerance_mtile must be positive")
    if type(minimum_flight_samples) is not int or minimum_flight_samples < 3:
        raise MiningManifestError("minimum_flight_samples must be at least 3")
    engine = engine or BattleEngine()
    if type(source_level) is not int or source_level != engine.ruleset.level:
        raise MiningManifestError(
            f"replay source level {source_level!r} does not match ruleset level {engine.ruleset.level}"
        )
    source = Path(cache_path).resolve()
    truth_source = Path(ground_truth_path).resolve()
    if not source.is_file() or not truth_source.is_file():
        raise MiningManifestError("cache and ground-truth action file must exist")
    try:
        truth = json.loads(truth_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MiningManifestError(f"cannot read ground-truth actions: {error}") from error
    events = truth.get("events")
    if not isinstance(events, list):
        raise MiningManifestError("ground-truth file must contain an events array")
    label_fps = _label_fps(truth)

    from collections import defaultdict
    from cr_bot.replay.cache import ReplayCacheReader
    from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

    tracks: dict[int, list[dict[str, object]]] = defaultdict(list)
    frame_times: dict[int, float] = {}
    level_confirmed = False
    for replay_frame in ReplayCacheReader(source):
        frame_times[int(replay_frame.frame_idx)] = float(replay_frame.video_time_s)
        level_confirmed = _validate_replay_source_level(
            replay_frame.analysis, source_level=source_level, engine=engine
        ) or level_confirmed
        for match in replay_frame.analysis.matches:
            detection = match.troop
            if (
                DIRECT_UNIT_TO_CARD.get(detection.class_name) != "fireball"
                or detection.track_id is None
                or float(detection.confidence) < threshold
            ):
                continue
            x_mtile, y_mtile = _detection_world_position(
                detection, replay_frame.analysis.arena_px
            )
            tracks[int(detection.track_id)].append(
                {
                    "frame_idx": int(replay_frame.frame_idx),
                    "video_time_s": float(replay_frame.video_time_s),
                    "x_mtile": x_mtile,
                    "y_mtile": y_mtile,
                    "confidence": float(detection.confidence),
                }
            )
    if not level_confirmed:
        raise MiningManifestError(
            f"replay cache has no exact full support-tower HP confirming Level {source_level}"
        )

    actions: list[dict[str, object]] = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        try:
            card_id = engine.ruleset.resolve_card_id(str(event.get("card", "")))
        except (KeyError, ValueError):
            continue
        cell = event.get("cell")
        side = event.get("side")
        if (
            card_id != "fireball"
            or side not in {"own", "enemy"}
            or not isinstance(cell, list)
            or len(cell) != 2
            or any(type(value) is not int for value in cell)
        ):
            continue
        frame_idx = _integer(event.get("frame_index"), "event.frame_index")
        action_time = _labeled_frame_video_time(
            frame_idx, label_fps=label_fps, replay_frame_times=frame_times
        )
        if action_time is not None:
            actions.append(
                {
                    "event_index": event_index,
                    "owner": 0 if side == "own" else 1,
                    "frame_idx": frame_idx,
                    "video_time_s": action_time,
                    "cell": (int(cell[0]), int(cell[1])),
                }
            )
    if not actions:
        raise MiningManifestError(
            "ground-truth file contains no localized, time-resolvable Fireball actions"
        )

    candidates: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    used_tracks: set[int] = set()
    card = engine.ruleset.card("fireball")
    speed = int(card.projectile.speed_mtile_per_s)  # type: ignore[union-attr]
    reference = engine.new_battle(seed=0, shuffle_decks=False)
    for action in actions:
        action_time = float(action["video_time_s"])
        owner = int(action["owner"])
        direction = -1 if owner == 0 else 1
        target_x, target_y = cell_center_mtile(action["cell"])  # type: ignore[arg-type]
        moving_options: list[tuple[float, int, list[dict[str, object]]]] = []
        for track_id, rows in tracks.items():
            if track_id in used_tracks or len(rows) < minimum_flight_samples:
                continue
            onset_delay = float(rows[0]["video_time_s"]) - action_time
            signed_displacement = direction * (
                int(rows[-1]["y_mtile"]) - int(rows[0]["y_mtile"])
            )
            if (
                0 <= onset_delay <= maximum_track_onset_delay_s
                and signed_displacement >= 5_000
            ):
                moving_options.append((onset_delay, track_id, rows))
        base = {
            "action_event_index": action["event_index"],
            "action_video_time_ms": round(action_time * 1_000),
            "owner": owner,
            "target_cell": list(action["cell"]),
            "target_x_mtile": target_x,
            "target_y_mtile": target_y,
        }
        if not moving_options:
            rejected.append({**base, "reason": "no_direction_consistent_flight_track"})
            continue
        _, flight_id, flight = min(moving_options, key=lambda item: (item[0], item[1]))
        flight_end = float(flight[-1]["video_time_s"])
        impact_options: list[tuple[int, float, int, list[dict[str, object]]]] = []
        for track_id, rows in tracks.items():
            if track_id == flight_id or track_id in used_tracks:
                continue
            gap = float(rows[0]["video_time_s"]) - flight_end
            span = max(
                max(int(row["x_mtile"]) for row in rows) - min(int(row["x_mtile"]) for row in rows),
                max(int(row["y_mtile"]) for row in rows) - min(int(row["y_mtile"]) for row in rows),
            )
            target_error = distance_mtile(
                target_x, target_y, int(rows[0]["x_mtile"]), int(rows[0]["y_mtile"])
            )
            if 0 <= gap <= maximum_impact_gap_s and span <= 1_500 and target_error <= target_tolerance_mtile:
                impact_options.append((target_error, gap, track_id, rows))
        if not impact_options:
            rejected.append(
                {**base, "flight_track_id": flight_id, "reason": "no_target_localized_impact_track"}
            )
            used_tracks.add(flight_id)
            continue
        target_error, _, impact_id, impact = min(
            impact_options, key=lambda item: (item[0], item[1], item[2])
        )
        used_tracks.update((flight_id, impact_id))
        impact_time = float(impact[0]["video_time_s"])
        observed_ms = round((impact_time - action_time) * 1_000)
        king = next(
            entity
            for entity in reference.entities.values()
            if entity.kind == "tower" and entity.owner == owner and entity.role == "king"
        )
        distance = distance_mtile(king.x_mtile, king.y_mtile, target_x, target_y)
        travel_per_tick = max(1, speed * engine.ruleset.tick_us // 1_000_000)
        simulated_ms = math.ceil(distance / travel_per_tick) * engine.ruleset.tick_us // 1_000
        candidates.append(
            {
                **base,
                "flight_track_id": flight_id,
                "impact_track_id": impact_id,
                "flight_frame_start": flight[0]["frame_idx"],
                "flight_frame_end": flight[-1]["frame_idx"],
                "impact_frame_idx": impact[0]["frame_idx"],
                "impact_video_time_ms": round(impact_time * 1_000),
                "observed_action_to_impact_ms": observed_ms,
                "simulated_action_to_impact_ms": simulated_ms,
                "absolute_error_ms": abs(observed_ms - simulated_ms),
                "impact_target_error_mtile": target_error,
                "minimum_flight_confidence": min(float(row["confidence"]) for row in flight),
                "impact_confidence": float(impact[0]["confidence"]),
            }
        )
    errors = [int(row["absolute_error_ms"]) for row in candidates]
    return {
        "schema_version": 1,
        "kind": "fireball_flight_candidate_report",
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "source_level": source_level,
        "cache_hash": _file_sha256(source),
        "ground_truth_hash": _file_sha256(truth_source),
        "ground_truth_fps": label_fps,
        "mechanics": {
            "fireball_action_to_impact": {
                "candidate_count": len(candidates),
                "mae_ms": sum(errors) / len(errors) if errors else None,
                "observed_ms": [int(row["observed_action_to_impact_ms"]) for row in candidates],
                "simulated_ms": [int(row["simulated_action_to_impact_ms"]) for row in candidates],
            }
        },
        "candidates": candidates,
        "rejected": rejected,
    }


def compile_replay_cache_hog_cannon_pulls(
    cache_path: str | Path,
    *,
    ground_truth_path: str | Path,
    corpus_id: str,
    group_id: str,
    source_level: int,
    engine: BattleEngine | None = None,
    confidence_threshold: float = 0.80,
    minimum_track_frames: int = 5,
    maximum_duration_s: float = 2.4,
    contamination_radius_mtile: int = 3_500,
    split_salt: str = "hog-cycle-pulls-v1",
    evidence_split: str | None = None,
) -> MiningResult:
    """Mine action-anchored Hog→Cannon pull trajectories.

    A Cannon detector track is accepted only when its onset follows a curated
    Cannon play of the same owner and agrees with the labeled placement cell.
    The Hog must belong to the opponent, overlap the Cannon track for a stable
    run, and measurably approach it.  Bounding-box bottom center is used as the
    Hog's ground contact point; Cannon position comes from the action cell.
    """

    if type(minimum_track_frames) is not int or minimum_track_frames < 3:
        raise MiningManifestError("minimum_track_frames must be at least 3")
    if type(maximum_duration_s) not in (int, float) or not math.isfinite(maximum_duration_s) or maximum_duration_s <= 0:
        raise MiningManifestError("maximum_duration_s must be finite and positive")
    if type(contamination_radius_mtile) is not int or contamination_radius_mtile < 1:
        raise MiningManifestError("contamination_radius_mtile must be positive")
    threshold = _confidence(confidence_threshold, "confidence_threshold")
    if evidence_split is not None and evidence_split not in _SPLITS:
        raise MiningManifestError(f"evidence_split must be one of {_SPLITS}")
    engine = engine or BattleEngine()
    if type(source_level) is not int or source_level != engine.ruleset.level:
        raise MiningManifestError(
            f"replay source level {source_level!r} does not match ruleset level "
            f"{engine.ruleset.level}"
        )
    source = Path(cache_path).resolve()
    truth_source = Path(ground_truth_path).resolve()
    if not source.is_file() or not truth_source.is_file():
        raise MiningManifestError("cache and ground-truth action file must exist")
    try:
        truth = json.loads(truth_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MiningManifestError(f"cannot read ground-truth actions: {error}") from error
    events = truth.get("events")
    if not isinstance(events, list):
        raise MiningManifestError("ground-truth file must contain an events array")
    label_fps = _label_fps(truth)
    cannon_actions: list[dict[str, object]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("card") != "cannon":
            continue
        side = event.get("side")
        cell = event.get("cell")
        localized_cell = (
            (cell[0], cell[1])
            if isinstance(cell, list)
            and len(cell) == 2
            and all(type(value) is int for value in cell)
            else None
        )
        if side not in {"own", "enemy"}:
            continue
        cannon_actions.append(
            {
                "event_index": index,
                "owner": 0 if side == "own" else 1,
                "frame_idx": _integer(event.get("frame_index"), "event.frame_index"),
                "cell": localized_cell,
            }
        )
    if not cannon_actions:
        raise MiningManifestError("ground-truth file contains no Cannon actions")

    from collections import defaultdict
    from cr_bot.replay.cache import ReplayCacheReader
    from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

    tracks: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    frame_units: dict[int, list[dict[str, object]]] = defaultdict(list)
    frame_times: dict[int, float] = {}
    level_confirmed = False
    for replay_frame in ReplayCacheReader(source):
        frame_times[int(replay_frame.frame_idx)] = float(replay_frame.video_time_s)
        level_confirmed = _validate_replay_source_level(
            replay_frame.analysis,
            source_level=source_level,
            engine=engine,
        ) or level_confirmed
        for match in replay_frame.analysis.matches:
            detection = match.troop
            card_id = DIRECT_UNIT_TO_CARD.get(detection.class_name)
            if (
                detection.team in {"ally", "enemy"}
                and detection.track_id is not None
                and float(detection.confidence) >= 0.5
            ):
                unit_x, unit_y = _detection_world_position(
                    detection,
                    replay_frame.analysis.arena_px,
                    ground_anchor=card_id != "cannon",
                    center_margin_mtile=(
                        int(engine.ruleset.card(card_id).collision_radius_mtile or 0)
                        if card_id in engine.ruleset.cards
                        else 0
                    ),
                )
                frame_units[int(replay_frame.frame_idx)].append(
                    {
                        "key": (
                            str(card_id or detection.class_name),
                            0 if detection.team == "ally" else 1,
                            int(detection.track_id),
                        ),
                        "x_mtile": unit_x,
                        "y_mtile": unit_y,
                    }
                )
            if card_id not in {"hog-rider", "cannon"}:
                continue
            if (
                detection.team not in {"ally", "enemy"}
                or detection.track_id is None
                or float(detection.confidence) < threshold
            ):
                continue
            owner = 0 if detection.team == "ally" else 1
            x, y = _detection_world_position(
                detection,
                replay_frame.analysis.arena_px,
                ground_anchor=card_id == "hog-rider",
                center_margin_mtile=int(
                    engine.ruleset.card(card_id).collision_radius_mtile or 0
                ),
            )
            maximum_hp = int(engine.ruleset.card(card_id).hitpoints or 1)
            tracks[(card_id, owner, int(detection.track_id))].append(
                {
                    "frame_idx": int(replay_frame.frame_idx),
                    "video_time_s": float(replay_frame.video_time_s),
                    "x_mtile": x,
                    "y_mtile": y,
                    "hp": _observed_hp(detection, maximum_hp),
                    "confidence": float(detection.confidence),
                }
            )

    if not level_confirmed:
        raise MiningManifestError(
            f"replay cache has no exact full support-tower HP confirming Level "
            f"{source_level}"
        )

    clips: list[dict[str, object]] = []
    cache_hash = _file_sha256(source)
    used_hogs: set[tuple[int, int]] = set()
    for cannon_key, cannon_rows in sorted(tracks.items()):
        if cannon_key[0] != "cannon" or len(cannon_rows) < minimum_track_frames:
            continue
        cannon_owner = cannon_key[1]
        onset_time = float(cannon_rows[0]["video_time_s"])
        timed_actions = [
            (
                action,
                _labeled_frame_video_time(
                    int(action["frame_idx"]),
                    label_fps=label_fps,
                    replay_frame_times=frame_times,
                ),
            )
            for action in cannon_actions
            if action["owner"] == cannon_owner
        ]
        matching_actions = [
            (action, action_time)
            for action, action_time in timed_actions
            if action_time is not None and 0 <= onset_time - action_time <= 2.0
        ]
        if not matching_actions:
            continue
        action, action_time = min(
            matching_actions,
            key=lambda row: onset_time - float(row[1]),
        )
        detected_x = int(cannon_rows[0]["x_mtile"])
        detected_y = int(cannon_rows[0]["y_mtile"])
        if action["cell"] is not None:
            cannon_x, cannon_y = cell_center_mtile(action["cell"])  # type: ignore[arg-type]
            if distance_mtile(cannon_x, cannon_y, detected_x, detected_y) > 2_000:
                continue
            placement_method = "localized_action_cell"
        else:
            onset_rows = cannon_rows[: min(5, len(cannon_rows))]
            cannon_x = sorted(int(row["x_mtile"]) for row in onset_rows)[len(onset_rows) // 2]
            cannon_y = sorted(int(row["y_mtile"]) for row in onset_rows)[len(onset_rows) // 2]
            if any(
                distance_mtile(
                    cannon_x,
                    cannon_y,
                    int(row["x_mtile"]),
                    int(row["y_mtile"]),
                ) > 500
                for row in onset_rows
            ):
                continue
            placement_method = "stable_detector_position"
        cannon_by_frame = {int(row["frame_idx"]): row for row in cannon_rows}

        candidates: list[tuple[tuple[str, int, int], list[dict[str, object]]]] = []
        for hog_key, hog_rows in tracks.items():
            if hog_key[0] != "hog-rider" or hog_key[1] == cannon_owner:
                continue
            common = [
                row for row in hog_rows if int(row["frame_idx"]) in cannon_by_frame
            ]
            if len(common) < minimum_track_frames:
                continue
            runs: list[list[dict[str, object]]] = []
            current: list[dict[str, object]] = []
            for row in common:
                if (
                    current
                    and float(row["video_time_s"])
                    - float(current[-1]["video_time_s"])
                    > 0.35
                ):
                    runs.append(current)
                    current = []
                current.append(row)
            if current:
                runs.append(current)
            for run in runs:
                if len(run) >= minimum_track_frames:
                    candidates.append((hog_key, run))
        if not candidates:
            continue
        hog_key, run = max(
            candidates,
            key=lambda item: (len(item[1]), -int(item[1][0]["frame_idx"])),
        )
        identity = (hog_key[1], hog_key[2])
        if identity in used_hogs:
            continue
        start_time = float(run[0]["video_time_s"])
        selected = [
            row
            for row in run
            if float(row["video_time_s"]) - start_time <= float(maximum_duration_s)
        ]
        raw_selected = selected
        uncontaminated: list[dict[str, object]] = []
        for row in selected:
            frame_index = int(row["frame_idx"])
            interferes = _pull_frame_is_contaminated(
                frame_units.get(frame_index, []),
                hog_key=hog_key,
                cannon_key=cannon_key,
                hog_position=(int(row["x_mtile"]), int(row["y_mtile"])),
                cannon_position=(cannon_x, cannon_y),
                radius_mtile=contamination_radius_mtile,
            )
            if interferes:
                break
            uncontaminated.append(row)
        selected = uncontaminated
        targeting_only = False
        if len(selected) < minimum_track_frames:
            if not raw_selected:
                continue
            raw_start_distance = distance_mtile(
                int(raw_selected[0]["x_mtile"]),
                int(raw_selected[0]["y_mtile"]),
                cannon_x,
                cannon_y,
            )
            raw_minimum_distance = min(
                distance_mtile(
                    int(row["x_mtile"]),
                    int(row["y_mtile"]),
                    cannon_x,
                    cannon_y,
                )
                for row in raw_selected
            )
            contact_gate = (
                int(engine.ruleset.card("hog-rider").collision_radius_mtile or 0)
                + int(engine.ruleset.card("cannon").collision_radius_mtile or 0)
                + int(engine.ruleset.card("hog-rider").range_mtile or 0)
                + 500
            )
            other_building_visible = any(
                unit["key"] != cannon_key
                and str(unit["key"][0]) in engine.ruleset.cards  # type: ignore[index]
                and engine.ruleset.card(str(unit["key"][0])).kind == "building"  # type: ignore[index]
                for row in raw_selected
                for unit in frame_units.get(int(row["frame_idx"]), [])
            )
            if (
                raw_start_distance - raw_minimum_distance < 1_000
                or raw_minimum_distance > contact_gate
                or other_building_visible
            ):
                continue
            # Collision-contaminated footage can still prove the discrete
            # target choice when the Hog conclusively reaches the only
            # detected building. Keep only the initial state; do not pretend
            # the omitted troops form a valid trajectory oracle.
            selected = [raw_selected[0]]
            targeting_only = True
        smoothed: list[dict[str, object]] = []
        for sample_index, row in enumerate(selected):
            copied = dict(row)
            if 0 < sample_index < len(selected) - 1:
                neighborhood = selected[sample_index - 1 : sample_index + 2]
                copied["x_mtile"] = sorted(int(item["x_mtile"]) for item in neighborhood)[1]
                copied["y_mtile"] = sorted(int(item["y_mtile"]) for item in neighborhood)[1]
            smoothed.append(copied)
        selected = smoothed
        start_distance = distance_mtile(
            int(selected[0]["x_mtile"]),
            int(selected[0]["y_mtile"]),
            cannon_x,
            cannon_y,
        )
        end_distance = distance_mtile(
            int(selected[-1]["x_mtile"]),
            int(selected[-1]["y_mtile"]),
            cannon_x,
            cannon_y,
        )
        if not targeting_only and start_distance - end_distance < 1_000:
            continue
        used_hogs.add(identity)
        first = selected[0]
        cannon_at_start = cannon_by_frame[int(first["frame_idx"])]
        clip_id = (
            f"{source.stem}:hog-cannon-pull:"
            f"{cannon_key[2]}:{hog_key[2]}"
        )
        track_samples = []
        for row in selected:
            relative_tick = round(
                (float(row["video_time_s"]) - start_time)
                * 1_000_000
                / engine.ruleset.tick_us
            )
            track_samples.append(
                {
                    "tick": relative_tick,
                    "x_mtile": int(row["x_mtile"]),
                    "y_mtile": int(row["y_mtile"]),
                    "confidence": float(row["confidence"]),
                }
            )
        clips.append(
            {
                "clip_id": clip_id,
                "group_id": group_id,
                **({"split": evidence_split} if evidence_split is not None else {}),
                "media_hash": cache_hash,
                "frame_start": int(selected[0]["frame_idx"]),
                "frame_end": int(selected[-1]["frame_idx"]),
                "method": (
                    "replay_cache_action_anchored_pull_ground_contact_v2:"
                    + placement_method
                    + f":declared_level_{source_level}_support_hp_conflict_checked"
                    + ":label_fps_to_video_time"
                    + (":targeting_only_contaminated_path" if targeting_only else "")
                ),
                "confidence": min(
                    [float(row["confidence"]) for row in selected]
                    + [float(cannon_at_start["confidence"])]
                ),
                "seed": int.from_bytes(
                    hashlib.sha256(clip_id.encode("utf-8")).digest()[:8], "big"
                ),
                "initial": {
                    "tick": 0,
                    "elapsed_us": 0,
                    "phase": "regulation",
                    "towers": [],
                    "entities": [
                        {
                            "track_id": f"hog:{hog_key[2]}",
                            "card_id": "hog-rider",
                            "owner": hog_key[1],
                            "x_mtile": int(first["x_mtile"]),
                            "y_mtile": int(first["y_mtile"]),
                            "hp": int(first["hp"]),
                            "confidence": float(first["confidence"]),
                        },
                        {
                            "track_id": f"cannon:{cannon_key[2]}",
                            "card_id": "cannon",
                            "owner": cannon_owner,
                            "x_mtile": cannon_x,
                            "y_mtile": cannon_y,
                            "hp": int(cannon_at_start["hp"]),
                            "confidence": float(cannon_at_start["confidence"]),
                        },
                    ],
                },
                "tracks": [
                    {
                        "track_id": f"hog:{hog_key[2]}",
                        "mechanic": (
                            "hog_cannon_pull_initial_state"
                            if targeting_only
                            else "hog_cannon_pull_trajectory"
                        ),
                        "confidence": min(float(row["confidence"]) for row in selected),
                        "samples": track_samples,
                    }
                ],
            }
        )
    if not clips:
        raise MiningManifestError("no action-anchored Hog/Cannon pulls met the gate")
    result = compile_observation_manifest(
        {
            "schema_version": MINING_SCHEMA_VERSION,
            "corpus_id": corpus_id,
            "split_salt": split_salt,
            "confidence_threshold": threshold,
            "position_tolerance_mtile": 1_000,
            "clips": clips,
        },
        engine=engine,
    )
    payload = corpus_to_dict(result.corpus)
    for case in payload["cases"]:
        case["traces"] = [
            {
                "trace_id": f"{case['case_id']}:target",
                "mechanic": "hog_cannon_pull_targeting",
                "included_event_kinds": ["target_changed"],
                "filters": {"card_id": "hog-rider"},
                "events": [
                    {
                        "tick": 0,
                        "kind": "target_changed",
                        "values": {
                            "card_id": "hog-rider",
                            "target_card_id": "cannon",
                        },
                        "tick_tolerance": 2,
                    }
                ],
            }
        ]
    corpus = validation_corpus_from_dict(payload)
    corpus = validation_corpus_from_dict(corpus_to_dict(corpus))
    return MiningResult(corpus=corpus, discarded=result.discarded)


__all__ = [
    "DiscardedClip",
    "MINING_SCHEMA_VERSION",
    "MiningManifestError",
    "MiningResult",
    "assigned_split",
    "compile_observation_manifest",
    "compile_replay_cache_movement",
    "compile_replay_cache_hog_cannon_pulls",
    "discover_replay_cache_interactions",
    "discover_replay_cache_interactions_batch",
    "merge_replay_interaction_reports",
    "discover_replay_cache_cannon_lifetimes",
    "corpus_to_dict",
    "load_observation_manifest",
]
