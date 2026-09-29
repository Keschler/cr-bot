"""Replay source matches and prepare correction and retention records."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import re

import numpy as np
import torch

from ..frontier_eval import _fresh_env, _observe_inputs, _recorded_action
from .artifacts import _check_deadline
from .records import CachedPrefix, ValidatedCorrection, DEFAULT_CONTEXT, DEFAULT_MIN_LCB
from .targets import action_from_record, validated_correction


_ENTRY_RE = re.compile(r"^(?P<split>[^:]+):(?P<match>.+):t(?P<t>[0-9]+)$")


def parse_entry_id(entry_id: str) -> tuple[str, str, int]:
    """Parse ``train:archetype#0:t247`` without relying on filenames."""

    match = _ENTRY_RE.match(str(entry_id))
    if match is None:
        raise ValueError(f"invalid frontier entry id: {entry_id!r}")
    return match.group("split"), match.group("match"), int(match.group("t"))


def match_json_path(root: Path, split: str, match_id: str) -> Path:
    """Resolve a materialized match JSON path."""

    path = root / f"{split}_{match_id.replace('#', '_')}_match.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _copy_input(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


def _required_observation_inputs(inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = (
        "raster", "global_features", "entities", "entity_mask", "hand_tokens",
        "opp_hand_probs", "opp_out_of_cycle", "opp_elixir_interval", "event_history",
    )
    out: dict[str, np.ndarray] = {}
    for name in required:
        value = np.asarray(inputs[name])
        out[name] = np.ascontiguousarray(value.copy())
    return out


def replay_selected_prefixes(
    match: Mapping[str, Any],
    offsets: Sequence[int],
    *,
    context: int = DEFAULT_CONTEXT,
) -> dict[int, list[dict[str, np.ndarray]]]:
    """Replay one source match and return fixed-length public input prefixes."""

    if context < 0:
        raise ValueError("context must be non-negative")
    result, _, _ = _replay_selected_prefixes_with_legal(
        match, offsets, context=context
    )
    return result


def _match_id_from_json(path: Path, match: Mapping[str, Any]) -> tuple[str, str] | None:
    """Recover the split/match key used by frontier entry ids.

    Materialised match files carry the archetype and match index in their
    config, so this remains correct even when an archetype itself contains an
    underscore.  The filename split is used only to keep train and sealed
    pools separate.
    """

    name = path.name
    suffix = "_match.json"
    if not name.endswith(suffix):
        return None
    split = name[:-len(suffix)].split("_", 1)[0]
    config = match.get("config", {})
    archetype = config.get("archetype")
    index = config.get("match_index")
    if split and archetype is not None and index is not None:
        return str(split), f"{archetype}#{index}"
    return None


def _load_verdict(root: Path, split: str, match_id: str, offset: int) -> dict[str, Any]:
    path = root / f"{split}_{match_id.replace('#', '_')}_t{offset}_verdict.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_frontier_entries(path: Path) -> list[dict[str, Any]]:
    """Load a frontier entry checkpoint and normalize it to plain dicts."""

    entries = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(entries, list):
        raise ValueError(f"frontier entries must be a list: {path}")
    return [dict(entry) for entry in entries]


def _replay_selected_prefixes_with_legal(
    match: Mapping[str, Any], offsets: Sequence[int], *, context: int,
    deadline_epoch: float | None = None,
) -> tuple[dict[int, list[dict[str, np.ndarray]]], dict[int, list[np.ndarray]], dict[int, list[bool]]]:
    """Replay helper retaining legality and WAIT availability."""

    requested = sorted({int(v) for v in offsets})
    if not requested:
        return {}, {}, {}
    decisions = match["decisions"]
    invalid = [offset for offset in requested if offset < 0 or offset >= len(decisions)]
    if invalid:
        raise ValueError(
            f"selected prefix offset(s) outside recorded decisions: {invalid}"
        )
    env = _fresh_env(dict(match))
    from simulator.public_state_estimator import PublicStateEstimator

    estimator = PublicStateEstimator()
    all_inputs: dict[int, dict[str, np.ndarray]] = {}
    all_legal: dict[int, np.ndarray] = {}
    all_wait: dict[int, bool] = {}
    maximum = requested[-1]
    for index in range(maximum + 1):
        _check_deadline(deadline_epoch)
        inputs, legal, _, _ = _observe_inputs(env, "cpu", estimator)
        if index in requested:
            expected_hash = decisions[index].get("state_hash")
            if not isinstance(expected_hash, str) or not expected_hash:
                raise ValueError(
                    f"selected prefix decision t={index} lacks a recorded state_hash"
                )
            actual_hash = env.state.state_hash()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"prefix replay diverged at t={index}: "
                    f"{actual_hash[:12]} != {expected_hash[:12]}"
                )
        all_inputs[index] = _required_observation_inputs(inputs)
        all_inputs[index]["legal_play"] = np.asarray(legal, dtype=bool).copy()
        all_wait[index] = bool(inputs.get("legal_wait", True))
        all_legal[index] = np.asarray(legal, dtype=bool).copy()
        if index < maximum:
            record = decisions[index]
            env.step_v2((_recorded_action(record["learner"], 0), _recorded_action(record["opponent"], 1)))
    outputs: dict[int, list[dict[str, np.ndarray]]] = {}
    legal_outputs: dict[int, list[np.ndarray]] = {}
    wait_outputs: dict[int, list[bool]] = {}
    for offset in requested:
        first = max(0, offset - context)
        sequence = [all_inputs[index] for index in range(first, offset + 1)]
        legal_sequence = [all_legal[index] for index in range(first, offset + 1)]
        wait_sequence = [all_wait[index] for index in range(first, offset + 1)]
        while len(sequence) < context + 1:
            sequence.insert(0, {name: _copy_input(value) for name, value in sequence[0].items()})
            legal_sequence.insert(0, legal_sequence[0].copy())
            wait_sequence.insert(0, wait_sequence[0])
        outputs[offset] = sequence
        legal_outputs[offset] = legal_sequence
        wait_outputs[offset] = wait_sequence
    return outputs, legal_outputs, wait_outputs


def prepare_records(
    root: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    context: int = DEFAULT_CONTEXT,
    min_lcb: float = DEFAULT_MIN_LCB,
    retention_count: int = 0,
    seed: int = 0,
    require_independent: bool = True,
    deadline_epoch: float | None = None,
) -> list[CachedPrefix]:
    """Build independently validated corrections and sampled ordinary replay."""

    if retention_count < 0:
        raise ValueError("retention_count must be non-negative")
    by_match: dict[tuple[str, str], dict[str, Any]] = {}
    match_paths: dict[tuple[str, str], Path] = {}
    selected: dict[tuple[str, str], set[int]] = {}
    # Validate every entry's source state, including entries whose verdict does
    # not pass the correction gate.  Otherwise a stale frontier file with zero
    # accepted corrections and no retention rows would never be replayed.
    verify_offsets: dict[tuple[str, str], set[int]] = {}
    corrections: list[ValidatedCorrection] = []
    requested_splits: set[str] = set()
    for raw in entries:
        _check_deadline(deadline_epoch)
        entry = dict(raw)
        split, match_id, offset = parse_entry_id(str(entry["entry_id"]))
        requested_splits.add(split)
        key = (split, match_id)
        if key not in by_match:
            path = match_json_path(root, split, match_id)
            by_match[key] = json.loads(path.read_text(encoding="utf-8"))
            match_paths[key] = path
        verify_offsets.setdefault(key, set()).add(offset)
        verdict = _load_verdict(root, split, match_id, offset)
        accepted, target, stats = validated_correction(
            verdict, min_lcb=min_lcb, require_independent=require_independent,
        )
        if accepted and target is not None:
            selected.setdefault(key, set()).add(offset)
            corrections.append(ValidatedCorrection(
                entry_id=str(entry["entry_id"]), match_key=key, offset=offset,
                verdict=verdict, action=target, statistics=stats,
            ))

    # Frontier entries are intentionally sparse.  Ordinary replay must draw
    # from the complete train rollout pool, including matches that contributed
    # no accepted frontier entry.  The validation root is a separate call, so
    # restricting to the entry splits here avoids accidentally mixing pools.
    for path in sorted(root.glob("*_match.json")):
        _check_deadline(deadline_epoch)
        try:
            match = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        key = _match_id_from_json(path, match)
        if key is None or (requested_splits and key[0] not in requested_splits):
            continue
        if key not in by_match:
            by_match[key] = match
            match_paths[key] = path
    if retention_count > 0:
        # One offset is drawn from each match in each round.  This gives every
        # eligible match an independent replay opportunity before any match
        # receives a second state, avoiding the old correction-bearing bias and
        # the old sorted-key truncation bias.
        candidates: dict[tuple[str, str], list[int]] = {}
        for key, match in sorted(by_match.items()):
            excluded = selected.get(key, set())
            values = [
                int(record["t"])
                for record in match.get("decisions", ())
                if int(record["t"]) >= context and int(record["t"]) not in excluded
            ]
            rng = np.random.default_rng(
                int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16)
            )
            rng.shuffle(values)
            candidates[key] = values
        chosen = 0
        while chosen < retention_count:
            progressed = False
            for key in sorted(candidates):
                if chosen >= retention_count:
                    break
                values = candidates[key]
                if not values:
                    continue
                selected.setdefault(key, set()).add(int(values.pop()))
                chosen += 1
                progressed = True
            if not progressed:
                break
    prefixes: dict[tuple[str, str], tuple[dict[int, list[dict[str, np.ndarray]]], dict[int, list[np.ndarray]], dict[int, list[bool]]]] = {}
    replay_offsets = {
        key: set(offsets) for key, offsets in verify_offsets.items()
    }
    for key, offsets in selected.items():
        replay_offsets.setdefault(key, set()).update(offsets)
    for key, offsets in replay_offsets.items():
        _check_deadline(deadline_epoch)
        prefixes[key] = _replay_selected_prefixes_with_legal(
            by_match[key], sorted(offsets), context=context, deadline_epoch=deadline_epoch,
        )
    records: list[CachedPrefix] = []
    for correction in corrections:
        key = correction.match_key
        offset = correction.offset
        verdict = correction.verdict
        stats = correction.statistics
        split, match_id = key
        match = by_match[key]
        seqs, legal, waits = prefixes[key]
        records.append(CachedPrefix(
            key=correction.entry_id, match_id=match_id, offset=offset,
            observations=seqs[offset], legal_play=legal[offset], legal_wait=waits[offset],
            recorded_action=action_from_record(match["decisions"][offset]["learner"]),
            correction_action=correction.action, branches=list(verdict["branches"]),
            paired_mean=float(stats["mean"]), paired_se=float(stats["se"]),
            correction_lcb=float(stats["lcb95"]), is_correction=True,
            match_path=str(match_paths[key]),
            opponent_spec={
                "strategy": str(match.get("config", {}).get("strategy", "wait")),
                "controller_seed": int(match.get("config", {}).get("controller_seed", 0)),
            },
            outcome_horizon=int(verdict.get("horizon", 128)),
            outcome_replicas=int(verdict.get("n_replicas", 4)),
            outcome_replica_seed_base=int(verdict.get("replica_seed_base", 0)),
        ))
    # ``CachedPrefix`` historically stored only ``match_id``.  Within one
    # split call this is sufficient and avoids changing the serialized public
    # key format.
    corrected = {(*correction.match_key, correction.offset) for correction in corrections}
    for key, offsets in selected.items():
        split, match_id = key
        match = by_match[key]
        seqs, legal, waits = prefixes[key]
        for offset in sorted(offsets):
            if (split, match_id, offset) in corrected:
                continue
            records.append(CachedPrefix(
                key=f"retention:{split}:{match_id}:t{offset}", match_id=match_id,
                offset=offset, observations=seqs[offset], legal_play=legal[offset],
                legal_wait=waits[offset], recorded_action=action_from_record(match["decisions"][offset]["learner"]),
                match_path=str(match_paths[key]),
                opponent_spec={
                    "strategy": str(match.get("config", {}).get("strategy", "wait")),
                    "controller_seed": int(match.get("config", {}).get("controller_seed", 0)),
                },
            ))
    corrections_out = [r for r in records if r.is_correction]
    retention_out = [r for r in records if not r.is_correction]
    return corrections_out + (retention_out if retention_count > 0 else [])
