"""Canonical JSON scenarios shared by tests, regressions, and evaluation.

The schema is intentionally small and versioned.  Automatically mined
failures are written as ``candidate`` scenarios; changing an expectation or
promoting a candidate to regression/gold is an explicit review operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .actions import SimAction, action_from_dict, action_to_dict


SCENARIO_SCHEMA_VERSION = 1
VALID_SPLITS = {"synthetic", "calibration", "validation", "regression", "heldout", "candidate"}


@dataclass(frozen=True, slots=True)
class ScheduledAction:
    tick: int
    action: SimAction


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    ruleset_id: str
    ruleset_hash: str
    engine_version: str
    seed: int
    decks: tuple[tuple[str, ...], tuple[str, ...]]
    actions: tuple[ScheduledAction, ...] = ()
    max_ticks: int | None = None
    shuffle_decks: bool = False
    split: str = "synthetic"
    tags: tuple[str, ...] = ()
    oracle: dict[str, Any] = field(default_factory=dict)
    initial_state: Mapping[str, Any] | None = None
    schema_version: int = SCENARIO_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCENARIO_SCHEMA_VERSION:
            raise ValueError(f"unsupported scenario schema: {self.schema_version}")
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario_id is required")
        if not isinstance(self.engine_version, str) or not self.engine_version:
            raise ValueError("engine_version is required")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"invalid scenario split: {self.split!r}")
        if len(self.decks) != 2 or any(len(deck) != 8 for deck in self.decks):
            raise ValueError("a scenario requires exactly two eight-card decks")
        if self.max_ticks is not None and (type(self.max_ticks) is not int or self.max_ticks <= 0):
            raise ValueError("max_ticks must be positive when provided")
        if type(self.seed) is not int:
            raise ValueError("scenario seed must be an integer")
        if type(self.shuffle_decks) is not bool:
            raise ValueError("shuffle_decks must be boolean")
        if self.initial_state is not None:
            canonical = json.loads(
                json.dumps(self.initial_state, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
            if not isinstance(canonical, dict):
                raise ValueError("initial_state must be a JSON object")
            object.__setattr__(self, "initial_state", _freeze_json(canonical))
        previous = -1
        for scheduled in self.actions:
            if type(scheduled.tick) is not int or scheduled.tick < 0:
                raise ValueError("scenario action ticks must be non-negative integers")
            if scheduled.tick < previous:
                raise ValueError("scenario actions must be sorted by tick")
            previous = scheduled.tick
        if self.split == "candidate" and self.oracle.get("promoted") is True:
            raise ValueError("candidate scenarios cannot claim promotion")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "ruleset_id": self.ruleset_id,
            "ruleset_hash": self.ruleset_hash,
            "engine_version": self.engine_version,
            "seed": self.seed,
            "decks": [list(deck) for deck in self.decks],
            "actions": [
                {"tick": scheduled.tick, "action": action_to_dict(scheduled.action)}
                for scheduled in self.actions
            ],
            "max_ticks": self.max_ticks,
            "shuffle_decks": self.shuffle_decks,
            "split": self.split,
            "tags": list(self.tags),
            "oracle": self.oracle,
        }
        if self.initial_state is not None:
            result["initial_state"] = _thaw_json(self.initial_state)
        return result

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.dumps(), encoding="utf-8")


def scenario_from_dict(raw: dict[str, Any]) -> Scenario:
    if set(raw) - {
        "schema_version",
        "scenario_id",
        "ruleset_id",
        "ruleset_hash",
        "engine_version",
        "seed",
        "decks",
        "actions",
        "max_ticks",
        "shuffle_decks",
        "split",
        "tags",
        "oracle",
        "initial_state",
    }:
        unknown = sorted(set(raw) - {
            "schema_version", "scenario_id", "ruleset_id", "ruleset_hash", "engine_version", "seed", "decks",
            "actions", "max_ticks", "shuffle_decks", "split", "tags", "oracle", "initial_state",
        })
        raise ValueError(f"unknown scenario fields: {unknown}")
    raw_decks = raw["decks"]
    if not isinstance(raw_decks, list) or len(raw_decks) != 2:
        raise ValueError("decks must contain two arrays")
    if any(
        not isinstance(deck, list)
        or any(not isinstance(card, str) or not card for card in deck)
        for deck in raw_decks
    ):
        raise ValueError("each deck must be an array of non-empty card IDs")
    actions_list: list[ScheduledAction] = []
    for row in raw.get("actions", []):
        if not isinstance(row, dict) or type(row.get("tick")) is not int:
            raise ValueError("each scheduled action requires an integer tick")
        if not isinstance(row.get("action"), dict):
            raise ValueError("each scheduled action requires an action object")
        actions_list.append(ScheduledAction(row["tick"], action_from_dict(row["action"])))
    actions = tuple(actions_list)
    seed = raw["seed"]
    if type(seed) is not int:
        raise ValueError("scenario seed must be an integer")
    shuffle_decks = raw.get("shuffle_decks", False)
    if type(shuffle_decks) is not bool:
        raise ValueError("shuffle_decks must be boolean")
    max_ticks = raw.get("max_ticks")
    if max_ticks is not None and type(max_ticks) is not int:
        raise ValueError("max_ticks must be an integer or null")
    return Scenario(
        scenario_id=str(raw["scenario_id"]),
        ruleset_id=str(raw["ruleset_id"]),
        ruleset_hash=str(raw["ruleset_hash"]),
        engine_version=str(raw["engine_version"]),
        seed=seed,
        decks=(tuple(raw_decks[0]), tuple(raw_decks[1])),
        actions=actions,
        max_ticks=max_ticks,
        shuffle_decks=shuffle_decks,
        split=str(raw.get("split", "synthetic")),
        tags=tuple(str(tag) for tag in raw.get("tags", [])),
        oracle=dict(raw.get("oracle", {})),
        initial_state=raw.get("initial_state"),
        schema_version=int(raw.get("schema_version", SCENARIO_SCHEMA_VERSION)),
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def load_scenario(path: str | Path) -> Scenario:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("scenario document must be an object")
    return scenario_from_dict(raw)
