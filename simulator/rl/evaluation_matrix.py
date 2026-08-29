"""Held-out evaluation matrices for recurrent prototype checkpoints.

The prototype evaluator intentionally has a narrow contract: one fixed player
deck, one deterministic opponent deck, and one controller.  This module is a
separate orchestration layer for a broader evaluation matrix.  It owns the
Cartesian product of opponent decks, opponent strategies, and reproducible
seeds, while keeping the result of every match distinct from its aggregate.

The public entry point is :func:`run_evaluation_matrix`.  It can run real
simulator matches from a prototype checkpoint, or it can receive an injected
``match_runner``.  The latter is useful for tests and for callers that own a
different simulator backend.  A match runner receives a sealed
:class:`MatchSpec` and returns a :class:`MatchResult`, an outcome string, or a
JSON-shaped mapping.

The actor path uses only the public V2 observation.  The opponent controller
is allowed to inspect authoritative simulator state because it is the
environment-side opponent, not an input to the learning actor.  The
``public-counter`` mode is retained as an explicit baseline; it is not
reported as neural actor performance.

This file deliberately does not modify :mod:`rl.prototype` or the simulator's
shared APIs.  It reuses their public checkpoint loader and the collector's
canonical V2 tensor batching helper where that avoids duplicating the model
contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import platform
from pathlib import Path
import random
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

from .domain_randomization import DomainRandomizationConfig, DomainRandomizationError
from .league import deterministic_seed
from .provenance import code_revision


EVALUATION_MATRIX_SCHEMA_VERSION = 2
EVALUATION_MATRIX_KIND = "recurrent_public_ppo_evaluation_matrix"
EVALUATION_PROVENANCE_SCHEMA_VERSION = 1
REPORT_COMPARISON_SCHEMA_VERSION = 1
REPORT_COMPARISON_KIND = "recurrent_public_ppo_evaluation_comparison"
DEFAULT_REPORT_COMPARISON_MAX_CELLS = 4096
_POLICY_MODES = frozenset(
    {"actor", "public-counter", "strategic-counter", "deterministic-counter"}
)
_BUILTIN_STRATEGIES = frozenset(
    {
        "deterministic-cycle",
        "deterministic-left",
        "deterministic-right",
        "random-legal",
        "wait",
    }
)


class EvaluationMatrixError(ValueError):
    """Raised when an evaluation matrix is malformed or cannot be run safely."""


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationMatrixError(f"{name} must be a non-empty string")
    return value.strip()


def _canonical_strategy_id(value: object) -> str:
    return _identifier(value, "strategy_id").casefold()


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise EvaluationMatrixError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise EvaluationMatrixError(f"{name} must be >= {minimum}")
    return value


def _sequence(value: object, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise EvaluationMatrixError(f"{name} must be a sequence")
    return tuple(value)


def _default_player_deck() -> tuple[str, ...]:
    try:
        from ..roster import PLAYER_DECK
    except ImportError:  # pragma: no cover - top-level ``rl`` layout
        from simulator.roster import PLAYER_DECK
    return tuple(PLAYER_DECK)


def _normalize_player_deck(
    value: object,
    *,
    name: str = "player_deck",
) -> tuple[str, ...]:
    """Validate a configurable learner deck without requiring a ruleset."""

    if value is None:
        value = _default_player_deck()
    cards = _sequence(value, name)
    if len(cards) != 8:
        raise EvaluationMatrixError(f"{name} must contain exactly eight cards")
    normalized = tuple(
        _identifier(card, f"{name}[{index}]")
        for index, card in enumerate(cards)
    )
    if len(set(normalized)) != 8:
        raise EvaluationMatrixError(f"{name} must not contain duplicate cards")
    return normalized


def _normalize_domain_randomization(
    value: object,
) -> DomainRandomizationConfig | None:
    """Normalize an optional, explicitly declared evaluation perturbation."""

    if value is None or isinstance(value, DomainRandomizationConfig):
        return value
    if isinstance(value, Mapping):
        try:
            return DomainRandomizationConfig.from_mapping(value)
        except DomainRandomizationError as error:
            raise EvaluationMatrixError(
                f"invalid domain_randomization: {error}"
            ) from error
    raise EvaluationMatrixError(
        "domain_randomization must be a DomainRandomizationConfig, object, or None"
    )


def _deck_composition_key(cards: Sequence[str]) -> tuple[str, ...]:
    """Return an order-independent, JSON-friendly deck identity."""

    return tuple(sorted(str(card) for card in cards))


def _file_fingerprint(path: str | Path) -> dict[str, object]:
    """Describe a file without making an injected test runner require it."""

    candidate = Path(path)
    result: dict[str, object] = {
        "path": str(path),
        "exists": candidate.is_file(),
    }
    if not candidate.is_file():
        return result
    try:
        size = candidate.stat().st_size
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EvaluationMatrixError(
            f"cannot fingerprint evaluation file {candidate}: {error}"
        ) from error
    result["size_bytes"] = int(size)
    result["sha256"] = digest.hexdigest()
    return result


def _troop_positions_by_player(raw_state: object) -> dict[str, list[dict[str, object]]]:
    """Return diagnostic troop coordinates split by authoritative owner."""

    try:
        from .prototype import _troop_locations
    except ImportError:  # pragma: no cover - top-level ``rl`` layout
        from rl.prototype import _troop_locations
    positions = _troop_locations(raw_state)
    return {
        "player_0": [row for row in positions if row.get("owner") == 0],
        "player_1": [row for row in positions if row.get("owner") == 1],
    }


def _json_safe(value: Any, *, path: str = "$", allow_path: bool = True) -> Any:
    """Convert scalar/container values to JSON-safe values, fail closed."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationMatrixError(f"non-finite JSON value at {path}")
        return value
    if allow_path and isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise EvaluationMatrixError(f"JSON object key at {path} is not a string")
            result[key] = _json_safe(child, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    # NumPy scalars are common in custom simulator adapters.  Importing NumPy
    # here would make the orchestration module needlessly non-optional, so use
    # its deliberately small scalar protocol when available.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (TypeError, ValueError):
            converted = None
        if converted is not None and converted is not value:
            return _json_safe(converted, path=path)
    raise EvaluationMatrixError(
        f"value at {path} is not JSON-safe: {type(value).__name__}"
    )


def _json_fingerprint(value: Any) -> str:
    """Return a stable digest for a JSON-shaped evaluation input."""

    safe = _json_safe(value)
    encoded = json.dumps(
        safe,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _utc_timestamp() -> str:
    """Return an unambiguous UTC timestamp for report provenance."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _runtime_provenance() -> dict[str, object]:
    """Describe the host runtime without affecting simulator behavior."""

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "code_revision": code_revision(),
    }


@dataclass(frozen=True, slots=True)
class OpponentDeckSpec:
    """One named eight-card opponent deck in the held-out matrix."""

    deck_id: str
    cards: tuple[str, ...]
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "deck_id", _identifier(self.deck_id, "deck_id"))
        cards = _sequence(self.cards, "cards")
        if len(cards) != 8:
            raise EvaluationMatrixError("cards must contain exactly eight cards")
        normalized_cards = tuple(_identifier(card, f"cards[{index}]") for index, card in enumerate(cards))
        if len(set(normalized_cards)) != len(normalized_cards):
            raise EvaluationMatrixError("cards must not contain duplicates")
        object.__setattr__(self, "cards", normalized_cards)

        tags = _sequence(self.tags, "tags")
        normalized_tags = tuple(_identifier(tag, f"tags[{index}]") for index, tag in enumerate(tags))
        if len(set(normalized_tags)) != len(normalized_tags):
            raise EvaluationMatrixError("tags must not contain duplicates")
        object.__setattr__(self, "tags", normalized_tags)

        if not isinstance(self.metadata, Mapping):
            raise EvaluationMatrixError("metadata must be a string mapping")
        metadata = {
            _identifier(key, "metadata key"): _identifier(value, f"metadata[{key!r}]")
            for key, value in self.metadata.items()
        }
        object.__setattr__(self, "metadata", dict(sorted(metadata.items())))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OpponentDeckSpec":
        if not isinstance(raw, Mapping):
            raise EvaluationMatrixError("opponent deck must be an object")
        return cls(
            deck_id=raw.get("deck_id", ""),
            cards=raw.get("cards", ()),
            tags=raw.get("tags", ()),
            metadata=raw.get("metadata", {}),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "deck_id": self.deck_id,
            "cards": list(self.cards),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


# A compatibility-friendly short name for callers that do not need the
# ``Spec`` suffix.  The explicit class above remains the canonical type.
OpponentDeck = OpponentDeckSpec


OpponentControllerFactory = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class OpponentStrategySpec:
    """Named opponent controller factory.

    ``factory`` is called with the match seed when it accepts one, otherwise
    with no arguments.  It must return either an object exposing
    ``choose_action(engine, state, player)`` or a callable with that same
    signature.  Built-in strategies may omit the factory.
    """

    strategy_id: str
    factory: OpponentControllerFactory | None = field(default=None, repr=False, compare=False)
    description: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        strategy_id = _canonical_strategy_id(self.strategy_id)
        object.__setattr__(self, "strategy_id", strategy_id)
        if self.factory is not None and not callable(self.factory):
            raise EvaluationMatrixError("strategy factory must be callable")
        if self.factory is None and strategy_id not in _BUILTIN_STRATEGIES:
            raise EvaluationMatrixError(
                f"unknown strategy {strategy_id!r}; provide a factory for custom strategies"
            )
        if not isinstance(self.description, str):
            raise EvaluationMatrixError("strategy description must be a string")
        if not isinstance(self.metadata, Mapping):
            raise EvaluationMatrixError("strategy metadata must be a string mapping")
        metadata = {
            _identifier(key, "strategy metadata key"): _identifier(
                value, f"strategy metadata[{key!r}]"
            )
            for key, value in self.metadata.items()
        }
        object.__setattr__(self, "metadata", dict(sorted(metadata.items())))

    @classmethod
    def from_name(cls, name: str) -> "OpponentStrategySpec":
        return cls(strategy_id=name)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OpponentStrategySpec":
        if not isinstance(raw, Mapping):
            raise EvaluationMatrixError("opponent strategy must be an object")
        return cls(
            strategy_id=raw.get("strategy_id", raw.get("name", "")),
            description=raw.get("description", ""),
            metadata=raw.get("metadata", {}),
        )

    def build(self, seed: int) -> Any:
        if self.factory is not None:
            return _invoke_factory(self.factory, seed)
        return _builtin_controller(self.strategy_id, seed)

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "description": self.description,
            "builtin": self.factory is None,
            "metadata": dict(self.metadata),
        }


OpponentStrategy = OpponentStrategySpec


def _invoke_factory(factory: OpponentControllerFactory, seed: int) -> Any:
    """Call a custom factory with seed when its signature permits it."""

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(seed)
    try:
        signature.bind(seed)
    except TypeError:
        return factory()
    return factory(seed)


@dataclass(frozen=True, slots=True)
class EvaluationMatrixConfig:
    """Validated, reproducible configuration for a matrix run."""

    checkpoint: str | Path
    opponent_decks: tuple[OpponentDeckSpec, ...]
    strategies: tuple[OpponentStrategySpec, ...]
    seeds: tuple[int, ...]
    policy_mode: str = "actor"
    target_player: int = 0
    max_decisions: int | None = None
    device: str | None = "auto"
    shuffle_decks: bool = True
    include_match_results: bool = True
    held_out: bool = True
    batch_size: int = 8
    held_out_source: str | Path | None = None
    excluded_deck_compositions: tuple[tuple[str, ...], ...] = ()
    player_deck: tuple[str, ...] | None = None
    domain_randomization: DomainRandomizationConfig | None = None

    def __post_init__(self) -> None:
        checkpoint = self.checkpoint
        if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
            raise EvaluationMatrixError("checkpoint must be a non-empty path")
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(
            self,
            "player_deck",
            _normalize_player_deck(self.player_deck),
        )
        object.__setattr__(
            self,
            "domain_randomization",
            _normalize_domain_randomization(self.domain_randomization),
        )

        decks = _sequence(self.opponent_decks, "opponent_decks")
        normalized_decks = tuple(
            deck if isinstance(deck, OpponentDeckSpec) else OpponentDeckSpec.from_mapping(deck)
            for deck in decks
        )
        if not normalized_decks:
            raise EvaluationMatrixError("opponent_decks must not be empty")
        if len({deck.deck_id for deck in normalized_decks}) != len(normalized_decks):
            raise EvaluationMatrixError("opponent_decks must have unique deck IDs")
        object.__setattr__(self, "opponent_decks", normalized_decks)

        strategies = _sequence(self.strategies, "strategies")
        normalized_strategies = tuple(
            strategy
            if isinstance(strategy, OpponentStrategySpec)
            else OpponentStrategySpec.from_name(strategy)
            if isinstance(strategy, str)
            else OpponentStrategySpec.from_mapping(strategy)
            for strategy in strategies
        )
        if not normalized_strategies:
            raise EvaluationMatrixError("strategies must not be empty")
        if len({strategy.strategy_id for strategy in normalized_strategies}) != len(normalized_strategies):
            raise EvaluationMatrixError("strategies must have unique strategy IDs")
        object.__setattr__(self, "strategies", normalized_strategies)

        seeds = tuple(_integer(seed, f"seeds[{index}]", minimum=0) for index, seed in enumerate(_sequence(self.seeds, "seeds")))
        if not seeds:
            raise EvaluationMatrixError("seeds must not be empty")
        if len(set(seeds)) != len(seeds):
            raise EvaluationMatrixError("seeds must be unique")
        object.__setattr__(self, "seeds", seeds)

        if self.policy_mode not in _POLICY_MODES:
            raise EvaluationMatrixError(
                f"policy_mode must be one of {sorted(_POLICY_MODES)}"
            )
        _integer(self.target_player, "target_player", minimum=0)
        if self.target_player not in (0, 1):
            raise EvaluationMatrixError("target_player must be 0 or 1")
        if self.max_decisions is not None:
            _integer(self.max_decisions, "max_decisions", minimum=1)
        if self.device is not None and (
            not isinstance(self.device, str) or not self.device.strip()
        ):
            raise EvaluationMatrixError("device must be a non-empty string or None")
        if type(self.shuffle_decks) is not bool:
            raise EvaluationMatrixError("shuffle_decks must be boolean")
        if type(self.include_match_results) is not bool:
            raise EvaluationMatrixError("include_match_results must be boolean")
        if type(self.held_out) is not bool:
            raise EvaluationMatrixError("held_out must be boolean")
        _integer(self.batch_size, "batch_size", minimum=1)

        if self.held_out_source is not None and (
            not isinstance(self.held_out_source, (str, Path))
            or not str(self.held_out_source).strip()
        ):
            raise EvaluationMatrixError(
                "held_out_source must be a non-empty path or None"
            )
        raw_exclusions = _sequence(
            self.excluded_deck_compositions,
            "excluded_deck_compositions",
        )
        normalized_exclusions: list[tuple[str, ...]] = []
        for index, raw_deck in enumerate(raw_exclusions):
            cards = _sequence(raw_deck, f"excluded_deck_compositions[{index}]")
            if len(cards) != 8:
                raise EvaluationMatrixError(
                    f"excluded_deck_compositions[{index}] must contain exactly eight cards"
                )
            normalized = tuple(
                _identifier(card, f"excluded_deck_compositions[{index}][{card_index}]")
                for card_index, card in enumerate(cards)
            )
            if len(set(normalized)) != len(normalized):
                raise EvaluationMatrixError(
                    f"excluded_deck_compositions[{index}] must not contain duplicate cards"
                )
            normalized_exclusions.append(_deck_composition_key(normalized))
        normalized_exclusions = sorted(set(normalized_exclusions))
        object.__setattr__(
            self,
            "held_out_source",
            self.held_out_source,
        )
        object.__setattr__(
            self,
            "excluded_deck_compositions",
            tuple(normalized_exclusions),
        )

    @property
    def match_count(self) -> int:
        return len(self.opponent_decks) * len(self.strategies) * len(self.seeds)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvaluationMatrixConfig":
        if not isinstance(raw, Mapping):
            raise EvaluationMatrixError("evaluation matrix config must be an object")
        return cls(
            checkpoint=raw.get("checkpoint", ""),
            opponent_decks=raw.get("opponent_decks", ()),
            strategies=raw.get("strategies", ()),
            seeds=raw.get("seeds", ()),
            policy_mode=raw.get("policy_mode", "actor"),
            target_player=raw.get("target_player", 0),
            max_decisions=raw.get("max_decisions"),
            device=raw.get("device", "auto"),
            shuffle_decks=raw.get("shuffle_decks", True),
            include_match_results=raw.get("include_match_results", True),
            held_out=raw.get("held_out", True),
            batch_size=raw.get("batch_size", 8),
            held_out_source=raw.get("held_out_source"),
            excluded_deck_compositions=raw.get("excluded_deck_compositions", ()),
            player_deck=raw.get("player_deck"),
            domain_randomization=raw.get("domain_randomization"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "checkpoint": str(self.checkpoint),
            "opponent_decks": [deck.as_dict() for deck in self.opponent_decks],
            "strategies": [strategy.as_dict() for strategy in self.strategies],
            "seeds": list(self.seeds),
            "policy_mode": self.policy_mode,
            "target_player": self.target_player,
            "max_decisions": self.max_decisions,
            "device": self.device,
            "shuffle_decks": self.shuffle_decks,
            "include_match_results": self.include_match_results,
            "held_out": self.held_out,
            "batch_size": self.batch_size,
            "held_out_source": (
                None if self.held_out_source is None else str(self.held_out_source)
            ),
            "excluded_deck_compositions": [
                list(cards) for cards in self.excluded_deck_compositions
            ],
            "player_deck": list(self.player_deck),
            "domain_randomization": (
                None
                if self.domain_randomization is None
                else self.domain_randomization.as_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class MatchSpec:
    """One immutable matrix cell passed to a match runner."""

    checkpoint: str | Path
    opponent_deck: OpponentDeckSpec
    strategy: OpponentStrategySpec
    seed: int
    policy_mode: str
    target_player: int
    max_decisions: int | None
    device: str | None
    shuffle_decks: bool
    player_deck: tuple[str, ...] | None = None
    domain_randomization: DomainRandomizationConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "player_deck",
            _normalize_player_deck(self.player_deck),
        )
        object.__setattr__(
            self,
            "domain_randomization",
            _normalize_domain_randomization(self.domain_randomization),
        )

    @property
    def cell_id(self) -> str:
        """Stable identity for one deck × strategy × seed matrix cell."""

        # The separators are intentionally only presentation; the structured
        # fields below remain in every report row so the ID never has to be
        # parsed to recover an evaluation input.
        return f"{self.opponent_deck.deck_id}::{self.strategy.strategy_id}::seed-{self.seed}"

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "checkpoint": str(self.checkpoint),
            "deck_id": self.opponent_deck.deck_id,
            "deck_cards": list(self.opponent_deck.cards),
            "opponent_deck": self.opponent_deck.as_dict(),
            "strategy_id": self.strategy.strategy_id,
            "opponent_strategy": self.strategy.as_dict(),
            "seed": self.seed,
            "policy_mode": self.policy_mode,
            "target_player": self.target_player,
            "actor_player": self.target_player,
            "opponent_player": 1 - self.target_player,
            "max_decisions": self.max_decisions,
            "device": self.device,
            "shuffle_decks": self.shuffle_decks,
            "player_deck": list(self.player_deck),
            "domain_randomization": (
                None
                if self.domain_randomization is None
                else self.domain_randomization.as_dict()
            ),
        }


MatchRunner = Callable[[MatchSpec], Any]


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Learner-perspective result for one completed or truncated match."""

    outcome: str
    decisions: int = 0
    return_value: float = 0.0
    winner: int | None = None
    terminal_reason: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in {"win", "loss", "draw", "truncated"}:
            raise EvaluationMatrixError(f"unsupported match outcome: {self.outcome!r}")
        _integer(self.decisions, "decisions", minimum=0)
        if isinstance(self.return_value, bool) or not isinstance(
            self.return_value, (int, float)
        ) or not math.isfinite(float(self.return_value)):
            raise EvaluationMatrixError("return_value must be finite")
        if self.winner is not None and self.winner not in (0, 1):
            raise EvaluationMatrixError("winner must be 0, 1, or None")
        if self.terminal_reason is not None and not isinstance(self.terminal_reason, str):
            raise EvaluationMatrixError("terminal_reason must be a string or None")
        if not isinstance(self.metrics, Mapping):
            raise EvaluationMatrixError("metrics must be a mapping")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        target_player: int,
    ) -> "MatchResult":
        if not isinstance(raw, Mapping):
            raise EvaluationMatrixError("match runner result must be an object")
        raw_outcome = raw.get("outcome")
        if raw_outcome is None:
            if bool(raw.get("truncated", False)):
                raw_outcome = "truncated"
            else:
                winner = raw.get("winner")
                raw_outcome = (
                    "win"
                    if winner == target_player
                    else "loss"
                    if winner == 1 - target_player
                    else "draw"
                )
        outcome = str(raw_outcome).casefold()
        if outcome in {"won", "win"}:
            outcome = "win"
        elif outcome in {"lost", "loss"}:
            outcome = "loss"
        elif outcome in {"draw", "tie"}:
            outcome = "draw"
        elif outcome in {"truncate", "truncated", "timeout", "censored"}:
            outcome = "truncated"
        metrics = raw.get("metrics", {})
        return cls(
            outcome=outcome,
            decisions=raw.get("decisions", raw.get("length", 0)),
            return_value=raw.get("return", raw.get("return_value", 0.0)),
            winner=raw.get("winner"),
            terminal_reason=raw.get("terminal_reason"),
            metrics=metrics,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "decisions": self.decisions,
            "return": float(self.return_value),
            "winner": self.winner,
            "terminal_reason": self.terminal_reason,
            "metrics": _json_safe(dict(self.metrics)),
        }


def _normalize_match_result(raw: Any, *, target_player: int) -> MatchResult:
    if isinstance(raw, MatchResult):
        return raw
    if isinstance(raw, str):
        return MatchResult(outcome=raw.casefold())
    if isinstance(raw, Mapping):
        return MatchResult.from_mapping(raw, target_player=target_player)
    raise EvaluationMatrixError(
        "match runner must return MatchResult, an outcome string, or a mapping"
    )


def _summary(results: Sequence[MatchResult]) -> dict[str, object]:
    wins = sum(result.outcome == "win" for result in results)
    losses = sum(result.outcome == "loss" for result in results)
    draws = sum(result.outcome == "draw" for result in results)
    truncated = sum(result.outcome == "truncated" for result in results)
    completed = wins + losses + draws
    matches = len(results)
    terminal_reasons: dict[str, int] = {}
    plays_by_card: dict[str, int] = {}
    rejected_actions = 0
    opponent_rejected_actions = 0
    traced_steps = 0
    crown_totals = {"player_0": 0, "player_1": 0}
    crown_rows = 0
    for result in results:
        reason = result.terminal_reason or "<unspecified>"
        terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
        metrics = result.metrics
        if not isinstance(metrics, Mapping):  # pragma: no cover - validated above
            continue
        raw_plays = metrics.get("target_plays_by_card", {})
        if isinstance(raw_plays, Mapping):
            for card, count in raw_plays.items():
                if isinstance(card, str) and type(count) is int and count >= 0:
                    plays_by_card[card] = plays_by_card.get(card, 0) + count
        raw_rejected = metrics.get("target_rejected_actions", 0)
        if type(raw_rejected) is int and raw_rejected >= 0:
            rejected_actions += raw_rejected
        raw_opponent_rejected = metrics.get("opponent_rejected_actions", 0)
        if type(raw_opponent_rejected) is int and raw_opponent_rejected >= 0:
            opponent_rejected_actions += raw_opponent_rejected
        raw_trace = metrics.get("target_play_trace", ())
        if isinstance(raw_trace, (list, tuple)):
            traced_steps += len(raw_trace)
        raw_crowns = metrics.get("crowns_end")
        if isinstance(raw_crowns, Mapping):
            values = [raw_crowns.get(player) for player in crown_totals]
            if all(type(value) is int and 0 <= value <= 3 for value in values):
                crown_rows += 1
                for player in crown_totals:
                    crown_totals[player] += int(raw_crowns[player])

    win_rate_interval: dict[str, float] | None
    if completed:
        # Wilson's interval behaves sensibly for small samples and at 0/1,
        # unlike a normal approximation.  Evaluation reports intentionally do
        # not depend on scipy so they remain usable in the simulator runtime.
        z = 1.959963984540054
        denominator = 1.0 + (z * z / completed)
        center = (wins / completed + (z * z / (2.0 * completed))) / denominator
        radius = (
            z
            * math.sqrt(
                (wins / completed) * (1.0 - wins / completed) / completed
                + (z * z / (4.0 * completed * completed))
            )
            / denominator
        )
        win_rate_interval = {
            "confidence": 0.95,
            "low": max(0.0, center - radius),
            "high": min(1.0, center + radius),
        }
    else:
        win_rate_interval = None
    return {
        "matches": matches,
        "completed": completed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "truncated": truncated,
        "win_rate": wins / completed if completed else 0.0,
        "win_rate_ci95": win_rate_interval,
        "completion_rate": completed / matches if matches else 0.0,
        "truncation_rate": truncated / matches if matches else 0.0,
        "all_wins": matches > 0 and wins == matches,
        "all_completed_wins": matches > 0 and completed == matches and wins == matches,
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
        "decisions_total": sum(result.decisions for result in results),
        "decisions_mean": (
            sum(result.decisions for result in results) / matches if matches else 0.0
        ),
        "decisions_min": min((result.decisions for result in results), default=0),
        "decisions_max": max((result.decisions for result in results), default=0),
        "target_plays_by_card": dict(sorted(plays_by_card.items())),
        "target_rejected_actions": rejected_actions,
        "opponent_rejected_actions": opponent_rejected_actions,
        "target_play_trace_entries": traced_steps,
        "crowns_end": crown_totals if crown_rows else None,
        "crowns_end_matches": crown_rows,
    }


def _evaluation_quality_gate(
    report: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, object]:
    """Return the fail-closed integrity gate for a promotion candidate.

    Win rate is deliberately reported as evidence only.  The gate checks that
    the evaluation artifact itself is trustworthy: it must be a verified
    held-out actor run with complete matches, no rejected actions, a public
    actor boundary, and a clean simulator-exploit audit.
    """

    total = report.get("total")
    if not isinstance(total, Mapping):
        total = {}
    held_out_audit = report.get("held_out_audit")
    if not isinstance(held_out_audit, Mapping):
        held_out_audit = {}

    matches = total.get("matches")
    truncated = total.get("truncated")
    target_rejected = total.get("target_rejected_actions")
    opponent_rejected = total.get("opponent_rejected_actions")
    held_out = report.get("held_out") is True
    held_out_split_verified = (
        held_out
        and held_out_audit.get("disjointness_verified") is True
        and held_out_audit.get("overlap") == []
    )
    actor_mode = report.get("policy_mode") == "actor"
    checks = {
        "non_empty_matrix": type(matches) is int and matches > 0,
        "held_out_evaluation": held_out,
        "held_out_split_verified": held_out_split_verified,
        "complete_matches": type(truncated) is int and truncated == 0,
        "no_rejected_actions": (
            type(target_rejected) is int
            and type(opponent_rejected) is int
            and target_rejected == 0
            and opponent_rejected == 0
        ),
        "actor_controls_actions": (
            not actor_mode or report.get("actor_controls_actions") is True
        ),
        "actor_public_inputs": (
            not actor_mode or report.get("actor_privileged_inputs") is False
        ),
        "simulation_exploit_audit_clean": audit.get("status") == "clean",
    }

    failures: list[str] = []
    if not checks["non_empty_matrix"]:
        failures.append("empty_matrix")
    if not checks["held_out_evaluation"]:
        failures.append("not_held_out")
    elif not checks["held_out_split_verified"]:
        failures.append("held_out_split_not_verified")
    if not checks["complete_matches"]:
        failures.append("truncated_matches")
    if not checks["no_rejected_actions"]:
        failures.append("rejected_actions")
    if not checks["actor_controls_actions"]:
        failures.append("actor_does_not_control_actions")
    if not checks["actor_public_inputs"]:
        failures.append("actor_privileged_input")
    if not checks["simulation_exploit_audit_clean"]:
        failures.append("simulation_exploit_audit")

    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "strength_evidence": {
            "win_rate": total.get("win_rate"),
            "wins": total.get("wins"),
            "completed": total.get("completed"),
            "used_as_gate": False,
        },
    }


_MISSING_REPORT_VALUE = object()
_OUTCOME_SCORES = {"loss": -1, "draw": 0, "win": 1}
_COMPARISON_PROVENANCE_FIELDS = (
    "kind",
    "schema_version",
    "policy_mode",
    "actor_controls_actions",
    "target_player",
    "player_deck",
    "opponent_decks",
    "strategies",
    "seeds",
    "max_decisions",
    "shuffle_decks",
    "domain_randomization",
    "held_out",
    "held_out_audit",
    "runner",
)
_REQUIRED_COMPARISON_PROVENANCE_FIELDS = (
    *_COMPARISON_PROVENANCE_FIELDS,
    "checkpoint_fingerprint",
)


def _comparison_report_field(
    report: Mapping[str, Any],
    field: str,
) -> Any:
    """Read a report field, accepting the duplicated config representation."""

    if field in report:
        return report[field]
    config = report.get("config")
    if isinstance(config, Mapping) and field in config:
        return config[field]
    return _MISSING_REPORT_VALUE


def _comparison_display_value(value: Any) -> Any:
    """Make a diagnostic value JSON-safe without exposing an internal sentinel."""

    if value is _MISSING_REPORT_VALUE:
        return "<missing>"
    return _json_safe(value)


def _comparison_values_equal(left: Any, right: Any) -> bool:
    if left is _MISSING_REPORT_VALUE or right is _MISSING_REPORT_VALUE:
        return left is right
    left_safe = _json_safe(left)
    right_safe = _json_safe(right)
    return json.dumps(left_safe, sort_keys=True, separators=(",", ":")) == json.dumps(
        right_safe,
        sort_keys=True,
        separators=(",", ":"),
    )


def _comparison_axis_value(
    report: Mapping[str, Any],
    field: str,
    *,
    max_cells: int,
) -> Any:
    value = _comparison_report_field(report, field)
    if value is _MISSING_REPORT_VALUE:
        return value
    if field in {"opponent_decks", "strategies", "seeds"}:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise EvaluationMatrixError(
                f"report field {field!r} must be a sequence"
            )
        if len(value) > max_cells:
            raise EvaluationMatrixError(
                f"report field {field!r} contains {len(value)} entries; "
                f"comparison limit is {max_cells}"
            )
    return value


def _comparison_provenance(
    report: Mapping[str, Any],
    *,
    max_cells: int,
) -> dict[str, Any]:
    """Extract only provenance shared by paired cells.

    Checkpoint identity is intentionally not part of the equality snapshot:
    comparing a candidate against a baseline normally means that those
    fingerprints differ.  Both identities are retained in the returned
    comparison and are required to be present by the quality gate.
    """

    snapshot: dict[str, Any] = {}
    for field in _COMPARISON_PROVENANCE_FIELDS:
        value = _comparison_axis_value(report, field, max_cells=max_cells)
        if field == "runner" and value is not _MISSING_REPORT_VALUE:
            if not isinstance(value, Mapping):
                raise EvaluationMatrixError("report field 'runner' must be an object")
            # Runner metadata can contain volatile paths or implementation
            # details.  These fields identify the simulator provenance that
            # must agree for a paired comparison.
            runner = {
                key: value[key]
                for key in (
                    "runner",
                    "checkpoint_format",
                    "ruleset_id",
                    "ruleset_hash",
                )
                if key in value
            }
            snapshot[field] = runner
        elif field == "held_out_audit" and value is not _MISSING_REPORT_VALUE:
            if not isinstance(value, Mapping):
                raise EvaluationMatrixError(
                    "report field 'held_out_audit' must be an object"
                )
            # The source path and exclusion list can legitimately differ when
            # two checkpoints were trained on different curricula.  Paired
            # evaluation only requires that the selected cells are identical
            # (checked separately through their deck signatures) and that each
            # report independently certifies its own held-out split.
            snapshot[field] = {
                key: value[key]
                for key in (
                    "selected_deck_compositions",
                    "overlap",
                    "disjointness_verified",
                )
                if key in value
            }
        else:
            snapshot[field] = value
    return snapshot


def _comparison_provenance_differences(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for field in sorted(set(baseline) | set(candidate)):
        left = baseline.get(field, _MISSING_REPORT_VALUE)
        right = candidate.get(field, _MISSING_REPORT_VALUE)
        if _comparison_values_equal(left, right):
            continue
        differences.append(
            {
                "field": field,
                "baseline": _comparison_display_value(left),
                "candidate": _comparison_display_value(right),
            }
        )
    return differences


def _comparison_target_player(report: Mapping[str, Any]) -> int:
    value = _comparison_report_field(report, "target_player")
    if value is _MISSING_REPORT_VALUE:
        # This default keeps the low-level comparator useful for small,
        # hand-written reports; a production quality gate still reports the
        # missing provenance field below.
        return 0
    if type(value) is not int or value not in (0, 1):
        raise EvaluationMatrixError("report target_player must be 0 or 1")
    return value


def _comparison_nested_row_value(
    row: Mapping[str, Any],
    direct_field: str,
    nested_field: str,
) -> Any:
    value = row.get(direct_field, _MISSING_REPORT_VALUE)
    if value is not _MISSING_REPORT_VALUE:
        return value
    nested = row.get("opponent_deck", _MISSING_REPORT_VALUE)
    if direct_field == "strategy_id":
        nested = row.get("opponent_strategy", _MISSING_REPORT_VALUE)
    if isinstance(nested, Mapping):
        return nested.get(nested_field, _MISSING_REPORT_VALUE)
    return _MISSING_REPORT_VALUE


def _comparison_cell_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    """Capture cell inputs while deliberately excluding checkpoint identity."""

    return {
        "deck_id": _comparison_nested_row_value(row, "deck_id", "deck_id"),
        "deck_cards": _comparison_nested_row_value(row, "deck_cards", "cards"),
        "strategy_id": _comparison_nested_row_value(
            row,
            "strategy_id",
            "strategy_id",
        ),
        "seed": row.get("seed", _MISSING_REPORT_VALUE),
        "policy_mode": row.get("policy_mode", _MISSING_REPORT_VALUE),
        "target_player": row.get("target_player", _MISSING_REPORT_VALUE),
        "max_decisions": row.get("max_decisions", _MISSING_REPORT_VALUE),
        "shuffle_decks": row.get("shuffle_decks", _MISSING_REPORT_VALUE),
        "player_deck": row.get("player_deck", _MISSING_REPORT_VALUE),
        "domain_randomization": row.get(
            "domain_randomization",
            _MISSING_REPORT_VALUE,
        ),
    }


def _comparison_rejected_metric(
    row: Mapping[str, Any],
    metrics: Mapping[str, Any],
    keys: Sequence[str],
    trace_keys: Sequence[str],
    *,
    side: str,
    cell_id: str,
) -> tuple[int, list[dict[str, Any]]]:
    """Read rejection counts and cross-check optional action traces."""

    values: list[int] = []
    issues: list[dict[str, Any]] = []
    for source_name, source in (("metrics", metrics), ("row", row)):
        for key in keys:
            if key not in source:
                continue
            value = source[key]
            if type(value) is not int or value < 0:
                issues.append(
                    {
                        "side": side,
                        "cell_id": cell_id,
                        "kind": "invalid_rejected_action_count",
                        "field": f"{source_name}.{key}",
                        "value": _comparison_display_value(value),
                    }
                )
                continue
            values.append(value)

    traced_rejections = 0
    for trace_key in trace_keys:
        if trace_key not in metrics:
            continue
        trace = metrics[trace_key]
        if not isinstance(trace, (list, tuple)):
            issues.append(
                {
                    "side": side,
                    "cell_id": cell_id,
                    "kind": "invalid_action_trace",
                    "field": f"metrics.{trace_key}",
                    "value": _comparison_display_value(trace),
                }
            )
            continue
        traced_rejections += sum(
            isinstance(item, Mapping) and item.get("accepted") is False
            for item in trace
        )
    values.append(traced_rejections)
    return max(values, default=0), issues


def _comparison_report_rows(
    report: Mapping[str, Any],
    *,
    side: str,
    max_cells: int,
) -> dict[str, Any]:
    """Validate and index per-cell rows, refusing unbounded input."""

    raw_matches = report.get("matches", _MISSING_REPORT_VALUE)
    if raw_matches is _MISSING_REPORT_VALUE:
        # Reports written with include_match_results=False omit the top-level
        # list, but older callers may retain match rows nested under matchup.
        raw_matchups = report.get("matchups", _MISSING_REPORT_VALUE)
        if not isinstance(raw_matchups, (list, tuple)):
            raise EvaluationMatrixError(
                f"{side} report must include per-cell 'matches' rows"
            )
        if len(raw_matchups) > max_cells:
            raise EvaluationMatrixError(
                f"{side} report has too many matchup groups for comparison"
            )
        nested_matches: list[Any] = []
        for matchup in raw_matchups:
            if not isinstance(matchup, Mapping):
                raise EvaluationMatrixError(
                    f"{side} report matchup must be an object"
                )
            nested = matchup.get("matches", ())
            if not isinstance(nested, (list, tuple)):
                raise EvaluationMatrixError(
                    f"{side} report matchup 'matches' must be a sequence"
                )
            nested_matches.extend(nested)
            if len(nested_matches) > max_cells:
                raise EvaluationMatrixError(
                    f"{side} report contains more than {max_cells} cells"
                )
        raw_matches = nested_matches
    if not isinstance(raw_matches, (list, tuple)):
        raise EvaluationMatrixError(f"{side} report 'matches' must be a sequence")
    if len(raw_matches) == 0:
        raise EvaluationMatrixError(f"{side} report contains no per-cell matches")
    if len(raw_matches) > max_cells:
        raise EvaluationMatrixError(
            f"{side} report contains {len(raw_matches)} cells; "
            f"comparison limit is {max_cells}"
        )

    target_player = _comparison_target_player(report)
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    issues: list[dict[str, Any]] = []
    safety_issues: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_matches):
        if not isinstance(raw_row, Mapping):
            raise EvaluationMatrixError(
                f"{side} report match row {index} must be an object"
            )
        raw_cell_id = raw_row.get("cell_id")
        if not isinstance(raw_cell_id, str) or not raw_cell_id.strip():
            raise EvaluationMatrixError(
                f"{side} report match row {index} has no non-empty cell_id"
            )
        cell_id = raw_cell_id.strip()
        if cell_id in rows:
            raise EvaluationMatrixError(
                f"{side} report contains duplicate cell_id {cell_id!r}"
            )
        metrics = raw_row.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise EvaluationMatrixError(
                f"{side} report cell {cell_id!r} metrics must be an object"
            )
        result = _normalize_match_result(raw_row, target_player=target_player)
        target_rejected, target_issues = _comparison_rejected_metric(
            raw_row,
            metrics,
            ("target_rejected_actions", "rejected_actions"),
            ("target_play_trace",),
            side=side,
            cell_id=cell_id,
        )
        opponent_rejected, opponent_issues = _comparison_rejected_metric(
            raw_row,
            metrics,
            ("opponent_rejected_actions",),
            ("opponent_play_trace",),
            side=side,
            cell_id=cell_id,
        )
        safety_issues.extend(target_issues)
        safety_issues.extend(opponent_issues)
        rows[cell_id] = {
            "cell_id": cell_id,
            "result": result,
            "signature": _comparison_cell_signature(raw_row),
            "target_rejected_actions": target_rejected,
            "opponent_rejected_actions": opponent_rejected,
        }
        order.append(cell_id)

    declared_ids = report.get("cell_ids", _MISSING_REPORT_VALUE)
    if declared_ids is not _MISSING_REPORT_VALUE:
        if not isinstance(declared_ids, (list, tuple)):
            raise EvaluationMatrixError(f"{side} report 'cell_ids' must be a sequence")
        if len(declared_ids) > max_cells:
            raise EvaluationMatrixError(
                f"{side} report declares more than {max_cells} cell_ids"
            )
        normalized_declared = []
        for index, value in enumerate(declared_ids):
            if not isinstance(value, str) or not value.strip():
                raise EvaluationMatrixError(
                    f"{side} report cell_ids[{index}] must be non-empty"
                )
            normalized_declared.append(value.strip())
        if len(set(normalized_declared)) != len(normalized_declared):
            raise EvaluationMatrixError(f"{side} report contains duplicate cell_ids")
        if set(normalized_declared) != set(order):
            issues.append(
                {
                    "side": side,
                    "kind": "declared_cell_ids_mismatch",
                    "declared": normalized_declared,
                    "rows": list(order),
                }
            )

    declared_size = report.get("matrix_size", _MISSING_REPORT_VALUE)
    if declared_size is not _MISSING_REPORT_VALUE:
        if type(declared_size) is not int or declared_size < 0:
            raise EvaluationMatrixError(
                f"{side} report matrix_size must be a non-negative integer"
            )
        if declared_size != len(order):
            issues.append(
                {
                    "side": side,
                    "kind": "declared_matrix_size_mismatch",
                    "declared": declared_size,
                    "rows": len(order),
                }
            )

    return {
        "rows": rows,
        "order": order,
        "issues": issues,
        "safety_issues": safety_issues,
    }


def _comparison_outcome_counts(
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    counts = {"win": 0, "loss": 0, "draw": 0, "truncated": 0}
    for row in rows.values():
        counts[row["result"].outcome] += 1
    return counts


def _comparison_safety(
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    truncated_ids = [
        cell_id
        for cell_id, row in rows.items()
        if row["result"].outcome == "truncated"
    ]
    target_rejected_ids = [
        cell_id
        for cell_id, row in rows.items()
        if row["target_rejected_actions"] > 0
    ]
    opponent_rejected_ids = [
        cell_id
        for cell_id, row in rows.items()
        if row["opponent_rejected_actions"] > 0
    ]
    target_total = sum(row["target_rejected_actions"] for row in rows.values())
    opponent_total = sum(row["opponent_rejected_actions"] for row in rows.values())
    return {
        "matches": len(rows),
        "truncated": len(truncated_ids),
        "truncated_cell_ids": truncated_ids,
        "target_rejected_actions": target_total,
        "target_rejected_cell_ids": target_rejected_ids,
        "opponent_rejected_actions": opponent_total,
        "opponent_rejected_cell_ids": opponent_rejected_ids,
        "rejected_actions": target_total + opponent_total,
    }


def compare_evaluation_reports(
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    max_cells: int = DEFAULT_REPORT_COMPARISON_MAX_CELLS,
    reject_truncations: bool = True,
    reject_rejected_actions: bool = True,
) -> dict[str, object]:
    """Compare two bounded, paired evaluation reports.

    Reports are paired by their stable ``cell_id`` rather than by list order.
    The helper never runs a match and refuses reports larger than
    ``max_cells``.  A valid comparison includes outcome deltas for the
    intersection of the cells; the quality gate passes only when the cell
    sets and cell inputs agree, shared provenance agrees, and the configured
    truncation/rejection checks are clean.  Different checkpoint fingerprints
    are expected and are reported as identities, not as a provenance mismatch.

    Malformed reports and reports without per-cell match rows raise
    :class:`EvaluationMatrixError`.  Valid but non-paired reports return a
    failed gate with bounded diagnostics so callers can surface the reason.
    """

    if not isinstance(baseline_report, Mapping):
        raise TypeError("baseline_report must be a mapping")
    if not isinstance(candidate_report, Mapping):
        raise TypeError("candidate_report must be a mapping")
    if type(max_cells) is not int or max_cells < 1:
        raise ValueError("max_cells must be a positive integer")
    if type(reject_truncations) is not bool:
        raise TypeError("reject_truncations must be boolean")
    if type(reject_rejected_actions) is not bool:
        raise TypeError("reject_rejected_actions must be boolean")

    baseline_rows = _comparison_report_rows(
        baseline_report,
        side="baseline",
        max_cells=max_cells,
    )
    candidate_rows = _comparison_report_rows(
        candidate_report,
        side="candidate",
        max_cells=max_cells,
    )
    baseline_index = baseline_rows["rows"]
    candidate_index = candidate_rows["rows"]
    baseline_order = baseline_rows["order"]
    candidate_ids = set(candidate_index)
    baseline_ids = set(baseline_index)
    paired_ids = [cell_id for cell_id in baseline_order if cell_id in candidate_ids]
    baseline_only = [cell_id for cell_id in baseline_order if cell_id not in candidate_ids]
    candidate_only = [
        cell_id for cell_id in candidate_rows["order"] if cell_id not in baseline_ids
    ]

    cell_definition_mismatches: list[dict[str, Any]] = []
    for cell_id in paired_ids:
        left_signature = baseline_index[cell_id]["signature"]
        right_signature = candidate_index[cell_id]["signature"]
        fields: list[dict[str, Any]] = []
        for field in sorted(set(left_signature) | set(right_signature)):
            left = left_signature.get(field, _MISSING_REPORT_VALUE)
            right = right_signature.get(field, _MISSING_REPORT_VALUE)
            if _comparison_values_equal(left, right):
                continue
            fields.append(
                {
                    "field": field,
                    "baseline": _comparison_display_value(left),
                    "candidate": _comparison_display_value(right),
                }
            )
        if fields:
            cell_definition_mismatches.append(
                {"cell_id": cell_id, "fields": fields}
            )

    structural_mismatches = [
        *baseline_rows["issues"],
        *candidate_rows["issues"],
    ]
    cell_alignment = {
        "identical": not (
            baseline_only
            or candidate_only
            or cell_definition_mismatches
            or structural_mismatches
        ),
        "baseline_count": len(baseline_index),
        "candidate_count": len(candidate_index),
        "paired_count": len(paired_ids),
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "cell_definition_mismatches": cell_definition_mismatches,
        "structural_mismatches": structural_mismatches,
    }

    baseline_provenance = _comparison_provenance(
        baseline_report,
        max_cells=max_cells,
    )
    candidate_provenance = _comparison_provenance(
        candidate_report,
        max_cells=max_cells,
    )
    provenance_mismatches = _comparison_provenance_differences(
        baseline_provenance,
        candidate_provenance,
    )
    missing_provenance: list[dict[str, str]] = []
    for side, report in (
        ("baseline", baseline_report),
        ("candidate", candidate_report),
    ):
        for field in _REQUIRED_COMPARISON_PROVENANCE_FIELDS:
            value = (
                report.get(field, _MISSING_REPORT_VALUE)
                if field == "checkpoint_fingerprint"
                else _comparison_report_field(report, field)
            )
            if value is _MISSING_REPORT_VALUE:
                missing_provenance.append({"side": side, "field": field})

    baseline_safety = _comparison_safety(baseline_index)
    candidate_safety = _comparison_safety(candidate_index)
    truncated_cell_ids = sorted(
        set(baseline_safety["truncated_cell_ids"])
        | set(candidate_safety["truncated_cell_ids"])
    )
    invalid_safety_metrics = [
        *baseline_rows["safety_issues"],
        *candidate_rows["safety_issues"],
    ]
    rejected_actions_present = bool(
        baseline_safety["rejected_actions"]
        or candidate_safety["rejected_actions"]
        or invalid_safety_metrics
    )

    per_cell: list[dict[str, Any]] = []
    transition_counts: dict[str, int] = {}
    improved = 0
    regressed = 0
    unchanged = 0
    comparable = 0
    outcome_delta_total = 0
    for cell_id in paired_ids:
        baseline_row = baseline_index[cell_id]
        candidate_row = candidate_index[cell_id]
        baseline_result = baseline_row["result"]
        candidate_result = candidate_row["result"]
        baseline_outcome = baseline_result.outcome
        candidate_outcome = candidate_result.outcome
        transition = f"{baseline_outcome}->{candidate_outcome}"
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
        if (
            baseline_outcome in _OUTCOME_SCORES
            and candidate_outcome in _OUTCOME_SCORES
        ):
            outcome_delta: int | None = (
                _OUTCOME_SCORES[candidate_outcome]
                - _OUTCOME_SCORES[baseline_outcome]
            )
            comparable += 1
            outcome_delta_total += outcome_delta
            if outcome_delta > 0:
                improved += 1
            elif outcome_delta < 0:
                regressed += 1
            else:
                unchanged += 1
            improved_value: bool | None = outcome_delta > 0
            regressed_value: bool | None = outcome_delta < 0
        else:
            # A truncation is not an outcome score and must not be presented
            # as an improvement or regression.
            outcome_delta = None
            improved_value = None
            regressed_value = None

        per_cell.append(
            {
                "cell_id": cell_id,
                "baseline_outcome": baseline_outcome,
                "candidate_outcome": candidate_outcome,
                "outcome_transition": transition,
                "outcome_delta": outcome_delta,
                "improved": improved_value,
                "regressed": regressed_value,
                "baseline_decisions": baseline_result.decisions,
                "candidate_decisions": candidate_result.decisions,
                "decisions_delta": candidate_result.decisions - baseline_result.decisions,
                "baseline_return": float(baseline_result.return_value),
                "candidate_return": float(candidate_result.return_value),
                "return_delta": float(
                    candidate_result.return_value - baseline_result.return_value
                ),
                "baseline_truncated": baseline_outcome == "truncated",
                "candidate_truncated": candidate_outcome == "truncated",
                "baseline_target_rejected_actions": baseline_row[
                    "target_rejected_actions"
                ],
                "candidate_target_rejected_actions": candidate_row[
                    "target_rejected_actions"
                ],
                "baseline_opponent_rejected_actions": baseline_row[
                    "opponent_rejected_actions"
                ],
                "candidate_opponent_rejected_actions": candidate_row[
                    "opponent_rejected_actions"
                ],
            }
        )

    baseline_checkpoint = baseline_report.get(
        "checkpoint_fingerprint",
        _MISSING_REPORT_VALUE,
    )
    candidate_checkpoint = candidate_report.get(
        "checkpoint_fingerprint",
        _MISSING_REPORT_VALUE,
    )
    provenance_complete = not missing_provenance
    checkpoint_identities_equal = _comparison_values_equal(
        baseline_checkpoint,
        candidate_checkpoint,
    )
    safety = {
        "truncations": {
            "baseline": baseline_safety["truncated"],
            "candidate": candidate_safety["truncated"],
            "either": len(truncated_cell_ids),
            "cell_ids": truncated_cell_ids,
        },
        "rejected_actions": {
            "baseline": {
                "target": baseline_safety["target_rejected_actions"],
                "opponent": baseline_safety["opponent_rejected_actions"],
                "total": baseline_safety["rejected_actions"],
                "cell_ids": sorted(
                    set(baseline_safety["target_rejected_cell_ids"])
                    | set(baseline_safety["opponent_rejected_cell_ids"])
                ),
                "invalid_metrics": baseline_rows["safety_issues"],
            },
            "candidate": {
                "target": candidate_safety["target_rejected_actions"],
                "opponent": candidate_safety["opponent_rejected_actions"],
                "total": candidate_safety["rejected_actions"],
                "cell_ids": sorted(
                    set(candidate_safety["target_rejected_cell_ids"])
                    | set(candidate_safety["opponent_rejected_cell_ids"])
                ),
                "invalid_metrics": candidate_rows["safety_issues"],
            },
            "any": rejected_actions_present,
            "invalid_metrics": invalid_safety_metrics,
        },
    }
    summary = {
        "paired_cells": len(paired_ids),
        "comparable_cells": comparable,
        "baseline": _comparison_outcome_counts(baseline_index),
        "candidate": _comparison_outcome_counts(candidate_index),
        "wins_delta": (
            _comparison_outcome_counts(candidate_index)["win"]
            - _comparison_outcome_counts(baseline_index)["win"]
        ),
        "losses_delta": (
            _comparison_outcome_counts(candidate_index)["loss"]
            - _comparison_outcome_counts(baseline_index)["loss"]
        ),
        "draws_delta": (
            _comparison_outcome_counts(candidate_index)["draw"]
            - _comparison_outcome_counts(baseline_index)["draw"]
        ),
        "truncated_delta": (
            _comparison_outcome_counts(candidate_index)["truncated"]
            - _comparison_outcome_counts(baseline_index)["truncated"]
        ),
        "improved_cells": improved,
        "regressed_cells": regressed,
        "unchanged_cells": unchanged,
        "outcome_delta_total": outcome_delta_total,
        "outcome_delta_mean": (
            outcome_delta_total / comparable if comparable else 0.0
        ),
        "outcome_transitions": dict(sorted(transition_counts.items())),
    }

    failures: list[str] = []
    if not cell_alignment["identical"]:
        failures.append("cells_not_identical")
    if provenance_mismatches:
        failures.append("provenance_mismatch")
    if not provenance_complete:
        failures.append("incomplete_provenance")
    if reject_truncations and safety["truncations"]["either"]:
        failures.append("truncated_matches")
    if reject_rejected_actions and safety["rejected_actions"]["any"]:
        failures.append("rejected_actions")
    if invalid_safety_metrics:
        failures.append("invalid_rejected_action_metrics")
    quality_gate = {
        "passed": not failures,
        "failures": failures,
        "checks": {
            "bounded": True,
            "identical_cells": cell_alignment["identical"],
            "provenance_match": not provenance_mismatches,
            "provenance_complete": provenance_complete,
            "checkpoint_provenance_present": (
                baseline_checkpoint is not _MISSING_REPORT_VALUE
                and candidate_checkpoint is not _MISSING_REPORT_VALUE
            ),
            "no_truncations": safety["truncations"]["either"] == 0,
            "no_rejected_actions": not safety["rejected_actions"]["any"],
        },
        "limits": {"max_cells": max_cells},
        "reject_truncations": reject_truncations,
        "reject_rejected_actions": reject_rejected_actions,
    }

    comparison: dict[str, Any] = {
        "kind": REPORT_COMPARISON_KIND,
        "schema_version": REPORT_COMPARISON_SCHEMA_VERSION,
        "baseline_checkpoint_fingerprint": _comparison_display_value(
            baseline_checkpoint
        ),
        "candidate_checkpoint_fingerprint": _comparison_display_value(
            candidate_checkpoint
        ),
        "checkpoint_identities_equal": checkpoint_identities_equal,
        "provenance": {
            "match": not provenance_mismatches,
            "complete": provenance_complete,
            "baseline": {
                field: _comparison_display_value(value)
                for field, value in baseline_provenance.items()
            },
            "candidate": {
                field: _comparison_display_value(value)
                for field, value in candidate_provenance.items()
            },
            "mismatches": provenance_mismatches,
            "missing": missing_provenance,
        },
        # These top-level aliases make the two gate inputs easy to consume in
        # shell/JSON tooling without requiring callers to know the nested shape.
        "provenance_match": not provenance_mismatches,
        "provenance_mismatches": provenance_mismatches,
        "cell_alignment": cell_alignment,
        "safety": safety,
        "summary": summary,
        "per_cell": per_cell,
        "quality_gate": quality_gate,
    }
    safe_comparison = _json_safe(comparison)
    try:
        json.dumps(safe_comparison, allow_nan=False)
    except (TypeError, ValueError) as error:  # pragma: no cover - guarded above
        raise EvaluationMatrixError("report comparison is not JSON-safe") from error
    return safe_comparison


def _make_default_deck_specs() -> tuple[OpponentDeckSpec, ...]:
    return (
        OpponentDeckSpec(
            "hog-cycle",
            (
                "hog-rider",
                "cannon",
                "musketeer",
                "skeletons",
                "ice-golem",
                "ice-spirit",
                "fireball",
                "log",
            ),
            tags=("cycle", "regression"),
        ),
        OpponentDeckSpec(
            "giant-beatdown",
            (
                "giant",
                "musketeer",
                "mini-pekka",
                "archers",
                "fireball",
                "zap",
                "knight",
                "bomber",
            ),
            tags=("beatdown", "ground"),
        ),
        OpponentDeckSpec(
            "balloon-air",
            (
                "balloon",
                "baby-dragon",
                "minions",
                "musketeer",
                "tombstone",
                "fireball",
                "arrows",
                "ice-golem",
            ),
            tags=("air", "beatdown"),
        ),
        OpponentDeckSpec(
            "xbow-siege",
            (
                "x-bow",
                "tesla",
                "archers",
                "knight",
                "fireball",
                "log",
                "skeletons",
                "ice-spirit",
            ),
            tags=("siege", "cycle"),
        ),
        OpponentDeckSpec(
            "bait",
            (
                "goblin-barrel",
                "princess",
                "goblin-gang",
                "skeleton-army",
                "rocket",
                "inferno-tower",
                "knight",
                "log",
            ),
            tags=("bait", "spell"),
        ),
    )


DEFAULT_OPPONENT_DECKS = _make_default_deck_specs()
DEFAULT_OPPONENT_STRATEGIES = (
    OpponentStrategySpec(
        "deterministic-cycle",
        description="Cheapest affordable card with alternating lanes.",
    ),
    OpponentStrategySpec(
        "random-legal",
        description="Seeded random legal affordable actions.",
    ),
)


def _builtin_controller(strategy_id: str, seed: int) -> Any:
    try:
        from ..actions import PlayCardAction, WaitAction
        from ..engine import DeterministicCycleController
    except ImportError:  # pragma: no cover - top-level ``rl`` layout
        from simulator.actions import PlayCardAction, WaitAction
        from simulator.engine import DeterministicCycleController

    if strategy_id == "deterministic-cycle":
        return DeterministicCycleController(lane="alternate")
    if strategy_id == "deterministic-left":
        return DeterministicCycleController(lane="left")
    if strategy_id == "deterministic-right":
        return DeterministicCycleController(lane="right")
    if strategy_id == "wait":
        class WaitController:
            def choose_action(self, _engine: Any, _state: Any, player: int) -> Any:
                return WaitAction(player)

        return WaitController()
    if strategy_id == "random-legal":
        class RandomLegalController:
            def __init__(self, random_seed: int) -> None:
                self._random = random.Random(random_seed)

            def choose_action(self, engine: Any, state: Any, player: int) -> Any:
                player_state = state.players[player]
                candidates: list[tuple[int, tuple[tuple[int, int], ...]]] = []
                for slot, card_id in enumerate(player_state.hand):
                    card = engine.ruleset.card(card_id)
                    if engine._effective_card_cost(player_state, card) > player_state.elixir_milli:
                        continue
                    legal = engine.legal_cells(state, player, card_id)
                    if legal:
                        candidates.append((slot, legal))
                if not candidates or self._random.random() < 0.15:
                    return WaitAction(player)
                slot, cells = self._random.choice(candidates)
                return PlayCardAction(player, slot, self._random.choice(cells))

        return RandomLegalController(seed)
    raise EvaluationMatrixError(f"unsupported built-in strategy: {strategy_id!r}")


def _controller_action(
    controller: Any,
    engine: Any,
    state: Any,
    player: int,
    *,
    public_observation: Any | None = None,
) -> Any:
    public_chooser = getattr(controller, "choose_public_action", None)
    if callable(public_chooser):
        if public_observation is None:
            raise EvaluationMatrixError(
                "public checkpoint opponent requires the viewer's V2 observation"
            )
        return public_chooser(public_observation, player=player)
    chooser = getattr(controller, "choose_action", None)
    if callable(chooser):
        action = chooser(engine, state, player)
    elif callable(controller):
        action = controller(engine, state, player)
    else:
        raise EvaluationMatrixError(
            "opponent factory must return choose_action(...) controller or callable"
        )
    return action


def _evaluation_batch_config(
    stored_config: Any,
    *,
    envs: int,
    horizon: int,
) -> Any:
    """Build a collector config for a complete evaluation horizon.

    ``sequence_length`` is a PPO-update chunking setting, not an inference
    setting.  A full-match cap is derived from the simulator duration and is
    therefore not guaranteed to be divisible by the training chunk length.
    The collector carries the recurrent hidden state one decision at a time,
    so disabling incompatible training-only chunking preserves evaluation
    behavior while allowing the complete cap to run.
    """

    try:
        from .prototype import PrototypeConfig
    except ImportError:  # pragma: no cover - top-level ``rl`` layout
        from rl.prototype import PrototypeConfig

    values = {
        **stored_config.as_dict(),
        "envs": envs,
        "horizon": horizon,
        "updates": 1,
        "seed": 0,
        "env_backend": "reference",
        "env_workers": None,
        "allow_provisional": True,
    }
    sequence_length = getattr(stored_config, "sequence_length", None)
    if sequence_length is not None and horizon % int(sequence_length):
        values["sequence_length"] = None
    return PrototypeConfig.from_mapping(values)


class _CheckpointMatchRunner:
    """Load one checkpoint once, then execute matrix cells sequentially."""

    def __init__(self, config: EvaluationMatrixConfig) -> None:
        try:
            from .prototype import load_prototype_checkpoint
        except ImportError:  # pragma: no cover - defensive package layout
            raise EvaluationMatrixError("cannot import the prototype checkpoint loader")
        try:
            self.learner, self.stored_config, self.metadata = load_prototype_checkpoint(
                config.checkpoint,
                device=config.device,
            )
        except Exception as error:
            raise EvaluationMatrixError(
                f"cannot load evaluation checkpoint {config.checkpoint}: {error}"
            ) from error

        try:
            from ..engine import BattleEngine
            from ..env import RewardConfig, SimulatorEnv
            from ..ruleset import load_ruleset
        except ImportError:  # pragma: no cover - top-level ``rl`` layout
            from simulator.engine import BattleEngine
            from simulator.env import RewardConfig, SimulatorEnv
            from simulator.ruleset import load_ruleset
        self._BattleEngine = BattleEngine
        self._RewardConfig = RewardConfig
        self._SimulatorEnv = SimulatorEnv
        self.ruleset = load_ruleset(self.stored_config.ruleset_id)
        self._player_deck = self._canonical_player_deck(config.player_deck)
        self.domain_randomization = config.domain_randomization
        self._torch = self._require_torch()
        self.learner.policy.eval()
        self.learner.critic.eval()

    @staticmethod
    def _require_torch() -> Any:
        try:
            import torch
        except ModuleNotFoundError as error:  # pragma: no cover - checkpoint loader already guards this
            raise EvaluationMatrixError("actor evaluation requires PyTorch") from error
        return torch

    def metadata_report(self) -> dict[str, object]:
        return {
            "checkpoint_format": self.metadata.get("checkpoint_format"),
            "checkpoint_fingerprint": _file_fingerprint(self.metadata.get("checkpoint_path", ""))
            if self.metadata.get("checkpoint_path")
            else None,
            "ruleset_id": self.ruleset.ruleset_id,
            "ruleset_hash": self.ruleset.content_hash,
            "actor_privileged_inputs": False,
            "critic_privileged_inputs": bool(self.learner.uses_privileged_critic),
            "domain_randomization": (
                None
                if self.domain_randomization is None
                else self.domain_randomization.as_dict()
            ),
            "runner": "simulator-reference",
        }

    def _canonical_opponent_deck(self, deck: OpponentDeckSpec) -> tuple[str, ...]:
        try:
            return tuple(self.ruleset.resolve_card_id(card) for card in deck.cards)
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationMatrixError(
                f"deck {deck.deck_id!r} contains a card unavailable in ruleset "
                f"{self.ruleset.ruleset_id!r}: {error}"
            ) from error

    def _canonical_player_deck(
        self,
        deck: Sequence[str],
    ) -> tuple[str, ...]:
        try:
            canonical = tuple(self.ruleset.resolve_card_id(card) for card in deck)
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationMatrixError(
                "player_deck contains a card unavailable in ruleset "
                f"{self.ruleset.ruleset_id!r}: {error}"
            ) from error
        if len(set(canonical)) != len(canonical):
            raise EvaluationMatrixError(
                "player_deck must not contain duplicate canonical cards"
            )
        return canonical

    def _decision_cap(self, spec: MatchSpec) -> int:
        if spec.max_decisions is not None:
            return spec.max_decisions
        duration_us = int(
            self.ruleset.match.regulation_us + self.ruleset.match.overtime_us
        )
        interval_us = int(self.stored_config.decision_interval_us)
        if spec.domain_randomization is not None:
            base_ticks = max(1, interval_us // int(self.ruleset.tick_us))
            minimum_ticks = max(
                1,
                base_ticks
                - spec.domain_randomization.decision_interval_jitter_ticks,
            )
            interval_us = minimum_ticks * int(self.ruleset.tick_us)
        return max(1, math.ceil(duration_us / interval_us))

    def _make_environment(
        self,
        spec: MatchSpec,
    ) -> Any:
        environment = self._SimulatorEnv(
            engine=self._BattleEngine(self.ruleset, validate_every_tick=False),
            decision_interval_us=int(self.stored_config.decision_interval_us),
            reward=self._RewardConfig.terminal_outcome(),
            expose_privileged_info=True,
            include_authoritative_state=False,
        )
        if spec.domain_randomization is None:
            return environment
        from .domain_randomization import DomainRandomizedEnv

        return DomainRandomizedEnv(
            environment,
            spec.domain_randomization,
            seed=deterministic_seed(
                spec.seed,
                "evaluation-domain-randomization",
                spec.cell_id,
            ),
        )

    def _actor_action(
        self,
        observation: Any,
        rollout_state: Any,
        *,
        reset: bool,
    ) -> tuple[Any, Any]:
        from .collector import _batch_observations
        from .learner import RecurrentRolloutState
        from cr_bot.domain.game_state import Action as PolicyAction

        raster, global_features, entities, entity_mask, masks = _batch_observations(
            [observation],
            device=self.learner.device,
        )
        reset_mask = self._torch.tensor(
            [[reset]],
            dtype=self._torch.bool,
            device=self.learner.device,
        )
        with self._torch.inference_mode():
            actions, final_hidden = self.learner.policy.act_deterministic(
                raster,
                global_features,
                entities,
                entity_mask,
                masks,
                reset_mask=reset_mask,
                hidden=rollout_state.hidden,
            )
        next_state = RecurrentRolloutState(final_hidden.detach())
        mode = int(actions.mode[0, 0].item())
        if mode == 0:
            return PolicyAction(kind="Wait"), next_state
        row = int(actions.placement[0, 0, 0].item())
        column = int(actions.placement[0, 0, 1].item())
        return (
            PolicyAction(
                kind="Play",
                card_idx=int(actions.card_slot[0, 0].item()),
                cell=(column, row),
            ),
            next_state,
        )

    def run_batch(self, specs: Sequence[MatchSpec]) -> list[MatchResult]:
        """Run actor cells together so one policy call serves all lanes.

        The simulator remains independent per lane, but the public V2 tensors
        are batched through the recurrent collector.  This is substantially
        faster than invoking a separate model forward for every matrix cell,
        especially on accelerators.  The sequential ``__call__`` path remains
        the fallback for public-counter evaluation and target-player-one
        matrices.
        """

        if not specs:
            return []
        if any(spec.policy_mode != "actor" for spec in specs):
            raise EvaluationMatrixError("run_batch supports actor policy cells only")
        if any(spec.target_player != 0 for spec in specs):
            raise EvaluationMatrixError("run_batch currently supports target_player=0 only")
        if any(spec.domain_randomization is not None for spec in specs):
            raise EvaluationMatrixError(
                "domain-randomized matrix cells must use sequential evaluation"
            )
        cap = self._decision_cap(specs[0])
        if any(self._decision_cap(spec) != cap for spec in specs):
            raise EvaluationMatrixError("batched matrix cells must share a decision cap")

        try:
            from .prototype import (
                _make_collector,
                _trace_decision,
            )
        except ImportError:  # pragma: no cover - top-level ``rl`` layout
            raise EvaluationMatrixError("cannot import the recurrent rollout collector")

        environments: list[Any] = []
        controllers: list[Any] = []
        for spec in specs:
            opponent_deck = self._canonical_opponent_deck(spec.opponent_deck)
            environment = self._SimulatorEnv(
                engine=self._BattleEngine(self.ruleset, validate_every_tick=False),
                decision_interval_us=int(self.stored_config.decision_interval_us),
                reward=self._RewardConfig.terminal_outcome(),
                # Diagnostics are returned through ``info`` only.  They do
                # not enter the actor's V2 tensors, so enabling them here
                # keeps batched matrix rows as auditable as sequential rows.
                expose_privileged_info=True,
                include_authoritative_state=False,
            )
            environment.reset_v2(
                seed=spec.seed,
                decks=(tuple(self._player_deck), opponent_deck),
                shuffle_decks=spec.shuffle_decks,
            )
            environments.append(environment)
            controllers.append(spec.strategy.build(spec.seed))

        environment_ids = {id(environment): index for index, environment in enumerate(environments)}

        def opponent_action(environment: Any, observation: Any, player: int) -> Any:
            try:
                index = environment_ids[id(environment)]
            except KeyError as error:  # pragma: no cover - collector invariant
                raise EvaluationMatrixError("unknown environment in batched matrix") from error
            return _controller_action(
                controllers[index],
                environment.engine,
                environment.state,
                player,
                public_observation=observation,
            )

        batch_config = _evaluation_batch_config(
            self.stored_config,
            envs=len(specs),
            horizon=cap,
        )
        last_results: list[Any | None] = [None] * len(specs)
        decisions: list[int] = [0] * len(specs)
        target_plays: list[dict[str, int]] = [dict() for _ in specs]
        opponent_plays: list[dict[str, int]] = [dict() for _ in specs]
        target_play_trace: list[list[dict[str, object]]] = [
            [] for _ in specs
        ]
        opponent_play_trace: list[list[dict[str, object]]] = [
            [] for _ in specs
        ]
        target_rejected: list[int] = [0] * len(specs)
        opponent_rejected: list[int] = [0] * len(specs)

        def on_decision(record: Any) -> None:
            lane = int(record.lane)
            last_results[lane] = record.result
            decisions[lane] += 1
            row = _trace_decision(
                record,
                target_player=0,
                include_positions=False,
            )
            target_action_kind = row.get("mode")
            if target_action_kind == "PLAY":
                target_play_trace[lane].append(row)
                if bool(row.get("accepted")):
                    card = row.get("played_card_id") or row.get("card_id")
                    if isinstance(card, str):
                        target_plays[lane][card] = target_plays[lane].get(card, 0) + 1
                else:
                    target_rejected[lane] += 1
            opponent_action = row.get("opponent_action")
            if isinstance(opponent_action, Mapping) and opponent_action.get("mode") == "PLAY":
                opponent_play_trace[lane].append(
                    {
                        "decision": row.get("decision"),
                        "physics_tick_before": row.get("physics_tick_before"),
                        "elapsed_us_before": row.get("elapsed_us_before"),
                        "card_id": row.get("opponent_card_id"),
                        "accepted": row.get("opponent_accepted"),
                        "policy_cell": row.get("opponent_policy_cell"),
                        "world_cell": row.get("opponent_world_cell"),
                        "played_world_cell": row.get("opponent_played_world_cell"),
                        "rejection_reason": row.get("opponent_rejection_reason"),
                    }
                )
                if bool(row.get("opponent_accepted")):
                    card = row.get("opponent_card_id")
                    if isinstance(card, str):
                        opponent_plays[lane][card] = opponent_plays[lane].get(card, 0) + 1
                else:
                    opponent_rejected[lane] += 1

        collector = _make_collector(
            self.learner,
            batch_config,
            deterministic=True,
            stop=True,
            freeze_completed_lanes=True,
            opponent_action=opponent_action,
        )
        rollout = collector.collect(environments, decision_callback=on_decision)
        returns = rollout.trajectory.rewards.detach().sum(dim=1).cpu().tolist()
        results: list[MatchResult] = []
        for index, spec in enumerate(specs):
            step = last_results[index]
            info = getattr(step, "info", {}) if step is not None else {}
            if not isinstance(info, Mapping):
                info = {}
            terminated = bool(getattr(step, "terminated", False))
            truncated = bool(getattr(step, "truncated", False))
            winner = info.get("winner")
            if truncated or not terminated:
                outcome = "truncated"
            elif winner == spec.target_player:
                outcome = "win"
            elif winner == 1 - spec.target_player:
                outcome = "loss"
            else:
                outcome = "draw"
            terminal_reason = info.get("terminal_reason")
            if outcome == "truncated" and not isinstance(terminal_reason, str):
                terminal_reason = "evaluation_cap"
            results.append(
                MatchResult(
                    outcome=outcome,
                    decisions=decisions[index] or cap,
                    return_value=float(returns[index]),
                    winner=winner,
                    terminal_reason=terminal_reason,
                    metrics={
                        "batched_actor_inference": True,
                        "target_rejected_actions": target_rejected[index],
                        "opponent_rejected_actions": opponent_rejected[index],
                        "target_plays_by_card": target_plays[index],
                        "opponent_plays_by_card": opponent_plays[index],
                        "target_play_trace": target_play_trace[index],
                        "opponent_play_trace": opponent_play_trace[index],
                        "tower_hp_end": self._tower_snapshot(environments[index]),
                        "crowns_end": self._crown_snapshot(environments[index]),
                        "domain_randomization": None,
                        "troop_positions_end": _troop_positions_by_player(
                            environments[index].state
                        ),
                    },
                )
            )
        return results

    @staticmethod
    def _tower_snapshot(environment: Any) -> dict[str, dict[str, dict[str, int]]]:
        towers: dict[str, dict[str, dict[str, int]]] = {
            "player_0": {},
            "player_1": {},
        }
        state = getattr(environment, "state", None)
        for entity in getattr(state, "entities", {}).values():
            if getattr(entity, "kind", None) != "tower":
                continue
            owner = getattr(entity, "owner", None)
            role = getattr(entity, "role", None)
            hp = getattr(entity, "hp", None)
            maximum = getattr(entity, "max_hp", None)
            if owner not in (0, 1) or not isinstance(role, str):
                continue
            if type(hp) is not int or type(maximum) is not int:
                continue
            towers[f"player_{owner}"][role] = {
                "hp": max(0, hp),
                "max_hp": max(0, maximum),
            }
        return towers

    @staticmethod
    def _variant_snapshot(environment: Any) -> dict[str, object] | None:
        variant = getattr(environment, "variant", None)
        if variant is None:
            return None
        as_dict = getattr(variant, "as_dict", None)
        if not callable(as_dict):
            return None
        value = as_dict()
        return value if isinstance(value, dict) else None

    @staticmethod
    def _crown_snapshot(environment: Any) -> dict[str, int]:
        """Return terminal crown totals in canonical world-player order."""

        state = getattr(environment, "state", None)
        players = getattr(state, "players", ())
        crowns: dict[str, int] = {}
        for player in (0, 1):
            try:
                value = players[player].crowns
            except (IndexError, KeyError, TypeError, AttributeError):
                continue
            if type(value) is int and 0 <= value <= 3:
                crowns[f"player_{player}"] = value
        return crowns

    def __call__(self, spec: MatchSpec) -> MatchResult:
        try:
            from .public_counter import (
                PublicCounterController,
                StrategicCounterController,
            )
            from .expert import DeterministicCounterController
            from .prototype import _trace_decision
            from cr_bot.domain.game_state import Action as PolicyAction
        except ImportError:  # pragma: no cover - defensive package layout
            raise EvaluationMatrixError("cannot import public policy action types")
        try:
            from ..engine import DeterministicCycleController
        except ImportError:  # pragma: no cover - top-level ``rl`` layout
            from simulator.engine import DeterministicCycleController

        opponent_deck = self._canonical_opponent_deck(spec.opponent_deck)
        decks = (
            (tuple(self._player_deck), opponent_deck)
            if spec.target_player == 0
            else (opponent_deck, tuple(self._player_deck))
        )
        environment = self._make_environment(spec)
        observations = environment.reset_v2(
            seed=spec.seed,
            decks=decks,
            shuffle_decks=spec.shuffle_decks,
        )
        opponent = spec.strategy.build(spec.seed)
        counter = None
        if spec.policy_mode == "public-counter":
            counter = PublicCounterController()
        elif spec.policy_mode == "strategic-counter":
            counter = StrategicCounterController()
        elif spec.policy_mode == "deterministic-counter":
            counter = DeterministicCounterController()
        rollout_state = (
            self.learner.initial_rollout_state(1)
            if spec.policy_mode == "actor"
            else None
        )
        reset = True
        total_return = 0.0
        target_rejected = 0
        opponent_rejected = 0
        target_plays: dict[str, int] = {}
        opponent_plays: dict[str, int] = {}
        target_play_trace: list[dict[str, object]] = []
        opponent_play_trace: list[dict[str, object]] = []
        cap = self._decision_cap(spec)

        for decision in range(cap):
            state_before = environment.state
            if state_before is None:  # pragma: no cover - environment invariant
                raise EvaluationMatrixError("evaluation environment lost its state")
            physics_tick_before = state_before.tick
            elapsed_us_before = state_before.elapsed_us
            hand_before = tuple(state_before.players[spec.target_player].hand)
            elixir_before = int(state_before.players[spec.target_player].elixir_milli)
            if spec.policy_mode == "actor":
                if rollout_state is None:  # pragma: no cover - construction invariant
                    raise EvaluationMatrixError("actor rollout state was not initialized")
                target_action, rollout_state = self._actor_action(
                    observations[spec.target_player],
                    rollout_state,
                    reset=reset,
                )
            else:
                if counter is None:  # pragma: no cover - construction invariant
                    raise EvaluationMatrixError("public policy was not initialized")
                if spec.policy_mode == "deterministic-counter":
                    target_action = counter.choose_action(
                        environment.engine,
                        environment.state,
                        spec.target_player,
                    )
                else:
                    target_action = counter.choose_action(
                        observations[spec.target_player],
                        player=spec.target_player,
                    )
            opponent_player = 1 - spec.target_player
            opponent_action = _controller_action(
                opponent,
                environment.engine,
                environment.state,
                opponent_player,
                public_observation=observations[opponent_player],
            )
            actions: list[Any] = [None, None]
            actions[spec.target_player] = target_action
            actions[opponent_player] = opponent_action
            step = environment.step_v2(actions)
            total_return += float(step.rewards[spec.target_player])
            info = step.info
            row = _trace_decision(
                SimpleNamespace(
                    decision_index=decision,
                    target_action=target_action,
                    opponent_action=opponent_action,
                    result=step,
                    state_after=environment.state,
                    physics_tick_before=physics_tick_before,
                    elapsed_us_before=elapsed_us_before,
                    hand_before=hand_before,
                    elixir_before=elixir_before,
                ),
                target_player=spec.target_player,
                include_positions=False,
            )
            if row.get("mode") == "PLAY":
                target_play_trace.append(row)
                if bool(row.get("accepted")):
                    card_id = row.get("played_card_id") or row.get("card_id")
                    if isinstance(card_id, str):
                        target_plays[card_id] = target_plays.get(card_id, 0) + 1
                else:
                    target_rejected += 1
            opponent_row = row.get("opponent_action")
            if isinstance(opponent_row, Mapping) and opponent_row.get("mode") == "PLAY":
                opponent_play_trace.append(
                    {
                        "decision": row.get("decision"),
                        "physics_tick_before": row.get("physics_tick_before"),
                        "elapsed_us_before": row.get("elapsed_us_before"),
                        "card_id": row.get("opponent_card_id"),
                        "accepted": row.get("opponent_accepted"),
                        "policy_cell": row.get("opponent_policy_cell"),
                        "world_cell": row.get("opponent_world_cell"),
                        "played_world_cell": row.get("opponent_played_world_cell"),
                        "rejection_reason": row.get("opponent_rejection_reason"),
                    }
                )
                if bool(row.get("opponent_accepted")):
                    card_id = row.get("opponent_card_id")
                    if isinstance(card_id, str):
                        opponent_plays[card_id] = opponent_plays.get(card_id, 0) + 1
                else:
                    opponent_rejected += 1
            reset = False
            observations = step.observations
            if step.terminated or step.truncated:
                winner = info.get("winner")
                if step.truncated:
                    outcome = "truncated"
                elif winner == spec.target_player:
                    outcome = "win"
                elif winner == opponent_player:
                    outcome = "loss"
                else:
                    outcome = "draw"
                return MatchResult(
                    outcome=outcome,
                    decisions=decision + 1,
                    return_value=total_return,
                    winner=winner,
                    terminal_reason=info.get("terminal_reason"),
                    metrics={
                        "target_rejected_actions": target_rejected,
                        "opponent_rejected_actions": opponent_rejected,
                        "target_plays_by_card": target_plays,
                        "opponent_plays_by_card": opponent_plays,
                        "target_play_trace": target_play_trace,
                        "opponent_play_trace": opponent_play_trace,
                        "tower_hp_end": self._tower_snapshot(environment),
                        "crowns_end": self._crown_snapshot(environment),
                        "domain_randomization": self._variant_snapshot(environment),
                        "troop_positions_end": _troop_positions_by_player(
                            environment.state
                        ),
                    },
                )

        return MatchResult(
            outcome="truncated",
            decisions=cap,
            return_value=total_return,
            terminal_reason="evaluation_cap",
            metrics={
                "target_rejected_actions": target_rejected,
                "opponent_rejected_actions": opponent_rejected,
                "target_plays_by_card": target_plays,
                "opponent_plays_by_card": opponent_plays,
                "target_play_trace": target_play_trace,
                "opponent_play_trace": opponent_play_trace,
                "tower_hp_end": self._tower_snapshot(environment),
                "crowns_end": self._crown_snapshot(environment),
                "domain_randomization": self._variant_snapshot(environment),
                "troop_positions_end": _troop_positions_by_player(
                    environment.state
                ),
            },
        )


def run_evaluation_matrix(
    config: EvaluationMatrixConfig,
    *,
    match_runner: MatchRunner | None = None,
    progress_callback: Callable[[int, int, MatchSpec, MatchResult], None] | None = None,
) -> dict[str, object]:
    """Run every deck × strategy × seed cell and return a JSON-safe report.

    The default runner loads ``config.checkpoint`` once and executes real
    reference-simulator matches.  An injected runner is called once per cell,
    in deterministic deck/strategy/seed order, and avoids checkpoint loading.
    ``progress_callback`` receives ``completed, total, spec, result`` after
    each cell.
    """

    if not isinstance(config, EvaluationMatrixConfig):
        raise TypeError("config must be an EvaluationMatrixConfig")
    if match_runner is not None and not callable(match_runner):
        raise TypeError("match_runner must be callable when provided")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable when provided")

    started = perf_counter()
    started_at_utc = _utc_timestamp()
    runner_setup_started = perf_counter()
    runner: MatchRunner
    runner_metadata: dict[str, object]
    default_runner: _CheckpointMatchRunner | None = None
    if match_runner is None:
        default_runner = _CheckpointMatchRunner(config)
        runner = default_runner
        runner_metadata = default_runner.metadata_report()
    else:
        runner = match_runner
        runner_metadata = {"runner": "injected"}
    runner_setup_seconds = perf_counter() - runner_setup_started

    rows: list[dict[str, object]] = []
    result_groups: dict[tuple[str, str], list[MatchResult]] = {
        (deck.deck_id, strategy.strategy_id): []
        for deck in config.opponent_decks
        for strategy in config.strategies
    }
    specs = [
        MatchSpec(
            checkpoint=config.checkpoint,
            opponent_deck=deck,
            strategy=strategy,
            seed=seed,
            policy_mode=config.policy_mode,
            target_player=config.target_player,
            max_decisions=config.max_decisions,
            device=config.device,
            shuffle_decks=config.shuffle_decks,
            player_deck=config.player_deck,
            domain_randomization=config.domain_randomization,
        )
        for deck in config.opponent_decks
        for strategy in config.strategies
        for seed in config.seeds
    ]

    completed = 0

    def record_result(spec: MatchSpec, raw_result: Any) -> None:
        nonlocal completed
        result = _normalize_match_result(
            raw_result,
            target_player=config.target_player,
        )
        # Force conversion now so a custom runner cannot cause a partially
        # JSON-safe report to escape this API.
        result.as_dict()
        result_groups[(spec.opponent_deck.deck_id, spec.strategy.strategy_id)].append(result)
        completed += 1
        row = {
            "matrix_index": completed - 1,
            **spec.as_dict(),
            "outcome": result.outcome,
            "decisions": result.decisions,
            "return": float(result.return_value),
            "winner": result.winner,
            "terminal_reason": result.terminal_reason,
            "metrics": _json_safe(dict(result.metrics)),
        }
        rows.append(_json_safe(row))
        if progress_callback is not None:
            progress_callback(completed, config.match_count, spec, result)

    can_batch = (
        default_runner is not None
        and config.policy_mode == "actor"
        and config.target_player == 0
        and config.batch_size > 1
        and config.domain_randomization is None
    )
    execution_mode = "batched_actor" if can_batch else "sequential"
    batch_count = (
        math.ceil(config.match_count / config.batch_size)
        if can_batch
        else config.match_count
    )
    execution_started = perf_counter()
    if can_batch:
        for start in range(0, len(specs), config.batch_size):
            batch = specs[start : start + config.batch_size]
            batch_results = default_runner.run_batch(batch)
            if len(batch_results) != len(batch):  # pragma: no cover - runner invariant
                raise EvaluationMatrixError("batched runner returned the wrong result count")
            for spec, result in zip(batch, batch_results, strict=True):
                record_result(spec, result)
    else:
        for spec in specs:
            record_result(spec, runner(spec))
    execution_seconds = perf_counter() - execution_started

    matchup_rows: list[dict[str, object]] = []
    for deck in config.opponent_decks:
        for strategy in config.strategies:
            results = result_groups[(deck.deck_id, strategy.strategy_id)]
            matchup: dict[str, object] = {
                "deck_id": deck.deck_id,
                "deck_cards": list(deck.cards),
                "deck_tags": list(deck.tags),
                "deck_metadata": dict(deck.metadata),
                "strategy_id": strategy.strategy_id,
                "strategy": strategy.as_dict(),
                "summary": _summary(results),
            }
            if config.include_match_results:
                matchup["matches"] = [
                    {
                        **MatchSpec(
                            checkpoint=config.checkpoint,
                            opponent_deck=deck,
                            strategy=strategy,
                            seed=config.seeds[index],
                            policy_mode=config.policy_mode,
                            target_player=config.target_player,
                            max_decisions=config.max_decisions,
                            device=config.device,
                            shuffle_decks=config.shuffle_decks,
                            player_deck=config.player_deck,
                            domain_randomization=config.domain_randomization,
                        ).as_dict(),
                        **result.as_dict(),
                    }
                    for index, result in enumerate(results)
                ]
            matchup_rows.append(_json_safe(matchup))

    total_results = [result for results in result_groups.values() for result in results]
    total_summary = _summary(total_results)
    wall_seconds = perf_counter() - started
    finished_at_utc = _utc_timestamp()
    checkpoint_fingerprint = _file_fingerprint(config.checkpoint)
    config_fingerprint = _json_fingerprint(config.as_dict())
    selected_compositions = {
        _deck_composition_key(deck.cards) for deck in config.opponent_decks
    }
    excluded_compositions = set(config.excluded_deck_compositions)
    overlap = sorted(selected_compositions & excluded_compositions)
    held_out_audit = {
        "source": (
            None if config.held_out_source is None else str(config.held_out_source)
        ),
        "excluded_deck_compositions": [
            list(cards) for cards in config.excluded_deck_compositions
        ],
        "selected_deck_compositions": [
            list(cards) for cards in sorted(selected_compositions)
        ],
        "overlap": [list(cards) for cards in overlap],
        "disjointness_verified": bool(
            config.held_out and config.held_out_source is not None and not overlap
        ),
    }
    report: dict[str, object] = {
        "kind": EVALUATION_MATRIX_KIND,
        "schema_version": EVALUATION_MATRIX_SCHEMA_VERSION,
        "checkpoint": str(config.checkpoint),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "config": config.as_dict(),
        "policy_mode": config.policy_mode,
        "actor_controls_actions": config.policy_mode == "actor",
        "actor_privileged_inputs": runner_metadata.get("actor_privileged_inputs"),
        "held_out": config.held_out,
        "target_player": config.target_player,
        "actor_player": config.target_player,
        "opponent_player": 1 - config.target_player,
        "player_deck": list(config.player_deck),
        "opponent_decks": [deck.as_dict() for deck in config.opponent_decks],
        "strategies": [strategy.as_dict() for strategy in config.strategies],
        "seeds": list(config.seeds),
        "max_decisions": config.max_decisions,
        "batch_size": config.batch_size,
        "domain_randomization": (
            None
            if config.domain_randomization is None
            else config.domain_randomization.as_dict()
        ),
        "matrix_size": config.match_count,
        "cell_ids": [spec.cell_id for spec in specs],
        "held_out_audit": held_out_audit,
        "runner": runner_metadata,
        "provenance": {
            "schema_version": EVALUATION_PROVENANCE_SCHEMA_VERSION,
            "config_fingerprint": config_fingerprint,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "matrix_order": "opponent_decks,strategies,seeds",
            "cell_identity": "deck_id::strategy_id::seed-{seed}",
            "actor_player": config.target_player,
            "opponent_player": 1 - config.target_player,
            "runtime": _runtime_provenance(),
            "runner": runner_metadata,
        },
        "timing": {
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "wall_seconds": wall_seconds,
            "runner_setup_seconds": runner_setup_seconds,
            "match_execution_seconds": execution_seconds,
            "execution_mode": execution_mode,
            "batch_count": batch_count,
            "matches_per_second": (
                config.match_count / execution_seconds if execution_seconds else 0.0
            ),
            "decisions_per_second": (
                total_summary["decisions_total"] / execution_seconds
                if execution_seconds
                else 0.0
            ),
            "includes_progress_callback": progress_callback is not None,
        },
        "total": total_summary,
        "matchups": matchup_rows,
        "warning": (
            "Held-out results are only as faithful as the selected simulator ruleset; "
            "the bundled V1 ruleset is provisional."
        ),
    }
    if config.include_match_results:
        report["matches"] = rows
    from .exploit_audit import audit_simulation_report

    report["simulation_exploit_audit"] = audit_simulation_report(report)
    report["quality_gate"] = _evaluation_quality_gate(
        report,
        report["simulation_exploit_audit"],
    )
    report = _json_safe(report)
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as error:  # pragma: no cover - guarded above
        raise EvaluationMatrixError("evaluation matrix report is not JSON-safe") from error
    return report


def evaluate_checkpoint_matrix(
    checkpoint: str | Path,
    *,
    opponent_decks: Sequence[OpponentDeckSpec | Mapping[str, Any]] = DEFAULT_OPPONENT_DECKS,
    strategies: Sequence[OpponentStrategySpec | str | Mapping[str, Any]] = DEFAULT_OPPONENT_STRATEGIES,
    seeds: Sequence[int] = (10_000,),
    policy_mode: str = "actor",
    target_player: int = 0,
    max_decisions: int | None = None,
    device: str | None = "auto",
    shuffle_decks: bool = True,
    include_match_results: bool = True,
    held_out: bool = True,
    batch_size: int = 8,
    held_out_source: str | Path | None = None,
    excluded_deck_compositions: Sequence[Sequence[str]] = (),
    player_deck: Sequence[str] | None = None,
    domain_randomization: DomainRandomizationConfig | Mapping[str, Any] | None = None,
    match_runner: MatchRunner | None = None,
    progress_callback: Callable[[int, int, MatchSpec, MatchResult], None] | None = None,
) -> dict[str, object]:
    """Convenience wrapper that builds :class:`EvaluationMatrixConfig`."""

    config = EvaluationMatrixConfig(
        checkpoint=checkpoint,
        opponent_decks=tuple(opponent_decks),
        strategies=tuple(strategies),
        seeds=tuple(seeds),
        policy_mode=policy_mode,
        target_player=target_player,
        max_decisions=max_decisions,
        device=device,
        shuffle_decks=shuffle_decks,
        include_match_results=include_match_results,
        held_out=held_out,
        batch_size=batch_size,
        held_out_source=held_out_source,
        excluded_deck_compositions=tuple(
            tuple(cards) for cards in excluded_deck_compositions
        ),
        player_deck=player_deck,
        domain_randomization=domain_randomization,
    )
    return run_evaluation_matrix(
        config,
        match_runner=match_runner,
        progress_callback=progress_callback,
    )


def write_evaluation_matrix_report(
    report: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Write an already-generated matrix report with strict JSON encoding."""

    safe = _json_safe(dict(report))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "DEFAULT_OPPONENT_DECKS",
    "DEFAULT_OPPONENT_STRATEGIES",
    "EVALUATION_MATRIX_KIND",
    "EVALUATION_MATRIX_SCHEMA_VERSION",
    "EVALUATION_PROVENANCE_SCHEMA_VERSION",
    "DEFAULT_REPORT_COMPARISON_MAX_CELLS",
    "REPORT_COMPARISON_KIND",
    "REPORT_COMPARISON_SCHEMA_VERSION",
    "EvaluationMatrixConfig",
    "EvaluationMatrixError",
    "MatchResult",
    "MatchRunner",
    "MatchSpec",
    "OpponentDeck",
    "OpponentDeckSpec",
    "OpponentStrategy",
    "OpponentStrategySpec",
    "compare_evaluation_reports",
    "evaluate_checkpoint_matrix",
    "run_evaluation_matrix",
    "write_evaluation_matrix_report",
]
