"""Serializable league scheduling, payoff, and opponent-sampling primitives.

This module does not load checkpoints or run physics.  It owns the
reproducible schedule boundary, directional payoff history, Elo bookkeeping,
PFSP selection, and fail-closed exploiter-reset handoff used by a match
service.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Literal, Mapping


LEAGUE_SCHEMA_VERSION = 1
AgentRole = Literal["main_agent", "exploiter"]
OpponentSource = Literal["main_agent", "exploiter", "historical"]
ConditioningMode = Literal["deck_id", "card_tokens"]
LeagueOutcome = Literal["win", "draw", "loss"]


class LeagueConfigurationError(ValueError):
    """Raised when a league scope or sampling policy is not fail-closed."""


class LeagueSamplingError(LeagueConfigurationError):
    """Raised when a valid configuration has no candidate for a requested draw."""


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LeagueConfigurationError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise LeagueConfigurationError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise LeagueConfigurationError(f"{name} must be >= {minimum}")
    return value


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LeagueConfigurationError(f"{name} must be a finite probability in [0, 1]")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise LeagueConfigurationError(f"{name} must be a finite probability in [0, 1]")
    return converted


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise LeagueConfigurationError(f"{name} must be a list of strings")
    result = tuple(_string(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise LeagueConfigurationError(f"{name} must not contain duplicates")
    return result


def _metadata(value: object, name: str) -> tuple[tuple[str, str], ...]:
    """Normalize JSON metadata to an immutable, deterministically ordered tuple."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        parsed: list[tuple[object, object]] = []
        for index, item in enumerate(value):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise LeagueConfigurationError(
                    f"{name}[{index}] must be a two-item key/value pair"
                )
            parsed.append((item[0], item[1]))
        items = parsed
    else:
        raise LeagueConfigurationError(f"{name} must be an object or key/value pairs")

    normalized: list[tuple[str, str]] = []
    for key, item in items:
        normalized.append((_string(key, f"{name}.key"), _string(item, f"{name}[{key!r}]")))
    normalized.sort()
    if len({key for key, _ in normalized}) != len(normalized):
        raise LeagueConfigurationError(f"{name} must not contain duplicate keys")
    return tuple(normalized)


def _metadata_dict(metadata: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return {key: value for key, value in metadata}


@dataclass(frozen=True, slots=True)
class DeckSpec:
    """One named eight-card deck in a deck-conditioned opponent scope."""

    deck_id: str
    cards: tuple[str, ...]
    tags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _string(self.deck_id, "deck_id")
        if len(self.cards) != 8:
            raise LeagueConfigurationError("deck cards must contain exactly eight cards")
        for index, card in enumerate(self.cards):
            _string(card, f"cards[{index}]")
        if len(set(self.cards)) != len(self.cards):
            raise LeagueConfigurationError("deck cards must not contain duplicates")
        if len(set(self.tags)) != len(self.tags):
            raise LeagueConfigurationError("deck tags must not contain duplicates")
        for index, tag in enumerate(self.tags):
            _string(tag, f"tags[{index}]")
        if not isinstance(self.metadata, tuple):
            raise LeagueConfigurationError("deck metadata must be normalized key/value pairs")
        _metadata(self.metadata, "metadata")

    def as_dict(self) -> dict[str, object]:
        return {
            "deck_id": self.deck_id,
            "cards": list(self.cards),
            "tags": list(self.tags),
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DeckSpec":
        if not isinstance(raw, Mapping):
            raise LeagueConfigurationError("deck specification must be an object")
        return cls(
            deck_id=raw.get("deck_id", ""),
            cards=_strings(raw.get("cards", []), "cards"),
            tags=_strings(raw.get("tags", []), "tags"),
            metadata=_metadata(raw.get("metadata"), "metadata"),
        )


@dataclass(frozen=True, slots=True)
class DeckConditionedOpponentScope:
    """Scope metadata for an opponent policy conditioned on deck identity."""

    scope_id: str
    ruleset_id: str
    decks: tuple[DeckSpec, ...]
    conditioning_mode: ConditioningMode = "card_tokens"
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _string(self.scope_id, "scope_id")
        _string(self.ruleset_id, "ruleset_id")
        if self.conditioning_mode not in {"deck_id", "card_tokens"}:
            raise LeagueConfigurationError(
                f"unsupported conditioning_mode: {self.conditioning_mode!r}"
            )
        if not self.decks:
            raise LeagueConfigurationError("opponent scope must contain at least one deck")
        if any(not isinstance(deck, DeckSpec) for deck in self.decks):
            raise LeagueConfigurationError("decks must contain DeckSpec values")
        deck_ids = [deck.deck_id for deck in self.decks]
        if len(set(deck_ids)) != len(deck_ids):
            raise LeagueConfigurationError("opponent scope deck IDs must be unique")
        if not isinstance(self.metadata, tuple):
            raise LeagueConfigurationError("scope metadata must be normalized key/value pairs")
        _metadata(self.metadata, "metadata")

    @property
    def deck_ids(self) -> tuple[str, ...]:
        return tuple(deck.deck_id for deck in self.decks)

    def deck(self, deck_id: str) -> DeckSpec:
        _string(deck_id, "deck_id")
        for deck in self.decks:
            if deck.deck_id == deck_id:
                return deck
        raise LeagueConfigurationError(
            f"deck {deck_id!r} is outside opponent scope {self.scope_id!r}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "scope_id": self.scope_id,
            "ruleset_id": self.ruleset_id,
            "conditioning_mode": self.conditioning_mode,
            "decks": [deck.as_dict() for deck in self.decks],
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DeckConditionedOpponentScope":
        if not isinstance(raw, Mapping):
            raise LeagueConfigurationError("opponent scope must be an object")
        raw_decks = raw.get("decks", [])
        if not isinstance(raw_decks, (list, tuple)):
            raise LeagueConfigurationError("scope decks must be a list of objects")
        return cls(
            scope_id=raw.get("scope_id", ""),
            ruleset_id=raw.get("ruleset_id", ""),
            decks=tuple(DeckSpec.from_mapping(item) for item in raw_decks),
            conditioning_mode=raw.get("conditioning_mode", "card_tokens"),
            metadata=_metadata(raw.get("metadata"), "metadata"),
        )


@dataclass(frozen=True, slots=True)
class HistoricalCheckpoint:
    """Metadata for a frozen policy artifact eligible for league sampling."""

    checkpoint_id: str
    agent_id: str
    role: AgentRole
    step: int
    artifact: str
    deck_scope_id: str
    deck_ids: tuple[str, ...]
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _string(self.checkpoint_id, "checkpoint_id")
        _string(self.agent_id, "agent_id")
        if self.role not in {"main_agent", "exploiter"}:
            raise LeagueConfigurationError(f"unsupported checkpoint role: {self.role!r}")
        _integer(self.step, "step", minimum=0)
        _string(self.artifact, "artifact")
        _string(self.deck_scope_id, "deck_scope_id")
        if not self.deck_ids:
            raise LeagueConfigurationError("historical checkpoint must name at least one deck")
        if len(set(self.deck_ids)) != len(self.deck_ids):
            raise LeagueConfigurationError("historical checkpoint deck IDs must be unique")
        for index, deck_id in enumerate(self.deck_ids):
            _string(deck_id, f"deck_ids[{index}]")
        if not isinstance(self.metadata, tuple):
            raise LeagueConfigurationError("checkpoint metadata must be normalized key/value pairs")
        _metadata(self.metadata, "metadata")

    def as_dict(self) -> dict[str, object]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "agent_id": self.agent_id,
            "role": self.role,
            "step": self.step,
            "artifact": self.artifact,
            "deck_scope_id": self.deck_scope_id,
            "deck_ids": list(self.deck_ids),
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HistoricalCheckpoint":
        if not isinstance(raw, Mapping):
            raise LeagueConfigurationError("historical checkpoint must be an object")
        return cls(
            checkpoint_id=raw.get("checkpoint_id", ""),
            agent_id=raw.get("agent_id", ""),
            role=raw.get("role", ""),
            step=raw.get("step", 0),
            artifact=raw.get("artifact", ""),
            deck_scope_id=raw.get("deck_scope_id", ""),
            deck_ids=_strings(raw.get("deck_ids", []), "deck_ids"),
            metadata=_metadata(raw.get("metadata"), "metadata"),
        )


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    """Immutable league scope and role-specific opponent-source weights."""

    league_id: str
    seed: int
    scope: DeckConditionedOpponentScope
    main_agent_id: str = "main"
    exploiter_ids: tuple[str, ...] = ()
    historical_checkpoints: tuple[HistoricalCheckpoint, ...] = ()
    main_agent_historical_probability: float = 0.75
    main_agent_exploiter_probability: float = 0.25
    exploiter_main_agent_probability: float = 0.50
    exploiter_historical_probability: float = 0.50
    exploiter_reset_interval: int | None = None

    def __post_init__(self) -> None:
        _string(self.league_id, "league_id")
        _integer(self.seed, "seed")
        if not isinstance(self.scope, DeckConditionedOpponentScope):
            raise LeagueConfigurationError("scope must be a DeckConditionedOpponentScope")
        _string(self.main_agent_id, "main_agent_id")
        if self.main_agent_id in self.exploiter_ids:
            raise LeagueConfigurationError("main_agent_id must not be an exploiter ID")
        if len(set(self.exploiter_ids)) != len(self.exploiter_ids):
            raise LeagueConfigurationError("exploiter_ids must not contain duplicates")
        for index, agent_id in enumerate(self.exploiter_ids):
            _string(agent_id, f"exploiter_ids[{index}]")
        if self.exploiter_reset_interval is not None:
            _integer(
                self.exploiter_reset_interval,
                "exploiter_reset_interval",
                minimum=1,
            )
            if not self.exploiter_ids:
                raise LeagueConfigurationError(
                    "exploiter_reset_interval requires exploiter IDs"
                )
        if len(
            {
                checkpoint.checkpoint_id
                for checkpoint in self.historical_checkpoints
            }
        ) != len(self.historical_checkpoints):
            raise LeagueConfigurationError("historical checkpoint IDs must be unique")
        for index, checkpoint in enumerate(self.historical_checkpoints):
            if not isinstance(checkpoint, HistoricalCheckpoint):
                raise LeagueConfigurationError(
                    f"historical_checkpoints[{index}] must be a HistoricalCheckpoint"
                )
            if checkpoint.deck_scope_id != self.scope.scope_id:
                raise LeagueConfigurationError(
                    f"checkpoint {checkpoint.checkpoint_id!r} belongs to a different deck scope"
                )
            unknown_decks = sorted(set(checkpoint.deck_ids) - set(self.scope.deck_ids))
            if unknown_decks:
                raise LeagueConfigurationError(
                    f"checkpoint {checkpoint.checkpoint_id!r} references decks outside scope: "
                    f"{unknown_decks}"
                )
        probabilities = (
            ("main_agent_historical_probability", self.main_agent_historical_probability),
            ("main_agent_exploiter_probability", self.main_agent_exploiter_probability),
            ("exploiter_main_agent_probability", self.exploiter_main_agent_probability),
            ("exploiter_historical_probability", self.exploiter_historical_probability),
        )
        for name, value in probabilities:
            _probability(value, name)
        if not math.isclose(
            self.main_agent_historical_probability
            + self.main_agent_exploiter_probability,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise LeagueConfigurationError(
                "main-agent opponent probabilities must sum to 1"
            )
        if not math.isclose(
            self.exploiter_main_agent_probability
            + self.exploiter_historical_probability,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise LeagueConfigurationError(
                "exploiter opponent probabilities must sum to 1"
            )
        if self.main_agent_historical_probability > 0.0 and not self.historical_checkpoints:
            raise LeagueConfigurationError(
                "main-agent historical probability requires historical checkpoints"
            )
        if self.exploiter_historical_probability > 0.0 and not self.historical_checkpoints:
            raise LeagueConfigurationError(
                "exploiter historical probability requires historical checkpoints"
            )
        if self.main_agent_exploiter_probability > 0.0 and not self.exploiter_ids:
            raise LeagueConfigurationError(
                "main-agent exploiter probability requires exploiter IDs"
            )

    def role_for(self, agent_id: str) -> AgentRole:
        _string(agent_id, "agent_id")
        if agent_id == self.main_agent_id:
            return "main_agent"
        if agent_id in self.exploiter_ids:
            return "exploiter"
        raise LeagueConfigurationError(
            f"agent {agent_id!r} is not declared in league {self.league_id!r}"
        )

    def source_probabilities(
        self, role: AgentRole
    ) -> tuple[tuple[OpponentSource, float], ...]:
        if role == "main_agent":
            return (
                ("historical", self.main_agent_historical_probability),
                ("exploiter", self.main_agent_exploiter_probability),
            )
        if role == "exploiter":
            return (
                ("main_agent", self.exploiter_main_agent_probability),
                ("historical", self.exploiter_historical_probability),
            )
        raise LeagueConfigurationError(f"unsupported agent role: {role!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": LEAGUE_SCHEMA_VERSION,
            "league_id": self.league_id,
            "seed": self.seed,
            "main_agent_id": self.main_agent_id,
            "exploiter_ids": list(self.exploiter_ids),
            "scope": self.scope.as_dict(),
            "historical_checkpoints": [
                checkpoint.as_dict() for checkpoint in self.historical_checkpoints
            ],
            "main_agent_historical_probability": self.main_agent_historical_probability,
            "main_agent_exploiter_probability": self.main_agent_exploiter_probability,
            "exploiter_main_agent_probability": self.exploiter_main_agent_probability,
            "exploiter_historical_probability": self.exploiter_historical_probability,
            "exploiter_reset_interval": self.exploiter_reset_interval,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LeagueConfig":
        if not isinstance(raw, Mapping):
            raise LeagueConfigurationError("league configuration must be an object")
        schema_version = raw.get("schema_version", LEAGUE_SCHEMA_VERSION)
        if schema_version != LEAGUE_SCHEMA_VERSION:
            raise LeagueConfigurationError(
                f"unsupported league schema: {schema_version!r}"
            )
        raw_checkpoints = raw.get("historical_checkpoints", [])
        if not isinstance(raw_checkpoints, (list, tuple)):
            raise LeagueConfigurationError(
                "historical_checkpoints must be a list of objects"
            )
        return cls(
            league_id=raw.get("league_id", ""),
            seed=raw.get("seed", 0),
            scope=DeckConditionedOpponentScope.from_mapping(raw.get("scope", {})),
            main_agent_id=raw.get("main_agent_id", "main"),
            exploiter_ids=_strings(raw.get("exploiter_ids", []), "exploiter_ids"),
            historical_checkpoints=tuple(
                HistoricalCheckpoint.from_mapping(item) for item in raw_checkpoints
            ),
            main_agent_historical_probability=raw.get(
                "main_agent_historical_probability", 0.75
            ),
            main_agent_exploiter_probability=raw.get(
                "main_agent_exploiter_probability", 0.25
            ),
            exploiter_main_agent_probability=raw.get(
                "exploiter_main_agent_probability", 0.50
            ),
            exploiter_historical_probability=raw.get(
                "exploiter_historical_probability", 0.50
            ),
            exploiter_reset_interval=raw.get("exploiter_reset_interval"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "LeagueConfig":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LeagueConfigurationError(
                f"cannot load league configuration {source}: {error}"
            ) from error
        return cls.from_mapping(raw)


@dataclass(frozen=True, slots=True)
class LeagueRunState:
    """Minimal serializable cursor for replaying a league schedule."""

    league_id: str
    next_match_index: int = 0
    exploiter_reset_count: int = 0

    def __post_init__(self) -> None:
        _string(self.league_id, "league_id")
        _integer(self.next_match_index, "next_match_index", minimum=0)
        _integer(self.exploiter_reset_count, "exploiter_reset_count", minimum=0)

    def after_match(self, count: int = 1) -> "LeagueRunState":
        _integer(count, "count", minimum=1)
        return LeagueRunState(
            league_id=self.league_id,
            next_match_index=self.next_match_index + count,
            exploiter_reset_count=self.exploiter_reset_count,
        )

    def after_exploiter_reset(self) -> "LeagueRunState":
        return LeagueRunState(
            league_id=self.league_id,
            next_match_index=self.next_match_index,
            exploiter_reset_count=self.exploiter_reset_count + 1,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": LEAGUE_SCHEMA_VERSION,
            "league_id": self.league_id,
            "next_match_index": self.next_match_index,
            "exploiter_reset_count": self.exploiter_reset_count,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LeagueRunState":
        if not isinstance(raw, Mapping):
            raise LeagueConfigurationError("league run state must be an object")
        schema_version = raw.get("schema_version", LEAGUE_SCHEMA_VERSION)
        if schema_version != LEAGUE_SCHEMA_VERSION:
            raise LeagueConfigurationError(
                f"unsupported league state schema: {schema_version!r}"
            )
        return cls(
            league_id=raw.get("league_id", ""),
            next_match_index=raw.get("next_match_index", 0),
            exploiter_reset_count=raw.get("exploiter_reset_count", 0),
        )


@dataclass(frozen=True, slots=True)
class OpponentSelection:
    """Serializable result of one deterministic opponent/deck draw."""

    match_index: int
    selection_seed: int
    learner_agent_id: str
    learner_role: AgentRole
    source: OpponentSource
    opponent_agent_id: str
    opponent_role: AgentRole
    checkpoint_id: str | None
    artifact: str | None
    deck_scope_id: str
    ruleset_id: str
    conditioning_mode: ConditioningMode
    deck_id: str
    deck_cards: tuple[str, ...]
    scope_metadata: tuple[tuple[str, str], ...] = ()
    checkpoint_metadata: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": LEAGUE_SCHEMA_VERSION,
            "match_index": self.match_index,
            "selection_seed": self.selection_seed,
            "learner_agent_id": self.learner_agent_id,
            "learner_role": self.learner_role,
            "source": self.source,
            "opponent_agent_id": self.opponent_agent_id,
            "opponent_role": self.opponent_role,
            "checkpoint_id": self.checkpoint_id,
            "artifact": self.artifact,
            "deck_scope_id": self.deck_scope_id,
            "ruleset_id": self.ruleset_id,
            "conditioning_mode": self.conditioning_mode,
            "deck_id": self.deck_id,
            "deck_cards": list(self.deck_cards),
            "scope_metadata": _metadata_dict(self.scope_metadata),
            "checkpoint_metadata": _metadata_dict(self.checkpoint_metadata),
        }


def deterministic_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 64-bit seed without using process-randomized ``hash``."""

    _integer(base_seed, "base_seed")
    try:
        encoded = json.dumps(
            [base_seed, *parts],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LeagueConfigurationError(
            "deterministic seed parts must be JSON-serializable"
        ) from error
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _unit_draw(seed: int, purpose: str) -> float:
    return deterministic_seed(seed, purpose) / float(1 << 64)


def _weighted_choice(
    seed: int,
    purpose: str,
    weighted: tuple[tuple[str, float], ...],
) -> str:
    available = tuple((item, weight) for item, weight in weighted if weight > 0.0)
    total = sum(weight for _, weight in available)
    if not available or total <= 0.0:
        raise LeagueSamplingError(f"no positive candidates available for {purpose}")
    draw = _unit_draw(seed, purpose) * total
    cumulative = 0.0
    for item, weight in available:
        cumulative += weight
        if draw < cumulative:
            return item
    return available[-1][0]


def _stable_choice(seed: int, purpose: str, values: tuple[str, ...]) -> str:
    if not values:
        raise LeagueSamplingError(f"no candidates available for {purpose}")
    index = deterministic_seed(seed, purpose) % len(values)
    return values[index]


class LeagueSampler:
    """Pure deterministic sampler over current roles and frozen checkpoints."""

    def __init__(self, config: LeagueConfig) -> None:
        if not isinstance(config, LeagueConfig):
            raise LeagueConfigurationError("config must be a LeagueConfig")
        self.config = config

    def seed_for(self, learner_agent_id: str, match_index: int) -> int:
        role = self.config.role_for(learner_agent_id)
        _integer(match_index, "match_index", minimum=0)
        return deterministic_seed(
            self.config.seed,
            self.config.league_id,
            learner_agent_id,
            role,
            match_index,
        )

    def sample(
        self,
        learner_agent_id: str,
        match_index: int,
        *,
        deck_id: str | None = None,
    ) -> OpponentSelection:
        """Select a role/checkpoint/deck reproducibly for one match index."""

        learner_role = self.config.role_for(learner_agent_id)
        _integer(match_index, "match_index", minimum=0)
        selection_seed = self.seed_for(learner_agent_id, match_index)
        source = _weighted_choice(
            selection_seed,
            "opponent-source",
            self.config.source_probabilities(learner_role),
        )

        checkpoint: HistoricalCheckpoint | None = None
        if source == "historical":
            candidates = self.config.historical_checkpoints
            if learner_role == "exploiter":
                candidates = tuple(
                    item for item in candidates if item.agent_id != learner_agent_id
                )
            if not candidates:
                raise LeagueSamplingError(
                    f"no historical checkpoint can oppose {learner_agent_id!r}"
                )
            checkpoint = candidates[
                deterministic_seed(selection_seed, "historical-checkpoint")
                % len(candidates)
            ]
            opponent_agent_id = checkpoint.agent_id
            opponent_role = checkpoint.role
            artifact = checkpoint.artifact
            allowed_decks = checkpoint.deck_ids
        elif source == "exploiter":
            candidates = tuple(
                agent_id
                for agent_id in self.config.exploiter_ids
                if agent_id != learner_agent_id
            )
            if not candidates:
                raise LeagueSamplingError(
                    f"no exploiter can oppose {learner_agent_id!r}"
                )
            opponent_agent_id = _stable_choice(
                selection_seed,
                "exploiter-agent",
                candidates,
            )
            opponent_role = "exploiter"
            artifact = None
            allowed_decks = self.config.scope.deck_ids
        elif source == "main_agent":
            opponent_agent_id = self.config.main_agent_id
            opponent_role = "main_agent"
            artifact = None
            allowed_decks = self.config.scope.deck_ids
        else:
            raise LeagueSamplingError(f"unsupported opponent source: {source!r}")

        if deck_id is not None:
            _string(deck_id, "deck_id")
            if deck_id not in allowed_decks:
                raise LeagueSamplingError(
                    f"deck {deck_id!r} is not available for selected opponent"
                )
            selected_deck_id = deck_id
        else:
            selected_deck_id = _stable_choice(
                selection_seed,
                "opponent-deck",
                tuple(allowed_decks),
            )
        selected_deck = self.config.scope.deck(selected_deck_id)

        return OpponentSelection(
            match_index=match_index,
            selection_seed=selection_seed,
            learner_agent_id=learner_agent_id,
            learner_role=learner_role,
            source=source,
            opponent_agent_id=opponent_agent_id,
            opponent_role=opponent_role,
            checkpoint_id=None if checkpoint is None else checkpoint.checkpoint_id,
            artifact=artifact,
            deck_scope_id=self.config.scope.scope_id,
            ruleset_id=self.config.scope.ruleset_id,
            conditioning_mode=self.config.scope.conditioning_mode,
            deck_id=selected_deck.deck_id,
            deck_cards=selected_deck.cards,
            scope_metadata=self.config.scope.metadata,
            checkpoint_metadata=() if checkpoint is None else checkpoint.metadata,
        )


def _rating(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LeagueConfigurationError(f"{name} must be a finite rating")
    converted = float(value)
    if not math.isfinite(converted):
        raise LeagueConfigurationError(f"{name} must be a finite rating")
    return converted


@dataclass(frozen=True, slots=True)
class LeagueRatingBook:
    """Immutable Elo ratings that can be checkpointed with league metadata."""

    ratings: tuple[tuple[str, float], ...] = ()
    default_rating: float = 1500.0
    k_factor: float = 32.0

    def __post_init__(self) -> None:
        _rating(self.default_rating, "default_rating")
        if isinstance(self.k_factor, bool) or not isinstance(self.k_factor, (int, float)):
            raise LeagueConfigurationError("k_factor must be a finite positive number")
        if not math.isfinite(float(self.k_factor)) or float(self.k_factor) <= 0.0:
            raise LeagueConfigurationError("k_factor must be a finite positive number")
        normalized: list[tuple[str, float]] = []
        for index, (agent_id, value) in enumerate(self.ratings):
            normalized.append((_string(agent_id, f"ratings[{index}].agent_id"), _rating(value, f"ratings[{index}]")))
        if len({agent_id for agent_id, _ in normalized}) != len(normalized):
            raise LeagueConfigurationError("rating agent IDs must be unique")
        if tuple(sorted(normalized)) != tuple(normalized):
            raise LeagueConfigurationError("ratings must be sorted by agent ID")

    def rating(self, agent_id: str) -> float:
        _string(agent_id, "agent_id")
        for current_id, value in self.ratings:
            if current_id == agent_id:
                return value
        return float(self.default_rating)

    def after_match(
        self,
        player_a: str,
        player_b: str,
        outcome_for_a: LeagueOutcome,
    ) -> "LeagueRatingBook":
        _string(player_a, "player_a")
        _string(player_b, "player_b")
        if player_a == player_b:
            raise LeagueConfigurationError("a match requires two distinct rating IDs")
        if outcome_for_a not in {"win", "draw", "loss"}:
            raise LeagueConfigurationError(f"unsupported league outcome: {outcome_for_a!r}")
        score_a = {"win": 1.0, "draw": 0.5, "loss": 0.0}[outcome_for_a]
        rating_a = self.rating(player_a)
        rating_b = self.rating(player_b)
        expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
        delta = float(self.k_factor) * (score_a - expected_a)
        values = dict(self.ratings)
        values[player_a] = rating_a + delta
        values[player_b] = rating_b - delta
        return LeagueRatingBook(
            ratings=tuple(sorted(values.items())),
            default_rating=self.default_rating,
            k_factor=self.k_factor,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": LEAGUE_SCHEMA_VERSION,
            "ratings": {agent_id: value for agent_id, value in self.ratings},
            "default_rating": self.default_rating,
            "k_factor": self.k_factor,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LeagueRatingBook":
        if not isinstance(raw, Mapping):
            raise LeagueConfigurationError("rating book must be an object")
        schema_version = raw.get("schema_version", LEAGUE_SCHEMA_VERSION)
        if schema_version != LEAGUE_SCHEMA_VERSION:
            raise LeagueConfigurationError(f"unsupported rating book schema: {schema_version!r}")
        raw_ratings = raw.get("ratings", {})
        if not isinstance(raw_ratings, Mapping):
            raise LeagueConfigurationError("ratings must be an object")
        return cls(
            ratings=tuple(sorted((_string(key, "rating agent ID"), _rating(value, "rating")) for key, value in raw_ratings.items())),
            default_rating=raw.get("default_rating", 1500.0),
            k_factor=raw.get("k_factor", 32.0),
        )


@dataclass(frozen=True, slots=True)
class LeaguePayoffStats:
    """Directional match outcomes for one learner/opponent pair."""

    wins: int = 0
    draws: int = 0
    losses: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("wins", self.wins),
            ("draws", self.draws),
            ("losses", self.losses),
        ):
            _integer(value, name, minimum=0)

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float | None:
        """Return the learner's scored win probability, or ``None`` if new."""

        if self.games == 0:
            return None
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def win_rate(self) -> float | None:
        """Alias for :attr:`score` using standard league terminology."""

        return self.score

    def after_match(self, outcome: LeagueOutcome) -> "LeaguePayoffStats":
        if outcome not in {"win", "draw", "loss"}:
            raise LeagueConfigurationError(f"unsupported league outcome: {outcome!r}")
        increments = {
            "win": (1, 0, 0),
            "draw": (0, 1, 0),
            "loss": (0, 0, 1),
        }[outcome]
        return LeaguePayoffStats(
            wins=self.wins + increments[0],
            draws=self.draws + increments[1],
            losses=self.losses + increments[2],
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LeaguePayoffStats":
        if not isinstance(raw, Mapping):
            raise LeagueConfigurationError("payoff statistics must be an object")
        return cls(
            wins=raw.get("wins", 0),
            draws=raw.get("draws", 0),
            losses=raw.get("losses", 0),
        )


PAYOFF_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LeaguePayoffBook:
    """Immutable, serializable directional payoff history."""

    records: tuple[tuple[str, str, LeaguePayoffStats], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            raise LeagueConfigurationError("payoff records must be a tuple")
        previous_key: tuple[str, str] | None = None
        for index, record in enumerate(self.records):
            if not isinstance(record, tuple) or len(record) != 3:
                raise LeagueConfigurationError(
                    f"payoff records[{index}] must be a three-item tuple"
                )
            learner_id = _string(record[0], f"payoff records[{index}].learner_agent_id")
            opponent_id = _string(record[1], f"payoff records[{index}].opponent_agent_id")
            if learner_id == opponent_id:
                raise LeagueConfigurationError("payoff records require distinct agents")
            stats = record[2]
            if not isinstance(stats, LeaguePayoffStats):
                raise LeagueConfigurationError(
                    f"payoff records[{index}].stats must be LeaguePayoffStats"
                )
            key = (learner_id, opponent_id)
            if previous_key is not None and key <= previous_key:
                raise LeagueConfigurationError(
                    "payoff records must be sorted by learner and opponent ID"
                )
            previous_key = key

    def stats(self, learner_agent_id: str, opponent_agent_id: str) -> LeaguePayoffStats:
        """Return directional stats, using zero games for an unseen pair."""

        learner_id = _string(learner_agent_id, "learner_agent_id")
        opponent_id = _string(opponent_agent_id, "opponent_agent_id")
        if learner_id == opponent_id:
            raise LeagueConfigurationError("payoff queries require distinct agents")
        for current_learner, current_opponent, stats in self.records:
            if current_learner == learner_id and current_opponent == opponent_id:
                return stats
        return LeaguePayoffStats()

    def after_match(
        self,
        learner_agent_id: str,
        opponent_agent_id: str,
        outcome: LeagueOutcome,
    ) -> "LeaguePayoffBook":
        """Return a new book with one learner-perspective outcome recorded."""

        learner_id = _string(learner_agent_id, "learner_agent_id")
        opponent_id = _string(opponent_agent_id, "opponent_agent_id")
        if learner_id == opponent_id:
            raise LeagueConfigurationError("a payoff match requires two distinct agents")
        if outcome not in {"win", "draw", "loss"}:
            raise LeagueConfigurationError(f"unsupported league outcome: {outcome!r}")

        key = (learner_id, opponent_id)
        values = {
            (current_learner, current_opponent): stats
            for current_learner, current_opponent, stats in self.records
        }
        values[key] = values.get(key, LeaguePayoffStats()).after_match(outcome)
        return LeaguePayoffBook(
            records=tuple(
                (current_learner, current_opponent, stats)
                for (current_learner, current_opponent), stats in sorted(values.items())
            )
        )

    def record(
        self,
        learner_agent_id: str,
        opponent_agent_id: str,
        outcome: LeagueOutcome,
    ) -> "LeaguePayoffBook":
        """Readable alias for :meth:`after_match`."""

        return self.after_match(learner_agent_id, opponent_agent_id, outcome)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": PAYOFF_SCHEMA_VERSION,
            "records": [
                {
                    "learner_agent_id": learner_id,
                    "opponent_agent_id": opponent_id,
                    **stats.as_dict(),
                }
                for learner_id, opponent_id, stats in self.records
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LeaguePayoffBook":
        if not isinstance(raw, Mapping):
            raise LeagueConfigurationError("payoff book must be an object")
        schema_version = raw.get("schema_version", PAYOFF_SCHEMA_VERSION)
        if schema_version != PAYOFF_SCHEMA_VERSION:
            raise LeagueConfigurationError(
                f"unsupported payoff book schema: {schema_version!r}"
            )
        raw_records = raw.get("records", [])
        if not isinstance(raw_records, (list, tuple)):
            raise LeagueConfigurationError("payoff records must be a list of objects")
        records: list[tuple[str, str, LeaguePayoffStats]] = []
        for index, item in enumerate(raw_records):
            if not isinstance(item, Mapping):
                raise LeagueConfigurationError(
                    f"payoff records[{index}] must be an object"
                )
            records.append(
                (
                    _string(item.get("learner_agent_id", ""), "learner_agent_id"),
                    _string(item.get("opponent_agent_id", ""), "opponent_agent_id"),
                    LeaguePayoffStats.from_mapping(item),
                )
            )
        return cls(records=tuple(sorted(records, key=lambda value: (value[0], value[1]))))

    @classmethod
    def from_json(cls, source: str | Path) -> "LeaguePayoffBook":
        path = Path(source)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LeagueConfigurationError(
                f"cannot load payoff book {path}: {error}"
            ) from error
        return cls.from_mapping(raw)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LeagueConfigurationError(f"{name} must be a finite positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise LeagueConfigurationError(f"{name} must be a finite positive number")
    return converted


@dataclass(frozen=True, slots=True)
class PFSPOpponentSampler:
    """Deterministically sample candidates using payoff-aware PFSP weights.

    For a tested matchup, the base weight is ``p * (1 - p)`` where ``p`` is
    the learner's scored win rate.  This emphasizes opponents near the
    learner's current level and down-weights solved matchups.  Unseen
    opponents receive the maximum base weight so discovery is preserved;
    ``minimum_weight`` and ``exploration_probability`` keep extreme records
    from disappearing completely.  Candidate IDs should distinguish frozen
    checkpoints when needed (for example ``main@step-100``).

    This class is intentionally independent of :class:`LeagueSampler`: the
    caller supplies the already-valid candidate IDs and can use the selected
    ID to resolve a full :class:`OpponentSelection`.  Existing deterministic
    league schedules therefore remain unchanged until a caller opts in.
    """

    payoff_book: LeaguePayoffBook = LeaguePayoffBook()
    seed: int = 0
    minimum_weight: float = 0.01
    unknown_opponent_weight: float = 0.25
    exploration_probability: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.payoff_book, LeaguePayoffBook):
            raise LeagueConfigurationError("payoff_book must be a LeaguePayoffBook")
        _integer(self.seed, "seed")
        _positive_number(self.minimum_weight, "minimum_weight")
        _positive_number(self.unknown_opponent_weight, "unknown_opponent_weight")
        _probability(self.exploration_probability, "exploration_probability")

    def seed_for(
        self,
        learner_agent_id: str,
        match_index: int,
        opponent_ids: tuple[str, ...],
    ) -> int:
        learner_id = _string(learner_agent_id, "learner_agent_id")
        _integer(match_index, "match_index", minimum=0)
        return deterministic_seed(
            self.seed,
            "pfsp",
            learner_id,
            match_index,
            opponent_ids,
        )

    @staticmethod
    def _candidate_tuple(opponent_ids: object) -> tuple[str, ...]:
        if not isinstance(opponent_ids, (list, tuple)):
            raise LeagueSamplingError("opponent_ids must be a non-empty list of strings")
        candidates = tuple(
            _string(value, f"opponent_ids[{index}]")
            for index, value in enumerate(opponent_ids)
        )
        if not candidates:
            raise LeagueSamplingError("opponent_ids must contain at least one candidate")
        if len(set(candidates)) != len(candidates):
            raise LeagueSamplingError("opponent_ids must not contain duplicates")
        return tuple(sorted(candidates))

    def base_weights(
        self,
        learner_agent_id: str,
        opponent_ids: list[str] | tuple[str, ...],
    ) -> tuple[tuple[str, float], ...]:
        """Return unnormalized PFSP weights for inspection and diagnostics."""

        learner_id = _string(learner_agent_id, "learner_agent_id")
        candidates = self._candidate_tuple(opponent_ids)
        if learner_id in candidates:
            raise LeagueSamplingError("opponent_ids must not contain the learner")
        weights: list[tuple[str, float]] = []
        for opponent_id in candidates:
            stats = self.payoff_book.stats(learner_id, opponent_id)
            if stats.score is None:
                weight = self.unknown_opponent_weight
            else:
                weight = max(self.minimum_weight, stats.score * (1.0 - stats.score))
            weights.append((opponent_id, weight))
        return tuple(weights)

    def weights(
        self,
        learner_agent_id: str,
        opponent_ids: list[str] | tuple[str, ...],
    ) -> tuple[tuple[str, float], ...]:
        """Return normalized sampling weights after uniform exploration mix."""

        base = self.base_weights(learner_agent_id, opponent_ids)
        total = sum(weight for _, weight in base)
        count = len(base)
        epsilon = self.exploration_probability
        return tuple(
            (
                opponent_id,
                (1.0 - epsilon) * (weight / total) + epsilon / count,
            )
            for opponent_id, weight in base
        )

    def sample(
        self,
        learner_agent_id: str,
        match_index: int,
        opponent_ids: list[str] | tuple[str, ...],
    ) -> str:
        """Return one reproducible opponent ID from the supplied candidates."""

        candidates = self._candidate_tuple(opponent_ids)
        weights = self.weights(learner_agent_id, candidates)
        selection_seed = self.seed_for(learner_agent_id, match_index, candidates)
        return _weighted_choice(selection_seed, "pfsp-opponent", weights)


@dataclass(frozen=True, slots=True)
class LeagueMatchRecord:
    """Serializable outcome and rating transition for one planned match."""

    league_id: str
    match_index: int
    selection_seed: int
    learner_agent_id: str
    opponent_agent_id: str
    opponent_rating_id: str
    outcome: LeagueOutcome
    learner_rating_before: float
    learner_rating_after: float
    opponent_rating_before: float
    opponent_rating_after: float
    selection: OpponentSelection

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": LEAGUE_SCHEMA_VERSION,
            "league_id": self.league_id,
            "match_index": self.match_index,
            "selection_seed": self.selection_seed,
            "learner_agent_id": self.learner_agent_id,
            "opponent_agent_id": self.opponent_agent_id,
            "opponent_rating_id": self.opponent_rating_id,
            "outcome": self.outcome,
            "learner_rating_before": self.learner_rating_before,
            "learner_rating_after": self.learner_rating_after,
            "opponent_rating_before": self.opponent_rating_before,
            "opponent_rating_after": self.opponent_rating_after,
            "selection": self.selection.as_dict(),
        }


class LeagueOrchestrator:
    """Plan and record deterministic league matches around a caller runner.

    The orchestrator owns schedule cursor and ratings.  It does not load a
    checkpoint or run physics; ``match_runner`` receives the sealed
    :class:`OpponentSelection` and returns the learner-perspective outcome.
    This keeps artifact loading, policy construction, and simulator readiness
    gates in the application that actually owns them.
    """

    def __init__(
        self,
        config: LeagueConfig,
        *,
        run_state: LeagueRunState | None = None,
        ratings: LeagueRatingBook | None = None,
        payoff_book: LeaguePayoffBook | None = None,
    ) -> None:
        if not isinstance(config, LeagueConfig):
            raise LeagueConfigurationError("config must be a LeagueConfig")
        if run_state is not None and run_state.league_id != config.league_id:
            raise LeagueConfigurationError("run state belongs to a different league")
        if ratings is not None and not isinstance(ratings, LeagueRatingBook):
            raise LeagueConfigurationError("ratings must be a LeagueRatingBook")
        if payoff_book is not None and not isinstance(payoff_book, LeaguePayoffBook):
            raise LeagueConfigurationError("payoff_book must be a LeaguePayoffBook")
        self.config = config
        self.sampler = LeagueSampler(config)
        self.run_state = run_state or LeagueRunState(config.league_id)
        self.ratings = ratings or LeagueRatingBook()
        self.payoff_book = payoff_book or LeaguePayoffBook()

    def exploiter_reset_due(self) -> bool:
        """Return whether the next match requires an exploiter reset."""

        interval = self.config.exploiter_reset_interval
        return bool(
            interval is not None
            and self.run_state.next_match_index > 0
            and self.run_state.exploiter_reset_count
            < self.run_state.next_match_index // interval
        )

    def reset_exploiters(
        self,
        callback: Callable[[tuple[str, ...]], object],
    ) -> bool:
        """Run and record the scheduled exploiter reset handoff.

        A configured reset cannot be silently skipped.  The callback owns
        learner/checkpoint state; this coordinator only supplies the declared
        exploiter IDs and advances the serializable reset counter after the
        callback succeeds.
        """

        if not self.exploiter_reset_due():
            return False
        if not callable(callback):
            raise LeagueConfigurationError(
                "exploiter reset callback is required when a reset is due"
            )
        callback(tuple(self.config.exploiter_ids))
        self.run_state = self.run_state.after_exploiter_reset()
        return True

    def plan(
        self,
        learner_agent_id: str,
        *,
        deck_id: str | None = None,
    ) -> OpponentSelection:
        """Plan the next match without advancing the cursor."""

        if self.run_state.league_id != self.config.league_id:
            raise LeagueConfigurationError("run state belongs to a different league")
        if self.exploiter_reset_due():
            raise LeagueConfigurationError(
                "periodic exploiter reset is due; call reset_exploiters before planning"
            )
        return self.sampler.sample(
            learner_agent_id,
            self.run_state.next_match_index,
            deck_id=deck_id,
        )

    def record(
        self,
        selection: OpponentSelection,
        outcome: LeagueOutcome,
    ) -> LeagueMatchRecord:
        """Record one completed planned match and advance the schedule."""

        if selection.deck_scope_id != self.config.scope.scope_id:
            raise LeagueConfigurationError("selection belongs to a different deck scope")
        if self.exploiter_reset_due():
            raise LeagueConfigurationError(
                "periodic exploiter reset is due; call reset_exploiters before recording"
            )
        if selection.match_index != self.run_state.next_match_index:
            raise LeagueConfigurationError(
                "league matches must be recorded in monotonically increasing order"
            )
        if outcome not in {"win", "draw", "loss"}:
            raise LeagueConfigurationError(f"unsupported league outcome: {outcome!r}")
        learner_id = selection.learner_agent_id
        opponent_id = selection.opponent_agent_id
        opponent_rating_id = (
            f"{opponent_id}@{selection.checkpoint_id}"
            if selection.checkpoint_id is not None
            else opponent_id
        )
        learner_before = self.ratings.rating(learner_id)
        opponent_before = self.ratings.rating(opponent_rating_id)
        updated = self.ratings.after_match(learner_id, opponent_rating_id, outcome)
        self.ratings = updated
        self.payoff_book = self.payoff_book.after_match(
            learner_id,
            opponent_rating_id,
            outcome,
        )
        self.run_state = self.run_state.after_match()
        return LeagueMatchRecord(
            league_id=self.config.league_id,
            match_index=selection.match_index,
            selection_seed=selection.selection_seed,
            learner_agent_id=learner_id,
            opponent_agent_id=opponent_id,
            opponent_rating_id=opponent_rating_id,
            outcome=outcome,
            learner_rating_before=learner_before,
            learner_rating_after=updated.rating(learner_id),
            opponent_rating_before=opponent_before,
            opponent_rating_after=updated.rating(opponent_rating_id),
            selection=selection,
        )

    def run_one(
        self,
        learner_agent_id: str,
        match_runner: Callable[[OpponentSelection], LeagueOutcome],
        *,
        deck_id: str | None = None,
        exploiter_reset_callback: Callable[[tuple[str, ...]], object] | None = None,
    ) -> LeagueMatchRecord:
        """Plan, delegate one match, and record the returned outcome."""

        if not callable(match_runner):
            raise TypeError("match_runner must be callable")
        if self.exploiter_reset_due():
            if exploiter_reset_callback is None:
                raise LeagueConfigurationError(
                    "periodic exploiter reset is due; provide exploiter_reset_callback"
                )
            self.reset_exploiters(exploiter_reset_callback)
        selection = self.plan(learner_agent_id, deck_id=deck_id)
        return self.record(selection, match_runner(selection))

    def sample_pfsp_opponent(
        self,
        learner_agent_id: str,
        opponent_ids: list[str] | tuple[str, ...],
        *,
        seed: int | None = None,
    ) -> str:
        """Sample a payoff-aware opponent at the current league cursor."""

        sampler = PFSPOpponentSampler(
            payoff_book=self.payoff_book,
            seed=self.config.seed if seed is None else seed,
        )
        return sampler.sample(
            learner_agent_id,
            self.run_state.next_match_index,
            opponent_ids,
        )


__all__ = [
    "AgentRole",
    "ConditioningMode",
    "DeckConditionedOpponentScope",
    "DeckSpec",
    "HistoricalCheckpoint",
    "LEAGUE_SCHEMA_VERSION",
    "LeagueConfig",
    "LeagueConfigurationError",
    "LeagueRunState",
    "LeagueSampler",
    "LeagueSamplingError",
    "LeagueMatchRecord",
    "LeagueOrchestrator",
    "LeagueOutcome",
    "LeaguePayoffBook",
    "LeaguePayoffStats",
    "LeagueRatingBook",
    "OpponentSelection",
    "OpponentSource",
    "PAYOFF_SCHEMA_VERSION",
    "PFSPOpponentSampler",
    "deterministic_seed",
]
