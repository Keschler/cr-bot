"""Pinned learned-opponent snapshots and train-only population sampling.

This is population infrastructure for the existing Hog 2.6 checkpoint contract.
It does not turn a Hog checkpoint into an opponent playing another deck. Model
imports and artifact validation happen on construction, not module import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, TYPE_CHECKING

from simulator.roster import PLAYER_DECK

if TYPE_CHECKING:
    from .model_v4 import ModelConfigV4


OPPONENT_POPULATION_SCHEMA_VERSION = 1


def _identity(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty trimmed string")


def _seed(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")


def _stable_seed(*values: Any) -> int:
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class LearnedOpponentMember:
    """One immutable snapshot identity, architecture and inference contract.

    Train/dev membership is explicit. Exact checkpoint bytes are checked when
    reading the manifest, and again by the opponent loader before first use.
    """

    member_id: str
    checkpoint: str
    sha256: str
    model_config: ModelConfigV4 | Mapping[str, Any]
    split: str
    deck: tuple[str, ...] = tuple(PLAYER_DECK)
    inference: str = "stochastic"
    sampling_seed: int = 0
    generation: int = 0

    def __post_init__(self) -> None:
        from .policy_opponent import exact_model_config, supported_opponent_deck

        _identity(self.member_id, "member_id")
        if self.split not in ("train", "dev"):
            raise ValueError("learned opponent split must be train or dev")
        if self.inference not in ("deterministic", "stochastic"):
            raise ValueError("learned opponent inference must be deterministic or stochastic")
        _seed(self.sampling_seed, "sampling_seed")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        if (
            not isinstance(self.sha256, str) or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("sha256 must be a lowercase 64-character hex digest")
        if not isinstance(self.checkpoint, (str, Path)) or not str(self.checkpoint):
            raise ValueError("checkpoint must be a nonempty file path")
        path = Path(self.checkpoint).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("checkpoint must be a file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError(f"checkpoint hash mismatch for population member {self.member_id}")
        object.__setattr__(self, "checkpoint", str(path))
        object.__setattr__(self, "deck", supported_opponent_deck(self.deck))
        object.__setattr__(self, "model_config", exact_model_config(self.model_config))

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id, "checkpoint": self.checkpoint,
            "sha256": self.sha256, "model_config": asdict(self.model_config),
            "split": self.split, "deck": list(self.deck), "inference": self.inference,
            "sampling_seed": self.sampling_seed, "generation": self.generation,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LearnedOpponentMember:
        expected = {
            "member_id", "checkpoint", "sha256", "model_config", "split", "deck",
            "inference", "sampling_seed", "generation",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("population member must contain every version-1 field exactly")
        return cls(**dict(value))

    def to_opponent_spec(self, *, episode_seed: int | None = None) -> dict[str, Any]:
        """Factory-compatible spec; the caller may derive a fresh episode stream."""

        if episode_seed is not None:
            _seed(episode_seed, "episode_seed")
        seed = self.sampling_seed if episode_seed is None else _stable_seed(
            "learned-opponent", self.member_id, self.sampling_seed, episode_seed
        )
        return {
            "kind": "v4-checkpoint", "archetype": f"v4-policy-{self.member_id}",
            "member_id": self.member_id, "population_split": self.split,
            "checkpoint": self.checkpoint, "sha256": self.sha256,
            "model_config": asdict(self.model_config), "deck": list(self.deck),
            "inference": self.inference, "sampling_seed": seed,
            "generation": self.generation,
        }


@dataclass(frozen=True, slots=True)
class TrainMatchupResult:
    """A completed train match, with score from the learner's perspective.

    Keeping the split and completion flags mandatory prevents development or
    censored outcomes from silently driving opponent selection.
    """

    match_id: str
    member_id: str
    learner_score: float
    split: str
    completed: bool
    truncated: bool

    def __post_init__(self) -> None:
        _identity(self.match_id, "match_id")
        _identity(self.member_id, "member_id")
        if self.split != "train":
            raise ValueError("population prioritization accepts train results only")
        if self.completed is not True or self.truncated is not False:
            raise ValueError("population prioritization needs complete nontruncated matches")
        if (
            isinstance(self.learner_score, bool)
            or not isinstance(self.learner_score, (int, float))
            or self.learner_score not in (0.0, 0.5, 1.0)
        ):
            raise ValueError("learner_score must be a terminal win/draw/loss score (1, .5, 0)")


@dataclass(frozen=True, slots=True)
class OpponentPopulation:
    """Append-only snapshot collection with a uniform coverage floor."""

    population_id: str
    members: tuple[LearnedOpponentMember, ...]
    schema_version: int = OPPONENT_POPULATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identity(self.population_id, "population_id")
        if type(self.schema_version) is not int or self.schema_version != OPPONENT_POPULATION_SCHEMA_VERSION:
            raise ValueError("unsupported opponent population schema_version")
        members = tuple(self.members)
        if not members or any(not isinstance(member, LearnedOpponentMember) for member in members):
            raise ValueError("population needs at least one typed learned opponent member")
        if len({member.member_id for member in members}) != len(members):
            raise ValueError("population member IDs must be unique")
        splits_by_checkpoint: dict[str, str] = {}
        for member in members:
            previous_split = splits_by_checkpoint.setdefault(member.sha256, member.split)
            if previous_split != member.split:
                raise ValueError("the same checkpoint cannot appear in both train and dev populations")
        object.__setattr__(self, "members", members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "population_id": self.population_id,
            "members": [member.to_dict() for member in self.members],
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OpponentPopulation:
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "population_id", "members"}:
            raise ValueError("population manifest must contain every version-1 field exactly")
        raw_members = value["members"]
        if not isinstance(raw_members, (list, tuple)):
            raise ValueError("population manifest members must be a sequence")
        return cls(
            population_id=value["population_id"], schema_version=value["schema_version"],
            members=tuple(LearnedOpponentMember.from_dict(member) for member in raw_members),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> OpponentPopulation:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_member(self, member: LearnedOpponentMember) -> OpponentPopulation:
        """Add a snapshot; existing identities cannot be replaced or evicted."""

        return replace(self, members=(*self.members, member))

    def _eligible(self, split: str) -> tuple[LearnedOpponentMember, ...]:
        if split not in ("train", "dev"):
            raise ValueError("population split must be train or dev")
        members = tuple(sorted(
            (member for member in self.members if member.split == split),
            key=lambda member: member.member_id,
        ))
        if not members:
            raise ValueError(f"population contains no {split} members")
        return members

    def sampling_weights(
        self, *, split: str = "train", train_results: Sequence[TrainMatchupResult] = (),
        coverage_fraction: float = 0.5,
    ) -> dict[str, float]:
        """Prioritize difficult TRAIN matchups while preserving every snapshot.

        A beta(1, 1) smoothed loss estimate provides the optional priority.
        Development scheduling is always uniform and never accepts result
        feedback. Each eligible member retains at least coverage_fraction/N.
        """

        members = self._eligible(split)
        if (
            isinstance(coverage_fraction, bool)
            or not isinstance(coverage_fraction, (int, float))
            or not math.isfinite(coverage_fraction) or not 0 < coverage_fraction <= 1
        ):
            raise ValueError("coverage_fraction must be finite and in (0, 1]")
        if split != "train" and train_results:
            raise ValueError("development opponent scheduling cannot use matchup result feedback")
        totals = {member.member_id: [0.0, 0] for member in members}
        seen: set[str] = set()
        for result in train_results:
            if not isinstance(result, TrainMatchupResult):
                raise TypeError("train_results must contain TrainMatchupResult values")
            if result.member_id not in totals:
                raise ValueError("matchup result references a non-training or unknown population member")
            if result.match_id in seen:
                raise ValueError("duplicate training matchup result")
            seen.add(result.match_id)
            totals[result.member_id][0] += 1.0 - result.learner_score
            totals[result.member_id][1] += 1
        difficulty = {
            member_id: (1.0 + losses) / (2.0 + count)
            for member_id, (losses, count) in totals.items()
        }
        total_difficulty = sum(difficulty.values())
        return {
            member.member_id: coverage_fraction / len(members)
            + (1.0 - coverage_fraction) * difficulty[member.member_id] / total_difficulty
            for member in members
        }

    def sample(
        self, *, seed: int, episode_index: int, split: str = "train",
        train_results: Sequence[TrainMatchupResult] = (), coverage_fraction: float = 0.5,
    ) -> LearnedOpponentMember:
        """Stateless reproducible draw, independent of process-global RNG."""

        _seed(seed, "seed")
        if type(episode_index) is not int or episode_index < 0:
            raise ValueError("episode_index must be a nonnegative integer")
        weights = self.sampling_weights(
            split=split, train_results=train_results, coverage_fraction=coverage_fraction
        )
        members = self._eligible(split)
        rng = random.Random(_stable_seed(self.population_id, seed, episode_index, split))
        return rng.choices(members, weights=[weights[member.member_id] for member in members], k=1)[0]

    def schedule(
        self, *, episodes: int, seed: int, split: str = "train",
        train_results: Sequence[TrainMatchupResult] = (), coverage_fraction: float = 0.5,
    ) -> tuple[LearnedOpponentMember, ...]:
        """Visit each member once before weighted extras in a finite workload."""

        _seed(seed, "seed")
        members = list(self._eligible(split))
        if type(episodes) is not int or episodes < len(members):
            raise ValueError("episodes must cover every member of the requested split")
        self.sampling_weights(
            split=split, train_results=train_results, coverage_fraction=coverage_fraction
        )
        rng = random.Random(_stable_seed(self.population_id, seed, "coverage", split))
        rng.shuffle(members)
        members.extend(
            self.sample(
                seed=seed, episode_index=index, split=split, train_results=train_results,
                coverage_fraction=coverage_fraction,
            )
            for index in range(len(members), episodes)
        )
        return tuple(members)


def make_population_match_start_entries(
    population: OpponentPopulation, *, pool_seed: int, episodes: int,
    train_results: Sequence[TrainMatchupResult] = (), coverage_fraction: float = 0.5,
) -> list[dict[str, Any]]:
    """Make genuine train-only match openings usable by the existing PPO lane.

    No teacher labels, frontier verdicts, rollout rewards or optimization are
    generated. Checkpoint compatibility is checked before returning a workload.
    Every training member is visited once, with optional weighted extra visits
    based exclusively on complete training outcomes.
    """

    from simulator.engine.core import BattleEngine
    from simulator.env import SimulatorEnv
    from simulator.ruleset import load_fixed_ruleset

    from .policy_opponent import _configured_actor
    from .raw_ppo import RAW_PPO_STARTS_VERSION

    if not isinstance(population, OpponentPopulation):
        raise TypeError("population must be an OpponentPopulation")
    _seed(pool_seed, "pool_seed")
    schedule = population.schedule(
        episodes=episodes, seed=pool_seed, split="train", train_results=train_results,
        coverage_fraction=coverage_fraction,
    )
    for member in population._eligible("train"):
        _configured_actor(member.checkpoint, member.sha256, member.model_config)
    weights = population.sampling_weights(
        split="train", train_results=train_results, coverage_fraction=coverage_fraction,
    )
    feedback_fingerprint = _fingerprint([
        asdict(result) for result in sorted(train_results, key=lambda result: result.match_id)
    ])
    population_fingerprint = population.fingerprint
    ruleset = load_fixed_ruleset()
    entries: list[dict[str, Any]] = []
    for index, member in enumerate(schedule):
        match_seed = _stable_seed(
            "learned-population-opening", population_fingerprint, pool_seed, index, member.member_id,
        )
        env = SimulatorEnv(BattleEngine(ruleset, validate_every_tick=False))
        env.reset_v2(seed=match_seed, decks=(tuple(PLAYER_DECK), member.deck), shuffle_decks=True)
        if env.state is None or env.state.tick != 0:
            raise RuntimeError("learned population reset did not produce a tick-zero battle")
        opponent = member.to_opponent_spec(episode_seed=match_seed)
        opponent.update({
            "population_id": population.population_id,
            "population_schema_version": population.schema_version,
        })
        state_hash = env.state.state_hash()
        match_hash = _fingerprint({
            "population_sha256": population_fingerprint, "opponent": opponent,
            "learner_deck": list(PLAYER_DECK), "match_seed": match_seed,
            "state_hash": state_hash,
        })
        entries.append({
            "entry_id": f"raw:{population.population_id}:{member.member_id}:{index}",
            "split": "train", "initial_state": env.save_state(),
            "learner_deck": list(PLAYER_DECK), "opponent_deck": list(member.deck),
            "opponent": opponent,
            "provenance": {
                "generator": RAW_PPO_STARTS_VERSION, "profile": "learned-population",
                "population_id": population.population_id,
                "population_schema_version": population.schema_version,
                "population_sha256": population_fingerprint,
                "population_member_id": member.member_id, "population_split": "train",
                "pool_seed": pool_seed, "match_index": index, "match_seed": match_seed,
                "state_hash": state_hash, "match_hash": match_hash,
                "training_feedback_sha256": feedback_fingerprint,
                "training_feedback_count": len(train_results),
                "coverage_fraction": coverage_fraction, "sampling_weights": dict(weights),
            },
        })
    return entries
