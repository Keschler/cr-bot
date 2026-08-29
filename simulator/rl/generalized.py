"""Generalized training and held-out evaluation orchestration.

The recurrent prototype originally trained on one fixed opponent deck and one
controller.  This module turns the simulator-side opponent pool and the
evaluation matrix into an executable workflow:

* every rollout segment receives a reproducible mix of deck archetypes and
  opponent controllers;
* one deterministic-cycle Hog deck is kept in every segment as a regression
  cell;
* checkpoints are resumed between segments, so optimizer state and recurrent
  policy weights continue across the curriculum;
* evaluation uses a different, held-out deck sample and reports every
  deck/strategy/seed cell separately.

The mainline path is actor-controlled PPO.  Heuristic controllers and frozen
checkpoints are opponent sources or explicitly opted-in teacher labels; they
do not choose the learner's strategic actions.

The learner still receives only the public V2 observation.  The opponent
callback is simulator-side and may inspect authoritative state only to choose
the opponent's own action.

Example::

    PYTHONPATH=..:../src outputs/venv/bin/python -m rl.generalized train \
      --allow-provisional --updates 20 --envs 4 --horizon 512 \
      --device cuda --checkpoint-out outputs/simulator/training/generalized.pt

Then evaluate the neural actor on the held-out matrix::

    PYTHONPATH=..:../src outputs/venv/bin/python -m rl.generalized evaluate \
      --checkpoint outputs/simulator/training/generalized.pt \
      --policy actor --seeds 10000,10001
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

from .evaluation_matrix import (
    EvaluationMatrixConfig,
    MatchResult,
    MatchSpec,
    OpponentDeckSpec as MatrixDeckSpec,
    OpponentStrategySpec,
    _file_fingerprint,
    compare_evaluation_reports,
    run_evaluation_matrix,
)
from .curriculum import (
    StrategicCurriculum,
    StrategicCurriculumStage,
    default_strategic_curriculum,
)
from .league import LeaguePayoffBook, PFSPOpponentSampler
from .opponent_pool import (
    ARCHETYPE_NAMES,
    OpponentPool,
    OpponentPoolError,
    OpponentScenario,
    make_opponent_controller,
)
from .prototype import PrototypeConfig, train_prototype


GENERALIZED_TRAINING_SCHEMA_VERSION = 1
GENERALIZED_TRAINING_KIND = "recurrent_public_ppo_generalized_training"
HELD_OUT_DECK_INDEX = 100_000
_SUPPORTED_STRATEGIES = frozenset(
    {
        "deterministic-cycle",
        "aggressive-pressure",
        "defensive-cycle",
        "beatdown",
        "siege-bait",
        "random-legal",
    }
)


class GeneralizedTrainingError(ValueError):
    """Raised when generalized orchestration cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class _TrainingReportAudit:
    """Parsed training-report exclusions and their provenance status.

    Older generalized reports contain enough schedule data to avoid reusing
    their decks, but not enough metadata to prove that they belong to the
    checkpoint being evaluated.  ``provenance_verified`` keeps that useful
    compatibility behavior while ensuring those reports cannot certify a
    held-out split.
    """

    deck_keys: frozenset[frozenset[str]]
    provenance_verified: bool


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise GeneralizedTrainingError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise GeneralizedTrainingError(f"{name} must be a non-negative integer")


def _names(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise GeneralizedTrainingError(f"{name} must be a sequence of names")
    result = tuple(str(value).strip().casefold().replace("_", "-") for value in values)
    if not result or any(not value for value in result):
        raise GeneralizedTrainingError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise GeneralizedTrainingError(f"{name} must not contain duplicates")
    return result


def _default_player_deck() -> tuple[str, ...]:
    try:
        from ..roster import PLAYER_DECK
    except ImportError:  # pragma: no cover - top-level ``rl`` layout
        from simulator.roster import PLAYER_DECK
    return tuple(PLAYER_DECK)


def _normalize_player_deck(value: object) -> tuple[str, ...]:
    if value is None:
        return _default_player_deck()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GeneralizedTrainingError(
            "player_deck must be a sequence of eight card identifiers"
        )
    cards = _names("player_deck", value)
    if len(cards) != 8:
        raise GeneralizedTrainingError("player_deck must contain exactly eight cards")
    return cards


def _mix_seed(seed: int, *parts: int) -> int:
    """Stable integer mixing for segment and lane scenario seeds."""

    value = seed & ((1 << 64) - 1)
    for part in parts:
        value ^= (int(part) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = (value * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    return value


def _default_prototype_config() -> PrototypeConfig:
    return PrototypeConfig(
        envs=4,
        horizon=512,
        updates=1,
        learning_rate=3e-4,
        update_epochs=3,
        sequence_minibatch_size=8,
        sequence_length=128,
        gamma=0.9995,
        gae_lambda=0.98,
        entropy_coef=0.01,
        dense_reward=False,
        potential_reward_weight=0.1,
        model_dim=128,
        encoder_dim=128,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff_dim=256,
        gru_hidden_dim=256,
        explicit_hand_features=True,
        spatial_placement_features=True,
    )


@dataclass(frozen=True, slots=True)
class GeneralizedTrainingConfig:
    """Settings for segmented opponent-diverse recurrent-PPO training.

    ``prototype_config`` carries the model and optimizer settings.  Its
    ``updates`` field is overridden by ``rollouts_per_scenario`` for each
    sampled scenario; ``segments`` is the generalized scenario budget.
    Recreating the environments at scenario boundaries is deliberate for this
    local-first runner, while keeping several rollouts on one scenario exposes
    the recurrent actor to opening, mid-match, and late-match states before
    the deck/controller is changed. A future persistent-lane runner can use
    the same schedule and report contract without changing the actor boundary.
    """

    prototype_config: PrototypeConfig = field(default_factory=_default_prototype_config)
    segments: int = 20
    rollouts_per_scenario: int = 1
    # The schedule index is separate from the number of segments in this
    # invocation.  A sidecar generalized report can fill it automatically on
    # resume; callers may set it explicitly when the sidecar is unavailable.
    segment_offset: int | None = None
    checkpoint: str | Path | None = None
    checkpoint_out: str | Path = "outputs/simulator/training/generalized-recurrent-prototype.pt"
    include_regression: bool = True
    threat_stratified: bool = False
    # The stage changes only the opponent distribution. It never supplies a
    # learner action or a strategic action mask.
    curriculum: StrategicCurriculum = field(default_factory=default_strategic_curriculum)
    use_curriculum: bool = True
    # Potential shaping is a temporary credit-assignment aid.  It reaches
    # zero at this global segment, after which the terminal objective is the
    # only reward. Zero disables annealing and preserves the configured
    # potential coefficient for the complete run.
    potential_reward_anneal_segments: int = 32
    expert_guidance: bool = False
    expert_teacher: str = "public-counter"
    train_archetypes: tuple[str, ...] = (
        "aggressive-pressure",
        "defensive-cycle",
        "beatdown",
        "air-beatdown",
        "siege-bait",
        "random-legal",
    )
    train_strategies: tuple[str, ...] = (
        "aggressive-pressure",
        "defensive-cycle",
        "beatdown",
        "siege-bait",
        "random-legal",
    )
    opponent_checkpoints: tuple[str | Path, ...] = ()
    # PFSP is deliberately opt-in. It selects among already frozen public
    # checkpoints; it does not turn simulator-side heuristics into learner
    # labels or silently change the actor's observation contract.
    pfsp: bool = False
    pfsp_payoff_book: str | Path | None = None
    pfsp_payoff_book_out: str | Path | None = None
    league_agent_id: str = "main"
    player_deck: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prototype_config, PrototypeConfig):
            raise GeneralizedTrainingError("prototype_config must be a PrototypeConfig")
        object.__setattr__(self, "player_deck", _normalize_player_deck(self.player_deck))
        _positive_int("segments", self.segments)
        _positive_int("rollouts_per_scenario", self.rollouts_per_scenario)
        if self.segment_offset is not None:
            _nonnegative_int("segment_offset", self.segment_offset)
        if self.checkpoint is not None and (
            not isinstance(self.checkpoint, (str, Path)) or not str(self.checkpoint).strip()
        ):
            raise GeneralizedTrainingError("checkpoint must be a non-empty path or None")
        if not isinstance(self.checkpoint_out, (str, Path)) or not str(self.checkpoint_out).strip():
            raise GeneralizedTrainingError("checkpoint_out must be a non-empty path")
        if type(self.include_regression) is not bool:
            raise GeneralizedTrainingError("include_regression must be boolean")
        if type(self.threat_stratified) is not bool:
            raise GeneralizedTrainingError("threat_stratified must be boolean")
        if not isinstance(self.curriculum, StrategicCurriculum):
            raise GeneralizedTrainingError(
                "curriculum must be a StrategicCurriculum"
            )
        if type(self.use_curriculum) is not bool:
            raise GeneralizedTrainingError("use_curriculum must be boolean")
        _nonnegative_int(
            "potential_reward_anneal_segments",
            self.potential_reward_anneal_segments,
        )
        if type(self.expert_guidance) is not bool:
            raise GeneralizedTrainingError("expert_guidance must be boolean")
        if type(self.pfsp) is not bool:
            raise GeneralizedTrainingError("pfsp must be boolean")
        if self.pfsp_payoff_book is not None and (
            not isinstance(self.pfsp_payoff_book, (str, Path))
            or not str(self.pfsp_payoff_book).strip()
        ):
            raise GeneralizedTrainingError(
                "pfsp_payoff_book must be a non-empty path or None"
            )
        if self.pfsp_payoff_book_out is not None and (
            not isinstance(self.pfsp_payoff_book_out, (str, Path))
            or not str(self.pfsp_payoff_book_out).strip()
        ):
            raise GeneralizedTrainingError(
                "pfsp_payoff_book_out must be a non-empty path or None"
            )
        if not isinstance(self.league_agent_id, str) or not self.league_agent_id.strip():
            raise GeneralizedTrainingError("league_agent_id must be a non-empty string")
        if self.expert_teacher not in {
            "public-counter",
            "strategic-counter",
            "deterministic-counter",
        }:
            raise GeneralizedTrainingError(
                "expert_teacher must be 'public-counter', 'strategic-counter', "
                "or 'deterministic-counter'"
            )
        if self.expert_guidance and not (
            self.prototype_config.behavior_cloning_coef > 0.0
            or self.prototype_config.behavior_cloning_factor_coef > 0.0
        ):
            raise GeneralizedTrainingError(
                "expert_guidance requires a positive joint or factor behavior-cloning coefficient"
            )
        _names("train_archetypes", self.train_archetypes)
        _names("train_strategies", self.train_strategies)
        if isinstance(self.opponent_checkpoints, (str, bytes)):
            raise GeneralizedTrainingError(
                "opponent_checkpoints must be a sequence of paths"
            )
        for index, checkpoint in enumerate(self.opponent_checkpoints):
            if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
                raise GeneralizedTrainingError(
                    f"opponent_checkpoints[{index}] must be a non-empty path"
                )
        if self.pfsp and not self.opponent_checkpoints:
            raise GeneralizedTrainingError(
                "pfsp requires at least one frozen opponent checkpoint"
            )
        if self.pfsp and len({str(path) for path in self.opponent_checkpoints}) != len(
            self.opponent_checkpoints
        ):
            raise GeneralizedTrainingError(
                "pfsp opponent checkpoints must have unique paths"
            )
        if not self.pfsp and (
            self.pfsp_payoff_book is not None or self.pfsp_payoff_book_out is not None
        ):
            raise GeneralizedTrainingError(
                "PFSP payoff-book paths require pfsp=True"
            )
        unknown_archetypes = set(self.train_archetypes) - set(ARCHETYPE_NAMES)
        if unknown_archetypes:
            raise GeneralizedTrainingError(
                f"unknown training archetypes: {sorted(unknown_archetypes)}"
            )


        unknown_strategies = set(self.train_strategies) - _SUPPORTED_STRATEGIES
        if unknown_strategies:
            raise GeneralizedTrainingError(
                f"unknown training strategies: {sorted(unknown_strategies)}"
            )
        if self.include_regression and self.prototype_config.envs < 1:
            raise GeneralizedTrainingError("at least one environment is required")
        if self.prototype_config.target_player != 0:
            raise GeneralizedTrainingError(
                "generalized training currently requires target_player=0 because "
                "the configurable player deck is assigned to player 0"
            )


def _curriculum_axes(
    config: GeneralizedTrainingConfig,
    segment_index: int,
) -> tuple[tuple[str, ...], tuple[str, ...], StrategicCurriculumStage | None]:
    """Resolve the stage's opponent axes without constraining learner actions."""

    if not config.use_curriculum:
        return tuple(config.train_archetypes), tuple(config.train_strategies), None
    stage = config.curriculum.stage_at(segment_index)
    # Explicit caller lists remain an upper bound. This keeps a small local
    # smoke run reproducible while the default schedule broadens over time.
    archetypes = tuple(
        name for name in config.train_archetypes if name in stage.archetypes
    )
    strategies = tuple(
        name for name in config.train_strategies if name in stage.strategies
    )
    # A custom narrow list may intentionally contain only a late-stage item.
    # In that case do not make the run empty merely because it is in an early
    # stage; the caller's explicit distribution wins.
    return (
        archetypes or tuple(config.train_archetypes),
        strategies or tuple(config.train_strategies),
        stage,
    )


def sample_training_scenarios(
    pool: OpponentPool,
    *,
    envs: int,
    segment_index: int,
    archetypes: Sequence[str],
    strategies: Sequence[str],
    include_regression: bool = True,
    threat_stratified: bool = False,
) -> tuple[OpponentScenario, ...]:
    """Sample one scenario per rollout lane in deterministic order.

    Lane zero is reserved for the pinned deterministic-cycle regression when
    requested.  The remaining lanes cycle through the configured archetypes
    and strategies, so a segment has both a stable anchor and varied opponents.
    When ``threat_stratified`` is true, the first available non-regression
    lanes are reserved for ``air-beatdown`` and ``beatdown`` before the
    remaining archetypes are sampled.  This makes air and ground defensive
    sequences present in every segment instead of relying on a long-period
    modulo schedule.
    ``segment_index`` is part of the sampler index; restarting with the same
    seed and configuration recreates the exact schedule.
    """

    _positive_int("envs", envs)
    _nonnegative_int("segment_index", segment_index)
    if type(threat_stratified) is not bool:
        raise GeneralizedTrainingError("threat_stratified must be boolean")
    archetype_names = _names("archetypes", archetypes)
    strategy_names = _names("strategies", strategies)
    scenarios: list[OpponentScenario] = []
    seen_decks: set[frozenset[str]] = set()
    base_index = segment_index * envs
    lane_start = 0
    if include_regression:
        regression = pool.sample(
            base_index,
            archetype="deterministic-cycle",
            strategy="deterministic-cycle",
        )
        scenarios.append(regression)
        seen_decks.add(frozenset(regression.deck.cards))
        lane_start = 1

    def sample_unique(
        episode_index: int,
        *,
        archetype: str,
        strategy: str,
    ) -> OpponentScenario:
        # Curated archetypes can have several templates, while random-legal
        # has a much larger stream.  Search the deterministic candidate
        # stream first so a normal-sized rollout receives distinct decks.
        # A large rollout can legitimately request more variants than a
        # curated archetype contains; retain a reproducible duplicate rather
        # than rejecting the entire training run.  The callback below assigns
        # duplicate compositions to their own scenario/controller state.
        fallback: OpponentScenario | None = None
        for attempt in range(256):
            candidate = pool.sample(
                episode_index + attempt * max(1, envs),
                archetype=archetype,
                strategy=strategy,
            )
            fallback = candidate
            deck_key = frozenset(candidate.deck.cards)
            if deck_key not in seen_decks:
                seen_decks.add(deck_key)
                return candidate
        if fallback is None:  # pragma: no cover - pool.sample always returns or raises
            raise GeneralizedTrainingError(
                "could not sample an opponent deck; "
                f"archetype={archetype!r}, episode_index={episode_index}"
            )
        return fallback

    threat_names = tuple(
        name
        for name in ("air-beatdown", "beatdown")
        if name in archetype_names
    )
    remaining_names = tuple(
        name for name in archetype_names if name not in threat_names
    )
    non_regression_lanes = max(1, envs - lane_start)
    for lane in range(lane_start, envs):
        lane_ordinal = lane - lane_start
        ordinal = segment_index * non_regression_lanes + lane_ordinal
        if threat_stratified and threat_names and lane_ordinal < len(threat_names):
            archetype = threat_names[lane_ordinal]
        elif threat_stratified and threat_names and not remaining_names:
            archetype = threat_names[ordinal % len(threat_names)]
        elif threat_stratified and remaining_names:
            archetype = remaining_names[
                (segment_index + lane_ordinal - len(threat_names))
                % len(remaining_names)
            ]
        else:
            archetype = archetype_names[ordinal % len(archetype_names)]
        strategy = strategy_names[(ordinal + segment_index) % len(strategy_names)]
        scenarios.append(
            sample_unique(
                base_index + lane,
                archetype=archetype,
                strategy=strategy,
            )
        )
    return tuple(scenarios)


def _scenario_checkpoint_assignments(
    scenarios: Sequence[OpponentScenario],
    checkpoint_opponents: Sequence[str | Path],
    *,
    pfsp_sampler: PFSPOpponentSampler | None = None,
    learner_agent_id: str = "main",
    match_index_base: int = 0,
) -> tuple[str | Path | None, ...]:
    """Assign frozen opponents exactly as the runtime callback does.

    The default is the historical round-robin assignment.  When a PFSP
    sampler is supplied, the candidate IDs are the checkpoint path strings and
    the selected path is still recorded verbatim in the report.  PFSP only
    changes which frozen actor supplies the opponent action; it never changes
    the learner's legal action set.
    """

    paths = tuple(checkpoint_opponents)
    assignments: list[str | Path | None] = []
    checkpoint_index = 0
    if type(match_index_base) is not int or match_index_base < 0:
        raise GeneralizedTrainingError("match_index_base must be non-negative")
    path_by_id = {str(path): path for path in paths}
    for index, scenario in enumerate(scenarios):
        is_regression = (
            index == 0
            and scenario.deck.archetype == "deterministic-cycle"
            and scenario.strategy == "deterministic-cycle"
        )
        if paths and not is_regression:
            if pfsp_sampler is None:
                selected = paths[checkpoint_index % len(paths)]
            else:
                selected_id = pfsp_sampler.sample(
                    learner_agent_id,
                    match_index_base + index,
                    tuple(path_by_id),
                )
                selected = path_by_id[selected_id]
            assignments.append(selected)
            checkpoint_index += 1
        else:
            assignments.append(None)
    return tuple(assignments)


def _scenario_assignment_rows(
    scenarios: Sequence[OpponentScenario],
    checkpoint_opponents: Sequence[str | Path],
    *,
    checkpoint_assignments: Sequence[str | Path | None] | None = None,
    pfsp_sampler: PFSPOpponentSampler | None = None,
    learner_agent_id: str = "main",
    match_index_base: int = 0,
) -> list[dict[str, object]]:
    """Serialize lane-to-opponent provenance for a generalized report."""

    assignments = (
        tuple(checkpoint_assignments)
        if checkpoint_assignments is not None
        else _scenario_checkpoint_assignments(
            scenarios,
            checkpoint_opponents,
            pfsp_sampler=pfsp_sampler,
            learner_agent_id=learner_agent_id,
            match_index_base=match_index_base,
        )
    )
    if len(assignments) != len(scenarios):
        raise GeneralizedTrainingError(
            "checkpoint_assignments must contain one entry per scenario"
        )
    rows: list[dict[str, object]] = []
    for lane, (scenario, checkpoint) in enumerate(zip(scenarios, assignments, strict=True)):
        row: dict[str, object] = {
            "lane": lane,
            "episode_index": scenario.episode_index,
            "selection_seed": scenario.selection_seed,
            "controller_seed": scenario.controller_seed,
            "deck_id": scenario.deck.deck_id,
            "deck_cards": list(scenario.deck.cards),
            "archetype": scenario.deck.archetype,
            "strategy": scenario.strategy,
            "source": "frozen-checkpoint" if checkpoint is not None else "simulator-controller",
            "checkpoint": None,
            "checkpoint_fingerprint": None,
        }
        if checkpoint is not None:
            row["checkpoint"] = str(checkpoint)
            row["checkpoint_fingerprint"] = _file_fingerprint(checkpoint)
        rows.append(row)
    return rows


def make_scenario_opponent_action(
    scenarios: Sequence[OpponentScenario],
    *,
    checkpoint_opponents: Sequence[str | Path] = (),
    checkpoint_assignments: Sequence[str | Path | None] | None = None,
    device: str | None = "auto",
) -> Callable[[Any, Any, int], Any]:
    """Build a callback for heuristic and optional frozen self-play lanes.

    When ``checkpoint_opponents`` is non-empty, each non-regression lane is
    assigned a frozen public actor in stable round-robin order. Heuristic lanes
    remain simulator-side and may inspect authoritative state; checkpoint
    lanes consume only the opponent viewer's public observation.
    """

    if not scenarios:
        raise GeneralizedTrainingError("scenarios must not be empty")
    by_deck: dict[tuple[str, ...], list[OpponentScenario]] = {}
    for scenario in scenarios:
        by_deck.setdefault(tuple(scenario.deck.cards), []).append(scenario)
    controllers: dict[int, Any] = {}
    assigned_scenarios: dict[int, OpponentScenario] = {}
    deck_assignment_counts: dict[tuple[str, ...], int] = {}
    checkpoint_paths = tuple(checkpoint_opponents)
    if checkpoint_assignments is None:
        checkpoint_assignments = _scenario_checkpoint_assignments(
            scenarios,
            checkpoint_paths,
        )
    else:
        checkpoint_assignments = tuple(checkpoint_assignments)
        if len(checkpoint_assignments) != len(scenarios):
            raise GeneralizedTrainingError(
                "checkpoint_assignments must contain one entry per scenario"
            )
    checkpoint_by_scenario: dict[int, str | Path] = {}
    for scenario, checkpoint in zip(
        scenarios,
        checkpoint_assignments,
        strict=True,
    ):
        if checkpoint is not None:
            checkpoint_by_scenario[id(scenario)] = checkpoint

    def choose(environment: Any, public_observation: Any, player: int) -> Any:
        state = getattr(environment, "state", None)
        if state is None:
            raise GeneralizedTrainingError("opponent callback received an uninitialized environment")
        players = getattr(state, "players", ())
        if player not in (0, 1) or len(players) != 2:
            raise GeneralizedTrainingError("opponent callback received an invalid player")
        deck = tuple(players[player].deck)
        key = id(environment)
        scenario = assigned_scenarios.get(key)
        if scenario is None:
            candidates = by_deck.get(deck)
            if not candidates:
                raise GeneralizedTrainingError(
                    f"environment opponent deck {deck!r} is not in the sampled scenario set"
                )
            candidate_index = deck_assignment_counts.get(deck, 0)
            scenario = candidates[min(candidate_index, len(candidates) - 1)]
            deck_assignment_counts[deck] = candidate_index + 1
            assigned_scenarios[key] = scenario
        elif tuple(scenario.deck.cards) != deck:
            raise GeneralizedTrainingError(
                f"environment opponent deck {deck!r} is not in the sampled scenario set"
            )
        checkpoint_path = checkpoint_by_scenario.get(id(scenario))
        if checkpoint_path is not None:
            controller = controllers.get(key)
            if controller is None:
                from .self_play import PublicCheckpointController

                controller = PublicCheckpointController(checkpoint_path, device=device)
                controllers[key] = controller
            return controller.choose_public_action(public_observation, player=player)
        controller = controllers.get(key)
        if controller is None:
            controller = scenario.build_controller()
            controllers[key] = controller
        return controller.choose_action(environment.engine, state, player)

    return choose


def _segment_config(
    base: PrototypeConfig,
    *,
    segment_index: int,
    rollouts_per_scenario: int,
    potential_reward_anneal_segments: int = 0,
) -> PrototypeConfig:
    _positive_int("rollouts_per_scenario", rollouts_per_scenario)
    _nonnegative_int(
        "potential_reward_anneal_segments",
        potential_reward_anneal_segments,
    )
    potential_weight = float(base.potential_reward_weight)
    if potential_reward_anneal_segments > 0:
        remaining = max(
            0.0,
            1.0 - float(segment_index) / float(potential_reward_anneal_segments),
        )
        potential_weight *= remaining
    return replace(
        base,
        updates=rollouts_per_scenario,
        seed=_mix_seed(base.seed, segment_index, 0x47524F57),
        potential_reward_weight=potential_weight,
    )


def _update_payoff_book_from_segment(
    payoff_book: LeaguePayoffBook,
    report: Mapping[str, object],
    checkpoint_assignments: Sequence[str | Path | None],
    *,
    learner_agent_id: str,
) -> tuple[LeaguePayoffBook, int]:
    """Record completed frozen-opponent matches emitted by a PPO segment.

    Rollout segments can contain several completed matches per lane.  The
    collector therefore emits lane-indexed terminal outcomes rather than
    guessing a single result from aggregate counters.  Older/fake reports may
    omit this optional field and simply produce zero updates.
    """

    updated = payoff_book
    recorded = 0
    raw_updates = report.get("update_rows", ())
    if raw_updates is None:
        return updated, recorded
    if isinstance(raw_updates, (str, bytes)) or not isinstance(raw_updates, Sequence):
        raise GeneralizedTrainingError("segment report update_rows must be a sequence")
    for update in raw_updates:
        if not isinstance(update, Mapping):
            raise GeneralizedTrainingError("segment report update row must be an object")
        raw_rollout = update.get("rollout", {})
        if not isinstance(raw_rollout, Mapping):
            raise GeneralizedTrainingError("segment report rollout must be an object")
        raw_outcomes = raw_rollout.get("match_outcomes", ())
        if raw_outcomes is None:
            continue
        if isinstance(raw_outcomes, (str, bytes)) or not isinstance(raw_outcomes, Sequence):
            raise GeneralizedTrainingError("match_outcomes must be a sequence")
        for raw_outcome in raw_outcomes:
            if not isinstance(raw_outcome, Mapping):
                raise GeneralizedTrainingError("match outcome must be an object")
            lane = raw_outcome.get("lane")
            outcome = raw_outcome.get("outcome")
            if type(lane) is not int or not 0 <= lane < len(checkpoint_assignments):
                raise GeneralizedTrainingError("match outcome lane is outside the segment")
            if outcome not in {"win", "draw", "loss"}:
                raise GeneralizedTrainingError("match outcome must be win, draw, or loss")
            opponent = checkpoint_assignments[lane]
            if opponent is None:
                # Simulator-controller regression lanes are not league agents.
                continue
            try:
                updated = updated.after_match(
                    learner_agent_id,
                    str(opponent),
                    outcome,
                )
            except ValueError as error:
                raise GeneralizedTrainingError(
                    f"cannot record PFSP outcome against {opponent!r}: {error}"
                ) from error
            recorded += 1
    return updated, recorded


def _resolve_segment_offset(config: GeneralizedTrainingConfig) -> tuple[int, str]:
    """Resolve the global scenario cursor for a fresh or resumed run.

    Generalized reports are normally written beside their checkpoint.  When
    that sidecar is present, its last schedule index is the only safe source
    of the next cursor.  An explicit ``segment_offset`` always wins and is
    useful when the report was moved or intentionally branched.
    """

    if config.segment_offset is not None:
        return config.segment_offset, "explicit"
    if config.checkpoint is None:
        return 0, "fresh-run"
    sidecar = Path(config.checkpoint).with_suffix(".json")
    if not sidecar.exists():
        return 0, "default-no-sidecar"
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GeneralizedTrainingError(
            f"cannot read generalized checkpoint sidecar {sidecar}: {error}"
        ) from error
    if not isinstance(raw, Mapping) or raw.get("kind") != GENERALIZED_TRAINING_KIND:
        return 0, "default-non-generalized-sidecar"
    indices = raw.get("segment_indices")
    if isinstance(indices, list) and indices and all(type(value) is int for value in indices):
        return max(indices) + 1, "generalized-sidecar"
    segments = raw.get("segments")
    if type(segments) is int and segments >= 0:
        # Reports written before segment_indices was introduced still carry
        # the number of schedule segments and can be resumed without replaying
        # the same prefix.
        return segments, "generalized-sidecar"
    raise GeneralizedTrainingError(
        f"generalized sidecar {sidecar} is missing a valid segment cursor"
    )


def train_generalized(
    config: GeneralizedTrainingConfig = GeneralizedTrainingConfig(),
    *,
    progress_callback: Callable[[int, int, tuple[OpponentScenario, ...], Mapping[str, object]], None]
    | None = None,
    progress_step_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Train across sampled deck/controller scenarios and save one checkpoint."""

    if not isinstance(config, GeneralizedTrainingConfig):
        raise TypeError("config must be a GeneralizedTrainingConfig")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable when provided")
    if progress_step_callback is not None and not callable(progress_step_callback):
        raise TypeError("progress_step_callback must be callable when provided")

    try:
        from ..ruleset import load_ruleset
    except ImportError:  # pragma: no cover - top-level ``rl`` layout
        from simulator.ruleset import load_ruleset

    ruleset = load_ruleset(config.prototype_config.ruleset_id)
    try:
        player_deck = tuple(
            ruleset.resolve_card_id(card) for card in config.player_deck
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GeneralizedTrainingError(
            "player_deck contains a card unavailable in ruleset "
            f"{ruleset.ruleset_id!r}: {error}"
        ) from error
    if len(set(player_deck)) != len(player_deck):
        raise GeneralizedTrainingError(
            "player_deck must not contain duplicate canonical cards"
        )
    pool = OpponentPool(
        ruleset,
        seed=_mix_seed(
            config.prototype_config.seed,
            config.curriculum.seed,
            0x43555252,
        ),
    )
    current_checkpoint = config.checkpoint
    starting_segment, segment_offset_source = _resolve_segment_offset(config)
    stage_reports: list[Mapping[str, object]] = []
    stage_scenarios: list[list[dict[str, object]]] = []
    stage_opponent_assignments: list[list[dict[str, object]]] = []
    stage_metadata: list[dict[str, object]] = []
    transitions_per_segment = (
        config.prototype_config.envs
        * config.prototype_config.horizon
        * config.rollouts_per_scenario
    )
    total_transitions = config.segments * transitions_per_segment

    pfsp_sampler: PFSPOpponentSampler | None = None
    payoff_book: LeaguePayoffBook | None = None
    payoff_book_source: str | None = None
    pfsp_matches_recorded = 0
    if config.pfsp:
        if config.pfsp_payoff_book is None:
            payoff_book = LeaguePayoffBook()
            payoff_book_source = "empty"
        else:
            try:
                payoff_book = LeaguePayoffBook.from_json(config.pfsp_payoff_book)
            except Exception as error:
                raise GeneralizedTrainingError(
                    f"cannot load PFSP payoff book {config.pfsp_payoff_book}: {error}"
                ) from error
            payoff_book_source = str(config.pfsp_payoff_book)
        pfsp_sampler = PFSPOpponentSampler(
            payoff_book=payoff_book,
            seed=_mix_seed(
                config.prototype_config.seed,
                config.curriculum.seed,
                0x50465350,
            ),
        )

    expert_action = None
    if config.expert_guidance:
        if config.expert_teacher == "public-counter":
            from .public_counter import public_counter_action

            expert_action = public_counter_action
        elif config.expert_teacher == "strategic-counter":
            from .public_counter import strategic_counter_action

            expert_action = strategic_counter_action
        else:
            from .expert import deterministic_counter_action

            expert_action = deterministic_counter_action

    segment_indices: list[int] = []
    for local_segment_index in range(config.segments):
        segment_index = starting_segment + local_segment_index
        segment_indices.append(segment_index)
        archetypes, strategies, stage = _curriculum_axes(config, segment_index)
        scenarios = sample_training_scenarios(
            pool,
            envs=config.prototype_config.envs,
            segment_index=segment_index,
            archetypes=archetypes,
            strategies=strategies,
            include_regression=config.include_regression,
            threat_stratified=config.threat_stratified,
        )
        checkpoint_assignments = _scenario_checkpoint_assignments(
            scenarios,
            config.opponent_checkpoints,
            pfsp_sampler=pfsp_sampler,
            learner_agent_id=config.league_agent_id,
            match_index_base=segment_index * config.prototype_config.envs,
        )
        pfsp_weights = None
        if pfsp_sampler is not None:
            pfsp_weights = dict(
                pfsp_sampler.weights(
                    config.league_agent_id,
                    tuple(str(path) for path in config.opponent_checkpoints),
                )
            )
        segment_config = _segment_config(
            config.prototype_config,
            segment_index=segment_index,
            rollouts_per_scenario=config.rollouts_per_scenario,
            potential_reward_anneal_segments=(
                config.potential_reward_anneal_segments
            ),
        )
        transition_offset = local_segment_index * transitions_per_segment

        def segment_progress(
            completed: int,
            *,
            offset: int = transition_offset,
        ) -> None:
            if progress_step_callback is not None:
                progress_step_callback(offset + completed, total_transitions)

        report = train_prototype(
            segment_config,
            checkpoint=current_checkpoint,
            checkpoint_out=config.checkpoint_out,
            progress_step_callback=segment_progress if progress_step_callback else None,
            player_deck=player_deck,
            opponent_decks=tuple(scenario.deck.cards for scenario in scenarios),
            opponent_action=make_scenario_opponent_action(
                scenarios,
                checkpoint_opponents=config.opponent_checkpoints,
                checkpoint_assignments=checkpoint_assignments,
                device=config.prototype_config.device,
            ),
            expert_guidance=config.expert_guidance,
            expert_action_callback=expert_action,
        )
        audit = report.get("simulation_exploit_audit")
        if isinstance(audit, Mapping) and audit.get("status") != "clean":
            raise GeneralizedTrainingError(
                "quarantining generalized segment because the simulator-exploit "
                f"audit is not clean: {audit!r}"
            )
        current_checkpoint = config.checkpoint_out
        stage_reports.append(report)
        if pfsp_sampler is not None and payoff_book is not None:
            payoff_book, recorded = _update_payoff_book_from_segment(
                payoff_book,
                report,
                checkpoint_assignments,
                learner_agent_id=config.league_agent_id,
            )
            pfsp_matches_recorded += recorded
            if recorded:
                # The sampler is immutable so that a report can always record
                # the exact weights used for a segment. Replacing it here
                # makes the next segment use the newly observed payoffs.
                pfsp_sampler = replace(pfsp_sampler, payoff_book=payoff_book)
        stage_scenarios.append([scenario.as_dict() for scenario in scenarios])
        stage_opponent_assignments.append(
            _scenario_assignment_rows(
                scenarios,
                config.opponent_checkpoints,
                checkpoint_assignments=checkpoint_assignments,
            )
        )
        stage_metadata.append(
            {
                "segment": segment_index,
                "stage_id": None if stage is None else stage.stage_id,
                "archetypes": list(archetypes),
                "strategies": list(strategies),
                "learner_actions": (
                    "actor-sampled"
                    if not (
                        config.expert_guidance
                        and config.prototype_config.expert_execution_probability > 0.0
                    )
                    else "mixed-teacher-and-actor"
                ),
                "actor_controls_actions": not (
                    config.expert_guidance
                    and config.prototype_config.expert_execution_probability > 0.0
                ),
                "potential_reward_weight": segment_config.potential_reward_weight,
                "pfsp_weights": pfsp_weights,
            }
        )
        if progress_callback is not None:
            progress_callback(local_segment_index + 1, config.segments, scenarios, report)

    if config.pfsp and config.pfsp_payoff_book_out is not None:
        if payoff_book is None:  # pragma: no cover - config invariant
            raise GeneralizedTrainingError("PFSP payoff book was not initialized")
        payoff_path = Path(config.pfsp_payoff_book_out)
        payoff_path.parent.mkdir(parents=True, exist_ok=True)
        payoff_path.write_text(payoff_book.to_json() + "\n", encoding="utf-8")

    final_report = stage_reports[-1] if stage_reports else {}
    aggregate = {
        "completed_matches": sum(int(report.get("outcomes", {}).get("completed_matches", 0)) for report in stage_reports),
        "wins": sum(int(report.get("outcomes", {}).get("wins", 0)) for report in stage_reports),
        "draws": sum(int(report.get("outcomes", {}).get("draws", 0)) for report in stage_reports),
        "losses": sum(int(report.get("outcomes", {}).get("losses", 0)) for report in stage_reports),
        "truncated_matches": sum(int(report.get("outcomes", {}).get("truncated_matches", 0)) for report in stage_reports),
    }
    report = {
        "kind": GENERALIZED_TRAINING_KIND,
        "schema_version": GENERALIZED_TRAINING_SCHEMA_VERSION,
        "checkpoint": str(current_checkpoint),
        # A fingerprint, when available, binds the generalized sidecar to the
        # exact output artifact.  It is optional so older sidecars remain
        # usable; their checkpoint path and identity metadata are still checked.
        "checkpoint_fingerprint": _file_fingerprint(current_checkpoint),
        "starting_checkpoint": None if config.checkpoint is None else str(config.checkpoint),
        "segments": config.segments,
        "rollouts_per_scenario": config.rollouts_per_scenario,
        "starting_segment": starting_segment,
        "segment_offset_source": segment_offset_source,
        "segment_indices": segment_indices,
        "envs": config.prototype_config.envs,
        "horizon": config.prototype_config.horizon,
        "sequence_length": config.prototype_config.sequence_length,
        "potential_reward_weight": config.prototype_config.potential_reward_weight,
        "potential_reward_anneal_segments": config.potential_reward_anneal_segments,
        "player_deck": list(player_deck),
        "transitions": total_transitions,
        "aggregate_outcomes": aggregate,
        "final_update": final_report.get("final_update"),
        "ruleset_id": ruleset.ruleset_id,
        "ruleset_hash": ruleset.content_hash,
        "actor_privileged_inputs": False,
        "critic_privileged_inputs": bool(config.prototype_config.use_privileged_critic),
        "actor_controls_actions": not (
            config.expert_guidance
            and config.prototype_config.expert_execution_probability > 0.0
        ),
        "include_regression": config.include_regression,
        "threat_stratified": config.threat_stratified,
        "curriculum_enabled": config.use_curriculum,
        "curriculum": config.curriculum.as_dict(),
        "stage_metadata": stage_metadata,
        "expert_guidance": config.expert_guidance,
        "expert_teacher": config.expert_teacher,
        "opponent_checkpoints": [str(path) for path in config.opponent_checkpoints],
        "pfsp": config.pfsp,
        "pfsp_payoff_book": (
            None if config.pfsp_payoff_book is None else str(config.pfsp_payoff_book)
        ),
        "pfsp_payoff_book_source": (
            None if not config.pfsp else payoff_book_source
        ),
        "pfsp_payoff_book_output": (
            None
            if config.pfsp_payoff_book_out is None
            else str(config.pfsp_payoff_book_out)
        ),
        "pfsp_matches_recorded": pfsp_matches_recorded,
        "pfsp_updates_payoff_book": pfsp_matches_recorded > 0,
        "pfsp_payoff_book_state": (
            None if payoff_book is None else payoff_book.as_dict()
        ),
        "train_archetypes": list(config.train_archetypes),
        "train_strategies": list(config.train_strategies),
        "scenario_schedule": stage_scenarios,
        "opponent_assignments": stage_opponent_assignments,
        "segment_reports": stage_reports,
        "warning": (
            "The generalized run uses the provisional simulator ruleset unless "
            "the selected ruleset reports training_ready=true."
        ),
    }
    from .exploit_audit import audit_simulation_report

    report["simulation_exploit_audit"] = audit_simulation_report(report)
    return report


def _read_training_report(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    """Read a generalized report and reject non-object JSON documents."""

    try:
        report_path = Path(path)
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise GeneralizedTrainingError(
            f"cannot read training report {path}: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise GeneralizedTrainingError(
            f"training report {report_path} must contain a JSON object"
        )
    return report_path, raw


def _report_card_key(
    cards: object,
    *,
    report_path: Path,
    ruleset: Any | None = None,
    field: str = "scenario deck",
) -> frozenset[str]:
    """Validate and canonicalize one report deck composition."""

    if isinstance(cards, (str, bytes)) or not isinstance(cards, Sequence):
        raise GeneralizedTrainingError(
            f"training report {report_path} contains an invalid {field}"
        )
    if len(cards) != 8 or any(
        not isinstance(card, str) or not card.strip() for card in cards
    ):
        raise GeneralizedTrainingError(
            f"training report {report_path} contains an invalid {field}"
        )
    normalized = tuple(card.strip() for card in cards)
    if len(set(normalized)) != 8:
        raise GeneralizedTrainingError(
            f"training report {report_path} contains an invalid {field}"
        )
    if ruleset is not None:
        try:
            normalized = tuple(ruleset.resolve_card_id(card) for card in normalized)
        except (KeyError, TypeError, ValueError) as error:
            raise GeneralizedTrainingError(
                f"training report {report_path} contains an unknown card in {field}: {error}"
            ) from error
        if len(set(normalized)) != 8:
            raise GeneralizedTrainingError(
                f"training report {report_path} contains duplicate canonical cards in {field}"
            )
    return frozenset(normalized)


def _training_schedule_deck_keys(
    raw: Mapping[str, Any],
    *,
    report_path: Path,
    ruleset: Any | None = None,
) -> set[frozenset[str]]:
    """Extract validated deck compositions from a report schedule."""

    schedule = raw.get("scenario_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise GeneralizedTrainingError(
            f"training report {report_path} has no usable scenario_schedule"
        )
    keys: set[frozenset[str]] = set()
    for segment in schedule:
        if not isinstance(segment, list) or not segment:
            raise GeneralizedTrainingError(
                f"training report {report_path} contains an invalid scenario segment"
            )
        for scenario in segment:
            if not isinstance(scenario, Mapping):
                raise GeneralizedTrainingError(
                    f"training report {report_path} contains a non-object scenario"
                )
            if (
                "schema_version" in scenario
                and (
                    type(scenario["schema_version"]) is not int
                    or scenario["schema_version"] != 1
                )
            ):
                raise GeneralizedTrainingError(
                    f"training report {report_path} contains an unsupported scenario schema"
                )
            deck = scenario.get("deck")
            if not isinstance(deck, Mapping):
                raise GeneralizedTrainingError(
                    f"training report {report_path} contains an invalid scenario deck"
                )
            if (
                "schema_version" in deck
                and (
                    type(deck["schema_version"]) is not int
                    or deck["schema_version"] != 1
                )
            ):
                raise GeneralizedTrainingError(
                    f"training report {report_path} contains an unsupported deck schema"
                )
            keys.add(
                _report_card_key(
                    deck.get("cards"),
                    report_path=report_path,
                    ruleset=ruleset,
                )
            )
    if not keys:
        raise GeneralizedTrainingError(
            f"training report {report_path} contains no scenario decks"
        )
    return keys


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == len("sha256:") + 64
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _is_sha256_file_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolved_path(path: str | Path) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=False)
    except OSError:
        return Path(path).expanduser().absolute()


def _checkpoint_reference_status(
    report_path: Path,
    report_checkpoint: str,
    checkpoint: str | Path,
) -> tuple[bool, bool]:
    """Return ``(matches, comparable)`` for a report/checkpoint reference."""

    actual = Path(checkpoint)
    reference = Path(report_checkpoint)
    candidates = (reference, report_path.parent / reference)
    actual_resolved = _resolved_path(actual)
    if any(_resolved_path(candidate) == actual_resolved for candidate in candidates):
        return True, True

    actual_fingerprint = _file_fingerprint(actual)
    reference_fingerprints = [
        _file_fingerprint(candidate)
        for candidate in candidates
        if candidate.is_file()
    ]
    if actual_fingerprint.get("exists") and reference_fingerprints:
        actual_sha = actual_fingerprint.get("sha256")
        return (
            any(
                fingerprint.get("sha256") == actual_sha
                for fingerprint in reference_fingerprints
            ),
            True,
        )
    return False, False


def _validate_checkpoint_fingerprint(
    raw: Mapping[str, Any],
    *,
    report_path: Path,
    checkpoint: str | Path | None,
) -> bool:
    """Validate an optional report fingerprint and return artifact status."""

    if "checkpoint_fingerprint" not in raw:
        return True
    fingerprint = raw["checkpoint_fingerprint"]
    if not isinstance(fingerprint, Mapping):
        raise GeneralizedTrainingError(
            f"training report {report_path} has invalid checkpoint_fingerprint metadata"
        )
    exists = fingerprint.get("exists")
    if type(exists) is not bool:
        raise GeneralizedTrainingError(
            f"training report {report_path} has invalid checkpoint_fingerprint.exists"
        )
    declared_sha = fingerprint.get("sha256")
    if exists and not _is_sha256_file_digest(declared_sha):
        raise GeneralizedTrainingError(
            f"training report {report_path} has an invalid checkpoint fingerprint"
        )
    if not exists:
        return False
    if checkpoint is None:
        return False
    actual = _file_fingerprint(checkpoint)
    if not actual.get("exists"):
        return False
    if actual.get("sha256") != declared_sha:
        raise GeneralizedTrainingError(
            f"training report {report_path} checkpoint fingerprint does not match {checkpoint}"
        )
    return True


def _validate_training_report_metadata(
    raw: Mapping[str, Any],
    *,
    report_path: Path,
    checkpoint: str | Path | None = None,
    ruleset: Any | None = None,
    player_deck: Sequence[str] | None = None,
) -> bool:
    """Validate report identity and return whether provenance is complete.

    The identity fields are optional for compatibility with early generalized
    reports.  If an old report omits one, its deck exclusions remain useful,
    but the caller must not label the resulting matrix as disjointly held out.
    A field that is present but contradictory is an error rather than a reason
    to silently certify the split.
    """

    verified = True

    if "kind" not in raw:
        verified = False
    elif raw["kind"] != GENERALIZED_TRAINING_KIND:
        raise GeneralizedTrainingError(
            f"training report {report_path} has unsupported kind {raw['kind']!r}"
        )

    if "schema_version" not in raw:
        verified = False
    elif (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != GENERALIZED_TRAINING_SCHEMA_VERSION
    ):
        raise GeneralizedTrainingError(
            f"training report {report_path} has unsupported schema_version"
        )

    for field, expected in (
        ("ruleset_id", None if ruleset is None else ruleset.ruleset_id),
        ("ruleset_hash", None if ruleset is None else ruleset.content_hash),
    ):
        if field not in raw:
            verified = False
            continue
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise GeneralizedTrainingError(
                f"training report {report_path} has invalid {field} metadata"
            )
        if field == "ruleset_hash" and not _is_sha256_digest(value):
            raise GeneralizedTrainingError(
                f"training report {report_path} has an invalid ruleset_hash"
            )
        if expected is not None and value != expected:
            raise GeneralizedTrainingError(
                f"training report {report_path} {field} does not match the evaluation ruleset"
            )

    if "player_deck" not in raw:
        verified = False
    else:
        report_deck = _report_card_key(
            raw["player_deck"],
            report_path=report_path,
            ruleset=ruleset,
            field="player_deck",
        )
        if player_deck is not None:
            expected_deck = _report_card_key(
                player_deck,
                report_path=report_path,
                ruleset=ruleset,
                field="evaluation player_deck",
            )
            if report_deck != expected_deck:
                raise GeneralizedTrainingError(
                    f"training report {report_path} player_deck does not match the evaluation deck"
                )

    if "checkpoint" not in raw:
        verified = False
    else:
        report_checkpoint = raw["checkpoint"]
        if not isinstance(report_checkpoint, str) or not report_checkpoint.strip():
            raise GeneralizedTrainingError(
                f"training report {report_path} has invalid checkpoint metadata"
            )
        if checkpoint is None:
            verified = False
        else:
            matches, comparable = _checkpoint_reference_status(
                report_path,
                report_checkpoint,
                checkpoint,
            )
            if comparable and not matches:
                raise GeneralizedTrainingError(
                    f"training report {report_path} checkpoint does not match {checkpoint}"
                )
            if not matches or not Path(checkpoint).is_file():
                verified = False

    if not _validate_checkpoint_fingerprint(
        raw,
        report_path=report_path,
        checkpoint=checkpoint,
    ):
        verified = False

    # These fields are emitted by newer prototype reports.  Validate them when
    # present without making them mandatory for older generalized sidecars.
    if "checkpoint_format" in raw:
        try:
            from .prototype import PROTOTYPE_CHECKPOINT_FORMAT
        except ImportError:  # pragma: no cover - top-level ``rl`` layout
            from rl.prototype import PROTOTYPE_CHECKPOINT_FORMAT
        if raw["checkpoint_format"] != PROTOTYPE_CHECKPOINT_FORMAT:
            raise GeneralizedTrainingError(
                f"training report {report_path} has unsupported checkpoint_format"
            )
    if (
        "actor_privileged_inputs" in raw
        and raw["actor_privileged_inputs"] is not False
    ):
        raise GeneralizedTrainingError(
            f"training report {report_path} does not identify a public-only actor"
        )
    return verified


def _load_training_report_audit(
    path: str | Path,
    *,
    checkpoint: str | Path | None = None,
    ruleset: Any | None = None,
    player_deck: Sequence[str] | None = None,
) -> _TrainingReportAudit:
    report_path, raw = _read_training_report(path)
    keys = _training_schedule_deck_keys(
        raw,
        report_path=report_path,
        ruleset=ruleset,
    )
    verified = _validate_training_report_metadata(
        raw,
        report_path=report_path,
        checkpoint=checkpoint,
        ruleset=ruleset,
        player_deck=player_deck,
    )
    return _TrainingReportAudit(frozenset(keys), verified)


def _load_training_deck_keys(path: str | Path) -> set[frozenset[str]]:
    """Read deck compositions from a generalized training report.

    This compatibility helper retains its historical set-returning API.  The
    held-out builder also consumes the audit status so legacy reports cannot
    certify disjointness merely because they contain a schedule.
    """

    return set(_load_training_report_audit(path).deck_keys)


def _matrix_deck(
    pool: OpponentPool,
    archetype: str,
    index: int,
    *,
    excluded_decks: set[frozenset[str]] | None = None,
) -> MatrixDeckSpec:
    excluded_decks = set() if excluded_decks is None else excluded_decks
    candidate_index = HELD_OUT_DECK_INDEX + index
    for _attempt in range(4_096):
        # Evaluation-only variants make finite curated archetypes (especially
        # air-beatdown) capable of producing a genuinely disjoint held-out
        # deck after training has covered every exact recipe.  The variant
        # sampler preserves the archetype core and is deterministic.
        deck = pool.sample_deck(
            candidate_index,
            archetype=archetype,
            allow_variants=True,
        )
        if frozenset(deck.cards) not in excluded_decks:
            break
        candidate_index += 1
    else:
        raise GeneralizedTrainingError(
            f"could not find a held-out {archetype!r} deck disjoint from training"
        )
    return MatrixDeckSpec(
        deck_id=f"heldout-{archetype}-{deck.deck_id}",
        cards=tuple(deck.cards),
        tags=tuple((*deck.tags, "held-out")),
        metadata={"archetype": archetype, "split": "held-out"},
    )


def build_heldout_matrix_config(
    checkpoint: str | Path,
    *,
    player_deck: Sequence[str] | None = None,
    seed: int = 0,
    archetypes: Sequence[str] = ARCHETYPE_NAMES,
    strategies: Sequence[str] = (
        "deterministic-cycle",
        "aggressive-pressure",
        "defensive-cycle",
        "beatdown",
        "siege-bait",
        "random-legal",
    ),
    seeds: Sequence[int] = (10_000,),
    policy_mode: str = "actor",
    device: str | None = "auto",
    max_decisions: int | None = None,
    batch_size: int = 8,
    include_match_results: bool = True,
    shuffle_decks: bool = True,
    training_report: str | Path | None = None,
) -> EvaluationMatrixConfig:
    """Create a held-out deck × controller × seed evaluation configuration."""

    try:
        from ..ruleset import load_fixed_ruleset
    except ImportError:  # pragma: no cover - top-level ``rl`` layout
        from simulator.ruleset import load_fixed_ruleset

    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=seed)
    archetype_names = _names("archetypes", archetypes)
    for archetype in archetype_names:
        if archetype not in ARCHETYPE_NAMES:
            raise GeneralizedTrainingError(f"unknown held-out archetype: {archetype!r}")
    strategy_names = _names("strategies", strategies)
    normalized_player_deck = _normalize_player_deck(player_deck)
    try:
        canonical_player_deck = tuple(
            ruleset.resolve_card_id(card) for card in normalized_player_deck
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GeneralizedTrainingError(
            f"player_deck contains a card unavailable in ruleset {ruleset.ruleset_id!r}: {error}"
        ) from error
    if len(set(canonical_player_deck)) != len(canonical_player_deck):
        raise GeneralizedTrainingError(
            "player_deck must not contain duplicate canonical cards"
        )

    if training_report is None:
        # Keep the historical optional argument, but do not let the matrix
        # claim a held-out split when no training provenance was supplied.
        excluded_decks: set[frozenset[str]] = set()
        held_out = False
    else:
        training_audit = _load_training_report_audit(
            training_report,
            checkpoint=checkpoint,
            ruleset=ruleset,
            player_deck=canonical_player_deck,
        )
        # A legacy schedule can still prevent accidental deck reuse.  It is
        # only a verified held-out source when all identity checks passed.
        excluded_decks = set(training_audit.deck_keys)
        held_out = training_audit.provenance_verified
    deck_specs_list: list[MatrixDeckSpec] = []
    for index, archetype in enumerate(archetype_names):
        deck = _matrix_deck(
            pool,
            archetype,
            index,
            excluded_decks=excluded_decks
            | {frozenset(candidate.cards) for candidate in deck_specs_list},
        )
        deck_specs_list.append(deck)
    deck_specs = tuple(deck_specs_list)
    strategy_specs = tuple(
        OpponentStrategySpec(
            strategy_id=strategy,
            factory=(lambda match_seed, selected=strategy: make_opponent_controller(selected, seed=match_seed)),
            description=f"simulator-side {strategy} controller",
        )
        for strategy in strategy_names
    )
    return EvaluationMatrixConfig(
        checkpoint=checkpoint,
        opponent_decks=deck_specs,
        strategies=strategy_specs,
        seeds=tuple(seeds),
        player_deck=player_deck,
        policy_mode=policy_mode,
        target_player=0,
        max_decisions=max_decisions,
        device=device,
        batch_size=batch_size,
        shuffle_decks=shuffle_decks,
        include_match_results=include_match_results,
        held_out=held_out,
        held_out_source=training_report,
        excluded_deck_compositions=tuple(
            tuple(cards) for cards in sorted(excluded_decks)
        ),
    )


def evaluate_heldout_matrix(
    checkpoint: str | Path,
    *,
    player_deck: Sequence[str] | None = None,
    seed: int = 0,
    archetypes: Sequence[str] = ARCHETYPE_NAMES,
    strategies: Sequence[str] = (
        "deterministic-cycle",
        "aggressive-pressure",
        "defensive-cycle",
        "beatdown",
        "siege-bait",
        "random-legal",
    ),
    seeds: Sequence[int] = (10_000,),
    policy_mode: str = "actor",
    device: str | None = "auto",
    max_decisions: int | None = None,
    batch_size: int = 8,
    include_match_results: bool = True,
    shuffle_decks: bool = True,
    training_report: str | Path | None = None,
    progress_callback: Callable[[int, int, MatchSpec, MatchResult], None] | None = None,
) -> dict[str, object]:
    config = build_heldout_matrix_config(
        checkpoint,
        player_deck=player_deck,
        seed=seed,
        archetypes=archetypes,
        strategies=strategies,
        seeds=seeds,
        policy_mode=policy_mode,
        device=device,
        max_decisions=max_decisions,
        batch_size=batch_size,
        include_match_results=include_match_results,
        shuffle_decks=shuffle_decks,
        training_report=training_report,
    )
    return run_evaluation_matrix(config, progress_callback=progress_callback)


def _csv_names(value: str) -> tuple[str, ...]:
    return _names("names", tuple(part for part in value.split(",") if part.strip()))


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise GeneralizedTrainingError("seeds must be comma-separated integers") from error
    if not values:
        raise GeneralizedTrainingError("seeds must not be empty")
    return values


def _csv_paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(part.strip()) for part in value.split(",") if part.strip())
    if not paths:
        raise GeneralizedTrainingError("opponent-checkpoints must not be empty")
    return paths


def _csv_deck(value: str) -> tuple[str, ...]:
    return _normalize_player_deck(_csv_names(value))


class _Progress:
    def __init__(self, label: str, total: int, *, stream: Any = None) -> None:
        self.label = label
        self.total = max(1, total)
        self.stream = sys.stderr if stream is None else stream
        self._last_filled = -1

    def update(self, completed: int) -> None:
        completed = max(0, min(self.total, int(completed)))
        width = 24
        filled = int(width * completed / self.total)
        if filled == self._last_filled and completed < self.total:
            return
        self._last_filled = filled
        self.stream.write(
            f"\r{self.label} [{'#' * filled}{'-' * (width - filled)}] "
            f"{completed}/{self.total}"
        )
        self.stream.flush()

    def close(self) -> None:
        self.stream.write("\n")
        self.stream.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rl.generalized")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train against sampled deck/controller scenarios")
    train.add_argument("--updates", type=int, default=20, help="number of sampled rollout segments")
    train.add_argument(
        "--rollouts-per-scenario",
        type=int,
        default=1,
        help=(
            "number of consecutive rollouts kept on each sampled scenario; "
            "use >1 to expose late-match states before resampling"
        ),
    )
    train.add_argument(
        "--segment-offset",
        type=int,
        help=(
            "global scenario index for this run; inferred from a generalized "
            "checkpoint sidecar when omitted"
        ),
    )
    train.add_argument("--envs", type=int, default=4)
    train.add_argument("--horizon", type=int, default=512)
    train.add_argument(
        "--env-backend",
        choices=("reference", "process", "packed-process"),
        default="reference",
        help=(
            "simulator lane backend; process variants can improve CPU throughput "
            "when their serialization overhead is amortized"
        ),
    )
    train.add_argument(
        "--env-workers",
        type=int,
        help="worker count for process or packed-process lane backends",
    )
    train.add_argument("--seed", type=int, default=0)
    defaults = GeneralizedTrainingConfig()
    train.add_argument(
        "--player-deck",
        default=",".join(defaults.player_deck),
        help="comma-separated eight-card learner deck",
    )
    train.add_argument(
        "--device",
        default="auto",
        help="policy device (default: auto; pass cpu to force host execution)",
    )
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--checkpoint-out", type=Path, default=Path("outputs/simulator/training/generalized-recurrent-prototype.pt"))
    train.add_argument("--allow-provisional", action="store_true")
    train.add_argument("--no-regression", action="store_true")
    train.add_argument(
        "--flat-curriculum",
        action="store_true",
        help="disable the staged opponent distribution and use --archetypes/--strategies as-is",
    )
    train.add_argument(
        "--threat-stratified",
        action="store_true",
        help=(
            "reserve non-regression lanes for air-beatdown and beatdown "
            "defensive sequences in every segment"
        ),
    )
    train.add_argument("--expert-guidance", action="store_true")
    train.add_argument(
        "--expert-teacher",
        choices=("public-counter", "strategic-counter", "deterministic-counter"),
        default="public-counter",
        help="teacher used for expert-guided public-actor imitation",
    )
    train.add_argument("--bc-coef", type=float, default=0.0)
    train.add_argument(
        "--bc-factor-coef",
        type=float,
        default=0.0,
        help="balanced mode/card/placement imitation loss for expert guidance",
    )
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--update-epochs", type=int, default=3)
    train.add_argument("--sequence-minibatch-size", type=int, default=8)
    train.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="recurrent PPO chunk length; must divide horizon (default: 128)",
    )
    train.add_argument("--gamma", type=float, default=0.9995)
    train.add_argument("--gae-lambda", type=float, default=0.98)
    train.add_argument(
        "--explicit-hand-features",
        action="store_true",
        help=(
            "project each public one-hot card-table hand slot independently; "
            "the Transformer processes entities only"
        ),
    )
    train.add_argument(
        "--direct-public-action-features",
        action="store_true",
        help="feed public global elixir/hand features directly to the WAIT/PLAY gate",
    )
    train.add_argument(
        "--direct-public-card-features",
        action="store_true",
        help="feed public global elixir/hand features directly to card-slot selection",
    )
    train.add_argument(
        "--contextual-public-card-features",
        action="store_true",
        help=(
            "include recurrent public-entity context in the direct card-slot head "
            "for state-dependent defense/pressure choices"
        ),
    )
    train.add_argument(
        "--direct-public-mask-features",
        action="store_true",
        help="feed public legality masks directly to the WAIT/PLAY gate",
    )
    train.add_argument(
        "--direct-public-context-features",
        action="store_true",
        help="use a nonlinear public hand/elixir/legality context for the WAIT/PLAY gate",
    )
    train.add_argument(
        "--direct-public-slot-card-features",
        action="store_true",
        help=(
            "score each public hand slot from its one-hot card identity; "
            "requires explicit hand features"
        ),
    )
    train.add_argument(
        "--spatial-placement-features",
        action="store_true",
        help="retain board-aligned raster features for card-conditioned placement",
    )
    model_group = train.add_mutually_exclusive_group()
    model_group.add_argument(
        "--strategic-model",
        dest="strategic_model",
        action="store_true",
        help=(
            "use the larger public recurrent actor with projected hand features "
            "and spatial placement features (default)"
        ),
    )
    model_group.add_argument(
        "--small-model",
        dest="strategic_model",
        action="store_false",
        help="use the compact model for fast smoke tests",
    )
    train.set_defaults(strategic_model=True)
    train.add_argument(
        "--imitation-only",
        action="store_true",
        help="use only supervised teacher-action loss during expert warm-start",
    )
    train.add_argument(
        "--expert-execution-probability",
        type=float,
        default=0.0,
        help="teacher-action execution probability; lower values enable DAgger states",
    )
    train.add_argument(
        "--deterministic-rollouts",
        action="store_true",
        help=(
            "use argmax actor actions during collection; useful for deterministic "
            "DAgger recovery-state training"
        ),
    )
    train.add_argument("--entropy-coef", type=float, default=0.01)
    train.add_argument("--belief-coef", type=float, default=0.05)
    train.add_argument("--no-belief-targets", action="store_true")
    train.add_argument(
        "--no-privileged-critic",
        action="store_true",
        help="use an actor-observation value head for a minimal smoke run",
    )
    train.add_argument(
        "--opponent-checkpoints",
        default="",
        help=(
            "comma-separated frozen public actor checkpoints; non-regression "
            "training lanes use them as self-play opponents"
        ),
    )
    train.add_argument(
        "--pfsp",
        action="store_true",
        help=(
            "sample frozen opponent checkpoints with payoff-aware PFSP; requires "
            "--opponent-checkpoints"
        ),
    )
    train.add_argument(
        "--pfsp-payoff-book",
        type=Path,
        help="optional JSON payoff history used by --pfsp (unseen opponents stay eligible)",
    )
    train.add_argument(
        "--pfsp-payoff-book-out",
        type=Path,
        help=(
            "optional JSON path receiving completed frozen-opponent outcomes; "
            "use with --pfsp to continue PFSP across runs"
        ),
    )
    train.add_argument(
        "--league-agent-id",
        default="main",
        help="learner ID used for directional PFSP payoff lookup",
    )
    train.add_argument("--archetypes", default=",".join(defaults.train_archetypes))
    train.add_argument("--strategies", default=",".join(defaults.train_strategies))
    reward_group = train.add_mutually_exclusive_group()
    reward_group.add_argument(
        "--dense-reward",
        dest="dense_reward",
        action="store_true",
        help="legacy per-step tower/crown shaping; prefer potential-reward-weight",
    )
    reward_group.add_argument(
        "--no-dense-reward",
        dest="dense_reward",
        action="store_false",
        help="disable legacy dense reward shaping",
    )
    train.set_defaults(dense_reward=False)
    train.add_argument(
        "--potential-reward-weight",
        type=float,
        default=0.1,
        help="temporary normalized tower/crown potential coefficient (0 disables it)",
    )
    train.add_argument(
        "--potential-reward-anneal-segments",
        type=int,
        default=32,
        help=(
            "global segment at which temporary potential shaping reaches zero; "
            "0 keeps the configured coefficient fixed"
        ),
    )
    train.add_argument("--json-out", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on a held-out matrix")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--seeds", default="10000")
    evaluate.add_argument(
        "--player-deck",
        default=",".join(_default_player_deck()),
        help="comma-separated eight-card learner deck",
    )
    evaluate.add_argument(
        "--device",
        default="auto",
        help="inference device (default: auto; pass cpu to force host execution)",
    )
    evaluate.add_argument(
        "--policy",
        choices=("actor", "public-counter", "strategic-counter", "deterministic-counter"),
        default="actor",
        help=(
            "target action source; deterministic-counter is an authoritative-state "
            "training-teacher diagnostic, not actor evidence"
        ),
    )
    evaluate.add_argument("--max-decisions", type=int)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument(
        "--opponent-checkpoints",
        default="",
        help=(
            "comma-separated frozen actor checkpoints for self-play comparison; "
            "when set, the matrix uses the fixed prototype deck"
        ),
    )
    evaluate.add_argument("--archetypes", default=",".join(ARCHETYPE_NAMES))
    evaluate.add_argument(
        "--strategies",
        default="deterministic-cycle,aggressive-pressure,defensive-cycle,beatdown,siege-bait,random-legal",
    )
    evaluate.add_argument("--no-match-results", action="store_true")
    evaluate.add_argument("--no-shuffle", action="store_true")
    evaluate.add_argument(
        "--training-report",
        type=Path,
        help=(
            "generalized training JSON whose sampled deck compositions must be "
            "excluded from the held-out matrix"
        ),
    )
    evaluate.add_argument("--json-out", type=Path)

    compare = subparsers.add_parser(
        "compare",
        help="compare two completed, paired evaluation-matrix reports",
    )
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument(
        "--max-cells",
        type=int,
        default=4096,
        help="maximum number of per-cell rows accepted from either report",
    )
    compare.add_argument(
        "--allow-truncations",
        action="store_true",
        help="do not fail the quality gate when either report contains truncations",
    )
    compare.add_argument(
        "--allow-rejected-actions",
        action="store_true",
        help="do not fail the quality gate when either report contains rejections",
    )
    compare.add_argument("--json-out", type=Path)
    return parser


def _write_json(path: Path | None, value: Mapping[str, object]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True)
    if path is None:
        print(encoded)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "train":
            prototype_config = PrototypeConfig(
                envs=args.envs,
                horizon=args.horizon,
                updates=1,
                seed=args.seed,
                device=args.device,
                env_backend=args.env_backend,
                env_workers=args.env_workers,
                learning_rate=args.learning_rate,
                update_epochs=args.update_epochs,
                sequence_minibatch_size=args.sequence_minibatch_size,
                sequence_length=args.sequence_length,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                imitation_only=args.imitation_only,
                expert_execution_probability=args.expert_execution_probability,
                deterministic_rollouts=args.deterministic_rollouts,
                entropy_coef=args.entropy_coef,
                dense_reward=args.dense_reward,
                potential_reward_weight=args.potential_reward_weight,
                behavior_cloning_coef=args.bc_coef,
                behavior_cloning_factor_coef=args.bc_factor_coef,
                direct_public_action_features=args.direct_public_action_features,
                direct_public_card_features=args.direct_public_card_features,
                contextual_public_card_features=args.contextual_public_card_features,
                direct_public_mask_features=args.direct_public_mask_features,
                direct_public_context_features=args.direct_public_context_features,
                direct_public_slot_card_features=args.direct_public_slot_card_features,
                model_dim=128 if args.strategic_model else 32,
                encoder_dim=128 if args.strategic_model else 32,
                transformer_heads=4,
                transformer_layers=2 if args.strategic_model else 1,
                transformer_ff_dim=256 if args.strategic_model else 64,
                gru_hidden_dim=256 if args.strategic_model else 32,
                explicit_hand_features=(
                    True if args.strategic_model else args.explicit_hand_features
                ),
                spatial_placement_features=(
                    True if args.strategic_model else args.spatial_placement_features
                ),
                belief_coef=args.belief_coef,
                use_privileged_critic=not args.no_privileged_critic,
                collect_belief_targets=not args.no_belief_targets,
                allow_provisional=args.allow_provisional,
            )
            config = GeneralizedTrainingConfig(
                prototype_config=prototype_config,
                player_deck=_csv_deck(args.player_deck),
                segments=args.updates,
                rollouts_per_scenario=args.rollouts_per_scenario,
                segment_offset=args.segment_offset,
                checkpoint=args.checkpoint,
                checkpoint_out=args.checkpoint_out,
                include_regression=not args.no_regression,
                threat_stratified=args.threat_stratified,
                use_curriculum=not args.flat_curriculum,
                potential_reward_anneal_segments=args.potential_reward_anneal_segments,
                expert_guidance=args.expert_guidance,
                expert_teacher=args.expert_teacher,
                opponent_checkpoints=(
                    _csv_paths(args.opponent_checkpoints)
                    if args.opponent_checkpoints.strip()
                    else ()
                ),
                pfsp=args.pfsp,
                pfsp_payoff_book=args.pfsp_payoff_book,
                pfsp_payoff_book_out=args.pfsp_payoff_book_out,
                league_agent_id=args.league_agent_id,
                train_archetypes=_csv_names(args.archetypes),
                train_strategies=_csv_names(args.strategies),
            )
            progress_total = (
                config.segments
                * config.prototype_config.envs
                * config.prototype_config.horizon
                * config.rollouts_per_scenario
            )
            progress = _Progress("generalized PPO", progress_total)
            try:
                report = train_generalized(
                    config,
                    progress_step_callback=lambda complete, _total: progress.update(complete),
                )
            finally:
                progress.close()
        elif args.command == "evaluate":
            seeds = _csv_ints(args.seeds)
            if args.opponent_checkpoints.strip():
                if args.policy != "actor":
                    raise GeneralizedTrainingError(
                        "--opponent-checkpoints requires --policy actor"
                    )
                from .self_play import build_self_play_matrix_config

                matrix_config = build_self_play_matrix_config(
                    args.checkpoint,
                    _csv_paths(args.opponent_checkpoints),
                    player_deck=_csv_deck(args.player_deck),
                    seeds=seeds,
                    device=args.device,
                    max_decisions=args.max_decisions,
                    batch_size=args.batch_size,
                    include_match_results=not args.no_match_results,
                    shuffle_decks=not args.no_shuffle,
                )
            else:
                matrix_config = build_heldout_matrix_config(
                    args.checkpoint,
                    player_deck=_csv_deck(args.player_deck),
                    seed=args.seed,
                    archetypes=_csv_names(args.archetypes),
                    strategies=_csv_names(args.strategies),
                    seeds=seeds,
                    policy_mode=args.policy,
                    device=args.device,
                    max_decisions=args.max_decisions,
                    batch_size=args.batch_size,
                    include_match_results=not args.no_match_results,
                    shuffle_decks=not args.no_shuffle,
                    training_report=args.training_report,
                )
            progress = _Progress("held-out matrix", matrix_config.match_count)
            try:
                report = run_evaluation_matrix(
                    matrix_config,
                    progress_callback=lambda complete, _total, _spec, _result: progress.update(complete),
                )
            finally:
                progress.close()
        else:
            _baseline_path, baseline_report = _read_training_report(args.baseline)
            _candidate_path, candidate_report = _read_training_report(args.candidate)
            report = compare_evaluation_reports(
                baseline_report,
                candidate_report,
                max_cells=args.max_cells,
                reject_truncations=not args.allow_truncations,
                reject_rejected_actions=not args.allow_rejected_actions,
            )
    except (GeneralizedTrainingError, OpponentPoolError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    _write_json(args.json_out, report)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "GENERALIZED_TRAINING_KIND",
    "GENERALIZED_TRAINING_SCHEMA_VERSION",
    "GeneralizedTrainingConfig",
    "GeneralizedTrainingError",
    "build_heldout_matrix_config",
    "evaluate_heldout_matrix",
    "main",
    "make_scenario_opponent_action",
    "sample_training_scenarios",
    "train_generalized",
]
