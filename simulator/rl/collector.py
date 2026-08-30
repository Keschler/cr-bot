"""Environment-native recurrent rollout collection.

The collector is intentionally small and explicit.  It connects
``SimulatorEnv.observe_v2()`` to the optional recurrent PPO learner without
changing the legacy V1 environment API:

* actor inputs are only public V2 tensors;
* action masks are copied from the public legality contract;
* the GRU reset mask is true after terminal/time-limit episode resets;
* exact simulator values are optional, caller-supplied critic features;
* optional belief targets are produced from simulator state for training only.

This is a rollout storage/bridge component, not a league orchestrator.  The
caller still chooses the opponent action policy and decides when to optimize
the returned :class:`~rl.learner.LearnerBatch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import math
from typing import Any, Callable, Mapping, Sequence

from ._compat import TORCH_AVAILABLE, TorchUnavailableError

_TORCH_IMPORT_ERROR: BaseException | None = None

if TORCH_AVAILABLE:
    try:
        import torch

        from .learner import (
            BeliefTargets,
            LearnerBatch,
            RecurrentPPOLearner,
            RecurrentRolloutState,
        )
        from .trajectory import ActionBatch, ActionMasks, RecurrentSequence, TrajectoryBatch
    except TorchUnavailableError as exc:  # pragma: no cover - defensive path
        TORCH_AVAILABLE = False
        _TORCH_IMPORT_ERROR = exc
else:
    torch = None  # type: ignore[assignment]
    BeliefTargets = Any  # type: ignore[misc,assignment]
    LearnerBatch = Any  # type: ignore[misc,assignment]
    RecurrentPPOLearner = Any  # type: ignore[misc,assignment]
    RecurrentRolloutState = Any  # type: ignore[misc,assignment]
    ActionBatch = Any  # type: ignore[misc,assignment]
    ActionMasks = Any  # type: ignore[misc,assignment]
    RecurrentSequence = Any  # type: ignore[misc,assignment]
    TrajectoryBatch = Any  # type: ignore[misc,assignment]


def _raise_torch_unavailable() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise TorchUnavailableError(
            "The recurrent rollout collector requires PyTorch. Install the "
            "optional torch dependency before using rl.collector."
        ) from _TORCH_IMPORT_ERROR
    raise TorchUnavailableError(
        "The recurrent rollout collector requires PyTorch. Install the "
        "optional torch dependency before using rl.collector."
    )


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    """Fixed-shape settings for one recurrent rollout batch."""

    horizon: int = 128
    target_player: int = 0
    seed: int = 0
    lane_offset: int = 0
    shuffle_decks: bool = True
    decks: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    lane_decks: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] | None = None
    collect_belief_targets: bool = False
    actor_only_observations: bool = False
    deterministic: bool = False
    # A teacher may provide labels when explicitly configured, but regular
    # PPO must execute the actor's sampled action by default.
    expert_execution_probability: float = 0.0
    stop_on_episode_end: bool = False
    freeze_completed_lanes: bool = False
    # Full model/simulator decision records are opt-in because serializing
    # logits and authoritative snapshots is intentionally much slower than a
    # normal rollout.  It has no effect on actor inputs or action execution.
    diagnostics: bool = False

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if type(self.target_player) is not int or self.target_player not in (0, 1):
            raise ValueError("target_player must be 0 or 1")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.lane_offset) is not int or self.lane_offset < 0:
            raise ValueError("lane_offset must be a non-negative integer")
        if type(self.shuffle_decks) is not bool:
            raise TypeError("shuffle_decks must be boolean")
        if type(self.actor_only_observations) is not bool:
            raise TypeError("actor_only_observations must be boolean")
        if type(self.deterministic) is not bool:
            raise TypeError("deterministic must be boolean")
        if (
            isinstance(self.expert_execution_probability, bool)
            or not math.isfinite(float(self.expert_execution_probability))
            or not 0.0 <= float(self.expert_execution_probability) <= 1.0
        ):
            raise ValueError("expert_execution_probability must be in [0, 1]")
        if type(self.stop_on_episode_end) is not bool:
            raise TypeError("stop_on_episode_end must be boolean")
        if type(self.freeze_completed_lanes) is not bool:
            raise TypeError("freeze_completed_lanes must be boolean")
        if type(self.diagnostics) is not bool:
            raise TypeError("diagnostics must be boolean")
        if self.decks is not None:
            if len(self.decks) != 2 or any(len(deck) != 8 for deck in self.decks):
                raise ValueError("decks must contain two eight-card decks")
            if any(not isinstance(card, str) or not card.strip() for deck in self.decks for card in deck):
                raise ValueError("decks must contain non-empty card identifiers")
        if self.lane_decks is not None:
            if not self.lane_decks:
                raise ValueError("lane_decks must contain at least one lane pair")
            if self.decks is not None:
                raise ValueError("provide either decks or lane_decks, not both")
            for lane, pair in enumerate(self.lane_decks):
                if len(pair) != 2 or any(len(deck) != 8 for deck in pair):
                    raise ValueError(
                        f"lane_decks[{lane}] must contain two eight-card decks"
                    )
                if any(
                    not isinstance(card, str) or not card.strip()
                    for deck in pair
                    for card in deck
                ):
                    raise ValueError(
                        f"lane_decks[{lane}] must contain non-empty card identifiers"
                    )


@dataclass(frozen=True, slots=True)
class RolloutStats:
    """Outcome counters observed while collecting a rollout.

    A rollout boundary is not treated as a game result.  ``truncated`` counts
    only simulator runner-tick limits; the caller can therefore distinguish a
    completed win/loss/draw from a batch that simply ended before the match.
    """

    completed_matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    truncated_matches: int = 0
    # Optional lane-indexed terminal outcomes let a generalized league update
    # PFSP without pretending that aggregate wins belong to one opponent.
    match_outcomes: tuple[tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("completed_matches", self.completed_matches),
            ("wins", self.wins),
            ("draws", self.draws),
            ("losses", self.losses),
            ("truncated_matches", self.truncated_matches),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.match_outcomes, tuple):
            raise ValueError("match_outcomes must be a tuple")
        for index, item in enumerate(self.match_outcomes):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"match_outcomes[{index}] must be a (lane, outcome) tuple")
            lane, outcome = item
            if type(lane) is not int or lane < 0:
                raise ValueError(f"match_outcomes[{index}] lane must be non-negative")
            if outcome not in {"win", "draw", "loss"}:
                raise ValueError(
                    f"match_outcomes[{index}] outcome must be win, draw, or loss"
                )

    @property
    def episode_boundaries(self) -> int:
        return self.completed_matches + self.truncated_matches

    def as_dict(self) -> dict[str, object]:
        return {
            "completed_matches": self.completed_matches,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "truncated_matches": self.truncated_matches,
            "episode_boundaries": self.episode_boundaries,
            "match_outcomes": [
                {"lane": lane, "outcome": outcome}
                for lane, outcome in self.match_outcomes
            ],
        }


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Collected recurrent data and the value bootstrap for its final step."""

    learner_batch: Any
    final_observations: tuple[Any, ...]
    next_rollout_state: Any
    bootstrap_values: Any
    stats: RolloutStats = RolloutStats()
    next_reset_mask: Any = None
    episode_counts: tuple[int, ...] = ()

    @property
    def trajectory(self) -> Any:
        """Return the stored trajectory for callers that need raw tensors."""

        return self.learner_batch.trajectory


@dataclass(frozen=True, slots=True)
class RolloutDecision:
    """CPU-side details for an optional diagnostic decision callback.

    The callback is intentionally outside the learner tensors.  It is useful
    for evaluation/replay tooling, while keeping the actor's public-observation
    boundary and the training trajectory unchanged.
    """

    decision_index: int
    lane: int
    target_action: Any
    opponent_action: Any | None
    result: Any
    state_after: Any
    physics_tick_before: int | None
    elapsed_us_before: int | None
    hand_before: tuple[str, ...]
    elixir_before: int | None
    # The remaining fields are JSON-shaped, CPU-side diagnostics.  Keeping
    # them optional preserves the cheap callback contract for existing users.
    state_before: Any | None = None
    action_masks: Mapping[str, Any] | None = None
    policy_diagnostics: Mapping[str, Any] | None = None
    teacher_action: Any | None = None


PrivilegedFeatureFn = Callable[[Any, int], Sequence[float]]
OpponentActionFn = Callable[[Any, Any, int], Any | None]
ExpertActionFn = Callable[[Any, Any, int], Any]
BatchStepFn = Callable[[Sequence[Sequence[Any | None]]], Sequence[Any]]
RolloutStepCallback = Callable[[int], None]
RolloutDecisionCallback = Callable[[RolloutDecision], None]


@dataclass(frozen=True, slots=True)
class _FrozenStep:
    """Synthetic zero-reward step used for a lane after its match ends."""

    rewards: tuple[float, float] = (0.0, 0.0)
    terminated: bool = True
    truncated: bool = False
    info: Mapping[str, object] = field(default_factory=dict)


def _card_id(card_id: str, card_count: int) -> int:
    from cr_bot.domain.card_metadata import CARD_METADATA

    metadata = CARD_METADATA.get(card_id)
    raw_id = metadata.get("id") if isinstance(metadata, dict) else None
    if type(raw_id) is not int or not 0 <= raw_id < card_count:
        return -100
    return raw_id


if TORCH_AVAILABLE:

    def _expert_should_execute(config: CollectorConfig, lane: int, timestep: int) -> bool:
        """Return a reproducible teacher-forcing decision for one lane/step."""

        probability = float(config.expert_execution_probability)
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        value = (int(config.seed) & ((1 << 64) - 1))
        for part in (lane, timestep):
            value ^= (int(part) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
            value = (
                value * 6364136223846793005 + 1442695040888963407
            ) & ((1 << 64) - 1)
        return (value / float(1 << 64)) < probability

    class RecurrentRolloutCollector:
        """Collect fixed-length public-observation rollouts from simulator lanes.

        ``opponent_action`` is called for the non-training player after the
        current V2 observation is produced.  Returning ``None`` means WAIT.
        The callback receives a public observation and may not access actor
        tensors or hidden simulator fields through this API.

        ``privileged_feature_fn`` is deliberately separate: it is a
        training-only callback for an asymmetric critic and receives the
        authoritative environment object.  It is required when the learner
        was configured with a privileged critic.
        """

        def __init__(
            self,
            learner: RecurrentPPOLearner,
            config: CollectorConfig | None = None,
            *,
            opponent_action: OpponentActionFn | None = None,
            expert_action: ExpertActionFn | None = None,
            privileged_feature_fn: PrivilegedFeatureFn | None = None,
            batch_step: BatchStepFn | None = None,
        ) -> None:
            if not isinstance(learner, RecurrentPPOLearner):
                raise TypeError("learner must be a RecurrentPPOLearner")
            self.learner = learner
            self.config = CollectorConfig() if config is None else config
            if not isinstance(self.config, CollectorConfig):
                raise TypeError("config must be a CollectorConfig")
            self.opponent_action = opponent_action
            if expert_action is not None and not callable(expert_action):
                raise TypeError("expert_action must be callable when provided")
            self.expert_action = expert_action
            self.privileged_feature_fn = privileged_feature_fn
            if batch_step is not None and not callable(batch_step):
                raise TypeError("batch_step must be callable when provided")
            self.batch_step = batch_step
            policy_config = self.learner.policy.config
            expected_dimensions = {
                "raster_channels": 21,
                "raster_height": 32,
                "raster_width": 18,
                "global_dim": 768,
                "entity_dim": 32,
                "card_slots": 4,
                "placement_rows": 32,
                "placement_cols": 18,
            }
            for name, expected in expected_dimensions.items():
                actual = int(getattr(policy_config, name))
                if actual != expected:
                    raise ValueError(
                        f"the recurrent policy {name}={actual} is incompatible with "
                        f"public V2 ({expected})"
                    )
            if int(policy_config.max_entities) < 128:
                raise ValueError(
                    "the recurrent policy must support the full V2 entity-token bound of 128"
                )
            if learner.uses_privileged_critic and privileged_feature_fn is None:
                raise ValueError(
                    "privileged_feature_fn is required when the learner uses a privileged critic"
                )

        def collect(
            self,
            environments: Sequence[Any],
            *,
            rollout_state: RecurrentRolloutState | None = None,
            reset_mask: Any | None = None,
            episode_counts: Sequence[int] | None = None,
            step_callback: RolloutStepCallback | None = None,
            decision_callback: RolloutDecisionCallback | None = None,
        ) -> RolloutResult:
            """Collect one ``B x horizon`` rollout from initialized env lanes.

            ``rollout_state`` and ``reset_mask`` allow consecutive batches to
            continue the same GRU stream when an environment itself is still
            in progress.  The bootstrap observation is evaluated for its
            value, but is not consumed twice by the returned continuation
            state.
            """

            if not environments:
                raise ValueError("at least one environment is required")
            if step_callback is not None and not callable(step_callback):
                raise TypeError("step_callback must be callable when provided")
            if decision_callback is not None and not callable(decision_callback):
                raise TypeError("decision_callback must be callable when provided")
            for index, environment in enumerate(environments):
                if not hasattr(environment, "observe_v2") or not (
                    hasattr(environment, "step_v2") or hasattr(environment, "step")
                ):
                    raise TypeError(f"environments[{index}] is not a SimulatorEnv-like object")
                if getattr(environment, "state", None) is None:
                    raise RuntimeError(
                        f"environments[{index}] must be reset before collection"
                    )
                if bool(environment.state.terminal):
                    raise RuntimeError(
                        f"environments[{index}] is terminal; reset it before collection"
                    )

            batch_size = len(environments)
            target_player = self.config.target_player
            if self.config.lane_decks is not None and len(self.config.lane_decks) != batch_size:
                raise ValueError(
                    "lane_decks must contain one deck pair per environment lane"
                )
            current_observations = [
                _collector_observations(
                    environment,
                    target_player=target_player,
                    actor_only=self.config.actor_only_observations,
                )
                for environment in environments
            ]
            if rollout_state is None:
                reset_before_step = torch.ones(
                    batch_size,
                    dtype=torch.bool,
                    device=self.learner.device,
                )
                rollout_state = self.learner.initial_rollout_state(batch_size)
            else:
                if not isinstance(rollout_state, RecurrentRolloutState):
                    raise TypeError("rollout_state must be a RecurrentRolloutState")
                if rollout_state.hidden.shape[1] != batch_size:
                    raise ValueError("rollout_state batch dimension must match environments")
                rollout_state = rollout_state.detach()
                if reset_mask is None:
                    reset_before_step = torch.zeros(
                        batch_size,
                        dtype=torch.bool,
                        device=self.learner.device,
                    )
                else:
                    reset_before_step = torch.as_tensor(
                        reset_mask,
                        dtype=torch.bool,
                        device=self.learner.device,
                    )
                    if reset_before_step.shape != (batch_size,):
                        raise ValueError("reset_mask must have shape [batch]")
            if episode_counts is None:
                episode_counts = [0] * batch_size
            else:
                if len(episode_counts) != batch_size or any(
                    type(value) is not int or value < 0 for value in episode_counts
                ):
                    raise ValueError("episode_counts must contain one non-negative integer per environment")
                episode_counts = list(episode_counts)
            completed_matches = wins = draws = losses = truncated_matches = 0
            match_outcomes: list[tuple[int, str]] = []
            frozen_lanes = [False] * batch_size
            frozen_infos: list[Mapping[str, object]] = [{} for _ in range(batch_size)]

            raster_steps: list[torch.Tensor] = []
            global_steps: list[torch.Tensor] = []
            entity_steps: list[torch.Tensor] = []
            entity_mask_steps: list[torch.Tensor] = []
            reset_steps: list[torch.Tensor] = []
            mode_masks: list[torch.Tensor] = []
            card_masks: list[torch.Tensor] = []
            placement_masks: list[torch.Tensor] = []
            action_steps: list[ActionBatch] = []
            log_prob_steps: list[torch.Tensor] = []
            value_steps: list[torch.Tensor] = []
            reward_steps: list[torch.Tensor] = []
            terminated_steps: list[torch.Tensor] = []
            truncated_steps: list[torch.Tensor] = []
            hidden_steps: list[torch.Tensor] = []
            privileged_steps: list[torch.Tensor] = []
            belief_elixir_steps: list[torch.Tensor] = []
            belief_hand_steps: list[torch.Tensor] = []
            belief_next_steps: list[torch.Tensor] = []
            behavior_cloning_weight_steps: list[torch.Tensor] = []
            behavior_cloning_action_steps: list[ActionBatch] = []
            # A nonterminal truncation resets the environment for collection,
            # but its value target is still V(s) at the final pre-reset state.
            # Keep those evaluations by rollout timestep so the learner can
            # distinguish them from the reset episode's first observation.
            boundary_bootstrap_values: dict[int, torch.Tensor] = {}

            for _time in range(self.config.horizon):
                observations = [item[target_player] for item in current_observations]
                raster, global_features, entities, entity_mask, masks = _batch_observations(
                    observations,
                    device=self.learner.device,
                    inference=self.config.actor_only_observations,
                )
                hidden_steps.append(rollout_state.hidden.detach().permute(1, 0, 2))
                reset_steps.append(reset_before_step.clone())
                raster_steps.append(raster[:, 0])
                global_steps.append(global_features[:, 0])
                entity_steps.append(entities[:, 0])
                entity_mask_steps.append(entity_mask[:, 0])
                mode_masks.append(masks.mode[:, 0])
                card_masks.append(masks.card[:, 0])
                placement_masks.append(masks.placement[:, 0])

                privileged = self._privileged_batch(environments, target_player)
                if privileged is not None:
                    privileged_steps.append(privileged[:, 0])
                belief = self._belief_batch(environments, target_player)
                if belief is not None:
                    belief_elixir_steps.append(belief[0])
                    belief_hand_steps.append(belief[1])
                    belief_next_steps.append(belief[2])

                # Rollout collection is inference, not a differentiable
                # optimization pass.  Keeping these graphs alive across a
                # long match needlessly multiplies memory use and can make a
                # first prototype look like it is leaking state between
                # updates.
                with torch.inference_mode():
                    step = self.learner.rollout_step(
                        rollout_state,
                        raster[:, 0],
                        global_features[:, 0],
                        entities[:, 0],
                        entity_mask[:, 0],
                        masks,
                        reset_mask=reset_before_step,
                        privileged_features=privileged,
                        deterministic=self.config.deterministic,
                        include_beliefs=False,
                        inference=self.config.actor_only_observations,
                        # Diagnostics need every card-placement branch so the
                        # trace can report meaningful alternatives.  The
                        # normal actor path retains the fast sampler.
                        fast_sampling=(
                            self.expert_action is None and not self.config.diagnostics
                        ),
                    )
                rollout_state = step.next_state
                # Decode the sampled action before selecting the executed
                # action. The expert path still needs the sampled action for
                # frozen lanes, where the simulator state is no longer live.
                decoded_actions = _decode_actions(step.actions)
                teacher_actions_raw: list[Any | None] = [None] * batch_size
                if self.expert_action is None:
                    stored_actions = step.actions
                    stored_log_probs = step.log_probs
                else:
                    expert_actions: list[Any] = []
                    executed_actions: list[Any] = []
                    expert_weights: list[float] = []
                    for lane, environment in enumerate(environments):
                        if frozen_lanes[lane]:
                            # A frozen lane contributes synthetic zero-reward
                            # padding; keep its sampled action because there
                            # is no live state from which to ask the teacher.
                            expert_actions.append(decoded_actions[lane])
                            executed_actions.append(decoded_actions[lane])
                            expert_weights.append(0.0)
                            continue
                        expert_action = self.expert_action(
                            environment,
                            current_observations[lane][target_player],
                            target_player,
                        )
                        if expert_action is None:
                            raise ValueError("expert_action must return an action, not None")
                        teacher_actions_raw[lane] = expert_action
                        expert_actions.append(expert_action)
                        executed_actions.append(
                            expert_action
                            if _expert_should_execute(self.config, lane, _time)
                            else decoded_actions[lane]
                        )
                        expert_weights.append(
                            _expert_action_weight(environment, expert_action, target_player)
                        )
                    teacher_actions = _encode_actions(
                        expert_actions,
                        device=self.learner.device,
                    )
                    stored_actions = _encode_actions(
                        executed_actions,
                        device=self.learner.device,
                    )
                    with torch.inference_mode():
                        stored_log_probs = self.learner.policy.log_prob(
                            step.output,
                            stored_actions,
                            masks,
                        )
                    behavior_cloning_action_steps.append(
                        ActionBatch(
                            mode=teacher_actions.mode[:, 0],
                            card_slot=teacher_actions.card_slot[:, 0],
                            placement=teacher_actions.placement[:, 0],
                        )
                    )
                    decoded_actions = executed_actions
                    behavior_cloning_weight_steps.append(
                        torch.as_tensor(
                            expert_weights,
                            dtype=torch.float32,
                            device=self.learner.device,
                        )
                    )
                policy_diagnostics: list[Mapping[str, Any] | None] = [None] * batch_size
                if self.config.diagnostics:
                    from .diagnostics import build_policy_diagnostics

                    for lane in range(batch_size):
                        if frozen_lanes[lane]:
                            continue
                        policy_diagnostics[lane] = build_policy_diagnostics(
                            self.learner.policy,
                            step.output,
                            masks,
                            stored_actions,
                            lane=lane,
                            time=0,
                            teacher_action=teacher_actions_raw[lane],
                            actor_actions=step.actions,
                            critic_value=step.values,
                            old_log_prob=stored_log_probs,
                        )
                action_steps.append(
                    ActionBatch(
                        mode=stored_actions.mode[:, 0],
                        card_slot=stored_actions.card_slot[:, 0],
                        placement=stored_actions.placement[:, 0],
                    )
                )
                log_prob_steps.append(stored_log_probs[:, 0])
                value_steps.append(step.values[:, 0])

                # The simulator needs Python actions, but decoding one lane
                # at a time would synchronize the accelerator once for every
                # scalar field.  Copy the complete action batch to the host
                # once, then build the per-lane objects without touching the
                # device again.
                simulator_actions = []
                opponent_actions: list[Any | None] = []
                decision_snapshots: list[
                    tuple[int | None, int | None, tuple[str, ...], int | None, Any | None]
                ] = []
                for lane, environment in enumerate(environments):
                    target_action = decoded_actions[lane]
                    if decision_callback is not None:
                        state = getattr(environment, "state", None)
                        players = getattr(state, "players", ()) if state is not None else ()
                        player_state = (
                            players[target_player]
                            if len(players) > target_player
                            else None
                        )
                        decision_snapshots.append(
                            (
                                getattr(state, "tick", None),
                                getattr(state, "elapsed_us", None),
                                tuple(getattr(player_state, "hand", ())),
                                (
                                    int(player_state.elixir_milli)
                                    if player_state is not None
                                    else None
                                ),
                                deepcopy(state) if self.config.diagnostics else None,
                            )
                        )
                    if frozen_lanes[lane]:
                        opponent_action = None
                    elif self.opponent_action is None:
                        opponent_action = None
                    else:
                        opponent_player = 1 - target_player
                        opponent_action = self.opponent_action(
                            environment,
                            current_observations[lane][opponent_player],
                            opponent_player,
                        )
                    opponent_actions.append(opponent_action)
                    row: list[Any | None] = [None, None]
                    if not frozen_lanes[lane]:
                        row[target_player] = target_action
                        row[1 - target_player] = opponent_action
                    simulator_actions.append(tuple(row))

                previously_frozen = tuple(frozen_lanes)
                if self.batch_step is None or (
                    self.config.freeze_completed_lanes and any(previously_frozen)
                ):
                    results = [
                        (
                            _FrozenStep(info=frozen_infos[lane])
                            if previously_frozen[lane]
                            else (
                                _step_v2_for_collector(
                                    environment,
                                    actions,
                                    target_player=target_player,
                                    actor_only=self.config.actor_only_observations,
                                )
                                if hasattr(environment, "step_v2")
                                else environment.step(actions)
                            )
                        )
                        for lane, (environment, actions) in enumerate(zip(
                            environments,
                            simulator_actions,
                            strict=True,
                        ))
                    ]
                else:
                    results = list(self.batch_step(simulator_actions))
                    if len(results) != batch_size:
                        raise ValueError(
                            "batch_step must return one result per environment"
                        )
                reward_steps.append(
                    torch.as_tensor(
                        [result.rewards[target_player] for result in results],
                        dtype=torch.float32,
                        device=self.learner.device,
                    )
                )
                terminated_values = [bool(result.terminated) for result in results]
                truncated_values = [bool(result.truncated) for result in results]
                terminated = torch.as_tensor(
                    terminated_values,
                    dtype=torch.bool,
                    device=self.learner.device,
                )
                truncated = torch.as_tensor(
                    truncated_values,
                    dtype=torch.bool,
                    device=self.learner.device,
                )
                terminated_steps.append(terminated)
                truncated_steps.append(truncated)

                for lane, result in enumerate(results):
                    if previously_frozen[lane]:
                        continue
                    if result.terminated and not result.truncated:
                        completed_matches += 1
                        winner = result.info.get("winner")
                        if winner == target_player:
                            wins += 1
                            match_outcomes.append((lane, "win"))
                        elif winner == 1 - target_player:
                            losses += 1
                            match_outcomes.append((lane, "loss"))
                        else:
                            draws += 1
                            match_outcomes.append((lane, "draw"))
                    elif result.truncated:
                        truncated_matches += 1

                if decision_callback is not None:
                    for lane, (target_action, opponent_action, result, snapshot) in enumerate(
                        zip(
                            decoded_actions,
                            opponent_actions,
                            results,
                            decision_snapshots,
                            strict=True,
                        )
                    ):
                        if previously_frozen[lane]:
                            continue
                        decision_callback(
                            RolloutDecision(
                                decision_index=_time,
                                lane=lane,
                                target_action=target_action,
                                opponent_action=opponent_action,
                                result=result,
                                state_after=getattr(environments[lane], "state", None),
                                physics_tick_before=snapshot[0],
                                elapsed_us_before=snapshot[1],
                                hand_before=snapshot[2],
                                elixir_before=snapshot[3],
                                state_before=snapshot[4],
                                action_masks=(
                                    None
                                    if policy_diagnostics[lane] is None
                                    else policy_diagnostics[lane].get("legal_action_mask")
                                ),
                                policy_diagnostics=policy_diagnostics[lane],
                                teacher_action=teacher_actions_raw[lane],
                            )
                        )

                post_step_observations: list[tuple[Any, Any]] = []
                for environment, result in zip(environments, results, strict=True):
                    if hasattr(result, "observations"):
                        post_step_observations.append(
                            _collector_observations_from_result(
                                result.observations,
                                environment,
                                target_player=target_player,
                                actor_only=self.config.actor_only_observations,
                            )
                        )
                    else:
                        post_step_observations.append(
                            _collector_observations(
                                environment,
                                target_player=target_player,
                                actor_only=self.config.actor_only_observations,
                            )
                        )

                nonterminal_truncation = [
                    truncated_value and not terminated_value
                    for terminated_value, truncated_value in zip(
                        terminated_values,
                        truncated_values,
                        strict=True,
                    )
                ]
                if any(nonterminal_truncation):
                    boundary_done = ~torch.as_tensor(
                        nonterminal_truncation,
                        dtype=torch.bool,
                        device=self.learner.device,
                    )
                    no_reset = torch.zeros(
                        batch_size,
                        dtype=torch.bool,
                        device=self.learner.device,
                    )
                    boundary_values, _ = self._bootstrap(
                        environments,
                        post_step_observations,
                        rollout_state,
                        boundary_done,
                        boundary_done,
                    )
                    boundary_bootstrap_values[_time] = boundary_values

                next_observations: list[tuple[Any, Any]] = []
                # These values originate in the CPU-side simulator results.
                # Keep the host copy for control flow instead of reading one
                # GPU boolean per lane below.
                next_reset_values = [
                    terminated_value or truncated_value
                    for terminated_value, truncated_value in zip(
                        terminated_values,
                        truncated_values,
                        strict=True,
                    )
                ]
                next_reset = terminated | truncated
                for lane, (environment, result) in enumerate(zip(environments, results, strict=True)):
                    reset_observations = None
                    if next_reset_values[lane] and not (
                        self.config.freeze_completed_lanes and previously_frozen[lane]
                    ):
                        episode_counts[lane] += 1
                        if self.config.freeze_completed_lanes:
                            frozen_lanes[lane] = True
                            frozen_infos[lane] = result.info
                        else:
                            reset_observations = _reset_environment(
                                environment,
                                seed=self._episode_seed(lane, episode_counts[lane]),
                                decks=(
                                    self.config.lane_decks[lane]
                                    if self.config.lane_decks is not None
                                    else self.config.decks
                                ),
                                shuffle_decks=self.config.shuffle_decks,
                            )
                    if reset_observations is not None:
                        observations = _collector_observations_from_result(
                            reset_observations,
                            environment,
                            target_player=target_player,
                            actor_only=self.config.actor_only_observations,
                        )
                    else:
                        observations = post_step_observations[lane]
                    if self.config.freeze_completed_lanes and frozen_lanes[lane]:
                        observations = tuple(
                            None
                            if observation is None
                            else _frozen_observation(observation)
                            for observation in observations
                        )
                    next_observations.append(observations)
                current_observations = next_observations
                reset_before_step = next_reset
                if step_callback is not None:
                    step_callback((_time + 1) * batch_size)
                if self.config.stop_on_episode_end and (
                    all(frozen_lanes)
                    if self.config.freeze_completed_lanes
                    else all(next_reset_values)
                ):
                    break

            bootstrap_values, next_rollout_state = self._bootstrap(
                environments,
                current_observations,
                rollout_state,
                reset_before_step,
                terminated_steps[-1] | truncated_steps[-1],
            )
            final_boundary_values = boundary_bootstrap_values.get(len(value_steps) - 1)
            if final_boundary_values is not None:
                final_nonterminal_truncation = (
                    truncated_steps[-1] & ~terminated_steps[-1]
                )
                bootstrap_values = torch.where(
                    final_nonterminal_truncation,
                    final_boundary_values,
                    bootstrap_values,
                )

            next_values = None
            learner_bootstrap_values = bootstrap_values
            if boundary_bootstrap_values:
                value_tensor = torch.stack(value_steps, dim=1)
                next_values = torch.zeros_like(value_tensor)
                if value_tensor.shape[1] > 1:
                    next_values[:, :-1] = value_tensor[:, 1:]
                next_values[:, -1] = bootstrap_values
                for timestep, boundary_values in boundary_bootstrap_values.items():
                    nonterminal_truncation = (
                        truncated_steps[timestep] & ~terminated_steps[timestep]
                    )
                    next_values[:, timestep] = torch.where(
                        nonterminal_truncation,
                        boundary_values,
                        next_values[:, timestep],
                    )
                # Explicit per-transition targets are required because a
                # reset observation is not the successor of a truncated
                # transition.  The public RolloutResult still exposes the
                # final vector for callers that only need the boundary value.
                learner_bootstrap_values = None
            trajectory = TrajectoryBatch(
                sequence=RecurrentSequence(
                    raster=torch.stack(raster_steps, dim=1),
                    global_features=torch.stack(global_steps, dim=1),
                    entities=torch.stack(entity_steps, dim=1),
                    entity_mask=torch.stack(entity_mask_steps, dim=1),
                    reset_mask=torch.stack(reset_steps, dim=1),
                    hidden_states=torch.stack(hidden_steps, dim=1),
                    initial_hidden=hidden_steps[0].permute(1, 0, 2).contiguous(),
                ),
                action_masks=ActionMasks(
                    mode=torch.stack(mode_masks, dim=1),
                    card=torch.stack(card_masks, dim=1),
                    placement=torch.stack(placement_masks, dim=1),
                ),
                actions=ActionBatch(
                    mode=torch.stack([item.mode for item in action_steps], dim=1),
                    card_slot=torch.stack([item.card_slot for item in action_steps], dim=1),
                    placement=torch.stack([item.placement for item in action_steps], dim=1),
                ),
                rewards=torch.stack(reward_steps, dim=1),
                terminated=torch.stack(terminated_steps, dim=1),
                truncated=torch.stack(truncated_steps, dim=1),
                old_log_probs=torch.stack(log_prob_steps, dim=1),
                values=torch.stack(value_steps, dim=1),
            )
            privileged_features = (
                torch.stack(privileged_steps, dim=1) if privileged_steps else None
            )
            belief_targets = None
            if belief_elixir_steps:
                belief_targets = BeliefTargets(
                    enemy_elixir=torch.stack(belief_elixir_steps, dim=1),
                    enemy_hand=torch.stack(belief_hand_steps, dim=1),
                    enemy_next_card=torch.stack(belief_next_steps, dim=1),
                )
            learner_batch = LearnerBatch(
                trajectory=trajectory,
                privileged_features=privileged_features,
                belief_targets=belief_targets,
                next_values=next_values,
                bootstrap_values=learner_bootstrap_values,
                behavior_cloning_weights=(
                    torch.stack(behavior_cloning_weight_steps, dim=1)
                    if behavior_cloning_weight_steps
                    else None
                ),
                behavior_cloning_actions=(
                    ActionBatch(
                        mode=torch.stack(
                            [item.mode for item in behavior_cloning_action_steps],
                            dim=1,
                        ),
                        card_slot=torch.stack(
                            [item.card_slot for item in behavior_cloning_action_steps],
                            dim=1,
                        ),
                        placement=torch.stack(
                            [item.placement for item in behavior_cloning_action_steps],
                            dim=1,
                        ),
                    )
                    if behavior_cloning_action_steps
                    else None
                ),
            )
            return RolloutResult(
                learner_batch=learner_batch,
                final_observations=tuple(item[target_player] for item in current_observations),
                next_rollout_state=next_rollout_state,
                bootstrap_values=bootstrap_values,
                stats=RolloutStats(
                    completed_matches=completed_matches,
                    wins=wins,
                    draws=draws,
                    losses=losses,
                    truncated_matches=truncated_matches,
                    match_outcomes=tuple(match_outcomes),
                ),
                next_reset_mask=reset_before_step.detach().clone(),
                episode_counts=tuple(episode_counts),
            )

        def _episode_seed(self, lane: int, episode: int) -> int:
            # Avoid process-randomized ``hash`` so a rollout schedule can be
            # reproduced after a checkpoint restart.
            value = self.config.seed & ((1 << 64) - 1)
            value ^= (
                lane + self.config.lane_offset + 0x9E3779B97F4A7C15
            ) & ((1 << 64) - 1)
            value = (value * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            value ^= (episode + 0xD1B54A32D192ED03) & ((1 << 64) - 1)
            return value

        def _privileged_batch(self, environments: Sequence[Any], viewer: int) -> Any | None:
            if self.privileged_feature_fn is None:
                return None
            rows = [self.privileged_feature_fn(environment, viewer) for environment in environments]
            if not rows:
                return None
            width = len(rows[0])
            if width < 1 or any(len(row) != width for row in rows):
                raise ValueError("privileged feature rows must share a non-empty width")
            return torch.as_tensor(rows, dtype=torch.float32, device=self.learner.device).unsqueeze(1)

        def _belief_batch(self, environments: Sequence[Any], viewer: int) -> tuple[Any, Any, Any] | None:
            if not self.config.collect_belief_targets:
                return None
            card_count = int(self.learner.policy.config.belief_card_count)
            elixir: list[float] = []
            hands: list[list[float]] = []
            next_cards: list[int] = []
            for environment in environments:
                state = getattr(environment, "state", None)
                if state is None:
                    raise RuntimeError("environment lost its authoritative state during collection")
                opponent = state.players[1 - viewer]
                maximum = max(1, int(environment.engine.ruleset.match.max_elixir_milli))
                elixir.append(min(1.0, max(0.0, opponent.elixir_milli / maximum)))
                hand = [0.0] * card_count
                for card in opponent.hand:
                    index = _card_id(card, card_count)
                    if index >= 0:
                        hand[index] = 1.0
                hands.append(hand)
                next_cards.append(
                    _card_id(opponent.draw_pile[0], card_count)
                    if opponent.draw_pile
                    else -100
                )
            return (
                torch.tensor(elixir, dtype=torch.float32, device=self.learner.device),
                torch.tensor(hands, dtype=torch.float32, device=self.learner.device),
                torch.tensor(next_cards, dtype=torch.long, device=self.learner.device),
            )

        def _bootstrap(
            self,
            environments: Sequence[Any],
            observations: Sequence[tuple[Any, Any]],
            rollout_state: RecurrentRolloutState,
            reset_mask: Any,
            last_done: Any,
        ) -> tuple[Any, RecurrentRolloutState]:
            # A terminal simulator snapshot intentionally has no legal PLAY
            # cells and may also disable WAIT. Mixed-lane collection still
            # evaluates one batched value row for a sibling lane, so replace
            # rows which cannot contribute a future return with a wait-only
            # public snapshot before batching. Nonterminal truncations keep
            # their real post-step observation and are bootstrapped normally.
            done_values = tuple(bool(value) for value in last_done.detach().cpu().tolist())
            target_observations = [
                (
                    _frozen_observation(item[self.config.target_player])
                    if done and item[self.config.target_player] is not None
                    else item[self.config.target_player]
                )
                for item, done in zip(observations, done_values, strict=True)
            ]
            raster, global_features, entities, entity_mask, masks = _batch_observations(
                target_observations,
                device=self.learner.device,
            )
            privileged = self._privileged_batch(environments, self.config.target_player)
            with torch.no_grad():
                output = self.learner.policy(
                    raster,
                    global_features,
                    entities,
                    entity_mask,
                    reset_mask=reset_mask.reshape(-1, 1),
                    hidden=rollout_state.hidden,
                    action_masks=masks,
                    include_beliefs=False,
                )
                values = self.learner._critic_values(output, privileged)
                bootstrap = values[:, 0].detach()
            # A terminal or actual environment time-limit transition has no
            # future return.  A collector-boundary nonterminal lane keeps its
            # value estimate for GAE bootstrapping.
            bootstrap = torch.where(last_done, torch.zeros_like(bootstrap), bootstrap)
            # ``output`` was evaluated only to obtain V(s_next).  The caller's
            # continuation state is the state before that next observation;
            # consuming it here would process the same observation twice on
            # the following rollout.
            return bootstrap, rollout_state.detach()

else:

    class RecurrentRolloutCollector:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_torch_unavailable()


def _batch_observations(
    observations: Sequence[Any],
    *,
    device: Any,
    inference: bool = False,
) -> tuple[Any, Any, Any, Any, Any]:
    if not TORCH_AVAILABLE:
        _raise_torch_unavailable()
    if not observations:
        raise ValueError("at least one observation is required")
    if type(inference) is not bool:
        raise TypeError("inference must be boolean")
    try:
        from ..observation_v2 import (
            OBSERVATION_V2_CONTRACT_HASH,
            OBSERVATION_V2_SCHEMA_VERSION,
        )
    except ImportError:  # pragma: no cover - top-level ``rl`` layout
        from simulator.observation_v2 import (
            OBSERVATION_V2_CONTRACT_HASH,
            OBSERVATION_V2_SCHEMA_VERSION,
        )
    for index, observation in enumerate(observations):
        if (
            getattr(observation, "schema_version", None) != OBSERVATION_V2_SCHEMA_VERSION
            or getattr(observation, "contract_hash", None) != OBSERVATION_V2_CONTRACT_HASH
        ):
            raise ValueError(
                f"observations[{index}] do not match the pinned public V2 contract"
            )
    import numpy as np

    def stack_to_device(
        values: Sequence[Any],
        *,
        dtype: Any,
        channels_last: bool = False,
    ) -> Any:
        # V2 arrays are fixed-shape NumPy snapshots.  Stack on the host first
        # so each field makes one contiguous device transfer instead of one
        # small transfer per lane.  A single-observation inference call is a
        # common deployment case; avoid np.stack's extra shape-building work
        # there while retaining the exact [batch, time, ...] result shape.
        if len(values) == 1:
            value = np.asarray(values[0])
            if not value.flags.c_contiguous:
                value = np.ascontiguousarray(value)
            elif not value.flags.writeable:
                value = value.copy(order="C")
            stacked = torch.from_numpy(value).unsqueeze(0).unsqueeze(1)
        else:
            stacked = torch.from_numpy(np.stack(values, axis=0)).unsqueeze(1)
        if channels_last and stacked.ndim == 5 and torch.device(device).type == "cpu":
            # The model's deployment convolution consumes flattened 4-D
            # frames.  Convert that view once at the observation boundary so
            # act_deterministic does not repeat the copy on every call.
            flat = stacked.reshape(
                stacked.shape[0] * stacked.shape[1],
                *stacked.shape[2:],
            )
            stacked = flat.contiguous(memory_format=torch.channels_last).reshape(
                stacked.shape
            )
        return stacked.to(device=device, dtype=dtype)

    board = stack_to_device(
        [getattr(observation, "board") for observation in observations],
        dtype=torch.float32,
        channels_last=inference,
    )
    global_features = stack_to_device(
        [getattr(observation, "global_vector") for observation in observations],
        dtype=torch.float32,
    )
    entities = stack_to_device(
        [getattr(observation, "entity_tokens") for observation in observations],
        dtype=torch.float32,
    )
    entity_mask = stack_to_device(
        [observation.entity_mask for observation in observations],
        dtype=torch.bool,
    )
    structured_masks = [observation.structured_action_masks() for observation in observations]
    mode_masks, card_masks, placement_masks = zip(*structured_masks, strict=True)
    masks = ActionMasks(
        mode=stack_to_device(mode_masks, dtype=torch.bool),
        card=stack_to_device(card_masks, dtype=torch.bool),
        placement=stack_to_device(placement_masks, dtype=torch.bool),
    )
    return board, global_features, entities, entity_mask, masks


def _reset_environment(environment: Any, **kwargs: Any) -> Any:
    """Reset through the V2 boundary when the environment provides it."""

    reset_v2 = getattr(environment, "reset_v2", None)
    if callable(reset_v2):
        return reset_v2(**kwargs)
    return environment.reset(**kwargs)


def _collector_observations(
    environment: Any,
    *,
    target_player: int,
    actor_only: bool,
) -> tuple[Any, Any]:
    """Return the collector's pair, optionally avoiding the unused view."""

    if actor_only:
        viewer_observation = getattr(environment, "observe_v2_for_viewer", None)
        if callable(viewer_observation):
            actor_observation = viewer_observation(target_player)
        else:
            actor_observation = tuple(environment.observe_v2())[target_player]
        pair: list[Any] = [None, None]
        pair[target_player] = actor_observation
        return tuple(pair)  # type: ignore[return-value]
    return tuple(environment.observe_v2())


def _collector_observations_from_result(
    raw_observations: Any,
    environment: Any,
    *,
    target_player: int,
    actor_only: bool,
) -> tuple[Any, Any]:
    """Normalize reset/step observations without rebuilding an identical view."""

    if isinstance(raw_observations, Sequence) and len(raw_observations) == 2:
        if actor_only:
            actor_observation = raw_observations[target_player]
            if actor_observation is None:
                return _collector_observations(
                    environment,
                    target_player=target_player,
                    actor_only=True,
                )
            pair: list[Any] = [None, None]
            pair[target_player] = actor_observation
            return tuple(pair)  # type: ignore[return-value]
        return tuple(raw_observations)
    return _collector_observations(
        environment,
        target_player=target_player,
        actor_only=actor_only,
    )


def _step_v2_for_collector(
    environment: Any,
    actions: Sequence[Any],
    *,
    target_player: int,
    actor_only: bool,
) -> Any:
    """Use the single-view step only for the restricted farm collector."""

    if actor_only:
        step_for_viewer = getattr(environment, "step_v2_for_viewer", None)
        if callable(step_for_viewer):
            return step_for_viewer(actions, viewer=target_player)
    return environment.step_v2(actions)


def _frozen_observation(observation: Any) -> Any:
    """Make a terminal lane safe to use as rollout padding.

    A terminal simulator snapshot may correctly expose no legal action at all.
    Batched collection still needs one more policy row while sibling lanes
    finish, so provide a wait-only public snapshot without changing the
    underlying environment state.
    """

    from dataclasses import replace
    import numpy as np

    legal_play = np.zeros_like(observation.legal_play, dtype=bool)
    return replace(observation, legal_play=legal_play, legal_wait=True)


def _decode_actions(actions: Any) -> list[Any]:
    """Decode one complete action batch with a single device-to-host copy."""

    from cr_bot.domain.game_state import Action as PolicyAction

    packed = torch.cat(
        (
            actions.mode[:, 0].reshape(-1, 1),
            actions.card_slot[:, 0].reshape(-1, 1),
            actions.placement[:, 0],
        ),
        dim=1,
    )
    host_rows = packed.detach().cpu().tolist()
    decoded: list[Any] = []
    for mode, card_slot, row, column in host_rows:
        if mode == 0:
            decoded.append(PolicyAction(kind="Wait"))
        else:
            decoded.append(
                PolicyAction(
                    kind="Play",
                    card_idx=int(card_slot),
                    cell=(int(column), int(row)),
                )
            )
    return decoded


def _encode_actions(actions: Sequence[Any], *, device: Any) -> Any:
    """Encode simulator/PolicyAction objects into the actor action ABI."""

    if not TORCH_AVAILABLE:
        _raise_torch_unavailable()
    from cr_bot.domain.game_state import Action as PolicyAction

    modes: list[int] = []
    card_slots: list[int] = []
    placements: list[tuple[int, int]] = []
    for action in actions:
        if isinstance(action, PolicyAction):
            kind = str(getattr(action, "kind", "Wait")).casefold()
            if kind in {"wait", "noop", "no-op"}:
                modes.append(0)
                card_slots.append(0)
                placements.append((0, 0))
                continue
            if kind != "play":
                raise ValueError(f"expert action kind is not supported: {kind!r}")
            card_slot = int(action.card_idx)
            cell = action.cell
        else:
            try:
                from ..actions import PlayCardAction, WaitAction
            except ImportError:  # pragma: no cover - top-level ``rl`` layout
                from simulator.actions import PlayCardAction, WaitAction

            if isinstance(action, WaitAction):
                modes.append(0)
                card_slots.append(0)
                placements.append((0, 0))
                continue
            if not isinstance(action, PlayCardAction):
                raise TypeError("expert actions must be WaitAction, PlayCardAction, or PolicyAction")
            card_slot = int(action.card_slot)
            cell = action.cell
        if not isinstance(cell, tuple) or len(cell) != 2:
            raise ValueError("expert play action cell must be a two-item tuple")
        # The model ABI stores placement as (row, column), while simulator
        # actions use world cells as (column, row).
        modes.append(1)
        card_slots.append(card_slot)
        placements.append((int(cell[1]), int(cell[0])))
    return ActionBatch(
        mode=torch.as_tensor(modes, dtype=torch.long, device=device).unsqueeze(1),
        card_slot=torch.as_tensor(card_slots, dtype=torch.long, device=device).unsqueeze(1),
        placement=torch.as_tensor(placements, dtype=torch.long, device=device).unsqueeze(1),
    )


def _expert_action_weight(environment: Any, action: Any, player: int) -> float:
    """Weight decisive teacher plays without erasing resource-saving waits.

    The old ``0.05`` versus ``100`` range made the supervised dataset behave
    as if waiting were almost never correct.  That is especially damaging for
    Hog cycle: the teacher must wait while Hog is held but still unaffordable,
    otherwise the actor spends the available elixir on a cheap card and keeps
    Hog masked forever.  These moderate ratios still make a rare Hog label
    visible while preserving the state-dependent WAIT signal.
    """

    kind = str(getattr(action, "kind", "")).casefold().replace("_", "-")
    if kind in {"wait", "noop", "no-op"} or action.__class__.__name__ == "WaitAction":
        state = getattr(environment, "state", None)
        player_state = (
            state.players[player]
            if state is not None and len(getattr(state, "players", ())) > player
            else None
        )
        if player_state is not None and "hog-rider" in getattr(player_state, "hand", ()):
            # This is the critical public state that was previously lost: a
            # held Hog is not yet affordable, so WAIT is the correct action.
            # Make those labels as durable as the decisive Hog PLAY label.
            # The affordability boundary is only one or two 250 ms decisions
            # wide, so a small WAIT weight lets the actor spend a cheap card
            # immediately before Hog becomes legal.
            hog = environment.engine.ruleset.card("hog-rider")
            if int(getattr(player_state, "elixir_milli", 0)) < int(
                environment.engine._effective_card_cost(player_state, hog)
            ):
                return 20.0
        return 0.5
    slot = getattr(action, "card_slot", None)
    if slot is None:
        slot = getattr(action, "card_idx", None)
    state = getattr(environment, "state", None)
    hand = getattr(state.players[player], "hand", ()) if state is not None else ()
    card_id = hand[int(slot)] if slot is not None and 0 <= int(slot) < len(hand) else None
    if card_id == "hog-rider":
        return 20.0
    if card_id == "fireball":
        # Fireball is useful but much less common than Hog in the teacher's
        # traces.  A modest boost keeps the label visible without making a
        # short imitation run over-select the spell in unrelated states.
        return 3.0
    if card_id == "musketeer":
        # Musketeer is rarer than the cycle cards, but it is the deck's
        # reusable air answer.  Without a little class weight, a balanced
        # factor loss can still converge to the numerically common cheap-card
        # labels and never reproduce this critical defensive decision.
        return 40.0
    return 1.0


__all__ = [
    "BatchStepFn",
    "CollectorConfig",
    "ExpertActionFn",
    "OpponentActionFn",
    "PrivilegedFeatureFn",
    "RecurrentRolloutCollector",
    "RolloutDecision",
    "RolloutDecisionCallback",
    "RolloutStepCallback",
    "RolloutResult",
    "RolloutStats",
    "TORCH_AVAILABLE",
    "TorchUnavailableError",
]
