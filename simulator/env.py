"""Training environment over the vision-policy-compatible boundary.

This is intentionally a small dependency-free API instead of pretending to be
Gymnasium when Gymnasium is not a project dependency.  Thin wrappers can map
the returned records into any RL framework without changing simulator state or
action semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
from typing import TYPE_CHECKING, Iterable, Sequence

from cr_bot.domain.game_state import Action as PolicyAction

from .actions import SimAction, WaitAction
from .engine import ENGINE_VERSION, BattleEngine
from .observation import (
    PINNED_OBSERVATION_CONTRACT_HASH,
    ObservationMemory,
    PolicyObservationV1,
    build_policy_observation,
    decode_policy_action,
)
from .packed_batch import pack_state, unpack_state
from .ruleset import load_ruleset
from .state import BattleState, battle_state_from_primitive
from .roster import PLAYER_DECK

if TYPE_CHECKING:
    from .observation_v2 import PolicyObservationV2


@dataclass(frozen=True, slots=True)
class RewardConfig:
    version: str = "tower-damage-crowns-v1"
    tower_damage_weight: float = 1.0
    crown_weight: float = 1.0
    win_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("reward version must be a non-empty string")
        for name in ("tower_damage_weight", "crown_weight", "win_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")

    def as_dict(self) -> dict[str, object]:
        """Return the exact reward semantics for run provenance."""

        return {
            "version": self.version,
            "tower_damage_weight": float(self.tower_damage_weight),
            "crown_weight": float(self.crown_weight),
            "win_weight": float(self.win_weight),
        }

    @classmethod
    def terminal_outcome(cls) -> "RewardConfig":
        """Return the sparse match objective used by serious policy training."""

        return cls(
            version="terminal-outcome-v1",
            tower_damage_weight=0.0,
            crown_weight=0.0,
            win_weight=1.0,
        )

    @classmethod
    def terminal_with_potential(cls, weight: float = 0.1) -> "RewardConfig":
        """Add a bounded potential difference to the terminal objective.

        The shaped component is the difference in normalized tower/crown
        potential between consecutive simulator decisions. It supplies local
        credit assignment without making a human card-choice heuristic part of
        the objective. A zero weight is rejected here so callers do not
        accidentally label a sparse run as shaped.
        """

        if isinstance(weight, bool) or not math.isfinite(float(weight)) or float(weight) <= 0.0:
            raise ValueError("potential shaping weight must be finite and positive")
        return cls(
            version="terminal-potential-v1",
            tower_damage_weight=float(weight),
            crown_weight=float(weight),
            win_weight=1.0,
        )


@dataclass(frozen=True, slots=True)
class EnvStep:
    observations: tuple[PolicyObservationV1, PolicyObservationV1]
    rewards: tuple[float, float]
    terminated: bool
    truncated: bool
    info: dict[str, object]


@dataclass(frozen=True, slots=True)
class EnvStepV2:
    """Environment step with the sanitized public V2 observation boundary."""

    observations: tuple["PolicyObservationV2", "PolicyObservationV2"]
    rewards: tuple[float, float]
    terminated: bool
    truncated: bool
    info: dict[str, object]


def _parallel_env_step_worker(
    payload: tuple[
        str,
        int,
        str,
        float,
        float,
        float,
        bool,
        bool,
        dict[str, object],
        Sequence[PolicyAction | SimAction | None],
    ],
) -> tuple[dict[str, object], tuple[float, float], bool, bool, dict[str, object]]:
    """Advance one serialized environment in an isolated worker.

    The worker deliberately receives only a canonical state and public action
    row. It reconstructs the pinned ruleset/engine and returns the canonical
    next state plus reward metadata. Observations are built by the parent after
    the state is installed, preserving the parent's temporal observation
    memories and the exact policy boundary.
    """

    (
        ruleset_id,
        decision_interval_us,
        reward_version,
        tower_damage_weight,
        crown_weight,
        win_weight,
        expose_privileged_info,
        validate_every_tick,
        raw_state,
        actions,
    ) = payload
    engine = BattleEngine(
        load_ruleset(ruleset_id),
        validate_every_tick=validate_every_tick,
    )
    environment = SimulatorEnv(
        engine=engine,
        decision_interval_us=decision_interval_us,
        reward=RewardConfig(
            version=reward_version,
            tower_damage_weight=tower_damage_weight,
            crown_weight=crown_weight,
            win_weight=win_weight,
        ),
        expose_privileged_info=expose_privileged_info,
    )
    state = battle_state_from_primitive(raw_state)
    engine.validate_state(state)
    environment.state = state
    rewards, terminated, truncated, info = environment._step_core(actions)
    state = environment.state
    if state is None:  # pragma: no cover - defensive worker invariant
        raise RuntimeError("parallel worker lost its authoritative state")
    # Event history is part of the replay identity, not privileged observation
    # data. It must survive every worker round trip even when the caller keeps
    # authoritative state hidden from the policy.
    return (
        state.to_primitive(include_events=True),
        rewards,
        terminated,
        truncated,
        info,
    )


def _packed_parallel_env_step_worker(
    payload: tuple[
        str,
        int,
        str,
        float,
        float,
        float,
        bool,
        bool,
        bytes,
        Sequence[PolicyAction | SimAction | None],
    ],
) -> tuple[bytes, tuple[float, float], bool, bool, dict[str, object]]:
    """Advance one lane through the bounded packed-state transport.

    This is intentionally a transport-only backend.  The worker still runs
    the readable Python engine, but its authoritative state crosses the
    process boundary through the fixed-stride packed ABI rather than a
    pickle/JSON-shaped object graph.  Keeping this separate from the existing
    process worker makes differential parity failures easy to diagnose.
    """

    (
        ruleset_id,
        decision_interval_us,
        reward_version,
        tower_damage_weight,
        crown_weight,
        win_weight,
        expose_privileged_info,
        validate_every_tick,
        packed_state,
        actions,
    ) = payload
    engine = BattleEngine(
        load_ruleset(ruleset_id),
        validate_every_tick=validate_every_tick,
    )
    environment = SimulatorEnv(
        engine=engine,
        decision_interval_us=decision_interval_us,
        reward=RewardConfig(
            version=reward_version,
            tower_damage_weight=tower_damage_weight,
            crown_weight=crown_weight,
            win_weight=win_weight,
        ),
        expose_privileged_info=expose_privileged_info,
    )
    state = unpack_state(packed_state)
    engine.validate_state(state)
    environment.state = state
    rewards, terminated, truncated, info = environment._step_core(actions)
    state = environment.state
    if state is None:  # pragma: no cover - defensive worker invariant
        raise RuntimeError("packed worker lost its authoritative state")
    return (
        pack_state(state).to_bytes(),
        rewards,
        terminated,
        truncated,
        info,
    )


class SimulatorEnv:
    """One deterministic two-player battle with a fixed policy cadence."""

    def __init__(
        self,
        engine: BattleEngine | None = None,
        *,
        decision_interval_us: int = 250_000,
        reward: RewardConfig = RewardConfig(),
        expose_privileged_info: bool = False,
        include_authoritative_state: bool = True,
    ) -> None:
        # Full invariant validation remains available through an explicitly
        # supplied strict engine and the audit CLI. Per-tick schema walks are
        # intentionally disabled on the default high-throughput training path.
        self.engine = engine or BattleEngine(validate_every_tick=False)
        if decision_interval_us <= 0 or decision_interval_us % self.engine.ruleset.tick_us:
            raise ValueError("decision interval must be a positive multiple of ruleset tick_us")
        self.decision_interval_ticks = decision_interval_us // self.engine.ruleset.tick_us
        self.reward_config = reward
        self.expose_privileged_info = expose_privileged_info
        if type(include_authoritative_state) is not bool:
            raise TypeError("include_authoritative_state must be boolean")
        self.include_authoritative_state = include_authoritative_state
        self.state: BattleState | None = None
        self._memories = (ObservationMemory(0), ObservationMemory(1))

    def reset(
        self,
        *,
        seed: int = 0,
        decks: tuple[Iterable[str], Iterable[str]] | None = None,
        shuffle_decks: bool = True,
    ) -> tuple[PolicyObservationV1, PolicyObservationV1]:
        self.state = self.engine.new_battle(
            decks or (PLAYER_DECK, PLAYER_DECK),
            seed=seed,
            shuffle_decks=shuffle_decks,
        )
        self._memories = (ObservationMemory(0), ObservationMemory(1))
        return self.observe()

    def reset_v2(
        self,
        *,
        seed: int = 0,
        decks: tuple[Iterable[str], Iterable[str]] | None = None,
        shuffle_decks: bool = True,
    ) -> tuple["PolicyObservationV2", "PolicyObservationV2"]:
        """Reset and return only sanitized public V2 observations."""

        self.state = self.engine.new_battle(
            decks or (PLAYER_DECK, PLAYER_DECK),
            seed=seed,
            shuffle_decks=shuffle_decks,
        )
        self._memories = (ObservationMemory(0), ObservationMemory(1))
        return self.observe_v2()

    def observe(self) -> tuple[PolicyObservationV1, PolicyObservationV1]:
        state = self._require_state()
        return tuple(
            build_policy_observation(
                state,
                self.engine.ruleset,
                viewer=viewer,
                memory=self._memories[viewer],
                legality_callback=lambda battle, action: self.engine.validate_action(battle, action) is None,
            )
            for viewer in (0, 1)
        )  # type: ignore[return-value]

    def observe_v2(self) -> tuple["PolicyObservationV2", "PolicyObservationV2"]:
        """Return the additive public hybrid observation for both viewers.

        The existing :meth:`observe` method remains the stable V1 training
        boundary.  V2 is an explicit opt-in so callers can migrate model
        inputs without changing legacy trainer behavior or serialized
        environment records.
        """

        from .observation_v2_adapter import build_policy_observation_v2

        state = self._require_state()
        observations = tuple(
            build_policy_observation_v2(
                state,
                self.engine.ruleset,
                viewer=viewer,
                memory=self._memories[viewer],
                legality_callback=lambda battle, action: self.engine.validate_action(battle, action) is None,
            )
            for viewer in (0, 1)
        )
        return observations  # type: ignore[return-value]

    def step(
        self,
        actions: Sequence[PolicyAction | SimAction | None],
    ) -> EnvStep:
        rewards, terminated, truncated, info = self._step_core(actions)
        return EnvStep(
            observations=self.observe(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def step_v2(
        self,
        actions: Sequence[PolicyAction | SimAction | None],
    ) -> EnvStepV2:
        """Advance physics and return only the sanitized V2 observations.

        The recurrent prototype uses this method so the legacy V1 projection
        is not even constructed on its actor-facing step path.  The
        authoritative state and reward semantics remain identical to
        :meth:`step`.
        """

        rewards, terminated, truncated, info = self._step_core(actions)
        return EnvStepV2(
            observations=self.observe_v2(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_core(
        self,
        actions: Sequence[PolicyAction | SimAction | None],
    ) -> tuple[tuple[float, float], bool, bool, dict[str, object]]:
        """Advance authoritative physics without projecting observations.

        The public :meth:`step` adds the policy observation projection. The
        process backend uses this core operation in workers and projects each
        observation once in the parent, avoiding two discarded NumPy
        allocations per lane and preserving parent-owned observation memory.
        """

        state = self._require_state()
        if state.terminal:
            raise RuntimeError("step() cannot advance a terminal environment; call reset()")
        if len(actions) != 2:
            raise ValueError("step requires one action (or None) per player")
        before = self._reward_snapshot(state)
        event_start = len(state.events)
        sim_actions: list[SimAction] = []
        for viewer, action in enumerate(actions):
            if action is None:
                sim_actions.append(WaitAction(viewer))
            elif isinstance(action, PolicyAction):
                sim_actions.append(decode_policy_action(action, viewer=viewer))
            else:
                if action.player != viewer:
                    raise ValueError("sim action player does not match sequence position")
                sim_actions.append(action)
        for offset in range(self.decision_interval_ticks):
            self.engine.step(state, sim_actions if offset == 0 else ())
            if state.terminal:
                break
        rewards = self._rewards(before, state)
        new_events = state.events[event_start:]
        info: dict[str, object] = {
            "ruleset_id": state.ruleset_id,
            "ruleset_hash": state.ruleset_hash,
            "engine_version": ENGINE_VERSION,
            "observation_contract_hash": PINNED_OBSERVATION_CONTRACT_HASH,
            "physics_tick": state.tick,
            "decision_interval_ticks": self.decision_interval_ticks,
            "reward_version": self.reward_config.version,
            "winner": state.winner,
            "terminal_reason": state.terminal_reason,
        }
        if self.expose_privileged_info:
            info["state_hash"] = state.state_hash()
            info["event_log_hash"] = state.event_log_hash()
            info["events"] = tuple(event.to_dict() for event in new_events)
            if self.include_authoritative_state:
                info["authoritative_state"] = state.to_primitive(include_events=False)
        return (
            rewards,
            state.terminal,
            state.terminal_reason == "runner_tick_limit",
            info,
        )

    def save_state(self) -> dict[str, object]:
        return self._require_state().to_primitive(include_events=True)

    def load_state(self, raw: dict[str, object]) -> tuple[PolicyObservationV1, PolicyObservationV1]:
        state = battle_state_from_primitive(raw)
        self.engine.validate_state(state)
        self.state = state
        self._memories = (ObservationMemory(0), ObservationMemory(1))
        return self.observe()

    def _require_state(self) -> BattleState:
        if self.state is None:
            raise RuntimeError("reset() or load_state() must be called first")
        return self.state

    @staticmethod
    def _reward_snapshot(state: BattleState) -> tuple[tuple[int, int], tuple[int, int]]:
        hp = [0, 0]
        maximum = [0, 0]
        for entity in state.entities.values():
            if entity.kind == "tower":
                hp[entity.owner] += max(0, entity.hp)
                maximum[entity.owner] += entity.max_hp
        return (tuple(hp), tuple(player.crowns for player in state.players))  # type: ignore[return-value]

    def _rewards(
        self,
        before: tuple[tuple[int, int], tuple[int, int]],
        state: BattleState,
    ) -> tuple[float, float]:
        before_hp, before_crowns = before
        after_hp, after_crowns = self._reward_snapshot(state)
        maximum_hp = [0, 0]
        for entity in state.entities.values():
            if entity.kind == "tower":
                maximum_hp[entity.owner] += entity.max_hp
        damage_by = [
            (before_hp[1 - player] - after_hp[1 - player]) / max(1, maximum_hp[1 - player])
            for player in (0, 1)
        ]
        crown_gain = [after_crowns[player] - before_crowns[player] for player in (0, 1)]
        rewards = [
            self.reward_config.tower_damage_weight * (damage_by[player] - damage_by[1 - player])
            + self.reward_config.crown_weight * (crown_gain[player] - crown_gain[1 - player])
            for player in (0, 1)
        ]
        if state.terminal and state.winner is not None:
            rewards[state.winner] += self.reward_config.win_weight
            rewards[1 - state.winner] -= self.reward_config.win_weight
        return float(rewards[0]), float(rewards[1])


class VectorSimulatorEnv:
    """Deterministic vector environment with reference and process backends.

    ``backend="reference"`` preserves the low-overhead sequential wrapper.
    ``backend="process"`` executes independent lanes concurrently in isolated
    worker processes. ``backend="packed-process"`` uses the bounded packed
    state ABI for that same worker boundary; it is a transport prototype, not
    the final SoA/JIT physics backend. All modes use the same parent-owned states,
    observations, actions, rewards, and canonical hashes. The process backend
    is a deterministic parallel reference backend; a future SoA/JIT backend
    can replace its worker implementation without changing this API.
    """

    def __init__(
        self,
        environments: Sequence[SimulatorEnv],
        *,
        backend: str = "reference",
        workers: int | None = None,
    ) -> None:
        if not environments:
            raise ValueError("at least one environment is required")
        hashes = {env.engine.ruleset.content_hash for env in environments}
        if len(hashes) != 1:
            raise ValueError("all vector environments must use the same ruleset")
        if backend not in {"reference", "process", "packed-process"}:
            raise ValueError("backend must be 'reference', 'process', or 'packed-process'")
        if workers is not None and (type(workers) is not int or workers <= 0):
            raise ValueError("workers must be a positive integer when provided")
        self.environments = tuple(environments)
        self.backend = backend
        self.workers = min(workers or len(self.environments), len(self.environments))
        self._executor: ProcessPoolExecutor | None = None
        if backend in {"process", "packed-process"}:
            # Do not fork after a CUDA/ROCm context has been initialized in the
            # parent.  Forkserver keeps workers isolated from accelerator state;
            # spawn is the portable fallback on platforms without forkserver.
            process_context = None
            for method in ("forkserver", "spawn", "fork"):
                try:
                    process_context = multiprocessing.get_context(method)
                    break
                except ValueError:
                    continue
            executor_kwargs = {"mp_context": process_context} if process_context else {}
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                **executor_kwargs,
            )

    @classmethod
    def create(
        cls,
        count: int,
        *,
        backend: str = "reference",
        workers: int | None = None,
        **environment_kwargs: object,
    ) -> "VectorSimulatorEnv":
        if count <= 0:
            raise ValueError("count must be positive")
        return cls(
            tuple(SimulatorEnv(**environment_kwargs) for _ in range(count)),
            backend=backend,
            workers=workers,
        )

    def reset(self, seeds: Sequence[int]) -> tuple[tuple[PolicyObservationV1, PolicyObservationV1], ...]:
        if len(seeds) != len(self.environments):
            raise ValueError("one seed is required per environment")
        return tuple(env.reset(seed=seed) for env, seed in zip(self.environments, seeds, strict=True))

    def reset_v2(self, seeds: Sequence[int]) -> tuple[tuple["PolicyObservationV2", "PolicyObservationV2"], ...]:
        """Reset all lanes and return the sanitized public V2 observations."""

        if len(seeds) != len(self.environments):
            raise ValueError("one seed is required per environment")
        return tuple(
            env.reset_v2(seed=seed)
            for env, seed in zip(self.environments, seeds, strict=True)
        )

    def step(self, actions: Sequence[Sequence[PolicyAction | SimAction | None]]) -> tuple[EnvStep, ...]:
        if len(actions) != len(self.environments):
            raise ValueError("one two-player action sequence is required per environment")
        if self.backend == "reference":
            return tuple(env.step(row) for env, row in zip(self.environments, actions, strict=True))
        if self.backend == "process":
            return self._step_process(actions)  # type: ignore[return-value]
        return self._step_packed_process(actions)  # type: ignore[return-value]

    def step_v2(
        self,
        actions: Sequence[Sequence[PolicyAction | SimAction | None]],
    ) -> tuple[EnvStepV2, ...]:
        """Advance all lanes and return only sanitized public V2 observations."""

        if len(actions) != len(self.environments):
            raise ValueError("one two-player action sequence is required per environment")
        if self.backend == "reference":
            return tuple(
                env.step_v2(row)
                for env, row in zip(self.environments, actions, strict=True)
            )
        if self.backend == "process":
            return self._step_process(actions, v2=True)  # type: ignore[return-value]
        return self._step_packed_process(actions, v2=True)  # type: ignore[return-value]

    def _step_process(
        self,
        actions: Sequence[Sequence[PolicyAction | SimAction | None]],
        *,
        v2: bool = False,
    ) -> tuple[EnvStep | EnvStepV2, ...]:
        executor = self._executor
        if executor is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("process backend is not initialized")
        payloads = []
        for environment, row in zip(self.environments, actions, strict=True):
            state = environment._require_state()
            reward = environment.reward_config
            payloads.append(
                (
                    state.ruleset_id,
                    environment.decision_interval_ticks * environment.engine.ruleset.tick_us,
                    reward.version,
                    reward.tower_damage_weight,
                    reward.crown_weight,
                    reward.win_weight,
                    environment.expose_privileged_info,
                    environment.engine.validate_every_tick,
                    # Event history is retained by the parent lane. The
                    # worker needs the event sequence counter, but only its
                    # current-step events need to cross the return boundary;
                    # sending the accumulated log would make every step more
                    # expensive as a match grows.
                    state.to_primitive(include_events=False),
                    tuple(row),
                )
            )
        results = executor.map(_parallel_env_step_worker, payloads)
        output: list[EnvStep | EnvStepV2] = []
        for environment, result in zip(self.environments, results, strict=True):
            raw_state, rewards, terminated, truncated, info = result
            previous_events = list(environment._require_state().events)
            restored_state = battle_state_from_primitive(raw_state)
            restored_state.events = previous_events + restored_state.events
            environment.state = restored_state
            # Observation memory belongs to the parent lane and is deliberately
            # not serialized through the worker. This keeps the public policy
            # boundary identical to the reference backend.
            if v2:
                output.append(
                    EnvStepV2(
                        observations=environment.observe_v2(),
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                )
            else:
                output.append(
                    EnvStep(
                        observations=environment.observe(),
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                )
        return tuple(output)

    def _step_packed_process(
        self,
        actions: Sequence[Sequence[PolicyAction | SimAction | None]],
        *,
        v2: bool = False,
    ) -> tuple[EnvStep | EnvStepV2, ...]:
        executor = self._executor
        if executor is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("packed process backend is not initialized")
        payloads = []
        for environment, row in zip(self.environments, actions, strict=True):
            state = environment._require_state()
            reward = environment.reward_config
            payloads.append(
                (
                    state.ruleset_id,
                    environment.decision_interval_ticks * environment.engine.ruleset.tick_us,
                    reward.version,
                    reward.tower_damage_weight,
                    reward.crown_weight,
                    reward.win_weight,
                    environment.expose_privileged_info,
                    environment.engine.validate_every_tick,
                    pack_state(state).to_bytes(),
                    tuple(row),
                )
            )
        results = executor.map(_packed_parallel_env_step_worker, payloads)
        output: list[EnvStep | EnvStepV2] = []
        for environment, result in zip(self.environments, results, strict=True):
            packed_state, rewards, terminated, truncated, info = result
            environment.state = unpack_state(packed_state)
            if v2:
                output.append(
                    EnvStepV2(
                        observations=environment.observe_v2(),
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                )
            else:
                output.append(
                    EnvStep(
                        observations=environment.observe(),
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                )
        return tuple(output)

    def close(self) -> None:
        """Stop process workers; safe to call for the reference backend."""

        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "VectorSimulatorEnv":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies
        try:
            self.close()
        except Exception:
            pass
