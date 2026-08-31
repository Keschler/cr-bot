"""Persistent CPU rollout workers for recurrent PPO.

The ordinary vector environment is intentionally state-transparent: every
decision returns a canonical state to the parent so callers can inspect and
replay it.  That contract is useful for tests and diagnostics, but it makes a
trainer pay a process-boundary round trip on every decision.  This module
provides a trainer-only alternative.  Workers own their environments and run
the existing public rollout collector locally; only one trajectory crosses
the boundary per PPO update.

The actor still receives exactly the same public V2 tensors, the critic still
receives the same optional privileged features, and the same simulator/action
callbacks are used for the supported default opponent.  This is a scheduling
optimization, not a second simulator or a second policy architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import multiprocessing
from time import perf_counter
from typing import Any, Mapping, Sequence


class RolloutFarmError(RuntimeError):
    """Raised when a persistent rollout worker cannot complete a command."""


RolloutOpponentSpec = tuple[str, int]


def _attach_shared_memory(name: str) -> Any:
    """Attach without tracker ownership on Python 3.13 and older hosts."""

    from multiprocessing.shared_memory import SharedMemory

    # ``track`` was added in Python 3.13.  Python 3.12 workers otherwise
    # register the parent's segment with their resource tracker and may unlink
    # it when the worker exits, breaking the next rollout segment.
    try:
        supports_track = "track" in inspect.signature(SharedMemory).parameters
    except (TypeError, ValueError):  # pragma: no cover - unusual extension type
        supports_track = True
    if supports_track:
        return SharedMemory(name=name, create=False, track=False)

    handle = SharedMemory(name=name, create=False)
    try:
        from multiprocessing import resource_tracker

        resource_tracker.unregister(handle._name, "shared_memory")
    except (AttributeError, KeyError):  # pragma: no cover - runtime variation
        pass
    return handle


class _SharedRolloutStorage:
    """Fixed-shape shared arrays for one complete PPO segment."""

    def __init__(self, handles: Mapping[str, Any], arrays: Mapping[str, Any]) -> None:
        self.handles = dict(handles)
        self.arrays = dict(arrays)

    @classmethod
    def create(
        cls,
        config: Any,
        *,
        expert_guidance: bool = False,
    ) -> "_SharedRolloutStorage":
        from multiprocessing.shared_memory import SharedMemory
        import numpy as np

        batch = int(config.envs)
        horizon = int(config.horizon)
        model_dim = int(config.gru_hidden_dim)
        layers = int(config.gru_layers)
        fields: dict[str, tuple[tuple[int, ...], Any]] = {
            "raster": ((batch, horizon, 21, 32, 18), np.float32),
            "global_features": ((batch, horizon, 768), np.float32),
            "entities": ((batch, horizon, 128, 32), np.float32),
            "entity_mask": ((batch, horizon, 128), np.bool_),
            "reset_mask": ((batch, horizon), np.bool_),
            "hidden_states": ((batch, horizon, layers, model_dim), np.float32),
            "initial_hidden": ((layers, batch, model_dim), np.float32),
            "mode_mask": ((batch, horizon, 2), np.bool_),
            "card_mask": ((batch, horizon, 4), np.bool_),
            "placement_mask": ((batch, horizon, 4, 32, 18), np.bool_),
            "action_mode": ((batch, horizon), np.int64),
            "action_card_slot": ((batch, horizon), np.int64),
            "action_placement": ((batch, horizon, 2), np.int64),
            "rewards": ((batch, horizon), np.float32),
            "terminated": ((batch, horizon), np.bool_),
            "truncated": ((batch, horizon), np.bool_),
            "old_log_probs": ((batch, horizon), np.float32),
            "values": ((batch, horizon), np.float32),
            # A rollout may contain nonterminal truncations.  Keep explicit
            # per-transition successor values in the transport rather than
            # accidentally substituting the reset episode's first value.
            "next_values": ((batch, horizon), np.float32),
            "next_values_present": ((batch,), np.bool_),
            "bootstrap_values": ((batch,), np.float32),
        }
        if bool(config.use_privileged_critic):
            fields["privileged_features"] = ((batch, horizon, 23), np.float32)
        if bool(config.collect_belief_targets):
            fields["belief_enemy_elixir"] = ((batch, horizon), np.float32)
            fields["belief_enemy_hand"] = ((batch, horizon, 128), np.float32)
            fields["belief_enemy_next_card"] = ((batch, horizon), np.int64)
        if expert_guidance:
            # Expert guidance is label-only by default: the worker computes a
            # training target from authoritative state, while the actor's
            # sampled action remains the environment action.  Keep the labels
            # in the same bounded shared-memory segment as the trajectory.
            fields["behavior_cloning_weights"] = ((batch, horizon), np.float32)
            fields["behavior_cloning_action_mode"] = ((batch, horizon), np.int64)
            fields["behavior_cloning_action_card_slot"] = ((batch, horizon), np.int64)
            fields["behavior_cloning_action_placement"] = (
                (batch, horizon, 2),
                np.int64,
            )

        handles: dict[str, Any] = {}
        arrays: dict[str, Any] = {}
        try:
            for name, (shape, dtype) in fields.items():
                size = int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)
                handle = SharedMemory(create=True, size=size)
                handles[name] = handle
                arrays[name] = np.ndarray(shape, dtype=dtype, buffer=handle.buf)
        except BaseException:
            for handle in handles.values():
                handle.close()
                handle.unlink()
            raise
        return cls(handles, arrays)

    def descriptor(self) -> dict[str, tuple[str, tuple[int, ...], str]]:
        return {
            name: (handle.name, tuple(array.shape), array.dtype.str)
            for name, handle in self.handles.items()
            for array in (self.arrays[name],)
        }

    @classmethod
    def attach(
        cls,
        descriptor: Mapping[str, Sequence[Any]],
    ) -> "_SharedRolloutStorage":
        from multiprocessing.shared_memory import SharedMemory
        import numpy as np

        handles: dict[str, Any] = {}
        arrays: dict[str, Any] = {}
        try:
            for name, raw_spec in descriptor.items():
                if not isinstance(raw_spec, Sequence) or len(raw_spec) != 3:
                    raise RolloutFarmError("invalid shared rollout field descriptor")
                shared_name, raw_shape, dtype = raw_spec
                shape = tuple(int(value) for value in raw_shape)
                handle = _attach_shared_memory(str(shared_name))
                handles[name] = handle
                arrays[name] = np.ndarray(shape, dtype=np.dtype(str(dtype)), buffer=handle.buf)
        except BaseException:
            for handle in handles.values():
                handle.close()
            raise
        return cls(handles, arrays)

    def close(self, *, unlink: bool = False) -> None:
        for handle in self.handles.values():
            try:
                handle.close()
            except OSError:
                pass
        if unlink:
            for handle in self.handles.values():
                try:
                    handle.unlink()
                except FileNotFoundError:
                    pass


def _worker_receive(connection: Any) -> Any:
    try:
        response = connection.recv()
    except (EOFError, OSError) as error:
        raise RolloutFarmError("rollout worker disconnected") from error
    if isinstance(response, tuple) and response and response[0] == "error":
        detail = response[2] if len(response) > 2 else "unknown worker error"
        raise RolloutFarmError(f"rollout worker failed: {detail}")
    return response


def _copy_batch_to_shared(
    storage: _SharedRolloutStorage,
    batch: Any,
    *,
    start: int,
    end: int,
) -> None:
    """Copy one worker's CPU batch into its preallocated lane slice."""

    import numpy as np

    def copy_field(name: str, value: Any, *, axis: int = 0) -> None:
        if value is None:
            raise RolloutFarmError(f"rollout batch omitted shared field {name!r}")
        source = value.detach().contiguous().numpy()
        target = storage.arrays[name]
        if axis == 0:
            target = target[start:end]
        elif axis == 1:
            target = target[:, start:end]
        else:  # pragma: no cover - all shared fields use axis 0 or 1
            raise RolloutFarmError(f"unsupported shared field axis {axis}")
        if source.shape != target.shape:
            raise RolloutFarmError(
                f"rollout field {name!r} has shape {source.shape}, "
                f"expected {target.shape}"
            )
        np.copyto(target, source, casting="no")

    trajectory = batch.trajectory
    sequence = trajectory.sequence
    copy_field("raster", sequence.raster)
    copy_field("global_features", sequence.global_features)
    copy_field("entities", sequence.entities)
    copy_field("entity_mask", sequence.entity_mask)
    copy_field("reset_mask", sequence.reset_mask)
    copy_field("hidden_states", sequence.hidden_states)
    copy_field("initial_hidden", sequence.initial_hidden, axis=1)

    masks = trajectory.action_masks
    copy_field("mode_mask", masks.mode)
    copy_field("card_mask", masks.card)
    copy_field("placement_mask", masks.placement)
    actions = trajectory.actions
    copy_field("action_mode", actions.mode)
    copy_field("action_card_slot", actions.card_slot)
    copy_field("action_placement", actions.placement)
    copy_field("rewards", trajectory.rewards)
    copy_field("terminated", trajectory.terminated)
    copy_field("truncated", trajectory.truncated)
    copy_field("old_log_probs", trajectory.old_log_probs)
    copy_field("values", trajectory.values)
    if batch.next_values is not None:
        copy_field("next_values", batch.next_values)
        storage.arrays["next_values_present"][start:end] = True
        bootstrap_values = batch.next_values[:, -1]
    else:
        storage.arrays["next_values"][start:end] = 0.0
        storage.arrays["next_values_present"][start:end] = False
        if batch.bootstrap_values is None:
            raise RolloutFarmError(
                "rollout batch omitted both next_values and bootstrap_values"
            )
        bootstrap_values = batch.bootstrap_values
    copy_field("bootstrap_values", bootstrap_values)

    if batch.privileged_features is not None:
        copy_field("privileged_features", batch.privileged_features)
    if batch.belief_targets is not None:
        copy_field("belief_enemy_elixir", batch.belief_targets.enemy_elixir)
        copy_field("belief_enemy_hand", batch.belief_targets.enemy_hand)
        copy_field("belief_enemy_next_card", batch.belief_targets.enemy_next_card)
    if "behavior_cloning_weights" in storage.arrays:
        if batch.behavior_cloning_weights is None or batch.behavior_cloning_actions is None:
            raise RolloutFarmError(
                "rollout batch omitted expert labels in an expert-guided farm"
            )
        copy_field("behavior_cloning_weights", batch.behavior_cloning_weights)
        copy_field(
            "behavior_cloning_action_mode",
            batch.behavior_cloning_actions.mode,
        )
        copy_field(
            "behavior_cloning_action_card_slot",
            batch.behavior_cloning_actions.card_slot,
        )
        copy_field(
            "behavior_cloning_action_placement",
            batch.behavior_cloning_actions.placement,
        )
    elif (
        batch.behavior_cloning_weights is not None
        or batch.behavior_cloning_actions is not None
    ):
        raise RolloutFarmError(
            "rollout farm was not configured to transport expert labels"
        )


def _rollout_worker(connection: Any) -> None:
    """Serve one contiguous group of rollout lanes in a long-lived process."""

    storages: tuple[_SharedRolloutStorage, ...] = ()
    try:
        command = connection.recv()
        if not isinstance(command, tuple) or len(command) != 9 or command[0] != "init":
            raise RolloutFarmError("rollout worker expected an init command")
        (
            _,
            raw_config,
            lane_indices,
            lane_decks,
            opponent_specs,
            policy_state,
            critic_state,
            storage_descriptor,
            expert_teacher,
        ) = command
        if not isinstance(raw_config, Mapping):
            raise RolloutFarmError("rollout worker received an invalid config")
        if not isinstance(lane_indices, Sequence) or not lane_indices:
            raise RolloutFarmError("rollout worker received no lane indices")
        if not isinstance(lane_decks, Sequence) or len(lane_decks) != len(lane_indices):
            raise RolloutFarmError("rollout worker received mismatched lane decks")
        if opponent_specs is not None and (
            not isinstance(opponent_specs, Sequence)
            or len(opponent_specs) != len(lane_indices)
        ):
            raise RolloutFarmError("rollout worker received mismatched opponent specs")
        if isinstance(storage_descriptor, Mapping):
            storage_descriptors = (storage_descriptor,)
        elif isinstance(storage_descriptor, Sequence) and storage_descriptor:
            storage_descriptors = tuple(storage_descriptor)
        else:
            raise RolloutFarmError("rollout worker received invalid shared storage")
        if any(not isinstance(item, Mapping) for item in storage_descriptors):
            raise RolloutFarmError("rollout worker received invalid shared storage descriptors")
        if expert_teacher is not None and expert_teacher not in {
            "public-counter",
            "strategic-counter",
            "deterministic-counter",
        }:
            raise RolloutFarmError("rollout worker received an invalid expert teacher")

        storages = tuple(
            _SharedRolloutStorage.attach(item) for item in storage_descriptors
        )

        # Keep one thread per rollout process.  The environment is Python
        # control flow and the small batch policy does not benefit from an
        # intra-op pool multiplied by every worker.
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # A process may have initialized the inter-op pool while importing
            # torch.  In that case the intra-op cap above is still effective.
            pass

        from .prototype import (
            PrototypeConfig,
            _make_collector,
            _make_environment,
            _model_and_learner,
            _opponent_callback,
            _simulator_modules,
        )

        values = dict(raw_config)
        values.update(
            {
                "envs": len(lane_indices),
                "device": "cpu",
                "env_backend": "reference",
                "updates": 1,
                "overlap_rollouts": False,
                "compile_policy": False,
            }
        )
        worker_config = PrototypeConfig.from_mapping(values)
        learner = _model_and_learner(worker_config)
        learner.policy.load_state_dict(policy_state, strict=True)
        learner.critic.load_state_dict(critic_state, strict=True)
        learner.policy.eval()
        learner.critic.eval()

        load_ruleset = _simulator_modules()[7]
        ruleset = load_ruleset(worker_config.ruleset_id)
        environments = [
            _make_environment(
                worker_config,
                ruleset,
                int(lane),
                player_deck=tuple(deck_pair[0]),
                opponent_deck=tuple(deck_pair[1]),
            )
            for lane, deck_pair in zip(lane_indices, lane_decks, strict=True)
        ]
        if opponent_specs is None:
            opponent_action = _opponent_callback()
        else:
            from .opponent_pool import make_opponent_controller

            controllers = {}
            for environment, raw_spec in zip(
                environments,
                opponent_specs,
                strict=True,
            ):
                if (
                    not isinstance(raw_spec, Sequence)
                    or isinstance(raw_spec, (str, bytes))
                    or len(raw_spec) != 2
                    or not isinstance(raw_spec[0], str)
                    or type(raw_spec[1]) is not int
                ):
                    raise RolloutFarmError(
                        "rollout worker received an invalid opponent spec"
                    )
                controllers[id(environment)] = make_opponent_controller(
                    raw_spec[0],
                    seed=raw_spec[1],
                )

            def opponent_action(
                environment: Any,
                _public_observation: Any,
                player: int,
            ) -> Any:
                controller = controllers.get(id(environment))
                if controller is None:
                    raise RolloutFarmError(
                        "rollout worker lost the controller for an environment"
                    )
                return controller.choose_action(environment.engine, environment.state, player)

        expert_action = None
        if expert_teacher == "public-counter":
            from .public_counter import public_counter_action

            expert_action = public_counter_action
        elif expert_teacher == "strategic-counter":
            from .public_counter import strategic_counter_action

            expert_action = strategic_counter_action
        elif expert_teacher == "deterministic-counter":
            from .expert import deterministic_counter_action

            expert_action = deterministic_counter_action

        episode_counts = [0] * len(environments)
        connection.send(("ready",))

        while True:
            command = connection.recv()
            if not isinstance(command, tuple) or not command:
                raise RolloutFarmError("rollout worker received an invalid command")
            kind = command[0]
            if kind == "close":
                connection.send(("closed",))
                return
            if kind == "weights" and len(command) == 4:
                _, policy_state, critic_state, seed = command
                learner.policy.load_state_dict(policy_state, strict=True)
                learner.critic.load_state_dict(critic_state, strict=True)
                learner.policy.eval()
                learner.critic.eval()
                if type(seed) is not int:
                    raise RolloutFarmError("rollout worker seed must be an integer")
                torch.manual_seed(seed)
                connection.send(("weights-loaded",))
                continue
            if kind != "collect" or len(command) not in (2, 3):
                raise RolloutFarmError(f"rollout worker received unknown command {kind!r}")
            raw_collect_config = command[1]
            if not isinstance(raw_collect_config, Mapping):
                raise RolloutFarmError("rollout worker received an invalid collect config")
            buffer_index = 0 if len(command) == 2 else command[2]
            if type(buffer_index) is not int or not 0 <= buffer_index < len(storages):
                raise RolloutFarmError("rollout worker received an invalid buffer index")
            collect_values = dict(raw_collect_config)
            collect_values.update(
                {
                    "envs": len(environments),
                    "device": "cpu",
                    "env_backend": "reference",
                    "updates": 1,
                    "overlap_rollouts": False,
                    "compile_policy": False,
                }
            )
            collect_config = PrototypeConfig.from_mapping(collect_values)
            collector = _make_collector(
                learner,
                collect_config,
                lane_decks=tuple(tuple(pair) for pair in lane_decks),
                opponent_action=opponent_action,
                expert_action=expert_action,
                batch_step=None,
                lane_offset=int(lane_indices[0]),
                actor_only_observations=True,
            )
            collect_started = perf_counter()
            result = collector.collect(
                environments,
                episode_counts=episode_counts,
            )
            episode_counts = list(result.episode_counts)
            _copy_batch_to_shared(
                storages[buffer_index],
                result.learner_batch,
                start=int(lane_indices[0]),
                end=int(lane_indices[-1]) + 1,
            )
            connection.send(
                (
                    "result",
                    result.stats.as_dict(),
                    tuple(episode_counts),
                    perf_counter() - collect_started,
                )
            )
    except BaseException as error:
        try:
            connection.send(("error", type(error).__name__, str(error)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        for storage in storages:
            storage.close()
        try:
            connection.close()
        except OSError:
            pass


def _cpu_state_dict(module: Any) -> dict[str, Any]:
    """Copy a module state to CPU before it is sent to rollout workers."""

    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in module.state_dict().items()
    }


def _cat(values: Sequence[Any], *, dim: int) -> Any:
    import torch

    if not values:
        raise RolloutFarmError("cannot concatenate an empty tensor sequence")
    return torch.cat(tuple(values), dim=dim)


def _cat_optional(values: Sequence[Any | None], *, dim: int) -> Any | None:
    if not values:
        return None
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise RolloutFarmError("rollout workers returned inconsistent optional fields")
    return _cat(present, dim=dim)


def _combine_batches(batches: Sequence[Any]) -> Any:
    """Concatenate worker learner batches without changing their contents."""

    if not batches:
        raise RolloutFarmError("rollout farm returned no learner batches")
    import torch

    from .learner import BeliefTargets, LearnerBatch
    from .trajectory import ActionBatch, ActionMasks, RecurrentSequence, TrajectoryBatch

    sequence = RecurrentSequence(
        raster=_cat([item.trajectory.sequence.raster for item in batches], dim=0),
        global_features=_cat(
            [item.trajectory.sequence.global_features for item in batches], dim=0
        ),
        entities=_cat([item.trajectory.sequence.entities for item in batches], dim=0),
        entity_mask=_cat(
            [item.trajectory.sequence.entity_mask for item in batches], dim=0
        ),
        reset_mask=_cat(
            [item.trajectory.sequence.reset_mask for item in batches], dim=0
        ),
        hidden_states=_cat_optional(
            [item.trajectory.sequence.hidden_states for item in batches], dim=0
        ),
        initial_hidden=_cat_optional(
            [item.trajectory.sequence.initial_hidden for item in batches], dim=1
        ),
    )
    trajectory = TrajectoryBatch(
        sequence=sequence,
        action_masks=ActionMasks(
            mode=_cat([item.trajectory.action_masks.mode for item in batches], dim=0),
            card=_cat([item.trajectory.action_masks.card for item in batches], dim=0),
            placement=_cat(
                [item.trajectory.action_masks.placement for item in batches],
                dim=0,
            ),
        ),
        actions=ActionBatch(
            mode=_cat([item.trajectory.actions.mode for item in batches], dim=0),
            card_slot=_cat(
                [item.trajectory.actions.card_slot for item in batches], dim=0
            ),
            placement=_cat(
                [item.trajectory.actions.placement for item in batches], dim=0
            ),
        ),
        rewards=_cat([item.trajectory.rewards for item in batches], dim=0),
        terminated=_cat([item.trajectory.terminated for item in batches], dim=0),
        truncated=_cat([item.trajectory.truncated for item in batches], dim=0),
        old_log_probs=_cat([item.trajectory.old_log_probs for item in batches], dim=0),
        values=_cat_optional([item.trajectory.values for item in batches], dim=0),
        advantages=_cat_optional(
            [item.trajectory.advantages for item in batches], dim=0
        ),
        returns=_cat_optional([item.trajectory.returns for item in batches], dim=0),
    )

    belief_batches = [item.belief_targets for item in batches]
    belief_targets = None
    if any(item is not None for item in belief_batches):
        if any(item is None for item in belief_batches):
            raise RolloutFarmError("rollout workers returned inconsistent belief targets")
        belief_targets = BeliefTargets(
            enemy_elixir=_cat_optional(
                [item.enemy_elixir for item in belief_batches if item is not None],
                dim=0,
            ),
            enemy_hand=_cat_optional(
                [item.enemy_hand for item in belief_batches if item is not None],
                dim=0,
            ),
            enemy_next_card=_cat_optional(
                [item.enemy_next_card for item in belief_batches if item is not None],
                dim=0,
            ),
        )
    next_value_batches = [item.next_values for item in batches]
    if any(value is not None for value in next_value_batches):
        explicit_next_values = []
        for item in batches:
            if item.next_values is not None:
                explicit_next_values.append(item.next_values)
                continue
            if item.bootstrap_values is None:
                raise RolloutFarmError(
                    "rollout worker omitted successor values at a mixed batch boundary"
                )
            values = item.trajectory.values
            fallback = torch.zeros_like(values)
            if values.shape[1] > 1:
                fallback[:, :-1] = values[:, 1:]
            fallback[:, -1] = item.bootstrap_values
            explicit_next_values.append(fallback)
        combined_next_values = _cat(explicit_next_values, dim=0)
        combined_bootstrap_values = None
    else:
        combined_next_values = None
        combined_bootstrap_values = (
            _cat([item.bootstrap_values for item in batches], dim=0)
            if all(item.bootstrap_values is not None for item in batches)
            else None
        )
    behavior_cloning_actions = None
    if any(item.behavior_cloning_actions is not None for item in batches):
        if not all(item.behavior_cloning_actions is not None for item in batches):
            raise RolloutFarmError("rollout workers returned inconsistent expert actions")
        behavior_cloning_actions = ActionBatch(
            mode=_cat(
                [item.behavior_cloning_actions.mode for item in batches], dim=0
            ),
            card_slot=_cat(
                [item.behavior_cloning_actions.card_slot for item in batches],
                dim=0,
            ),
            placement=_cat(
                [item.behavior_cloning_actions.placement for item in batches],
                dim=0,
            ),
        )
    return LearnerBatch(
        trajectory=trajectory,
        privileged_features=_cat_optional(
            [item.privileged_features for item in batches], dim=0
        ),
        belief_targets=belief_targets,
        next_values=combined_next_values,
        bootstrap_values=combined_bootstrap_values,
        behavior_cloning_log_probs=_cat_optional(
            [item.behavior_cloning_log_probs for item in batches], dim=0
        ),
        behavior_cloning_actions=behavior_cloning_actions,
        behavior_cloning_weights=_cat_optional(
            [item.behavior_cloning_weights for item in batches], dim=0
        ),
    )


def _batch_from_shared(storage: _SharedRolloutStorage, config: Any) -> Any:
    """Build a learner batch from shared arrays without IPC tensor copies."""

    import torch

    from .learner import BeliefTargets, LearnerBatch
    from .trajectory import ActionBatch, ActionMasks, RecurrentSequence, TrajectoryBatch

    def tensor(name: str) -> Any:
        return torch.from_numpy(storage.arrays[name])

    sequence = RecurrentSequence(
        raster=tensor("raster"),
        global_features=tensor("global_features"),
        entities=tensor("entities"),
        entity_mask=tensor("entity_mask"),
        reset_mask=tensor("reset_mask"),
        hidden_states=tensor("hidden_states"),
        initial_hidden=tensor("initial_hidden"),
    )
    trajectory = TrajectoryBatch(
        sequence=sequence,
        action_masks=ActionMasks(
            mode=tensor("mode_mask"),
            card=tensor("card_mask"),
            placement=tensor("placement_mask"),
        ),
        actions=ActionBatch(
            mode=tensor("action_mode"),
            card_slot=tensor("action_card_slot"),
            placement=tensor("action_placement"),
        ),
        rewards=tensor("rewards"),
        terminated=tensor("terminated"),
        truncated=tensor("truncated"),
        old_log_probs=tensor("old_log_probs"),
        values=tensor("values"),
    )
    belief_targets = None
    if bool(config.collect_belief_targets):
        belief_targets = BeliefTargets(
            enemy_elixir=tensor("belief_enemy_elixir"),
            enemy_hand=tensor("belief_enemy_hand"),
            enemy_next_card=tensor("belief_enemy_next_card"),
        )
    next_values_present = storage.arrays["next_values_present"]
    if bool(next_values_present.any()):
        values = tensor("values")
        fallback = torch.zeros_like(values)
        if values.shape[1] > 1:
            fallback[:, :-1] = values[:, 1:]
        fallback[:, -1] = tensor("bootstrap_values")
        next_values = torch.where(
            torch.from_numpy(next_values_present).reshape(-1, 1),
            tensor("next_values"),
            fallback,
        )
        bootstrap_values = None
    else:
        next_values = None
        bootstrap_values = tensor("bootstrap_values")
    behavior_cloning_actions = None
    behavior_cloning_weights = None
    if "behavior_cloning_weights" in storage.arrays:
        behavior_cloning_weights = tensor("behavior_cloning_weights")
        behavior_cloning_actions = ActionBatch(
            mode=tensor("behavior_cloning_action_mode"),
            card_slot=tensor("behavior_cloning_action_card_slot"),
            placement=tensor("behavior_cloning_action_placement"),
        )
    return LearnerBatch(
        trajectory=trajectory,
        privileged_features=(
            tensor("privileged_features")
            if bool(config.use_privileged_critic)
            else None
        ),
        belief_targets=belief_targets,
        next_values=next_values,
        bootstrap_values=bootstrap_values,
        behavior_cloning_actions=behavior_cloning_actions,
        behavior_cloning_weights=behavior_cloning_weights,
    )


@dataclass(frozen=True, slots=True)
class RolloutFarmResult:
    """A combined PPO batch plus worker-side episode bookkeeping."""

    learner_batch: Any
    stats: Any
    episode_counts: tuple[int, ...]
    startup_seconds: float
    worker_collect_seconds: tuple[float, ...]
    collect_wall_seconds: float


class RolloutFarm:
    """Keep simulator/collector workers alive across PPO updates."""

    def __init__(
        self,
        config: Any,
        learner: Any,
        lane_decks: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
        opponent_specs: Sequence[RolloutOpponentSpec] | None = None,
        *,
        expert_teacher: str | None = None,
        double_buffer: bool = False,
    ) -> None:
        if len(lane_decks) != int(config.envs):
            raise RolloutFarmError("lane_decks must match the configured environment count")
        self.config = config
        self.lane_decks = tuple(lane_decks)
        if opponent_specs is not None:
            if len(opponent_specs) != int(config.envs):
                raise RolloutFarmError(
                    "opponent_specs must match the configured environment count"
                )
            normalized_specs: list[RolloutOpponentSpec] = []
            for index, raw_spec in enumerate(opponent_specs):
                if (
                    not isinstance(raw_spec, Sequence)
                    or isinstance(raw_spec, (str, bytes))
                    or len(raw_spec) != 2
                    or not isinstance(raw_spec[0], str)
                    or type(raw_spec[1]) is not int
                ):
                    raise RolloutFarmError(
                        f"opponent_specs[{index}] must be a (strategy, integer seed) pair"
                    )
                normalized_specs.append((raw_spec[0], raw_spec[1]))
            self.opponent_specs: tuple[RolloutOpponentSpec, ...] = tuple(normalized_specs)
        else:
            self.opponent_specs = None
        if expert_teacher is not None and expert_teacher not in {
            "public-counter",
            "strategic-counter",
            "deterministic-counter",
        }:
            raise RolloutFarmError("expert_teacher must name a supported built-in teacher")
        self.expert_teacher = expert_teacher
        self.connections: list[Any] = []
        self.processes: list[Any] = []
        self.groups: list[tuple[int, ...]] = []
        self.closed = False
        if type(double_buffer) is not bool:
            raise RolloutFarmError("double_buffer must be boolean")
        self.double_buffer = double_buffer
        self.storages = tuple(
            _SharedRolloutStorage.create(
                config,
                expert_guidance=expert_teacher is not None,
            )
            for _ in range(2 if double_buffer else 1)
        )
        # Keep the singular attribute for diagnostics and existing callers.
        self.storage = self.storages[0]
        started = perf_counter()
        try:
            self._start(learner)
        except BaseException:
            for storage in self.storages:
                storage.close(unlink=True)
            raise
        self.startup_seconds = perf_counter() - started

    def _start(self, learner: Any) -> None:
        context = None
        # CPU workers can safely fork after the learner is constructed and
        # inherit the already-imported model/runtime through copy-on-write.
        # CUDA contexts must never be forked; keep forkserver/spawn first for
        # accelerator-backed parent learners.
        methods = (
            ("fork", "forkserver", "spawn")
            if getattr(getattr(learner, "device", None), "type", None) == "cpu"
            else ("forkserver", "spawn", "fork")
        )
        for method in methods:
            try:
                context = multiprocessing.get_context(method)
                break
            except ValueError:
                continue
        if context is None:  # pragma: no cover - supported hosts provide one
            raise RolloutFarmError("no multiprocessing context is available")

        worker_count = min(
            int(self.config.env_workers or self.config.envs),
            int(self.config.envs),
        )
        chunk_size = (int(self.config.envs) + worker_count - 1) // worker_count
        self.groups = [
            tuple(range(start, min(start + chunk_size, int(self.config.envs))))
            for start in range(0, int(self.config.envs), chunk_size)
        ]
        policy_state = _cpu_state_dict(learner.policy)
        critic_state = _cpu_state_dict(learner.critic)
        for group in self.groups:
            parent, child = context.Pipe(duplex=True)
            process = context.Process(target=_rollout_worker, args=(child,))
            process.daemon = True
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        try:
            for group, connection in zip(self.groups, self.connections, strict=True):
                connection.send(
                    (
                        "init",
                        self.config.as_dict(),
                        group,
                        tuple(self.lane_decks[lane] for lane in group),
                        None
                        if self.opponent_specs is None
                        else tuple(self.opponent_specs[lane] for lane in group),
                        policy_state,
                        critic_state,
                        tuple(storage.descriptor() for storage in self.storages),
                        self.expert_teacher,
                    )
                )
            for connection in self.connections:
                response = _worker_receive(connection)
                if not isinstance(response, tuple) or not response or response[0] != "ready":
                    raise RolloutFarmError(f"rollout worker failed to start: {response!r}")
        except BaseException:
            self.close()
            raise

    def sync_weights(self, learner: Any, *, seed: int) -> None:
        """Publish the post-update actor and critic to every worker."""

        policy_state = _cpu_state_dict(learner.policy)
        critic_state = _cpu_state_dict(learner.critic)
        for worker_index, connection in enumerate(self.connections):
            connection.send(
                (
                    "weights",
                    policy_state,
                    critic_state,
                    int(seed) + worker_index,
                )
            )
        for connection in self.connections:
            response = _worker_receive(connection)
            if not isinstance(response, tuple) or not response or response[0] != "weights-loaded":
                raise RolloutFarmError(f"rollout worker rejected weights: {response!r}")

    def collect(self, config: Any, *, buffer_index: int = 0) -> RolloutFarmResult:
        """Collect one segment in parallel into the shared trajectory."""

        if type(buffer_index) is not int or not 0 <= buffer_index < len(self.storages):
            raise RolloutFarmError("buffer_index is outside the rollout farm buffers")
        collect_started = perf_counter()
        for connection in self.connections:
            connection.send(("collect", config.as_dict(), buffer_index))
        group_stats: list[Mapping[str, object] | None] = [None] * len(self.groups)
        group_counts: list[tuple[int, ...] | None] = [None] * len(self.groups)
        worker_seconds: list[float | None] = [None] * len(self.groups)
        for index, connection in enumerate(self.connections):
            response = _worker_receive(connection)
            if not isinstance(response, tuple) or len(response) != 4 or response[0] != "result":
                raise RolloutFarmError(f"rollout worker returned an invalid result: {response!r}")
            _, group_stats[index], group_counts[index], worker_seconds[index] = response

        combined = _batch_from_shared(self.storages[buffer_index], config)
        stats = {
            "completed_matches": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "truncated_matches": 0,
            "episode_boundaries": 0,
            "match_outcomes": [],
        }
        episode_counts = [0] * int(self.config.envs)
        for index, (group, raw_stats, counts) in enumerate(zip(
            self.groups,
            group_stats,
            group_counts,
            strict=True,
        )):
            if raw_stats is None or counts is None:
                raise RolloutFarmError("rollout worker omitted statistics")
            if not isinstance(worker_seconds[index], (int, float)):
                raise RolloutFarmError("rollout worker omitted collection timing")
            for name in (
                "completed_matches",
                "wins",
                "draws",
                "losses",
                "truncated_matches",
            ):
                stats[name] = int(stats[name]) + int(raw_stats.get(name, 0))
            for raw_outcome in raw_stats.get("match_outcomes", ()):
                if not isinstance(raw_outcome, Mapping):
                    raise RolloutFarmError("rollout worker returned an invalid outcome")
                stats["match_outcomes"].append(
                    {
                        "lane": int(group[int(raw_outcome["lane"])]),
                        "outcome": str(raw_outcome["outcome"]),
                    }
                )
            for local_lane, global_lane in enumerate(group):
                episode_counts[global_lane] = int(counts[local_lane])
        stats["episode_boundaries"] = int(stats["completed_matches"]) + int(
            stats["truncated_matches"]
        )
        from .collector import RolloutStats, RolloutResult

        rollout_stats = RolloutStats(
            completed_matches=int(stats["completed_matches"]),
            wins=int(stats["wins"]),
            draws=int(stats["draws"]),
            losses=int(stats["losses"]),
            truncated_matches=int(stats["truncated_matches"]),
            match_outcomes=tuple(
                (int(item["lane"]), str(item["outcome"]))
                for item in stats["match_outcomes"]
            ),
        )
        return RolloutFarmResult(
            learner_batch=combined,
            stats=rollout_stats,
            episode_counts=tuple(episode_counts),
            startup_seconds=self.startup_seconds,
            worker_collect_seconds=tuple(
                float(value) for value in worker_seconds if value is not None
            ),
            collect_wall_seconds=perf_counter() - collect_started,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        connections, self.connections = self.connections, []
        processes, self.processes = self.processes, []
        for connection in connections:
            try:
                connection.send(("close",))
            except (BrokenPipeError, EOFError, OSError):
                continue
        for connection in connections:
            try:
                _worker_receive(connection)
            except RolloutFarmError:
                pass
            try:
                connection.close()
            except OSError:
                pass
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():  # pragma: no cover - defensive cleanup
                process.terminate()
                process.join(timeout=1.0)
        for storage in self.storages:
            storage.close(unlink=True)

    def __enter__(self) -> "RolloutFarm":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies
        try:
            self.close()
        except Exception:
            pass
