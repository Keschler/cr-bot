"""Exact-state checkpoint diagnosis for recurrent-policy regressions.

The comparator follows one deterministic reference trajectory (the known-good
checkpoint), records every pre-action state, and probes every checkpoint on
that same state and public-observation history.  Candidate actions are then
executed in cloned environments with the same opponent action, so an action
difference is separated from the later state divergence it causes.

This is an investigation tool, not a gameplay controller.  The actor remains
the only learner action source; the strategic controller is logged as a
training/evaluation reference and is never executed for the learner.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .diagnostics import (
    DIAGNOSTIC_TRACE_SCHEMA_VERSION,
    action_equal,
    build_policy_diagnostics,
    state_snapshot,
    tower_damage,
)
from .evaluation_matrix import (
    EvaluationMatrixError,
    MatchSpec,
    _controller_action,
)
from .generalized import build_heldout_matrix_config
from .provenance import code_revision


COMPARE_SCHEMA_VERSION = 1
COMPARE_KIND = "recurrent_public_ppo_exact_state_checkpoint_diagnosis"


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(child) for child in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _safe(item())
        except (TypeError, ValueError):
            return None
    return str(value)


def _fingerprint_observation(observation: Any) -> str:
    digest = hashlib.sha256()
    for name in ("board", "global_vector", "entity_tokens", "entity_mask"):
        value = getattr(observation, name, None)
        if value is None:
            continue
        digest.update(memoryview(value).tobytes())
    for mode, card, placement in (observation.structured_action_masks(),):
        for value in (mode, card, placement):
            digest.update(memoryview(value).tobytes())
    return f"sha256:{digest.hexdigest()}"


def _compact_action(action: Any) -> dict[str, Any]:
    from .diagnostics import _action_descriptor

    return _safe(_action_descriptor(action))


def _state_difference(before: Mapping[str, Any], after: Mapping[str, Any], *, target_player: int) -> dict[str, Any]:
    """Summarize the immediate consequence of one candidate action."""

    before_units = {int(row["uid"]): row for row in before.get("units", ()) if isinstance(row, Mapping) and type(row.get("uid")) is int}
    after_units = {int(row["uid"]): row for row in after.get("units", ()) if isinstance(row, Mapping) and type(row.get("uid")) is int}
    added = sorted(set(after_units) - set(before_units))
    removed = sorted(set(before_units) - set(after_units))
    moved: list[dict[str, Any]] = []
    for uid in sorted(set(before_units) & set(after_units)):
        old = before_units[uid]
        new = after_units[uid]
        if (old.get("x_mtile"), old.get("y_mtile"), old.get("hp")) != (new.get("x_mtile"), new.get("y_mtile"), new.get("hp")):
            moved.append(
                {
                    "uid": uid,
                    "card_id": new.get("card_id"),
                    "owner": new.get("owner"),
                    "before": {key: old.get(key) for key in ("x_mtile", "y_mtile", "hp")},
                    "after": {key: new.get(key) for key in ("x_mtile", "y_mtile", "hp")},
                }
            )
    return {
        "tower_damage_to_opponent": tower_damage(before, after, player=1 - target_player),
        "tower_damage_to_self": tower_damage(before, after, player=target_player),
        "units_added": added,
        "units_removed": removed,
        "units_moved_or_damaged": moved[:32],
        "major_consequence": bool(
            tower_damage(before, after, player=target_player)
            or tower_damage(before, after, player=1 - target_player)
            or removed
        ),
    }


def _relative_state_difference(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    target_player: int,
) -> dict[str, Any]:
    """Describe what the candidate branch changed relative to the good branch.

    ``_state_difference`` includes ordinary physics and the opponent's action.
    This second view is the causal comparison needed for regression reports:
    both branches start from one state and receive the same opponent action,
    so extra tower damage or branch-only units are attributable to the
    learner action up to the one-step counterfactual boundary.
    """

    reference_units = {
        int(row["uid"]): row
        for row in reference.get("units", ())
        if isinstance(row, Mapping) and type(row.get("uid")) is int
    }
    candidate_units = {
        int(row["uid"]): row
        for row in candidate.get("units", ())
        if isinstance(row, Mapping) and type(row.get("uid")) is int
    }
    added = sorted(set(candidate_units) - set(reference_units))
    removed = sorted(set(reference_units) - set(candidate_units))
    changed: list[dict[str, Any]] = []
    for uid in sorted(set(reference_units) & set(candidate_units)):
        old = reference_units[uid]
        new = candidate_units[uid]
        old_fields = (old.get("x_mtile"), old.get("y_mtile"), old.get("hp"))
        new_fields = (new.get("x_mtile"), new.get("y_mtile"), new.get("hp"))
        if old_fields != new_fields:
            changed.append(
                {
                    "uid": uid,
                    "card_id": new.get("card_id"),
                    "owner": new.get("owner"),
                    "reference": {
                        "x_mtile": old.get("x_mtile"),
                        "y_mtile": old.get("y_mtile"),
                        "hp": old.get("hp"),
                    },
                    "candidate": {
                        "x_mtile": new.get("x_mtile"),
                        "y_mtile": new.get("y_mtile"),
                        "hp": new.get("hp"),
                    },
                }
            )

    def hp_delta(player: int) -> int:
        reference_rows = reference.get("tower_hp", {}).get(f"player_{player}", {})
        candidate_rows = candidate.get("tower_hp", {}).get(f"player_{player}", {})
        delta = 0
        for role, reference_row in reference_rows.items():
            if not isinstance(reference_row, Mapping):
                continue
            candidate_row = candidate_rows.get(role)
            if not isinstance(candidate_row, Mapping):
                continue
            old_hp = reference_row.get("hp")
            new_hp = candidate_row.get("hp")
            if type(old_hp) is int and type(new_hp) is int:
                # Positive means the candidate branch has the lower tower HP.
                delta += max(0, int(old_hp) - int(new_hp))
        return delta

    extra_self_damage = hp_delta(target_player)
    extra_opponent_damage = hp_delta(1 - target_player)
    return {
        "additional_tower_damage_to_self": extra_self_damage,
        "additional_tower_damage_to_opponent": extra_opponent_damage,
        "units_added": added,
        "units_removed": removed,
        "units_changed": changed[:32],
        "major_consequence": bool(
            extra_self_damage
            or extra_opponent_damage
            or added
            or removed
            or changed
        ),
    }


def _categories(
    before: Mapping[str, Any],
    *,
    good_action: Mapping[str, Any],
    candidate_action: Mapping[str, Any],
    teacher_action: Mapping[str, Any] | None,
    consequence: Mapping[str, Any],
    causal_difference: Mapping[str, Any] | None = None,
) -> list[str]:
    categories: set[str] = set()
    if candidate_action != good_action:
        if candidate_action.get("mode") != good_action.get("mode"):
            categories.add("mode-head-regression")
            if candidate_action.get("mode") == "PLAY" and good_action.get("mode") == "WAIT":
                categories.add("action-too-early")
            elif candidate_action.get("mode") == "WAIT" and good_action.get("mode") == "PLAY":
                categories.add("action-too-late")
        elif candidate_action.get("card_slot") != good_action.get("card_slot"):
            categories.add("card-selection-head-regression")
        if (
            candidate_action.get("mode") == "PLAY"
            and good_action.get("mode") == "PLAY"
            and candidate_action.get("world_cell") != good_action.get("world_cell")
        ):
            categories.add("placement-head-regression")
    if teacher_action is not None and candidate_action != teacher_action:
        categories.add("teacher-disagreement")
    is_divergence = candidate_action != good_action
    units = [
        unit
        for unit in before.get("units", ())
        if isinstance(unit, Mapping) and unit.get("owner") == 1
    ]
    threat = [unit for unit in units if isinstance(unit.get("y_mtile"), int) and unit["y_mtile"] >= 15000]
    if threat and is_divergence:
        if candidate_action.get("mode") == "WAIT":
            categories.add("missed-defense")
        elif candidate_action.get("mode") == "PLAY" and good_action.get("mode") == "WAIT":
            categories.add("threat-response")
        if any("air" in str(unit.get("card_id", "")).casefold() for unit in threat):
            categories.add("air-threat-response")
        else:
            categories.add("ground-threat-response")
    if (
        is_divergence
        and candidate_action.get("mode") == "PLAY"
        and good_action.get("mode") == "WAIT"
        and not threat
        and causal_difference is not None
        and not causal_difference.get("major_consequence")
    ):
        categories.add("potential-unnecessary-action")
    if causal_difference and causal_difference.get("additional_tower_damage_to_self", 0):
        categories.add("bad-immediate-consequence")
    return sorted(categories)


class ExactStateComparator:
    """Probe named checkpoints on one shared deterministic state stream."""

    def __init__(
        self,
        checkpoints: Mapping[str, str | Path],
        *,
        device: str | None = "auto",
        lookahead_steps: int = 8,
    ) -> None:
        try:
            from .prototype import load_prototype_checkpoint, _privileged_features
            from .collector import _batch_observations, _decode_actions
            from .learner import RecurrentRolloutState
        except ImportError as error:  # pragma: no cover
            raise EvaluationMatrixError("the recurrent diagnostic stack is unavailable") from error
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise EvaluationMatrixError("exact-state diagnosis requires PyTorch") from error
        self._load_checkpoint = load_prototype_checkpoint
        self._privileged_features = _privileged_features
        self._batch_observations = _batch_observations
        self._decode_actions = _decode_actions
        self._rollout_state_type = RecurrentRolloutState
        self.torch = torch
        self.device = device
        if type(lookahead_steps) is not int or lookahead_steps < 0:
            raise EvaluationMatrixError("lookahead_steps must be a non-negative integer")
        self.lookahead_steps = lookahead_steps
        self.checkpoints: dict[str, dict[str, Any]] = {}
        for label, path in checkpoints.items():
            learner, config, metadata = load_prototype_checkpoint(path, device=device)
            learner.policy.eval()
            learner.critic.eval()
            self.checkpoints[label] = {
                "path": str(path),
                "learner": learner,
                "config": config,
                "metadata": metadata,
                "hidden": learner.initial_rollout_state(1),
            }
        if self.checkpoints and all(
            item["learner"].device.type == "cpu" for item in self.checkpoints.values()
        ):
            # Exact-state diagnosis is intentionally single-lane.  One host
            # thread avoids repeatedly starting large BLAS teams for the
            # small batch-1 actor forwards.
            self.torch.set_num_threads(1)
            try:
                self.torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        if "good" not in self.checkpoints or "bad" not in self.checkpoints:
            raise EvaluationMatrixError("checkpoints must include good and bad labels")
        configs = {item["config"].ruleset_id for item in self.checkpoints.values()}
        if len(configs) != 1:
            raise EvaluationMatrixError("checkpoints use different rulesets")
        try:
            from ..engine import BattleEngine
            from ..env import RewardConfig, SimulatorEnv
            from ..ruleset import load_ruleset
        except ImportError:  # pragma: no cover
            from simulator.engine import BattleEngine
            from simulator.env import RewardConfig, SimulatorEnv
            from simulator.ruleset import load_ruleset
        self.BattleEngine = BattleEngine
        self.RewardConfig = RewardConfig
        self.SimulatorEnv = SimulatorEnv
        self.ruleset = load_ruleset(next(iter(configs)))

    def _environment(self, spec: MatchSpec) -> Any:
        config = self.checkpoints["good"]["config"]
        try:
            opponent_deck = tuple(self.ruleset.resolve_card_id(card) for card in spec.opponent_deck.cards)
            player_deck = tuple(self.ruleset.resolve_card_id(card) for card in spec.player_deck)
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationMatrixError(f"cannot canonicalize diagnosis deck: {error}") from error
        decks = (player_deck, opponent_deck) if spec.target_player == 0 else (opponent_deck, player_deck)
        environment = self.SimulatorEnv(
            engine=self.BattleEngine(self.ruleset, validate_every_tick=False),
            decision_interval_us=int(config.decision_interval_us),
            reward=self.RewardConfig.terminal_outcome(),
            expose_privileged_info=True,
            include_authoritative_state=False,
        )
        environment.reset_v2(seed=spec.seed, decks=decks, shuffle_decks=spec.shuffle_decks)
        return environment

    def _policy_step(
        self,
        label: str,
        environment: Any,
        observation: Any,
        hidden: Any,
        *,
        target_player: int,
        reset: bool,
    ) -> tuple[Any, Any, Any]:
        """Run one deterministic actor step without mutating checkpoint state."""

        item = self.checkpoints[label]
        learner = item["learner"]
        raster, global_features, entities, entity_mask, masks = self._batch_observations(
            [observation], device=learner.device, inference=True
        )
        reset_mask = self.torch.tensor([reset], dtype=self.torch.bool, device=learner.device)
        privileged = None
        if learner.uses_privileged_critic:
            privileged = self.torch.as_tensor(
                [self._privileged_features(environment, target_player)],
                dtype=self.torch.float32,
                device=learner.device,
            ).unsqueeze(1)
        with self.torch.inference_mode():
            step = learner.rollout_step(
                hidden,
                raster[:, 0],
                global_features[:, 0],
                entities[:, 0],
                entity_mask[:, 0],
                masks,
                reset_mask=reset_mask,
                privileged_features=privileged,
                deterministic=True,
                include_beliefs=False,
                inference=True,
                fast_sampling=False,
            )
        return self._decode_actions(step.actions)[0], masks, step

    def _probe(
        self,
        label: str,
        environment: Any,
        observation: Any,
        *,
        reset: bool,
        target_player: int,
        reference_action: Any | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        item = self.checkpoints[label]
        learner = item["learner"]
        action, masks, step = self._policy_step(
            label,
            environment,
            observation,
            item["hidden"],
            target_player=target_player,
            reset=reset,
        )
        with self.torch.inference_mode():
            diag = build_policy_diagnostics(
                learner.policy,
                step.output,
                masks,
                step.actions,
                critic_value=step.values,
                old_log_prob=step.log_probs,
                lane=0,
                time=0,
                reference_action=reference_action,
            )
        item["hidden"] = step.next_state
        return action, masks, diag

    def _restore_clone(
        self,
        clone: Any,
        state: Mapping[str, Any] | Any,
        *,
        memories: Any | None = None,
    ) -> None:
        # ``SimulatorEnv.load_state`` intentionally returns the legacy V1
        # observation for backwards compatibility.  Some valid V2-only card
        # aliases cannot be represented by that old projection, so restore the
        # authoritative state directly for a physics-only counterfactual.
        try:
            from ..state import battle_state_from_primitive
            from ..observation import ObservationMemory
        except ImportError:  # pragma: no cover
            from simulator.state import battle_state_from_primitive
            from simulator.observation import ObservationMemory
        if isinstance(state, Mapping):
            clone.state = battle_state_from_primitive(deepcopy(dict(state)))
        else:
            clone.state = deepcopy(state)
        if memories is None:
            clone._memories = (ObservationMemory(0), ObservationMemory(1))
        else:
            clone._memories = deepcopy(memories)
            # The serialized branch state deliberately omits the full event
            # log. Rebind the copied public-memory bookkeeping to the branch's
            # event list so the next observation consumes only newly emitted
            # branch events instead of resetting the public history.
            for memory in clone._memories:
                if hasattr(memory, "_processed_event_list"):
                    memory._processed_event_list = clone.state.events
                if hasattr(memory, "_processed_event_mutation_revision"):
                    memory._processed_event_mutation_revision = getattr(
                        clone.state.events,
                        "mutation_revision",
                        None,
                    )
        clone._persistent_observation_cache = None

    def _candidate_step(
        self,
        spec: MatchSpec,
        state: Mapping[str, Any],
        target_action: Any,
        opponent_action: Any,
        clone: Any,
        memories: Any | None = None,
    ) -> tuple[dict[str, Any], tuple[tuple[float, float], bool, bool, Mapping[str, Any]]]:
        self._restore_clone(clone, state, memories=memories)
        actions: list[Any] = [None, None]
        actions[spec.target_player] = target_action
        actions[1 - spec.target_player] = opponent_action
        # Physics-only stepping avoids constructing two public observations per
        # candidate.  The real reference trajectory still uses step_v2 below.
        result = clone._step_core(actions)
        return state_snapshot(clone.state) or {}, result

    def _counterfactual_lookahead(
        self,
        label: str,
        clone: Any,
        hidden: Any,
        spec: MatchSpec,
        opponent: Any,
    ) -> dict[str, Any]:
        """Follow a branch with its actor and opponent controllers.

        This is intentionally a diagnostic continuation, not an alternative
        gameplay policy. It starts after the candidate's one-step action and
        reports later tower damage/termination attributable to that branch.
        The initial state and current opponent action are still identical for
        every checkpoint; subsequent observations are allowed to react to the
        branch so a bad placement can be observed causing a later failure.
        """

        start = state_snapshot(clone.state) or {}
        current_hidden = hidden.detach()
        total_reward = 0.0
        rows: list[dict[str, Any]] = []
        try:
            branch_opponent = deepcopy(opponent)
        except Exception:  # pragma: no cover - custom controllers may not copy
            branch_opponent = opponent
        for offset in range(self.lookahead_steps):
            state = getattr(clone, "state", None)
            if state is None or bool(getattr(state, "terminal", False)):
                break
            observations = clone.observe_v2()
            action, _masks, policy_step = self._policy_step(
                label,
                clone,
                observations[spec.target_player],
                current_hidden,
                target_player=spec.target_player,
                reset=False,
            )
            current_hidden = policy_step.next_state
            opponent_action = _controller_action(
                branch_opponent,
                clone.engine,
                clone.state,
                1 - spec.target_player,
                public_observation=observations[1 - spec.target_player],
            )
            before = state_snapshot(clone.state) or {}
            actions: list[Any] = [None, None]
            actions[spec.target_player] = action
            actions[1 - spec.target_player] = opponent_action
            rewards, terminated, truncated, info = clone._step_core(actions)
            after = state_snapshot(clone.state) or {}
            reward = float(rewards[spec.target_player])
            total_reward += reward
            rows.append(
                {
                    "offset": offset + 1,
                    "action": _compact_action(action),
                    "opponent_action": _compact_action(opponent_action),
                    "reward": reward,
                    "tower_damage_to_self": tower_damage(
                        before,
                        after,
                        player=spec.target_player,
                    ),
                    "tower_damage_to_opponent": tower_damage(
                        before,
                        after,
                        player=1 - spec.target_player,
                    ),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "winner": info.get("winner") if isinstance(info, Mapping) else None,
                    "terminal_reason": (
                        info.get("terminal_reason") if isinstance(info, Mapping) else None
                    ),
                }
            )
        final = state_snapshot(clone.state) or {}
        return {
            "kind": "candidate-policy-continuation",
            "steps_requested": self.lookahead_steps,
            "steps_completed": len(rows),
            "approximate": True,
            "initial_state": start,
            "final_state": final,
            "tower_damage_to_self": tower_damage(
                start,
                final,
                player=spec.target_player,
            ),
            "tower_damage_to_opponent": tower_damage(
                start,
                final,
                player=1 - spec.target_player,
            ),
            "total_reward": total_reward,
            "terminated": bool(getattr(clone.state, "terminal", False)),
            "rows": rows,
        }

    def _reset_hidden(self) -> None:
        for item in self.checkpoints.values():
            item["hidden"] = item["learner"].initial_rollout_state(1)

    def compare_match(self, spec: MatchSpec) -> dict[str, Any]:
        try:
            from .public_counter import StrategicCounterController
        except ImportError as error:  # pragma: no cover
            raise EvaluationMatrixError("strategic teacher is unavailable") from error
        environment = self._environment(spec)
        observations = environment.observe_v2()
        candidate_environments = {
            label: self._environment(spec) for label in self.checkpoints
        }
        opponent = spec.strategy.build(spec.seed)
        teacher = StrategicCounterController()
        self._reset_hidden()
        cap = spec.max_decisions
        if cap is None:
            cap = max(1, math.ceil((self.ruleset.match.regulation_us + self.ruleset.match.overtime_us) / int(self.checkpoints["good"]["config"].decision_interval_us)))
        rows: list[dict[str, Any]] = []
        total_return: dict[str, float] = {label: 0.0 for label in self.checkpoints}
        reset = True
        for decision in range(cap):
            state = environment.state
            if state is None:
                raise EvaluationMatrixError("diagnosis environment lost its state")
            # Events are useful in the emitted trace, but are not required to
            # restore deterministic physics and make the serialized copy grow
            # with the entire match history.  The canonical state fields still
            # include RNG state, so replay remains exact.
            raw_state = state.to_primitive(include_events=False)
            before = state_snapshot(state)
            if before is None:
                raise EvaluationMatrixError("could not snapshot diagnosis state")
            source_memories = deepcopy(getattr(environment, "_memories", None))
            actor_rows: dict[str, dict[str, Any]] = {}
            candidate_actions: dict[str, Any] = {}
            branch_hidden: dict[str, Any] = {}
            for label in self.checkpoints:
                action, _masks, diag = self._probe(
                    label,
                    environment,
                    observations[spec.target_player],
                    reset=reset,
                    target_player=spec.target_player,
                    reference_action=(candidate_actions.get("good") if label != "good" else None),
                )
                candidate_actions[label] = action
                branch_hidden[label] = self.checkpoints[label]["hidden"].detach()
                actor_rows[label] = {"action": _compact_action(action), **_safe(diag)}
            teacher_action = teacher.choose_action(observations[spec.target_player], player=spec.target_player)
            opponent_action = _controller_action(
                opponent,
                environment.engine,
                environment.state,
                1 - spec.target_player,
                public_observation=observations[1 - spec.target_player],
            )
            consequences: dict[str, Any] = {}
            for label, action in candidate_actions.items():
                after, step = self._candidate_step(
                    spec,
                    raw_state,
                    action,
                    opponent_action,
                    candidate_environments[label],
                    memories=source_memories,
                )
                difference = _state_difference(before, after, target_player=spec.target_player)
                rewards, terminated, truncated, info = step
                consequences[label] = {
                    "state_after": after,
                    "state_difference": difference,
                    "reward": float(rewards[spec.target_player]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "events": _safe(info.get("events", ())) if isinstance(info, Mapping) else [],
                }
                total_return[label] += float(rewards[spec.target_player])
            good_after = consequences["good"]["state_after"]
            good_action = actor_rows["good"]["action"]
            teacher_descriptor = _compact_action(teacher_action)
            for label, actor in actor_rows.items():
                actor["action_differs_from_good"] = label != "good" and actor["action"] != good_action
                actor["teacher_agreement"] = actor["action"] == teacher_descriptor
                causal_difference = _relative_state_difference(
                    good_after,
                    consequences[label]["state_after"],
                    target_player=spec.target_player,
                )
                consequences[label]["state_difference_vs_good"] = causal_difference
                actor["suspicious_categories"] = _categories(
                    before,
                    good_action=good_action,
                    candidate_action=actor["action"],
                    teacher_action=teacher_descriptor,
                    consequence=consequences[label]["state_difference"],
                    causal_difference=causal_difference,
                )
            divergent_labels = [
                label
                for label in self.checkpoints
                if label != "good" and actor_rows[label]["action_differs_from_good"]
            ]
            for label in self.checkpoints:
                consequences[label]["follow_on_consequence"] = None
            if self.lookahead_steps and divergent_labels:
                lookahead_labels = ["good", *divergent_labels]
                for label in lookahead_labels:
                    consequences[label]["follow_on_consequence"] = self._counterfactual_lookahead(
                        label,
                        candidate_environments[label],
                        branch_hidden[label],
                        spec,
                        opponent,
                    )
                good_follow_on = consequences["good"]["follow_on_consequence"]
                if isinstance(good_follow_on, Mapping):
                    good_final = good_follow_on.get("final_state")
                    if isinstance(good_final, Mapping):
                        for label in divergent_labels:
                            candidate_follow_on = consequences[label].get(
                                "follow_on_consequence"
                            )
                            if not isinstance(candidate_follow_on, Mapping):
                                continue
                            candidate_final = candidate_follow_on.get("final_state")
                            if isinstance(candidate_final, Mapping):
                                candidate_follow_on["state_difference_vs_good"] = (
                                    _relative_state_difference(
                                        good_final,
                                        candidate_final,
                                        target_player=spec.target_player,
                                    )
                                )
            rows.append(
                {
                    "decision": decision,
                    "game_time_us": before.get("elapsed_us"),
                    "state_hash": state.state_hash(),
                    "public_observation_fingerprint": _fingerprint_observation(observations[spec.target_player]),
                    "same_state_for_all_checkpoints": True,
                    "state_before": before,
                    "hand": before["players"][spec.target_player]["hand"],
                    "elixir_milli": before["players"][spec.target_player]["elixir_milli"],
                    "tower_hp": before["tower_hp"],
                    "strategic_teacher_action": teacher_descriptor,
                    "opponent_action": _compact_action(opponent_action),
                    "action_comparison": {
                        label: {
                            "action": actor["action"],
                            "chosen_action_probability": actor.get("chosen_action_probability"),
                            "reference_action_probability": actor.get("reference_action_probability"),
                            "reference_action_log_probability": actor.get("reference_action_log_probability"),
                        }
                        for label, actor in actor_rows.items()
                    },
                    "checkpoints": actor_rows,
                    "consequences": consequences,
                }
            )
            good_action = candidate_actions["good"]
            actions: list[Any] = [None, None]
            actions[spec.target_player] = good_action
            actions[1 - spec.target_player] = opponent_action
            step = environment.step_v2(actions)
            observations = step.observations
            reset = False
            if step.terminated or step.truncated:
                break
        outcome = "truncated"
        winner = None
        terminal_reason = "evaluation_cap"
        if rows:
            final_state = environment.state
            if final_state is not None and final_state.terminal:
                winner = final_state.winner
                terminal_reason = final_state.terminal_reason
                outcome = "win" if winner == spec.target_player else "loss" if winner == 1 - spec.target_player else "draw"
        category_counts: Counter[str] = Counter()
        divergence_counts: Counter[str] = Counter()
        suspects: list[dict[str, Any]] = []
        for row in rows:
            for label, actor in row["checkpoints"].items():
                divergent = label != "good" and actor.get("action_differs_from_good")
                if divergent:
                    for category in actor.get("suspicious_categories", ()):
                        category_counts[category] += 1
                    divergence_counts[label] += 1
                    consequence = row["consequences"][label]["state_difference"]
                    causal_difference = row["consequences"][label].get(
                        "state_difference_vs_good",
                        {},
                    )
                    score = (
                        1000.0 * float(
                            causal_difference.get("additional_tower_damage_to_self", 0)
                        )
                        + 10.0 * len(causal_difference.get("units_removed", ()))
                        + 1.0 * len(actor.get("suspicious_categories", ()))
                    )
                    follow_on = row["consequences"][label].get(
                        "follow_on_consequence"
                    )
                    follow_on_difference = (
                        follow_on.get("state_difference_vs_good", {})
                        if isinstance(follow_on, Mapping)
                        else {}
                    )
                    score += 1000.0 * float(
                        follow_on_difference.get(
                            "additional_tower_damage_to_self", 0
                        )
                    )
                    good_actor = row["checkpoints"]["good"]
                    suspects.append(
                        {
                            "score": score,
                            "decision": row["decision"],
                            "checkpoint": label,
                            "action": actor["action"],
                            "good_action": row["checkpoints"]["good"]["action"],
                            "chosen_action_probability": actor.get(
                                "chosen_action_probability"
                            ),
                            "good_action_probability": good_actor.get(
                                "chosen_action_probability"
                            ),
                            "reference_action_probability": actor.get(
                                "reference_action_probability"
                            ),
                            "top_alternative_actions": actor.get(
                                "top_alternative_actions", []
                            ),
                            "teacher_action": row["strategic_teacher_action"],
                            "categories": actor.get("suspicious_categories", ()),
                            "consequence": consequence,
                            "causal_difference_vs_good": causal_difference,
                            "follow_on_consequence": follow_on,
                        }
                    )
        suspects.sort(key=lambda item: (float(item["score"]), -int(item["decision"])), reverse=True)
        return {
            "cell_id": spec.cell_id,
            "deck_id": spec.opponent_deck.deck_id,
            "strategy_id": spec.strategy.strategy_id,
            "seed": spec.seed,
            "outcome_on_good_trajectory": outcome,
            "winner_on_good_trajectory": winner,
            "terminal_reason": terminal_reason,
            "decisions": len(rows),
            "returns_under_one_step_counterfactuals": total_return,
            "action_divergences_from_good": dict(divergence_counts),
            "failure_categories": dict(sorted(category_counts.items())),
            "top_suspects": suspects[:12],
            "trace": rows,
        }

    def compare(self, specs: Sequence[MatchSpec]) -> dict[str, Any]:
        matches = [self.compare_match(spec) for spec in specs]
        aggregate: Counter[str] = Counter()
        divergences: Counter[str] = Counter()
        for match in matches:
            aggregate.update(match["failure_categories"])
            divergences.update(match["action_divergences_from_good"])
        root_cause = self._root_cause(matches)
        return {
            "kind": COMPARE_KIND,
            "schema_version": COMPARE_SCHEMA_VERSION,
            "diagnostic_trace_schema_version": DIAGNOSTIC_TRACE_SCHEMA_VERSION,
            "code_revision": code_revision(),
            "same_states": True,
            "state_source": "known-good-checkpoint-trajectory",
            "actor_controls_actions": True,
            "checkpoints": {
                label: {"path": item["path"], "checkpoint_revision": item["metadata"].get("code_revision")}
                for label, item in self.checkpoints.items()
            },
            "matches": matches,
            "aggregate_failure_categories": dict(sorted(aggregate.items())),
            "aggregate_action_divergences": dict(sorted(divergences.items())),
            "root_cause_assessment": root_cause,
        }

    @staticmethod
    def _root_cause(matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        bad_rows = sum(
            int(match.get("action_divergences_from_good", {}).get("bad", 0))
            for match in matches
        )
        recovery_rows = sum(
            int(match.get("action_divergences_from_good", {}).get("recovery", 0))
            for match in matches
        )
        per_checkpoint: dict[str, dict[str, Any]] = {}
        for label in ("bad", "recovery"):
            counts: Counter[str] = Counter()
            extra_self_damage = 0
            follow_on_self_damage = 0
            follow_on_additional_self_damage = 0
            follow_on_cases = 0
            divergences_for_label = 0
            for match in matches:
                divergences_for_label += int(
                    match.get("action_divergences_from_good", {}).get(label, 0)
                )
                for row in match.get("trace", ()):
                    actor = row.get("checkpoints", {}).get(label, {})
                    if not actor.get("action_differs_from_good"):
                        continue
                    counts.update(actor.get("suspicious_categories", ()))
                    consequence = row.get("consequences", {}).get(label, {})
                    causal = consequence.get("state_difference_vs_good", {})
                    extra_self_damage += int(
                        causal.get("additional_tower_damage_to_self", 0) or 0
                    )
                    follow_on = consequence.get("follow_on_consequence")
                    if isinstance(follow_on, Mapping):
                        follow_on_cases += 1
                        follow_on_self_damage += int(
                            follow_on.get("tower_damage_to_self", 0) or 0
                        )
                        follow_on_difference = follow_on.get(
                            "state_difference_vs_good", {}
                        )
                        if isinstance(follow_on_difference, Mapping):
                            follow_on_additional_self_damage += int(
                                follow_on_difference.get(
                                    "additional_tower_damage_to_self", 0
                                )
                                or 0
                            )
            per_checkpoint[label] = {
                "divergent_decisions": divergences_for_label,
                "category_counts": dict(sorted(counts.items())),
                "additional_immediate_self_tower_damage": extra_self_damage,
                "follow_on_cases": follow_on_cases,
                "follow_on_self_tower_damage": follow_on_self_damage,
                "follow_on_additional_self_tower_damage": follow_on_additional_self_damage,
            }
        bad_counts = Counter(per_checkpoint["bad"]["category_counts"])
        mode = int(bad_counts.get("mode-head-regression", 0))
        card = int(bad_counts.get("card-selection-head-regression", 0))
        placement = int(bad_counts.get("placement-head-regression", 0))
        teacher = int(bad_counts.get("teacher-disagreement", 0))
        if bad_rows == 0:
            statement = "No bad-vs-good action divergence was observed on the selected exact states."
        elif mode > max(card, placement):
            statement = (
                "The bad checkpoint is dominated by WAIT/PLAY mode-head changes; "
                "pair the trace with the update where mode entropy, teacher "
                "disagreement, and mode gradients moved."
            )
        elif card > placement:
            statement = (
                "The bad checkpoint is dominated by card-selection changes; "
                "inspect card-head gradients and teacher labels before changing PPO hyperparameters."
            )
        elif placement > card:
            statement = (
                "The bad checkpoint is dominated by placement changes; inspect "
                "placement-head logits, masks, and gradients before changing PPO hyperparameters."
            )
        else:
            statement = "The observed regression is mixed; the trace is insufficient to attribute it to one head without additional held-out states."
        return {
            "status": "evidence-ranked, not proof" if bad_rows else "no-divergence",
            "statement": statement,
            "bad_divergent_decisions": bad_rows,
            "recovery_divergent_decisions": recovery_rows,
            "card_head_regression_count": card,
            "placement_head_regression_count": placement,
            "teacher_disagreement_count": teacher,
            "per_checkpoint": per_checkpoint,
            "required_next_check": "validate the same category and consequence improvements on the held-out seeds after the smallest fix",
        }


def compare_checkpoints(
    good: str | Path,
    bad: str | Path,
    recovery: str | Path | None = None,
    *,
    archetypes: Sequence[str],
    strategies: Sequence[str],
    seeds: Sequence[int],
    player_deck: Sequence[str] | None = None,
    max_decisions: int | None = None,
    device: str | None = "auto",
    lookahead_steps: int = 8,
) -> dict[str, Any]:
    """Build the exact same matrix for good/bad/recovery probes."""

    config = build_heldout_matrix_config(
        good,
        player_deck=player_deck,
        archetypes=archetypes,
        strategies=strategies,
        seeds=seeds,
        policy_mode="actor",
        max_decisions=max_decisions,
        device=device,
        batch_size=1,
        include_match_results=False,
        shuffle_decks=False,
    )
    checkpoints: dict[str, str | Path] = {"good": good, "bad": bad}
    if recovery is not None:
        checkpoints["recovery"] = recovery
    comparator = ExactStateComparator(
        checkpoints,
        device=device,
        lookahead_steps=lookahead_steps,
    )
    specs = [
        MatchSpec(
            checkpoint=good,
            opponent_deck=deck,
            strategy=strategy,
            seed=seed,
            policy_mode="actor",
            target_player=config.target_player,
            max_decisions=config.max_decisions,
            device=device,
            shuffle_decks=config.shuffle_decks,
            player_deck=config.player_deck,
        )
        for deck in config.opponent_decks
        for strategy in config.strategies
        for seed in config.seeds
    ]
    result = comparator.compare(specs)
    result["matrix"] = {
        "archetypes": list(archetypes),
        "strategies": list(strategies),
        "seeds": list(seeds),
        "max_decisions": max_decisions,
        "shuffle_decks": config.shuffle_decks,
        "player_deck": list(config.player_deck),
    }
    return _safe(result)


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated value must not be empty")
    return values


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in _csv(value))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rl.diagnose")
    parser.add_argument("--good", type=Path, required=True)
    parser.add_argument("--bad", type=Path, required=True)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--archetypes", default="aggressive-pressure,defensive-cycle,beatdown,air-beatdown,siege-bait,random-legal")
    parser.add_argument("--strategies", default="deterministic-cycle")
    parser.add_argument("--seeds", default="10000")
    parser.add_argument("--player-deck")
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--lookahead-decisions",
        type=int,
        default=8,
        help="candidate-policy continuation steps for later tower consequences (0 disables)",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    player_deck = None if args.player_deck is None else _csv(args.player_deck)
    report = compare_checkpoints(
        args.good,
        args.bad,
        args.recovery,
        archetypes=_csv(args.archetypes),
        strategies=_csv(args.strategies),
        seeds=_ints(args.seeds),
        player_deck=player_deck,
        max_decisions=args.max_decisions,
        device=args.device,
        lookahead_steps=args.lookahead_decisions,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is None:
        print(encoded)
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["COMPARE_KIND", "COMPARE_SCHEMA_VERSION", "compare_checkpoints", "main"]
