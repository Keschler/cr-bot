"""Public saving-plan hypotheses, independently certified as whole sequences.

The first WAIT is never assigned the value of later scripted actions.  A
sequence capture has its own nonterminal schema: factual prefix followed by
the observations and acknowledged actions actually produced by the branch.
This experimental low-level API is not wired into replay search or training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .frontier_eval import BranchSpec, action_key
from .hand_features import V4_LEARNER_DECK
from .tactical_data import PUBLIC_KEYS, array_digest, canonical, sha256_file, validate_target
from .tactical_plans import PlanDecision, PlanPublicState, PlanStatus, _bodies
from .simulator_teacher import TeacherTarget, nearest_legal_cell

VERSION = "v4-correction-sequence-v1"
_HAND = {card[1]: (card[0], card[2]) for card in V4_LEARNER_DECK}
_REGEN = .25 / 2.8  # Public fixed-rules normal-elixir lower bound per decision.


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SavingPlanConfig:
    max_decisions: int = 48
    deadline_decisions: int = 32
    allow_spirit_stall: bool = True
    minimum_elixir_per_decision: float = _REGEN
    decision_interval_us: int = 250_000

    def __post_init__(self) -> None:
        if (type(self.max_decisions) is not int or not 2 <= self.max_decisions <= 48
                or type(self.deadline_decisions) is not int
                or not 1 <= self.deadline_decisions < self.max_decisions):
            raise ValueError("saving plans need a bounded deadline before expiry")
        if (self.decision_interval_us != 250_000 or type(self.allow_spirit_stall) is not bool
                or not math.isfinite(self.minimum_elixir_per_decision)
                or not 0 < self.minimum_elixir_per_decision <= _REGEN):
            raise ValueError("saving plans require native cadence and conservative public regeneration")


@dataclass(frozen=True, slots=True)
class SavingPlanSpec:
    defender: str
    created_step: int
    config: SavingPlanConfig = SavingPlanConfig()

    def __post_init__(self) -> None:
        if self.defender not in ("musketeer", "fireball") or type(self.created_step) is not int or self.created_step < 0:
            raise ValueError("saving plans require a known defender and observation index")

    def identity(self) -> dict[str, Any]:
        value = dict(version=VERSION, defender=self.defender, created_step=self.created_step,
                     config=asdict(self.config))
        return dict(value, plan_id=_digest(value))


def _slot(state: PlanPublicState, name: str) -> int | None:
    # Card identity establishes membership; token fields 3/4 report affordability.
    return next((i for i, row in enumerate(state.hand_tokens)
                 if _HAND.get(int(round(float(row[0]) * 127)), (None,))[0] == name), None)


def propose_saving_plans(inputs: Mapping[str, Any], *, own_elixir: float, decision_step: int,
                         config: SavingPlanConfig | None = None) -> tuple[SavingPlanSpec, ...]:
    state = PlanPublicState.from_inputs(inputs, decision_step, own_elixir)
    enemies = _bodies(state, 1)
    if (not state.legal_wait or not any(body.is_air for body in enemies)
            or any(not body.is_air and body.y >= .70 for body in enemies)):
        return ()
    return tuple(SavingPlanSpec(name, decision_step, config or SavingPlanConfig())
                 for name in ("musketeer", "fireball")
                 if _slot(state, name) is not None and state.own_elixir < 4)


class SavingPlanController:
    """Wait for an in-hand defender, optionally stall within its deadline budget."""

    def __init__(self, proposal: SavingPlanSpec):
        self.proposal = proposal
        self._last: PlanDecision | None = None
        self._pending: PlanDecision | None = None
        self._terminal: PlanDecision | None = None
        self._threat = None
        self._stalled = False
        self.completed = False
        self.defender_plays = 0

    def _decision(self, state: PlanPublicState, status: PlanStatus, reason: str,
                  target: TeacherTarget | None = None) -> PlanDecision:
        result = PlanDecision(target, status, reason, self.proposal.identity()["plan_id"],
                              int(self._stalled), state.decision_step)
        self._last = result
        if result.active:
            self._pending = result
        else:
            self._terminal = result
        return result

    def step(self, state: PlanPublicState) -> PlanDecision:
        if self._terminal is not None:
            return self._terminal
        if self._last is not None and state.decision_step == self._last.decision_step:
            return self._last
        if self._pending is not None:
            return self._decision(state, PlanStatus.INTERRUPTED, "unacknowledged_action")
        expected = self.proposal.created_step if self._last is None else self._last.decision_step + 1
        if state.decision_step != expected:
            return self._decision(state, PlanStatus.INTERRUPTED, "observation_gap")
        elapsed = state.decision_step - self.proposal.created_step
        if self.completed:
            return self._decision(state, PlanStatus.COMPLETE, "defender_executed")
        if elapsed >= self.proposal.config.max_decisions:
            return self._decision(state, PlanStatus.INTERRUPTED, "plan_expired")
        enemies = _bodies(state, 1)
        if any(not body.is_air and body.y >= .70 for body in enemies):
            return self._decision(state, PlanStatus.INTERRUPTED, "urgent_ground_threat")
        slot = _slot(state, self.proposal.defender)
        if slot is None:
            return self._decision(state, PlanStatus.INTERRUPTED, "defender_missing")
        air = [body for body in enemies if body.is_air]
        if self._threat is not None:
            matches = [body for body in air if body.card_id == self._threat.card_id
                       and math.hypot(body.x - self._threat.x, body.y - self._threat.y) <= 2.5]
            if len(matches) != 1:
                return self._decision(state, PlanStatus.INTERRUPTED,
                                      "threat_gone" if not matches else "ambiguous_threat")
            threat = matches[0]
        else:
            threat = next(iter(sorted(air, key=lambda body: (-body.y, body.card_id, body.x))), None)
            if threat is not None and sum(body.card_id == threat.card_id and
                                         math.hypot(body.x - threat.x, body.y - threat.y) <= 2.5
                                         for body in air) != 1:
                return self._decision(state, PlanStatus.INTERRUPTED, "ambiguous_threat")
        self._threat = threat
        # The arm identity is explicitly WAIT then plan, even when already affordable.
        if elapsed == 0:
            return self._wait(state, "initial_wait")
        if threat is not None and state.own_elixir >= 4 and state.hand_tokens[slot, 4] > .5:
            row = int(threat.y * 32)
            col = int(threat.x * 18)
            if self.proposal.defender == "musketeer":
                row = max(17, min(30, row + 3))
                col += 2 if col < 9 else -2
            cell = nearest_legal_cell(row, col, state.legal_play[slot], max_radius=2)
            if cell is not None:
                return self._decision(state, PlanStatus.PLAY, "saved_defender",
                                      TeacherTarget(1, slot, *cell, 0))
        # Check legal affordable defense before handing an urgent scene back to actor.
        if any(body.y >= .70 for body in enemies):
            return self._decision(state, PlanStatus.INTERRUPTED, "urgent_threat")
        if elapsed >= self.proposal.config.deadline_decisions:
            return self._decision(state, PlanStatus.INTERRUPTED, "defense_deadline")
        if threat is not None and self.proposal.config.allow_spirit_stall and not self._stalled:
            spirit = _slot(state, "ice-spirit")
            remaining = self.proposal.config.deadline_decisions - elapsed
            budget = state.own_elixir - 1 + remaining * self.proposal.config.minimum_elixir_per_decision
            if (spirit is not None and state.own_elixir >= 1 and state.hand_tokens[spirit, 4] > .5
                    and budget >= 4 and threat.y >= .48):
                cell = nearest_legal_cell(max(16, int(threat.y * 32) + 1), int(threat.x * 18),
                                          state.legal_play[spirit], max_radius=2)
                if cell is not None:
                    return self._decision(state, PlanStatus.PLAY, "budgeted_spirit_stall",
                                          TeacherTarget(1, spirit, *cell, 0))
        return self._wait(state, "awaiting_affordable_visible_defense")

    def _wait(self, state: PlanPublicState, reason: str) -> PlanDecision:
        if not state.legal_wait:
            return self._decision(state, PlanStatus.INTERRUPTED, "wait_illegal")
        return self._decision(state, PlanStatus.WAIT, reason, TeacherTarget(0, 0, 0, 0, 0))

    def acknowledge(self, decision: PlanDecision | None = None, *, accepted: bool = True) -> None:
        decision = self._last if decision is None else decision
        if decision is None or decision is not self._pending or self._terminal is not None:
            raise ValueError("acknowledgement requires the current unacknowledged action")
        self._pending = None
        if not accepted:
            self._terminal = PlanDecision(None, PlanStatus.INTERRUPTED, "execution_rejected",
                                          decision.plan_id, decision.step_index, decision.decision_step)
        elif decision.reason == "saved_defender":
            self.completed = True
            self.defender_plays += 1
        elif decision.reason == "budgeted_spirit_stall":
            self._stalled = True


def _run(*, env: Any, hidden: Any, policy: Any, candidate: Mapping[str, Any], factual: BranchSpec,
         plan: SavingPlanSpec, opponent_spec: Mapping[str, Any], replica_seed: int, horizon: int,
         origin_inputs: Mapping[str, Any], own_elixir: float, arm: str, capture: bool = False) -> dict[str, Any]:
    """Use frontier continuation ordering, including actor memory on forced rows."""
    import torch
    from simulator.public_state_estimator import PublicStateEstimator
    from .frontier import tower_fracs
    from .frontier_eval import _sim_action, _step_inputs, continuation_value, make_opponent
    from .model_v4 import masks_from_legal_play
    from .tactical_data import _encode_action, active_public_mask
    from .v4_ppo.runtime import configure_v4_runtime, observe_v4_inputs

    if arm not in ("factual", "wait", "plan") or type(horizon) is not int or horizon < 1:
        raise ValueError("unknown sequence arm or invalid horizon")
    if opponent_spec.get("kind") == "v4-checkpoint" or "checkpoint" in opponent_spec:
        raise ValueError("unrestored learned opponents are excluded")
    if env.state.state_hash() != candidate["state_hash"] or plan.created_step != candidate["t"]:
        raise ValueError("sequence origin differs")
    child = env.fork()
    if child.decision_interval_ticks * child.engine.ruleset.tick_us != 250_000:
        raise ValueError("sequence rollouts require native cadence")
    before = tower_fracs(child.state)
    opponent = make_opponent(dict(opponent_spec), replica_seed, env=child)
    torch.manual_seed(replica_seed & 0xFFFFFFFF)
    configure_v4_runtime(policy)
    policy.eval()
    estimator = PublicStateEstimator()
    carried = hidden.detach().clone()
    controller = SavingPlanController(plan) if arm == "plan" else None
    rows: list[dict[str, Any]] = []
    terminal = truncated = False
    winner = None
    ran = 0
    with torch.no_grad():
        for offset in range(horizon):
            if child.state.terminal:
                break
            if offset == 0:
                inputs, elixir = origin_inputs, own_elixir
                sampled = factual if arm == "factual" else BranchSpec("wait", 0)
            else:
                obs, inputs, _, elixir = observe_v4_inputs(child, estimator)
                tensors = _step_inputs(inputs, "cpu")
                masks = masks_from_legal_play(torch.as_tensor(np.array(obs.legal_play, copy=True)).reshape(1, 1, 4, 32, 18), bool(inputs["legal_wait"]))
                _, acts, _, _, _, nxt = policy.rollout_sample(
                    tensors["raster"], tensors["global_features"], tensors["entities"],
                    tensors["entity_mask"], tensors["hand_tokens"], tensors["opp_hand_probs"],
                    tensors["opp_out_of_cycle"], tensors["opp_elixir_interval"], tensors["event_history"],
                    masks, reset_mask=torch.zeros((1, 1), dtype=torch.bool), hidden=carried)
                carried = nxt.detach()
                sampled = BranchSpec("actor", int(acts.mode[0, 0]), int(acts.card_slot[0, 0]),
                                     int(acts.placement[0, 0, 0]), int(acts.placement[0, 0, 1]))
            decision = controller.step(PlanPublicState.from_inputs(inputs, int(candidate["t"]) + offset, elixir)) if controller else None
            controlled = decision is not None and decision.active
            if controlled:
                target = decision.target
                selected = BranchSpec("plan", target.mode, target.card_slot, target.row, target.col)
            else:
                selected = sampled
            mine = _sim_action(selected, 0)
            opp = opponent.choose_action(child.engine, child.state, 1)
            executed = _encode_action(mine)
            valid = bool(inputs["legal_wait"]) or bool(np.asarray(inputs["legal_play"]).any())
            if valid:
                validate_target(executed, inputs["legal_play"], bool(inputs["legal_wait"]))
            if capture:
                air, siege = active_public_mask(inputs)
                rows.append(dict(**{key: np.array(inputs[key], copy=True) for key in PUBLIC_KEYS},
                    state_hash=child.state.state_hash(), elapsed_us=int(child.state.elapsed_us), own_elixir=elixir,
                    executed_actions=executed, opponent_actions=_encode_action(opp), label_valid=valid,
                    active=controlled and valid, air_threat=air, xbow_present=siege))
            before_events = len(child.state.events)
            _, terminated, truncated, info = child._step_core((mine, opp))
            events = child.state.events[before_events:]
            rejected = any(event.kind == "action_rejected" and event.get("player") == 0 for event in events)
            played = [str(event.get("card_id")) for event in events
                      if event.kind == "card_played" and event.get("player") == 0]
            expected = [] if selected.mode == 0 else [str(_HAND.get(int(round(float(inputs["hand_tokens"][selected.slot, 0]) * 127)), ("",))[0])]
            rejected = rejected or played != expected
            if controlled:
                controller.acknowledge(decision, accepted=not rejected)
            if rejected:
                raise ValueError("sequence branch action rejected")
            ran += 1
            if terminated or truncated:
                terminal = bool(terminated) and not bool(truncated)
                winner = info.get("winner") if terminal else None
                break
    if env.state.state_hash() != candidate["state_hash"]:
        raise ValueError("sequence branch mutated its origin")
    result = dict(value=continuation_value(before, tower_fracs(child.state), winner), winner=winner,
                  terminal=terminal, truncated=bool(truncated), decisions_run=ran,
                  plan_completed=bool(controller and controller.completed),
                  defender_plays=controller.defender_plays if controller else 0,
                  plan_reason=controller._last.reason if controller and controller._last else None)
    if capture:
        result["rows"] = rows
    return result


def _paired(left: list[float], right: list[float]) -> dict[str, float]:
    values = np.asarray(left, float) - np.asarray(right, float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("inf")
    return dict(mean=mean, se=se, lcb95=mean - 1.96 * se)


def evaluate_saving_sequence(*, env: Any, hidden: Any, policy: Any, candidate: Mapping[str, Any],
                             factual: BranchSpec, plan: SavingPlanSpec, opponent_spec: Mapping[str, Any],
                             config: Any, origin_inputs: Mapping[str, Any], own_elixir: float) -> dict[str, Any]:
    from .frontier import _stable_seed
    discovery = list(range(config.discovery_replicas))
    confirmation = list(range(config.discovery_replicas, config.discovery_replicas + config.confirmation_replicas))
    seeds = {i: _stable_seed(VERSION, config.seed, candidate["state_hash"], i) & 0xFFFFFFFF for i in discovery + confirmation}
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("sequence replica seeds collided")
    identity = plan.identity()
    result = dict(version=VERSION, accepted=False, target=None, target_kind="sequence", t=candidate["t"],
                  state_hash=candidate["state_hash"], plan_id=identity["plan_id"], plan_identity=identity,
                  frozen_plan_id=identity["plan_id"], discovery_replica_indices=discovery,
                  confirmation_replica_indices=confirmation, discovery_horizon=config.discovery_horizon,
                  confirmation_horizon=config.confirmation_horizon, arms={}, comparisons={})
    for arm in ("factual", "wait", "plan"):
        result["arms"][arm] = {}
        for phase, indices, horizon in (("discovery", discovery, config.discovery_horizon),
                                         ("confirmation", confirmation, config.confirmation_horizon)):
            replicas = [dict(replica_index=i, replica_seed=seeds[i], **_run(env=env, hidden=hidden,
                policy=policy, candidate=candidate, factual=factual, plan=plan, opponent_spec=opponent_spec,
                replica_seed=seeds[i], horizon=horizon, origin_inputs=origin_inputs, own_elixir=own_elixir, arm=arm))
                for i in indices]
            result["arms"][arm][f"{phase}_replicas"] = replicas
    confirmed = {arm: [row["value"] for row in report["confirmation_replicas"]] for arm, report in result["arms"].items()}
    result["comparisons"] = {"plan_vs_factual": _paired(confirmed["plan"], confirmed["factual"]),
                             "plan_vs_wait": _paired(confirmed["plan"], confirmed["wait"]),
                             "wait_vs_factual": _paired(confirmed["wait"], confirmed["factual"])}
    result["wait_only_accepted"] = result["comparisons"]["wait_vs_factual"]["lcb95"] > config.min_lcb
    result["accepted"] = all(result["comparisons"][name]["lcb95"] > config.min_lcb
                             for name in ("plan_vs_factual", "plan_vs_wait")) and all(
        row["plan_completed"] and row["defender_plays"] >= 1 for row in result["arms"]["plan"]["confirmation_replicas"])
    result["rejection"] = None if result["accepted"] else "sequence_confirmation_not_established"
    return result


def validate_sequence_certificate(certificate: Mapping[str, Any], *, min_lcb: float,
                                  expected_state_hash: str | None = None,
                                  expected_plan_id: str | None = None) -> dict[str, Any]:
    if (certificate.get("version") != VERSION or certificate.get("accepted") is not True
            or certificate.get("target_kind") != "sequence" or certificate.get("target") is not None
            or not math.isfinite(min_lcb) or min_lcb < 0):
        raise ValueError("sequence certificate is not an accepted whole-sequence target")
    identity = dict(certificate["plan_identity"])
    plan_id = identity.pop("plan_id")
    if (_digest(identity) != plan_id or certificate.get("plan_id") != plan_id
            or certificate.get("frozen_plan_id") != plan_id
            or expected_plan_id is not None and expected_plan_id != plan_id
            or expected_state_hash is not None and expected_state_hash != certificate.get("state_hash")):
        raise ValueError("sequence frozen plan/origin identity differs")
    phase_seeds = {}
    for phase in ("discovery", "confirmation"):
        indices = certificate[f"{phase}_replica_indices"]
        if len(indices) < 2 or len(set(indices)) != len(indices) or any(type(i) is not int for i in indices):
            raise ValueError("sequence replica indices are invalid")
        paired = []
        for arm in ("factual", "wait", "plan"):
            rows = certificate["arms"][arm][f"{phase}_replicas"]
            if len(rows) != len(indices):
                raise ValueError("sequence replica evidence incomplete")
            seeds = []
            for index, row in zip(indices, rows, strict=True):
                seed = row.get("replica_seed")
                if (row.get("replica_index") != index or type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF
                        or not math.isfinite(float(row["value"]))):
                    raise ValueError("sequence actual replica seed/index/value differs")
                seeds.append(seed)
            if len(set(seeds)) != len(seeds):
                raise ValueError("sequence actual seeds overlap")
            paired.append(seeds)
        if paired[0] != paired[1] or paired[0] != paired[2]:
            raise ValueError("sequence arms are not paired on actual seeds")
        phase_seeds[phase] = set(paired[0])
    if phase_seeds["discovery"] & phase_seeds["confirmation"] or set(certificate["discovery_replica_indices"]) & set(certificate["confirmation_replica_indices"]):
        raise ValueError("sequence confirmation overlaps discovery")
    values = {arm: [row["value"] for row in certificate["arms"][arm]["confirmation_replicas"]]
              for arm in ("factual", "wait", "plan")}
    stats = {"plan_vs_factual": _paired(values["plan"], values["factual"]),
             "plan_vs_wait": _paired(values["plan"], values["wait"]),
             "wait_vs_factual": _paired(values["wait"], values["factual"])}
    if canonical(stats) != canonical(certificate.get("comparisons")):
        raise ValueError("sequence paired statistics differ")
    if (any(stats[name]["lcb95"] <= min_lcb for name in ("plan_vs_factual", "plan_vs_wait"))
            or not all(row.get("plan_completed") is True and row.get("defender_plays", 0) >= 1
                       for row in certificate["arms"]["plan"]["confirmation_replicas"])):
        raise ValueError("sequence independent advantage/execution not established")
    return stats


def capture_saving_sequence(*, env: Any, hidden: Any, policy: Any, candidate: Mapping[str, Any],
                            factual: BranchSpec, plan: SavingPlanSpec, opponent_spec: Mapping[str, Any],
                            replica_seed: int, horizon: int, source_arrays: Mapping[str, Any],
                            source_metadata: Mapping[str, Any]) -> dict[str, Any]:
    from .v4_ppo.runtime import observe_v4_inputs
    from simulator.public_state_estimator import PublicStateEstimator
    t = int(candidate["t"])
    if (env.state.state_hash() != candidate["state_hash"]
            or str(source_arrays["state_hash"][t]) != candidate["state_hash"]
            or action_key(factual) != tuple(int(x) for x in source_arrays["executed_actions"][t][:4])):
        raise ValueError("sequence capture exact origin/action differs")
    _, observed, _, elixir = observe_v4_inputs(env, PublicStateEstimator())
    if any(not np.array_equal(observed[key], source_arrays[key][t]) for key in PUBLIC_KEYS):
        raise ValueError("sequence capture public origin differs")
    if float(source_arrays["own_elixir"][t]) != float(elixir):
        raise ValueError("sequence capture own elixir differs")
    summary = _run(env=env, hidden=hidden, policy=policy, candidate=candidate, factual=factual,
        plan=plan, opponent_spec=opponent_spec, replica_seed=replica_seed, horizon=horizon,
        origin_inputs=observed, own_elixir=elixir, arm="plan", capture=True)
    rows = summary.pop("rows")
    keys = (*PUBLIC_KEYS, "state_hash", "elapsed_us", "own_elixir", "executed_actions", "opponent_actions",
            "label_valid", "air_threat", "xbow_present")
    arrays = {key: np.concatenate((np.array(source_arrays[key][:t], copy=True), np.asarray([row[key] for row in rows])), axis=0)
              for key in keys}
    n = len(arrays["label_valid"])
    arrays["active"] = np.asarray([False] * t + [row["active"] for row in rows], dtype=bool)
    arrays["reset_mask"] = np.zeros(n, bool)
    arrays["reset_mask"][0] = True
    arrays["targets_cycle"] = np.array(arrays["executed_actions"], copy=True)
    arrays["targets_cycle"][~arrays["label_valid"]] = -1
    arrays["outcome_values"] = np.zeros((n, 3, 4), np.float32)
    arrays["outcome_mask"] = np.zeros((n, 3, 4), bool)
    metadata = dict(schema=VERSION, job_id=f"{source_metadata['job_id']}_sequence_{plan.identity()['plan_id'][:12]}_{replica_seed}",
        origin_job_id=source_metadata["job_id"], source_episode_sha256=candidate.get("source_episode_sha256", source_metadata.get("sha256")),
        origin_t=t, origin_state_hash=candidate["state_hash"], prefix_length=t, plan_id=plan.identity()["plan_id"],
        plan_identity=plan.identity(), capture_replica_seed=replica_seed, plan_completed=summary["plan_completed"],
        split=source_metadata["split"], family=source_metadata.get("family") or source_metadata["opponent"]["archetype"],
        decisions=n, horizon=horizon, capture_summary=summary)
    return dict(arrays=arrays, metadata=metadata, summary=summary)


def save_sequence_capture(path: str | Path, capture: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    arrays, metadata = capture["arrays"], dict(capture["metadata"])
    metadata["arrays_digest"] = array_digest(arrays, tuple(sorted(arrays)))
    metadata["metadata_digest"] = _digest({k: v for k, v in metadata.items() if k != "metadata_digest"})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays, metadata_json=np.asarray(canonical(metadata)))
    temporary.replace(path)
    load_sequence_capture(path)
    return dict(path=str(path), sha256=sha256_file(path), arrays_digest=metadata["arrays_digest"], metadata_digest=metadata["metadata_digest"])


def load_sequence_capture(path: str | Path, *, expected_sha: str | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise ValueError("sequence capture file hash differs")
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {key: payload[key] for key in payload.files if key != "metadata_json"}
    if (metadata.get("schema") != VERSION or metadata.get("metadata_digest") != _digest({k: v for k, v in metadata.items() if k != "metadata_digest"})
            or metadata.get("arrays_digest") != array_digest(arrays, tuple(sorted(arrays)))):
        raise ValueError("sequence capture schema/content seal differs")
    n, prefix = metadata["decisions"], metadata["prefix_length"]
    required = (*PUBLIC_KEYS, "reset_mask", "label_valid", "active", "executed_actions", "opponent_actions",
                "targets_cycle", "outcome_values", "outcome_mask", "own_elixir", "elapsed_us", "state_hash", "air_threat", "xbow_present")
    if any(key not in arrays or len(arrays[key]) != n for key in required) or not 0 <= prefix < n:
        raise ValueError("sequence capture chronology is incomplete")
    if (prefix != metadata["origin_t"] or str(arrays["state_hash"][prefix]) != metadata["origin_state_hash"]
            or np.any(arrays["active"][:prefix]) or not np.array_equal(arrays["reset_mask"], np.arange(n) == 0)
            or np.any(np.diff(arrays["elapsed_us"]) != 250_000)):
        raise ValueError("sequence prefix/origin/reset/cadence differs")
    identity = dict(metadata["plan_identity"])
    if identity.pop("plan_id") != metadata["plan_id"] or _digest(identity) != metadata["plan_id"]:
        raise ValueError("sequence capture frozen plan differs")
    for i in range(n):
        if arrays["active"][i] and not arrays["label_valid"][i]:
            raise ValueError("invalid sequence observation has a target")
        if arrays["label_valid"][i]:
            validate_target(arrays["executed_actions"][i], arrays["legal_play"][i], bool(arrays["legal_wait"][i]))
            validate_target(arrays["targets_cycle"][i], arrays["legal_play"][i], bool(arrays["legal_wait"][i]))
        if arrays["active"][i] and not np.array_equal(arrays["targets_cycle"][i], arrays["executed_actions"][i]):
            raise ValueError("sequence target differs from its executed action")
    return arrays, metadata
