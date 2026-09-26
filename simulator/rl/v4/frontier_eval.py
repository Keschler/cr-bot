"""Data-driven frontier discovery for V4 (Stage 3): counterfactual evaluation.

For each retained candidate state this module replays the recorded action
prefix to reach the exact simulator state (verified by ``state_hash``), plans
a bounded branch set that always includes the exact simx12 action, and runs
every branch forward with an ACTIVE opponent and frozen-simx12 continuation:

    "Starting from the same state, if we change only this first decision,
     what happens afterward when both players continue playing normally?"

Matched conditions across first-action branches (common random numbers):
fresh fork per branch, policy sampling reseeded per replica, stateless
heuristic opponents recomputed from state (seeded only for random-legal).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

try:
    from .frontier import (
        FRONTIER_VERSION,
        GRID_COLS,
        GRID_ROWS,
        _stable_seed,
        tower_fracs,
    )
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.rl.v4.frontier import (
        FRONTIER_VERSION,
        GRID_COLS,
        GRID_ROWS,
        _stable_seed,
        tower_fracs,
    )


FRONTIER_EVAL_VERSION: str = "frontier-eval-v4-3"

# Terminal outcome bonus in continuation value (tower-fraction units).
WIN_BONUS: float = 2.0

# Horizon-stability acceptance: shortest H with top-1 agreement vs H=256 and
# regret-sign agreement above these floors is adopted for full evaluation.
# Sign agreement uses a dead zone: regrets within SIGN_DEADZONE of zero on
# both horizons count as agreement (noise around solved states must not force
# longer horizons).
HORIZON_CANDIDATES: tuple[int, ...] = (64, 128, 256)
TOP1_AGREE_FLOOR: float = 0.80
SIGN_AGREE_FLOOR: float = 0.85
SIGN_DEADZONE: float = 0.03


@dataclass(slots=True)
class EvalConfig:
    """Counterfactual evaluation knobs."""

    horizon: int = 128
    n_replicas: int = 4
    max_branches: int = 7
    replica_seed_base: int = 0
    # ``replica_indices`` lets a caller reserve disjoint discovery and
    # confirmation replicas while keeping the seed derivation stable.  When
    # omitted, ``range(n_replicas)`` is used.
    replica_indices: tuple[int, ...] | None = None
    # A confirmation pass may freeze the action selected by a discovery pass.
    # It still evaluates the complete branch set for diagnostics, but records
    # paired metrics for this action without selecting a new target.
    frozen_action: dict[str, Any] | None = None
    # Extra opponent specs for the robustness round; None -> collection spec.
    opponent_specs: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class ClassifyConfig:
    """Solved / frontier / hopeless cut points (value units).

    The competence gate of v1 ("sim must beat weak baselines") was removed
    after scale calibration: the largest stable regrets found (+0.1..+1.1)
    sit exactly in states where sim plays WORSE than passivity, and those
    are the most learnable mistakes, not hopeless ones.  What remains is a
    position-quality floor: the best continuation must not lose a tower
    (``best_mean > best_floor``), excluding garbage-time damage control in
    already-lost positions.
    """

    eps_solve: float = 0.03
    eps_frontier: float = 0.08
    margin_robust: float = 0.05
    tau_stability: float = 0.75
    best_floor: float = -1.0
    # Retained for diagnostics (stored on verdicts, never gating).
    margin_competent: float = 0.05


@dataclass(slots=True)
class BranchSpec:
    source: str  # sim | rule | policy-top | wait | random
    mode: int
    slot: int = 0
    row: int = 0
    col: int = 0


def _sim_action(branch: BranchSpec, player: int = 0) -> Any:
    from simulator.actions import PlayCardAction, WaitAction

    if branch.mode == 0:
        return WaitAction(player)
    return PlayCardAction(player, int(branch.slot), (int(branch.col), int(branch.row)))


def _recorded_action(rec: dict[str, Any], player: int) -> Any:
    """Rebuild a SimAction from a recorded prefix entry (both players)."""

    from simulator.actions import PlayCardAction, WaitAction

    if "kind" in rec:  # opponent-style record
        if rec["kind"] == "play":
            return PlayCardAction(player, int(rec["slot"]), (int(rec["col"]), int(rec["row"])))
        return WaitAction(player)
    if int(rec.get("mode", 0)) == 1:  # learner-style record
        return PlayCardAction(player, int(rec["slot"]), (int(rec["col"]), int(rec["row"])))
    return WaitAction(player)


def _fresh_env(match: dict[str, Any]) -> Any:
    from simulator.engine.core import BattleEngine
    from simulator.env import SimulatorEnv
    from simulator.ruleset import load_fixed_ruleset

    cfg = match["config"]
    env = SimulatorEnv(BattleEngine(load_fixed_ruleset(), validate_every_tick=False))
    env.reset_v2(
        seed=int(cfg["match_seed"]),
        decks=(tuple(cfg["learner_deck"]), tuple(cfg["opponent_deck"])),
        shuffle_decks=True,
    )
    return env


def _step_inputs(policy_inputs: dict[str, np.ndarray], device: Any) -> dict[str, Any]:
    from .v4_ppo.runtime import batch_v4_inputs

    return batch_v4_inputs(policy_inputs, device)


def _observe_inputs(env: Any, device: Any, estimator: Any = None) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], float]:
    try:
        from .v4_ppo.runtime import observe_v4_inputs
    except ImportError:  # pragma: no cover
        from simulator.rl.v4.v4_ppo.runtime import observe_v4_inputs
    from simulator.public_state_estimator import PublicStateEstimator

    if estimator is None:
        estimator = PublicStateEstimator()
    obs, inputs, hand, elixir = observe_v4_inputs(env, estimator)
    legal = np.array(obs.legal_play, dtype=bool, copy=True)
    return inputs, legal, hand, elixir


def replay_to_state(
    match: dict[str, Any], t: int, policy: Any, device: Any = None
) -> tuple[Any, Any]:
    """Replay the recorded prefix to reach decision ``t`` exactly.

    Force recorded actions and advance the actor memory once per public
    observation, including decision ``t``. Supports every GRU layer without
    retaining a full tensor prefix. Raises on ``state_hash`` mismatch.
    Returns ``(env, hidden)`` with hidden ``[layers, 1, hidden]``.
    """

    import torch

    decisions = match["decisions"]
    if not (0 <= t < len(decisions)):
        raise ValueError(f"candidate t={t} outside recorded prefix len={len(decisions)}")
    env = _fresh_env(match)
    from simulator.public_state_estimator import PublicStateEstimator
    from .v4_ppo import configure_v4_runtime
    configure_v4_runtime(policy)
    policy.eval()
    estimator = PublicStateEstimator()
    hidden = policy.initial_hidden(1, device=device)
    with torch.no_grad():
        for i in range(t + 1):
            rec = decisions[i]
            inputs, _, _, _ = _observe_inputs(env, device, estimator)
            _, _, hidden = policy.forward_step(_step_inputs(inputs, device), hidden=hidden)
            if i == t:
                break
            env.step_v2((
                _recorded_action(rec["learner"], 0),
                _recorded_action(rec["opponent"], 1),
            ))
    actual = env.state.state_hash()
    if actual != decisions[t]["state_hash"]:
        raise ValueError(
            f"prefix replay diverged at t={t}: {actual[:12]} != {decisions[t]['state_hash'][:12]}"
        )
    return env, hidden


def plan_branches(
    env: Any,
    policy: Any,
    hidden: Any,
    candidate: dict[str, Any],
    *,
    max_branches: int = 7,
    rng_seed: int = 0,
    device: Any = None,
) -> list[BranchSpec]:
    """Bounded branch set; always includes the exact recorded simx12 action."""

    import torch

    try:
        from .model_v4 import masks_from_legal_play
        from .simulator_teacher import teacher_label
    except ImportError:  # pragma: no cover
        from simulator.rl.v4.model_v4 import masks_from_legal_play
        from simulator.rl.v4.simulator_teacher import teacher_label

    if max_branches < 2:
        raise ValueError("max_branches must be >= 2")
    inputs, legal, hand, elixir = _observe_inputs(env, device)
    tensors = _step_inputs(inputs, device)
    legal_t = torch.as_tensor(legal, device=device).unsqueeze(0).unsqueeze(0)
    masks = masks_from_legal_play(legal_t, bool(inputs["legal_wait"]))
    with torch.no_grad():
        # replay_to_state returns memory AFTER the current observation. Reuse
        # it directly; starting a fresh GRU or advancing it twice is incorrect.
        zt, spatial, entities, hands = policy.encoder(
            tensors["raster"], tensors["global_features"], tensors["entities"],
            tensors["entity_mask"], tensors["hand_tokens"], tensors["opp_hand_probs"],
            tensors["opp_out_of_cycle"], tensors["opp_elixir_interval"],
            tensors["event_history"],
        )
        logits = policy.heads(zt, hidden[-1].unsqueeze(1), entities, hands, spatial)
        mode_p = torch.softmax(logits.mode[0, 0].float(), dim=-1).cpu().numpy()
        card_p = torch.softmax(logits.card[0, 0].float(), dim=-1).cpu().numpy()
        place = torch.softmax(logits.placement[0, 0].float().reshape(4, -1), dim=-1).cpu().numpy()

    sim_rec = candidate["learner"]
    branches = [BranchSpec(source="sim", mode=int(sim_rec["mode"]), slot=int(sim_rec["slot"]), row=int(sim_rec["row"]), col=int(sim_rec["col"]))]
    seen = {("sim", branches[0].mode, branches[0].slot, branches[0].row, branches[0].col)}

    def add(source: str, mode: int, slot: int = 0, row: int = 0, col: int = 0) -> None:
        if len(branches) >= max_branches:
            return
        if mode == 0 and not bool(inputs["legal_wait"]):
            return
        if mode == 1 and not bool(legal[slot, row, col]):
            return
        key = (source, mode, slot, row, col)
        triple = (mode, slot, row, col)
        if any((b.mode, b.slot, b.row, b.col) == triple for b in branches):
            return
        seen.add(key)
        branches.append(BranchSpec(source=source, mode=mode, slot=slot, row=row, col=col))

    try:
        rule = teacher_label(
            hand_tokens=np.asarray(inputs["hand_tokens"], dtype=np.float32),
            entity_tokens=np.asarray(inputs["entities"], dtype=np.float32),
            entity_mask=np.asarray(inputs["entity_mask"], dtype=bool),
            legal_play=legal,
            legal_wait=bool(inputs["legal_wait"]),
            own_elixir=float(elixir),
        )
        add("rule", int(rule.mode), int(rule.card_slot), int(rule.row), int(rule.col))
    except Exception:
        pass
    # Policy top-offs: best legal cell per slot, slots ordered by card prob.
    for slot in list(np.argsort(-np.asarray(card_p, dtype=float), kind="stable")):
        if len(branches) >= max_branches:
            break
        slot = int(slot)
        if not bool(legal[slot].any()):
            continue
        masked = np.where(legal[slot].reshape(-1), place[slot], -1.0)
        flat = int(np.argmax(masked))
        if float(masked[flat]) < 0.0:
            continue
        add("policy-top", 1, slot, flat // GRID_COLS, flat % GRID_COLS)
        if len([b for b in branches if b.source == "policy-top"]) >= 2:
            break
    add("wait", 0)
    # Weak random baseline: seeded legal play distinct from the above.
    legal_cells = [
        (s, r, c)
        for s in range(legal.shape[0])
        for r in range(legal.shape[1])
        for c in range(legal.shape[2])
        if bool(legal[s, r, c])
    ]
    if legal_cells:
        rng = random.Random(rng_seed)
        for _ in range(8):
            s, r, c = rng.choice(legal_cells)
            before = len(branches)
            add("random", 1, s, r, c)
            if len(branches) > before:
                break
    return branches


def make_opponent(spec: dict[str, Any], seed: int) -> Any:
    try:
        from ..foundation.opponent_pool import make_opponent_controller
    except ImportError:  # pragma: no cover
        from simulator.rl.foundation.opponent_pool import make_opponent_controller
    return make_opponent_controller(str(spec["strategy"]), seed=int(seed))


def continuation_value(
    before: dict[str, list[float]], after: dict[str, list[float]], winner: Any
) -> float:
    """Tower-differential change plus terminal outcome bonus."""

    enemy_gain = sum(before.get("enemy", [])) - sum(after.get("enemy", []))
    own_loss = sum(before.get("own", [])) - sum(after.get("own", []))
    outcome = 1.0 if winner == 0 else (-1.0 if winner == 1 else 0.0)
    return round(float(enemy_gain - own_loss + WIN_BONUS * outcome), 5)


def run_branch(
    *,
    env: Any,
    hidden: Any,
    branch: BranchSpec,
    opponent_spec: dict[str, Any],
    replica_seed: int,
    horizon: int,
    policy: Any,
    device: Any = None,
) -> dict[str, Any]:
    """Force the first learner action, then active play on both sides."""

    import torch

    try:
        from .model_v4 import masks_from_legal_play
        from .v4_ppo.runtime import observe_v4_inputs
    except ImportError:  # pragma: no cover
        from simulator.rl.v4.model_v4 import masks_from_legal_play
        from simulator.rl.v4.v4_ppo.runtime import observe_v4_inputs
    from simulator.public_state_estimator import PublicStateEstimator

    child = env.fork()  # full fork: continuation builds observations
    torch.manual_seed(replica_seed & 0xFFFFFFFF)
    try:
        torch.cuda.manual_seed_all(replica_seed & 0xFFFFFFFF)
    except Exception:
        pass
    before = tower_fracs(child.state)
    opponent = make_opponent(opponent_spec, replica_seed)
    from .v4_ppo import configure_v4_runtime
    configure_v4_runtime(policy)
    policy.eval()
    estimator = PublicStateEstimator()
    cur_hidden = hidden.detach()
    winner: Any = None
    terminal = False
    ran = 0
    first = _sim_action(branch, 0)
    with torch.no_grad():
        opp = opponent.choose_action(child.engine, child.state, 1)
        result = child.step_v2((first, opp))
        ran = 1
        if result.terminated or result.truncated:
            terminal = bool(result.terminated) and not bool(result.truncated)
            winner = result.info.get("winner") if terminal else None
        else:
            for _ in range(horizon - 1):
                if child.state is None or child.state.terminal:
                    break
                obs, packed, _, _ = observe_v4_inputs(child, estimator)
                tensors = _step_inputs(packed, device)
                legal = torch.as_tensor(
                    np.array(obs.legal_play, dtype=bool, copy=True), device=device
                ).unsqueeze(0).unsqueeze(0)
                masks = masks_from_legal_play(legal, bool(packed["legal_wait"]))
                reset = torch.zeros((1, 1), dtype=torch.bool, device=device)
                _, acts, _, _, _, nxt = policy.rollout_sample(
                    tensors["raster"], tensors["global_features"], tensors["entities"],
                    tensors["entity_mask"], tensors["hand_tokens"], tensors["opp_hand_probs"],
                    tensors["opp_out_of_cycle"], tensors["opp_elixir_interval"],
                    tensors["event_history"], masks, reset_mask=reset, hidden=cur_hidden,
                )
                cur_hidden = nxt.detach()
                a_mode = int(acts.mode[0, 0])
                if a_mode == 0:
                    mine = _sim_action(BranchSpec(source="*", mode=0), 0)
                else:
                    mine = _sim_action(
                        BranchSpec(source="*", mode=1, slot=int(acts.card_slot[0, 0]),
                                   row=int(acts.placement[0, 0, 0]), col=int(acts.placement[0, 0, 1])), 0,
                    )
                opp = opponent.choose_action(child.engine, child.state, 1)
                result = child.step_v2((mine, opp))
                ran += 1
                if result.terminated or result.truncated:
                    terminal = bool(result.terminated) and not bool(result.truncated)
                    winner = result.info.get("winner") if terminal else None
                    break
    after = tower_fracs(child.state)
    return {
        "source": branch.source,
        "mode": branch.mode,
        "slot": branch.slot,
        "row": branch.row,
        "col": branch.col,
        "value": continuation_value(before, after, winner),
        "winner": winner,
        "terminal": terminal,
        "decisions_run": ran,
    }


def action_identity(branch: dict[str, Any] | BranchSpec) -> dict[str, Any]:
    """Return the complete action identity carried by a branch report.

    ``source`` is provenance only.  Two policy-top branches can have the same
    source while representing different card/placement decisions, so callers
    must use this identity (or :func:`action_key`) when freezing a target.
    """

    if isinstance(branch, BranchSpec):
        return {
            "source": str(branch.source),
            "mode": int(branch.mode),
            "slot": int(branch.slot),
            "row": int(branch.row),
            "col": int(branch.col),
        }
    return {
        "source": str(branch.get("source", "")),
        "mode": int(branch.get("mode", 0)),
        "slot": int(branch.get("slot", 0)),
        "row": int(branch.get("row", 0)),
        "col": int(branch.get("col", 0)),
    }


def action_key(action: dict[str, Any] | BranchSpec) -> tuple[int, int, int, int]:
    """Return the source-independent action key ``(mode, slot, row, col)``."""

    value = action_identity(action)
    return value["mode"], value["slot"], value["row"], value["col"]


def _replica_indices(eval_cfg: EvalConfig) -> tuple[int, ...]:
    if eval_cfg.replica_indices is None:
        indices = tuple(range(int(eval_cfg.n_replicas)))
    else:
        indices = tuple(int(value) for value in eval_cfg.replica_indices)
    if not indices:
        raise ValueError("at least one replica is required")
    if any(value < 0 for value in indices):
        raise ValueError("replica indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise ValueError("replica indices must be unique")
    return indices


def _winning_action(verdict: dict[str, Any]) -> dict[str, Any] | None:
    """Recover a verdict's exact winning action, including old verdicts.

    New verdicts persist ``best_action`` directly.  For legacy verdicts that
    only retained ``best_source``, select the highest-mean branch among that
    source rather than taking the first matching branch.
    """

    stored = verdict.get("best_action")
    if isinstance(stored, dict):
        return action_identity(stored)
    source = verdict.get("best_source")
    if source is None:
        return None
    candidates = [
        branch for branch in verdict.get("branches", ())
        if branch.get("source") == source
    ]
    if not candidates:
        return None
    best = max(enumerate(candidates), key=lambda item: (float(item[1].get("mean", -float("inf"))), -item[0]))[1]
    return action_identity(best)


def _comparison_identity(verdict: dict[str, Any]) -> tuple[Any, ...] | None:
    action = _winning_action(verdict)
    # A source label is provenance, never an action identity.  Legacy reports
    # that omitted branch coordinates are therefore unavailable for a strict
    # horizon comparison instead of being counted as agreeing by source.
    return ("action", *action_key(action)) if action is not None else None


def _branch_values_by_replica(branch: dict[str, Any]) -> dict[int, float]:
    replicas = branch.get("replicas")
    if isinstance(replicas, list):
        values: dict[int, float] = {}
        for position, replica in enumerate(replicas):
            if not isinstance(replica, dict) or "value" not in replica:
                continue
            index = int(replica.get("replica_index", position))
            values[index] = float(replica["value"])
        if values:
            return values
    return {
        index: float(value)
        for index, value in enumerate(branch.get("values", ()))
    }


def fixed_action_metrics(
    verdict: dict[str, Any],
    frozen_action: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute paired alternative-minus-sim metrics for one fixed action.

    The function is intentionally JSON-safe and keeps the replica indices used
    in the calculation.  It is shared by frontier generation and the audit so
    independent confirmation can never silently fall back to a selected
    action's aggregate mean.
    """

    if frozen_action is None:
        return {"available": False, "reason": "frozen action is missing"}
    target = action_key(frozen_action)
    branches = list(verdict.get("branches", ()))
    sim = next((branch for branch in branches if branch.get("source") == "sim"), None)
    alternative = next((branch for branch in branches if action_key(branch) == target), None)
    if sim is None or alternative is None:
        return {
            "available": False,
            "reason": "frozen action or sim branch is absent",
            "frozen_action": action_identity(frozen_action),
        }
    sim_values = _branch_values_by_replica(sim)
    alt_values = _branch_values_by_replica(alternative)
    indices = sorted(set(sim_values) & set(alt_values))
    if not indices:
        return {
            "available": False,
            "reason": "frozen action has no paired replicas",
            "frozen_action": action_identity(frozen_action),
        }
    deltas = [alt_values[index] - sim_values[index] for index in indices]
    mean = float(sum(deltas) / len(deltas))
    variance = (
        sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
        if len(deltas) > 1 else 0.0
    )
    se = float(variance / len(deltas)) ** 0.5
    return {
        "available": True,
        "frozen_action": action_identity(frozen_action),
        "action_source": alternative.get("source"),
        "replica_indices": indices,
        "n": len(indices),
        "deltas": [round(value, 6) for value in deltas],
        "mean_delta": round(mean, 6),
        "sd_delta": round(variance ** 0.5, 6),
        "se_delta": round(se, 6),
        "lcb95": round(mean - 1.96 * se, 6),
        "ucb95": round(mean + 1.96 * se, 6),
        "positive_fraction": round(sum(value > 0.0 for value in deltas) / len(deltas), 6),
    }


def _summary_stats(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, float(variance / len(values)) ** 0.5


def merge_independent_confirmation(
    discovery: dict[str, Any],
    confirmation: dict[str, Any],
    *,
    frozen_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach an independent held-out evaluation to a discovery verdict.

    Discovery statistics remain the selection/classification statistics.  The
    confirmation replicas are appended with explicit indices and seeds, while
    each branch retains discovery/confirmation aggregates separately.  This
    prevents a downstream learner from mistaking a held-out result for another
    selected-action sample and makes the independence boundary auditable.
    """

    if discovery.get("state_hash") != confirmation.get("state_hash"):
        raise ValueError("cannot merge confirmations for different states")
    raw_frozen = frozen_action or discovery.get("best_action") or _winning_action(discovery)
    if raw_frozen is None:
        raise ValueError("discovery verdict has no winning action to freeze")
    frozen = action_identity(raw_frozen)
    discovery_indices = {int(value) for value in discovery.get("replica_indices", ())}
    confirmation_indices = {int(value) for value in confirmation.get("replica_indices", ())}
    if not discovery_indices:
        discovery_indices = {
            int(replica.get("replica_index", position))
            for branch in discovery.get("branches", ())
            for position, replica in enumerate(branch.get("replicas", ()))
            if isinstance(replica, dict)
        }
    if not confirmation_indices:
        confirmation_indices = {
            int(replica.get("replica_index", position))
            for branch in confirmation.get("branches", ())
            for position, replica in enumerate(branch.get("replicas", ()))
            if isinstance(replica, dict)
        }
    if discovery_indices & confirmation_indices:
        raise ValueError("discovery and confirmation replicas must be disjoint")
    confirmation_by_key = {
        action_key(branch): branch for branch in confirmation.get("branches", ())
    }
    merged_branches: list[dict[str, Any]] = []
    for original in discovery.get("branches", ()):
        key = action_key(original)
        held_out = confirmation_by_key.get(key)
        if held_out is None:
            raise ValueError(f"confirmation is missing branch action {key}")
        item = dict(original)
        discovery_values = [float(value) for value in original.get("values", ())]
        confirmation_values = [float(value) for value in held_out.get("values", ())]
        discovery_replicas = list(original.get("replicas", ()))
        confirmation_replicas = list(held_out.get("replicas", ()))
        # Legacy verdicts can be merged too; synthesize provenance records for
        # their compact values using the declared replica index ranges.
        if not discovery_replicas:
            discovery_replicas = [
                {"replica_index": index, "value": value}
                for index, value in zip(sorted(discovery_indices), discovery_values)
            ]
        if not confirmation_replicas:
            confirmation_replicas = [
                {"replica_index": index, "value": value}
                for index, value in zip(sorted(confirmation_indices), confirmation_values)
            ]
        all_values = discovery_values + confirmation_values
        all_replicas = discovery_replicas + confirmation_replicas
        discovery_mean, discovery_se = _summary_stats(discovery_values)
        confirmation_mean, confirmation_se = _summary_stats(confirmation_values)
        all_mean, all_se = _summary_stats(all_values)
        item["discovery_values"] = [round(value, 5) for value in discovery_values]
        item["confirmation_values"] = [round(value, 5) for value in confirmation_values]
        # Keep the historical compact fields scoped to discovery.  Consumers
        # that have not learned the confirmation schema therefore retain the
        # same selection semantics; ``all_values`` is explicit opt-in.
        item["values"] = [round(value, 5) for value in discovery_values]
        item["discovery_replicas"] = discovery_replicas
        item["confirmation_replicas"] = confirmation_replicas
        item["replicas"] = all_replicas
        item["replica_indices"] = sorted(discovery_indices)
        item["all_values"] = [round(value, 5) for value in all_values]
        item["all_replica_indices"] = sorted(discovery_indices | confirmation_indices)
        # ``mean``/``se`` remain discovery values for backwards-compatible
        # selection.  The explicit aggregate fields are for diagnostics.
        item["mean"] = round(discovery_mean, 5)
        item["se"] = round(discovery_se, 5)
        item["mean_discovery"] = round(discovery_mean, 5)
        item["se_discovery"] = round(discovery_se, 5)
        item["mean_confirmation"] = round(confirmation_mean, 5)
        item["se_confirmation"] = round(confirmation_se, 5)
        item["mean_all"] = round(all_mean, 5)
        item["se_all"] = round(all_se, 5)
        item["outcomes"] = list(original.get("outcomes", ())) + list(held_out.get("outcomes", ()))
        merged_branches.append(item)
    result = dict(discovery)
    result["branches"] = merged_branches
    result["best_action"] = frozen if frozen else discovery.get("best_action")
    result["discovery_replica_indices"] = sorted(discovery_indices)
    result["confirmation_replica_indices"] = sorted(confirmation_indices)
    result["discovery_n_replicas"] = len(discovery_indices)
    result["confirmation_n_replicas"] = len(confirmation_indices)
    result["frozen_action"] = frozen
    # Confirmation confidence is held-out only.  Computing it from the
    # merged branch replica list would reintroduce the discovery samples that
    # selected the action and make the bound optimistic.
    result["frozen_confirmation"] = fixed_action_metrics(confirmation, frozen)
    result["confirmation"] = {
        "version": FRONTIER_EVAL_VERSION,
        "replica_indices": sorted(confirmation_indices),
        "n_replicas": len(confirmation_indices),
        "frozen_action": frozen,
        **result["frozen_confirmation"],
    }
    return result


def evaluate_candidate(
    *,
    match: dict[str, Any],
    candidate: dict[str, Any],
    policy: Any,
    eval_cfg: EvalConfig,
    classify_cfg: ClassifyConfig | None = None,
    device: Any = None,
) -> dict[str, Any]:
    """Full replica evaluation of one candidate state (JSON-serializable)."""

    classify_cfg = classify_cfg or ClassifyConfig()
    env, hidden = replay_to_state(match, int(candidate["t"]), policy, device)
    if eval_cfg.opponent_specs:
        opp_specs = eval_cfg.opponent_specs
    else:
        mc = match["config"]
        opp_specs = [{"strategy": mc["strategy"], "controller_seed": mc.get("controller_seed", 0)}]
    replica_indices = _replica_indices(eval_cfg)
    branches = plan_branches(
        env, policy, hidden, candidate,
        max_branches=eval_cfg.max_branches,
        rng_seed=_stable_seed("branches", candidate["state_hash"]),
        device=device,
    )
    per_branch: dict[int, list[float]] = {i: [] for i in range(len(branches))}
    outcomes: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(branches))}
    for replica_position, replica_index in enumerate(replica_indices):
        replica_seed = _stable_seed(
            "frontier-eval", eval_cfg.replica_seed_base, candidate["state_hash"], replica_index
        ) & 0xFFFFFFFF
        for i, branch in enumerate(branches):
            out = run_branch(
                env=env, hidden=hidden, branch=branch,
                opponent_spec=opp_specs[replica_position % len(opp_specs)],
                replica_seed=replica_seed,
                horizon=eval_cfg.horizon, policy=policy, device=device,
            )
            per_branch[i].append(out["value"])
            outcomes[i].append(
                {
                    "replica_index": int(replica_index),
                    "replica_seed": int(replica_seed),
                    **{k: out[k] for k in ("value", "winner", "terminal", "decisions_run")},
                }
            )
    branch_reports = []
    for i, branch in enumerate(branches):
        vals = per_branch[i]
        mean = float(sum(vals) / len(vals))
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        branch_reports.append(
            {
                "source": branch.source, "mode": branch.mode, "slot": branch.slot,
                "row": branch.row, "col": branch.col,
                "mean": round(mean, 5), "se": round((var / len(vals)) ** 0.5, 5),
                "values": [round(v, 5) for v in vals],
                # ``values`` remains the compact legacy representation.  The
                # replica records are the authoritative representation for
                # held-out confirmation and retain the seed/index provenance.
                "replica_indices": [int(value) for value in replica_indices],
                "replicas": outcomes[i],
                "outcomes": outcomes[i],
            }
        )
    sim_idx = next(i for i, b in enumerate(branches) if b.source == "sim")
    sim_mean = branch_reports[sim_idx]["mean"]
    alt_reports = [rep for i, rep in enumerate(branch_reports) if i != sim_idx]
    best = max(alt_reports, key=lambda rep: rep["mean"]) if alt_reports else None
    weak_reports = [rep for rep in alt_reports if rep["source"] in ("wait", "random")]
    weak_best = max((rep["mean"] for rep in weak_reports), default=None)
    # Per-replica best-alt vs sim (ranking stability with matched seeds).
    stable_hits = 0
    for position in range(len(replica_indices)):
        sim_v = per_branch[sim_idx][position]
        alt_v = max((per_branch[i][position] for i in range(len(branches)) if i != sim_idx), default=sim_v)
        if alt_v - sim_v > classify_cfg.margin_robust:
            stable_hits += 1
    best_action = action_identity(best) if best else action_identity(branches[sim_idx])
    verdict = {
        "version": FRONTIER_EVAL_VERSION,
        "match_id": candidate["match_id"],
        "archetype": candidate["archetype"],
        "t": candidate["t"],
        "tick": candidate["tick"],
        "state_hash": candidate["state_hash"],
        "tags": candidate["tags"],
        "filter": candidate["filter"],
        "horizon": eval_cfg.horizon,
        "n_replicas": len(replica_indices),
        "replica_indices": [int(value) for value in replica_indices],
        "opponent_specs": opp_specs,
        "branches": branch_reports,
        "sim_mean": round(sim_mean, 5),
        "best_mean": round(best["mean"], 5) if best else round(sim_mean, 5),
        "best_source": best["source"] if best else "sim",
        "best_action": best_action,
        "regret": round((best["mean"] - sim_mean), 5) if best else 0.0,
        "stability": round(stable_hits / max(1, len(replica_indices)), 4),
        "weak_best": round(weak_best, 5) if weak_best is not None else None,
        "competence": round(sim_mean - weak_best, 5) if weak_best is not None else None,
    }
    if eval_cfg.frozen_action is not None:
        verdict["frozen_action"] = action_identity(eval_cfg.frozen_action)
        verdict["frozen_confirmation"] = fixed_action_metrics(verdict, eval_cfg.frozen_action)
    verdict["class"], verdict["class_reason"] = classify_verdict(verdict, classify_cfg)
    return verdict


def classify_verdict(
    verdict: dict[str, Any], cfg: ClassifyConfig | None = None
) -> tuple[str, str]:
    """solved | frontier | hopeless from regret, stability, best position."""

    cfg = cfg or ClassifyConfig()
    regret = float(verdict.get("regret", 0.0))
    stability = float(verdict.get("stability", 0.0))
    best_mean = float(verdict.get("best_mean", 0.0))
    if best_mean <= cfg.best_floor:
        return "hopeless", f"lost position (best continuation {best_mean:.3f} <= {cfg.best_floor})"
    if regret < cfg.eps_solve:
        return "solved", f"regret {regret:.3f} < {cfg.eps_solve}"
    if stability < cfg.tau_stability:
        return "hopeless", f"unstable ranking (stability {stability:.2f})"
    if regret >= cfg.eps_frontier:
        return "frontier", (
            f"regret {regret:.3f} >= {cfg.eps_frontier}, stability {stability:.2f}, "
            f"best {best_mean:.3f}"
        )
    return "hopeless", f"regret {regret:.3f} below frontier floor {cfg.eps_frontier}"


def is_ranking_flip(base_regret: float, alt_regret: float) -> bool:
    """True only on genuine +<->- flips (emergence neutral<->signed is not a flip)."""

    base_pos = float(base_regret) > SIGN_DEADZONE
    alt_neg = float(alt_regret) < -SIGN_DEADZONE
    base_neg = float(base_regret) < -SIGN_DEADZONE
    alt_pos = float(alt_regret) > SIGN_DEADZONE
    return bool((base_pos and alt_neg) or (base_neg and alt_pos))


def evaluate_robustness(
    *,
    match: dict[str, Any],
    verdict: dict[str, Any],
    policy: Any,
    strategies: Sequence[str],
    n_replicas: int = 2,
    horizon: int = 128,
    classify_cfg: ClassifyConfig | None = None,
    device: Any = None,
) -> dict[str, Any]:
    """Re-evaluate a frontier verdict under alternate opponent controllers.

    Decks and the recorded prefix are unchanged (replay forces recorded
    actions, so the state stays valid); only the continuation opponent
    policy varies.  A true best-vs-sim sign flip under any alternate
    demotes the state to hopeless (ranking not robust across opponents).
    """

    classify_cfg = classify_cfg or ClassifyConfig()
    candidate = {
        "match_id": verdict["match_id"],
        "archetype": verdict["archetype"],
        "t": verdict["t"],
        "tick": verdict["tick"],
        "state_hash": verdict["state_hash"],
        "tags": verdict["tags"],
        "filter": verdict["filter"],
        "learner": {"mode": 0, "slot": 0, "row": 0, "col": 0},
    }
    # Recover the exact sim action from the verdict's sim branch.
    sim_branch = next(b for b in verdict["branches"] if b["source"] == "sim")
    candidate["learner"] = {
        "mode": sim_branch["mode"], "slot": sim_branch["slot"],
        "row": sim_branch["row"], "col": sim_branch["col"],
    }
    per_strategy: dict[str, Any] = {}
    flips: list[str] = []
    for strategy in strategies:
        alt = evaluate_candidate(
            match=match, candidate=candidate, policy=policy,
            eval_cfg=EvalConfig(
                horizon=horizon, n_replicas=n_replicas,
                opponent_specs=[{"strategy": strategy, "controller_seed": 0}],
            ),
            classify_cfg=classify_cfg,
            device=device,
        )
        per_strategy[strategy] = {
            "regret": alt["regret"], "stability": alt["stability"],
            "best_source": alt["best_source"], "sim_mean": alt["sim_mean"],
            "best_mean": alt["best_mean"],
        }
        if is_ranking_flip(float(verdict["regret"]), float(alt["regret"])):
            flips.append(strategy)
    out = {"strategies": per_strategy, "flips": flips}
    if flips:
        out["demoted"] = True
        out["reason"] = f"ranking flips under alternate opponents: {flips}"
    else:
        out["demoted"] = False
    return out


def horizon_stability(
    reports_by_horizon: dict[int, list[dict[str, Any]]],
    reference: int = 256,
) -> dict[str, Any]:
    """Compare candidate rankings across horizons; recommend shortest stable H.

    Ranking identity is the complete action key.  ``best_source`` is retained
    only as a legacy fallback for verdicts produced before exact action
    identities were persisted.  Missing states and missing action identities
    are reported as unavailable instead of disappearing from the denominator.
    """

    ref = {v["state_hash"]: v for v in reports_by_horizon.get(reference, [])}
    summary: dict[str, Any] = {"reference": reference, "horizons": {}}
    for horizon in sorted(reports_by_horizon):
        if horizon == reference:
            continue
        top1 = sign = compared = 0
        deltas: list[float] = []
        candidates = reports_by_horizon[horizon]
        candidate_hashes = {v.get("state_hash") for v in candidates}
        unavailable = len(candidate_hashes - set(ref))
        unavailable += len(set(ref) - candidate_hashes)
        for v in reports_by_horizon[horizon]:
            r = ref.get(v["state_hash"])
            if r is None:
                continue
            short_action = _comparison_identity(v)
            reference_action = _comparison_identity(r)
            if short_action is None or reference_action is None:
                unavailable += 1
                continue
            compared += 1
            top1 += int(short_action == reference_action)
            # One-sided sign: only true +<->- flips count.  Emergence
            # (neutral at short H, signed at long H) is the horizon doing
            # its job — delayed consequences materializing — not instability.
            v_pos = float(v["regret"]) > SIGN_DEADZONE
            v_neg = float(v["regret"]) < -SIGN_DEADZONE
            r_pos = float(r["regret"]) > SIGN_DEADZONE
            r_neg = float(r["regret"]) < -SIGN_DEADZONE
            sign += int(not ((v_pos and r_neg) or (v_neg and r_pos)))
            deltas.append(abs(float(v["regret"]) - float(r["regret"])))
        summary["horizons"][horizon] = {
            # ``n`` remains the historical matched-state field.  New callers
            # should use ``compared`` and ``unavailable`` for gate decisions.
            "n": compared,
            "compared": compared,
            "available": compared,
            "unavailable": unavailable,
            "n_candidates": len(candidates),
            "n_reference": len(ref),
            "top1_agree": round(top1 / max(1, compared), 4),
            "sign_agree": round(sign / max(1, compared), 4),
            "mean_abs_regret_delta": round(sum(deltas) / max(1, len(deltas)), 5),
        }
    chosen = reference
    for horizon in sorted(h for h in reports_by_horizon if h != reference):
        s = summary["horizons"][horizon]
        if (
            s["unavailable"] == 0
            and s["top1_agree"] >= TOP1_AGREE_FLOOR
            and s["sign_agree"] >= SIGN_AGREE_FLOOR
        ):
            chosen = horizon
            break
    summary["chosen_horizon"] = chosen
    return summary


def _audit_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("verdict")
    return value if isinstance(value, dict) else payload


def _audit_state_hash(payload: dict[str, Any]) -> str | None:
    verdict = _audit_verdict(payload)
    if verdict.get("state_hash") is not None:
        return str(verdict["state_hash"])
    source = payload.get("source")
    if isinstance(source, dict) and source.get("state_hash") is not None:
        return str(source["state_hash"])
    return None


def _audit_frozen_action(payload: dict[str, Any]) -> dict[str, Any] | None:
    verdict = _audit_verdict(payload)
    confirmation = payload.get("frozen_confirmation")
    if isinstance(confirmation, dict) and isinstance(confirmation.get("frozen_action"), dict):
        return action_identity(confirmation["frozen_action"])
    if isinstance(verdict.get("frozen_action"), dict):
        return action_identity(verdict["frozen_action"])
    return _winning_action(verdict)


def _payload_fixed_metrics(
    payload: dict[str, Any], frozen: dict[str, Any]
) -> dict[str, Any]:
    explicit = payload.get("frozen_confirmation")
    if isinstance(explicit, dict):
        explicit_action = explicit.get("frozen_action")
        if not isinstance(explicit_action, dict) or action_key(explicit_action) != action_key(frozen):
            return {"available": False, "reason": "frozen action differs"}
        # An unavailable held-out comparison must stay unavailable. Falling
        # back to all search replicas would let discovery samples pass the gate.
        return explicit
    return fixed_action_metrics(_audit_verdict(payload), frozen)


def fixed_action_horizon_gate(
    reports_by_horizon: dict[int, list[dict[str, Any]]],
    *,
    reference: int = 112,
    required_horizons: Sequence[int] | None = None,
    top1_floor: float = TOP1_AGREE_FLOOR,
    sign_floor: float = SIGN_AGREE_FLOOR,
    deadzone: float = SIGN_DEADZONE,
) -> dict[str, Any]:
    """Gate horizon selection using one frozen action per state.

    ``reports_by_horizon`` accepts raw verdicts or teacher-audit payloads
    containing ``verdict`` and ``frozen_confirmation``.  The action is frozen
    from the reference report (or its explicit confirmation field), then the
    exact same action is looked up at each comparison horizon.  A missing
    state, action, sim branch, or paired replica is counted as unavailable and
    fails the gate.  This makes the result safe to call from a shell gate.
    """

    available_horizons = {int(value) for value in reports_by_horizon}
    if required_horizons is None:
        required = tuple(sorted(value for value in available_horizons if value != int(reference)))
    else:
        required = tuple(dict.fromkeys(int(value) for value in required_horizons))
    reference_rows = {
        state: payload
        for payload in reports_by_horizon.get(int(reference), ())
        if (state := _audit_state_hash(payload)) is not None
    }
    comparisons: dict[str, Any] = {}
    if int(reference) not in available_horizons:
        return {
            "pass": False,
            "reference": int(reference),
            "required_horizons": list(required),
            "chosen_horizon": None,
            "reason": "reference horizon is unavailable",
            "comparisons": comparisons,
            "horizons": comparisons,
        }

    for horizon in required:
        rows = {
            state: payload
            for payload in reports_by_horizon.get(horizon, ())
            if (state := _audit_state_hash(payload)) is not None
        }
        compared = 0
        top1 = 0
        sign = 0
        unavailable_states = set(reference_rows) ^ set(rows)
        unavailable = len(unavailable_states)
        for state, reference_payload in reference_rows.items():
            current = rows.get(state)
            frozen = _audit_frozen_action(reference_payload)
            current_verdict = _audit_verdict(current) if current is not None else None
            if current_verdict is None or frozen is None:
                continue
            reference_metrics = _payload_fixed_metrics(reference_payload, frozen)
            current_metrics = _payload_fixed_metrics(current, frozen)
            if not reference_metrics.get("available") or not current_metrics.get("available"):
                unavailable += 1
                continue
            compared += 1
            top1 += int(_winning_action(current_verdict) is not None and action_key(_winning_action(current_verdict)) == action_key(frozen))
            reference_delta = float(reference_metrics["mean_delta"])
            current_delta = float(current_metrics["mean_delta"])
            reference_pos = reference_delta > deadzone
            reference_neg = reference_delta < -deadzone
            current_pos = current_delta > deadzone
            current_neg = current_delta < -deadzone
            sign += int(not ((reference_pos and current_neg) or (reference_neg and current_pos)))
        top1_agree = top1 / max(1, compared)
        sign_agree = sign / max(1, compared)
        passed = (
            unavailable == 0
            and compared > 0
            and top1_agree >= float(top1_floor)
            and sign_agree >= float(sign_floor)
        )
        comparisons[str(horizon)] = {
            "horizon": horizon,
            "n_reference": len(reference_rows),
            "n_candidates": len(rows),
            "compared": compared,
            "available": compared,
            "unavailable": unavailable,
            "top1_agree": round(top1_agree, 4),
            "sign_agree": round(sign_agree, 4),
            "pass": passed,
        }

    passing = [horizon for horizon in required if comparisons.get(str(horizon), {}).get("pass")]
    gate_pass = bool(required) and len(passing) == len(required)
    reason = "ok" if gate_pass else "one or more fixed-action horizon comparisons failed"
    return {
        "pass": gate_pass,
        "reference": int(reference),
        "required_horizons": list(required),
        "chosen_horizon": min(passing) if gate_pass else int(reference),
        "reason": reason,
        "comparisons": comparisons,
        "horizons": comparisons,
    }


# A concise alias for shell/runner callers that prefer the shorter name.
horizon_confidence_gate = fixed_action_horizon_gate

# Public spelling for runners that should not depend on the compatibility
# underscore used by the legacy helper.
winning_action = _winning_action


# ---------------------------------------------------------------------------
# Distribution build + sealed split + PPO episode loading.
# ---------------------------------------------------------------------------


def headroom_bin(regret: float, cuts: tuple[float, float]) -> str:
    if regret < cuts[0]:
        return "modest"
    if regret < cuts[1]:
        return "medium"
    return "large"


def build_distribution(
    verdicts: Sequence[dict[str, Any]],
    *,
    max_train: int = 512,
    per_signature_cap: int = 3,
    per_archetype_cap: int = 96,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stratified frontier train set: headroom x archetype, deduplicated."""

    frontier = [v for v in verdicts if v.get("class") == "frontier"]
    if not frontier:
        return [], {"n_verdicts": len(verdicts), "n_frontier": 0, "n_train": 0}
    regrets = sorted(float(v["regret"]) for v in frontier)
    cuts = (regrets[len(regrets) // 3], regrets[2 * len(regrets) // 3])
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for v in frontier:
        key = (headroom_bin(float(v["regret"]), cuts), str(v.get("archetype", "?")))
        buckets.setdefault(key, []).append(v)
    for group in buckets.values():
        group.sort(key=lambda v: -float(v["regret"]))
    picked: list[dict[str, Any]] = []
    sig_counts: dict[str, int] = {}
    arch_counts: dict[str, int] = {}
    # Round-robin across buckets so modest/medium/large and archetypes mix.
    while len(picked) < max_train:
        progressed = False
        for key in sorted(buckets):
            group = buckets[key]
            while group:
                v = group.pop(0)
                sig = f"{v.get('archetype')}#{v.get('state_hash', '')[:16]}"
                arch = str(v.get("archetype", "?"))
                if sig_counts.get(sig, 0) >= per_signature_cap:
                    continue
                if arch_counts.get(arch, 0) >= per_archetype_cap:
                    continue
                sig_counts[sig] = sig_counts.get(sig, 0) + 1
                arch_counts[arch] = arch_counts.get(arch, 0) + 1
                v = dict(v)
                v["headroom"] = key[0]
                picked.append(v)
                progressed = True
                break
            if len(picked) >= max_train:
                break
        if not progressed:
            break
    report = {
        "n_verdicts": len(verdicts),
        "n_frontier": len(frontier),
        "n_train": len(picked),
        "regret_cuts": [round(c, 4) for c in cuts],
        "by_headroom": {},
        "by_archetype": dict(arch_counts),
        "by_class": {},
    }
    for v in verdicts:
        report["by_class"][v.get("class", "?")] = report["by_class"].get(v.get("class", "?"), 0) + 1
    for v in picked:
        report["by_headroom"][v["headroom"]] = report["by_headroom"].get(v["headroom"], 0) + 1
    return picked, report


def materialize_entries(
    matches: dict[str, dict[str, Any]],
    verdicts: Sequence[dict[str, Any]],
    policy: Any,
    *,
    split: str,
    device: Any = None,
) -> list[dict[str, Any]]:
    """Replay frontier verdicts once more; capture resettable PPO entries."""

    entries: list[dict[str, Any]] = []
    for v in verdicts:
        match = matches[v["match_id"]]
        env, hidden = replay_to_state(match, int(v["t"]), policy, device)
        import torch

        entries.append(
            {
                "entry_id": f"{split}:{v['match_id']}:t{v['t']}",
                "split": split,
                "initial_state": env.save_state(),
                "hidden0": hidden.detach().cpu(),
                "learner_deck": match["config"]["learner_deck"],
                "opponent_deck": match["config"]["opponent_deck"],
                "opponent": {
                    "archetype": match["config"]["archetype"],
                    "strategy": match["config"]["strategy"],
                    "controller_seed": int(match["config"].get("controller_seed", 0)),
                },
                "tags": v["tags"],
                "headroom": v.get("headroom", "unknown"),
                "verdict": {
                    k: v[k]
                    for k in (
                        "regret", "sim_mean", "best_mean", "best_source", "best_action",
                        "stability", "competence", "horizon", "n_replicas", "class",
                    )
                    if k in v
                },
                "provenance": {
                    "frontier_version": FRONTIER_VERSION,
                    "eval_version": FRONTIER_EVAL_VERSION,
                    "state_hash": v["state_hash"],
                    "match_seed": match["config"]["match_seed"],
                    "best_action": v.get("best_action"),
                    "discovery_replica_indices": v.get("discovery_replica_indices", v.get("replica_indices", [])),
                    "confirmation_replica_indices": v.get("confirmation_replica_indices", []),
                },
            }
        )
    return entries


def load_frontier_state(env: Any, primitive: dict[str, Any]) -> None:
    """Restore a materialized entry state without strict-V1 observation.

    Same validation as :meth:`SimulatorEnv.load_state` (event-inclusive
    snapshot, complete event history) but skips the eager V1 ``observe()``:
    V1 ``vision_v1_exact`` cannot represent exotic opponent-hand cards
    (e.g. bomb-tower in a variant deck) while the V2 actor path tolerates
    them.  Memories restart fresh, exactly like post-``reset()``; the PPO
    consumer re-accumulates them by stepping (documented step-0 shift).
    """

    from simulator.observation import ObservationMemory as EnvMemory
    from simulator.state import battle_state_from_primitive
    events = primitive.get("events") if isinstance(primitive, dict) else None
    if not isinstance(events, list):
        raise ValueError("frontier entry requires an event-inclusive snapshot")
    state = battle_state_from_primitive(primitive)
    env.engine.validate_state(state)
    event_sequences = [event.sequence for event in state.events]
    if event_sequences != list(range(state.event_sequence)):
        raise ValueError("frontier entry requires a complete event history")
    env.state = state
    env._memories = (EnvMemory(0), EnvMemory(1))
    env._persistent_observation_cache = None


def reset_to_frontier_entry(env: Any, entry: dict[str, Any]) -> dict[str, Any]:
    """Reset a live env to a frontier entry; return hidden + opponent."""

    import torch

    load_frontier_state(env, entry["initial_state"])
    hidden = torch.as_tensor(entry["hidden0"]).contiguous()
    opp = entry["opponent"]
    opponent = make_opponent(opp, int(opp.get("controller_seed", 0)))
    return {"hidden": hidden, "opponent": opponent, "entry_id": entry["entry_id"]}


def assert_split_disjoint(train: Sequence[dict[str, Any]], sealed: Sequence[dict[str, Any]]) -> None:
    train_hashes = {e["provenance"]["state_hash"] for e in train}
    sealed_hashes = {e["provenance"]["state_hash"] for e in sealed}
    overlap = train_hashes & sealed_hashes
    if overlap:
        raise ValueError(f"train/sealed overlap: {len(overlap)} shared state hashes")
    train_seeds = {e["provenance"]["match_seed"] for e in train}
    sealed_seeds = {e["provenance"]["match_seed"] for e in sealed}
    if train_seeds & sealed_seeds:
        raise ValueError("train/sealed share match seeds; sealed must use different seeds")


__all__ = [
    "BranchSpec",
    "ClassifyConfig",
    "EvalConfig",
    "FRONTIER_EVAL_VERSION",
    "HORIZON_CANDIDATES",
    "SIGN_AGREE_FLOOR",
    "TOP1_AGREE_FLOOR",
    "WIN_BONUS",
    "assert_split_disjoint",
    "build_distribution",
    "action_identity",
    "action_key",
    "classify_verdict",
    "continuation_value",
    "evaluate_candidate",
    "evaluate_robustness",
    "headroom_bin",
    "horizon_stability",
    "fixed_action_horizon_gate",
    "horizon_confidence_gate",
    "fixed_action_metrics",
    "merge_independent_confirmation",
    "is_ranking_flip",
    "load_frontier_state",
    "make_opponent",
    "materialize_entries",
    "plan_branches",
    "replay_to_state",
    "reset_to_frontier_entry",
    "run_branch",
    "winning_action",
]
