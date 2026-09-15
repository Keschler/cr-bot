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
    from simulator.rl.frontier import (
        FRONTIER_VERSION,
        GRID_COLS,
        GRID_ROWS,
        _stable_seed,
        tower_fracs,
    )


FRONTIER_EVAL_VERSION: str = "frontier-eval-v4-0"

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
    import torch

    F = torch.float32
    return {
        "raster": torch.as_tensor(policy_inputs["raster"], dtype=F, device=device).unsqueeze(0).unsqueeze(0),
        "global_features": torch.as_tensor(policy_inputs["global_features"], dtype=F, device=device).unsqueeze(0).unsqueeze(0),
        "entities": torch.as_tensor(policy_inputs["entities"], dtype=F, device=device).unsqueeze(0).unsqueeze(0),
        "entity_mask": torch.as_tensor(np.asarray(policy_inputs["entity_mask"], dtype=bool), device=device).unsqueeze(0).unsqueeze(0),
        "hand_tokens": torch.as_tensor(np.asarray(policy_inputs["hand_tokens"], dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0),
        "opp_hand_probs": torch.as_tensor(np.asarray(policy_inputs["opp_hand_probs"], dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0),
        "opp_out_of_cycle": torch.as_tensor(np.asarray(policy_inputs["opp_out_of_cycle"], dtype=bool), device=device).unsqueeze(0).unsqueeze(0),
        "opp_elixir_interval": torch.as_tensor(np.asarray(policy_inputs["opp_elixir_interval"], dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0),
        "event_history": torch.as_tensor(np.asarray(policy_inputs["event_history"], dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0),
    }


def _observe_inputs(env: Any, device: Any) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], float]:
    try:
        from .v4_ppo import _v4_step_inputs
    except ImportError:  # pragma: no cover
        from simulator.rl.v4_ppo import _v4_step_inputs
    from simulator.public_state_estimator import PublicStateEstimator

    obs = env.observe_v2_for_viewer(0)
    state = env.state
    hand = list(state.players[0].hand[:4])
    elixir = float(state.players[0].elixir_milli) / 1000.0
    inputs = _v4_step_inputs(obs, hand, elixir, PublicStateEstimator().snapshot())
    legal = np.array(obs.legal_play, dtype=bool, copy=True)
    return inputs, legal, hand, elixir


def replay_to_state(
    match: dict[str, Any], t: int, policy: Any, device: Any = None
) -> tuple[Any, Any]:
    """Replay the recorded prefix to reach decision ``t`` exactly.

    Single loop: force both recorded actions per decision while stacking
    policy inputs; one batched ``policy.forward`` then yields the exact
    recurrent hidden state at ``t``.  Raises on ``state_hash`` mismatch.
    Returns ``(env, hidden)`` with hidden ``[layers, 1, hidden]``.
    """

    import torch

    decisions = match["decisions"]
    if not (0 <= t < len(decisions)):
        raise ValueError(f"candidate t={t} outside recorded prefix len={len(decisions)}")
    env = _fresh_env(match)
    stacked: list[dict[str, Any]] = []
    with torch.no_grad():
        for i in range(t + 1):
            rec = decisions[i]
            inputs, _, _, _ = _observe_inputs(env, device)
            stacked.append(_step_inputs(inputs, device))
            if i == t:
                break
            env.step_v2(
                (
                    _recorded_action(rec["learner"], 0),
                    _recorded_action(rec["opponent"], 1),
                )
            )
    actual = env.state.state_hash()
    if actual != decisions[t]["state_hash"]:
        raise ValueError(
            f"prefix replay diverged at t={t}: {actual[:12]} != {decisions[t]['state_hash'][:12]}"
        )
    with torch.no_grad():
        seq = {k: torch.cat([s[k] for s in stacked], dim=1) for k in stacked[0]}
        reset = torch.zeros((1, t + 1), dtype=torch.bool, device=device)
        reset[:, 0] = True
        _, _, ht = policy.forward(
            seq["raster"], seq["global_features"], seq["entities"], seq["entity_mask"],
            seq["hand_tokens"], seq["opp_hand_probs"], seq["opp_out_of_cycle"],
            seq["opp_elixir_interval"], seq["event_history"], reset,
        )
        layers = policy.initial_hidden(1, device=device).shape[0]
        hidden = ht[:, t : t + 1, :].transpose(0, 1).contiguous()
        assert hidden.shape[0] == layers, f"hidden layout {tuple(hidden.shape)}"
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
        from simulator.rl.model_v4 import masks_from_legal_play
        from simulator.rl.simulator_teacher import teacher_label

    if max_branches < 2:
        raise ValueError("max_branches must be >= 2")
    inputs, legal, hand, elixir = _observe_inputs(env, device)
    tensors = _step_inputs(inputs, device)
    legal_t = torch.as_tensor(legal, device=device).unsqueeze(0).unsqueeze(0)
    masks = masks_from_legal_play(legal_t)
    reset = torch.zeros((1, 1), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, _, _ = policy.forward(
            tensors["raster"], tensors["global_features"], tensors["entities"],
            tensors["entity_mask"], tensors["hand_tokens"], tensors["opp_hand_probs"],
            tensors["opp_out_of_cycle"], tensors["opp_elixir_interval"],
            tensors["event_history"], reset,
        )
        mode_p = torch.softmax(logits.mode[0, 0].float(), dim=-1).cpu().numpy()
        card_p = torch.softmax(logits.card[0, 0].float(), dim=-1).cpu().numpy()
        place = torch.softmax(logits.placement[0, 0].float().reshape(4, -1), dim=-1).cpu().numpy()

    sim_rec = candidate["learner"]
    branches = [BranchSpec(source="sim", mode=int(sim_rec["mode"]), slot=int(sim_rec["slot"]), row=int(sim_rec["row"]), col=int(sim_rec["col"]))]
    seen = {("sim", branches[0].mode, branches[0].slot, branches[0].row, branches[0].col)}

    def add(source: str, mode: int, slot: int = 0, row: int = 0, col: int = 0) -> None:
        if len(branches) >= max_branches:
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
            legal_wait=True,
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
        from .opponent_pool import make_opponent_controller
    except ImportError:  # pragma: no cover
        from simulator.rl.opponent_pool import make_opponent_controller
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
        from .v4_ppo import _v4_step_inputs
    except ImportError:  # pragma: no cover
        from simulator.rl.model_v4 import masks_from_legal_play
        from simulator.rl.v4_ppo import _v4_step_inputs
    from simulator.public_state_estimator import PublicStateEstimator

    child = env.fork()  # full fork: continuation builds observations
    torch.manual_seed(replica_seed & 0xFFFFFFFF)
    try:
        torch.cuda.manual_seed_all(replica_seed & 0xFFFFFFFF)
    except Exception:
        pass
    before = tower_fracs(child.state)
    opponent = make_opponent(opponent_spec, replica_seed)
    estimator = PublicStateEstimator()  # unfed, matching collection
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
            terminal = bool(result.terminated)
            winner = result.info.get("winner")
        else:
            for _ in range(horizon - 1):
                if child.state is None or child.state.terminal:
                    break
                obs = child.observe_v2_for_viewer(0)
                st = child.state
                hand = list(st.players[0].hand[:4])
                elixir = float(st.players[0].elixir_milli) / 1000.0
                tensors = _step_inputs(_v4_step_inputs(obs, hand, elixir, estimator.snapshot()), device)
                legal = torch.as_tensor(
                    np.array(obs.legal_play, dtype=bool, copy=True), device=device
                ).unsqueeze(0).unsqueeze(0)
                masks = masks_from_legal_play(legal)
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
                    terminal = bool(result.terminated)
                    winner = result.info.get("winner")
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
    branches = plan_branches(
        env, policy, hidden, candidate,
        max_branches=eval_cfg.max_branches,
        rng_seed=_stable_seed("branches", candidate["state_hash"]),
        device=device,
    )
    per_branch: dict[int, list[float]] = {i: [] for i in range(len(branches))}
    outcomes: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(branches))}
    for r in range(eval_cfg.n_replicas):
        replica_seed = _stable_seed(
            "frontier-eval", eval_cfg.replica_seed_base, candidate["state_hash"], r
        ) & 0xFFFFFFFF
        for i, branch in enumerate(branches):
            out = run_branch(
                env=env, hidden=hidden, branch=branch,
                opponent_spec=opp_specs[r % len(opp_specs)],
                replica_seed=replica_seed,
                horizon=eval_cfg.horizon, policy=policy, device=device,
            )
            per_branch[i].append(out["value"])
            outcomes[i].append({k: out[k] for k in ("value", "winner", "terminal", "decisions_run")})
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
    for r in range(eval_cfg.n_replicas):
        sim_v = per_branch[sim_idx][r]
        alt_v = max((per_branch[i][r] for i in range(len(branches)) if i != sim_idx), default=sim_v)
        if alt_v - sim_v > classify_cfg.margin_robust:
            stable_hits += 1
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
        "n_replicas": eval_cfg.n_replicas,
        "opponent_specs": opp_specs,
        "branches": branch_reports,
        "sim_mean": round(sim_mean, 5),
        "best_mean": round(best["mean"], 5) if best else round(sim_mean, 5),
        "best_source": best["source"] if best else "sim",
        "regret": round((best["mean"] - sim_mean), 5) if best else 0.0,
        "stability": round(stable_hits / max(1, eval_cfg.n_replicas), 4),
        "weak_best": round(weak_best, 5) if weak_best is not None else None,
        "competence": round(sim_mean - weak_best, 5) if weak_best is not None else None,
    }
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
    if regret < cfg.eps_solve:
        return "solved", f"regret {regret:.3f} < {cfg.eps_solve}"
    if stability < cfg.tau_stability:
        return "hopeless", f"unstable ranking (stability {stability:.2f})"
    if best_mean <= cfg.best_floor:
        return "hopeless", f"lost position (best continuation {best_mean:.3f} <= {cfg.best_floor})"
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
    """Compare candidate rankings across horizons; recommend shortest stable H."""

    ref = {v["state_hash"]: v for v in reports_by_horizon.get(reference, [])}
    summary: dict[str, Any] = {"reference": reference, "horizons": {}}
    for horizon in sorted(reports_by_horizon):
        if horizon == reference:
            continue
        top1 = sign = both = 0
        deltas: list[float] = []
        for v in reports_by_horizon[horizon]:
            r = ref.get(v["state_hash"])
            if r is None:
                continue
            both += 1
            top1 += int(v["best_source"] == r["best_source"])
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
            "n": both,
            "top1_agree": round(top1 / max(1, both), 4),
            "sign_agree": round(sign / max(1, both), 4),
            "mean_abs_regret_delta": round(sum(deltas) / max(1, len(deltas)), 5),
        }
    chosen = reference
    for horizon in sorted(h for h in reports_by_horizon if h != reference):
        s = summary["horizons"][horizon]
        if s["top1_agree"] >= TOP1_AGREE_FLOOR and s["sign_agree"] >= SIGN_AGREE_FLOOR:
            chosen = horizon
            break
    summary["chosen_horizon"] = chosen
    return summary


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
                "verdict": {k: v[k] for k in ("regret", "sim_mean", "best_mean", "best_source", "stability", "competence", "horizon", "n_replicas", "class")},
                "provenance": {
                    "frontier_version": FRONTIER_VERSION,
                    "eval_version": FRONTIER_EVAL_VERSION,
                    "state_hash": v["state_hash"],
                    "match_seed": match["config"]["match_seed"],
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
    "classify_verdict",
    "continuation_value",
    "evaluate_candidate",
    "evaluate_robustness",
    "headroom_bin",
    "horizon_stability",
    "is_ranking_flip",
    "load_frontier_state",
    "make_opponent",
    "materialize_entries",
    "plan_branches",
    "replay_to_state",
    "reset_to_frontier_entry",
    "run_branch",
]
