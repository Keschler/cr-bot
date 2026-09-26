"""Paired branch comparisons, stochastic continuations, and full-match evaluation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..frontier import _stable_seed
from ..frontier_eval import BranchSpec, load_frontier_state, make_opponent, run_branch
from .actions import _entry_continuation_hidden, _policy_argmax_at_entry
from .constants import ADAPTIVE_REPLICA_CAP, FRONTIER_PPO_VERSION, PAIRED_SIGN_K
from .environment import entry_input_tensors, make_frontier_env
from .statistics import classify_paired_sign, paired_delta_stats


def sealed_choice_regret(
    *,
    policy: Any,
    champion: Any,
    entries: Sequence[dict[str, Any]],
    n_replicas: int = 3,
    horizon: int = 128,
    device: Any = None,
    adaptive_replicas: bool = False,
    max_replicas: int = ADAPTIVE_REPLICA_CAP,
    se_k: float = PAIRED_SIGN_K,
) -> list[dict[str, Any]]:
    """Metric A: policy action vs champion action under champion continuation.

    Both branches share each replica's seed (common random numbers), so the
    per-replica paired differences ``policy_value - champ_value`` are the
    unit for uncertainty: each changed state reports paired mean/std/SE plus
    a confidence sign (improved/regressed/unresolved at ``se_k`` SEs).

    With ``adaptive_replicas``, volatile changed states whose paired
    interval still covers zero are extended by doubling (8 -> 16 -> 32 up to
    ``max_replicas``); same-action states never extend since their paired
    deltas are identically zero. Replica ``r`` always uses the same stable
    seed, so extension only appends new replicas deterministically.
    """

    if n_replicas < 1:
        raise ValueError("n_replicas >= 1 required")
    if max_replicas < n_replicas:
        raise ValueError("max_replicas must be >= n_replicas")

    out = []
    import torch

    for entry in entries:
        env = make_frontier_env()
        load_frontier_state(env, entry["initial_state"])
        hidden0 = _entry_continuation_hidden(champion, env, device)
        a_champ = _policy_argmax_at_entry(champion, entry, device)
        a_pol = _policy_argmax_at_entry(policy, entry, device)
        same = (a_champ.mode == a_pol.mode == 0) or (
            (a_champ.mode, a_champ.slot, a_champ.row, a_champ.col)
            == (a_pol.mode, a_pol.slot, a_pol.row, a_pol.col)
        )
        opp_spec = entry["opponent"]
        vals_champ, vals_pol = [], []

        def _run_replica(r: int) -> None:
            replica_seed = _stable_seed("sealed-choice", entry["entry_id"], r) & 0xFFFFFFFF
            oc = run_branch(env=env, hidden=hidden0,
                            branch=BranchSpec(source="c", mode=a_champ.mode, slot=a_champ.slot, row=a_champ.row, col=a_champ.col),
                            opponent_spec=opp_spec, replica_seed=replica_seed,
                            horizon=horizon, policy=champion, device=device)
            op = run_branch(env=env, hidden=hidden0,
                            branch=BranchSpec(source="p", mode=a_pol.mode, slot=a_pol.slot, row=a_pol.row, col=a_pol.col),
                            opponent_spec=opp_spec, replica_seed=replica_seed,
                            horizon=horizon, policy=champion, device=device)
            vals_champ.append(oc["value"])
            vals_pol.append(op["value"])

        for r in range(n_replicas):
            _run_replica(r)
        extended = False
        while adaptive_replicas and not same and len(vals_champ) < max_replicas:
            diffs = [p - c for p, c in zip(vals_pol, vals_champ)]
            stats = paired_delta_stats(diffs)
            if classify_paired_sign(stats["mean"], stats["se"], same_action=False, k=se_k) != "unresolved":
                break
            target = min(max_replicas, len(vals_champ) * 2)
            for r in range(len(vals_champ), target):
                _run_replica(r)
            extended = True
        diffs = [p - c for p, c in zip(vals_pol, vals_champ)]
        stats = paired_delta_stats(diffs)
        paired_mean = stats["mean"]
        sign = classify_paired_sign(
            stats["mean"], stats["se"], same_action=same, k=se_k)
        mean_c = sum(vals_champ) / len(vals_champ)
        mean_p = sum(vals_pol) / len(vals_pol)
        out.append(
            {
                "entry_id": entry["entry_id"],
                "same_action": same,
                "champ_action": [a_champ.mode, a_champ.slot, a_champ.row, a_champ.col],
                "policy_action": [a_pol.mode, a_pol.slot, a_pol.row, a_pol.col],
                "champ_mean": round(mean_c, 5),
                "policy_mean": round(mean_p, 5),
                "delta": round(paired_mean, 5),
                # Per-replica values for uncertainty reporting (8+ replica
                # standard); means above must equal these lists' means.
                "champ_values": [round(v, 5) for v in vals_champ],
                "policy_values": [round(v, 5) for v in vals_pol],
                # Paired deltas under CRN + their uncertainty. paired_mean
                # equals delta; sign is the confidence label at se_k SEs.
                "paired_deltas": [round(v, 5) for v in diffs],
                "paired_mean": round(paired_mean, 5),
                "paired_std": round(stats["std"], 5) if stats["std"] is not None else None,
                "paired_se": round(stats["se"], 5) if stats["se"] is not None else None,
                "n_replicas": len(vals_champ),
                "replicas_extended": extended,
                "sign": sign,
            }
        )
    return out


def sealed_rollout_returns(
    *,
    policy: Any,
    entries: Sequence[dict[str, Any]],
    n_replicas: int = 3,
    horizon: int = 128,
    device: Any = None,
    base_seed: int = 0,
) -> list[dict[str, Any]]:
    """Metric B: stochastic policy rollouts from sealed entries (CRN-ready seeds)."""

    import torch

    from ..v4_ppo import configure_v4_runtime
    configure_v4_runtime(policy)
    policy.eval()

    from ..model_v4 import masks_from_legal_play
    from ..v4_ppo import _v4_step_inputs, decode_v4_sim_actions
    from simulator.public_state_estimator import PublicStateEstimator

    out = []
    for entry in entries:
        rep_returns, rep_winners, rep_lens = [], [], []
        for r in range(n_replicas):
            replica_seed = (_stable_seed("sealed-rollout", base_seed, entry["entry_id"], r) & 0xFFFFFFFF)
            torch.manual_seed(replica_seed)
            try:
                torch.cuda.manual_seed_all(replica_seed)
            except Exception:
                pass
            env = make_frontier_env()
            load_frontier_state(env, entry["initial_state"])
            hidden = policy.initial_hidden(1, device=device)
            opponent = make_opponent(entry["opponent"], replica_seed)
            estimator = PublicStateEstimator()
            total = 0.0
            winner: Any = None
            ran = 0
            with torch.no_grad():
                for _ in range(horizon):
                    if env.state is None or env.state.terminal:
                        break
                    obs = env.observe_v2_for_viewer(0)
                    st = env.state
                    hand = list(st.players[0].hand[:4])
                    elixir = float(st.players[0].elixir_milli) / 1000.0
                    estimator.sync_from_env(env, viewer=0)
                    tensors = _v4_step_inputs(obs, hand, elixir, estimator.snapshot())
                    args = entry_input_tensors(tensors, device)
                    legal = torch.as_tensor(
                        np.array(obs.legal_play, dtype=bool, copy=True), device=device).unsqueeze(0).unsqueeze(0)
                    masks = masks_from_legal_play(legal, bool(tensors["legal_wait"]))
                    reset = torch.zeros((1, 1), dtype=torch.bool, device=device)
                    _, acts, _, _, _, nxt = policy.rollout_sample(
                        args["raster"], args["global_features"], args["entities"],
                        args["entity_mask"], args["hand_tokens"], args["opp_hand_probs"],
                        args["opp_out_of_cycle"], args["opp_elixir_interval"],
                        args["event_history"], masks, reset_mask=reset, hidden=hidden,
                    )
                    hidden = nxt.detach()
                    flat = acts
                    from simulator.rl.v4.model_v4 import V4ActionBatch as _AB

                    mine = decode_v4_sim_actions(
                        _AB(mode=flat.mode[:, 0], card_slot=flat.card_slot[:, 0],
                            placement=flat.placement[:, 0], wait_duration=flat.wait_duration[:, 0]),
                        player=0,
                    )[0]
                    opp = opponent.choose_action(env.engine, env.state, 1)
                    result = env.step_v2((mine, opp))
                    total += result.rewards[0]
                    ran += 1
                    if result.terminated or result.truncated:
                        winner = result.info.get("winner")
                        break
            rep_returns.append(round(total, 4))
            rep_winners.append(winner)
            rep_lens.append(ran)
        out.append(
            {
                "entry_id": entry["entry_id"],
                "mean_return": round(sum(rep_returns) / len(rep_returns), 4),
                "returns": rep_returns,
                "winners": rep_winners,
                "mean_len": round(sum(rep_lens) / len(rep_lens), 1),
            }
        )
    return out


def full_match_record(
    *,
    policy: Any,
    opponent_specs: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    device: Any = None,
    deterministic: bool = True,
    max_decisions: int = 1200,
) -> list[dict[str, Any]]:
    """Metric C: paired deterministic full matches vs the pinned pool."""

    import torch

    from ..v4_ppo import configure_v4_runtime
    configure_v4_runtime(policy)
    policy.eval()

    from simulator.roster import PLAYER_DECK

    from ..model_v4 import masks_from_legal_play
    from ..v4_ppo import _v4_step_inputs, decode_v4_sim_actions
    from simulator.public_state_estimator import PublicStateEstimator

    out = []
    for spec in opponent_specs:
        for match_seed in seeds:
            torch.manual_seed(_stable_seed("full-match", spec["archetype"], match_seed) & 0xFFFFFFFF)
            env = make_frontier_env()
            env.reset_v2(
                seed=int(match_seed),
                decks=(tuple(PLAYER_DECK), tuple(spec["deck"])),
                shuffle_decks=True,
            )
            opponent = make_opponent(
                {"strategy": spec["strategy"], "controller_seed": spec["controller_seed"]},
                int(match_seed),
            )
            hidden = policy.initial_hidden(1, device=device)
            estimator = PublicStateEstimator()
            winner: Any = None
            ran = 0
            completed = False
            with torch.no_grad():
                for _ in range(max_decisions):
                    if env.state is None or env.state.terminal:
                        break
                    obs = env.observe_v2_for_viewer(0)
                    st = env.state
                    hand = list(st.players[0].hand[:4])
                    elixir = float(st.players[0].elixir_milli) / 1000.0
                    estimator.sync_from_env(env, viewer=0)
                    tensors = _v4_step_inputs(obs, hand, elixir, estimator.snapshot())
                    args = entry_input_tensors(tensors, device)
                    legal = torch.as_tensor(
                        np.array(obs.legal_play, dtype=bool, copy=True), device=device).unsqueeze(0).unsqueeze(0)
                    masks = masks_from_legal_play(legal, bool(tensors["legal_wait"]))
                    reset = torch.zeros((1, 1), dtype=torch.bool, device=device)
                    if deterministic:
                        logits, _, hidden = policy.forward_step(args, hidden=hidden, reset_mask=reset)
                        acts = policy.act_deterministic(logits, masks)
                    else:
                        _, acts, _, _, _, nxt = policy.rollout_sample(
                            args["raster"], args["global_features"], args["entities"],
                            args["entity_mask"], args["hand_tokens"], args["opp_hand_probs"],
                            args["opp_out_of_cycle"], args["opp_elixir_interval"],
                            args["event_history"], masks, reset_mask=reset, hidden=hidden,
                        )
                        hidden = nxt.detach()
                    from simulator.rl.v4.model_v4 import V4ActionBatch as _AB

                    mine = decode_v4_sim_actions(
                        _AB(mode=acts.mode[:, 0], card_slot=acts.card_slot[:, 0],
                            placement=acts.placement[:, 0], wait_duration=acts.wait_duration[:, 0]),
                        player=0,
                    )[0]
                    opp = opponent.choose_action(env.engine, env.state, 1)
                    result = env.step_v2((mine, opp))
                    ran += 1
                    if result.terminated or result.truncated:
                        completed = bool(result.terminated) and not bool(result.truncated)
                        winner = result.info.get("winner") if completed else None
                        break
            out.append(
                {
                    "archetype": spec["archetype"],
                    "seed": int(match_seed),
                    "winner": winner,
                    "decisions": ran,
                    "result": ("win" if winner == 0 else ("loss" if winner == 1 else "draw")) if completed else "truncated",
                    "terminated": completed,
                    "runtime_version": FRONTIER_PPO_VERSION,
                }
            )
    return out
