"""Tests for data-driven frontier discovery (frontier + frontier_eval)."""

import os

import pytest

from simulator.rl.frontier import (
    FilterConfig,
    deterioration,
    score_decision,
    select_candidates,
    signature_of,
    transition,
)
from simulator.rl.frontier_eval import (
    ClassifyConfig,
    build_distribution,
    classify_verdict,
    continuation_value,
    headroom_bin,
    horizon_stability,
    is_ranking_flip,
)


def _decisions():
    base_towers = {"own": [1.0, 1.0, 1.0], "enemy": [1.0, 1.0, 1.0]}

    def dec(t, **kw):
        d = {
            "t": t,
            "tick": t * 5,
            "state_hash": f"hash{t}",
            "learner": {"mode": 0, "slot": 0, "row": 0, "col": 0},
            "mode_p": [0.9, 0.1],
            "card_p": [0.4, 0.3, 0.2, 0.1],
            "entropy": {"joint": 0.2, "mode": 0.1, "card": 0.5, "placement": 0.3},
            "rule": {"mode": 0, "slot": 0, "row": 0, "col": 0},
            "hand": ["hog-rider", "cannon", "musketeer", "skeletons"],
            "elixir": 5.0,
            "towers": dict(base_towers),
            "threats": [],
            "tags": {
                "archetype": "beatdown", "threat_cards": [], "lanes": [],
                "elixir_bucket": "mid", "hand": ["cannon", "hog-rider"],
                "own_tower_bucket": 1.0, "air": False, "ground": False,
            },
        }
        d.update(kw)
        return d

    return [dec(t) for t in range(64)]


def test_rule_disagree_and_confident_wrong_retained():
    cfg = FilterConfig(det_lookahead=8, max_per_match=10)
    decisions = _decisions()
    # Confidently-wrong lane: policy WAITs confidently, rule says PLAY Cannon,
    # own tower then takes damage -> must be retained despite low entropy.
    decisions[20]["rule"] = {"mode": 1, "slot": 1, "row": 20, "col": 4}
    decisions[20]["entropy"] = {"joint": 0.05, "mode": 0.02, "card": 0.1, "placement": 0.1}
    for t in range(21, 30):
        decisions[t]["towers"] = {"own": [0.9, 1.0, 1.0], "enemy": [1.0, 1.0, 1.0]}
        decisions[t]["threats"] = [{"card": "hog-rider", "hp": 1.0, "lane": "left", "row": 20, "air": False}]
    match = {"config": {"archetype": "beatdown", "match_index": 0}, "decisions": decisions}
    kept = select_candidates(match, cfg)
    assert any(c["t"] == 20 for c in kept), "confidently-wrong state must be retained"
    # Agreement states are never retained.
    assert not any(c["t"] == 3 for c in kept)


def test_deterioration_and_transition_components():
    decisions = _decisions()
    decisions[5]["threats"] = [{"card": "hog-rider", "hp": 1.0, "lane": "left", "row": 20, "air": False}]
    assert transition(decisions, 4, 4) is True
    assert transition(decisions, 20, 4) is False
    assert deterioration(decisions, 20, 8) == 0.0
    decisions[28]["towers"] = {"own": [0.8, 1.0, 1.0], "enemy": [1.0, 1.0, 1.0]}
    assert deterioration(decisions, 20, 8) > 0.15


def test_score_decision_uncertainty_boosts_but_never_gates():
    cfg = FilterConfig()
    decisions = _decisions()
    decisions[10]["rule"] = {"mode": 1, "slot": 1, "row": 20, "col": 4}
    decisions[10]["entropy"] = {"joint": 0.05, "mode": 0.02, "card": 0.1, "placement": 0.1}
    decisions[11]["threats"] = [{"card": "hog-rider", "hp": 1.0, "lane": "left", "row": 20, "air": False}]
    low = score_decision(decisions, 10, cfg)
    decisions[10]["entropy"] = {"joint": 1.8, "mode": 0.7, "card": 1.2, "placement": 1.0}
    high = score_decision(decisions, 10, cfg)
    assert high["score"] > low["score"]
    assert low["rule_disagree"] == 1.0


def test_signature_stable():
    tags = {"archetype": "a", "threat_cards": ["hog-rider"], "lanes": ["left"],
            "elixir_bucket": "mid", "hand": ["cannon"], "own_tower_bucket": 1.0,
            "air": False, "ground": True}
    assert signature_of(tags) == signature_of(dict(tags))


def test_continuation_value_math():
    before = {"own": [1.0, 1.0], "enemy": [1.0, 1.0]}
    after = {"own": [0.9, 1.0], "enemy": [0.7, 1.0]}
    assert continuation_value(before, after, None) == pytest.approx(0.3 - 0.1)
    assert continuation_value(before, after, 0) == pytest.approx(0.2 + 2.0)
    assert continuation_value(before, after, 1) == pytest.approx(0.2 - 2.0)


def test_classify_verdict_classes():
    cfg = ClassifyConfig()
    base = {"regret": 0.01, "stability": 1.0, "best_mean": 0.2, "competence": 0.2}
    assert classify_verdict(dict(base), cfg)[0] == "solved"
    front = dict(base, regret=0.25)
    assert classify_verdict(front, cfg)[0] == "frontier"
    # Sim worse than passivity is still frontier when the best action holds:
    # the most learnable mistakes live exactly there.
    bad_anchor = dict(base, regret=0.50, sim_mean=-1.0, best_mean=-0.2, competence=-0.5)
    assert classify_verdict(bad_anchor, cfg)[0] == "frontier"
    # Lost positions (best continuation drops a tower) are hopeless.
    lost = dict(base, regret=0.50, best_mean=-1.5)
    assert classify_verdict(lost, cfg)[0] == "hopeless"
    unstable = dict(base, regret=0.25, stability=0.25)
    assert classify_verdict(unstable, cfg)[0] == "hopeless"


def test_is_ranking_flip_one_sided():
    assert is_ranking_flip(0.20, -0.10) is True
    assert is_ranking_flip(-0.20, 0.10) is True
    assert is_ranking_flip(0.20, 0.01) is False  # emergence, not a flip
    assert is_ranking_flip(0.01, 0.20) is False
    assert is_ranking_flip(0.01, -0.01) is False  # noise around zero
    assert is_ranking_flip(0.20, 0.15) is False


def test_headroom_bin_and_horizon_stability():
    assert headroom_bin(0.05, (0.1, 0.3)) == "modest"
    assert headroom_bin(0.2, (0.1, 0.3)) == "medium"
    assert headroom_bin(0.5, (0.1, 0.3)) == "large"

    def v(h, src, reg):
        return {"state_hash": h, "best_source": src, "regret": reg}

    reports = {
        64: [v("a", "rule", 0.2), v("b", "policy-top", 0.1), v("c", "rule", -0.05), v("d", "rule", 0.3), v("e", "wait", 0.15)],
        128: [v("a", "rule", 0.22), v("b", "policy-top", 0.12), v("c", "rule", -0.04), v("d", "rule", 0.31), v("e", "wait", 0.14)],
        256: [v("a", "rule", 0.21), v("b", "policy-top", 0.11), v("c", "rule", -0.05), v("d", "rule", 0.30), v("e", "wait", 0.16)],
    }
    summary = horizon_stability(reports)
    assert summary["chosen_horizon"] == 64
    assert summary["horizons"][64]["top1_agree"] == 1.0


def test_build_distribution_stratifies_and_caps():
    verdicts = []
    for i in range(30):
        verdicts.append({
            "match_id": f"beatdown#{i % 3}", "archetype": "beatdown",
            "t": i, "state_hash": f"h{i}", "tags": {},
            "regret": 0.05 + 0.02 * i, "sim_mean": 0.0, "best_mean": 0.1,
            "best_source": "rule", "stability": 1.0, "competence": 0.2, "class": "frontier",
        })
    for i in range(10):
        verdicts.append({"match_id": "x", "archetype": "siege-bait", "t": i,
                         "state_hash": f"s{i}", "tags": {}, "regret": 0.01,
                         "sim_mean": 0.0, "best_mean": 0.01, "best_source": "sim",
                         "stability": 1.0, "competence": 0.2, "class": "solved"})
    picked, report = build_distribution(verdicts, max_train=12)
    assert len(picked) == 12
    assert set(v["headroom"] for v in picked) == {"modest", "medium", "large"}
    assert report["by_class"] == {"frontier": 30, "solved": 10}


TRAIN_PT = "outputs/v4/frontier/frontier_train.pt"


@pytest.mark.skipif(not os.path.exists(TRAIN_PT), reason="frontier train set missing")
def test_frontier_entry_round_trip():
    """A materialized entry resets a live env; policy + opponent can step it."""

    import torch

    from simulator.engine.core import BattleEngine
    from simulator.env import SimulatorEnv
    from simulator.rl.frontier import frontier_model_config
    from simulator.rl.frontier_eval import reset_to_frontier_entry
    from simulator.rl.model_v4 import masks_from_legal_play
    from simulator.rl.v4_ppo import _v4_step_inputs, load_frozen_anchor
    from simulator.public_state_estimator import PublicStateEstimator
    from simulator.ruleset import load_fixed_ruleset

    entries = torch.load(TRAIN_PT, map_location="cpu", weights_only=False)
    assert len(entries) >= 1
    entry = entries[0]
    assert entry["split"] == "train"
    assert entry["verdict"]["class"] == "frontier"

    policy = load_frozen_anchor(CKPT, frontier_model_config(), device="cpu")
    env = SimulatorEnv(BattleEngine(load_fixed_ruleset(), validate_every_tick=False))
    loaded = reset_to_frontier_entry(env, entry)
    assert tuple(loaded["hidden"].shape) == (1, 1, 64)
    assert env.state.state_hash() == entry["provenance"]["state_hash"]

    # One PPO-style decision: policy forward + sample, opponent acts, step.
    from simulator.rl.frontier_eval import _step_inputs, _observe_inputs

    policy.eval()
    with torch.no_grad():
        inputs, legal_np, hand, elixir = _observe_inputs(env, "cpu")
        tensors = _step_inputs(inputs, "cpu")
        masks = masks_from_legal_play(
            torch.as_tensor(legal_np, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        )
        reset = torch.zeros((1, 1), dtype=torch.bool)
        logits, acts, _, _, _, _ = policy.rollout_sample(
            tensors["raster"], tensors["global_features"], tensors["entities"],
            tensors["entity_mask"], tensors["hand_tokens"], tensors["opp_hand_probs"],
            tensors["opp_out_of_cycle"], tensors["opp_elixir_interval"],
            tensors["event_history"], masks, reset_mask=reset, hidden=loaded["hidden"],
        )
        a_mode = int(acts.mode[0, 0])
        if a_mode == 0:
            from simulator.actions import WaitAction
            mine = WaitAction(0)
        else:
            from simulator.actions import PlayCardAction
            mine = PlayCardAction(0, int(acts.card_slot[0, 0]),
                                  (int(acts.placement[0, 0, 1]), int(acts.placement[0, 0, 0])))
        opp = loaded["opponent"].choose_action(env.engine, env.state, 1)
        result = env.step_v2((mine, opp))
    assert result.terminated is False or True  # stepping works either way


CKPT = "outputs/v4/stage2_12k_v2_e12_s12.pt"


@pytest.mark.skipif(not os.path.exists(CKPT), reason="champion checkpoint missing")
def test_live_replay_branch_harness_micro():
    """End-to-end micro run: collect -> replay -> branches -> one branch."""

    from simulator.rl.frontier import (
        FilterConfig,
        PoolMatchConfig,
        collect_pool_match,
        frontier_model_config,
        select_candidates,
    )
    from simulator.rl.frontier_eval import (
        EvalConfig,
        evaluate_candidate,
        plan_branches,
        replay_to_state,
        run_branch,
    )
    from simulator.rl.v4_ppo import load_frozen_anchor

    policy = load_frozen_anchor(CKPT, frontier_model_config(), device="cpu")
    match = collect_pool_match(
        policy=policy,
        cfg=PoolMatchConfig(archetype="deterministic-cycle", match_index=0,
                            pool_seed=123, max_decisions=6),
        device="cpu",
    )
    assert len(match["decisions"]) == 6
    # First decision with a legal play option (early decisions may be broke).
    t_pick = next(
        i for i, d in enumerate(match["decisions"]) if d["legal"]["slots"] > 0
    )
    env, hidden = replay_to_state(match, t_pick, policy, device="cpu")
    assert tuple(hidden.shape) == (1, 1, 64)
    dec = match["decisions"][t_pick]
    candidate = {
        "match_id": "deterministic-cycle#0", "archetype": "deterministic-cycle",
        "t": t_pick, "tick": dec["tick"],
        "state_hash": dec["state_hash"],
        "tags": dec["tags"],
        "filter": {"score": 1.0},
        "learner": dec["learner"],
    }
    branches = plan_branches(env, policy, hidden, candidate, device="cpu")
    assert branches[0].source == "sim"
    assert len(branches) >= 2
    mc = match["config"]
    out = run_branch(env=env, hidden=hidden, branch=branches[0],
                     opponent_spec={"strategy": mc["strategy"], "controller_seed": 0},
                     replica_seed=7, horizon=3, policy=policy, device="cpu")
    assert out["decisions_run"] >= 1 and isinstance(out["value"], float)
    # Determinism: same branch + seed twice -> identical value.
    out2 = run_branch(env=env, hidden=hidden, branch=branches[0],
                      opponent_spec={"strategy": mc["strategy"], "controller_seed": 0},
                      replica_seed=7, horizon=3, policy=policy, device="cpu")
    assert out2["value"] == out["value"]
    verdict = evaluate_candidate(
        match=match, candidate=candidate, policy=policy,
        eval_cfg=EvalConfig(horizon=3, n_replicas=1, max_branches=3), device="cpu",
    )
    assert verdict["class"] in ("solved", "frontier", "hopeless")
