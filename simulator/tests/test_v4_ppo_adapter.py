"""Contract tests for the V4 PPO adapter surface (model_v4 sampling/entropy).

Pins the exact behaviors PPO relies on: legal sampling, logprob equality
with re-evaluation, WAIT/PLAY factorization, reset semantics, finite
entropy/gradients, critic isolation, and fail-closed inconsistent masks.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from simulator.rl.model_v4 import (
    ModelConfigV4,
    RecurrentV4Policy,
    V4ValueHead,
    masks_from_legal_play,
)
from simulator.rl.trajectory import ActionMasks, RecurrentSequence
from simulator.rl.v4_ppo import decode_v4_sim_actions, load_frozen_anchor


def _tiny_config() -> ModelConfigV4:
    return ModelConfigV4(
        model_dim=32,
        spatial_channels=16,
        fused_dim=64,
        gru_hidden_dim=64,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ff_dim=64,
        spatial_head_dim=8,
    )


def _inputs(batch: int = 3, time: int = 1, device: str = "cpu") -> dict:
    return {
        "raster": torch.zeros(batch, time, 21, 32, 18, device=device),
        "global_features": torch.zeros(batch, time, 768, device=device),
        "entities": torch.zeros(batch, time, 128, 32, device=device),
        "entity_mask": torch.zeros(batch, time, 128, dtype=torch.bool, device=device),
        "hand_tokens": torch.zeros(batch, time, 4, 16, device=device),
        "opp_hand_probs": torch.full((batch, time, 128), 1.0 / 128, device=device),
        "opp_out_of_cycle": torch.zeros(batch, time, 128, dtype=torch.bool, device=device),
        "opp_elixir_interval": torch.zeros(batch, time, 2, device=device),
        "event_history": torch.zeros(batch, time, 16, 8, device=device),
        "reset_mask": torch.ones(batch, time, dtype=torch.bool, device=device),
    }


def _masks(batch: int = 3, wait_only_rows: tuple[int, ...] = ()) -> ActionMasks:
    legal = torch.zeros(batch, 1, 4, 32, 18, dtype=torch.bool)
    legal[:, 0, 0, :4, :4] = True
    legal[:, 0, 1, 10:12, 8:10] = True
    legal[:, 0, 2, 20:22, 4:6] = True
    legal[:, 0, 3, 25:27, 12:14] = True
    for row in wait_only_rows:
        legal[row, 0] = False
    return masks_from_legal_play(legal)


def _policy() -> RecurrentV4Policy:
    torch.manual_seed(0)
    return RecurrentV4Policy(_tiny_config())


def test_sampled_actions_always_legal() -> None:
    torch.manual_seed(1)
    policy = _policy()
    data = _inputs(batch=8)
    masks = _masks(batch=8, wait_only_rows=(0, 5))
    h0 = policy.initial_hidden(8)
    logits, _, _, _, _, _ = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], masks, reset_mask=data["reset_mask"], hidden=h0,
    )
    for _ in range(4):
        actions, _, _ = policy.sample_action(logits, masks)[:3]
        mode = actions.mode[:, 0]
        card = actions.card_slot[:, 0]
        place = actions.placement[:, 0]
        for i in range(8):
            if i in (0, 5):
                assert int(mode[i]) == 0  # nothing legal: WAIT forced by mask
            if int(mode[i]) == 0:
                continue
            assert bool(masks.card[i, 0, int(card[i])])
            assert bool(masks.placement[i, 0, int(card[i]), int(place[i][0]), int(place[i][1])])


def test_recomputed_logprob_equals_rollout_logprob() -> None:
    torch.manual_seed(2)
    policy = _policy()
    data = _inputs()
    masks = _masks()
    h0 = policy.initial_hidden(3)
    logits, actions, rollout_lp, _, _, _ = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], masks, reset_mask=data["reset_mask"], hidden=h0,
    )
    recomputed = policy.log_prob(logits, masks, actions)
    assert bool(((rollout_lp - recomputed).abs().max() < 1e-5).item())


def test_wait_play_factorization() -> None:
    torch.manual_seed(3)
    policy = _policy()
    data = _inputs()
    masks = _masks(wait_only_rows=(0,))
    h0 = policy.initial_hidden(3)
    logits, _, _, _, _, _ = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], masks, reset_mask=data["reset_mask"], hidden=h0,
    )
    actions, joint, _ = policy.sample_action(logits, masks)
    # Card/placement logits must not move WAIT-row joint logprobs.
    perturbed = type(logits)(
        mode=logits.mode,
        wait_duration=logits.wait_duration,
        card=logits.card + 50.0,
        placement=logits.placement - 50.0,
    )
    joint_perturbed = policy.log_prob(perturbed, masks, actions)
    is_wait = actions.mode == 0
    assert bool(is_wait.any())
    assert bool(
        ((joint - joint_perturbed).abs()[is_wait].max() < 1e-4).item()
    )
    # ...but the mode distribution must matter everywhere: shifting only
    # the PLAY logit rescales every row's joint logprob.
    play_shift = torch.zeros_like(logits.mode)
    play_shift[..., 1] += 5.0
    joint_mode_perturbed = policy.log_prob(
        type(logits)(
            mode=logits.mode + play_shift,
            wait_duration=logits.wait_duration,
            card=logits.card,
            placement=logits.placement,
        ),
        masks,
        actions,
    )
    assert bool(((joint - joint_mode_perturbed).abs().max() > 1e-3).item())


def test_recurrent_reset_matches_fresh_start() -> None:
    torch.manual_seed(4)
    policy = _policy()
    data = _inputs(batch=2, time=2)
    masks = _masks(batch=2)
    masks_t = type(masks)(
        mode=masks.mode.expand(2, 2, 2).contiguous(),
        card=masks.card.expand(2, 2, 4).contiguous(),
        placement=masks.placement.expand(2, 2, 4, 32, 18).contiguous(),
    )
    h0 = policy.initial_hidden(2)
    logits_full, _, _, _, _, _ = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], masks_t,
        reset_mask=torch.tensor([[True, True], [True, False]]), hidden=h0,
    )
    # Lane 0 resets at both steps, so its t=1 output must equal a fresh
    # single step.  (Lane 1 carries hidden at t=1 and must DIFFER, which
    # is asserted implicitly by the carryover test below.)
    single, _, _, _, _, _ = policy.rollout_sample(
        data["raster"][:, 1:2].contiguous(),
        data["global_features"][:, 1:2].contiguous(),
        data["entities"][:, 1:2].contiguous(),
        data["entity_mask"][:, 1:2].contiguous(),
        data["hand_tokens"][:, 1:2].contiguous(),
        data["opp_hand_probs"][:, 1:2].contiguous(),
        data["opp_out_of_cycle"][:, 1:2].contiguous(),
        data["opp_elixir_interval"][:, 1:2].contiguous(),
        data["event_history"][:, 1:2].contiguous(),
        type(masks)(
            mode=masks.mode[:, :1].contiguous(),
            card=masks.card[:, :1].contiguous(),
            placement=masks.placement[:, :1].contiguous(),
        ),
        reset_mask=torch.ones(2, 1, dtype=torch.bool), hidden=h0,
    )
    assert bool(
        ((logits_full.mode[0, 1] - single.mode[0, 0]).abs().max() < 1e-5).item()
    )
    # And lane 1 (hidden carried at t=1) must differ from a fresh start,
    # proving the plumbing actually carries state.
    assert bool(
        ((logits_full.mode[1, 1] - single.mode[1, 0]).abs().max() > 1e-4).item()
    )


def test_hidden_carryover_matches_unrolled_forward() -> None:
    torch.manual_seed(5)
    policy = _policy()
    data = _inputs(batch=2, time=3)
    masks = _masks(batch=2)
    masks_t = type(masks)(
        mode=masks.mode.expand(2, 3, 2).contiguous(),
        card=masks.card.expand(2, 3, 4).contiguous(),
        placement=masks.placement.expand(2, 3, 4, 32, 18).contiguous(),
    )
    reset = torch.zeros(2, 3, dtype=torch.bool)
    h0 = policy.initial_hidden(2)
    logits_seq, _, _, _, _, _ = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], masks_t, reset_mask=reset, hidden=h0,
    )
    # Manual carryover one step at a time must reproduce the unrolled pass.
    hidden = h0
    modes = []
    for t in range(3):
        sl = lambda x: x[:, t : t + 1].contiguous()  # noqa: E731
        mt = type(masks)(
            mode=masks.mode[:, :1].contiguous(),
            card=masks.card[:, :1].contiguous(),
            placement=masks.placement[:, :1].contiguous(),
        )
        logits_t, _, _, _, _, hidden = policy.rollout_sample(
            sl(data["raster"]), sl(data["global_features"]), sl(data["entities"]),
            sl(data["entity_mask"]), sl(data["hand_tokens"]), sl(data["opp_hand_probs"]),
            sl(data["opp_out_of_cycle"]), sl(data["opp_elixir_interval"]),
            sl(data["event_history"]), mt,
            reset_mask=torch.zeros(2, 1, dtype=torch.bool), hidden=hidden,
        )
        modes.append(logits_t.mode[:, 0])
    for t in range(3):
        assert bool(
            ((logits_seq.mode[:, t] - modes[t]).abs().max() < 1e-5).item()
        )


def test_entropy_finite_on_degenerate_batches() -> None:
    torch.manual_seed(6)
    policy = _policy()
    data = _inputs(batch=4)
    # All rows WAIT-only (broke) plus one normal row mixture.
    masks = _masks(batch=4, wait_only_rows=(0, 1, 2, 3))
    h0 = policy.initial_hidden(4)
    logits, _, _, entropy, _, _ = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], masks, reset_mask=data["reset_mask"], hidden=h0,
    )
    for key, value in entropy.items():
        assert bool(torch.isfinite(value).all()), key
        assert bool((value >= 0.0).all()), key


def test_inconsistent_masks_fail_closed_to_wait() -> None:
    torch.manual_seed(7)
    policy = _policy()
    data = _inputs(batch=2)
    legal = torch.zeros(2, 1, 4, 32, 18, dtype=torch.bool)
    legal[:, 0, 0, :2, :2] = True
    legal[:, 0, 1, :2, :2] = True
    legal[:, 0, 2, :2, :2] = True
    legal[:, 0, 3, :2, :2] = True
    masks = masks_from_legal_play(legal)
    # Corrupt: claim PLAY legal but no card legal on row 0.
    bad_card = masks.card.clone()
    bad_card[0, 0] = False
    bad = ActionMasks(mode=masks.mode, card=bad_card, placement=masks.placement)
    h0 = policy.initial_hidden(2)
    logits, actions, joint, entropy, _, _ = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], bad, reset_mask=data["reset_mask"], hidden=h0,
    )
    assert int(actions.mode[0, 0]) == 0  # forced WAIT, still legal
    assert bool(torch.isfinite(joint).all())
    assert bool(torch.isfinite(entropy["joint"]).all())


def test_objective_gradients_finite() -> None:
    torch.manual_seed(8)
    from simulator.rl.objectives import PPOObjectiveConfig, ppo_objective

    policy = _policy()
    critic = V4ValueHead(64)
    data = _inputs(batch=4, time=2)
    masks = _masks(batch=4)
    masks_t = type(masks)(
        mode=masks.mode.expand(4, 2, 2).contiguous(),
        card=masks.card.expand(4, 2, 4).contiguous(),
        placement=masks.placement.expand(4, 2, 4, 32, 18).contiguous(),
    )
    h0 = policy.initial_hidden(4)
    logits, actions, old_lp, entropy, ht, h1 = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], masks_t,
        reset_mask=torch.zeros(4, 2, dtype=torch.bool), hidden=h0,
    )
    values = critic(ht.detach())
    advantages = torch.randn(4, 2)
    returns = advantages + values.detach()
    new_lp = policy.log_prob(logits, masks_t, actions)
    anchor_lp = policy.log_prob(logits, masks_t, actions).detach()
    result = ppo_objective(
        old_log_probs=old_lp.detach(),
        new_log_probs=new_lp,
        advantages=advantages,
        values=values,
        returns=returns,
        entropy=entropy["joint"],
        old_values=values.detach(),
        behavior_cloning_log_probs=anchor_lp,
        config=PPOObjectiveConfig(bc_coef=0.1),
    )
    result.total_loss.backward()
    for param in (*policy.parameters(), *critic.parameters()):
        if param.grad is not None:
            assert bool(torch.isfinite(param.grad).all())


def test_critic_isolated_from_actor_inference() -> None:
    torch.manual_seed(9)
    policy = _policy()
    critic = V4ValueHead(64)
    data = _inputs()
    masks = _masks()
    h0 = policy.initial_hidden(3)
    logits_a, _, _, _, _, _ = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], masks, reset_mask=data["reset_mask"], hidden=h0,
    )
    _ = critic(torch.randn(3, 1, 64))
    torch.manual_seed(9)
    logits_b, _, _, _, _, _ = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], masks, reset_mask=data["reset_mask"], hidden=h0,
    )
    assert bool(((logits_a.mode - logits_b.mode).abs().max() == 0.0).item())
    # Critic training on detached features must not move actor parameters.
    before = [p.detach().clone() for p in policy.parameters()]
    ht = torch.randn(3, 1, 64, requires_grad=True)
    loss = critic(ht).square().mean()
    opt = torch.optim.SGD(critic.parameters(), lr=0.1)
    opt.zero_grad()
    loss.backward()
    opt.step()
    for got, want in zip(policy.parameters(), before):
        assert bool(torch.equal(got.detach(), want))


def test_trajectory_v4_fields_validate() -> None:
    batch, time = 2, 3
    base = {
        "raster": torch.zeros(batch, time, 21, 32, 18),
        "global_features": torch.zeros(batch, time, 768),
        "entities": torch.zeros(batch, time, 128, 32),
        "entity_mask": torch.zeros(batch, time, 128, dtype=torch.bool),
        "reset_mask": torch.ones(batch, time, dtype=torch.bool),
    }
    full = dict(
        base,
        hand_tokens=torch.zeros(batch, time, 4, 16),
        opp_hand_probs=torch.full((batch, time, 128), 1.0 / 128),
        opp_out_of_cycle=torch.zeros(batch, time, 128, dtype=torch.bool),
        opp_elixir_interval=torch.zeros(batch, time, 2),
        event_history=torch.zeros(batch, time, 16, 8),
    )
    seq = RecurrentSequence(**full)
    assert seq.hand_tokens is not None and seq.event_history is not None
    seq_min = RecurrentSequence(**base)  # prototype path unaffected
    assert seq_min.hand_tokens is None
    with pytest.raises(ValueError, match="hand_tokens"):
        RecurrentSequence(**dict(full, hand_tokens=torch.zeros(batch, time, 4, 15)))


def _lane(source: str, archetype: str, seed: int) -> dict:
    from simulator.engine import BattleEngine
    from simulator.engine.match import DeterministicCycleController
    from simulator.env import SimulatorEnv
    from simulator.public_state_estimator import PublicStateEstimator
    from simulator.rl.basic_scenarios import BasicMechanicsScenarioEnv, BasicScenarioConfig
    from simulator.rl.opponent_pool import OpponentPool
    from simulator.roster import PLAYER_DECK
    from simulator.ruleset import load_fixed_ruleset

    ruleset = load_fixed_ruleset()
    opponent = OpponentPool(ruleset, seed=seed + 100).sample(
        0, archetype=archetype, strategy="deterministic-cycle"
    )
    base = SimulatorEnv(
        engine=BattleEngine(ruleset, validate_every_tick=False),
        decision_interval_us=250_000,
    )
    env = BasicMechanicsScenarioEnv(
        base, BasicScenarioConfig(source=source, target_player=0, decision_limit=16)
    )
    decks = (tuple(PLAYER_DECK), tuple(opponent.deck.cards))
    env.reset_v2(seed=seed, decks=decks, shuffle_decks=True)
    return {
        "env": env,
        "opponent": DeterministicCycleController(lane="alternate"),
        "estimator": PublicStateEstimator(),
        "decks": decks,
        "next_seed": seed + 1000,
        "needs_reset": True,
    }


def test_decode_v4_sim_actions_types() -> None:
    from simulator.actions import PlayCardAction, WaitAction
    from simulator.rl.model_v4 import V4ActionBatch

    actions = V4ActionBatch(
        mode=torch.tensor([0, 1]),
        card_slot=torch.tensor([0, 2]),
        placement=torch.tensor([[0, 0], [17, 3]]),
        wait_duration=torch.tensor([1, 0]),
    )
    decoded = decode_v4_sim_actions(actions, player=0)
    assert isinstance(decoded[0], WaitAction) and decoded[0].player == 0
    assert isinstance(decoded[1], PlayCardAction)
    assert decoded[1].player == 0 and decoded[1].card_slot == 2
    assert tuple(decoded[1].cell) == (3, 17)


def test_load_frozen_anchor_freezes_and_verifies(tmp_path) -> None:
    torch.manual_seed(11)
    policy = RecurrentV4Policy(_tiny_config())
    path = tmp_path / "anchor.pt"
    torch.save(policy.state_dict(), path)
    anchor = load_frozen_anchor(str(path), _tiny_config())
    assert not anchor.training
    assert all(not p.requires_grad for p in anchor.parameters())
    assert hasattr(anchor, "anchor_sha256")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_anchor(str(path), _tiny_config(), expected_sha256="0" * 64)
    reloaded = load_frozen_anchor(
        str(path), _tiny_config(), expected_sha256=anchor.anchor_sha256
    )
    assert not reloaded.training


def test_v4_ppo_dry_run_collect_and_update() -> None:
    """End-to-end wiring proof on live sims: collect, update, stay finite."""

    from simulator.rl.v4_ppo import (
        collect_v4_rollout,
        load_frozen_anchor,
        update_v4_ppo,
    )

    torch.manual_seed(21)
    policy = RecurrentV4Policy(_tiny_config())
    critic = V4ValueHead(64)
    anchor_path = "/tmp/v4_ppo_dryrun_anchor.pt"
    torch.save(policy.state_dict(), anchor_path)
    anchor = load_frozen_anchor(anchor_path, _tiny_config())
    anchor_before = [p.detach().clone() for p in anchor.parameters()]
    lanes = [
        _lane("ground-defense", "beatdown", 5),
        _lane("isolated-offense", "aggressive-pressure", 6),
    ]
    actor_opt = torch.optim.Adam(policy.parameters(), lr=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4)
    try:
        batch, info = collect_v4_rollout(
            policy, critic, anchor, lanes, n_decisions=4, device=torch.device("cpu")
        )
    finally:
        import os

        os.remove(anchor_path)
    assert info["illegal_attempts"] == 0
    assert 0.0 <= info["wait_fraction"] <= 1.0
    assert batch.trajectory.sequence.hand_tokens is not None
    assert batch.trajectory.sequence.event_history is not None
    assert batch.anchor_actions is not None
    report = update_v4_ppo(
        policy, critic, actor_opt, critic_opt, batch, ppo_epochs=1, bc_coef=0.05
    )
    assert bool(torch.isfinite(torch.tensor(report.policy_loss)))
    assert bool(torch.isfinite(torch.tensor(report.value_loss)))
    assert bool(torch.isfinite(torch.tensor(report.behavior_cloning_loss)))
    assert isinstance(report.quarantined, bool) and isinstance(
        report.quarantine_reasons, list
    )
    for param, before in zip(anchor.parameters(), anchor_before):
        assert bool(torch.equal(param.detach(), before))
    assert set(report.per_head_entropy) >= {"joint", "mode", "duration", "card", "placement"}
    assert set(report.grad_norms) >= {"mode", "card", "placement"}
    assert bool(torch.isfinite(torch.tensor(report.policy_grad_norm)))
    assert bool(torch.isfinite(torch.tensor(report.bc_grad_norm)))
    assert report.policy_grad_norm >= 0.0 and report.bc_grad_norm >= 0.0
    assert 0.0 <= (report.anchor_mode_agree or 0.0) <= 1.0
    assert bool(torch.isfinite(torch.tensor(report.returns_mean)))
    assert bool(torch.isfinite(torch.tensor(report.advantages_std)))


def test_log_prob_finite_on_broke_wait_rows() -> None:
    """WAIT rows naming an illegal dummy slot must score finite, not NaN.

    Regression test for the ``-inf * is_play=0`` forward NaN: gathering an
    illegal slot's -inf and zeroing it with the WAIT gate poisons the value
    even though the row contributes nothing.  Low-elixir states are core
    curriculum, so PPO logprobs must be finite there.
    """

    torch.manual_seed(12)
    policy = _policy()
    data = _inputs(batch=4)
    # Rows 0-1: WAIT-only with nothing legal anywhere (fully broke).
    masks = _masks(batch=4, wait_only_rows=(0, 1))
    h0 = policy.initial_hidden(4)
    logits, actions, joint, _, _, _ = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], masks, reset_mask=data["reset_mask"], hidden=h0,
    )
    assert int(actions.mode[0, 0]) == 0 and int(actions.mode[1, 0]) == 0
    assert bool(torch.isfinite(joint).all())
    recomputed = policy.log_prob(logits, masks, actions)
    assert bool(torch.isfinite(recomputed).all())
    assert bool(((joint - recomputed).abs().max() < 1e-5).item())


def _forward_logits(policy, batch: int = 3):
    data = _inputs(batch=batch)
    masks = _masks(batch=batch)
    h0 = policy.initial_hidden(batch)
    logits, _, _, _, _, _ = policy.rollout_sample(
        data["raster"], data["global_features"], data["entities"],
        data["entity_mask"], data["hand_tokens"], data["opp_hand_probs"],
        data["opp_out_of_cycle"], data["opp_elixir_interval"],
        data["event_history"], masks, reset_mask=data["reset_mask"], hidden=h0,
    )
    return logits, masks


def test_per_head_kl_identity_and_shift() -> None:
    from simulator.rl.v4_ppo import per_head_kl

    torch.manual_seed(13)
    policy = _policy()
    logits, masks = _forward_logits(policy)
    same = per_head_kl(
        {"mode": logits.mode, "duration": logits.wait_duration,
         "card": logits.card, "placement": logits.placement},
        logits, masks,
    )
    for key, value in same.items():
        assert bool(torch.isfinite(value).all()), key
        assert bool((value.abs().max() < 1e-5).item()), key
    # Temperature scaling barely moves near-uniform maps; spike one legal
    # cell per slot instead for a decisive distribution shift.
    cells = 32 * 18
    flat_mask = masks.placement.reshape(3, 1, 4, cells)
    first_cell = flat_mask.float().argmax(dim=-1, keepdim=True)
    spike_flat = torch.zeros(3, 1, 4, cells)
    spike_flat.scatter_(-1, first_cell.clamp(0, cells - 1), 6.0)
    spike = torch.where(
        flat_mask, spike_flat, torch.zeros_like(spike_flat)
    ).reshape_as(logits.placement)
    spiked = type(logits)(
        mode=logits.mode,
        wait_duration=logits.wait_duration,
        card=logits.card,
        placement=logits.placement + spike,
    )
    moved = per_head_kl(
        {"mode": logits.mode, "duration": logits.wait_duration,
         "card": logits.card, "placement": logits.placement},
        spiked, masks,
    )
    assert float(moved["mode"].mean()) < 1e-5
    assert float(moved["placement"].mean()) > 1e-3


def test_per_head_kl_finite_with_illegal_entries() -> None:
    """KL over masked heads must stay finite (0-mass entries contribute 0).

    Regression test: the first smoke reported NaN mode/card KL because
    0 * (-inf - -inf) on illegal entries was never guarded.
    """

    from simulator.rl.v4_ppo import per_head_kl

    torch.manual_seed(14)
    policy = _policy()
    logits, masks = _forward_logits(policy)
    # Force illegal entries: WAIT-only mode row and empty card slots.
    mode = masks.mode.clone()
    mode[0, 0, 1] = False
    card = masks.card.clone()
    card[1, 0] = False
    placement = masks.placement.clone()
    placement[1, 0] = False
    sparse = ActionMasks(mode=mode, card=card, placement=placement)
    shifted = type(logits)(
        mode=logits.mode + torch.randn_like(logits.mode) * 0.5,
        wait_duration=logits.wait_duration,
        card=logits.card + torch.randn_like(logits.card) * 0.5,
        placement=logits.placement + torch.randn_like(logits.placement) * 0.5,
    )
    moved = per_head_kl(
        {"mode": logits.mode, "duration": logits.wait_duration,
         "card": logits.card, "placement": logits.placement},
        shifted, sparse,
    )
    for key, value in moved.items():
        assert bool(torch.isfinite(value).all()), key


def test_anchor_agreement_metrics() -> None:
    from simulator.rl.model_v4 import V4ActionBatch
    from simulator.rl.v4_ppo import _anchor_agreement

    post = V4ActionBatch(
        mode=torch.tensor([0, 1, 1, 1]),
        card_slot=torch.tensor([0, 2, 2, 1]),
        placement=torch.tensor([[0, 0], [5, 5], [5, 6], [9, 9]]),
        wait_duration=torch.tensor([1, 0, 0, 0]),
    )
    anchor = V4ActionBatch(
        mode=torch.tensor([0, 1, 1, 0]),
        card_slot=torch.tensor([0, 2, 1, 0]),
        placement=torch.tensor([[0, 0], [5, 5], [0, 0], [0, 0]]),
        wait_duration=torch.tensor([1, 0, 0, 2]),
    )
    result = _anchor_agreement(post, anchor)
    assert result["mode"] == 0.75
    assert result["card"] == 0.5  # rows 1,2 PLAY both sides; slot differs on row 2
    assert result["place"] == 1.0  # only row 1 same-card PLAY; exact cell


def test_decode_drift_tracks_duration_flips() -> None:
    from simulator.rl.model_v4 import V4ActionBatch
    from simulator.rl.v4_ppo import _decode_drift

    first = V4ActionBatch(
        mode=torch.tensor([0, 0, 1, 0]),
        card_slot=torch.tensor([0, 0, 2, 0]),
        placement=torch.tensor([[0, 0], [0, 0], [5, 5], [0, 0]]),
        wait_duration=torch.tensor([1, 1, 0, 2]),
    )
    second = V4ActionBatch(
        mode=torch.tensor([0, 0, 1, 0]),
        card_slot=torch.tensor([0, 0, 2, 0]),
        placement=torch.tensor([[0, 0], [0, 0], [5, 5], [0, 0]]),
        wait_duration=torch.tensor([1, 3, 0, 2]),
    )
    drift = _decode_drift(first, second)
    assert drift["mode_flip"] == 0.0
    assert drift["card_flip"] == 0.0
    assert drift["placement_move"] == 0.0
    assert drift["duration_flip"] == pytest.approx(1 / 3)


def test_potential_anneal_schedule() -> None:
    from simulator.rl.v4_ppo import potential_anneal_weight

    assert potential_anneal_weight(0.1, 0, 5) == 0.1
    assert potential_anneal_weight(0.1, 4, 5) == 0.0
    assert potential_anneal_weight(0.1, 2, 5) == 0.05
    assert potential_anneal_weight(0.1, 2, 5, anneal=False) == 0.1
    assert potential_anneal_weight(0.0, 0, 1) == 0.0
    with pytest.raises(ValueError, match="non-negative integer"):
        potential_anneal_weight(0.1, -1, 5)
    with pytest.raises(ValueError, match="finite non-negative"):
        potential_anneal_weight(float("nan"), 0, 5)


def test_deterministic_collect_is_reproducible() -> None:
    from simulator.rl.v4_ppo import collect_v4_rollout

    def run_once() -> list:
        torch.manual_seed(31)
        policy = RecurrentV4Policy(_tiny_config())
        critic = V4ValueHead(64)
        lanes = [
            _lane("ground-defense", "beatdown", 5),
            _lane("isolated-offense", "aggressive-pressure", 6),
        ]
        batch, _ = collect_v4_rollout(
            policy, critic, None, lanes, n_decisions=4,
            device=torch.device("cpu"), deterministic_policy=True,
        )
        return [
            batch.trajectory.actions.mode.tolist(),
            batch.trajectory.actions.card_slot.tolist(),
            batch.trajectory.old_log_probs.tolist(),
        ]

    first, second = run_once(), run_once()
    assert first == second


def test_monte_carlo_returns_and_distribution() -> None:
    from simulator.rl.v4_ppo import monte_carlo_returns, return_distribution

    rewards = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    terminated = torch.tensor(
        [[False, True, False], [False, False, False]]
    )
    returns, stats = monte_carlo_returns(rewards, terminated, gamma=1.0)
    assert torch.allclose(returns[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(returns[1], torch.tensor([2.0, 2.0, 2.0]))
    # Only lane 1's pre-cut steps depend on post-cut future (2 of 6 steps).
    assert stats["truncated_tail_fraction"] == pytest.approx(1 / 3)
    dist = return_distribution(returns)
    assert dist["n"] == 6
    assert dist["median"] == 1.5
    assert dist["positive_fraction"] == pytest.approx(2 / 3)
    assert dist["near_zero_fraction"] == pytest.approx(1 / 3)
    assert dist["negative_fraction"] == 0.0
    with pytest.raises(ValueError, match="gamma"):
        monte_carlo_returns(rewards, terminated, gamma=0.0)


def test_fit_value_head_learns_linear_target() -> None:
    from simulator.rl.v4_ppo import V4ValueHead, fit_value_head

    torch.manual_seed(3)
    critic = V4ValueHead(8)
    mapping = torch.randn(8)
    features = torch.randn(48, 1, 8)
    targets = (features.squeeze(1) @ mapping).unsqueeze(1)
    held_features = torch.randn(16, 1, 8)
    held_targets = (held_features.squeeze(1) @ mapping).unsqueeze(1)
    before = float(((critic(features) - targets) ** 2).mean())
    report = fit_value_head(
        critic, features, targets,
        heldout_recurrent=held_features, heldout_targets=held_targets,
        lr=1e-2, epochs=10, seed=0,
    )
    after = float(((critic(features) - targets) ** 2).mean())
    assert after < before
    assert report["best"]["epoch"] >= 0
    assert "heldout_ev" in report["history"][-1]
    assert report["history"][-1]["heldout_loss"] >= 0.0
