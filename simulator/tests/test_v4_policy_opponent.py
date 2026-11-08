"""Frozen model metadata, public legality and episode RNG isolation."""

from dataclasses import asdict, replace
import hashlib

import pytest

torch = pytest.importorskip("torch")

from simulator.actions import PlayCardAction
from simulator.rl.v4.frontier_ppo.environment import make_frontier_env
from simulator.rl.v4.model_v4 import ModelConfigV4, RecurrentV4Policy, masks_from_legal_play
from simulator.rl.v4.policy_opponent import V4PolicyOpponent, _sample_action
from simulator.roster import PLAYER_DECK


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def checkpoint(tmp_path):
    config = ModelConfigV4(
        model_dim=8, spatial_channels=4, fused_dim=16, gru_hidden_dim=16,
        transformer_heads=2, transformer_layers=1, transformer_ff_dim=16,
        spatial_head_dim=4,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        actor = RecurrentV4Policy(config)
        path = tmp_path / "small-actor.pt"
        torch.save(actor.state_dict(), path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), config


def _env():
    env = make_frontier_env()
    env.reset_v2(seed=7, decks=(PLAYER_DECK, PLAYER_DECK), shuffle_decks=True)
    return env


def test_stochastic_opponents_share_only_frozen_weights_and_keep_rng_private(checkpoint):
    path, digest, config = checkpoint
    env = _env()
    global_state = torch.get_rng_state().clone()
    opponents = [
        V4PolicyOpponent(env, str(path), digest, model_config=asdict(config), inference="stochastic", seed=seed)
        for seed in (42, 42, 43)
    ]
    assert torch.equal(torch.get_rng_state(), global_state)
    assert opponents[0].actor is opponents[1].actor
    assert opponents[0].hidden.data_ptr() != opponents[1].hidden.data_ptr()
    assert opponents[0].generator is not opponents[1].generator
    traces = [[] for _ in opponents]
    for _ in range(24):
        for opponent, trace in zip(opponents, traces):
            # Interleave unrelated caller randomness: it must not affect the opponent.
            torch.rand(3)
            before = torch.get_rng_state().clone()
            action = opponent.choose_action(env.engine, env.state, 1)
            assert torch.equal(torch.get_rng_state(), before)
            assert env.engine.validate_action(env.state, action) is None
            if isinstance(action, PlayCardAction):
                observation = env.observe_v2_for_viewer(1)
                col, row = action.cell
                assert observation.legal_play[action.card_slot, 31 - row, 17 - col]
            trace.append(action)
    assert traces[0] == traces[1]
    assert traces[0] != traces[2]
    assert not any(parameter.requires_grad for parameter in opponents[0].actor.parameters())


def test_metadata_envelope_and_factory_use_exact_configuration(checkpoint, tmp_path):
    from simulator.rl.v4.frontier_eval import make_opponent

    path, _, config = checkpoint
    envelope = tmp_path / "envelope.pt"
    torch.save({
        "model_config": asdict(config),
        "state_dict": torch.load(path, weights_only=True),
    }, envelope)
    digest = hashlib.sha256(envelope.read_bytes()).hexdigest()
    env = _env()
    opponent = make_opponent({
        "kind": "v4-checkpoint", "checkpoint": str(envelope), "sha256": digest,
        "model_config": asdict(config), "deck": list(PLAYER_DECK),
        "inference": "stochastic", "sampling_seed": 44,
    }, 0, env=env)
    assert opponent.actor.config.model_dim == 8
    assert opponent.actor.config.fixed_wait
    assert opponent.inference == "stochastic"
    assert opponent.seed == 44
    assert env.engine.validate_action(env.state, opponent.choose_action(env.engine, env.state, 1)) is None
    with pytest.raises(ValueError, match="model_config does not match"):
        V4PolicyOpponent(env, str(envelope), digest, model_config=replace(config, model_dim=16))


def test_wrong_deck_or_observation_contract_fails_before_policy_execution(checkpoint):
    path, digest, config = checkpoint
    env = _env()
    wrong = ("x-bow", *PLAYER_DECK[1:])
    with pytest.raises(ValueError, match="canonical Hog"):
        V4PolicyOpponent(env, str(path), digest, model_config=config, deck=wrong)
    env.reset_v2(seed=7, decks=(PLAYER_DECK, wrong))
    with pytest.raises(ValueError, match="environment deck"):
        V4PolicyOpponent(env, str(path), digest, model_config=config)
    with pytest.raises(ValueError, match="public input contract"):
        V4PolicyOpponent(_env(), str(path), digest, model_config=replace(config, event_dim=16))


def test_isolated_sampler_matches_native_fixed_wait_factorization(checkpoint):
    from simulator.rl.v4.v4_ppo.runtime import _v4_step_inputs, batch_v4_inputs
    from simulator.public_state_estimator import PublicStateEstimator

    path, digest, config = checkpoint
    env = _env()
    opponent = V4PolicyOpponent(env, str(path), digest, model_config=config)
    observation = env.observe_v2_for_viewer(1)
    estimator = PublicStateEstimator()
    estimator.sync_from_env(env, viewer=1)
    own = env.state.players[1]
    inputs = batch_v4_inputs(_v4_step_inputs(
        observation, own.hand, own.elixir_milli / 1000.0, estimator.snapshot(),
    ), "cpu")
    legal = torch.as_tensor(observation.legal_play.copy()).unsqueeze(0).unsqueeze(0)
    masks = masks_from_legal_play(legal, bool(observation.legal_wait))
    with torch.no_grad():
        logits, _, _ = opponent.actor.forward_step(inputs, hidden=opponent.hidden)
        generator = torch.Generator().manual_seed(49)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(49)
            for _ in range(24):
                expected, _, _ = opponent.actor.sample_action(logits, masks)
                actual = _sample_action(logits, masks, generator)
                assert torch.equal(actual.mode, expected.mode)
                assert torch.equal(actual.card_slot, expected.card_slot)
                assert torch.equal(actual.placement, expected.placement)
                assert torch.equal(actual.wait_duration, expected.wait_duration)

