"""Focused contracts for the retained from-scratch V4 bootstrap."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulator.rl.v4.bootstrap import (  # noqa: E402
    BootstrapConfig,
    generate_bootstrap_dataset,
    to_torch_batch,
)
from simulator.rl.v4.hand_features import (  # noqa: E402
    HandEncodingError,
    build_hand_tokens,
)
from simulator.rl.v4.simulator_teacher import teacher_label  # noqa: E402
from simulator.rl.v4.supervised import V4SupervisedWeights, v4_supervised_loss


def test_hand_encoder_is_fixed_and_rejects_unknown_cards() -> None:
    first = build_hand_tokens(
        ["hog-rider", "cannon", "musketeer", "skeletons"], 7.0
    )
    second = build_hand_tokens(
        ["hog-rider", "cannon", "musketeer", "skeletons"], 7.0
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 16)
    with pytest.raises(HandEncodingError, match="outside the V4 learner deck"):
        build_hand_tokens(["unknown", "cannon", "musketeer", "skeletons"], 7.0)


def test_bootstrap_collection_is_deterministic_and_public() -> None:
    config = BootstrapConfig(n_states=2, seed=19, max_decisions=128)
    first = generate_bootstrap_dataset(config)
    second = generate_bootstrap_dataset(config)
    assert [sample.provenance["state_hash"] for sample in first] == [
        sample.provenance["state_hash"] for sample in second
    ]
    for sample in first:
        assert sample.raster.shape == (21, 32, 18)
        assert sample.global_features.shape == (768,)
        assert sample.entity_tokens.shape == (128, 32)
        assert sample.opp_hand_probs.shape == (128,)
        assert bool(np.isfinite(sample.raster).all())


def test_bootstrap_collection_balances_opponent_archetypes() -> None:
    config = BootstrapConfig(n_states=12, seed=7, max_decisions=128)
    samples = generate_bootstrap_dataset(config)
    assert Counter(sample.family for sample in samples) == {
        archetype: 2 for archetype in config.archetypes
    }


def test_bootstrap_batch_preserves_masks_and_action_targets() -> None:
    samples = generate_bootstrap_dataset(BootstrapConfig(n_states=3, seed=2, max_decisions=128))
    batch = to_torch_batch(samples)
    count = len(samples)
    assert count >= 3  # support can add records beyond the preferred quota
    assert tuple(batch["raster"].shape) == (count, 1, 21, 32, 18)
    assert tuple(batch["actions"].mode.shape) == (count, 1)
    assert tuple(batch["masks"].mode.shape) == (count, 1, 2)
    assert tuple(batch["mode_weights"].shape) == (count, 1)
    assert float(batch["mode_weights"].mean()) == pytest.approx(1.0)
    assert torch.isfinite(batch["soft_placement"]).all()
    for index, sample in enumerate(samples):
        mode = int(batch["actions"].mode[index, 0])
        assert bool(batch["masks"].mode[index, 0, mode])
        if mode == 1:
            slot = int(batch["actions"].card_slot[index, 0])
            row, col = (int(value) for value in batch["actions"].placement[index, 0])
            assert bool(batch["masks"].placement[index, 0, slot, row, col])


@pytest.fixture
def fake_bootstrap_matches(monkeypatch):
    """Fast full episodes with a long WAIT opening and late PLAY labels."""
    import simulator.rl.v4.bootstrap as bootstrap

    environments = []
    modes = {"play_after": 90}

    class FakeEnv:
        def __init__(self, engine, **kwargs):
            self.engine = engine
            self.steps = 0
            self.state = SimpleNamespace(terminal=False, phase="regulation", elapsed_us=0)
            environments.append(self)

        def reset_v2(self, **kwargs):
            pass

        def step_v2(self, actions):
            self.steps += 1
            self.state.elapsed_us = self.steps * 250_000
            self.state.terminal = self.steps == 100

    controller = SimpleNamespace(choose_action=lambda *args: None)
    scenario = SimpleNamespace(
        deck=SimpleNamespace(deck_id="test-deck", cards=("test-card",) * 8),
        build_controller=lambda: controller,
    )
    monkeypatch.setattr(bootstrap, "load_fixed_ruleset", lambda: object())
    monkeypatch.setattr(bootstrap, "BattleEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(bootstrap, "SimulatorEnv", FakeEnv)
    monkeypatch.setattr(bootstrap, "PublicStateEstimator", lambda: object())
    monkeypatch.setattr(
        bootstrap, "OpponentPool",
        lambda *args, **kwargs: SimpleNamespace(sample=lambda *args, **kwargs: scenario),
    )

    def capture(env, estimator, *, archetype, episode, decision, seed):
        sample = SimpleNamespace(
            family=archetype,
            target=SimpleNamespace(mode=int(decision >= modes["play_after"])),
            provenance={
                "episode": episode, "decision": decision, "seed": seed,
                "elapsed_us": env.state.elapsed_us, "phase": env.state.phase,
                "state_hash": f"{episode}:{decision}",
            },
        )
        return sample, None

    monkeypatch.setattr(bootstrap, "_capture_sample", capture)
    return environments, modes


def test_bootstrap_samples_late_plays_and_multiple_complete_matches(fake_bootstrap_matches):
    environments, _ = fake_bootstrap_matches
    config = BootstrapConfig(
        n_states=20, seed=7, max_decisions=200,
        archetypes=("first", "second"), samples_per_episode=4,
    )
    first = generate_bootstrap_dataset(config)
    second = generate_bootstrap_dataset(config)
    assert [sample.provenance for sample in first] == [sample.provenance for sample in second]
    assert Counter((sample.family, sample.target.mode) for sample in first) == {
        ("first", 0): 5, ("first", 1): 5,
        ("second", 0): 5, ("second", 1): 5,
    }
    episodes = Counter(sample.provenance["episode"] for sample in first)
    assert len(episodes) >= 6
    assert max(episodes.values()) <= 4
    assert all(env.steps == 100 for env in environments)
    assert all(sample.provenance["decision"] >= 90 for sample in first if sample.target.mode)
    assert any(sample.provenance["decision"] >= 50 for sample in first if not sample.target.mode)


def test_bootstrap_impossible_play_quota_fails_with_bounded_work(fake_bootstrap_matches):
    environments, modes = fake_bootstrap_matches
    modes["play_after"] = 101
    with pytest.raises(RuntimeError, match="(?i)(play|quota|target|label|collect)"):
        generate_bootstrap_dataset(BootstrapConfig(
            n_states=4, seed=0, max_decisions=200,
            archetypes=("first",), samples_per_episode=4,
        ))
    assert 1 <= len(environments) <= 8
    assert all(env.steps == 100 for env in environments)


def test_bootstrap_rejects_reserved_opponent_deck(fake_bootstrap_matches, monkeypatch):
    import simulator.rl.v4.bootstrap as bootstrap

    monkeypatch.setattr(
        bootstrap, "load_real_decks",
        lambda: (SimpleNamespace(split="eval", cards=("test-card",) * 8),),
    )
    with pytest.raises(RuntimeError, match="overlaps final evaluation"):
        generate_bootstrap_dataset(BootstrapConfig(
            n_states=2, seed=0, max_decisions=200,
            archetypes=("first",), samples_per_episode=4,
        ))


def test_bootstrap_retains_each_observed_episode_mode_and_restores_frequency(
    fake_bootstrap_matches,
) -> None:
    samples = generate_bootstrap_dataset(BootstrapConfig(
        n_states=5, seed=3, max_decisions=200,
        archetypes=("first",), samples_per_episode=4,
    ))
    groups = Counter((sample.provenance["episode"], sample.target.mode) for sample in samples)
    assert set(groups) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(samples) > 5  # preserving support may exceed the preferred quota
    assert sum(sample.mode_weight for sample in samples) == pytest.approx(len(samples))
    wait_weight = sum(sample.mode_weight for sample in samples if sample.target.mode == 0)
    play_weight = sum(sample.mode_weight for sample in samples if sample.target.mode == 1)
    assert wait_weight / (wait_weight + play_weight) == pytest.approx(0.9)
    assert play_weight / (wait_weight + play_weight) == pytest.approx(0.1)
    for sample in samples:
        assert sample.provenance["eligible_mode_count"] == (90 if sample.target.mode == 0 else 10)
        assert sample.provenance["retained_mode_count"] == groups[
            sample.provenance["episode"], sample.target.mode
        ]


def test_bootstrap_fixed_batch_runs_predeclared_complete_matches(fake_bootstrap_matches) -> None:
    environments, _ = fake_bootstrap_matches
    config = BootstrapConfig(
        n_states=5, seed=11, max_decisions=200,
        archetypes=("first", "second"), samples_per_episode=4,
        episodes_per_archetype=3,
    )
    summaries = []
    samples = generate_bootstrap_dataset(config, episode_summaries=summaries)
    assert len(environments) == 6
    assert all(env.steps == 100 for env in environments)
    assert len(summaries) == 6
    assert all(summary.terminal and summary.decisions == 100 for summary in summaries)
    assert Counter(sample.family for sample in samples) == {"first": 12, "second": 12}
    assert {sample.provenance["episode"] for sample in samples} == set(range(6))
    assert len(samples) == 24  # --states is advisory for a fixed match batch


def test_bootstrap_records_or_rejects_decision_cap_truncation(fake_bootstrap_matches) -> None:
    _, modes = fake_bootstrap_matches
    modes["play_after"] = 1
    base = dict(
        n_states=2, seed=5, max_decisions=3,
        archetypes=("first",), samples_per_episode=4,
        episodes_per_archetype=1,
    )
    with pytest.raises(RuntimeError, match="decision cap before termination"):
        generate_bootstrap_dataset(BootstrapConfig(**base, require_terminal=True))

    summaries = []
    samples = generate_bootstrap_dataset(
        BootstrapConfig(**base, require_terminal=False),
        episode_summaries=summaries,
    )
    assert len(summaries) == 1
    assert summaries[0].decisions == 3
    assert summaries[0].terminal is False
    assert all(sample.provenance["episode_terminal"] is False for sample in samples)


def test_rules_teacher_returns_a_legal_public_target() -> None:
    hand = build_hand_tokens(
        ["hog-rider", "cannon", "musketeer", "skeletons"], 7.0
    )
    entities = np.zeros((128, 32), dtype=np.float32)
    entity_mask = np.zeros((128,), dtype=bool)
    entities[0, 1:7] = [1.0, 0.5, 0.75, 1.0, 0.0, 0.0]
    entity_mask[0] = True
    legal = np.zeros((4, 32, 18), dtype=bool)
    legal[1, 20:, :] = True
    target = teacher_label(
        hand_tokens=hand,
        entity_tokens=entities,
        entity_mask=entity_mask,
        legal_play=legal,
        legal_wait=True,
        own_elixir=7.0,
    )
    assert target.mode in (0, 1)
    if target.mode:
        assert bool(legal[target.card_slot, target.row, target.col])


def test_bootstrap_mode_weights_reconstruct_unsampled_teacher_loss() -> None:
    from simulator.rl.foundation.trajectory import ActionMasks
    from simulator.rl.v4.model_v4 import V4ActionBatch, V4Logits

    def mode_loss(n_wait: int, mode_weights: torch.Tensor | None = None) -> float:
        modes = torch.tensor([0] * n_wait + [1], dtype=torch.long).reshape(-1, 1)
        n = n_wait + 1
        logits = V4Logits(
            mode=torch.tensor([[[2.0, -2.0]]] * n),
            wait_duration=torch.zeros(n, 1, 4),
            card=torch.zeros(n, 1, 1),
            placement=torch.zeros(n, 1, 1, 1, 1),
        )
        masks = ActionMasks(
            mode=torch.ones(n, 1, 2, dtype=torch.bool),
            card=torch.ones(n, 1, 1, dtype=torch.bool),
            placement=torch.ones(n, 1, 1, 1, 1, dtype=torch.bool),
        )
        actions = V4ActionBatch(
            mode=modes,
            card_slot=torch.zeros(n, 1, dtype=torch.long),
            placement=torch.zeros(n, 1, 2, dtype=torch.long),
            wait_duration=torch.zeros(n, 1, dtype=torch.long),
        )
        loss, _ = v4_supervised_loss(
            logits, masks, actions,
            mode_weights=mode_weights,
            weights=V4SupervisedWeights(
                w_wait_duration=0.0, w_card=0.0, w_placement=0.0
            ),
        )
        return float(loss)

    full_match_loss = mode_loss(9, torch.ones(10, 1))
    retained_loss = mode_loss(1, torch.tensor([[1.8], [0.2]]))
    assert retained_loss == pytest.approx(full_match_loss, abs=1e-6)
    assert mode_loss(1) != pytest.approx(full_match_loss)
    # Existing callers without sampling weights retain their equal-mode loss.
    assert mode_loss(9) == pytest.approx(mode_loss(1), abs=1e-6)
