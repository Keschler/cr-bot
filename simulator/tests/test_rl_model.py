from __future__ import annotations

import importlib

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


def test_rl_package_reports_optional_torch_dependency() -> None:
    import rl

    assert rl.TORCH_AVAILABLE is (torch is not None)
    if torch is None:
        with pytest.raises(rl.TorchUnavailableError, match="PyTorch"):
            importlib.import_module("rl.model")


requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


def _config():
    from rl import ModelConfig

    return ModelConfig(
        raster_channels=3,
        raster_height=8,
        raster_width=6,
        global_dim=7,
        entity_dim=5,
        max_entities=10,
        model_dim=16,
        encoder_dim=12,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ff_dim=32,
        gru_hidden_dim=14,
        gru_layers=2,
        card_slots=3,
        placement_rows=5,
        placement_cols=4,
    )


def _inputs(config, *, batch: int = 2, time: int = 3, entities: int = 6):
    raster = torch.randn(
        batch,
        time,
        config.raster_channels,
        config.raster_height,
        config.raster_width,
    )
    global_features = torch.randn(batch, time, config.global_dim)
    entity_features = torch.randn(batch, time, entities, config.entity_dim)
    entity_mask = torch.ones(batch, time, entities, dtype=torch.bool)
    if time > 1:
        entity_mask[0, 1, 2:] = False
    if time > 2:
        entity_mask[1, 2, :] = False
    reset_mask = torch.zeros(batch, time, dtype=torch.bool)
    reset_mask[:, 0] = True
    return raster, global_features, entity_features, entity_mask, reset_mask


def _masks(config, *, batch: int = 2, time: int = 3):
    from rl import ActionMasks

    mode = torch.ones(batch, time, 2, dtype=torch.bool)
    card = torch.ones(batch, time, config.card_slots, dtype=torch.bool)
    card[..., 1] = False
    placement = torch.zeros(
        batch,
        time,
        config.card_slots,
        config.placement_rows,
        config.placement_cols,
        dtype=torch.bool,
    )
    placement[..., 0, 1, 2] = True
    placement[..., 0, 2, 3] = True
    placement[..., 2, 4, 0] = True
    return ActionMasks(mode=mode, card=card, placement=placement)


@requires_torch
def test_hybrid_recurrent_policy_shapes_and_trajectory_storage() -> None:
    from rl import ActionBatch, RecurrentHybridPolicy, RecurrentSequence, TrajectoryBatch

    config = _config()
    raster, global_features, entities, entity_mask, reset_mask = _inputs(config)
    policy = RecurrentHybridPolicy(config)
    output = policy(raster, global_features, entities, entity_mask, reset_mask=reset_mask)

    assert output.encoded_features.shape == (2, 3, config.encoder_dim)
    assert output.recurrent_features.shape == (2, 3, config.gru_hidden_dim)
    assert output.mode_logits.shape == (2, 3, 2)
    assert output.card_logits.shape == (2, 3, config.card_slots)
    assert output.placement_logits.shape == (
        2,
        3,
        config.card_slots,
        config.placement_rows,
        config.placement_cols,
    )
    assert output.final_hidden.shape == (config.gru_layers, 2, config.gru_hidden_dim)

    sequence = RecurrentSequence(
        raster=raster,
        global_features=global_features,
        entities=entities,
        entity_mask=entity_mask,
        reset_mask=reset_mask,
        hidden_states=torch.zeros(2, 3, config.gru_layers, config.gru_hidden_dim),
        initial_hidden=policy.initial_hidden(2),
    )
    actions = ActionBatch(
        mode=torch.zeros(2, 3, dtype=torch.long),
        card_slot=torch.zeros(2, 3, dtype=torch.long),
        placement=torch.zeros(2, 3, 2, dtype=torch.long),
    )
    masks = _masks(config)
    trajectory = TrajectoryBatch(
        sequence=sequence,
        action_masks=masks,
        actions=actions,
        rewards=torch.zeros(2, 3),
        terminated=torch.zeros(2, 3, dtype=torch.bool),
        truncated=torch.zeros(2, 3, dtype=torch.bool),
        old_log_probs=torch.zeros(2, 3),
    )
    assert trajectory.batch_size == 2
    assert trajectory.time_steps == 3


@requires_torch
def test_explicit_hand_features_condition_card_and_placement_heads() -> None:
    from rl import ModelConfig, RecurrentHybridPolicy

    config = _config()
    values = {
        field: getattr(config, field)
        for field in config.__dataclass_fields__
    }
    values.update(
        global_dim=16,
        hand_feature_offset=1,
        hand_card_count=3,
    )
    config = ModelConfig(
        **values
    )
    policy = RecurrentHybridPolicy(config)
    assert policy.encoder.hand_projection is not None
    assert policy.hand_action_projection is not None
    assert policy.action_head.hand_card_score is not None

    raster, global_features, entities, entity_mask, reset_mask = _inputs(config)
    output = policy(
        raster,
        global_features,
        entities,
        entity_mask,
        reset_mask=reset_mask,
    )
    assert output.hand_features is not None
    assert output.hand_features.shape == (2, 3, config.card_slots, config.gru_hidden_dim)
    assert output.card_logits.shape == (2, 3, config.card_slots)


@requires_torch
def test_hand_table_features_stay_outside_the_entity_transformer() -> None:
    from rl import ModelConfig, RecurrentHybridPolicy

    values = {
        field: getattr(_config(), field)
        for field in _config().__dataclass_fields__
    }
    values.update(
        global_dim=10,
        hand_feature_offset=1,
        hand_card_count=3,
    )
    config = ModelConfig(**values)
    policy = RecurrentHybridPolicy(config).eval()
    assert policy.encoder.hand_projection is not None

    raster = torch.randn(1, 1, config.raster_channels, config.raster_height, config.raster_width)
    global_features = torch.zeros(1, 1, config.global_dim)
    # Slot zero contains table row 1, slot one contains table row 2, and slot
    # two is empty. These remain one-hot table features for the hand heads.
    global_features[0, 0, 1 + 1] = 1.0
    global_features[0, 0, 1 + config.hand_card_count + 2] = 1.0
    entities = torch.randn(1, 1, 2, config.entity_dim)
    entity_mask = torch.ones(1, 1, 2, dtype=torch.bool)

    captured: dict[str, torch.Tensor] = {}

    def capture_transformer_input(module, args, kwargs):
        del module, kwargs
        captured["tokens"] = args[0].detach()

    handle = policy.encoder.entity_transformer.register_forward_pre_hook(
        capture_transformer_input,
        with_kwargs=True,
    )
    try:
        with torch.inference_mode():
            policy(raster, global_features, entities, entity_mask)
    finally:
        handle.remove()

    assert captured["tokens"].shape == (1, 2 + 1, config.model_dim)
    expected_hand = policy.encoder.public_hand_features(global_features)
    assert expected_hand is not None
    assert expected_hand.shape == (1, 1, config.card_slots, config.model_dim)


@requires_torch
def test_deterministic_fast_action_matches_full_reference_path() -> None:
    from rl import ActionMasks, ModelConfig, RecurrentHybridPolicy
    from rl.learner import _deterministic_action

    values = {
        field: getattr(_config(), field)
        for field in _config().__dataclass_fields__
    }
    values.update(
        global_dim=16,
        hand_feature_offset=1,
        hand_card_count=3,
        spatial_placement_features=True,
        placement_rows=5,
        placement_cols=4,
    )
    config = ModelConfig(**values)
    policy = RecurrentHybridPolicy(config).eval()
    raster, global_features, entities, entity_mask, reset_mask = _inputs(
        config,
        batch=2,
        time=1,
        entities=6,
    )
    mode = torch.tensor(
        [
            [[True, False]],
            [[False, True]],
        ],
        dtype=torch.bool,
    )  # Exercise both the WAIT fast branch and PLAY placement decoding.
    card = torch.ones(2, 1, config.card_slots, dtype=torch.bool)
    placement = torch.zeros(
        2,
        1,
        config.card_slots,
        config.placement_rows,
        config.placement_cols,
        dtype=torch.bool,
    )
    placement[:, 0, :, 1, 2] = True
    placement[:, 0, :, 3, 1] = True
    masks = ActionMasks(mode=mode, card=card, placement=placement)

    with torch.inference_mode():
        reference = policy(
            raster,
            global_features,
            entities,
            entity_mask,
            reset_mask=reset_mask,
            action_masks=masks,
            include_beliefs=False,
        )
        reference_actions, _log_probs, _entropy = _deterministic_action(
            policy,
            reference,
            masks,
        )
        mha_backend = getattr(torch.backends, "mha", None)
        if (
            mha_backend is None
            or not hasattr(mha_backend, "get_fastpath_enabled")
            or not hasattr(mha_backend, "set_fastpath_enabled")
        ):
            fast_actions, fast_hidden = policy.act_deterministic(
                raster,
                global_features,
                entities,
                entity_mask,
                masks,
                reset_mask=reset_mask,
            )
        else:
            previous_mha_fastpath = bool(mha_backend.get_fastpath_enabled())
            mha_backend.set_fastpath_enabled(True)
            try:
                fast_actions, fast_hidden = policy.act_deterministic(
                    raster,
                    global_features,
                    entities,
                    entity_mask,
                    masks,
                    reset_mask=reset_mask,
                )
                assert mha_backend.get_fastpath_enabled() is True
            finally:
                mha_backend.set_fastpath_enabled(previous_mha_fastpath)

    assert reference.belief_logits is None
    assert torch.equal(fast_actions.mode, reference_actions.mode)
    assert torch.equal(fast_actions.card_slot, reference_actions.card_slot)
    assert torch.equal(fast_actions.placement, reference_actions.placement)
    torch.testing.assert_close(fast_hidden, reference.final_hidden)

    wait_masks = ActionMasks(
        mode=torch.tensor([[[True, False]], [[True, False]]]),
        card=torch.zeros_like(card),
        placement=torch.zeros_like(placement),
    )
    with torch.inference_mode():
        wait_reference = policy(
            raster,
            global_features,
            entities,
            entity_mask,
            reset_mask=reset_mask,
            action_masks=wait_masks,
            include_beliefs=False,
        )
        wait_reference_actions, _log_probs, _entropy = _deterministic_action(
            policy,
            wait_reference,
            wait_masks,
        )
        wait_fast_actions, wait_fast_hidden = policy.act_deterministic(
            raster,
            global_features,
            entities,
            entity_mask,
            wait_masks,
            reset_mask=reset_mask,
        )

    assert torch.equal(wait_fast_actions.mode, wait_reference_actions.mode)
    assert torch.equal(wait_fast_actions.card_slot, wait_reference_actions.card_slot)
    assert torch.equal(wait_fast_actions.placement, wait_reference_actions.placement)
    torch.testing.assert_close(wait_fast_hidden, wait_reference.final_hidden)


@requires_torch
def test_inference_compacts_padded_entities_without_changing_actions() -> None:
    from rl import ActionMasks, ModelConfig, RecurrentHybridPolicy
    from rl.learner import _deterministic_action
    from rl.model import _inference_entity_tokens

    config = _config()
    policy = RecurrentHybridPolicy(config).eval()
    raster, global_features, entities, _entity_mask, reset_mask = _inputs(
        config,
        batch=2,
        time=1,
        entities=10,
    )
    entity_mask = torch.zeros(2, 1, 10, dtype=torch.bool)
    entity_mask[:, :, :3] = True
    entity_mask[1, :, 2:] = False
    masks = ActionMasks(
        mode=torch.ones(2, 1, 2, dtype=torch.bool),
        card=torch.ones(2, 1, config.card_slots, dtype=torch.bool),
        placement=torch.ones(
            2,
            1,
            config.card_slots,
            config.placement_rows,
            config.placement_cols,
            dtype=torch.bool,
        ),
    )

    compact_entities, compact_mask = _inference_entity_tokens(
        entities,
        entity_mask,
        config,
    )
    assert compact_entities.shape[2] == 3
    assert compact_mask.shape[2] == 3

    with torch.inference_mode():
        reference = policy(
            raster,
            global_features,
            entities,
            entity_mask,
            reset_mask=reset_mask,
            hidden=None,
            action_masks=masks,
            include_beliefs=False,
        )
        reference_actions, _log_probs, _entropy = _deterministic_action(
            policy,
            reference,
            masks,
        )
        fast_actions, fast_hidden = policy.act_deterministic(
            raster,
            global_features,
            entities,
            entity_mask,
            masks,
            reset_mask=reset_mask,
        )

    assert torch.equal(fast_actions.mode, reference_actions.mode)
    assert torch.equal(fast_actions.card_slot, reference_actions.card_slot)
    assert torch.equal(fast_actions.placement, reference_actions.placement)
    torch.testing.assert_close(fast_hidden, reference.final_hidden)


@requires_torch
def test_dense_inference_transformer_layout_preserves_selected_actions() -> None:
    from rl import ActionMasks, ModelConfig, RecurrentHybridPolicy
    from rl.learner import _deterministic_action

    config = _config()
    policy = RecurrentHybridPolicy(config).eval()
    raster, global_features, entities, entity_mask, reset_mask = _inputs(
        config,
        batch=3,
        time=1,
        entities=10,
    )
    masks = ActionMasks(
        mode=torch.ones(3, 1, 2, dtype=torch.bool),
        card=torch.ones(3, 1, config.card_slots, dtype=torch.bool),
        placement=torch.ones(
            3,
            1,
            config.card_slots,
            config.placement_rows,
            config.placement_cols,
            dtype=torch.bool,
        ),
    )

    with torch.inference_mode():
        reference = policy(
            raster,
            global_features,
            entities,
            entity_mask,
            reset_mask=reset_mask,
            action_masks=masks,
            include_beliefs=False,
        )
        reference_actions, _log_probs, _entropy = _deterministic_action(
            policy,
            reference,
            masks,
        )
        fast_actions, fast_hidden = policy.act_deterministic(
            raster,
            global_features,
            entities,
            entity_mask,
            masks,
            reset_mask=reset_mask,
        )

    assert torch.equal(fast_actions.mode, reference_actions.mode)
    assert torch.equal(fast_actions.card_slot, reference_actions.card_slot)
    assert torch.equal(fast_actions.placement, reference_actions.placement)
    torch.testing.assert_close(fast_hidden, reference.final_hidden)


@requires_torch
def test_one_step_gru_matches_direct_reset_semantics() -> None:
    from rl import GRURecurrentCore

    torch.manual_seed(17)
    core = GRURecurrentCore(input_dim=5, hidden_dim=7, layers=2).eval()
    features = torch.randn(3, 1, 5)
    hidden = torch.randn(2, 3, 7)
    reset_mask = torch.tensor([[True], [False], [True]])
    reset_hidden = hidden.masked_fill(reset_mask[:, 0].reshape(1, 3, 1), 0.0)
    expected_output, expected_hidden = core.gru(features, reset_hidden)

    with torch.inference_mode():
        output, final_hidden = core(features, hidden=hidden, reset_mask=reset_mask)

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(final_hidden, expected_hidden)


@requires_torch
def test_spatial_placement_variant_retains_board_aligned_features() -> None:
    from rl import ModelConfig, RecurrentHybridPolicy

    values = {
        field: getattr(_config(), field)
        for field in _config().__dataclass_fields__
    }
    values.update(
        spatial_placement_features=True,
        spatial_placement_dim=6,
    )
    config = ModelConfig(**values)
    policy = RecurrentHybridPolicy(config)
    raster, global_features, entities, entity_mask, reset_mask = _inputs(config)
    output = policy(
        raster,
        global_features,
        entities,
        entity_mask,
        reset_mask=reset_mask,
    )

    assert output.spatial_features is not None
    assert output.spatial_features.shape == (
        2,
        3,
        config.model_dim,
        config.raster_height,
        config.raster_width,
    )
    assert policy.action_head.spatial_placement_key is not None
    assert policy.action_head.spatial_placement_query is not None
    assert output.placement_logits.shape == (
        2,
        3,
        config.card_slots,
        config.placement_rows,
        config.placement_cols,
    )


@requires_torch
def test_contextual_public_card_head_keeps_entity_context_in_card_selection() -> None:
    from rl import ModelConfig, RecurrentHybridPolicy

    values = {
        field: getattr(_config(), field)
        for field in _config().__dataclass_fields__
    }
    values.update(
        direct_public_card_features=True,
        contextual_public_card_features=True,
    )
    config = ModelConfig(**values)
    policy = RecurrentHybridPolicy(config)

    assert policy.action_head.public_card_head is not None
    assert policy.action_head.public_card_head[0].in_features == (
        config.global_dim + config.gru_hidden_dim
    )
    raster, global_features, entities, entity_mask, reset_mask = _inputs(config)
    output = policy(
        raster,
        global_features,
        entities,
        entity_mask,
        reset_mask=reset_mask,
    )
    assert output.card_logits.shape == (2, 3, config.card_slots)


@requires_torch
def test_contextual_public_card_head_includes_projected_hand_slots() -> None:
    from rl import ModelConfig, RecurrentHybridPolicy

    values = {
        field: getattr(_config(), field)
        for field in _config().__dataclass_fields__
    }
    values.update(
        global_dim=16,
        hand_feature_offset=1,
        hand_card_count=3,
        direct_public_card_features=True,
        contextual_public_card_features=True,
    )
    config = ModelConfig(**values)
    policy = RecurrentHybridPolicy(config)

    assert policy.action_head.public_card_head is not None
    assert policy.action_head.public_card_head[0].in_features == (
        config.global_dim
        + config.gru_hidden_dim
        + config.card_slots * config.gru_hidden_dim
    )
    raster, global_features, entities, entity_mask, reset_mask = _inputs(config)
    output = policy(
        raster,
        global_features,
        entities,
        entity_mask,
        reset_mask=reset_mask,
    )
    assert output.card_logits.shape == (2, 3, config.card_slots)


@requires_torch
def test_public_card_context_starts_as_behavior_preserving_residual() -> None:
    from rl import ModelConfig, RecurrentHybridPolicy

    values = {
        field: getattr(_config(), field)
        for field in _config().__dataclass_fields__
    }
    base_config = ModelConfig(**values)
    residual_values = dict(values)
    residual_values.update(
        direct_public_card_features=True,
        contextual_public_card_features=True,
    )
    residual_config = ModelConfig(**residual_values)

    base = RecurrentHybridPolicy(base_config).eval()
    residual = RecurrentHybridPolicy(residual_config).eval()
    residual.load_state_dict(base.state_dict(), strict=False)
    assert residual.action_head.public_card_head is not None
    assert torch.count_nonzero(residual.action_head.public_card_head[-1].weight) == 0
    assert torch.count_nonzero(residual.action_head.public_card_head[-1].bias) == 0

    inputs = _inputs(base_config)
    with torch.inference_mode():
        base_output = base(*inputs[:4], reset_mask=inputs[4])
        residual_output = residual(*inputs[:4], reset_mask=inputs[4])
    assert torch.equal(residual_output.card_logits, base_output.card_logits)


@requires_torch
def test_action_masks_remove_illegal_card_and_placement_probability() -> None:
    from rl import ActionMasks, MaskedAutoregressivePolicy, ModelConfig

    config = _config()
    head = MaskedAutoregressivePolicy(config.gru_hidden_dim, config)
    features = torch.zeros(2, 1, config.gru_hidden_dim)
    logits = head(features)
    masks = _masks(config, batch=2, time=1)
    masked = head.masked_log_probs(logits, masks)

    assert torch.isneginf(masked.card[..., 1]).all()
    assert torch.isneginf(masked.placement[..., 0, 0, 0]).all()
    assert torch.isfinite(masked.placement[..., 0, 1, 2]).all()

    illegal_mode = torch.tensor([[[True, False]], [[True, False]]])
    illegal_masks = ActionMasks(
        mode=illegal_mode,
        card=masks.card,
        placement=masks.placement,
    )
    from rl import ActionBatch

    with pytest.raises(ValueError, match="illegal WAIT/PLAY mode"):
        head.log_prob(
            logits,
            ActionBatch(
                mode=torch.ones(2, 1, dtype=torch.long),
                card_slot=torch.zeros(2, 1, dtype=torch.long),
                placement=torch.tensor([[[1, 2]], [[1, 2]]]),
            ),
            illegal_masks,
        )


@requires_torch
def test_log_prob_is_sum_of_selected_mode_card_and_placement_terms() -> None:
    from rl import ActionBatch, MaskedAutoregressivePolicy

    config = _config()
    head = MaskedAutoregressivePolicy(config.gru_hidden_dim, config)
    features = torch.randn(2, 2, config.gru_hidden_dim)
    logits = head(features)
    masks = _masks(config, batch=2, time=2)
    actions = ActionBatch(
        mode=torch.tensor([[1, 0], [1, 1]]),
        card_slot=torch.tensor([[0, 0], [2, 0]]),
        placement=torch.tensor([[[1, 2], [0, 0]], [[4, 0], [2, 3]]]),
    )

    joint = head.log_prob(logits, actions, masks)
    masked = head.masked_log_probs(logits, masks)
    expected = masked.mode.gather(-1, actions.mode.unsqueeze(-1)).squeeze(-1)
    expected = expected.clone()
    play = actions.mode == 1
    selected_cards = actions.card_slot[play]
    selected_placement = actions.placement[play]
    selected_card_log_prob = masked.card[play].gather(
        -1,
        selected_cards.unsqueeze(-1),
    ).squeeze(-1)
    selected_placement_log_prob = masked.placement[play][
        torch.arange(selected_cards.shape[0]),
        selected_cards,
    ]
    selected_placement_log_prob = selected_placement_log_prob[
        torch.arange(selected_cards.shape[0]),
        selected_placement[:, 0],
        selected_placement[:, 1],
    ]
    expected[play] += selected_card_log_prob + selected_placement_log_prob
    torch.testing.assert_close(joint, expected)


@requires_torch
def test_sampling_returns_legal_hierarchical_actions_and_diagnostics() -> None:
    from rl import MaskedAutoregressivePolicy

    config = _config()
    head = MaskedAutoregressivePolicy(config.gru_hidden_dim, config)
    logits = head(torch.randn(2, 3, config.gru_hidden_dim))
    masks = _masks(config)

    actions, log_probs, entropy = head.sample(logits, masks)

    assert actions.mode.shape == (2, 3)
    assert actions.card_slot.shape == (2, 3)
    assert actions.placement.shape == (2, 3, 2)
    assert log_probs.shape == entropy.shape == (2, 3)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()
    selected_mode = masks.mode.gather(-1, actions.mode.unsqueeze(-1)).squeeze(-1)
    assert selected_mode.all()
    play = actions.mode == head.PLAY
    if bool(play.any().item()):
        selected_cards = actions.card_slot[play]
        assert masks.card[play].gather(-1, selected_cards.unsqueeze(-1)).all()
        selected_mask = masks.placement[play, selected_cards]
        selected_rows = actions.placement[play, 0]
        selected_cols = actions.placement[play, 1]
        assert selected_mask[
            torch.arange(selected_cards.shape[0]), selected_rows, selected_cols
        ].all()


@requires_torch
def test_privileged_critic_and_belief_heads_keep_shapes_separate() -> None:
    from rl import OpponentBeliefHeads, PrivilegedCritic

    belief = OpponentBeliefHeads(hidden_dim=7, card_count=11)
    recurrent = torch.randn(2, 3, 7)
    belief_logits = belief(recurrent)
    assert belief_logits.enemy_elixir.shape == (2, 3)
    assert belief_logits.enemy_hand.shape == (2, 3, 11)
    assert belief_logits.enemy_next_card.shape == (2, 3, 11)

    critic = PrivilegedCritic(recurrent_dim=7, privileged_dim=13)
    value = critic(recurrent, torch.randn(2, 3, 13))
    assert value.shape == (2, 3)


@requires_torch
def test_gru_reset_discards_previous_episode_hidden_state() -> None:
    from rl import GRURecurrentCore

    torch.manual_seed(7)
    core = GRURecurrentCore(input_dim=6, hidden_dim=9, layers=2)
    features = torch.randn(2, 4, 6)
    hidden = torch.randn(2, 2, 9)
    reset_mask = torch.zeros(2, 4, dtype=torch.bool)
    reset_mask[0, 2] = True
    output, _ = core(features, hidden=hidden, reset_mask=reset_mask)

    fresh_hidden = core.initial_state(1, dtype=features.dtype)
    fresh_output, _ = core(
        features[0:1, 2:],
        hidden=fresh_hidden,
        reset_mask=torch.zeros(1, 2, dtype=torch.bool),
    )
    torch.testing.assert_close(output[0:1, 2:], fresh_output)
