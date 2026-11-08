"""Factual tactical labels, public tracking uncertainty, and optional heads."""

from dataclasses import asdict, replace
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from simulator.rl.v4.tactical_outcomes import (
    OUTCOME_NAMES,
    PublicTacticalMeasurement as Measurement,
    PublicTrackedEntity as Entity,
    TacticalOutcomeConfig,
    TacticalOutcomeTargets,
    build_tactical_outcome_head,
    build_tactical_outcome_targets,
    encode_tactical_actions,
    normalized_tower_health,
    tactical_outcome_loss,
)


def _measurement(seconds, own=1.0, enemy=1.0, **kwargs):
    return Measurement(seconds, own, enemy, **kwargs)


def test_target_deadlines_use_match_seconds_and_only_requested_factual_anchors():
    measurements = [
        _measurement(5), _measurement(6.99, .95, .95),
        _measurement(7.1, .8, .7), _measurement(10, .7, .65),
        _measurement(11.3, .6, .5), _measurement(17, .4, .3),
    ]
    targets = build_tactical_outcome_targets(measurements, [0, 3])
    assert targets.anchor_indices == (0, 3)
    assert targets.horizons_seconds == (2., 6., 12.)
    np.testing.assert_array_equal(targets.future_indices, [[2, -1, 5], [-1, -1, -1]])
    np.testing.assert_allclose(targets.measured_seconds[0, [0, 2]], [7.1, 17])
    assert np.isnan(targets.measured_seconds[1]).all()
    np.testing.assert_allclose(targets.values[0, [0, 2], :2], [[.2, .3], [.6, .7]])
    assert targets.mask[0, 0, :2].all() and targets.mask[0, 2, :2].all()
    assert not targets.mask[0, 1].any() and not targets.mask[1].any()
    assert not targets.mask[..., 2:].any()  # Empty cohorts are unsupported, not dead.


@pytest.mark.parametrize("boundary", ["terminated", "truncated"])
def test_only_true_terminal_snapshots_absorb_later_horizons(boundary):
    measurements = [
        _measurement(0), _measurement(2, .9, .8),
        _measurement(4, .7, .5, **{boundary: True}),
    ]
    targets = build_tactical_outcome_targets(measurements, [0])
    np.testing.assert_allclose(targets.values[0, 0, :2], [.1, .2])
    if boundary == "terminated":
        np.testing.assert_array_equal(targets.future_indices, [[1, 2, 2]])
        np.testing.assert_allclose(targets.measured_seconds, [[2, 4, 4]])
        np.testing.assert_allclose(targets.values[0, 1:, :2], [[.3, .5], [.3, .5]])
        assert targets.mask[0, :, :2].all()
    else:
        np.testing.assert_array_equal(targets.future_indices, [[1, -1, -1]])
        assert not targets.mask[0, 1:].any()


def test_terminal_after_deadline_does_not_override_sampling_tolerance():
    targets = build_tactical_outcome_targets([
        _measurement(0), _measurement(2.4, .7, .5, terminated=True),
    ], [0])
    np.testing.assert_array_equal(targets.future_indices, [[-1, 1, 1]])
    assert not targets.mask[0, 0].any()


def test_survival_uses_fixed_anchor_cohort_and_ignores_replacement_spawns():
    anchor = _measurement(0, entities=(
        Entity(0, "own", is_defender=True), Entity(1, "own", is_defender=True),
        Entity(10, "enemy", is_air=True), Entity(11, "enemy", is_air=True),
        Entity(20, "enemy"), Entity(21, "own", is_air=True),
    ))
    future = _measurement(2, entities=(
        Entity(0, "own", is_defender=True),
        Entity(1, "own", is_defender=True, alive=False),
        Entity(2, "own", is_defender=True),
        Entity(10, "enemy", is_air=True, alive=False),
        Entity(11, "enemy", is_air=True), Entity(12, "enemy", is_air=True),
    ))
    targets = build_tactical_outcome_targets([anchor, future], [0], horizons_seconds=(2,))
    np.testing.assert_allclose(targets.values[0, 0, 2:], [.5, .5])
    assert targets.mask.all()


def test_missing_tracks_are_unknown_unless_public_measurement_proves_death():
    anchor = _measurement(0, entities=(
        Entity(0, "own", is_defender=True), Entity(1, "own", is_defender=True),
        Entity(10, "enemy", is_air=True),
    ))
    future = _measurement(2, entities=(Entity(0, "own", is_defender=True),))
    unknown = build_tactical_outcome_targets([anchor, future], [0], horizons_seconds=(2,))
    assert not unknown.mask[0, 0, 2:].any()
    known = build_tactical_outcome_targets(
        [anchor, replace(future, entities_complete=True)], [0], horizons_seconds=(2,),
    )
    assert known.mask.all()
    np.testing.assert_allclose(known.values[0, 0, 2:], [0, .5])


def test_public_death_evidence_persists_after_tombstone_disappears():
    measurements = [
        _measurement(0, entities=(Entity(10, "enemy", is_air=True),)),
        _measurement(1, entities=(Entity(10, "enemy", is_air=True, alive=False),)),
        _measurement(2),
    ]
    targets = build_tactical_outcome_targets(measurements, [0], horizons_seconds=(2,))
    assert targets.mask[0, 0, 2] and targets.values[0, 0, 2] == 0
    measurements[-1] = replace(measurements[-1], entities=(Entity(10, "enemy", is_air=True),))
    with pytest.raises(ValueError, match="known death"):
        build_tactical_outcome_targets(measurements, [0])


def test_unsupported_entities_and_missing_hp_mask_independent_outcomes():
    targets = build_tactical_outcome_targets([
        _measurement(0, own=None), _measurement(2, .6, .7),
    ], [0], horizons_seconds=(2,))
    np.testing.assert_array_equal(targets.mask, [[[False, True, False, False]]])
    assert targets.values[0, 0, 1] == pytest.approx(.3)
    assert targets.future_indices[0, 0] == 1


def test_normalized_health_uses_fixed_total_even_after_a_tower_is_destroyed():
    assert normalized_tower_health([200, 100, 100], maximum_total_hp=400) == 1
    assert normalized_tower_health([200, 0, 0], maximum_total_hp=400) == .5
    assert normalized_tower_health(None, maximum_total_hp=400) is None
    with pytest.raises(ValueError, match="exceeds"):
        normalized_tower_health([401], maximum_total_hp=400)
    with pytest.raises(ValueError, match="maximum_total_hp"):
        normalized_tower_health(None, maximum_total_hp=0)


def test_empty_inputs_are_well_shaped_and_targets_are_readonly_copies():
    empty = build_tactical_outcome_targets([], [])
    assert empty.values.shape == (0, 3, len(OUTCOME_NAMES))
    values = np.zeros((1, 1, 4), np.float32)
    targets = TacticalOutcomeTargets(values, np.ones_like(values, bool), [[1]], [[2]], (0,), (2,))
    values[:] = 1
    assert not targets.values.any()
    with pytest.raises(ValueError, match="read-only"):
        targets.values[0, 0, 0] = 1


@pytest.mark.parametrize("measurements, anchors, message", [
    ([_measurement(0), _measurement(0)], [0], "strictly increasing"),
    ([_measurement(0, terminated=True), _measurement(2)], [], "boundaries must be final"),
    ([_measurement(0), _measurement(2, truncated=True)], [1], "episode boundary"),
    ([_measurement(0), _measurement(2)], [0, 0], "strictly increasing"),
    ([_measurement(0)], [1], "existing measurements"),
])
def test_target_builder_rejects_ambiguous_episode_or_anchor_alignment(measurements, anchors, message):
    with pytest.raises(ValueError, match=message):
        build_tactical_outcome_targets(measurements, anchors)


def test_measurement_and_config_reject_invalid_contracts():
    with pytest.raises(ValueError, match="unique"):
        _measurement(0, entities=(Entity(1, "own"), Entity(1, "enemy")))
    with pytest.raises(ValueError, match="mutually exclusive"):
        _measurement(0, terminated=True, truncated=True)
    with pytest.raises(ValueError, match="strictly increasing"):
        TacticalOutcomeConfig(horizons_seconds=(6, 2))
    with pytest.raises(ValueError, match="track_id"):
        Entity(-1, "own")


def test_public_target_utilities_import_without_torch():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; sys.modules['torch'] = None; "
        "from simulator.rl.v4.tactical_outcomes import build_tactical_outcome_targets; "
        "assert build_tactical_outcome_targets([], []).values.shape == (0, 3, 4)",
    ], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _actions(torch, mode, slot, placement, duration=None):
    mode = torch.as_tensor(mode, dtype=torch.long)
    return SimpleNamespace(
        mode=mode, card_slot=torch.as_tensor(slot, dtype=torch.long),
        placement=torch.as_tensor(placement, dtype=torch.long),
        wait_duration=(torch.zeros_like(mode) if duration is None
                       else torch.as_tensor(duration, dtype=torch.long)),
    )


def test_action_encoding_gates_wait_placeholders_and_uses_row_column_cell_centers():
    torch = pytest.importorskip("torch")
    config = TacticalOutcomeConfig(recurrent_dim=5, placement_rows=4, placement_cols=2)
    actions = _actions(torch, [0, 1], [-99, 3], [[-99, 99], [3, 0]])
    encoded = encode_tactical_actions(actions, config=config)
    torch.testing.assert_close(encoded, torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, .875, .25],
    ]))
    assert encoded.shape == (2, config.action_dim)
    with pytest.raises(ValueError, match="in range"):
        encode_tactical_actions(_actions(torch, [1], [4], [[0, 0]]), config=config)
    with pytest.raises(ValueError, match="fixed WAIT"):
        encode_tactical_actions(_actions(torch, [0], [0], [[0, 0]], duration=[1]), config=config)


def test_head_is_action_conditioned_has_finite_gradients_and_state_dict_roundtrips():
    torch = pytest.importorskip("torch")
    config = TacticalOutcomeConfig(recurrent_dim=5, hidden_dim=8, horizons_seconds=(2., 6.))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        head = build_tactical_outcome_head(config)
    features = torch.ones((2, 1, 5), requires_grad=True)
    actions = _actions(torch, [[0], [1]], [[-1], [2]], [[[-1, -1]], [[3, 4]]])
    encoding = encode_tactical_actions(actions, config=config)
    logits = head(features, encoding)
    assert logits.shape == (2, 1, 2, 4)
    assert not torch.equal(logits[0], logits[1])
    values, mask = torch.zeros_like(logits), torch.ones_like(logits, dtype=torch.bool)
    mask[1, 0, 1] = False
    loss, metrics = tactical_outcome_loss(logits, values, mask)
    loss.backward()
    assert metrics["valid_targets"] == 12
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in head.parameters())
    payload = io.BytesIO()
    torch.save({"config": asdict(config), "state_dict": head.state_dict()}, payload)
    payload.seek(0)
    saved = torch.load(payload, weights_only=True)
    restored = build_tactical_outcome_head(TacticalOutcomeConfig(**saved["config"]))
    restored.load_state_dict(saved["state_dict"])
    torch.testing.assert_close(restored(features.detach(), encoding), logits.detach())


def test_loss_ignores_censored_values_and_counts_only_valid_elements():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 2, 4), requires_grad=True)
    values = torch.full_like(logits, float("nan"))
    values[0, 0, 1] = .2
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, 0, 1] = True
    loss, metrics = tactical_outcome_loss(logits, values, mask)
    assert loss.item() == pytest.approx(.25)  # Smooth-L1(.5-.2), beta=.1.
    assert metrics["valid_targets"] == metrics["enemy_tower_damage_count"] == 1
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 1
    assert logits.grad[0, 0, 1] > 0


def test_all_masked_loss_is_finite_differentiable_zero_even_with_invalid_placeholders():
    torch = pytest.importorskip("torch")
    head = build_tactical_outcome_head(TacticalOutcomeConfig(recurrent_dim=2, hidden_dim=4))
    features = torch.ones((2, 2), requires_grad=True)
    encoding = encode_tactical_actions(_actions(torch, [0, 0], [-1, -1], [[-1, -1], [-1, -1]]))
    logits = head(features, encoding)
    values = torch.full_like(logits, float("nan"))
    loss, metrics = tactical_outcome_loss(logits, values, torch.zeros_like(logits, dtype=torch.bool))
    assert loss.item() == 0 and metrics["valid_targets"] == 0
    loss.backward()
    assert torch.isfinite(features.grad).all() and not features.grad.any()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and not parameter.grad.any() for parameter in head.parameters())
    bad_logits = torch.full((1, 1, 4), float("nan"), requires_grad=True)
    loss, _ = tactical_outcome_loss(bad_logits, torch.full_like(bad_logits, float("nan")),
                                   torch.zeros_like(bad_logits, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(bad_logits.grad).all()


def test_loss_accepts_typed_targets_and_rejects_unmasked_nonfinite_labels():
    torch = pytest.importorskip("torch")
    targets = build_tactical_outcome_targets([_measurement(0), _measurement(2, .9, .8)],
                                            [0], horizons_seconds=(2,))
    logits = torch.zeros((1, 1, 4), requires_grad=True)
    loss, metrics = tactical_outcome_loss(logits, targets)
    assert torch.isfinite(loss) and metrics["valid_targets"] == 2
    with pytest.raises(ValueError, match="already contain"):
        tactical_outcome_loss(logits, targets, targets.mask)
    with pytest.raises(ValueError, match="finite fractions"):
        tactical_outcome_loss(logits, torch.full_like(logits, float("nan")),
                              torch.ones_like(logits, dtype=torch.bool))
