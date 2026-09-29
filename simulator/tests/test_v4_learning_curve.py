import copy
import json

import pytest

from simulator.rl.v4.learning_curve import (
    LearningCurveConfig, LearningCurveHooks, TrainingState,
    paired_match_statistics, run_final_evaluation, run_learning_curve,
    select_development_checkpoint, sha256_file, smoke_hooks, validate_entry_split,
)


def test_censored_matches_are_not_draws():
    baseline = [dict(seed=1, result='loss'), dict(seed=2, result='win')]
    candidate = [dict(seed=1, result='win'), dict(seed=2, result='truncated')]
    result = paired_match_statistics(baseline, candidate, max_decisions=10)
    assert result['n_paired'] == 1
    assert result['n_censored'] == 1
    assert result['paired_deltas'] == [1.0]
    assert result['candidate']['draws'] == 0
    assert result['paired_std_score_delta'] == float('inf')
    assert result['paired_se_score_delta'] == float('inf')
    assert result['paired_cluster_count'] == 1
    assert result['paired_inference_valid'] is False
    assert result['paired_censoring_bounds'] == {
        'lower': 0.0, 'upper': 1.0, 'n_total_pairs': 2,
    }


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [
        ([dict(seed=1, result="win")], []),
        ([], [dict(seed=1, result="win")]),
        ([dict(seed=1, result="win")], [dict(seed=2, result="win")]),
    ],
)
def test_paired_matches_reject_missing_or_mismatched_rows(baseline, candidate):
    with pytest.raises(ValueError, match="paired evaluation"):
        paired_match_statistics(baseline, candidate, max_decisions=10)


def test_paired_matches_reject_rows_without_identity():
    with pytest.raises(ValueError, match="explicit seed identity"):
        paired_match_statistics(
            [dict(result="win")], [dict(result="loss")], max_decisions=10
        )


def test_paired_matches_require_seed_and_real_deck_identity():
    with pytest.raises(ValueError, match="explicit seed identity"):
        paired_match_statistics(
            [dict(archetype="hog", result="win")],
            [dict(archetype="hog", result="loss")],
            max_decisions=10,
        )


def test_paired_match_uncertainty_clusters_shared_match_seed():
    baseline = [
        dict(archetype='a', deck_id='d1', seed=7, result='loss'),
        dict(archetype='a', deck_id='d2', seed=7, result='loss'),
        dict(archetype='a', deck_id='d1', seed=8, result='loss'),
        dict(archetype='a', deck_id='d2', seed=8, result='loss'),
    ]
    candidate = [
        dict(archetype='a', deck_id='d1', seed=7, result='win'),
        dict(archetype='a', deck_id='d2', seed=7, result='win'),
        dict(archetype='a', deck_id='d1', seed=8, result='loss'),
        dict(archetype='a', deck_id='d2', seed=8, result='loss'),
    ]
    result = paired_match_statistics(baseline, candidate, max_decisions=10)
    assert result['paired_cluster_count'] == 2
    assert result['paired_se_score_delta'] > 0
    assert result['paired_se_score_delta'] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="deck ID"):
        paired_match_statistics(
            [dict(archetype="hog", seed=7, real_deck_split="eval", result="win")],
            [dict(archetype="hog", seed=7, real_deck_split="eval", result="loss")],
            max_decisions=10,
        )


def test_paired_matches_reject_zero_completed_pairs():
    rows = [dict(seed=1, result="truncated")]
    with pytest.raises(ValueError, match="zero completed pairs"):
        paired_match_statistics(rows, rows, max_decisions=10)


def test_source_fingerprint_covers_nested_runtime_changes(tmp_path):
    from simulator.rl.v4.learning_curve import dirty_source_fingerprint

    package = tmp_path / 'runtime'
    package.mkdir()
    implementation = package / 'rollout.py'
    implementation.write_text('VERSION = 1\n')
    first = dirty_source_fingerprint([package])
    implementation.write_text('VERSION = 2\n')
    second = dirty_source_fingerprint([package])
    assert first['files'] != second['files']


def test_default_source_fingerprint_includes_ppo_packages():
    from simulator.rl.v4.learning_curve import dirty_source_fingerprint

    files = dirty_source_fingerprint()['files']
    for package in ('v4_ppo', 'frontier_ppo'):
        prefix = f'simulator/rl/v4/{package}/'
        assert prefix + '__init__.py' in files
        assert len([name for name in files if name.startswith(prefix)]) > 1


def test_split_rejects_shared_match_despite_different_states():
    train = [{'provenance': {'state_hash': 'a', 'match_id': 'shared'}}]
    sealed = [{'provenance': {'state_hash': 'b', 'match_id': 'shared'}}]
    with pytest.raises(ValueError, match='match origins'):
        validate_entry_split(train, sealed)


def test_split_allows_shared_development_deck_identity_with_fresh_states():
    train = [{
        'provenance': {'state_hash': 'a', 'match_id': 'train-match'},
        'opponent_deck_id': 'shared-deck',
    }]
    sealed = [{
        'provenance': {'state_hash': 'b', 'match_id': 'sealed-match'},
        'opponent_deck_id': 'shared-deck',
    }]
    result = validate_entry_split(train, sealed, require_deck_identity=True)
    assert result['shared_deck_count'] == 1


def test_split_allows_shared_catalog_train_deck_with_fresh_matches():
    from simulator.rl.v4.real_decks import real_decks_for

    deck = real_decks_for('train', 'beatdown')[0]
    def entry(state, match):
        return {
            'provenance': {'state_hash': state, 'match_id': match},
            'opponent_deck_id': deck.deck_id,
            'opponent_deck': list(deck.cards),
            'real_deck_split': 'train',
        }

    result = validate_entry_split(
        [entry('a', 'train-match')],
        [entry('b', 'sealed-match')],
        require_deck_identity=True,
        require_catalog_identity=True,
    )
    assert result['shared_deck_count'] == 1


def test_split_rejects_reserved_eval_deck_in_sealed_development_entries():
    from simulator.rl.v4.real_decks import real_decks_for

    deck = real_decks_for('eval', 'beatdown')[0]
    entries = [{
        'provenance': {'state_hash': 'sealed-state', 'match_id': 'sealed-match'},
        'opponent_deck_id': deck.deck_id,
        'real_deck_split': 'eval',
    }]
    with pytest.raises(ValueError, match='reserved evaluation-deck'):
        validate_entry_split(
            [], entries,
            require_deck_identity=True,
            require_catalog_identity=True,
        )


def test_split_rejects_unknown_catalog_deck_in_strict_comparison_entries():
    entries = [{
        'provenance': {'state_hash': 'train-state', 'match_id': 'train-match'},
        'opponent_deck_id': 'not-a-catalog-deck',
    }]
    with pytest.raises(ValueError, match='catalog_deck'):
        validate_entry_split(
            entries, [],
            require_deck_identity=True,
            require_catalog_identity=True,
        )


def test_learning_curve_default_pool_is_disjoint_from_reserved_eval_catalog():
    from simulator.rl.v4.learning_curve import (
        _default_final_opponents, _default_opponents, _validate_opponent_specs,
    )
    from simulator.rl.v4.real_decks import real_decks_for

    development = _default_opponents()
    final = _default_final_opponents()
    development_ids = {row['deck_id'] for row in development}
    final_ids = {row['deck_id'] for row in final}
    assert len(development_ids) == 30
    assert len(final_ids) == 10
    assert development_ids.isdisjoint(final_ids)
    assert {row['real_deck_split'] for row in development} == {'train'}
    eval_deck = real_decks_for('eval', 'beatdown')[0]
    with pytest.raises(ValueError, match='does not match'):
        _validate_opponent_specs([{
            'deck_id': 'not-a-catalog-id', 'deck': list(eval_deck.cards),
        }], final=True)


def test_checkpoint_selection_rejects_censored_points_and_ignores_sealed_diagnostics():
    points = [
        {
            'update': 0,
            'checkpoint': {'path': 'zero.pt'},
            'evaluation': {
                'full_match': {'candidate': {'completed': 4, 'truncated': 1, 'score_mean_completed': 1.0}},
                'sealed_choice': {'summary': {'mean_delta': -100.0}},
            },
        },
        {
            'update': 1,
            'checkpoint': {'path': 'one.pt'},
            'evaluation': {
                'full_match': {'candidate': {'completed': 4, 'score_mean_completed': 0.75}},
                'sealed_choice': {'summary': {'mean_delta': -1000.0}},
            },
        },
    ]
    with pytest.raises(ValueError, match='censored'):
        select_development_checkpoint(points)
    points[0]['evaluation']['full_match']['candidate']['truncated'] = 0
    points[0]['evaluation']['full_match']['candidate']['score_mean_completed'] = 0.5
    selected = select_development_checkpoint(points)
    assert selected['selected_update'] == 1
    assert selected['metric'] == 'full_match.candidate.score_mean_completed'
    assert 'sealed_choice' in selected['excludes']


def test_checkpoint_selection_rejects_fully_censored_scheduled_point():
    points = [
        {
            'update': 0,
            'checkpoint': {'path': 'zero.pt'},
            'evaluation': {
                'full_match': {
                    'baseline': {'completed': 0, 'truncated': 2},
                    'candidate': {'completed': 0, 'truncated': 2},
                },
            },
        },
        {
            'update': 1,
            'checkpoint': {'path': 'one.pt'},
            'evaluation': {
                'full_match': {
                    'baseline': {'completed': 2, 'truncated': 0},
                    'candidate': {
                        'completed': 2,
                        'truncated': 0,
                        'score_mean_completed': 0.5,
                    },
                },
            },
        },
    ]
    with pytest.raises(ValueError, match='censored'):
        select_development_checkpoint(points)


def test_checkpoint_selection_tie_breaks_by_cumulative_training_cost():
    def point(update, cost):
        return {
            'update': update,
            'checkpoint': {'path': f'{update}.pt'},
            'cumulative_training_cost': cost,
            'evaluation': {
                'full_match': {
                    'baseline': {'completed': 2, 'truncated': 0},
                    'candidate': {
                        'completed': 2,
                        'truncated': 0,
                        'score_mean_completed': 0.5,
                    },
                },
            },
        }

    selected = select_development_checkpoint([point(4, 512), point(2, 256)])
    assert selected['selected_update'] == 2


def test_final_evaluation_uses_frozen_actors_without_training_or_selection(tmp_path):
    from simulator.rl.v4.real_decks import real_decks_for

    primary = tmp_path / 'primary.pt'
    comparison = tmp_path / 'comparison.pt'
    comparison_two = tmp_path / 'comparison_two.pt'
    primary.write_bytes(b'primary')
    comparison.write_bytes(b'comparison')
    comparison_two.write_bytes(b'comparison-two')
    deck = real_decks_for('eval', 'beatdown')[0]
    calls = {'load': 0, 'evaluate': 0}

    def load_policy(**_kwargs):
        calls['load'] += 1
        return object()

    def forbidden(**_kwargs):
        raise AssertionError('final evaluation must not train or select')

    seen_deterministic: list[bool] = []

    def evaluate_matches(**kwargs):
        calls['evaluate'] += 1
        seen_deterministic.append(bool(kwargs['config'].deterministic_policy))
        return [
            {
                'archetype': deck.archetype,
                'deck_id': deck.deck_id,
                'seed': seed,
                'result': 'win',
                'decisions': 1,
            }
            for seed in kwargs['seeds']
        ]

    hooks = LearningCurveHooks(
        load_policy=load_policy,
        clone_policy=forbidden,
        make_training_state=forbidden,
        collect_rollout=forbidden,
        update=forbidden,
        evaluate_matches=evaluate_matches,
        evaluate_choice=forbidden,
        save_checkpoint=forbidden,
    )
    config = LearningCurveConfig(
        checkpoint=primary,
        comparison_checkpoints=(comparison, comparison_two),
        opponent_specs=({'archetype': deck.archetype, 'deck_id': deck.deck_id},),
        eval_seeds=(3, 4),
        output_dir=tmp_path / 'final',
    )
    report = run_final_evaluation(config, hooks=hooks)
    assert report['final_evaluation'] is True
    assert report['checkpoint_selection']['performed'] is False
    assert len(report['actors']) == 3
    assert {(row['baseline_actor'], row['candidate_actor']) for row in report['pairwise']} == {
        (0, 1), (0, 2), (1, 2)
    }
    assert report['protocol']['winner_claims_permitted'] is True
    assert calls == {'load': 3, 'evaluate': 3}
    assert seen_deterministic == [False, False, False]
    assert (config.output_dir / 'final_evaluation.json').is_file()


def test_development_curve_can_run_without_sealed_entries(tmp_path):
    from simulator.rl.v4.real_decks import real_decks_for

    checkpoint = tmp_path / 'input.pt'
    checkpoint.write_bytes(b'input')
    deck = real_decks_for('train', 'beatdown')[0]
    train_path = tmp_path / 'train.pt'
    train_path.write_bytes(b'placeholder')
    import torch

    torch.save([{
        'entry_id': 'train-entry',
        'provenance': {'state_hash': 'state', 'match_seed': 11},
        'opponent_deck_id': deck.deck_id,
        'real_deck_split': 'train',
    }], train_path)

    class Policy:
        pass

    def matches(**kwargs):
        return [
            {
                'archetype': deck.archetype,
                'deck_id': deck.deck_id,
                'seed': seed,
                'result': 'win',
                'decisions': 1,
            }
            for seed in kwargs['seeds']
        ]

    def save_checkpoint(**kwargs):
        kwargs['path'].write_bytes(b'point')

    hooks = LearningCurveHooks(
        load_policy=lambda **_kwargs: Policy(),
        clone_policy=lambda policy: policy,
        make_training_state=lambda **_kwargs: TrainingState(),
        collect_rollout=lambda **_kwargs: ({}, {}),
        update=lambda **_kwargs: {},
        evaluate_matches=matches,
        evaluate_choice=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError('sealed choice must not run without sealed entries')
        ),
        save_checkpoint=save_checkpoint,
    )
    config = LearningCurveConfig(
        checkpoint=checkpoint,
        train_entries=train_path,
        opponent_specs=({'archetype': deck.archetype, 'deck_id': deck.deck_id},),
        eval_seeds=(3, 4),
        update_schedule=(0,),
        output_dir=tmp_path / 'development',
    )
    report = run_learning_curve(config, hooks=hooks)
    assert report['provenance']['separation']['sealed_entry_count'] == 0
    assert report['checkpoint_selection']['performed'] is True


def test_scripted_frontier_entries_require_explicit_opt_in(tmp_path):
    import torch
    from simulator.rl.v4.real_decks import real_decks_for

    checkpoint = tmp_path / 'actor.pt'
    checkpoint.write_bytes(b'actor')
    entries = tmp_path / 'scripted.pt'
    torch.save([{
        'entry_id': 'train:scripted:t0',
        'provenance': {'state_hash': 'scripted-state', 'match_seed': 91},
        'opponent_deck_id': 'scripted-opponent',
    }], entries)
    deck = real_decks_for('train', 'beatdown')[0]

    def matches(**kwargs):
        return [{
            'archetype': deck.archetype, 'deck_id': deck.deck_id,
            'seed': seed, 'result': 'draw', 'terminated': True,
        } for seed in kwargs['seeds']]

    hooks = LearningCurveHooks(
        load_policy=lambda **_kwargs: object(),
        clone_policy=lambda policy: policy,
        make_training_state=lambda **_kwargs: TrainingState(),
        collect_rollout=lambda **_kwargs: ({}, {}),
        update=lambda **_kwargs: {},
        evaluate_matches=matches,
        save_checkpoint=lambda **kwargs: kwargs['path'].write_bytes(b'point'),
    )
    config_args = dict(
        checkpoint=checkpoint,
        train_entries=entries,
        opponent_specs=({'archetype': deck.archetype, 'deck_id': deck.deck_id},),
        eval_seeds=(3, 4),
        evaluation_only=True,
        output_dir=tmp_path / 'curve',
    )
    with pytest.raises(ValueError, match='catalog_deck'):
        run_learning_curve(LearningCurveConfig(**config_args), hooks=hooks)
    report = run_learning_curve(
        LearningCurveConfig(**config_args, allow_scripted_frontier_entries=True),
        hooks=hooks,
    )
    assert report['checkpoint_selection']['performed'] is True
    assert report['provenance']['separation']['development_catalog_only'] is False


def test_default_match_hook_exposes_seeded_sampling_mode(monkeypatch):
    import simulator.rl.v4.frontier_ppo as frontier_ppo
    from simulator.rl.v4.learning_curve import _default_hooks

    seen: list[bool] = []

    def fake_full_match_record(**kwargs):
        seen.append(bool(kwargs['deterministic']))
        return []

    monkeypatch.setattr(frontier_ppo, 'full_match_record', fake_full_match_record)
    hooks = _default_hooks()
    hooks.evaluate_matches(
        policy=object(), opponent_specs=(), seeds=(),
        config=LearningCurveConfig(deterministic_policy=False),
    )
    hooks.evaluate_matches(
        policy=object(), opponent_specs=(), seeds=(),
        config=LearningCurveConfig(deterministic_policy=True),
    )
    assert seen == [False, True]


def test_curve_artifacts_and_overwrite_protection(tmp_path):
    config = LearningCurveConfig(output_dir=tmp_path / 'curve', smoke=True,
                                 update_schedule=(0, 1), eval_matches=2)
    report = run_learning_curve(config, hooks=smoke_hooks())
    saved = json.loads((config.output_dir / 'learning_curve.json').read_text())
    assert saved == report
    assert len(report['points']) == 2
    for name in ('config', 'provenance'):
        artifact = report['artifacts'][name]
        assert artifact['sha256'] == sha256_file(artifact['path'])
    assert 'sha256' not in report['artifacts']['curve']
    with pytest.raises(FileExistsError, match='not empty|non-empty|already'):
        run_learning_curve(config, hooks=smoke_hooks())


def test_comparison_checkpoint_is_paired_directly_with_final_curve_point(tmp_path):
    comparison = tmp_path / 'comparison.pt'
    comparison.write_bytes(b'smoke comparison fixture')
    config = LearningCurveConfig(
        output_dir=tmp_path / 'paired', smoke=True,
        update_schedule=(0, 1), eval_seeds=(1, 2, 3),
        comparison_checkpoints=(comparison,),
    )
    report = run_learning_curve(config, hooks=smoke_hooks())
    direct = report['comparison_checkpoints'][0]['versus_final_curve_point']
    assert direct['curve_update'] == 1
    assert direct['full_match']['n_paired'] == 3
    assert direct['full_match']['n_censored'] == 0


def test_quarantined_update_does_not_create_candidate_checkpoint(tmp_path):
    hooks = smoke_hooks()
    hooks.update = lambda **kwargs: {'quarantined': True, 'quarantine_reasons': ['test']}
    config = LearningCurveConfig(output_dir=tmp_path / 'quarantine', smoke=True,
                                 update_schedule=(0, 1, 2), eval_matches=2)
    report = run_learning_curve(config, hooks=hooks)
    assert len(report['points']) == 1
    assert report['protocol']['quarantine_stop']['stopped_at_update'] == 1
    assert report['protocol']['quarantine_stop']['weights_restored'] is True
    assert not (config.output_dir / 'checkpoint_update_000001.pt').exists()


def test_quarantined_update_restores_the_pre_update_policy(tmp_path):
    """A quarantined update must not leave its weights in the model.

    The diagnostics that trigger quarantine are computed after
    ``optimizer.step()``, so the update has already moved the policy. The run
    has to put the parameters and optimizer state back, otherwise a quarantined
    run ends holding exactly the weights that failed its own checks.
    """

    import torch

    from simulator.rl.v4.learning_curve import (
        _capture_pre_update_state,
        _restore_pre_update_state,
    )

    policy = torch.nn.Linear(3, 2)
    critic = torch.nn.Linear(2, 1)
    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)

    def take_a_step() -> None:
        loss = policy(torch.ones(1, 3)).sum() + critic(torch.ones(1, 2)).sum()
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()

    # The snapshot the loop takes before each update.
    snapshot = _capture_pre_update_state(
        policy=policy,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )
    expected_policy = copy.deepcopy(policy.state_dict())
    expected_critic = copy.deepcopy(critic.state_dict())

    # The update that trips quarantine.
    for _ in range(20):
        take_a_step()
    assert any(
        not torch.equal(expected_policy[name], value)
        for name, value in policy.state_dict().items()
    )

    _restore_pre_update_state(
        policy=policy,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        snapshot=snapshot,
    )
    restored_policy = policy.state_dict()
    restored_critic = critic.state_dict()
    assert all(
        torch.equal(expected_policy[name], value) for name, value in restored_policy.items()
    )
    assert all(
        torch.equal(expected_critic[name], value) for name, value in restored_critic.items()
    )


def test_pre_update_state_tolerates_absent_components():
    """The hooks boundary allows a run with no critic or optimizer."""

    from simulator.rl.v4.learning_curve import (
        _capture_pre_update_state,
        _restore_pre_update_state,
    )

    class Policy:
        def __init__(self) -> None:
            self.values = {"w": 1}

        def state_dict(self):
            return dict(self.values)

        def load_state_dict(self, state):
            self.values = dict(state)

    policy = Policy()
    snapshot = _capture_pre_update_state(
        policy=policy,
        critic=None,
        actor_optimizer=None,
        critic_optimizer=None,
    )
    policy.values["w"] = 99
    _restore_pre_update_state(
        policy=policy,
        critic=None,
        actor_optimizer=None,
        critic_optimizer=None,
        snapshot=snapshot,
    )
    assert policy.values == {"w": 1}


def test_extra_evaluation_points_do_not_change_training_randomness(tmp_path):
    import numpy as np
    draws = []
    for label, schedule in [('sparse', (0, 2)), ('dense', (0, 1, 2))]:
        observed = []
        hooks = smoke_hooks()
        original = hooks.collect_rollout
        def collect(_original=original, **kwargs):
            observed.append(float(np.random.random()))
            return _original(**kwargs)
        hooks.collect_rollout = collect
        config = LearningCurveConfig(output_dir=tmp_path / label, smoke=True,
                                     update_schedule=schedule, eval_matches=2)
        run_learning_curve(config, hooks=hooks)
        draws.append(observed)
    assert len(draws[0]) == 2
    assert draws[0] == draws[1]
