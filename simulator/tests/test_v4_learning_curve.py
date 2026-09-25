import json

import pytest

from simulator.rl.v4.learning_curve import (
    LearningCurveConfig, paired_match_statistics, run_learning_curve,
    sha256_file, smoke_hooks, validate_entry_split,
)


def test_censored_matches_are_not_draws():
    baseline = [dict(seed=1, result='loss'), dict(seed=2, result='win')]
    candidate = [dict(seed=1, result='win'), dict(seed=2, result='truncated')]
    result = paired_match_statistics(baseline, candidate, max_decisions=10)
    assert result['n_paired'] == 1
    assert result['n_censored'] == 1
    assert result['paired_deltas'] == [1.0]
    assert result['candidate']['draws'] == 0


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


def test_quarantined_update_does_not_create_candidate_checkpoint(tmp_path):
    hooks = smoke_hooks()
    hooks.update = lambda **kwargs: {'quarantined': True, 'quarantine_reasons': ['test']}
    config = LearningCurveConfig(output_dir=tmp_path / 'quarantine', smoke=True,
                                 update_schedule=(0, 1, 2), eval_matches=2)
    report = run_learning_curve(config, hooks=hooks)
    assert len(report['points']) == 1
    assert report['protocol']['quarantine_stop']['stopped_at_update'] == 1
    assert not (config.output_dir / 'checkpoint_update_000001.pt').exists()


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
