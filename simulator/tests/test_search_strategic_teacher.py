from types import SimpleNamespace

import numpy as np
import pytest

from simulator.rl.v4 import search_teacher as st
from simulator.rl.v4.search_dataset import group_search_sequences, _record_to_inference_sample


def test_placement_support_contains_only_scored_cells():
    result = st.SearchResult('hash', 0, 'test', 'test', 4, branches=[
        st.BranchScore(st.BranchAction('play', 0, 1, 1), 2., {}, 4),
        st.BranchScore(st.BranchAction('play', 0, 2, 2), 1., {}, 4),
    ])
    target = st.tested_placement_target(result, slot=0, legal=np.ones((4, 4), bool))
    assert np.count_nonzero(target) == 2
    assert target[1, 1] > target[2, 2] > 0
    assert target.sum() == pytest.approx(1)


def test_recurrent_runs_split_at_gaps_and_duplicates():
    samples = [SimpleNamespace(provenance={'seq_id': 'match', 'sequence_offset': i})
               for i in (0, 1, 3, 3, 4)]
    assert [len(run) for run in group_search_sequences(samples)] == [2, 1, 2]


def test_malformed_public_history_is_rejected():
    with pytest.raises(ValueError, match='event_history'):
        _record_to_inference_sample({'event_history': np.zeros((1, 8))})


def test_active_continuation_reserves_first_decision_and_copies_controller(monkeypatch):
    class Controller:
        def __init__(self):
            self.calls = 0
        def choose_public_action(self, observation, player):
            self.calls += 1
            return 'learner'
    class Env:
        state = SimpleNamespace(terminal=False)
        engine = object()
        def __init__(self):
            self.actions = []
        def observe_v2_for_viewer(self, viewer):
            return object()
        def step_v2(self, actions):
            self.actions.append(actions)
            return SimpleNamespace(terminated=False, truncated=False)
    opponent = SimpleNamespace(choose_action=lambda *args: 'opponent')
    monkeypatch.setattr(st, '_active_opponent', lambda **kwargs: opponent)
    controller = Controller()
    env = Env()
    assert st._rollout_active(env, 3, learner_controller=controller) == 3
    assert env.actions[0][0] != 'learner'
    assert env.actions[1:] == [('learner', 'opponent')] * 2
    assert controller.calls == 0


def test_active_dataset_trains_a_real_recurrent_sequence():
    import torch
    from scripts.run_v4_stage2_proof import _proof_config, _train
    from simulator.rl.v4.model_v4 import RecurrentV4Policy
    from simulator.rl.v4.search_dataset import SearchDatasetConfig, generate_search_dataset
    champion = RecurrentV4Policy(_proof_config()).eval()
    config = SearchDatasetConfig(n_states=3, continuation_mode='active',
                                 continuation_horizon=2, max_branches=2,
                                 active_match_decisions=3)
    samples, stats = generate_search_dataset(config, champion)
    assert stats['active']
    assert [len(seq) for seq in group_search_sequences(samples)] == [3]
    assert not champion.training
    trained = _train(samples, epochs=1, batch_size=1, lr=1e-4, seed=0, device='cpu')
    assert all(torch.isfinite(p).all() for p in trained.parameters())
