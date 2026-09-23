"""Public bookkeeping must agree for dense observations and restored prefixes."""
from types import SimpleNamespace
import numpy as np
import pytest

from simulator.actions import WaitAction
from simulator.events import SimEvent
from simulator.observation import ObservationMemory
from simulator.public_state_estimator import PublicEvent, PublicStateEstimator, belief_index
from simulator.rl.v4.frontier_ppo import make_frontier_env
from simulator.roster import PLAYER_DECK


def env():
    result = make_frontier_env()
    result.reset_v2(seed=4, decks=(tuple(PLAYER_DECK), tuple(PLAYER_DECK)), shuffle_decks=False)
    return result


def append_event(environment, kind, **data):
    st = environment.state
    st.events.append(SimEvent.create(st.tick, st.event_sequence, kind, **data))
    st.event_sequence += 1


def assert_snapshot_same(a, b):
    assert a.tick_seconds == b.tick_seconds
    assert a.opp_elixir_lo == pytest.approx(b.opp_elixir_lo)
    assert a.opp_elixir_hi == pytest.approx(b.opp_elixir_hi)
    for key in ('opp_hand_probs', 'opp_out_of_cycle', 'event_history'):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))


def test_counter_returns_after_four_other_plays():
    est = PublicStateEstimator()
    cards = ['cannon', 'skeletons', 'ice-spirit', 'ice-golem', 'log']
    for i, key in enumerate(cards):
        est.observe_opponent_spend(tick_seconds=float(i), card_key=key, elixir_cost=0.)
        assert est.out_of_cycle()[belief_index('cannon')] == (i < 4)
        assert est.out_of_cycle()[belief_index(key)]
    assert est.hand_belief()[belief_index('cannon')] > 0
    assert not est.out_of_cycle()[belief_index('balloon')]  # not yet observed != impossible


def test_belief_ids_are_metadata_ids_without_hash_collisions():
    from cr_bot.domain.card_metadata import CARD_METADATA
    indices = [belief_index(key) for key in CARD_METADATA]
    assert len(set(indices)) == len(indices)
    assert all(belief_index(key) == value['id'] for key, value in CARD_METADATA.items())
    with pytest.raises(ValueError, match='vocabulary'):
        belief_index('unrecognized-card')


def test_spending_subtracts_from_both_bounds():
    est = PublicStateEstimator()
    est.observe(PublicEvent('own_deploy', 14., 0.))
    assert est.elixir_interval() == pytest.approx((5., 10.))
    est.observe_opponent_spend(tick_seconds=14., card_key='hog-rider', elixir_cost=4.)
    assert est.elixir_interval() == pytest.approx((1., 6.))


def test_sync_dense_and_restored_public_prefix_match():
    game = env()
    dense = PublicStateEstimator()
    dense.sync_from_env(game)
    for i, card in enumerate(['cannon', 'skeletons', 'ice-spirit', 'ice-golem', 'log']):
        append_event(game, 'card_played', player=1, card_id=card, cost_milli=1000)
        game.step_v2((WaitAction(0), WaitAction(1)))
        dense.sync_from_env(game)
    before = game.state.state_hash()
    snapshot = dense.sync_from_env(game)
    assert not snapshot.opp_out_of_cycle[belief_index('cannon')]
    assert snapshot.event_history.any()
    sparse = PublicStateEstimator().sync_from_env(game)
    assert_snapshot_same(snapshot, sparse)
    assert_snapshot_same(snapshot, dense.sync_from_env(game))
    assert game.state.state_hash() == before
    restored = game.fork()
    assert_snapshot_same(snapshot, dense.sync_from_env(restored))
    # Same-seed reset must not retain cards or events from the old episode.
    game.reset_v2(seed=4, decks=(tuple(PLAYER_DECK), tuple(PLAYER_DECK)), shuffle_decks=False)
    clean = dense.sync_from_env(game)
    assert not clean.opp_out_of_cycle.any()
    assert not clean.event_history.any()


def test_sync_uses_phase_aware_public_elixir_and_effective_cost():
    game = env()
    game.state.elapsed_us = 120_000_000
    game.state.tick = 2400
    append_event(game, 'card_played', player=1, card_id='mirror', cost_milli=7000)
    game.state.elapsed_us += 2_800_000
    game.state.tick += 56
    snapshot = PublicStateEstimator().sync_from_env(game)
    public = ObservationMemory(0)
    public.update(game.state, game.engine.ruleset)
    assert snapshot.opp_elixir_hi == pytest.approx(public.opponent_elixir_est)
    assert snapshot.opp_elixir_hi > 4.5  # double-elixir regen after the observed spend


def test_sync_never_reads_opponent_private_player_state():
    game = env()
    append_event(game, 'card_played', player=1, card_id='cannon', cost_milli=3000)
    game.step_v2((WaitAction(0), WaitAction(1)))
    st = game.state
    # Deliberately omit all players/hand/elixir fields from the state boundary.
    public = SimpleNamespace(seed=st.seed, ruleset_id=st.ruleset_id, ruleset_hash=st.ruleset_hash,
        tick=st.tick, elapsed_us=st.elapsed_us, event_sequence=st.event_sequence,
        events=st.events, entities=st.entities)
    snapshot = PublicStateEstimator().sync_from_env(SimpleNamespace(state=public, engine=game.engine))
    assert snapshot.opp_out_of_cycle[belief_index('cannon')]


def test_tower_damage_own_deploy_death_events_are_classified_once():
    game = env()
    tower = next(e for e in game.state.entities.values() if e.kind == 'tower' and e.owner == 0)
    append_event(game, 'card_played', player=0, card_id='skeletons', cost_milli=1000)
    append_event(game, 'damage_applied', target_uid=tower.uid, damage=200)
    append_event(game, 'entity_died', player=1, card_id='skeletons', uid=999)
    est = PublicStateEstimator()
    first = est.sync_from_env(game)
    from simulator.observation_v3 import EVENT_TYPES
    for kind in ('own_deploy', 'action_taken', 'own_tower_damaged', 'unit_died'):
        assert first.event_history[:, EVENT_TYPES.index(kind)].sum() == 1
    assert_snapshot_same(first, est.sync_from_env(game))


def test_rewritten_history_and_viewer_switch_reconstruct():
    game = env()
    append_event(game, 'card_played', player=1, card_id='cannon', cost_milli=3000)
    est = PublicStateEstimator()
    assert est.sync_from_env(game).opp_out_of_cycle[belief_index('cannon')]
    previous = game.state.events[-1]
    game.state.events[-1] = SimEvent.create(previous.tick, previous.sequence, 'card_played',
        player=1, card_id='log', cost_milli=2000)
    changed = est.sync_from_env(game)
    assert changed.opp_out_of_cycle[belief_index('log')]
    assert not changed.opp_out_of_cycle[belief_index('cannon')]
    assert not est.sync_from_env(game, viewer=1).opp_out_of_cycle.any()
