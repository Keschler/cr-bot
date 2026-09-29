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
from simulator.state import EntityState, StatusState


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
    game.state.entities[999] = EntityState(
        uid=999, card_id='skeletons', owner=1, kind='troop',
        x_mtile=5000, y_mtile=17000, hp=0, max_hp=1,
        spawn_tick=0, alive=False,
    )
    append_event(game, 'card_played', player=0, card_id='skeletons', cost_milli=1000)
    append_event(game, 'damage_applied', target_uid=tower.uid, damage=200)
    append_event(game, 'entity_died', player=1, card_id='skeletons', uid=999)
    est = PublicStateEstimator()
    first = est.sync_from_env(game)
    from simulator.observation_v3 import EVENT_TYPES
    for kind in ('own_deploy', 'action_taken', 'own_tower_damaged', 'unit_died'):
        assert first.event_history[:, EVENT_TYPES.index(kind)].sum() == 1
    assert_snapshot_same(first, est.sync_from_env(game))


def test_sync_omits_concealed_death_from_public_history():
    game = env()
    game.state.entities[999] = EntityState(
        uid=999, card_id='skeletons', owner=1, kind='troop',
        x_mtile=5000, y_mtile=17000, hp=0, max_hp=1,
        spawn_tick=0, alive=False,
        statuses=[StatusState(kind='stealth', remaining_us=1000)],
    )
    append_event(game, 'entity_died', player=1, card_id='skeletons', uid=999)
    snapshot = PublicStateEstimator().sync_from_env(game)
    from simulator.observation_v3 import EVENT_TYPES
    assert snapshot.event_history[:, EVENT_TYPES.index('unit_died')].sum() == 0


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


def test_death_of_a_concealed_body_is_not_part_of_the_public_stream() -> None:
    """A hidden body that dies was never observed, so its death is not public.

    The event carries no card identity, but it still told the actor that
    something died at a time when the board showed nothing, which is a
    hidden-information leak. A visible body dying is still reported.
    """

    from simulator.engine import BattleEngine
    from simulator.public_state_estimator import EVENT_TYPE_INDEX
    from simulator.ruleset import load_ruleset

    ruleset = load_ruleset("v1")
    deck = ("hog-rider", "cannon", "musketeer", "skeletons", "fireball", "log", "ice-golem", "ice-spirit")
    death_index = EVENT_TYPE_INDEX["unit_died"]

    def death_count(*, concealed: bool) -> int:
        engine = BattleEngine(ruleset)
        state = engine.new_battle(decks=(deck, deck), seed=3, shuffle_decks=False)
        card = ruleset.card("royal-ghost")
        flags = {"stealth_active": True} if concealed else {}
        ghost = EntityState(
            uid=state.next_uid,
            card_id="royal-ghost",
            owner=1,
            kind="troop",
            x_mtile=4_000,
            y_mtile=6_000,
            hp=int(card.hitpoints),
            max_hp=int(card.hitpoints),
            spawn_tick=state.tick,
            parent_uid=None,
            **flags,
        )
        state.next_uid += 1
        state.entities[ghost.uid] = ghost
        engine.step(state)
        ghost.hp = 0
        engine.step(state)
        estimator = PublicStateEstimator()
        estimator.sync_from_env(SimpleNamespace(state=state, engine=engine))
        return int(estimator.snapshot().event_history[:, death_index].sum())

    assert death_count(concealed=False) == 1
    assert death_count(concealed=True) == 0


def test_tower_damage_magnitude_is_the_exact_engine_damage() -> None:
    """Pin the precision the actor actually receives, so the gap stays visible.

    The actor receives the exact ``damage_applied`` magnitude, and the tower HP
    globals are normalized by the exact Level-11 maximum so the actor recovers
    integer HP. A phone can only estimate both. This is a sim-to-real precision
    gap, not a privilege leak, and it is documented in simulator/README.md. The
    test exists so that closing the gap cannot happen silently: any change to
    quantize these values has to update the documentation in the same commit.
    """

    from simulator.engine import BattleEngine
    from simulator.public_state_estimator import EVENT_TYPE_INDEX
    from simulator.ruleset import load_ruleset

    ruleset = load_ruleset("v1")
    deck = ("hog-rider", "cannon", "musketeer", "skeletons", "fireball", "log", "ice-golem", "ice-spirit")
    engine = BattleEngine(ruleset)
    state = engine.new_battle(decks=(deck, deck), seed=17, shuffle_decks=False)
    enemy_tower = next(
        entity
        for entity in state.entities.values()
        if entity.kind == "tower" and entity.owner == 1
    )
    before = enemy_tower.hp
    engine.step(state)
    # Apply a known authoritative amount through the normal damage path.
    engine._deal_damage(state, enemy_tower, 123, None, "fireball", 1)

    estimator = PublicStateEstimator()
    estimator.sync_from_env(SimpleNamespace(state=state, engine=engine))
    history = estimator.snapshot().event_history
    index = EVENT_TYPE_INDEX["enemy_tower_damaged"]
    assert any(row[index] == 1.0 for row in history), (
        "the enemy tower damage event must reach the actor memory"
    )
    # The magnitude column is the exact engine damage on a 0.1 kHP scale, so the
    # actor can still distinguish a Fireball from a Log. A phone derives the
    # same event from HP-bar differences instead.
    magnitudes = [row[7] for row in history if row[7] > 0.0]
    assert magnitudes, "the damage magnitude column must be populated"
    assert max(magnitudes) == pytest.approx(123.0 / 1000.0 / 10.0, rel=1e-5)
    assert enemy_tower.hp == before - 123
