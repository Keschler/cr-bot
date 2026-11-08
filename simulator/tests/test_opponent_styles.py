from __future__ import annotations

import copy
import sys

import pytest

from simulator.actions import PlayCardAction, WaitAction, action_to_dict
from simulator.engine import BattleEngine
from simulator.rl.foundation.opponent_pool import (
    OpponentPool,
    OpponentPoolError,
    make_opponent_controller,
)
from simulator.rl.foundation.opponent_styles import (
    DEV_STYLE_IDS,
    OPPONENT_STYLE_VERSION,
    STYLE_ARCHETYPES,
    TRAIN_STYLE_IDS,
    make_styled_opponent,
    normalize_style_strategy,
    resolve_style_config,
    style_identity,
    style_ids,
)
from simulator.rl.v4.frontier_eval import make_opponent
from simulator.ruleset import load_fixed_ruleset


_PRIMARY = {
    "aggressive-pressure": "hog-rider",
    "defensive-cycle": "x-bow",
    "beatdown": "golem",
    "air-beatdown": "lava-hound",
    "siege-bait": "goblin-barrel",
}


def _battle(archetype: str, *, player: int = 0, elixir: int | None = None):
    engine = BattleEngine(load_fixed_ruleset(), validate_every_tick=True)
    deck = OpponentPool(engine.ruleset, seed=91).sample_deck(0, archetype=archetype).cards
    state = engine.new_battle((deck, deck), seed=81, shuffle_decks=False)
    if elixir is not None:
        state.players[player].hand = [_PRIMARY[archetype], "skeletons"]
        state.players[player].elixir_milli = elixir
    return engine, state


@pytest.mark.parametrize("archetype", STYLE_ARCHETYPES)
@pytest.mark.parametrize("player", (0, 1))
def test_reserve_changes_same_state_commitment_without_abandoning_primary(archetype, player):
    cost = load_fixed_ruleset().card(_PRIMARY[archetype]).elixir_milli
    engine, state = _battle(archetype, player=player, elixir=cost)
    tempo = make_styled_opponent(archetype, "tempo", seed=7)
    patient = make_styled_opponent(archetype, "patient", seed=7)

    assert isinstance(tempo.choose_action(engine, state, player), PlayCardAction)
    assert patient.choose_action(engine, state, player) == WaitAction(player)
    state.players[player].elixir_milli = 10_000
    action = patient.choose_action(engine, state, player)
    assert isinstance(action, PlayCardAction)
    assert state.players[player].hand[action.card_slot] == _PRIMARY[archetype]
    assert engine.validate_action(state, action) is None


@pytest.mark.parametrize("player", (0, 1))
def test_visible_expensive_commitment_releases_reserve_and_changes_pressure_lane(player):
    engine, state = _battle("aggressive-pressure", player=player, elixir=4_000)
    patient = make_styled_opponent("pressure", "patient", seed=7)
    split = make_styled_opponent("aggressive", "split-pressure", seed=7)
    assert patient.choose_action(engine, state, player) == WaitAction(player)
    engine._spawn_single_at(
        state, engine.ruleset.card("golem"), owner=1 - player,
        x_mtile=3_500, y_mtile=8_000 if player == 0 else 23_000,
    )
    # Make the left tower the attractive damage target for the patient style.
    enemy_towers = [e for e in state.entities.values() if e.kind == "tower" and e.owner != player]
    for tower in enemy_towers:
        if tower.role != "king" and tower.x_mtile < 9_000:
            tower.hp //= 2
    weak_lane_action = patient.choose_action(engine, state, player)
    split_action = split.choose_action(engine, state, player)
    assert isinstance(weak_lane_action, PlayCardAction)
    assert isinstance(split_action, PlayCardAction)
    assert weak_lane_action.cell[0] == 3
    assert split_action.cell[0] == 14
    assert engine.validate_action(state, split_action) is None


@pytest.mark.parametrize("player", (0, 1))
def test_surviving_support_releases_reserve_and_sets_counterpush_lane(player):
    engine, state = _battle("aggressive-pressure", player=player, elixir=4_000)
    controller = make_styled_opponent("aggressive", "counterpush", seed=0)
    assert controller.choose_action(engine, state, player) == WaitAction(player)
    survivor = engine._spawn_single_at(
        state, engine.ruleset.card("musketeer"), owner=player,
        x_mtile=14_500, y_mtile=20_000 if player == 0 else 11_000,
    )
    survivor.deploy_remaining_us = 0
    survivor.hp = survivor.max_hp * 2 // 3
    action = controller.choose_action(engine, state, player)
    assert isinstance(action, PlayCardAction)
    assert action.cell[0] == 14


@pytest.mark.parametrize("player", (0, 1))
def test_patient_tank_formation_starts_deeper_and_seed_varies_legal_formations(player):
    engine, state = _battle("beatdown", player=player, elixir=10_000)
    tempo = make_styled_opponent("beatdown", "tempo", seed=0).choose_action(engine, state, player)
    patient = make_styled_opponent("beatdown", "patient", seed=0).choose_action(engine, state, player)
    assert isinstance(tempo, PlayCardAction) and isinstance(patient, PlayCardAction)
    assert patient.cell[1] > tempo.cell[1] if player == 0 else patient.cell[1] < tempo.cell[1]
    variants = []
    for seed in range(16):
        action = make_styled_opponent("beatdown", "counterpush", seed=seed).choose_action(engine, state, player)
        assert engine.validate_action(state, action) is None
        variants.append(action.cell)
    assert len(set(variants)) > 1


@pytest.mark.parametrize("archetype", STYLE_ARCHETYPES)
@pytest.mark.parametrize("style", (*TRAIN_STYLE_IDS, *DEV_STYLE_IDS))
def test_bounded_games_have_legal_actions_and_active_win_conditions(archetype, style):
    engine, state = _battle(archetype)
    controllers = [make_styled_opponent(archetype, style, seed=51) for _ in (0, 1)]
    for own in state.players:
        own.elixir_milli = 10_000
    attacks = [0, 0]
    primary_cards = {
        "hog-rider", "battle-ram", "ram-rider", "x-bow", "mortar", "goblin-drill",
        "goblin-barrel", "golem", "giant", "electro-giant", "lava-hound", "balloon",
    }
    for _ in range(100):  # 25 seconds of real decision/physics cadence.
        if state.terminal:
            break
        actions = [controller.choose_action(engine, state, player) for player, controller in enumerate(controllers)]
        for player, action in enumerate(actions):
            assert engine.validate_action(state, action) is None
            if isinstance(action, PlayCardAction):
                attacks[player] += state.players[player].hand[action.card_slot] in primary_cards
        engine.step(state, actions)
        for _ in range(engine.decision_interval_ticks - 1):
            engine.step(state)
    assert min(attacks) >= 1
    assert not any(event.kind == "action_rejected" for event in state.events)


@pytest.mark.parametrize("style", (*TRAIN_STYLE_IDS, *DEV_STYLE_IDS))
def test_styled_games_reproduce_without_rng_history(style):
    def play():
        engine, state = _battle("defensive-cycle")
        controllers = [make_styled_opponent("defensive-cycle", style, seed=51) for _ in (0, 1)]
        transcript = []
        for _ in range(20):
            actions = [controller.choose_action(engine, state, player) for player, controller in enumerate(controllers)]
            assert actions[0] == controllers[0].choose_action(engine, state, 0)
            transcript.append([action_to_dict(action) for action in actions])
            engine.step(state, actions)
            for _ in range(engine.decision_interval_ticks - 1):
                engine.step(state)
        return transcript, state.state_hash()

    assert play() == play()


@pytest.mark.parametrize("archetype", STYLE_ARCHETYPES)
def test_crossed_threat_defense_and_no_style_factory_remain_historical(archetype):
    engine, state = _battle(archetype)
    engine._spawn_single_at(
        state, engine.ruleset.card("giant"), owner=1, x_mtile=4_500, y_mtile=20_000,
    )
    baseline = make_opponent_controller(archetype, seed=13)
    expected = baseline.choose_action(engine, state, 0)
    assert make_opponent({"strategy": archetype}, 13).choose_action(engine, state, 0) == expected
    for style in (*TRAIN_STYLE_IDS, *DEV_STYLE_IDS):
        assert make_styled_opponent(archetype, style, seed=13).choose_action(engine, state, 0) == expected


def test_style_identity_pins_parameters_version_seed_and_disjoint_pools():
    assert set(style_ids()) == set(TRAIN_STYLE_IDS)
    assert set(style_ids("dev")) == set(DEV_STYLE_IDS)
    assert set(TRAIN_STYLE_IDS).isdisjoint(DEV_STYLE_IDS)
    identities = [style_identity(arch, style, seed=10) for arch in STYLE_ARCHETYPES for style in (*TRAIN_STYLE_IDS, *DEV_STYLE_IDS)]
    assert len({identity["sha256"] for identity in identities}) == len(identities)
    identity = style_identity("tank_support", "patient", seed=10)
    assert identity["version"] == OPPONENT_STYLE_VERSION
    assert identity == style_identity("beatdown", "patient", seed=10)
    assert identity != style_identity("beatdown", "patient", seed=11)
    assert identity != style_identity("beatdown", "patient", seed=10, config={"reserve_milli": 1_000})
    config = resolve_style_config("patient")
    assert resolve_style_config(None, config.as_dict()) == config


@pytest.mark.parametrize("alias,canonical", (
    ("AGGRESSIVE", "aggressive-pressure"), ("Pressure", "aggressive-pressure"),
    ("Defensive", "defensive-cycle"), ("tank support", "beatdown"),
    ("beatdown_tank_support", "beatdown"), ("air beatdown", "air-beatdown"),
    ("Siege", "siege-bait"), ("Bait", "siege-bait"),
))
def test_historical_aliases_normalize(alias, canonical):
    assert normalize_style_strategy(alias) == canonical


@pytest.mark.parametrize("config", (
    {"reserve_milli": -1}, {"reserve_milli": True}, {"reserve_milli": 3_001},
    {"cycle_floor_milli": 0}, {"formation_depth_rows": 3},
    {"punish_commitment_milli": 1}, {"lane_policy": "random"},
    {"version": "future"}, {"typo": 1}, {"style_id": "tempo"},
))
def test_invalid_style_configuration_fails_closed(config):
    with pytest.raises(OpponentPoolError):
        resolve_style_config("patient", config)


def test_unknown_style_strategy_split_and_bad_seed_fail_closed():
    for strategy in ("random", "cycle", "typo"):
        with pytest.raises(OpponentPoolError):
            make_styled_opponent(strategy, "tempo")
    with pytest.raises(OpponentPoolError):
        resolve_style_config("typo")
    with pytest.raises(OpponentPoolError):
        style_ids("test")
    with pytest.raises(OpponentPoolError):
        style_identity("beatdown", "tempo", seed=True)


def test_factory_uses_pinned_controller_seed_and_verifies_identity():
    identity = style_identity("beatdown", "patient", seed=99)
    spec = {"strategy": "tank-support", "style": "patient", "controller_seed": 99, "style_identity": identity}
    assert make_opponent(spec, 22).identity == identity
    config_spec = {"strategy": "beatdown", "style_config": resolve_style_config("patient").as_dict(), "controller_seed": 99}
    assert make_opponent(config_spec, 22).identity == identity
    changed = copy.deepcopy(spec)
    changed["style_identity"]["parameters"]["reserve_milli"] = 0
    with pytest.raises(ValueError, match="identity"):
        make_opponent(changed, 22)
    with pytest.raises(ValueError, match="requires"):
        make_opponent({"strategy": "beatdown", "style_identity": identity}, 22)
    with pytest.raises(ValueError, match="cannot also"):
        make_opponent({"kind": "v4-checkpoint", "style": "tempo"}, 22)


def test_style_does_not_inspect_hidden_enemy_hand_or_elixir():
    engine, state = _battle("aggressive-pressure", elixir=6_000)
    controller = make_styled_opponent("aggressive", "patient", seed=81)
    expected = controller.choose_action(engine, state, 0)
    state.players[1].elixir_milli = 0
    state.players[1].hand.reverse()
    assert controller.choose_action(engine, state, 0) == expected


def test_style_module_has_no_torch_import_dependency():
    # Avoid a subprocess and its environment variability: inspect the module's
    # actual imported bindings, allowing other tests to have imported Torch.
    module = sys.modules["simulator.rl.foundation.opponent_styles"]
    assert all(getattr(value, "__name__", "") != "torch" for value in vars(module).values())
