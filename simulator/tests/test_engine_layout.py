"""Structural contracts for the composable battle-engine package."""

from simulator.engine import BattleEngine, DeterministicCycleController, ENGINE_VERSION
from simulator.engine.abilities import AbilitiesMixin
from simulator.engine.collision import CollisionMixin
from simulator.engine.combat import CombatMixin
from simulator.engine.deaths import DeathsMixin
from simulator.engine.deployment import DeploymentMixin
from simulator.engine.match import MatchMixin
from simulator.engine.movement import MovementMixin
from simulator.engine.projectiles import ProjectilesMixin
from simulator.engine.scheduler import SchedulerMixin
from simulator.engine.spawning import SpawningMixin
from simulator.engine.status import StatusMixin
from simulator.engine.targeting import TargetingMixin


def test_battle_engine_composes_the_focused_mechanics_modules() -> None:
    expected = (
        SchedulerMixin,
        DeploymentMixin,
        TargetingMixin,
        MovementMixin,
        CollisionMixin,
        CombatMixin,
        ProjectilesMixin,
        StatusMixin,
        AbilitiesMixin,
        SpawningMixin,
        DeathsMixin,
        MatchMixin,
    )

    assert BattleEngine.__mro__[1 : 1 + len(expected)] == expected


def test_public_engine_api_and_controller_remain_available() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=17, shuffle_decks=False)

    assert state.engine_version == ENGINE_VERSION
    assert engine.decision_interval_ticks > 0
    assert isinstance(DeterministicCycleController(), DeterministicCycleController)
