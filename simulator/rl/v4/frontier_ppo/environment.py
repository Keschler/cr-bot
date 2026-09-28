"""Sparse-reward simulator construction and pinned evaluation opponents."""

from __future__ import annotations

from typing import Any, Sequence



def make_frontier_env() -> Any:
    from simulator.engine.core import BattleEngine
    from simulator.env import RewardConfig, SimulatorEnv
    from simulator.ruleset import load_fixed_ruleset

    return SimulatorEnv(
        BattleEngine(load_fixed_ruleset(), validate_every_tick=False),
        reward=RewardConfig.terminal_outcome(),
    )


def pinned_opponent_decks(
    *, pool_seed: int = 0, episode_index: int = 500
) -> list[dict[str, Any]]:
    """Pinned-deck specs for the 6 discovery archetypes (eval pairing)."""

    from simulator.ruleset import load_fixed_ruleset

    from ...foundation.opponent_pool import OpponentPool

    from simulator.rl.v4.frontier import DISCOVERY_ARCHETYPES

    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=pool_seed)
    specs = []
    for arch in DISCOVERY_ARCHETYPES:
        sampled = pool.sample(episode_index, archetype=arch, allow_variants=False)
        specs.append(
            {
                "archetype": arch,
                "strategy": sampled.strategy,
                "deck": list(sampled.deck.cards),
                "controller_seed": int(sampled.controller_seed),
            }
        )
    return specs


def heldout_opponent_decks(
    *, pool_seed: int = 20261117, episode_index: int = 700
) -> list[dict[str, Any]]:
    """Evaluation-only variants and random legal decks from a separate seed.

    The deterministic-cycle deck cannot vary, so it is excluded. This pool
    tests deck transfer and one independent controller style; it is not a
    substitute for independent learned opponents.
    """

    from simulator.ruleset import load_fixed_ruleset
    from ...foundation.opponent_pool import OpponentPool
    from simulator.rl.v4.frontier import DISCOVERY_ARCHETYPES

    pool = OpponentPool(load_fixed_ruleset(), seed=pool_seed)
    specs = []
    for offset, archetype in enumerate((*DISCOVERY_ARCHETYPES[1:], "random-legal")):
        scenario = pool.sample(
            episode_index + offset, archetype=archetype, allow_variants=True
        )
        specs.append({
            "archetype": archetype,
            "strategy": scenario.strategy,
            "deck": list(scenario.deck.cards),
            "controller_seed": int(scenario.controller_seed),
        })
    return specs


def real_eval_opponent_decks() -> list[dict[str, Any]]:
    """Complete held-out base-card decks from the pinned source catalog."""

    from ..frontier import _stable_seed
    from ..real_decks import REAL_DECK_ARCHETYPES, real_decks_for

    return [
        {
            "archetype": deck.archetype,
            "strategy": deck.archetype,
            "deck": list(deck.cards),
            "deck_id": deck.deck_id,
            "source_url": deck.source_url,
            "controller_seed": _stable_seed("real-eval", deck.deck_id) & 0xFFFFFFFF,
        }
        for archetype in REAL_DECK_ARCHETYPES
        for deck in real_decks_for("eval", archetype)
    ]


def entry_input_tensors(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    """Add batch/time dimensions, preserving boolean masks and float features."""
    from ..v4_ppo.runtime import batch_v4_inputs

    return batch_v4_inputs(inputs, device)
