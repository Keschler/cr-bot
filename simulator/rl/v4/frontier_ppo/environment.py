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


def entry_input_tensors(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    """Add batch/time dimensions, preserving boolean masks and float features."""
    import numpy as np
    import torch

    boolean_keys = {"entity_mask", "opp_out_of_cycle"}
    return {
        key: torch.as_tensor(
            np.asarray(value, dtype=bool if key in boolean_keys else np.float32),
            device=device,
        ).unsqueeze(0).unsqueeze(0)
        for key, value in inputs.items()
        if key not in ("legal_play", "legal_wait")
    }
