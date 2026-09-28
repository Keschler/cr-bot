"""Match-opening training states for a reward-only V4 PPO baseline."""

from __future__ import annotations

from typing import Any, Sequence
import hashlib
from pathlib import Path

from .frontier import DISCOVERY_ARCHETYPES, PoolMatchConfig, _stable_seed, make_pool_match_start
from .real_decks import REAL_DECK_ARCHETYPES, catalog_sha256


RAW_PPO_STARTS_VERSION = "raw-ppo-starts-v3"
MIXED_ARCHETYPES = (*DISCOVERY_ARCHETYPES, "random-legal")


def make_match_start_entries(
    *, pool_seed: int, matches_per_archetype: int, profile: str = "pinned",
    policy_opponents: Sequence[str | Path] = (),
) -> list[dict[str, Any]]:
    """Create the same pinned match openings used by frontier collection.

    These entries have no search verdict, teacher target, or privileged actor
    feature. The PPO collector plays forward from decision zero and receives
    only simulator rewards.
    """

    if type(pool_seed) is not int or pool_seed < 0:
        raise ValueError("pool_seed must be a non-negative integer")
    if type(matches_per_archetype) is not int or matches_per_archetype < 1:
        raise ValueError("matches_per_archetype must be a positive integer")
    if profile not in ("pinned", "mixed", "real"):
        raise ValueError("profile must be pinned, mixed, or real")
    entries: list[dict[str, Any]] = []
    archetypes = (
        REAL_DECK_ARCHETYPES if profile == "real"
        else DISCOVERY_ARCHETYPES if profile == "pinned"
        else MIXED_ARCHETYPES
    )
    for archetype in archetypes:
        for match_index in range(matches_per_archetype):
            config = PoolMatchConfig(
                archetype=archetype,
                match_index=match_index,
                pool_seed=pool_seed,
                allow_variants=profile == "mixed",
                deck_profile="real" if profile == "real" else "scripted",
            )
            env, scenario, match_seed, learner_deck = make_pool_match_start(config)
            if env.state is None:
                raise RuntimeError("raw PPO match reset returned no state")
            entries.append({
                "entry_id": f"raw:{archetype}:{match_index}",
                "split": "train",
                "initial_state": env.save_state(),
                "learner_deck": list(learner_deck),
                "opponent_deck": list(scenario.deck.cards),
                "opponent": {
                    "archetype": archetype,
                    "strategy": scenario.strategy,
                    "controller_seed": int(scenario.controller_seed),
                },
                "provenance": {
                    "generator": RAW_PPO_STARTS_VERSION,
                    "profile": profile,
                    "real_deck_catalog_sha256": catalog_sha256() if profile == "real" else None,
                    "real_deck_id": scenario.deck.deck_id if profile == "real" else None,
                    "pool_seed": pool_seed,
                    "match_index": match_index,
                    "match_seed": match_seed,
                    "state_hash": env.state.state_hash(),
                },
            })
    if policy_opponents:
        from simulator.engine.core import BattleEngine
        from simulator.env import SimulatorEnv
        from simulator.roster import PLAYER_DECK
        from simulator.ruleset import load_fixed_ruleset

        ruleset = load_fixed_ruleset()
        for checkpoint in policy_opponents:
            path = Path(checkpoint).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"policy opponent checkpoint is not a file: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            archetype = f"v4-policy-{digest[:12]}"
            for match_index in range(matches_per_archetype):
                match_seed = _stable_seed("policy-opponent", pool_seed, digest, match_index)
                env = SimulatorEnv(BattleEngine(ruleset, validate_every_tick=False))
                env.reset_v2(
                    seed=match_seed, decks=(tuple(PLAYER_DECK), tuple(PLAYER_DECK)),
                    shuffle_decks=True,
                )
                if env.state is None:
                    raise RuntimeError("policy opponent match reset returned no state")
                entries.append({
                    "entry_id": f"raw:{archetype}:{match_index}",
                    "split": "train",
                    "initial_state": env.save_state(),
                    "learner_deck": list(PLAYER_DECK),
                    "opponent_deck": list(PLAYER_DECK),
                    "opponent": {
                        "kind": "v4-checkpoint", "archetype": archetype,
                        "checkpoint": str(path), "sha256": digest,
                    },
                    "provenance": {
                        "generator": RAW_PPO_STARTS_VERSION, "profile": profile,
                        "pool_seed": pool_seed, "match_index": match_index,
                        "match_seed": match_seed, "state_hash": env.state.state_hash(),
                    },
                })
    return entries
