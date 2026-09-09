"""Simulator-grounded distillation states (paper Stage 1, real-simulator revision).

Stage 1 asks whether V4 can learn basic tactical decisions from realistic
simulator observations.  The synthetic generator in :mod:`rl.distillation`
hand-makes every feature (random entity splats, uniform beliefs, zero events);
this module instead runs :class:`BasicMechanicsScenarioEnv`, captures the same
public observation the policy would receive during training or live inference
(raster, visible entities, hand, elixir, tower state, legal actions), and
labels it with the same rules teacher.

What is real here and what is not:

* real: board raster, global vector, entity tokens/mask, hand order, own
  elixir, tower HP variation, legal masks, prelude dynamics, threat identity
  and placement.  Every tensor comes from the simulator public boundary.
* still default: opponent belief and event history come from a fresh
  :class:`PublicStateEstimator` (uniform hand belief, ``[0, 10]`` elixir
  interval, empty history), exactly like the synthetic defaults.  No
  privileged state (opponent hand, exact opponent elixir, hidden cooldowns)
  enters the sample.
* fixed scope: the learner deck stays the 2.6 Hog path (paper section 8.2);
  only the opponent deck varies by scenario archetype.

Each sample is deterministic and order-independent: ``(seed, family, index,
attempt)`` derives the scenario source, opponent deck, and environment seed,
and a fresh environment is built per sample so no cross-sample RNG state can
leak in.  Consumers only see ``DistillationSample`` + ``to_torch_batch``,
unchanged from the synthetic schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

try:
    from ..public_state_estimator import PublicStateEstimator
    from ..roster import PLAYER_DECK as ROSTER_PLAYER_DECK
    from ..ruleset import load_fixed_ruleset
    from .basic_scenarios import BasicMechanicsScenarioEnv, BasicScenarioConfig
    from .distillation import (
        PLAYER_DECK as FIXED_DECK_TABLE,
        TEACHER_VERSION,
        DistillationConfig,
        DistillationSample,
        family_for_index,
    )
    from .opponent_pool import OpponentPool
    from .simulator_teacher import TeacherError, soft_placement_target, teacher_label
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.public_state_estimator import PublicStateEstimator
    from simulator.roster import PLAYER_DECK as ROSTER_PLAYER_DECK
    from simulator.ruleset import load_fixed_ruleset
    from simulator.rl.basic_scenarios import BasicMechanicsScenarioEnv, BasicScenarioConfig
    from simulator.rl.distillation import (
        PLAYER_DECK as FIXED_DECK_TABLE,
        TEACHER_VERSION,
        DistillationConfig,
        DistillationSample,
        family_for_index,
    )
    from simulator.rl.opponent_pool import OpponentPool
    from simulator.rl.simulator_teacher import TeacherError, soft_placement_target, teacher_label


SIM_GENERATOR_VERSION: str = "sim-v4-distill-0"
"""Provenance identity for real-simulator distillation states."""

OPPONENT_STRATEGY: str = "deterministic-cycle"

# The eight reporting families are broader than the five executable short-state
# sources, so some sources serve two families.  The mapping is recorded in
# every sample's provenance; reporting stays comparable with the synthetic
# baseline while the underlying states are real simulator observations.
FAMILY_TO_SOURCE: dict[str, str] = {
    "offense": "isolated-offense",
    "ground-defense": "ground-defense",
    "air-defense": "air-defense",
    "spell-value": "spell-situations",
    "kiting-cycle": "kiting-cycling-elixir",
    "low-elixir": "kiting-cycling-elixir",
    "counterpush": "isolated-offense",
    "bridge-defense": "ground-defense",
}

# Opponent archetypes per scenario source, mirroring the basic-scenario test
# contract so air-defense states contain actual air threats and spell states
# contain clusterable bodies.
SOURCE_ARCHETYPE: dict[str, str] = {
    "isolated-offense": "aggressive-pressure",
    "ground-defense": "beatdown",
    "air-defense": "air-beatdown",
    "spell-situations": "siege-bait",
    "kiting-cycling-elixir": "defensive-cycle",
}

_MAX_TEACHER_ATTEMPTS: int = 8

# The kiting-cycling-elixir source randomizes target elixir in 3-7k, so the
# mapped ``low-elixir`` family could never observe broke states without help.
# Those states carry almost all WAIT supervision (unaffordable hand), which
# the mode/duration heads need.  The override below restores the synthetic
# low-elixir range (0-3) deterministically; it touches only the public elixir
# scalar, is recorded in provenance, and leaves board/entities/legal intact.
LOW_ELIXIR_OVERRIDE_MILLI: tuple[int, int] = (0, 3_000)


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass
class _SimContext:
    """Shared read-only simulator state for one dataset generation run."""

    ruleset: Any
    pool: OpponentPool
    target_player: int

    @classmethod
    def build(cls, seed: int, target_player: int) -> "_SimContext":
        if type(target_player) is not int or target_player not in (0, 1):
            raise ValueError("target_player must be 0 or 1")
        ruleset = load_fixed_ruleset()
        return cls(ruleset=ruleset, pool=OpponentPool(ruleset, seed=seed), target_player=target_player)


def build_hand_tokens(hand: tuple[str, ...] | list[str], elixir: float) -> np.ndarray:
    """Encode the simulator hand in the ``HAND_TOKEN_FEATURES`` layout.

    Column semantics (identity, cost, slot, affordability, class flags, cycle
    position) match :func:`rl.distillation` sample generation exactly: class
    flags follow the fixed-deck table there rather than live metadata
    derivation, so teacher behavior stays comparable across generators.
    """

    by_key = {row[0]: row for row in FIXED_DECK_TABLE}
    canonical_order = {card: index for index, card in enumerate(ROSTER_PLAYER_DECK)}
    tokens = np.zeros((4, 16), dtype=np.float32)
    for slot in range(4):
        card = hand[slot]
        try:
            _, policy_id, cost, win, spell, building, swarm, air = by_key[card]
        except KeyError as error:
            raise TeacherError(f"hand card {card!r} is outside the fixed player deck") from error
        affordable = cost <= elixir
        tokens[slot, 0] = policy_id / 127.0
        tokens[slot, 1] = cost / 10.0
        tokens[slot, 2] = slot / 3.0
        tokens[slot, 3] = 1.0 if affordable else 0.0
        tokens[slot, 4] = 1.0 if affordable else 0.0
        tokens[slot, 5] = 1.0 if win else 0.0
        tokens[slot, 6] = 1.0 if spell else 0.0
        tokens[slot, 7] = 1.0 if building else 0.0
        tokens[slot, 8] = 1.0 if swarm else 0.0
        tokens[slot, 9] = 1.0 if air else 0.0
        tokens[slot, 10] = float(canonical_order[card]) / 7.0
    return tokens


def _make_scenario_env(context: _SimContext, source: str) -> BasicMechanicsScenarioEnv:
    try:
        from ..engine import BattleEngine
        from ..env import SimulatorEnv
    except ImportError:  # pragma: no cover - top-level ``rl`` imports
        from simulator.engine import BattleEngine
        from simulator.env import SimulatorEnv
    base = SimulatorEnv(
        engine=BattleEngine(context.ruleset, validate_every_tick=False),
        decision_interval_us=250_000,
    )
    return BasicMechanicsScenarioEnv(
        base,
        BasicScenarioConfig(
            source=source,
            target_player=context.target_player,
            decision_limit=64,
        ),
    )


def _sample_with_context(
    config: DistillationConfig,
    index: int,
    context: _SimContext,
) -> DistillationSample:
    """Generate one deterministic real-simulator state (order-independent)."""

    family = family_for_index(config, index)
    try:
        source = FAMILY_TO_SOURCE[family]
    except KeyError as error:  # pragma: no cover - mix validation covers families
        raise TeacherError(f"no scenario source mapped for family {family!r}") from error
    archetype = SOURCE_ARCHETYPE[source]
    last_error: Exception | None = None
    for attempt in range(_MAX_TEACHER_ATTEMPTS):
        env_seed = _stable_seed(config.seed, "sim-state", family, index, attempt)
        opponent = context.pool.sample(
            index + attempt * 1_000_003,
            archetype=archetype,
            strategy=OPPONENT_STRATEGY,
        )
        environment = _make_scenario_env(context, source)
        try:
            environment.reset_v2(
                seed=env_seed,
                decks=(tuple(ROSTER_PLAYER_DECK), opponent.deck.cards),
                shuffle_decks=True,
            )
        except Exception as error:
            last_error = error
            continue
        state = environment.state
        if state is None:  # pragma: no cover - reset invariant
            last_error = TeacherError("scenario reset produced no state")
            continue
        player_state = state.players[context.target_player]
        hand = list(player_state.hand)
        if len(hand) != 4:
            last_error = TeacherError(f"simulator hand has {len(hand)} cards, expected 4")
            continue
        elixir_overridden = False
        if family == "low-elixir":
            try:
                from ..fixed import DeterministicRng
            except ImportError:  # pragma: no cover - top-level ``rl`` imports
                from simulator.fixed import DeterministicRng
            elixir_rng = DeterministicRng(
                _stable_seed(config.seed, "sim-elixir", family, index, attempt)
            )
            low, high = LOW_ELIXIR_OVERRIDE_MILLI
            player_state.elixir_milli = low + elixir_rng.randbelow(high - low + 1)
            player_state.elixir_remainder = 0
            elixir_overridden = True
        # Capture the public observation after any elixir override so the
        # global vector and legality masks reflect the labeled state.
        observation = environment.observe_v2_for_viewer(context.target_player)
        elixir = float(player_state.elixir_milli) / 1000.0
        try:
            hand_tokens = build_hand_tokens(hand, elixir)
            legal_play = np.array(observation.legal_play, dtype=bool)
            target = teacher_label(
                hand_tokens=hand_tokens,
                entity_tokens=np.asarray(observation.entity_tokens),
                entity_mask=np.asarray(observation.entity_mask),
                legal_play=legal_play,
                legal_wait=bool(observation.legal_wait),
                own_elixir=elixir,
            )
        except TeacherError as error:
            last_error = error
            continue
        if target.mode == 1:
            soft = soft_placement_target(
                target.row, target.col, np.asarray(legal_play[target.card_slot])
            )
        else:
            soft = np.zeros((config.rows, config.cols), dtype=np.float32)
        estimator = PublicStateEstimator()
        snapshot = estimator.snapshot()
        audit_latest = environment.scenario_audit()["latest"] or {}
        return DistillationSample(
            family=family,
            raster=np.array(observation.board, dtype=np.float32),
            global_features=np.array(observation.global_vector, dtype=np.float32),
            entity_tokens=np.array(observation.entity_tokens, dtype=np.float32),
            entity_mask=np.array(observation.entity_mask, dtype=bool),
            hand_tokens=hand_tokens,
            opp_hand_probs=np.array(snapshot.opp_hand_probs, dtype=np.float32),
            opp_out_of_cycle=np.array(snapshot.opp_out_of_cycle, dtype=bool),
            opp_elixir_interval=np.asarray(
                [snapshot.opp_elixir_lo, snapshot.opp_elixir_hi], dtype=np.float32
            ),
            event_history=np.array(snapshot.event_history, dtype=np.float32),
            legal_play=legal_play,
            legal_wait=bool(observation.legal_wait),
            own_elixir=float(elixir),
            target=target,
            soft_placement=soft,
            provenance={
                "generator": SIM_GENERATOR_VERSION,
                "teacher": TEACHER_VERSION,
                "seed": config.seed,
                "family": family,
                "index": index,
                "attempt": attempt,
                "source": source,
                "archetype": archetype,
                "opponent_strategy": OPPONENT_STRATEGY,
                "opponent_deck_id": opponent.deck.deck_id,
                "opponent_deck": list(opponent.deck.cards),
                "target_player": context.target_player,
                "env_seed": env_seed,
                "state_hash": state.state_hash(),
                "target_hand": hand,
                "target_elixir_milli": int(player_state.elixir_milli),
                "elixir_override_milli": list(LOW_ELIXIR_OVERRIDE_MILLI)
                if elixir_overridden
                else None,
                "setup_cards": list(audit_latest.get("setup_cards", [])),
                "threat_uids": list(audit_latest.get("threat_uids", [])),
            },
        )
    raise TeacherError(
        f"no labelable simulator state for index {index} after {_MAX_TEACHER_ATTEMPTS} attempts"
    ) from last_error


def generate_sim_sample(
    config: DistillationConfig,
    index: int,
    *,
    target_player: int = 0,
) -> DistillationSample:
    """Generate one deterministic real-simulator state (order-independent)."""

    return _sample_with_context(config, index, _SimContext.build(config.seed, target_player))


def generate_sim_dataset(
    config: DistillationConfig,
    *,
    target_player: int = 0,
) -> list[DistillationSample]:
    """Generate ``n_states`` deterministic real-simulator states."""

    context = _SimContext.build(config.seed, target_player)
    return [_sample_with_context(config, index, context) for index in range(config.n_states)]


__all__ = [
    "FAMILY_TO_SOURCE",
    "OPPONENT_STRATEGY",
    "SIM_GENERATOR_VERSION",
    "SOURCE_ARCHETYPE",
    "build_hand_tokens",
    "generate_sim_dataset",
    "generate_sim_sample",
]
