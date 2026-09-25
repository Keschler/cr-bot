"""Simulator reset records and search eligibility predicates."""

from __future__ import annotations

from typing import Any
import numpy as np
from simulator.observation_v3 import (
    BELIEF_CARD_COUNT,
    EVENT_DIM,
    EVENT_HISTORY_LEN,
    HAND_TOKEN_FEATURES,
)
from simulator.roster import PLAYER_DECK as ROSTER_PLAYER_DECK
from simulator.rl.v4.distillation import DistillationConfig, family_for_index
from simulator.rl.foundation.opponent_pool import OpponentPool
from simulator.rl.v4.simulator_distillation import (
    FAMILY_TO_SOURCE,
    SOURCE_ARCHETYPE,
    build_hand_tokens,
)
from simulator.rl.v4.simulator_teacher import TeacherError, TeacherTarget, teacher_label
from simulator.rl.v4.timing_teacher import timing_teacher_label
from .config import SEARCH_GENERATOR_VERSION, OPPONENT_STRATEGY, _SPELL_COL, _stable_seed


def _hand_keys(record: dict[str, Any]) -> list[str]:
    if "hand_keys" in record:
        return list(record["hand_keys"])
    return list(record["provenance"]["target_hand"])


def _enemy_count(record: dict[str, Any]) -> int:
    mask = np.asarray(record["entity_mask"])
    tokens = np.asarray(record["entity_tokens"])
    return int(((tokens[:, 1] > 0.5) & mask).sum())


def _spell_legal(record: dict[str, Any]) -> bool:
    hand = np.asarray(record["hand_tokens"])
    legal = np.asarray(record["legal_play"]).reshape(4, -1).any(axis=1)
    return bool(((hand[:, _SPELL_COL] > 0.5) & legal).any())


def _base_label(record: dict[str, Any]) -> TeacherTarget:
    return teacher_label(
        hand_tokens=np.asarray(record["hand_tokens"]),
        entity_tokens=np.asarray(record["entity_tokens"]),
        entity_mask=np.asarray(record["entity_mask"]),
        legal_play=np.asarray(record["legal_play"]),
        legal_wait=bool(record["legal_wait"]),
        own_elixir=float(record["own_elixir"]),
    )


def _new_sim_env(context_ruleset: Any, pool: OpponentPool, seed: int, index: int, distill_seed: int,
                 family_mix: dict[str, float] | None = None):
    """Deterministic sim-natural reset (replayable by index).

    Uses the same executable short-state scenario wrapper as Stage-1
    sim-natural distillation (setup cards + prelude), so searched sim
    states carry real threats, spell clusters, and elixir variation
    instead of empty-arena openings.  Bare resets have no enemies, which
    starves spell opportunities and makes WAIT trivially optimal.
    """

    try:
        from ....engine import BattleEngine
        from ....env import SimulatorEnv
        from ...foundation.basic_scenarios import BasicMechanicsScenarioEnv, BasicScenarioConfig
    except ImportError:  # pragma: no cover
        from simulator.engine import BattleEngine
        from simulator.env import SimulatorEnv
        from simulator.rl.foundation.basic_scenarios import BasicMechanicsScenarioEnv, BasicScenarioConfig
    distill_config = DistillationConfig(n_states=max(1, index + 1), seed=distill_seed,
                                          family_mix=dict(family_mix) if family_mix else None)
    family = family_for_index(distill_config, index)
    source = FAMILY_TO_SOURCE[family]
    opponent = pool.sample(index, archetype=SOURCE_ARCHETYPE[source], strategy=OPPONENT_STRATEGY)
    base = SimulatorEnv(
        engine=BattleEngine(context_ruleset, validate_every_tick=False),
        decision_interval_us=250_000,
    )
    env = BasicMechanicsScenarioEnv(
        base,
        BasicScenarioConfig(source=source, target_player=0, decision_limit=64),
    )
    env.reset_v2(
        seed=_stable_seed(seed, "search-sim", family, index),
        decks=(tuple(ROSTER_PLAYER_DECK), tuple(opponent.deck.cards)),
        shuffle_decks=True,
    )
    return env, family, source


def _record_from_env(
    env: Any,
    family: str,
    source: str,
    context_ruleset: Any,
    *,
    allow_partial_hand: bool = False,
    public_memory: Any = None,
) -> dict[str, Any] | None:
    """Observe + label a live reset state (shared by sim build and replay)."""

    state = env.state
    if state is None or state.terminal:
        return None
    try:
        observation = env.observe_v2_for_viewer(0)
    except (ValueError, TypeError):
        return None
    player_state = state.players[0]
    hand = list(player_state.hand)
    if len(hand) != 4 and not allow_partial_hand:
        return None
    elixir = float(player_state.elixir_milli) / 1000.0
    try:
        if len(hand) == 4:
            hand_tokens = build_hand_tokens(hand, elixir)
        else:
            # Match the active continuation encoder's zero rows for cards
            # waiting on a refill.  The legal mask still protects those
            # missing slots from accidental actions.
            hand_tokens = np.zeros((4, HAND_TOKEN_FEATURES.__len__()), dtype=np.float32)
            for slot, card in enumerate(hand[:4]):
                row = np.asarray(build_hand_tokens([card] * 4, elixir)[0], dtype=np.float32)
                row = np.array(row, copy=True)
                row[2] = np.float32(slot / 3.0)
                hand_tokens[slot] = row
    except TeacherError:
        return None
    legal_play = np.array(observation.legal_play, dtype=bool)
    tick_seconds = float(state.tick) * float(context_ruleset.tick_us) / 1_000_000.0
    try:
        timing = timing_teacher_label(
            hand_tokens=hand_tokens,
            entity_tokens=np.asarray(observation.entity_tokens),
            entity_mask=np.asarray(observation.entity_mask),
            legal_play=legal_play,
            legal_wait=bool(observation.legal_wait),
            own_elixir=elixir,
            tick_seconds=tick_seconds,
        )
    except TeacherError:
        return None
    legal_cards = int(legal_play.reshape(4, -1).any(axis=1).sum())
    foe_cards = sorted(
        str(getattr(entity, "card_id", ""))
        for entity in state.entities.values()
        if entity.alive and entity.owner == 1 and entity.kind != "tower"
    )
    return {
        "family": family,
        "raster": np.array(observation.board, dtype=np.float32),
        "global_features": np.array(observation.global_vector, dtype=np.float32),
        "entity_tokens": np.array(observation.entity_tokens, dtype=np.float32),
        "entity_mask": np.array(observation.entity_mask, dtype=bool),
        "hand_tokens": hand_tokens,
        "legal_play": legal_play,
        "legal_wait": bool(observation.legal_wait),
        "own_elixir": float(elixir),
        "target": timing,
        "mode": int(timing.mode),
        "hand_keys": hand,
        "seq_id": "",
        "offset": 0,
        "legal_cards": legal_cards,
        "play_legal": bool(legal_cards > 0),
        "event_history": np.zeros((EVENT_HISTORY_LEN, EVENT_DIM), dtype=np.float32),
        "opp_hand_probs": np.full((BELIEF_CARD_COUNT,), 1.0 / BELIEF_CARD_COUNT, dtype=np.float32),
        "opp_out_of_cycle": np.zeros((BELIEF_CARD_COUNT,), dtype=bool),
        "opp_elixir_interval": np.asarray([0.0, 10.0], dtype=np.float32),
        "provenance": {
            "generator": SEARCH_GENERATOR_VERSION,
            "timing_family": family,
            "state_hash": state.state_hash(),
            "source": source,
            "target_elixir_milli": int(player_state.elixir_milli),
            "target_hand": list(hand),
            "legal_cards": legal_cards,
            "tick_seconds": round(tick_seconds, 3),
            "foe_cards": list(foe_cards),
        },
    }


def _disagrees(record: dict[str, Any]) -> bool:
    pred, target = record["champ"], record["target"]
    if pred["mode"] != target.mode:
        return True
    return bool(pred["mode"] == 1 and pred["card"] != target.card_slot)


def _spell_opp(record: dict[str, Any]) -> bool:
    return _spell_legal(record) and _enemy_count(record) >= 1


def _ambiguous(record: dict[str, Any]) -> bool:
    return record["target"].mode == 0 and bool(record["play_legal"])

