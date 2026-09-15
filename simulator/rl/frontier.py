"""Data-driven frontier discovery for V4 (Stage 3): pool collection + cheap filter.

Pass 1 runs the frozen simx12 champion through long/full trajectories against
the diverse heuristic opponent pool and records compact per-decision records.
Pass 2 cheap-filters naturally visited states that may be frontier mistakes
using policy/rule disagreement, uncertainty, close action probabilities,
subsequent tower/elixir deterioration, and tactical transitions.  Uncertainty
is deliberately NOT required: confidently wrong decisions are in scope.

Expensive counterfactual evaluation lives in :mod:`frontier_eval`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

try:
    from .model_v4 import ModelConfigV4
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.rl.model_v4 import ModelConfigV4


FRONTIER_VERSION: str = "frontier-v4-0"

CHAMPION_CHECKPOINT: str = "outputs/v4/stage2_12k_v2_e12_s12.pt"

# Discovery opponent pool: pinned archetype decks (``allow_variants=False``).
# Sealed evaluation uses held-out variants + different seeds (see frontier_eval).
DISCOVERY_ARCHETYPES: tuple[str, ...] = (
    "deterministic-cycle",
    "aggressive-pressure",
    "defensive-cycle",
    "beatdown",
    "air-beatdown",
    "siege-bait",
)

# Full regulation + overtime at the 250ms decision cadence.
FULL_MATCH_DECISIONS: int = 1200

GRID_ROWS: int = 32
GRID_COLS: int = 18


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:16], 16)


def frontier_model_config() -> ModelConfigV4:
    """Champion-matching model config (mirrors the PPO smoke proof config)."""

    return ModelConfigV4(
        model_dim=32,
        spatial_channels=16,
        fused_dim=64,
        gru_hidden_dim=64,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ff_dim=64,
        spatial_head_dim=8,
    )


@dataclass(slots=True)
class PoolMatchConfig:
    """One discovery trajectory: opponent identity + budget."""

    archetype: str
    strategy: str = ""
    allow_variants: bool = False
    match_index: int = 0
    pool_seed: int = 0
    max_decisions: int = FULL_MATCH_DECISIONS
    learner_deck: str = "roster"  # "roster" -> simulator PLAYER_DECK

    def resolved_strategy(self) -> str:
        return self.strategy or self.archetype


@dataclass(slots=True)
class FilterConfig:
    """Cheap-filter thresholds (all documented; pilot-tunable)."""

    det_lookahead: int = 32
    det_threshold: float = 0.02
    transition_window: int = 4
    max_per_match: int = 40
    min_spacing: int = 8
    global_top_k: int = 2000
    per_signature_cap: int = 25
    # Opening decisions carry no game information (full elixir, empty arena);
    # disagreements there are unmeasurable style, not tactical mistakes.
    min_t: int = 16


# ---------------------------------------------------------------------------
# Authoritative state summaries (towers / threats / tags).
# ---------------------------------------------------------------------------


def tower_fracs(state: Any) -> dict[str, list[float]]:
    """Per-tower HP fractions by side, sorted king-last for stability."""

    own: list[float] = []
    enemy: list[float] = []
    for entity in state.entities.values():
        if entity.kind != "tower" or not entity.alive:
            frac = 0.0
        elif entity.kind != "tower":
            continue
        else:
            frac = max(0.0, min(1.0, float(entity.hp) / max(1, int(entity.max_hp))))
        if entity.kind != "tower":
            continue
        (own if entity.owner == 0 else enemy).append(frac)
    return {"own": sorted(own), "enemy": sorted(enemy)}


def threat_summary(state: Any) -> list[dict[str, Any]]:
    """Compact enemy non-tower bodies: card, hp frac, lane, advancement."""

    out: list[dict[str, Any]] = []
    for entity in state.entities.values():
        if not entity.alive or entity.owner != 1 or entity.kind == "tower":
            continue
        row = int(entity.y_mtile) // 1000
        col = int(entity.x_mtile) // 1000
        out.append(
            {
                "card": str(entity.card_id),
                "hp": round(max(0.0, min(1.0, float(entity.hp) / max(1, int(entity.max_hp)))), 4),
                "lane": "left" if col < GRID_COLS // 2 else "right",
                # Advancement toward the learner's side (row 31 = learner king).
                "row": max(0, min(GRID_ROWS - 1, row)),
                "air": bool(getattr(entity, "is_air", False)),
            }
        )
    out.sort(key=lambda t: (-t["row"], t["card"]))
    return out


def balance_tags(
    *,
    archetype: str,
    hand: Sequence[str],
    elixir: float,
    towers: dict[str, list[float]],
    threats: Sequence[dict[str, Any]],
    tick: int,
    tick_limit: int,
) -> dict[str, Any]:
    """Stratification tags for balancing + near-duplicate signatures."""

    threat_cards = sorted({t["card"] for t in threats})
    lanes = sorted({t["lane"] for t in threats})
    airs = sum(1 for t in threats if t["air"])
    phase = tick / max(1, tick_limit)
    return {
        "archetype": archetype,
        "threat_cards": threat_cards,
        "threat_count": len(threats),
        "multi_threat": len(threats) >= 2,
        "air": airs > 0 and airs == len(threats),
        "ground": airs == 0 and len(threats) > 0,
        "lanes": lanes,
        "elixir_bucket": "low" if elixir < 4.0 else ("mid" if elixir < 7.0 else "high"),
        "hand": sorted(hand[:4]),
        "own_tower_bucket": round(min(towers["own"]) if towers["own"] else 1.0, 1),
        "enemy_tower_bucket": round(min(towers["enemy"]) if towers["enemy"] else 1.0, 1),
        "phase": "early" if phase < 0.33 else ("mid" if phase < 0.66 else "late"),
    }


def signature_of(tags: dict[str, Any]) -> str:
    parts = [
        tags.get("archetype", "?"),
        "+".join(tags.get("threat_cards", [])) or "calm",
        "+".join(tags.get("lanes", [])) or "-",
        str(tags.get("elixir_bucket", "?")),
        "+".join(tags.get("hand", [])),
        str(tags.get("own_tower_bucket", "?")),
        "air" if tags.get("air") else ("ground" if tags.get("ground") else "none"),
    ]
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Pool collection (pass 1).
# ---------------------------------------------------------------------------


def _policy_probs(logits: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    mode_p = torch.softmax(logits.mode[0, 0].float(), dim=-1).cpu().numpy()
    card_p = torch.softmax(logits.card[0, 0].float(), dim=-1).cpu().numpy()
    place = logits.placement[0, 0].float()
    place_p = torch.softmax(place.reshape(4, -1), dim=-1).cpu().numpy()
    return (
        np.asarray(mode_p, dtype=np.float64),
        np.asarray(card_p, dtype=np.float64),
        np.asarray(place_p, dtype=np.float64),
    )


def _top_cells(
    place_p: np.ndarray, legal_play: np.ndarray, per_slot: int = 2
) -> list[list[dict[str, Any]]]:
    top: list[list[dict[str, Any]]] = []
    for slot in range(place_p.shape[0]):
        masked = np.where(np.asarray(legal_play[slot]).reshape(-1), place_p[slot], -1.0)
        order = np.argsort(-masked, kind="stable")[:per_slot]
        cells = []
        for flat in order:
            if float(masked[flat]) < 0.0:
                continue
            cells.append(
                {
                    "row": int(flat) // GRID_COLS,
                    "col": int(flat) % GRID_COLS,
                    "p": round(float(place_p[slot, flat]), 4),
                }
            )
        top.append(cells)
    return top


def collect_pool_match(
    *,
    policy: Any,
    cfg: PoolMatchConfig,
    device: Any = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one long trajectory; return a JSON-serializable match record."""

    import torch

    from simulator.actions import PlayCardAction, WaitAction
    from simulator.engine.core import BattleEngine
    from simulator.env import SimulatorEnv
    from simulator.roster import PLAYER_DECK
    from simulator.ruleset import load_fixed_ruleset
    from simulator.public_state_estimator import PublicStateEstimator

    try:
        from .opponent_pool import OpponentPool
        from .simulator_teacher import teacher_label
        from .v4_ppo import _v4_step_inputs
        from .model_v4 import masks_from_legal_play
    except ImportError:  # pragma: no cover
        from simulator.rl.opponent_pool import OpponentPool
        from simulator.rl.simulator_teacher import teacher_label
        from simulator.rl.v4_ppo import _v4_step_inputs
        from simulator.rl.model_v4 import masks_from_legal_play

    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=_stable_seed(cfg.pool_seed, "pool", cfg.archetype))
    scenario = pool.sample(
        cfg.match_index,
        archetype=cfg.archetype,
        strategy=cfg.resolved_strategy() or None,
        allow_variants=cfg.allow_variants,
    )
    learner_deck = tuple(PLAYER_DECK) if cfg.learner_deck == "roster" else tuple(cfg.learner_deck)
    decks = (learner_deck, tuple(scenario.deck.cards))
    match_seed = _stable_seed(cfg.pool_seed, "match", cfg.archetype, cfg.match_index)

    torch.manual_seed(match_seed & 0xFFFFFFFF)
    opponent = scenario.build_controller()
    base = SimulatorEnv(BattleEngine(ruleset, validate_every_tick=False))
    env = base
    env.reset_v2(seed=match_seed, decks=decks, shuffle_decks=True)
    estimator = PublicStateEstimator()
    hidden = policy.initial_hidden(1, device=device)
    needs_reset = True

    decisions: list[dict[str, Any]] = []
    winner: Any = None
    terminal_reason: str = ""
    with torch.no_grad():
        for t in range(cfg.max_decisions):
            state = env.state
            if state is None or state.terminal:
                break
            obs = env.observe_v2_for_viewer(0)
            hand = list(state.players[0].hand[:4])
            elixir = float(state.players[0].elixir_milli) / 1000.0
            snap = estimator.snapshot()
            inputs = _v4_step_inputs(obs, hand, elixir, snap)
            F = torch.float32
            raster = torch.as_tensor(inputs["raster"], dtype=F, device=device).unsqueeze(0).unsqueeze(0)
            glob = torch.as_tensor(inputs["global_features"], dtype=F, device=device).unsqueeze(0).unsqueeze(0)
            entities = torch.as_tensor(inputs["entities"], dtype=F, device=device).unsqueeze(0).unsqueeze(0)
            entity_mask = torch.as_tensor(np.asarray(inputs["entity_mask"], dtype=bool), device=device).unsqueeze(0).unsqueeze(0)
            hand_t = torch.as_tensor(np.asarray(inputs["hand_tokens"], dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0)
            ohp = torch.as_tensor(np.asarray(inputs["opp_hand_probs"], dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0)
            ooc = torch.as_tensor(np.asarray(inputs["opp_out_of_cycle"], dtype=bool), device=device).unsqueeze(0).unsqueeze(0)
            oel = torch.as_tensor(np.asarray(inputs["opp_elixir_interval"], dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0)
            ehist = torch.as_tensor(np.asarray(inputs["event_history"], dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0)
            legal = torch.as_tensor(np.asarray(inputs["legal_play"], dtype=bool), device=device).unsqueeze(0).unsqueeze(0)
            masks = masks_from_legal_play(legal)
            reset_mask = torch.ones((1, 1), dtype=torch.bool, device=device) if needs_reset else torch.zeros((1, 1), dtype=torch.bool, device=device)
            logits, actions, joint_lp, entropy, ht, next_hidden = policy.rollout_sample(
                raster, glob, entities, entity_mask, hand_t, ohp, ooc, oel, ehist,
                masks,
                reset_mask=reset_mask,
                hidden=hidden,
            )
            mode_p, card_p, place_p = _policy_probs(logits)
            ent = {k: round(float(torch.as_tensor(v).mean()), 4) for k, v in entropy.items()}
            legal_np = np.asarray(inputs["legal_play"], dtype=bool)
            try:
                rule = teacher_label(
                    hand_tokens=np.asarray(inputs["hand_tokens"], dtype=np.float32),
                    entity_tokens=np.asarray(inputs["entities"], dtype=np.float32),
                    entity_mask=np.asarray(inputs["entity_mask"], dtype=bool),
                    legal_play=legal_np,
                    legal_wait=bool(inputs["legal_wait"]),
                    own_elixir=float(elixir),
                )
                rule_rec = {"mode": int(rule.mode), "slot": int(rule.card_slot), "row": int(rule.row), "col": int(rule.col)}
            except Exception:
                rule_rec = {"mode": -1, "slot": 0, "row": 0, "col": 0}

            # Decode manually to keep the exact (mode, slot, row, col) triple.
            a_mode = int(actions.mode[0, 0])
            a_slot = int(actions.card_slot[0, 0])
            a_row = int(actions.placement[0, 0, 0])
            a_col = int(actions.placement[0, 0, 1])
            if a_mode == 0:
                learner_action = WaitAction(0)
            else:
                learner_action = PlayCardAction(0, a_slot, (a_col, a_row))
            opp_action = opponent.choose_action(env.engine, env.state, 1)
            towers = tower_fracs(env.state)
            threats = threat_summary(env.state)
            tags = balance_tags(
                archetype=cfg.archetype,
                hand=hand,
                elixir=elixir,
                towers=towers,
                threats=threats,
                tick=int(env.state.tick),
                tick_limit=int(getattr(env.state, "tick_limit", FULL_MATCH_DECISIONS * 5) or FULL_MATCH_DECISIONS * 5),
            )
            decisions.append(
                {
                    "t": t,
                    "state_hash": env.state.state_hash(),
                    "learner": {"mode": a_mode, "slot": a_slot, "row": a_row, "col": a_col},
                    "opponent": {
                        "kind": "play" if isinstance(opp_action, PlayCardAction) else "wait",
                        "slot": int(getattr(opp_action, "card_slot", 0)),
                        "col": int(opp_action.cell[0]) if isinstance(opp_action, PlayCardAction) else 0,
                        "row": int(opp_action.cell[1]) if isinstance(opp_action, PlayCardAction) else 0,
                    },
                    "mode_p": [round(float(mode_p[0]), 4), round(float(mode_p[1]), 4)],
                    "card_p": [round(float(v), 4) for v in card_p],
                    "top_cells": _top_cells(place_p, legal_np),
                    "entropy": ent,
                    "rule": rule_rec,
                    "hand": hand,
                    "elixir": round(elixir, 3),
                    "towers": {k: [round(v, 4) for v in fracs] for k, fracs in towers.items()},
                    "threats": threats,
                    "legal": {
                        "slots": int(legal_np.reshape(4, -1).any(axis=1).sum()),
                        "cells": int(legal_np.sum()),
                    },
                    "tags": tags,
                    "tick": int(env.state.tick),
                }
            )
            result = env.step_v2((learner_action, opp_action))
            needs_reset = False
            hidden = next_hidden.detach()
            if result.terminated or result.truncated:
                winner = result.info.get("winner")
                terminal_reason = str(result.info.get("terminal_reason", ""))
                break
    if verbose:
        print(f"match {cfg.archetype}#{cfg.match_index}: {len(decisions)} decisions winner={winner}", flush=True)
    return {
        "version": FRONTIER_VERSION,
        "config": {
            "archetype": cfg.archetype,
            "strategy": scenario.strategy,
            "allow_variants": cfg.allow_variants,
            "match_index": cfg.match_index,
            "pool_seed": cfg.pool_seed,
            "max_decisions": cfg.max_decisions,
            "match_seed": match_seed,
            "learner_deck": list(learner_deck),
            "opponent_deck": list(scenario.deck.cards),
            "opponent_deck_id": scenario.deck.deck_id,
            "controller_seed": scenario.controller_seed,
        },
        "decisions": decisions,
        "outcome": {"winner": winner, "terminal_reason": terminal_reason, "decisions": len(decisions)},
    }


# ---------------------------------------------------------------------------
# Cheap filter (pass 2).
# ---------------------------------------------------------------------------


def _rule_disagree(dec: dict[str, Any]) -> float:
    rule = dec.get("rule", {})
    if rule.get("mode", -1) < 0:
        return 0.0
    learn = dec["learner"]
    if int(rule["mode"]) != int(learn["mode"]):
        return 1.0
    if int(rule["mode"]) == 1 and int(rule["slot"]) != int(learn["slot"]):
        return 0.5
    return 0.0


def _close_call(dec: dict[str, Any]) -> float:
    mode_p = dec.get("mode_p", [1.0, 0.0])
    margin = abs(float(mode_p[0]) - float(mode_p[1]))
    score = max(0.0, 1.0 - margin * 2.0)
    if int(dec["learner"]["mode"]) == 1:
        card_p = sorted((float(v) for v in dec.get("card_p", [])), reverse=True)
        if len(card_p) >= 2:
            score = max(score, max(0.0, 1.0 - (card_p[0] - card_p[1]) * 2.0))
    return round(score, 4)


def _tower_side(towers: dict[str, list[float]], side: str) -> float:
    return float(sum(towers.get(side, [])))


def deterioration(decisions: Sequence[dict[str, Any]], t: int, lookahead: int) -> float:
    """Subsequent own-tower loss + enemy threat growth after decision t."""

    base = decisions[t]
    end = decisions[min(len(decisions) - 1, t + lookahead)]
    own_loss = max(0.0, _tower_side(base["towers"], "own") - _tower_side(end["towers"], "own"))
    base_hp = sum(th["hp"] for th in base["threats"])
    end_hp = sum(th["hp"] for th in end["threats"])
    threat_growth = max(0.0, end_hp - base_hp)
    new_threats = max(0, len(end["threats"]) - len(base["threats"]))
    return round(own_loss + 0.5 * threat_growth + 0.25 * min(2, new_threats), 4)


def transition(decisions: Sequence[dict[str, Any]], t: int, window: int) -> bool:
    """Tactical transition within (t, t+window]: new threat, first tower damage, tower lost."""

    base = decisions[t]
    base_threat = len(base["threats"])
    base_own = _tower_side(base["towers"], "own")
    base_n_own = len(base["towers"].get("own", []))
    for nxt in decisions[t + 1 : min(len(decisions), t + 1 + window)]:
        if len(nxt["threats"]) > base_threat:
            return True
        if _tower_side(nxt["towers"], "own") < base_own - 1e-9:
            return True
        if len(nxt["towers"].get("own", [])) < base_n_own:
            return True
    return False


def score_decision(
    decisions: Sequence[dict[str, Any]], t: int, cfg: FilterConfig
) -> dict[str, Any]:
    dec = decisions[t]
    disagree = _rule_disagree(dec)
    ent = float(dec.get("entropy", {}).get("joint", 0.0))
    close = _close_call(dec)
    det = deterioration(decisions, t, cfg.det_lookahead)
    trans = transition(decisions, t, cfg.transition_window)
    score = (
        1.0 * disagree
        + 0.5 * min(ent, 2.0) / 2.0
        + 0.5 * close
        + 2.0 * min(det, 1.0)
        + 1.0 * (1.0 if trans else 0.0)
    )
    return {
        "t": t,
        "state_hash": dec["state_hash"],
        "score": round(score, 4),
        "rule_disagree": disagree,
        "entropy": round(ent, 4),
        "close_call": close,
        "deterioration": det,
        "transition": bool(trans),
        "signature": signature_of(dec["tags"]),
    }


def select_candidates(
    match: dict[str, Any], cfg: FilterConfig
) -> list[dict[str, Any]]:
    """Filter one match trajectory to retained candidate states."""

    decisions = match["decisions"]
    if not decisions:
        return []
    scored = [
        score_decision(decisions, t, cfg)
        for t in range(len(decisions))
        if t >= cfg.min_t
    ]
    # Gate: policy/rule disagreement plus (deterioration or transition).
    # Uncertainty and close calls boost ranking but never gate.
    gated = [
        s
        for s in scored
        if s["rule_disagree"] >= 0.5
        and (s["deterioration"] >= cfg.det_threshold or s["transition"])
    ]
    gated.sort(key=lambda s: (-s["score"], decisions[s["t"]]["tick"]))
    kept: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    sig_counts: dict[str, int] = {}
    for s in gated:
        if len(kept) >= cfg.max_per_match:
            break
        if s["state_hash"] in seen_hashes:
            continue
        if kept and s["t"] - kept[-1]["t"] < cfg.min_spacing:
            continue
        if sig_counts.get(s["signature"], 0) >= cfg.per_signature_cap:
            continue
        seen_hashes.add(s["state_hash"])
        sig_counts[s["signature"]] = sig_counts.get(s["signature"], 0) + 1
        dec = decisions[s["t"]]
        kept.append(
            {
                "match_id": f"{match['config']['archetype']}#{match['config']['match_index']}",
                "archetype": match["config"]["archetype"],
                "t": s["t"],
                "tick": dec["tick"],
                "state_hash": s["state_hash"],
                "filter": s,
                "tags": dec["tags"],
                "learner": dec["learner"],
                "hand": dec["hand"],
                "elixir": dec["elixir"],
            }
        )
    return kept


def merge_candidates(
    per_match: Sequence[Sequence[dict[str, Any]]], cfg: FilterConfig
) -> list[dict[str, Any]]:
    all_cands = [c for group in per_match for c in group]
    all_cands.sort(key=lambda c: (-c["filter"]["score"], c["match_id"], c["t"]))
    return all_cands[: cfg.global_top_k]


# ---------------------------------------------------------------------------
# Serialization.
# ---------------------------------------------------------------------------


def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh)


def load_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


__all__ = [
    "CHAMPION_CHECKPOINT",
    "DISCOVERY_ARCHETYPES",
    "FRONTIER_VERSION",
    "FRONTIER_VERSION",
    "FULL_MATCH_DECISIONS",
    "FilterConfig",
    "PoolMatchConfig",
    "balance_tags",
    "collect_pool_match",
    "deterioration",
    "frontier_model_config",
    "load_json",
    "merge_candidates",
    "save_json",
    "score_decision",
    "select_candidates",
    "signature_of",
    "threat_summary",
    "tower_fracs",
    "transition",
]
