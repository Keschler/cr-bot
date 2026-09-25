"""Deterministic search selection, quotas and balance statistics."""

from __future__ import annotations

from collections import Counter
from typing import Any
import numpy as np
from simulator.rl.v4.simulator_teacher import TeacherError
from .config import SearchDatasetConfig, _hash_order
from .records import _disagrees, _spell_opp, _ambiguous


def _select_sim_records(
    sim_records: list[dict[str, Any]], config: SearchDatasetConfig,
) -> list[dict[str, Any]]:
    """Bound search work while retaining spell, ambiguous and disagreement states."""
    # Sim states: select indices, then rebuild + search while envs are live.
    # Caps scale with requested states (ratios fixed at the validated 1500
    # baseline: spell .20 / ambiguous 1/3 / disagree .2667 / fill .40) so
    # large targeted datasets keep the same oversampling shape.
    _cap_scale = max(1.0, config.n_states / 1500.0)
    caps = {
        "spell": int(round(300 * _cap_scale)),
        "ambiguous": int(round(500 * _cap_scale)),
        "disagree": int(round(400 * _cap_scale)),
        "fill": int(round(600 * _cap_scale)),
    }
    counts = dict.fromkeys(caps, 0)
    ordered_sim = sorted(
        sim_records, key=lambda r: _hash_order(config.seed, str(r["provenance"]["state_hash"]))
    )
    sim_to_search: list[dict[str, Any]] = []
    for record in ordered_sim:
        classes = []
        if _spell_opp(record):
            classes.append("spell")
        if _ambiguous(record):
            classes.append("ambiguous")
        if _disagrees(record):
            classes.append("disagree")
        classes.append("fill")
        for choice in classes:
            if counts[choice] < caps[choice]:
                counts[choice] += 1
                sim_to_search.append(record)
                break
    return sim_to_search


def _select_candidates(candidates: list[dict[str, Any]], config: SearchDatasetConfig) -> list[dict[str, Any]]:
    """Select adopted examples and fill quotas in the original deterministic order."""
    n_total = min(config.n_states, len(candidates))
    by_adopted = [c for c in candidates if c["adopted"]]
    need_adopted = min(len(by_adopted), int(round(n_total * config.min_adopted_share)))
    # High-regret adopted states first (still fully deterministic): the
    # outcome gap is the training signal, so large-regret outcomes outrank
    # hash order.  Actor-regret preferred where the champion played legally.
    def _adopted_key(c: dict[str, Any]) -> tuple[float, int]:
        search = c["search"]
        actor_score = search.get("actor_score")
        if actor_score is not None:
            regret = float(search["result"].best.score - actor_score)
        else:
            regret = float(c["regret_rule"])
        return (-regret, _hash_order(config.seed, c["record"]["provenance"]["state_hash"]))
    def _adopted_family(c: dict[str, Any]) -> str:
        return c["record"]["family"] if not c["sim"] else f"sim:{c['record']['family']}"
    # Round-robin across families (regret order within family): without a
    # cap, one high-regret family (e.g. preserve-defense cannon flips) can
    # dominate the adopted slice and wash out rule behavior the frozen
    # transition gate still checks.  Priority with diversity.
    _adopted_by_fam: dict[str, list[dict[str, Any]]] = {}
    for c in by_adopted:
        _adopted_by_fam.setdefault(_adopted_family(c), []).append(c)
    for lst in _adopted_by_fam.values():
        lst.sort(key=_adopted_key)
    selected: list[dict[str, Any]] = []
    while len(selected) < need_adopted:
        progressed = False
        for fam in sorted(_adopted_by_fam):
            lst = _adopted_by_fam[fam]
            if lst and len(selected) < need_adopted:
                selected.append(lst.pop(0))
                progressed = True
        if not progressed:
            break
    taken = {id(c) for c in selected}
    rest = sorted(
        [c for c in candidates if id(c) not in taken],
        key=lambda c: _hash_order(config.seed, c["record"]["provenance"]["state_hash"]),
    )
    for entry in rest:
        if len(selected) >= n_total:
            break
        selected.append(entry)
    quotas = [
        (config.min_spell_share, lambda c: c["spell"]),
        (config.min_transition_share, lambda c: c["transition"]),
        (config.min_sim_share, lambda c: c["sim"]),
    ]
    pool_left = [c for c in rest if id(c) not in {id(s) for s in selected}]

    def share(pred) -> float:
        return sum(1 for c in selected if pred(c)) / max(1, len(selected))

    for minimum, pred in quotas:
        while share(pred) < minimum and pool_left:
            found = next((c for c in pool_left if pred(c)), None)
            if found is None:
                break
            pool_left.remove(found)
            if len(selected) < n_total:
                selected.append(found)
            else:
                victim = next((s for s in selected if not pred(s)), None)
                if victim is None:
                    break
                selected.remove(victim)
                pool_left.append(victim)
                selected.append(found)
    return selected


def _search_stats(selected: list[dict[str, Any]]) -> dict[str, Any]:
    families = Counter(
        c["record"]["family"] if not c["sim"] else f"sim:{c['record']['family']}"
        for c in selected
    )
    waits = [c for c in selected if c["target"].mode == 0]
    regrets = [c["regret_rule"] for c in selected]
    return {
        "n": len(selected),
        "n_wait": len(waits),
        "wait_share": len(waits) / max(1, len(selected)),
        "adopted_share": sum(1 for c in selected if c["adopted"]) / max(1, len(selected)),
        "spell_share": sum(1 for c in selected if c["spell"]) / max(1, len(selected)),
        "transition_share": sum(1 for c in selected if c["transition"]) / max(1, len(selected)),
        "sim_share": sum(1 for c in selected if c["sim"]) / max(1, len(selected)),
        "family_n": dict(families),
        "mean_regret_rule": float(np.mean(regrets)) if regrets else 0.0,
    }


def _assert_search_balance(stats: dict[str, Any], config: SearchDatasetConfig, n_total: int) -> None:
    problems: list[str] = []
    if stats["n"] != n_total:
        problems.append(f"selected {stats['n']} != {n_total} requested")
    if stats["adopted_share"] < config.min_adopted_share - 1e-9:
        problems.append(f"adopted {stats['adopted_share']:.3f} below minimum")
    if stats["spell_share"] < config.min_spell_share - 1e-9:
        problems.append(f"spell {stats['spell_share']:.3f} below minimum")
    if stats["transition_share"] < config.min_transition_share - 1e-9:
        problems.append(f"transition {stats['transition_share']:.3f} below minimum")
    if stats["sim_share"] < config.min_sim_share - 1e-9:
        problems.append(f"sim {stats['sim_share']:.3f} below minimum")
    if problems:
        raise TeacherError("search balance unsatisfiable: " + "; ".join(problems))

