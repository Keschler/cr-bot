"""Deterministic search selection, quotas and balance statistics."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import ceil
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
    return _satisfy_quotas(selected, rest, config, n_total)


def _satisfy_quotas(
    selected: list[dict[str, Any]], rest: list[dict[str, Any]],
    config: SearchDatasetConfig, n_total: int,
) -> list[dict[str, Any]]:
    """Solve the four lower bounds jointly, grouping by their 16 Boolean masks.

    Records contain NumPy arrays, so membership and removal must never use
    dictionary equality. Work with group counts instead. Preserve the preferred
    selection unchanged if it satisfies all quotas, and prefer its counts when
    repair is necessary. Exhaustive bounded count search makes failure mean
    infeasibility rather than failure of a greedy sequence of replacements.
    """
    fields = ("adopted", "spell", "transition", "sim")
    required = tuple(
        ceil(n_total * getattr(config, f"min_{field}_share") - 1e-9)
        for field in fields
    )
    if all(sum(bool(c[field]) for c in selected) >= need
           for field, need in zip(fields, required)):
        return selected
    taken = {id(c) for c in selected}
    ordered = selected + [c for c in rest if id(c) not in taken]
    groups: dict[tuple[bool, ...], list[dict[str, Any]]] = {}
    preferred: Counter = Counter()
    for c in ordered:
        mask = tuple(bool(c[field]) for field in fields)
        groups.setdefault(mask, []).append(c)
        if id(c) in taken:
            preferred[mask] += 1
    # Multi-quota records first; stable Boolean order breaks ties.
    masks = sorted(groups, key=lambda mask: (-sum(mask), mask))
    capacities = [len(groups[mask]) for mask in masks]
    suffix_total = [0] * (len(masks) + 1)
    suffix_quota = [[0] * len(fields) for _ in suffix_total]
    for i in range(len(masks) - 1, -1, -1):
        suffix_total[i] = suffix_total[i + 1] + capacities[i]
        suffix_quota[i] = [
            suffix_quota[i + 1][j] + capacities[i] * masks[i][j]
            for j in range(len(fields))
        ]

    @lru_cache(maxsize=100_000)
    def solve(i: int, slots: int, needs: tuple[int, ...]) -> tuple[int, ...] | None:
        if slots < 0 or suffix_total[i] < slots:
            return None
        if any(need > min(slots, suffix_quota[i][j]) for j, need in enumerate(needs)):
            return None
        if i == len(masks):
            return () if slots == 0 else None
        mask = masks[i]
        lower = max(0, slots - suffix_total[i + 1])
        upper = min(slots, capacities[i])
        for j, need in enumerate(needs):
            if mask[j]:
                lower = max(lower, need - suffix_quota[i + 1][j])
            else:
                upper = min(upper, slots - need)
        # Try the existing count first, then nearby counts deterministically.
        for count in sorted(range(lower, upper + 1),
                            key=lambda value: (abs(value - preferred[mask]), -value)):
            tail = solve(i + 1, slots - count,
                         tuple(max(0, need - count * mask[j])
                               for j, need in enumerate(needs)))
            if tail is not None:
                return (count,) + tail
        return None

    counts = solve(0, n_total, required)
    if counts is None:
        raise TeacherError("search balance unsatisfiable: no joint quota selection exists")
    chosen = {id(c) for mask, count in zip(masks, counts) for c in groups[mask][:count]}
    return [c for c in ordered if id(c) in chosen]



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

