from __future__ import annotations

from itertools import combinations
from typing import Any

from cr_bot.domain.card_metadata import CARD_METADATA


V2_TIER_WEIGHTS = {
    "luna_marker": 1,
    "luna_temporal": 1,
    "luna_specialized": 1,
    "terra_residual": 2,
    "terra_verify": 2,
    "sol_specialized": 3,
}


def cell_distance(left: dict[str, Any], right: dict[str, Any]) -> int:
    """Return coordinate-wise (Chebyshev) grid distance between decisions."""

    return max(abs(left["cell"][0] - right["cell"][0]), abs(left["cell"][1] - right["cell"][1]))


def is_legal_own_cell(decision: dict[str, Any], card: str) -> bool:
    """Apply only card-rule legality that is available without reference labels."""

    base = card[4:] if card.startswith("evo-") else card
    kind = CARD_METADATA[base]["kind"]
    row = int(decision["cell"][1])
    if base == "log":
        return row >= 16
    if kind in {"troop", "building"}:
        return row >= 16
    return True


def direct_legal(decision: dict[str, Any], card: str) -> bool:
    return decision.get("confidence") == "direct" and is_legal_own_cell(decision, card)


def agreeing_pairs(
    attempts: list[tuple[str, dict[str, Any]]], card: str, *, tolerance: int
) -> list[tuple[int, int, int]]:
    pairs = []
    for left in range(len(attempts)):
        if not direct_legal(attempts[left][1], card):
            continue
        for right in range(left + 1, len(attempts)):
            if not direct_legal(attempts[right][1], card):
                continue
            distance = cell_distance(attempts[left][1], attempts[right][1])
            if distance <= tolerance:
                pairs.append((distance, left, right))
    return sorted(pairs)


def select_medoid(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> tuple[str, dict[str, Any]] | None:
    eligible = [item for item in attempts if direct_legal(item[1], card)]
    if not eligible:
        eligible = [item for item in attempts if is_legal_own_cell(item[1], card)]
    if not eligible:
        return None
    ranked = []
    for order, item in enumerate(eligible):
        total = sum(cell_distance(item[1], other[1]) for other in eligible)
        ranked.append((total, order, item))
    return min(ranked, key=lambda row: (row[0], row[1]))[2]


def route_after_primary(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> str:
    """Return accept or the next fixed escalation tier without label access."""

    if len(attempts) != 2:
        raise ValueError("primary routing requires exactly two attempts")
    return "accept" if agreeing_pairs(attempts, card, tolerance=1) else "luna_tiebreak"


def route_after_tiebreak(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> str:
    if len(attempts) < 3:
        raise ValueError("tiebreak routing requires at least three attempts")
    return "accept" if agreeing_pairs(attempts, card, tolerance=1) else "terra"


def route_after_terra(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> str:
    if len(attempts) < 4:
        raise ValueError("Terra routing requires at least four attempts")
    return "accept" if agreeing_pairs(attempts, card, tolerance=1) else "sol"


def tier_family(tier: str) -> str:
    for family in ("luna", "terra", "sol"):
        if tier.startswith(f"{family}_"):
            return family
    raise ValueError(f"unknown localization tier {tier!r}")


def best_v2_cluster(
    attempts: list[tuple[str, dict[str, Any]]],
    card: str,
    *,
    minimum_size: int = 2,
    minimum_families: int = 1,
    required_families: frozenset[str] = frozenset(),
) -> list[int]:
    """Return the strongest pairwise-close blind agreement clique.

    Family diversity ranks ahead of raw vote count so a cross-model agreement
    cannot be displaced by a larger set of correlated Luna prompt variants.
    All inputs are sealed worker decisions; no score or reference coordinate is
    accepted by this function.
    """

    eligible = [
        index
        for index, (_, decision) in enumerate(attempts)
        if direct_legal(decision, card)
    ]
    ranked: list[tuple[tuple[Any, ...], list[int]]] = []
    for size in range(minimum_size, len(eligible) + 1):
        for candidate in combinations(eligible, size):
            if any(
                cell_distance(attempts[left][1], attempts[right][1]) > 1
                for left, right in combinations(candidate, 2)
            ):
                continue
            families = {tier_family(attempts[index][0]) for index in candidate}
            if len(families) < minimum_families or not required_families <= families:
                continue
            weight = sum(V2_TIER_WEIGHTS[attempts[index][0]] for index in candidate)
            dispersion = sum(
                cell_distance(attempts[left][1], attempts[right][1])
                for left, right in combinations(candidate, 2)
            )
            # Earlier attempt indices are the final deterministic tie-breaker.
            key = (
                len(families),
                len(candidate),
                weight,
                -dispersion,
                tuple(-index for index in candidate),
            )
            ranked.append((key, list(candidate)))
    if not ranked:
        return []
    return max(ranked, key=lambda row: row[0])[1]


def route_v2_initial(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> str:
    frozen_roles = [
        "luna_marker",
        "luna_temporal",
        "luna_specialized",
        "terra_residual",
    ]
    tiers = [tier for tier, _ in attempts]
    if len(tiers) != len(set(tiers)) or any(tier not in frozen_roles for tier in tiers):
        raise ValueError("v2 initial routing contains an unknown or duplicate role")
    cluster = best_v2_cluster(
        attempts,
        card,
        minimum_size=3,
        minimum_families=2,
        required_families=frozenset({"terra"}),
    )
    return "accept" if cluster else "terra_verify"


def route_v2_after_terra_verify(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> str:
    frozen_roles = {
        "luna_marker",
        "luna_temporal",
        "luna_specialized",
        "terra_residual",
        "terra_verify",
    }
    tiers = [tier for tier, _ in attempts]
    if len(tiers) != len(set(tiers)) or any(tier not in frozen_roles for tier in tiers):
        raise ValueError("v2 Terra routing contains an unknown or duplicate role")
    cluster = best_v2_cluster(
        attempts,
        card,
        minimum_size=3,
        minimum_families=2,
        required_families=frozenset({"terra"}),
    )
    return "accept" if cluster else "sol_specialized"


def _validate_role_subset(
    attempts: list[tuple[str, dict[str, Any]]], allowed: set[str], stage: str
) -> None:
    tiers = [tier for tier, _ in attempts]
    if len(tiers) != len(set(tiers)) or any(tier not in allowed for tier in tiers):
        raise ValueError(f"{stage} routing contains an unknown or duplicate role")


def _v3_has_cross_family_clique(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> bool:
    return bool(
        best_v2_cluster(
            attempts,
            card,
            minimum_size=3,
            minimum_families=2,
            required_families=frozenset({"terra"}),
        )
    )


def best_v4_cluster(
    attempts: list[tuple[str, dict[str, Any]]],
    card: str,
    *,
    required_families: frozenset[str] = frozenset({"terra"}),
) -> list[int]:
    """Return a confidence-aware cross-family localization clique.

    Direct decisions may agree within one cell. An inferred decision may
    participate only when at least two other decisions are direct and every
    coordinate in the clique is identical. This prevents a subjective
    confidence word from forcing an escalation when independent model
    families report the exact same legal cell, without weakening the normal
    tolerance for fully direct evidence.
    """

    eligible = [
        index
        for index, (_, decision) in enumerate(attempts)
        if is_legal_own_cell(decision, card)
    ]
    ranked: list[tuple[tuple[Any, ...], list[int]]] = []
    for size in range(3, len(eligible) + 1):
        for candidate_tuple in combinations(eligible, size):
            candidate = list(candidate_tuple)
            distances = [
                cell_distance(attempts[left][1], attempts[right][1])
                for left, right in combinations(candidate, 2)
            ]
            direct_count = sum(
                attempts[index][1].get("confidence") == "direct"
                for index in candidate
            )
            all_direct = direct_count == len(candidate)
            if all_direct:
                if any(distance > 1 for distance in distances):
                    continue
            elif direct_count < 2 or any(distance != 0 for distance in distances):
                continue
            families = {tier_family(attempts[index][0]) for index in candidate}
            if len(families) < 2 or not required_families <= families:
                continue
            weight = sum(V2_TIER_WEIGHTS[attempts[index][0]] for index in candidate)
            key = (
                len(families),
                direct_count,
                len(candidate),
                weight,
                -sum(distances),
                tuple(-index for index in candidate),
            )
            ranked.append((key, candidate))
    if not ranked:
        return []
    return max(ranked, key=lambda row: row[0])[1]


def _v4_has_cross_family_clique(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> bool:
    return bool(best_v4_cluster(attempts, card))


def route_v4_initial(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> str:
    """Route after three initial perspectives using exact inferred consensus."""

    _validate_role_subset(
        attempts,
        {"luna_marker", "luna_temporal", "terra_residual"},
        "v4 initial",
    )
    return "accept" if _v4_has_cross_family_clique(attempts, card) else "luna_specialized"


def route_v4_after_luna_specialized(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> str:
    _validate_role_subset(
        attempts,
        {"luna_marker", "luna_temporal", "terra_residual", "luna_specialized"},
        "v4 Luna-specialized",
    )
    return "accept" if _v4_has_cross_family_clique(attempts, card) else "terra_verify"


def route_v4_after_terra_verify(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> str:
    _validate_role_subset(
        attempts,
        {
            "luna_marker",
            "luna_temporal",
            "terra_residual",
            "luna_specialized",
            "terra_verify",
        },
        "v4 Terra",
    )
    return "accept" if _v4_has_cross_family_clique(attempts, card) else "sol_specialized"


def select_v4_consensus(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> tuple[tuple[str, dict[str, Any]], list[int]] | None:
    """Select a medoid only from a qualified v4 cross-family clique."""

    cluster = best_v4_cluster(attempts, card, required_families=frozenset())
    if not cluster:
        return None
    representatives = []
    for index in cluster:
        total_distance = sum(
            cell_distance(attempts[index][1], attempts[other][1])
            for other in cluster
        )
        tier, decision = attempts[index]
        inferred_penalty = decision.get("confidence") != "direct"
        representatives.append(
            (total_distance, inferred_penalty, -V2_TIER_WEIGHTS[tier], index)
        )
    selected_index = min(representatives)[3]
    return attempts[selected_index], cluster


def route_v3_initial(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> str:
    """Route the batched v3 cascade after its three initial perspectives."""

    _validate_role_subset(
        attempts,
        {"luna_marker", "luna_temporal", "terra_residual"},
        "v3 initial",
    )
    return "accept" if _v3_has_cross_family_clique(attempts, card) else "luna_specialized"


def route_v3_after_luna_specialized(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> str:
    _validate_role_subset(
        attempts,
        {"luna_marker", "luna_temporal", "terra_residual", "luna_specialized"},
        "v3 Luna-specialized",
    )
    return "accept" if _v3_has_cross_family_clique(attempts, card) else "terra_verify"


def route_v3_after_terra_verify(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> str:
    _validate_role_subset(
        attempts,
        {
            "luna_marker",
            "luna_temporal",
            "terra_residual",
            "luna_specialized",
            "terra_verify",
        },
        "v3 Terra",
    )
    return "accept" if _v3_has_cross_family_clique(attempts, card) else "sol_specialized"


def select_v2_consensus(
    attempts: list[tuple[str, dict[str, Any]]], card: str
) -> tuple[tuple[str, dict[str, Any]], list[int]] | None:
    """Select only inside the best agreement clique, never across dissenters."""

    cluster = best_v2_cluster(attempts, card, minimum_size=2)
    if cluster:
        representatives = []
        for index in cluster:
            total_distance = sum(
                cell_distance(attempts[index][1], attempts[other][1])
                for other in cluster
            )
            tier = attempts[index][0]
            representatives.append(
                (total_distance, -V2_TIER_WEIGHTS[tier], index)
            )
        selected_index = min(representatives)[2]
        return attempts[selected_index], cluster

    eligible = [
        index
        for index, (_, decision) in enumerate(attempts)
        if direct_legal(decision, card)
    ]
    if not eligible:
        eligible = [
            index
            for index, (_, decision) in enumerate(attempts)
            if is_legal_own_cell(decision, card)
        ]
    if not eligible:
        return None
    selected_index = max(
        eligible,
        key=lambda index: (V2_TIER_WEIGHTS[attempts[index][0]], index),
    )
    return attempts[selected_index], [selected_index]
