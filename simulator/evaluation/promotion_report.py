"""Formal champion-challenger promotion rule (paper gates G0-G6).

A challenger is promoted only when *all* mandatory gates pass on one sealed
paired suite.  Training reward, teacher agreement, PPO KL, entropy, loss
curves, Elo, or crowns on one script are diagnostics; none promotes alone.
Thresholds are policy requirements chosen *before* the challenger result is
visible, not tuned after looking at a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    """Predeclared practical margins and regression tolerances."""

    eps_match: float = 0.02
    eps_tact: float = 0.0
    rho_stratum: float = 0.05
    rho_cat: float = 0.01

    def __post_init__(self) -> None:
        for name in ("eps_match", "eps_tact", "rho_stratum", "rho_cat"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PromotionReport:
    """Machine-readable promotion artifact for one challenger."""

    champion_hash: str
    challenger_hash: str
    contract_hash: str
    manifest_hash: str
    gates: tuple[GateResult, ...]
    summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "gates", tuple(self.gates))

    @property
    def promoted(self) -> bool:
        return bool(self.gates) and all(gate.passed for gate in self.gates)

    def as_dict(self) -> dict[str, object]:
        return {
            "champion_hash": self.champion_hash,
            "challenger_hash": self.challenger_hash,
            "contract_hash": self.contract_hash,
            "manifest_hash": self.manifest_hash,
            "promoted": self.promoted,
            "gates": [
                {"name": gate.name, "passed": gate.passed, "detail": gate.detail}
                for gate in self.gates
            ],
            "summary": self.summary,
        }


def evaluate_promotion(
    *,
    champion_hash: str,
    challenger_hash: str,
    contract_hash: str,
    manifest_hash: str,
    thresholds: PromotionThresholds,
    validity_clean: bool,
    match_lcb: float,
    tact_lcb: float,
    stratum_lcbs: dict[str, float],
    cat_ucb_diff: float,
    head_to_head_diff: float,
    integrity_ok: bool,
    notes: str = "",
) -> PromotionReport:
    """Apply gates G0-G6 to precomputed paired statistics.

    ``match_lcb``/``tact_lcb`` are the lower confidence bounds of the
    challenger-minus-champion improvement; ``stratum_lcbs`` the simultaneous
    per-stratum lower bounds; ``cat_ucb_diff`` the upper bound of the
    catastrophic-rate difference ``Rcat(C) - Rcat(B)``; ``head_to_head_diff``
    the paired side-swapped direct-match difference.
    """

    gates = (
        GateResult(
            "G0-validity",
            bool(validity_clean),
            "same sealed contracts, public-only actor, complete horizons, "
            "deterministic replay, zero rejected/fallback, exploit audit clean",
        ),
        GateResult(
            "G1-primary-match",
            float(match_lcb) > float(thresholds.eps_match),
            f"LCB95(dmatch)={float(match_lcb):.4f} > eps={thresholds.eps_match}",
        ),
        GateResult(
            "G2-tactical",
            float(tact_lcb) >= float(thresholds.eps_tact),
            f"LCB95(dtact)={float(tact_lcb):.4f} >= eps={thresholds.eps_tact}",
        ),
        GateResult(
            "G3-no-archetype-collapse",
            all(float(v) > -float(thresholds.rho_stratum) for v in stratum_lcbs.values())
            and bool(stratum_lcbs),
            f"min stratum LCB={min((float(v) for v in stratum_lcbs.values()), default=float('nan')):.4f}",
        ),
        GateResult(
            "G4-catastrophic",
            float(cat_ucb_diff) < float(thresholds.rho_cat),
            f"UCB(Rcat(C)-Rcat(B))={float(cat_ucb_diff):.4f} < rho={thresholds.rho_cat}",
        ),
        GateResult(
            "G5-head-to-head",
            float(head_to_head_diff) >= 0.0,
            f"direct paired diff={float(head_to_head_diff):.4f} >= 0",
        ),
        GateResult(
            "G6-integrity",
            bool(integrity_ok),
            "no collapse of card usage, WAIT duration, placement diversity, "
            "or legal-action support",
        ),
    )
    summary = (
        "PROMOTED" if all(gate.passed for gate in gates) else "REJECTED"
    )
    if notes:
        summary = f"{summary}: {notes}"
    return PromotionReport(
        champion_hash=champion_hash,
        challenger_hash=challenger_hash,
        contract_hash=contract_hash,
        manifest_hash=manifest_hash,
        gates=gates,
        summary=summary,
    )


__all__ = [
    "GateResult",
    "PromotionReport",
    "PromotionThresholds",
    "evaluate_promotion",
]
