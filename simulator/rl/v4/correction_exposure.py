"""Declared correction exposure and diagnostic gates, without model imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class CorrectionGateConfig:
    """Screening thresholds, never a certificate of full-game strength."""

    min_nll_drop: float = 1.0
    min_mode_probability: float = 0.1
    min_card_probability: float = 0.1
    min_neighborhood_probability: float = 0.1
    max_mean_retention_kl: float = 0.02
    max_row_retention_kl: float = 0.5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if "probability" in name and value > 1:
                raise ValueError(f"{name} must be at most one")


def exposure_schedule(factual: Mapping[str, Sequence[int]], sequences: Mapping[str, Sequence[int]], *,
                      exposures: int, optimizer_steps: int, seed: int) -> list[list[tuple[str, int]]]:
    """Add target windows to existing updates, preserving their exact count.

    A factual target already occurs in the chronological pass. A certified
    branch target has one base presentation here, plus the requested extras.
    """
    if type(exposures) is not int or exposures < 0 or type(optimizer_steps) is not int or optimizer_steps <= 0:
        raise ValueError("exposures must be nonnegative and optimizer_steps positive")
    if set(factual) & set(sequences):
        raise ValueError("factual and branch source identities overlap")
    targets = [(job, int(step)) for source, repeats in ((factual, exposures), (sequences, exposures + 1))
               for job in sorted(source) for step in sorted(source[job]) for _ in range(repeats)]
    result: list[list[tuple[str, int]]] = [[] for _ in range(optimizer_steps)]
    if targets:
        order = np.random.default_rng(seed).permutation(len(targets))
        for index, selected in enumerate(order):
            result[index * optimizer_steps // len(targets)].append(targets[int(selected)])
    return result


def correction_learning_gate(train: Mapping[str, Any], calibration: Mapping[str, Any], *,
                             config: CorrectionGateConfig = CorrectionGateConfig()) -> dict[str, Any]:
    """Decide whether label uptake warrants fresh games, without promotion."""
    def uptake(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
        rows = diagnostic.get("factors", [])
        if not rows:
            return dict(status="unmeasured", passed=False, rows=0, reasons=["no independently certified labels"])
        failures = []
        drops = []
        for row in rows:
            before, after = row["before"], row["after"]
            values = (before["nll"], after["nll"], after["target_mode_probability"])
            if not all(math.isfinite(float(value)) for value in values):
                failures.append("nonfinite diagnostic")
                continue
            drops.append(before["nll"] - after["nll"])
            if after["nll"] > before["nll"] + 1e-6:
                failures.append("a target became less likely")
            if after["target_mode_probability"] < config.min_mode_probability:
                failures.append("target mode remains unlikely")
            if row["target"][0] == 1:
                card, near = after["card_probability_given_play"], after["nearby_cell_probability_given_card"]
                if card is None or not math.isfinite(card) or card < config.min_card_probability:
                    failures.append("target card remains unlikely")
                if near is None or not math.isfinite(near) or near < config.min_neighborhood_probability:
                    failures.append("target placement neighborhood remains unlikely")
        mean_drop = sum(drops) / len(drops) if drops else None
        if mean_drop is None or mean_drop < config.min_nll_drop:
            failures.append("mean target NLL reduction below declared threshold")
        return dict(status="measured", passed=not failures, rows=len(rows), mean_nll_drop=mean_drop,
                    reasons=sorted(set(failures)))

    retention_reasons = []
    for split, diagnostic in (("train", train), ("calibration", calibration)):
        if not diagnostic.get("inactive_rows", 0):
            retention_reasons.append(f"{split} retention unmeasured")
        for key, limit in (("retention_kl", config.max_mean_retention_kl),
                           ("retention_kl_max", config.max_row_retention_kl)):
            value = diagnostic.get(key)
            if value is None or not math.isfinite(value) or value > limit:
                retention_reasons.append(f"{split} {key} outside declared limit")
    training, held_out = uptake(train), uptake(calibration)
    return dict(thresholds=asdict(config), train_memorization=training, held_out_label_uptake=held_out,
                retention=dict(passed=not retention_reasons, reasons=retention_reasons),
                eligible_for_fresh_games=training["passed"] and held_out["passed"] and not retention_reasons,
                selected_actor_changed=False, used_for_epoch_selection=False,
                limitation="Label uptake and recorded-state retention do not establish full-game strength.")
