"""Pure paired evaluation statistics and pre/post report summaries."""

from __future__ import annotations

from typing import Any, Sequence

from .constants import CATASTROPHIC_DELTA, PAIRED_SIGN_K, TRIM_FRACTION


def paired_delta_stats(diffs: Sequence[float]) -> dict[str, float | int | None]:
    """Mean / sample-std / standard-error of paired replica deltas.

    Uses the sample standard deviation (ddof=1); ``n < 2`` has unknown uncertainty (std=se=None).
    """
    import math
    import statistics

    vals = [float(v) for v in diffs]
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "std": None, "se": None, "n": 0}
    mean = sum(vals) / n
    if n < 2:
        return {"mean": mean, "std": None, "se": None, "n": n}
    std = statistics.stdev(vals)
    return {"mean": mean, "std": std, "se": std / math.sqrt(n), "n": n}


def classify_paired_sign(
    mean: float, se: float | None, *, same_action: bool, k: float = PAIRED_SIGN_K
) -> str:
    """Confidence label for one state's paired delta.

    ``same``: identical first action (paired deltas identically zero).
    Otherwise the ``k``-SE interval around the mean decides: fully above
    zero is ``improved``, fully below is ``regressed``, anything covering
    zero is ``unresolved``.
    """
    if same_action:
        return "same"
    if se is None:
        return "unresolved"
    if mean - k * se > 0:
        return "improved"
    if mean + k * se < 0:
        return "regressed"
    return "unresolved"


def _sign_of_entry(entry: dict[str, Any], k: float = PAIRED_SIGN_K) -> str:
    """Read a stored entry's sign, recomputing from paired stats if needed."""
    # Raw replicas take precedence over labels stored by older versions that
    # incorrectly treated a singleton as a zero-uncertainty estimate.
    diffs = entry.get("paired_deltas")
    if diffs is not None:
        stats = paired_delta_stats([float(v) for v in diffs])
        return classify_paired_sign(
            stats["mean"], stats["se"],
            same_action=bool(entry.get("same_action")), k=k,
        )
    if entry.get("same_action"):
        return "same"
    if entry.get("n_replicas", 2) < 2:
        return "unresolved"
    sign = entry.get("sign")
    if sign in ("improved", "regressed", "unresolved"):
        return sign
    return "unresolved"


def summarize_choice_regret(
    choice: Sequence[dict[str, Any]],
    *,
    se_k: float = PAIRED_SIGN_K,
    trim_fraction: float = TRIM_FRACTION,
) -> dict[str, Any]:
    """Aggregate sealed statistics that survive single-state dominance.

    Reports mean delta with its across-entry SE, median, trimmed mean
    (``trim_fraction`` cut from each tail; at least one entry per tail when
    ``n >= 6``), confident improved/regressed/unresolved counts from the
    per-entry paired intervals, catastrophic regressions (confident +
    ``paired_mean <= CATASTROPHIC_DELTA``), and a leave-top-mover-out
    dominance check so aggregate means dominated by one volatile state are
    flagged rather than claimed as improvement.
    """
    import math
    import statistics

    rows = list(choice)
    n = len(rows)
    if n == 0:
        return {
            "n": 0, "n_same": 0, "n_changed": 0,
            "mean_delta": 0.0, "mean_se": 0.0,
            "median_delta": 0.0, "trimmed_mean": 0.0,
            "n_improved": 0, "n_regressed": 0, "n_unresolved": 0,
            "n_catastrophic": 0, "catastrophic_ids": [],
            "regressed_ids": [], "improved_ids": [],
            "top_mover_id": None, "top_mover_delta": 0.0,
            "mean_without_top": 0.0, "single_state_dominated": False,
        }
    deltas = [float(r["delta"]) for r in rows]
    mean = sum(deltas) / n
    mean_se = (statistics.stdev(deltas) / math.sqrt(n)) if n >= 2 else 0.0
    median = statistics.median(deltas)
    cut = int(n * trim_fraction)
    if n >= 6:
        cut = max(1, cut)
    else:
        cut = 0
    trimmed = sorted(deltas)[cut : n - cut] if cut else sorted(deltas)
    trimmed_mean = sum(trimmed) / len(trimmed)
    improved_ids, regressed_ids, unresolved_ids = [], [], []
    catastrophic_ids = []
    for r in rows:
        sign = _sign_of_entry(r, k=se_k)
        if sign == "improved":
            improved_ids.append(r["entry_id"])
        elif sign == "regressed":
            regressed_ids.append(r["entry_id"])
            paired_mean = r.get("paired_mean", r["delta"])
            if float(paired_mean) <= CATASTROPHIC_DELTA:
                catastrophic_ids.append(r["entry_id"])
        elif sign == "unresolved":
            unresolved_ids.append(r["entry_id"])
    n_same = sum(1 for r in rows if r.get("same_action"))
    top_idx = max(range(n), key=lambda i: abs(deltas[i]))
    top_id = rows[top_idx]["entry_id"]
    top_delta = deltas[top_idx]
    rest = deltas[:top_idx] + deltas[top_idx + 1 :]
    mean_without_top = sum(rest) / len(rest) if rest else 0.0
    dominated = (
        (mean > 0) != (mean_without_top > 0)
        or abs(mean - mean_without_top) > mean_se
    ) if n >= 2 else False
    return {
        "n": n,
        "n_same": n_same,
        "n_changed": n - n_same,
        "mean_delta": round(mean, 5),
        "mean_se": round(mean_se, 5),
        "median_delta": round(median, 5),
        "trimmed_mean": round(trimmed_mean, 5),
        "n_improved": len(improved_ids),
        "n_regressed": len(regressed_ids),
        "n_unresolved": len(unresolved_ids),
        "n_catastrophic": len(catastrophic_ids),
        "catastrophic_ids": catastrophic_ids,
        "regressed_ids": regressed_ids,
        "improved_ids": improved_ids,
        "top_mover_id": top_id,
        "top_mover_delta": round(top_delta, 5),
        "mean_without_top": round(mean_without_top, 5),
        "single_state_dominated": bool(dominated),
    }


def build_comparison_summary(
    pre: dict[str, Any],
    post: dict[str, Any],
    gates_before: dict[str, Any],
    gates_after: dict[str, Any],
    eval_only_checkpoint: Any = None,
) -> dict[str, Any]:
    """Shared pre/post summary for training and eval-only report paths.

    Pure function (no I/O, no models) so eval-only checkpoint wiring --
    reference ``pre`` preserved, fresh ``post`` compared against it -- is
    unit-testable without running the simulator.
    """
    return {
        "choice_delta_pre": pre["choice_mean_delta"],
        "choice_delta_post": post["choice_mean_delta"],
        "train_choice_delta_pre": pre.get("train_choice_mean_delta"),
        "train_choice_delta_post": post.get("train_choice_mean_delta"),
        "rollout_return_pre": pre["rollout_mean_return"],
        "rollout_return_post": post["rollout_mean_return"],
        "match_wld_pre": pre["match_wld"],
        "match_wld_post": post["match_wld"],
        "gates_green_pre": bool(gates_before.get("accepted")),
        "gates_green_post": bool(gates_after.get("accepted")),
        "choice_summary_pre": pre.get("choice_summary"),
        "choice_summary_post": post.get("choice_summary"),
        "train_choice_summary_pre": pre.get("train_choice_summary"),
        "train_choice_summary_post": post.get("train_choice_summary"),
        "eval_only_checkpoint": eval_only_checkpoint,
    }
