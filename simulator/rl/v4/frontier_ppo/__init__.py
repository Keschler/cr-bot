"""Frontier PPO: train on frontier entries + controlled pre/post evaluation.

Training episodes start from materialized frontier entries (naturally
reached, demonstrably mistaken states) on bare simulator envs with the sparse
terminal-outcome reward.  Entry loads use zero-restart recurrent semantics
(``reset=True``), identical to the PPO smoke setup, so the update unroll
from ``initial_hidden=zeros`` stays exact.

Evaluation (identical protocol pre/post, paired seeds throughout):

* sealed choice-regret: each sealed entry's frozen-sim action vs the
  policy-under-test's action, both scored by frozen-champion continuation
  with matched replica seeds;
* sealed rollout return: stochastic rollouts from sealed entries with common
  random numbers across the two policies;
* full-match W/L/D: fresh deterministic matches vs a configured opponent
  pool, paired by (archetype, seed)."""

from .constants import (
    ADAPTIVE_REPLICA_CAP,
    CATASTROPHIC_DELTA,
    FRONTIER_PPO_EPISODE_CAP,
    FRONTIER_PPO_VERSION,
    PAIRED_SIGN_K,
    TRIM_FRACTION,
)
from .statistics import (
    paired_delta_stats,
    classify_paired_sign,
    _sign_of_entry,
    summarize_choice_regret,
    build_comparison_summary,
)
from .environment import (
    make_frontier_env,
    pinned_opponent_decks,
    heldout_opponent_decks,
    real_eval_opponent_decks,
)
from .collection import (
    collect_frontier_rollout,
)
from .actions import (
    _policy_argmax_at_entry,
    _entry_continuation_hidden,
    action_logit_margins,
)
from .evaluation import (
    sealed_choice_regret,
    sealed_rollout_returns,
    full_match_record,
)

__all__ = [
    'ADAPTIVE_REPLICA_CAP',
    'CATASTROPHIC_DELTA',
    'FRONTIER_PPO_EPISODE_CAP',
    'FRONTIER_PPO_VERSION',
    'PAIRED_SIGN_K',
    'TRIM_FRACTION',
    'action_logit_margins',
    'build_comparison_summary',
    'classify_paired_sign',
    'collect_frontier_rollout',
    'full_match_record',
    'make_frontier_env',
    'paired_delta_stats',
    'pinned_opponent_decks',
    'heldout_opponent_decks',
    'real_eval_opponent_decks',
    'sealed_choice_regret',
    'sealed_rollout_returns',
    'summarize_choice_regret',
]
