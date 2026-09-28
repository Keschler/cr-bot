"""Fixed-decision recurrent PPO for the compact V4 actor.

Public histories are synchronized before observation. Each action advances one
base simulator decision; WAIT duration is explicitly excluded from the runtime
action distribution. Frozen-anchor targets use the anchor's own recurrent state.
Collectors save actual successor values before any episode reset, including at
rollout tails. The critic consumes detached actor features.

Rollouts use full-batch recurrent updates and zero memory at collection starts.
Frontier starts reconstruct public bookkeeping from event history, but are still
restart scenarios rather than full-match recurrent prefixes. Full-match evaluation
preserves recurrent memory throughout the match."""

from ...foundation._compat import TorchUnavailableError
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise TorchUnavailableError("rl.v4.v4_ppo requires PyTorch. Install torch to run V4 PPO.") from exc
    raise

from ..model_v4 import V4ValueHead
from .config import (
    V4OpponentLane,
    SMOKE_OPPONENT_LANES,
    V4PPOConfig,
    V4_RUNTIME_VERSION,
)

from .records import (
    V4RolloutBatch,
    V4UpdateReport,
)

from .runtime import (
    load_frozen_anchor,
    configure_v4_runtime,
    decode_v4_sim_actions,
    _hand_tokens_partial,
    _v4_step_inputs,
    _forward_sequence,
)

from .rollout_utils import (
    post_step_value,
    rollout_next_values,
)

from .update import (
    update_v4_ppo,
)

from .diagnostics import (
    _head_grad_norms,
    per_head_kl,
    _term_grad_norm,
    _anchor_agreement,
    _decode_drift,
    _explained_variance,
)

from .value import (
    potential_anneal_weight,
    monte_carlo_returns,
    return_distribution,
    fit_value_head,
)

__all__ = [
    "SMOKE_OPPONENT_LANES",
    "V4OpponentLane",
    "V4PPOConfig",
    "V4RolloutBatch",
    "V4UpdateReport",
    "decode_v4_sim_actions",
    "fit_value_head",
    "load_frozen_anchor",
    "monte_carlo_returns",
    "per_head_kl",
    "potential_anneal_weight",
    "return_distribution",
    "update_v4_ppo",
]
