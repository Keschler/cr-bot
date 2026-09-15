"""Minimal V4 PPO smoke path: frozen-anchor short-scenario training.

This module wires the already-validated pieces into the smallest safe PPO
loop for the 117k-parameter V4 actor; it deliberately does NOT replicate
the prototype training stack:

* rollout uses :meth:`RecurrentV4Policy.rollout_sample` (masked, fail-closed)
  with per-lane public estimators, so inputs match distillation exactly;
* the objective is :func:`objectives.ppo_objective` plus a BC term on
  frozen-anchor actions (no new loss math);
* the critic is :class:`V4ValueHead` on detached actor features (actor
  parameters never move under the value loss; the privileged critic stays
  the deferred stronger option);
* opponents are scripted archetype lanes (self-play-lite flagged off until
  mirrored-V4 opponent quality is validated);
* rewards are the native short-scenario potentials (validated, anti-exploit
  history) with tower/crown/wincome components logged separately.

Deliberate simplifications for the smoke (documented follow-ups): full-batch
updates (no recurrent minibatching), fresh estimator per episode (matches
distillation inputs exactly; evolving beliefs are a separate project),
uniform single-step-equivalent GRU carryover with no burn-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Sequence

from ._compat import TorchUnavailableError

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise TorchUnavailableError(
            "rl.v4_ppo requires PyTorch. Install torch to run the V4 PPO smoke."
        ) from exc
    raise

import numpy as np

from .distillation import PLAYER_DECK as FIXED_PLAYER_DECK
from .model_v4 import (
    ModelConfigV4,
    RecurrentV4Policy,
    V4ActionBatch,
    V4ValueHead,
    masks_from_legal_play,
)
from .objectives import PPOObjectiveConfig, behavior_cloning_loss, compute_gae, ppo_objective
from .trajectory import ActionMasks, RecurrentSequence, TrajectoryBatch


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4OpponentLane:
    """One scripted opponent lane: deck archetype, play strategy, source.

    ``strategy="deterministic-cycle"`` is the legible anchor controller;
    every other lane uses a distinct heuristic strategy so PPO cannot
    memorize one cycle.  ``controller_lane`` only applies to the
    deterministic anchor (heuristic factories manage their own lanes).
    """

    archetype: str
    source: str
    strategy: str = "deterministic-cycle"
    controller_lane: str = "alternate"


SMOKE_OPPONENT_LANES: tuple[V4OpponentLane, ...] = (
    V4OpponentLane("beatdown", "ground-defense", "beatdown"),
    V4OpponentLane("beatdown", "ground-defense", "deterministic-cycle", "left"),
    V4OpponentLane("air-beatdown", "air-defense", "air-beatdown"),
    V4OpponentLane("aggressive-pressure", "isolated-offense", "aggressive-pressure"),
    V4OpponentLane("defensive-cycle", "kiting-cycling-elixir", "defensive-cycle"),
    V4OpponentLane("siege-bait", "spell-situations", "siege-bait"),
    V4OpponentLane("random-legal", "isolated-offense", "random-legal"),
    V4OpponentLane("aggressive-pressure", "bridge-defense", "aggressive-pressure"),
)
"""Default smoke mix: seven distinct heuristic behaviors plus one
deterministic anchor lane.  No controller family holds a majority."""


@dataclass(frozen=True, slots=True)
class V4PPOConfig:
    """Knobs for one short-scenario V4 PPO smoke run."""

    n_updates: int = 5
    decisions_per_rollout: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    bc_coef: float = 0.05
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    ppo_epochs: int = 2
    seed: int = 0
    device: str = "cpu"
    # Quarantine thresholds are initial values to calibrate, except the
    # hard ones (illegal actions, non-finite losses) which are exact.
    max_approx_kl: float = 0.03
    max_mean_abs_log_ratio: float = 0.05
    max_entropy_collapse_fraction: float = 0.50
    max_placement_grad_ratio: float = 10.0

    def __post_init__(self) -> None:
        if type(self.n_updates) is not int or self.n_updates <= 0:
            raise ValueError("n_updates must be a positive integer")
        if type(self.decisions_per_rollout) is not int or self.decisions_per_rollout <= 0:
            raise ValueError("decisions_per_rollout must be a positive integer")
        if type(self.ppo_epochs) is not int or self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be a positive integer")
        for name in (
            "gamma",
            "gae_lambda",
            "clip_epsilon",
            "value_coef",
            "entropy_coef",
            "bc_coef",
            "actor_lr",
            "critic_lr",
            "max_approx_kl",
            "max_mean_abs_log_ratio",
            "max_entropy_collapse_fraction",
            "max_placement_grad_ratio",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= float(self.gae_lambda) <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")


# ---------------------------------------------------------------------------
# Frozen anchor.
# ---------------------------------------------------------------------------


def load_frozen_anchor(
    path: str,
    config: ModelConfigV4 | None = None,
    *,
    expected_sha256: str | None = None,
    device: Any = None,
) -> RecurrentV4Policy:
    """Load simx12 (or any V4 checkpoint) as a frozen BC anchor.

    The anchor is set to eval mode with every parameter frozen; an optional
    sha256 over the checkpoint file pins exactly which weights train
    against.  Run provenance should record the returned digest.
    """

    import hashlib

    with open(path, "rb") as handle:
        payload = handle.read()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"anchor checkpoint hash mismatch: expected {expected_sha256}, got {digest}"
        )
    policy = RecurrentV4Policy(config or ModelConfigV4())
    policy.eval()
    with torch.no_grad():
        policy(
            torch.zeros(1, 1, 21, 32, 18),
            torch.zeros(1, 1, 768),
            torch.zeros(1, 1, 128, 32),
            torch.zeros(1, 1, 128, dtype=torch.bool),
            torch.zeros(1, 1, 4, 16),
            torch.full((1, 1, 128), 1.0 / 128),
            torch.zeros(1, 1, 128, dtype=torch.bool),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 16, 8),
            torch.ones(1, 1, dtype=torch.bool),
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    # The placement cell-bias materializes lazily on first forward, so a
    # checkpoint saved before any forward legitimately lacks it.  Tolerate
    # exactly that key and nothing else (fail-closed on any other drift).
    result = policy.load_state_dict(state, strict=False)
    unexpected = [k for k in result.unexpected_keys]
    missing = [k for k in result.missing_keys if k != "heads.cell_bias"]
    if unexpected or missing:
        raise RuntimeError(
            f"anchor checkpoint incompatible: unexpected={unexpected} missing={missing}"
        )
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)
    if device is not None:
        policy = policy.to(device)
    policy.anchor_sha256 = digest  # type: ignore[attr-defined]
    return policy


# ---------------------------------------------------------------------------
# Action decoding (public actions only).
# ---------------------------------------------------------------------------


def decode_v4_sim_actions(
    actions: V4ActionBatch, *, player: int = 0
) -> list[Any]:
    """Decode one batch element of V4 actions to simulator actions."""

    try:
        from ..actions import PlayCardAction, WaitAction
    except ImportError:  # pragma: no cover - top-level ``rl`` imports
        from simulator.actions import PlayCardAction, WaitAction
    if type(player) is not int or player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    decoded: list[Any] = []
    mode = actions.mode.reshape(-1)
    slot = actions.card_slot.reshape(-1)
    place = actions.placement.reshape(-1, 2)
    for index in range(int(mode.numel())):
        if int(mode[index]) == 0:
            decoded.append(WaitAction(player))
        else:
            decoded.append(
                PlayCardAction(
                    player,
                    int(slot[index]),
                    (int(place[index][1]), int(place[index][0])),
                )
            )
    return decoded


# ---------------------------------------------------------------------------
# Rollout collection.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class V4RolloutBatch:
    """One collected rollout: PPO tensors plus anchor/diagnostic material."""

    trajectory: TrajectoryBatch
    anchor_actions: V4ActionBatch | None
    recurrent: torch.Tensor | None = None
    old_head_logits: dict[str, torch.Tensor] | None = None
    anchor_log_probs: torch.Tensor | None = None
    per_head_entropy: dict[str, torch.Tensor] | None = None
    wait_fraction: float = 0.0
    card_histogram: dict[str, float] | None = None
    illegal_attempts: int = 0


def _hand_tokens_partial(hand: Sequence[str], elixir: float) -> np.ndarray:
    """Encode a live hand that may transiently hold fewer than four cards.

    Fresh-reset distillation always sees full hands, but mid-rollout the
    refill cooldown leaves three-card hands.  Present cards keep their true
    slot encoding; missing slots stay zero (their mask rows are empty, so
    the sampler can never select them).
    """

    try:
        from .simulator_distillation import build_hand_tokens
    except ImportError:  # pragma: no cover
        from simulator.rl.simulator_distillation import build_hand_tokens
    tokens = np.zeros((4, 16), dtype=np.float32)
    for slot, card in enumerate(list(hand)[:4]):
        row = np.array(build_hand_tokens([card] * 4, float(elixir))[0], copy=True)
        row[2] = slot / 3.0
        tokens[slot] = row
    return tokens


def _v4_step_inputs(
    observation: Any,
    hand: Sequence[str],
    elixir: float,
    snapshot: Any,
) -> dict[str, np.ndarray]:
    """Map one public observation + estimator snapshot to V4 inputs."""

    return {
        "raster": np.array(observation.board, dtype=np.float32),
        "global_features": np.array(observation.global_vector, dtype=np.float32),
        "entities": np.array(observation.entity_tokens, dtype=np.float32),
        "entity_mask": np.array(observation.entity_mask, dtype=bool),
        "hand_tokens": _hand_tokens_partial(hand, float(elixir)),
        "opp_hand_probs": np.array(snapshot.opp_hand_probs, dtype=np.float32),
        "opp_out_of_cycle": np.array(snapshot.opp_out_of_cycle, dtype=bool),
        "opp_elixir_interval": np.asarray(
            [snapshot.opp_elixir_lo, snapshot.opp_elixir_hi], dtype=np.float32
        ),
        "event_history": np.array(snapshot.event_history, dtype=np.float32),
        "legal_play": np.array(observation.legal_play, dtype=bool),
        "legal_wait": bool(observation.legal_wait),
    }


def collect_v4_rollout(
    policy: RecurrentV4Policy,
    critic: V4ValueHead,
    anchor: RecurrentV4Policy | None,
    lanes: Sequence[dict[str, Any]],
    *,
    n_decisions: int,
    device: Any = None,
    deterministic_anchor: bool = True,
    deterministic_policy: bool = False,
) -> tuple[V4RolloutBatch, dict[str, Any]]:
    """Collect one fixed-length V4 rollout across scenario lanes.

    Each lane dict carries ``env`` (a reset ``BasicMechanicsScenarioEnv``),
    ``opponent`` (a controller with ``choose_action(engine, state, player)``),
    and ``estimator`` (a fresh ``PublicStateEstimator``).  Envs that
    terminate mid-rollout reset in place with the lane's next seed and a
    ``reset_mask`` break, so every lane contributes exactly ``n_decisions``.
    """

    device = device or next(policy.parameters()).device
    n_lanes = len(lanes)
    if n_lanes < 1:
        raise ValueError("at least one rollout lane is required")
    if type(n_decisions) is not int or n_decisions < 1:
        raise ValueError("n_decisions must be a positive integer")
    if type(deterministic_anchor) is not bool:
        raise TypeError("deterministic_anchor must be boolean")
    if type(deterministic_policy) is not bool:
        raise TypeError("deterministic_policy must be boolean")
    policy.eval()
    critic.eval()
    if anchor is not None:
        anchor.eval()
    hidden = policy.initial_hidden(n_lanes, device=device)
    steps: list[dict[str, torch.Tensor]] = []
    illegal_attempts = 0
    wait_count = 0
    episode_wins = 0
    episode_losses = 0
    episode_draws = 0
    all_rewards: list[float] = []
    lane_outcomes: list[list[int]] = [[0, 0, 0] for _ in range(n_lanes)]
    card_counts: dict[int, int] = {}
    with torch.no_grad():
        for _ in range(n_decisions):
            batch_inputs: dict[str, list[np.ndarray]] = {
                key: [] for key in (
                    "raster", "global_features", "entities", "entity_mask",
                    "hand_tokens", "opp_hand_probs", "opp_out_of_cycle",
                    "opp_elixir_interval", "event_history", "legal_play",
                )
            }
            reset_rows: list[bool] = []
            legal_list: list[np.ndarray] = []
            for lane in lanes:
                env = lane["env"]
                state = env.state
                if state is None:
                    raise RuntimeError("rollout lane lost its authoritative state")
                observation = env.observe_v2_for_viewer(0)
                player_state = state.players[0]
                hand = list(player_state.hand)
                elixir = float(player_state.elixir_milli) / 1000.0
                snapshot = lane["estimator"].snapshot()
                packed = _v4_step_inputs(observation, hand, elixir, snapshot)
                for key, value in packed.items():
                    if key == "legal_wait":
                        continue
                    batch_inputs[key].append(value)
                legal_list.append(packed["legal_play"])
                reset_rows.append(bool(lane.get("needs_reset", True)))
                lane["needs_reset"] = False
            array_inputs = {
                key: torch.as_tensor(np.stack(values)).unsqueeze(1).to(device)
                for key, values in batch_inputs.items()
                if key != "legal_play"
            }
            # Boolean arrays need explicit dtype routing.
            array_inputs["entity_mask"] = array_inputs["entity_mask"].to(torch.bool)
            array_inputs["opp_out_of_cycle"] = array_inputs["opp_out_of_cycle"].to(torch.bool)
            legal = torch.as_tensor(np.stack(legal_list)).unsqueeze(1).to(device)
            masks = masks_from_legal_play(legal)
            reset_mask = torch.tensor(reset_rows, dtype=torch.bool, device=device).reshape(-1, 1)
            logits, actions, log_probs, entropy, ht, next_hidden = policy.rollout_sample(
                array_inputs["raster"],
                array_inputs["global_features"],
                array_inputs["entities"],
                array_inputs["entity_mask"],
                array_inputs["hand_tokens"],
                array_inputs["opp_hand_probs"],
                array_inputs["opp_out_of_cycle"],
                array_inputs["opp_elixir_interval"],
                array_inputs["event_history"],
                masks,
                reset_mask=reset_mask,
                hidden=hidden,
            )
            if deterministic_policy:
                # Eval-fair comparison: argmax actions scored under the same
                # masks (always finite: argmax selections are legal).
                actions = policy.act_deterministic(logits, masks)
                log_probs = policy.log_prob(logits, masks, actions)
                entropy = policy.action_entropy(logits, masks)
            values = critic(ht)
            anchor_actions: V4ActionBatch | None = None
            if anchor is not None:
                if deterministic_anchor:
                    anchor_actions = anchor.act_deterministic(logits, masks)
                    # Anchor decode runs under no-grad; the learner scores
                    # these actions differentiably at update time.
                else:
                    sampled, _, _ = anchor.sample_action(logits, masks)
                    anchor_actions = sampled
            sim_actions = decode_v4_sim_actions(
                V4ActionBatch(
                    mode=actions.mode[:, 0].reshape(-1),
                    card_slot=actions.card_slot[:, 0].reshape(-1),
                    placement=actions.placement[:, 0].reshape(-1, 2),
                    wait_duration=actions.wait_duration[:, 0].reshape(-1),
                ),
                player=0,
            )
            rewards: list[float] = []
            terminated: list[bool] = []
            truncated: list[bool] = []
            for lane_index, lane in enumerate(lanes):
                env = lane["env"]
                opponent_action = lane["opponent"].choose_action(env.engine, env.state, 1)
                result = env.step_v2((sim_actions[lane_index], opponent_action))
                rewards.append(float(result.rewards[0]))
                all_rewards.append(float(result.rewards[0]))
                terminated.append(bool(result.terminated))
                truncated.append(bool(result.truncated))
                if bool(result.terminated):
                    winner = result.info.get("winner", None) if isinstance(result.info, dict) else None
                    if winner == 0:
                        episode_wins += 1
                        lane_outcomes[lane_index][0] += 1
                    elif winner == 1:
                        episode_losses += 1
                        lane_outcomes[lane_index][1] += 1
                    else:
                        episode_draws += 1
                        lane_outcomes[lane_index][2] += 1
                if bool(result.terminated) or bool(result.truncated):
                    seed = int(lane.get("next_seed", 0))
                    lane["env"].reset_v2(seed=seed, decks=lane["decks"], shuffle_decks=True)
                    lane["estimator"].reset()
                    lane["needs_reset"] = True
                    lane["next_seed"] = seed + 1
                if int(actions.mode[lane_index, 0]) == 0:
                    wait_count += 1
                else:
                    card_counts[int(actions.card_slot[lane_index, 0])] = (
                        card_counts.get(int(actions.card_slot[lane_index, 0]), 0) + 1
                    )
                # Defense in depth: decoded actions must honor legality even
                # though sampling already guarantees it.
                played = int(actions.mode[lane_index, 0]) == 1
                if played:
                    slot = int(actions.card_slot[lane_index, 0])
                    row = int(actions.placement[lane_index, 0, 0])
                    col = int(actions.placement[lane_index, 0, 1])
                    if not bool(legal[lane_index, 0, slot, row, col]):
                        illegal_attempts += 1
            hidden = next_hidden.detach()
            steps.append(
                {
                    "inputs": {k: v.detach().cpu() for k, v in array_inputs.items()},
                    "recurrent": ht.detach().cpu(),
                    "logits_mode": logits.mode.detach().cpu(),
                    "logits_duration": logits.wait_duration.detach().cpu(),
                    "logits_card": logits.card.detach().cpu(),
                    "logits_placement": logits.placement.detach().cpu(),
                    "masks_mode": masks.mode.detach().cpu(),
                    "masks_card": masks.card.detach().cpu(),
                    "masks_placement": masks.placement.detach().cpu(),
                    "actions_mode": actions.mode.detach().cpu(),
                    "actions_card": actions.card_slot.detach().cpu(),
                    "actions_placement": actions.placement.detach().cpu(),
                    "actions_duration": actions.wait_duration.detach().cpu(),
                    "anchor_mode": anchor_actions.mode.detach().cpu()
                    if anchor_actions is not None
                    else torch.zeros_like(actions.mode.detach().cpu()),
                    "anchor_card": anchor_actions.card_slot.detach().cpu()
                    if anchor_actions is not None
                    else torch.zeros_like(actions.card_slot.detach().cpu()),
                    "anchor_placement": anchor_actions.placement.detach().cpu()
                    if anchor_actions is not None
                    else torch.zeros_like(actions.placement.detach().cpu()),
                    "anchor_duration": anchor_actions.wait_duration.detach().cpu()
                    if anchor_actions is not None
                    else torch.zeros_like(actions.wait_duration.detach().cpu()),
                    "anchor_present": torch.ones(n_lanes, 1, dtype=torch.bool)
                    if anchor_actions is not None
                    else torch.zeros(n_lanes, 1, dtype=torch.bool),
                    "log_probs": log_probs.detach().cpu(),
                    "entropy": {k: v.detach().cpu() for k, v in entropy.items()},
                    "values": values.detach().cpu(),
                    "rewards": torch.tensor(rewards, dtype=torch.float32),
                    "terminated": torch.tensor(terminated, dtype=torch.bool),
                    "truncated": torch.tensor(truncated, dtype=torch.bool),
                    "reset": reset_mask.detach().cpu(),
                }
            )
    first_hidden = policy.initial_hidden(n_lanes, device=torch.device("cpu"))
    batch = _stack_rollout(steps, first_hidden, n_lanes, n_decisions, device)
    total_actions = n_lanes * n_decisions
    histogram = (
        {str(slot): count / total_actions for slot, count in sorted(card_counts.items())}
        if total_actions
        else {}
    )
    n_episodes = episode_wins + episode_losses + episode_draws
    return batch, {
        "wait_fraction": wait_count / total_actions if total_actions else 0.0,
        "card_histogram": histogram,
        "illegal_attempts": illegal_attempts,
        "median_reward": float(np.median(all_rewards)) if all_rewards else 0.0,
        "episode_wins": episode_wins,
        "episode_losses": episode_losses,
        "episode_draws": episode_draws,
        "episode_win_rate": episode_wins / n_episodes if n_episodes else 0.0,
        "lane_outcomes": lane_outcomes,
    }


def _stack_rollout(
    steps: list[dict[str, torch.Tensor]],
    first_hidden: torch.Tensor,
    n_lanes: int,
    n_decisions: int,
    device: Any,
) -> V4RolloutBatch:
    """Stack per-decision rows into trajectory + anchor containers.

    Per-step rows carry the singleton time axis used by the forward call;
    stacking squeezes it first so the batch is exactly
    ``[lanes, decisions, ...]``.  Rewards/flags are stored 1-D per step.
    """

    def stack(key: str) -> torch.Tensor:
        rows = [
            row[key].squeeze(1) if row[key].ndim > 1 else row[key]
            for row in steps
        ]
        return torch.stack(rows, dim=1).to(device)

    def stacked_input(key: str) -> torch.Tensor:
        return torch.stack(
            [row["inputs"][key].squeeze(1) for row in steps], dim=1
        ).to(device)

    sequence = RecurrentSequence(
        raster=stacked_input("raster"),
        global_features=stacked_input("global_features"),
        entities=stacked_input("entities"),
        entity_mask=stacked_input("entity_mask").to(torch.bool),
        reset_mask=torch.stack(
            [row["reset"].squeeze(1) for row in steps], dim=1
        ).to(device, dtype=torch.bool),
        # Smoke episodes reset explicitly and re-evaluate from the stored
        # initial state; per-step hidden snapshots and burn-in windows are
        # the documented follow-up, not smoke requirements.
        initial_hidden=first_hidden.to(device),
        hand_tokens=stacked_input("hand_tokens"),
        opp_hand_probs=stacked_input("opp_hand_probs"),
        opp_out_of_cycle=stacked_input("opp_out_of_cycle").to(torch.bool),
        opp_elixir_interval=stacked_input("opp_elixir_interval"),
        event_history=stacked_input("event_history"),
    )
    masks = ActionMasks(
        mode=stack("masks_mode"),
        card=stack("masks_card"),
        placement=stack("masks_placement"),
    )
    actions = V4ActionBatch(
        mode=stack("actions_mode"),
        card_slot=stack("actions_card"),
        placement=stack("actions_placement"),
        wait_duration=stack("actions_duration"),
    )
    anchor_present = stack("anchor_present")
    anchor_actions = None
    if bool(anchor_present.any().item()):
        anchor_actions = V4ActionBatch(
            mode=stack("anchor_mode"),
            card_slot=stack("anchor_card"),
            placement=stack("anchor_placement"),
            wait_duration=stack("anchor_duration"),
        )
    trajectory = TrajectoryBatch(
        sequence=sequence,
        action_masks=masks,
        actions=actions,
        rewards=stack("rewards"),
        terminated=stack("terminated").to(torch.bool),
        truncated=stack("truncated").to(torch.bool),
        old_log_probs=stack("log_probs"),
        values=stack("values"),
    )
    entropies = {
        key: torch.stack(
            [row["entropy"][key].squeeze(1) for row in steps], dim=1
        ).to(device)
        for key in ("joint", "mode", "duration", "card", "placement")
    }
    old_head_logits = {
        key: torch.stack(
            [row[f"logits_{key}"].squeeze(1) for row in steps], dim=1
        ).to(device)
        for key in ("mode", "duration", "card", "placement")
    }
    return V4RolloutBatch(
        trajectory=trajectory,
        anchor_actions=anchor_actions,
        recurrent=torch.stack(
            [row["recurrent"].squeeze(1) for row in steps], dim=1
        ).to(device),
        old_head_logits=old_head_logits,
        per_head_entropy=entropies,
    )


# ---------------------------------------------------------------------------
# PPO update + quarantine.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class V4UpdateReport:
    """Diagnostics for one PPO update (losses, movement, quarantine flags)."""

    policy_loss: float
    value_loss: float
    entropy: float
    per_head_entropy: dict[str, float]
    behavior_cloning_loss: float
    approx_kl: float
    mean_abs_log_ratio: float
    clip_fraction: float
    grad_norms: dict[str, float]
    explained_variance: float
    mode_flip_fraction: float
    card_flip_fraction: float
    placement_move_fraction: float
    duration_flip_fraction: float
    per_head_kl: dict[str, float]
    policy_grad_norm: float
    bc_grad_norm: float
    returns_mean: float
    returns_std: float
    advantages_mean: float
    advantages_std: float
    anchor_mode_agree: float | None
    anchor_card_agree: float | None
    anchor_place_within1: float | None
    quarantined: bool
    quarantine_reasons: list[str]


def update_v4_ppo(
    policy: RecurrentV4Policy,
    critic: V4ValueHead,
    actor_optimizer: Any,
    critic_optimizer: Any,
    batch: V4RolloutBatch,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ppo_config: PPOObjectiveConfig | None = None,
    bc_coef: float = 0.05,
    ppo_epochs: int = 1,
) -> V4UpdateReport:
    """Run PPO epochs over one collected rollout.

    Advantages come from :func:`compute_gae` with critic bootstrapping on
    nonterminal truncations.  The critic trains on detached actor features,
    so value updates never move actor parameters (isolation is asserted by
    tests).  Entropy uses the current policy (not stored values).  Returns
    the final epoch's diagnostics plus quarantine flags; the caller decides
    whether to keep the updated weights.
    """

    objective_config = ppo_config or PPOObjectiveConfig(bc_coef=bc_coef)
    if type(ppo_epochs) is not int or ppo_epochs < 1:
        raise ValueError("ppo_epochs must be a positive integer")
    trajectory = batch.trajectory
    sequence = trajectory.sequence
    device = sequence.raster.device
    policy.train()
    critic.train()
    if batch.recurrent is None:
        raise ValueError("rollout batch lacks stored recurrent features")
    recurrent = batch.recurrent.to(device)
    with torch.no_grad():
        # Bootstrap the final step on mid-rollout cuts only; terminated
        # episodes bootstrap zero.  Short-scenario horizons terminate
        # explicitly, so cuts are the only bootstrap path in the smoke.
        final_live = trajectory.truncated[:, -1] & ~trajectory.terminated[:, -1]
        bootstrap = critic(recurrent[:, -1:, :].detach()).squeeze(1)
        final_value = torch.where(
            final_live, bootstrap, torch.zeros_like(bootstrap)
        )
    stored_values = trajectory.values.to(device)
    next_values = torch.zeros_like(stored_values)
    if stored_values.shape[1] > 1:
        next_values[:, :-1] = stored_values[:, 1:]
    next_values[:, -1] = final_value
    advantages, returns = compute_gae(
        trajectory.rewards.to(device),
        trajectory.values.to(device),
        next_values,
        trajectory.terminated.to(device),
        trajectory.truncated.to(device),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    report: V4UpdateReport | None = None
    with torch.no_grad():
        pre_logits, _, _ = _forward_sequence(policy, sequence, device)
        pre_decoded = policy.act_deterministic(pre_logits, trajectory.action_masks)
    for _ in range(ppo_epochs):
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        logits, ht, _ = _forward_sequence(policy, sequence, device)
        new_log_probs = policy.log_prob(logits, trajectory.action_masks, trajectory.actions)
        values = critic(ht.detach())
        anchor_log_probs: torch.Tensor | None = None
        if batch.anchor_actions is not None and float(objective_config.bc_coef) > 0.0:
            anchor_log_probs = policy.log_prob(
                logits, trajectory.action_masks, batch.anchor_actions
            )
        new_entropy = policy.action_entropy(logits, trajectory.action_masks)["joint"]
        result = ppo_objective(
            old_log_probs=trajectory.old_log_probs.to(device),
            new_log_probs=new_log_probs,
            advantages=advantages,
            values=values,
            returns=returns,
            entropy=new_entropy,
            old_values=trajectory.values.to(device),
            behavior_cloning_log_probs=anchor_log_probs,
            config=objective_config,
        )
        result.total_loss.backward(retain_graph=True)
        grad_norms = _head_grad_norms(policy, critic)
        # Per-term actor gradient norms answer "PPO vs BC: who is actually
        # moving the policy?"  Separate autograd passes (graph retained
        # above); the stepping backward below reuses it a final time.
        actor_params = [p for p in policy.parameters() if p.requires_grad]
        policy_grad_norm = _term_grad_norm(result.policy_loss, actor_params)
        if anchor_log_probs is not None:
            bc_measured = behavior_cloning_loss(
                policy.log_prob(logits, trajectory.action_masks, batch.anchor_actions),
                torch.ones_like(anchor_log_probs),
            )
            bc_grad_norm = _term_grad_norm(bc_measured, actor_params)
        else:
            bc_grad_norm = 0.0
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        result.total_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
        with torch.no_grad():
            explained = _explained_variance(returns, values)
        per_head = {
            key: float(value.detach().mean().item())
            for key, value in (batch.per_head_entropy or {}).items()
        }
        post_logits, _, _ = _forward_sequence(policy, sequence, device)
        post_decoded = policy.act_deterministic(
            post_logits, trajectory.action_masks
        )
        drift = _decode_drift(pre_decoded, post_decoded)
        placement_ratio = grad_norms.get("placement", 0.0) / max(grad_norms.get("mode", 1e-8), 1e-8)
        reasons: list[str] = []
        if not bool(torch.isfinite(result.total_loss).item()):
            reasons.append("non-finite total loss")
        if float(result.approx_kl.item()) > 0.03:
            reasons.append(f"approx-kl {float(result.approx_kl.item()):.4f} above 0.03")
        if float(result.mean_abs_log_ratio.item()) > 0.05:
            reasons.append(
                f"mean-abs-log-ratio {float(result.mean_abs_log_ratio.item()):.4f} above 0.05"
            )
        if drift["mode_flip"] > 0.15:
            reasons.append(f"mode flip {drift['mode_flip']:.3f} above 0.15")
        if drift["card_flip"] > 0.25:
            reasons.append(f"card flip {drift['card_flip']:.3f} above 0.25")
        if drift["placement_move"] > 0.40:
            reasons.append(
                f"placement move {drift['placement_move']:.3f} above 0.40"
            )
        head_kl: dict[str, float] = {}
        if batch.old_head_logits is not None:
            with torch.no_grad():
                final_logits, _, _ = _forward_sequence(policy, sequence, device)
                for key, value in per_head_kl(
                    {k: v.to(device) for k, v in batch.old_head_logits.items()},
                    final_logits,
                    trajectory.action_masks,
                ).items():
                    head_kl[key] = float(value.detach().mean().item())
            for key, value in head_kl.items():
                if not np.isfinite(value):
                    reasons.append(f"non-finite per-head KL on {key}")
                    break
            placement_kl = head_kl.get("placement", 0.0)
            mode_kl = head_kl.get("mode", 0.0)
            if placement_kl > 20.0 * max(mode_kl, 1e-6) and placement_kl > 5e-3:
                reasons.append(
                    f"placement KL {placement_kl:.4f} dominates mode KL "
                    f"{mode_kl:.4f} (update-22 pattern)"
                )
        if placement_ratio > 10.0 and grad_norms.get("mode", 0.0) > 1e-6:
            # Landscape warning only (calibrated on the micro smoke): a
            # confident champion legitimately concentrates gradients where
            # uncertainty lives, and a near-zero mode gradient makes any
            # ratio explode mechanically.  It quarantines only alongside
            # movement, which is already covered above.
            per_head["placement_grad_warning"] = placement_ratio
        post_entropy = policy.action_entropy(logits, trajectory.action_masks)
        for key, value in post_entropy.items():
            pre = float(batch.per_head_entropy[key].detach().mean().item()) if batch.per_head_entropy is not None and key in batch.per_head_entropy else float("nan")
            post = float(value.detach().mean().item())
            if pre > 1e-6:
                collapse = 1.0 - post / pre
                per_head[f"{key}_collapse"] = collapse
                if key == "joint" and collapse > 0.50:
                    reasons.append(
                        f"joint entropy collapse {collapse:.2f} above 0.50"
                    )
        anchor_agree = (
            _anchor_agreement(post_decoded, batch.anchor_actions)
            if batch.anchor_actions is not None
            else {"mode": None, "card": None, "place": None}
        )
        report = V4UpdateReport(
            policy_loss=float(result.policy_loss.item()),
            value_loss=float(result.value_loss.item()),
            entropy=float(result.entropy.item()),
            per_head_entropy=per_head,
            behavior_cloning_loss=float(result.behavior_cloning_loss.item()),
            approx_kl=float(result.approx_kl.item()),
            mean_abs_log_ratio=float(result.mean_abs_log_ratio.item()),
            clip_fraction=float(result.clip_fraction.item()),
            grad_norms=grad_norms,
            explained_variance=explained,
            mode_flip_fraction=drift["mode_flip"],
            card_flip_fraction=drift["card_flip"],
            placement_move_fraction=drift["placement_move"],
            duration_flip_fraction=drift["duration_flip"],
            per_head_kl=head_kl,
            policy_grad_norm=policy_grad_norm,
            bc_grad_norm=bc_grad_norm,
            returns_mean=float(returns.detach().float().mean().item()),
            returns_std=float(returns.detach().float().std(unbiased=False).item()),
            advantages_mean=float(advantages.detach().float().mean().item()),
            advantages_std=float(advantages.detach().float().std(unbiased=False).item()),
            anchor_mode_agree=anchor_agree["mode"],
            anchor_card_agree=anchor_agree["card"],
            anchor_place_within1=anchor_agree["place"],
            quarantined=bool(reasons),
            quarantine_reasons=reasons,
        )
    assert report is not None
    return report


def _forward_sequence(
    policy: RecurrentV4Policy, sequence: RecurrentSequence, device: Any
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Re-evaluate stored V4 inputs (requires the retained V4 fields)."""

    for name in (
        "hand_tokens", "opp_hand_probs", "opp_out_of_cycle",
        "opp_elixir_interval", "event_history",
    ):
        if getattr(sequence, name) is None:
            raise ValueError(f"sequence lacks retained V4 field {name!r}")
    logits, zt, ht = policy(
        sequence.raster.to(device),
        sequence.global_features.to(device),
        sequence.entities.to(device),
        sequence.entity_mask.to(device),
        sequence.hand_tokens.to(device),
        sequence.opp_hand_probs.to(device),
        sequence.opp_out_of_cycle.to(device),
        sequence.opp_elixir_interval.to(device),
        sequence.event_history.to(device),
        sequence.reset_mask.to(device),
    )
    hidden = policy.initial_hidden(sequence.batch_size, device=device)
    _, next_hidden = policy.recurrent.forward_with_hidden(
        zt, sequence.reset_mask.to(device), hidden=hidden
    )
    return logits, ht, next_hidden


def _head_grad_norms(
    policy: RecurrentV4Policy, critic: V4ValueHead
) -> dict[str, float]:
    """Per-head gradient norms (the update-22 early-warning signal)."""

    def norm(modules: list[Any]) -> float:
        total = 0.0
        for module in modules:
            for param in module.parameters():
                if param.grad is not None:
                    total += float(param.grad.detach().float().norm().item()) ** 2
        return total**0.5

    heads = policy.heads
    return {
        "mode": norm([heads.mode_head]),
        "duration": norm([heads.duration_head]),
        "card": norm([heads.card_query, heads.card_key, heads.card_score]),
        "placement": norm([heads.spatial_key, heads.placement_query]),
        "encoder": norm([policy.encoder]),
        "recurrent": norm([policy.recurrent]),
        "critic": norm([critic]),
    }


def per_head_kl(
    old_logits: dict[str, torch.Tensor],
    new_logits: Any,
    masks: ActionMasks,
) -> dict[str, torch.Tensor]:
    """Exact per-head KL(old || new) on identical masks, ``[B, T]`` each.

    Mode/duration/card compare full distributions; placement averages the
    per-card map KL over slots with a legal cell (empty maps contribute
    zero — they can never be selected).  Identical masks on both sides is
    required; the smoke reuses the stored batch masks by construction.
    """

    from .model_v4 import _masked_log_softmax

    def kl(old_logp: torch.Tensor, new_logp: torch.Tensor) -> torch.Tensor:
        # Illegal entries hold -inf on both sides with zero mass; their
        # 0 * (-inf - -inf) NaN contributes nothing by definition of KL.
        old_prob = old_logp.exp()
        terms = old_prob * (old_logp - new_logp)
        return torch.where(
            torch.isfinite(terms), terms, torch.zeros_like(terms)
        ).sum(dim=-1)

    mode_old = _masked_log_softmax(old_logits["mode"], masks.mode)
    mode_new = _masked_log_softmax(new_logits.mode, masks.mode)
    out: dict[str, torch.Tensor] = {
        "mode": kl(mode_old, mode_new),
        "duration": kl(
            torch.log_softmax(old_logits["duration"], dim=-1),
            torch.log_softmax(new_logits.wait_duration, dim=-1),
        ),
    }
    # Broke WAIT rows may hold no legal card at all; those rows contribute
    # zero card KL (they can never select a card) instead of raising.
    card_ok = masks.card.any(dim=-1)
    card_term = torch.zeros_like(old_logits["card"][..., 0])
    if bool(card_ok.any().item()):
        card_term[card_ok] = kl(
            _masked_log_softmax(old_logits["card"][card_ok], masks.card[card_ok]),
            _masked_log_softmax(new_logits.card[card_ok], masks.card[card_ok]),
        )
    out["card"] = card_term
    rows, cols = new_logits.placement.shape[-2:]
    slots = masks.card.shape[-1]
    cells = rows * cols

    def selected(logits: torch.Tensor) -> torch.Tensor:
        flat = logits.reshape(logits.shape[:-3] + (slots, cells))
        mask = masks.placement.reshape(masks.placement.shape[:-3] + (slots, cells))
        safe = torch.where(
            mask.any(dim=-1, keepdim=True),
            torch.where(mask, flat, torch.full_like(flat, float("-inf"))),
            torch.zeros_like(flat),
        )
        return torch.log_softmax(safe, dim=-1)

    old_maps = selected(old_logits["placement"])
    new_maps = selected(new_logits.placement)
    per_cell = old_maps.exp() * (old_maps - new_maps)
    per_cell = torch.where(torch.isfinite(per_cell), per_cell, torch.zeros_like(per_cell))
    legal = masks.placement.reshape(masks.placement.shape[:-3] + (slots, cells)).any(dim=-1)
    base = (
        (per_cell.sum(dim=-1) * legal.to(per_cell.dtype)).sum(dim=-1)
        / legal.sum(dim=-1).clamp_min(1).to(per_cell.dtype)
    )
    out["placement"] = torch.where(
        legal.any(dim=-1),
        base,
        torch.zeros_like(base),
    )
    return out


def potential_anneal_weight(
    initial_weight: float, update: int, n_updates: int, *, anneal: bool = True
) -> float:
    """Potential-shaping weight for one smoke update (sparse backbone).

    Annealed linearly from ``initial_weight`` to exactly 0.0 at the final
    update so training ends on the sparse terminal objective.  With
    ``anneal=False`` the weight stays constant (ablation only).
    """

    if not np.isfinite(float(initial_weight)) or float(initial_weight) < 0.0:
        raise ValueError("initial_weight must be a finite non-negative number")
    if type(update) is not int or update < 0:
        raise ValueError("update must be a non-negative integer")
    if type(n_updates) is not int or n_updates <= 0:
        raise ValueError("n_updates must be a positive integer")
    if type(anneal) is not bool:
        raise TypeError("anneal must be boolean")
    if not anneal:
        return float(initial_weight)
    return max(0.0, float(initial_weight) * (1.0 - update / max(1, n_updates - 1)))


def _term_grad_norm(loss: torch.Tensor, params: list[Any]) -> float:
    """L2 gradient norm of one loss term over actor parameters (measured)."""

    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True
    )
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().float().norm().item()) ** 2
    return total**0.5


def _anchor_agreement(
    decoded: Any, anchor_actions: Any
) -> dict[str, float | None]:
    """Argmax agreement of the updated policy against frozen anchor actions.

    Mode agreement over all rows; card agreement over rows PLAY on both
    sides; placement within-1 over same-card PLAY rows.  ``None`` when the
    denominator is empty.
    """

    mode_a = decoded.mode.reshape(-1)
    mode_b = anchor_actions.mode.reshape(-1)
    n_rows = int(mode_a.numel())
    if n_rows == 0:
        return {"mode": None, "card": None, "place": None}
    mode_agree = float((mode_a == mode_b).float().mean().item())
    both_play = (mode_a == 1) & (mode_b == 1)
    if bool(both_play.any().item()):
        card_a = decoded.card_slot.reshape(-1)[both_play]
        card_b = anchor_actions.card_slot.reshape(-1)[both_play]
        card_agree: float | None = float((card_a == card_b).float().mean().item())
        same_card = card_a == card_b
        if bool(same_card.any().item()):
            place_a = decoded.placement.reshape(-1, 2)[both_play][same_card]
            place_b = anchor_actions.placement.reshape(-1, 2)[both_play][same_card]
            chebyshev = (place_a - place_b).abs().max(dim=-1).values
            place_within1: float | None = float((chebyshev <= 1).float().mean().item())
        else:
            place_within1 = None
    else:
        card_agree = None
        place_within1 = None
    return {"mode": mode_agree, "card": card_agree, "place": place_within1}


def _decode_drift(first: Any, second: Any) -> dict[str, float]:
    """Behavioral movement between two deterministic decodes.

    Fractions over rows: mode flips; card flips (rows PLAY on both sides);
    placement Chebyshev moves >1 (rows PLAY on both sides with the same
    card, so pure placement drift is isolated from card switches);
    duration flips (rows WAIT on both sides, the only rows where the
    duration head executes).
    """

    import torch

    first_mode = first.mode.reshape(-1)
    second_mode = second.mode.reshape(-1)
    n_rows = int(first_mode.numel())
    if n_rows == 0:
        return {"mode_flip": 0.0, "card_flip": 0.0, "placement_move": 0.0, "duration_flip": 0.0}
    mode_flip = float((first_mode != second_mode).float().mean().item())
    both_play = (first_mode == 1) & (second_mode == 1)
    if bool(both_play.any().item()):
        first_card = first.card_slot.reshape(-1)[both_play]
        second_card = second.card_slot.reshape(-1)[both_play]
        card_flip = float((first_card != second_card).float().mean().item())
        same_card = first_card == second_card
        if bool(same_card.any().item()):
            first_place = first.placement.reshape(-1, 2)[both_play][same_card]
            second_place = second.placement.reshape(-1, 2)[both_play][same_card]
            chebyshev = (first_place - second_place).abs().max(dim=-1).values
            placement_move = float((chebyshev > 1).float().mean().item())
        else:
            placement_move = 0.0
    else:
        card_flip = 0.0
        placement_move = 0.0
    both_wait = (first_mode == 0) & (second_mode == 0)
    if bool(both_wait.any().item()):
        duration_flip = float(
            (first.wait_duration.reshape(-1)[both_wait]
             != second.wait_duration.reshape(-1)[both_wait])
            .float()
            .mean()
            .item()
        )
    else:
        duration_flip = 0.0
    return {
        "mode_flip": mode_flip,
        "card_flip": card_flip,
        "placement_move": placement_move,
        "duration_flip": duration_flip,
    }


def _explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    """Fraction of return variance explained by the critic."""

    returns = returns.detach().float()
    values = values.detach().float()
    variance = returns.var(unbiased=False)
    if not bool(torch.isfinite(variance).item()) or float(variance.item()) <= 0.0:
        return 0.0
    residual = (returns - values).var(unbiased=False)
    return max(0.0, 1.0 - float(residual.item()) / float(variance.item()))


def monte_carlo_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    *,
    gamma: float = 0.99,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rewards-to-go with zero bootstrap at rollout cuts.

    Terminated episodes yield exact returns.  Trailing nonterminal tails
    bootstrap zero, which biases them toward zero; the returned
    ``truncated_tail_fraction`` quantifies how much of the data is
    affected.  If it dominates, the honest fix is longer horizons, not a
    bigger model.
    """

    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    rewards = rewards.detach().float()
    terminated = terminated.detach().to(torch.bool)
    batch, time = rewards.shape
    out = torch.zeros_like(rewards)
    running = torch.zeros(batch, dtype=rewards.dtype, device=rewards.device)
    for step in range(time - 1, -1, -1):
        running = rewards[:, step] + float(gamma) * running * (~terminated[:, step]).to(
            rewards.dtype
        )
        out[:, step] = running
        running = torch.where(terminated[:, step], torch.zeros_like(running), running)
    # Steps biased by the zero bootstrap: nonterminal-final lanes, steps
    # strictly before the cut edge and after the lane's last termination.
    biased = 0
    for lane in range(batch):
        terms = torch.nonzero(terminated[lane], as_tuple=False).reshape(-1)
        last_term = int(terms.max().item()) if int(terms.numel()) > 0 else -1
        if not bool(terminated[lane, -1].item()):
            biased += max(0, (time - 1) - max(last_term + 1, 0))
    stats = {
        "truncated_tail_fraction": biased / max(1, batch * time),
    }
    return out, stats


def return_distribution(values: torch.Tensor) -> dict[str, float]:
    """Quantile/sparsity summary of a return (or reward) tensor."""

    flat = values.detach().float().reshape(-1)
    if int(flat.numel()) == 0:
        raise ValueError("empty tensor has no distribution")
    quantiles = torch.quantile(
        flat, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=flat.dtype)
    ).tolist()
    near_zero = float((flat.abs() < 1e-6).float().mean().item())
    return {
        "n": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "near_zero_fraction": near_zero,
        "positive_fraction": float((flat > 1e-6).float().mean().item()),
        "negative_fraction": float((flat < -1e-6).float().mean().item()),
    }


def fit_value_head(
    critic: V4ValueHead,
    recurrent: torch.Tensor,
    targets: torch.Tensor,
    *,
    heldout_recurrent: torch.Tensor | None = None,
    heldout_targets: torch.Tensor | None = None,
    lr: float = 1e-3,
    epochs: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    """Supervised critic warmup on fixed Monte-Carlo targets (actor frozen).

    Only the value head trains, on caller-provided recurrent features, so
    this cannot move the actor by construction.  Tracks train and held-out
    loss plus explained variance every epoch and returns the best-held-out
    state (in-memory copy) for the caller to persist.
    """

    generator = torch.Generator().manual_seed(int(seed))
    try:
        device = next(critic.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    recurrent = recurrent.detach().float().to(device)
    targets = targets.detach().float().to(device)
    if recurrent.shape[:2] != targets.shape[:2]:
        raise ValueError("recurrent features and targets must share batch/time")
    critic.train()
    optimizer = torch.optim.Adam(critic.parameters(), lr=float(lr))
    flat_x = recurrent.reshape(-1, recurrent.shape[-1])
    flat_y = targets.reshape(-1)
    if heldout_recurrent is not None and heldout_targets is not None:
        held_x = heldout_recurrent.detach().float().to(device).reshape(-1, recurrent.shape[-1])
        held_y = heldout_targets.detach().float().to(device).reshape(-1)
    else:
        held_x, held_y = None, None
    best = {"heldout_loss": float("inf"), "epoch": -1, "state": None}
    history: list[dict[str, float]] = []
    order_base = torch.randperm(flat_x.shape[0], generator=generator)
    for epoch in range(int(epochs)):
        order = order_base[torch.randperm(flat_x.shape[0], generator=generator)]
        for start in range(0, int(flat_x.shape[0]), 256):
            chunk = order[start : start + 256]
            optimizer.zero_grad(set_to_none=True)
            loss = (critic(flat_x[chunk].unsqueeze(1)).squeeze(1) - flat_y[chunk]).square().mean()
            loss.backward()
            optimizer.step()
        critic.eval()
        with torch.no_grad():
            train_pred = critic(flat_x.unsqueeze(1)).squeeze(1)
            train_loss = float(((train_pred - flat_y).square()).mean().item())
            row: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_ev": _explained_variance(flat_y, train_pred.reshape_as(flat_y)),
            }
            if held_x is not None:
                held_pred = critic(held_x.unsqueeze(1)).squeeze(1)
                held_loss = float(((held_pred - held_y).square()).mean().item())
                row["heldout_loss"] = held_loss
                row["heldout_ev"] = _explained_variance(held_y, held_pred.reshape_as(held_y))
                if held_loss < best["heldout_loss"]:
                    best = {
                        "heldout_loss": held_loss,
                        "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()},
                    }
        critic.train()
        history.append(row)
    if best["state"] is not None:
        critic.load_state_dict(best["state"])
    return {"history": history, "best": best}


__all__ = [
    "SMOKE_OPPONENT_LANES",
    "V4OpponentLane",
    "V4PPOConfig",
    "V4RolloutBatch",
    "V4UpdateReport",
    "collect_v4_rollout",
    "decode_v4_sim_actions",
    "fit_value_head",
    "load_frozen_anchor",
    "monte_carlo_returns",
    "per_head_kl",
    "potential_anneal_weight",
    "return_distribution",
    "update_v4_ppo",
]
