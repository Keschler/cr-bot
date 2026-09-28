"""Deterministic from-scratch V4 bootstrap data from real simulator matches.

This is the only supervised bootstrap path retained by V4.  It runs ordinary
simulator environments, records the public actor observation, and applies the
small rules teacher used by frontier auditing.  There is no synthetic feature
generator and no search or timing curriculum hidden behind this API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Sequence

import numpy as np

try:
    from ...actions import PlayCardAction, WaitAction
    from ...engine import BattleEngine
    from ...env import SimulatorEnv
    from ...public_state_estimator import PublicStateEstimator
    from ...roster import PLAYER_DECK
    from ...ruleset import load_fixed_ruleset
    from ..foundation.opponent_pool import OpponentPool
    from .hand_features import build_hand_tokens
    from .simulator_teacher import (
        TeacherError,
        TeacherTarget,
        soft_placement_target,
        teacher_label,
    )
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.actions import PlayCardAction, WaitAction
    from simulator.engine import BattleEngine
    from simulator.env import SimulatorEnv
    from simulator.public_state_estimator import PublicStateEstimator
    from simulator.roster import PLAYER_DECK
    from simulator.ruleset import load_fixed_ruleset
    from simulator.rl.foundation.opponent_pool import OpponentPool
    from simulator.rl.v4.hand_features import build_hand_tokens
    from simulator.rl.v4.simulator_teacher import (
        TeacherError,
        TeacherTarget,
        soft_placement_target,
        teacher_label,
    )


BOOTSTRAP_VERSION = "v4-bootstrap-sim-v2"
BOOTSTRAP_ARCHETYPES: tuple[str, ...] = (
    "deterministic-cycle",
    "aggressive-pressure",
    "defensive-cycle",
    "beatdown",
    "air-beatdown",
    "siege-bait",
)


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Settings for deterministic public-state collection."""

    n_states: int
    seed: int = 0
    max_decisions: int = 1440
    samples_per_episode: int = 64
    archetypes: tuple[str, ...] = BOOTSTRAP_ARCHETYPES

    def __post_init__(self) -> None:
        if type(self.n_states) is not int or self.n_states <= 0:
            raise ValueError("n_states must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.max_decisions) is not int or self.max_decisions <= 0:
            raise ValueError("max_decisions must be a positive integer")
        if type(self.samples_per_episode) is not int or self.samples_per_episode < 2:
            raise ValueError("samples_per_episode must be an integer of at least two")
        if not self.archetypes or any(
            not isinstance(value, str) or not value for value in self.archetypes
        ):
            raise ValueError("archetypes must contain non-empty strings")
        if len(set(self.archetypes)) != len(self.archetypes):
            raise ValueError("archetypes must not contain duplicates")


@dataclass(slots=True)
class BootstrapSample:
    """One public simulator observation with a rules-teacher target."""

    family: str
    raster: np.ndarray
    global_features: np.ndarray
    entity_tokens: np.ndarray
    entity_mask: np.ndarray
    hand_tokens: np.ndarray
    opp_hand_probs: np.ndarray
    opp_out_of_cycle: np.ndarray
    opp_elixir_interval: np.ndarray
    event_history: np.ndarray
    legal_play: np.ndarray
    legal_wait: bool
    target: TeacherTarget
    soft_placement: np.ndarray
    provenance: dict[str, object]


def _target_action(target: TeacherTarget, *, player: int) -> Any:
    if target.mode == 0:
        return WaitAction(player)
    return PlayCardAction(player, target.card_slot, (target.col, target.row))


def _fallback_action(
    legal_play: np.ndarray,
    legal_wait: bool,
    *,
    player: int,
    allow_countdown_wait: bool = False,
) -> Any:
    if legal_wait:
        return WaitAction(player)
    legal = np.argwhere(np.asarray(legal_play, dtype=bool))
    if len(legal):
        slot, row, col = (int(value) for value in legal[0])
        return PlayCardAction(player, slot, (col, row))
    if allow_countdown_wait:
        # Countdown WAIT is a physical simulator action used only to advance
        # to the active phase; it is never emitted as a bootstrap label.
        return WaitAction(player)
    raise RuntimeError("active bootstrap state has no legal action")


def _capture_sample(
    env: SimulatorEnv,
    estimator: PublicStateEstimator,
    *,
    archetype: str,
    episode: int,
    decision: int,
    seed: int,
) -> tuple[BootstrapSample, Any] | None:
    state = env.state
    if state is None:
        raise RuntimeError("bootstrap environment has no state")
    estimator.sync_from_env(env, viewer=0)
    observation = env.observe_v2_for_viewer(0)
    player = state.players[0]
    hand = tuple(player.hand[:4])
    if len(hand) != 4:
        return None
    elixir = float(player.elixir_milli) / 1000.0
    try:
        hand_tokens = build_hand_tokens(hand, elixir)
        legal_play = np.asarray(observation.legal_play, dtype=bool)
        legal_wait = bool(observation.legal_wait)
        target = teacher_label(
            hand_tokens=hand_tokens,
            entity_tokens=np.asarray(observation.entity_tokens),
            entity_mask=np.asarray(observation.entity_mask),
            legal_play=legal_play,
            legal_wait=legal_wait,
            own_elixir=elixir,
        )
    except TeacherError:
        if bool(observation.legal_wait):
            raise
        # Before the active phase the observation contract intentionally
        # exposes no WAIT action; the caller advances that short prelude with
        # a physical WAIT and does not record it as training data.
        return None
    if target.mode == 1:
        soft = soft_placement_target(
            target.row, target.col, np.asarray(legal_play[target.card_slot])
        )
    else:
        soft = np.zeros(legal_play.shape[1:], dtype=np.float32)
    snapshot = estimator.snapshot()
    sample = BootstrapSample(
        family=archetype,
        raster=np.array(observation.board, dtype=np.float32, copy=True),
        global_features=np.array(observation.global_vector, dtype=np.float32, copy=True),
        entity_tokens=np.array(observation.entity_tokens, dtype=np.float32, copy=True),
        entity_mask=np.array(observation.entity_mask, dtype=bool, copy=True),
        hand_tokens=np.array(hand_tokens, dtype=np.float32, copy=True),
        opp_hand_probs=np.array(snapshot.opp_hand_probs, dtype=np.float32, copy=True),
        opp_out_of_cycle=np.array(snapshot.opp_out_of_cycle, dtype=bool, copy=True),
        opp_elixir_interval=np.array(
            [snapshot.opp_elixir_lo, snapshot.opp_elixir_hi], dtype=np.float32, copy=True
        ),
        event_history=np.array(snapshot.event_history, dtype=np.float32, copy=True),
        legal_play=np.array(legal_play, copy=True),
        legal_wait=legal_wait,
        target=target,
        soft_placement=np.asarray(soft, dtype=np.float32),
        provenance={
            "generator": BOOTSTRAP_VERSION,
            "seed": seed,
            "episode": episode,
            "decision": decision,
            "elapsed_us": state.elapsed_us,
            "phase": state.phase,
            "archetype": archetype,
            "state_hash": state.state_hash(),
        },
    )
    return sample, _target_action(target, player=0)


def generate_bootstrap_dataset(config: BootstrapConfig) -> list[BootstrapSample]:
    """Sample whole trajectories with balanced modes and bounded match influence.

    Independent per-mode reservoirs sample uniformly over each trajectory.
    Quotas control retained records, never how early a trajectory stops.
    """

    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=config.seed)
    samples: list[BootstrapSample] = []
    quotas = {
        archetype: config.n_states // len(config.archetypes)
        + (index < config.n_states % len(config.archetypes))
        for index, archetype in enumerate(config.archetypes)
    }
    mode_cap = config.samples_per_episode // 2
    remaining = {
        archetype: {
            1: (quota + (index % 2 == 0)) // 2,
            0: quota - (quota + (index % 2 == 0)) // 2,
        }
        for index, (archetype, quota) in enumerate(quotas.items())
    }
    episode_limits = {
        archetype: max(8, 8 * ((max(targets.values()) + mode_cap - 1) // mode_cap))
        for archetype, targets in remaining.items()
    }
    family_episodes = dict.fromkeys(config.archetypes, 0)
    episode = 0
    while len(samples) < config.n_states:
        archetype = config.archetypes[episode % len(config.archetypes)]
        if not any(remaining[archetype].values()):
            episode += 1
            continue
        if family_episodes[archetype] >= episode_limits[archetype]:
            raise RuntimeError(
                f"bootstrap could not fill PLAY/WAIT quotas for {archetype} "
                f"after {family_episodes[archetype]} episodes: "
                f"missing {remaining[archetype]}; increase max_decisions or "
                "inspect teacher action coverage"
            )
        family_episodes[archetype] += 1
        capacities = {mode: min(mode_cap, count) for mode, count in remaining[archetype].items()}
        reservoirs: dict[int, list[BootstrapSample]] = {0: [], 1: []}
        seen = {0: 0, 1: 0}
        rng = random.Random(_stable_seed(config.seed, "bootstrap-reservoir", episode))
        episode_seed = _stable_seed(config.seed, "bootstrap-episode", episode)
        scenario = pool.sample(
            episode,
            archetype=archetype,
            strategy="deterministic-cycle",
            allow_variants=False,
        )
        env = SimulatorEnv(
            BattleEngine(ruleset, validate_every_tick=False),
            decision_interval_us=250_000,
        )
        decks = (tuple(PLAYER_DECK), tuple(scenario.deck.cards))
        env.reset_v2(seed=episode_seed, decks=decks, shuffle_decks=True)
        estimator = PublicStateEstimator()
        opponent = scenario.build_controller()
        for decision in range(config.max_decisions):
            if env.state is None or env.state.terminal:
                break
            captured = _capture_sample(
                env,
                estimator,
                archetype=archetype,
                episode=episode,
                decision=decision,
                seed=config.seed,
            )
            if captured is None:
                observation = env.observe_v2_for_viewer(0)
                learner_action = _fallback_action(
                    np.asarray(observation.legal_play, dtype=bool),
                    bool(observation.legal_wait),
                    player=0,
                    allow_countdown_wait=(env.state.phase == "countdown"),
                )
            else:
                sample, learner_action = captured
                mode = sample.target.mode
                seen[mode] += 1
                bucket = reservoirs[mode]
                if len(bucket) < capacities[mode]:
                    bucket.append(sample)
                elif capacities[mode]:
                    replacement = rng.randrange(seen[mode])
                    if replacement < capacities[mode]:
                        bucket[replacement] = sample
            opponent_action = opponent.choose_action(env.engine, env.state, 1)
            env.step_v2((learner_action, opponent_action))
        retained = reservoirs[0] + reservoirs[1]
        retained.sort(key=lambda sample: int(sample.provenance["decision"]))
        samples.extend(retained)
        for mode in (0, 1):
            remaining[archetype][mode] -= len(reservoirs[mode])
        episode += 1
    return samples


def to_torch_batch(samples: Sequence[BootstrapSample]) -> dict[str, object]:
    """Convert samples to one-step V4 actor tensors."""

    if not samples:
        raise ValueError("cannot batch an empty bootstrap sample sequence")
    import torch

    from .model_v4 import V4ActionBatch, masks_from_legal_play

    def stack(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(getattr(sample, name), dtype=dtype) for sample in samples]
        ).unsqueeze(1)

    legal_play = torch.stack(
        [torch.as_tensor(sample.legal_play, dtype=torch.bool) for sample in samples]
    ).unsqueeze(1)
    mode = torch.tensor(
        [sample.target.mode for sample in samples], dtype=torch.long
    ).reshape(-1, 1)
    masks = masks_from_legal_play(
        legal_play,
        torch.tensor([sample.legal_wait for sample in samples], dtype=torch.bool).reshape(-1, 1),
    )
    actions = V4ActionBatch(
        mode=mode,
        card_slot=torch.tensor(
            [sample.target.card_slot for sample in samples], dtype=torch.long
        ).reshape(-1, 1),
        placement=torch.tensor(
            [[sample.target.row, sample.target.col] for sample in samples], dtype=torch.long
        ).reshape(-1, 1, 2),
        wait_duration=torch.tensor(
            [sample.target.wait_duration_idx for sample in samples], dtype=torch.long
        ).reshape(-1, 1),
    )
    return {
        "raster": stack("raster", torch.float32),
        "global_features": stack("global_features", torch.float32),
        "entities": stack("entity_tokens", torch.float32),
        "entity_mask": stack("entity_mask", torch.bool),
        "hand_tokens": stack("hand_tokens", torch.float32),
        "opp_hand_probs": stack("opp_hand_probs", torch.float32),
        "opp_out_of_cycle": stack("opp_out_of_cycle", torch.bool),
        "opp_elixir_interval": stack("opp_elixir_interval", torch.float32),
        "event_history": stack("event_history", torch.float32),
        "legal_play": legal_play,
        "masks": masks,
        "actions": actions,
        "soft_placement": torch.stack(
            [torch.as_tensor(sample.soft_placement, dtype=torch.float32) for sample in samples]
        ).unsqueeze(1),
        "reset_mask": torch.ones(len(samples), 1, dtype=torch.bool),
        "families": [sample.family for sample in samples],
    }


def move_batch_to_device(batch: dict[str, object], device: object) -> dict[str, object]:
    """Move tensor and validated dataclass fields to ``device``."""

    import dataclasses
    import torch

    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            moved[key] = dataclasses.replace(
                value,
                **{
                    field.name: (
                        getattr(value, field.name).to(device)
                        if isinstance(getattr(value, field.name), torch.Tensor)
                        else getattr(value, field.name)
                    )
                    for field in dataclasses.fields(value)
                },
            )
        else:
            moved[key] = value
    return moved


__all__ = [
    "BOOTSTRAP_ARCHETYPES",
    "BOOTSTRAP_VERSION",
    "BootstrapConfig",
    "BootstrapSample",
    "generate_bootstrap_dataset",
    "move_batch_to_device",
    "to_torch_batch",
]
