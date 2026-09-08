"""Simulator-grounded distillation state generator (paper Stage 1).

V4 bootstraps from *simulator-generated* supervised states rather than
unverified expert video.  This module is the v0 state source: it synthesizes
stratified tactical states across the eight tactical families with
deterministic seeds, public-only features, exact legality, and provenance
on every sample.  The teacher (``simulator_teacher``) labels each state.

The raster channels 0-2 carry presence/threat splats so the spatial
placement decoder has learnable geometry; all other channels are zero.
A future revision swaps this generator for short-horizon
``BasicMechanicsScenarioEnv`` states without changing the sample schema:
consumers only see ``DistillationSample`` + ``to_torch_batch``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

try:
    from ..observation_v3 import BELIEF_CARD_COUNT, EVENT_DIM, EVENT_HISTORY_LEN
    from .simulator_teacher import (
        TeacherTarget,
        soft_placement_target,
        teacher_label,
    )
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.observation_v3 import BELIEF_CARD_COUNT, EVENT_DIM, EVENT_HISTORY_LEN
    from simulator.rl.simulator_teacher import (
        TeacherTarget,
        soft_placement_target,
        teacher_label,
    )


GENERATOR_VERSION: str = "synthetic-v4-distill-0"
TEACHER_VERSION: str = "rules-v4-0"

TACTICAL_FAMILIES: tuple[str, ...] = (
    "offense",
    "ground-defense",
    "air-defense",
    "spell-value",
    "kiting-cycle",
    "low-elixir",
    "counterpush",
    "bridge-defense",
)

_DEFAULT_MIX: dict[str, float] = {
    "offense": 0.15,
    "ground-defense": 0.20,
    "air-defense": 0.15,
    "spell-value": 0.15,
    "kiting-cycle": 0.10,
    "low-elixir": 0.10,
    "counterpush": 0.05,
    "bridge-defense": 0.10,
}

# Fixed 2.6 Hog player deck: (card key, legacy policy id, elixir cost,
# win_condition, spell, building, swarm, air_attacker).
PLAYER_DECK: tuple[tuple[str, int, int, bool, bool, bool, bool, bool], ...] = (
    ("hog-rider", 49, 4, True, False, False, False, False),
    ("cannon", 114, 3, False, False, True, False, False),
    ("musketeer", 72, 4, False, False, False, False, True),
    ("skeletons", 96, 1, False, False, False, True, False),
    ("ice-golem", 51, 2, False, False, False, False, False),
    ("ice-spirit", 52, 1, False, False, False, False, False),
    ("fireball", 28, 4, False, True, False, False, False),
    ("log", 59, 2, False, True, False, False, False),
)


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass
class DistillationSample:
    """One supervised state: public features, teacher target, provenance."""

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
    own_elixir: float
    target: TeacherTarget
    soft_placement: np.ndarray
    provenance: dict[str, object]


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    """How many states to generate and under which generator identity."""

    n_states: int
    seed: int
    family_mix: dict[str, float] | None = None
    rows: int = 32
    cols: int = 18
    raster_channels: int = 21
    global_dim: int = 768
    entity_dim: int = 32
    max_entities: int = 128

    def __post_init__(self) -> None:
        if type(self.n_states) is not int or self.n_states <= 0:
            raise ValueError("n_states must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        mix = dict(_DEFAULT_MIX) if self.family_mix is None else dict(self.family_mix)
        if set(mix) != set(TACTICAL_FAMILIES):
            raise ValueError(f"family_mix must cover exactly {TACTICAL_FAMILIES}")
        if any(not isinstance(w, float) or w <= 0.0 for w in mix.values()):
            raise ValueError("family_mix weights must be positive floats")
        object.__setattr__(self, "family_mix", mix)


def _family_for_index(config: DistillationConfig, index: int) -> str:
    draw = (_stable_seed(config.seed, "family", index) % 1000) / 1000.0
    cumulative = 0.0
    total = sum(config.family_mix.values())
    for family in TACTICAL_FAMILIES:
        cumulative += config.family_mix[family] / total
        if draw < cumulative:
            return family
    return TACTICAL_FAMILIES[-1]


def _sample_hand(rng: np.random.Generator, elixir: float) -> tuple[np.ndarray, list[int]]:
    order = rng.permutation(len(PLAYER_DECK))[:4]
    hand = np.zeros((4, 16), dtype=np.float32)
    for slot, deck_idx in enumerate(order):
        _, policy_id, cost, win, spell, building, swarm, air = PLAYER_DECK[int(deck_idx)]
        affordable = cost <= elixir
        hand[slot, 0] = policy_id / 127.0
        hand[slot, 1] = cost / 10.0
        hand[slot, 2] = slot / 3.0
        hand[slot, 3] = 1.0 if affordable else 0.0
        hand[slot, 4] = 1.0 if affordable else 0.0
        hand[slot, 5] = 1.0 if win else 0.0
        hand[slot, 6] = 1.0 if spell else 0.0
        hand[slot, 7] = 1.0 if building else 0.0
        hand[slot, 8] = 1.0 if swarm else 0.0
        hand[slot, 9] = 1.0 if air else 0.0
        hand[slot, 10] = float(deck_idx) / 7.0
    return hand, [int(i) for i in order]


def _legal_for_hand(
    hand: np.ndarray, rows: int, cols: int
) -> tuple[np.ndarray, list[bool]]:
    legal = np.zeros((4, rows, cols), dtype=bool)
    is_spell = hand[:, 6] > 0.5
    affordable = hand[:, 3] > 0.5
    for slot in range(4):
        if not affordable[slot]:
            continue
        if is_spell[slot]:
            legal[slot] = True
        else:
            legal[slot, rows // 2 + 1 :, :] = True
    return legal, [bool(affordable[s]) for s in range(4)]


def _entity_row(
    rng: np.random.Generator,
    *,
    side: int,
    x: float,
    y: float,
    hp: float,
    is_air: bool = False,
    is_building: bool = False,
) -> np.ndarray:
    row = np.zeros((32,), dtype=np.float32)
    row[0] = float(rng.integers(0, 125)) / 124.0
    row[1] = float(side)
    row[2] = float(x)
    row[3] = float(y)
    row[4] = float(hp)
    row[5] = 1.0 if is_air else 0.0
    row[6] = 1.0 if is_building else 0.0
    row[9] = 1.0
    row[10] = 1.0
    row[11] = 0.0 if x < 0.5 else 1.0
    row[19] = 1.0
    row[28] = 1.0
    row[29] = float(rng.uniform(0.0, 5.0))
    return row


def _sample_entities(
    rng: np.random.Generator, family: str, config: DistillationConfig
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((config.max_entities, config.entity_dim), dtype=np.float32)
    mask = np.zeros((config.max_entities,), dtype=bool)

    def add(side: int, x: float, y: float, hp: float, **kwargs) -> None:
        idx = int(mask.sum())
        if idx >= config.max_entities:
            return
        tokens[idx] = _entity_row(rng, side=side, x=x, y=y, hp=hp, **kwargs)
        mask[idx] = True

    if family == "offense":
        for _ in range(int(rng.integers(1, 3))):
            add(0, float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.5)), 0.8)
        if bool(rng.integers(0, 2)):
            add(1, float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.25, 0.45)), 0.6)
    elif family == "ground-defense":
        add(1, float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.55, 0.7)), 1.0)
        for _ in range(int(rng.integers(1, 3))):
            add(1, float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.55, 0.85)), 0.5)
    elif family == "air-defense":
        for _ in range(int(rng.integers(1, 3))):
            add(
                1,
                float(rng.uniform(0.2, 0.8)),
                float(rng.uniform(0.5, 0.75)),
                0.6,
                is_air=True,
            )
    elif family == "spell-value":
        cx, cy = float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.15, 0.4))
        for _ in range(int(rng.integers(3, 6))):
            add(
                1,
                min(max(cx + float(rng.uniform(-0.04, 0.04)), 0.0), 1.0),
                min(max(cy + float(rng.uniform(-0.04, 0.04)), 0.0), 1.0),
                0.5,
            )
    elif family == "kiting-cycle":
        add(1, float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.4, 0.55)), 0.7)
        add(0, float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.6, 0.8)), 0.4)
    elif family == "low-elixir":
        add(1, float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.45, 0.6)), 0.7)
    elif family == "counterpush":
        add(0, float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.3, 0.5)), 0.3)
        if bool(rng.integers(0, 2)):
            add(1, float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.6, 0.8)), 0.4)
    elif family == "bridge-defense":
        for _ in range(int(rng.integers(1, 3))):
            add(1, float(rng.uniform(0.25, 0.75)), float(rng.uniform(0.46, 0.54)), 0.8)
    return tokens, mask


def _elixir_for_family(rng: np.random.Generator, family: str) -> float:
    if family == "low-elixir":
        return float(rng.uniform(0.0, 3.0))
    if family in ("offense", "counterpush"):
        return float(rng.uniform(5.0, 10.0))
    return float(rng.uniform(3.0, 10.0))


def _raster_from_entities(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    rows: int,
    cols: int,
    channels: int,
) -> np.ndarray:
    raster = np.zeros((channels, rows, cols), dtype=np.float32)
    for row in tokens[mask]:
        side, x, y, hp = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        cell = (min(int(y * rows), rows - 1), min(int(x * cols), cols - 1))
        raster[1 if side < 0.5 else 0, cell[0], cell[1]] = 1.0
        if side > 0.5 and y > 0.45:
            raster[2, cell[0], cell[1]] = max(raster[2, cell[0], cell[1]], np.float32(hp))
    return raster


def generate_sample(
    config: DistillationConfig, index: int
) -> DistillationSample:
    """Generate one deterministic supervised state (order-independent)."""

    family = _family_for_index(config, index)
    rng = np.random.default_rng(_stable_seed(config.seed, family, index))
    elixir = _elixir_for_family(rng, family)
    hand, _ = _sample_hand(rng, elixir)
    legal_play, _ = _legal_for_hand(hand, config.rows, config.cols)
    entity_tokens, entity_mask = _sample_entities(rng, family, config)
    target = teacher_label(
        hand_tokens=hand,
        entity_tokens=entity_tokens,
        entity_mask=entity_mask,
        legal_play=legal_play,
        legal_wait=True,
        own_elixir=elixir,
    )
    if target.mode == 1:
        soft = soft_placement_target(
            target.row, target.col, np.asarray(legal_play[target.card_slot])
        )
    else:
        soft = np.zeros((config.rows, config.cols), dtype=np.float32)
    probs = np.full((BELIEF_CARD_COUNT,), 1.0 / BELIEF_CARD_COUNT, dtype=np.float32)
    out = np.zeros((BELIEF_CARD_COUNT,), dtype=bool)
    glob = np.zeros((config.global_dim,), dtype=np.float32)
    glob[0] = np.float32(elixir / 10.0)
    return DistillationSample(
        family=family,
        raster=_raster_from_entities(
            entity_tokens,
            entity_mask,
            rows=config.rows,
            cols=config.cols,
            channels=config.raster_channels,
        ),
        global_features=glob,
        entity_tokens=entity_tokens,
        entity_mask=entity_mask,
        hand_tokens=hand,
        opp_hand_probs=probs,
        opp_out_of_cycle=out,
        opp_elixir_interval=np.asarray([0.0, 10.0], dtype=np.float32),
        event_history=np.zeros((EVENT_HISTORY_LEN, EVENT_DIM), dtype=np.float32),
        legal_play=legal_play,
        legal_wait=True,
        own_elixir=float(elixir),
        target=target,
        soft_placement=soft,
        provenance={
            "generator": GENERATOR_VERSION,
            "teacher": TEACHER_VERSION,
            "seed": config.seed,
            "family": family,
            "index": index,
        },
    )


def generate_dataset(config: DistillationConfig) -> list[DistillationSample]:
    """Generate ``n_states`` deterministic supervised states."""

    return [generate_sample(config, index) for index in range(config.n_states)]


def to_torch_batch(samples: list[DistillationSample]) -> dict[str, object]:
    """Stack samples into ``[B, T=1]`` training tensors for the V4 actor."""

    import torch

    from .trajectory import ActionMasks
    from .model_v4 import V4ActionBatch

    def stack(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(getattr(sample, name), dtype=dtype) for sample in samples]
        ).unsqueeze(1)

    batch = int(len(samples))
    rows, cols = samples[0].legal_play.shape[1], samples[0].legal_play.shape[2]
    raster = stack("raster", torch.float32)
    glob = stack("global_features", torch.float32)
    entities = stack("entity_tokens", torch.float32)
    entity_mask = stack("entity_mask", torch.bool)
    hand = stack("hand_tokens", torch.float32)
    probs = stack("opp_hand_probs", torch.float32)
    out = stack("opp_out_of_cycle", torch.bool)
    elixir = stack("opp_elixir_interval", torch.float32)
    events = stack("event_history", torch.float32)
    legal_play = torch.stack(
        [torch.as_tensor(sample.legal_play, dtype=torch.bool) for sample in samples]
    ).unsqueeze(1)
    mode = torch.tensor([s.target.mode for s in samples], dtype=torch.long).reshape(batch, 1)
    card_mask = legal_play.reshape(batch, 1, 4, -1).any(dim=-1)
    mode_mask = torch.stack(
        (
            torch.ones(batch, 1, dtype=torch.bool),
            card_mask.any(dim=-1),
        ),
        dim=-1,
    )
    masks = ActionMasks(mode=mode_mask, card=card_mask, placement=legal_play)
    actions = V4ActionBatch(
        mode=mode,
        card_slot=torch.tensor(
            [s.target.card_slot for s in samples], dtype=torch.long
        ).reshape(batch, 1),
        placement=torch.tensor(
            [[s.target.row, s.target.col] for s in samples], dtype=torch.long
        ).reshape(batch, 1, 2),
        wait_duration=torch.tensor(
            [s.target.wait_duration_idx for s in samples], dtype=torch.long
        ).reshape(batch, 1),
    )
    soft = torch.stack(
        [torch.as_tensor(sample.soft_placement, dtype=torch.float32) for sample in samples]
    ).reshape(batch, 1, rows, cols)
    reset = torch.ones(batch, 1, dtype=torch.bool)
    return {
        "raster": raster,
        "global_features": glob,
        "entities": entities,
        "entity_mask": entity_mask,
        "hand_tokens": hand,
        "opp_hand_probs": probs,
        "opp_out_of_cycle": out,
        "opp_elixir_interval": elixir,
        "event_history": events,
        "masks": masks,
        "actions": actions,
        "soft_placement": soft,
        "reset_mask": reset,
        "families": [sample.family for sample in samples],
    }


__all__ = [
    "GENERATOR_VERSION",
    "PLAYER_DECK",
    "TACTICAL_FAMILIES",
    "TEACHER_VERSION",
    "DistillationConfig",
    "DistillationSample",
    "generate_dataset",
    "generate_sample",
    "to_torch_batch",
]
