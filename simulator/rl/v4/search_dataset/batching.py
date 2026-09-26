"""Recurrent sequence grouping and tensor batching."""

from __future__ import annotations

from typing import Sequence
import numpy as np
from simulator.rl.v4.distillation import DistillationSample
from .config import _stable_seed


def group_search_sequences(
    samples: Sequence[DistillationSample],
    *,
    sequence_length: int | None = None,
) -> list[list[DistillationSample]]:
    """Group search samples into ordered recurrent trajectories.

    Active records carry a stable ``seq_id`` and ``sequence_offset`` in their
    provenance.  Samples without a sequence id are independent states and
    therefore become one step trajectories.  A positive ``sequence_length``
    optionally windows long trajectories; each window is a complete training
    sequence and starts with a reset, so no hidden state is fabricated across
    windows.
    """

    if sequence_length is not None:
        if type(sequence_length) is not int or sequence_length < 0:
            raise ValueError("sequence_length must be a non-negative integer or None")
        window = int(sequence_length) or None
    else:
        window = None
    grouped: dict[str, list[tuple[int, DistillationSample]]] = {}
    for index, sample in enumerate(samples):
        provenance = sample.provenance
        seq_id = str(provenance.get("seq_id", ""))
        if not seq_id:
            # Independent sim-natural records may legitimately have identical
            # state hashes in a legacy checkpoint.  Their list position keeps
            # them independent rather than accidentally sharing hidden state.
            seq_id = f"__independent__:{index}"
        offset = int(
            provenance.get(
                "sequence_offset",
                provenance.get("offset", index),
            )
        )
        grouped.setdefault(seq_id, []).append((offset, sample))
    trajectories: list[list[DistillationSample]] = []
    for seq_id in sorted(grouped):
        runs: list[list[DistillationSample]] = []
        previous = None
        for offset, sample in sorted(grouped[seq_id], key=lambda item: item[0]):
            # Missing decisions and repeated training examples cannot share GRU state.
            if previous is None or offset != previous + 1:
                runs.append([])
            runs[-1].append(sample)
            previous = offset
        for ordered in runs:
            if window is None:
                trajectories.append(ordered)
            else:
                trajectories.extend(
                    ordered[start : start + window]
                    for start in range(0, len(ordered), window)
                )
    return trajectories


def iter_search_sequence_batches(
    samples: Sequence[DistillationSample],
    *,
    batch_size: int,
    sequence_length: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[list[list[DistillationSample]]]:
    """Return deterministic batches of equal-length recurrent trajectories.

    Sequences of different lengths are kept in separate batches so padding
    cannot contribute fabricated labels to the supervised loss.  The return
    value is a list of batches; each batch is suitable for
    :func:`to_torch_sequence_batch`.
    """

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    trajectories = group_search_sequences(samples, sequence_length=sequence_length)
    by_length: dict[int, list[list[DistillationSample]]] = {}
    for trajectory in trajectories:
        if trajectory:
            by_length.setdefault(len(trajectory), []).append(trajectory)
    batches: list[list[list[DistillationSample]]] = []
    for length in sorted(by_length):
        group = by_length[length]
        if shuffle:
            rng = np.random.default_rng(_stable_seed(int(seed), "sequence-order", length))
            order = rng.permutation(len(group)).tolist()
            group = [group[int(index)] for index in order]
        batches.extend(
            group[start : start + batch_size]
            for start in range(0, len(group), batch_size)
        )
    if shuffle:
        rng = np.random.default_rng(_stable_seed(int(seed), "batch-order"))
        order = rng.permutation(len(batches)).tolist()
        batches = [batches[int(index)] for index in order]
    return batches


def to_torch_sequence_batch(
    sequences: Sequence[Sequence[DistillationSample]],
) -> dict[str, object]:
    """Stack equal-length sample trajectories into ``[B, T, ...]`` tensors."""

    import torch

    from ..model_v4 import V4ActionBatch, masks_from_legal_play

    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("sequences must contain at least one non-empty trajectory")
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError("all recurrent trajectories in a batch must have equal length")
    batch = len(sequences)
    time = next(iter(lengths))
    first = sequences[0][0]
    rows, cols = first.legal_play.shape[1], first.legal_play.shape[2]

    def stack(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack(
            [
                torch.stack(
                    [torch.as_tensor(getattr(sample, name), dtype=dtype) for sample in sequence]
                )
                for sequence in sequences
            ]
        )

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
        [
            torch.stack(
                [torch.as_tensor(sample.legal_play, dtype=torch.bool) for sample in sequence]
            )
            for sequence in sequences
        ]
    )
    mode = torch.tensor(
        [[sample.target.mode for sample in sequence] for sequence in sequences],
        dtype=torch.long,
    )
    legal_wait = torch.tensor(
        [[sample.legal_wait for sample in sequence] for sequence in sequences],
        dtype=torch.bool,
    )
    masks = masks_from_legal_play(legal_play, legal_wait)
    actions = V4ActionBatch(
        mode=mode,
        card_slot=torch.tensor(
            [[sample.target.card_slot for sample in sequence] for sequence in sequences],
            dtype=torch.long,
        ),
        placement=torch.tensor(
            [
                [[sample.target.row, sample.target.col] for sample in sequence]
                for sequence in sequences
            ],
            dtype=torch.long,
        ),
        wait_duration=torch.tensor(
            [[sample.target.wait_duration_idx for sample in sequence] for sequence in sequences],
            dtype=torch.long,
        ),
    )
    soft = torch.stack(
        [
            torch.stack(
                [torch.as_tensor(sample.soft_placement, dtype=torch.float32) for sample in sequence]
            )
            for sequence in sequences
        ]
    ).reshape(batch, time, rows, cols)
    reset = torch.zeros(batch, time, dtype=torch.bool)
    reset[:, 0] = True
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
        "families": [[sample.family for sample in sequence] for sequence in sequences],
        "sequence_ids": [
            str(sequence[0].provenance.get("seq_id", "")) for sequence in sequences
        ],
    }
