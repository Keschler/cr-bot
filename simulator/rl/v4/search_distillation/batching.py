"""Convert cached recurrent prefixes into policy input batches."""

from __future__ import annotations
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..model_v4 import RecurrentV4Policy, V4Logits, masks_from_legal_play
from .records import CachedPrefix


def _stack_records(records: Sequence[CachedPrefix], *, context: int, device: torch.device) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot stack an empty record list")
    expected = context + 1
    def selected(record: CachedPrefix) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray], list[bool]]:
        if min(len(record.observations), len(record.legal_play), len(record.legal_wait)) < expected:
            raise ValueError("record does not contain the requested context")
        return (record.observations[-expected:], record.legal_play[-expected:],
                record.legal_wait[-expected:])
    names = (
        "raster", "global_features", "entities", "entity_mask", "hand_tokens",
        "opp_hand_probs", "opp_out_of_cycle", "opp_elixir_interval", "event_history",
    )
    output: dict[str, Any] = {}
    for name in names:
        dtype = torch.bool if name == "entity_mask" or name == "opp_out_of_cycle" else torch.float32
        output[name] = torch.stack(
            [torch.stack([torch.as_tensor(obs[name], dtype=dtype) for obs in selected(r)[0]]) for r in records]
        ).to(device)
    legal = torch.stack(
        [torch.stack([torch.as_tensor(value, dtype=torch.bool) for value in selected(r)[1]]) for r in records]
    ).to(device)
    legal_wait = torch.tensor([selected(r)[2] for r in records], dtype=torch.bool, device=device)
    output["legal_play"] = legal
    output["masks"] = masks_from_legal_play(legal, legal_wait)
    output["reset_mask"] = torch.zeros(len(records), expected, dtype=torch.bool, device=device)
    output["reset_mask"][:, 0] = True
    return output


def _forward(policy: RecurrentV4Policy, batch: Mapping[str, Any]) -> tuple[V4Logits, Any]:
    logits, _, _ = policy(
        batch["raster"], batch["global_features"], batch["entities"],
        batch["entity_mask"], batch["hand_tokens"], batch["opp_hand_probs"],
        batch["opp_out_of_cycle"], batch["opp_elixir_interval"],
        batch["event_history"], batch["reset_mask"],
    )
    return logits, batch["masks"]
