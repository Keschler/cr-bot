"""Frozen champion loading and batched prediction."""

from __future__ import annotations

from typing import Any
import numpy as np
from simulator.observation_v3 import BELIEF_CARD_COUNT, EVENT_DIM, EVENT_HISTORY_LEN
from simulator.rl.v4.distillation import DistillationSample
from simulator.rl.v4.simulator_teacher import TeacherTarget
from .config import SEARCH_GENERATOR_VERSION


def load_champion_actor(path: str, config=None):
    """Load a frozen champion checkpoint into a fresh V4 actor.

    The placement head materializes its cell-bias parameter lazily on
    first forward, so a dummy forward runs before the strict load.
    """

    import torch

    try:
        from ..model_v4 import ModelConfigV4, RecurrentV4Policy
    except ImportError:  # pragma: no cover
        from simulator.rl.v4.model_v4 import ModelConfigV4, RecurrentV4Policy
    policy = RecurrentV4Policy(config or ModelConfigV4())
    policy.eval()
    with torch.no_grad():
        zeros = torch.zeros(1, 1, 21, 32, 18)
        globals_768 = torch.zeros(1, 1, 768)
        entities = torch.zeros(1, 1, 128, 32)
        entity_mask = torch.zeros(1, 1, 128, dtype=torch.bool)
        hand = torch.zeros(1, 1, 4, 16)
        probs = torch.full((1, 1, 128), 1.0 / 128)
        out = torch.zeros(1, 1, 128, dtype=torch.bool)
        interval = torch.zeros(1, 1, 2)
        events = torch.zeros(1, 1, 16, 8)
        reset = torch.ones(1, 1, dtype=torch.bool)
        policy(
            zeros, globals_768, entities, entity_mask, hand, probs, out,
            interval, events, reset,
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def _record_to_inference_sample(record: dict[str, Any]) -> DistillationSample:
    event_history = np.asarray(
        record.get("event_history", np.zeros((EVENT_HISTORY_LEN, EVENT_DIM), dtype=np.float32)),
        dtype=np.float32,
    )
    if event_history.shape != (EVENT_HISTORY_LEN, EVENT_DIM):
        raise ValueError("invalid event_history shape")
    opp_probs = np.asarray(
        record.get("opp_hand_probs", np.full((BELIEF_CARD_COUNT,), 1.0 / BELIEF_CARD_COUNT)),
        dtype=np.float32,
    )
    if opp_probs.shape != (BELIEF_CARD_COUNT,):
        raise ValueError("invalid opp_hand_probs shape")
    opp_out = np.asarray(
        record.get("opp_out_of_cycle", np.zeros((BELIEF_CARD_COUNT,), dtype=bool)),
        dtype=bool,
    )
    if opp_out.shape != (BELIEF_CARD_COUNT,):
        raise ValueError("invalid opp_out_of_cycle shape")
    opp_interval = np.asarray(record.get("opp_elixir_interval", [0.0, 10.0]), dtype=np.float32)
    if opp_interval.shape != (2,):
        raise ValueError("invalid opp_elixir_interval shape")
    return DistillationSample(
        family=record["family"],
        raster=np.ascontiguousarray(record["raster"], dtype=np.float32),
        global_features=np.ascontiguousarray(record["global_features"], dtype=np.float32),
        entity_tokens=np.ascontiguousarray(record["entity_tokens"], dtype=np.float32),
        entity_mask=np.ascontiguousarray(record["entity_mask"], dtype=bool),
        hand_tokens=np.ascontiguousarray(record["hand_tokens"], dtype=np.float32),
        opp_hand_probs=opp_probs,
        opp_out_of_cycle=opp_out,
        opp_elixir_interval=opp_interval,
        event_history=event_history,
        legal_play=np.ascontiguousarray(record["legal_play"], dtype=bool),
        legal_wait=bool(record["legal_wait"]),
        own_elixir=float(record["own_elixir"]),
        target=TeacherTarget(mode=0, card_slot=0, row=0, col=0, wait_duration_idx=0),
        soft_placement=np.zeros((32, 18), dtype=np.float32),
        provenance={"generator": SEARCH_GENERATOR_VERSION, "index": -1},
    )


def _predict_records(
    policy: Any, records: list[dict[str, Any]], device: Any = None
) -> list[dict[str, int]]:
    """Batched champion inference: one (mode, card, row, col) per record.

    ``device`` defaults to whatever device the policy lives on, keeping
    model and batch tensors consistent without caller plumbing.
    """

    import torch

    try:
        from ..distillation import move_batch_to_device, to_torch_batch
    except ImportError:  # pragma: no cover
        from simulator.rl.v4.distillation import move_batch_to_device, to_torch_batch
    if device is None:
        try:
            device = next(policy.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    preds: list[dict[str, int]] = []
    was_training = policy.training
    policy.eval()
    active = any(record.get("seq_id", "").startswith("active:") for record in records)
    hidden = None
    previous_seq = None
    previous_offset = None
    chunk_size = 1 if active else 256
    with torch.no_grad():
        for start in range(0, len(records), chunk_size):
            chunk = [_record_to_inference_sample(r) for r in records[start : start + chunk_size]]
            batch = move_batch_to_device(to_torch_batch(chunk), device)
            if active:
                record = records[start]
                seq = record.get("seq_id")
                offset = int(record.get("sequence_offset", 0))
                reset = seq != previous_seq or previous_offset is None or offset != previous_offset + 1
                batch["reset_mask"].fill_(reset)
                if reset:
                    hidden = None
                previous_seq, previous_offset = seq, offset
            if active:
                logits, _, hidden = policy.forward_step(
                    batch, hidden=hidden, reset_mask=batch["reset_mask"]
                )
            else:
                logits, _, _ = policy(
                    batch["raster"], batch["global_features"], batch["entities"],
                    batch["entity_mask"], batch["hand_tokens"], batch["opp_hand_probs"],
                    batch["opp_out_of_cycle"], batch["opp_elixir_interval"],
                    batch["event_history"], batch["reset_mask"],
                )
            _, cols = logits.placement.shape[-2:]
            for i in range(len(chunk)):
                mode = int(logits.mode[i, 0].argmax())
                card = int(logits.card[i, 0].argmax())
                flat = logits.placement[i, 0, card].reshape(-1)
                mask = batch["masks"].placement[i, 0, card].reshape(-1)
                safe = torch.where(mask, flat, torch.full_like(flat, float("-inf")))
                cell = int(safe.argmax())
                preds.append(
                    {"mode": mode, "card": card, "row": cell // cols, "col": cell % cols}
                )
    policy.train(was_training)
    return preds

