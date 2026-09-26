"""Zero-restart action selection and confidence diagnostics for frontier entries."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..frontier_eval import BranchSpec, load_frontier_state
from .constants import FRONTIER_PPO_VERSION
from .environment import entry_input_tensors, make_frontier_env


def _policy_argmax_at_entry(
    policy: Any, entry: dict[str, Any], device: Any = None
) -> BranchSpec:
    """Deterministic action of a policy at a sealed entry state."""

    import torch

    from ..v4_ppo import configure_v4_runtime
    configure_v4_runtime(policy)
    policy.eval()

    from ..frontier_eval import _observe_inputs
    from ..model_v4 import masks_from_legal_play

    env = make_frontier_env()
    load_frontier_state(env, entry["initial_state"])
    inputs, legal, _, _ = _observe_inputs(env, device)
    tensors = entry_input_tensors(inputs, device)
    legal_t = torch.as_tensor(legal, device=device).unsqueeze(0).unsqueeze(0)
    masks = masks_from_legal_play(legal_t, bool(inputs["legal_wait"]))
    reset = torch.zeros((1, 1), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, _, _ = policy.forward(
            tensors["raster"], tensors["global_features"], tensors["entities"],
            tensors["entity_mask"], tensors["hand_tokens"], tensors["opp_hand_probs"],
            tensors["opp_out_of_cycle"], tensors["opp_elixir_interval"],
            tensors["event_history"], reset,
        )
        acts = policy.act_deterministic(logits, masks)
    mode = int(acts.mode[0, 0])
    return BranchSpec(
        source="policy", mode=mode,
        slot=int(acts.card_slot[0, 0]),
        row=int(acts.placement[0, 0, 0]),
        col=int(acts.placement[0, 0, 1]),
    )


def _entry_continuation_hidden(champion, env, device):
    """Fresh recurrent state after the first observation, matching entry actions."""
    import torch
    from ..frontier_eval import _observe_inputs, _step_inputs
    from ..v4_ppo import configure_v4_runtime
    configure_v4_runtime(champion)
    champion.eval()
    current_inputs, _, _, _ = _observe_inputs(env, device)
    with torch.no_grad():
        _, _, hidden = champion.forward_step(_step_inputs(current_inputs, device))
    return hidden


def action_logit_margins(
    policy: Any, entries: Sequence[dict[str, Any]], device: Any = None
) -> dict[str, Any]:
    """Per-entry chosen-vs-runner-up margins (direct confidence readout).

    Same zero-start forward convention as :func:`_policy_argmax_at_entry`
    so margins describe the exact logits the policy acts on.  Margins are
    probability gaps after legality masking: mode (2-way), card (legal
    slots), placement (legal cells of the chosen slot).  Duration excluded
    (WAIT-only head, rarely decisive).
    """

    import torch

    from ..v4_ppo import configure_v4_runtime
    configure_v4_runtime(policy)
    policy.eval()

    from ..frontier_eval import _observe_inputs
    from ..model_v4 import masks_from_legal_play

    per_entry = []
    for entry in entries:
        env = make_frontier_env()
        load_frontier_state(env, entry["initial_state"])
        inputs, legal, _, _ = _observe_inputs(env, device)
        tensors = entry_input_tensors(inputs, device)
        legal_t = torch.as_tensor(legal, device=device).unsqueeze(0).unsqueeze(0)
        masks = masks_from_legal_play(legal_t, bool(inputs["legal_wait"]))
        reset = torch.zeros((1, 1), dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, _, _ = policy.forward(
                tensors["raster"], tensors["global_features"], tensors["entities"],
                tensors["entity_mask"], tensors["hand_tokens"], tensors["opp_hand_probs"],
                tensors["opp_out_of_cycle"], tensors["opp_elixir_interval"],
                tensors["event_history"], reset,
            )
            mode_logp = torch.where(
                masks.mode, logits.mode[0, 0].float(),
                torch.full_like(logits.mode[0, 0], float("-inf")),
            )
            mode_p = torch.softmax(mode_logp, dim=-1).cpu().numpy().reshape(-1)
            mode_sorted = sorted(float(v) for v in mode_p)
            mode_margin = mode_sorted[-1] - (mode_sorted[-2] if len(mode_sorted) > 1 else 0.0)
            # The card distribution is undefined when no card can be played.
            card_margin = None
            if bool(masks.card.any().item()):
                card_logp = torch.where(
                    masks.card, logits.card[0, 0].float(),
                    torch.full_like(logits.card[0, 0], float("-inf")),
                )
                card_p = torch.softmax(card_logp, dim=-1).cpu().numpy().reshape(-1)
                card_sorted = sorted(float(v) for v in card_p)
                card_margin = card_sorted[-1] - (card_sorted[-2] if len(card_sorted) > 1 else 0.0)
            chosen_mode = int(mode_p.argmax())
            placement_margin = None
            if chosen_mode == 1:
                slot = int(card_p.argmax())
                cell_logp = torch.where(
                    masks.placement[0, 0, slot],
                    logits.placement[0, 0, slot].float(),
                    torch.full_like(logits.placement[0, 0, slot], float("-inf")),
                )
                cell_p = torch.softmax(cell_logp.reshape(-1), dim=-1).cpu().numpy()
                cell_sorted = sorted(float(v) for v in cell_p)
                placement_margin = cell_sorted[-1] - (
                    cell_sorted[-2] if len(cell_sorted) > 1 else 0.0)
        per_entry.append(
            {
                "entry_id": entry["entry_id"],
                "chosen_mode": chosen_mode,
                "mode_margin": round(float(mode_margin), 4),
                "card_margin": round(float(card_margin), 4) if card_margin is not None else None,
                "placement_margin": round(float(placement_margin), 4) if placement_margin is not None else None,
            }
        )

    def _summary(vals: list[float]) -> dict[str, float | int | None]:
        s = sorted(vals)
        n = len(s)
        if n == 0:
            # Placement margins are intentionally absent when every selected
            # action is WAIT.  Keep the report schema explicit while marking
            # undefined statistics as null instead of dividing by zero.
            return {
                "n": 0,
                "mean": None,
                "p10": None,
                "median": None,
                "p90": None,
                "frac_close_lt_010": None,
                "frac_close_lt_025": None,
            }
        return {
            "n": n,
            "mean": round(sum(s) / n, 4),
            "p10": round(s[max(0, n // 10)], 4),
            "median": round(s[n // 2], 4),
            "p90": round(s[min(n - 1, 9 * n // 10)], 4),
            "frac_close_lt_010": round(sum(1 for v in s if v < 0.10) / n, 4),
            "frac_close_lt_025": round(sum(1 for v in s if v < 0.25) / n, 4),
        }

    return {
        "version": FRONTIER_PPO_VERSION,
        "per_entry": per_entry,
        "mode": _summary([e["mode_margin"] for e in per_entry]),
        "card": _summary([e["card_margin"] for e in per_entry if e["card_margin"] is not None]),
        "placement": _summary([e["placement_margin"] for e in per_entry if e["placement_margin"] is not None]),
        "n_play": sum(1 for e in per_entry if e["chosen_mode"] == 1),
        "n_wait": sum(1 for e in per_entry if e["chosen_mode"] == 0),
    }
