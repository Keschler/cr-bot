"""JSON display data derived from authoritative state without mutating it.

Coordinates retain the simulator orientation (player zero at the bottom) and
are converted from milli-tiles to tiles. Neural dependencies stay lazy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from simulator.actions import PlayCardAction, UseAbilityAction


def project_actions(actions: Iterable[Any], state: Any) -> list[dict[str, Any]]:
    result = []
    for action in actions:
        if isinstance(action, Mapping):
            result.append(dict(action))
            continue
        row: dict[str, Any] = {"player": action.player, "kind": "wait"}
        if isinstance(action, PlayCardAction):
            hand = state.players[action.player].hand
            row.update(kind="play", slot=action.card_slot, col=action.cell[0], row=action.cell[1])
            if 0 <= action.card_slot < len(hand):
                row["card_id"] = hand[action.card_slot]
        elif isinstance(action, UseAbilityAction):
            row.update(kind="ability", entity_uid=action.entity_uid)
        result.append(row)
    return result


def project_policy(logits: Any, masks: Any, actions: Any, *, lane: int = 0) -> dict[str, Any]:
    """Project masked distributions already computed for a pre-action state.

The placement heatmap is conditional on PLAY, marginalized over legal card
slots. WAIT duration is excluded because simulator V4 decisions use a fixed
interval. This function neither samples actions nor invokes the policy.
"""
    import torch

    def probabilities(values: Any, legal: Any) -> Any:
        values = values.detach().cpu().double()
        legal = legal.detach().cpu().bool()
        if not bool(legal.any()):
            return torch.zeros_like(values)
        return torch.softmax(values.masked_fill(~legal, float("-inf")), dim=-1)

    mode = probabilities(logits.mode[lane, 0], masks.mode[lane, 0])
    card = probabilities(logits.card[lane, 0], masks.card[lane, 0])
    placement_logits = logits.placement[lane, 0].detach().cpu()
    placement_mask = masks.placement[lane, 0].detach().cpu()
    maps = torch.stack([
        probabilities(values.flatten(), legal.flatten())
        for values, legal in zip(placement_logits, placement_mask, strict=True)
    ])
    heatmap = (maps * card.unsqueeze(-1)).sum(dim=0)

    def entropy(probabilities: Any) -> float:
        positive = probabilities[probabilities > 0]
        return float(-(positive * positive.log()).sum())

    selected_slot = int(actions.card_slot[lane, 0])
    action_mode = int(actions.mode[lane, 0])
    action_probability = float(mode[action_mode])
    if action_mode == 1:
        row, col = (int(value) for value in actions.placement[lane, 0])
        cell = row * placement_logits.shape[-1] + col
        action_probability *= float(card[selected_slot] * maps[selected_slot, cell])
    return {
        "distribution": "masked",
        "wait_probability": float(mode[0]),
        "play_probability": float(mode[1]),
        "slot_probabilities": card.tolist(),
        "heatmap": heatmap.tolist(),
        "selected_slot": selected_slot if action_mode == 1 else None,
        "entropy": entropy(mode) + float(mode[1]) * (
            entropy(card) + sum(float(card[i]) * entropy(maps[i]) for i in range(len(card)))
        ),
        "action_probability": action_probability,
    }


def project_state(
    state: Any,
    ruleset: Any,
    *,
    index: int = 0,
    events: Iterable[Any] = (),
    actions: Iterable[Any] = (),
    policy: Mapping[str, Any] | None = None,
    reward: float | None = None,
) -> dict[str, Any]:
    """Build one detached JSON frame; authoritative state is never modified."""
    entities = []
    for entity in sorted(state.entities.values(), key=lambda item: item.uid):
        if not entity.alive:
            continue
        definition = ruleset.towers.get(entity.card_id) or ruleset.cards.get(entity.card_id)
        statuses = [status.kind for status in entity.statuses if status.remaining_us > 0]
        for name, active in (
            ("shield", entity.shield_hp > 0),
            ("stealth", entity.stealth_active),
            ("burrow", entity.burrow_active),
            ("concealed", entity.concealed_active),
            ("dash", entity.dash_remaining_us > 0),
            ("jump", entity.jump_remaining_us > 0),
            ("clone", entity.is_clone),
            ("charge", entity.attack_charge_active or entity.charge_active),
        ):
            if active:
                statuses.append(name)
        entities.append({
            "uid": entity.uid, "card_id": entity.card_id, "owner": entity.owner,
            "kind": entity.kind, "role": entity.role,
            "x": entity.x_mtile / 1000, "y": entity.y_mtile / 1000,
            "hp": entity.hp, "max_hp": entity.max_hp,
            "target_uid": entity.target_uid,
            "range": (getattr(definition, "range_mtile", None) or 0) / 1000,
            "radius": (getattr(definition, "collision_radius_mtile", None) or 0) / 1000,
            "statuses": statuses, "deploying": entity.deploy_remaining_us > 0,
        })
    return {
        "index": index, "tick": state.tick, "time_s": state.elapsed_us / 1_000_000,
        "phase": state.phase, "terminal": state.terminal, "winner": state.winner,
        "players": [{
            "player": player, "hand": list(value.hand), "deck": list(value.deck),
            "next_card": value.draw_pile[0] if value.draw_pile else None,
            "elixir": value.elixir_milli / 1000, "crowns": value.crowns,
        } for player, value in enumerate(state.players)],
        "entities": entities,
        "projectiles": [{
            "uid": value.uid, "owner": value.owner, "card_id": value.source_card_id,
            "x": value.x_mtile / 1000, "y": value.y_mtile / 1000,
            "target_x": value.target_x_mtile / 1000, "target_y": value.target_y_mtile / 1000,
            "radius": value.radius_mtile / 1000,
        } for value in sorted(state.projectiles.values(), key=lambda item: item.uid) if value.alive],
        "effects": [{
            "uid": value.uid, "owner": value.owner, "card_id": value.source_card_id,
            "x": value.x_mtile / 1000, "y": value.y_mtile / 1000,
            "radius": value.radius_mtile / 1000, "remaining_s": value.remaining_us / 1_000_000,
        } for value in sorted(state.effects.values(), key=lambda item: item.uid) if value.alive],
        "events": [event.to_dict() if hasattr(event, "to_dict") else dict(event) for event in events],
        "actions": project_actions(actions, state),
        "policy": dict(policy) if policy is not None else None,
        "reward": float(reward) if reward is not None else None,
    }
