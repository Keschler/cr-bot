"""Decision-level diagnostics for recurrent actor/PPO investigations.

This module is deliberately observation agnostic: it only turns already
computed policy tensors and simulator snapshots into JSON-shaped records.  It
does not choose or execute actions.  The authoritative state fields are
marked as diagnostic-only by the callers; they are never fed back into the
actor.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence


DIAGNOSTIC_TRACE_SCHEMA_VERSION = 1


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_scalar(value: Any) -> Any:
    """Convert common tensor/numpy scalar values without importing either."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_scalar(item())
        except (TypeError, ValueError):
            return None
    return value


def _action_descriptor(action: Any) -> dict[str, Any]:
    """Return a stable, model/simulator independent action description."""

    if action is None:
        return {"mode": "WAIT"}
    if isinstance(action, Mapping):
        raw_kind = action.get("mode", action.get("kind", "WAIT"))
        slot = action.get("card_slot", action.get("card_idx"))
        cell = action.get("world_cell", action.get("cell"))
    else:
        raw_kind = getattr(action, "kind", "Wait")
        slot = getattr(action, "card_idx", None)
        if slot is None:
            slot = getattr(action, "card_slot", None)
        cell = getattr(action, "cell", None)
    kind = str(raw_kind).strip().casefold().replace("_", "-")
    if kind == "play":
        kind = "play"
    if kind in {"wait", "noop", "no-op"}:
        return {"mode": "WAIT"}
    if kind != "play":
        return {"mode": kind.upper()}
    row: dict[str, Any] = {
        "mode": "PLAY",
        "card_slot": None if slot is None else int(slot),
    }
    if isinstance(cell, Sequence) and not isinstance(cell, (str, bytes)) and len(cell) == 2:
        row["world_cell"] = [int(cell[0]), int(cell[1])]
    else:
        row["world_cell"] = None
    return row


def action_equal(left: Any, right: Any) -> bool:
    """Compare actions at the environment ABI level."""

    return _action_descriptor(left) == _action_descriptor(right)


def serialize_action_masks(masks: Any, *, lane: int = 0, time: int = 0) -> dict[str, Any]:
    """Serialize the complete legal-action mask for one decision."""

    mode = masks.mode[lane, time].detach().cpu().tolist()
    card = masks.card[lane, time].detach().cpu().tolist()
    placement = masks.placement[lane, time].detach().cpu()
    legal_cells: list[list[list[int]]] = []
    for card_index in range(int(placement.shape[0])):
        cells = placement[card_index].nonzero(as_tuple=False).tolist()
        legal_cells.append([[int(row), int(column)] for row, column in cells])
    return {
        "mode": [bool(value) for value in mode],
        "card": [bool(value) for value in card],
        "placement": legal_cells,
        "placement_coordinate_order": "[row,column]",
    }


def _distribution_rows(log_probs: Any, *, names: Sequence[Any] | None = None, limit: int = 5) -> list[dict[str, Any]]:
    values = log_probs.detach().cpu()
    if values.ndim != 1:
        values = values.reshape(-1)
    finite = torch_isfinite(values)
    candidates = [index for index, valid in enumerate(finite) if valid]
    candidates.sort(key=lambda index: float(values[index]), reverse=True)
    rows: list[dict[str, Any]] = []
    for index in candidates[:limit]:
        log_probability = float(values[index])
        probability = math.exp(log_probability)
        rows.append(
            {
                "index": int(index),
                "label": None if names is None or index >= len(names) else names[index],
                "log_probability": log_probability,
                "probability": probability,
            }
        )
    return rows


def torch_isfinite(value: Any) -> list[bool]:
    """Avoid a module-level torch import while preserving ``-inf`` masks."""

    return [math.isfinite(float(item)) for item in value.tolist()]


def _entropy(log_probs: Any) -> float:
    finite = torch_isfinite_tensor(log_probs)
    probabilities = log_probs.exp()
    terms = torch_where(finite, probabilities * log_probs, probabilities.new_zeros(probabilities.shape))
    return float((-terms).sum().item())


def torch_isfinite_tensor(value: Any) -> Any:
    import torch

    return torch.isfinite(value)


def torch_where(mask: Any, left: Any, right: Any) -> Any:
    import torch

    return torch.where(mask, left, right)


def _state_entities(raw_state: Any) -> tuple[Any, ...]:
    if raw_state is None:
        return ()
    if isinstance(raw_state, Mapping):
        entities = raw_state.get("entities", ())
        return tuple(entities) if isinstance(entities, (list, tuple)) else ()
    entities = getattr(raw_state, "entities", {})
    return tuple(entities.values()) if isinstance(entities, Mapping) else ()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _player(raw_state: Any, index: int) -> Any:
    players = raw_state.get("players", ()) if isinstance(raw_state, Mapping) else getattr(raw_state, "players", ())
    if not isinstance(players, (list, tuple)) or not 0 <= index < len(players):
        return None
    return players[index]


def _tower_hp(raw_state: Any) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for entity in _state_entities(raw_state):
        if _field(entity, "kind") != "tower":
            continue
        owner = _field(entity, "owner")
        role = _field(entity, "role")
        hp = _field(entity, "hp")
        maximum = _field(entity, "max_hp")
        if type(owner) is not int or not isinstance(role, str) or type(hp) is not int or type(maximum) is not int:
            continue
        result.setdefault(f"player_{owner}", {})[role] = {"hp": hp, "max_hp": maximum}
    return result


def _units(raw_state: Any) -> list[dict[str, Any]]:
    """Capture alive non-tower entities and their authoritative positions."""

    rows: list[dict[str, Any]] = []
    for entity in _state_entities(raw_state):
        if not bool(_field(entity, "alive", False)) or _field(entity, "kind") == "tower":
            continue
        uid = _field(entity, "uid")
        owner = _field(entity, "owner")
        if type(uid) is not int or type(owner) is not int:
            continue
        row: dict[str, Any] = {
            "uid": uid,
            "owner": owner,
            "card_id": _field(entity, "card_id"),
            "kind": _field(entity, "kind"),
            "role": _field(entity, "role"),
            "x_mtile": _field(entity, "x_mtile"),
            "y_mtile": _field(entity, "y_mtile"),
            "hp": _field(entity, "hp"),
            "max_hp": _field(entity, "max_hp"),
            "target_uid": _field(entity, "target_uid"),
            "deploy_remaining_us": _field(entity, "deploy_remaining_us"),
        }
        rows.append({key: _json_scalar(value) for key, value in row.items()})
    rows.sort(key=lambda item: int(item["uid"]))
    return rows


def state_snapshot(raw_state: Any) -> dict[str, Any] | None:
    """Return a compact authoritative diagnostic snapshot.

    The snapshot intentionally includes hidden simulator state for debugging,
    but callers must not use it as actor input.
    """

    if raw_state is None:
        return None
    to_primitive = getattr(raw_state, "to_primitive", None)
    if callable(to_primitive):
        raw_state = to_primitive(include_events=False)
    players: list[dict[str, Any]] = []
    for index in (0, 1):
        player = _player(raw_state, index)
        players.append(
            {
                "player": index,
                "hand": list(_field(player, "hand", ()) or ()),
                "elixir_milli": _json_scalar(_field(player, "elixir_milli")),
                "crowns": _json_scalar(_field(player, "crowns")),
            }
        )
    return {
        "tick": _json_scalar(_field(raw_state, "tick")),
        "elapsed_us": _json_scalar(_field(raw_state, "elapsed_us")),
        "phase": _field(raw_state, "phase"),
        "players": players,
        "tower_hp": _tower_hp(raw_state),
        "units": _units(raw_state),
    }


def _action_for_index(action_batch: Any, lane: int, time: int) -> dict[str, Any]:
    mode = int(action_batch.mode[lane, time].detach().cpu().item())
    if mode == 0:
        return {"mode": "WAIT"}
    row = int(action_batch.placement[lane, time, 0].detach().cpu().item())
    column = int(action_batch.placement[lane, time, 1].detach().cpu().item())
    return {
        "mode": "PLAY",
        "card_slot": int(action_batch.card_slot[lane, time].detach().cpu().item()),
        "policy_cell": [row, column],
        "world_cell": [column, row],
    }


def _descriptor_log_probability(
    factor_log_probs: Any,
    masks: Any,
    descriptor: Mapping[str, Any],
    *,
    lane: int,
    time: int,
) -> Any:
    """Return the masked joint log-probability of a decoded action.

    This is used by checkpoint comparison to score the known-good action under
    every candidate policy.  It intentionally works from the already computed
    factor distributions, so it cannot execute an action or alter recurrent
    state.
    """

    mode = factor_log_probs.mode[lane, time]
    if descriptor.get("mode") == "WAIT":
        return mode[0]
    if descriptor.get("mode") != "PLAY":
        return mode[0].new_full((), float("-inf"))
    card = descriptor.get("card_slot")
    world_cell = descriptor.get("world_cell")
    if not isinstance(card, int) or not isinstance(world_cell, Sequence) or len(world_cell) != 2:
        return mode[0].new_full((), float("-inf"))
    column, row = (int(world_cell[0]), int(world_cell[1]))
    placement = factor_log_probs.placement[lane, time]
    if not (
        0 <= card < int(placement.shape[0])
        and 0 <= row < int(placement.shape[-2])
        and 0 <= column < int(placement.shape[-1])
    ):
        return mode[0].new_full((), float("-inf"))
    if not bool(masks.card[lane, time, card].item()):
        return mode[0].new_full((), float("-inf"))
    if not bool(masks.placement[lane, time, card, row, column].item()):
        return mode[0].new_full((), float("-inf"))
    return mode[1] + factor_log_probs.card[lane, time, card] + placement[card, row, column]


def build_policy_diagnostics(
    policy: Any,
    output: Any,
    masks: Any,
    actions: Any,
    *,
    lane: int = 0,
    time: int = 0,
    teacher_action: Any | None = None,
    actor_actions: Any | None = None,
    critic_value: Any | None = None,
    old_log_prob: Any | None = None,
    ppo_ratio: Any | None = None,
    ppo_clipped: bool | None = None,
    reference_action: Any | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Build probabilities, alternatives, masks, and teacher agreement."""

    factor_log_probs = policy.action_head.masked_log_probs(output.logits, masks)
    selected = _action_for_index(actions, lane, time)
    selected_mode = int(actions.mode[lane, time].detach().cpu().item())
    selected_card = int(actions.card_slot[lane, time].detach().cpu().item())
    selected_row = int(actions.placement[lane, time, 0].detach().cpu().item())
    selected_column = int(actions.placement[lane, time, 1].detach().cpu().item())
    mode_lp = factor_log_probs.mode[lane, time]
    card_lp = factor_log_probs.card[lane, time]
    placement_lp = factor_log_probs.placement[lane, time]
    selected_lp = mode_lp[selected_mode]
    if selected_mode == 1:
        selected_lp = selected_lp + card_lp[selected_card] + placement_lp[selected_card, selected_row, selected_column]
    selected_log_probability = float(selected_lp.detach().cpu().item())
    mode_names = ("WAIT", "PLAY")
    alternatives: list[dict[str, Any]] = []
    wait_probability = float(mode_lp[0].exp().detach().cpu().item())
    alternatives.append({"mode": "WAIT", "probability": wait_probability, "log_probability": float(mode_lp[0].detach().cpu().item())})
    play_mode_lp = mode_lp[1]
    for card_index in range(int(card_lp.shape[0])):
        if not bool(masks.card[lane, time, card_index].item()):
            continue
        flat = placement_lp[card_index].reshape(-1)
        for cell_index in flat.topk(min(top_k, flat.numel())).indices.detach().cpu().tolist():
            row, column = divmod(int(cell_index), int(placement_lp.shape[-1]))
            if not bool(masks.placement[lane, time, card_index, row, column].item()):
                continue
            log_probability = float((play_mode_lp + card_lp[card_index] + flat[cell_index]).detach().cpu().item())
            alternatives.append(
                {
                    "mode": "PLAY",
                    "card_slot": card_index,
                    "policy_cell": [row, column],
                    "world_cell": [column, row],
                    "probability": math.exp(log_probability),
                    "log_probability": log_probability,
                }
            )
    alternatives.sort(key=lambda row: float(row["probability"]), reverse=True)
    teacher_descriptor = None if teacher_action is None else _action_descriptor(teacher_action)
    reference_descriptor = None if reference_action is None else _action_descriptor(reference_action)
    reference_log_probability = None
    if reference_descriptor is not None:
        reference_log_probability = _finite(
            _descriptor_log_probability(
                factor_log_probs,
                masks,
                reference_descriptor,
                lane=lane,
                time=time,
            )
        )
    actor_descriptor = None
    if actor_actions is not None:
        actor_descriptor = _action_for_index(actor_actions, lane, time)
    executed_descriptor = selected
    agreement_descriptor = (
        executed_descriptor if actor_descriptor is None else actor_descriptor
    )
    agreement = (
        None
        if teacher_descriptor is None
        else action_equal(agreement_descriptor, teacher_action)
    )
    value = None if critic_value is None else _finite(critic_value[lane, time] if getattr(critic_value, "ndim", 0) > 1 else critic_value[lane])
    old_lp_value = None if old_log_prob is None else _finite(old_log_prob[lane, time] if getattr(old_log_prob, "ndim", 0) > 1 else old_log_prob[lane])
    ratio_value = None
    if ppo_ratio is not None:
        ratio_value = _finite(ppo_ratio[lane, time] if getattr(ppo_ratio, "ndim", 0) > 1 else ppo_ratio[lane])
    return {
        "legal_action_mask": serialize_action_masks(masks, lane=lane, time=time),
        "actor_action": actor_descriptor,
        "executed_action": executed_descriptor,
        "chosen_action_probability": math.exp(selected_log_probability),
        "chosen_action_log_probability": selected_log_probability,
        "top_mode_alternatives": _distribution_rows(mode_lp, names=mode_names, limit=top_k),
        "top_card_alternatives": _distribution_rows(card_lp, limit=top_k),
        "top_placement_alternatives": [
            {
                **row,
                "card_slot": selected_card,
                "policy_cell": [
                    int(row["index"]) // int(placement_lp.shape[-1]),
                    int(row["index"]) % int(placement_lp.shape[-1]),
                ],
                "world_cell": [
                    int(row["index"]) % int(placement_lp.shape[-1]),
                    int(row["index"]) // int(placement_lp.shape[-1]),
                ],
            }
            for row in _distribution_rows(placement_lp[selected_card].reshape(-1), limit=top_k)
        ],
        "top_alternative_actions": alternatives[:top_k],
        "factor_entropy": {
            "mode": _entropy(mode_lp),
            "card": _entropy(card_lp),
            "placement_selected_card": _entropy(placement_lp[selected_card].reshape(-1)),
            "joint": _entropy(mode_lp) + (_entropy(card_lp) + _entropy(placement_lp[selected_card].reshape(-1)) if selected_mode == 1 else 0.0),
        },
        "strategic_teacher_action": teacher_descriptor,
        "actor_teacher_agreement": agreement,
        "critic_value_prediction": value,
        "old_log_probability": old_lp_value,
        "ppo_probability_ratio": ratio_value,
        "ppo_clipping_occurred": ppo_clipped,
        "reference_action": reference_descriptor,
        "reference_action_probability": (
            None
            if reference_log_probability is None
            else math.exp(reference_log_probability)
        ),
        "reference_action_log_probability": reference_log_probability,
    }


def tower_damage(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None, *, player: int) -> int:
    """Return positive damage dealt to ``player`` between two snapshots."""

    if not before or not after:
        return 0
    before_rows = before.get("tower_hp", {}).get(f"player_{player}", {})
    after_rows = after.get("tower_hp", {}).get(f"player_{player}", {})
    damage = 0
    for role, row in before_rows.items():
        if not isinstance(row, Mapping):
            continue
        old_hp = row.get("hp")
        new_hp = after_rows.get(role, {}).get("hp") if isinstance(after_rows.get(role), Mapping) else None
        if type(old_hp) is int and type(new_hp) is int:
            damage += max(0, old_hp - new_hp)
    return damage


def classify_decision(row: Mapping[str, Any]) -> list[str]:
    """Conservative, evidence-labelled categories for a suspicious decision.

    A teacher disagreement is not, by itself, a gameplay failure, and a play
    that does not damage a tower during the next 250 ms is not necessarily
    unnecessary.  The evaluator therefore only emits head/timing/failure
    labels when a reference action or an explicit invalid transition exists.
    Threat labels are attached to an already suspicious row as context; they
    are not counted as failures for every ordinary defensive state.
    """

    categories: list[str] = []
    policy = row.get("policy", {})
    if not isinstance(policy, Mapping):
        policy = {}
    action = policy.get("executed_action") or {
        "mode": row.get("mode"),
        "card_slot": row.get("card_slot"),
        "world_cell": row.get("world_cell"),
    }
    reference = policy.get("reference_action")
    if policy.get("actor_teacher_agreement") is False:
        categories.append("teacher_disagreement")
    if isinstance(action, Mapping) and isinstance(reference, Mapping):
        if action.get("mode") != reference.get("mode"):
            categories.append("mode-head-regression")
            if action.get("mode") == "PLAY" and reference.get("mode") == "WAIT":
                # A reference WAIT versus candidate PLAY identifies a timing
                # divergence, not its quality.  Waiting may preserve elixir
                # for a combination, while an earlier defensive deployment
                # may also be correct.  This one-step trace has no
                # counterfactual continuation, so keep the label neutral.
                categories.append("wait-to-play-divergence")
            elif action.get("mode") == "WAIT" and reference.get("mode") == "PLAY":
                categories.append("play-to-wait-divergence")
        elif action.get("mode") == "PLAY" and reference.get("mode") == "PLAY":
            if action.get("card_slot") != reference.get("card_slot"):
                categories.append("card-selection-head-regression")
            if action.get("world_cell") != reference.get("world_cell"):
                categories.append("placement-head-regression")
    if row.get("action_status") == "rejected":
        categories.append("invalid-action")
    before = row.get("state_before")
    suspicious_reference = isinstance(reference, Mapping)
    suspicious_teacher = policy.get("actor_teacher_agreement") is False
    if isinstance(before, Mapping):
        enemy_units = [unit for unit in before.get("units", ()) if isinstance(unit, Mapping) and unit.get("owner") != row.get("target_player", 0)]
        near = [unit for unit in enemy_units if isinstance(unit.get("y_mtile"), int) and unit["y_mtile"] >= 15000]
        if near and (suspicious_reference or suspicious_teacher):
            if isinstance(action, Mapping) and action.get("mode") == "WAIT":
                categories.append("missed-defense")
            elif isinstance(action, Mapping) and action.get("mode") == "PLAY" and action.get("card_slot") is not None:
                categories.append("threat-response")
            if any("air" in str(unit.get("card_id", "")).casefold() for unit in near):
                categories.append("air-threat-response")
            else:
                categories.append("ground-threat-response")
        player = row.get("target_player", 0)
        players = before.get("players", ())
        player_row = players[player] if isinstance(players, list) and isinstance(player, int) and player < len(players) else {}
        if (
            suspicious_reference
            and isinstance(player_row, Mapping)
            and isinstance(action, Mapping)
            and action.get("mode") == "PLAY"
            and isinstance(reference, Mapping)
            and reference.get("mode") == "WAIT"
            and isinstance(player_row.get("elixir_milli"), int)
            and player_row["elixir_milli"] >= 8000
        ):
            categories.append("potential-elixir-overcommitment")
        if (
            suspicious_reference
            and isinstance(action, Mapping)
            and action.get("mode") == "PLAY"
            and isinstance(reference, Mapping)
            and reference.get("mode") == "WAIT"
            and not near
            and row.get("tower_damage_to_opponent", 0) == 0
        ):
            categories.append("potential-unnecessary-action")
    return sorted(set(categories))


def annotate_trace(rows: Sequence[Mapping[str, Any]], *, target_player: int = 0) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Add classifications and return category counts."""

    annotated: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        item = dict(row)
        item["target_player"] = target_player
        before = item.get("state_before")
        after = item.get("state_after")
        item["tower_damage_to_opponent"] = tower_damage(before, after, player=1 - target_player)
        item["tower_damage_to_self"] = tower_damage(before, after, player=target_player)
        categories = classify_decision(item)
        item["suspicious_categories"] = categories
        for category in categories:
            counts[category] += 1
        annotated.append(item)
    return annotated, dict(sorted(counts.items()))


def ppo_ratio_diagnostics(old_log_probs: Any, new_log_probs: Any, advantages: Any, clip_epsilon: float) -> tuple[Any, Any]:
    """Return per-transition PPO ratio and exact objective clipping mask."""

    import torch

    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = ((advantages >= 0) & (ratio > 1.0 + clip_epsilon)) | ((advantages < 0) & (ratio < 1.0 - clip_epsilon))
    return ratio, clipped


def explained_variance(values: Any, returns: Any) -> float | None:
    """Compute the standard value-function explained variance."""

    import torch

    values = values.detach().float().reshape(-1)
    returns = returns.detach().float().reshape(-1)
    variance = torch.var(returns, unbiased=False)
    if not bool(torch.isfinite(variance).item()) or float(variance.item()) <= 1e-12:
        return None
    result = 1.0 - torch.var(returns - values, unbiased=False) / variance
    return _finite(result)


__all__ = [
    "DIAGNOSTIC_TRACE_SCHEMA_VERSION",
    "action_equal",
    "annotate_trace",
    "build_policy_diagnostics",
    "explained_variance",
    "ppo_ratio_diagnostics",
    "serialize_action_masks",
    "state_snapshot",
    "tower_damage",
]
