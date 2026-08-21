"""Typed actions accepted by the headless simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


Cell = tuple[int, int]  # (column, row), matching cr_bot.domain.game_state.Action


@dataclass(frozen=True, slots=True)
class WaitAction:
    player: int


@dataclass(frozen=True, slots=True)
class PlayCardAction:
    player: int
    card_slot: int
    cell: Cell


@dataclass(frozen=True, slots=True)
class UseAbilityAction:
    player: int
    entity_uid: int


SimAction: TypeAlias = WaitAction | PlayCardAction | UseAbilityAction


def action_to_dict(action: SimAction) -> dict[str, object]:
    if isinstance(action, WaitAction):
        return {"kind": "wait", "player": action.player}
    if isinstance(action, PlayCardAction):
        return {
            "kind": "play",
            "player": action.player,
            "card_slot": action.card_slot,
            "cell": list(action.cell),
        }
    return {
        "kind": "ability",
        "player": action.player,
        "entity_uid": action.entity_uid,
    }


def action_from_dict(raw: dict[str, object]) -> SimAction:
    kind = str(raw.get("kind", "")).lower()
    player_raw = raw["player"]
    if type(player_raw) is not int:
        raise ValueError("action player must be integer 0 or 1")
    player = player_raw
    if kind == "wait":
        return WaitAction(player)
    if kind == "play":
        cell = raw.get("cell")
        if (
            not isinstance(cell, list)
            or len(cell) != 2
            or any(type(value) is not int for value in cell)
        ):
            raise ValueError("play action cell must be [column, row]")
        slot = raw["card_slot"]
        if type(slot) is not int:
            raise ValueError("play action card_slot must be an integer")
        return PlayCardAction(player, slot, (cell[0], cell[1]))
    if kind == "ability":
        uid = raw["entity_uid"]
        if type(uid) is not int:
            raise ValueError("ability entity_uid must be an integer")
        return UseAbilityAction(player, uid)
    raise ValueError(f"unknown action kind: {kind!r}")
