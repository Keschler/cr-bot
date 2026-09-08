from __future__ import annotations

from typing import Any

from ..models.frames import FrontendFrame
from ..runners.pump import _try_import_decide_with_scores
from ._common import CorrectionUnprocessableError, _finite_number


def _live_policy_imports() -> tuple[Any, Any, Any]:
    """Lazily resolve (observation_fn, decide_fn, action_to_dict_fn)."""
    try:
        from simulator.physical_lab.policy_bridge import (
            observation_v2_from_game_state,
        )
    except ImportError:
        try:
            from physical_lab.policy_bridge import (  # type: ignore
                observation_v2_from_game_state,
            )
        except ImportError as error:
            raise CorrectionUnprocessableError(
                "policy bridge is not importable"
            ) from error
    decide_fn = _try_import_decide_with_scores()
    if decide_fn is None:
        raise CorrectionUnprocessableError("scored decision entry point is missing")
    try:
        from simulator.physical_lab.prototype_controller import action_to_dict
    except ImportError:
        try:
            from physical_lab.prototype_controller import (  # type: ignore
                action_to_dict,
            )
        except ImportError as error:
            raise CorrectionUnprocessableError(
                "action serializer is not importable"
            ) from error
    return observation_v2_from_game_state, decide_fn, action_to_dict


def _corrected_observation(frame: FrontendFrame, rows: list[dict]) -> Any:
    """Rebuild a V2 observation for edited rows (pure; no tracker/state)."""
    try:
        from cr_bot.domain.game_state import (
            Detection,
            GameState,
            HudState,
            Match,
            PrincessTowerState,
        )
    except ImportError as error:
        raise CorrectionUnprocessableError(
            "game-state models are not importable"
        ) from error
    record = frame.record if isinstance(frame.record, dict) else {}
    visual = record.get("visual_state")
    if not isinstance(visual, dict):
        raise CorrectionUnprocessableError("frame has no extracted visual state")
    time_left = _finite_number(visual.get("time_left_s"))
    if time_left is None:
        raise CorrectionUnprocessableError("frame has no readable match clock")
    arena = visual.get("arena_px")
    if (
        not isinstance(arena, (list, tuple))
        or len(arena) != 4
        or any(_finite_number(v) is None for v in arena)
    ):
        raise CorrectionUnprocessableError("frame has no arena calibration")
    hand = visual.get("hand")
    hand_cards = [
        (hand[i] if isinstance(hand, (list, tuple)) and i < len(hand) else None)
        for i in range(4)
    ]
    elixir = _finite_number(visual.get("elixir"))
    if elixir is None:
        raise CorrectionUnprocessableError("frame has no elixir reading")

    def triplet(key: str) -> list:
        values = visual.get(key)
        if not isinstance(values, (list, tuple)):
            raise CorrectionUnprocessableError(f"frame has no {key}")
        out = list(values)[:3]
        while len(out) < 3:
            out.append(None)
        return out

    hp_self = triplet("tower_hp_self")
    hp_enemy = triplet("tower_hp_enemy")

    def alive(value: Any) -> bool:
        number = _finite_number(value)
        return number is not None and number > 0

    seen: list[int] = []
    raw_seen = visual.get("seen_enemy_cards")
    if isinstance(raw_seen, (list, tuple, set)):
        for card in raw_seen:
            try:
                seen.append(int(card))
            except (TypeError, ValueError, OverflowError):
                continue
    own_units: list = []
    enemy_units: list = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        team = str(row.get("team", "")).strip().lower()
        if team not in ("ally", "enemy"):
            raise CorrectionUnprocessableError("edited row has an invalid team")
        try:
            box = [float(v) for v in row["box"]]
            center = [float(v) for v in row["center"]]
        except (TypeError, ValueError, KeyError):
            raise CorrectionUnprocessableError("edited row has an invalid box")
        detection = Detection(
            track_id=row.get("track_id"),
            class_name=str(row.get("class_name")),
            team=team,
            confidence=float(row.get("confidence", 1.0)),
            x1=box[0],
            y1=box[1],
            x2=box[2],
            y2=box[3],
            center_x=center[0],
            center_y=center[1],
            estimated_hp=row.get("estimated_hp"),
        )
        match = Match(troop=detection, bar=None)
        (own_units if team == "ally" else enemy_units).append(match)
    game_state = GameState(
        hud=HudState(
            time_left_s=time_left,
            overtime=bool(visual.get("overtime", False)),
            elixir_self=elixir,
            hand_cards=hand_cards,
            next_card=visual.get("next_card"),
            tower_hp_self=hp_self,
            tower_hp_enemy=hp_enemy,
            princess_towers=PrincessTowerState(
                own_left_alive=alive(hp_self[0]),
                own_right_alive=alive(hp_self[2]),
                enemy_left_alive=alive(hp_enemy[0]),
                enemy_right_alive=alive(hp_enemy[2]),
            ),
        ),
        total_remaining_s=_finite_number(visual.get("total_remaining_s")) or time_left,
        own_units=own_units,
        enemy_units=enemy_units,
        seen_enemy_cards=seen,
        elixir_enemy_est=_finite_number(visual.get("enemy_elixir_est")) or 0.0,
        own_king_active=bool(visual.get("own_king_active", False)),
        enemy_king_active=bool(visual.get("enemy_king_active", False)),
        started=True,
    )
    observation_fn, _, _ = _live_policy_imports()
    try:
        observation = observation_fn(
            game_state, arena_px=tuple(float(v) for v in arena), legal_wait=True
        )
    except Exception as error:
        raise CorrectionUnprocessableError(
            f"edited state cannot be turned into an observation: {error}"
        ) from error
    if observation is None:
        raise CorrectionUnprocessableError("edited state yields no observation")
    return observation
