from __future__ import annotations

from dataclasses import dataclass

from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.features.action_space import ACTION_GRID


@dataclass(frozen=True, slots=True)
class SpellTargetObservation:
    card: str
    time_left_s: float
    phase: str
    quality: float
    cell: tuple[int, int] | None
    center_x: float
    center_y: float

    @property
    def key(self) -> str:
        cell = self.cell if self.cell is not None else ("none", "none")
        return (
            f"{self.card}:{self.phase}:{self.time_left_s:.2f}:"
            f"{cell[0]}:{cell[1]}:{self.center_x:.1f}:{self.center_y:.1f}"
        )


def sense_spell_target_observations(
    locator,
    *,
    frame,
    arena_px,
    card: str,
    time_left_s: float,
) -> list[SpellTargetObservation]:
    if frame is None or arena_px is None:
        return []

    cost = CARD_METADATA.get(card, {}).get("elixir_cost")
    deploy = locator.locate(frame, arena_px, card, cost)
    release = locator.locate_released(frame, arena_px, card, cost)
    observations: list[SpellTargetObservation] = []

    if deploy is not None:
        observations.append(
            SpellTargetObservation(
                card=card,
                time_left_s=time_left_s,
                phase="aim",
                quality=0.6,
                cell=ACTION_GRID.pixel_to_cell(deploy.center_x, deploy.center_y, arena_px),
                center_x=deploy.center_x,
                center_y=deploy.center_y,
            )
        )

    if release is not None:
        observations.append(
            SpellTargetObservation(
                card=card,
                time_left_s=time_left_s,
                phase="release",
                quality=1.0,
                cell=ACTION_GRID.pixel_to_cell(release.center_x, release.center_y, arena_px),
                center_x=release.center_x,
                center_y=release.center_y,
            )
        )

    return observations
