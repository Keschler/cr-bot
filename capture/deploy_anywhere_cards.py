"""
Clash Royale cards with map-wide placement.

This file contains the current troop and spell cards that can be played
anywhere on the battlefield, using project-friendly lowercase card IDs.

Notes:
- `mirror` is intentionally omitted.
- Spells with placement restrictions such as `the-log`, `barbarian-barrel`,
  and `royal-delivery` are not included.
"""

from __future__ import annotations

ANYWHERE_TROOPS = [
    "miner",
    "goblin-drill"
]

ANYWHERE_SPELLS = [
    "arrows",
    "clone",
    "earthquake",
    "fireball",
    "freeze",
    "giant-snowball",
    "goblin-barrel",
    "goblin-curse",
    "graveyard",
    "lightning",
    "poison",
    "rage",
    "rocket",
    "tornado",
    "void",
    "zap",
]


def get_anywhere_troops() -> list[str]:
    """Return a copy of the troop cards that can be deployed anywhere."""
    return ANYWHERE_TROOPS.copy()



def get_anywhere_spells() -> list[str]:
    """Return a copy of the spell cards that can be cast anywhere."""
    return ANYWHERE_SPELLS.copy()
