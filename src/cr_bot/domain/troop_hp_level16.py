"""Full-spawn unit HP for YOLO detector labels at displayed level 16.

This module keeps the existing detector-label lookup API, but now derives HP
from ``card_metadata.py`` using the shared detector-label-to-card mapping.

Notes:
- Keys match the unit labels emitted by the YOLO/KataCR detector.
- Where a detector label maps cleanly to a card in ``CARD_METADATA``, the
  value comes directly from that card's ``hitpoints`` field.
- A small fallback table remains for split-unit forms or detector-only labels
  that do not exist as top-level cards in ``CARD_METADATA``.
"""

from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD


FALLBACK_HP_LEVEL16: dict[str, int] = {
    "elixir-golem-big": 2508,
    "elixir-golem-mid": 1220,
    "elixir-golem-small": 576,
    "goblin-brawler": 1644,
    "golemite": 1664,
    "hog": 1339,
    "lava-pup": 345,
    "phoenix-egg": 382,
    "phoenix-small": 1343,
    "rascal-boy": 3100,
    "rascal-girl": 417,
    "royal-guardian": 2556,
    "skeleton": 130,
    "skeleton-evolution": 130,
}


def _build_unit_hp_level16() -> dict[str, int]:
    hp_by_label = dict(FALLBACK_HP_LEVEL16)

    for label, card_key in DIRECT_UNIT_TO_CARD.items():
        if card_key is None:
            continue
        metadata = CARD_METADATA.get(card_key)
        if metadata is None:
            raise KeyError(f"{card_key} is not included in CARD_METADATA")

        hitpoints = metadata.get("hitpoints")
        if hitpoints is None:
            continue

        hp_by_label[label] = int(hitpoints)

    return hp_by_label


UNIT_HP_LEVEL16: dict[str, int] = _build_unit_hp_level16()


def get_unit_hp_level16(label: str) -> int | None:
    """Return full-spawn HP for a YOLO detector label at level 16."""

    return UNIT_HP_LEVEL16.get(label)
