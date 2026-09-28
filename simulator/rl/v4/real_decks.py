"""Pinned, source-linked base-card opponent decks for V4 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from simulator.ruleset import load_fixed_ruleset


CATALOG_PATH = Path(__file__).with_name("real_decks.json")
REAL_DECK_CATALOG_VERSION = 1
REAL_DECK_ARCHETYPES = (
    "aggressive-pressure", "defensive-cycle", "beatdown", "air-beatdown", "siege-bait"
)


@dataclass(frozen=True, slots=True)
class RealDeck:
    deck_id: str
    split: str
    archetype: str
    cards: tuple[str, ...]
    source_url: str


def catalog_sha256() -> str:
    return hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()


def _source_cards(url: str) -> tuple[str, ...]:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if parsed.scheme != "https" or parsed.netloc not in {
        "royaleapi.com", "www.deckshop.pro"
    }:
        raise ValueError(f"unapproved real-deck source: {url}")
    if parsed.netloc == "royaleapi.com":
        prefix = "/decks/stats/"
        if not path.startswith(prefix):
            raise ValueError(f"source is not a RoyaleAPI deck page: {url}")
    else:
        prefix = "/deck/detail/"
        if not path.startswith(prefix):
            raise ValueError(f"source is not a Deck Shop deck page: {url}")
    slug = unquote(path[len(prefix):])
    if "/" in slug:
        raise ValueError(f"source must point to one deck: {url}")
    cards = tuple("log" if card == "the-log" else card for card in slug.split(","))
    if len(cards) != 8 or any(
        card.startswith(("evo-", "hero-")) or card.endswith(("-ev1", "-hero"))
        for card in cards
    ):
        raise ValueError(f"source deck has unsupported card forms: {url}")
    return cards


@lru_cache(maxsize=1)
def load_real_decks() -> tuple[RealDeck, ...]:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if payload.get("version") != REAL_DECK_CATALOG_VERSION:
        raise ValueError("unsupported real-deck catalog version")
    if not isinstance(payload.get("captured_on"), str) or not payload["captured_on"]:
        raise ValueError("real-deck catalog needs a capture date")
    ruleset = load_fixed_ruleset()
    rows: list[RealDeck] = []
    identifiers: set[str] = set()
    identities: set[frozenset[str]] = set()
    for raw in payload["decks"]:
        row = RealDeck(
            deck_id=str(raw["id"]), split=str(raw["split"]),
            archetype=str(raw["archetype"]), cards=tuple(raw["cards"]),
            source_url=str(raw["source_url"]),
        )
        identity = frozenset(row.cards)
        if not row.deck_id or row.deck_id in identifiers:
            raise ValueError(f"duplicate or empty real-deck ID: {row.deck_id}")
        if row.split not in ("train", "eval") or row.archetype not in REAL_DECK_ARCHETYPES:
            raise ValueError(f"invalid split or archetype for real deck {row.deck_id}")
        if len(row.cards) != 8 or len(identity) != 8:
            raise ValueError(f"real deck {row.deck_id} must have eight unique cards")
        if any(card not in ruleset.interaction_set for card in row.cards):
            raise ValueError(f"real deck {row.deck_id} contains unsupported cards")
        if identity != frozenset(_source_cards(row.source_url)):
            raise ValueError(f"real deck {row.deck_id} differs from its source URL")
        if identity in identities:
            raise ValueError(f"real deck {row.deck_id} duplicates another deck")
        identifiers.add(row.deck_id)
        identities.add(identity)
        rows.append(row)
    for split in ("train", "eval"):
        for archetype in REAL_DECK_ARCHETYPES:
            if not any(row.split == split and row.archetype == archetype for row in rows):
                raise ValueError(f"real catalog lacks {split}/{archetype}")
    return tuple(rows)


def real_decks_for(split: str, archetype: str) -> tuple[RealDeck, ...]:
    if split not in ("train", "eval") or archetype not in REAL_DECK_ARCHETYPES:
        raise ValueError("unknown real-deck split or archetype")
    return tuple(
        deck for deck in load_real_decks()
        if deck.split == split and deck.archetype == archetype
    )


def select_real_deck(split: str, archetype: str, match_index: int) -> RealDeck:
    if type(match_index) is not int or match_index < 0:
        raise ValueError("match_index must be a non-negative integer")
    choices = real_decks_for(split, archetype)
    return choices[match_index % len(choices)]
