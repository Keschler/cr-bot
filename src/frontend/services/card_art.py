from __future__ import annotations

import re
from pathlib import Path

from .paths import REPO_ROOT


CARD_ART_CANDIDATES = (
    REPO_ROOT / "assets/templates/cr-api-assets/cards-gold",
    REPO_ROOT / "capture/templates/cr-api-assets/cards-gold",
)
# Extractor/ruleset names that differ from the card-art file names.
CARD_ART_ALIASES = {"log": "the-log"}
_CARD_SAFE_RE = re.compile(r"[^a-z0-9-]+")


def _card_art_dir() -> Path | None:
    for candidate in CARD_ART_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return None


def _card_art_candidates(name: str) -> list[str]:
    norm = (name or "").strip().lower().replace("_", "-").replace(" ", "-")
    norm = _CARD_SAFE_RE.sub("", norm).strip("-")
    if not norm:
        return []
    options = [norm]
    alias = CARD_ART_ALIASES.get(norm)
    if alias and alias not in options:
        options.append(alias)
    return options
