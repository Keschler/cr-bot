from __future__ import annotations

from pathlib import Path

CARD_PREFIXES = (
    "card_champion_",
    "card_common_",
    "card_epic_",
    "card_evolution_",
    "card_legendary_",
    "card_rare_",
)

FOLDER_ALIASES = {
    "angry_barbarian": "elite-barbarians",
    "archer": "archers",
    "arrow": "arrows",
    "assassin": "bandit",
    "axeman": "executioner",
    "babydragon": "baby-dragon",
    "barbarian": "barbarians",
    "bat": "bats",
    "bats": "bats",
    "battleram": "battle-ram",
    "blowdart_goblin": "dart-goblin",
    "bombtower": "bomb-tower",
    "dark_witch": "night-witch",
    "dart_gob": "dart-goblin",
    "ghost": "royal-ghost",
    "goblin_ref": "goblins",
    "goblindrill": "goblin-drill",
    "goblins": "goblins",
    "lava_golem": "lava-hound",
    "log": "log",
    "megaminion": "mega-minion",
    "minion": "minion-horde",
    "minipekka": "mini-pekka",
    "moving_cannon": "cannon-cart",
    "motherwitch": "mother-witch",
    "ramrider": "ram-rider",
    "ragebarbarian": "lumberjack",
    "recruit": "royal-recruits",
    "recruits": "royal-recruits",
    "royal_hog": "royal-hogs",
    "rascal": "rascals",
    "skeleton": "skeletons",
    "skeleton_balloon": "skeleton-barrel",
    "skeleton_dragon": "skeleton-dragons",
    "snowball": "giant-snowball",
    "spear_goblin": "spear-goblins",
    "superhog_rider": "hog-rider",
    "superminipekka": "mini-pekka",
    "wallbreaker": "wall-breakers",
    "xbow": "x-bow",
    "zapmachine": "zappies",
    "minizapmachine": "zappies",
}

FOLDER_MULTI_ALIASES = {
    "barbarian": ["barbarians", "barbarian-hut"],
    "dark_magic": ["void"],
    "goblin_curse": ["goblin-curse", "vines"],
    "goblin_science": ["goblinstein"],
    "gob_bush": ["suspicious-bush"],
    "goblins": ["goblins", "goblin-gang"],
    "minion": ["minions", "minion-horde"],
    "minizapmachine": ["zappies", "sparky"],
    "musketeer": ["musketeer", "three-musketeers"],
    "skeleton": ["skeletons", "skeleton-army"],
    "skeleton_warrior": ["guards"],
    "spear_goblin": ["spear-goblins", "goblin-gang"],
    "zapmachine": ["zappies", "sparky"],
}

GROUND_TRUTH_ALIASES = {
    "evo-dart-goblin": "dart-goblin",
    "evo-knight": "knight",
    "minion-hord": "minion-horde",
    "old-musketeer": "musketeer",
    "the-log": "log",
}


def folder_to_card_key(folder: str | Path) -> str | None:
    keys = folder_to_card_keys(folder)
    return keys[0] if keys else None


def folder_to_card_keys(folder: str | Path) -> list[str]:
    name = Path(folder).name
    if name.startswith("card_tower_"):
        return []
    for prefix in CARD_PREFIXES:
        if name.startswith(prefix):
            raw = name.removeprefix(prefix)
            if raw in FOLDER_MULTI_ALIASES:
                return FOLDER_MULTI_ALIASES[raw]
            return [FOLDER_ALIASES.get(raw, raw.replace("_", "-"))]
    return []


def normalize_card_key(card: str) -> str:
    card = card.strip().lower().replace("_", "-")
    if card.startswith("evo-"):
        card = card.removeprefix("evo-")
    return GROUND_TRUTH_ALIASES.get(card, card)
