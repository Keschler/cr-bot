"""HP-based size buckets for troop YOLO classes.

These buckets are intended for gameplay grouping rather than exact stat
reproduction. The labels are based on approximate battlefield HP tiers:

- ``small``: fragile units, roughly under 1000 HP
- ``middle``: standard fighters, roughly 1000-1999 HP
- ``large``: tanks and heavy units, roughly 2000+ HP

Non-troop YOLO classes such as towers, buildings, spells, HUD markers, and
projectile-only objects are intentionally excluded.
"""

TROOP_HP_SIZE = {
    "archer": "small",
    "archer-evolution": "small",
    "archer-queen": "middle",
    "baby-dragon": "middle",
    "bandit": "small",
    "barbarian": "small",
    "barbarian-evolution": "small",
    "battle-healer": "middle",
    "battle-ram": "middle",
    "battle-ram-evolution": "middle",
    "bat": "small",
    "bat-evolution": "small",
    "balloon": "middle",
    "bomber": "small",
    "bomber-evolution": "small",
    "bowler": "middle",
    "cannon-cart": "middle",
    "dark-prince": "middle",
    "dart-goblin": "small",
    "electro-dragon": "middle",
    "electro-giant": "large",
    "electro-spirit": "small",
    "electro-wizard": "small",
    "elite-barbarian": "small",
    "elixir-golem-big": "large",
    "elixir-golem-mid": "middle",
    "elixir-golem-small": "small",
    "executioner": "middle",
    "fire-spirit": "small",
    "firecracker": "small",
    "firecracker-evolution": "small",
    "fisherman": "small",
    "flying-machine": "small",
    "giant": "large",
    "giant-skeleton": "large",
    "goblin": "small",
    "goblin-brawler": "small",
    "goblin-giant": "large",
    "golemite": "middle",
    "golden-knight": "middle",
    "golem": "large",
    "guard": "small",
    "heal-spirit": "small",
    "hog": "small",
    "hog-rider": "middle",
    "hunter": "small",
    "ice-golem": "middle",
    "ice-spirit": "small",
    "ice-spirit-evolution": "small",
    "ice-wizard": "small",
    "inferno-dragon": "middle",
    "knight": "middle",
    "knight-evolution": "middle",
    "lava-hound": "large",
    "lava-pup": "middle",
    "little-prince": "small",
    "lumberjack": "middle",
    "magic-archer": "small",
    "mega-knight": "large",
    "mega-minion": "small",
    "mighty-miner": "large",
    "miner": "middle",
    "mini-pekka": "middle",
    "minion": "small",
    "monk": "large",
    "mother-witch": "small",
    "musketeer": "small",
    "night-witch": "small",
    "pekka": "large",
    "phoenix-big": "middle",
    "phoenix-egg": "middle",
    "phoenix-small": "middle",
    "princess": "small",
    "prince": "middle",
    "ram-rider": "middle",
    "rascal-boy": "large",
    "rascal-girl": "small",
    "royal-ghost": "small",
    "royal-guardian": "middle",
    "royal-giant": "large",
    "royal-giant-evolution": "large",
    "royal-hog": "small",
    "royal-recruit": "large",
    "royal-recruit-evolution": "large",
    "skeleton": "small",
    "skeleton-barrel": "middle",
    "skeleton-dragon": "small",
    "skeleton-evolution": "small",
    "skeleton-king": "large",
    "sparky": "middle",
    "spear-goblin": "small",
    "valkyrie": "middle",
    "valkyrie-evolution": "middle",
    "wall-breaker": "small",
    "wall-breaker-evolution": "small",
    "witch": "small",
    "wizard": "small",
    "zappy": "small",
}

SMALL_TROOPS = sorted(
    troop for troop, size in TROOP_HP_SIZE.items() if size == "small"
)
MIDDLE_TROOPS = sorted(
    troop for troop, size in TROOP_HP_SIZE.items() if size == "middle"
)
LARGE_TROOPS = sorted(
    troop for troop, size in TROOP_HP_SIZE.items() if size == "large"
)
ALL_TROOPS = sorted(TROOP_HP_SIZE)


def get_troop_hp_size(label: str) -> str | None:
    return TROOP_HP_SIZE.get(label)
