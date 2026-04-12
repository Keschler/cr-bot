"""Full-spawn troop HP for YOLO troop labels at displayed level 16.

The map keys match the troop labels used by the KataCR/YOLO model in this
repo. Use these values when a troop has no detectable HP bar and should be
treated as being at full health.

Most values were derived from RoyaleAPI's current static Clash Royale card and
character datasets by converting internal rarity-based stat ladders to
displayed card level 16:

- Common: index 15
- Rare: index 13
- Epic: index 10
- Legendary: index 7
- Champion: index 5

Notes:
- Evolution labels currently reuse the base troop HP.
- ``phoenix-big``, ``phoenix-small``, and ``phoenix-egg`` use their distinct
  form HP values.
- ``little-prince`` and ``royal-guardian`` are manual level-16 values because
  the RoyaleAPI troop dataset in use here does not expose those labels
  directly.
- ``hog`` is an ambiguous KataCR label with no direct current stat entry; it
  is mapped to the standalone hog unit HP used by ``royal-hog``.
"""


TROOP_HP_LEVEL16: dict[str, int] = {
    "archer": 486,
    "archer-evolution": 486,
    "archer-queen": 1600,
    "baby-dragon": 1843,
    "balloon": 2688,
    "bandit": 1447,
    "barbarian": 1071,
    "barbarian-evolution": 1071,
    "bat": 130,
    "bat-evolution": 130,
    "battle-healer": 2745,
    "battle-ram": 1545,
    "battle-ram-evolution": 1545,
    "bomber": 531,
    "bomber-evolution": 531,
    "bowler": 3328,
    "cannon-cart": 1428,
    "dark-prince": 1920,
    "dart-goblin": 416,
    "electro-dragon": 1520,
    "electro-giant": 6169,
    "electro-spirit": 368,
    "electro-wizard": 1138,
    "elite-barbarian": 2143,
    "elixir-golem-big": 2508,
    "elixir-golem-mid": 1220,
    "elixir-golem-small": 576,
    "executioner": 2048,
    "fire-spirit": 368,
    "firecracker": 486,
    "firecracker-evolution": 486,
    "fisherman": 1389,
    "flying-machine": 983,
    "giant": 6542,
    "giant-skeleton": 5478,
    "goblin": 323,
    "goblin-brawler": 1644,
    "goblin-giant": 5337,
    "golden-knight": 2880,
    "golem": 8192,
    "golemite": 1664,
    "guard": 130,
    "heal-spirit": 369,
    "hog": 1339,
    "hog-rider": 2712,
    "hunter": 1341,
    "ice-golem": 1915,
    "ice-spirit": 368,
    "ice-spirit-evolution": 368,
    "ice-wizard": 1098,
    "inferno-dragon": 2065,
    "knight": 2822,
    "knight-evolution": 2822,
    "lava-hound": 6079,
    "lava-pup": 345,
    "little-prince": 1026,
    "lumberjack": 2045,
    "magic-archer": 849,
    "mega-knight": 6369,
    "mega-minion": 1339,
    "mighty-miner": 3600,
    "miner": 1930,
    "mini-pekka": 2176,
    "minion": 368,
    "monk": 3200,
    "mother-witch": 849,
    "musketeer": 1152,
    "night-witch": 1447,
    "pekka": 6016,
    "phoenix-big": 1679,
    "phoenix-egg": 382,
    "phoenix-small": 1343,
    "prince": 3072,
    "princess": 416,
    "ram-rider": 945,
    "rascal-boy": 3100,
    "rascal-girl": 417,
    "royal-ghost": 1930,
    "royal-giant": 4908,
    "royal-giant-evolution": 4908,
    "royal-guardian": 2556,
    "royal-hog": 1339,
    "royal-recruit": 850,
    "royal-recruit-evolution": 850,
    "skeleton": 130,
    "skeleton-barrel": 850,
    "skeleton-dragon": 899,
    "skeleton-evolution": 130,
    "skeleton-king": 3680,
    "sparky": 2316,
    "spear-goblin": 212,
    "valkyrie": 3051,
    "valkyrie-evolution": 3051,
    "wall-breaker": 529,
    "wall-breaker-evolution": 529,
    "witch": 1341,
    "wizard": 1152,
    "zappy": 847,
}


def get_troop_hp_level16(label: str) -> int | None:
    """Return full-spawn HP for a YOLO troop label at level 16."""

    return TROOP_HP_LEVEL16.get(label)
