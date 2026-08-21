"""Build the versioned opponent-card ruleset from the checked-in card catalog.

The base ``2026-08-04`` ruleset is intentionally small and high-confidence.  V1
also needs an executable *roster* surface so scenario generation cannot silently
skip an opponent card.  This module creates that surface from the repository's
Level-11 card catalog and labels every generated field as provisional.  It is
therefore useful for exercising dispatch, observations, placement, and fuzzing,
but it cannot satisfy the training-readiness gate until exact stats and
mechanic-specific evidence replace the generated values.

The output is deterministic JSON.  It is generated once and checked in as
``rulesets/2026-08-04-roster.json``; runtime simulation never reaches for the
network or imports this builder implicitly.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from cr_bot.domain.card_metadata import CARD_METADATA

from .data_reconciliation import (
    LEVEL11_SOURCE_ID,
    apply_official_overrides,
    load_level11_source,
    load_official_overrides,
    official_override_rows,
    source_sha256,
)
from .roster import PLAYER_DECK, load_opponent_roster
from .ruleset import FIXED_RULESET_ID, calculate_content_hash, ruleset_path


ROSTER_RULESET_ID = "2026-08-04-roster"
CATALOG_SOURCE_ID = "local-card-metadata-2026-08-14"
CATALOG_GENERATED_AT = "2026-08-14"
GOBLIN_MACHINE_SOURCE_ID = "royaleapi-goblin-machine-2024"
LEVEL11_SOURCE_PAYLOAD = load_level11_source()
DECKSHOP_SOURCE_ID = "deckshop-battle-healer-2026-08-14"
DECKSHOP_SOURCE_PATH = Path(__file__).resolve().parent / "sources" / "deckshop_level11.json"
DECKSHOP_CORE_SOURCE_ID = "deckshop-core-level11-2026-08-15"
DECKSHOP_CORE_SOURCE_PATH = Path(__file__).resolve().parent / "sources" / "deckshop_core_level11.json"
DECKSHOP_CORE_SOURCE_PAYLOAD = json.loads(DECKSHOP_CORE_SOURCE_PATH.read_text(encoding="utf-8"))
DECKSHOP_HEAL_SPIRIT_SOURCE_ID = "deckshop-heal-spirit-2026-08-16"
DECKSHOP_HEAL_SPIRIT_SOURCE_PATH = (
    Path(__file__).resolve().parent / "sources" / "deckshop_heal_spirit_level11.json"
)
DECKSHOP_HEAL_SPIRIT_SOURCE_PAYLOAD = json.loads(
    DECKSHOP_HEAL_SPIRIT_SOURCE_PATH.read_text(encoding="utf-8")
)
SPLIT_SOURCE_ID = "deckmelon-split-level11-2026-08-16"
SPLIT_SOURCE_PATH = (
    Path(__file__).resolve().parent / "sources" / "deckmelon_split_level11.json"
)
SPLIT_SOURCE_PAYLOAD = json.loads(SPLIT_SOURCE_PATH.read_text(encoding="utf-8"))
GOBLIN_BRAWLER_SOURCE_ID = "deckmelon-goblin-cage-level11-2026-08-16"
GOBLIN_BRAWLER_SOURCE_PATH = (
    Path(__file__).resolve().parent / "sources" / "deckmelon_goblin_cage_level11.json"
)
GOBLIN_BRAWLER_SOURCE_PAYLOAD = json.loads(
    GOBLIN_BRAWLER_SOURCE_PATH.read_text(encoding="utf-8")
)

# ``CARD_METADATA`` predates the simulator's explicit card-kind contract and
# has a few spawners represented as troops.  These are gameplay buildings in
# the V1 base-card surface.  Furnace is intentionally *not* in this set:
# Supercell's August 2025 rework made it a moving troop whose cauldron spawns
# Fire Spirits.  Keeping that exception here makes the source catalog
# immutable and leaves the correction auditable in the generated ruleset's
# provenance.
BUILDING_IDS = frozenset(
    {
        "barbarian-hut",
        "bomb-tower",
        "cannon",
        "elixir-collector",
        "goblin-cage",
        "goblin-drill",
        "goblin-hut",
        "inferno-tower",
        "mortar",
        "tesla",
        "tombstone",
        "x-bow",
    }
)

# Deterministic formations for the cards which deploy multiple bodies.  Cards
# not listed here remain one body; the list is deliberately conservative until
# child-card identity and spawn timing are sourced per card.
SPAWN_COUNTS: Mapping[str, int] = {
    "archers": 2,
    "barbarians": 5,
    "bats": 5,
    "elite-barbarians": 2,
    "goblin-gang": 5,
    "goblins": 3,
    "guards": 3,
    "minion-horde": 6,
    "minions": 3,
    "rascals": 3,
    "royal-hogs": 4,
    "royal-recruits": 6,
    "skeleton-army": 15,
    "spear-goblins": 3,
    "three-musketeers": 3,
    "zappies": 3,
}

# Level-11 shield pools.  DeckShop's current Level-11 pages report these
# values and explicitly model shields as a separate layer; a hit that breaks a
# shield does not spill its excess damage into body HP.
SHIELD_DEFINITIONS: Mapping[str, Mapping[str, int]] = {
    "dark-prince": {"hitpoints": 240},
    "guards": {"hitpoints": 256},
    "royal-recruits": {"hitpoints": 240},
}

# Mixed formations must create the actual child bodies, not clones of the
# parent card's aggregate stats.  The hidden child definitions are installed
# below and remain outside the playable interaction set.
SPAWN_CHILDREN_DEFINITIONS: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "goblin-gang": (
        {"card_id": "goblin", "count": 3},
        {"card_id": "spear-goblin", "count": 3},
    ),
    "rascals": (
        {"card_id": "rascal-boy", "count": 1},
        {"card_id": "rascal-girl", "count": 2},
    ),
}

DEATH_RAGE_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "lumberjack": {
        "duration_us": 6_000_000,
        "tick_interval_us": 100_000,
        "radius_mtile": 3_000,
        "speed_multiplier_milli": 1400,
        "hit_speed_multiplier_milli": 1400,
        "targets": ["air", "ground"],
    },
}

DEPLOY_EFFECT_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "electro-wizard": {
        "kind": "stun",
        "duration_us": 500_000,
        "radius_mtile": 1_000,
        "speed_multiplier_milli": 0,
        "hit_speed_multiplier_milli": 0,
        "targets": ["air", "ground", "building", "crown_tower"],
    },
    "ice-wizard": {
        "kind": "freeze",
        "duration_us": 1_500_000,
        "radius_mtile": 1_000,
        "speed_multiplier_milli": 0,
        "hit_speed_multiplier_milli": 0,
        "targets": ["air", "ground", "building", "crown_tower"],
    },
}

JUMP_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "mega-knight": {
        "min_range_mtile": 1_500,
        "max_range_mtile": 4_000,
        "duration_us": 400_000,
        "damage": 268,
        "radius_mtile": 1_300,
        "spawn_damage": True,
    },
}

SPAWNER_DEFINITIONS: Mapping[str, Mapping[str, int | str | None]] = {
    "barbarian-hut": {
        "card_id": "barbarians",
        # The pinned Level-11 source reports one three-Barbarian wave every
        # 14 seconds.  This used to be a 7-second placeholder, which doubled
        # the defensive pressure of every Barbarian Hut trace.
        "interval_us": 14_000_000,
        "start_delay_us": 1_000_000,
        "max_alive": 6,
        "count": 3,
    },
    "furnace": {
        "card_id": "fire-spirit",
        # August's official spawn-speed override is applied below as a
        # field-level provenance row; the generated component is kept in the
        # same fixed V1 table so runtime never needs the network.  The moving
        # Furnace emits one Spirit per cadence and has no sourced live-child
        # cap; ``None`` deliberately means unbounded while the parent lives.
        "interval_us": 5_000_000,
        "start_delay_us": 1_000_000,
        "max_alive": None,
        "count": 1,
    },
    "goblin-hut": {
        "card_id": "spear-goblins",
        "interval_us": 2_200_000,
        "start_delay_us": 1_000_000,
        "max_alive": 6,
        "count": 3,
    },
    "goblin-drill": {
        "card_id": "goblins",
        "interval_us": 3_500_000,
        "start_delay_us": 1_000_000,
        "max_alive": 6,
        "count": 1,
    },
    "tombstone": {
        "card_id": "skeletons",
        "interval_us": 3_500_000,
        "start_delay_us": 1_000_000,
        "max_alive": 8,
        "count": 2,
    },
    "witch": {
        "card_id": "skeletons",
        "interval_us": 7_000_000,
        "start_delay_us": 1_000_000,
        "max_alive": 8,
        "count": 4,
    },
    "night-witch": {
        "card_id": "bats",
        "interval_us": 5_000_000,
        "start_delay_us": 1_000_000,
        "max_alive": 8,
        "count": 2,
    },
}

STATUS_DEFINITIONS: Mapping[str, Mapping[str, int | str]] = {
    "freeze": {
        "kind": "freeze",
        "duration_us": 1_100_000,
        "speed_multiplier_milli": 0,
        "hit_speed_multiplier_milli": 0,
    },
    "giant-snowball": {
        "kind": "slow",
        "duration_us": 2_000_000,
        "speed_multiplier_milli": 500,
        "hit_speed_multiplier_milli": 500,
    },
    "ice-wizard": {
        "kind": "slow",
        "duration_us": 1_500_000,
        "speed_multiplier_milli": 650,
        "hit_speed_multiplier_milli": 650,
    },
    "poison": {
        "kind": "poison",
        "duration_us": 8_000_000,
        "speed_multiplier_milli": 1_000,
        "hit_speed_multiplier_milli": 1_000,
        "damage_per_tick": 100,
        "tick_interval_us": 1_000_000,
    },
    "zap": {
        "kind": "stun",
        "duration_us": 500_000,
        "speed_multiplier_milli": 0,
        "hit_speed_multiplier_milli": 0,
    },
    "lightning": {
        "kind": "stun",
        "duration_us": 500_000,
        "speed_multiplier_milli": 0,
        "hit_speed_multiplier_milli": 0,
    },
    # Both electric dragons apply the attack-reset stun on every bolt.  The
    # target selection is owned by the chain/multi-target components below;
    # keeping the status in the shared table ensures projectile and direct
    # impacts use the same deterministic status semantics.
    "electro-dragon": {
        "kind": "stun",
        "duration_us": 500_000,
        "speed_multiplier_milli": 0,
        "hit_speed_multiplier_milli": 0,
    },
    "electro-wizard": {
        "kind": "stun",
        "duration_us": 500_000,
        "speed_multiplier_milli": 0,
        "hit_speed_multiplier_milli": 0,
    },
}

# Targeting components are intentionally data-driven.  A normal splash attack
# damages every legal victim in a radius; these cards instead select discrete
# victims.  Electro Dragon hits the initial target and up to two additional
# targets, with each hop limited to its three-and-a-half-tile chain range.
# Electro Wizard fires two independent bolts and never splashes every unit in
# its three-tile sight radius.  Exact tie-breaking/bolt frame spacing remains
# a held-out-video unknown, but the component makes the strategic outcomes
# executable and testable instead of silently using AoE.
CHAIN_ATTACK_DEFINITIONS: Mapping[str, Mapping[str, int | str]] = {
    "electro-dragon": {
        "max_targets": 3,
        "chain_range_mtile": 3_500,
        "selection": "nearest",
    },
}

MULTI_TARGET_ATTACK_DEFINITIONS: Mapping[str, Mapping[str, int | str]] = {
    "electro-wizard": {
        "max_targets": 2,
        "selection": "nearest",
        "range_mtile": 5_000,
    },
}

REFLECTION_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    # Level-11 reflection values are kept separate from Electro Giant's
    # ordinary melee damage.  The current structured/third-party snapshot
    # reports 192 body and 128 Crown-Tower damage, a 3.5-tile zap radius, and
    # a half-second attack reset; held-out footage still owns exact immunity
    # and projectile-source edge cases.
    "electro-giant": {
        "damage": 192,
        "crown_tower_damage": 128,
        "radius_mtile": 3_500,
        "targets": ["air", "ground", "building", "crown_tower"],
        "stun_duration_us": 500_000,
    },
}

# Generic charge attacks share one movement/impact component.  The Level-11
# source supplies the charged hit values; the card-specific run thresholds and
# doubled medium movement speed are explicit, auditable V1 assumptions until
# high-confidence footage fits the exact charge trigger/reset frames.  Keeping
# this data out of card-specific engine branches means Prince, Dark Prince,
# Battle Ram, and Ram Rider receive identical state/serialization semantics.
CHARGE_ATTACK_DEFINITIONS: Mapping[str, Mapping[str, int | bool]] = {
    "prince": {
        "charge_distance_mtile": 2_500,
        "charged_speed_mtile_per_s": 2_400,
        "charge_damage": 783,
        "reset_on_hit": True,
    },
    "dark-prince": {
        "charge_distance_mtile": 3_000,
        "charged_speed_mtile_per_s": 2_400,
        "charge_damage": 532,
        "reset_on_hit": True,
    },
    "battle-ram": {
        "charge_distance_mtile": 3_500,
        "charged_speed_mtile_per_s": 2_400,
        "charge_damage": 573,
        "reset_on_hit": True,
    },
    "ram-rider": {
        "charge_distance_mtile": 2_500,
        "charged_speed_mtile_per_s": 2_400,
        "charge_damage": 501,
        "reset_on_hit": True,
    },
}

# Bandit is a movement dash rather than a normal charge: when an eligible
# target is inside the dash window she traverses to melee range in one
# deterministic movement step and the next impact uses the separate dash
# damage.  The exact visual dash frame/landing offset remains a video-fit
# unknown, so the executable values are isolated in this component.
DASH_DEFINITIONS: Mapping[str, Mapping[str, int | bool]] = {
    "bandit": {
        "dash_range_mtile": 6_000,
        "dash_damage": 388,
        "min_dash_distance_mtile": 1_000,
        "reset_on_hit": True,
    },
}

# Fisherman's long-range hook is represented independently from his short
# melee attack range.  The pull only moves troop bodies in V1; building/tower
# hook eligibility and exact reel timing remain explicit unknowns.
HOOK_DEFINITIONS: Mapping[str, Mapping[str, int | bool]] = {
    "fisherman": {
        "hook_range_mtile": 7_000,
        "pull_distance_mtile": 1_200,
        "pull_troops_only": True,
    },
}

# Inferno attacks ramp while their beam remains locked on one target.  The
# first value is active immediately after acquisition; later values begin at
# the elapsed-time thresholds.  These Level-11 schedules are isolated as a
# component because the ordinary ``damage`` field is only the stage-one
# value.  Target loss, hard crowd control, and retargeting reset the timer.
# Stage timing is a provisional, video-fit value and remains listed in the
# uncertainty ledger until clean pre-Evolution footage promotes it.
RAMP_ATTACK_DEFINITIONS: Mapping[str, Mapping[str, object]] = {
    "inferno-dragon": {
        "damage_schedule": [35, 120, 422],
        "stage_thresholds_us": [0, 2_000_000, 4_000_000],
        "reset_on_target_loss": True,
    },
    "inferno-tower": {
        "damage_schedule": [43, 158, 847],
        "stage_thresholds_us": [0, 1_500_000, 5_250_000],
        "reset_on_target_loss": True,
    },
}

# Phoenix leaves a targetable egg once per life.  November 2025's official
# balance note makes the reborn body use the original Level-11 HP and damage;
# the March 2026 official note also pins the egg to 317 HP and a 3.8 s hatch
# window. Exact targetability/frame ordering remains an explicit video-fit
# target.
REVIVE_DEFINITIONS: Mapping[str, Mapping[str, int | str]] = {
    "phoenix": {
        "egg_card_id": "phoenix-egg",
        "egg_hitpoints": 317,
        "egg_lifetime_us": 3_800_000,
        "revived_hitpoints": 1_052,
        "revived_damage": 217,
        "max_revives": 1,
    },
}

DEATH_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    # Balloon drops its bomb when the body is destroyed.  The Level-11
    # DeckShop snapshot reports 240 death damage; the bomb is a normal area
    # impact and therefore also damages Crown Towers at the same value.
    "balloon": {
        "damage": 240,
        "crown_tower_damage": 240,
        "radius_mtile": 1_500,
        "targets": ["air", "ground", "building", "crown_tower"],
        "knockback_mtile": 0,
    },
    "golem": {
        "damage": 225,
        "crown_tower_damage": 225,
        "radius_mtile": 2_000,
        "targets": ["air", "ground", "building", "crown_tower"],
        "knockback_mtile": 0,
        "spawn_children": [{"card_id": "golemite", "count": 2}],
    },
    "elixir-golem": {
        "damage": 0,
        "crown_tower_damage": 0,
        "radius_mtile": 0,
        "targets": ["ground"],
        "knockback_mtile": 0,
        "spawn_children": [{"card_id": "elixir-golemite", "count": 2}],
    },
    "lava-hound": {
        "damage": 0,
        "crown_tower_damage": 0,
        "radius_mtile": 0,
        "targets": ["air", "ground", "building", "crown_tower"],
        "knockback_mtile": 0,
        "spawn_children": [{"card_id": "lava-pup", "count": 6}],
    },
    "goblin-giant": {
        "damage": 0,
        "crown_tower_damage": 0,
        "radius_mtile": 0,
        "targets": ["ground"],
        "knockback_mtile": 0,
        "spawn_children": [{"card_id": "spear-goblin", "count": 2}],
    },
    "golemite": {
        # The child has its own death burst.  It is kept as a separate hidden
        # card so nested Golem splits are resolved by the same death queue as
        # every other entity rather than by a special-case recursion.
        "damage": 99,
        "crown_tower_damage": 99,
        "radius_mtile": 2_000,
        "targets": ["air", "ground", "building", "crown_tower"],
        "knockback_mtile": 0,
    },
    "elixir-golemite": {
        "damage": 0,
        "crown_tower_damage": 0,
        "radius_mtile": 0,
        "targets": ["ground"],
        "knockback_mtile": 0,
        "spawn_children": [{"card_id": "elixir-blob", "count": 2}],
    },
    # Battle Ram is a carrier: whether it reaches a building or is destroyed
    # on the way, the two Barbarians emerge from the broken ram.  The body
    # collision/charge is still represented by the generic troop component;
    # this definition owns the deterministic child stream.
    "battle-ram": {
        "damage": 0,
        "crown_tower_damage": 0,
        "radius_mtile": 0,
        "targets": ["ground"],
        "spawn_card_id": "barbarian",
        "spawn_count": 2,
    },
    "giant-skeleton": {
        "damage": 500,
        "crown_tower_damage": 125,
        "radius_mtile": 1_000,
        "targets": ["air", "ground", "building", "crown_tower"],
        "knockback_mtile": 0,
    },
    "goblin-cage": {
        "damage": 0,
        "crown_tower_damage": 0,
        "radius_mtile": 0,
        "targets": ["ground"],
        # Goblin Cage releases one Goblin Brawler, not the playable Goblins
        # formation.  The hidden one-body definition is added below so the
        # death stream cannot accidentally emit three/four generic Goblins.
        "spawn_card_id": "goblin-brawler",
        "spawn_count": 1,
    },
    "skeleton-barrel": {
        # The current Level-11 DeckShop page reports 145 death damage.  The
        # barrel still releases its three Skeletons in the same death event.
        "damage": 145,
        "crown_tower_damage": 145,
        "radius_mtile": 1_500,
        "targets": ["air", "ground", "building", "crown_tower"],
        "spawn_card_id": "skeletons",
        "spawn_count": 3,
    },
    "witch": {
        "damage": 0,
        "crown_tower_damage": 0,
        "radius_mtile": 0,
        "targets": ["ground"],
        "spawn_card_id": "skeletons",
        "spawn_count": 3,
    },
    # Suspicious Bush is a transport-like troop rather than a normal attacker:
    # it releases two private Bush Goblins when it reaches a building or is
    # destroyed.  ``bush-goblin`` is an internal child definition added below
    # and is intentionally not part of the playable opponent interaction set.
    "suspicious-bush": {
        "damage": 0,
        "crown_tower_damage": 0,
        "radius_mtile": 0,
        "targets": ["ground"],
        "spawn_card_id": "bush-goblin",
        "spawn_count": 2,
    },
    # These Level-11 values come from the official July 2024 balance note;
    # the 10-second low-health fuse is an official June 2025 change and is
    # represented in the card mechanics below.
    "goblin-demolisher": {
        "damage": 404,
        "crown_tower_damage": 404,
        "radius_mtile": 2_500,
        "targets": ["air", "ground", "building", "crown_tower"],
        "knockback_mtile": 0,
    },
    "phoenix": {
        # Level-11 death damage and the reduced Crown-Tower amount are
        # corroborated by the fixed card snapshot; radius/knockback remain
        # provisional until the visual oracle fits the burst edge.
        "damage": 163,
        "crown_tower_damage": 49,
        "radius_mtile": 1_500,
        "targets": ["air", "ground", "building", "crown_tower"],
        "knockback_mtile": 1_500,
    },
}

# Some troops transport independent bodies which attack while the carrier is
# alive.  The child identities are hidden, non-playable forms in the fixed
# roster, but they must still exist in authoritative state so projectile
# timing, targeting, and death/release interactions are simulated.  Offsets
# are deliberately deterministic world-space placeholders until isolated
# high-frame-rate footage resolves the exact rendering/formation geometry.
CARRIER_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "goblin-giant": {
        "child_card_id": "spear-goblin",
        "count": 2,
        "offsets_mtile": [[-450, 0], [450, 0]],
        "release_on_death": True,
    },
}

# Cannon Cart's May 2025 rework combines the wheel/shield and body health
# pools.  Once the single pool reaches 50% the same UID becomes a stationary
# building; it is not a death/spawn event.  The component is intentionally
# separate from ``death`` so damage crossing the threshold can preserve the
# remaining shared HP and the building can then receive normal lifetime decay.
HEALTH_TRANSFORM_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "cannon-cart": {
        "threshold_permille": 500,
        "target_card_id": "cannon-cart-building",
        "preserve_hp": True,
        "preserve_max_hp": True,
        "lifetime_us": 30_000_000,
    },
}

SPAWN_ON_IMPACT: Mapping[str, Mapping[str, int | str]] = {
    "barbarian-barrel": {"card_id": "barbarian", "count": 1},
    "goblin-barrel": {"card_id": "goblins", "count": 3},
    "graveyard": {"card_id": "skeletons", "count": 5},
    "royal-delivery": {"card_id": "royal-recruits", "count": 1},
}

ELIXIR_GENERATION: Mapping[str, Mapping[str, int]] = {
    # The current Level-11 card table reports a 12-second collector cycle.
    "elixir-collector": {"interval_us": 12_000_000, "amount_milli": 1_000},
}

# These structures create pressure only through their child/resource stream.
# Some legacy stat tables carry attack-looking fields for them; those fields
# must not become phantom turrets in the generated V1 artifact.
PASSIVE_SPAWNER_IDS = frozenset(
    {
        "barbarian-hut",
        "goblin-cage",
        # Goblin Drill's legacy table exposes its emergence/spawn damage as a
        # generic ``damage`` field.  It is not a repeating turret attack;
        # keep that interaction evidence-gated instead of simulating a
        # phantom melee weapon.
        "goblin-drill",
        "goblin-hut",
        "tombstone",
        "elixir-collector",
    }
)

# Persistent effects are represented as shared components rather than special
# branches in individual cards.  These are intentionally provisional catalog
# values: the generated ruleset keeps the uncertainty marker and the fidelity
# gate remains closed until each duration, pulse, victim set, and spawn stream
# is reconciled against held-out footage.
PERSISTENT_EFFECT_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "poison": {
        "duration_us": 8_000_000,
        "tick_interval_us": 1_000_000,
        "targets": ["air", "ground", "building", "crown_tower"],
        # The structured Level-11 source exposes Poison's per-second body
        # damage as 92.  The official June Crown-Tower total is 168, which is
        # eight deterministic 21-damage pulses.
        "damage_per_tick": 92,
        "crown_damage_per_tick": 21,
    },
    "earthquake": {
        "duration_us": 3_000_000,
        "tick_interval_us": 1_000_000,
        "targets": ["ground", "building", "crown_tower"],
        "damage_per_tick": 82,
        # Three one-second pulses produce the official 147 Crown-Tower total.
        "crown_damage_per_tick": 49,
    },
    "graveyard": {
        # The official June 2026 note pins the current field to 12 total
        # Skeletons; the older structured row reports 19 and is deliberately
        # overridden below rather than silently treated as truth.
        "duration_us": 9_000_000,
        "tick_interval_us": 500_000,
        "targets": ["air", "ground", "building", "crown_tower"],
        "damage_per_tick": 0,
        "crown_damage_per_tick": 0,
        "spawn": {
            "card_id": "skeletons",
            "count": 1,
            "max_spawns": 12,
        },
    },
    # Void is a field, not a one-shot projectile.  The official August 2026
    # patch gives the three target-count damage tiers and a 1.2 s hit
    # frequency.  The fixed component applies three pulses (the fourth
    # second of field lifetime is visual/temporal slack); this remains an
    # explicitly audited assumption until a high-confidence trace confirms
    # the exact first/last pulse timestamps.
    "void": {
        "duration_us": 4_000_000,
        "tick_interval_us": 1_200_000,
        "max_pulses": 3,
        "radius_mtile": 2_500,
        "targets": ["air", "ground", "building", "crown_tower"],
        "damage_per_tick": 696,
        "crown_damage_per_tick": 97,
        "damage_by_target_count": {"1": 696, "2-4": 294, "5+": 153},
        "crown_damage_by_target_count": {"1": 97, "2-4": 51, "5+": 35},
    },
    "tornado": {
        # DeckShop's current Level-11 page reports a 1.1 s field with damage
        # every 0.5 s.  The two positive pulses are split from the displayed
        # per-second total so integer damage remains deterministic.  The
        # third pulse keeps the pull active through the final 0.1 s of field
        # lifetime but carries no additional damage; exact game pulse timing
        # and pull force remain an explicit video-fit target.
        "duration_us": 1_100_000,
        "duration_anchor": "creation",
        "tick_interval_us": 500_000,
        "radius_mtile": 5_500,
        "targets": ["air", "ground", "crown_tower"],
        "damage_per_tick": 0,
        "crown_damage_per_tick": 0,
        "damage_schedule": [42, 42],
        "crown_damage_schedule": [14, 13],
        "pull_to_center_mtile": 1_000,
    },
    "rage": {
        # Rage deals its one-shot area damage on impact and leaves a 4.5 s
        # aura.  The one-element schedules intentionally make later pulses
        # damage-free while still refreshing a friendly buff for units which
        # enter the field after impact.  Units retain the buff for one second
        # after leaving, matching the current post-2025 behavior.
        "duration_us": 4_500_000,
        "duration_anchor": "creation",
        "tick_interval_us": 100_000,
        "radius_mtile": 3_000,
        "targets": ["air", "ground", "building", "crown_tower"],
        "damage_per_tick": 0,
        "crown_damage_per_tick": 0,
        "damage_schedule": [179],
        "crown_damage_schedule": [45],
        "friendly_status": {
            "kind": "rage",
            "duration_us": 1_000_000,
            "speed_multiplier_milli": 1_300,
            "hit_speed_multiplier_milli": 1_300,
            "linger_us": 1_000_000,
        },
        "friendly_targets": ["air", "ground", "building"],
    },
    "goblin-curse": {
        # The current card page reports 210 total body damage over six
        # one-second pulses (35 per pulse), 42 total Crown-Tower damage, and
        # a three-tile field.  The official August 2026 patch additionally
        # pins a 15% enemy slowdown.  A cursed troop is converted into one
        # ordinary Goblin if it dies while this status is active.
        "duration_us": 6_000_000,
        "duration_anchor": "after_immediate",
        "tick_interval_us": 1_000_000,
        "radius_mtile": 3_000,
        "targets": ["air", "ground", "crown_tower"],
        "damage_per_tick": 35,
        "crown_damage_per_tick": 7,
        "status": {
            "kind": "slow",
            "duration_us": 1_100_000,
            "speed_multiplier_milli": 850,
            "hit_speed_multiplier_milli": 1_000,
            "on_death_spawn_card_id": "goblin",
            "on_death_spawn_count": 1,
        },
    },
}


def _seconds_to_us(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    return int(round(float(value) * 1_000_000))


def _tiles_to_mtile(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    return int(round(float(value) * 1_000))


def _speed(value: object) -> int | None:
    if value is None:
        return None
    # The source catalog stores the qualitative speed used by the policy
    # stack.  These are executable conversions only; each generated card
    # carries a high-impact uncertainty until video fitting replaces it.
    # The pinned structured source uses both ``very-fast`` and ``very fast``
    # spellings.  Treat them as the same qualitative value; falling through
    # to ``medium`` would silently make every very-fast card (Bats, Goblins,
    # Spirits, ...) path at the wrong speed.
    normalized = str(value).strip().casefold().replace("-", " ")
    return {
        "slow": 700,
        "medium": 1_200,
        "fast": 1_800,
        "very fast": 2_400,
    }.get(normalized, 1_200)


def _formation(count: int, radius: int = 500) -> list[list[int]]:
    if count <= 0:
        return []
    result: list[list[int]] = []
    columns = 5 if count > 10 else max(1, min(5, count))
    for index in range(count):
        row, col = divmod(index, columns)
        x = (col - (columns - 1) // 2) * radius
        y = (row - ((count - 1) // columns) // 2) * radius
        result.append([x, y])
    return result


def _kind(card_id: str, metadata: Mapping[str, Any]) -> str:
    if card_id in BUILDING_IDS:
        return "building"
    return str(metadata.get("kind", "troop"))


def _targets(
    card_id: str,
    metadata: Mapping[str, Any],
    kind: str,
    source: Mapping[str, Any] | None = None,
) -> list[str]:
    if kind == "spell":
        return ["air", "ground", "building", "crown_tower"]
    # Prefer the Level-11 structured target classes.  The source uses the
    # human-facing plural ``buildings`` for Hog/Balloon/Royal Hogs; expand it
    # to the two simulator classes so a building-targeting troop can reach a
    # Crown Tower without relying on legacy metadata flags.
    source_targets = source.get("targets") if source is not None else None
    if isinstance(source_targets, list) and source_targets:
        normalized: list[str] = []
        for raw_target in source_targets:
            value = str(raw_target).strip().casefold().replace("-", "_")
            if value in {"building", "buildings"}:
                values = ("building", "crown_tower")
            elif value in {"crown_tower", "crown_towers", "tower", "towers"}:
                values = ("crown_tower",)
            elif value in {"troop", "troops", "units"}:
                values = ("air", "ground")
            elif value in {"air", "ground"}:
                values = (value,)
            else:
                values = ()
            for target in values:
                if target not in normalized:
                    normalized.append(target)
        if normalized:
            return normalized
    if bool(metadata.get("targets_buildings_only")):
        return ["building", "crown_tower"]
    targets: list[str] = []
    if bool(metadata.get("can_attack_air")):
        targets.append("air")
    if bool(metadata.get("can_attack_ground", True)):
        targets.append("ground")
    # A passive/spawner building still needs a non-empty schema target set;
    # engine dispatch treats missing attack values as non-attacking.
    return targets or ["ground"]


def _generated_card(card_id: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    kind = _kind(card_id, metadata)
    spell = kind == "spell"
    is_air = bool(metadata.get("is_air")) and not spell
    source = LEVEL11_SOURCE_PAYLOAD.get("cards", {}).get(card_id, {})
    # DeckShop corroboration is used only where a fresh Level-11 page supplies
    # a scalar that disagrees with the older structured snapshot.  Official
    # patch overrides are still applied after this merge.
    deckshop_source = DECKSHOP_CORE_SOURCE_PAYLOAD.get("cards", {}).get(card_id, {})
    if deckshop_source:
        source = {**source, **deckshop_source}
    # The repository metadata table is level 16.  Prefer the pinned Level-11
    # snapshot for every scalar it actually contains; metadata remains only a
    # clearly provisional fallback for cards absent from that snapshot.
    source_cost = source.get("elixir_cost")
    cost = int(source_cost if source_cost is not None else (metadata.get("elixir_cost") or 0)) * 1_000
    source_hp = source.get("hitpoints")
    hp = None if spell else int(source_hp if source_hp is not None else (metadata.get("hitpoints") or 1))
    source_damage = source.get("damage")
    damage = int(source_damage if source_damage is not None else (metadata.get("damage") or 0))
    hit_speed = source.get("attack_interval_s")
    if hit_speed is None:
        hit_speed = metadata.get("hit_speed")
    interval = None if spell or hit_speed is None else _seconds_to_us(hit_speed)
    deploy = 0 if spell else _seconds_to_us(metadata.get("deploy_time"), default=1_000_000)
    raw_range = source.get("range_tiles")
    if raw_range is None:
        raw_range = metadata.get("range")
    range_mtile = None if raw_range is None or spell else _tiles_to_mtile(raw_range)
    if spell:
        range_mtile = None
    source_radius = source.get("radius_tiles")
    # Never use attack range as a splash radius.  The old fallback made a
    # Firecracker's six-tile projectile, an Electro Dragon's chain range, and
    # similar non-area mechanics damage every unit in their sight radius.  A
    # missing radius remains ``None`` and is visible to reconciliation/video
    # readiness instead of becoming a plausible-looking but wrong constant.
    radius_value = source_radius if source_radius is not None else metadata.get("radius")
    # Firecracker's projectile is a splash burst behind the target.  The
    # structured scalar table omits that visual radius, so keep the
    # component-specific V1 radius explicit instead of falling back to its
    # six-tile acquisition range.
    if card_id == "firecracker" and radius_value is None:
        radius_value = 1.5
    area = (
        _tiles_to_mtile(radius_value, default=0)
        if spell
        else None
        if radius_value is None
        else _tiles_to_mtile(radius_value, default=0)
    )
    # Electro Wizard's three-tile value is its lightning acquisition/visual
    # radius, not a splash radius.  Its attack selects two discrete victims;
    # the multi-target component below replaces the old all-in-radius AoE
    # fallback.
    if card_id in MULTI_TARGET_ATTACK_DEFINITIONS:
        area = None
    source_projectile = source.get("projectile")
    projectile_needed = (
        spell
        or bool(source_projectile)
        if isinstance(source_projectile, bool)
        else spell or (not spell and raw_range is not None and float(raw_range) > 1.5)
    )
    # Wall Breakers are suicide splash troops, not ranged projectiles.  The
    # legacy structured row marks them as ``projectile`` because its generic
    # combat schema uses that bit for splash attacks.  In Clash Royale the
    # body runs into its building target and the explosion is resolved at the
    # contact point, so keep the existing area radius but remove the flight
    # phase.
    if card_id == "wall-breakers":
        projectile_needed = False
    projectile = (
        {
            "projectile_id": f"{card_id}-projectile",
            "speed_mtile_per_s": 12_000,
            "radius_mtile": 0,
            "start_radius_mtile": 0,
            "homing": not spell and float(raw_range or 0) <= 6.0,
        }
        if projectile_needed
        else None
    )
    placement = str(metadata.get("placement_class") or ("spell_anywhere" if spell else "own_ground"))
    if placement not in {"own_ground", "spell_anywhere", "spells", "restricted_spell"}:
        placement = "spell_anywhere" if spell else "own_ground"
    source_count = source.get("spawn_count")
    # An explicit Level-11 child count outranks the old metadata fallback.
    # This matters for current Goblins (4) and Goblin Gang (6), among other
    # formations whose composition has changed over the game's lifetime.
    count = (
        0
        if spell
        else int(source_count)
        if isinstance(source_count, int) and source_count > 0
        else SPAWN_COUNTS.get(card_id, 1)
    )
    source_lifetime = source.get("duration_s")
    lifetime = (
        _seconds_to_us(source_lifetime)
        if kind == "building" and source_lifetime is not None
        else (30_000_000 if kind == "building" else None)
    )
    mechanics = {
        "placement_class": placement,
        "movement_layer": "air" if is_air else None,
        "building_only": bool(metadata.get("targets_buildings_only")),
        "spawn_layout_mtile": _formation(count, 450 if count > 1 else 500),
        "death": None,
        "suicide_on_attack": bool(source.get("suicide")),
        "crown_tower_connection": "normal",
        "projectile_mode": "ballistic_to_point" if spell else ("homing" if projectile else "none"),
        "impact_mode": "at_destination" if spell else ("on_target" if projectile else "melee"),
        "status": None,
        "knockback_mtile": 0,
        "piercing": False,
        "spell_origin": "own-king-tower" if spell else None,
        "lifetime_decay": "linear_hp" if kind == "building" else None,
        "lifetime_start": "placement" if kind == "building" else None,
        "targetable_during_deploy": kind == "building",
        "persistent_effect": None,
    }
    if card_id == "battle-healer":
        # The August patch changed the radius and removed self-healing but did
        # not publish a new per-hit amount.  Independent Level-11 references
        # converge on 100 HP per attack (50 HPS at the new 2 s hit speed); the
        # value remains listed in UNKNOWN_BEHAVIORS until video confirms it.
        mechanics.update({
            "heal_radius_mtile": 3_000,
            "heal_amount": 100,
            "self_heal": False,
        })
    if card_id == "heal-spirit":
        # Heal Spirit is a suicide projectile whose impact also heals nearby
        # friendly troops.  Keep this separate from Battle Healer's melee
        # aura: the source body is already dead when the jump resolves and
        # friendly buildings/towers are not recipients.
        heal_source = DECKSHOP_HEAL_SPIRIT_SOURCE_PAYLOAD.get("cards", {}).get(
            card_id, {}
        )
        mechanics["heal_on_impact"] = {
            "amount": int(heal_source.get("healing") or 532),
            "radius_mtile": _tiles_to_mtile(
                heal_source.get("radius_tiles"), default=1_500
            ),
            "targets": [
                str(value)
                for value in heal_source.get(
                    "friendly_targets", ["air", "ground"]
                )
            ],
            "exclude_source": True,
        }
    if card_id == "suspicious-bush":
        mechanics.update({
            "stealth": True,
            "trigger_on_target": True,
        })
    if card_id in SHIELD_DEFINITIONS:
        mechanics["shield"] = dict(SHIELD_DEFINITIONS[card_id])
    if card_id == "royal-ghost":
        mechanics.update({
            "stealth": True,
            "stealth_recloak_us": 1_500_000,
        })
    if card_id == "miner":
        # Miner is the explicit deployment exception to normal own-territory
        # troop placement: the card selects a legal ground destination and
        # the body tunnels there before becoming targetable.
        mechanics.update({
            "placement_class": "miner_anywhere",
            "burrow": {
                "duration_us": 1_000_000,
                "target_anywhere": True,
                "targetable_during_burrow": False,
            },
        })
    if card_id in SPAWN_CHILDREN_DEFINITIONS:
        mechanics["spawn_children"] = [deepcopy(dict(row)) for row in SPAWN_CHILDREN_DEFINITIONS[card_id]]
    if card_id == "magic-archer":
        mechanics.update({
            "piercing": True,
            "line_piercing": {
                "length_mtile": 10_000,
                "width_mtile": 250,
            },
        })
    if card_id == "executioner":
        mechanics["returning_projectile"] = {
            "return_speed_mtile_per_s": 12_000,
            "return_radius_mtile": 1_000,
        }
        # An Executioner axe is a fixed outbound/return path, not a homing
        # missile.  The endpoint still follows the acquired target at launch.
        if projectile is not None:
            projectile["homing"] = False
    if card_id == "hunter":
        mechanics["pellets"] = {"count": 10, "spread_mtile": 900}
    if card_id == "bowler":
        mechanics.update({
            "knockback_mtile": 1_500,
            "knockback_direction": "projectile_travel",
        })
    if card_id == "mega-knight":
        mechanics["jump"] = dict(JUMP_DEFINITIONS[card_id])
        mechanics["impact_targets"] = ["ground", "building", "crown_tower"]
        projectile = None
        mechanics["projectile_mode"] = "none"
        mechanics["impact_mode"] = "melee"
    if card_id in DEPLOY_EFFECT_DEFINITIONS:
        mechanics["deploy_effect"] = deepcopy(DEPLOY_EFFECT_DEFINITIONS[card_id])
    if card_id in DEATH_RAGE_DEFINITIONS:
        mechanics["death_rage"] = deepcopy(DEATH_RAGE_DEFINITIONS[card_id])
    if card_id == "mother-witch":
        mechanics["status"] = {
            "kind": "mother-witch-curse",
            "duration_us": 5_000_000,
            "speed_multiplier_milli": 1_000,
            "hit_speed_multiplier_milli": 1_000,
            "on_death_spawn_card_id": "cursed-hog",
            "on_death_spawn_count": 1,
        }
    if card_id == "ram-rider":
        mechanics["snare"] = {
            "duration_us": 1_500_000,
            "speed_multiplier_milli": 550,
            "hit_speed_multiplier_milli": 1_000,
            "targets": ["air", "ground"],
        }
    if card_id == "goblin-demolisher":
        mechanics.update({
            "charge_threshold_permille": 500,
            "charge_duration_us": 10_000_000,
            "charged_speed_mtile_per_s": 2_400,
            "charge_range_mtile": 800,
            "trigger_on_building_contact": True,
        })
    if card_id in CHARGE_ATTACK_DEFINITIONS:
        mechanics["charge_attack"] = dict(CHARGE_ATTACK_DEFINITIONS[card_id])
    if card_id in DASH_DEFINITIONS:
        mechanics["dash"] = dict(DASH_DEFINITIONS[card_id])
    if card_id in HOOK_DEFINITIONS:
        mechanics["hook"] = dict(HOOK_DEFINITIONS[card_id])
    if card_id in RAMP_ATTACK_DEFINITIONS:
        mechanics["ramp_attack"] = deepcopy(RAMP_ATTACK_DEFINITIONS[card_id])
    if card_id in REVIVE_DEFINITIONS:
        mechanics["revive"] = dict(REVIVE_DEFINITIONS[card_id])
    if card_id == "firecracker":
        mechanics["recoil_mtile"] = 1_500
    if card_id == "sparky":
        # Sparky charges its first shot for four seconds and immediately
        # starts the next charge after firing.  The engine treats this as a
        # reusable wind-up (rather than adding the four seconds to the normal
        # attack interval) via ``attack_windup_mode``.
        mechanics["attack_windup_mode"] = "recharge"
    if card_id == "wall-breakers":
        # Movement/target acquisition remains buildings-only, while the
        # contact explosion damages all normal unit layers and Crown Towers.
        mechanics["impact_targets"] = ["air", "ground", "building", "crown_tower"]
    if card_id == "battle-ram":
        # A Battle Ram is consumed by its first building/tower impact.  Its
        # two Barbarians are then released through the normal death queue;
        # leaving the generic body alive after the impact incorrectly lets a
        # Ram continue attacking as an ordinary troop and suppresses the
        # carrier interaction that Hog-cycle counterplay depends on.
        mechanics["suicide_on_attack"] = True
    if card_id == "goblin-machine":
        # The machine has an independent rocket weapon.  Official August
        # 2026 notes set its 5-second hit speed and 350-tile/s travel speed;
        # the fixed Level-11 damage remains the post-June-2024 391 body
        # damage with the documented 50% Crown-Tower reduction (rounded to
        # the nearest integer for the game's integer damage model).
        mechanics["secondary_attack"] = {
            "min_range_mtile": 2_500,
            "max_range_mtile": 5_000,
            "attack_interval_us": 5_000_000,
            "first_hit_delay_us": 0,
            "damage": 391,
            "crown_tower_damage": 196,
            "area_radius_mtile": 1_500,
            "projectile_speed_mtile_per_s": 350_000,
            "projectile_radius_mtile": 0,
            "targets": ["air", "ground", "building", "crown_tower"],
        }
    if card_id == "clone":
        # Clone is an impact-only spell.  It does not damage the arena; it
        # copies friendly troop bodies in its radius as one-HP entities.  The
        # explicit component keeps buildings and existing clones out of the
        # generic spell victim path and gives the engine a deterministic place
        # to refine spawn offsets/eligibility when held-out footage resolves
        # those details.
        mechanics["clone"] = {
            "copy_kind": "troop",
            "clone_hp": 1,
            "clone_max_hp": 1,
            "exclude_clones": True,
        }
    if card_id == "lightning":
        # Lightning selects at most three highest-current-HP eligible targets
        # in its radius and resets their attack state.  The strike spacing is
        # intentionally left as a source/video unknown; the target set and
        # damage are still deterministic at the impact tick.
        mechanics.update({
            "target_limit": 3,
            "target_selection": "highest_hp",
            "reset_attack": True,
        })
    if card_id in SPAWNER_DEFINITIONS:
        mechanics["spawn"] = dict(SPAWNER_DEFINITIONS[card_id])
    if card_id in STATUS_DEFINITIONS:
        mechanics["status"] = dict(STATUS_DEFINITIONS[card_id])
        if card_id in {"electro-dragon", "electro-wizard"}:
            mechanics["reset_attack"] = True
    if card_id in CHAIN_ATTACK_DEFINITIONS:
        mechanics["chain_attack"] = dict(CHAIN_ATTACK_DEFINITIONS[card_id])
    if card_id in MULTI_TARGET_ATTACK_DEFINITIONS:
        mechanics["multi_target_attack"] = dict(MULTI_TARGET_ATTACK_DEFINITIONS[card_id])
    if card_id in REFLECTION_DEFINITIONS:
        mechanics["reflection"] = deepcopy(REFLECTION_DEFINITIONS[card_id])
    if card_id in DEATH_DEFINITIONS:
        mechanics["death"] = dict(DEATH_DEFINITIONS[card_id])
    if card_id in CARRIER_DEFINITIONS:
        mechanics["carrier"] = deepcopy(CARRIER_DEFINITIONS[card_id])
    if card_id in HEALTH_TRANSFORM_DEFINITIONS:
        mechanics["health_transform"] = dict(HEALTH_TRANSFORM_DEFINITIONS[card_id])
    if card_id in SPAWN_ON_IMPACT:
        mechanics["spawn_on_impact"] = dict(SPAWN_ON_IMPACT[card_id])
    if card_id in ELIXIR_GENERATION:
        mechanics["elixir_generation"] = dict(ELIXIR_GENERATION[card_id])
    if card_id in PERSISTENT_EFFECT_DEFINITIONS:
        mechanics["persistent_effect"] = deepcopy(
            PERSISTENT_EFFECT_DEFINITIONS[card_id]
        )
        # A persistent component owns the temporal spawn/effect stream.  It
        # must not also execute the old one-shot impact spawn path.
        mechanics.pop("spawn_on_impact", None)
    uncertainty = {
        "field": "generated_definition",
        "reason": (
            "Scalar values use the pinned Level-11 structured snapshot where available; "
            "card-specific mechanics and held-out current-ruleset evidence are not yet fully reconciled."
            if source
            else "No Level-11 structured row exists for this card; values fall back to the repository catalog."
        ),
        "impact": "high",
        "resolution": "Replace with field-level official/structured/video evidence and promote only after held-out tests pass.",
    }
    if card_id in CARRIER_DEFINITIONS:
        uncertainty = {
            "field": "carrier_child_timing",
            "reason": (
                "DeckShop identifies two carried Spear Goblins; the executable "
                "attached-body stream and release behavior are data-driven, while "
                "exact offsets, orientation, and first-hit timing remain unresolved."
            ),
            "impact": "high",
            "resolution": (
                "Fit isolated Level-11 Goblin Giant footage for child positions, "
                "ranged attacks while carried, and release timing on carrier death."
            ),
        }
    provenance = {
        "identity": [CATALOG_SOURCE_ID],
        "catalog_values": [CATALOG_SOURCE_ID],
        "generated_mechanics": [CATALOG_SOURCE_ID],
    }
    if source:
        provenance["level_11_stats"] = [LEVEL11_SOURCE_ID]
    if card_id == "goblin-machine":
        provenance.setdefault("generated_mechanics", []).append(GOBLIN_MACHINE_SOURCE_ID)
    if card_id in CARRIER_DEFINITIONS:
        provenance.setdefault("generated_mechanics", []).append(SPLIT_SOURCE_ID)
    if deckshop_source:
        level11_sources = provenance.setdefault("level_11_stats", [])
        if DECKSHOP_CORE_SOURCE_ID not in level11_sources:
            level11_sources.append(DECKSHOP_CORE_SOURCE_ID)
    if card_id == "battle-healer":
        provenance.setdefault("level_11_stats", []).append(DECKSHOP_SOURCE_ID)
    if card_id == "heal-spirit":
        provenance.setdefault("generated_mechanics", []).append(
            DECKSHOP_HEAL_SPIRIT_SOURCE_ID
        )
    raw = {
        "name": card_id.replace("-", " ").title(),
        "aliases": [card_id.replace("-", " "), card_id.replace("-", "_")],
        "kind": kind,
        "elixir_milli": cost,
        "deploy_time_us": deploy,
        "spawn_count": count,
        "hitpoints": hp,
        "damage": damage,
        # Never derive Crown Tower damage by dividing troop damage.  A number
        # of spells (Clone, Graveyard, Goblin Barrel, Freeze, ...) have no
        # direct tower damage; their spawned bodies/effects are separate
        # components.  The official June table then overrides the spells with
        # explicit Level-11 values where applicable.
        "crown_tower_damage": (
            int(source.get("tower_damage"))
            if spell and source.get("tower_damage") is not None
            else (0 if spell else None)
        ),
        "attack_interval_us": interval,
        "first_hit_delay_us": 0 if interval is not None else None,
        "move_speed_mtile_per_s": None
        if spell or kind == "building"
        else _speed(source.get("speed") or metadata.get("move_speed")),
        "range_mtile": range_mtile,
        "sight_range_mtile": None if spell else max(range_mtile or 4_000, 6_000),
        "collision_radius_mtile": None if spell else (600 if kind == "building" else 400),
        "mass": None if spell else (0 if kind == "building" else 1),
        "lifetime_us": lifetime,
        "targets": _targets(card_id, metadata, kind, source),
        "projectile": projectile,
        "area_radius_mtile": area or None,
        "mechanics": mechanics,
        "provenance": provenance,
        "uncertainties": [uncertainty],
    }
    if card_id in PASSIVE_SPAWNER_IDS:
        # Child/resource-only structures do not independently acquire or fire
        # at targets.  Keep their target vocabulary for placement/effect
        # filtering, but make the attack component absent and auditable.
        raw.update(
            {
                "damage": None,
                "attack_interval_us": None,
                "first_hit_delay_us": None,
                "range_mtile": None,
                "sight_range_mtile": None,
                "projectile": None,
                "area_radius_mtile": None,
            }
        )
    raw, _ = apply_official_overrides(card_id, raw)
    if card_id == "sparky":
        raw["first_hit_delay_us"] = 4_000_000
    return raw


def _phoenix_egg_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    """Create Phoenix's internal, non-playable targetable egg definition."""

    phoenix = deepcopy(cards["phoenix"])
    egg_mechanics = dict(phoenix["mechanics"])
    egg_mechanics.pop("revive", None)
    egg_mechanics.update(
        {
            "building_only": False,
            "movement_layer": None,
            "spawn_layout_mtile": [[0, 0]],
            "death": None,
            "placement_class": "own_ground",
            "projectile_mode": "none",
            "impact_mode": "melee",
            "lifetime_decay": None,
            "lifetime_start": "placement",
            "targetable_during_deploy": True,
            "revive_egg": {"hatch_card_id": "phoenix"},
        }
    )
    phoenix.update(
        {
            "name": "Phoenix Egg",
            "aliases": ["phoenix egg", "phoenix_egg"],
            "kind": "building",
            "elixir_milli": 0,
            "deploy_time_us": 0,
            "spawn_count": 1,
            "hitpoints": 317,
            "damage": None,
            "crown_tower_damage": None,
            "attack_interval_us": None,
            "first_hit_delay_us": None,
            "move_speed_mtile_per_s": None,
            "range_mtile": None,
            "sight_range_mtile": None,
            "collision_radius_mtile": 400,
            "mass": 0,
            "lifetime_us": 3_800_000,
            "targets": ["air", "ground", "building", "crown_tower"],
            "projectile": None,
            "area_radius_mtile": None,
            "mechanics": egg_mechanics,
            "provenance": {
                **dict(phoenix.get("provenance", {})),
                "identity": [CATALOG_SOURCE_ID],
                "generated_mechanics": [
                    CATALOG_SOURCE_ID,
                    "deckmelon-phoenix-2026-08-15",
                    "official-march-2026",
                ],
            },
            "uncertainties": [
                {
                    "field": "phoenix_egg_hatch",
                    "reason": "Official March 2026 values pin egg HP/lifetime; exact targetability and hatch frame require held-out video confirmation.",
                    "impact": "high",
                    "resolution": "Measure isolated Phoenix deaths and egg damage/hatch timing in both HUD variants.",
                }
            ],
        }
    )
    return phoenix


def _bush_goblin_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    """Create the non-playable Bush Goblin child used by Suspicious Bush."""

    raw = deepcopy(cards["goblins"])
    raw.update(
        {
            "name": "Bush Goblin",
            "aliases": ["bush goblin", "bush_goblin"],
            "kind": "troop",
            "elixir_milli": 0,
            "deploy_time_us": 0,
            "spawn_count": 1,
            "hitpoints": 337,
            "damage": 256,
            "attack_interval_us": 1_400_000,
            "first_hit_delay_us": 0,
            "move_speed_mtile_per_s": 1_200,
            "range_mtile": 800,
            "sight_range_mtile": 6_000,
            "area_radius_mtile": None,
            "targets": ["ground"],
            "projectile": None,
            "mechanics": {
                **dict(raw["mechanics"]),
                "building_only": False,
                "spawn_layout_mtile": [[0, 0]],
                "death": None,
                "projectile_mode": "none",
                "impact_mode": "melee",
                "suicide_on_attack": False,
                "placement_class": "own_ground",
                "stealth": False,
                "trigger_on_target": False,
            },
            "provenance": {
                **dict(raw.get("provenance", {})),
                "identity": [CATALOG_SOURCE_ID],
                "level_11_stats": ["official-august-2026"],
                "generated_mechanics": [CATALOG_SOURCE_ID, "official-august-2026"],
            },
            "uncertainties": [
                {
                    "field": "bush_goblin_child_definition",
                    "reason": "Internal child card; its 337 HP is official, while movement and attack geometry are independently sourced.",
                    "impact": "high",
                    "resolution": "Promote after isolated Suspicious Bush footage confirms child timing and trajectory.",
                }
            ],
        }
    )
    return raw


def _goblin_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    """Create the one-body Goblin produced by Goblin Curse.

    The playable ``goblins`` card is a four-body formation.  A transformed
    troop is a single Goblin, so it needs its own internal definition rather
    than reusing the group card's spawn count and layout.
    """

    raw = deepcopy(cards["goblins"])
    raw.update(
        {
            "name": "Goblin",
            "aliases": ["goblin", "goblin_child", "goblin-curse-goblin"],
            "elixir_milli": 0,
            "deploy_time_us": 0,
            "spawn_count": 1,
            "mechanics": {
                **dict(raw["mechanics"]),
                "spawn_layout_mtile": [[0, 0]],
            },
            "provenance": {
                **dict(raw.get("provenance", {})),
                "identity": [CATALOG_SOURCE_ID],
                "generated_mechanics": [CATALOG_SOURCE_ID],
            },
            "uncertainties": [
                {
                    "field": "goblin_curse_child_definition",
                    "reason": "Single Goblin body derived from the Level-11 Goblins row.",
                    "impact": "medium",
                    "resolution": "Confirm transformed Goblin spawn position and deploy timing against isolated Goblin Curse footage.",
                }
            ],
        }
    )
    return raw


def _barbarian_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    """Create one Barbarian body for carrier/death-spawn components.

    The public ``barbarians`` card is a five-body formation.  Reusing that
    group definition for Battle Ram/Barbarian Barrel would accidentally emit
    five-body formations per child, so carrier components use this hidden
    one-body form instead.
    """

    raw = deepcopy(cards["barbarians"])
    raw.update(
        {
            "name": "Barbarian",
            "aliases": ["barbarian", "barbarian_child"],
            "spawn_count": 1,
            "mechanics": {
                **dict(raw["mechanics"]),
                "spawn_layout_mtile": [[0, 0]],
            },
            "provenance": {
                **dict(raw.get("provenance", {})),
                "identity": [CATALOG_SOURCE_ID],
                "generated_mechanics": [CATALOG_SOURCE_ID],
            },
            "uncertainties": [
                {
                    "field": "carrier_child_definition",
                    "reason": "Single Barbarian body derived from the Level-11 Barbarians group row.",
                    "impact": "medium",
                    "resolution": "Confirm carrier child spacing and exact body identity against isolated footage.",
                }
            ],
        }
    )
    return raw


def _split_child_raw(
    cards: Mapping[str, Any],
    *,
    parent_card_id: str,
    child_card_id: str,
    name: str,
    hitpoints: int,
    damage: int,
    attack_interval_us: int,
    range_mtile: int,
    move_speed_mtile_per_s: int,
    targets: list[str],
    movement_layer: str | None,
    projectile: dict[str, Any] | None,
    uncertainty: str,
    first_hit_delay_us: int = 0,
    source_id: str = SPLIT_SOURCE_ID,
) -> dict[str, Any]:
    """Build one hidden body from a public split-card definition.

    Split forms are not playable interaction-set cards, but they are still
    first-class entities: they can acquire targets, attack, die, and (for
    Elixir Golemites) recursively split.  Keeping their definitions in the
    pinned ruleset makes nested death streams deterministic and serializable.
    """

    parent = deepcopy(cards[parent_card_id])
    mechanics = dict(parent.get("mechanics", {}))
    # A child must not inherit a parent's transport, spawner, revive, or
    # health-form component.  Its own death component is installed below.
    for key in (
        "spawn",
        "spawn_on_impact",
        "elixir_generation",
        "persistent_effect",
        "revive",
        "revive_egg",
        "health_transform",
        "charge_attack",
        "dash",
        "hook",
        "ramp_attack",
        "secondary_attack",
        "clone",
        "spawn_children",
        "shield",
        "stealth_recloak_us",
        "burrow",
        "line_piercing",
        "returning_projectile",
        "pellets",
        "jump",
        "deploy_effect",
        "death_rage",
        "snare",
    ):
        mechanics.pop(key, None)
    mechanics.update(
        {
            "placement_class": "own_ground",
            "movement_layer": movement_layer,
            "building_only": targets == ["building", "crown_tower"],
            "spawn_layout_mtile": [[0, 0]],
            "death": dict(DEATH_DEFINITIONS.get(child_card_id, {})) or None,
            "suicide_on_attack": False,
            "crown_tower_connection": "normal",
            "projectile_mode": "homing" if projectile is not None else "none",
            "impact_mode": "on_target" if projectile is not None else "melee",
            "status": None,
            "knockback_mtile": 0,
            "piercing": False,
            "spell_origin": None,
            "lifetime_decay": None,
            "lifetime_start": None,
            "targetable_during_deploy": True,
        }
    )
    parent.update(
        {
            "name": name,
            "aliases": [child_card_id.replace("-", " "), child_card_id.replace("-", "_")],
            "kind": "troop",
            "elixir_milli": 0,
            "deploy_time_us": 0,
            "spawn_count": 1,
            "hitpoints": hitpoints,
            "damage": damage,
            "crown_tower_damage": None,
            "attack_interval_us": attack_interval_us,
            "first_hit_delay_us": first_hit_delay_us,
            "move_speed_mtile_per_s": move_speed_mtile_per_s,
            "range_mtile": range_mtile,
            "sight_range_mtile": max(range_mtile, 6_000),
            "collision_radius_mtile": 400,
            "mass": 1,
            "lifetime_us": None,
            "targets": list(targets),
            "projectile": deepcopy(projectile),
            "area_radius_mtile": None,
            "mechanics": mechanics,
            "provenance": {
                "identity": [source_id],
                "level_11_stats": [source_id],
                "generated_mechanics": [source_id],
            },
            "uncertainties": [
                {
                    "field": f"{child_card_id}.split_child_definition",
                    "reason": uncertainty,
                    "impact": "high",
                    "resolution": "Fit child spawn offsets, first-hit timing, and targeting against isolated 120 FPS pre-Evolution footage.",
                }
            ],
        }
    )
    return parent


def _golemite_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    row = SPLIT_SOURCE_PAYLOAD["cards"]["golemite"]
    return _split_child_raw(
        cards,
        parent_card_id="golem",
        child_card_id="golemite",
        name="Golemite",
        hitpoints=int(row["hitpoints"]),
        damage=int(row["damage"]),
        attack_interval_us=_seconds_to_us(row["attack_interval_s"]),
        range_mtile=_tiles_to_mtile(row["range_tiles"]),
        move_speed_mtile_per_s=_speed(row["speed"]) or 1_200,
        targets=["building", "crown_tower"],
        movement_layer=None,
        projectile=None,
        uncertainty="DeckMelon supplies Level-11 Golemite body values; the official August 2025 patch corroborates its damage change, while child death radius and spawn geometry remain unresolved.",
    )


def _elixir_golemite_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    row = SPLIT_SOURCE_PAYLOAD["cards"]["elixir-golemite"]
    return _split_child_raw(
        cards,
        parent_card_id="elixir-golem",
        child_card_id="elixir-golemite",
        name="Elixir Golemite",
        hitpoints=int(row["hitpoints"]),
        damage=int(row["damage"]),
        attack_interval_us=_seconds_to_us(row["attack_interval_s"]),
        range_mtile=_tiles_to_mtile(row["range_tiles"]),
        move_speed_mtile_per_s=_speed(row["speed"]) or 1_200,
        targets=["building", "crown_tower"],
        movement_layer=None,
        projectile=None,
        uncertainty="DeckMelon supplies Level-11 Elixir Golemite values; exact split offsets and elixir-on-death transfer timing remain unresolved.",
    )


def _elixir_blob_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    row = SPLIT_SOURCE_PAYLOAD["cards"]["elixir-blob"]
    return _split_child_raw(
        cards,
        parent_card_id="elixir-golem",
        child_card_id="elixir-blob",
        name="Elixir Blob",
        hitpoints=int(row["hitpoints"]),
        damage=int(row["damage"]),
        attack_interval_us=_seconds_to_us(row["attack_interval_s"]),
        range_mtile=_tiles_to_mtile(row["range_tiles"]),
        move_speed_mtile_per_s=_speed(row["speed"]) or 1_200,
        targets=["building", "crown_tower"],
        movement_layer=None,
        projectile=None,
        uncertainty="DeckMelon supplies Level-11 Elixir Blob values; exact split offsets and enemy elixir award timing remain unresolved.",
    )


def _lava_pup_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    row = SPLIT_SOURCE_PAYLOAD["cards"]["lava-pup"]
    raw = _split_child_raw(
        cards,
        parent_card_id="lava-hound",
        child_card_id="lava-pup",
        name="Lava Pup",
        hitpoints=int(row["hitpoints"]),
        damage=int(row["damage"]),
        attack_interval_us=_seconds_to_us(row["attack_interval_s"]),
        range_mtile=_tiles_to_mtile(row["range_tiles"]),
        move_speed_mtile_per_s=_speed(row["speed"]) or 1_200,
        targets=["air", "ground", "building", "crown_tower"],
        movement_layer="air",
        projectile=None,
        uncertainty="DeckMelon supplies Level-11 Lava Pup values; exact post-burst offsets and first-hit timing require clean air-lane footage.",
    )
    provenance = {
        str(key): list(value) if isinstance(value, (list, tuple)) else [str(value)]
        for key, value in dict(raw.get("provenance", {})).items()
    }
    provenance.setdefault("level_11_stats", []).append("official-december-2025-lava-pup")
    raw["provenance"] = provenance
    return raw


def _spear_goblin_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    row = SPLIT_SOURCE_PAYLOAD["cards"]["spear-goblin"]
    parent = cards["spear-goblins"]
    projectile = deepcopy(parent.get("projectile"))
    return _split_child_raw(
        cards,
        parent_card_id="spear-goblins",
        child_card_id="spear-goblin",
        name="Spear Goblin",
        hitpoints=int(row["hitpoints"]),
        damage=int(row["damage"]),
        attack_interval_us=_seconds_to_us(row["attack_interval_s"]),
        range_mtile=_tiles_to_mtile(row["range_tiles"]),
        move_speed_mtile_per_s=_speed(row["speed"]) or 2_400,
        targets=["air", "ground"],
        movement_layer=None,
        projectile=projectile,
        uncertainty="DeckShop identifies Goblin Giant's two carried Spear Goblins; the child scalar values are corroborated by the Level-11 Spear Goblins row, while carried-body timing remains unresolved.",
    )


def _rascal_boy_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    """Build the melee Rascal Boy child of the three-body Rascals card."""

    return _split_child_raw(
        cards,
        parent_card_id="rascals",
        child_card_id="rascal-boy",
        name="Rascal Boy",
        hitpoints=1_940,
        damage=217,
        attack_interval_us=1_500_000,
        range_mtile=800,
        move_speed_mtile_per_s=_speed("medium") or 1_200,
        targets=["ground"],
        movement_layer=None,
        projectile=None,
        source_id=CATALOG_SOURCE_ID,
        uncertainty=(
            "Rascal Boy's Level-11 aggregate body values are corroborated by "
            "the pinned Rascals row; exact child formation offsets and split "
            "animation require held-out footage."
        ),
    )


def _rascal_girl_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    """Build the ranged Rascal Girl child of the three-body Rascals card."""

    return _split_child_raw(
        cards,
        parent_card_id="rascals",
        child_card_id="rascal-girl",
        name="Rascal Girl",
        hitpoints=202,
        damage=120,
        attack_interval_us=1_500_000,
        range_mtile=5_500,
        move_speed_mtile_per_s=_speed("medium") or 1_200,
        targets=["air", "ground"],
        movement_layer=None,
        projectile=deepcopy(cards["rascals"].get("projectile")),
        source_id=CATALOG_SOURCE_ID,
        uncertainty=(
            "Rascal Girl's Level-11 body values follow the standard ranged "
            "Goblin-class child profile; exact projectile timing and child "
            "formation offsets remain a video-fit target."
        ),
    )


def _cursed_hog_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    """Create Mother Witch's non-playable building-targeting Hog."""

    return _split_child_raw(
        cards,
        parent_card_id="hog-rider",
        child_card_id="cursed-hog",
        name="Cursed Hog",
        hitpoints=629,
        damage=53,
        attack_interval_us=1_200_000,
        range_mtile=800,
        move_speed_mtile_per_s=_speed("very-fast") or 2_400,
        targets=["building", "crown_tower"],
        movement_layer=None,
        projectile=None,
        source_id=CATALOG_SOURCE_ID,
        uncertainty=(
            "Cursed Hog Level-11 body values are corroborated by the fixed "
            "Mother Witch child snapshot; exact conversion offset and first "
            "target frame remain held-out video targets."
        ),
    )


def _goblin_brawler_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    """Create Goblin Cage's hidden one-body Goblin Brawler form.

    ``goblins`` is a multi-body playable formation and cannot be reused for
    the Cage death stream.  DeckMelon's Level-11 Goblin Cage page provides the
    Brawler's independent body values; the child remains hidden from the
    interaction roster because players never deploy it directly.
    """

    row = GOBLIN_BRAWLER_SOURCE_PAYLOAD["cards"]["goblin-brawler"]
    return _split_child_raw(
        cards,
        parent_card_id="goblins",
        child_card_id="goblin-brawler",
        name="Goblin Brawler",
        hitpoints=int(row["hitpoints"]),
        damage=int(row["damage"]),
        attack_interval_us=_seconds_to_us(row["attack_interval_s"]),
        range_mtile=_tiles_to_mtile(row["range_tiles"]),
        move_speed_mtile_per_s=_speed(row["speed"]) or 1_800,
        targets=[str(value) for value in row["targets"]],
        movement_layer=None,
        projectile=None,
        first_hit_delay_us=_seconds_to_us(row.get("first_hit_delay_s")),
        source_id=GOBLIN_BRAWLER_SOURCE_ID,
        uncertainty=(
            "DeckMelon supplies the Level-11 Goblin Brawler body values and "
            "first-hit delay; exact Cage release offset, activation frame, "
            "and post-release retarget behavior remain unresolved."
        ),
    )


def _cannon_cart_building_raw(cards: Mapping[str, Any]) -> dict[str, Any]:
    """Create Cannon Cart's hidden stationary form.

    The post-May-2025 card has one shared health pool: the public cart body
    (1,809 HP at Level 11) changes kind at 50% HP instead of dropping a
    separate shield.  The broken form keeps the cart's 212 damage / 0.9 s
    weapon and uses the normal building lifetime/linear-decay machinery.  Its
    exact first-hit frame and the way the client renders the shared HP bar are
    still held-out validation targets, so this definition remains explicitly
    provisional even though the transform trigger itself is official.
    """

    source = deepcopy(cards["cannon-cart"])
    source.update(
        {
            "name": "Cannon Cart (building)",
            "aliases": [
                "cannon-cart-building",
                "cannon_cart_building",
                "broken-cannon-cart",
            ],
            "kind": "building",
            "elixir_milli": 0,
            "deploy_time_us": 0,
            "spawn_count": 1,
            # Keep the shared pool's maximum on the hidden form.  The engine
            # preserves the live entity's HP/max HP during transformation;
            # this value only makes the internal card definition executable.
            "hitpoints": 1809,
            "lifetime_us": 30_000_000,
            "move_speed_mtile_per_s": None,
            "collision_radius_mtile": 600,
            "mass": 0,
            "mechanics": {
                **dict(source["mechanics"]),
                "movement_layer": None,
                "placement_class": "own_ground",
                "lifetime_decay": "linear_hp",
                "lifetime_start": "transform",
                "targetable_during_deploy": True,
                "health_transform": None,
            },
            "provenance": {
                **dict(source.get("provenance", {})),
                "identity": [
                    "official-may-2025-cannon-cart-rework",
                    "official-january-2025-cannon-cart",
                ],
                "generated_mechanics": [
                    "official-may-2025-cannon-cart-rework",
                    "official-august-2024-cannon-cart",
                ],
                "level_11_stats": [
                    "local-level11-card-stats-2026-04-12",
                    "deckshop-cannon-cart-2026-08-16",
                ],
            },
            "uncertainties": [
                {
                    "field": "health_transform.stationary_form",
                    "reason": (
                        "Supercell specifies the 50% shared-health transition and "
                        "the cart weapon speed, but does not publish the exact "
                        "first-hit frame, HP-bar rendering, or decay reference "
                        "for the hidden stationary form."
                    ),
                    "impact": "high",
                    "resolution": (
                        "Measure a Level-11 Cannon Cart crossing the threshold at "
                        "120 FPS and OCR HP/lifetime until the building expires."
                    ),
                }
            ],
        }
    )
    return source


def build_roster_ruleset_raw(base_raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a canonical roster-complete ruleset payload.

    Existing high-confidence base-card definitions are retained byte-for-byte
    apart from the interaction-set/version metadata.  Missing eligible cards
    receive generated provisional definitions with explicit uncertainty.
    """

    if base_raw is None:
        base_raw = json.loads(ruleset_path("2026-08-04").read_text(encoding="utf-8"))
    raw = deepcopy(dict(base_raw))
    roster = load_opponent_roster()
    cards = dict(raw["cards"])
    for card_id in roster.eligible_cards:
        metadata = CARD_METADATA.get(card_id)
        if metadata is None:
            raise ValueError(f"eligible roster card missing from card catalog: {card_id}")
        if card_id not in cards:
            cards[card_id] = _generated_card(card_id, metadata)
        # The immutable base ruleset already contains a handful of cards with
        # hand-curated definitions.  Apply fresh DeckShop scalar corroboration
        # to those rows as well; otherwise an old Log value would survive just
        # because it happened to be present in the base artifact.
        corroboration = DECKSHOP_CORE_SOURCE_PAYLOAD.get("cards", {}).get(card_id, {})
        if corroboration:
            cards[card_id] = deepcopy(cards[card_id])
            official_fields = {
                str(row.get("field"))
                for row in official_override_rows(card_id)
                if isinstance(row.get("field"), str)
            }
            for field in ("damage", "hitpoints", "attack_interval_us", "move_speed"):
                # Tier-A patch values always outrank a later/stale third-party
                # page.  Furnace is the important regression: DeckShop still
                # shows the pre-nerf 896 HP while Supercell's August 2025 note
                # pins the live base body to 727 HP.
                if (
                    field in corroboration
                    and field in cards[card_id]
                    and field not in official_fields
                ):
                    cards[card_id][field] = corroboration[field]
            provenance = {
                str(key): list(value) if isinstance(value, (list, tuple)) else [str(value)]
                for key, value in dict(cards[card_id].get("provenance", {})).items()
            }
            level11_sources = provenance.setdefault("level_11_stats", [])
            if DECKSHOP_CORE_SOURCE_ID not in level11_sources:
                level11_sources.append(DECKSHOP_CORE_SOURCE_ID)
            cards[card_id]["provenance"] = provenance
        # Components can be added to an existing hand-curated base row as
        # well as to generated cards.  Cannon Cart is present in the base
        # artifact, so applying this after the scalar merge prevents a stale
        # row from silently bypassing the official health transformation.
        if card_id in HEALTH_TRANSFORM_DEFINITIONS:
            cards[card_id] = deepcopy(cards[card_id])
            mechanics = dict(cards[card_id].get("mechanics", {}))
            mechanics["health_transform"] = dict(HEALTH_TRANSFORM_DEFINITIONS[card_id])
            cards[card_id]["mechanics"] = mechanics
            provenance = {
                str(key): list(value) if isinstance(value, (list, tuple)) else [str(value)]
                for key, value in dict(cards[card_id].get("provenance", {})).items()
            }
            generated = provenance.setdefault("generated_mechanics", [])
            for source_id in (
                "official-may-2025-cannon-cart-rework",
                "official-january-2025-cannon-cart",
            ):
                if source_id not in generated:
                    generated.append(source_id)
            cards[card_id]["provenance"] = provenance
        if card_id in DEATH_DEFINITIONS and DEATH_DEFINITIONS[card_id].get(
            "spawn_children"
        ):
            cards[card_id] = deepcopy(cards[card_id])
            provenance = {
                str(key): list(value) if isinstance(value, (list, tuple)) else [str(value)]
                for key, value in dict(cards[card_id].get("provenance", {})).items()
            }
            generated = provenance.setdefault("generated_mechanics", [])
            if SPLIT_SOURCE_ID not in generated:
                generated.append(SPLIT_SOURCE_ID)
            cards[card_id]["provenance"] = provenance
        if card_id == "goblin-cage":
            cards[card_id] = deepcopy(cards[card_id])
            provenance = {
                str(key): list(value) if isinstance(value, (list, tuple)) else [str(value)]
                for key, value in dict(cards[card_id].get("provenance", {})).items()
            }
            generated = provenance.setdefault("generated_mechanics", [])
            if GOBLIN_BRAWLER_SOURCE_ID not in generated:
                generated.append(GOBLIN_BRAWLER_SOURCE_ID)
            cards[card_id]["provenance"] = provenance
    # Internal child forms are executable entities but are not playable cards;
    # keeping them outside ``interaction_set`` preserves the fixed V1 roster
    # contract while preventing generic Goblin stats from being substituted.
    cards.setdefault("bush-goblin", _bush_goblin_raw(cards))
    cards.setdefault("barbarian", _barbarian_raw(cards))
    cards.setdefault("goblin", _goblin_raw(cards))
    cards.setdefault("phoenix-egg", _phoenix_egg_raw(cards))
    cards.setdefault("golemite", _golemite_raw(cards))
    cards.setdefault("elixir-golemite", _elixir_golemite_raw(cards))
    cards.setdefault("elixir-blob", _elixir_blob_raw(cards))
    cards.setdefault("lava-pup", _lava_pup_raw(cards))
    cards.setdefault("spear-goblin", _spear_goblin_raw(cards))
    cards.setdefault("rascal-boy", _rascal_boy_raw(cards))
    cards.setdefault("rascal-girl", _rascal_girl_raw(cards))
    cards.setdefault("cursed-hog", _cursed_hog_raw(cards))
    cards.setdefault("goblin-brawler", _goblin_brawler_raw(cards))
    cards.setdefault("cannon-cart-building", _cannon_cart_building_raw(cards))
    raw["ruleset_id"] = ROSTER_RULESET_ID
    raw["cards"] = {card_id: cards[card_id] for card_id in sorted(cards)}
    raw["interaction_set"] = sorted(roster.eligible_cards)
    raw["metadata"] = {
        **dict(raw.get("metadata", {})),
        "status": "roster-complete-provisional",
        "roster_id": roster.roster_id,
        "release_cutoff_exclusive": roster.release_cutoff_exclusive.isoformat(),
        # Never embed wall-clock time in a ruleset payload: rebuilding the
        # artifact must reproduce the same content hash byte-for-byte.
        "generated_at": CATALOG_GENERATED_AT,
        "training_ready": False,
        "training_blockers": [
            "exact per-card release-date lineage",
            "field-level Level-11 reconciliation",
            "card-specific mechanic tests and held-out video evidence",
        ],
    }
    raw["sources"] = {
        **dict(raw.get("sources", {})),
        LEVEL11_SOURCE_ID: {
            "confidence_tier": "B",
            "kind": "pinned-structured-level11-snapshot",
            "url": LEVEL11_SOURCE_PAYLOAD.get("source_url"),
            "retrieved_at": LEVEL11_SOURCE_PAYLOAD.get("retrieved_at", "2026-04-12"),
            "published_at": None,
            "sha256": source_sha256(Path(__file__).resolve().parent / "sources" / "level11_card_stats.json"),
            "lineage": "RoyaleAPI cr-api-data-derived Level-11 card snapshot",
            "note": "Comparison source only; official patch overrides and video evidence outrank it.",
        },
        DECKSHOP_SOURCE_ID: {
            "confidence_tier": "B",
            "kind": "independent-level11-card-page",
            "url": "https://www.deckshop.pro/card/detail/battle-healer",
            "retrieved_at": "2026-08-14",
            "published_at": None,
            "sha256": source_sha256(DECKSHOP_SOURCE_PATH),
            "lineage": "DeckShop",
            "note": "Independent Battle Healer Level-11 corroboration; official patch notes remain higher priority.",
        },
        DECKSHOP_CORE_SOURCE_ID: {
            "confidence_tier": "B",
            "kind": "independent-level11-card-pages",
            "url": "https://www.deckshop.pro/card/detail/fireball",
            "retrieved_at": "2026-08-15",
            "published_at": None,
            "sha256": source_sha256(DECKSHOP_CORE_SOURCE_PATH),
            "lineage": "DeckShop",
            "note": "Independent Furnace/Fireball/The Log/Goblin Curse Level-11 corroboration; official Crown Tower reductions remain higher priority.",
        },
        DECKSHOP_HEAL_SPIRIT_SOURCE_ID: {
            "confidence_tier": "B",
            "kind": "independent-level11-card-page",
            "url": "https://www.deckshop.pro/card/detail/heal-spirit",
            "retrieved_at": "2026-08-16",
            "published_at": None,
            "sha256": source_sha256(DECKSHOP_HEAL_SPIRIT_SOURCE_PATH),
            "lineage": "DeckShop",
            "note": "DeckShop Level-11 healing amount; the executable radius/recipient classes remain explicit provisional assumptions, while official Spirit HP and Crown-Tower rules remain higher priority.",
        },
        "deckmelon-phoenix-2026-08-15": {
            "confidence_tier": "B",
            "kind": "independent-level11-card-page",
            "url": "https://deckmelon.com/cards/phoenix",
            "retrieved_at": "2026-08-15",
            "published_at": None,
            "sha256": None,
            "lineage": "DeckMelon",
            "note": "Phoenix Level-11 body, egg, and death-damage corroboration; official Supercell rebirth values remain higher priority.",
        },
        "official-march-2026": {
            "confidence_tier": "A",
            "kind": "official-patch-note",
            "url": "https://supercell.com/en/games/clashroyale/blog/release-notes/march-balance-changes-2026/",
            "retrieved_at": "2026-08-15",
            "published_at": "2026-03-04",
            "sha256": None,
            "lineage": "Supercell official March 2026 balance changes",
            "note": "Phoenix Egg HP 240→317 and lifetime 4.3s→3.8s at Level 11.",
        },
        GOBLIN_MACHINE_SOURCE_ID: {
            "confidence_tier": "B",
            "kind": "independent-card-mechanics-reference",
            "url": "https://royaleapi.com/blog/goblin-machine-new-card-2024-summer-update?lang=de",
            "retrieved_at": "2026-08-15",
            "published_at": "2024-06-15",
            "sha256": None,
            "lineage": "RoyaleAPI",
            "note": "Goblin Machine two-weapon architecture, 2.5–5 tile rocket annulus, 1.5 tile splash, and air/ground rocket targeting; official later patches override changed numeric values.",
        },
        "official-may-2025-cannon-cart-rework": {
            "confidence_tier": "A",
            "kind": "official-patch-note",
            "url": "https://supercell.com/en/games/clashroyale/blog/release-notes/may-balance-changes-2/",
            "retrieved_at": "2026-08-16",
            "published_at": "2025-05-05",
            "sha256": None,
            "lineage": "Supercell",
            "note": "Cannon Cart shield and body health are combined; the cart becomes a building at 50% HP.",
        },
        "official-january-2025-cannon-cart": {
            "confidence_tier": "A",
            "kind": "official-patch-note",
            "url": "https://supercell.com/en/games/clashroyale/blog/release-notes/january-balance-changes-2/",
            "retrieved_at": "2026-08-16",
            "published_at": "2025-01-07",
            "sha256": None,
            "lineage": "Supercell",
            "note": "The stationary Cannon Cart form's 824 HP was aligned with the moving form's shield before the May shared-pool rework.",
        },
        "official-august-2024-cannon-cart": {
            "confidence_tier": "A",
            "kind": "official-patch-note",
            "url": "https://supercell.com/en/games/clashroyale/blog/news/august-balance-changes/",
            "retrieved_at": "2026-08-16",
            "published_at": "2024-08-06",
            "sha256": None,
            "lineage": "Supercell",
            "note": "Shielded Cannon hit speed was changed to 0.9 seconds to align with the broken form.",
        },
        "deckshop-cannon-cart-2026-08-16": {
            "confidence_tier": "B",
            "kind": "independent-level11-card-page",
            "url": "https://www.deckshop.pro/card/detail/cannon-cart",
            "retrieved_at": "2026-08-16",
            "published_at": None,
            "sha256": None,
            "lineage": "DeckShop",
            "note": "Current Level-11 Cannon Cart scalar page: 1,809 HP, 212 damage, 0.9 s hit speed, 5.5 range.",
        },
        SPLIT_SOURCE_ID: {
            "confidence_tier": "B",
            "kind": "independent-level11-split-card-pages",
            "url": SPLIT_SOURCE_PAYLOAD.get("source_url"),
            "retrieved_at": SPLIT_SOURCE_PAYLOAD.get("retrieved_at", CATALOG_GENERATED_AT),
            "published_at": None,
            "sha256": source_sha256(SPLIT_SOURCE_PATH),
            "lineage": "DeckMelon and DeckShop",
            "note": "Level-11 internal child forms for Golem, Elixir Golem, Lava Hound, and Goblin Giant. Child geometry, split timing, and elixir transfer remain video-validation targets.",
        },
        GOBLIN_BRAWLER_SOURCE_ID: {
            "confidence_tier": "B",
            "kind": "independent-level11-card-page",
            "url": GOBLIN_BRAWLER_SOURCE_PAYLOAD.get("source_url"),
            "retrieved_at": GOBLIN_BRAWLER_SOURCE_PAYLOAD.get(
                "retrieved_at", CATALOG_GENERATED_AT
            ),
            "published_at": None,
            "sha256": source_sha256(GOBLIN_BRAWLER_SOURCE_PATH),
            "lineage": "DeckMelon",
            "note": "Level-11 Goblin Brawler body released by Goblin Cage: one ground-targeting fast melee body. Release offset, activation frame, and retarget behavior remain video-validation targets.",
        },
        "official-august-2025-golemite": {
            "confidence_tier": "A",
            "kind": "official-patch-note",
            "url": "https://supercell.com/en/games/clashroyale/blog/release-notes/august-balance-changes-2/",
            "retrieved_at": "2026-08-16",
            "published_at": "2025-08-04",
            "sha256": None,
            "lineage": "Supercell",
            "note": "Official August 2025 Golemite damage change from 48 to 84 at the fixed reference level.",
        },
        "official-december-2025-lava-pup": {
            "confidence_tier": "A",
            "kind": "official-patch-note",
            "url": "https://supercell.com/en/games/clashroyale/blog/release-notes/december-balance-changes/",
            "retrieved_at": "2026-08-16",
            "published_at": "2025-12-01",
            "sha256": None,
            "lineage": "Supercell",
            "note": "Official December 2025 Lava Pup HP change from 217 to 215; the fixed V1 source snapshot predates that change and remains explicitly marked for reconciliation.",
        },
        CATALOG_SOURCE_ID: {
            "confidence_tier": "E",
            "kind": "local-catalog-generated-definition",
            "url": None,
            "retrieved_at": "2026-08-14",
            "published_at": None,
            "sha256": None,
            "lineage": "src/cr_bot/domain/card_metadata.py -> simulator/catalog.py",
            "note": "Provisional dispatch surface only; never sufficient for fidelity readiness.",
        },
    }
    for source_id, source in load_official_overrides().get("source_records", {}).items():
        raw["sources"].setdefault(
            source_id,
            {
                "confidence_tier": source.get("confidence_tier", "A"),
                "kind": source.get("kind", "official-patch-note"),
                "url": source.get("url"),
                "retrieved_at": source.get("retrieved_at", CATALOG_GENERATED_AT),
                "published_at": source.get("published_at"),
                "sha256": source.get("sha256"),
                "lineage": source.get("lineage"),
                "note": source.get("note"),
            },
        )
    raw["content_hash"] = calculate_content_hash(raw)
    return raw


def write_roster_ruleset(path: Path | None = None) -> Path:
    """Write the generated payload atomically and return its path."""

    destination = path or ruleset_path(ROSTER_RULESET_ID)
    payload = build_roster_ruleset_raw()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def build_fixed_ruleset_raw() -> dict[str, Any]:
    """Build the immutable V1 artifact from the already generated roster.

    V1 intentionally has one constant card table.  The date-stamped source
    records remain provenance for future migrations, but callers should load
    ``v1`` rather than selecting a balance version at runtime.
    """

    raw = build_roster_ruleset_raw()
    raw["ruleset_id"] = FIXED_RULESET_ID
    raw["metadata"] = {
        **dict(raw.get("metadata", {})),
        "status": "v1-fixed",
        "fixed_data": True,
        "versioning_policy": "constant-until-v2",
    }
    raw["content_hash"] = calculate_content_hash(raw)
    return raw


def write_fixed_ruleset(path: Path | None = None) -> Path:
    """Write the canonical V1 ruleset artifact."""

    destination = path or ruleset_path(FIXED_RULESET_ID)
    payload = build_fixed_ruleset_raw()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


__all__ = [
    "BUILDING_IDS",
    "CHARGE_ATTACK_DEFINITIONS",
    "DASH_DEFINITIONS",
    "CATALOG_SOURCE_ID",
    "CATALOG_GENERATED_AT",
    "DECKSHOP_SOURCE_ID",
    "DECKSHOP_HEAL_SPIRIT_SOURCE_ID",
    "DEATH_DEFINITIONS",
    "GOBLIN_BRAWLER_SOURCE_ID",
    "HEALTH_TRANSFORM_DEFINITIONS",
    "SPLIT_SOURCE_ID",
    "ELIXIR_GENERATION",
    "ROSTER_RULESET_ID",
    "FIXED_RULESET_ID",
    "SPAWN_COUNTS",
    "SHIELD_DEFINITIONS",
    "SPAWN_CHILDREN_DEFINITIONS",
    "DEATH_RAGE_DEFINITIONS",
    "DEPLOY_EFFECT_DEFINITIONS",
    "JUMP_DEFINITIONS",
    "SPAWNER_DEFINITIONS",
    "SPAWN_ON_IMPACT",
    "PERSISTENT_EFFECT_DEFINITIONS",
    "PASSIVE_SPAWNER_IDS",
    "STATUS_DEFINITIONS",
    "HOOK_DEFINITIONS",
    "RAMP_ATTACK_DEFINITIONS",
    "REVIVE_DEFINITIONS",
    "build_roster_ruleset_raw",
    "build_fixed_ruleset_raw",
    "write_fixed_ruleset",
    "write_roster_ruleset",
]
