"""Deterministic generated scenarios for the complete opponent roster."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .actions import PlayCardAction
from .engine import ENGINE_VERSION
from .roster import PLAYER_DECK, load_opponent_roster
from .ruleset import Ruleset
from .scenario import Scenario, ScheduledAction


SCENARIO_FACTORY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GeneratedScenario:
    scenario: Scenario
    card_id: str
    mechanic: str
    seed: int


def _stable_seed(card_id: str, mechanic: str, index: int) -> int:
    digest = hashlib.sha256(f"{card_id}:{mechanic}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def card_mechanics(ruleset: Ruleset, card_id: str) -> tuple[str, ...]:
    """Derive the mechanic inventory from a card's executable components."""

    card = ruleset.card(card_id)
    mechanics = {
        "deployment",
        "target_legality",
        "lifecycle",
    }
    if card.kind == "troop":
        mechanics.update({"movement", "target_acquisition", "attack", "damage", "death"})
        if card.mechanics.get("movement_layer") == "air":
            mechanics.add("air_navigation")
        if card.area_radius_mtile is not None:
            mechanics.add("area_damage")
    elif card.kind == "building":
        mechanics.update({"building_navigation", "lifetime"})
        # Huts, Drill, Cage and the Collector are entities whose executable
        # behavior is a child/resource stream, not a turret attack.  Do not
        # generate phantom target/attack cases from stale legacy damage
        # columns.  Active buildings retain the full combat inventory.
        passive_spawner = (
            card.damage is None
            and card.attack_interval_us is None
            and card.range_mtile is None
            and (
                card.mechanics.get("spawn") is not None
                or card.mechanics.get("elixir_generation") is not None
                or card.mechanics.get("death") is not None
            )
        )
        if passive_spawner:
            mechanics.add("passive_spawner")
        else:
            mechanics.update({"target_acquisition", "attack"})
    elif card.kind == "spell":
        mechanics.update({"spell_geometry", "effect_timing", "victim_selection"})
    if card.projectile is not None:
        mechanics.update({"projectile_origin", "projectile_motion"})
    if card.mechanics.get("status") is not None:
        mechanics.add("status_effect")
    if card.mechanics.get("heal_on_impact") is not None:
        mechanics.add("heal_effect")
    if card.mechanics.get("health_transform") is not None:
        mechanics.update({"health_threshold_transform", "form_change"})
    if card.mechanics.get("spawn") is not None:
        mechanics.add("periodic_spawn")
    if card.mechanics.get("spawn_on_impact") is not None:
        mechanics.add("impact_spawn")
    if card.mechanics.get("persistent_effect") is not None:
        mechanics.add("persistent_area_effect")
        persistent = card.mechanics["persistent_effect"]
        if persistent.get("friendly_status") is not None:
            mechanics.add("friendly_aura")
        if persistent.get("status") is not None:
            mechanics.add("status_effect")
            if persistent["status"].get("on_death_spawn_card_id") is not None:
                mechanics.add("death_effect")
    if card.mechanics.get("clone") is not None:
        mechanics.add("clone_component")
    if card.mechanics.get("elixir_generation") is not None:
        mechanics.add("resource_generation")
    death = card.mechanics.get("death")
    if death is not None:
        mechanics.add("death_effect")
        if death.get("spawn_children"):
            mechanics.add("death_split")
    if card.mechanics.get("target_limit") is not None:
        mechanics.add("target_selection")
    if card.mechanics.get("chain_attack") is not None:
        mechanics.update({"chain_targeting", "status_effect"})
    if card.mechanics.get("multi_target_attack") is not None:
        mechanics.update({"multi_targeting", "status_effect"})
    if card.mechanics.get("reflection") is not None:
        mechanics.update({"reflected_damage", "status_effect"})
    if card.mechanics.get("charge_attack") is not None:
        mechanics.update({"charge_attack", "charge_movement"})
    if card.mechanics.get("dash") is not None:
        mechanics.update({"dash_movement", "dash_attack"})
    if card.mechanics.get("hook") is not None:
        mechanics.update({"hook_targeting", "hook_pull"})
    if card.mechanics.get("recoil_mtile") is not None:
        mechanics.update({"recoil", "area_damage"})
    if card.mechanics.get("ramp_attack") is not None:
        mechanics.update({"ramp_attack", "ramp_reset"})
    if card.mechanics.get("revive") is not None:
        mechanics.update({"revive", "revive_egg"})
    if card.mechanics.get("carrier") is not None:
        mechanics.update({"carrier", "carrier_release"})
    if card.mechanics.get("shield") is not None:
        mechanics.add("shield")
    # Suspicious Bush is permanently hidden until contact; only cards with an
    # authored re-cloak delay receive the attack/reveal lifecycle case.
    if card.mechanics.get("stealth_recloak_us") is not None:
        mechanics.add("stealth_lifecycle")
    if card.mechanics.get("burrow") is not None:
        mechanics.add("burrow")
    if card.mechanics.get("spawn_children") is not None:
        mechanics.add("spawn_composition")
    if card.mechanics.get("line_piercing") is not None:
        mechanics.add("line_piercing")
    if card.mechanics.get("returning_projectile") is not None:
        mechanics.add("returning_projectile")
    if card.mechanics.get("pellets") is not None:
        mechanics.add("pellet_spread")
    if int(card.mechanics.get("knockback_mtile") or 0) > 0:
        mechanics.add("knockback")
    if card.mechanics.get("jump") is not None:
        mechanics.add("jump_landing")
    if card.mechanics.get("deploy_effect") is not None:
        mechanics.add("deployment_effect")
    if card.mechanics.get("death_rage") is not None:
        mechanics.add("death_rage")
    if card.mechanics.get("snare") is not None:
        mechanics.add("snare")
    status = card.mechanics.get("status") or {}
    if status.get("on_death_spawn_card_id") is not None:
        mechanics.add("death_transform")
    death_component = card.mechanics.get("death") or {}
    if death_component.get("spawn_card_id") is not None:
        mechanics.add("death_spawn")
    return tuple(sorted(mechanics))


def _test_cell(card_kind: str, player: int) -> tuple[int, int]:
    if card_kind == "spell":
        return (8, 12 if player == 1 else 19)
    if card_kind == "building":
        return (8, 9 if player == 1 else 20)
    return (3, 8 if player == 1 else 23)


def _pair_test_cell(card_kind: str, player: int, pair_index: int, variant: int) -> tuple[int, int]:
    """Return a separated legal fixture cell for an opponent-card pair.

    The ordinary roster fixture intentionally places one card at a canonical
    anchor.  Pair scenarios need two anchors that remain distinct when both
    cards are buildings or troops.  Keep the offsets in the same lane for the
    canonical case, and mirror that lane for odd variants so the pair matrix
    exercises both bridge approaches without relying on random placement.
    """

    if type(pair_index) is not int or pair_index not in (0, 1):
        raise ValueError("pair_index must be zero or one")
    lane_left = _variant_pattern(variant)[0] == 0
    if card_kind == "spell":
        row = 12 if player == 1 else 19
        columns = (8, 10) if lane_left else (9, 11)
    elif card_kind == "building":
        row = 9 if player == 1 else 22
        columns = (8, 10) if lane_left else (9, 11)
    else:
        row = 8 if player == 1 else 23
        columns = (3, 5) if lane_left else (12, 14)
    return columns[pair_index], row


# Scenario variants are deliberately deterministic rather than random.  The
# factory is used in CI and in nightly million-case sweeps, so a failing case
# must be reproducible from its card/mechanic/variant identity alone.  Variant
# zero is the long-standing canonical fixture; later variants move the same
# legal setup through both bridge lanes and nudge its timing.  Keeping the
# perturbation small is important: the branch under test should change, not
# disappear because a support body was placed on the wrong side of a bridge.
_VARIANT_PATTERNS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 8),
    (0, 16),
    (1, 24),
    (0, 32),
    (1, 40),
)


def _variant_pattern(variant: int) -> tuple[int, int]:
    """Return ``(lane_index, tick_delta)`` for a variant.

    The pattern repeats after six entries, but the variant number remains in
    the scenario ID/seed.  This lets callers request arbitrarily large fuzz
    batches without introducing non-deterministic placement generation.
    """

    if type(variant) is not int or variant < 0:
        raise ValueError("variant must be a non-negative integer")
    return _VARIANT_PATTERNS[variant % len(_VARIANT_PATTERNS)]


def _variant_cell(cell: tuple[int, int], variant: int) -> tuple[int, int]:
    """Move a policy-grid cell while preserving its legal territory.

    Troop fixtures start around column three and are shifted to the mirrored
    bridge around column fourteen.  Building/spell fixtures start near the
    center (column eight) and
    receive the mirrored bridge anchor instead.  Rows intentionally remain
    unchanged: a one-cell shift near a Crown Tower can overlap its footprint
    or fall on a terrain mask even though the canonical cell is legal.
    """

    column, row = cell
    if variant == 0:
        next_column = column
    else:
        # The calibrated fixture uses the left bridge (around column 3).
        # The only other legal full-lane route is the mirrored bridge around
        # column 14; central columns do not cross the river.  Map every
        # fixture coordinate to one of those two bridge lanes while retaining
        # the small +/-1 offsets used by multi-victim setups.
        lane_anchor = 14 if _variant_pattern(variant)[0] else 3
        next_column = lane_anchor + (column - 3 if column <= 4 else 0)
    return (
        max(0, min(17, int(next_column))),
        int(row),
    )


def _variant_actions(
    actions: tuple[ScheduledAction, ...],
    variant: int,
) -> tuple[ScheduledAction, ...]:
    """Apply a reproducible geometry/timing perturbation to scheduled plays."""

    _, tick_delta = _variant_pattern(variant)
    if variant == 0:
        return actions
    shifted: list[ScheduledAction] = []
    for scheduled in actions:
        action = scheduled.action
        if isinstance(action, PlayCardAction):
            action = PlayCardAction(
                action.player,
                action.card_slot,
                _variant_cell(action.cell, variant),
            )
        shifted.append(ScheduledAction(scheduled.tick + tick_delta, action))
    return tuple(sorted(shifted, key=lambda row: (row.tick, row.action.player)))


def _support_card_for_opponent(ruleset: Ruleset, card_id: str) -> str:
    """Choose a stable, cheap opponent troop for friendly-effect fixtures."""

    # Knight and Skeletons are both base cards in the frozen roster and are
    # intentionally used only as deterministic setup bodies.  Avoid putting
    # the same card in two hand slots when the card under test is one of them.
    for candidate in ("knight", "skeletons", "hog-rider"):
        if candidate in ruleset.cards and candidate != card_id:
            return candidate
    raise ValueError(f"no deterministic support troop is available for {card_id}")


# These branches are about a troop interacting with another troop rather than
# simply walking into a Crown Tower.  A Hog is intentionally *not* used as the
# fixture body here: Hog Rider is buildings-only, so it can make a hook,
# reflection, chain, splash, or death test pass while the actual target set is
# empty.  Musketeer is in the fixed player deck and is a legal air/ground
# target for every one of these deterministic probes.
_TROOP_TARGET_MECHANICS = frozenset(
    {
        "area_damage",
        "carrier",
        "carrier_release",
        "dash_attack",
        "dash_movement",
        "death",
        "death_effect",
        "death_split",
        "death_spawn",
        "form_change",
        "health_threshold_transform",
        "heal_effect",
        "hook_pull",
        "hook_targeting",
        "multi_targeting",
        "chain_targeting",
        "ramp_attack",
        "ramp_reset",
        "reflected_damage",
        "recoil",
        "revive",
        "revive_egg",
        "status_effect",
        "stealth_lifecycle",
        "target_selection",
        "burrow",
        "line_piercing",
        "returning_projectile",
        "pellet_spread",
        "knockback",
        "jump_landing",
        "deployment_effect",
        "death_rage",
        "snare",
        "death_transform",
        "shield",
    }
)

_TROOP_VICTIM_MECHANICS = _TROOP_TARGET_MECHANICS | frozenset(
    {
        "attack",
        "damage",
        "projectile_origin",
        "projectile_motion",
        "target_acquisition",
        "target_legality",
    }
)

_MULTI_VICTIM_MECHANICS = frozenset(
    {"area_damage", "chain_targeting", "multi_targeting", "target_selection"}
)


def _generated_support_plan(
    ruleset: Ruleset,
    card: Any,
    card_id: str,
    mechanic: str,
    *,
    variant: int = 0,
) -> tuple[tuple[ScheduledAction, ...], tuple[dict[str, Any], ...], tuple[int, int]]:
    """Build deterministic target/setup actions for one mechanic case.

    The original one-action factory proved that a card could be accepted, but
    it left active buildings with no troop to acquire and most spells with an
    empty victim set.  These fixtures remain ordinary legal card plays, never
    privileged entity injection: Hog is used as an enemy target, while a
    Knight/Skeleton is used for friendly Clone/Rage/Mirror setup.
    """

    support_actions: list[ScheduledAction] = []
    required_support: list[dict[str, Any]] = []
    main_tick = 520
    main_cell = _test_cell(card.kind, 1)

    persistent = card.mechanics.get("persistent_effect") or {}
    friendly_setup = bool(
        card.card_id == "mirror"
        or card.mechanics.get("clone") is not None
        or persistent.get("friendly_status") is not None
    )
    passive_spawner = (
        card.kind == "building"
        and card.damage is None
        and card.attack_interval_us is None
        and card.range_mtile is None
        and (
            card.mechanics.get("spawn") is not None
            or card.mechanics.get("elixir_generation") is not None
            or card.mechanics.get("death") is not None
        )
    )
    # Active structures need a moving body for their combat, projectile and
    # target-selection cases.  Death-effect structures (notably Goblin Cage)
    # also need an attacker so their destruction branch is actually reached.
    building_target_case = card.kind == "building" and (
        not passive_spawner
        or card.mechanics.get("death") is not None
    )
    # Most troops naturally reach a Crown Tower in the ordinary generated
    # case.  Components whose behavior is specifically about another troop
    # receive a Hog body so hooks, reflections, dashes and status payloads
    # cannot pass vacuously against an empty lane.  Charge cases deliberately
    # run against a tower so they have room to accumulate their travel meter.
    troop_target_case = card.kind == "troop" and mechanic in _TROOP_VICTIM_MECHANICS
    if friendly_setup:
        # Keep the friendly body inside Clone/Rage's impact radius after its
        # one-second deployment without letting it walk all the way to a
        # tower before the effect lands.
        main_tick = 440
    elif troop_target_case:
        # A Hog placed at tick 400 reaches the bridge quickly.  Deploying the
        # exceptional troop at tick 480 catches it at hook/dash range rather
        # than after it has already collided with the Crown Tower.
        main_tick = 480
    enemy_target_setup = (
        card.kind == "spell"
        and card.mechanics.get("placement_class") == "spell_anywhere"
        and not friendly_setup
    ) or (
        card.kind == "spell"
        and mechanic == "knockback"
        and not friendly_setup
    ) or (
        building_target_case
        and mechanic
        not in {"deployment", "lifecycle", "building_navigation", "lifetime", "passive_spawner", "periodic_spawn", "resource_generation"}
    ) or troop_target_case

    if friendly_setup:
        support_card = _support_card_for_opponent(ruleset, card_id)
        support_cell = (3, 9)
        support_actions.append(
            ScheduledAction(400, PlayCardAction(1, 1, support_cell))
        )
        required_support.append({"player": 1, "card_id": support_card})
        main_cell = support_cell
    elif enemy_target_setup:
        # The player Hog is placed on the same lane.  Spells can target its
        # own side directly; buildings wait until tick 520 so the Hog is fully
        # deployed and walking when the defensive building appears.
        support_cell = (3, 17)
        # A building-targeting Hog is the useful opponent for a defensive
        # building.  Troop-target branches instead need Musketeer so the
        # tested entity can acquire and damage a troop (and eventually die),
        # while Cannon Cart/Electro Giant retain their ranged/reflective
        # Musketeer fixture.
        use_cannon = (
            card.kind == "troop"
            and bool(card.mechanics.get("building_only"))
            and mechanic in _TROOP_VICTIM_MECHANICS
            and "cannon" in ruleset.cards
        )
        use_ice_golem = (
            mechanic in {"hook_pull", "hook_targeting"}
            and "ice-golem" in ruleset.cards
        )
        use_musketeer = (
            (
                card.kind == "troop"
                and mechanic in _TROOP_VICTIM_MECHANICS
            )
            or card_id in {"cannon-cart", "electro-giant"}
            or (card.kind == "spell" and mechanic == "knockback")
        ) and "musketeer" in ruleset.cards
        if use_cannon:
            support_card = "cannon"
            support_slot = 0
            support_cell = (3, 20)
        elif use_ice_golem:
            support_card = "ice-golem"
            support_slot = 2
            support_cell = (3, 23)
        else:
            support_card = "musketeer" if use_musketeer else "hog-rider"
            support_slot = 1 if support_card == "musketeer" else 0
        # Building-only troops need to reach a building before the defensive
        # Cannon is played.  If the Cannon is already live at tick 400 its
        # first three shots plus a Princess Tower shot can kill short-lived
        # rams and golems before their authored attack ever occurs.  Playing
        # the legal support building after the opponent troop is on the lane
        # preserves the real target-acquisition/attack ordering while still
        # using the fixed player deck (no privileged entity injection).
        support_tick = (
            520
            if use_cannon
            else 470
            if mechanic in {"jump_landing", "deployment_effect", "knockback"}
            else 400
        )
        support_actions.append(ScheduledAction(support_tick, PlayCardAction(0, support_slot, support_cell)))
        required_support.append({"player": 0, "card_id": support_card})
        if card.kind == "spell":
            main_cell = (
                # The target body starts at row 17 and is already walking
                # toward its tower while the ballistic spell travels from
                # the king tower.  A row-17 impact therefore lands behind
                # it by several tiles, making status-only fixtures pass
                # vacuously.  Place status-effect impacts at the predicted
                # row-12 position so the generated oracle observes the
                # actual status application rather than an empty radius.
                (support_cell[0], 12)
                if mechanic == "status_effect"
                else support_cell
                if card.mechanics.get("placement_class") == "spell_anywhere"
                else (support_cell[0], 12)
            )
        elif card.kind == "troop":
            # Put troop-victim branches just across the river.  This is still
            # a legal player-1 deployment cell, but lets a short-lived Spirit,
            # swarm, or ranged body reach its real target before the Crown
            # Tower deletes it.
            main_cell = (3, 14)

    if card.kind == "spell" and enemy_target_setup:
        # Put the impact near the just-deployed target before it walks away;
        # this is especially important for Freeze/Snowball/Zap and the
        # persistent poison/curse fields, whose status branch otherwise sees
        # an empty radius after a 45-second delay.
        # Troop deployment takes one canonical second.  Scheduling the spell
        # after that deployment (rather than in the same tick) is essential
        # for impact-only components such as Log knockback to observe a real
        # victim instead of passing vacuously at an empty coordinate.
        main_tick = 460

    # Heal Spirit needs both a friendly body to receive the heal and an enemy
    # body to trigger its suicide impact.  The owner-1 setup card remains in
    # the opponent deck, while the fixed player Musketeer supplies the legal
    # enemy target selected above.
    if mechanic == "heal_effect":
        friendly_card = _support_card_for_opponent(ruleset, card_id)
        support_actions.append(
            ScheduledAction(400, PlayCardAction(1, 1, (3, 14)))
        )
        required_support.append({"player": 1, "card_id": friendly_card})

    # Area, chain, multi-target, and Lightning selection cases need more than
    # one legal victim.  Play two additional fixed-deck bodies on the same
    # lane after Musketeer so the engine must perform its real candidate
    # ordering rather than only hitting a Crown Tower.
    if mechanic in _MULTI_VICTIM_MECHANICS and enemy_target_setup and not (
        card.kind == "troop" and bool(card.mechanics.get("building_only"))
    ):
        support_actions.append(
            ScheduledAction(401, PlayCardAction(0, 1, (2, 17)))
        )
        required_support.append({"player": 0, "card_id": "ice-golem"})
        support_actions.append(
            ScheduledAction(402, PlayCardAction(0, 0, (4, 17)))
        )
        required_support.append({"player": 0, "card_id": "hog-rider"})

    main_action = ScheduledAction(main_tick, PlayCardAction(1, 0, main_cell))
    actions = tuple(sorted((*support_actions, main_action), key=lambda row: (row.tick, row.action.player)))
    varied_actions = _variant_actions(actions, variant)
    varied_main_cell = _variant_cell(main_cell, variant)
    return varied_actions, tuple(required_support), varied_main_cell


def _required_event_kinds(card: Any, mechanic: str) -> tuple[str, ...]:
    """Return event obligations for branches that a generated fixture targets.

    A card-play obligation only proves that the action was accepted.  These
    small event-level obligations make the synthetic matrix fail when a
    scenario silently exercises only deployment while claiming to cover an
    attack, spell victim, status, spawn, or transformation component.
    """

    if card.kind == "building":
        passive_spawner = (
            card.damage is None
            and card.attack_interval_us is None
            and card.range_mtile is None
            and (
                card.mechanics.get("spawn") is not None
                or card.mechanics.get("elixir_generation") is not None
                or card.mechanics.get("death") is not None
            )
        )
        mapping = {
            "attack": ("attack_started",),
            "damage": ("damage_applied",),
            "area_damage": ("damage_applied",),
            "projectile_origin": ("projectile_spawned",),
            "projectile_motion": ("projectile_resolved",),
            "target_acquisition": ("target_changed",),
            "target_legality": ("target_changed",) if not passive_spawner else (),
            "death": ("entity_died",),
            "death_effect": ("entity_died",),
            "death_split": (
                ("carrier_released",)
                if card.mechanics.get("carrier") is not None
                else ("death_spawn",)
            ),
            "carrier": ("carrier_child_created",),
            "carrier_release": ("carrier_released",),
            "lifetime": ("building_expired",),
            "periodic_spawn": ("entity_spawned",),
            "resource_generation": ("elixir_generated",),
            # A shield case is not complete when the child merely spawned:
            # it must take a real incoming hit that consumes the layer and
            # then expose the broken transition.  The target fixture below
            # uses the fixed-deck Musketeer for this branch.
            "shield": ("shield_damaged", "shield_broken"),
        }
        return mapping.get(mechanic, ())
    if card.kind == "troop":
        mapping = {
            # Trigger/contact troops (Suspicious Bush) consume themselves on
            # contact instead of entering the ordinary attack scheduler.
            "attack": (
                ("entity_triggered",)
                if card.mechanics.get("trigger_on_target")
                else ("attack_started",)
            ),
            "damage": ("damage_applied",),
            "area_damage": ("damage_applied",),
            "projectile_origin": ("projectile_spawned",),
            "projectile_motion": ("projectile_resolved",),
            "target_acquisition": ("target_changed",),
            "target_legality": ("target_changed",),
            "periodic_spawn": ("entity_spawned",),
            "status_effect": ("status_applied",),
            "death": (
                ("entity_transformed",)
                if card.mechanics.get("health_transform") is not None
                else ("entity_died",)
            ),
            "death_effect": ("entity_died",),
            "death_split": (
                ("carrier_released",)
                if card.mechanics.get("carrier") is not None
                else ("death_spawn",)
            ),
            "carrier": ("carrier_child_created",),
            "carrier_release": ("carrier_released",),
            "chain_targeting": ("chain_hit",),
            "multi_targeting": ("multi_target_hit",),
            "recoil": ("recoil_applied",),
            "ramp_attack": ("ramp_stage_changed",),
            "ramp_reset": ("ramp_reset",),
            "revive": ("phoenix_death_rebirth_started",),
            # The generated action-boundary fixture always proves that the
            # egg was materialized.  A dedicated state fixture/test covers
            # the delayed hatch without relying on a tower not selecting it.
            "revive_egg": ("phoenix_egg_created",),
            "dash_attack": ("dash_started",),
            "charge_movement": ("charge_started",),
            "dash_movement": ("dash_started",),
            "hook_pull": ("hook_pulled",),
            "hook_targeting": ("hook_pulled",),
            "reflected_damage": ("reflected_damage",),
            "health_threshold_transform": ("entity_transformed",),
            "form_change": ("entity_transformed",),
            "shield": ("shield_damaged", "shield_broken"),
            "stealth_lifecycle": ("stealth_broken", "stealth_started"),
            "burrow": ("burrow_started", "burrow_emerged"),
            "spawn_composition": ("entity_created",),
            "line_piercing": ("piercing_hit",),
            "returning_projectile": ("projectile_return_started", "piercing_hit"),
            "pellet_spread": ("projectile_spawned",),
            "knockback": ("knockback_applied",),
            "jump_landing": ("jump_started", "jump_landed"),
            "deployment_effect": ("deployment_effect",),
            "death_rage": ("death_rage_created",),
            "snare": ("status_applied",),
            "death_transform": ("death_transform",),
            "death_spawn": ("death_spawn",),
        }
        return mapping.get(mechanic, ())
    if card.kind == "spell":
        if card.card_id == "mirror":
            return ("card_mirrored",)
        mapping = {
            "projectile_motion": ("projectile_resolved",),
            "spell_geometry": ("projectile_resolved",),
            "victim_selection": ("projectile_resolved",),
            "effect_timing": ("projectile_resolved",),
            "status_effect": ("status_applied",),
            "persistent_area_effect": ("area_effect_created",),
            "friendly_aura": ("status_applied",),
            "impact_spawn": ("entity_spawned",),
            "clone_component": ("entity_cloned",),
            "target_selection": ("projectile_resolved",),
        }
        return mapping.get(mechanic, ())
    return ()


def _required_event_matches(card: Any, mechanic: str) -> tuple[dict[str, Any], ...]:
    """Return source/card-specific event predicates for a generated case.

    Event-kind obligations prevent vacuous deployment-only cases.  These
    predicates go one step further: a Musketeer support shot must not satisfy
    a generated Golem ``death`` case, and a child spawned by a different card
    must not satisfy a periodic-spawn obligation.
    """

    card_id = str(card.card_id)
    mixed_children = tuple(card.mechanics.get("spawn_children") or ())
    child_ids = tuple(str(row.get("card_id")) for row in mixed_children)

    def event_source_card_id(kind: str) -> str:
        """Return the child body which owns a generated event.

        Goblin Gang and Rascals intentionally do not create an aggregate
        parent entity.  Their attack/death/projectile events are emitted by
        the materialized child bodies, so generated predicates must follow
        that ownership instead of falsely asking for an impossible parent
        event.
        """

        if not child_ids:
            return card_id
        if kind in {"projectile_origin", "projectile_motion", "pellet_spread", "line_piercing", "returning_projectile"}:
            for child_id in child_ids:
                if child_id in {"spear-goblin", "rascal-girl"}:
                    return child_id
        return child_ids[0]

    source_card_id = event_source_card_id(mechanic)
    if card_id == "mirror":
        return ({"kind": "card_mirrored", "filters": {"player": 1}},)
    if mechanic == "attack" and card.mechanics.get("trigger_on_target"):
        return ({"kind": "entity_triggered", "filters": {"card_id": card_id}},)
    if mechanic in {"attack", "charge_attack", "dash_attack"}:
        return ({"kind": "attack_started", "filters": {"card_id": source_card_id}},)
    if mechanic in {"damage", "area_damage"}:
        return ({"kind": "damage_applied", "filters": {"source_card_id": source_card_id}},)
    if mechanic == "projectile_origin":
        return ({"kind": "projectile_spawned", "filters": {"card_id": source_card_id, "player": 1}},)
    if mechanic in {"projectile_motion", "spell_geometry", "victim_selection", "effect_timing"}:
        return ({"kind": "projectile_resolved", "filters": {"card_id": source_card_id}},)
    if mechanic == "death" and card.mechanics.get("health_transform") is not None:
        return ({"kind": "entity_transformed", "filters": {"source_card_id": card_id}},)
    if mechanic in {"death", "death_effect"} and card.kind != "spell":
        card_filter: object = source_card_id
        if child_ids:
            card_filter = {"one_of": list(child_ids)}
        return ({"kind": "entity_died", "filters": {"card_id": card_filter, "player": 1}},)
    if mechanic == "death_effect" and card.kind == "spell":
        return ({"kind": "area_effect_created", "filters": {"card_id": card_id}},)
    if mechanic == "death_split":
        carrier = card.mechanics.get("carrier")
        if carrier is not None:
            return ({"kind": "carrier_released", "filters": {"parent_card_id": card_id}},)
        return ({"kind": "death_spawn", "filters": {"parent_card_id": card_id}},)
    if mechanic == "carrier":
        child_id = str((card.mechanics.get("carrier") or {}).get("child_card_id") or "")
        return ({"kind": "carrier_child_created", "filters": {"card_id": child_id}},)
    if mechanic == "carrier_release":
        return ({"kind": "carrier_released", "filters": {"parent_card_id": card_id}},)
    if mechanic == "chain_targeting":
        return ({"kind": "chain_hit", "filters": {"source_card_id": card_id, "target_index": 2}},)
    if mechanic == "multi_targeting":
        # One hit can be the ordinary primary attack; index two proves the
        # component actually selected a second legal victim.
        return ({"kind": "multi_target_hit", "filters": {"source_card_id": card_id, "target_index": 2}},)
    if mechanic == "recoil":
        return ({"kind": "recoil_applied", "filters": {"source_card_id": card_id}},)
    if mechanic in {"ramp_attack", "ramp_reset"}:
        return ({"kind": "ramp_stage_changed", "filters": {"card_id": card_id}},)
    if mechanic in {"revive"}:
        return ({"kind": "phoenix_death_rebirth_started", "filters": {"card_id": card_id}},)
    if mechanic == "revive_egg":
        return ({"kind": "phoenix_egg_created", "filters": {"card_id": "phoenix-egg", "player": 1}},)
    if mechanic in {"dash_movement"}:
        return ({"kind": "dash_started", "filters": {"card_id": card_id}},)
    if mechanic in {"charge_movement"}:
        return ({"kind": "charge_started", "filters": {"card_id": card_id}},)
    if mechanic in {"hook_pull", "hook_targeting"}:
        return ({"kind": "hook_pulled", "filters": {"card_id": card_id}},)
    if mechanic in {"health_threshold_transform", "form_change"}:
        return ({"kind": "entity_transformed", "filters": {"source_card_id": card_id}},)
    if mechanic in {"movement", "air_navigation"} and card.mechanics.get("health_transform") is not None:
        return ({"kind": "entity_transformed", "filters": {"source_card_id": card_id}},)
    if mechanic == "heal_effect":
        return ({"kind": "healing_impact_resolved", "filters": {"source_card_id": card_id}},)
    if mechanic == "periodic_spawn":
        spawn = card.mechanics.get("spawn") or {}
        child_id = spawn.get("card_id")
        if child_id is not None:
            # Spawn events carry the child card and parent UID; the parent
            # card can be recovered from the authoritative entity pool but is
            # intentionally not duplicated in the flat event payload.
            return ({"kind": "entity_spawned", "filters": {"card_id": str(child_id)}},)
    if mechanic == "resource_generation":
        # The flat resource event carries the authoritative parent UID and
        # owner, not a redundant card ID.  No other V1 component emits this
        # event, so the kind itself is the source-specific predicate.
        return ({"kind": "elixir_generated", "filters": {}},)
    if mechanic == "persistent_area_effect":
        return ({"kind": "area_effect_created", "filters": {"card_id": card_id}},)
    if mechanic == "impact_spawn":
        spawn = card.mechanics.get("spawn_on_impact") or {}
        child_id = spawn.get("card_id")
        if child_id is not None:
            return ({"kind": "entity_spawned", "filters": {"card_id": str(child_id)}},)
        return ()
    if mechanic == "clone_component":
        return ({"kind": "entity_cloned", "filters": {"player": 1}},)
    if mechanic == "status_effect":
        return ({"kind": "status_applied", "filters": {}},)
    if mechanic == "shield":
        return (
            # The fixed support body may die before its shot lands; the
            # Princess Tower is an equally valid authoritative shield
            # attacker.  This fixture has only the tested shield target, so
            # an unfiltered shield event remains source-specific without
            # hard-coding which attacker wins the race.
            {"kind": "shield_damaged", "filters": {}},
            {"kind": "shield_broken", "filters": {}},
        )
    if mechanic == "stealth_lifecycle":
        return (
            {"kind": "stealth_broken", "filters": {"card_id": card_id}},
            {"kind": "stealth_started", "filters": {"card_id": card_id}},
        )
    if mechanic == "burrow":
        return (
            {"kind": "burrow_started", "filters": {"card_id": card_id}},
            {"kind": "burrow_emerged", "filters": {"card_id": card_id}},
        )
    if mechanic == "spawn_composition":
        children = card.mechanics.get("spawn_children") or ()
        if not children:
            return ({"kind": "entity_created", "filters": {"card_id": card_id, "player": 1}},)
        return tuple(
            {
                "kind": "entity_created",
                "filters": {"card_id": str(child.get("card_id")), "player": 1},
            }
            for child in children
        )
    if mechanic == "line_piercing":
        return ({"kind": "piercing_hit", "filters": {"source_card_id": card_id}},)
    if mechanic == "returning_projectile":
        return ({"kind": "projectile_return_started", "filters": {"card_id": card_id}},)
    if mechanic == "pellet_spread":
        return ({"kind": "projectile_spawned", "filters": {"card_id": card_id, "player": 1, "pellet_index": 0}},)
    if mechanic == "knockback":
        return ({"kind": "knockback_applied", "filters": {}},)
    if mechanic == "jump_landing":
        return (
            {"kind": "jump_started", "filters": {"card_id": card_id}},
            {"kind": "jump_landed", "filters": {"card_id": card_id}},
        )
    if mechanic == "deployment_effect":
        return ({"kind": "deployment_effect", "filters": {"card_id": card_id}},)
    if mechanic == "death_rage":
        return ({"kind": "death_rage_created", "filters": {"card_id": card_id}},)
    if mechanic == "snare":
        return ({"kind": "status_applied", "filters": {"status": "slow"}},)
    if mechanic == "death_transform":
        return ({"kind": "death_transform", "filters": {"source_card_id": card_id}},)
    if mechanic == "death_spawn":
        return ({"kind": "death_spawn", "filters": {"parent_card_id": card_id}},)
    return ()


def generate_card_scenarios(
    ruleset: Ruleset,
    card_id: str,
    *,
    per_mechanic: int = 1,
) -> tuple[GeneratedScenario, ...]:
    """Generate legal, runnable scenarios for one opponent card.

    The opponent card is always slot zero and is played after enough clock time
    for every V1 elixir cost.  This keeps the factory independent of random
    initial hand order and makes generated cases comparable across cards.  The
    required-play and event oracles are part of each generated case so
    validation can prove that the declared branch ran instead of merely
    surviving a rejected action or targetless deployment.
    """

    if card_id not in ruleset.cards:
        raise KeyError(card_id)
    if type(per_mechanic) is not int or per_mechanic < 1:
        raise ValueError("per_mechanic must be positive")
    card = ruleset.card(card_id)
    rows: list[GeneratedScenario] = []
    for mechanic_index, mechanic in enumerate(card_mechanics(ruleset, card_id)):
        for variant in range(per_mechanic):
            seed = _stable_seed(card_id, mechanic, variant)
            actions, required_support, main_cell = _generated_support_plan(
                ruleset, card, card_id, mechanic, variant=variant
            )
            support_card = _support_card_for_opponent(ruleset, card_id)
            required_events = _required_event_kinds(card, mechanic)
            required_event_matches = _required_event_matches(card, mechanic)
            required_state_checks: list[dict[str, Any]] = []
            if (
                card.kind == "troop"
                and mechanic in {"movement", "air_navigation"}
                and card.mechanics.get("health_transform") is None
            ):
                mixed_children = card.mechanics.get("spawn_children") or ()
                movement_card_id = (
                    str(mixed_children[0].get("card_id"))
                    if mixed_children
                    else card_id
                )
                required_state_checks.append(
                    {
                        "type": "entity_moved",
                        "player": 1,
                        "card_id": movement_card_id,
                        "from_cell": list(main_cell),
                    }
                )
            # Lifetime begins on placement for buildings in V1.  Schedule the
            # stop after the action, deployment delay, and the full declared
            # lifetime so a long-lived collector cannot pass as “tested” only
            # because the scenario ended early.
            lifetime_ticks = 0
            if mechanic == "lifetime" and card.lifetime_us is not None:
                main_action_tick = max(
                    action.tick for action in actions if action.action.player == 1
                )
                lifetime_ticks = (
                    main_action_tick
                    + (int(card.deploy_time_us) + int(card.lifetime_us) + ruleset.tick_us - 1)
                    // ruleset.tick_us
                    + 2
                )
            # Death/release fixtures include a legal fixed-deck Musketeer
            # counter.  High-HP bodies (Golem, Lava Hound, and carriers) need
            # more than the ordinary 30-second probe to cross their full
            # health pool, and Phoenix needs enough time for its egg stream to
            # be observed even when the tower wins first.
            branch_ticks = (
                1_800
                if mechanic
                in {
                    "death",
                    "death_effect",
                    "death_split",
                    "death_spawn",
                    "death_rage",
                    "death_transform",
                    "carrier_release",
                    "revive",
                    "revive_egg",
                }
                else 0
            )
            player_deck = (
                _rotated_fixed_player_deck("cannon")
                if card.kind == "troop"
                and bool(card.mechanics.get("building_only"))
                and mechanic in _TROOP_VICTIM_MECHANICS
                else PLAYER_DECK
            )
            opponent_candidates = [
                support_card,
                *(
                    candidate
                    for candidate in ruleset.interaction_set
                    if candidate not in {card_id, support_card}
                ),
            ]
            scenario_id = f"generated:{ruleset.ruleset_id}:{card_id}:{mechanic}:{variant}"
            scenario = Scenario(
                scenario_id=scenario_id,
                ruleset_id=ruleset.ruleset_id,
                ruleset_hash=ruleset.content_hash,
                engine_version=ENGINE_VERSION,
                seed=seed,
                decks=(player_deck, (card_id,) + tuple(opponent_candidates[:7])),
                actions=actions,
                # A lifetime case must run long enough to observe a 30-second
                # structure expiry.  Death/release cases use the longer branch
                # window above; other cases still receive 45 seconds so
                # delayed waves and projectiles have room to resolve.
                max_ticks=max(lifetime_ticks, branch_ticks, 900),
                shuffle_decks=False,
                split="synthetic",
                tags=("generated", "opponent", card.kind, mechanic),
                oracle={
                    "type": "property_and_invariant",
                    "mechanic": mechanic,
                    "card_id": card_id,
                    "source_confidence": "synthetic",
                    "required_card_plays": [
                        {"player": 1, "card_id": card_id},
                    ],
                    "required_support_card_plays": list(required_support),
                    "required_event_kinds": list(required_events),
                    "required_event_matches": list(required_event_matches),
                    "required_state_checks": required_state_checks,
                },
            )
            rows.append(GeneratedScenario(scenario, card_id, mechanic, seed))
    return tuple(rows)


def generate_roster_scenarios(
    ruleset: Ruleset,
    *,
    per_mechanic: int = 1,
    card_ids: Iterable[str] | None = None,
) -> tuple[GeneratedScenario, ...]:
    roster = load_opponent_roster()
    selected = tuple(card_ids) if card_ids is not None else roster.eligible_cards
    unknown = sorted(set(selected) - set(roster.eligible_cards))
    if unknown:
        raise ValueError(f"cards outside eligible roster: {unknown}")
    result: list[GeneratedScenario] = []
    for card_id in selected:
        result.extend(generate_card_scenarios(ruleset, card_id, per_mechanic=per_mechanic))
    return tuple(result)


def _rotated_fixed_player_deck(card_id: str) -> tuple[str, ...]:
    """Put one fixed Hog-cycle card in deterministic hand slot zero."""

    if card_id not in PLAYER_DECK:
        raise ValueError(f"interaction card is outside the fixed player deck: {card_id}")
    index = PLAYER_DECK.index(card_id)
    return PLAYER_DECK[index:] + PLAYER_DECK[:index]


def generate_interaction_scenarios(
    ruleset: Ruleset,
    *,
    opponent_card_ids: Iterable[str] | None = None,
    player_card_ids: Iterable[str] = PLAYER_DECK,
    variants: int = 1,
) -> tuple[GeneratedScenario, ...]:
    """Generate deterministic fixed-deck × opponent interaction probes.

    These are action-boundary scenarios rather than privileged state fixtures:
    both cards are in hand slot zero of valid eight-card decks, both plays are
    scheduled after the maximum V1 elixir cost is affordable, and every case
    records a required-play oracle.  Rotating the *same* fixed player deck
    exposes each of its eight cards without introducing an alternative deck.
    The matrix is intentionally synthetic; it proves instantiation, legal
    placement, target dispatch, projectile/effect creation, and determinism.
    Real-game truth remains the separate video-fidelity gate.
    """

    if type(variants) is not int or variants < 1:
        raise ValueError("variants must be positive")
    roster = load_opponent_roster()
    selected_opponents = (
        tuple(opponent_card_ids)
        if opponent_card_ids is not None
        else roster.eligible_cards
    )
    unknown = sorted(set(selected_opponents) - set(roster.eligible_cards))
    if unknown:
        raise ValueError(f"cards outside eligible roster: {unknown}")
    selected_players = tuple(player_card_ids)
    if not selected_players:
        raise ValueError("player_card_ids must not be empty")
    if len(set(selected_players)) != len(selected_players):
        raise ValueError("player_card_ids must be unique")
    for card_id in selected_players:
        if card_id not in PLAYER_DECK:
            raise ValueError(f"player card outside fixed deck: {card_id}")
        if card_id not in ruleset.cards:
            raise ValueError(f"player card outside ruleset: {card_id}")

    result: list[GeneratedScenario] = []
    for opponent_id in selected_opponents:
        opponent = ruleset.card(opponent_id)
        opponent_deck = (opponent_id,) + tuple(
            candidate
            for candidate in roster.eligible_cards
            if candidate != opponent_id
        )[:7]
        for player_id in selected_players:
            player = ruleset.card(player_id)
            player_deck = _rotated_fixed_player_deck(player_id)
            for variant in range(variants):
                # Keep the two cards on opposite legal territories for the
                # base probe.  A later variant can be added without changing
                # the identity of this first matrix row.
                player_cell = _test_cell(player.kind, 0)
                opponent_cell = _test_cell(opponent.kind, 1)
                player_tick = 400
                opponent_tick = 400
                # Instant spells should see the already-deployed opposing
                # body when possible; the placement remains legal for every
                # spell class because it uses that spell's own territory.
                if player.kind == "spell":
                    player_tick = 440
                if opponent.kind == "spell":
                    opponent_tick = 440
                actions = tuple(sorted(
                    (
                        ScheduledAction(player_tick, PlayCardAction(0, 0, player_cell)),
                        ScheduledAction(opponent_tick, PlayCardAction(1, 0, opponent_cell)),
                    ),
                    key=lambda scheduled: (scheduled.tick, scheduled.action.player),
                ))
                actions = _variant_actions(actions, variant)
                scenario_id = (
                    f"interaction:{ruleset.ruleset_id}:{player_id}:{opponent_id}:{variant}"
                )
                scenario = Scenario(
                    scenario_id=scenario_id,
                    ruleset_id=ruleset.ruleset_id,
                    ruleset_hash=ruleset.content_hash,
                    engine_version=ENGINE_VERSION,
                    seed=_stable_seed(f"{player_id}:{opponent_id}", "interaction", variant),
                    decks=(player_deck, opponent_deck),
                    actions=actions,
                    # The pair is injected at 20–22 seconds.  Ten additional
                    # seconds is enough to exercise deploy, first movement,
                    # projectile/effect creation, and early collision without
                    # letting a synthetic matrix spend minutes in a crowded
                    # all-card soak trace.
                    max_ticks=600,
                    shuffle_decks=False,
                    split="synthetic",
                    tags=("generated", "interaction", "fixed-player-deck", opponent.kind),
                    oracle={
                        "type": "pairwise_action_boundary",
                        "mechanic": "pairwise_interaction",
                        "card_id": opponent_id,
                        "player_card_id": player_id,
                        "opponent_card_id": opponent_id,
                        "source_confidence": "synthetic",
                        "required_card_plays": [
                            {"player": 0, "card_id": player_id},
                            {"player": 1, "card_id": opponent_id},
                        ],
                    },
                )
                result.append(GeneratedScenario(scenario, opponent_id, "pairwise_interaction", scenario.seed))
    return tuple(result)


def generate_opponent_pair_scenarios(
    ruleset: Ruleset,
    *,
    opponent_card_ids: Iterable[str] | None = None,
    variants: int = 1,
) -> tuple[GeneratedScenario, ...]:
    """Generate unordered two-opponent-card interaction probes.

    The fixed player remains the classic Hog-cycle deck.  Each scenario puts
    two distinct eligible opponent cards in hand slots zero and schedules both
    plays against a Hog.  This is deliberately a synthetic exercisability
    matrix: it proves that every pair can coexist in one legal match and that
    both card paths are accepted, while real-video reports remain the authority
    for timing, geometry, and strategic outcomes.

    Pair rows are unordered to avoid duplicating ``A+B`` as ``B+A``.  The
    second play waits long enough for the first card's elixir cost to
    regenerate, which means even the most expensive eligible card cannot be
    silently rejected by the action boundary.
    """

    if type(variants) is not int or variants < 1:
        raise ValueError("variants must be positive")
    roster = load_opponent_roster()
    selected = (
        tuple(opponent_card_ids)
        if opponent_card_ids is not None
        else roster.eligible_cards
    )
    if len(selected) < 2:
        raise ValueError("at least two opponent cards are required")
    if len(set(selected)) != len(selected):
        raise ValueError("opponent_card_ids must be unique")
    unknown = sorted(set(selected) - set(roster.eligible_cards))
    if unknown:
        raise ValueError(f"cards outside eligible roster: {unknown}")

    result: list[GeneratedScenario] = []
    normal_interval_ticks = max(
        1,
        (ruleset.match.normal_elixir_interval_us + ruleset.tick_us - 1)
        // ruleset.tick_us,
    )
    for first_index, first_id in enumerate(selected[:-1]):
        first_card = ruleset.card(first_id)
        for second_id in selected[first_index + 1 :]:
            second_card = ruleset.card(second_id)
            opponent_deck = (first_id, second_id) + tuple(
                candidate
                for candidate in roster.eligible_cards
                if candidate not in {first_id, second_id}
            )[:6]
            for variant in range(variants):
                delta = _variant_pattern(variant)[1]
                # Reach the ten-elixir cap before the first opponent play.
                # This is the shortest common setup that keeps every eligible
                # card (including Three Musketeers) legal while avoiding a
                # hidden cost-dependent rejection in the pair matrix.
                setup_ticks = (
                    (ruleset.match.max_elixir_milli - ruleset.match.initial_elixir_milli)
                    // 1_000
                ) * normal_interval_ticks
                hog_tick = setup_ticks + delta
                first_tick = hog_tick
                # Start at the maximum normal elixir after the long setup;
                # wait for the exact cost to regenerate before slot zero is
                # used a second time.  The one-tick margin avoids a rounding
                # boundary when an interval is not divisible by tick_us.
                second_tick = (
                    first_tick
                    + int(first_card.elixir_milli // 1_000) * normal_interval_ticks
                    + 1
                )
                actions = (
                    ScheduledAction(
                        hog_tick,
                        PlayCardAction(0, 0, _variant_cell(_test_cell("troop", 0), variant)),
                    ),
                    ScheduledAction(
                        first_tick,
                        PlayCardAction(
                            1,
                            0,
                            _pair_test_cell(first_card.kind, 1, 0, variant),
                        ),
                    ),
                    ScheduledAction(
                        second_tick,
                        PlayCardAction(
                            1,
                            0,
                            _pair_test_cell(second_card.kind, 1, 1, variant),
                        ),
                    ),
                )
                actions = tuple(sorted(actions, key=lambda row: (row.tick, row.action.player)))
                scenario_id = (
                    f"opponent-pair:{ruleset.ruleset_id}:{first_id}:{second_id}:{variant}"
                )
                scenario = Scenario(
                    scenario_id=scenario_id,
                    ruleset_id=ruleset.ruleset_id,
                    ruleset_hash=ruleset.content_hash,
                    engine_version=ENGINE_VERSION,
                    seed=_stable_seed(
                        f"{first_id}:{second_id}",
                        "opponent_pair_interaction",
                        variant,
                    ),
                    decks=(PLAYER_DECK, opponent_deck),
                    actions=actions,
                    # The pair oracle only requires both action-boundary
                    # plays.  Stop immediately after the second card's tick;
                    # long combat traces belong to card-mechanic scenarios or
                    # real-video validation and would make this exhaustive
                    # coexistence matrix needlessly quadratic in runtime.
                    max_ticks=second_tick + 1,
                    shuffle_decks=False,
                    split="synthetic",
                    tags=(
                        "generated",
                        "opponent-pair",
                        "fixed-player-deck",
                        first_card.kind,
                        second_card.kind,
                    ),
                    oracle={
                        "type": "opponent_pair_action_boundary",
                        "mechanic": "opponent_pair_interaction",
                        "card_id": first_id,
                        "first_opponent_card_id": first_id,
                        "second_opponent_card_id": second_id,
                        "source_confidence": "synthetic",
                        "required_card_plays": [
                            {"player": 0, "card_id": "hog-rider"},
                            {"player": 1, "card_id": first_id},
                            {"player": 1, "card_id": second_id},
                        ],
                    },
                )
                result.append(
                    GeneratedScenario(
                        scenario,
                        first_id,
                        "opponent_pair_interaction",
                        scenario.seed,
                    )
                )
    return tuple(result)


def generated_manifest(rows: Iterable[GeneratedScenario]) -> dict[str, Any]:
    cases = [row.scenario.to_dict() for row in rows]
    cases.sort(key=lambda row: row["scenario_id"])
    card_ids: set[str] = set()
    pair_keys: set[tuple[str, str]] = set()
    for row in cases:
        oracle = row["oracle"]
        for field in ("card_id", "first_opponent_card_id", "second_opponent_card_id"):
            value = oracle.get(field)
            if isinstance(value, str) and value:
                card_ids.add(value)
        first = oracle.get("first_opponent_card_id")
        second = oracle.get("second_opponent_card_id")
        if isinstance(first, str) and isinstance(second, str) and first != second:
            pair_keys.add(tuple(sorted((first, second))))
    summary: dict[str, Any] = {
        "scenario_count": len(cases),
        "card_count": len(card_ids),
        "mechanic_count": len({str(row["oracle"]["mechanic"]) for row in cases}),
    }
    if pair_keys:
        summary["unordered_pair_count"] = len(pair_keys)
    return {
        "schema_version": SCENARIO_FACTORY_SCHEMA_VERSION,
        "kind": "simulator_generated_scenario_manifest",
        "cases": cases,
        "summary": summary,
    }


def write_generated_manifest(path: str | Path, rows: Iterable[GeneratedScenario]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = generated_manifest(rows)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


__all__ = [
    "GeneratedScenario",
    "SCENARIO_FACTORY_SCHEMA_VERSION",
    "card_mechanics",
    "generate_card_scenarios",
    "generate_interaction_scenarios",
    "generate_opponent_pair_scenarios",
    "generate_roster_scenarios",
    "generated_manifest",
    "write_generated_manifest",
]
