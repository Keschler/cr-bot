"""Authoritative, integer-only battle state for the simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any

from .events import SimEvent


def _required_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return value


def _required_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be boolean")
    return value


def _required_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


@dataclass(slots=True)
class StatusState:
    kind: str
    remaining_us: int
    magnitude_permille: int = 1_000
    damage_per_tick: int = 0
    tick_interval_us: int = 0
    tick_remainder_us: int = 0
    # Some effects change the identity of a troop when it dies while the
    # status is active (Goblin Curse).  These fields are intentionally
    # scalar so the event and replay state stay canonical JSON.
    on_death_spawn_card_id: str | None = None
    on_death_spawn_count: int = 0
    on_death_spawn_owner: int | None = None
    # ``None`` preserves the legacy constructor meaning (same multiplier for
    # movement and attack speed).  Goblin Curse uses 850 for movement and
    # 1000 for attack speed.
    hit_speed_magnitude_permille: int | None = None
    source_level_multiplier_permille: int = 1_000


@dataclass(slots=True)
class EntityState:
    uid: int
    card_id: str
    owner: int
    kind: str
    x_mtile: int
    y_mtile: int
    hp: int
    max_hp: int
    spawn_tick: int
    # Per-entity level scaling is used by Mirror.  Normal Level-11 bodies use
    # 1000; a Level-12 mirrored body uses the game's 10% level step (1100).
    level_multiplier_permille: int = 1_000
    role: str | None = None
    target_uid: int | None = None
    pending_target_uid: int | None = None
    deploy_remaining_us: int = 0
    attack_cooldown_us: int = 0
    # Initial attack loading is independent from the repeat-attack cooldown.
    # It is advanced after deployment even when no target is currently in
    # range, allowing a body to retain a preloaded first attack in state.
    attack_load_remaining_us: int = 0
    windup_remaining_us: int = 0
    # Optional independent attack channel for multi-weapon troops such as
    # Goblin Machine.  Keeping the rocket clock separate from the melee clock
    # allows both attacks to progress and resolve in the same simulation tick.
    secondary_attack_cooldown_us: int = 0
    secondary_windup_remaining_us: int = 0
    secondary_pending_target_uid: int | None = None
    secondary_attack_time_remainder: int = 0
    secondary_attack_count: int = 0
    lifetime_remaining_us: int | None = None
    lifetime_decay_remainder: int = 0
    spawn_cooldown_us: int = 0
    spawn_time_remainder: int = 0
    spawned_count: int = 0
    # Proximity-triggered spawners serialize their current visible-enemy gate
    # so a paused cadence resumes identically after replay restoration.
    spawner_active: bool = False
    movement_remainder: int = 0
    attack_time_remainder: int = 0
    navigation_target_uid: int | None = None
    navigation_revision: int = -1
    navigation_goal_x_mtile: int = 0
    navigation_goal_y_mtile: int = 0
    navigation_cursor: int = 0
    navigation_waypoints: list[tuple[int, int]] = field(default_factory=list)
    statuses: list[StatusState] = field(default_factory=list)
    alive: bool = True
    death_effect_done: bool = False
    attack_count: int = 0
    # Optional phase state for cards whose behavior changes at a health
    # threshold (for example Goblin Demolisher).  ``charge_active`` is
    # latched: healing above the threshold does not undo an already-triggered
    # phase in the live game.
    charge_active: bool = False
    charge_remaining_us: int | None = None
    # Clone is represented as a normal card body with one HP, but retaining
    # this flag keeps clone provenance observable and prevents downstream
    # truth-mining from confusing a copied body with the original entity.
    is_clone: bool = False
    # Generic movement-charge state is separate from ``charge_active`` above,
    # which is reserved for health-threshold phases such as Goblin
    # Demolisher.  Keeping the two components independent makes serialized
    # replays unambiguous.
    attack_charge_active: bool = False
    attack_charge_distance_mtile: int = 0
    # Bandit's dash is a distinct one-impact phase; it must not share the
    # movement-charge fields because a dash can arm without a long run.
    dash_attack_active: bool = False
    # A dash remains an explicit short-lived phase after the body reaches its
    # landing point.  Damage, hard crowd control, and knockback are ignored
    # while this clock is positive; the dash attack resolves only after the
    # phase has ended.
    dash_remaining_us: int = 0
    # Electro Giant reflection is once per concrete attack instance.  Keep
    # the last source/attack pair in authoritative state so multi-projectile
    # attacks cannot reflect independently while remaining replayable.
    last_reflection_source_uid: int | None = None
    last_reflection_attack_instance_id: int | None = None
    # Inferno attacks ramp their damage while a target remains locked.  The
    # elapsed time and stage are authoritative so a replay can be resumed
    # without reconstructing hidden beam state from events.
    ramp_elapsed_us: int = 0
    ramp_stage: int = 0
    # Phoenix bodies may leave one egg; reborn bodies and cloned bodies have
    # this eligibility cleared so the lifecycle cannot recurse indefinitely.
    revive_eligible: bool = True
    # An egg only hatches when its lifetime expires.  Damage sets HP to zero
    # without setting this flag, allowing the normal death path to destroy it.
    hatch_due: bool = False
    # Transported child bodies (currently Goblin Giant's two Spear Goblins)
    # remain first-class entities so they can attack while attached.  The
    # carrier relation is authoritative and serializable; a ``None`` value
    # means the body has been released and follows ordinary navigation.
    carried_by_uid: int | None = None
    carried_offset_x_mtile: int = 0
    carried_offset_y_mtile: int = 0
    # Shield is a separate damage pool.  Clash Royale shield damage is
    # consumed by the shield (excess does not spill into body HP); keeping
    # both values in authoritative state makes shield breaks replayable and
    # lets vision validation compare the two-layer lifecycle.
    shield_hp: int = 0
    shield_max_hp: int = 0
    # Royal Ghost starts hidden, becomes visible when it attacks, then
    # re-enters stealth after the authored re-cloak delay.  A boolean plus a
    # countdown avoids encoding lifecycle state in a lossy status list.
    stealth_active: bool = False
    stealth_remaining_us: int = 0
    # Burrowed/tunnel cards (Miner) are present at their destination but do
    # not move, acquire targets, or receive ordinary target selection until
    # the tunnel phase ends.
    burrow_active: bool = False
    # Tesla's underground state differs from stealth: ordinary spells and
    # troops cannot affect it, while Earthquake and Freeze are exceptions.
    concealed_active: bool = False
    river_airborne_active: bool = False
    # Mega Knight-style jump/landing attacks use an explicit in-flight phase
    # so a jump cannot be mistaken for ordinary path movement.
    jump_remaining_us: int = 0
    jump_target_uid: int | None = None
    jump_landing_x_mtile: int = 0
    jump_landing_y_mtile: int = 0
    # Parent provenance is authoritative for capped spawners.  Counting by
    # owner/card alone incorrectly lets one Furnace (or similar building)
    # consume another parent's cap.
    parent_uid: int | None = None


@dataclass(slots=True)
class ProjectileState:
    uid: int
    source_uid: int | None
    source_card_id: str
    owner: int
    x_mtile: int
    y_mtile: int
    target_x_mtile: int
    target_y_mtile: int
    damage: int
    crown_damage: int
    speed_mtile_per_s: int
    # Some spells have an authored fall/deploy delay that is independent of
    # their normalized projectile travel speed (currently Royal Delivery).
    # Keeping the remaining delay in authoritative state makes the impact
    # frame deterministic across snapshots and replays.
    impact_delay_remaining_us: int = 0
    # Preserve a versioned ruleset projectile speed code alongside the
    # normalized physical speed when a card supplies one.
    speed_code: int | None = None
    homing: bool = False
    radius_mtile: int = 0
    target_uid: int | None = None
    status_kind: str | None = None
    status_duration_us: int = 0
    status_magnitude_permille: int = 1_000
    status_hit_speed_magnitude_permille: int = 1_000
    status_damage_per_tick: int = 0
    status_tick_interval_us: int = 0
    knockback_mtile: int = 0
    piercing: bool = False
    # Some cards have a movement/primary target class different from the
    # projectile's impact class (Goblin Machine melee targets ground while
    # its rocket targets air and ground).  An empty tuple preserves the
    # legacy card-definition lookup.
    allowed_targets: tuple[str, ...] = ()
    hit_uids: list[int] = field(default_factory=list)
    alive: bool = True
    movement_remainder: int = 0
    level_multiplier_permille: int = 1_000
    # Fixed line geometry for piercing projectiles (Magic Archer).  The
    # legacy target coordinates remain the endpoint for ordinary projectiles.
    origin_x_mtile: int = 0
    origin_y_mtile: int = 0
    line_end_x_mtile: int = 0
    line_end_y_mtile: int = 0
    direction_x_mtile: int = 0
    direction_y_mtile: int = 0
    # Executioner's axe returns to its source and can hit a second time.
    returning: bool = False
    return_phase: bool = False
    # Hunter's fan is represented by independent pellet projectiles; indices
    # make event streams and generated truth unambiguous.
    pellet_index: int = 0
    # Electro Spirit's later bounces are authoritative delayed impacts.
    chain_target_uids: list[int] = field(default_factory=list)
    chain_next_index: int = 0
    chain_delay_us: int = 0
    chain_delay_remaining_us: int = 0
    # Start of the most recent movement segment.  Piercing collisions must be
    # evaluated on that segment, not repeatedly from the launch point.
    previous_x_mtile: int | None = None
    previous_y_mtile: int | None = None
    # All pellets emitted by one attacker shot share this identifier.  It is
    # intentionally optional for old serialized projectiles and single-hit
    # fixtures that do not need attack-level aggregation.
    attack_instance_id: int | None = None


@dataclass(slots=True)
class AreaEffectState:
    """A deterministic persistent effect anchored to an arena position.

    Effects are deliberately data-only.  The engine applies their damage,
    status, displacement, and optional spawn component in UID order.  Keeping
    the component in authoritative state makes persistent spells replayable
    and lets the same implementation serve Poison, Earthquake, Graveyard,
    and future area mechanics.
    """

    uid: int
    source_uid: int | None
    source_card_id: str
    owner: int
    x_mtile: int
    y_mtile: int
    radius_mtile: int
    remaining_us: int
    tick_interval_us: int
    tick_remainder_us: int = 0
    initial_delay_remaining_us: int = 0
    damage_per_tick: int = 0
    crown_damage_per_tick: int = 0
    status_kind: str | None = None
    status_duration_us: int = 0
    status_magnitude_permille: int = 1_000
    status_hit_speed_magnitude_permille: int = 1_000
    status_damage_per_tick: int = 0
    status_tick_interval_us: int = 0
    knockback_mtile: int = 0
    pull_to_center_mtile: int = 0
    allowed_targets: tuple[str, ...] = ()
    spawn_card_id: str | None = None
    spawn_count: int = 0
    max_spawns: int = 0
    spawned_count: int = 0
    pulses_applied: int = 0
    max_pulses: int | None = None
    alive: bool = True
    level_multiplier_permille: int = 1_000
    # Optional per-pulse damage schedules.  Tuples are canonical in the
    # authoritative state; JSON round-trips normalize lists back to tuples.
    # A schedule may end before the area lifetime (for example Tornado keeps
    # its pull active during a short final tail after its one damage pulse).
    damage_schedule: tuple[int, ...] = ()
    crown_damage_schedule: tuple[int, ...] = ()
    # Optional friendly aura component (Rage is the first consumer).  The
    # normal ``status_*`` fields above apply to enemy victims; these fields
    # let a persistent effect damage enemies while refreshing a buff on
    # friendly troops/buildings in the same radius.
    friendly_status_kind: str | None = None
    friendly_status_duration_us: int = 0
    friendly_status_magnitude_permille: int = 1_000
    friendly_status_linger_us: int = 0
    friendly_allowed_targets: tuple[str, ...] = ()
    # Optional status-triggered child stream (currently Goblin Curse).
    status_on_death_spawn_card_id: str | None = None
    status_on_death_spawn_count: int = 0


@dataclass(slots=True)
class PlayerState:
    deck: tuple[str, ...]
    hand: list[str]
    draw_pile: list[str]
    elixir_milli: int
    elixir_remainder: int = 0
    crowns: int = 0
    king_active: bool = False
    cards_played: int = 0
    seen_enemy_cards: list[str] = field(default_factory=list)
    last_played_card_id: str | None = None
    # The visible Next card can be temporarily held out of the playable hand
    # after rapid plays.  Zero means the card is ready to replace the next
    # played card; a positive value is the remaining hand-loading cooldown.
    next_card_cooldown_us: int = 0


class EventHistory(list[SimEvent]):
    """Append-only event transport with an O(1) mutation revision.

    The authoritative engine only appends events, but restore and validation
    tooling can replace or edit a retained history in place.  Tracking those
    edits lets consumers detect a rewrite without hashing the complete log on
    every observation or persistent-worker synchronization.
    """

    __slots__ = ("mutation_revision",)

    def __init__(self, values: object = ()) -> None:
        super().__init__(values)  # type: ignore[arg-type]
        self.mutation_revision = 0

    def append(self, value: SimEvent) -> None:
        list.append(self, value)
        self.mutation_revision += 1

    def extend(self, values: object) -> None:
        additions = list(values)  # type: ignore[arg-type]
        if not additions:
            return
        list.extend(self, additions)
        self.mutation_revision += 1

    def insert(self, index: int, value: SimEvent) -> None:
        list.insert(self, index, value)
        self.mutation_revision += 1

    def __setitem__(self, key: int | slice, value: object) -> None:
        if isinstance(key, slice):
            value = list(value)  # type: ignore[arg-type]
        list.__setitem__(self, key, value)  # type: ignore[index]
        self.mutation_revision += 1

    def __delitem__(self, key: int | slice) -> None:
        list.__delitem__(self, key)
        self.mutation_revision += 1

    def clear(self) -> None:
        if not self:
            return
        list.clear(self)
        self.mutation_revision += 1

    def pop(self, index: int = -1) -> SimEvent:
        value = list.pop(self, index)
        self.mutation_revision += 1
        return value

    def remove(self, value: SimEvent) -> None:
        list.remove(self, value)
        self.mutation_revision += 1

    def reverse(self) -> None:
        list.reverse(self)
        self.mutation_revision += 1

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        list.sort(self, key=key, reverse=reverse)
        self.mutation_revision += 1

    def __iadd__(self, values: object) -> "EventHistory":
        self.extend(values)
        return self

    def __imul__(self, multiplier: int) -> "EventHistory":
        list.__imul__(self, multiplier)
        self.mutation_revision += 1
        return self


@dataclass(slots=True)
class BattleState:
    schema_version: int
    engine_version: str
    ruleset_id: str
    ruleset_hash: str
    seed: int
    rng_state: int
    tick: int
    elapsed_us: int
    phase: str
    players: list[PlayerState]
    entities: dict[int, EntityState]
    projectiles: dict[int, ProjectileState]
    next_uid: int
    navigation_revision: int = 0
    winner: int | None = None
    terminal: bool = False
    terminal_reason: str | None = None
    event_sequence: int = 0
    events: EventHistory = field(default_factory=EventHistory)
    # Appended after the original default fields to preserve positional
    # construction compatibility for serialized/research fixtures.
    effects: dict[int, AreaEffectState] = field(default_factory=dict)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "events" and not isinstance(value, EventHistory):
            value = EventHistory(value)
        object.__setattr__(self, name, value)

    def to_primitive(self, *, include_events: bool = False) -> dict[str, Any]:
        entities = [asdict(self.entities[uid]) for uid in sorted(self.entities)]
        projectiles = [asdict(self.projectiles[uid]) for uid in sorted(self.projectiles)]
        effects = [asdict(self.effects[uid]) for uid in sorted(self.effects)]
        raw: dict[str, Any] = {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "ruleset_id": self.ruleset_id,
            "ruleset_hash": self.ruleset_hash,
            "seed": self.seed,
            "rng_state": self.rng_state,
            "tick": self.tick,
            "elapsed_us": self.elapsed_us,
            "phase": self.phase,
            "players": [asdict(player) for player in self.players],
            "entities": entities,
            "projectiles": projectiles,
            "effects": effects,
            "next_uid": self.next_uid,
            "navigation_revision": self.navigation_revision,
            "winner": self.winner,
            "terminal": self.terminal,
            "terminal_reason": self.terminal_reason,
            "event_sequence": self.event_sequence,
        }
        if include_events:
            raw["events"] = [event.to_dict() for event in self.events]
        return raw

    def canonical_json(self, *, include_events: bool = False) -> str:
        return json.dumps(
            self.to_primitive(include_events=include_events),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def state_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    def event_log_hash(self) -> str:
        encoded = json.dumps(
            [event.to_dict() for event in self.events],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def replay_hash(self) -> str:
        return hashlib.sha256(
            self.canonical_json(include_events=True).encode("ascii")
        ).hexdigest()


def battle_state_from_primitive(raw: dict[str, Any]) -> BattleState:
    players = [
        PlayerState(
            deck=tuple(row["deck"]),
            hand=list(row["hand"]),
            draw_pile=list(row["draw_pile"]),
            elixir_milli=_required_int(row["elixir_milli"], "player.elixir_milli"),
            elixir_remainder=_required_int(
                row.get("elixir_remainder", 0), "player.elixir_remainder"
            ),
            crowns=_required_int(row.get("crowns", 0), "player.crowns"),
            king_active=_required_bool(row.get("king_active", False), "player.king_active"),
            cards_played=_required_int(
                row.get("cards_played", 0), "player.cards_played"
            ),
            seen_enemy_cards=list(row.get("seen_enemy_cards", [])),
            last_played_card_id=row.get("last_played_card_id"),
            next_card_cooldown_us=_required_int(
                row.get("next_card_cooldown_us", 0),
                "player.next_card_cooldown_us",
            ),
        )
        for row in raw["players"]
    ]
    entities: dict[int, EntityState] = {}
    for row in raw["entities"]:
        entity_row = dict(row)
        entity_row.setdefault("lifetime_decay_remainder", 0)
        entity_row.setdefault("spawn_cooldown_us", 0)
        entity_row.setdefault("spawn_time_remainder", 0)
        entity_row.setdefault("spawned_count", 0)
        entity_row.setdefault("spawner_active", False)
        entity_row.setdefault("navigation_target_uid", None)
        entity_row.setdefault("navigation_revision", -1)
        entity_row.setdefault("navigation_goal_x_mtile", 0)
        entity_row.setdefault("navigation_goal_y_mtile", 0)
        entity_row.setdefault("navigation_cursor", 0)
        entity_row.setdefault("charge_active", False)
        entity_row.setdefault("charge_remaining_us", None)
        entity_row.setdefault("attack_load_remaining_us", 0)
        entity_row.setdefault("is_clone", False)
        entity_row.setdefault("attack_charge_active", False)
        entity_row.setdefault("attack_charge_distance_mtile", 0)
        entity_row.setdefault("dash_attack_active", False)
        entity_row.setdefault("dash_remaining_us", 0)
        entity_row.setdefault("last_reflection_source_uid", None)
        entity_row.setdefault("last_reflection_attack_instance_id", None)
        entity_row.setdefault("ramp_elapsed_us", 0)
        entity_row.setdefault("ramp_stage", 0)
        entity_row.setdefault("revive_eligible", True)
        entity_row.setdefault("hatch_due", False)
        entity_row.setdefault("carried_by_uid", None)
        entity_row.setdefault("carried_offset_x_mtile", 0)
        entity_row.setdefault("carried_offset_y_mtile", 0)
        entity_row.setdefault("shield_hp", 0)
        entity_row.setdefault("shield_max_hp", 0)
        entity_row.setdefault("stealth_active", False)
        entity_row.setdefault("stealth_remaining_us", 0)
        entity_row.setdefault("burrow_active", False)
        entity_row.setdefault("concealed_active", False)
        entity_row.setdefault("river_airborne_active", False)
        entity_row.setdefault("jump_remaining_us", 0)
        entity_row.setdefault("jump_target_uid", None)
        entity_row.setdefault("jump_landing_x_mtile", 0)
        entity_row.setdefault("jump_landing_y_mtile", 0)
        entity_row.setdefault("parent_uid", None)
        entity_row.setdefault("secondary_attack_cooldown_us", 0)
        entity_row.setdefault("secondary_windup_remaining_us", 0)
        entity_row.setdefault("secondary_pending_target_uid", None)
        entity_row.setdefault("secondary_attack_time_remainder", 0)
        entity_row.setdefault("secondary_attack_count", 0)
        entity_row.setdefault("level_multiplier_permille", 1_000)
        entity_row["navigation_waypoints"] = [
            tuple(point) for point in entity_row.get("navigation_waypoints", [])
        ]
        statuses = []
        for status in row.get("statuses", []):
            status_row = dict(status)
            status_row.setdefault("source_level_multiplier_permille", 1_000)
            statuses.append(StatusState(**status_row))
        entity_row["statuses"] = statuses
        entity = EntityState(**entity_row)
        if type(entity.uid) is not int:
            raise ValueError("entity.uid must be an integer")
        if entity.uid in entities:
            raise ValueError(f"duplicate entity UID: {entity.uid}")
        entities[entity.uid] = entity
    projectiles: dict[int, ProjectileState] = {}
    for row in raw["projectiles"]:
        projectile_row = dict(row)
        projectile_row.setdefault("speed_code", None)
        projectile_row.setdefault("impact_delay_remaining_us", 0)
        projectile_row.setdefault("homing", False)
        projectile_row.setdefault("status_hit_speed_magnitude_permille", 1_000)
        projectile_row.setdefault("level_multiplier_permille", 1_000)
        projectile_row.setdefault("chain_target_uids", [])
        projectile_row.setdefault("chain_next_index", 0)
        projectile_row.setdefault("chain_delay_us", 0)
        projectile_row.setdefault("chain_delay_remaining_us", 0)
        projectile_row.setdefault("previous_x_mtile", None)
        projectile_row.setdefault("previous_y_mtile", None)
        projectile_row.setdefault("attack_instance_id", None)
        projectile_row.setdefault("allowed_targets", ())
        projectile_row.setdefault("origin_x_mtile", projectile_row.get("x_mtile", 0))
        projectile_row.setdefault("origin_y_mtile", projectile_row.get("y_mtile", 0))
        projectile_row.setdefault("line_end_x_mtile", projectile_row.get("target_x_mtile", 0))
        projectile_row.setdefault("line_end_y_mtile", projectile_row.get("target_y_mtile", 0))
        projectile_row.setdefault("direction_x_mtile", 0)
        projectile_row.setdefault("direction_y_mtile", 0)
        projectile_row.setdefault("returning", False)
        projectile_row.setdefault("return_phase", False)
        projectile_row.setdefault("pellet_index", 0)
        projectile_row["allowed_targets"] = tuple(projectile_row["allowed_targets"])
        projectile_row["chain_target_uids"] = list(projectile_row["chain_target_uids"])
        projectile = ProjectileState(**projectile_row)
        if type(projectile.uid) is not int:
            raise ValueError("projectile.uid must be an integer")
        if projectile.uid in projectiles:
            raise ValueError(f"duplicate projectile UID: {projectile.uid}")
        projectiles[projectile.uid] = projectile
    effects: dict[int, AreaEffectState] = {}
    for row in raw.get("effects", []):
        effect_row = dict(row)
        effect_row["allowed_targets"] = tuple(effect_row.get("allowed_targets", ()))
        effect_row["damage_schedule"] = tuple(effect_row.get("damage_schedule", ()))
        effect_row["crown_damage_schedule"] = tuple(
            effect_row.get("crown_damage_schedule", ())
        )
        effect_row["friendly_allowed_targets"] = tuple(
            effect_row.get("friendly_allowed_targets", ())
        )
        effect_row.setdefault("pulses_applied", 0)
        effect_row.setdefault("initial_delay_remaining_us", 0)
        effect_row.setdefault("max_pulses", None)
        effect_row.setdefault("friendly_status_kind", None)
        effect_row.setdefault("friendly_status_duration_us", 0)
        effect_row.setdefault("friendly_status_magnitude_permille", 1_000)
        effect_row.setdefault("friendly_status_linger_us", 0)
        effect_row.setdefault("level_multiplier_permille", 1_000)
        effect = AreaEffectState(**effect_row)
        if type(effect.uid) is not int:
            raise ValueError("effect.uid must be an integer")
        if effect.uid in effects:
            raise ValueError(f"duplicate effect UID: {effect.uid}")
        effects[effect.uid] = effect
    events = []
    for row in raw.get("events", []):
        events.append(
            SimEvent(
                _required_int(row["tick"], "event.tick"),
                _required_int(row["sequence"], "event.sequence"),
                _required_str(row["kind"], "event.kind"),
                tuple(sorted(dict(row.get("data", {})).items())),
            )
        )
    return BattleState(
        schema_version=_required_int(raw["schema_version"], "schema_version"),
        engine_version=_required_str(raw["engine_version"], "engine_version"),
        ruleset_id=_required_str(raw["ruleset_id"], "ruleset_id"),
        ruleset_hash=_required_str(raw["ruleset_hash"], "ruleset_hash"),
        seed=_required_int(raw["seed"], "seed"),
        rng_state=_required_int(raw["rng_state"], "rng_state"),
        tick=_required_int(raw["tick"], "tick"),
        elapsed_us=_required_int(raw["elapsed_us"], "elapsed_us"),
        phase=_required_str(raw["phase"], "phase"),
        players=players,
        entities=entities,
        projectiles=projectiles,
        next_uid=_required_int(raw["next_uid"], "next_uid"),
        effects=effects,
        navigation_revision=_required_int(
            raw.get("navigation_revision", 0), "navigation_revision"
        ),
        winner=raw.get("winner"),
        terminal=_required_bool(raw.get("terminal", False), "terminal"),
        terminal_reason=raw.get("terminal_reason"),
        event_sequence=_required_int(raw.get("event_sequence", 0), "event_sequence"),
        events=events,
    )
