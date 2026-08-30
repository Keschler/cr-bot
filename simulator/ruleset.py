"""Pinned, immutable simulator rulesets with field-level provenance.

The JSON files in :mod:`simulator.rulesets` are runtime inputs, not a live
cache.  Loading a ruleset never accesses the network.  A ruleset's SHA-256
digest covers its canonical JSON payload (excluding the ``content_hash``
field), making simulations and saved scenarios reproducible.

All authoritative numeric values use integers.  Durations are microseconds,
positions and distances are milli-tiles, and elixir is milli-elixir.  Values
which have not been established are represented by ``None`` and accompanied
by an :class:`Uncertainty`; the loader never manufactures a default game
fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
import unicodedata


RULESET_SCHEMA_VERSION = 1
DEFAULT_RULESET_ID = "2026-08-04"
FIXED_RULESET_ID = "v1"
# V1 is intentionally a fixed, non-timestamped artifact.  Date IDs remain
# accepted for provenance/compatibility while later releases can reintroduce
# timestamped balance snapshots without changing the V1 runtime contract.
_RULESET_ID_RE = re.compile(r"(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|v[0-9]+)(?:-[a-z0-9-]+)?\Z")
_CONTENT_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CARD_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_CARD_KINDS = frozenset({"troop", "building", "spell"})
_TARGETS = frozenset({"air", "ground", "building", "crown_tower"})


class RulesetError(ValueError):
    """Raised when a pinned ruleset is malformed or has been modified."""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    confidence_tier: str
    kind: str
    url: str | None
    retrieved_at: str
    published_at: str | None = None
    sha256: str | None = None
    lineage: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class Uncertainty:
    field: str
    reason: str
    impact: str
    resolution: str


@dataclass(frozen=True, slots=True)
class ProjectileDefinition:
    projectile_id: str
    speed_mtile_per_s: int
    radius_mtile: int
    start_radius_mtile: int
    homing: bool


@dataclass(frozen=True, slots=True)
class CardDefinition:
    card_id: str
    name: str
    aliases: tuple[str, ...]
    kind: str
    elixir_milli: int
    deploy_time_us: int
    spawn_count: int
    hitpoints: int | None
    damage: int | None
    crown_tower_damage: int | None
    attack_interval_us: int | None
    first_hit_delay_us: int | None
    move_speed_mtile_per_s: int | None
    range_mtile: int | None
    sight_range_mtile: int | None
    collision_radius_mtile: int | None
    mass: int | None
    lifetime_us: int | None
    targets: tuple[str, ...]
    projectile: ProjectileDefinition | None
    area_radius_mtile: int | None
    mechanics: Mapping[str, Any]
    provenance: Mapping[str, tuple[str, ...]]
    uncertainties: tuple[Uncertainty, ...]


@dataclass(frozen=True, slots=True)
class TowerDefinition:
    tower_id: str
    name: str
    aliases: tuple[str, ...]
    hitpoints: int
    damage: int
    attack_interval_us: int
    first_hit_delay_us: int
    range_mtile: int
    sight_range_mtile: int
    collision_radius_mtile: int
    targets: tuple[str, ...]
    projectile: ProjectileDefinition
    activated_at_start: bool
    crown_value: int
    provenance: Mapping[str, tuple[str, ...]]
    uncertainties: tuple[Uncertainty, ...]


@dataclass(frozen=True, slots=True)
class MatchRules:
    initial_elixir_milli: int
    max_elixir_milli: int
    deck_size: int
    hand_size: int
    regulation_us: int
    overtime_us: int
    normal_elixir_interval_us: int
    double_elixir_interval_us: int
    triple_elixir_interval_us: int
    tiebreak_enabled: bool


@dataclass(frozen=True, slots=True)
class ArenaRules:
    width_mtile: int
    height_mtile: int
    grid_columns: int
    grid_rows: int
    river_y_min_mtile: int
    river_y_max_mtile: int
    bridge_x_ranges_mtile: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class Ruleset:
    schema_version: int
    ruleset_id: str
    level: int
    tick_us: int
    content_hash: str
    match: MatchRules
    arena: ArenaRules
    cards: Mapping[str, CardDefinition]
    towers: Mapping[str, TowerDefinition]
    interaction_set: tuple[str, ...]
    sources: Mapping[str, SourceRecord]
    metadata: Mapping[str, Any]
    uncertainties: tuple[Uncertainty, ...]
    _card_aliases: Mapping[str, str] = field(repr=False)
    _tower_aliases: Mapping[str, str] = field(repr=False)

    def resolve_card_id(self, card_id_or_alias: str) -> str:
        if type(card_id_or_alias) is str and card_id_or_alias in self.cards:
            return card_id_or_alias
        normalized = normalize_identifier(card_id_or_alias)
        try:
            return self._card_aliases[normalized]
        except KeyError as error:
            raise KeyError(f"unknown card for ruleset {self.ruleset_id}: {card_id_or_alias!r}") from error

    def resolve_tower_id(self, tower_id_or_alias: str) -> str:
        if type(tower_id_or_alias) is str and tower_id_or_alias in self.towers:
            return tower_id_or_alias
        normalized = normalize_identifier(tower_id_or_alias)
        try:
            return self._tower_aliases[normalized]
        except KeyError as error:
            raise KeyError(f"unknown tower for ruleset {self.ruleset_id}: {tower_id_or_alias!r}") from error

    def card(self, card_id_or_alias: str) -> CardDefinition:
        return self.cards[self.resolve_card_id(card_id_or_alias)]

    def tower(self, tower_id_or_alias: str) -> TowerDefinition:
        return self.towers[self.resolve_tower_id(tower_id_or_alias)]

    def verify_hash(self) -> None:
        """Re-read the pinned file and verify it still has this exact digest."""

        path = ruleset_path(self.ruleset_id)
        raw = _read_json(path)
        actual = calculate_content_hash(raw)
        if self.content_hash != actual:
            raise RulesetError(
                f"ruleset hash changed on disk: loaded={self.content_hash}, current={actual}"
            )


def normalize_identifier(value: str) -> str:
    """Normalize user/detector spellings while retaining explicit alias control."""

    if not isinstance(value, str):
        raise TypeError("identifier must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized


def ruleset_path(ruleset_id: str = DEFAULT_RULESET_ID) -> Path:
    normalized = normalize_identifier(ruleset_id)
    if not _RULESET_ID_RE.fullmatch(normalized):
        raise RulesetError(f"invalid ruleset id: {ruleset_id!r}")
    return Path(__file__).resolve().parent / "rulesets" / f"{normalized}.json"


def available_rulesets(*, include_provisional: bool = False) -> tuple[str, ...]:
    """Return release-ready bundled rulesets by default.

    Provisional roster-complete artifacts are loadable by explicit ID, but
    they are intentionally excluded from the historical default listing so a
    caller cannot mistake an executable dispatch surface for a fidelity-ready
    balance release.  Tooling that wants to inspect every artifact can opt in.
    """

    directory = Path(__file__).resolve().parent / "rulesets"
    result: list[str] = []
    for path in sorted(directory.glob("*.json")):
        if not _RULESET_ID_RE.fullmatch(path.stem):
            continue
        if not include_provisional and (
            path.stem.endswith("-roster") or path.stem == "v1"
        ):
            continue
        result.append(path.stem)
    return tuple(result)


def calculate_content_hash(raw: Mapping[str, Any]) -> str:
    """Return the canonical payload digest, excluding ``content_hash`` itself."""

    payload = dict(raw)
    payload.pop("content_hash", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@lru_cache(maxsize=None)
def load_ruleset(
    ruleset_id_or_path: str | Path = DEFAULT_RULESET_ID,
    *,
    verify_hash: bool = True,
) -> Ruleset:
    """Load and validate an immutable, network-free ruleset.

    Passing a ruleset ID loads the corresponding bundled JSON.  A ``Path`` is
    accepted for tests and ruleset-build tooling.  String paths are not
    accepted accidentally; this prevents identifiers from becoming a path
    traversal surface.
    """

    if isinstance(ruleset_id_or_path, Path):
        path = ruleset_id_or_path.resolve()
    else:
        path = ruleset_path(ruleset_id_or_path)
    raw = _read_json(path)
    _reject_floats(raw)
    declared_hash = _require_str(raw, "content_hash")
    if not _CONTENT_HASH_RE.fullmatch(declared_hash):
        raise RulesetError("content_hash must be sha256:<64 lowercase hex characters>")
    actual_hash = calculate_content_hash(raw)
    if verify_hash and declared_hash != actual_hash:
        raise RulesetError(
            f"ruleset content hash mismatch for {path}: declared={declared_hash}, actual={actual_hash}"
        )
    return _parse_ruleset(raw, path=path, content_hash=actual_hash if not verify_hash else declared_hash)


def load_fixed_ruleset() -> Ruleset:
    """Load the single constant V1 runtime artifact."""

    return load_ruleset(FIXED_RULESET_ID)


def _read_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RulesetError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except FileNotFoundError as error:
        raise RulesetError(f"unknown ruleset: {path}") from error
    except json.JSONDecodeError as error:
        raise RulesetError(f"invalid ruleset JSON at {path}: {error}") from error
    if not isinstance(raw, dict):
        raise RulesetError("ruleset root must be a JSON object")
    return raw


def _reject_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise RulesetError(f"floating-point value forbidden at {path}; use integer canonical units")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_floats(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_floats(child, f"{path}[{index}]")


def _parse_ruleset(raw: dict[str, Any], *, path: Path, content_hash: str) -> Ruleset:
    allowed = {
        "schema_version", "ruleset_id", "level", "tick_us", "content_hash", "units",
        "match", "arena", "sources", "cards", "towers", "interaction_set", "metadata",
        "uncertainties",
    }
    _reject_unknown(raw, allowed, "ruleset")
    schema_version = _require_int(raw, "schema_version", minimum=1)
    if schema_version != RULESET_SCHEMA_VERSION:
        raise RulesetError(
            f"unsupported ruleset schema {schema_version}; expected {RULESET_SCHEMA_VERSION}"
        )
    ruleset_id = _require_str(raw, "ruleset_id")
    if not _RULESET_ID_RE.fullmatch(ruleset_id):
        raise RulesetError(f"invalid ruleset_id: {ruleset_id!r}")
    if path.parent == ruleset_path(ruleset_id).parent and path.stem != ruleset_id:
        raise RulesetError(f"ruleset_id {ruleset_id!r} does not match filename {path.name!r}")
    _validate_units(_require_dict(raw, "units"))

    sources_raw = _require_dict(raw, "sources")
    sources = {source_id: _parse_source(source_id, row) for source_id, row in sources_raw.items()}
    if not sources:
        raise RulesetError("ruleset must declare at least one source")

    cards_raw = _require_dict(raw, "cards")
    cards = {card_id: _parse_card(card_id, row, sources) for card_id, row in cards_raw.items()}
    for card in cards.values():
        transform = card.mechanics.get("health_transform")
        if transform is not None and str(transform.get("target_card_id")) not in cards:
            raise RulesetError(
                f"{card.card_id}.mechanics.health_transform references undefined "
                f"card {transform.get('target_card_id')!r}"
            )
        death = card.mechanics.get("death")
        if death is not None:
            legacy_child = death.get("spawn_card_id")
            if legacy_child is not None and str(legacy_child) not in cards:
                raise RulesetError(
                    f"{card.card_id}.mechanics.death references undefined child "
                    f"{legacy_child!r}"
                )
            for child in death.get("spawn_children", ()):
                child_id = str(child.get("card_id"))
                if child_id not in cards:
                    raise RulesetError(
                        f"{card.card_id}.mechanics.death references undefined child "
                        f"{child_id!r}"
                    )
    towers_raw = _require_dict(raw, "towers")
    towers = {tower_id: _parse_tower(tower_id, row, sources) for tower_id, row in towers_raw.items()}
    card_aliases = _build_alias_index(cards, kind="card")
    tower_aliases = _build_alias_index(towers, kind="tower")

    interaction_set = tuple(_require_str_list(raw, "interaction_set"))
    if len(set(interaction_set)) != len(interaction_set):
        raise RulesetError("interaction_set contains duplicate card IDs")
    missing = sorted(set(interaction_set) - set(cards))
    if missing:
        raise RulesetError(f"interaction_set references undefined cards: {missing}")

    return Ruleset(
        schema_version=schema_version,
        ruleset_id=ruleset_id,
        level=_require_int(raw, "level", minimum=1),
        tick_us=_require_int(raw, "tick_us", minimum=1),
        content_hash=content_hash,
        match=_parse_match(_require_dict(raw, "match")),
        arena=_parse_arena(_require_dict(raw, "arena")),
        cards=MappingProxyType(cards),
        towers=MappingProxyType(towers),
        interaction_set=interaction_set,
        sources=MappingProxyType(sources),
        metadata=_deep_freeze(_require_dict(raw, "metadata")),
        uncertainties=_parse_uncertainties(raw.get("uncertainties", []), "ruleset"),
        _card_aliases=MappingProxyType(card_aliases),
        _tower_aliases=MappingProxyType(tower_aliases),
    )


def _parse_source(source_id: str, value: Any) -> SourceRecord:
    row = _as_dict(value, f"sources.{source_id}")
    _reject_unknown(
        row,
        {"confidence_tier", "kind", "url", "retrieved_at", "published_at", "sha256", "lineage", "note"},
        f"sources.{source_id}",
    )
    if not _CARD_ID_RE.fullmatch(source_id):
        raise RulesetError(f"invalid source ID: {source_id!r}")
    tier = _require_str(row, "confidence_tier")
    if tier not in {"A", "B", "C", "D", "E"}:
        raise RulesetError(f"invalid confidence tier for source {source_id}: {tier!r}")
    sha256 = _optional_str(row, "sha256")
    if sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise RulesetError(f"invalid source sha256 for {source_id}")
    return SourceRecord(
        source_id=source_id,
        confidence_tier=tier,
        kind=_require_str(row, "kind"),
        url=_optional_str(row, "url"),
        retrieved_at=_require_str(row, "retrieved_at"),
        published_at=_optional_str(row, "published_at"),
        sha256=sha256,
        lineage=_optional_str(row, "lineage"),
        note=_optional_str(row, "note"),
    )


def _parse_projectile(value: Any, context: str) -> ProjectileDefinition | None:
    if value is None:
        return None
    row = _as_dict(value, context)
    _reject_unknown(
        row,
        {
            "projectile_id",
            "speed_mtile_per_s",
            "radius_mtile",
            "start_radius_mtile",
            "homing",
        },
        context,
    )
    return ProjectileDefinition(
        projectile_id=_require_str(row, "projectile_id"),
        speed_mtile_per_s=_require_int(row, "speed_mtile_per_s", minimum=1),
        radius_mtile=_require_int(row, "radius_mtile", minimum=0),
        start_radius_mtile=_require_int(row, "start_radius_mtile", minimum=0),
        homing=_require_bool(row, "homing"),
    )


def _parse_card(card_id: str, value: Any, sources: Mapping[str, SourceRecord]) -> CardDefinition:
    context = f"cards.{card_id}"
    row = _as_dict(value, context)
    expected = {
        "name", "aliases", "kind", "elixir_milli", "deploy_time_us", "spawn_count",
        "hitpoints", "damage", "crown_tower_damage", "attack_interval_us",
        "first_hit_delay_us", "move_speed_mtile_per_s", "range_mtile", "sight_range_mtile",
        "collision_radius_mtile", "mass", "lifetime_us", "targets", "projectile", "area_radius_mtile",
        "mechanics", "provenance", "uncertainties",
    }
    _reject_unknown(row, expected, context)
    if not _CARD_ID_RE.fullmatch(card_id):
        raise RulesetError(f"invalid card ID: {card_id!r}")
    kind = _require_str(row, "kind")
    if kind not in _CARD_KINDS:
        raise RulesetError(f"invalid card kind at {context}.kind: {kind!r}")
    targets = tuple(_require_str_list(row, "targets"))
    invalid_targets = sorted(set(targets) - _TARGETS)
    if invalid_targets:
        raise RulesetError(f"invalid targets at {context}: {invalid_targets}")
    provenance = _parse_provenance(_require_dict(row, "provenance"), sources, context)
    mechanics = _require_dict(row, "mechanics")
    _validate_mechanics(mechanics, context)
    return CardDefinition(
        card_id=card_id,
        name=_require_str(row, "name"),
        aliases=tuple(_require_str_list(row, "aliases")),
        kind=kind,
        elixir_milli=_require_int(row, "elixir_milli", minimum=0),
        deploy_time_us=_require_int(row, "deploy_time_us", minimum=0),
        spawn_count=_require_int(row, "spawn_count", minimum=0),
        hitpoints=_optional_int(row, "hitpoints", minimum=1),
        damage=_optional_int(row, "damage", minimum=0),
        crown_tower_damage=_optional_int(row, "crown_tower_damage", minimum=0),
        attack_interval_us=_optional_int(row, "attack_interval_us", minimum=1),
        first_hit_delay_us=_optional_int(row, "first_hit_delay_us", minimum=0),
        move_speed_mtile_per_s=_optional_int(row, "move_speed_mtile_per_s", minimum=1),
        range_mtile=_optional_int(row, "range_mtile", minimum=0),
        sight_range_mtile=_optional_int(row, "sight_range_mtile", minimum=0),
        collision_radius_mtile=_optional_int(row, "collision_radius_mtile", minimum=0),
        mass=_optional_int(row, "mass", minimum=0),
        lifetime_us=_optional_int(row, "lifetime_us", minimum=1),
        targets=targets,
        projectile=_parse_projectile(row.get("projectile"), f"{context}.projectile"),
        area_radius_mtile=_optional_int(row, "area_radius_mtile", minimum=0),
        mechanics=_deep_freeze(mechanics),
        provenance=MappingProxyType(provenance),
        uncertainties=_parse_uncertainties(row.get("uncertainties", []), context),
    )


def _validate_mechanics(row: dict[str, Any], context: str) -> None:
    required = {
        "placement_class", "movement_layer", "building_only", "spawn_layout_mtile", "death",
        "suicide_on_attack", "crown_tower_connection", "projectile_mode", "impact_mode", "status",
        "knockback_mtile", "piercing", "spell_origin",
    }
    missing = sorted(required - set(row))
    unknown = sorted(
        set(row)
        - required
        - {
            "lifetime_decay",
            "lifetime_start",
            "targetable_during_deploy",
            "knockback_direction",
            "spawn",
            "spawn_on_impact",
            "elixir_generation",
            "persistent_effect",
            "clone",
            "target_limit",
            "target_selection",
            "reset_attack",
            "chain_attack",
            "multi_target_attack",
            "reflection",
            "charge_attack",
            "dash",
            "hook",
            "recoil_mtile",
            "ramp_attack",
            "revive",
            "revive_egg",
            # Explicit patch-note fields which do not fit the compact
            # executable scalar schema yet.  They remain typed provenance
            # data until their dedicated component is implemented.
            "heal_radius_mtile",
            "self_heal",
            "enemy_slowdown_milli",
            "tower_spawn_damage",
            "spawn_child_hitpoints",
            "spawn_range",
            "damage_by_target_count",
            "crown_tower_damage_by_target_count",
            "projectile_speed_code",
            "spirit_one_shot",
            "heal_amount",
            "stealth",
            "trigger_on_target",
            "charge_threshold_permille",
            "charge_duration_us",
            "charged_speed_mtile_per_s",
            "charge_range_mtile",
            "trigger_on_building_contact",
            "attack_windup_mode",
            "impact_targets",
            "secondary_attack",
            "heal_on_impact",
            "health_transform",
            "carrier",
            "shield",
            "stealth_recloak_us",
            "burrow",
            "spawn_children",
            "line_piercing",
            "returning_projectile",
            "pellets",
            "jump",
            "deploy_effect",
            "death_rage",
            "snare",
            "river_jump",
            "concealment",
            "rolling_range_mtile",
            "spawn_stagger_us",
            "impact_delay_us",
            "mirror_spawn_layout",
            "primary_targets",
            "spread_targets",
            "bayonet",
            "min_attack_range_mtile",
            "counts_as_troop",
            "hook_pullable",
            "pullable_by_area_effect",
            "cloneable_by_clone",
            "cannot_hit_jumping",
            "building_footprint_size",
            # Balance values for currently excluded Hero/Evolution/champion
            # variants.  These are typed metadata blocks only; the V1 engine
            # remains fail-closed until the corresponding action/component is
            # implemented.
            "ability",
            "hero_ability",
            "evolution",
            "monster",
        }
    )
    if missing or unknown:
        raise RulesetError(f"{context}.mechanics keys mismatch: missing={missing}, unknown={unknown}")
    for name in ("building_only", "suicide_on_attack", "piercing"):
        if not isinstance(row[name], bool):
            raise RulesetError(f"{context}.mechanics.{name} must be boolean")
    for name in (
        "counts_as_troop",
        "hook_pullable",
        "pullable_by_area_effect",
        "cloneable_by_clone",
        "cannot_hit_jumping",
    ):
        if name in row and not isinstance(row[name], bool):
            raise RulesetError(f"{context}.mechanics.{name} must be boolean")
    if "building_footprint_size" in row:
        _require_int(row, "building_footprint_size", minimum=1)
    layout = row["spawn_layout_mtile"]
    if layout or row.get("placement_class") not in {"spell_anywhere", "spells", "restricted_spell"}:
        _validate_offsets(layout, f"{context}.mechanics.spawn_layout_mtile")
    if "target_limit" in row:
        _require_int(row, "target_limit", minimum=1)
    if "target_selection" in row and row["target_selection"] not in {
        "highest_hp",
        "nearest",
    }:
        raise RulesetError(
            f"{context}.mechanics.target_selection must be 'highest_hp' or 'nearest'"
        )
    if "reset_attack" in row and not isinstance(row["reset_attack"], bool):
        raise RulesetError(f"{context}.mechanics.reset_attack must be boolean")
    if "attack_windup_mode" in row and row["attack_windup_mode"] not in {"recharge"}:
        raise RulesetError(
            f"{context}.mechanics.attack_windup_mode must be 'recharge'"
        )
    if "impact_targets" in row:
        values = row["impact_targets"]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or value not in _TARGETS for value in values)
        ):
            raise RulesetError(
                f"{context}.mechanics.impact_targets must be a non-empty list of valid target classes"
            )
    heal_on_impact = row.get("heal_on_impact")
    if heal_on_impact is not None:
        heal_context = f"{context}.mechanics.heal_on_impact"
        heal_row = _as_dict(heal_on_impact, heal_context)
        _reject_unknown(
            heal_row,
            {"amount", "radius_mtile", "targets", "exclude_source"},
            heal_context,
        )
        _require_int(heal_row, "amount", minimum=1)
        _require_int(heal_row, "radius_mtile", minimum=0)
        targets = heal_row.get("targets")
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(value, str) or value not in {"air", "ground"} for value in targets)
        ):
            raise RulesetError(
                f"{heal_context}.targets must be a non-empty list containing only 'air' or 'ground'"
            )
        if not isinstance(heal_row.get("exclude_source", True), bool):
            raise RulesetError(f"{heal_context}.exclude_source must be boolean")
    health_transform = row.get("health_transform")
    if health_transform is not None:
        transform_context = f"{context}.mechanics.health_transform"
        transform_row = _as_dict(health_transform, transform_context)
        _reject_unknown(
            transform_row,
            {
                "threshold_permille",
                "target_card_id",
                "preserve_hp",
                "preserve_max_hp",
                "lifetime_us",
            },
            transform_context,
        )
        threshold = _require_int(transform_row, "threshold_permille", minimum=1)
        if threshold > 1_000:
            raise RulesetError(
                f"{transform_context}.threshold_permille must not exceed 1000"
            )
        target_card_id = _require_str(transform_row, "target_card_id")
        if not _CARD_ID_RE.fullmatch(target_card_id):
            raise RulesetError(
                f"{transform_context}.target_card_id is not a valid card ID"
            )
        for name in ("preserve_hp", "preserve_max_hp"):
            if not isinstance(transform_row.get(name, True), bool):
                raise RulesetError(f"{transform_context}.{name} must be boolean")
        if "lifetime_us" in transform_row:
            _require_int(transform_row, "lifetime_us", minimum=1)
    secondary = row.get("secondary_attack")
    if secondary is not None:
        secondary_context = f"{context}.mechanics.secondary_attack"
        secondary = _as_dict(secondary, secondary_context)
        secondary_required = {
                "min_range_mtile",
                "max_range_mtile",
                "attack_interval_us",
                "first_hit_delay_us",
                "damage",
                "crown_tower_damage",
                "area_radius_mtile",
                "projectile_speed_mtile_per_s",
                "projectile_radius_mtile",
                "targets",
            }
        missing_secondary = sorted(secondary_required - set(secondary))
        unknown_secondary = sorted(
            set(secondary) - secondary_required - {"status", "troops_only"}
        )
        if missing_secondary or unknown_secondary:
            raise RulesetError(
                f"{secondary_context} keys mismatch: missing={missing_secondary}, "
                f"unknown={unknown_secondary}"
            )
        for name in (
            "min_range_mtile",
            "max_range_mtile",
            "attack_interval_us",
            "first_hit_delay_us",
            "damage",
            "crown_tower_damage",
            "area_radius_mtile",
            "projectile_speed_mtile_per_s",
            "projectile_radius_mtile",
        ):
            _require_int(
                secondary,
                name,
                minimum=0
                if name in {
                    "min_range_mtile", "first_hit_delay_us", "crown_tower_damage",
                    "area_radius_mtile", "projectile_radius_mtile",
                }
                else 1,
            )
        if secondary["max_range_mtile"] < secondary["min_range_mtile"]:
            raise RulesetError(f"{secondary_context}.max_range_mtile must be >= min_range_mtile")
        if "troops_only" in secondary and type(secondary["troops_only"]) is not bool:
            raise RulesetError(f"{secondary_context}.troops_only must be boolean")
        targets = secondary.get("targets")
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(value, str) or value not in _TARGETS for value in targets)
        ):
            raise RulesetError(
                f"{secondary_context}.targets must be a non-empty list of valid target classes"
            )
        _validate_status(secondary.get("status"), f"{secondary_context}.status")

    # Cross-cutting card mechanics introduced by the base-card audit are kept
    # in the permissive mechanics object for forward compatibility, but their
    # shapes still need strict validation so malformed generated rulesets fail
    # at load time rather than during a replay.
    if "spawn_stagger_us" in row:
        _require_int(row, "spawn_stagger_us", minimum=0)
    if "impact_delay_us" in row:
        _require_int(row, "impact_delay_us", minimum=0)
    if "mirror_spawn_layout" in row and not isinstance(row["mirror_spawn_layout"], bool):
        raise RulesetError(f"{context}.mechanics.mirror_spawn_layout must be boolean")
    if "spread_targets" in row and not isinstance(row["spread_targets"], bool):
        raise RulesetError(f"{context}.mechanics.spread_targets must be boolean")
    if "min_attack_range_mtile" in row:
        _require_int(row, "min_attack_range_mtile", minimum=0)
    for name in ("primary_targets",):
        if name in row:
            values = row[name]
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or value not in _TARGETS for value in values)
            ):
                raise RulesetError(
                    f"{context}.mechanics.{name} must be a non-empty list of valid target classes"
                )
    bayonet = row.get("bayonet")
    if bayonet is not None:
        bayonet_context = f"{context}.mechanics.bayonet"
        bayonet_row = _as_dict(bayonet, bayonet_context)
        _reject_unknown(
            bayonet_row,
            {"range_mtile", "damage", "crown_tower_damage", "targets"},
            bayonet_context,
        )
        _require_int(bayonet_row, "range_mtile", minimum=0)
        _require_int(bayonet_row, "damage", minimum=0)
        _require_int(bayonet_row, "crown_tower_damage", minimum=0)
        targets = bayonet_row.get("targets")
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(value, str) or value not in _TARGETS for value in targets)
        ):
            raise RulesetError(
                f"{bayonet_context}.targets must be a non-empty list of valid target classes"
            )
    river_jump = row.get("river_jump")
    if river_jump is not None:
        river_context = f"{context}.mechanics.river_jump"
        river_row = _as_dict(river_jump, river_context)
        _reject_unknown(river_row, {"duration_us"}, river_context)
        _require_int(river_row, "duration_us", minimum=1)
    concealment = row.get("concealment")
    if concealment is not None:
        concealment_context = f"{context}.mechanics.concealment"
        concealment_row = _as_dict(concealment, concealment_context)
        _reject_unknown(
            concealment_row,
            {"reveal_range_mtile", "starts_concealed", "earthquake_hits", "freeze_suppresses_reveal"},
            concealment_context,
        )
        _require_int(concealment_row, "reveal_range_mtile", minimum=0)
        for name in ("starts_concealed", "earthquake_hits", "freeze_suppresses_reveal"):
            if not isinstance(concealment_row.get(name), bool):
                raise RulesetError(f"{concealment_context}.{name} must be boolean")

    # Keep balance-only fields for unsupported special variants explicit and
    # integer-valued.  They are deliberately separate from the executable
    # action components above: loading a future card with one of these blocks
    # must not imply that the current engine can play its Hero/Evolution
    # ability.
    balance_blocks = {
        "ability": {"charge_damage"},
        "hero_ability": {"duration_us"},
        "evolution": {"spear_damage", "rage_duration_us"},
        "monster": {"hitpoints"},
    }
    for name, allowed in balance_blocks.items():
        block = row.get(name)
        if block is None:
            continue
        block_context = f"{context}.mechanics.{name}"
        block = _as_dict(block, block_context)
        _reject_unknown(block, allowed, block_context)
        if not block:
            raise RulesetError(f"{block_context} must not be empty")
        for field_name in block:
            _require_int(block, field_name, minimum=1)
    for component_name in ("chain_attack", "multi_target_attack"):
        component = row.get(component_name)
        if component is None:
            continue
        component_context = f"{context}.mechanics.{component_name}"
        component = _as_dict(component, component_context)
        unknown_component_fields = sorted(
            set(component) - {"max_targets", "chain_range_mtile", "selection", "range_mtile", "chain_delay_us"}
        )
        if unknown_component_fields:
            raise RulesetError(
                f"unknown fields at {component_context}: {unknown_component_fields}"
            )
        missing_component_fields = sorted({"max_targets", "selection"} - set(component))
        if component_name == "chain_attack" and "chain_range_mtile" not in component:
            missing_component_fields.append("chain_range_mtile")
        if component_name == "multi_target_attack" and "range_mtile" not in component:
            missing_component_fields.append("range_mtile")
        if missing_component_fields:
            raise RulesetError(
                f"missing fields at {component_context}: {sorted(set(missing_component_fields))}"
            )
        _require_int(component, "max_targets", minimum=2)
        if component.get("chain_range_mtile") is not None:
            _require_int(component, "chain_range_mtile", minimum=1)
        if component.get("range_mtile") is not None:
            _require_int(component, "range_mtile", minimum=1)
        if component.get("chain_delay_us") is not None:
            _require_int(component, "chain_delay_us", minimum=0)
        if component.get("selection") not in {"nearest", "highest_hp"}:
            raise RulesetError(
                f"{component_context}.selection must be 'nearest' or 'highest_hp'"
            )
    reflection = row.get("reflection")
    if reflection is not None:
        reflection_context = f"{context}.mechanics.reflection"
        reflection = _as_dict(reflection, reflection_context)
        _reject_unknown(
            reflection,
            {"damage", "crown_tower_damage", "radius_mtile", "targets", "stun_duration_us"},
            reflection_context,
        )
        for name in ("damage", "crown_tower_damage", "radius_mtile", "stun_duration_us"):
            _require_int(reflection, name, minimum=0)
        targets = reflection.get("targets")
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(value, str) or value not in _TARGETS for value in targets)
        ):
            raise RulesetError(
                f"{reflection_context}.targets must be a non-empty list of valid target classes"
            )
    charge_attack = row.get("charge_attack")
    if charge_attack is not None:
        charge_context = f"{context}.mechanics.charge_attack"
        charge_attack = _as_dict(charge_attack, charge_context)
        _reject_unknown(
            charge_attack,
            {
                "charge_distance_mtile",
                "charged_speed_mtile_per_s",
                "charge_damage",
                "reset_on_hit",
            },
            charge_context,
        )
        for name in (
            "charge_distance_mtile",
            "charged_speed_mtile_per_s",
            "charge_damage",
        ):
            _require_int(charge_attack, name, minimum=1)
        if not isinstance(charge_attack.get("reset_on_hit"), bool):
            raise RulesetError(f"{charge_context}.reset_on_hit must be boolean")
    dash = row.get("dash")
    if dash is not None:
        dash_context = f"{context}.mechanics.dash"
        dash = _as_dict(dash, dash_context)
        _reject_unknown(
            dash,
            {
                "dash_range_mtile",
                "dash_damage",
                "duration_us",
                "min_dash_distance_mtile",
                "reset_on_hit",
            },
            dash_context,
        )
        for name in (
            "dash_range_mtile",
            "dash_damage",
            "duration_us",
            "min_dash_distance_mtile",
        ):
            if name not in dash:
                continue
            _require_int(dash, name, minimum=1)
        if not isinstance(dash.get("reset_on_hit"), bool):
            raise RulesetError(f"{dash_context}.reset_on_hit must be boolean")
    hook = row.get("hook")
    if hook is not None:
        hook_context = f"{context}.mechanics.hook"
        hook = _as_dict(hook, hook_context)
        _reject_unknown(
            hook,
            {
                "hook_range_mtile",
                "min_hook_range_mtile",
                "pull_distance_mtile",
                "pull_troops_only",
            },
            hook_context,
        )
        _require_int(hook, "hook_range_mtile", minimum=1)
        if "min_hook_range_mtile" in hook:
            _require_int(hook, "min_hook_range_mtile", minimum=0)
        _require_int(hook, "pull_distance_mtile", minimum=0)
        if not isinstance(hook.get("pull_troops_only"), bool):
            raise RulesetError(f"{hook_context}.pull_troops_only must be boolean")
    ramp_attack = row.get("ramp_attack")
    if ramp_attack is not None:
        ramp_context = f"{context}.mechanics.ramp_attack"
        ramp_attack = _as_dict(ramp_attack, ramp_context)
        _reject_unknown(
            ramp_attack,
            {"damage_schedule", "stage_thresholds_us", "reset_on_target_loss"},
            ramp_context,
        )
        damage_schedule = ramp_attack.get("damage_schedule")
        thresholds = ramp_attack.get("stage_thresholds_us")
        if (
            not isinstance(damage_schedule, list)
            or not damage_schedule
            or any(type(value) is not int or value <= 0 for value in damage_schedule)
        ):
            raise RulesetError(
                f"{ramp_context}.damage_schedule must be a non-empty list of positive integers"
            )
        if (
            not isinstance(thresholds, list)
            or len(thresholds) != len(damage_schedule)
            or any(type(value) is not int or value < 0 for value in thresholds)
            or thresholds[0] != 0
            or any(left >= right for left, right in zip(thresholds, thresholds[1:]))
        ):
            raise RulesetError(
                f"{ramp_context}.stage_thresholds_us must be strictly increasing, start at 0, "
                "and match damage_schedule length"
            )
        if not isinstance(ramp_attack.get("reset_on_target_loss"), bool):
            raise RulesetError(f"{ramp_context}.reset_on_target_loss must be boolean")
    revive = row.get("revive")
    if revive is not None:
        revive_context = f"{context}.mechanics.revive"
        revive = _as_dict(revive, revive_context)
        _reject_unknown(
            revive,
            {
                "egg_card_id",
                "egg_hitpoints",
                "egg_lifetime_us",
                "revived_hitpoints",
                "revived_damage",
                "max_revives",
            },
            revive_context,
        )
        for name in (
            "egg_hitpoints",
            "egg_lifetime_us",
            "revived_hitpoints",
            "revived_damage",
            "max_revives",
        ):
            _require_int(revive, name, minimum=1)
        if not _CARD_ID_RE.fullmatch(_require_str(revive, "egg_card_id")):
            raise RulesetError(f"{revive_context}.egg_card_id is not a valid card ID")
    revive_egg = row.get("revive_egg")
    if revive_egg is not None:
        egg_context = f"{context}.mechanics.revive_egg"
        revive_egg = _as_dict(revive_egg, egg_context)
        _reject_unknown(revive_egg, {"hatch_card_id"}, egg_context)
        hatch_card_id = _require_str(revive_egg, "hatch_card_id")
        if not _CARD_ID_RE.fullmatch(hatch_card_id):
            raise RulesetError(f"{egg_context}.hatch_card_id is not a valid card ID")
    for name in (
        "heal_radius_mtile",
        "enemy_slowdown_milli",
        "tower_spawn_damage",
        "spawn_child_hitpoints",
        "projectile_speed_code",
        "heal_amount",
        "charge_threshold_permille",
        "charge_duration_us",
        "charged_speed_mtile_per_s",
        "charge_range_mtile",
        "recoil_mtile",
    ):
        if name in row and (
            not isinstance(row[name], int) or isinstance(row[name], bool) or row[name] < 0
        ):
            raise RulesetError(f"{context}.mechanics.{name} must be a non-negative integer")
    for name in (
        "self_heal",
        "spirit_one_shot",
        "stealth",
        "trigger_on_target",
        "trigger_on_building_contact",
    ):
        if name in row and not isinstance(row[name], bool):
            raise RulesetError(f"{context}.mechanics.{name} must be boolean")
    for name in ("stealth_recloak_us",):
        if name in row and (
            type(row[name]) is not int or row[name] < 0
        ):
            raise RulesetError(f"{context}.mechanics.{name} must be a non-negative integer")
    shield = row.get("shield")
    if shield is not None:
        shield_context = f"{context}.mechanics.shield"
        shield_row = _as_dict(shield, shield_context)
        _reject_unknown(shield_row, {"hitpoints"}, shield_context)
        _require_int(shield_row, "hitpoints", minimum=1)
    burrow = row.get("burrow")
    if burrow is not None:
        burrow_context = f"{context}.mechanics.burrow"
        burrow_row = _as_dict(burrow, burrow_context)
        _reject_unknown(burrow_row, {"duration_us", "target_anywhere", "targetable_during_burrow"}, burrow_context)
        _require_int(burrow_row, "duration_us", minimum=1)
        for name in ("target_anywhere", "targetable_during_burrow"):
            if not isinstance(burrow_row.get(name, False), bool):
                raise RulesetError(f"{burrow_context}.{name} must be boolean")
    spawn_children = row.get("spawn_children")
    if spawn_children is not None:
        if not isinstance(spawn_children, list) or not spawn_children:
            raise RulesetError(f"{context}.mechanics.spawn_children must be a non-empty list")
        for index, child in enumerate(spawn_children):
            child_context = f"{context}.mechanics.spawn_children[{index}]"
            child_row = _as_dict(child, child_context)
            unknown_child_fields = sorted(set(child_row) - {"card_id", "count", "offsets_mtile"})
            if unknown_child_fields:
                raise RulesetError(f"unknown fields at {child_context}: {unknown_child_fields}")
            child_id = _require_str(child_row, "card_id")
            if not _CARD_ID_RE.fullmatch(child_id):
                raise RulesetError(f"{child_context}.card_id is not a valid card ID")
            count = _require_int(child_row, "count", minimum=1)
            offsets = child_row.get("offsets_mtile")
            if offsets is not None and (
                not isinstance(offsets, list)
                or len(offsets) != count
                or any(
                    not isinstance(offset, list)
                    or len(offset) != 2
                    or any(type(value) is not int for value in offset)
                    for offset in offsets
                )
            ):
                raise RulesetError(f"{child_context}.offsets_mtile has invalid layout")
    line_piercing = row.get("line_piercing")
    if line_piercing is not None:
        line_context = f"{context}.mechanics.line_piercing"
        line_row = _as_dict(line_piercing, line_context)
        _reject_unknown(line_row, {"length_mtile", "width_mtile"}, line_context)
        _require_int(line_row, "length_mtile", minimum=1)
        _require_int(line_row, "width_mtile", minimum=0)
    returning = row.get("returning_projectile")
    if returning is not None:
        return_context = f"{context}.mechanics.returning_projectile"
        return_row = _as_dict(returning, return_context)
        _reject_unknown(return_row, {"return_speed_mtile_per_s", "return_radius_mtile"}, return_context)
        _require_int(return_row, "return_speed_mtile_per_s", minimum=1)
        _require_int(return_row, "return_radius_mtile", minimum=0)
    pellets = row.get("pellets")
    if pellets is not None:
        pellet_context = f"{context}.mechanics.pellets"
        pellet_row = _as_dict(pellets, pellet_context)
        _reject_unknown(pellet_row, {"count", "spread_mtile"}, pellet_context)
        _require_int(pellet_row, "count", minimum=2)
        _require_int(pellet_row, "spread_mtile", minimum=0)
    jump = row.get("jump")
    if jump is not None:
        jump_context = f"{context}.mechanics.jump"
        jump_row = _as_dict(jump, jump_context)
        _reject_unknown(
            jump_row,
            {"min_range_mtile", "max_range_mtile", "duration_us", "damage", "radius_mtile", "spawn_damage"},
            jump_context,
        )
        for name in ("min_range_mtile", "max_range_mtile", "duration_us", "damage", "radius_mtile"):
            _require_int(jump_row, name, minimum=0 if name in {"damage", "radius_mtile"} else 1)
        if jump_row["max_range_mtile"] < jump_row["min_range_mtile"]:
            raise RulesetError(f"{jump_context}.max_range_mtile must be >= min_range_mtile")
        if not isinstance(jump_row.get("spawn_damage", True), bool):
            raise RulesetError(f"{jump_context}.spawn_damage must be boolean")
    deploy_effect = row.get("deploy_effect")
    if deploy_effect is not None:
        deploy_context = f"{context}.mechanics.deploy_effect"
        deploy_row = _as_dict(deploy_effect, deploy_context)
        deploy_required = {"kind", "duration_us", "radius_mtile", "speed_multiplier_milli", "hit_speed_multiplier_milli", "targets"}
        deploy_optional = {"damage", "crown_tower_damage", "knockback_mtile"}
        missing_deploy = sorted(deploy_required - set(deploy_row))
        unknown_deploy = sorted(set(deploy_row) - deploy_required - deploy_optional)
        if missing_deploy or unknown_deploy:
            raise RulesetError(
                f"{deploy_context} keys mismatch: missing={missing_deploy}, unknown={unknown_deploy}"
            )
        _require_str(deploy_row, "kind")
        _require_int(deploy_row, "duration_us", minimum=0)
        _require_int(deploy_row, "radius_mtile", minimum=0)
        _require_int(deploy_row, "speed_multiplier_milli", minimum=0)
        _require_int(deploy_row, "hit_speed_multiplier_milli", minimum=0)
        for name in deploy_optional:
            if name in deploy_row:
                _require_int(deploy_row, name, minimum=0)
        targets = deploy_row.get("targets")
        if not isinstance(targets, list) or not targets or any(value not in _TARGETS for value in targets):
            raise RulesetError(f"{deploy_context}.targets must contain valid target classes")
    death_rage = row.get("death_rage")
    if death_rage is not None:
        rage_context = f"{context}.mechanics.death_rage"
        rage_row = _as_dict(death_rage, rage_context)
        _reject_unknown(rage_row, {"duration_us", "tick_interval_us", "radius_mtile", "speed_multiplier_milli", "hit_speed_multiplier_milli", "targets"}, rage_context)
        for name in ("duration_us", "tick_interval_us", "radius_mtile", "speed_multiplier_milli", "hit_speed_multiplier_milli"):
            _require_int(rage_row, name, minimum=1 if name != "radius_mtile" else 0)
        targets = rage_row.get("targets")
        if not isinstance(targets, list) or not targets or any(value not in _TARGETS for value in targets):
            raise RulesetError(f"{rage_context}.targets must contain valid target classes")
    snare = row.get("snare")
    if snare is not None:
        snare_context = f"{context}.mechanics.snare"
        snare_row = _as_dict(snare, snare_context)
        _reject_unknown(snare_row, {"duration_us", "speed_multiplier_milli", "hit_speed_multiplier_milli", "targets"}, snare_context)
        for name in ("duration_us", "speed_multiplier_milli", "hit_speed_multiplier_milli"):
            _require_int(snare_row, name, minimum=0 if name != "duration_us" else 1)
        targets = snare_row.get("targets")
        if not isinstance(targets, list) or not targets or any(value not in _TARGETS for value in targets):
            raise RulesetError(f"{snare_context}.targets must contain valid target classes")
    if "charge_threshold_permille" in row and row["charge_threshold_permille"] > 1_000:
        raise RulesetError(
            f"{context}.mechanics.charge_threshold_permille must not exceed 1000"
        )
    if "spawn_range" in row and row["spawn_range"] not in {"short", "long"}:
        raise RulesetError(f"{context}.mechanics.spawn_range must be 'short' or 'long'")
    clone = row.get("clone")
    if clone is not None:
        clone_row = _as_dict(clone, f"{context}.mechanics.clone")
        _reject_unknown(
            clone_row,
            {"copy_kind", "clone_hp", "clone_max_hp", "exclude_clones"},
            f"{context}.mechanics.clone",
        )
        if clone_row.get("copy_kind") != "troop":
            raise RulesetError(
                f"{context}.mechanics.clone.copy_kind must be 'troop'"
            )
        _require_int(clone_row, "clone_hp", minimum=1)
        _require_int(clone_row, "clone_max_hp", minimum=1)
        if clone_row["clone_hp"] > clone_row["clone_max_hp"]:
            raise RulesetError(
                f"{context}.mechanics.clone.clone_hp must not exceed clone_max_hp"
            )
        if not isinstance(clone_row.get("exclude_clones", True), bool):
            raise RulesetError(
                f"{context}.mechanics.clone.exclude_clones must be boolean"
            )
    for name in ("damage_by_target_count", "crown_tower_damage_by_target_count"):
        if name in row:
            values = row[name]
            if not isinstance(values, dict) or any(
                not isinstance(key, str)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for key, value in values.items()
            ):
                raise RulesetError(
                    f"{context}.mechanics.{name} must map string target buckets to non-negative integers"
                )
    knockback = row["knockback_mtile"]
    if not isinstance(knockback, int) or isinstance(knockback, bool) or knockback < 0:
        raise RulesetError(f"{context}.mechanics.knockback_mtile must be a non-negative integer")
    if row.get("knockback_direction") not in {None, "radial", "projectile_travel"}:
        raise RulesetError(
            f"{context}.mechanics.knockback_direction must be null, 'radial', or "
            "'projectile_travel'"
        )
    if row.get("lifetime_decay") not in {None, "linear_hp"}:
        raise RulesetError(
            f"{context}.mechanics.lifetime_decay must be null or 'linear_hp'"
        )
    if row.get("lifetime_start") not in {None, "placement", "active", "transform"}:
        raise RulesetError(
            f"{context}.mechanics.lifetime_start must be null, 'placement', 'active', or 'transform'"
        )
    targetable_during_deploy = row.get("targetable_during_deploy", False)
    if not isinstance(targetable_during_deploy, bool):
        raise RulesetError(
            f"{context}.mechanics.targetable_during_deploy must be boolean"
        )
    spawn = row.get("spawn")
    if spawn is not None:
        spawn_row = _as_dict(spawn, f"{context}.mechanics.spawn")
        spawn_required = {
            "card_id", "interval_us", "start_delay_us", "max_alive", "count"
        }
        spawn_allowed = spawn_required | {
            "activation_range_mtile",
            "requires_visible_enemy",
            "child_deploy_time_us",
            "child_spawn_stagger_us",
        }
        missing_spawn = sorted(spawn_required - set(spawn_row))
        unknown_spawn = sorted(set(spawn_row) - spawn_allowed)
        if missing_spawn or unknown_spawn:
            raise RulesetError(
                f"{context}.mechanics.spawn keys mismatch: "
                f"missing={missing_spawn}, unknown={unknown_spawn}"
            )
        _require_str(spawn_row, "card_id")
        _require_int(spawn_row, "interval_us", minimum=1)
        _require_int(spawn_row, "start_delay_us", minimum=0)
        max_alive = spawn_row.get("max_alive")
        if max_alive is not None:
            _require_int(spawn_row, "max_alive", minimum=1)
        _require_int(spawn_row, "count", minimum=1)
        if "activation_range_mtile" in spawn_row:
            _require_int(spawn_row, "activation_range_mtile", minimum=1)
        if "child_deploy_time_us" in spawn_row:
            _require_int(spawn_row, "child_deploy_time_us", minimum=0)
        if "child_spawn_stagger_us" in spawn_row:
            _require_int(spawn_row, "child_spawn_stagger_us", minimum=0)
        if "requires_visible_enemy" in spawn_row and not isinstance(
            spawn_row["requires_visible_enemy"], bool
        ):
            raise RulesetError(
                f"{context}.mechanics.spawn.requires_visible_enemy must be boolean"
            )
    spawn_on_impact = row.get("spawn_on_impact")
    if spawn_on_impact is not None:
        impact_row = _as_dict(
            spawn_on_impact,
            f"{context}.mechanics.spawn_on_impact",
        )
        impact_context = f"{context}.mechanics.spawn_on_impact"
        unknown_impact = sorted(
            set(impact_row) - {"card_id", "count", "child_deploy_time_us"}
        )
        if unknown_impact:
            raise RulesetError(
                f"unknown fields at {impact_context}: {unknown_impact}"
            )
        _require_str(impact_row, "card_id")
        _require_int(impact_row, "count", minimum=1)
        if "child_deploy_time_us" in impact_row:
            _require_int(
                impact_row,
                "child_deploy_time_us",
                minimum=0,
            )
    carrier = row.get("carrier")
    if carrier is not None:
        carrier_context = f"{context}.mechanics.carrier"
        carrier_row = _as_dict(carrier, carrier_context)
        _reject_unknown(
            carrier_row,
            {"child_card_id", "count", "offsets_mtile", "release_on_death"},
            carrier_context,
        )
        child_card_id = _require_str(carrier_row, "child_card_id")
        if not _CARD_ID_RE.fullmatch(child_card_id):
            raise RulesetError(f"{carrier_context}.child_card_id is not a valid card ID")
        count = _require_int(carrier_row, "count", minimum=1)
        offsets = carrier_row.get("offsets_mtile")
        if (
            not isinstance(offsets, list)
            or len(offsets) != count
            or any(
                not isinstance(offset, list)
                or len(offset) != 2
                or any(type(value) is not int for value in offset)
                for offset in offsets
            )
        ):
            raise RulesetError(
                f"{carrier_context}.offsets_mtile must contain one integer [x, y] pair per child"
            )
        if not isinstance(carrier_row.get("release_on_death", True), bool):
            raise RulesetError(f"{carrier_context}.release_on_death must be boolean")
    generation = row.get("elixir_generation")
    if generation is not None:
        generation_row = _as_dict(
            generation,
            f"{context}.mechanics.elixir_generation",
        )
        _reject_unknown(
            generation_row,
            {"interval_us", "amount_milli"},
            f"{context}.mechanics.elixir_generation",
        )
        _require_int(generation_row, "interval_us", minimum=1)
        _require_int(generation_row, "amount_milli", minimum=1)
    persistent = row.get("persistent_effect")
    if persistent is not None:
        persistent_row = _as_dict(
            persistent,
            f"{context}.mechanics.persistent_effect",
        )
        persistent_allowed = {
            "duration_us",
            "duration_anchor",
            "initial_delay_us",
            "tick_interval_us",
            "radius_mtile",
            "damage_per_tick",
            "crown_damage_per_tick",
            "building_damage_per_tick",
            "targets",
            "status",
            "knockback_mtile",
            "pull_to_center_mtile",
            "spawn",
            "max_pulses",
            "damage_schedule",
            "crown_damage_schedule",
            "damage_by_target_count",
            "crown_damage_by_target_count",
            "friendly_status",
            "friendly_targets",
        }
        unknown_persistent = sorted(set(persistent_row) - persistent_allowed)
        if unknown_persistent:
            raise RulesetError(
                f"unknown fields at {context}.mechanics.persistent_effect: "
                f"{unknown_persistent}"
            )
        _require_int(persistent_row, "duration_us", minimum=1)
        _require_int(persistent_row, "tick_interval_us", minimum=1)
        if "initial_delay_us" in persistent_row:
            _require_int(persistent_row, "initial_delay_us", minimum=0)
        if persistent_row.get("duration_anchor") not in {None, "after_immediate", "creation"}:
            raise RulesetError(
                f"{context}.mechanics.persistent_effect.duration_anchor must be null, 'after_immediate', or 'creation'"
            )
        for name in (
            "radius_mtile",
            "damage_per_tick",
            "crown_damage_per_tick",
            "building_damage_per_tick",
            "knockback_mtile",
            "pull_to_center_mtile",
        ):
            if name in persistent_row:
                _require_int(persistent_row, name, minimum=0)
        if "max_pulses" in persistent_row:
            _require_int(persistent_row, "max_pulses", minimum=1)
        for name in ("damage_schedule", "crown_damage_schedule"):
            if name in persistent_row:
                values = persistent_row[name]
                if (
                    not isinstance(values, list)
                    or not values
                    or any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 0
                        for value in values
                    )
                ):
                    raise RulesetError(
                        f"{context}.mechanics.persistent_effect.{name} must be a non-empty list of non-negative integers"
                    )
        friendly_status = persistent_row.get("friendly_status")
        if friendly_status is not None:
            friendly_row = _as_dict(
                friendly_status,
                f"{context}.mechanics.persistent_effect.friendly_status",
            )
            _reject_unknown(
                friendly_row,
                {
                    "kind",
                    "duration_us",
                    "speed_multiplier_milli",
                    "hit_speed_multiplier_milli",
                    "linger_us",
                },
                f"{context}.mechanics.persistent_effect.friendly_status",
            )
            _require_str(friendly_row, "kind")
            _require_int(friendly_row, "duration_us", minimum=1)
            _require_int(friendly_row, "linger_us", minimum=0)
            for name in ("speed_multiplier_milli", "hit_speed_multiplier_milli"):
                _require_int(friendly_row, name, minimum=0)
                if friendly_row[name] > 2_000:
                    raise RulesetError(
                        f"{context}.mechanics.persistent_effect.friendly_status.{name} must not exceed 2000"
                    )
        friendly_targets = persistent_row.get("friendly_targets")
        if friendly_targets is not None:
            if not isinstance(friendly_targets, list) or not friendly_targets:
                raise RulesetError(
                    f"{context}.mechanics.persistent_effect.friendly_targets must be a non-empty list"
                )
            invalid_friendly_targets = sorted(set(friendly_targets) - _TARGETS)
            if invalid_friendly_targets or any(
                not isinstance(item, str) for item in friendly_targets
            ):
                raise RulesetError(
                    f"invalid friendly persistent effect targets at {context}: {invalid_friendly_targets}"
                )
        for name in ("damage_by_target_count", "crown_damage_by_target_count"):
            if name in persistent_row:
                values = persistent_row[name]
                if not isinstance(values, dict) or any(
                    not isinstance(key, str)
                    or not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for key, value in values.items()
                ):
                    raise RulesetError(
                        f"{context}.mechanics.persistent_effect.{name} must map target buckets to non-negative integers"
                    )
        targets = persistent_row.get("targets", [])
        if not isinstance(targets, list) or not targets:
            raise RulesetError(
                f"{context}.mechanics.persistent_effect.targets must be non-empty"
            )
        invalid_targets = sorted(set(targets) - _TARGETS)
        if invalid_targets or any(not isinstance(item, str) for item in targets):
            raise RulesetError(
                f"invalid persistent effect targets at {context}: {invalid_targets}"
            )
        _validate_status(
            persistent_row.get("status"),
            f"{context}.mechanics.persistent_effect.status",
        )
        spawn_effect = persistent_row.get("spawn")
        if spawn_effect is not None:
            spawn_row = _as_dict(
                spawn_effect,
                f"{context}.mechanics.persistent_effect.spawn",
            )
            _reject_unknown(
                spawn_row,
                {"card_id", "count", "max_spawns", "offsets_mtile"},
                f"{context}.mechanics.persistent_effect.spawn",
            )
            _require_str(spawn_row, "card_id")
            _require_int(spawn_row, "count", minimum=1)
            _require_int(spawn_row, "max_spawns", minimum=1)
            if "offsets_mtile" in spawn_row:
                _validate_offsets(spawn_row["offsets_mtile"], f"{context}.mechanics.persistent_effect.spawn.offsets_mtile")
    death = row["death"]
    if death is not None:
        death_row = _as_dict(death, f"{context}.mechanics.death")
        allowed = {
            "damage",
            "crown_tower_damage",
            "radius_mtile",
            "targets",
            "status",
            "knockback_mtile",
            "spawn_card_id",
            "spawn_count",
            "spawn_offsets_mtile",
            "spawn_children",
            "opponent_elixir_milli",
            "owner_elixir_milli",
            "delay_us",
        }
        unknown_death = sorted(set(death_row) - allowed)
        if unknown_death:
            raise RulesetError(
                f"{context}.mechanics.death has unknown fields: {unknown_death}"
            )
        for name in ("damage", "crown_tower_damage", "radius_mtile", "knockback_mtile"):
            if name in death_row and (
                type(death_row[name]) is not int or death_row[name] < 0
            ):
                raise RulesetError(
                    f"{context}.mechanics.death.{name} must be a non-negative integer"
                )
        if "opponent_elixir_milli" in death_row:
            _require_int(death_row, "opponent_elixir_milli", minimum=1)
        if "owner_elixir_milli" in death_row:
            _require_int(death_row, "owner_elixir_milli", minimum=1)
        if "delay_us" in death_row:
            _require_int(death_row, "delay_us", minimum=1)
        if "spawn_card_id" in death_row:
            _require_str(death_row, "spawn_card_id")
            _require_int(death_row, "spawn_count", minimum=1)
            if "spawn_offsets_mtile" in death_row:
                offsets = death_row["spawn_offsets_mtile"]
                _validate_offsets(offsets, f"{context}.mechanics.death.spawn_offsets_mtile")
                if len(offsets) != int(death_row["spawn_count"]):
                    raise RulesetError(
                        f"{context}.mechanics.death.spawn_offsets_mtile must contain exactly spawn_count entries"
                    )
        elif "spawn_count" in death_row:
            raise RulesetError(
                f"{context}.mechanics.death.spawn_count requires spawn_card_id"
            )
        elif "spawn_offsets_mtile" in death_row:
            raise RulesetError(
                f"{context}.mechanics.death.spawn_offsets_mtile requires spawn_card_id"
            )
        spawn_children = death_row.get("spawn_children")
        if spawn_children is not None:
            if not isinstance(spawn_children, list) or not spawn_children:
                raise RulesetError(
                    f"{context}.mechanics.death.spawn_children must be a non-empty list"
                )
            for index, child in enumerate(spawn_children):
                child_context = f"{context}.mechanics.death.spawn_children[{index}]"
                child_row = _as_dict(child, child_context)
                unknown_child = sorted(set(child_row) - {"card_id", "count", "offsets_mtile"})
                missing_child = sorted({"card_id", "count"} - set(child_row))
                if unknown_child or missing_child:
                    raise RulesetError(
                        f"{child_context} keys mismatch: "
                        f"missing={missing_child}, unknown={unknown_child}"
                    )
                child_id = _require_str(child_row, "card_id")
                if not _CARD_ID_RE.fullmatch(child_id):
                    raise RulesetError(
                        f"{child_context}.card_id is not a valid card ID"
                    )
                _require_int(child_row, "count", minimum=1)
                if "offsets_mtile" in child_row:
                    offsets = child_row["offsets_mtile"]
                    _validate_offsets(offsets, f"{child_context}.offsets_mtile")
                    if len(offsets) != int(child_row["count"]):
                        raise RulesetError(
                            f"{child_context}.offsets_mtile must contain exactly count entries"
                        )
        death_targets = death_row.get("targets")
        if not isinstance(death_targets, list) or not death_targets:
            raise RulesetError(f"{context}.mechanics.death.targets must be non-empty")
        invalid_death_targets = sorted(set(death_targets) - _TARGETS)
        if invalid_death_targets or any(not isinstance(item, str) for item in death_targets):
            raise RulesetError(
                f"invalid death targets at {context}: {invalid_death_targets}"
            )
        _validate_status(death_row.get("status"), f"{context}.mechanics.death.status")
    _validate_status(row["status"], f"{context}.mechanics.status")


def _validate_status(value: Any, context: str) -> None:
    if value is None:
        return
    row = _as_dict(value, context)
    expected = {
        "kind",
        "duration_us",
        "speed_multiplier_milli",
        "hit_speed_multiplier_milli",
    }
    optional = {
        "damage_per_tick",
        "tick_interval_us",
        "on_death_spawn_card_id",
        "on_death_spawn_count",
        # A few projectile/status definitions carry their victim vocabulary
        # alongside the effect (Ram Rider's bola is the current consumer).
        # Keep it optional so ordinary status rows remain compact.
        "targets",
    }
    unknown = sorted(set(row) - expected - optional)
    missing = sorted(expected - set(row))
    if unknown or missing:
        raise RulesetError(
            f"{context} keys mismatch: missing={missing}, unknown={unknown}"
        )
    _require_str(row, "kind")
    _require_int(row, "duration_us", minimum=1)
    for name in ("speed_multiplier_milli", "hit_speed_multiplier_milli"):
        multiplier = _require_int(row, name, minimum=0)
        if multiplier > 1_000:
            raise RulesetError(f"{context}.{name} must not exceed 1000")
    for name in ("damage_per_tick", "tick_interval_us"):
        if name in row:
            _require_int(row, name, minimum=0)
    if int(row.get("damage_per_tick", 0)) and int(row.get("tick_interval_us", 0)) <= 0:
        raise RulesetError(
            f"{context}.tick_interval_us must be positive when damage_per_tick is used"
        )
    if "on_death_spawn_card_id" in row:
        _require_str(row, "on_death_spawn_card_id")
        _require_int(row, "on_death_spawn_count", minimum=1)
    elif "on_death_spawn_count" in row:
        raise RulesetError(
            f"{context}.on_death_spawn_count requires on_death_spawn_card_id"
        )
    if "targets" in row:
        targets = row["targets"]
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(value, str) or value not in _TARGETS for value in targets)
        ):
            raise RulesetError(
                f"{context}.targets must be a non-empty list of valid target classes"
            )


def _parse_tower(tower_id: str, value: Any, sources: Mapping[str, SourceRecord]) -> TowerDefinition:
    context = f"towers.{tower_id}"
    row = _as_dict(value, context)
    expected = {
        "name", "aliases", "hitpoints", "damage", "attack_interval_us", "first_hit_delay_us",
        "range_mtile", "sight_range_mtile", "collision_radius_mtile", "targets", "projectile",
        "activated_at_start", "crown_value", "provenance", "uncertainties",
    }
    _reject_unknown(row, expected, context)
    if not _CARD_ID_RE.fullmatch(tower_id):
        raise RulesetError(f"invalid tower ID: {tower_id!r}")
    targets = tuple(_require_str_list(row, "targets"))
    invalid_targets = sorted(set(targets) - _TARGETS)
    if invalid_targets:
        raise RulesetError(f"invalid targets at {context}: {invalid_targets}")
    projectile = _parse_projectile(row.get("projectile"), f"{context}.projectile")
    if projectile is None:
        raise RulesetError(f"{context}.projectile is required")
    return TowerDefinition(
        tower_id=tower_id,
        name=_require_str(row, "name"),
        aliases=tuple(_require_str_list(row, "aliases")),
        hitpoints=_require_int(row, "hitpoints", minimum=1),
        damage=_require_int(row, "damage", minimum=1),
        attack_interval_us=_require_int(row, "attack_interval_us", minimum=1),
        first_hit_delay_us=_require_int(row, "first_hit_delay_us", minimum=0),
        range_mtile=_require_int(row, "range_mtile", minimum=0),
        sight_range_mtile=_require_int(row, "sight_range_mtile", minimum=0),
        collision_radius_mtile=_require_int(row, "collision_radius_mtile", minimum=0),
        targets=targets,
        projectile=projectile,
        activated_at_start=_require_bool(row, "activated_at_start"),
        crown_value=_require_int(row, "crown_value", minimum=1),
        provenance=MappingProxyType(_parse_provenance(_require_dict(row, "provenance"), sources, context)),
        uncertainties=_parse_uncertainties(row.get("uncertainties", []), context),
    )


def _parse_match(row: dict[str, Any]) -> MatchRules:
    expected = {
        "initial_elixir_milli", "max_elixir_milli", "deck_size", "hand_size", "regulation_us",
        "overtime_us", "normal_elixir_interval_us", "double_elixir_interval_us",
        "triple_elixir_interval_us", "tiebreak_enabled",
    }
    _reject_unknown(row, expected, "match")
    match = MatchRules(
        initial_elixir_milli=_require_int(row, "initial_elixir_milli", minimum=0),
        max_elixir_milli=_require_int(row, "max_elixir_milli", minimum=1),
        deck_size=_require_int(row, "deck_size", minimum=1),
        hand_size=_require_int(row, "hand_size", minimum=1),
        regulation_us=_require_int(row, "regulation_us", minimum=1),
        overtime_us=_require_int(row, "overtime_us", minimum=0),
        normal_elixir_interval_us=_require_int(row, "normal_elixir_interval_us", minimum=1),
        double_elixir_interval_us=_require_int(row, "double_elixir_interval_us", minimum=1),
        triple_elixir_interval_us=_require_int(row, "triple_elixir_interval_us", minimum=1),
        tiebreak_enabled=_require_bool(row, "tiebreak_enabled"),
    )
    if match.initial_elixir_milli > match.max_elixir_milli:
        raise RulesetError("initial elixir exceeds cap")
    if match.hand_size >= match.deck_size:
        raise RulesetError("hand_size must be smaller than deck_size")
    return match


def _parse_arena(row: dict[str, Any]) -> ArenaRules:
    expected = {
        "width_mtile", "height_mtile", "grid_columns", "grid_rows", "river_y_min_mtile",
        "river_y_max_mtile", "bridge_x_ranges_mtile",
    }
    _reject_unknown(row, expected, "arena")
    raw_bridges = row.get("bridge_x_ranges_mtile")
    if not isinstance(raw_bridges, list) or not raw_bridges:
        raise RulesetError("arena.bridge_x_ranges_mtile must be a non-empty array")
    bridges: list[tuple[int, int]] = []
    for index, value in enumerate(raw_bridges):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(part, int) or isinstance(part, bool) for part in value)
            or value[0] < 0
            or value[1] <= value[0]
        ):
            raise RulesetError(f"invalid bridge range at arena.bridge_x_ranges_mtile[{index}]")
        bridges.append((value[0], value[1]))
    arena = ArenaRules(
        width_mtile=_require_int(row, "width_mtile", minimum=1),
        height_mtile=_require_int(row, "height_mtile", minimum=1),
        grid_columns=_require_int(row, "grid_columns", minimum=1),
        grid_rows=_require_int(row, "grid_rows", minimum=1),
        river_y_min_mtile=_require_int(row, "river_y_min_mtile", minimum=0),
        river_y_max_mtile=_require_int(row, "river_y_max_mtile", minimum=1),
        bridge_x_ranges_mtile=tuple(bridges),
    )
    if arena.river_y_max_mtile <= arena.river_y_min_mtile:
        raise RulesetError("river maximum must exceed minimum")
    if arena.river_y_max_mtile > arena.height_mtile:
        raise RulesetError("river lies outside arena")
    if any(end > arena.width_mtile for _, end in bridges):
        raise RulesetError("bridge lies outside arena")
    return arena


def _parse_provenance(
    row: dict[str, Any], sources: Mapping[str, SourceRecord], context: str
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for field_name, value in row.items():
        if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
            raise RulesetError(f"{context}.provenance.{field_name} must be a non-empty string array")
        unknown_sources = sorted(set(value) - set(sources))
        if unknown_sources:
            raise RulesetError(
                f"{context}.provenance.{field_name} references unknown sources: {unknown_sources}"
            )
        result[field_name] = tuple(value)
    if not result:
        raise RulesetError(f"{context}.provenance may not be empty")
    return result


def _parse_uncertainties(value: Any, context: str) -> tuple[Uncertainty, ...]:
    if not isinstance(value, list):
        raise RulesetError(f"{context}.uncertainties must be an array")
    result: list[Uncertainty] = []
    for index, item in enumerate(value):
        row = _as_dict(item, f"{context}.uncertainties[{index}]")
        _reject_unknown(row, {"field", "reason", "impact", "resolution"}, f"{context}.uncertainties[{index}]")
        result.append(
            Uncertainty(
                field=_require_str(row, "field"),
                reason=_require_str(row, "reason"),
                impact=_require_str(row, "impact"),
                resolution=_require_str(row, "resolution"),
            )
        )
    return tuple(result)


def _build_alias_index(definitions: Mapping[str, Any], *, kind: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for definition_id, definition in definitions.items():
        for candidate in (definition_id, definition.name, *definition.aliases):
            alias = normalize_identifier(candidate)
            if not alias:
                raise RulesetError(f"empty {kind} alias on {definition_id}")
            previous = result.get(alias)
            if previous is not None and previous != definition_id:
                raise RulesetError(
                    f"ambiguous {kind} alias {candidate!r}: {previous!r} and {definition_id!r}"
                )
            result[alias] = definition_id
    return result


def _validate_units(row: dict[str, Any]) -> None:
    expected = {"time", "position", "elixir", "health", "damage", "multiplier"}
    _reject_unknown(row, expected, "units")
    required_values = {
        "time": "microseconds",
        "position": "milli-tiles",
        "elixir": "milli-elixir",
        "health": "integer-hitpoints",
        "damage": "integer-hitpoints",
        "multiplier": "permille",
    }
    for key, expected_value in required_values.items():
        actual = _require_str(row, key)
        if actual != expected_value:
            raise RulesetError(f"units.{key} must be {expected_value!r}, got {actual!r}")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _reject_unknown(row: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise RulesetError(f"unknown fields at {context}: {unknown}")
    missing = sorted(allowed - set(row))
    if missing:
        raise RulesetError(f"missing fields at {context}: {missing}")


def _as_dict(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RulesetError(f"{context} must be an object")
    return value


def _validate_offsets(value: Any, context: str) -> None:
    if not isinstance(value, list) or not value:
        raise RulesetError(f"{context} must be a non-empty list")
    if any(
        not isinstance(point, list)
        or len(point) != 2
        or any(type(coordinate) is not int for coordinate in point)
        for point in value
    ):
        raise RulesetError(f"{context} must contain integer [x, y] pairs")


def _require_dict(row: Mapping[str, Any], key: str) -> dict[str, Any]:
    return _as_dict(row.get(key), key)


def _require_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise RulesetError(f"{key} must be a non-empty string")
    return value


def _optional_str(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RulesetError(f"{key} must be null or a non-empty string")
    return value


def _require_bool(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise RulesetError(f"{key} must be boolean")
    return value


def _require_int(row: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RulesetError(f"{key} must be an integer >= {minimum}")
    return value


def _optional_int(row: Mapping[str, Any], key: str, *, minimum: int) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RulesetError(f"{key} must be null or an integer >= {minimum}")
    return value


def _require_str_list(row: Mapping[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise RulesetError(f"{key} must be an array of non-empty strings")
    return value
