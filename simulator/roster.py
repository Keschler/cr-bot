"""Versioned V1 card-roster and card-to-mechanic coverage contracts.

The roster is deliberately separate from the Level-11 balance ruleset. A card
can be eligible for opponent play before its complete mechanics are implemented;
the coverage report then keeps the simulator fail-closed instead of silently
dropping that card from training.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Mapping


ROSTER_SCHEMA_VERSION = 1
OPPONENT_RELEASE_CUTOFF = date(2025, 12, 1)
EVOLUTION_CUTOFF = date(2023, 6, 19)
PLAYER_DECK: tuple[str, ...] = (
    "hog-rider",
    "musketeer",
    "ice-golem",
    "ice-spirit",
    "cannon",
    "skeletons",
    "fireball",
    "log",
)


class RosterError(ValueError):
    """Raised when the roster cannot safely describe the V1 scope."""


@dataclass(frozen=True, slots=True)
class OpponentRoster:
    roster_id: str
    ruleset_id: str
    release_cutoff_exclusive: date
    eligible_cards: tuple[str, ...]
    excluded_cards: Mapping[str, str]
    catalog_source: str
    release_date_status: str

    @property
    def all_classified_cards(self) -> frozenset[str]:
        return frozenset(self.eligible_cards) | frozenset(self.excluded_cards)


def roster_path() -> Path:
    return Path(__file__).with_name("rosters") / "opponent-base-pre-2025-12-01.json"


def load_opponent_roster(path: Path | None = None) -> OpponentRoster:
    path = path or roster_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RosterError(f"cannot load opponent roster {path}: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != ROSTER_SCHEMA_VERSION:
        raise RosterError(f"roster {path} must use schema_version {ROSTER_SCHEMA_VERSION}")
    try:
        cutoff = date.fromisoformat(str(raw["release_cutoff_exclusive"]))
        eligible = tuple(str(card) for card in raw["eligible_cards"])
        excluded = {str(card): str(reason) for card, reason in raw["excluded_cards"].items()}
    except (KeyError, TypeError, ValueError) as error:
        raise RosterError(f"malformed roster {path}: {error}") from error
    if cutoff != OPPONENT_RELEASE_CUTOFF:
        raise RosterError(
            f"roster cutoff {cutoff.isoformat()} differs from V1 {OPPONENT_RELEASE_CUTOFF.isoformat()}"
        )
    if not eligible or len(set(eligible)) != len(eligible):
        raise RosterError("eligible_cards must be non-empty and unique")
    if set(eligible) & set(excluded):
        raise RosterError("a card cannot be both eligible and excluded")
    if any(not card or card != card.strip() for card in (*eligible, *excluded)):
        raise RosterError("card IDs must be non-empty canonical strings")
    return OpponentRoster(
        roster_id=str(raw["roster_id"]),
        ruleset_id=str(raw["ruleset_id"]),
        release_cutoff_exclusive=cutoff,
        eligible_cards=eligible,
        excluded_cards=excluded,
        catalog_source=str(raw["catalog_source"]),
        release_date_status=str(raw["release_date_status"]),
    )


def validate_roster_against_catalog(
    roster: OpponentRoster,
    catalog: Mapping[str, Any],
    *,
    require_release_verification: bool = False,
) -> dict[str, Any]:
    """Return a fail-closed catalog classification report.

    ``require_release_verification`` is used by release/readiness CI. The
    checked-in V1 catalog currently records a conservative pre-cutoff
    classification source, but a final held-out release must replace that
    source with exact, independently reproducible release-date evidence.
    """

    catalog_cards = frozenset(catalog)
    classified = roster.all_classified_cards
    missing = sorted(catalog_cards - classified)
    unknown = sorted(classified - catalog_cards)
    duplicate_player_cards = sorted(set(PLAYER_DECK) - catalog_cards)
    release_unverified = []
    if require_release_verification and roster.release_date_status != "exact":
        release_unverified = list(roster.eligible_cards)
    return {
        "schema_version": ROSTER_SCHEMA_VERSION,
        "roster_id": roster.roster_id,
        "catalog_count": len(catalog_cards),
        "eligible_count": len(roster.eligible_cards),
        "excluded_count": len(roster.excluded_cards),
        "missing_catalog_classification": missing,
        "unknown_roster_cards": unknown,
        "missing_player_cards": duplicate_player_cards,
        "release_date_unverified": release_unverified,
        "complete": not (missing or unknown or duplicate_player_cards or release_unverified),
    }


def build_mechanic_coverage(
    roster: OpponentRoster,
    card_definitions: Mapping[str, Mapping[str, Any]],
    implemented_cards: set[str],
    *,
    fidelity_ready_cards: set[str] | None = None,
) -> dict[str, Any]:
    """Build executable and fidelity-ready card-to-mechanic coverage.

    ``implemented_cards`` answers whether the engine can instantiate and
    advance a card.  ``fidelity_ready_cards`` is deliberately separate:
    generated/provisional definitions can be executable while still being
    forbidden from training.  Existing callers which do not provide the
    optional set retain the historical executable-only behavior.
    """

    rows: list[dict[str, Any]] = []
    for card_id in roster.eligible_cards:
        definition = card_definitions.get(card_id, {})
        kind = str(definition.get("kind", "unknown"))
        required = ["card_identity", "deployment", "target_legality", "lifecycle"]
        if kind == "troop":
            required.extend(("movement", "target_acquisition", "attack", "damage", "death"))
            if definition.get("is_air"):
                required.append("air_navigation")
            if definition.get("is_splash"):
                required.append("area_damage")
        elif kind == "building":
            required.extend(("building_navigation", "target_acquisition", "attack", "lifetime"))
        elif kind == "spell":
            required.extend(("spell_geometry", "effect_timing", "victim_selection"))
        else:
            required.append("kind_definition")
        # Preserve component-specific coverage in the roster report as well as
        # in the generated-scenario manifest.  Callers that provide only the
        # historical ``kind`` shorthand retain the old contract; the CLI
        # passes the authoritative mechanics mapping so charge/chain/reflection
        # consumers cannot disappear from readiness output.
        mechanics = definition.get("mechanics")
        if isinstance(mechanics, Mapping):
            if mechanics.get("charge_attack") is not None:
                required.extend(("charge_attack", "charge_movement"))
            if mechanics.get("chain_attack") is not None:
                required.append("chain_targeting")
            if mechanics.get("multi_target_attack") is not None:
                required.append("multi_targeting")
            if mechanics.get("reflection") is not None:
                required.append("reflected_damage")
            if mechanics.get("persistent_effect") is not None:
                required.append("persistent_area_effect")
            if mechanics.get("dash") is not None:
                required.extend(("dash_movement", "dash_attack"))
            if mechanics.get("hook") is not None:
                required.extend(("hook_targeting", "hook_pull"))
            if mechanics.get("recoil_mtile") is not None:
                required.append("recoil")
            if mechanics.get("ramp_attack") is not None:
                required.extend(("ramp_attack", "ramp_reset"))
            if mechanics.get("revive") is not None:
                required.extend(("revive", "revive_egg"))
        executable = card_id in implemented_cards
        fidelity_ready = executable and (
            fidelity_ready_cards is None or card_id in fidelity_ready_cards
        )
        rows.append(
            {
                "card_id": card_id,
                "kind": kind,
                "implemented": executable,
                "fidelity_ready": fidelity_ready,
                "required_mechanics": sorted(set(required)),
                "status": (
                    "fidelity_ready"
                    if fidelity_ready
                    else "implemented_provisional"
                    if executable
                    else "missing"
                ),
            }
        )
    missing = [row["card_id"] for row in rows if not row["implemented"]]
    not_ready = [row["card_id"] for row in rows if not row["fidelity_ready"]]
    return {
        "schema_version": ROSTER_SCHEMA_VERSION,
        "roster_id": roster.roster_id,
        "card_count": len(rows),
        "implemented_card_count": len(rows) - len(missing),
        "missing_card_count": len(missing),
        "all_cards_implemented": not missing,
        "fidelity_ready_card_count": len(rows) - len(not_ready),
        "fidelity_not_ready_cards": not_ready,
        "all_cards_fidelity_ready": not not_ready,
        "cards": rows,
    }


__all__ = [
    "EVOLUTION_CUTOFF",
    "OPPONENT_RELEASE_CUTOFF",
    "OpponentRoster",
    "PLAYER_DECK",
    "ROSTER_SCHEMA_VERSION",
    "RosterError",
    "build_mechanic_coverage",
    "load_opponent_roster",
    "roster_path",
    "validate_roster_against_catalog",
]
