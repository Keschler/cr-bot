"""Explicit, fail-closed scopes for simulator training and evaluation.

The full V1 ruleset is intentionally broader than the first useful Hog
training experiment.  A :class:`TrainingProfile` records the exact card and
mechanic scope of a run, while :func:`validate_training_profile` requires a
matching readiness artifact for serious training.  Provisional smoke runs
remain possible only when the caller explicitly selects the smoke purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping

from .engine import ENGINE_VERSION
from .roster import PLAYER_DECK
from .ruleset import Ruleset, load_ruleset


PROFILE_SCHEMA_VERSION = 1
TrainingPurpose = Literal["smoke", "training", "evaluation"]


class TrainingProfileError(ValueError):
    """Raised when a training scope cannot be proven compatible and ready."""


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    """Immutable scope and evidence identity for one learner run."""

    profile_id: str
    ruleset_id: str
    player_deck: tuple[str, ...] = PLAYER_DECK
    opponent_decks: tuple[tuple[str, ...], ...] = ()
    required_cards: tuple[str, ...] = ()
    required_mechanics: tuple[str, ...] = ()
    purpose: TrainingPurpose = "training"
    readiness_report: Path | None = None

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise TrainingProfileError("profile_id must not be empty")
        if not self.ruleset_id.strip():
            raise TrainingProfileError("ruleset_id must not be empty")
        if self.purpose not in {"smoke", "training", "evaluation"}:
            raise TrainingProfileError(f"unsupported training purpose: {self.purpose!r}")
        if len(self.player_deck) != 8:
            raise TrainingProfileError("player_deck must contain exactly eight cards")
        for deck_index, deck in enumerate(self.opponent_decks):
            if len(deck) != 8:
                raise TrainingProfileError(
                    f"opponent_decks[{deck_index}] must contain exactly eight cards"
                )
        for field_name in ("required_cards", "required_mechanics"):
            values = getattr(self, field_name)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise TrainingProfileError(f"{field_name} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise TrainingProfileError(f"{field_name} must not contain duplicates")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base_dir: Path | None = None) -> "TrainingProfile":
        """Parse a JSON-compatible profile mapping without loading a ruleset."""

        if not isinstance(raw, Mapping):
            raise TrainingProfileError("training profile must be an object")
        schema_version = raw.get("schema_version", PROFILE_SCHEMA_VERSION)
        if schema_version != PROFILE_SCHEMA_VERSION:
            raise TrainingProfileError(f"unsupported training profile schema: {schema_version!r}")

        def strings(field_name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
            value = raw.get(field_name, list(default))
            if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
                raise TrainingProfileError(f"{field_name} must be a list of strings")
            return tuple(value)

        raw_opponent_decks = raw.get("opponent_decks", [])
        if not isinstance(raw_opponent_decks, (list, tuple)):
            raise TrainingProfileError("opponent_decks must be a list of eight-card lists")
        opponent_decks: list[tuple[str, ...]] = []
        for index, deck in enumerate(raw_opponent_decks):
            if not isinstance(deck, (list, tuple)) or any(not isinstance(item, str) for item in deck):
                raise TrainingProfileError(f"opponent_decks[{index}] must be a list of strings")
            opponent_decks.append(tuple(deck))

        readiness_raw = raw.get("readiness_report")
        readiness_report = None
        if readiness_raw is not None:
            if not isinstance(readiness_raw, str) or not readiness_raw.strip():
                raise TrainingProfileError("readiness_report must be a non-empty path")
            readiness_report = Path(readiness_raw)
            if base_dir is not None and not readiness_report.is_absolute():
                readiness_report = base_dir / readiness_report

        return cls(
            profile_id=str(raw.get("profile_id", "")),
            ruleset_id=str(raw.get("ruleset_id", "")),
            player_deck=strings("player_deck", PLAYER_DECK),
            opponent_decks=tuple(opponent_decks),
            required_cards=strings("required_cards"),
            required_mechanics=strings("required_mechanics"),
            purpose=raw.get("purpose", "training"),
            readiness_report=readiness_report,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingProfile":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingProfileError(f"cannot load training profile {source}: {error}") from error
        return cls.from_mapping(raw, base_dir=source.parent)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "ruleset_id": self.ruleset_id,
            "player_deck": list(self.player_deck),
            "opponent_decks": [list(deck) for deck in self.opponent_decks],
            "required_cards": list(self.required_cards),
            "required_mechanics": list(self.required_mechanics),
            "purpose": self.purpose,
            "readiness_report": None if self.readiness_report is None else str(self.readiness_report),
        }


def _readiness_report(profile: TrainingProfile) -> dict[str, Any]:
    if profile.readiness_report is None:
        raise TrainingProfileError(
            f"profile {profile.profile_id!r} requires a readiness_report for {profile.purpose}"
        )
    try:
        raw = json.loads(profile.readiness_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingProfileError(
            f"cannot load readiness report {profile.readiness_report}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise TrainingProfileError("readiness report must contain an object")
    return raw


def validate_training_profile(
    profile: TrainingProfile,
    *,
    ruleset: Ruleset | str | None = None,
) -> dict[str, object]:
    """Validate deck scope, ruleset identity, and readiness for a run.

    ``smoke`` profiles only validate executable scope and may use provisional
    rulesets. ``training`` and ``evaluation`` profiles require a matching
    scoped readiness report whose declared card/mechanic rows all pass.
    """

    loaded = load_ruleset(profile.ruleset_id) if ruleset is None or isinstance(ruleset, str) else ruleset
    if loaded.ruleset_id != profile.ruleset_id:
        raise TrainingProfileError(
            f"profile ruleset {profile.ruleset_id!r} does not match loaded {loaded.ruleset_id!r}"
        )
    if tuple(profile.player_deck) != tuple(PLAYER_DECK):
        raise TrainingProfileError("V1 profiles must use the fixed Hog-cycle player deck")

    declared_cards = set(loaded.cards)
    decks = (profile.player_deck, *profile.opponent_decks)
    unknown_deck_cards = sorted({card for deck in decks for card in deck if card not in declared_cards})
    if unknown_deck_cards:
        raise TrainingProfileError(f"profile references unknown ruleset cards: {unknown_deck_cards}")

    required_cards = set(profile.required_cards)
    missing_required_cards = sorted(required_cards - declared_cards)
    if missing_required_cards:
        raise TrainingProfileError(f"profile required_cards are absent from ruleset: {missing_required_cards}")

    result: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": profile.profile_id,
        "purpose": profile.purpose,
        "ruleset_id": loaded.ruleset_id,
        "ruleset_hash": loaded.content_hash,
        "engine_version": ENGINE_VERSION,
        "training_ready": False,
        "required_cards": sorted(required_cards),
        "required_mechanics": sorted(profile.required_mechanics),
    }
    if profile.purpose == "smoke":
        result["training_ready"] = True
        result["readiness_source"] = "explicit_provisional_smoke"
        return result

    if not profile.required_cards and not profile.required_mechanics:
        raise TrainingProfileError(
            "serious profiles must declare at least one required card or mechanic"
        )

    report = _readiness_report(profile)
    identity = (
        report.get("ruleset_id"),
        report.get("ruleset_hash"),
        report.get("engine_version"),
    )
    expected = (loaded.ruleset_id, loaded.content_hash, ENGINE_VERSION)
    if identity != expected:
        raise TrainingProfileError(
            f"readiness identity {identity!r} does not match expected {expected!r}"
        )
    if report.get("profile_id") not in {None, profile.profile_id}:
        raise TrainingProfileError("readiness report belongs to a different training profile")
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        raise TrainingProfileError("readiness report summary must be an object")
    if summary.get("ready") is not True and report.get("ready") is not True:
        raise TrainingProfileError("readiness report is not ready")

    mechanics = report.get("mechanics", {})
    if not isinstance(mechanics, Mapping):
        raise TrainingProfileError("readiness report mechanics must be an object")
    failed_mechanics = []
    for mechanic in profile.required_mechanics:
        row = mechanics.get(mechanic, {})
        status = row.get("status") if isinstance(row, Mapping) else None
        if status not in {"heldout_validated", "ready"}:
            failed_mechanics.append(mechanic)
    if failed_mechanics:
        raise TrainingProfileError(f"required mechanics are not ready: {failed_mechanics}")

    cards = report.get("cards", {})
    if isinstance(cards, Mapping):
        failed_cards = []
        for card in required_cards:
            row = cards.get(card, {})
            status = row.get("status") if isinstance(row, Mapping) else None
            if status not in {"fidelity_ready", "ready"}:
                failed_cards.append(card)
        if failed_cards:
            raise TrainingProfileError(f"required cards are not ready: {failed_cards}")
    elif required_cards:
        raise TrainingProfileError("readiness report has no card readiness rows")

    result["training_ready"] = True
    result["readiness_source"] = str(profile.readiness_report)
    return result


__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "TrainingProfile",
    "TrainingProfileError",
    "validate_training_profile",
]
