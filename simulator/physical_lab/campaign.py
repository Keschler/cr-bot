"""Versioned interaction campaigns for the physical-fidelity lab.

Campaign files are the immutable bridge between a planned physical sweep and
the artifacts produced by each phone run.  A campaign case contains the exact
deck order, opening hand, logical actions, and measurements that must be used
for both the physical run and its simulator replay.  Results are deliberately
stored outside the campaign manifest so that a captured case can never be
silently rewritten when the simulator changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..engine import ENGINE_VERSION, BattleEngine
from ..ruleset import load_fixed_ruleset, load_ruleset
from ..validation import apply_fidelity_gate, run_fidelity_corpus
from .planner import hog_cannon_probe
from .schema import (
    EvidenceSplit,
    ExperimentSpec,
    InitialConditions,
    MeasurementSpec,
    PhysicalAction,
    PhysicalLabError,
    Trigger,
    TriggerType,
    canonical_hash,
)


CAMPAIGN_SCHEMA_VERSION = 1
CAMPAIGN_LEVELS = (
    "isolated",
    "single_troop",
    "paired_interaction",
    "multi_card",
    "complex",
)
DEFAULT_CAMPAIGN_ID = "physical-fidelity-interaction-sweep-v5"
DEFAULT_CAMPAIGN_ROOT = Path("outputs/simulator/fidelity_media/physical_lab/campaigns")

# Phone A is the ASUS AI2302. Its first three human deck slots turn Musketeer
# into the Hero variant, so the regular Musketeer is placed in human slot 4.
PHONE_A_REGULAR_MUSKETEER_DECK: tuple[str, ...] = (
    "hog-rider",
    "cannon",
    "skeletons",
    "musketeer",
    "ice-golem",
    "ice-spirit",
    "fireball",
    "log",
)
PHONE_A_CARD_CONSTRAINTS: Mapping[str, Any] = {
    "model_ids": ["ASUS_AI2302"],
    "restricted_cards": ["archers", "musketeer"],
    "restricted_human_slots": [4, 8],
    "restricted_zero_based_min_slot": 3,
    "reason": "human slots 1-3 activate special Hero/Evolution variants on Phone A",
}

# Phone B is the Samsung SM-G970F/beyond0. Its first three human deck slots
# activate Hero/Evolution variants for these five cards, so all five are
# deliberately placed in human slots 4-8. This is a physical-device
# constraint, not a simulator card rule.
PHONE_B_REGULAR_MUSKETEER_DECK: tuple[str, ...] = (
    "hog-rider",
    "fireball",
    "log",
    "cannon",
    "skeletons",
    "musketeer",
    "ice-golem",
    "ice-spirit",
)
PHONE_B_CARD_CONSTRAINTS: Mapping[str, Any] = {
    "model_ids": ["SM-G970F", "beyond0"],
    "restricted_cards": ["cannon", "skeletons", "musketeer", "ice-golem", "ice-spirit"],
    "restricted_human_slots": [4, 8],
    "restricted_zero_based_min_slot": 3,
    "reason": "human slots 1-3 activate Hero/Evolution variants on Phone B",
}


def _name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhysicalLabError(f"{field_name} must be a non-empty string")
    return value.strip()


def _identifier(value: object, field_name: str) -> str:
    value = _name(value, field_name)
    if not all(character.isalnum() or character in "_.:-" for character in value):
        raise PhysicalLabError(f"{field_name} contains unsupported characters: {value!r}")
    return value


def _copy_json(value: object, field_name: str) -> Any:
    try:
        return json.loads(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise PhysicalLabError(f"{field_name} must be JSON-compatible: {error}") from error


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create an artifact once; refuse to overwrite a different artifact."""

    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as error:
            raise PhysicalLabError(f"cannot read existing immutable artifact {path}: {error}") from error
        if existing != encoded:
            raise PhysicalLabError(f"immutable campaign artifact already exists with different contents: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _validate_card(card_id: str) -> str:
    card_id = _identifier(card_id, "card_id").lower()
    ruleset = load_fixed_ruleset()
    if card_id not in ruleset.cards:
        raise PhysicalLabError(f"card_id is not present in the fixed ruleset: {card_id!r}")
    return card_id


def _validate_deck(deck: Sequence[str], field_name: str = "deck") -> tuple[str, ...]:
    if not isinstance(deck, (list, tuple)) or len(deck) != 8:
        raise PhysicalLabError(f"{field_name} must contain exactly eight cards")
    parsed = tuple(_validate_card(card) for card in deck)
    if len(set(parsed)) != 8:
        raise PhysicalLabError(f"{field_name} must contain eight unique cards")
    return parsed


@dataclass(frozen=True, slots=True)
class DeckMutation:
    """One deterministic replacement in a side's ordered eight-card deck."""

    side: str
    slot: int
    card_id: str
    reason: str = "interaction variant"

    def __post_init__(self) -> None:
        side = _name(self.side, "mutation.side").upper()
        if side not in {"A", "B"}:
            raise PhysicalLabError("mutation.side must be A or B")
        object.__setattr__(self, "side", side)
        if type(self.slot) is not int or not 0 <= self.slot < 8:
            raise PhysicalLabError("mutation.slot must be an integer between 0 and 7")
        object.__setattr__(self, "card_id", _validate_card(self.card_id))
        object.__setattr__(self, "reason", _name(self.reason, "mutation.reason"))

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "slot": self.slot,
            "card_id": self.card_id,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DeckMutation":
        if not isinstance(raw, Mapping):
            raise PhysicalLabError("deck mutation must be an object")
        allowed = {"side", "slot", "card_id", "reason"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PhysicalLabError(f"unknown deck mutation fields: {unknown}")
        return cls(
            side=raw.get("side"),
            slot=raw.get("slot"),
            card_id=raw.get("card_id"),
            reason=raw.get("reason", "interaction variant"),
        )


def apply_deck_mutations(
    base_decks: Mapping[str, Sequence[str]],
    mutations: Sequence[DeckMutation],
) -> dict[str, tuple[str, ...]]:
    """Apply mutations and validate both resulting decks and opening hands."""

    decks = {
        side: list(_validate_deck(base_decks.get(side, ()), f"base_decks.{side}"))
        for side in ("A", "B")
    }
    occupied: set[tuple[str, int]] = set()
    for mutation in mutations:
        key = (mutation.side, mutation.slot)
        if key in occupied:
            raise PhysicalLabError(f"duplicate mutation for {mutation.side} slot {mutation.slot}")
        occupied.add(key)
        decks[mutation.side][mutation.slot] = mutation.card_id
    return {
        side: _validate_deck(deck, f"resulting_decks.{side}")
        for side, deck in decks.items()
    }


def _opening_slots(decks: Mapping[str, Sequence[str]]) -> dict[str, dict[str, int]]:
    return {
        side: {card_id: slot for slot, card_id in enumerate(tuple(deck)[:4])}
        for side, deck in decks.items()
    }


@dataclass(frozen=True, slots=True)
class CampaignCase:
    """One immutable physical interaction case."""

    case_id: str
    level: str
    description: str
    base_decks: Mapping[str, tuple[str, ...]]
    mutations: tuple[DeckMutation, ...]
    spec: ExperimentSpec
    expected_mechanics: tuple[str, ...] = ()
    result_relpath: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        level = _name(self.level, "level").lower()
        if level not in CAMPAIGN_LEVELS:
            raise PhysicalLabError(f"unsupported campaign level: {level!r}")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "description", _name(self.description, "description"))
        parsed_base_decks = {
            side: _validate_deck(self.base_decks.get(side, ()), f"base_decks.{side}")
            for side in ("A", "B")
        }
        object.__setattr__(self, "base_decks", parsed_base_decks)
        object.__setattr__(self, "mutations", tuple(self.mutations))
        if any(not isinstance(item, DeckMutation) for item in self.mutations):
            raise PhysicalLabError("mutations must contain DeckMutation records")
        if not isinstance(self.spec, ExperimentSpec):
            raise PhysicalLabError("spec must be an ExperimentSpec")
        if self.spec.experiment_id != self.case_id:
            raise PhysicalLabError("campaign case_id must match spec.experiment_id")
        object.__setattr__(self, "expected_mechanics", tuple(_identifier(item, "expected_mechanic") for item in self.expected_mechanics))
        if self.result_relpath is None:
            object.__setattr__(self, "result_relpath", f"{self.case_id}/fidelity-corpus.json")
        else:
            result_relpath = _name(self.result_relpath, "result_relpath")
            relative = Path(result_relpath)
            if relative.is_absolute() or ".." in relative.parts:
                raise PhysicalLabError("result_relpath must remain relative to the campaign result root")
            object.__setattr__(self, "result_relpath", result_relpath)

        expected_decks = apply_deck_mutations(self.base_decks, self.mutations)
        for side in ("A", "B"):
            deck = self.spec.initial_conditions.decks.get(side)
            if deck is None or tuple(deck) != expected_decks[side]:
                raise PhysicalLabError(f"campaign case spec lacks a valid {side} deck")
            for action in self.spec.actions:
                if action.side != side:
                    continue
                if action.card_id not in deck:
                    raise PhysicalLabError(
                        f"case action {action.action_id!r} card is absent from side {side} deck"
                    )
                if action.card_slot is not None and deck[action.card_slot] != action.card_id:
                    raise PhysicalLabError(
                        f"case action {action.action_id!r} card_slot does not match side {side} deck"
                    )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "case_id": self.case_id,
            "level": self.level,
            "description": self.description,
            "base_decks": {side: list(deck) for side, deck in self.base_decks.items()},
            "mutations": [item.to_dict() for item in self.mutations],
            "spec": self.spec.to_dict(include_hash=True),
            "opening_hand": {
                side: list(self.spec.initial_conditions.decks[side][:4])
                for side in ("A", "B")
            },
            "expected_mechanics": list(self.expected_mechanics),
            "result_relpath": self.result_relpath,
        }
        if include_hash:
            payload["case_hash"] = canonical_hash(payload)
        return payload

    def case_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def save(self, path: str | Path) -> None:
        _write_immutable_json(Path(path), self.to_dict(include_hash=True))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CampaignCase":
        if not isinstance(raw, Mapping):
            raise PhysicalLabError("campaign case must be an object")
        allowed = {
            "schema_version",
            "case_id",
            "level",
            "description",
            "base_decks",
            "mutations",
            "spec",
            "opening_hand",
            "expected_mechanics",
            "result_relpath",
            "case_hash",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PhysicalLabError(f"unknown campaign case fields: {unknown}")
        if raw.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
            raise PhysicalLabError("unsupported campaign case schema")
        mutations_raw = raw.get("mutations", [])
        if not isinstance(mutations_raw, list):
            raise PhysicalLabError("campaign case mutations must be an array")
        base_decks_raw = raw.get("base_decks")
        if not isinstance(base_decks_raw, Mapping):
            raise PhysicalLabError("campaign case base_decks must be an object")
        spec_raw = raw.get("spec")
        if not isinstance(spec_raw, Mapping):
            raise PhysicalLabError("campaign case spec must be an object")
        expected_raw = raw.get("expected_mechanics", [])
        if not isinstance(expected_raw, list):
            raise PhysicalLabError("campaign case expected_mechanics must be an array")
        case = cls(
            case_id=raw.get("case_id"),
            level=raw.get("level"),
            description=raw.get("description"),
            base_decks={
                str(side).upper(): tuple(deck)
                for side, deck in base_decks_raw.items()
            },
            mutations=tuple(DeckMutation.from_dict(item) for item in mutations_raw),
            spec=ExperimentSpec.from_dict(spec_raw),
            expected_mechanics=tuple(expected_raw),
            result_relpath=raw.get("result_relpath"),
        )
        declared_hash = raw.get("case_hash")
        if declared_hash != case.case_hash():
            raise PhysicalLabError(
                f"campaign case hash mismatch: declared={declared_hash!r}, actual={case.case_hash()!r}"
            )
        opening_hand = raw.get("opening_hand")
        if opening_hand != {
            side: list(case.spec.initial_conditions.decks[side][:4]) for side in ("A", "B")
        }:
            raise PhysicalLabError("campaign case opening_hand does not match its deck order")
        return case


@dataclass(frozen=True, slots=True)
class InteractionCampaign:
    """An ordered simple-to-complex campaign and its canonical hash."""

    campaign_id: str
    cases: tuple[CampaignCase, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_SCHEMA_VERSION:
            raise PhysicalLabError("unsupported interaction campaign schema")
        object.__setattr__(self, "campaign_id", _identifier(self.campaign_id, "campaign_id"))
        cases = tuple(self.cases)
        if not cases:
            raise PhysicalLabError("interaction campaign must contain at least one case")
        if any(not isinstance(case, CampaignCase) for case in cases):
            raise PhysicalLabError("campaign cases must contain CampaignCase records")
        if len({case.case_id for case in cases}) != len(cases):
            raise PhysicalLabError("campaign case IDs must be unique")
        if tuple(sorted(cases, key=lambda item: (CAMPAIGN_LEVELS.index(item.level), item.case_id))) != cases:
            raise PhysicalLabError("campaign cases must be ordered from simple to complex")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "metadata", _copy_json(self.metadata or {}, "metadata"))

    @property
    def ruleset_id(self) -> str:
        return self.cases[0].spec.ruleset_id

    @property
    def ruleset_hash(self) -> str:
        return self.cases[0].spec.ruleset_hash

    @property
    def engine_version(self) -> str:
        return self.cases[0].spec.engine_version

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": "physical_lab_interaction_campaign",
            "campaign_id": self.campaign_id,
            "ruleset_id": self.ruleset_id,
            "ruleset_hash": self.ruleset_hash,
            "engine_version": self.engine_version,
            "metadata": self.metadata,
            "cases": [case.to_dict(include_hash=True) for case in self.cases],
        }
        if include_hash:
            payload["campaign_hash"] = canonical_hash(payload)
        return payload

    def campaign_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def save(self, path: str | Path) -> None:
        _write_immutable_json(Path(path), self.to_dict(include_hash=True))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InteractionCampaign":
        if not isinstance(raw, Mapping):
            raise PhysicalLabError("interaction campaign must be an object")
        allowed = {
            "schema_version",
            "kind",
            "campaign_id",
            "ruleset_id",
            "ruleset_hash",
            "engine_version",
            "metadata",
            "cases",
            "campaign_hash",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PhysicalLabError(f"unknown campaign fields: {unknown}")
        if raw.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
            raise PhysicalLabError("unsupported interaction campaign schema")
        if raw.get("kind") != "physical_lab_interaction_campaign":
            raise PhysicalLabError("unsupported interaction campaign kind")
        cases_raw = raw.get("cases")
        if not isinstance(cases_raw, list):
            raise PhysicalLabError("interaction campaign cases must be an array")
        campaign = cls(
            campaign_id=raw.get("campaign_id"),
            cases=tuple(CampaignCase.from_dict(item) for item in cases_raw),
            metadata=raw.get("metadata", {}),
        )
        if raw.get("ruleset_id") != campaign.ruleset_id or raw.get("ruleset_hash") != campaign.ruleset_hash:
            raise PhysicalLabError("campaign ruleset metadata does not match its cases")
        if raw.get("engine_version") != campaign.engine_version:
            raise PhysicalLabError("campaign engine metadata does not match its cases")
        if raw.get("campaign_hash") != campaign.campaign_hash():
            raise PhysicalLabError("interaction campaign hash does not match its contents")
        return campaign


def load_campaign(path: str | Path) -> InteractionCampaign:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhysicalLabError(f"cannot load interaction campaign {source}: {error}") from error
    return InteractionCampaign.from_dict(raw)


def materialize_campaign(campaign: InteractionCampaign, root: str | Path) -> dict[str, object]:
    """Write the campaign and per-case immutable artifacts under ``root``."""

    destination = Path(root).resolve() / campaign.campaign_id
    manifest_path = destination / "campaign.json"
    campaign.save(manifest_path)
    cases: dict[str, str] = {}
    for case in campaign.cases:
        case_path = destination / "cases" / case.case_id / "case.json"
        case.save(case_path)
        cases[case.case_id] = str(case_path)
    return {
        "campaign_id": campaign.campaign_id,
        "campaign_hash": campaign.campaign_hash(),
        "manifest_path": str(manifest_path),
        "case_paths": cases,
    }


def _case_spec(
    *,
    case_id: str,
    level: str,
    description: str,
    decks: Mapping[str, Sequence[str]],
    actions: Sequence[PhysicalAction],
    measurements: Sequence[MeasurementSpec],
    metadata: Mapping[str, Any],
    evidence_split: EvidenceSplit = EvidenceSplit.CALIBRATION,
) -> ExperimentSpec:
    template = hog_cannon_probe(
        experiment_id=case_id,
        evidence_split=evidence_split,
        metadata={"campaign_level": level, "campaign_description": description, **dict(metadata)},
    )
    parsed_decks = {side: _validate_deck(decks[side], f"decks.{side}") for side in ("A", "B")}
    return replace(
        template,
        initial_conditions=InitialConditions(
            tower_state="default",
            requested_elixir_milli={"A": 10_000, "B": 10_000},
            decks=parsed_decks,
            hand_slots=_opening_slots(parsed_decks),
        ),
        actions=tuple(actions),
        measurements=tuple(measurements),
    )


def _action(
    action_id: str,
    side: str,
    card_id: str,
    cell: tuple[int, int],
    slot: int,
    match_time_us: int,
) -> PhysicalAction:
    return PhysicalAction(
        action_id=action_id,
        side=side,
        card_id=card_id,
        arena_cell=cell,
        card_slot=slot,
        trigger=Trigger(TriggerType.MATCH_TIME_US, value=match_time_us),
    )


def build_default_campaign(
    *,
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    evidence_split: EvidenceSplit = EvidenceSplit.CALIBRATION,
) -> InteractionCampaign:
    """Build the deterministic first sweep used by the physical operators.

    The first two cases isolate one troop.  Later cases add a defender and
    then multiple cards.  Every case keeps the cards used by that case in the
    first four ordered deck slots so Fixed Deck Order makes the opening hand
    explicit on the host (Phone B).
    """

    base = {"A": PHONE_A_REGULAR_MUSKETEER_DECK, "B": PHONE_B_REGULAR_MUSKETEER_DECK}
    cases: list[CampaignCase] = []

    definitions = [
        (
            "isolated-hog",
            "isolated",
            "One Hog Rider deployment with no configured defender.",
            (),
            (
                _action("deploy-hog", "A", "hog-rider", (3, 20), 0, 0),
            ),
            (MeasurementSpec("hog_isolated_movement"), MeasurementSpec("hog_tower_contact")),
            ("hog-rider",),
        ),
        (
            "isolated-archers",
            "single_troop",
            "Place regular Archers in safe human slot 4 and measure one deployment.",
            (
                DeckMutation(
                    "A",
                    3,
                    "archers",
                    "keep an evolvable troop out of the first three device slots",
                ),
            ),
            (
                _action("deploy-archers", "A", "archers", (4, 20), 3, 0),
            ),
            (MeasurementSpec("archers_first_target"), MeasurementSpec("archers_projectile_timing")),
            ("archers",),
        ),
        (
            "hog-cannon-pull",
            "paired_interaction",
            "Canonical Hog bridge crossing followed by a Cannon pull.",
            (),
            (
                PhysicalAction(
                    action_id="deploy-hog",
                    side="A",
                    card_id="hog-rider",
                    arena_cell=(3, 20),
                    card_slot=0,
                    trigger=Trigger(TriggerType.MATCH_TIME_US, value=0),
                ),
                PhysicalAction(
                    action_id="deploy-cannon",
                    side="B",
                    card_id="cannon",
                    arena_cell=(8, 13),
                    card_slot=3,
                    trigger=Trigger(
                        TriggerType.AFTER_OBSERVATION,
                        value=17_000,
                        event="hog_crosses_y_mtile",
                    ),
                ),
            ),
            (
                MeasurementSpec("hog_cannon_targeting", requires_direct_timing=True),
                MeasurementSpec("hog_cannon_pull_trajectory"),
                MeasurementSpec("cannon_lifetime_hp_decay"),
            ),
            ("hog-rider", "cannon"),
        ),
        (
            "hog-musketeer-support",
            "multi_card",
            "Hog pressure followed by a friendly Musketeer support deployment.",
            (DeckMutation("A", 1, "archers", "replace the unused opening Cannon slot"),),
            (
                _action("deploy-hog", "A", "hog-rider", (3, 20), 0, 0),
                _action("deploy-musketeer", "A", "musketeer", (4, 20), 3, 2_000_000),
                _action("deploy-cannon", "B", "cannon", (8, 13), 3, 4_000_000),
            ),
            (
                MeasurementSpec("hog_support_survival"),
                MeasurementSpec("musketeer_target_switch"),
                MeasurementSpec("tower_hit_count"),
            ),
            ("hog-rider", "musketeer", "cannon"),
        ),
        (
            "three-card-pressure",
            "complex",
            "Three timed card plays with a spell and a building interaction.",
            (
                DeckMutation("A", 1, "fireball", "add a spell to the opening hand"),
                DeckMutation("A", 6, "archers", "remove the displaced spell duplicate"),
                DeckMutation("B", 0, "archers", "change the opponent opening response"),
            ),
            (
                _action("deploy-hog", "A", "hog-rider", (3, 20), 0, 0),
                _action("deploy-archers", "B", "archers", (10, 18), 0, 2_000_000),
                _action("cast-fireball", "A", "fireball", (10, 18), 1, 4_000_000),
                _action("deploy-cannon", "B", "cannon", (8, 13), 3, 6_000_000),
            ),
            (
                MeasurementSpec("spell_area_damage"),
                MeasurementSpec("multi_card_targeting"),
                MeasurementSpec("cannon_lifetime_hp_decay"),
                MeasurementSpec("tower_hit_count"),
            ),
            ("hog-rider", "archers", "fireball", "cannon"),
        ),
    ]

    for case_id, level, description, mutations, actions, measurements, mechanics in definitions:
        decks = apply_deck_mutations(base, mutations)
        spec = _case_spec(
            case_id=case_id,
            level=level,
            description=description,
            decks=decks,
            actions=actions,
            measurements=measurements,
            evidence_split=evidence_split,
            metadata={
                "campaign_id": campaign_id,
                "deck_mutations": [item.to_dict() for item in mutations],
                "fixed_deck_host": "B",
                "opening_hand_source": "first_four_ordered_deck_cards",
                "phone_card_constraints": {
                    "A": PHONE_A_CARD_CONSTRAINTS,
                    "B": PHONE_B_CARD_CONSTRAINTS,
                },
            },
        )
        cases.append(
            CampaignCase(
                case_id=case_id,
                level=level,
                description=description,
                base_decks={side: tuple(deck) for side, deck in base.items()},
                mutations=tuple(mutations),
                spec=spec,
                expected_mechanics=mechanics,
            )
        )
    return InteractionCampaign(
        campaign_id=campaign_id,
        cases=tuple(cases),
        metadata={
            "purpose": "sim-to-real physical interaction sweep",
            "ordering": "isolated-to-complex",
            "fixed_deck_order_host": "B",
            "phone_card_constraints": {
                "A": PHONE_A_CARD_CONSTRAINTS,
                "B": PHONE_B_CARD_CONSTRAINTS,
            },
            "immutable_case_artifacts": True,
        },
    )


def evaluate_campaign(
    campaign: InteractionCampaign,
    *,
    results_root: str | Path,
    split: str | None = None,
    min_observations: int = 1,
    min_agreement_rate: float | None = None,
) -> dict[str, Any]:
    """Re-evaluate every stored physical corpus with the current simulator."""

    root = Path(results_root).resolve()
    engine = BattleEngine(load_ruleset(campaign.ruleset_id))
    rows: list[dict[str, Any]] = []
    for case in campaign.cases:
        corpus_path = root / (case.result_relpath or f"{case.case_id}/fidelity-corpus.json")
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "case_hash": case.case_hash(),
            "level": case.level,
            "corpus_path": str(corpus_path),
        }
        if not corpus_path.is_file():
            row.update({"status": "missing_artifact", "error": "fidelity corpus not found"})
            rows.append(row)
            continue
        try:
            report = run_fidelity_corpus(
                engine,
                corpus_path,
                split=split or case.spec.evidence_split.value,
            )
            report = apply_fidelity_gate(
                report,
                min_observations=min_observations,
                min_agreement_rate=min_agreement_rate,
                required_mechanics=case.expected_mechanics,
            )
            row.update(
                {
                    "status": "evaluated",
                    "corpus_hash": report.corpus_hash,
                    "report": report.to_dict(),
                    "gate": dict(report.gate or {}),
                }
            )
        except (OSError, PhysicalLabError, ValueError) as error:
            row.update({"status": "rejected_artifact", "error": str(error)})
        rows.append(row)
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "kind": "physical_lab_campaign_evaluation",
        "campaign_id": campaign.campaign_id,
        "campaign_hash": campaign.campaign_hash(),
        "engine_version": ENGINE_VERSION,
        "ruleset_id": campaign.ruleset_id,
        "ruleset_hash": campaign.ruleset_hash,
        "results_root": str(root),
        "split": split,
        "case_count": len(rows),
        "evaluated_case_count": len(evaluated),
        "missing_or_rejected_case_count": len(rows) - len(evaluated),
        "cases": rows,
    }


def write_campaign_evaluation(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write a sealed batch evaluation and return its content hash."""

    unsigned = dict(payload)
    unsigned.pop("evaluation_hash", None)
    sealed = {**unsigned, "evaluation_hash": canonical_hash(unsigned)}
    _write_immutable_json(Path(path), sealed)
    return str(sealed["evaluation_hash"])


__all__ = [
    "CAMPAIGN_LEVELS",
    "CAMPAIGN_SCHEMA_VERSION",
    "DEFAULT_CAMPAIGN_ID",
    "DEFAULT_CAMPAIGN_ROOT",
    "PHONE_B_CARD_CONSTRAINTS",
    "PHONE_B_REGULAR_MUSKETEER_DECK",
    "PHONE_A_CARD_CONSTRAINTS",
    "PHONE_A_REGULAR_MUSKETEER_DECK",
    "CampaignCase",
    "DeckMutation",
    "InteractionCampaign",
    "apply_deck_mutations",
    "build_default_campaign",
    "evaluate_campaign",
    "load_campaign",
    "materialize_campaign",
    "write_campaign_evaluation",
]
