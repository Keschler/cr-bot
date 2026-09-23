"""Deck- and receipt-aware card identity adjudication for physical runs.

The vision model supplies a class label, but a physical-fidelity experiment
also knows each player's declared deck and the cards that the runner actually
accepted.  This module keeps those sources of truth separate from the model
confidence and makes the conservative decision used by extraction:

* an accepted placement receipt can override a detector class near that
  placement;
* an unreceipted detector class must belong to the observed owner's declared
  deck when a deck is available; and
* an impossible or unmapped class is rejected rather than silently renamed.

Rejected rows remain readable in the normalized stream so detector failures
are auditable and can be used to improve the model later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


# A detector sample can arrive a little before the input receipt because the
# video and input streams are independently buffered.  The larger post-receipt
# window covers the first visible sample of a newly placed troop at the
# extractor's normal ~10 Hz cadence without claiming an old troop as the new
# action when a capture starts in the middle of a battle.
PLACEMENT_IDENTITY_BEFORE_US = 250_000
PLACEMENT_IDENTITY_AFTER_US = 2_000_000


@dataclass(frozen=True, slots=True)
class KnownPlacement:
    """One accepted physical placement on the internal match-time axis."""

    action_id: str
    owner: str
    card_id: str
    match_time_us: int
    arena_cell: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    """The auditable result of reconciling one detector identity."""

    accepted: bool
    card_id: str | None
    raw_card_id: str | None
    source: str
    reason: str
    matched_action_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "accepted": self.accepted,
            "card_id": self.card_id,
            "raw_card_id": self.raw_card_id,
            "source": self.source,
            "reason": self.reason,
        }
        if self.matched_action_id is not None:
            result["matched_action_id"] = self.matched_action_id
        return result


class KnownCardIdentity:
    """Resolve detector card labels using the run's known card universe."""

    def __init__(
        self,
        *,
        decks: Mapping[str, Sequence[str]] | None = None,
        placements: Sequence[KnownPlacement] = (),
        placements_authoritative: bool = False,
    ) -> None:
        self.decks = {
            str(side).upper(): frozenset(str(card).lower() for card in cards)
            for side, cards in (decks or {}).items()
            if str(side).upper() in {"A", "B"}
        }
        self.placements = tuple(
            placement
            for placement in placements
            if placement.owner.upper() in {"A", "B"}
        )
        self.placements_authoritative = bool(placements_authoritative)

    @property
    def enabled(self) -> bool:
        return bool(self.decks or self.placements)

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "decks": {
                side: sorted(cards)
                for side, cards in sorted(self.decks.items())
            },
            "placements": [
                {
                    "action_id": placement.action_id,
                    "owner": placement.owner.upper(),
                    "card_id": placement.card_id,
                    "match_time_us": placement.match_time_us,
                    "arena_cell": (
                        None
                        if placement.arena_cell is None
                        else list(placement.arena_cell)
                    ),
                }
                for placement in self.placements
            ],
            "placements_authoritative": self.placements_authoritative,
            "placement_window_us": {
                "before": PLACEMENT_IDENTITY_BEFORE_US,
                "after": PLACEMENT_IDENTITY_AFTER_US,
            },
        }

    def _placement_candidates(
        self,
        *,
        owner: str,
        match_time_us: int | None,
    ) -> list[KnownPlacement]:
        if match_time_us is None:
            return []
        owner = owner.upper()
        candidates = []
        for placement in self.placements:
            if placement.owner.upper() != owner:
                continue
            delta = int(match_time_us) - placement.match_time_us
            if -PLACEMENT_IDENTITY_BEFORE_US <= delta <= PLACEMENT_IDENTITY_AFTER_US:
                candidates.append(placement)
        return candidates

    def placement_lineage(
        self,
        *,
        owner: str,
        card_id: str,
        match_time_us: int | None,
    ) -> KnownPlacement | None:
        """Return the latest causally available placement of this card."""

        if match_time_us is None:
            return None
        owner = owner.upper()
        card_id = card_id.lower()
        candidates = [
            placement
            for placement in self.placements
            if placement.owner.upper() == owner
            and placement.card_id.lower() == card_id
            and placement.match_time_us - PLACEMENT_IDENTITY_BEFORE_US <= match_time_us
        ]
        return max(candidates, key=lambda item: item.match_time_us, default=None)

    @staticmethod
    def _can_have_a_tracked_entity(card_id: str) -> bool:
        """Avoid using a spell receipt to relabel an unrelated troop box."""

        try:
            from cr_bot.domain.card_metadata import CARD_METADATA

            kind = CARD_METADATA.get(card_id, {}).get("kind")
            if kind is not None:
                return kind != "spell"
        except (ImportError, AttributeError):
            pass
        # This fallback covers the common projectile/effect labels when the
        # optional cr_bot source environment is unavailable to a unit test.
        return card_id not in {
            "arrows",
            "barbarian-barrel",
            "clone",
            "earthquake",
            "fireball",
            "freeze",
            "giant-snowball",
            "graveyard",
            "lightning",
            "log",
            "poison",
            "rage",
            "rocket",
            "tornado",
            "zap",
        }

    def resolve(
        self,
        *,
        owner: str,
        raw_card_id: str | None,
        match_time_us: int | None,
    ) -> IdentityDecision:
        """Return a conservative, provenance-bearing identity decision."""

        owner = owner.upper()
        raw = None if raw_card_id is None else str(raw_card_id).lower()
        candidates = self._placement_candidates(
            owner=owner,
            match_time_us=match_time_us,
        )

        # A unique accepted placement is stronger than the model class.  If
        # multiple placements overlap, only an exact class match is allowed to
        # select one; otherwise we fall back to the deck gate rather than
        # assigning a card to the wrong action.
        exact = [
            placement
            for placement in candidates
            if str(placement.card_id).lower() == raw
        ]
        if exact:
            placement = min(
                exact,
                key=lambda item: abs(int(match_time_us) - item.match_time_us),
            )
            return IdentityDecision(
                accepted=True,
                card_id=str(placement.card_id).lower(),
                raw_card_id=raw,
                source="placement_receipt_verified",
                reason="detector class agrees with accepted placement receipt",
                matched_action_id=placement.action_id,
            )

        if len(candidates) == 1 and self._can_have_a_tracked_entity(candidates[0].card_id):
            placement = candidates[0]
            return IdentityDecision(
                accepted=True,
                card_id=str(placement.card_id).lower(),
                raw_card_id=raw,
                source="placement_receipt_override",
                reason="detector class overridden by unique accepted placement receipt",
                matched_action_id=placement.action_id,
            )

        allowed = self.decks.get(owner)
        if raw is None:
            return IdentityDecision(
                accepted=False,
                card_id=None,
                raw_card_id=None,
                source="rejected",
                reason="detector class is unmapped and no unique placement receipt matches",
            )
        if allowed is not None and raw not in allowed:
            return IdentityDecision(
                accepted=False,
                card_id=None,
                raw_card_id=raw,
                source="rejected",
                reason="detector card is absent from the observed owner's declared deck",
            )
        lineage = self.placement_lineage(
            owner=owner,
            card_id=raw,
            match_time_us=match_time_us,
        )
        if self.placements_authoritative and lineage is None:
            return IdentityDecision(
                accepted=False,
                card_id=None,
                raw_card_id=raw,
                source="rejected",
                reason="detector card has no causal placement in the authoritative action log",
            )
        if lineage is not None:
            return IdentityDecision(
                accepted=True,
                card_id=raw,
                raw_card_id=raw,
                source="placement_lineage",
                reason="detector card is causally backed by an accepted placement receipt",
                matched_action_id=lineage.action_id,
            )
        if allowed is not None:
            return IdentityDecision(
                accepted=True,
                card_id=raw,
                raw_card_id=raw,
                source="declared_deck",
                reason="detector card is present in the observed owner's declared deck",
            )
        return IdentityDecision(
            accepted=True,
            card_id=raw,
            raw_card_id=raw,
            source="unconstrained_detector",
            reason="no declared deck is available for the observed owner",
        )


__all__ = [
    "IdentityDecision",
    "KnownCardIdentity",
    "KnownPlacement",
    "PLACEMENT_IDENTITY_AFTER_US",
    "PLACEMENT_IDENTITY_BEFORE_US",
]
