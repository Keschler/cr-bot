from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from cr_bot.annotation_harness import EVIDENCE_KEYS, OWN_CONFIRMATION_KEYS
from cr_bot.domain.card_metadata import CARD_METADATA


@dataclass(frozen=True)
class ModelSpec:
    model: str
    reasoning_effort: str
    cost_multiplier: float


# The current Codex rate card is token-based, and all three token columns have
# the same model ratio. The Plus UI also reports local-message ranges per
# five-hour window; both ends of those ranges yield the same ratio to Terra,
# so we use that ratio as the normalized accounting weight for raw worker
# tokens:
#
#   Sol   10–100 messages -> 2.5x Terra (25/10 == 200/100)
#   Terra 25–200 messages -> 1.0x baseline
#   Luna  250–2,000       -> 0.1x Terra (25/250 == 200/2,000)
#
# These weights are for comparing model usage in pipeline state; they are not
# a promise of a fixed number of messages available to a particular account.
MODEL_COST_MULTIPLIERS: dict[str, float] = {
    "gpt-5.6-sol": 2.5,
    "gpt-5.6-terra": 1.0,
    "gpt-5.6-luna": 0.1,
}


MODEL_PROFILES: dict[str, dict[str, ModelSpec]] = {
    "terra-efficient": {
        "own_slot_primary": ModelSpec("gpt-5.6-terra", "low", 1),
        "own_primary": ModelSpec("gpt-5.6-terra", "medium", 1),
        "own_completeness": ModelSpec("gpt-5.6-terra", "low", 1),
        "own_adjudication": ModelSpec("gpt-5.6-terra", "low", 1),
        "own_release_review": ModelSpec("gpt-5.6-terra", "low", 1),
        "enemy_spells": ModelSpec("gpt-5.6-terra", "medium", 1),
        "enemy_spell_confirmation": ModelSpec("gpt-5.6-terra", "low", 1),
        "enemy_spell_recovery": ModelSpec("gpt-5.6-terra", "medium", 1),
        "enemy_spell_boundary": ModelSpec("gpt-5.6-terra", "low", 1),
        "enemy_existence": ModelSpec("gpt-5.6-terra", "medium", 1),
        "enemy_side_check": ModelSpec("gpt-5.6-terra", "low", 1),
        "enemy_simultaneous_recovery": ModelSpec(
            "gpt-5.6-terra", "medium", 1
        ),
        "enemy_side_escalation": ModelSpec(
            "gpt-5.6-terra", "medium", 1
        ),
        "enemy_card": ModelSpec("gpt-5.6-terra", "medium", 1),
    },
    "terra-recall": {
        "own_slot_primary": ModelSpec("gpt-5.6-terra", "low", 1),
        "own_primary": ModelSpec("gpt-5.6-terra", "medium", 1),
        "own_completeness": ModelSpec("gpt-5.6-terra", "low", 1),
        "own_adjudication": ModelSpec("gpt-5.6-terra", "medium", 1),
        "own_release_review": ModelSpec("gpt-5.6-terra", "low", 1),
        "enemy_spells": ModelSpec("gpt-5.6-terra", "medium", 1),
        "enemy_spell_confirmation": ModelSpec("gpt-5.6-terra", "low", 1),
        "enemy_spell_recovery": ModelSpec("gpt-5.6-terra", "medium", 1),
        "enemy_spell_boundary": ModelSpec("gpt-5.6-terra", "low", 1),
        "enemy_existence": ModelSpec("gpt-5.6-terra", "medium", 1),
        "enemy_side_check": ModelSpec("gpt-5.6-terra", "low", 1),
        "enemy_simultaneous_recovery": ModelSpec(
            "gpt-5.6-terra", "medium", 1
        ),
        "enemy_side_escalation": ModelSpec(
            "gpt-5.6-luna", "medium", 0.1
        ),
        "enemy_card": ModelSpec("gpt-5.6-terra", "medium", 1),
    },
    "hybrid-accuracy": {
        # Once deterministic before/after card-return matching fixes the
        # release outcome, Luna-low identifies the departing own card at one
        # tenth of Terra's normalized local-message cost without changing
        # event recall.
        "own_slot_primary": ModelSpec("gpt-5.6-luna", "low", 0.1),
        "own_primary": ModelSpec("gpt-5.6-terra", "medium", 1),
        "own_completeness": ModelSpec("gpt-5.6-terra", "low", 1),
        "own_adjudication": ModelSpec("gpt-5.6-terra", "low", 1),
        "own_release_review": ModelSpec("gpt-5.6-terra", "low", 1),
        "enemy_spells": ModelSpec("gpt-5.6-terra", "medium", 1),
        "enemy_spell_confirmation": ModelSpec("gpt-5.6-terra", "low", 1),
        # Luna-low was cheaper on isolated onset sheets but unstable once both
        # lane searches were present (one false recovery and one miss). Terra
        # remains the fail-closed production choice for this precision gate.
        "enemy_spell_recovery": ModelSpec("gpt-5.6-terra", "low", 1),
        "enemy_spell_boundary": ModelSpec("gpt-5.6-terra", "low", 1),
        # Sol is worth its 2.5x price for sequence-aware existence decisions:
        # Terra repeatedly missed small actors occluded by an older large unit.
        "enemy_existence": ModelSpec("gpt-5.6-sol", "medium", 2.5),
        # Side is a much smaller direct-evidence task after temporal dedupe.
        "enemy_side_check": ModelSpec("gpt-5.6-terra", "low", 1),
        # Recover an enemy actor hidden by a simultaneous own deployment.
        "enemy_simultaneous_recovery": ModelSpec(
            "gpt-5.6-terra", "medium", 1
        ),
        # Luna is reserved for a cheap escalation after the direct Terra side
        # check leaves a row unresolved.
        "enemy_side_escalation": ModelSpec(
            "gpt-5.6-luna", "medium", 0.1
        ),
        "enemy_card": ModelSpec("gpt-5.6-terra", "medium", 1),
    },
    "sol-experimental": {
        stage: ModelSpec("gpt-5.6-sol", "medium", 2.5)
        for stage in (
            "own_primary",
            "own_slot_primary",
            "own_completeness",
            "own_adjudication",
            "own_release_review",
            "enemy_spells",
            "enemy_spell_confirmation",
            "enemy_spell_recovery",
            "enemy_spell_boundary",
            "enemy_existence",
            "enemy_side_check",
            "enemy_simultaneous_recovery",
            "enemy_side_escalation",
            "enemy_card",
        )
    },
    "luna-experimental": {
        stage: ModelSpec("gpt-5.6-luna", "medium", 0.1)
        for stage in (
            "own_primary",
            "own_slot_primary",
            "own_completeness",
            "own_adjudication",
            "own_release_review",
            "enemy_spells",
            "enemy_spell_confirmation",
            "enemy_spell_recovery",
            "enemy_spell_boundary",
            "enemy_existence",
            "enemy_side_check",
            "enemy_simultaneous_recovery",
            "enemy_side_escalation",
            "enemy_card",
        )
    },
}


def normalize_enemy_unit_decision_roles(
    document: dict[str, object],
    package: dict[str, object],
) -> bool:
    """Normalize package ownership, which is metadata rather than judgment."""
    bursts = package.get("bursts")
    decisions = document.get("burst_decisions")
    if not isinstance(bursts, list) or not isinstance(decisions, list):
        return False
    roles = {
        row.get("burst_id"): row.get("package_role")
        for row in bursts
        if isinstance(row, dict)
    }
    changed = False
    for row in decisions:
        if not isinstance(row, dict):
            continue
        role = roles.get(row.get("burst_id"))
        if role == "context_burst" and row.get("decision") != "context_only":
            row["decision"] = "context_only"
            row["accepted_onset_ids"] = []
            changed = True
        elif role == "owned_burst" and row.get("decision") == "context_only":
            row["decision"] = "rejected"
            row["accepted_onset_ids"] = []
            changed = True
    return changed


def validate_enemy_unit_decisions(
    document: dict[str, object],
    package: dict[str, object],
) -> None:
    bursts = package.get("bursts")
    if not isinstance(bursts, list):
        raise ValueError("enemy-unit package bursts must be a list")
    expected = {
        row["burst_id"]: row
        for row in bursts
        if isinstance(row, dict) and isinstance(row.get("burst_id"), str)
    }
    if len(expected) != len(bursts):
        raise ValueError("enemy-unit package has invalid burst rows")
    decisions = document.get("burst_decisions")
    if not isinstance(decisions, list):
        raise ValueError("burst_decisions must be a list")
    decision_ids = [
        row.get("burst_id") if isinstance(row, dict) else None
        for row in decisions
    ]
    if (
        len(decision_ids) != len(set(decision_ids))
        or set(decision_ids) != set(expected)
    ):
        raise ValueError("burst_decisions must cover package bursts exactly")
    for row in decisions:
        if not isinstance(row, dict):
            raise ValueError("burst decision must be an object")
        burst = expected[row["burst_id"]]
        role = burst.get("package_role")
        decision = row.get("decision")
        if role == "context_burst":
            if decision != "context_only":
                raise ValueError("context burst must use context_only")
        elif role == "owned_burst":
            if decision not in {"accepted", "rejected"}:
                raise ValueError(
                    "owned burst must use accepted or rejected"
                )
        else:
            raise ValueError("burst has unknown package_role")
        if row.get("accepted_onset_ids") != []:
            raise ValueError(
                "accepted_onset_ids must stay empty; merger creates IDs"
            )
    if document.get("onsets") != []:
        raise ValueError(
            "enemy-unit decision worker must not create onset records"
        )


def validate_own_semantic_decisions(
    document: dict[str, object],
    package: dict[str, object],
    *,
    require_candidate_coverage: bool,
) -> None:
    candidates = package.get("candidates")
    events = document.get("events")
    if not isinstance(candidates, list) or not isinstance(events, list):
        raise ValueError("own candidates and events must be lists")
    expected = {
        row["candidate_id"]: row
        for row in candidates
        if isinstance(row, dict) and isinstance(row.get("candidate_id"), str)
    }
    if len(expected) != len(candidates):
        raise ValueError("own package has invalid or duplicate candidates")

    event_candidates: set[str] = set()
    groups: dict[int, list[tuple[dict[str, Any], int]]] = {}
    for index, raw in enumerate(events):
        if not isinstance(raw, dict):
            raise ValueError(f"own event {index} must be an object")
        candidate_id = raw.get("candidate_id")
        candidate = expected.get(candidate_id)
        if candidate is None:
            raise ValueError(f"own event {index}: unknown candidate")
        event_candidates.add(candidate_id)
        card = raw.get("card")
        if not isinstance(card, str):
            raise ValueError(f"own event {index}: card is required")
        card = card.strip().lower().replace("_", "-")
        if card == "the-log":
            card = "log"
        metadata = CARD_METADATA.get(card)
        if metadata is None:
            raise ValueError(f"own event {index}: unknown card {card!r}")
        frame = raw.get("event_frame_index")
        sampled = candidate.get("discovery_frame_indices")
        if (
            not isinstance(frame, int)
            or isinstance(frame, bool)
            or not isinstance(sampled, list)
            or not sampled
            or min(abs(frame - int(value)) for value in sampled) > 1
        ):
            raise ValueError(
                f"own event {index}: frame is not supported by labeled evidence"
            )
        artifact = candidate.get("discovery_artifact")
        if raw.get("verification_artifacts") != [artifact]:
            raise ValueError(
                f"own event {index}: verification must cite discovery evidence"
            )
        confirmation_frame = raw.get("confirmation_frame_index")
        if (
            not isinstance(confirmation_frame, int)
            or confirmation_frame not in sampled
            or not frame + 5 <= confirmation_frame <= frame + 15
            or raw.get("confirmation_artifacts") != [artifact]
        ):
            raise ValueError(
                f"own event {index}: invalid labeled confirmation evidence"
            )
        observation = raw.get("transition_observation")
        if not isinstance(observation, dict):
            raise ValueError(
                f"own event {index}: transition observation is required"
            )
        before = observation.get("elixir_before")
        after = observation.get("elixir_after")
        delta = observation.get("observed_elixir_delta")
        total = observation.get("total_released_cost")
        compensated = observation.get("regeneration_compensated")
        if (
            not isinstance(before, int)
            or not 0 <= before <= 10
            or not isinstance(after, int)
            or not 0 <= after <= 10
            or not isinstance(delta, int)
            or delta != max(0, before - after)
            or not isinstance(total, int)
            or total < int(metadata["elixir_cost"])
            or abs(total - delta) > 1
            or compensated is not (total > delta)
        ):
            raise ValueError(
                f"own event {index}: inconsistent elixir accounting"
            )
        if not isinstance(observation.get("occupied_slots_before"), list) or not isinstance(
            observation.get("cooldown_slots_after"), list
        ):
            raise ValueError(f"own event {index}: hand-slot evidence is required")
        evidence = raw.get("evidence")
        confirmation = raw.get("own_confirmation")
        if (
            not isinstance(evidence, dict)
            or set(evidence) != set(EVIDENCE_KEYS)
            or any(
                value is not True and value is not False and value is not None
                for value in evidence.values()
            )
            or (
                evidence.get("elixir_drop") is not True
                and not (
                    compensated is True
                    and evidence.get("elixir_drop") is False
                )
            )
            or evidence.get("hand_transition") is not True
            or evidence.get("deployment_onset") is not True
            or not isinstance(confirmation, dict)
            or set(confirmation) != OWN_CONFIRMATION_KEYS
            or any(
                value is not True and value is not False and value is not None
                for value in confirmation.values()
            )
            or confirmation.get("release_confirmed") is not True
            or confirmation.get("elixir_spend_persisted") is not True
            or confirmation.get("hand_cycle_completed") is not True
            or confirmation.get("post_release_effect") is not True
        ):
            raise ValueError(
                f"own event {index}: release persistence is not confirmed"
            )
        spell_release = raw.get("spell_release")
        if metadata["kind"] == "spell":
            if (
                not isinstance(spell_release, dict)
                or spell_release.get("targeting_overlay_cleared") is not True
                or spell_release.get("projectile_or_impact_visible") is not True
            ):
                raise ValueError(
                    f"own event {index}: spell release is not directly confirmed"
                )
        elif spell_release is not None:
            raise ValueError(
                f"own event {index}: non-spell has spell release evidence"
            )
        groups.setdefault(frame, []).append(
            (raw, int(metadata["elixir_cost"]))
        )

    for frame, rows in groups.items():
        totals = {
            row["transition_observation"]["total_released_cost"]
            for row, _ in rows
        }
        if len(totals) != 1 or totals.pop() != sum(cost for _, cost in rows):
            raise ValueError(
                f"own events at frame {frame}: grouped cost is inconsistent"
            )

    if not require_candidate_coverage:
        return
    rejected = document.get("rejected_candidates")
    pending = document.get("pending_at_end")
    if not isinstance(rejected, list) or not isinstance(pending, list):
        raise ValueError("own primary output requires rejected and pending rows")

    def decision_ids(rows: list[object], label: str) -> set[str]:
        values = [
            row.get("candidate_id") if isinstance(row, dict) else None
            for row in rows
        ]
        if (
            any(value not in expected for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"own {label} rows contain invalid candidates")
        return set(values)

    rejected_ids = decision_ids(rejected, "rejected")
    pending_ids = decision_ids(pending, "pending")
    categories = [event_candidates, rejected_ids, pending_ids]
    if any(
        left & right
        for index, left in enumerate(categories)
        for right in categories[index + 1 :]
    ):
        raise ValueError("own candidate appears in multiple decision categories")
    if set().union(*categories) != set(expected):
        raise ValueError("own decisions must cover every package candidate")


def validate_own_adjudication_decisions(
    document: dict[str, object],
    package: dict[str, object],
) -> None:
    proposals = package.get("proposals")
    events = document.get("events")
    rejected = document.get("rejected_proposals")
    if (
        not isinstance(proposals, list)
        or not isinstance(events, list)
        or not isinstance(rejected, list)
    ):
        raise ValueError("own adjudication proposals and decisions must be lists")
    expected = {
        row["proposal_id"]: row
        for row in proposals
        if isinstance(row, dict) and isinstance(row.get("proposal_id"), str)
    }
    if len(expected) != len(proposals):
        raise ValueError("own adjudication has invalid proposals")
    decided = [
        row.get("proposal_id") if isinstance(row, dict) else None
        for row in [*events, *rejected]
    ]
    if len(decided) != len(set(decided)) or set(decided) != set(expected):
        raise ValueError("own adjudication must cover every proposal exactly")
    candidate_rows: dict[str, dict[str, Any]] = {}
    semantic_events = []
    for row in events:
        proposal = expected[row["proposal_id"]]
        if row.get("candidate_id") not in proposal["candidate_ids"]:
            raise ValueError("own adjudication selected an unrelated candidate")
        if row.get("card") != proposal["card"]:
            raise ValueError("own adjudication changed proposal card identity")
        for candidate in proposal["candidate_evidence"]:
            candidate_rows[candidate["candidate_id"]] = candidate
        semantic_events.append(
            {key: value for key, value in row.items() if key != "proposal_id"}
        )
    validate_own_semantic_decisions(
        {"events": semantic_events},
        {"candidates": list(candidate_rows.values())},
        require_candidate_coverage=False,
    )


def validate_own_release_review_decisions(
    document: dict[str, object],
    package: dict[str, object],
) -> None:
    proposed = package.get("reviews")
    decisions = document.get("decisions")
    if not isinstance(proposed, list) or not isinstance(decisions, list):
        raise ValueError("own release reviews and decisions must be lists")
    expected = {
        row["event_id"]: row
        for row in proposed
        if isinstance(row, dict) and isinstance(row.get("event_id"), str)
    }
    if len(expected) != len(proposed):
        raise ValueError("own release-review package has invalid review rows")
    rows = {
        row["event_id"]: row
        for row in decisions
        if isinstance(row, dict) and isinstance(row.get("event_id"), str)
    }
    if len(rows) != len(decisions) or set(rows) != set(expected):
        raise ValueError("own release review must cover every event exactly once")
    check_keys = {
        "release_confirmed",
        "elixir_spend_persisted",
        "hand_cycle_completed",
        "post_release_effect",
    }
    decision_keys = {
        "event_id",
        "decision",
        "confirmation_frame_index",
        "confirmation_artifacts",
        "checks",
        "reason",
    }
    for event_id, row in rows.items():
        if set(row) != decision_keys:
            raise ValueError(
                f"{event_id}: release decision must contain exactly "
                f"{', '.join(sorted(decision_keys))}"
            )
        source = expected[event_id]
        if row["confirmation_frame_index"] != source["confirmation_frame_index"]:
            raise ValueError(f"{event_id}: confirmation frame changed")
        if row["confirmation_artifacts"] != source["confirmation_artifacts"]:
            raise ValueError(f"{event_id}: confirmation artifacts changed")
        decision = row["decision"]
        if decision not in {"released", "canceled", "unresolved"}:
            raise ValueError(f"{event_id}: invalid release decision {decision!r}")
        checks = row["checks"]
        if not isinstance(checks, dict) or set(checks) != check_keys:
            raise ValueError(f"{event_id}: checks have invalid keys")
        if any(
            value is not None and not isinstance(value, bool)
            for value in checks.values()
        ):
            raise ValueError(f"{event_id}: release checks must be boolean or null")
        if decision == "released":
            required = (
                checks["release_confirmed"],
                checks["elixir_spend_persisted"],
                checks["post_release_effect"],
            )
            if required != (True, True, True):
                raise ValueError(
                    f"{event_id}: released requires release, spend, and effect"
                )
            if row["reason"] is not None:
                raise ValueError(f"{event_id}: released reason must be null")
        else:
            if not isinstance(row["reason"], str) or not row["reason"].strip():
                raise ValueError(
                    f"{event_id}: canceled/unresolved requires a visual reason"
                )
            if decision == "canceled" and checks["release_confirmed"] is not False:
                raise ValueError(
                    f"{event_id}: canceled requires release_confirmed false"
                )


def validate_own_slot_interval_decisions(
    document: dict[str, object],
    package: dict[str, object],
) -> None:
    """Validate the isolated hand-slot empty-interval worker contract."""

    if document.get("stage") != "own_slot_intervals_chunk":
        raise ValueError("own slot output has the wrong stage")
    if document.get("run_id") != package.get("run_id"):
        raise ValueError("own slot output run_id does not match package")
    if document.get("target_range") != package.get("target_range"):
        raise ValueError("own slot output target_range does not match package")

    intervals = package.get("intervals")
    decisions = document.get("decisions")
    if not isinstance(intervals, list) or not isinstance(decisions, list):
        raise ValueError("own slot intervals and decisions must be lists")
    expected = {
        row["interval_id"]: row
        for row in intervals
        if isinstance(row, dict) and isinstance(row.get("interval_id"), str)
    }
    rows = {
        row["interval_id"]: row
        for row in decisions
        if isinstance(row, dict) and isinstance(row.get("interval_id"), str)
    }
    if len(expected) != len(intervals):
        raise ValueError("own slot package has invalid or duplicate intervals")
    if len(rows) != len(decisions) or set(rows) != set(expected):
        raise ValueError(
            "own slot decisions must cover every interval exactly once"
        )

    decision_keys = {
        "interval_id",
        "decision",
        "card",
        "event_frame_index",
        "confirmation_frame_index",
        "artifact",
        "reason",
    }
    for interval_id, row in rows.items():
        if set(row) != decision_keys:
            raise ValueError(
                f"{interval_id}: decision must contain exactly "
                f"{', '.join(sorted(decision_keys))}"
            )
        source = expected[interval_id]
        if row.get("artifact") != source.get("artifact"):
            raise ValueError(f"{interval_id}: artifact changed")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{interval_id}: a visual reason is required")

        decision = row.get("decision")
        if decision not in {"released", "canceled", "unresolved"}:
            raise ValueError(
                f"{interval_id}: invalid interval decision {decision!r}"
            )
        return_evidence = source.get("return_evidence")
        if not isinstance(return_evidence, dict):
            raise ValueError(
                f"{interval_id}: deterministic return evidence is required"
            )
        constraint = return_evidence.get("outcome_constraint")
        if constraint not in {"released", "canceled"}:
            raise ValueError(
                f"{interval_id}: invalid deterministic outcome constraint"
            )
        if decision != constraint:
            raise ValueError(
                f"{interval_id}: decision {decision!r} conflicts with "
                f"deterministic {constraint!r} constraint"
            )
        if decision != "released":
            if any(
                row.get(key) is not None
                for key in (
                    "card",
                    "event_frame_index",
                    "confirmation_frame_index",
                )
            ):
                raise ValueError(
                    f"{interval_id}: canceled/unresolved timing must be null"
                )
            continue

        card = row.get("card")
        if not isinstance(card, str) or card not in CARD_METADATA:
            raise ValueError(
                f"{interval_id}: released card must be a canonical slug"
            )
        event_frame = row.get("event_frame_index")
        confirmation_frame = row.get("confirmation_frame_index")
        sampled = source.get("sampled_frame_indices")
        if not isinstance(sampled, list) or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in sampled
        ):
            raise ValueError(f"{interval_id}: invalid sampled frame metadata")
        if (
            not isinstance(event_frame, int)
            or isinstance(event_frame, bool)
            or event_frame not in sampled
        ):
            raise ValueError(
                f"{interval_id}: released frame is not sampled"
            )
        if (
            not isinstance(confirmation_frame, int)
            or isinstance(confirmation_frame, bool)
            or confirmation_frame not in sampled
            or not event_frame + 5
            <= confirmation_frame
            <= event_frame + 15
        ):
            raise ValueError(
                f"{interval_id}: confirmation must be sampled 5-15 frames later"
            )


def validate_enemy_identity_decisions(
    document: dict[str, object],
    package: dict[str, object],
) -> None:
    onsets = package.get("onsets")
    decisions = document.get("decisions")
    if not isinstance(onsets, list) or not isinstance(decisions, list):
        raise ValueError("identity onsets and decisions must be lists")
    expected = {
        row["onset_id"]: row
        for row in onsets
        if isinstance(row, dict) and isinstance(row.get("onset_id"), str)
    }
    rows = {
        row["onset_id"]: row
        for row in decisions
        if isinstance(row, dict) and isinstance(row.get("onset_id"), str)
    }
    if (
        len(expected) != len(onsets)
        or len(rows) != len(decisions)
        or set(rows) != set(expected)
    ):
        raise ValueError("identity decisions must cover package onsets exactly")
    for onset_id, row in rows.items():
        onset = expected[onset_id]
        if not isinstance(row.get("event_exists"), bool):
            raise ValueError(f"{onset_id}: event_exists must be boolean")
        if not isinstance(row.get("existence_evidence"), dict):
            raise ValueError(f"{onset_id}: existence evidence is required")
        side = row.get("side")
        if side not in {"own", "enemy", "unresolved"}:
            raise ValueError(f"{onset_id}: invalid side {side!r}")
        if row.get("card") is not None:
            raise ValueError(f"{onset_id}: side stage cannot identify a card")
        side_evidence = row.get("side_evidence")
        if not isinstance(side_evidence, dict):
            raise ValueError(f"{onset_id}: side evidence is required")
        if (
            row["event_exists"]
            and side in {"own", "enemy"}
            and side_evidence.get("direct") is not True
        ):
            raise ValueError(f"{onset_id}: retained side is not direct")
        if (
            row.get("identity_frame_index") is not None
            or row.get("identity_artifacts") != []
        ):
            raise ValueError(
                f"{onset_id}: side gate must not create identity evidence"
            )


def validate_enemy_spell_decisions(
    document: dict[str, object],
    package: dict[str, object],
) -> None:
    onsets = document.get("onsets")
    if not isinstance(onsets, list):
        raise ValueError("enemy spell onsets must be a list")
    primary = {
        row["candidate_id"]: row
        for row in package.get("primary_windows", [])
        if isinstance(row, dict)
    }
    allowed_artifacts = {
        row["verification_artifact"]
        for key in ("primary_windows", "boundary_windows")
        for row in package.get(key, [])
        if isinstance(row, dict)
    }
    seen: set[str] = set()
    owned_start, owned_end = package["owned_event_range"]
    for index, row in enumerate(onsets):
        if not isinstance(row, dict):
            raise ValueError(f"enemy spell {index} must be an object")
        onset_id = row.get("onset_id")
        if not isinstance(onset_id, str) or onset_id in seen:
            raise ValueError(f"enemy spell {index}: onset_id must be unique")
        seen.add(onset_id)
        candidate = primary.get(row.get("candidate_id"))
        frame = row.get("event_frame_index")
        if (
            candidate is None
            or not isinstance(frame, int)
            or not owned_start <= frame < owned_end
            or not candidate["inspection_start_frame"]
            <= frame
            < candidate["inspection_end_frame_exclusive"]
        ):
            raise ValueError(f"enemy spell {index}: unsupported event frame")
        artifacts = row.get("verification_artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or any(value not in allowed_artifacts for value in artifacts)
        ):
            raise ValueError(f"enemy spell {index}: invalid cited evidence")
        evidence = row.get("evidence")
        if (
            row.get("kind") != "spell"
            or row.get("absence_confirmed") is not True
            or row.get("persistence_confirmed") is not True
            or not isinstance(evidence, dict)
            or evidence.get("impact_sequence") is not True
            or row.get("identity_artifacts") != []
        ):
            raise ValueError(f"enemy spell {index}: sequence is not confirmed")


def normalize_enemy_spell_decision_artifacts(
    document: dict[str, object],
    package: dict[str, object],
) -> bool:
    """Bind each blind spell decision to its package-owned primary sheet.

    Workers decide whether and when a spell exists.  The candidate ID already
    names exactly one primary arena window, so accepting a copied-path typo or
    invented alias adds no semantic information.  Canonicalizing the citation
    here keeps the evidence contract deterministic without changing a decision.
    """

    primary = {
        row["candidate_id"]: row["verification_artifact"]
        for row in package.get("primary_windows", [])
        if isinstance(row, dict)
        and isinstance(row.get("candidate_id"), str)
        and isinstance(row.get("verification_artifact"), str)
    }
    changed = False
    for row in document.get("onsets", []):
        if not isinstance(row, dict):
            continue
        artifact = primary.get(row.get("candidate_id"))
        if artifact is not None and row.get("verification_artifacts") != [artifact]:
            row["verification_artifacts"] = [artifact]
            changed = True
    return changed


ENEMY_SPELL_CONFIRMATION_CHECKS = frozenset(
    {
        "absent_before",
        "coherent_sequence",
        "independent_spell_object_or_resolution",
        "not_unit_attack_or_ability",
        "not_targeting_overlay_or_floating_label",
        "enemy_direction_or_origin",
        "resolved_after",
        "boundary_truncated",
    }
)


def normalize_enemy_spell_confirmation_artifacts(
    document: dict[str, object],
    package: dict[str, object],
) -> bool:
    """Bind each confirmation decision to its review-owned evidence sheets.

    The review ID already selects exactly one sealed review.  Replacing a
    worker's copied-path typo with that review's package-owned artifact list is
    therefore structural normalization, not a semantic annotation change.
    """

    expected = {
        row["review_id"]: row["confirmation_artifacts"]
        for row in package.get("reviews", [])
        if isinstance(row, dict)
        and isinstance(row.get("review_id"), str)
        and isinstance(row.get("confirmation_artifacts"), list)
    }
    changed = False
    for row in document.get("decisions", []):
        if not isinstance(row, dict):
            continue
        artifacts = expected.get(row.get("review_id"))
        if artifacts is not None and row.get("confirmation_artifacts") != artifacts:
            row["confirmation_artifacts"] = list(artifacts)
            changed = True
    return changed


def validate_enemy_spell_confirmation_decisions(
    document: dict[str, object],
    package: dict[str, object],
) -> None:
    """Validate exact, side-aware spell confirmation decisions.

    Broad spell scanning is deliberately recall-oriented.  This second pass
    must cover every triggered exact review and fails closed unless the visual
    sequence is both an independent spell and directly attributable to the
    enemy.  A final-frame spell may be retained without forward resolution
    only when its review is explicitly marked as the segment-end sentinel.
    """

    reviews = package.get("reviews")
    decisions = document.get("decisions")
    if not isinstance(reviews, list) or not isinstance(decisions, list):
        raise ValueError("spell confirmation reviews and decisions must be lists")
    expected = {
        row["review_id"]: row
        for row in reviews
        if isinstance(row, dict) and isinstance(row.get("review_id"), str)
    }
    rows = {
        row["review_id"]: row
        for row in decisions
        if isinstance(row, dict) and isinstance(row.get("review_id"), str)
    }
    if (
        len(expected) != len(reviews)
        or len(rows) != len(decisions)
        or set(rows) != set(expected)
    ):
        raise ValueError(
            "spell confirmation decisions must cover reviews exactly"
        )
    segment_end = int(package["segment"]["end_frame_exclusive"])
    for review_id, row in rows.items():
        review = expected[review_id]
        decision = row.get("decision")
        if decision not in {"confirmed", "rejected", "unresolved"}:
            raise ValueError(f"{review_id}: invalid spell confirmation decision")
        if row.get("card") is not None:
            raise ValueError(f"{review_id}: confirmation stage cannot identify a card")
        artifacts = row.get("confirmation_artifacts")
        if artifacts != review.get("confirmation_artifacts"):
            raise ValueError(f"{review_id}: exact spell artifacts are required")
        checks = row.get("checks")
        if (
            not isinstance(checks, dict)
            or set(checks) != ENEMY_SPELL_CONFIRMATION_CHECKS
            or any(value not in {True, False, None} for value in checks.values())
        ):
            raise ValueError(f"{review_id}: invalid spell confirmation checks")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{review_id}: visual reason is required")
        frame = row.get("event_frame_index")
        effect_class = row.get("effect_class")
        if decision != "confirmed":
            if frame is not None or effect_class != "unresolved":
                raise ValueError(
                    f"{review_id}: rejected/unresolved spell must fail closed"
                )
            continue
        if (
            not isinstance(frame, int)
            or frame not in review.get("sampled_frame_indices", [])
        ):
            raise ValueError(
                f"{review_id}: confirmed onset must use a labeled review frame"
            )
        if effect_class not in {
            "directional_projectile",
            "rolling_object",
            "area_impact",
        }:
            raise ValueError(f"{review_id}: invalid confirmed spell effect class")
        required_true = (
            "absent_before",
            "coherent_sequence",
            "independent_spell_object_or_resolution",
            "not_unit_attack_or_ability",
            "not_targeting_overlay_or_floating_label",
            "enemy_direction_or_origin",
        )
        if any(checks[key] is not True for key in required_true):
            raise ValueError(f"{review_id}: spell sequence is not directly confirmed")
        if checks["resolved_after"] is not True:
            boundary_ok = (
                review.get("segment_end_sentinel") is True
                and checks["boundary_truncated"] is True
                # Compression/first-pixel ambiguity can place the visible
                # projectile one source frame before the final-frame sentinel.
                and frame >= segment_end - 2
            )
            if not boundary_ok:
                raise ValueError(
                    f"{review_id}: spell lacks forward resolution evidence"
                )
        elif checks["boundary_truncated"] is True:
            raise ValueError(
                f"{review_id}: resolved spell cannot be boundary truncated"
            )


def validate_enemy_unit_scan_decisions(
    document: dict[str, object],
    package: dict[str, object],
) -> None:
    """Validate direct enemy-unit discoveries from exhaustive arena sheets.

    This pass complements marker-focused discovery.  It is intentionally
    allowed to emit no rows, but every emitted row must be tied to a primary
    scan window and must carry direct temporal and enemy-side evidence.
    """

    onsets = document.get("onsets")
    if not isinstance(onsets, list):
        raise ValueError("enemy unit scan onsets must be a list")
    primary = {
        row["candidate_id"]: row
        for row in package.get("primary_windows", [])
        if isinstance(row, dict)
        and isinstance(row.get("candidate_id"), str)
    }
    allowed_artifacts = {
        row["verification_artifact"]
        for key in ("primary_windows", "boundary_windows")
        for row in package.get(key, [])
        if isinstance(row, dict)
        and isinstance(row.get("verification_artifact"), str)
    }
    owned_start, owned_end = package["owned_event_range"]
    seen_ids: set[str] = set()
    seen_frames: set[int] = set()
    required_evidence = {
        "absent_before",
        "independent_after",
        "persistent_or_resolved_after",
        "direct_enemy_side",
    }
    for index, row in enumerate(onsets):
        if not isinstance(row, dict):
            raise ValueError(f"enemy unit scan {index} must be an object")
        onset_id = row.get("onset_id")
        frame = row.get("event_frame_index")
        if (
            not isinstance(onset_id, str)
            or onset_id in seen_ids
            or not isinstance(frame, int)
            or frame in seen_frames
        ):
            raise ValueError(
                f"enemy unit scan {index}: onset ID and frame must be unique"
            )
        seen_ids.add(onset_id)
        seen_frames.add(frame)
        candidate = primary.get(row.get("candidate_id"))
        if (
            candidate is None
            or not owned_start <= frame < owned_end
            or not candidate["inspection_start_frame"]
            <= frame
            < candidate["inspection_end_frame_exclusive"]
        ):
            raise ValueError(
                f"enemy unit scan {index}: unsupported event frame"
            )
        artifacts = row.get("verification_artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or any(value not in allowed_artifacts for value in artifacts)
        ):
            raise ValueError(
                f"enemy unit scan {index}: invalid cited evidence"
            )
        evidence = row.get("evidence")
        if (
            row.get("kind") != "unit_or_building"
            or row.get("side") != "enemy"
            or not isinstance(evidence, dict)
            or set(evidence) != required_evidence
            or any(evidence[key] is not True for key in required_evidence)
        ):
            raise ValueError(
                f"enemy unit scan {index}: direct sequence evidence required"
            )
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError(
                f"enemy unit scan {index}: visual reason is required"
            )


def validate_enemy_card_decisions(
    document: dict[str, object],
    package: dict[str, object],
) -> None:
    targets = package.get("targets")
    cards = document.get("cards")
    if not isinstance(targets, list) or not isinstance(cards, list):
        raise ValueError("enemy card targets and decisions must be lists")
    expected = {
        row["onset_id"]: row
        for row in targets
        if isinstance(row, dict) and isinstance(row.get("onset_id"), str)
    }
    rows = {
        row["onset_id"]: row
        for row in cards
        if isinstance(row, dict) and isinstance(row.get("onset_id"), str)
    }
    if (
        len(expected) != len(targets)
        or len(rows) != len(cards)
        or set(rows) != set(expected)
    ):
        raise ValueError("enemy card decisions must cover targets exactly")
    for onset_id, row in rows.items():
        target = expected[onset_id]
        card = row.get("card")
        metadata_card = (
            card[4:]
            if isinstance(card, str) and card.startswith("evo-")
            else card
        )
        if card == "the-log" or metadata_card not in CARD_METADATA:
            raise ValueError(f"{onset_id}: unresolved or invalid card identity")
        expected_kind = target["kind"]
        actual_kind = CARD_METADATA[metadata_card]["kind"]
        if expected_kind == "spell":
            if (
                actual_kind != "spell"
                or row.get("identity_frame_index") is not None
                or row.get("identity_artifacts") != []
            ):
                raise ValueError(f"{onset_id}: invalid spell identity evidence")
        elif (
            actual_kind not in {"troop", "building"}
            or row.get("visibility") != "clear"
            or row.get("identity_frame_index")
            != target.get("identity_frame_index")
            or row.get("identity_artifacts")
            != target.get("identity_artifacts")
            or not isinstance(row.get("identity_artifacts"), list)
            or not row.get("identity_artifacts")
            or (
                len(row["identity_artifacts"]) < 2
                and target.get("identity_render_options", {}).get("mode")
                != "neighbor_candidates"
            )
        ):
            raise ValueError(f"{onset_id}: invalid delayed identity evidence")
        if row.get("confidence") != "direct":
            raise ValueError(f"{onset_id}: card identity is not direct")


def validate_enemy_existence_decisions(
    document: dict[str, object],
    package: dict[str, object],
) -> None:
    candidates = package.get("candidates")
    decisions = document.get("decisions")
    if not isinstance(candidates, list) or not isinstance(decisions, list):
        raise ValueError("existence candidates and decisions must be lists")
    expected = {
        row["onset_id"]: row
        for row in candidates
        if isinstance(row, dict) and isinstance(row.get("onset_id"), str)
    }
    rows = {
        row["onset_id"]: row
        for row in decisions
        if isinstance(row, dict) and isinstance(row.get("onset_id"), str)
    }
    if (
        len(expected) != len(candidates)
        or len(rows) != len(decisions)
        or set(rows) != set(expected)
    ):
        raise ValueError(
            "existence decisions must cover package candidates exactly"
        )
    for onset_id, row in rows.items():
        exists = row.get("overlap_event_exists")
        if not isinstance(exists, bool):
            raise ValueError(f"{onset_id}: invalid existence verdict")
        evidence = row.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"{onset_id}: existence evidence is required")
        frame = row.get("event_frame_index")
        if not exists:
            if frame is not None:
                raise ValueError(f"{onset_id}: rejected event has a frame")
            continue
        approximate = expected[onset_id].get("event_frame_index")
        if (
            not isinstance(frame, int)
            or not isinstance(approximate, int)
            or abs(frame - approximate) > 25
        ):
            raise ValueError(
                f"{onset_id}: corrected event frame is outside evidence horizon"
            )
        resolved = evidence.get(
            "secondary_persists_or_resolves_after",
            evidence.get("secondary_persists_after"),
        )
        schema_version = package.get("decision_schema_version")
        if schema_version == 2:
            direct_actor = evidence.get("direct_new_actor") is True
            side_valid = row.get("side") == "unresolved"
        else:
            # Compatibility for retained benchmark artifacts produced before
            # existence and side were split into independent decisions.
            direct_actor = evidence.get("direct_enemy_side") is True
            side_valid = row.get("side") == "enemy"
        if (
            evidence.get("secondary_absent_before") is not True
            or evidence.get("secondary_appears_at_marker") is not True
            or resolved is not True
            or not direct_actor
            or not side_valid
        ):
            raise ValueError(
                f"{onset_id}: accepted event lacks direct sequence evidence"
            )
        sampled = expected[onset_id].get("sampled_frame_indices")
        if (
            schema_version == 2
            and (
                not isinstance(sampled, list)
                or not sampled
                or frame not in sampled
            )
        ):
            raise ValueError(
                f"{onset_id}: corrected frame is not a labeled sampled frame"
            )


def validate_enemy_side_check_decisions(
    document: dict[str, object],
    package: dict[str, object],
) -> None:
    candidates = package.get("candidates")
    decisions = document.get("decisions")
    if not isinstance(candidates, list) or not isinstance(decisions, list):
        raise ValueError("side candidates and decisions must be lists")
    expected_ids = {
        row["onset_id"]
        for row in candidates
        if isinstance(row, dict) and isinstance(row.get("onset_id"), str)
    }
    rows = {
        row["onset_id"]: row
        for row in decisions
        if isinstance(row, dict) and isinstance(row.get("onset_id"), str)
    }
    if (
        len(expected_ids) != len(candidates)
        or len(rows) != len(decisions)
        or set(rows) != expected_ids
    ):
        raise ValueError("side decisions must cover candidates exactly")
    for onset_id, row in rows.items():
        side = row.get("side")
        if side not in {"own", "enemy", "unresolved"}:
            raise ValueError(f"{onset_id}: invalid side {side!r}")
        direct = row.get("direct")
        if side in {"own", "enemy"} and direct is not True:
            raise ValueError(f"{onset_id}: decided side lacks direct evidence")
        if side == "unresolved" and direct is not False:
            raise ValueError(
                f"{onset_id}: unresolved side must use direct=false"
            )
        if package.get("decision_schema_version") != 2:
            continue
        team_indicator = row.get("team_indicator")
        origin = row.get("origin")
        motion = row.get("motion")
        if team_indicator not in {"red", "blue", None}:
            raise ValueError(f"{onset_id}: invalid team indicator")
        if origin not in {"upper", "lower", "own-release", None}:
            raise ValueError(f"{onset_id}: invalid origin")
        if motion not in {"downward", "upward", None}:
            raise ValueError(f"{onset_id}: invalid motion")
        if side == "enemy" and not (
            team_indicator == "red"
            and (origin == "upper" or motion == "downward")
        ):
            raise ValueError(
                f"{onset_id}: enemy side lacks red plus origin/direction"
            )
        if side == "own" and not (
            team_indicator == "blue"
            or origin in {"lower", "own-release"}
            or motion == "upward"
        ):
            raise ValueError(
                f"{onset_id}: own side lacks direct own evidence"
            )
        if side == "unresolved" and any(
            value is not None for value in (team_indicator, origin, motion)
        ):
            raise ValueError(
                f"{onset_id}: unresolved side must not invent evidence"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_state(
    path: Path,
    *,
    run_id: str,
    profile: str,
    allow_profile_change: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "pipeline_version": 1,
            "workflow_version": 7,
            "run_id": run_id,
            "profile": profile,
            "status": "running",
            "jobs": {},
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("run_id") != run_id:
        raise ValueError("pipeline state run_id does not match manifest")
    if state.get("profile") != profile and not allow_profile_change:
        raise ValueError(
            "pipeline profile changed; use a new run or explicitly archive "
            "pipeline_state.json"
        )
    if state.get("profile") != profile:
        previous = state.get("profile")
        history = state.setdefault("profile_history", [])
        history.append({"from": previous, "to": profile})
        state["profile"] = profile
    if not isinstance(state.get("jobs"), dict):
        raise ValueError("pipeline state jobs must be an object")
    return state


def atomic_write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def accumulated_weighted_tokens(state: dict[str, Any]) -> float:
    return sum(
        float(row.get("weighted_tokens") or 0)
        for row in state.get("jobs", {}).values()
        if row.get("status") == "succeeded"
    )


def job_fingerprint(
    *,
    package: Path,
    prompt: Path,
    model_spec: ModelSpec,
) -> dict[str, Any]:
    package_document = json.loads(package.read_text(encoding="utf-8"))
    run_dir = package.resolve().parent.parent
    evidence_hashes: dict[str, str] = {}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
            return
        if isinstance(value, list):
            for nested in value:
                visit(nested)
            return
        if not isinstance(value, str):
            return
        candidate = Path(value)
        if candidate.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            return
        resolved = (
            candidate if candidate.is_absolute() else run_dir / candidate
        ).resolve()
        if not resolved.is_relative_to(run_dir):
            raise ValueError(f"package image escapes run directory: {value!r}")
        if not resolved.is_file():
            raise FileNotFoundError(f"package image does not exist: {resolved}")
        evidence_hashes[value] = sha256_file(resolved)

    visit(package_document)
    repository_root = Path(__file__).resolve().parents[2]
    worker_path = (
        repository_root
        / "scripts"
        / "codex_annotation"
        / "run_model_worker.py"
    )
    return {
        "fingerprint_version": 2,
        "package_sha256": sha256_file(package),
        "evidence_sha256": dict(sorted(evidence_hashes.items())),
        # Keep the template hash separate from run_model_worker's hash of the
        # fully rendered prompt.  Reusing the same key made every completed job
        # look stale on the next invocation.
        "prompt_template_sha256": sha256_file(prompt),
        "model": model_spec.model,
        "reasoning_effort": model_spec.reasoning_effort,
        "cost_multiplier": model_spec.cost_multiplier,
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "worker_sha256": sha256_file(worker_path),
        "workflow_version": 7,
    }


def completed_job_matches(
    row: dict[str, Any] | None,
    *,
    fingerprint: dict[str, Any],
    output: Path,
) -> bool:
    if not isinstance(row, dict) or row.get("status") != "succeeded":
        return False
    if any(row.get(key) != value for key, value in fingerprint.items()):
        return False
    return output.is_file() and row.get("output_sha256") == sha256_file(output)
