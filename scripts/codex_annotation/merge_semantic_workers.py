from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.annotation_stages import WORKFLOW_VERSION
from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.eval.action_eval import CARD_ALIASES


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _card(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("card must be a non-empty canonical slug")
    normalized = value.strip().lower().replace("_", "-")
    if normalized.startswith("evo-"):
        normalized = normalized[4:]
    if normalized == "the-log":
        normalized = "log"
    normalized = CARD_ALIASES.get(normalized, normalized)
    if normalized not in CARD_METADATA:
        raise ValueError(f"unknown canonical card slug {value!r}")
    return normalized


def _event_id(side: str, frame: int, card: str) -> str:
    return f"event-{side}-{frame:06d}-{card}"


def _normalize_pending_addition(
    row: dict[str, Any],
    *,
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Attach a deferred held-card release to evidence surrounding its release."""
    normalized = dict(row)
    frame = int(normalized["event_frame_index"])
    candidates = [
        candidate
        for candidate in manifest["candidate_discovery"]["own_candidates"]
        if candidate["inspection_start_frame"]
        <= frame
        < candidate["inspection_end_frame_exclusive"]
    ]
    if not candidates:
        raise ValueError(f"pending own event frame {frame} has no own candidate")
    candidate = min(
        candidates,
        key=lambda value: abs(value["approximate_frame_index"] - frame),
    )
    candidate_id = candidate["candidate_id"]
    suffix = candidate_id.replace(":", "-")
    review_index = _read(run_dir / "review_index.json")
    earliest = frame + max(1, round(0.5 * float(manifest["fps"])))
    latest = frame + max(1, round(1.5 * float(manifest["fps"])))
    release_rows = [
        review
        for review in review_index.get("reviews", [])
        if review.get("event_id") == f"release-{suffix}"
        and review.get("purpose") == "own_confirmation"
        and review["start_frame"] <= latest
        and review["end_frame_exclusive"] > earliest
    ]
    if not release_rows:
        raise ValueError(
            f"pending own event frame {frame} has no bounded release sheet "
            f"for {candidate_id}"
        )
    release = min(release_rows, key=lambda value: value["start_frame"])
    confirmation_frame = max(earliest, release["start_frame"])
    if confirmation_frame >= release["end_frame_exclusive"]:
        raise ValueError(f"release sheet for {candidate_id} misses confirmation")
    normalized.update(
        {
            "candidate_id": candidate_id,
            "verification_artifacts": [f"reviews/verify-{suffix}.jpg"],
            "confirmation_frame_index": confirmation_frame,
            "confirmation_artifacts": [
                f"reviews/{Path(release['path']).name}"
            ],
        }
    )
    return normalized


def _own_event(row: dict[str, Any]) -> dict[str, Any]:
    card = _card(row["card"])
    frame = int(row["event_frame_index"])
    return {
        "event_id": _event_id("own", frame, card),
        "candidate_id": row["candidate_id"],
        "side": "own",
        "card": card,
        "event_frame_index": frame,
        "evidence": row["evidence"],
        "ambiguity": "none",
        "verification_artifacts": row["verification_artifacts"],
        "confirmation_frame_index": row["confirmation_frame_index"],
        "confirmation_artifacts": row["confirmation_artifacts"],
        "own_confirmation": row["own_confirmation"],
        "identity_frame_index": None,
        "identity_artifacts": [],
    }


def _enemy_event(
    onset: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    card = _card(identity["card"])
    frame = int(onset["event_frame_index"])
    kind = CARD_METADATA[card]["kind"]
    is_spell = kind == "spell"
    expected_kind = "spell" if is_spell else "unit_or_building"
    if onset.get("kind") != expected_kind:
        raise ValueError(
            f"{onset['onset_id']}: onset kind {onset.get('kind')!r} "
            f"does not match card {card!r}"
        )
    observed = onset["evidence"]
    evidence = {
        "elixir_drop": observed["elixir_drop"],
        "hand_transition": observed["hand_transition"],
        "deployment_onset": observed["deployment_onset"],
        "first_visible_object": observed["first_visible_object"],
        "side_direction": bool(
            identity.get("side_evidence", {}).get("direct")
        ),
        "impact_sequence": observed["impact_sequence"],
    }
    return {
        "event_id": _event_id("enemy", frame, card),
        "candidate_id": onset["candidate_id"],
        "side": "enemy",
        "card": card,
        "event_frame_index": frame,
        "evidence": evidence,
        "ambiguity": "none",
        "verification_artifacts": onset["verification_artifacts"],
        "confirmation_frame_index": None,
        "confirmation_artifacts": [],
        "own_confirmation": None,
        "identity_frame_index": (
            None if is_spell else identity["identity_frame_index"]
        ),
        "identity_artifacts": (
            [] if is_spell else identity["identity_artifacts"]
        ),
    }


def _ensure_event_context(
    event: dict[str, Any],
    *,
    run_dir: Path,
    reviews: list[dict[str, Any]],
    segment_start: int,
    segment_end: int,
) -> dict[str, Any]:
    """Add adjacent indexed review sheets needed by final validation.

    An event may sit exactly on a non-overlapping scan-sheet boundary.  The
    worker's selected sheet then proves the event and its future, while the
    immediately preceding indexed arena sheet is needed for the fixed absence
    context.  Adding that already-blind evidence is deterministic and does not
    alter the semantic decision.
    """

    side = event["side"]
    purpose = "own_context" if side == "own" else "arena"
    frame = int(event["event_frame_index"])
    if side == "own":
        required = set(
            range(max(segment_start, frame - 1), min(segment_end, frame + 3))
        )
    else:
        required = set(
            range(max(segment_start, frame - 4), min(segment_end, frame + 11))
        )

    allowed_verification_refs: set[str] = set()
    indexed_by_ref = {}
    for row in reviews:
        path = Path(row["path"])
        try:
            reference = str(path.resolve().relative_to(run_dir))
        except ValueError:
            continue
        if row.get("purpose") in {"arena", "full", "own_context"}:
            allowed_verification_refs.add(reference)
        if row.get("purpose") == purpose:
            indexed_by_ref[reference] = row
    artifacts = [
        reference
        for reference in event["verification_artifacts"]
        if reference in allowed_verification_refs
    ]
    covered = {
        value
        for reference in artifacts
        if reference in indexed_by_ref
        for value in range(
            indexed_by_ref[reference]["start_frame"],
            indexed_by_ref[reference]["end_frame_exclusive"],
        )
    }
    missing = required - covered
    available = {
        reference: row
        for reference, row in indexed_by_ref.items()
        if reference not in artifacts
    }
    while missing:
        candidates = [
            (
                -len(
                    missing
                    & set(range(row["start_frame"], row["end_frame_exclusive"]))
                ),
                abs(row["start_frame"] - frame),
                reference,
                row,
            )
            for reference, row in available.items()
            if missing
            & set(range(row["start_frame"], row["end_frame_exclusive"]))
        ]
        if not candidates:
            raise ValueError(
                f"{event['event_id']}: indexed {purpose} evidence cannot cover "
                f"required context; missing frame {min(missing)}"
            )
        _, _, reference, row = min(candidates)
        artifacts.append(reference)
        missing -= set(range(row["start_frame"], row["end_frame_exclusive"]))
        available.pop(reference)
    return {**event, "verification_artifacts": artifacts}


def _ensure_candidate_support(
    event: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Bind a corrected enemy onset to the scan window containing its frame."""

    if event["side"] != "enemy":
        return event
    frame = int(event["event_frame_index"])
    windows = manifest["candidate_discovery"]["enemy_scan_windows"]
    current = next(
        (row for row in windows if row["candidate_id"] == event["candidate_id"]),
        None,
    )
    if current is not None and (
        current["inspection_start_frame"]
        <= frame
        < current["inspection_end_frame_exclusive"]
    ):
        return event
    supporting = [
        row
        for row in windows
        if row["inspection_start_frame"]
        <= frame
        < row["inspection_end_frame_exclusive"]
    ]
    if not supporting:
        raise ValueError(
            f"{event['event_id']}: corrected enemy frame has no supporting scan window"
        )
    selected = min(
        supporting,
        key=lambda row: (
            0 if row["candidate_id"].endswith(":p1") else 1,
            row["inspection_end_frame_exclusive"]
            - row["inspection_start_frame"],
            row["candidate_id"],
        ),
    )
    return {**event, "candidate_id": selected["candidate_id"]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge independent own, enemy-onset, and enemy-identity judgments."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model-label", default="split-workers")
    parser.add_argument("--reasoning-effort", default="mixed")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    own = _read(run_dir / "own_semantics.json")
    onsets = _read(run_dir / "enemy_onsets.json")
    identities = _read(run_dir / "enemy_identities.json")
    pending_path = run_dir / "own_pending_adjudication.json"
    pending = _read(pending_path) if pending_path.is_file() else None
    cards_path = run_dir / "enemy_cards.json"
    cards = _read(cards_path) if cards_path.is_file() else None
    run_id = manifest["run_id"]
    documents = [
        ("own_semantics", own),
        ("enemy_onsets", onsets),
        ("enemy_identities", identities),
    ]
    if pending is not None:
        documents.append(("own_pending_adjudication", pending))
    if cards is not None:
        documents.append(("enemy_cards", cards))
    for name, document in documents:
        if document.get("run_id") != run_id:
            raise ValueError(f"{name} run_id does not match manifest")

    onset_rows = onsets.get("onsets")
    identity_rows = identities.get("decisions")
    if not isinstance(onset_rows, list) or not isinstance(identity_rows, list):
        raise ValueError("onsets and identity decisions must be lists")
    identity_by_id = {
        row["onset_id"]: row for row in identity_rows if isinstance(row, dict)
    }
    onset_ids = {
        row["onset_id"] for row in onset_rows if isinstance(row, dict)
    }
    if len(identity_by_id) != len(identity_rows) or set(identity_by_id) != onset_ids:
        raise ValueError("enemy identities must cover every onset exactly once")

    own_rows = own.get("events")
    if not isinstance(own_rows, list):
        raise ValueError("own events must be a list")
    if pending is not None:
        remove = {
            (
                row["candidate_id"],
                _card(row["card"]),
                int(row["event_frame_index"]),
            )
            for row in pending.get("remove_events", [])
        }
        own_rows = [
            row
            for row in own_rows
            if (
                row["candidate_id"],
                _card(row["card"]),
                int(row["event_frame_index"]),
            )
            not in remove
        ]
        own_rows.extend(
            _normalize_pending_addition(
                row,
                run_dir=run_dir,
                manifest=manifest,
            )
            for row in pending.get("add_events", [])
        )
    card_by_id = None
    if cards is not None:
        card_rows = cards.get("cards")
        if not isinstance(card_rows, list):
            raise ValueError("enemy_cards cards must be a list")
        card_by_id = {
            row["onset_id"]: row for row in card_rows if isinstance(row, dict)
        }
        retained = {
            onset_id
            for onset_id, row in identity_by_id.items()
            if row.get("event_exists") is True
            and row.get("side") == "enemy"
        }
        if len(card_by_id) != len(card_rows) or set(card_by_id) != retained:
            raise ValueError("enemy cards must cover every retained enemy onset")
    events = [_own_event(row) for row in own_rows]
    for row in onset_rows:
        identity = identity_by_id[row["onset_id"]]
        if identity.get("event_exists") is not True:
            continue
        if identity.get("side") == "own":
            continue
        if identity.get("side") != "enemy":
            raise ValueError(
                f"{row['onset_id']}: identity side must be own or enemy"
            )
        if card_by_id is not None:
            identity = {
                **identity,
                **card_by_id[row["onset_id"]],
            }
        events.append(_enemy_event(row, identity))
    review_index = _read(run_dir / "review_index.json")
    segment_start = int(manifest["segment"]["start_frame"])
    segment_end = int(manifest["segment"]["end_frame_exclusive"])
    events = [_ensure_candidate_support(event, manifest) for event in events]
    events = [
        _ensure_event_context(
            event,
            run_dir=run_dir,
            reviews=review_index.get("reviews", []),
            segment_start=segment_start,
            segment_end=segment_end,
        )
        for event in events
    ]
    events.sort(key=lambda row: (row["event_frame_index"], row["side"], row["card"]))
    event_ids = [row["event_id"] for row in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("merged semantic events contain duplicate event IDs")

    output = {
        "run_id": run_id,
        "stage": "verification",
        "stage_version": WORKFLOW_VERSION,
        "annotation_session_id": args.session_id,
        "model": args.model_label,
        "reasoning_effort": args.reasoning_effort,
        "instructions": (
            "Merged from independent own-release, enemy-onset, and enemy-identity "
            "workers; localization is intentionally excluded."
        ),
        "events": events,
        "rejected_candidates": own.get("rejected_candidates", []),
        "adjudications": [],
    }
    atomic_write_json(run_dir / "verification.json", output)
    print(
        json.dumps(
            {
                "output": str(run_dir / "verification.json"),
                "own_events": len(own_rows),
                "enemy_events": sum(
                    1
                    for row in identity_rows
                    if row.get("event_exists") is True
                    and row.get("side") == "enemy"
                ),
                "total_events": len(events),
            }
        )
    )


if __name__ == "__main__":
    main()
