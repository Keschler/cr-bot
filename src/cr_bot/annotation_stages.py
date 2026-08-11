from __future__ import annotations

from copy import deepcopy
import fcntl
from pathlib import Path
from typing import Any

from cr_bot.annotation_harness import (
    AMBIGUITIES,
    ANNOTATION_TYPE,
    EVIDENCE_KEYS,
    LOCATION_RULES,
    OWN_CONFIRMATION_KEYS,
    atomic_write_json,
    sha256_file,
    utc_now_iso,
)
from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.features.action_space import ACTION_GRID, get_card_deploy_mask


WORKFLOW_VERSION = 7
VERIFICATION_PURPOSES = {"arena", "own_context", "full"}
OWN_CONFIRMATION_PURPOSES = {"own_confirmation"}
IDENTITY_PURPOSES = {"identity"}
LOCALIZATION_MACRO_PURPOSES = {"macro"}
LOCALIZATION_GRID_PURPOSES = {"grid"}
COMPLETENESS_PURPOSES = {"arena", "own_context", "full"}
OWN_CONFIRMATION_REQUIRED_TRUE = {
    "release_confirmed",
    "elixir_spend_persisted",
    "post_release_effect",
}


def write_stage_templates(run_dir: Path, manifest: dict[str, Any]) -> None:
    run_id = manifest["run_id"]
    segment = manifest["segment"]
    verification = {
        "run_id": run_id,
        "stage": "verification",
        "stage_version": WORKFLOW_VERSION,
        "annotation_session_id": "",
        "model": "",
        "reasoning_effort": "",
        "instructions": (
            "Decide event existence, side, card, and exact source frame. Own events "
            "require a later post-release HUD+arena confirmation that rejects canceled "
            "drags. A separate fresh-session release review must corroborate every own "
            "event before verification can be checkpointed. Enemy troops/buildings require "
            "a later, grid-free identity review "
            "after spawn effects clear. Do not assign a location frame or cell."
        ),
        "events": [],
        "rejected_candidates": [],
        "adjudications": [],
    }
    release_review = {
        "run_id": run_id,
        "stage": "release_review",
        "stage_version": WORKFLOW_VERSION,
        "annotation_session_id": "",
        "model": "",
        "reasoning_effort": "",
        "instructions": (
            "Use a fresh Codex conversation to independently inspect only the cited "
            "own_confirmation sheets. For every proposed own event, confirm that a "
            "release—not a canceled drag—is visible. Do not change event identity, "
            "timing, or localization in this stage."
        ),
        "reviews": [],
    }
    localization = {
        "run_id": run_id,
        "stage": "localization",
        "stage_version": WORKFLOW_VERSION,
        "annotation_session_id": "",
        "model": "",
        "reasoning_effort": "",
        "instructions": (
            "Populate only after verification is checkpointed. Each event requires "
            "a coarse macro review followed by a tight labeled grid review."
        ),
        "locations": [],
        "adjudications": [],
    }
    completeness = {
        "run_id": run_id,
        "stage": "completeness",
        "stage_version": WORKFLOW_VERSION,
        "annotation_session_id": "",
        "model": "",
        "reasoning_effort": "",
        "instructions": (
            "Review the complete interval independently for each side. Ranges must "
            "cover the full half-open segment and unresolved candidates must be empty."
        ),
        "sweeps": [
            {
                "sweep_id": "own-final",
                "side": "own",
                "reviewed_ranges": [],
                "review_artifacts": [],
                "discovered_event_ids": [],
                "unresolved_candidate_ids": [],
                "completed": False,
                "notes": "",
            },
            {
                "sweep_id": "enemy-final",
                "side": "enemy",
                "reviewed_ranges": [],
                "review_artifacts": [],
                "discovered_event_ids": [],
                "unresolved_candidate_ids": [],
                "completed": False,
                "notes": "",
            },
        ],
        "segment_to_cover": [
            segment["start_frame"],
            segment["end_frame_exclusive"],
        ],
    }
    checkpoints = {
        "run_id": run_id,
        "workflow_version": WORKFLOW_VERSION,
        "release_review": None,
        "verification": None,
        "localization": None,
        "completeness": None,
    }
    atomic_write_json(run_dir / "verification.json", verification)
    atomic_write_json(run_dir / "release_review.json", release_review)
    atomic_write_json(run_dir / "localization.json", localization)
    atomic_write_json(run_dir / "completeness.json", completeness)
    atomic_write_json(run_dir / "checkpoints.json", checkpoints)
    atomic_write_json(run_dir / "review_index.json", {"run_id": run_id, "reviews": []})


def record_review(
    *,
    run_dir: Path,
    output_path: Path,
    purpose: str,
    start_frame: int,
    end_frame: int,
    candidate_id: str | None,
    event_id: str | None,
) -> None:
    index_path = run_dir / "review_index.json"
    if not index_path.exists():
        return
    lock_path = run_dir / ".review_index.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        index = _read_object(index_path)
        resolved = output_path.resolve()
        reviews = [
            row
            for row in index.get("reviews", [])
            if Path(row["path"]).resolve() != resolved
        ]
        reviews.append(
            {
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "purpose": purpose,
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame,
                "candidate_id": candidate_id,
                "event_id": event_id,
                "created_at": utc_now_iso(),
            }
        )
        index["reviews"] = reviews
        atomic_write_json(index_path, index)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def require_verification_checkpoint(run_dir: Path, event_id: str) -> dict[str, Any]:
    manifest = _read_object(run_dir / "manifest.json")
    if manifest.get("workflow_version") != WORKFLOW_VERSION:
        raise ValueError(
            f"event-scoped localization requires a staged v{WORKFLOW_VERSION} run"
        )
    checkpoints = _read_object(run_dir / "checkpoints.json")
    checkpoint = checkpoints.get("verification")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint verification before rendering localization")
    verification_path = run_dir / "verification.json"
    if checkpoint.get("sha256") != sha256_file(verification_path):
        raise ValueError("verification changed after its checkpoint; checkpoint it again")
    verification = _read_object(verification_path)
    event = next(
        (row for row in verification.get("events", []) if row.get("event_id") == event_id),
        None,
    )
    if event is None:
        raise KeyError(f"unknown verified event_id: {event_id}")
    return event


def checkpoint_stage(run_dir: Path, stage: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest = _read_object(run_dir / "manifest.json")
    if manifest.get("workflow_version") != WORKFLOW_VERSION:
        raise ValueError(
            f"stage checkpoints require a newly prepared v{WORKFLOW_VERSION} run"
        )
    checkpoints_path = run_dir / "checkpoints.json"
    checkpoints = _read_object(checkpoints_path)
    if stage == "release_review":
        verification = _read_object(run_dir / "verification.json")
        events = validate_verification(run_dir, manifest, verification)
        document = _read_object(run_dir / "release_review.json")
        reviews = validate_release_review(
            run_dir, manifest, document, verification["events"]
        )
        checkpoint = {
            "sha256": sha256_file(run_dir / "release_review.json"),
            "verification_sha256": sha256_file(run_dir / "verification.json"),
            "review_count": len(reviews),
            "annotation_session_id": document["annotation_session_id"],
            "model": document["model"],
            "reasoning_effort": document["reasoning_effort"],
            "checkpointed_at": utc_now_iso(),
        }
        checkpoints["release_review"] = checkpoint
        checkpoints["verification"] = None
        checkpoints["localization"] = None
        checkpoints["completeness"] = None
    elif stage == "verification":
        document = _read_object(run_dir / "verification.json")
        events = validate_verification(run_dir, manifest, document)
        if any(event["side"] == "own" for event in events):
            _current_release_review(run_dir, checkpoints, document)
        checkpoint = {
            "sha256": sha256_file(run_dir / "verification.json"),
            "event_count": len(events),
            "annotation_session_id": document["annotation_session_id"],
            "model": document["model"],
            "reasoning_effort": document["reasoning_effort"],
            "checkpointed_at": utc_now_iso(),
        }
        checkpoints["verification"] = checkpoint
        checkpoints["localization"] = None
        checkpoints["completeness"] = None
    elif stage == "localization":
        verification, verification_checkpoint = _current_verification(
            run_dir, checkpoints
        )
        document = _read_object(run_dir / "localization.json")
        locations = validate_localization(
            run_dir, manifest, document, verification["events"]
        )
        checkpoint = {
            "sha256": sha256_file(run_dir / "localization.json"),
            "verification_sha256": verification_checkpoint["sha256"],
            "location_count": len(locations),
            "annotation_session_id": document["annotation_session_id"],
            "model": document["model"],
            "reasoning_effort": document["reasoning_effort"],
            "checkpointed_at": utc_now_iso(),
        }
        checkpoints["localization"] = checkpoint
        checkpoints["completeness"] = None
    elif stage == "completeness":
        verification, verification_checkpoint = _current_verification(
            run_dir, checkpoints
        )
        _, localization_checkpoint = _current_localization(
            run_dir, checkpoints, verification_checkpoint
        )
        document = _read_object(run_dir / "completeness.json")
        sweeps = validate_completeness(
            run_dir,
            manifest,
            document,
            verification["events"],
            verification_session_id=verification["annotation_session_id"],
        )
        checkpoint = {
            "sha256": sha256_file(run_dir / "completeness.json"),
            "verification_sha256": verification_checkpoint["sha256"],
            "localization_sha256": localization_checkpoint["sha256"],
            "sweep_count": len(sweeps),
            "annotation_session_id": document["annotation_session_id"],
            "model": document["model"],
            "reasoning_effort": document["reasoning_effort"],
            "checkpointed_at": utc_now_iso(),
        }
        checkpoints["completeness"] = checkpoint
    else:
        raise ValueError(
            "stage must be release_review, verification, localization, or completeness"
        )
    atomic_write_json(checkpoints_path, checkpoints)
    return checkpoint


def validate_verification(
    run_dir: Path, manifest: dict[str, Any], document: dict[str, Any]
) -> list[dict[str, Any]]:
    _validate_stage_header(manifest, document, "verification")
    _validate_stage_provenance(document, "verification")
    events = document.get("events")
    if not isinstance(events, list):
        raise ValueError("verification events must be a list")
    start, end = _segment_bounds(manifest)
    validated: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    semantic_keys = {
        "event_id",
        "candidate_id",
        "side",
        "card",
        "event_frame_index",
        "evidence",
        "ambiguity",
        "verification_artifacts",
        "confirmation_frame_index",
        "confirmation_artifacts",
        "own_confirmation",
        "identity_frame_index",
        "identity_artifacts",
    }
    for index, raw in enumerate(events):
        if not isinstance(raw, dict) or set(raw) != semantic_keys:
            raise ValueError(
                f"verification event {index} must contain exactly "
                f"{', '.join(sorted(semantic_keys))}"
            )
        event = deepcopy(raw)
        event_id = event["event_id"]
        if not isinstance(event_id, str) or not event_id or event_id in event_ids:
            raise ValueError(f"verification event {index}: event_id must be unique")
        if not isinstance(event["candidate_id"], str) or not event["candidate_id"]:
            raise ValueError(f"verification event {index}: candidate_id is required")
        if event["side"] not in {"own", "enemy"}:
            raise ValueError(f"verification event {index}: invalid side")
        base_card = _base_card(event["card"])
        if base_card not in CARD_METADATA:
            raise ValueError(
                f"verification event {index}: unknown canonical card slug {event['card']!r}"
            )
        frame = event["event_frame_index"]
        if not isinstance(frame, int) or isinstance(frame, bool) or not start <= frame < end:
            raise ValueError(
                f"verification event {index}: event_frame_index outside segment"
            )
        expected_event_id = (
            f"event-{event['side']}-{frame:06d}-{event['card']}"
        )
        if event_id != expected_event_id:
            raise ValueError(
                f"verification event {index}: event_id must be {expected_event_id!r}"
            )
        event_ids.add(event_id)
        candidate_id = event["candidate_id"]
        allowed_candidate_prefixes = (
            ("own:", "completeness:own:")
            if event["side"] == "own"
            else ("enemy-scan:", "enemy-boundary:", "completeness:enemy:")
        )
        if not candidate_id.startswith(allowed_candidate_prefixes):
            raise ValueError(
                f"verification event {index}: candidate {candidate_id!r} "
                f"does not support side {event['side']!r}"
            )
        _validate_candidate_frame(manifest, index, candidate_id, frame)
        ambiguity = event["ambiguity"]
        if ambiguity not in AMBIGUITIES or ambiguity != "none":
            raise ValueError(
                f"verification event {index}: unresolved semantic ambiguity {ambiguity!r}"
            )
        evidence = event["evidence"]
        _validate_evidence(index, event["side"], base_card, evidence)
        artifact_rows = _validate_artifacts(
            run_dir,
            event["verification_artifacts"],
            allowed_purposes=VERIFICATION_PURPOSES,
            event_id=None,
            label=f"verification event {index}",
        )
        _validate_verification_context(
            index=index,
            side=event["side"],
            event_frame=frame,
            segment_start=start,
            segment_end=end,
            artifact_rows=artifact_rows,
        )
        _validate_own_confirmation(
            run_dir=run_dir,
            manifest=manifest,
            index=index,
            event=event,
            segment_start=start,
            segment_end=end,
        )
        _validate_identity_evidence(
            run_dir=run_dir,
            manifest=manifest,
            index=index,
            event=event,
            base_card=base_card,
            segment_start=start,
            segment_end=end,
        )
        validated.append(event)
    return validated


def validate_release_review(
    run_dir: Path,
    manifest: dict[str, Any],
    document: dict[str, Any],
    verification_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _validate_stage_header(manifest, document, "release_review")
    _validate_stage_provenance(document, "release_review")
    own_events = {
        event["event_id"]: event
        for event in verification_events
        if event["side"] == "own"
    }
    reviews = document.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("release_review reviews must be a list")
    if {row.get("event_id") for row in reviews if isinstance(row, dict)} != set(own_events):
        raise ValueError("release_review must cover every proposed own event exactly once")
    keys = {
        "event_id",
        "decision",
        "confirmation_frame_index",
        "confirmation_artifacts",
        "checks",
    }
    for index, raw in enumerate(reviews):
        if not isinstance(raw, dict) or set(raw) != keys:
            raise ValueError(
                f"release_review {index} must contain exactly {', '.join(sorted(keys))}"
            )
        event = own_events.get(raw["event_id"])
        if event is None:
            raise ValueError(f"release_review {index}: unknown own event_id")
        if raw["decision"] != "released":
            raise ValueError(
                f"release_review {index}: accepted own events must be independently "
                "confirmed as released; remove or reject the event first"
            )
        if raw["confirmation_frame_index"] != event["confirmation_frame_index"]:
            raise ValueError(
                f"release_review {index}: confirmation_frame_index must match verification"
            )
        if raw["confirmation_artifacts"] != event["confirmation_artifacts"]:
            raise ValueError(
                f"release_review {index}: confirmation_artifacts must match verification"
            )
        if raw["checks"] != event["own_confirmation"]:
            raise ValueError(f"release_review {index}: checks must match verification")
        _validate_own_confirmation(
            run_dir=run_dir,
            manifest=manifest,
            index=index,
            event=event,
            segment_start=_segment_bounds(manifest)[0],
            segment_end=_segment_bounds(manifest)[1],
        )
    return reviews


def validate_localization(
    run_dir: Path,
    manifest: dict[str, Any],
    document: dict[str, Any],
    verified_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _validate_stage_header(manifest, document, "localization")
    _validate_stage_provenance(document, "localization")
    locations = document.get("locations")
    if not isinstance(locations, list):
        raise ValueError("localization locations must be a list")
    by_event = {row["event_id"]: row for row in verified_events}
    if {row.get("event_id") for row in locations if isinstance(row, dict)} != set(
        by_event
    ):
        raise ValueError("localization must cover every verified event exactly once")
    start, end = _segment_bounds(manifest)
    validated = []
    location_keys = {
        "event_id",
        "location_frame_index",
        "location_rule",
        "cell",
        "ambiguity",
        "unavailable_reason",
        "macro_review_artifacts",
        "grid_review_artifacts",
        "adjudication_artifacts",
    }
    for index, raw in enumerate(locations):
        if not isinstance(raw, dict) or set(raw) != location_keys:
            raise ValueError(
                f"location {index} must contain exactly "
                f"{', '.join(sorted(location_keys))}"
            )
        location = deepcopy(raw)
        event_id = location["event_id"]
        if event_id not in by_event:
            raise ValueError(f"location {index}: unknown event_id")
        frame = location["location_frame_index"]
        if not isinstance(frame, int) or isinstance(frame, bool) or not start <= frame < end:
            raise ValueError(f"location {index}: invalid location_frame_index")
        rule = location["location_rule"]
        if rule not in LOCATION_RULES:
            raise ValueError(f"location {index}: invalid location_rule")
        cell = location["cell"]
        if cell is not None and not _valid_cell(cell):
            raise ValueError(f"location {index}: invalid [column,row] cell")
        if cell is not None:
            verified = by_event[event_id]
            _validate_legal_deployment_cell(
                index=index,
                side=verified["side"],
                card=verified["card"],
                cell=cell,
            )
        _validate_artifacts(
            run_dir,
            location["macro_review_artifacts"],
            allowed_purposes=LOCALIZATION_MACRO_PURPOSES,
            event_id=event_id,
            label=f"location {index} macro review",
        )
        _validate_artifacts(
            run_dir,
            location["grid_review_artifacts"],
            allowed_purposes=LOCALIZATION_GRID_PURPOSES,
            event_id=event_id,
            label=f"location {index} grid review",
        )
        if cell is None:
            if rule != "unavailable" or location["ambiguity"] != "unscorable":
                raise ValueError(
                    f"location {index}: null cell requires unavailable/unscorable"
                )
            if not isinstance(location["unavailable_reason"], str) or not location[
                "unavailable_reason"
            ].strip():
                raise ValueError(
                    f"location {index}: unavailable location needs a reason"
                )
            _validate_artifacts(
                run_dir,
                location["adjudication_artifacts"],
                allowed_purposes=LOCALIZATION_MACRO_PURPOSES
                | LOCALIZATION_GRID_PURPOSES,
                event_id=event_id,
                label=f"location {index} adjudication",
            )
            cited = set(location["macro_review_artifacts"]) | set(
                location["grid_review_artifacts"]
            )
            if cited & set(location["adjudication_artifacts"]):
                raise ValueError(
                    f"location {index}: adjudication must cite a separate review artifact"
                )
        else:
            if rule == "unavailable" or location["ambiguity"] != "none":
                raise ValueError(
                    f"location {index}: resolved cell requires a concrete rule and no ambiguity"
                )
            if location["unavailable_reason"] is not None:
                raise ValueError(
                    f"location {index}: resolved cell cannot have unavailable_reason"
                )
            if location["adjudication_artifacts"] not in ([], None):
                _validate_artifacts(
                    run_dir,
                    location["adjudication_artifacts"],
                    allowed_purposes=LOCALIZATION_MACRO_PURPOSES
                    | LOCALIZATION_GRID_PURPOSES,
                    event_id=event_id,
                    label=f"location {index} adjudication",
                )
        validated.append(location)
    return validated


def validate_completeness(
    run_dir: Path,
    manifest: dict[str, Any],
    document: dict[str, Any],
    verified_events: list[dict[str, Any]],
    *,
    verification_session_id: str,
) -> list[dict[str, Any]]:
    _validate_stage_header(manifest, document, "completeness")
    _validate_stage_provenance(document, "completeness")
    if document["annotation_session_id"] == verification_session_id:
        raise ValueError(
            "completeness must use a fresh annotation_session_id, separate from verification"
        )
    sweeps = document.get("sweeps")
    if not isinstance(sweeps, list):
        raise ValueError("completeness sweeps must be a list")
    by_side = {
        row.get("side"): row for row in sweeps if isinstance(row, dict)
    }
    if set(by_side) != {"own", "enemy"}:
        raise ValueError("exactly one own and one enemy completeness sweep are required")
    start, end = _segment_bounds(manifest)
    event_ids = {row["event_id"] for row in verified_events}
    for side in ("own", "enemy"):
        sweep = by_side[side]
        if sweep.get("completed") is not True:
            raise ValueError(f"{side} completeness sweep is not completed")
        unresolved = sweep.get("unresolved_candidate_ids")
        if not isinstance(unresolved, list) or unresolved:
            raise ValueError(
                f"{side} completeness sweep has unresolved candidates: {unresolved!r}"
            )
        discovered = sweep.get("discovered_event_ids")
        if not isinstance(discovered, list) or not set(discovered) <= event_ids:
            raise ValueError(
                f"{side} completeness discovered_event_ids must reference verified events"
            )
        expected_purposes = {"own_context"} if side == "own" else {"arena"}
        artifact_rows = _validate_artifacts(
            run_dir,
            sweep.get("review_artifacts"),
            allowed_purposes=expected_purposes,
            event_id=None,
            label=f"{side} completeness sweep",
        )
        artifact_ranges = sorted(
            {
                (row["start_frame"], row["end_frame_exclusive"])
                for row in artifact_rows
            }
        )
        declared_ranges = sweep.get("reviewed_ranges")
        valid_declared_ranges = (
            isinstance(declared_ranges, list)
            and all(
                isinstance(item, list)
                and len(item) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in item
                )
                for item in declared_ranges
            )
        )
        if (
            not valid_declared_ranges
            or sorted(tuple(item) for item in declared_ranges) != artifact_ranges
        ):
            raise ValueError(
                f"{side} reviewed_ranges must exactly match its indexed review artifacts"
            )
        if not _ranges_cover(declared_ranges, start, end):
            raise ValueError(
                f"{side} completeness artifacts do not cover [{start}, {end})"
            )
    return sweeps


def assemble_staged_decisions(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoints = _read_object(run_dir / "checkpoints.json")
    verification, verification_checkpoint = _current_verification(
        run_dir, checkpoints
    )
    localization, localization_checkpoint = _current_localization(
        run_dir, checkpoints, verification_checkpoint
    )
    completeness, completeness_checkpoint = _current_completeness(
        run_dir,
        checkpoints,
        verification_checkpoint,
        localization_checkpoint,
    )
    verified_events = validate_verification(run_dir, manifest, verification)
    locations = validate_localization(
        run_dir, manifest, localization, verified_events
    )
    sweeps = validate_completeness(
        run_dir,
        manifest,
        completeness,
        verified_events,
        verification_session_id=verification["annotation_session_id"],
    )
    locations_by_event = {row["event_id"]: row for row in locations}
    events = []
    for verified in verified_events:
        location = locations_by_event[verified["event_id"]]
        events.append(
            {
                "candidate_id": verified["candidate_id"],
                "event_id": verified["event_id"],
                "side": verified["side"],
                "card": verified["card"],
                "event_frame_index": verified["event_frame_index"],
                "location_frame_index": location["location_frame_index"],
                "location_rule": location["location_rule"],
                "cell": location["cell"],
                "evidence": verified["evidence"],
                "ambiguity": (
                    "unscorable"
                    if location["ambiguity"] == "unscorable"
                    else "none"
                ),
                "review_artifacts": (
                    verified["verification_artifacts"]
                    + verified["confirmation_artifacts"]
                    + verified["identity_artifacts"]
                    + location["macro_review_artifacts"]
                    + location["grid_review_artifacts"]
                    + (location["adjudication_artifacts"] or [])
                ),
            }
        )
    decisions = {
        "run_id": manifest["run_id"],
        "events": events,
        "rejected_candidates": verification.get("rejected_candidates", []),
        "completeness_sweeps": [
            {
                "side": sweep["side"],
                "completed": True,
                "notes": sweep.get("notes", ""),
            }
            for sweep in sweeps
        ],
        "adjudications": verification.get("adjudications", [])
        + localization.get("adjudications", []),
    }
    return decisions, checkpoints


def _current_verification(
    run_dir: Path, checkpoints: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = checkpoints.get("verification")
    if not isinstance(checkpoint, dict):
        raise ValueError("verification checkpoint is required")
    path = run_dir / "verification.json"
    if checkpoint.get("sha256") != sha256_file(path):
        raise ValueError("verification checkpoint is stale")
    return _read_object(path), checkpoint


def _current_release_review(
    run_dir: Path, checkpoints: dict[str, Any], verification: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = checkpoints.get("release_review")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "a fresh-session release_review checkpoint is required before verification"
        )
    verification_path = run_dir / "verification.json"
    if checkpoint.get("verification_sha256") != sha256_file(verification_path):
        raise ValueError("release_review depends on an obsolete verification document")
    path = run_dir / "release_review.json"
    if checkpoint.get("sha256") != sha256_file(path):
        raise ValueError("release_review checkpoint is stale")
    review = _read_object(path)
    if review.get("annotation_session_id") == verification.get("annotation_session_id"):
        raise ValueError("release_review must use a fresh annotation_session_id")
    validate_release_review(
        run_dir,
        _read_object(run_dir / "manifest.json"),
        review,
        verification["events"],
    )
    return review, checkpoint


def _current_localization(
    run_dir: Path,
    checkpoints: dict[str, Any],
    verification_checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = checkpoints.get("localization")
    if not isinstance(checkpoint, dict):
        raise ValueError("localization checkpoint is required")
    path = run_dir / "localization.json"
    if checkpoint.get("sha256") != sha256_file(path):
        raise ValueError("localization checkpoint is stale")
    if checkpoint.get("verification_sha256") != verification_checkpoint.get("sha256"):
        raise ValueError("localization depends on an obsolete verification checkpoint")
    return _read_object(path), checkpoint


def _current_completeness(
    run_dir: Path,
    checkpoints: dict[str, Any],
    verification_checkpoint: dict[str, Any],
    localization_checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = checkpoints.get("completeness")
    if not isinstance(checkpoint, dict):
        raise ValueError("completeness checkpoint is required")
    path = run_dir / "completeness.json"
    if checkpoint.get("sha256") != sha256_file(path):
        raise ValueError("completeness checkpoint is stale")
    if checkpoint.get("verification_sha256") != verification_checkpoint.get("sha256"):
        raise ValueError("completeness depends on obsolete verification")
    if checkpoint.get("localization_sha256") != localization_checkpoint.get("sha256"):
        raise ValueError("completeness depends on obsolete localization")
    return _read_object(path), checkpoint


def _validate_stage_header(
    manifest: dict[str, Any], document: dict[str, Any], stage: str
) -> None:
    if document.get("run_id") != manifest.get("run_id"):
        raise ValueError(f"{stage} run_id does not match manifest")
    if document.get("stage") != stage:
        raise ValueError(f"expected {stage} stage document")
    if document.get("stage_version") != WORKFLOW_VERSION:
        raise ValueError(f"{stage} stage_version must be {WORKFLOW_VERSION}")


def _validate_stage_provenance(document: dict[str, Any], stage: str) -> None:
    for key in ("annotation_session_id", "model", "reasoning_effort"):
        value = document.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{stage} requires non-empty {key}")


def _validate_verification_context(
    *,
    index: int,
    side: str,
    event_frame: int,
    segment_start: int,
    segment_end: int,
    artifact_rows: list[dict[str, Any]],
) -> None:
    required_purpose = "own_context" if side == "own" else "arena"
    relevant = [row for row in artifact_rows if row.get("purpose") == required_purpose]
    if not relevant:
        raise ValueError(
            f"verification event {index}: {side} event requires {required_purpose} evidence"
        )
    covered = {
        frame
        for row in relevant
        for frame in range(row["start_frame"], row["end_frame_exclusive"])
    }
    if side == "own":
        required_start = max(segment_start, event_frame - 1)
        required_end = min(segment_end, event_frame + 3)
    else:
        required_start = max(segment_start, event_frame - 4)
        required_end = min(segment_end, event_frame + 11)
    missing = [
        frame for frame in range(required_start, required_end) if frame not in covered
    ]
    if missing:
        raise ValueError(
            f"verification event {index}: evidence misses required context "
            f"[{required_start}, {required_end}); missing frames begin at {missing[0]}"
        )


def _validate_candidate_frame(
    manifest: dict[str, Any], index: int, candidate_id: str, event_frame: int
) -> None:
    """Prevent a compact candidate from being reused for an unrelated later play."""
    if candidate_id.startswith("completeness:"):
        return
    discovery = manifest.get("candidate_discovery", {})
    candidates = (
        discovery.get("own_candidates", [])
        + discovery.get("enemy_scan_windows", [])
    )
    candidate = next(
        (row for row in candidates if row.get("candidate_id") == candidate_id), None
    )
    if candidate is None:
        raise ValueError(
            f"verification event {index}: candidate {candidate_id!r} is not in the manifest"
        )
    if candidate_id.startswith("own:"):
        approximate = candidate.get("approximate_frame_index")
        # Manifest candidates are frame-local transition hints. Independent
        # slot-interval events use a completeness:own: ID and bypass this
        # association; ordinary candidates remain tightly bounded.
        support_radius = max(
            1,
            round(2.0 * float(manifest["fps"])),
        )
        segment_start, segment_end = _segment_bounds(manifest)
        start = max(segment_start, int(approximate) - support_radius)
        end = min(segment_end, int(approximate) + support_radius + 1)
    else:
        start = candidate.get("inspection_start_frame")
        end = candidate.get("inspection_end_frame_exclusive")
    if not isinstance(start, int) or not isinstance(end, int) or not start <= event_frame < end:
        raise ValueError(
            f"verification event {index}: event frame {event_frame} is outside "
            f"candidate {candidate_id!r} range [{start}, {end})"
        )


def _validate_identity_evidence(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    index: int,
    event: dict[str, Any],
    base_card: str,
    segment_start: int,
    segment_end: int,
) -> None:
    kind = CARD_METADATA[base_card]["kind"]
    requires_identity = event["side"] == "enemy" and kind != "spell"
    identity_frame = event["identity_frame_index"]
    identity_artifacts = event["identity_artifacts"]
    if not requires_identity:
        if identity_frame is not None or identity_artifacts != []:
            raise ValueError(
                f"verification event {index}: identity evidence is reserved for "
                "enemy troops and buildings"
            )
        return

    event_frame = event["event_frame_index"]
    clear_delay = max(1, round(0.5 * float(manifest["fps"])))
    earliest_clear_frame = min(segment_end - 1, event_frame + clear_delay)
    if (
        not isinstance(identity_frame, int)
        or isinstance(identity_frame, bool)
        or not segment_start <= identity_frame < segment_end
    ):
        raise ValueError(
            f"verification event {index}: enemy troop/building requires a valid "
            "identity_frame_index"
        )
    if identity_frame < earliest_clear_frame:
        raise ValueError(
            f"verification event {index}: identity_frame_index must be at least "
            f"{earliest_clear_frame}, after the spawn effect has had time to clear"
        )
    latest_identity_frame = min(
        segment_end - 1, event_frame + max(clear_delay, round(6.0 * float(manifest["fps"])))
    )
    if identity_frame > latest_identity_frame:
        raise ValueError(
            f"verification event {index}: identity_frame_index must be no later than "
            f"{latest_identity_frame}, while continuously tracking the same unit"
        )
    rows = _validate_artifacts(
        run_dir,
        identity_artifacts,
        allowed_purposes=IDENTITY_PURPOSES,
        event_id=f"identity-enemy-{event_frame:06d}",
        label=f"verification event {index} identity",
    )
    distinct_ranges = {
        (row["start_frame"], row["end_frame_exclusive"]) for row in rows
    }
    if len(distinct_ranges) < 2:
        raise ValueError(
            f"verification event {index}: enemy identity requires two separate "
            "post-effect views of the same body"
        )
    if not any(
        row["start_frame"] <= identity_frame < row["end_frame_exclusive"]
        for row in rows
    ):
        raise ValueError(
            f"verification event {index}: identity artifacts must contain "
            f"identity frame {identity_frame}"
        )


def _validate_own_confirmation(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    index: int,
    event: dict[str, Any],
    segment_start: int,
    segment_end: int,
) -> None:
    confirmation_frame = event["confirmation_frame_index"]
    confirmation_artifacts = event["confirmation_artifacts"]
    confirmation = event["own_confirmation"]
    if event["side"] != "own":
        if (
            confirmation_frame is not None
            or confirmation_artifacts != []
            or confirmation is not None
        ):
            raise ValueError(
                f"verification event {index}: own confirmation fields must be "
                "empty for enemy events"
            )
        return

    event_frame = event["event_frame_index"]
    release_delay = max(1, round(0.5 * float(manifest["fps"])))
    earliest_confirmation = event_frame + release_delay
    if earliest_confirmation >= segment_end:
        raise ValueError(
            f"verification event {index}: own event is too close to the segment "
            "end for post-release confirmation"
        )
    if (
        not isinstance(confirmation_frame, int)
        or isinstance(confirmation_frame, bool)
        or not segment_start <= confirmation_frame < segment_end
    ):
        raise ValueError(
            f"verification event {index}: own event requires a valid "
            "confirmation_frame_index"
        )
    if confirmation_frame < earliest_confirmation:
        raise ValueError(
            f"verification event {index}: confirmation_frame_index must be at "
            f"least {earliest_confirmation}, after release or cancellation resolves"
        )
    latest_confirmation = min(
        segment_end - 1,
        event_frame + max(release_delay, round(1.5 * float(manifest["fps"]))),
    )
    if confirmation_frame > latest_confirmation:
        raise ValueError(
            f"verification event {index}: confirmation_frame_index must be no later "
            f"than {latest_confirmation}, before later plays can contaminate it"
        )
    rows = _validate_artifacts(
        run_dir,
        confirmation_artifacts,
        allowed_purposes=OWN_CONFIRMATION_PURPOSES,
        event_id=f"release-{event['candidate_id'].replace(':', '-')}",
        label=f"verification event {index} own confirmation",
    )
    if not any(
        row["start_frame"] <= confirmation_frame < row["end_frame_exclusive"]
        for row in rows
    ):
        raise ValueError(
            f"verification event {index}: confirmation artifacts must contain "
            f"confirmation frame {confirmation_frame}"
        )
    if any(
        row["start_frame"] < event_frame
        or row["end_frame_exclusive"] > latest_confirmation + 1
        for row in rows
    ):
        raise ValueError(
            f"verification event {index}: confirmation sheets must stay within the "
            "bounded post-onset window"
        )
    if (
        not isinstance(confirmation, dict)
        or set(confirmation) != OWN_CONFIRMATION_KEYS
        or any(
            confirmation[key] is not True
            for key in OWN_CONFIRMATION_REQUIRED_TRUE
        )
        or confirmation["hand_cycle_completed"] not in {True, False, None}
    ):
        raise ValueError(
            f"verification event {index}: own confirmation requires release, "
            "persistent elixir spend, and a post-release effect; hand cycle may "
            "remain unresolved while the replacement-slot cooldown is visible"
        )


def _validate_legal_deployment_cell(
    *, index: int, side: str, card: str, cell: list[int]
) -> None:
    base_card = _base_card(card)
    assert base_card is not None
    mask = get_card_deploy_mask(base_card)
    column, row = cell
    mask_row = row if side == "own" else ACTION_GRID.rows - 1 - row
    if not bool(mask[mask_row, column]):
        raise ValueError(
            f"location {index}: cell {cell} is not a legal {side} deployment "
            f"for {card}"
        )


def _validate_evidence(
    index: int, side: str, base_card: str, evidence: Any
) -> None:
    if not isinstance(evidence, dict) or set(evidence) != set(EVIDENCE_KEYS):
        raise ValueError(f"verification event {index}: invalid evidence keys")
    if any(
        value is not True and value is not False and value is not None
        for value in evidence.values()
    ):
        raise ValueError(f"verification event {index}: invalid evidence value")
    kind = CARD_METADATA[base_card]["kind"]
    if side == "own":
        valid = evidence["elixir_drop"] is True and (
            evidence["hand_transition"] is True
            or evidence["deployment_onset"] is True
        )
    elif kind == "spell":
        valid = (
            evidence["side_direction"] is True
            and evidence["impact_sequence"] is True
        )
    else:
        valid = (
            evidence["first_visible_object"] is True
            and evidence["deployment_onset"] is True
            and evidence["side_direction"] is True
        )
    if not valid:
        raise ValueError(f"verification event {index}: insufficient direct evidence")


def _validate_artifacts(
    run_dir: Path,
    artifacts: Any,
    *,
    allowed_purposes: set[str],
    event_id: str | None,
    label: str,
) -> list[dict[str, Any]]:
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or any(not isinstance(path, str) or not path for path in artifacts)
    ):
        raise ValueError(f"{label}: non-empty artifact path list required")
    review_index = _read_object(run_dir / "review_index.json")
    indexed = {
        str(Path(row["path"]).resolve()): row
        for row in review_index.get("reviews", [])
    }
    validated_rows = []
    for artifact in artifacts:
        path = Path(artifact)
        if not path.is_absolute():
            path = run_dir / path
        resolved = str(path.resolve())
        row = indexed.get(resolved)
        if row is None:
            raise ValueError(f"{label}: artifact was not rendered by the harness: {path}")
        if row.get("purpose") not in allowed_purposes:
            raise ValueError(
                f"{label}: {path.name} has purpose {row.get('purpose')!r}, "
                f"expected one of {sorted(allowed_purposes)}"
            )
        if event_id is not None and row.get("event_id") != event_id:
            raise ValueError(f"{label}: artifact is not scoped to event {event_id}")
        if not path.is_file() or row.get("sha256") != sha256_file(path):
            raise ValueError(f"{label}: artifact is missing or changed: {path}")
        validated_rows.append(row)
    return validated_rows


def _ranges_cover(ranges: Any, start: int, end: int) -> bool:
    if not isinstance(ranges, list):
        return False
    normalized = []
    for item in ranges:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in item)
            or item[0] < start
            or item[1] > end
            or item[0] >= item[1]
        ):
            return False
        normalized.append((item[0], item[1]))
    cursor = start
    for range_start, range_end in sorted(normalized):
        if range_start > cursor:
            return False
        cursor = max(cursor, range_end)
    return cursor >= end


def _base_card(card: Any) -> str | None:
    if not isinstance(card, str):
        return None
    return card[4:] if card.startswith("evo-") else card


def _valid_cell(cell: Any) -> bool:
    return (
        isinstance(cell, list)
        and len(cell) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in cell)
        and 0 <= cell[0] < ACTION_GRID.cols
        and 0 <= cell[1] < ACTION_GRID.rows
    )


def _segment_bounds(manifest: dict[str, Any]) -> tuple[int, int]:
    segment = manifest["segment"]
    return segment["start_frame"], segment["end_frame_exclusive"]


def _read_object(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value
