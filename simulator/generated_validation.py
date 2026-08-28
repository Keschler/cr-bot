"""Execute and audit generated roster scenarios.

Generated scenarios are not real-game truth.  They are the broad synthetic
coverage layer that proves every declared card/component can be instantiated,
advanced, serialized, and replayed deterministically before expensive video
fidelity work is spent on it.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import combinations
import json
import multiprocessing
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable, Mapping

from .engine import ENGINE_VERSION, BattleEngine
from .geometry import cell_center_mtile
from .roster import PLAYER_DECK, load_opponent_roster
from .runner import run_scenario
from .ruleset import Ruleset, load_ruleset
from .scenario import Scenario, scenario_from_dict
from .scenario_factory import GeneratedScenario, card_mechanics


GENERATED_VALIDATION_SCHEMA_VERSION = 1


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def load_generated_manifest(path: str | Path) -> tuple[dict[str, Any], tuple[Scenario, ...]]:
    """Load a generated scenario manifest without silently accepting junk."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load generated manifest {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("generated manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported generated manifest schema")
    if payload.get("kind") != "simulator_generated_scenario_manifest":
        raise ValueError("manifest is not a generated scenario manifest")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("generated manifest cases must be a non-empty array")
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"generated case {index} must be an object")
        scenario = scenario_from_dict(raw)
        if scenario.scenario_id in seen:
            raise ValueError(f"duplicate generated scenario ID: {scenario.scenario_id}")
        seen.add(scenario.scenario_id)
        if scenario.split != "synthetic":
            raise ValueError(
                f"generated scenario {scenario.scenario_id} must use synthetic split"
            )
        scenarios.append(scenario)
    return payload, tuple(scenarios)


def _scenario_labels(scenario: Scenario) -> tuple[str, str]:
    oracle = scenario.oracle
    card_id = str(oracle.get("card_id") or "unknown")
    mechanic = str(oracle.get("mechanic") or "unknown")
    return card_id, mechanic


def _required_card_play_errors(scenario: Scenario, state: object) -> list[str]:
    """Return missing generated-card exercise obligations.

    A deterministic run can be perfectly repeatable while every scheduled
    play is rejected (for example a high-cost card played before enough
    elixir has accumulated).  Generated coverage must distinguish that case
    from an exercised mechanic.  The factory records these obligations in
    ``oracle.required_card_plays``; hand-authored scenarios may omit them.
    """

    obligations: list[tuple[str, object]] = [
        ("required_card_plays", scenario.oracle.get("required_card_plays", [])),
        (
            "required_support_card_plays",
            scenario.oracle.get("required_support_card_plays", []),
        ),
    ]
    played = {
        (event.get("player"), event.get("card_id"))
        for event in getattr(state, "events", ())
        if event.kind == "card_played"
    }
    errors: list[str] = []
    for field_name, raw_required in obligations:
        if raw_required is None:
            continue
        if not isinstance(raw_required, list):
            errors.append(f"oracle.{field_name} must be an array")
            continue
        for index, row in enumerate(raw_required):
            if not isinstance(row, dict):
                errors.append(f"{field_name}[{index}] must be an object")
                continue
            player = row.get("player")
            card_id = row.get("card_id")
            if type(player) is not int or not isinstance(card_id, str):
                errors.append(f"{field_name}[{index}] has invalid player/card_id")
                continue
            if (player, card_id) not in played:
                errors.append(
                    f"required card was not played: player={player} card={card_id}"
                )
    return errors


def _required_event_errors(scenario: Scenario, state: object) -> list[str]:
    """Return missing event obligations for a generated mechanic case."""

    raw_required = scenario.oracle.get("required_event_kinds", [])
    if raw_required is None:
        return []
    if not isinstance(raw_required, list):
        return ["oracle.required_event_kinds must be an array"]
    present = {event.kind for event in getattr(state, "events", ())}
    errors: list[str] = []
    for index, kind in enumerate(raw_required):
        if not isinstance(kind, str) or not kind:
            errors.append(f"required_event_kinds[{index}] must be a non-empty string")
        elif kind not in present:
            errors.append(f"required event was not emitted: kind={kind}")
    return errors


def _required_event_match_errors(scenario: Scenario, state: object) -> list[str]:
    """Check card/source-specific event predicates recorded by the factory."""

    raw_required = scenario.oracle.get("required_event_matches", [])
    if raw_required is None:
        return []
    if not isinstance(raw_required, list):
        return ["oracle.required_event_matches must be an array"]
    events = tuple(getattr(state, "events", ()))
    errors: list[str] = []
    for index, raw in enumerate(raw_required):
        if not isinstance(raw, dict):
            errors.append(f"required_event_matches[{index}] must be an object")
            continue
        kind = raw.get("kind")
        filters = raw.get("filters", {})
        if not isinstance(kind, str) or not kind:
            errors.append(f"required_event_matches[{index}].kind must be a non-empty string")
            continue
        if not isinstance(filters, dict):
            errors.append(f"required_event_matches[{index}].filters must be an object")
            continue
        matched = False
        for event in events:
            if event.kind != kind:
                continue
            def matches_filter(key: object, expected: object) -> bool:
                actual = event.get(str(key))
                if isinstance(expected, dict) and set(expected) == {"one_of"}:
                    candidates = expected.get("one_of")
                    return isinstance(candidates, list) and actual in candidates
                return actual == expected

            if all(matches_filter(key, value) for key, value in filters.items()):
                matched = True
                break
        if not matched:
            rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(filters.items()))
            errors.append(
                f"required event predicate was not emitted: kind={kind}"
                + (f" ({rendered})" if rendered else "")
            )
    return errors


def _required_state_errors(scenario: Scenario, state: object) -> list[str]:
    """Validate final-state obligations such as path movement.

    Movement is continuous state, not a discrete event.  The generated oracle
    therefore records the legal deployment cell and requires at least one
    entity from that play to leave its spawn center.  Dead entities remain in
    the authoritative pool, so a fast collision/death cannot hide movement.
    """

    raw_required = scenario.oracle.get("required_state_checks", [])
    if raw_required is None:
        return []
    if not isinstance(raw_required, list):
        return ["oracle.required_state_checks must be an array"]
    errors: list[str] = []
    for index, raw in enumerate(raw_required):
        if not isinstance(raw, dict):
            errors.append(f"required_state_checks[{index}] must be an object")
            continue
        if raw.get("type") != "entity_moved":
            errors.append(f"required_state_checks[{index}] has unsupported type")
            continue
        player = raw.get("player")
        card_id = raw.get("card_id")
        cell = raw.get("from_cell")
        if type(player) is not int or player not in (0, 1):
            errors.append(f"required_state_checks[{index}].player is invalid")
            continue
        if not isinstance(card_id, str) or not isinstance(cell, list) or len(cell) != 2:
            errors.append(f"required_state_checks[{index}] has invalid card/cell")
            continue
        try:
            origin = cell_center_mtile((int(cell[0]), int(cell[1])))
        except (TypeError, ValueError):
            errors.append(f"required_state_checks[{index}].from_cell is invalid")
            continue
        candidates = [
            entity
            for entity in getattr(state, "entities", {}).values()
            if entity.owner == player
            and entity.card_id == card_id
            and entity.kind == "troop"
        ]
        if not candidates or not any(
            (entity.x_mtile, entity.y_mtile) != origin for entity in candidates
        ):
            errors.append(
                f"required entity did not move: player={player} card={card_id} origin={origin}"
            )
    return errors


def _behavioral_obligation_fields(scenario: Scenario) -> tuple[list[str], list[str]]:
    """Return malformed obligation fields and the fields that are present."""

    malformed: list[str] = []
    present: list[str] = []
    for field_name in (
        "required_event_kinds",
        "required_event_matches",
        "required_state_checks",
    ):
        raw = scenario.oracle.get(field_name, [])
        if raw is None:
            continue
        if not isinstance(raw, list):
            malformed.append(f"oracle.{field_name} must be an array")
        elif raw:
            present.append(field_name)
    return malformed, present


def _validate_one_scenario(
    engine: BattleEngine,
    scenario: Scenario,
    *,
    repeats: int,
) -> dict[str, Any]:
    """Validate one case and return a JSON-compatible row.

    Keeping the case operation independent from aggregation is intentional:
    the same function is used by the serial path and by process workers.  A
    worker gets a fresh engine for each case, so mutable engine state can
    never leak between scenarios and each result is reproducible from the
    scenario payload alone.
    """

    card_id, mechanic = _scenario_labels(scenario)
    row: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "card_id": card_id,
        "mechanic": mechanic,
        "seed": scenario.seed,
        "max_ticks": scenario.max_ticks,
        "repeats": repeats,
    }
    hashes: list[dict[str, str]] = []
    error: str | None = None
    for repeat in range(repeats):
        try:
            state = run_scenario(engine, scenario)
            # Fast matrix runs disable the per-tick audit to make hundreds of
            # thousands of scenarios practical. Always keep a final
            # authoritative schema/invariant check in that mode.
            if not engine.validate_every_tick:
                engine.validate_state(state)
            exercise_errors = _required_card_play_errors(scenario, state)
            exercise_errors.extend(_required_event_errors(scenario, state))
            exercise_errors.extend(_required_event_match_errors(scenario, state))
            exercise_errors.extend(_required_state_errors(scenario, state))
            if exercise_errors:
                error = "; ".join(exercise_errors)
                break
            hashes.append(
                {
                    "state_hash": state.state_hash(),
                    "event_log_hash": state.event_log_hash(),
                    "replay_hash": state.replay_hash(),
                }
            )
            if repeat == 0:
                row.update(
                    {
                        "final_tick": state.tick,
                        "terminal": state.terminal,
                        "winner": state.winner,
                        "event_count": len(state.events),
                        "entity_count": len(state.entities),
                        "projectile_count": len(state.projectiles),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - report every generated failure
            error = f"{type(exc).__name__}: {exc}"
            break
    deterministic = bool(hashes) and len(
        {json.dumps(row, sort_keys=True) for row in hashes}
    ) == 1
    passed = error is None and len(hashes) == repeats and deterministic
    row.update(
        {
            "passed": passed,
            "deterministic": deterministic,
            "hashes": hashes,
        }
    )
    if error is not None:
        row["error"] = error
    return row


def _validate_one_scenario_worker(
    payload: tuple[dict[str, Any], str, bool, int],
) -> dict[str, Any]:
    """Process-pool entry point; all inputs and output are JSON-compatible."""

    raw_scenario, ruleset_id, validate_every_tick, repeats = payload
    scenario = scenario_from_dict(raw_scenario)
    engine = BattleEngine(
        load_ruleset(ruleset_id),
        validate_every_tick=validate_every_tick,
    )
    return _validate_one_scenario(engine, scenario, repeats=repeats)


def validate_generated_scenarios(
    engine: BattleEngine,
    scenarios: Iterable[Scenario | GeneratedScenario],
    *,
    repeats: int = 2,
    workers: int = 1,
) -> dict[str, Any]:
    """Run every generated scenario and require repeated hash identity.

    ``workers`` enables an optional process pool for large synthetic matrices.
    The default remains serial for low-latency CI.  Parallel execution does
    not change the simulation: each worker reconstructs the pinned ruleset and
    scenario, and rows are sorted by scenario ID before aggregation.
    """

    if type(repeats) is not int or repeats < 2:
        raise ValueError("repeats must be at least two")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    materialized: list[Scenario] = [
        item.scenario if isinstance(item, GeneratedScenario) else item
        for item in scenarios
    ]
    rows: list[dict[str, Any]]
    card_counts: Counter[str] = Counter()
    mechanic_counts: Counter[str] = Counter()
    started_ns = perf_counter_ns()
    if workers == 1 or len(materialized) <= 1:
        rows = [
            _validate_one_scenario(engine, scenario, repeats=repeats)
            for scenario in materialized
        ]
    else:
        payloads = [
            (
                scenario.to_dict(),
                engine.ruleset.ruleset_id,
                engine.validate_every_tick,
                repeats,
            )
            for scenario in materialized
        ]
        # Python 3.14 defaults to a forkserver on POSIX.  That mode needs a
        # filesystem socket and is unavailable in some hermetic CI runners;
        # fork is safe here because workers receive immutable JSON payloads
        # and construct their own engine.  Fall back to the platform default
        # where fork is not provided (for example Windows).
        try:
            process_context = multiprocessing.get_context("fork")
        except ValueError:
            process_context = None
        executor_kwargs = {"mp_context": process_context} if process_context else {}
        with ProcessPoolExecutor(max_workers=workers, **executor_kwargs) as executor:
            rows = list(executor.map(_validate_one_scenario_worker, payloads))
    for row in rows:
        card_counts[row["card_id"]] += 1
        mechanic_counts[row["mechanic"]] += 1
    elapsed_ns = perf_counter_ns() - started_ns
    rows.sort(key=lambda row: row["scenario_id"])
    failures = [row for row in rows if not row["passed"]]
    behavioral_gaps = []
    behavioral_malformed = []
    for scenario in materialized:
        malformed, present = _behavioral_obligation_fields(scenario)
        if malformed:
            behavioral_malformed.append(
                {"scenario_id": scenario.scenario_id, "errors": malformed}
            )
        if not present:
            card_id, mechanic = _scenario_labels(scenario)
            behavioral_gaps.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "card_id": card_id,
                    "mechanic": mechanic,
                }
            )
    pair_keys = {
        tuple(
            sorted(
                (
                    str(scenario.oracle["first_opponent_card_id"]),
                    str(scenario.oracle["second_opponent_card_id"]),
                )
            )
        )
        for scenario in materialized
        if scenario.oracle.get("first_opponent_card_id")
        and scenario.oracle.get("second_opponent_card_id")
        and scenario.oracle.get("first_opponent_card_id")
        != scenario.oracle.get("second_opponent_card_id")
    }
    pair_card_ids = {
        card_id
        for scenario in materialized
        for card_id in (
            scenario.oracle.get("first_opponent_card_id"),
            scenario.oracle.get("second_opponent_card_id"),
        )
        if isinstance(card_id, str) and card_id
    }
    report: dict[str, Any] = {
        "schema_version": GENERATED_VALIDATION_SCHEMA_VERSION,
        "kind": "simulator_generated_scenario_validation",
        "engine_version": ENGINE_VERSION,
        "state_validation": (
            "every_tick" if engine.validate_every_tick else "final_state"
        ),
        "ruleset_id": engine.ruleset.ruleset_id,
        "ruleset_hash": engine.ruleset.content_hash,
        "scenario_count": len(rows),
        "card_count": len(card_counts),
        "mechanic_count": len(mechanic_counts),
        "passed_count": sum(row["passed"] for row in rows),
        "failed_count": len(failures),
        "determinism_failures": sum(not row["deterministic"] for row in rows),
        "workers": workers,
        "elapsed_ns": elapsed_ns,
        "card_counts": dict(sorted(card_counts.items())),
        "mechanic_counts": dict(sorted(mechanic_counts.items())),
        "failures": failures,
        "behavioral_obligation_count": len(materialized) - len(behavioral_gaps),
        "behavioral_obligation_gap_count": len(behavioral_gaps),
        "behavioral_obligation_gaps": sorted(
            behavioral_gaps, key=lambda row: row["scenario_id"]
        ),
        "behavioral_obligation_malformed": sorted(
            behavioral_malformed, key=lambda row: row["scenario_id"]
        ),
        "cases": rows,
    }
    if pair_keys:
        report["unordered_pair_count"] = len(pair_keys)
        report["pair_card_count"] = len(pair_card_ids)
    return report


def validate_generated_behavioral_obligations(
    scenarios: Iterable[Scenario],
) -> dict[str, Any]:
    """Fail closed when a generated case has no behavioral oracle.

    ``required_card_plays`` proves that a card action was accepted, but it is
    not a mechanic oracle.  Release validation therefore requires at least
    one event-kind, event-predicate, or final-state check for every roster
    mechanic case.  Fixed-deck interaction and unordered-pair manifests are
    intentionally action-boundary scopes and are reported as not applicable.
    This gate is separate from matrix membership so focused manifests remain
    convenient during development.
    """

    gaps: list[dict[str, str]] = []
    malformed: list[dict[str, Any]] = []
    materialized = tuple(scenarios)
    scopes = {
        (
            "opponent_pairs"
            if scenario.oracle.get("first_opponent_card_id") is not None
            else "fixed_deck_interactions"
            if scenario.oracle.get("player_card_id") is not None
            else "roster_mechanics"
        )
        for scenario in materialized
    }
    if len(scopes) != 1:
        return {
            "passed": False,
            "applicable": False,
            "scope": "mixed",
            "scenario_count": len(materialized),
            "behavioral_obligation_count": 0,
            "behavioral_obligation_gap_count": 0,
            "gaps": [],
            "malformed": [],
            "errors": ["behavioral obligation gate received mixed scopes"],
        }
    scope = next(iter(scopes), "roster_mechanics")
    if scope != "roster_mechanics":
        # Interaction and unordered-pair manifests are explicitly action-
        # boundary matrices. Their required card-play oracle is the intended
        # contract; mechanic-specific behavior belongs to the roster scope.
        return {
            "passed": True,
            "applicable": False,
            "scope": scope,
            "scenario_count": len(materialized),
            "behavioral_obligation_count": 0,
            "behavioral_obligation_gap_count": 0,
            "gaps": [],
            "malformed": [],
            "errors": [],
        }
    for scenario in materialized:
        field_errors, present = _behavioral_obligation_fields(scenario)
        if field_errors:
            malformed.append(
                {"scenario_id": scenario.scenario_id, "errors": field_errors}
            )
        if not present:
            card_id, mechanic = _scenario_labels(scenario)
            gaps.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "card_id": card_id,
                    "mechanic": mechanic,
                }
            )
    errors: list[str] = []
    if malformed:
        errors.append(f"malformed behavioral obligation fields: {len(malformed)}")
    if gaps:
        errors.append(f"generated cases without behavioral obligations: {len(gaps)}")
    return {
        "passed": not errors,
        "applicable": True,
        "scope": scope,
        "scenario_count": len(materialized),
        "behavioral_obligation_count": len(materialized) - len(gaps),
        "behavioral_obligation_gap_count": len(gaps),
        "gaps": sorted(gaps, key=lambda row: row["scenario_id"]),
        "malformed": sorted(malformed, key=lambda row: row["scenario_id"]),
        "errors": errors,
    }


def validate_complete_generated_coverage(
    payload: Mapping[str, Any],
    scenarios: Iterable[Scenario],
    *,
    ruleset: Ruleset,
) -> dict[str, Any]:
    """Require a generated manifest to cover the complete declared V1 scope.

    The ordinary validator intentionally accepts focused manifests for fast
    iteration.  This separate gate is used by release/readiness commands and
    derives its expected keys from the checked-in roster and ruleset rather
    than trusting the manifest's summary fields.
    """

    materialized = tuple(scenarios)
    roster = load_opponent_roster()
    eligible = frozenset(roster.eligible_cards)
    oracles = tuple(scenario.oracle for scenario in materialized)
    pair_rows = tuple(
        oracle
        for oracle in oracles
        if isinstance(oracle.get("first_opponent_card_id"), str)
        and isinstance(oracle.get("second_opponent_card_id"), str)
    )
    interaction_rows = tuple(
        oracle
        for oracle in oracles
        if isinstance(oracle.get("player_card_id"), str)
        and isinstance(oracle.get("opponent_card_id"), str)
    )

    if pair_rows and interaction_rows:
        scope = "mixed"
    elif pair_rows:
        scope = "opponent_pairs"
    elif interaction_rows:
        scope = "fixed_deck_interactions"
    else:
        scope = "roster_mechanics"

    observed_cards: set[str] = set()
    observed_keys: set[tuple[str, ...]] = set()
    if scope == "opponent_pairs":
        for oracle in pair_rows:
            first = oracle["first_opponent_card_id"]
            second = oracle["second_opponent_card_id"]
            observed_cards.update((first, second))
            if first != second:
                observed_keys.add(tuple(sorted((first, second))))
        expected_keys = {
            tuple(pair)
            for pair in combinations(sorted(eligible), 2)
        }
        key_label = "unordered_opponent_pairs"
    elif scope == "fixed_deck_interactions":
        for oracle in interaction_rows:
            player = oracle["player_card_id"]
            opponent = oracle["opponent_card_id"]
            observed_cards.add(opponent)
            observed_keys.add((player, opponent))
        expected_keys = {
            (player, opponent)
            for player in PLAYER_DECK
            for opponent in sorted(eligible)
        }
        key_label = "fixed_deck_interactions"
    elif scope == "roster_mechanics":
        for oracle in oracles:
            card_id = oracle.get("card_id")
            mechanic = oracle.get("mechanic")
            if isinstance(card_id, str):
                observed_cards.add(card_id)
            if isinstance(card_id, str) and isinstance(mechanic, str):
                observed_keys.add((card_id, mechanic))
        expected_keys = {
            (card_id, mechanic)
            for card_id in sorted(eligible)
            for mechanic in card_mechanics(ruleset, card_id)
        }
        key_label = "roster_mechanics"
    else:
        expected_keys = set()
        key_label = "generated_scope"

    missing_cards = sorted(eligible - observed_cards)
    unexpected_cards = sorted(observed_cards - eligible)
    missing_keys = sorted(expected_keys - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_keys)
    errors: list[str] = []
    if scope == "mixed":
        errors.append("manifest mixes incompatible generated coverage scopes")
    if ruleset.ruleset_id != "v1":
        errors.append(
            f"complete generated coverage requires ruleset v1, got {ruleset.ruleset_id}"
        )
    if missing_cards:
        errors.append(f"missing eligible cards: {', '.join(missing_cards)}")
    if unexpected_cards:
        errors.append(f"unexpected cards: {', '.join(unexpected_cards)}")
    if missing_keys:
        errors.append(f"missing {key_label}: {len(missing_keys)}")
    if unexpected_keys:
        errors.append(f"unexpected {key_label}: {len(unexpected_keys)}")

    summary = payload.get("summary") if isinstance(payload, Mapping) else None
    declared_case_count = summary.get("scenario_count") if isinstance(summary, Mapping) else None
    if declared_case_count != len(materialized):
        errors.append("manifest summary.scenario_count does not match its cases")
    return {
        "passed": not errors,
        "scope": scope,
        "ruleset_id": ruleset.ruleset_id,
        "roster_id": roster.roster_id,
        "expected_card_count": len(eligible),
        "observed_card_count": len(observed_cards),
        "missing_cards": missing_cards,
        "unexpected_cards": unexpected_cards,
        "key_label": key_label,
        "expected_coverage_count": len(expected_keys),
        "observed_coverage_count": len(observed_keys),
        "missing_coverage_count": len(missing_keys),
        "unexpected_coverage_count": len(unexpected_keys),
        "errors": errors,
    }


def write_generated_validation_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Atomically write a canonical report with its content hash."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["report_hash"] = _manifest_hash(payload)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


__all__ = [
    "GENERATED_VALIDATION_SCHEMA_VERSION",
    "load_generated_manifest",
    "validate_complete_generated_coverage",
    "validate_generated_behavioral_obligations",
    "validate_generated_scenarios",
    "write_generated_validation_report",
]
