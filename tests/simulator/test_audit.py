from __future__ import annotations

import json

import pytest

from simulator.actions import PlayCardAction, WaitAction
from simulator.audit import (
    DeterminismAuditError,
    LegalFuzzController,
    run_determinism_audit,
    run_soak_audit,
)
from simulator.engine import ENGINE_VERSION, BattleEngine


def test_legal_fuzz_controller_forces_wait_then_chooses_a_legal_play() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=11, shuffle_decks=False)
    controller = LegalFuzzController(
        99,
        random_wait_permille=0,
        force_wait_every=7,
    )

    first = controller.choose_action(engine, state, 0)
    second = controller.choose_action(engine, state, 0)

    assert first == WaitAction(0)
    assert isinstance(second, PlayCardAction)
    assert engine.validate_action(state, second) is None
    assert controller.decision_counts == (2, 0)


def test_audit_is_repeatable_json_safe_and_exercises_wait_and_play() -> None:
    kwargs = {
        "seed_count": 3,
        "seed_start": 17,
        "max_ticks_per_seed": 31,
        "decision_interval_ticks": 2,
        "controller_factory": lambda seed: LegalFuzzController(
            seed,
            random_wait_permille=0,
            force_wait_every=3,
        ),
    }

    first = run_determinism_audit(**kwargs)
    second = run_determinism_audit(**kwargs)

    assert first.to_dict() == second.to_dict()
    assert json.loads(json.dumps(first.to_dict(), sort_keys=True)) == first.to_dict()
    assert first.seed_count == 3
    assert first.engine_version == ENGINE_VERSION
    assert first.to_dict()["engine_version"] == ENGINE_VERSION
    assert first.to_dict()["revision_guard"]["status"] == "stable"
    assert first.to_dict()["code_revision"]["commit"]
    assert first.to_dict()["run_code_revision"] == first.to_dict()["revision_guard"]["start"]
    assert [run.seed for run in first.runs] == [17, 18, 19]
    assert first.total_ticks == 93
    assert first.total_actions == 96
    assert first.total_actions == first.total_card_plays + first.total_waits
    assert first.total_card_plays > 0
    assert first.total_waits > 0
    assert first.total_events > 0
    assert first.completions == 0
    assert len({run.final_hash for run in first.runs}) == 3
    assert first.to_dict()["final_hashes"] == [
        {
            "seed": run.seed,
            "state_hash": run.final_hash,
            "event_log_hash": run.event_log_hash,
            "replay_hash": run.replay_hash,
        }
        for run in first.runs
    ]


def test_audit_reports_completed_matches() -> None:
    class _ShortMatchEngine(BattleEngine):
        def step(self, state, actions=()):  # type: ignore[no-untyped-def]
            events = super().step(state, actions)
            if state.tick == 3 and not state.terminal:
                state.terminal = True
                state.phase = "ended"
                state.terminal_reason = "test_complete"
            return events

    report = run_determinism_audit(
        seed_count=2,
        max_ticks_per_seed=20,
        decision_interval_ticks=2,
        engine_factory=_ShortMatchEngine,
    )

    assert report.total_ticks == 6
    assert report.completions == 2
    assert all(run.completed for run in report.runs)
    assert all(run.terminal_reason == "test_complete" for run in report.runs)


def test_divergence_includes_seed_and_tick() -> None:
    creations = 0

    class _DivergingEngine(BattleEngine):
        def step(self, state, actions=()):  # type: ignore[no-untyped-def]
            events = super().step(state, actions)
            if state.tick == 2:
                state.rng_state ^= 1
            return events

    def factory() -> BattleEngine:
        nonlocal creations
        creations += 1
        return BattleEngine() if creations == 1 else _DivergingEngine()

    with pytest.raises(DeterminismAuditError, match=r"seed=41, tick=2") as raised:
        run_determinism_audit(
            seed_count=1,
            seed_start=41,
            max_ticks_per_seed=5,
            decision_interval_ticks=1,
            engine_factory=factory,
        )

    assert raised.value.seed == 41
    assert raised.value.tick == 2
    assert raised.value.first_hash != raised.value.second_hash


def test_soak_mode_respects_deterministic_tick_budget() -> None:
    report = run_soak_audit(
        seed_count=3,
        seed_start=100,
        tick_budget=17,
        max_ticks_per_seed=50,
        decision_interval_ticks=2,
    )

    assert report.mode == "soak"
    assert report.tick_budget == 17
    assert report.max_ticks_per_seed == 5
    assert report.total_ticks == 15
    assert [run.seed for run in report.runs] == [100, 101, 102]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed_count": 0}, "seed_count"),
        ({"tick_budget": 0}, "tick_budget"),
        ({"seed_count": 3, "tick_budget": 2}, "at least one tick"),
        ({"decision_interval_ticks": 0}, "decision_interval_ticks"),
    ],
)
def test_soak_rejects_unbounded_or_empty_requests(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_soak_audit(**kwargs)
