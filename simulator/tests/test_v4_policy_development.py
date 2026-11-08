"""Independent contracts for paired, resumable native development screens."""

from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from simulator.rl.v4 import policy_development as development


@pytest.fixture(scope="module")
def opponents():
    return development.training_opponents()


def _job(spec, *, seed=2026100801, arm="defense"):
    return dict(
        cell_id=f"{spec['deck_id']}__{spec.get('style', 'fixed')}__{seed}",
        arm=asdict(development.DevelopmentArm(arm, arm)),
        opponent=deepcopy(spec),
        seed=seed,
    )


def _row(job, *, result="win", opponent_invalid=0):
    """A complete lightweight native record, without loading a checkpoint."""

    from simulator.engine import ENGINE_VERSION
    from simulator.observation_v3 import OBSERVATION_V3_CONTRACT_HASH
    from simulator.rl.v4.frontier import _stable_seed
    from simulator.rl.v4.frontier_ppo.constants import FRONTIER_PPO_VERSION
    from simulator.roster import PLAYER_DECK
    from simulator.ruleset import load_fixed_ruleset

    spec = job["opponent"]
    identity = dict(
        kind=spec.get("kind", "scripted"), strategy=spec["strategy"],
        deck_id=spec["deck_id"], deck=list(spec["deck"]),
        controller_seed=spec["controller_seed"],
    )
    if "style_identity" in spec:
        identity["style"] = deepcopy(spec["style_identity"])
    winner = {"win": 0, "loss": 1, "draw": None}[result]
    record = dict(
        archetype=spec["archetype"], deck_id=spec["deck_id"], real_deck_split="train",
        opponent=identity, controller_seed=spec["controller_seed"], strategy=spec["strategy"],
        seed=job["seed"],
        policy_seed=_stable_seed("full-match", json.dumps(identity, sort_keys=True), job["seed"]) & 0xFFFFFFFF,
        deterministic_policy=True, learner_player=0, learner_deck=list(PLAYER_DECK),
        winner=winner, decisions=720, result=result, terminated=True, truncated=False,
        cap_reached=False, terminal_reason="regulation_crowns" if winner is not None else "tiebreak_draw",
        max_decisions=1440, decision_interval_us=250000,
        crowns=[int(winner == 0), int(winner == 1)], tower_hp=[], tower_hp_reduction=[0, 0],
        invalid_actions=[0, opponent_invalid], card_plays=[1, 0], spent_elixir_milli=[1000, 0],
        initial_state_hash="a" * 64, final_state_hash="b" * 64,
        event_log_hash="c" * 64, action_trace_sha256="d" * 64,
        runtime_version=FRONTIER_PPO_VERSION, engine_version=ENGINE_VERSION,
        ruleset_hash=load_fixed_ruleset().content_hash,
        observation_contract_hash=OBSERVATION_V3_CONTRACT_HASH,
    )
    diagnostics = dict(
        decisions=720, attempted_cards={"skeletons": 1}, accepted_cards={"skeletons": 1},
        waits=719, action_trace_sha256="d" * 64,
        rejected_actions=0,
        ground_only_air_wait_seconds=0., longest_ground_only_air_wait_seconds=0.,
        cycle_rescue_plays=0,
    )
    return dict(job=deepcopy(job), native_record=record, diagnostics=diagnostics)


def test_valid_complete_native_row_is_accepted(opponents) -> None:
    job = _job(opponents[0])
    development.validate_match(_row(job), job)


@pytest.mark.parametrize("field,value", [
    ("winner", 1),
    ("result", "loss"),
    ("seed", 2026100802),
    ("deck_id", "another-deck"),
    ("controller_seed", 0),
    ("strategy", "siege-bait"),
    ("archetype", "siege-bait"),
    ("learner_player", 1),
    ("learner_deck", ["skeletons"] * 8),
    ("real_deck_split", "eval"),
    ("max_decisions", 1),
    ("decision_interval_us", 500000),
    ("policy_seed", 0),
    ("deterministic_policy", False),
    ("terminated", False),
    ("truncated", True),
    ("cap_reached", True),
    ("decisions", 1441),
    ("invalid_actions", [1, 0]),
    ("card_plays", [2, 0]),
    ("action_trace_sha256", "e" * 64),
])
def test_native_contract_mutations_are_rejected(opponents, field, value) -> None:
    job = _job(opponents[0])
    row = _row(job)
    row["native_record"][field] = value
    with pytest.raises(ValueError):
        development.validate_match(row, job)


@pytest.mark.parametrize("field,value", [
    ("kind", "v4-checkpoint"),
    ("strategy", "siege-bait"),
    ("deck_id", "another-deck"),
    ("controller_seed", 0),
    ("deck", ["skeletons"] * 8),
])
def test_opponent_execution_identity_mutations_are_rejected(opponents, field, value) -> None:
    job = _job(opponents[0])
    row = _row(job)
    row["native_record"]["opponent"][field] = value
    with pytest.raises(ValueError):
        development.validate_match(row, job)


@pytest.mark.parametrize("field,value", [
    ("decisions", 719),
    ("waits", 720),
    ("attempted_cards", {"log": 1}),
    ("accepted_cards", {"log": 1}),
    ("rejected_actions", 1),
    ("action_trace_sha256", "e" * 64),
])
def test_execution_diagnostic_mutations_are_rejected(opponents, field, value) -> None:
    job = _job(opponents[0])
    row = _row(job)
    row["diagnostics"][field] = value
    with pytest.raises(ValueError):
        development.validate_match(row, job)


def test_opponent_warning_is_retained(opponents) -> None:
    job = _job(opponents[0])
    development.validate_match(_row(job, opponent_invalid=1), job)


def test_training_styles_have_exact_disjoint_identity(opponents) -> None:
    styled = development.training_opponents(["tempo", "patient"])
    assert len(styled) == len(opponents) * 2
    for spec in styled:
        identity = spec["style_identity"]
        assert identity["style_id"] == spec["style"]
        assert identity["seed"] == spec["controller_seed"]
        assert identity["split"] == spec["style_split"] == "train"
        development.validate_match(_row(_job(spec)), _job(spec))
    dev = development.training_opponents(["adaptive"], style_split="dev")
    assert len(dev) == len(opponents)
    assert all(spec["style_identity"]["split"] == "dev" for spec in dev)
    assert {spec["style"] for spec in styled}.isdisjoint(spec["style"] for spec in dev)
    with pytest.raises(ValueError):
        development.training_opponents(["adaptive"], style_split="train")
    with pytest.raises(ValueError):
        development.training_opponents(["tempo"], style_split="dev")


def test_saved_style_identity_must_match_execution() -> None:
    job = _job(development.training_opponents(["tempo"])[0])
    row = _row(job)
    row["native_record"]["opponent"]["style"]["parameters"]["reserve_milli"] += 1
    with pytest.raises(ValueError):
        development.validate_match(row, job)


def test_teacher_adapter_uses_public_inputs_and_public_own_elixir_only() -> None:
    import torch
    from simulator.rl.v4.hand_features import build_hand_tokens
    from simulator.rl.v4.model_v4 import masks_from_legal_play
    from simulator.tests.test_v4_defense_teacher import _entities, _entity, _legal

    class OwnOnlyPlayers:
        def __getitem__(self, index):
            assert index == 0, "teacher must not read enemy private player state"
            return SimpleNamespace(elixir_milli=6000)

    tokens, mask = _entities([_entity("balloon", y=.75)])
    args = dict(
        hand_tokens=torch.as_tensor(build_hand_tokens(["cannon", "log", "ice-golem", "hog-rider"], 6))[None, None],
        entities=torch.as_tensor(tokens)[None, None], entity_mask=torch.as_tensor(mask)[None, None],
    )
    masks = masks_from_legal_play(torch.as_tensor(_legal())[None, None], True)
    env = SimpleNamespace(state=SimpleNamespace(players=OwnOnlyPlayers()))
    teacher = development._PublicTeacherPolicy("cycle", [env])
    action = teacher.act_deterministic(args, masks)
    assert action.mode.tolist() == [[1]]
    assert action.card_slot.tolist() == [[2]]
    assert action.placement.tolist() == [[[29, 5]]]
    assert action.wait_duration.tolist() == [[0]]
    assert teacher.cycle_rescues == 1


def test_pairing_uses_cells_and_half_draw_score(opponents) -> None:
    base = [_row(_job(opponents[0]), result="loss"), _row(_job(opponents[1]), result="draw")]
    candidate = [_row(_job(opponents[1], arm="cycle")), _row(_job(opponents[0], arm="cycle"))]
    comparison = development.paired_summary(base, candidate)
    assert comparison["pairs"] == 2
    assert comparison["net_wins"] == 1.5
    assert comparison["mean_score_delta"] == .75
    assert comparison["favorable"] == 2
    assert comparison["unfavorable"] == 0
    assert comparison["family_net"] == {"hog": 1., "pekka-bridge": .5}


@pytest.mark.parametrize("field,value", [
    ("initial_state_hash", "e" * 64), ("policy_seed", 0),
    ("controller_seed", 0), ("decision_interval_us", 500000),
    ("ruleset_hash", "e" * 64), ("observation_contract_hash", "e" * 64),
    ("runtime_version", "different-runtime"), ("engine_version", "different-engine"),
])
def test_pairing_rejects_different_execution_conditions(opponents, field, value) -> None:
    base = _row(_job(opponents[0]))
    candidate = _row(_job(opponents[0], arm="cycle"))
    candidate["native_record"][field] = value
    with pytest.raises(ValueError):
        development.paired_summary([base], [candidate])


def test_pairing_rejects_missing_or_duplicate_cells(opponents) -> None:
    base = _row(_job(opponents[0]))
    candidate = _row(_job(opponents[0], arm="cycle"))
    with pytest.raises(ValueError):
        development.paired_summary([base], [])
    with pytest.raises(ValueError):
        development.paired_summary([base, deepcopy(base)], [candidate])


def test_empty_pairing_reports_no_score_estimate() -> None:
    comparison = development.paired_summary([], [])
    assert comparison["pairs"] == 0
    assert comparison["mean_score_delta"] is None


@pytest.mark.parametrize("net,lavaloon,bait,eligible", [
    (2., 1., -2.5, True), (1.5, 1., 0., False),
    (2., .5, 0., False), (3., 1., -3., False),
])
def test_cycle_gate_uses_predeclared_inclusive_boundaries(net, lavaloon, bait, eligible) -> None:
    gate = dict(minimum_net_wins=2, minimum_lavaloon_net_wins=1,
                reject_any_family_delta_at_or_below=-3, learner_illegal=0, genuine_terminal=True)
    comparison = dict(net_wins=net, family_net={"lavaloon": lavaloon, "bait": bait})
    result = development.cycle_gate(comparison, gate)
    assert result["eligible"] is eligible
    assert result["regressions"] == ({"bait": bait} if bait <= -3 else {})
    assert result["gate"] == gate


class _SynchronousPool:
    """Real Future contract without processes or native matches."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, job):
        future = Future()
        try:
            future.set_result(function(job))
        except BaseException as error:
            future.set_exception(error)
        return future


@pytest.fixture
def fake_screen(monkeypatch, opponents):
    calls = []
    def execute(job):
        calls.append(deepcopy(job))
        return _row(job)
    monkeypatch.setattr(development, "ProcessPoolExecutor", _SynchronousPool)
    monkeypatch.setattr(development, "execute_match", execute)
    monkeypatch.setattr(development, "runtime_source_fingerprint", lambda: {"sha256": "source-fixture"})
    return dict(arms=[development.DevelopmentArm("defense", "defense"),
                      development.DevelopmentArm("cycle", "cycle")],
                opponents=[opponents[0]], seeds=[2026100801], workers=1), calls


def test_completed_resume_reuses_exact_records(tmp_path, fake_screen) -> None:
    kwargs, calls = fake_screen
    first = development.run_development_screen(out=tmp_path, **kwargs)
    assert first["raw_records"] == 2
    assert len(calls) == 2
    second = development.run_development_screen(out=tmp_path, **kwargs)
    assert len(calls) == 2
    assert second["raw_digest_sha256"] == first["raw_digest_sha256"]
    assert second["comparisons"] == first["comparisons"]
    assert second["selected_actor_changed"] is False


def test_completed_resume_rejects_changed_raw_record(tmp_path, fake_screen) -> None:
    kwargs, calls = fake_screen
    development.run_development_screen(out=tmp_path, **kwargs)
    path = next((tmp_path / "cycle").glob("*.json"))
    row = json.loads(path.read_text())
    # This extra diagnostic leaves action/outcome validation intact; the saved
    # content hash must still reject silent replacement of completed evidence.
    row["diagnostics"]["review_note"] = "changed after completion"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        development.run_development_screen(out=tmp_path, **kwargs)
    assert len(calls) == 2


@pytest.mark.parametrize("mutation", ["missing-seal", "extra-record", "missing-record"])
def test_completed_resume_requires_exact_sealed_grid(tmp_path, fake_screen, mutation) -> None:
    kwargs, calls = fake_screen
    development.run_development_screen(out=tmp_path, **kwargs)
    path = next((tmp_path / "cycle").glob("*.json"))
    if mutation == "missing-seal":
        (tmp_path / "raw_digests.json").unlink()
    elif mutation == "extra-record":
        path.with_name("extra-cell.json").write_bytes(path.read_bytes())
    else:
        path.unlink()
    with pytest.raises(ValueError):
        development.run_development_screen(out=tmp_path, **kwargs)
    assert len(calls) == 2


def test_opponent_warning_exclusion_is_symmetric_even_without_remaining_pairs(
    tmp_path, fake_screen, monkeypatch,
) -> None:
    kwargs, calls = fake_screen
    def execute(job):
        calls.append(deepcopy(job))
        return _row(job, opponent_invalid=int(job["arm"]["kind"] == "cycle"))
    monkeypatch.setattr(development, "execute_match", execute)
    report = development.run_development_screen(out=tmp_path, **kwargs)
    assert report["status"] == "complete_with_opponent_warning"
    assert report["arms"]["defense"]["games"] == report["arms"]["cycle"]["games"] == 1
    assert len(report["paired_exclusion"]["cells"]) == 1
    assert report["paired_exclusion"]["comparisons"]["cycle"]["pairs"] == 0
    assert report["paired_exclusion"]["comparisons"]["cycle"]["mean_score_delta"] is None


def test_development_screen_rejects_reserved_deck_relabel(tmp_path, fake_screen) -> None:
    from simulator.rl.v4.catalog_evaluation import catalog_opponents

    kwargs, calls = fake_screen
    reserved = dict(catalog_opponents()["eval"][0], real_deck_split="train")
    with pytest.raises(ValueError):
        development.run_development_screen(out=tmp_path, **dict(kwargs, opponents=[reserved]))
    assert not calls


def test_development_screen_rejects_wrong_cards_for_training_identity(tmp_path, fake_screen, opponents) -> None:
    kwargs, calls = fake_screen
    spec = dict(opponents[0], deck=list(opponents[1]["deck"]))
    with pytest.raises(ValueError):
        development.run_development_screen(out=tmp_path, **dict(kwargs, opponents=[spec]))
    assert not calls


def test_resume_rejects_changed_source(tmp_path, fake_screen, monkeypatch) -> None:
    kwargs, calls = fake_screen
    development.run_development_screen(out=tmp_path, **kwargs)
    monkeypatch.setattr(development, "runtime_source_fingerprint", lambda: {"sha256": "changed-source"})
    with pytest.raises(ValueError):
        development.run_development_screen(out=tmp_path, **kwargs)
    assert len(calls) == 2


def test_native_source_scope_covers_reachable_tracker_mapping() -> None:
    fingerprint = development.runtime_source_fingerprint()
    files = fingerprint["files"]
    assert "src/cr_bot/trackers/__init__.py" in files
    assert "src/cr_bot/trackers/direct_unit_to_card.py" in files
    assert "simulator/rl/v4/simulator_teacher.py" in files
    assert "scripts/run_v4_policy_development.py" in files
    assert not any("/tests/" in path or "/physical_lab/" in path for path in files)
    assert fingerprint["sha256"] == development.digest(files)
