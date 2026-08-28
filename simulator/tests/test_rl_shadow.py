"""Contract tests for the public-observation shadow runner.

These tests keep the shadow runner's media boundaries and actor-only safety
properties explicit without opening real media, cameras, or ADB devices.

Proposed API
============

``rl.shadow.ShadowPolicyRunner`` accepts a learner with the existing recurrent
policy interface.  It exposes:

* ``consume(step, *, frame_idx, time_s)``.  It consumes one
  ``MatchSessionStep``-shaped object, returns a ``ShadowPrediction`` for an
  emitted in-game step, and returns ``None`` for a lobby, countdown, or
  end-of-match step.  A non-game step clears the recurrent episode boundary;
  the next emitted step is sent with ``reset_mask=True``.
* ``ShadowPrediction.to_dict()``.  It is the JSON-compatible decision record.
  Shadow mode records predictions only and never dispatches a phone action.

``rl.shadow.run_shadow_media`` accepts exactly one of ``video``,
``replay_cache``, or ``video_device``.  For deterministic tests it accepts an
injected ``runner`` and ``source_factory``.  The factory is called as
``source_factory(source_kind, source_value)`` and yields either
``(frame_idx, time_s, match_session_step)`` or
``(frame_idx, time_s, frame, analysis)``.  The normal implementation is
responsible for connecting those source kinds to the existing
detector/cache/camera pipeline, and the tests never invoke it.

``rl.prototype._parser()`` exposes a ``shadow`` subcommand with a required
mutually-exclusive media-source group.

The learner interface has one deliberate implementation choice worth making
explicit: shadow inference must be actor-only.  The current runner performs a
deterministic policy forward and bypasses the learner's value/critic path, so
there is no ``privileged_features`` object to pass at all.  The tests assert
the observable equivalent—deterministic action selection, public tensor-only
policy inputs, and zero critic/tap calls.  If the implementation later routes
through ``learner.rollout_step``, it must pass ``deterministic=True`` and
``privileged_features=None`` and preserve the same assertions.

If the production API changes, update this contract and its assumptions
together.  No production files, detectors, cameras, or ADB devices are used
by this test module.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pytest


try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


def _shadow_module() -> Any:
    """Import the proposed module with a useful failure while it is absent."""

    try:
        return importlib.import_module("rl.shadow")
    except ModuleNotFoundError as error:
        if error.name == "rl.shadow":
            pytest.fail(
                "the rl.shadow module is not implemented"
            )
        raise


def _v1_observation() -> Any:
    from simulator.observation import (
        ACTION_MASK_SHAPE,
        BOARD_SHAPE,
        GLOBAL_VECTOR_SHAPE,
        PolicyObservationV1,
    )

    board = np.zeros(BOARD_SHAPE, dtype=np.float32)
    board[0, 0, 0] = np.float32(0.25)
    global_vector = np.zeros(GLOBAL_VECTOR_SHAPE, dtype=np.float32)
    global_vector[0] = np.float32(0.5)
    spatial_masks = np.zeros(ACTION_MASK_SHAPE, dtype=bool)
    legal_play = np.zeros(ACTION_MASK_SHAPE, dtype=bool)
    legal_play[2, :, :] = True
    return PolicyObservationV1(
        board=board,
        global_vector=global_vector,
        spatial_masks=spatial_masks,
        legal_play=legal_play,
        legal_wait=True,
    )


def _public_observation() -> Any:
    from simulator.observation_v2 import ENTITY_TOKEN_DIM, PolicyObservationV2

    rows = np.zeros((1, ENTITY_TOKEN_DIM), dtype=np.float32)
    rows[0, 0] = np.float32(0.75)
    return PolicyObservationV2.from_v1(
        _v1_observation(),
        public_entity_rows=rows,
    )


def _match_step(*, in_game: bool = True, should_emit: bool = True) -> Any:
    """Return a detector-free MatchSessionStep-shaped fake."""

    return SimpleNamespace(
        analysis=SimpleNamespace(arena_px=(0.0, 0.0, 1.0, 1.0)),
        game_state=(
            SimpleNamespace(
                # These values must never enter the actor tensors or trace.
                secret_enemy_hand=("cannon", "log"),
                exact_enemy_elixir=7.25,
                hud=SimpleNamespace(
                    hand_cards=("hog-rider", "cannon", "musketeer", "skeletons")
                ),
            )
            if in_game
            else None
        ),
        in_game=in_game,
        should_emit=should_emit,
    )


def _patch_public_match_adapter(monkeypatch: pytest.MonkeyPatch, shadow: Any, observation: Any) -> list[Any]:
    """Replace only the public match-step conversion, never a detector."""

    from simulator.physical_lab import policy_bridge

    seen: list[Any] = []

    def convert(step: Any) -> Any | None:
        seen.append(step)
        if not bool(getattr(step, "in_game", False)) or not bool(
            getattr(step, "should_emit", False)
        ):
            return None
        return observation

    # Accommodate either a module-qualified lookup or a deliberately re-exported
    # function while keeping the contract tied to the named bridge helper.
    monkeypatch.setattr(policy_bridge, "observation_v2_from_match_step", convert)
    monkeypatch.setattr(shadow, "observation_v2_from_match_step", convert, raising=False)
    return seen


@dataclass
class _FakeLearner:
    """Small recurrent learner double recording the public call boundary."""

    device: Any

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.initial_state_calls = 0
        self.next_states: list[Any] = []
        self.critic_calls = 0
        self.rollout_calls: list[dict[str, Any]] = []

    def initial_rollout_state(self, batch_size: int) -> Any:
        from rl.learner import RecurrentRolloutState

        self.initial_state_calls += 1
        return RecurrentRolloutState(
            torch.full(
                (1, batch_size, 2),
                float(self.initial_state_calls),
                dtype=torch.float32,
            )
        )

    def rollout_step(
        self,
        state: Any,
        raster: Any,
        global_features: Any,
        entities: Any,
        entity_mask: Any,
        action_masks: Any,
        *,
        reset_mask: Any,
        privileged_features: Any,
        deterministic: bool,
    ) -> Any:
        from rl.learner import RecurrentRolloutState
        from rl.trajectory import ActionBatch

        self.calls.append(
            {
                "state": state,
                "raster": raster.detach().clone(),
                "global_features": global_features.detach().clone(),
                "entities": entities.detach().clone(),
                "entity_mask": entity_mask.detach().clone(),
                "action_masks": action_masks,
                "reset_mask": reset_mask.detach().clone(),
                "privileged_features": privileged_features,
                "deterministic": deterministic,
            }
        )
        next_state = RecurrentRolloutState(
            torch.full(
                (1, 1, 2),
                float(10 + len(self.calls)),
                dtype=torch.float32,
            )
        )
        self.next_states.append(next_state)
        return SimpleNamespace(
            actions=ActionBatch(
                # PLAY, slot 2, tensor order (row, column) -> public action
                # cell (column, row) = (7, 12).
                mode=torch.tensor([[1]], dtype=torch.long),
                card_slot=torch.tensor([[2]], dtype=torch.long),
                placement=torch.tensor([[[12, 7]]], dtype=torch.long),
            ),
            next_state=next_state,
        )


def _fake_learner() -> _FakeLearner:
    return _FakeLearner(device=torch.device("cpu"))


def test_policy_bridge_v2_match_step_skips_boundaries_and_delegates_public_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from simulator.physical_lab import policy_bridge

    sentinel = object()
    calls: list[tuple[Any, tuple[float, ...]]] = []

    def fake_from_game_state(
        game_state: Any,
        *,
        arena_px: tuple[float, ...],
        legal_wait: bool | None = None,
    ) -> Any:
        assert legal_wait is None
        calls.append((game_state, arena_px))
        return sentinel

    monkeypatch.setattr(policy_bridge, "observation_v2_from_game_state", fake_from_game_state)

    assert policy_bridge.observation_v2_from_match_step(_match_step(in_game=False)) is None
    assert (
        policy_bridge.observation_v2_from_match_step(
            _match_step(in_game=True, should_emit=False)
        )
        is None
    )

    live_step = _match_step()
    assert policy_bridge.observation_v2_from_match_step(live_step) is sentinel
    assert calls == [
        (
            live_step.game_state,
            (0.0, 0.0, 1.0, 1.0),
        )
    ]


def test_policy_bridge_v2_game_state_adapter_only_attaches_public_entity_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from simulator.physical_lab import policy_bridge

    public_rows = np.zeros((1, 32), dtype=np.float32)
    public_rows[0, 0] = np.float32(0.875)
    state = SimpleNamespace(
        secret_enemy_hand=("cannon", "log"),
        exact_enemy_elixir=8.5,
    )
    seen: list[tuple[Any, int]] = []

    monkeypatch.setattr(policy_bridge, "observation_from_game_state", lambda *args, **kwargs: _v1_observation())

    def public_rows_only(game_state: Any, *, viewer: int = 0, **_: Any) -> np.ndarray:
        seen.append((game_state, viewer))
        return public_rows

    monkeypatch.setattr(policy_bridge, "build_public_entity_rows", public_rows_only)

    observation = policy_bridge.observation_v2_from_game_state(
        state,
        arena_px=(0.0, 0.0, 1.0, 1.0),
    )

    assert seen == [(state, 0)]
    assert observation.entity_mask[0]
    assert observation.entity_tokens[0, 0] == np.float32(0.875)
    serialized = json.dumps(
        {
            "entity_tokens": observation.entity_tokens.tolist(),
            "entity_mask": observation.entity_mask.tolist(),
        },
        allow_nan=False,
    )
    assert "cannon" not in serialized
    assert "8.5" not in serialized


@requires_torch
def test_shadow_runner_uses_public_deterministic_rollout_and_maps_play_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = _shadow_module()
    from simulator.physical_lab import policy_bridge

    observation = _public_observation()
    adapter_calls = _patch_public_match_adapter(monkeypatch, shadow, observation)

    def fail_if_tapped(*_: Any, **__: Any) -> None:
        raise AssertionError("shadow inference must never dispatch a phone tap")

    monkeypatch.setattr(policy_bridge, "dispatch_policy_action", fail_if_tapped)
    learner = _fake_learner()
    runner = shadow.ShadowPolicyRunner(learner)
    step = _match_step()

    decision = runner.process_step(step, frame_index=17, timestamp_s=1.5)

    assert adapter_calls == [step]
    assert len(learner.calls) == 1
    call = learner.calls[0]
    assert call["deterministic"] is True
    assert call["privileged_features"] is None
    assert call["reset_mask"].reshape(-1).tolist() == [True]
    assert call["raster"].shape[0] == 1
    assert float(call["raster"].reshape(-1)[0]) == pytest.approx(0.25)
    assert float(call["entities"].reshape(-1)[0]) == pytest.approx(0.75)
    assert "secret_enemy_hand" not in repr(call)
    assert "exact_enemy_elixir" not in repr(call)
    assert decision == {
        "frame_index": 17,
        "timestamp_s": 1.5,
        "action": {
            "kind": "Play",
            "card_slot": 2,
            "cell": [7, 12],
        },
    }


@requires_torch
def test_shadow_runner_continues_hidden_and_resets_after_non_game_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = _shadow_module()
    observation = _public_observation()
    _patch_public_match_adapter(monkeypatch, shadow, observation)
    learner = _fake_learner()
    runner = shadow.ShadowPolicyRunner(learner)

    runner.process_step(_match_step(), frame_index=1, timestamp_s=0.1)
    runner.process_step(_match_step(), frame_index=2, timestamp_s=0.2)

    assert len(learner.calls) == 2
    # The second decision receives the first decision's recurrent output.  A
    # concrete learner may detach/wrap the object, so compare hidden values.
    torch.testing.assert_close(
        learner.calls[1]["state"].hidden,
        learner.next_states[0].hidden,
    )
    assert learner.calls[1]["reset_mask"].reshape(-1).tolist() == [False]

    assert runner.process_step(
        _match_step(in_game=False, should_emit=False),
        frame_index=3,
        timestamp_s=0.3,
    ) is None
    assert len(learner.calls) == 2

    runner.process_step(_match_step(), frame_index=4, timestamp_s=0.4)
    assert len(learner.calls) == 3
    # The boundary must be explicit at the next policy call, so stale hidden
    # state cannot be mistaken for the beginning of a new match.
    assert learner.calls[2]["reset_mask"].reshape(-1).tolist() == [True]


@requires_torch
def test_shadow_runner_trace_is_json_safe_and_contains_decision_basics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = _shadow_module()
    observation = _public_observation()
    _patch_public_match_adapter(monkeypatch, shadow, observation)
    runner = shadow.ShadowPolicyRunner(_fake_learner())
    assert runner.process_step(_match_step(), frame_index=8, timestamp_s=2.25) is not None

    payload = runner.trace_payload()
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["kind"] == "recurrent_public_ppo_shadow_trace"
    assert isinstance(decoded["decisions"], list)
    assert decoded["decisions"] == [
        {
            "frame_index": 8,
            "timestamp_s": 2.25,
            "action": {
                "kind": "Play",
                "card_slot": 2,
                "cell": [7, 12],
            },
        }
    ]


@dataclass
class _FakeMediaRunner:
    calls: list[tuple[Any, int, float]]

    def process_step(self, step: Any, *, frame_index: int, timestamp_s: float) -> dict[str, Any]:
        self.calls.append((step, frame_index, timestamp_s))
        return {
            "frame_index": frame_index,
            "timestamp_s": timestamp_s,
            "action": {"kind": "Wait"},
        }

    def consume(self, step: Any, *, frame_idx: int, time_s: float) -> Any:
        self.calls.append((step, frame_idx, time_s))
        return SimpleNamespace(
            to_dict=lambda: {
                "frame_idx": frame_idx,
                "time_s": time_s,
                "action_kind": "WAIT",
                "action_log_prob": 0.0,
                "entropy": 0.0,
            }
        )

    def trace_payload(self) -> dict[str, Any]:
        return {
            "kind": "recurrent_public_ppo_shadow_trace",
            "decisions": [
                {
                    "frame_index": frame_index,
                    "timestamp_s": timestamp_s,
                    "action": {"kind": "Wait"},
                }
                for _step, frame_index, timestamp_s in self.calls
            ],
            "actor_privileged_inputs": False,
            "critic_privileged_inputs": False,
            "taps_sent": 0,
        }


@pytest.mark.parametrize(
    ("source_argument", "source_kind", "source_value"),
    (
        ("video", "video", "match.mp4"),
        ("replay_cache", "replay-cache", "match.pkl.gz"),
        ("video_device", "video-device", "/dev/video37"),
    ),
)
def test_run_shadow_media_dispatches_each_source_without_detector_or_adb(
    source_argument: str,
    source_kind: str,
    source_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = _shadow_module()
    media_calls: list[tuple[str, object]] = []
    fake_runner = _FakeMediaRunner(calls=[])
    step = _match_step()
    source_input: str | Path = (
        tmp_path / source_value if source_kind in {"video", "replay-cache"} else source_value
    )

    def source_steps(kind: str, value: object) -> Iterable[tuple[int, float, Any]]:
        expected_value = (
            Path(source_input) if source_kind in {"video", "replay-cache"} else source_input
        )
        assert kind == source_kind
        assert value == expected_value
        media_calls.append((kind, value))
        yield (23, 4.5, step)

    report = shadow.run_shadow_media(
        "checkpoint.pt",
        **{source_argument: source_input},
        runner=fake_runner,
        source_factory=source_steps,
        trace_path=tmp_path / f"{source_kind}.json",
        max_frames=1,
    )

    expected_source = Path(source_input) if source_kind in {"video", "replay-cache"} else source_input
    assert media_calls == [(source_kind, expected_source)]
    assert fake_runner.calls == [(step, 23, 4.5)]
    assert report["kind"] == "recurrent_public_ppo_shadow_trace"
    assert report["source"] == str(expected_source)
    assert report["source_kind"] == source_kind
    assert report["decisions"][0]["action"] == {"kind": "Wait"}
    assert report["actor_privileged_inputs"] is False
    assert report["critic_privileged_inputs"] is False
    assert report["taps_sent"] == 0
    trace_path = tmp_path / f"{source_kind}.json"
    assert json.loads(trace_path.read_text(encoding="utf-8")) == report
    json.dumps(report, allow_nan=False)


def test_run_shadow_media_rejects_zero_or_multiple_sources() -> None:
    shadow = _shadow_module()

    with pytest.raises(shadow.ShadowConfigurationError, match="exactly one"):
        shadow.run_shadow_media("checkpoint.pt")
    with pytest.raises(shadow.ShadowConfigurationError, match="exactly one"):
        shadow.run_shadow_media(
            "checkpoint.pt",
            video="match.mp4",
            replay_cache="match.pkl.gz",
        )


def test_run_shadow_media_max_seconds_is_relative_to_source_start(tmp_path: Path) -> None:
    shadow = _shadow_module()
    source = tmp_path / "match.mp4"
    source.write_bytes(b"fake media")
    runner = _FakeMediaRunner(calls=[])
    step = _match_step()

    def source_steps(kind: str, value: object) -> Iterable[tuple[int, float, Any]]:
        assert kind == "video"
        assert value == source
        yield (1, 60.0, step)
        yield (2, 60.5, step)
        yield (3, 60.8, step)

    report = shadow.run_shadow_media(
        "checkpoint.pt",
        video=source,
        runner=runner,
        source_factory=source_steps,
        max_seconds=0.6,
    )

    assert [frame_idx for _step, frame_idx, _time_s in runner.calls] == [1, 2]
    assert report["source_kind"] == "video"


def test_run_shadow_media_reports_stale_checkpoint_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rl import prototype

    shadow = _shadow_module()
    loaded_metadata = {
        "checkpoint_format": "recurrent-public-ppo-prototype-v1",
        "_checkpoint_ruleset_id": "v1",
        "_checkpoint_ruleset_hash": "sha256:" + "a" * 64,
        "_runtime_ruleset_id": "v1",
        "_runtime_ruleset_hash": "sha256:" + "b" * 64,
        "_checkpoint_ruleset_match": False,
        "_stale_ruleset_allowed": True,
        "critic_observation": {"privileged_inputs": True},
    }

    def load_checkpoint(
        checkpoint: Any,
        *,
        device: str | None,
        allow_stale_ruleset: bool,
    ) -> tuple[Any, Any, dict[str, Any]]:
        assert checkpoint == "checkpoint.pt"
        assert device == "cpu"
        assert allow_stale_ruleset is True
        return object(), object(), loaded_metadata

    class NoTraceRunner:
        emitted_observations = 0
        invalid_observations = 0
        matches_started = 0
        match_boundaries = 0

        def process_step(
            self,
            step: Any,
            *,
            frame_index: int,
            timestamp_s: float,
        ) -> None:
            return None

    monkeypatch.setattr(prototype, "load_shadow_prototype_checkpoint", load_checkpoint)
    monkeypatch.setattr(shadow, "ShadowPolicyRunner", lambda learner: NoTraceRunner())

    def source_steps(kind: str, value: object) -> Iterable[tuple[int, float, Any]]:
        assert kind == "video"
        assert value == Path("match.mp4")
        yield (1, 1.0, _match_step())

    report = shadow.run_shadow_media(
        "checkpoint.pt",
        video="match.mp4",
        device="cpu",
        allow_stale_ruleset=True,
        source_factory=source_steps,
    )

    assert report["checkpoint_validation"] == {
        "mode": "shadow-stale-allowed",
        "status": "stale",
        "checkpoint_ruleset_id": "v1",
        "checkpoint_ruleset_hash": "sha256:" + "a" * 64,
        "runtime_ruleset_id": "v1",
        "runtime_ruleset_hash": "sha256:" + "b" * 64,
        "hash_match": False,
        "stale_checkpoint_used": True,
    }
    assert "older content hash" in report["warning"]


def test_prototype_shadow_parser_requires_exactly_one_media_source() -> None:
    from rl import prototype

    parser = prototype._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["shadow", "--checkpoint", "checkpoint.pt"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "shadow",
                "--checkpoint",
                "checkpoint.pt",
                "--video",
                "match.mp4",
                "--replay-cache",
                "match.pkl.gz",
            ]
        )

    for option, value in (
        ("--video", "match.mp4"),
        ("--replay-cache", "match.pkl.gz"),
        ("--video-device", "/dev/video37"),
    ):
        args = parser.parse_args(
            ["shadow", "--checkpoint", "checkpoint.pt", option, value]
        )
        assert args.command == "shadow"
        assert str(getattr(args, option.removeprefix("--").replace("-", "_"))) == value
        assert args.device == "cpu"

    stale_args = parser.parse_args(
        [
            "shadow",
            "--checkpoint",
            "checkpoint.pt",
            "--video",
            "match.mp4",
            "--allow-stale-ruleset",
        ]
    )
    assert stale_args.allow_stale_ruleset is True

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--allow-stale-ruleset"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["evaluate", "--checkpoint", "checkpoint.pt", "--allow-stale-ruleset"]
        )


def test_prototype_shadow_cli_forwards_stale_ruleset_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from rl import prototype
    import rl.shadow as shadow

    report = {"kind": "shadow-test"}
    calls: dict[str, Any] = {}

    def run_shadow(checkpoint: Any, **kwargs: Any) -> dict[str, Any]:
        calls["checkpoint"] = checkpoint
        calls.update(kwargs)
        return report

    monkeypatch.setattr(shadow, "run_shadow_media", run_shadow)

    assert prototype.main(
        [
            "shadow",
            "--checkpoint",
            "checkpoint.pt",
            "--video",
            "match.mp4",
            "--allow-stale-ruleset",
        ]
    ) == 0
    assert calls["checkpoint"] == Path("checkpoint.pt")
    assert calls["device"] == "cpu"
    assert calls["allow_stale_ruleset"] is True
    assert json.loads(capsys.readouterr().out) == report
