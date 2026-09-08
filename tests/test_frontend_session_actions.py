"""Pump-level tests for own/enemy tracker event forwarding (no models)."""

from types import SimpleNamespace

import pytest

from src.frontend import server as frontend_server
from src.frontend.services import session_manager as session_mgr
from src.frontend.session import (
    FrontendFrame,
    FrontendSession,
    INFERENCE_DEVICES,
    _OffsetFrameSource,
    _run_pump_loop,
    resolve_inference_devices,
)


def test_frame_at_endpoint_serves_history_jpegs():
    previous = session_mgr._session
    session = FrontendSession(mode="video")
    try:
        session_mgr._session = session
        assert frontend_server.api_frame_at(3).status_code == 204
        session.push(
            FrontendFrame(
                frame_index=3, timestamp_s=1.5, jpeg_bytes=b"fake-jpeg-bytes"
            )
        )
        response = frontend_server.api_frame_at(3)
        assert response.status_code == 200
        assert response.body == b"fake-jpeg-bytes"
        assert response.media_type == "image/jpeg"
        assert frontend_server.api_frame_at(99).status_code == 204
        session.push(FrontendFrame(frame_index=4, timestamp_s=2.0, jpeg_bytes=None))
        assert frontend_server.api_frame_at(4).status_code == 204
    finally:
        session_mgr._session = previous


def _fake_source(indices):
    state = {"frames": list(indices), "closed": False}

    class Source:
        def next_frame(self):
            if not state["frames"]:
                return None
            i = state["frames"].pop(0)
            return SimpleNamespace(image=None, frame_index=i, timestamp_s=float(i))

        def close(self):
            state["closed"] = True

    return Source(), state


def _drain(source):
    out = []
    while True:
        frame = source.next_frame()
        if frame is None:
            return out
        out.append(frame.frame_index)


def test_offset_source_skips_to_start_frame():
    source, state = _fake_source([0, 1, 2, 3, 4])
    wrapped = _OffsetFrameSource(source, 2)
    assert _drain(wrapped) == [2, 3, 4]
    wrapped.close()
    assert state["closed"] is True


def test_offset_source_zero_start_passes_through():
    source, _ = _fake_source([0, 1])
    assert _drain(_OffsetFrameSource(source, 0)) == [0, 1]


def test_offset_source_beyond_eof_yields_nothing():
    source, state = _fake_source([0, 1])
    wrapped = _OffsetFrameSource(source, 99)
    assert _drain(wrapped) == []
    wrapped.close()
    assert state["closed"] is True


def test_resolve_inference_devices_cpu_and_invalid():
    assert INFERENCE_DEVICES == ("auto", "cpu", "cuda")
    assert resolve_inference_devices("cpu") == ("cpu", "cpu")
    assert resolve_inference_devices("CPU ") == ("cpu", "cpu")
    for bad in ("tpu", "gpu", "cuda:1", 0, ["cpu"]):
        try:
            resolve_inference_devices(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"device {bad!r} should be rejected")


def test_resolve_inference_devices_missing_means_auto():
    expected = resolve_inference_devices("auto")
    assert resolve_inference_devices(None) == expected
    assert resolve_inference_devices("") == expected
    assert resolve_inference_devices("  ") == expected


def test_resolve_inference_devices_auto_and_cuda_follow_environment():
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except ImportError:
        cuda = False
    yolo, actor = resolve_inference_devices("auto")
    assert isinstance(yolo, str) and yolo
    assert actor == ("cuda" if cuda else "cpu")
    if cuda:
        assert resolve_inference_devices("cuda") == ("cuda", "cuda")
    else:
        try:
            resolve_inference_devices("cuda")
        except ValueError:
            pass
        else:
            raise AssertionError("cuda must be rejected without CUDA")


def test_video_start_rejects_bad_device_before_file_checks():
    request = SimpleNamespace(
        video_path="missing.mp4",
        frame_stride=1,
        start_frame=0,
        max_frames=None,
        checkpoint=None,
        device="tpu",
        yolo_image_size=896,
        adapt_rois=False,
        roi_set=None,
    )
    try:
        frontend_server.api_video_start(request)
    except Exception as error:
        assert getattr(error, "status_code", None) == 400
        assert "device" in str(getattr(error, "detail", error))
    else:
        raise AssertionError("bad device should fail with 400")


def test_live_start_rejects_bad_device():
    request = SimpleNamespace(
        serial="emulator-5554",
        transport="stream",
        checkpoint=None,
        device="tpu",
        calibration=None,
        execute=False,
        confirm_live=False,
    )
    try:
        frontend_server.api_live_start(request)
    except Exception as error:
        assert getattr(error, "status_code", None) == 400
        assert "device" in str(getattr(error, "detail", error))
    else:
        raise AssertionError("bad device should fail with 400")


def test_offset_source_uses_fast_forward_when_available():
    source, _ = _fake_source([0, 1, 2, 3, 4])
    calls = []
    source.fast_forward_to = lambda index: calls.append(index)
    wrapped = _OffsetFrameSource(source, 2)
    assert _drain(wrapped) == [2, 3, 4]
    assert calls == [2]


def test_offset_source_fast_forward_matches_sequential_skip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    try:
        from simulator.physical_lab.prototype_controller import VideoFrameSource
    except ImportError:
        from physical_lab.prototype_controller import VideoFrameSource  # type: ignore
    video = tmp_path / "seek.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video writer is unavailable")
    try:
        for index in range(40):
            image = np.full((48, 64, 3), (index * 6) % 256, dtype=np.uint8)
            cv2.putText(
                image, str(index), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
            )
            writer.write(image)
    finally:
        writer.release()
    sequential = VideoFrameSource(video)
    fast = VideoFrameSource(video)
    try:
        first_sequential = _OffsetFrameSource(sequential, 25).next_frame()
        wrapped_fast = _OffsetFrameSource(fast, 25)
        first_fast = wrapped_fast.next_frame()
        assert first_sequential is not None and first_fast is not None
        assert int(first_fast.frame_index) == 25
        assert int(first_sequential.frame_index) == 25
        assert first_fast.timestamp_s == first_sequential.timestamp_s
        assert np.array_equal(first_fast.image, first_sequential.image)
        # The remainder after the seek stays exact.
        assert int(wrapped_fast.next_frame().frame_index) == 26
    finally:
        sequential.close()
        fast.close()


def _source(frames):
    state = {"frames": list(frames)}

    class Source:
        def next_frame(self):
            return state["frames"].pop(0) if state["frames"] else None

        def close(self):
            state["closed"] = True

    return Source(), state


def _own(card, **kwargs):
    values = {
        "card": card,
        "slot_idx": 0,
        "cell": (9, 20),
        "video_time_s": 2.0,
        "time_left_s": 100.0,
        "played_via": None,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def _enemy(event_id, confirmed, **kwargs):
    values = {
        "event_id": event_id,
        "card": "knight",
        "cost": 3,
        "cell": (4, 6),
        "track_id": 7,
        "video_time_s": 2.0,
        "time_left_s": 100.0,
        "clock_confirmed": confirmed,
        "frame_confirmed": False,
        "avg_confidence": 0.8,
        "is_spell": False,
        "played_via": None,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def _step():
    return SimpleNamespace(
        in_game=True,
        should_emit=True,
        game_state=SimpleNamespace(hud=SimpleNamespace(hand_cards=(), elixir_self=5.0)),
    )


def test_pump_forwards_new_tracker_events_only():
    session = FrontendSession(mode="video")
    frames = [
        SimpleNamespace(image=None, frame_index=0, timestamp_s=1.0),
        SimpleNamespace(image=None, frame_index=1, timestamp_s=2.0),
        SimpleNamespace(image=None, frame_index=2, timestamp_s=3.0),
    ]
    source, _ = _source(frames)
    own_tracker = SimpleNamespace(actions=[])
    enemy_tracker = SimpleNamespace(detected_card_plays=[])
    calls = {"n": 0}

    class MatchSession:
        own_action_tracker = own_tracker
        enemy_card_tracker = enemy_tracker

        def process(self, analysis, *, frame, now_s):
            calls["n"] += 1
            if calls["n"] == 2:
                own_tracker.actions.append(_own("hog-rider"))
                enemy_tracker.detected_card_plays.append(_enemy("u1", False))
            if calls["n"] == 3:
                # Replaced tracker (match reset) restarts the own baseline;
                # enemy reconciliation prunes the unconfirmed play.
                self.own_action_tracker = SimpleNamespace(
                    actions=[_own("musketeer", video_time_s=3.0)]
                )
                enemy_tracker.detected_card_plays[:] = [_enemy("e1", True, video_time_s=3.0)]
            return _step()

    def record_fn(source_frame, step, *, action, result):
        return SimpleNamespace(
            as_dict=lambda: {"frame_index": source_frame.frame_index, "result": result}
        )

    summary = _run_pump_loop(
        frontend_session=session,
        frame_source=source,
        detector=object(),
        actor=SimpleNamespace(reset=lambda: None),
        match_session=MatchSession(),
        observation_builder=lambda step: None,
        dispatch_fn=lambda *args, **kwargs: None,
        normalize_frame_fn=lambda image: image,
        process_frame_fn=lambda *args, **kwargs: object(),
        filter_live_analysis_fn=None,
        action_to_dict_fn=lambda action: {"kind": "wait"},
        record_fn=record_fn,
        detection_filter=None,
        execute=False,
        phone=None,
        calibration=None,
        max_frames=3,
        poll_interval_s=0.0,
        min_action_interval_s=0.0,
        post_action_delay_s=0.0,
        stop_event=None,
    )

    assert summary["frames"] == 3
    history = session.history
    assert len(history) == 3
    # Frame 0: nothing tracked yet.
    assert history[0].own_actions == []
    assert history[0].enemy_plays == []
    # Frame 1: new own play forwarded; unconfirmed enemy play excluded.
    assert [a["card"] for a in history[1].own_actions] == ["hog-rider"]
    assert history[1].own_actions[0]["cell"] == [9, 20]
    assert history[1].enemy_plays == []
    # Frame 2: replaced own tracker restarts baseline (no re-send of hog);
    # only the confirmed enemy play is forwarded.
    assert [a["card"] for a in history[2].own_actions] == ["musketeer"]
    assert [p["event_id"] for p in history[2].enemy_plays] == ["e1"]
    assert history[2].enemy_plays[0]["clock_confirmed"] is True


def test_pump_survives_missing_trackers():
    session = FrontendSession(mode="video")
    frames = [SimpleNamespace(image=None, frame_index=0, timestamp_s=1.0)]
    source, _ = _source(frames)

    class MatchSession:
        def process(self, analysis, *, frame, now_s):
            return _step()

    def record_fn(source_frame, step, *, action, result):
        return SimpleNamespace(as_dict=lambda: {"frame_index": 0})

    _run_pump_loop(
        frontend_session=session,
        frame_source=source,
        detector=object(),
        actor=SimpleNamespace(reset=lambda: None),
        match_session=MatchSession(),
        observation_builder=lambda step: None,
        dispatch_fn=lambda *args, **kwargs: None,
        normalize_frame_fn=lambda image: image,
        process_frame_fn=lambda *args, **kwargs: object(),
        filter_live_analysis_fn=None,
        action_to_dict_fn=lambda action: {"kind": "wait"},
        record_fn=record_fn,
        detection_filter=None,
        execute=False,
        phone=None,
        calibration=None,
        max_frames=1,
        poll_interval_s=0.0,
        min_action_interval_s=0.0,
        post_action_delay_s=0.0,
        stop_event=None,
    )
    assert session.history[0].own_actions == []
    assert session.history[0].enemy_plays == []


# ---------------------------------------------------------------------------
# Label correction / what-if re-evaluation
# ---------------------------------------------------------------------------


def _labeled_frame(**overrides):
    visual_state = {
        "hand": ["knight", "archers", "fireball", "zap"],
        "next_card": "giant",
        "elixir": 5.0,
        "enemy_elixir_est": 4.0,
        "time_left_s": 120.0,
        "total_remaining_s": 120.0,
        "overtime": False,
        "tower_hp_self": [4424.0, 7032.0, 4424.0],
        "tower_hp_enemy": [4424.0, 7032.0, 4424.0],
        "own_king_active": False,
        "enemy_king_active": False,
        "seen_enemy_cards": [],
        "arena_px": [30, 314, 1010, 1480],
    }
    detections = [
        {
            "class_name": "knight",
            "team": "enemy",
            "confidence": 0.9,
            "track_id": 7,
            "box": [400.0, 800.0, 460.0, 900.0],
            "center": [430.0, 850.0],
            "estimated_hp": 1452.0,
        },
        {
            "class_name": "archers",
            "team": "ally",
            "confidence": 0.8,
            "track_id": 3,
            "box": [400.0, 1200.0, 450.0, 1280.0],
            "center": [425.0, 1240.0],
            "estimated_hp": 304.0,
        },
    ]
    values = {
        "frame_index": 10,
        "timestamp_s": 5.0,
        "jpeg_bytes": None,
        "record": {"visual_state": visual_state},
        "in_game": True,
        "emitted": True,
        "frame_width": 1080,
        "frame_height": 2400,
        "detections": detections,
    }
    values.update(overrides)
    return FrontendFrame(**values)


def _labeled_session():
    session = FrontendSession(mode="video")
    session.push(_labeled_frame())
    return session


def test_reevaluate_missing_frame_raises_not_found():
    from src.frontend.session import FrameNotFoundError, reevaluate_frame

    session = FrontendSession(mode="video")
    try:
        reevaluate_frame(session, 99, {"updates": []})
    except FrameNotFoundError:
        pass
    else:
        raise AssertionError("evicted frame should raise FrameNotFoundError")


def test_reevaluate_requires_live_actor():
    from src.frontend.session import NoActorError, reevaluate_frame

    session = _labeled_session()
    assert session.actor is None
    try:
        reevaluate_frame(session, 10, {"updates": []})
    except NoActorError:
        pass
    else:
        raise AssertionError("missing actor should raise NoActorError")


def test_reevaluate_rejects_non_emitted_frame():
    from src.frontend.session import CorrectionUnprocessableError, reevaluate_frame

    session = FrontendSession(mode="video")
    session.push(_labeled_frame(emitted=False))
    session.actor = object()
    try:
        reevaluate_frame(session, 10, {"updates": []})
    except CorrectionUnprocessableError:
        pass
    else:
        raise AssertionError("non-emitted frame should be rejected")


def test_reevaluate_rejects_malformed_edits():
    from src.frontend.session import reevaluate_frame

    bad_edits = [
        "not-a-dict",
        {"bogus": []},
        {"updates": "not-a-list"},
        {"updates": [{"target": {"track": 7}}]},
        {"updates": [{"target": {"track": 7}, "team": "neutral"}]},
        {"updates": [{"target": {"track": 7}, "class_name": "not-a-card"}]},
        {"deletes": [{"nontarget": 1}]},
        {"adds": [{"box": [0, 0, 10], "class_name": "knight", "team": "ally"}]},
        {"adds": [{"box": [50, 50, 10, 10], "class_name": "knight", "team": "ally"}]},
        {"adds": [{"box": [0, 0, 50, 50], "class_name": "knight"}]},
        {},
    ]
    for edits in bad_edits:
        session = _labeled_session()
        session.actor = object()
        try:
            reevaluate_frame(session, 10, edits)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{edits!r} should be rejected")
        assert session.find_frame(10).corrected is None


def test_reevaluate_rejects_unmatched_target():
    from src.frontend.session import CorrectionUnprocessableError, reevaluate_frame

    session = _labeled_session()
    session.actor = object()
    try:
        reevaluate_frame(session, 10, {"updates": [{"target": {"track": 999}, "team": "ally"}]})
    except CorrectionUnprocessableError:
        pass
    else:
        raise AssertionError("unmatched target should be rejected")


def test_reevaluate_move_only_update_repositions():
    from src.frontend.session import _apply_correction_edits

    base = _labeled_frame().detections
    rows, applied = _apply_correction_edits(
        base,
        {"updates": [{"target": {"track": 7}, "box": [500.0, 900.0, 560.0, 1000.0]}]},
        frame_width=1080,
        frame_height=2400,
    )
    moved = next(r for r in rows if r["track_id"] == 7)
    assert moved["box"] == [500.0, 900.0, 560.0, 1000.0]
    assert moved["center"] == [530.0, 950.0]
    assert moved["class_name"] == "knight" and moved["team"] == "enemy"
    assert applied["counts"] == {"updated": 1, "deleted": 0, "added": 0}
    # Same box is a no-op and rejected; out-of-frame rejected too.
    for bad in (
        {"updates": [{"target": {"track": 7}, "box": [400.0, 800.0, 460.0, 900.0]}]},
        {"updates": [{"target": {"track": 7}, "box": [0.0, 0.0, 2000.0, 100.0]}]},
        {"updates": [{"target": {"track": 7}, "box": [60.0, 60.0, 10.0, 10.0]}]},
    ):
        try:
            _apply_correction_edits(base, bad, frame_width=1080, frame_height=2400)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should be rejected")


def test_reevaluate_forward_stores_correction_and_restores_hidden():
    import torch

    from simulator.rl.model import AutoregressiveLogits, RecurrentPolicyOutput
    from src.frontend.session import clear_reevaluation, reevaluate_frame

    placement = torch.zeros(1, 1, 4, 32, 18)

    class FakePolicy:
        def __init__(self):
            self.forward_calls = 0

        def initial_hidden(self, batch_size, *, device=None, dtype=None):
            return torch.zeros(1, batch_size, 8)

        def forward(self, raster, global_features, entities, entity_mask, **kwargs):
            self.forward_calls += 1
            return RecurrentPolicyOutput(
                logits=AutoregressiveLogits(
                    mode=torch.zeros(1, 1, 2),
                    card=torch.zeros(1, 1, 4),
                    placement=placement,
                ),
                encoded_features=torch.zeros(1, 1, 8),
                recurrent_features=torch.zeros(1, 1, 8),
                final_hidden=torch.ones(1, 1, 8),
                belief_logits=None,
            )

    class FakeActor:
        def __init__(self, policy):
            self.policy = policy
            self._hidden = None
            self._torch = torch

        @property
        def device(self):
            return torch.device("cpu")

    session = _labeled_session()
    actor = FakeActor(FakePolicy())
    session.actor = actor
    sentinel = torch.zeros(1, 1, 8)
    actor._hidden = sentinel
    correction = reevaluate_frame(
        session,
        10,
        {
            "updates": [{"target": {"track": 7}, "class_name": "musketeer"}],
            "deletes": [{"target": {"label": "archers", "center": [425.0, 1240.0]}}],
            "adds": [
                {
                    "box": [600.0, 900.0, 660.0, 1000.0],
                    "class_name": "giant",
                    "team": "enemy",
                }
            ],
        },
    )
    assert actor.policy.forward_calls == 1
    assert actor._hidden is sentinel
    assert correction["revised"] is True
    assert correction["applied"]["counts"] == {"updated": 1, "deleted": 1, "added": 1}
    assert len(correction["suggestions"]) > 0
    assert isinstance(correction["action"], dict)
    stored = session.find_frame(10).corrected
    assert stored is not None and stored["applied"]["counts"]["added"] == 1
    assert clear_reevaluation(session, 10) is True
    assert session.find_frame(10).corrected is None
    assert clear_reevaluation(session, 99) is False


def test_frame_json_carries_detections_and_correction():
    frame = _labeled_frame()
    payload = frontend_server._frame_to_json(frame)
    assert len(payload["detections"]) == 2
    assert payload["corrected"] is None
    frame.corrected = {"revised": True}
    assert frontend_server._frame_to_json(frame)["corrected"] == {"revised": True}


def test_labels_endpoint_shape():
    payload = frontend_server.api_labels()
    assert payload["teams"] == ["ally", "enemy"]
    assert isinstance(payload["labels"], list)
    assert "knight" in payload["labels"]
    assert not [label for label in payload["labels"] if "bar" in label.lower()]


def test_reevaluate_endpoints_status_paths():
    from src.frontend.server import (
        ReevaluateRequest,
        api_frame_reevaluate,
        api_frame_reevaluate_revert,
    )

    previous = session_mgr._session
    session = _labeled_session()
    try:
        session_mgr._session = session
        try:
            api_frame_reevaluate(99, ReevaluateRequest(updates=[]))
        except Exception as error:
            assert getattr(error, "status_code", None) == 404
        else:
            raise AssertionError("evicted frame should 404")
        try:
            api_frame_reevaluate_revert(99)
        except Exception as error:
            assert getattr(error, "status_code", None) == 404
        else:
            raise AssertionError("evicted revert should 404")
        # Frame present but no live actor.
        try:
            api_frame_reevaluate(10, ReevaluateRequest(updates=[]))
        except Exception as error:
            assert getattr(error, "status_code", None) == 409
        else:
            raise AssertionError("missing actor should 409")
        # Malformed edits with an actor present (no forward reached).
        session.actor = object()
        try:
            api_frame_reevaluate(10, ReevaluateRequest(updates=[{"team": "x"}]))
        except Exception as error:
            assert getattr(error, "status_code", None) == 400
        else:
            raise AssertionError("malformed edits should 400")
        # Unmatched target.
        try:
            api_frame_reevaluate(
                10, ReevaluateRequest(deletes=[{"target": {"track": 999}}])
            )
        except Exception as error:
            assert getattr(error, "status_code", None) == 422
        else:
            raise AssertionError("unmatched target should 422")
        assert session.find_frame(10).corrected is None
        # Revert on a present frame without a correction succeeds.
        assert api_frame_reevaluate_revert(10) == {"frame_index": 10, "reverted": True}
    finally:
        session_mgr._session = previous
