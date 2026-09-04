"""Pump-level tests for own/enemy tracker event forwarding (no models)."""

from types import SimpleNamespace

from src.frontend import server as frontend_server
from src.frontend.session import (
    FrontendFrame,
    FrontendSession,
    _OffsetFrameSource,
    _run_pump_loop,
)


def test_frame_at_endpoint_serves_history_jpegs():
    previous = frontend_server._session
    session = FrontendSession(mode="video")
    try:
        frontend_server._session = session
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
        frontend_server._session = previous


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
