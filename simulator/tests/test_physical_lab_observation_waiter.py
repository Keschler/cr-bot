from __future__ import annotations

from collections import deque

import pytest

from simulator.physical_lab.observation_waiter import (
    HogBridgeObservationWaiter,
    LiveTrackObservationSource,
    TrackObservation,
)
from simulator.physical_lab.devices import Frame
from simulator.physical_lab.schema import PhysicalLabError


def _row(
    y: int,
    timestamp: int,
    *,
    track_id: int = 7,
    x: int = 3_500,
    confidence: float = 0.99,
    card_id: str = "hog-rider",
    team: str = "ally",
) -> TrackObservation:
    return TrackObservation(track_id, card_id, team, confidence, x, y, timestamp)


def _waiter(batches, clock_values=None) -> HogBridgeObservationWaiter:
    queue = deque(batches)
    clocks = deque(clock_values or range(0, 1_000_000, 10))
    return HogBridgeObservationWaiter(
        lambda: queue.popleft() if queue else (),
        monotonic_clock_us=lambda: clocks.popleft(),
        poll=lambda: None,
    )


def test_waiter_returns_first_crossing_timestamp_after_three_confirmations() -> None:
    waiter = _waiter(
        [
            (_row(17_200, 100),),
            (_row(17_000, 120),),
            (_row(16_900, 140),),
            (_row(16_800, 160),),
        ]
    )

    assert waiter("hog_crosses_y_mtile", 17_000, 10_000) == 120


@pytest.mark.parametrize(
    ("event", "value"),
    (("hog_crosses_y_mtile", 17_001), ("elapsed_match_time_us", 17_000)),
)
def test_waiter_rejects_any_other_event_contract(event: str, value: int) -> None:
    with pytest.raises(PhysicalLabError, match="only supports"):
        _waiter([])(event, value, 10_000)


def test_waiter_fails_closed_on_ambiguous_hogs() -> None:
    with pytest.raises(PhysicalLabError, match="ambiguous"):
        _waiter([(_row(17_200, 100), _row(17_250, 100, track_id=8))])(
            "hog_crosses_y_mtile", 17_000, 10_000
        )


def test_waiter_fails_closed_on_track_switch() -> None:
    waiter = _waiter([(_row(17_200, 100),), (_row(17_000, 120, track_id=8),)])
    with pytest.raises(PhysicalLabError, match="switched"):
        waiter("hog_crosses_y_mtile", 17_000, 10_000)


def test_waiter_fails_closed_on_nonmonotonic_crossing() -> None:
    waiter = _waiter(
        [
            (_row(17_200, 100),),
            (_row(16_900, 120),),
            (_row(16_950, 140),),
        ]
    )
    with pytest.raises(PhysicalLabError, match="not monotonic"):
        waiter("hog_crosses_y_mtile", 17_000, 10_000)


def test_waiter_rejects_crossing_outside_reviewed_bridge() -> None:
    waiter = _waiter([(_row(17_200, 100, x=8_000),), (_row(17_000, 120, x=8_000),)])
    with pytest.raises(PhysicalLabError, match="outside"):
        waiter("hog_crosses_y_mtile", 17_000, 10_000)


@pytest.mark.parametrize(
    "batch",
    [
        (_row(17_200, 100, confidence=0.97),),
        (_row(17_200, 100, card_id="ram-rider"),),
        (_row(17_200, 100, team="enemy"),),
    ],
)
def test_unverified_identity_team_or_confidence_cannot_trigger(batch) -> None:
    waiter = _waiter([batch], clock_values=(0, 1, 20_000))
    with pytest.raises(TimeoutError):
        waiter("hog_crosses_y_mtile", 17_000, 10_000)


def test_waiter_requires_a_pre_boundary_sample() -> None:
    with pytest.raises(PhysicalLabError, match="pre-boundary"):
        _waiter([(_row(17_000, 120),)])("hog_crosses_y_mtile", 17_000, 10_000)


def test_live_source_records_each_polled_frame_exactly_once() -> None:
    frame = Frame("A", 4, 123, payload=b"synthetic")

    class Recorder:
        def __init__(self) -> None:
            self.frames = []

        def record_frame(self, item: Frame) -> None:
            self.frames.append(item)

    recorder = Recorder()
    expected = (_row(17_200, 123),)
    source = LiveTrackObservationSource(
        lambda: frame,
        recorder=recorder,
        analyze_frame=lambda observed: expected if observed is frame else (),
    )

    assert source() == expected
    assert recorder.frames == [frame]
