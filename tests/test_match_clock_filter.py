from cr_bot.trackers.match_clock import MatchClockFilter


def test_does_not_initialise_from_zero_timer_values():
    clock_filter = MatchClockFilter()

    for now_s in range(10):
        clock_filter.initialise(0.0, now_s)

    assert not clock_filter.initialised
    assert clock_filter.initial_seen_values == []


def test_initialises_from_valid_timer_values():
    clock_filter = MatchClockFilter()

    for now_s in range(10):
        clock_filter.initialise(173.0, now_s)

    assert clock_filter.initialised
    assert clock_filter.last_time_left_s == 173.0
