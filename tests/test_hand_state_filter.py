from cr_bot.trackers.hand_state_filter import HandStateFilter


def _state(*cards):
    return {
        **{
            f"card_{idx}": (card, 99.0)
            for idx, card in enumerate(cards, start=1)
        },
        "next_card": ("cannon", 99.0),
    }


def _hand_names(state):
    return [state[f"card_{idx}"][0] for idx in range(1, 5)]


def test_rejects_transient_full_hand_change():
    filter_ = HandStateFilter()
    accepted = _state("log", "ice-spirit", "ice-golem", "hog-rider")
    transient = _state("bats", "rune-giant", "night-witch", "baby-dragon")

    filter_.update(accepted)
    filtered = filter_.update(transient)
    recovered = filter_.update(accepted)

    assert _hand_names(filtered) == _hand_names(accepted)
    assert _hand_names(recovered) == _hand_names(accepted)


def test_accepts_stable_full_hand_replacement_after_confirmation():
    filter_ = HandStateFilter()
    stale = _state("bats", "rune-giant", "night-witch", "baby-dragon")
    actual = _state("log", "ice-spirit", "ice-golem", "hog-rider")

    filter_.update(stale)
    first = filter_.update(actual)
    second = filter_.update(actual)
    third = filter_.update(actual)

    assert _hand_names(first) == _hand_names(stale)
    assert _hand_names(second) == _hand_names(stale)
    assert _hand_names(third) == _hand_names(actual)


def test_normal_single_slot_change_is_immediate():
    filter_ = HandStateFilter()
    before = _state("log", "ice-spirit", "ice-golem", "hog-rider")
    after = _state("log", "ice-spirit", "ice-golem", "None")

    filter_.update(before)

    assert _hand_names(filter_.update(after)) == _hand_names(after)
