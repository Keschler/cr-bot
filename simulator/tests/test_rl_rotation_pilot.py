from simulator.rl.rotation_pilot import build_pool


def test_rotation_pilot_pool_is_unique_and_reproducible():
    decks = build_pool(100)
    assert len(decks) == 12
    assert len(set(map(frozenset, decks))) == 12
    assert decks == build_pool(100)


def test_rotation_pilot_pool_size_is_not_lane_count():
    decks = build_pool(100, size=10)
    assert len(decks) == 10
    assert len(set(map(frozenset, decks))) == 10
