from cr_bot.trackers.tower_hp_filter import TowerHPFilter


def update_support_right(tower_filter, value):
    return tower_filter.update({"enemy_support_right": value})["enemy_support_right"]


def test_rejects_overlong_ocr_instead_of_trimming_to_low_suffix():
    tower_filter = TowerHPFilter()

    assert update_support_right(tower_filter, 147424) == 4424
    assert update_support_right(tower_filter, 147424) == 4424
    assert update_support_right(tower_filter, 147424) == 4424
    assert update_support_right(tower_filter, 4424) == 4424


def test_rejects_repeated_leading_digit_dropout():
    tower_filter = TowerHPFilter()

    for _ in range(9):
        assert update_support_right(tower_filter, 4067) == 4424
    assert update_support_right(tower_filter, 4067) == 4067
    assert update_support_right(tower_filter, 467) == 4067
    assert update_support_right(tower_filter, 467) == 4067
    assert update_support_right(tower_filter, 467) == 4067


def test_allows_zero_for_destroyed_tower():
    tower_filter = TowerHPFilter()

    assert update_support_right(tower_filter, 0) == 4424
    assert update_support_right(tower_filter, 0) == 4424
    assert update_support_right(tower_filter, 0) == 0


def test_allows_stable_small_upward_ocr_correction():
    tower_filter = TowerHPFilter()

    for _ in range(9):
        assert update_support_right(tower_filter, 4020) == 4424
    assert update_support_right(tower_filter, 4020) == 4020
    assert update_support_right(tower_filter, 4067) == 4020
    assert update_support_right(tower_filter, 4067) == 4020
    assert update_support_right(tower_filter, 4067) == 4067


def test_rejects_large_upward_change():
    tower_filter = TowerHPFilter()

    assert update_support_right(tower_filter, 424) == 4424
    assert update_support_right(tower_filter, 424) == 4424
    assert update_support_right(tower_filter, 424) == 4424


def test_requires_longer_confirmation_for_large_drop():
    tower_filter = TowerHPFilter()

    for _ in range(3):
        update_support_right(tower_filter, 3453)
    for _ in range(8):
        assert update_support_right(tower_filter, 3053) == 4424


def test_accepts_sustained_large_drop():
    tower_filter = TowerHPFilter()

    for _ in range(9):
        assert update_support_right(tower_filter, 3453) == 4424
    assert update_support_right(tower_filter, 3453) == 3453
