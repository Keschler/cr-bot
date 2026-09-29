from __future__ import annotations

from random import Random

from simulator.fixed import move_towards


def test_move_towards_is_equivariant_under_full_arena_mirror() -> None:
    width_mtile = 18_000
    height_mtile = 32_000
    mirror = lambda point: (
        width_mtile - point[0],
        height_mtile - point[1],
    )
    rng = Random(20260929)

    for _ in range(2_000):
        start = (rng.randrange(width_mtile + 1), rng.randrange(height_mtile + 1))
        target = (rng.randrange(width_mtile + 1), rng.randrange(height_mtile + 1))
        distance = rng.randrange(0, 5_000)
        moved = move_towards(*start, *target, distance)

        assert move_towards(*mirror(start), *mirror(target), distance) == mirror(moved)


def test_move_towards_truncates_each_signed_component_symmetrically() -> None:
    assert move_towards(0, 0, 10, 10, 3) == (2, 2)
    assert move_towards(0, 0, -10, -10, 3) == (-2, -2)
