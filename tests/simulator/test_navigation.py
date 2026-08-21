from __future__ import annotations

from simulator.fixed import distance_mtile
from simulator.geometry import cell_center_mtile, is_basic_deploy_cell
from simulator.navigation import (
    NavigationObstacle,
    _fixed_visibility_edges,
    plan_route,
    point_is_walkable,
    segment_is_walkable,
)
from simulator.ruleset import load_ruleset


def test_route_crosses_river_only_through_a_bridge() -> None:
    arena = load_ruleset().arena
    route = plan_route(
        arena,
        (9_000, 23_000),
        (9_000, 9_000),
        agent_radius_mtile=600,
    )

    assert len(route) >= 3
    for start, end in zip(route, route[1:]):
        assert segment_is_walkable(
            arena,
            start,
            end,
            agent_radius_mtile=600,
        )
    crossings = []
    for start, end in zip(route, route[1:]):
        if (start[1] - 16_000) * (end[1] - 16_000) <= 0 and start[1] != end[1]:
            x = start[0] + (end[0] - start[0]) * (16_000 - start[1]) // (
                end[1] - start[1]
            )
            crossings.append(x)
    assert crossings
    assert all(3_000 <= x <= 4_000 or 14_000 <= x <= 15_000 for x in crossings)


def test_route_avoids_an_inflated_building_obstacle() -> None:
    arena = load_ruleset().arena
    obstacle = NavigationObstacle(100, 9_000, 24_000, 600)
    route = plan_route(
        arena,
        (5_000, 24_000),
        (13_000, 24_000),
        agent_radius_mtile=600,
        obstacles=(obstacle,),
    )

    assert len(route) >= 3
    assert any(y != 24_000 for _, y in route[1:-1])
    for start, end in zip(route, route[1:]):
        assert segment_is_walkable(
            arena,
            start,
            end,
            agent_radius_mtile=600,
            obstacles=(obstacle,),
        )


def test_repeated_topology_reuses_fixed_visibility_without_changing_route() -> None:
    arena = load_ruleset().arena
    obstacles = (NavigationObstacle(100, 9_000, 24_000, 600),)
    _fixed_visibility_edges.cache_clear()

    first = plan_route(
        arena,
        (5_000, 24_000),
        (13_000, 24_000),
        agent_radius_mtile=600,
        obstacles=obstacles,
    )
    after_first = _fixed_visibility_edges.cache_info()
    second = plan_route(
        arena,
        (5_100, 24_000),
        (13_100, 24_000),
        agent_radius_mtile=600,
        obstacles=obstacles,
    )
    after_second = _fixed_visibility_edges.cache_info()

    assert first and second
    assert after_first.misses == 1
    assert after_second.misses == 1
    assert after_second.hits == after_first.hits + 1


def test_route_is_equivariant_under_full_arena_mirror() -> None:
    arena = load_ruleset().arena
    mirror = lambda point: (arena.width_mtile - point[0], arena.height_mtile - point[1])
    obstacle = NavigationObstacle(100, 8_500, 23_500, 600)
    route = plan_route(
        arena,
        (4_500, 25_000),
        (13_000, 20_000),
        agent_radius_mtile=450,
        obstacles=(obstacle,),
    )
    mirrored_route = plan_route(
        arena,
        mirror((4_500, 25_000)),
        mirror((13_000, 20_000)),
        agent_radius_mtile=450,
        obstacles=(NavigationObstacle(100, *mirror((8_500, 23_500)), 600),),
    )

    assert tuple(mirror(point) for point in route) == mirrored_route


def test_target_obstacle_can_be_excluded_for_melee_approach() -> None:
    arena = load_ruleset().arena
    target = NavigationObstacle(200, 9_000, 24_000, 600)
    blocked = plan_route(
        arena,
        (5_000, 24_000),
        (9_000, 24_000),
        agent_radius_mtile=600,
        obstacles=(target,),
    )
    approach = plan_route(
        arena,
        (5_000, 24_000),
        (9_000, 24_000),
        agent_radius_mtile=600,
        obstacles=(),
    )

    assert blocked == ()
    assert approach == ((5_000, 24_000), (9_000, 24_000))
    assert distance_mtile(*approach[-1], 9_000, 24_000) == 0


def test_every_walkable_hog_deploy_cell_has_a_mirrored_legal_tower_route() -> None:
    """Exhaust the policy grid instead of proving routing with one lane."""

    arena = load_ruleset().arena
    radius = 600
    enemy_princess_towers = ((3_500, 6_500), (14_500, 6_500))
    mirror = lambda point: (
        arena.width_mtile - point[0],
        arena.height_mtile - point[1],
    )
    route_count = 0
    for row in range(17, 32):
        for column in range(18):
            cell = (column, row)
            start = cell_center_mtile(cell)
            if not is_basic_deploy_cell(0, cell) or not point_is_walkable(
                arena, *start, radius
            ):
                continue
            for goal in enemy_princess_towers:
                route = plan_route(
                    arena,
                    start,
                    goal,
                    agent_radius_mtile=radius,
                )
                mirrored_route = plan_route(
                    arena,
                    mirror(start),
                    mirror(goal),
                    agent_radius_mtile=radius,
                )

                assert route and route[0] == start and route[-1] == goal
                assert tuple(mirror(point) for point in route) == mirrored_route
                assert all(
                    segment_is_walkable(
                        arena,
                        segment_start,
                        segment_end,
                        agent_radius_mtile=radius,
                    )
                    for segment_start, segment_end in zip(route, route[1:])
                )
                route_count += 1

    assert route_count == 380
