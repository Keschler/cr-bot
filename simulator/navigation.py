"""Deterministic obstacle-aware arena navigation.

The planner uses a small visibility graph rather than a frame-rate-dependent
physics library.  Static river gates and dynamically inflated circular
building/tower obstacles become candidate waypoints; Dijkstra then selects a
stable shortest route.  Integer arithmetic and explicit tie-breaking keep the
result portable and replayable.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import heapq
from math import isqrt
from typing import Iterable, Protocol

from .fixed import ceil_div, distance_mtile


NAVIGATION_CLEARANCE_MTILE = 50
NAVIGATION_SAMPLE_MTILE = 250
# Ceil(sqrt(1/2) * 1000): diagonal obstacle waypoints must be outside the
# inflated circle after integer rounding, not one or two milli-tiles inside it.
_DIAGONAL_PERMILLE = 708


class ArenaGeometry(Protocol):
    width_mtile: int
    height_mtile: int
    river_y_min_mtile: int
    river_y_max_mtile: int
    bridge_x_ranges_mtile: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class NavigationObstacle:
    uid: int
    x_mtile: int
    y_mtile: int
    radius_mtile: int


def point_is_walkable(arena: ArenaGeometry, x: int, y: int, radius: int = 0) -> bool:
    """Return whether a ground-unit center can occupy an arena point."""

    if not (radius <= x < arena.width_mtile - radius):
        return False
    if not (radius <= y < arena.height_mtile - radius):
        return False
    if arena.river_y_min_mtile < y < arena.river_y_max_mtile:
        return any(start <= x <= end for start, end in arena.bridge_x_ranges_mtile)
    return True


def _segment_intersects_circle(
    start: tuple[int, int],
    end: tuple[int, int],
    obstacle: NavigationObstacle,
    inflated_radius: int,
) -> bool:
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    px = obstacle.x_mtile - x0
    py = obstacle.y_mtile - y0
    length_sq = dx * dx + dy * dy
    radius_sq = inflated_radius * inflated_radius
    if length_sq == 0:
        return px * px + py * py < radius_sq
    projection = px * dx + py * dy
    if projection <= 0:
        return px * px + py * py < radius_sq
    if projection >= length_sq:
        qx = obstacle.x_mtile - x1
        qy = obstacle.y_mtile - y1
        return qx * qx + qy * qy < radius_sq
    cross = px * dy - py * dx
    return cross * cross < radius_sq * length_sq


def _segment_is_walkable_uncached(
    arena: ArenaGeometry,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    agent_radius_mtile: int,
    obstacles: Iterable[NavigationObstacle] = (),
    allow_river_crossing: bool = False,
) -> bool:
    """Check terrain and dynamic obstacles along one straight segment."""

    if not point_is_walkable(arena, *start, agent_radius_mtile):
        return False
    if not point_is_walkable(arena, *end, agent_radius_mtile):
        return False
    low_y = min(start[1], end[1])
    high_y = max(start[1], end[1])
    if (
        not allow_river_crossing
        and high_y > arena.river_y_min_mtile
        and low_y < arena.river_y_max_mtile
    ):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        span = abs(dx) if dy == 0 else abs(dy)
        samples = max(1, ceil_div(span, NAVIGATION_SAMPLE_MTILE))
        # The arena rectangle is convex and both endpoints already passed the
        # bounds check. Outside the open river strip every intermediate point
        # is therefore terrain-valid. Binary-search only the sample indices
        # whose y coordinate can be in the river; this is exactly equivalent
        # to checking all samples and avoids dozens of redundant calls on
        # long lane-to-tower visibility edges.
        def ascending_river_indices(start_y: int, delta_y: int) -> tuple[int, int]:
            if delta_y == 0:
                return (
                    (0, samples + 1)
                    if arena.river_y_min_mtile < start_y < arena.river_y_max_mtile
                    else (0, 0)
                )
            first = ceil_div(
                (arena.river_y_min_mtile - start_y + 1) * samples,
                delta_y,
            )
            stop = ceil_div(
                (arena.river_y_max_mtile - start_y) * samples,
                delta_y,
            )
            return max(0, first), min(samples + 1, stop)

        if dy >= 0:
            first, stop = ascending_river_indices(start[1], dy)
        else:
            # Reverse the exact integer sample sequence. For j=samples-i,
            # start_y + floor(dy*i/samples) equals
            # end_y + floor((-dy)*j/samples).
            reverse_first, reverse_stop = ascending_river_indices(end[1], -dy)
            first = max(0, samples - reverse_stop + 1)
            stop = min(samples + 1, samples - reverse_first + 1)

        for index in range(first, stop):
            x = start[0] + dx * index // samples
            y = start[1] + dy * index // samples
            if not point_is_walkable(arena, x, y, agent_radius_mtile):
                return False
    for obstacle in obstacles:
        inflated = agent_radius_mtile + obstacle.radius_mtile + NAVIGATION_CLEARANCE_MTILE
        if _segment_intersects_circle(start, end, obstacle, inflated):
            return False
    return True


@lru_cache(maxsize=131_072)
def _segment_is_walkable_cached(
    arena: ArenaGeometry,
    start: tuple[int, int],
    end: tuple[int, int],
    agent_radius_mtile: int,
    obstacle_rows: tuple[NavigationObstacle, ...],
    allow_river_crossing: bool,
) -> bool:
    return _segment_is_walkable_uncached(
        arena,
        start,
        end,
        agent_radius_mtile=agent_radius_mtile,
        obstacles=obstacle_rows,
        allow_river_crossing=allow_river_crossing,
    )


def segment_is_walkable(
    arena: ArenaGeometry,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    agent_radius_mtile: int,
    obstacles: Iterable[NavigationObstacle] = (),
    allow_river_crossing: bool = False,
) -> bool:
    """Check terrain and dynamic obstacles, reusing exact immutable queries."""

    return _segment_is_walkable_cached(
        arena,
        start,
        end,
        agent_radius_mtile,
        tuple(obstacles),
        allow_river_crossing,
    )


def _obstacle_waypoints(
    arena: ArenaGeometry,
    obstacle: NavigationObstacle,
    agent_radius: int,
) -> tuple[tuple[int, int], ...]:
    inflated = agent_radius + obstacle.radius_mtile + NAVIGATION_CLEARANCE_MTILE
    # Eight straight visibility edges form chords. Put vertices on the
    # circumscribed octagon (1/cos(22.5°) ~= 1.0824) so adjacent chords stay
    # outside the actual inflated collision circle.
    radius = ceil_div(inflated * 1_083, 1_000)
    diagonal = radius * _DIAGONAL_PERMILLE // 1_000
    offsets = (
        (-radius, 0),
        (-diagonal, -diagonal),
        (0, -radius),
        (diagonal, -diagonal),
        (radius, 0),
        (diagonal, diagonal),
        (0, radius),
        (-diagonal, diagonal),
    )
    return tuple(
        (obstacle.x_mtile + dx, obstacle.y_mtile + dy)
        for dx, dy in offsets
        if point_is_walkable(
            arena,
            obstacle.x_mtile + dx,
            obstacle.y_mtile + dy,
            agent_radius,
        )
    )


def _bridge_waypoints(arena: ArenaGeometry, radius: int) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for start, end in arena.bridge_x_ranges_mtile:
        center = (start + end) // 2
        for y in (
            arena.river_y_min_mtile - NAVIGATION_SAMPLE_MTILE,
            (arena.river_y_min_mtile + arena.river_y_max_mtile) // 2,
            arena.river_y_max_mtile + NAVIGATION_SAMPLE_MTILE,
        ):
            if point_is_walkable(arena, center, y, radius):
                result.append((center, y))
    return tuple(result)


@lru_cache(maxsize=256)
def _fixed_visibility_edges(
    arena: ArenaGeometry,
    agent_radius_mtile: int,
    obstacle_rows: tuple[NavigationObstacle, ...],
) -> tuple[
    tuple[tuple[int, int], ...],
    dict[tuple[tuple[int, int], tuple[int, int]], int],
]:
    """Cache visibility among topology-fixed bridge/structure waypoints."""

    raw_nodes = list(_bridge_waypoints(arena, agent_radius_mtile))
    for obstacle in obstacle_rows:
        raw_nodes.extend(_obstacle_waypoints(arena, obstacle, agent_radius_mtile))
    nodes = tuple(dict.fromkeys(raw_nodes))
    edges: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            if segment_is_walkable(
                arena,
                left,
                right,
                agent_radius_mtile=agent_radius_mtile,
                obstacles=obstacle_rows,
            ):
                key = (left, right) if left < right else (right, left)
                edges[key] = distance_mtile(*left, *right)
    return nodes, edges


@lru_cache(maxsize=256)
def _fixed_visibility_adjacency(
    arena: ArenaGeometry,
    agent_radius_mtile: int,
    obstacle_rows: tuple[NavigationObstacle, ...],
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[tuple[int, int], int], ...],
]:
    """Cache the topology adjacency used by every route through one layout."""

    nodes, edges = _fixed_visibility_edges(
        arena,
        agent_radius_mtile,
        obstacle_rows,
    )
    indices = {node: index for index, node in enumerate(nodes)}
    adjacency: list[list[tuple[int, int]]] = [[] for _ in nodes]
    for (left, right), cost in edges.items():
        adjacency[indices[left]].append((indices[right], cost))
        adjacency[indices[right]].append((indices[left], cost))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: item[0])
    return nodes, tuple(tuple(neighbors) for neighbors in adjacency)


def _plan_route_uncached(
    arena: ArenaGeometry,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    agent_radius_mtile: int,
    obstacles: Iterable[NavigationObstacle] = (),
) -> tuple[tuple[int, int], ...]:
    """Return a stable shortest waypoint route, including start and goal.

    An empty tuple means there is no legal route.  The caller should stop and
    retry after the obstacle topology changes rather than crossing terrain.
    """

    obstacle_rows = tuple(sorted(obstacles, key=lambda item: item.uid))
    if segment_is_walkable(
        arena,
        start,
        goal,
        agent_radius_mtile=agent_radius_mtile,
        obstacles=obstacle_rows,
    ):
        return start, goal

    fixed_nodes, fixed_adjacency = _fixed_visibility_adjacency(
        arena, agent_radius_mtile, obstacle_rows
    )
    raw_nodes = [start, goal]
    raw_nodes.extend(fixed_nodes)
    nodes = tuple(dict.fromkeys(raw_nodes))
    node_indices = {node: index for index, node in enumerate(nodes)}
    start_index = node_indices[start]
    goal_index = node_indices[goal]
    route_dx = goal[0] - start[0]
    route_dy = goal[1] - start[1]

    tie_ranks: list[tuple[int, int, int, int]] = []
    heuristics: list[int] = []
    for index, node in enumerate(nodes):
        node_dx = node[0] - start[0]
        node_dy = node[1] - start[1]
        # Cross/dot/distance are invariant under a 180-degree arena mirror.
        # The final index only distinguishes geometrically equivalent nodes.
        tie_ranks.append(
            (
                route_dx * node_dy - route_dy * node_dx,
                route_dx * node_dx + route_dy * node_dy,
                node_dx * node_dx + node_dy * node_dy,
                index,
            )
        )
        heuristics.append(distance_mtile(*node, *goal))

    # The previous implementation scanned every node and re-tested every
    # dynamic edge on every Dijkstra iteration.  The fixed visibility graph is
    # already cached by obstacle topology, so construct its adjacency once and
    # evaluate only the start/goal boundary edges for this route.  The sorted
    # neighbor order preserves the old index-order relaxation/tie behavior.
    fixed_node_indices = tuple(node_indices[node] for node in fixed_nodes)
    fixed_indices = {node: index for index, node in enumerate(fixed_nodes)}
    start_neighbors: list[tuple[int, int]] = []
    goal_costs: dict[int, int] = {}

    for node in fixed_nodes:
        node_index = node_indices[node]
        if node_index == goal_index:
            continue
        if segment_is_walkable(
            arena,
            start,
            node,
            agent_radius_mtile=agent_radius_mtile,
            obstacles=obstacle_rows,
        ):
            start_neighbors.append((node_index, distance_mtile(*start, *node)))
        if segment_is_walkable(
            arena,
            node,
            goal,
            agent_radius_mtile=agent_radius_mtile,
            obstacles=obstacle_rows,
        ):
            goal_costs[node_index] = distance_mtile(*node, *goal)
    start_neighbors.sort(key=lambda item: item[0])

    infinity = 1 << 62
    distances = [infinity] * len(nodes)
    previous: list[int | None] = [None] * len(nodes)
    visited = [False] * len(nodes)
    distances[start_index] = 0
    queue: list[tuple[int, int, tuple[int, int, int, int], int]] = [
        (heuristics[start_index], 0, tie_ranks[start_index], start_index)
    ]
    while queue:
        _priority, current_distance, _rank, current = heapq.heappop(queue)
        if visited[current] or current_distance != distances[current]:
            continue
        if current == goal_index:
            break
        visited[current] = True
        if current == start_index:
            neighbors = start_neighbors
        else:
            fixed_index = fixed_indices.get(nodes[current])
            if fixed_index is None:
                continue
            neighbors = [
                (fixed_node_indices[neighbor], cost)
                for neighbor, cost in fixed_adjacency[fixed_index]
                if fixed_node_indices[neighbor] != current
            ]
            goal_cost = goal_costs.get(current)
            if goal_cost is not None:
                neighbors.append((goal_index, goal_cost))
        for neighbor, cost in neighbors:
            if visited[neighbor]:
                continue
            candidate = distances[current] + cost
            if candidate < distances[neighbor] or (
                candidate == distances[neighbor]
                and (
                    previous[neighbor] is None
                    or tie_ranks[current] < tie_ranks[previous[neighbor]]
                )
            ):
                distances[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(
                    queue,
                    (
                        candidate + heuristics[neighbor],
                        candidate,
                        tie_ranks[neighbor],
                        neighbor,
                    ),
                )

    if distances[goal_index] == infinity:
        return ()
    indices = [goal_index]
    while indices[-1] != start_index:
        parent = previous[indices[-1]]
        if parent is None:
            return ()
        indices.append(parent)
    indices.reverse()
    return tuple(nodes[index] for index in indices)


@lru_cache(maxsize=32_768)
def _plan_route_cached(
    arena: ArenaGeometry,
    start: tuple[int, int],
    goal: tuple[int, int],
    agent_radius_mtile: int,
    obstacle_rows: tuple[NavigationObstacle, ...],
) -> tuple[tuple[int, int], ...]:
    """Memoize pure route results across lanes with the same topology."""

    return _plan_route_uncached(
        arena,
        start,
        goal,
        agent_radius_mtile=agent_radius_mtile,
        obstacles=obstacle_rows,
    )


def plan_route(
    arena: ArenaGeometry,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    agent_radius_mtile: int,
    obstacles: Iterable[NavigationObstacle] = (),
) -> tuple[tuple[int, int], ...]:
    """Return a stable route, reusing identical pure route queries."""

    obstacle_rows = tuple(sorted(obstacles, key=lambda item: item.uid))
    return _plan_route_cached(
        arena,
        start,
        goal,
        agent_radius_mtile,
        obstacle_rows,
    )


def next_waypoint(
    arena: ArenaGeometry,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    agent_radius_mtile: int,
    obstacles: Iterable[NavigationObstacle] = (),
) -> tuple[int, int]:
    route = plan_route(
        arena,
        start,
        goal,
        agent_radius_mtile=agent_radius_mtile,
        obstacles=obstacles,
    )
    return route[1] if len(route) >= 2 else start


__all__ = [
    "NAVIGATION_CLEARANCE_MTILE",
    "NavigationObstacle",
    "next_waypoint",
    "plan_route",
    "point_is_walkable",
    "segment_is_walkable",
]
