"""Integer arithmetic helpers used by the deterministic simulator core.

The authoritative simulation never stores Python floating-point values.  World
coordinates are milli-tiles, elixir is milli-elixir, and durations are integer
microseconds.  Conversion to floats is reserved for observation/reporting
adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt


POSITION_SCALE = 1_000
ELIXIR_SCALE = 1_000
SECOND_US = 1_000_000
PERMILLE = 1_000


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)


def distance_mtile(x0: int, y0: int, x1: int, y1: int) -> int:
    """Return a deterministic integer Euclidean distance in milli-tiles."""

    return isqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def move_towards(
    x: int,
    y: int,
    target_x: int,
    target_y: int,
    distance: int,
) -> tuple[int, int]:
    """Move at most ``distance`` milli-tiles without floating-point math."""

    if distance <= 0:
        return x, y
    dx = target_x - x
    dy = target_y - y
    remaining = isqrt(dx * dx + dy * dy)
    if remaining <= distance:
        return target_x, target_y
    if remaining == 0:
        return x, y
    next_x = x + dx * distance // remaining
    next_y = y + dy * distance // remaining
    # Integer division can produce no progress for extremely small diagonal
    # steps. The simulator's normal step is much larger, but keep the helper
    # total for fuzzed rulesets.
    if next_x == x and dx:
        next_x += 1 if dx > 0 else -1
    if next_y == y and dy:
        next_y += 1 if dy > 0 else -1
    return next_x, next_y


@dataclass(slots=True)
class DeterministicRng:
    """Small SplitMix64 generator with explicit, serializable state."""

    state: int

    _MASK = (1 << 64) - 1

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (value ^ (value >> 31)) & self._MASK

    def randbelow(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        # Rejection avoids modulo bias and, more importantly here, has stable
        # behavior independent of Python's random implementation.
        limit = ((1 << 64) // bound) * bound
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound

    def shuffle(self, values: list[str]) -> None:
        for index in range(len(values) - 1, 0, -1):
            other = self.randbelow(index + 1)
            values[index], values[other] = values[other], values[index]
