"""Stable event records emitted by the simulator.

Events are deliberately flat and JSON-safe so fidelity tooling can compare
them without importing or mutating the authoritative battle state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


EventValue: TypeAlias = str | int | bool | None


@dataclass(frozen=True, slots=True)
class SimEvent:
    tick: int
    sequence: int
    kind: str
    data: tuple[tuple[str, EventValue], ...] = ()

    @classmethod
    def create(
        cls,
        tick: int,
        sequence: int,
        kind: str,
        **data: EventValue,
    ) -> "SimEvent":
        return cls(tick, sequence, kind, tuple(sorted(data.items())))

    def get(self, key: str, default: EventValue = None) -> EventValue:
        return dict(self.data).get(key, default)

    def to_dict(self) -> dict[str, EventValue | dict[str, EventValue]]:
        return {
            "tick": self.tick,
            "sequence": self.sequence,
            "kind": self.kind,
            "data": dict(self.data),
        }
