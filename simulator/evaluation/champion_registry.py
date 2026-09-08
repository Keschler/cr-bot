"""Immutable champion registry: the monotonic chain of accepted policies.

The current champion is immutable during one evaluation generation.  A
challenger that passes every mandatory gate appends a new entry; a failed
challenger changes nothing.  V3 remains a permanent historical anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True, slots=True)
class ChampionEntry:
    """One frozen champion checkpoint with its exact experiment identity."""

    name: str
    checkpoint_path: str
    checkpoint_sha256: str
    code_revision: str
    contract_hash: str
    manifest_hash: str
    summary: str = ""

    def __post_init__(self) -> None:
        for name in (
            "name",
            "checkpoint_path",
            "checkpoint_sha256",
            "code_revision",
            "contract_hash",
            "manifest_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.summary, str):
            raise TypeError("summary must be a string")

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "code_revision": self.code_revision,
            "contract_hash": self.contract_hash,
            "manifest_hash": self.manifest_hash,
            "summary": self.summary,
        }


class ChampionRegistry:
    """Append-only ordered registry; index 0 is the oldest anchor (V3)."""

    def __init__(self, entries: list[ChampionEntry] | tuple[ChampionEntry, ...] = ()) -> None:
        self._entries = list(entries)
        for entry in self._entries:
            if not isinstance(entry, ChampionEntry):
                raise TypeError("entries must be ChampionEntry objects")

    def __len__(self) -> int:
        return len(self._entries)

    def current(self) -> ChampionEntry:
        if not self._entries:
            raise ValueError("champion registry is empty")
        return self._entries[-1]

    def history(self) -> tuple[ChampionEntry, ...]:
        return tuple(self._entries)

    def promote(self, entry: ChampionEntry) -> ChampionEntry:
        """Append a validated challenger as the new champion.

        Promotion itself is decided by :mod:`evaluation.promotion_report`;
        this method only enforces monotonicity (a challenger must cite the
        exact contract generation it was evaluated under).
        """

        if not isinstance(entry, ChampionEntry):
            raise TypeError("entry must be a ChampionEntry")
        self._entries.append(entry)
        return entry

    def chain_hash(self) -> str:
        """Digest of the whole promotion chain (tamper-evident history)."""

        encoded = json.dumps(
            [entry.as_dict() for entry in self._entries],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_json(self) -> str:
        return json.dumps(
            {"entries": [entry.as_dict() for entry in self._entries]},
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> "ChampionRegistry":
        raw = json.loads(payload)
        entries = [ChampionEntry(**item) for item in raw.get("entries", [])]
        return cls(entries)


__all__ = ["ChampionEntry", "ChampionRegistry"]
