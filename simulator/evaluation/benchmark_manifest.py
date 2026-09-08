"""Sealed promotion manifests: fresh paired cells sampled after freezing.

A promotion suite is valid only when its exact cells did not exist until
*after* the challenger checkpoint hash was fixed.  Callers pass the frozen
challenger hash in; the manifest hash covers it, proving the ordering.
Once revealed, a manifest is retired and may never serve as promotion
evidence again (it may become diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True, slots=True)
class EvalCell:
    """One paired evaluation cell shared by champion and challenger."""

    cell_id: str
    stratum: str
    deck_id: str
    strategy_id: str
    side: int
    pair_id: str
    seed_env: int
    seed_policy: int
    horizon_decisions: int

    def __post_init__(self) -> None:
        for name in ("cell_id", "stratum", "deck_id", "strategy_id", "pair_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.side) is not int or self.side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        for name in ("seed_env", "seed_policy", "horizon_decisions"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "stratum": self.stratum,
            "deck_id": self.deck_id,
            "strategy_id": self.strategy_id,
            "side": self.side,
            "pair_id": self.pair_id,
            "seed_env": self.seed_env,
            "seed_policy": self.seed_policy,
            "horizon_decisions": self.horizon_decisions,
        }


@dataclass(frozen=True, slots=True)
class SealedManifest:
    """Frozen cell list with its own hash and the challenger it seals."""

    manifest_hash: str
    challenger_hash: str
    contract_hash: str
    cells: tuple[EvalCell, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_hash, str) or not self.manifest_hash:
            raise ValueError("manifest_hash must be a non-empty string")
        if not isinstance(self.challenger_hash, str) or not self.challenger_hash:
            raise ValueError("challenger_hash must be a non-empty string")
        if not isinstance(self.contract_hash, str) or not self.contract_hash:
            raise ValueError("contract_hash must be a non-empty string")
        object.__setattr__(self, "cells", tuple(self.cells))
        for cell in self.cells:
            if not isinstance(cell, EvalCell):
                raise TypeError("cells must be EvalCell objects")

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_hash": self.manifest_hash,
            "challenger_hash": self.challenger_hash,
            "contract_hash": self.contract_hash,
            "cells": [cell.as_dict() for cell in self.cells],
        }


_RETIRED_MANIFESTS: set[str] = set()


def _stable_int(seed_material: str) -> int:
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sample_sealed_manifest(
    *,
    challenger_hash: str,
    contract_hash: str,
    strata_counts: dict[str, int],
    deck_ids: dict[str, list[str]],
    strategy_ids: dict[str, list[str]],
    manifest_seed: int,
    horizon_decisions: int = 0,
) -> SealedManifest:
    """Sample fresh cells deterministically; every stratum gets side swaps.

    ``deck_ids``/``strategy_ids`` map stratum -> pool.  Each draw creates a
    side-0 cell and its side-1 mirror sharing one ``pair_id`` so later
    statistics can bootstrap the block instead of the individual games.
    """

    if not isinstance(challenger_hash, str) or not challenger_hash:
        raise ValueError("challenger_hash must be a non-empty string")
    if not isinstance(contract_hash, str) or not contract_hash:
        raise ValueError("contract_hash must be a non-empty string")
    if type(manifest_seed) is not int or manifest_seed < 0:
        raise ValueError("manifest_seed must be a non-negative integer")
    if not strata_counts:
        raise ValueError("strata_counts must not be empty")

    cells: list[EvalCell] = []
    for stratum, count in strata_counts.items():
        if type(count) is not int or count <= 0:
            raise ValueError(f"count for stratum {stratum!r} must be positive")
        decks = deck_ids.get(stratum)
        strategies = strategy_ids.get(stratum)
        if not decks or not strategies:
            raise ValueError(f"stratum {stratum!r} needs non-empty deck/strategy pools")
        for draw in range(count):
            deck = decks[_stable_int(f"{manifest_seed}\x1f{stratum}\x1fdeck\x1f{draw}") % len(decks)]
            strategy = strategies[
                _stable_int(f"{manifest_seed}\x1f{stratum}\x1fstrategy\x1f{draw}") % len(strategies)
            ]
            pair_id = f"{stratum}::draw-{draw:04d}"
            for side in (0, 1):
                seed_env = _stable_int(f"{manifest_seed}\x1f{pair_id}\x1fenv\x1f{side}")
                seed_policy = _stable_int(f"{manifest_seed}\x1f{pair_id}\x1fpolicy\x1f{side}")
                cell_id = f"{pair_id}::side-{side}"
                cells.append(
                    EvalCell(
                        cell_id=cell_id,
                        stratum=stratum,
                        deck_id=deck,
                        strategy_id=strategy,
                        side=side,
                        pair_id=pair_id,
                        seed_env=seed_env,
                        seed_policy=seed_policy,
                        horizon_decisions=horizon_decisions,
                    )
                )
    encoded = json.dumps(
        {
            "challenger_hash": challenger_hash,
            "contract_hash": contract_hash,
            "manifest_seed": manifest_seed,
            "cells": [cell.as_dict() for cell in cells],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest_hash = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    if manifest_hash in _RETIRED_MANIFESTS:
        raise ValueError("manifest seed reuses a retired manifest; choose a fresh seed")
    return SealedManifest(
        manifest_hash=manifest_hash,
        challenger_hash=challenger_hash,
        contract_hash=contract_hash,
        cells=tuple(cells),
    )


def retire_manifest(manifest_hash: str) -> None:
    """Mark a revealed suite as spent; it may never promote again."""

    if not isinstance(manifest_hash, str) or not manifest_hash:
        raise ValueError("manifest_hash must be a non-empty string")
    _RETIRED_MANIFESTS.add(manifest_hash)


def is_retired(manifest_hash: str) -> bool:
    return manifest_hash in _RETIRED_MANIFESTS


def manifest_cells(manifest: SealedManifest) -> tuple[EvalCell, ...]:
    if not isinstance(manifest, SealedManifest):
        raise TypeError("manifest must be a SealedManifest")
    return manifest.cells


__all__ = [
    "EvalCell",
    "SealedManifest",
    "is_retired",
    "manifest_cells",
    "retire_manifest",
    "sample_sealed_manifest",
]
