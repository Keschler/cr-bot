"""Sealed promotion manifests: fresh paired cells sampled after freezing.

A promotion suite is valid only when sampled after the challenger checkpoint
hash was fixed. Callers supply that hash; its ordering must be established by
the surrounding experiment record. The durable registry prevents a retired
manifest or identical cell set from being sampled for another challenger.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator


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


_DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "outputs/simulator/evaluation/manifest_registry.json"


@contextmanager
def _registry(registry_path: str | Path | None) -> Iterator[tuple[Path, dict[str, object]]]:
    """Lock and persist the manifest ledger across independent processes."""

    path = Path(registry_path) if registry_path is not None else _DEFAULT_REGISTRY
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / f"{path.name}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data: dict[str, object] = json.loads(path.read_text()) if path.exists() else {}
        data.setdefault("issued_cells", {})
        data.setdefault("retired_manifests", [])
        data.setdefault("retired_cells", [])
        try:
            yield path, data
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _save_registry(path: Path, data: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


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
    registry_path: str | Path | None = None,
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
    # Cell identity excludes the challenger: a retired suite cannot be reused
    # merely by sealing identical draws for a new checkpoint.
    cell_payload = json.dumps([cell.as_dict() for cell in cells], sort_keys=True, separators=(",", ":")).encode()
    cell_hash = f"sha256:{hashlib.sha256(cell_payload).hexdigest()}"
    with _registry(registry_path) as (path, data):
        if manifest_hash in data["retired_manifests"] or cell_hash in data["retired_cells"]:
            raise ValueError("manifest seed reuses a retired manifest or cell set; choose a fresh seed")
        issued = data["issued_cells"]
        if cell_hash in issued.values() and issued.get(manifest_hash) != cell_hash:
            raise ValueError("cell set was already issued for another challenger; choose a fresh seed")
        issued[manifest_hash] = cell_hash
        _save_registry(path, data)
    return SealedManifest(
        manifest_hash=manifest_hash,
        challenger_hash=challenger_hash,
        contract_hash=contract_hash,
        cells=tuple(cells),
    )


def retire_manifest(manifest_hash: str, *, registry_path: str | Path | None = None) -> None:
    """Mark a revealed suite as spent; it may never promote again."""

    if not isinstance(manifest_hash, str) or not manifest_hash:
        raise ValueError("manifest_hash must be a non-empty string")
    with _registry(registry_path) as (path, data):
        cell_hash = data["issued_cells"].get(manifest_hash)
        if cell_hash is None:
            raise ValueError("cannot retire an unknown manifest")
        data["retired_manifests"] = sorted(set(data["retired_manifests"]) | {manifest_hash})
        data["retired_cells"] = sorted(set(data["retired_cells"]) | {cell_hash})
        _save_registry(path, data)


def is_retired(manifest_hash: str, *, registry_path: str | Path | None = None) -> bool:
    with _registry(registry_path) as (_, data):
        return manifest_hash in data["retired_manifests"]


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
