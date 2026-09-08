"""Frozen V4 experiment contracts (paper Stage 0).

A V4 experiment generation is identified by the exact simulator/ruleset
revision, the exact observation/action contract, and the code revision that
produced a checkpoint.  Every promotion artifact records these hashes so a
"better" claim can never silently mix revisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Final


ACTION_SEMANTICS_VERSION: Final = "v4-hierarchical-1"
"""WAIT/PLAY -> (WAIT duration | card slot -> card-conditioned placement)."""

LEGAlITY_SEMANTICS_VERSION: Final = "center-cell-impossible-only-1"
"""Only impossible actions are masked; strategy stays learned."""

POLICY_OBSERVATION_VERSION: Final = "public-v3-1"
"""Public-only V3 observation contract consumed by the V4 actor."""


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    """Immutable identity of one V4 experiment generation."""

    code_revision: str
    tracked_worktree_state: str
    engine_version: str
    ruleset_id: str
    ruleset_hash: str
    observation_schema_version: str
    observation_contract_hash: str
    action_semantics_version: str = ACTION_SEMANTICS_VERSION
    legality_semantics_version: str = LEGAlITY_SEMANTICS_VERSION
    contract_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "code_revision",
            "tracked_worktree_state",
            "engine_version",
            "ruleset_id",
            "ruleset_hash",
            "observation_schema_version",
            "observation_contract_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.contract_hash, str):
            raise TypeError("contract_hash must be a string")
        manifest = contract_manifest(self)
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        expected = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        if self.contract_hash and self.contract_hash != expected:
            raise ValueError("contract_hash does not match the contract manifest")
        if not self.contract_hash:
            object.__setattr__(self, "contract_hash", expected)


def contract_manifest(contract: ExperimentContract) -> dict[str, str]:
    """Return the canonical JSON-compatible manifest (hash excludes itself)."""

    return {
        "code_revision": contract.code_revision,
        "tracked_worktree_state": contract.tracked_worktree_state,
        "engine_version": contract.engine_version,
        "ruleset_id": contract.ruleset_id,
        "ruleset_hash": contract.ruleset_hash,
        "observation_schema_version": contract.observation_schema_version,
        "observation_contract_hash": contract.observation_contract_hash,
        "action_semantics_version": contract.action_semantics_version,
        "legality_semantics_version": contract.legality_semantics_version,
    }


def seal_current_contracts(
    *,
    code_revision: str,
    tracked_worktree_state: str = "clean",
    ruleset_id: str = "v1",
) -> ExperimentContract:
    """Freeze the live simulator/ruleset/observation identity for one generation.

    Heavy simulator imports stay inside this function so importing the
    evaluation harness never loads the engine, ruleset JSON, or torch.
    """

    try:
        from ..engine._base import ENGINE_VERSION
        from ..observation_v3 import (
            OBSERVATION_V3_CONTRACT_HASH,
            OBSERVATION_V3_SCHEMA_VERSION,
        )
        from ..ruleset import load_ruleset
    except ImportError:  # pragma: no cover - top-level ``evaluation`` imports
        from simulator.engine._base import ENGINE_VERSION
        from simulator.observation_v3 import (
            OBSERVATION_V3_CONTRACT_HASH,
            OBSERVATION_V3_SCHEMA_VERSION,
        )
        from simulator.ruleset import load_ruleset

    if not isinstance(code_revision, str) or not code_revision:
        raise ValueError("code_revision must be a non-empty string")
    ruleset = load_ruleset(ruleset_id)
    return ExperimentContract(
        code_revision=code_revision,
        tracked_worktree_state=tracked_worktree_state,
        engine_version=ENGINE_VERSION,
        ruleset_id=ruleset.ruleset_id,
        ruleset_hash=ruleset.content_hash,
        observation_schema_version=OBSERVATION_V3_SCHEMA_VERSION,
        observation_contract_hash=OBSERVATION_V3_CONTRACT_HASH,
    )


__all__ = [
    "ACTION_SEMANTICS_VERSION",
    "LEGAlITY_SEMANTICS_VERSION",
    "POLICY_OBSERVATION_VERSION",
    "ExperimentContract",
    "contract_manifest",
    "seal_current_contracts",
]
