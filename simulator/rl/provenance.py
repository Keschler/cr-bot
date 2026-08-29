"""Small, dependency-free runtime provenance helpers for RL artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any


def code_revision() -> dict[str, Any]:
    """Return the current Git revision and whether tracked work is dirty.

    Artifact writers call this only at report/checkpoint creation time.  Git
    is optional at runtime: a report remains serializable when the package is
    copied without repository metadata, but it explicitly records that the
    revision was unavailable instead of inventing an identity.
    """

    repository = Path(__file__).resolve().parents[1]
    try:
        revision_result = subprocess.run(
            ("git", "rev-parse", "--verify", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        revision = revision_result.stdout.strip()
        if not revision:
            raise RuntimeError("git returned an empty revision")
        status_result = subprocess.run(
            # Generated datasets and local captures are intentionally ignored;
            # this flag answers whether tracked source changed.
            ("git", "status", "--porcelain", "--untracked-files=no"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return {
            "commit": None,
            "worktree_dirty": None,
            "source": "unavailable",
        }
    return {
        "commit": revision,
        "worktree_dirty": bool(status_result.stdout.strip()),
        "source": "git",
    }


__all__ = ["code_revision"]
