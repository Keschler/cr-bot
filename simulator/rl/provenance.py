"""Small, dependency-free runtime provenance helpers for RL artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping


def _tracked_worktree_hash(repository: Path) -> str:
    """Hash tracked content changes relative to ``HEAD``.

    ``HEAD`` alone is insufficient in a shared checkout: a simulator fix can
    be present in the worktree before its commit is created.  ``git diff
    HEAD`` includes both staged and unstaged tracked changes while excluding
    untracked captures, checkpoints, and other local artifacts.
    """

    diff_result = subprocess.run(
        ("git", "diff", "--no-ext-diff", "--binary", "HEAD", "--"),
        cwd=repository,
        check=True,
        capture_output=True,
        timeout=5.0,
    )
    return "sha256:" + hashlib.sha256(diff_result.stdout).hexdigest()


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
        tracked_hash = _tracked_worktree_hash(repository)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return {
            "commit": None,
            "worktree_dirty": None,
            "source": "unavailable",
        }
    return {
        "commit": revision,
        "worktree_dirty": bool(status_result.stdout.strip()),
        "tracked_worktree_hash": tracked_hash,
        "source": "git",
    }


def revision_changed(
    start: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    """Return whether two Git provenance snapshots identify different code.

    Compare both committed ``HEAD`` and tracked worktree content.  Older
    artifacts may not contain a worktree hash; comparing two such snapshots
    retains compatibility, while mixing an old snapshot with a hashed one
    fails closed.  If Git is unavailable in either snapshot, there is no
    reliable identity to compare and this helper returns ``False``; the
    snapshots still preserve that limitation in the artifact.
    """

    start_commit = start.get("commit")
    current_commit = current.get("commit")
    if not isinstance(start_commit, str) or not isinstance(current_commit, str):
        return False
    if start_commit != current_commit:
        return True
    start_hash = start.get("tracked_worktree_hash")
    current_hash = current.get("tracked_worktree_hash")
    if isinstance(start_hash, str) and isinstance(current_hash, str):
        return start_hash != current_hash
    # A new hashed snapshot compared with an old un-hashed artifact cannot
    # prove that the same tracked source produced both artifacts.
    return (isinstance(start_hash, str)) != (isinstance(current_hash, str))


__all__ = ["code_revision", "revision_changed"]
