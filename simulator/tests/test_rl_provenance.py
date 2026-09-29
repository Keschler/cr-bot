from __future__ import annotations


def test_code_revision_is_explicit_when_git_is_available() -> None:
    from simulator.rl.foundation.provenance import code_revision, revision_changed

    revision = code_revision()

    assert revision["source"] in {"git", "unavailable"}
    if revision["source"] == "git":
        assert isinstance(revision["commit"], str)
        assert len(revision["commit"]) >= 7
        assert type(revision["worktree_dirty"]) is bool
        assert isinstance(revision["tracked_worktree_hash"], str)
        assert revision["tracked_worktree_hash"].startswith("sha256:")
    else:
        assert revision["commit"] is None
        assert revision["worktree_dirty"] is None

    stable = {"source": "git", "commit": "abc1234", "worktree_dirty": True}
    assert revision_changed(stable, dict(stable)) is False
    assert revision_changed(
        stable,
        {"source": "git", "commit": "def5678", "worktree_dirty": True},
    ) is True
    assert revision_changed(
        stable,
        {"source": "unavailable", "commit": None, "worktree_dirty": None},
    ) is False
    assert revision_changed(
        stable,
        {
            **stable,
            "tracked_worktree_hash": "sha256:" + "0" * 64,
        },
    ) is True
    hashed = {
        **stable,
        "tracked_worktree_hash": "sha256:" + "1" * 64,
    }
    assert revision_changed(hashed, dict(hashed)) is False
    assert revision_changed(
        hashed,
        {**hashed, "tracked_worktree_hash": "sha256:" + "2" * 64},
    ) is True


def test_unavailable_provenance_is_unknown_not_stable() -> None:
    """An identity Git cannot supply must not pass the drift guard.

    ``revision_changed`` answers "did the code change", and a missing identity
    has no answer, so it stays ``False`` for callers that only need that
    boolean. Callers that key an exit code on the guard use
    ``revision_guard_status``, which reports ``unknown`` instead of letting an
    unpackaged copy exit clean.
    """

    from simulator.rl.foundation.provenance import revision_guard_status

    stable = {"source": "git", "commit": "abc1234", "worktree_dirty": True}
    unavailable = {"source": "unavailable", "commit": None, "worktree_dirty": None}
    drifted = {"source": "git", "commit": "def5678", "worktree_dirty": True}

    assert revision_guard_status(stable, stable) == "stable"
    assert revision_guard_status(stable, drifted) == "drifted"
    assert revision_guard_status(stable, unavailable) == "unknown"
    assert revision_guard_status(unavailable, stable) == "unknown"
    assert revision_guard_status(unavailable, unavailable) == "unknown"
