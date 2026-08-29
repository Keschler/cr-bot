from __future__ import annotations


def test_code_revision_is_explicit_when_git_is_available() -> None:
    from rl.provenance import code_revision

    revision = code_revision()

    assert revision["source"] in {"git", "unavailable"}
    if revision["source"] == "git":
        assert isinstance(revision["commit"], str)
        assert len(revision["commit"]) >= 7
        assert type(revision["worktree_dirty"]) is bool
    else:
        assert revision["commit"] is None
        assert revision["worktree_dirty"] is None
