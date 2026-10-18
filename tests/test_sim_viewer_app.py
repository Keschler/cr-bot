"""Standalone HTTP contracts, including bounded artifact access."""

from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from sim_viewer.app import create_app
from sim_viewer.library import ViewerPaths


class FakeMatches:
    def __init__(self):
        self.config = None
        self.closed = False

    def start_replay(self, path):
        return {"id": "replay", "state": "running", "path": str(path)}

    def start_exhibition(self, config):
        self.config = config
        return {"id": "exhibition", "state": "running"}

    def status(self, match_id):
        if match_id == "missing":
            raise KeyError(match_id)
        return {"id": match_id, "state": "complete"}

    def frames(self, match_id, *, after, limit):
        return [{"index": 0, "tick": 10, "time_s": .5}] if after < 0 else []

    def stop(self, match_id):
        return {"id": match_id, "state": "stopped"}

    def close(self):
        self.closed = True


@pytest.fixture
def viewer(tmp_path):
    root = tmp_path / "v4"
    root.mkdir()
    manager = FakeMatches()
    with TestClient(create_app(paths=ViewerPaths((root,), root / "viewer"), manager=manager)) as client:
        yield client, manager, root
    assert manager.closed


def test_application_is_separate_and_serves_modules_and_geometry(viewer):
    client, _, _ = viewer
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/health").json()["application"] == "V4 Arena Lab"
    grid = client.get("/api/geometry").json()
    assert (grid["width"], grid["height"]) == (18, 32)
    assert grid["river"] == {"y_min": 15, "y_max": 17}
    assert len(grid["towers"]) == 6
    assert client.get("/api/video/start").status_code == 404


def test_replay_path_and_frame_query_validation(viewer, tmp_path):
    client, _, root = viewer
    path = root / "test_match.json"
    path.write_text("{}")
    assert client.post("/api/replays", json={"path": str(path)}).status_code == 202
    assert client.post("/api/replays", json={"path": str(tmp_path / "outside.json")}).status_code == 422
    assert client.get("/api/matches/missing").status_code == 404
    assert client.get("/api/matches/test/frames?limit=201").status_code == 422
    result = client.get("/api/matches/test/frames?after=-1").json()
    assert result["frames"][0]["time_s"] == .5


def test_exhibition_requires_opponent_checkpoint_and_resolves_paths(viewer):
    client, manager, root = viewer
    actor = root / "actor.pt"
    actor.touch()
    request = {"checkpoint": str(actor), "opponent": "checkpoint"}
    assert client.post("/api/exhibitions", json=request).status_code == 422
    request["opponent_checkpoint"] = str(actor)
    assert client.post("/api/exhibitions", json=request).status_code == 202
    assert manager.config["checkpoint"] == str(actor.resolve())
    assert client.post("/api/exhibitions", json={**request, "max_decisions": 1500}).status_code == 422


def test_server_import_does_not_load_torch():
    code = "import sys; from sim_viewer.app import app; [r.endpoint() for r in app.routes if getattr(r, 'path', '') in ('/api/catalog', '/api/geometry')]; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
