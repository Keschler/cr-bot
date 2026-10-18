"""Independent FastAPI application for local V4 simulator matches."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .library import ArtifactLibrary, REPO_ROOT, ViewerPaths


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)


class ExhibitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: str = Field(min_length=1)
    opponent: Literal["scripted", "checkpoint"] = "scripted"
    opponent_checkpoint: str | None = None
    archetype: str = "deterministic-cycle"
    learner_deck: list[str] | None = Field(default=None, min_length=8, max_length=8)
    opponent_deck: list[str] | None = Field(default=None, min_length=8, max_length=8)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    max_decisions: int = Field(default=1200, ge=1, le=1440)


def create_app(*, paths: ViewerPaths | None = None, manager: Any = None) -> FastAPI:
    paths = paths or ViewerPaths.from_environment()
    library = ArtifactLibrary(paths)
    if manager is None:
        from .matches import MatchManager

        manager = MatchManager(paths.storage)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        manager.close()

    app = FastAPI(title="V4 Arena Lab", lifespan=lifespan)
    app.state.library = library
    app.state.matches = manager

    @app.exception_handler(ValueError)
    async def invalid(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(FileNotFoundError)
    async def missing_file(request: Request, error: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Artifact not found: {error}"})

    @app.exception_handler(KeyError)
    async def missing_id(request: Request, error: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "Match or run not found"})

    from .matches import MatchBusyError

    @app.exception_handler(MatchBusyError)
    async def busy(request: Request, error: MatchBusyError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    static = Path(__file__).with_name("static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    if static.is_dir():
        app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "application": "V4 Arena Lab", "roots": [str(root) for root in paths.roots]}

    @app.get("/api/library")
    def artifacts(query: str = "", limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        return library.listing(query, limit)

    @app.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        from simulator.roster import PLAYER_DECK
        from simulator.ruleset import load_fixed_ruleset

        ruleset = load_fixed_ruleset()
        # Discovery needs JSON metadata only; importing the RL package would
        # initialize Torch just to fill a dropdown.
        raw_decks = json.loads((REPO_ROOT / "simulator/rl/v4/real_decks.json").read_text(encoding="utf-8"))
        decks = [
            {"id": deck["id"], "name": deck["id"], "cards": deck["cards"], "archetype": deck["archetype"], "split": deck["split"]}
            for deck in raw_decks["decks"]
        ]
        return {
            "checkpoints": library.checkpoints(),
            "archetypes": ["deterministic-cycle", "aggressive-pressure", "defensive-cycle", "beatdown", "air-beatdown", "siege-bait", "random-legal"],
            "decks": decks,
            "cards": sorted(ruleset.interaction_set),
            "learner_deck": list(PLAYER_DECK),
            "roots": [str(root) for root in paths.roots],
        }

    @app.get("/api/geometry")
    def geometry() -> dict[str, Any]:
        from simulator.geometry import GRID_COLS, GRID_ROWS, RIVER_ROWS, TOWER_SITES, is_ground_cell
        from simulator.ruleset import load_fixed_ruleset

        ruleset = load_fixed_ruleset()
        return {
            "width": GRID_COLS, "height": GRID_ROWS,
            "river": {"y_min": min(RIVER_ROWS), "y_max": max(RIVER_ROWS) + 1},
            "bridges": [
                {"x_min": col, "x_max": col + 1, "y_min": min(RIVER_ROWS), "y_max": max(RIVER_ROWS) + 1}
                for col in range(GRID_COLS) if is_ground_cell((col, min(RIVER_ROWS)))
            ],
            "towers": [{"owner": site.owner, "role": site.role, "x": site.x_mtile / 1000, "y": site.y_mtile / 1000} for site in TOWER_SITES],
            "tick_s": ruleset.tick_us / 1_000_000,
        }

    @app.get("/api/card-icon")
    def card_icon(name: str = Query(min_length=1, max_length=100)) -> FileResponse:
        # Share the project's existing naming adapter and local assets.
        from frontend.services.card_art import _card_art_candidates, _card_art_dir

        art = _card_art_dir()
        if art is not None:
            for candidate in _card_art_candidates(name):
                path = art / f"{candidate}.png"
                if path.is_file():
                    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})
        raise HTTPException(404, detail="Card art is unavailable")

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        return library.run(run_id)

    @app.post("/api/replays", status_code=202)
    def replay(body: ReplayRequest) -> dict[str, Any]:
        path = paths.resolve(body.path, suffixes=(".json", ".jsonl"))
        return manager.start_replay(path)

    @app.post("/api/exhibitions", status_code=202)
    def exhibition(body: ExhibitionRequest) -> dict[str, Any]:
        config = body.model_dump()
        config["checkpoint"] = str(paths.resolve(body.checkpoint, suffixes=(".pt",)))
        if body.opponent == "checkpoint":
            if not body.opponent_checkpoint:
                raise ValueError("Choose a checkpoint for the opponent")
            config["opponent_checkpoint"] = str(paths.resolve(body.opponent_checkpoint, suffixes=(".pt",)))
        return manager.start_exhibition(config)

    @app.get("/api/matches/{match_id}")
    def status(match_id: str) -> dict[str, Any]:
        return manager.status(match_id)

    @app.get("/api/matches/{match_id}/frames")
    def frames(match_id: str, after: int = Query(default=-1, ge=-1), limit: int = Query(default=100, ge=1, le=200)) -> dict[str, Any]:
        return {"frames": manager.frames(match_id, after=after, limit=limit), "status": manager.status(match_id)}

    @app.post("/api/matches/{match_id}/stop")
    def stop(match_id: str) -> dict[str, Any]:
        return manager.stop(match_id)

    @app.get("/api/matches/{match_id}/stream")
    async def stream(match_id: str, request: Request, after: int = Query(default=-1, ge=-1)) -> StreamingResponse:
        import asyncio

        manager.status(match_id)

        async def events():
            cursor = after
            while not await request.is_disconnected():
                batch = manager.frames(match_id, after=cursor, limit=100)
                for frame in batch:
                    cursor = frame["index"]
                    yield f"event: frame\ndata: {json.dumps(frame, allow_nan=False)}\n\n"
                current = manager.status(match_id)
                yield f"event: status\ndata: {json.dumps(current, allow_nan=False)}\n\n"
                if current["state"] != "running" and not batch:
                    break
                await asyncio.sleep(0.25)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()
