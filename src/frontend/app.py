from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.assets import router as assets_router
from .api.corrections import router as corrections_router
from .api.frames import router as frames_router
from .api.live import router as live_router
from .api.roi import router as roi_router
from .api.stream import router as stream_router
from .api.system import router as system_router
from .api.video import router as video_router
from .services.paths import STATIC_DIR


def create_app() -> FastAPI:
    app = FastAPI(title="cr-bot frontend")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system_router)
    app.include_router(video_router)
    app.include_router(live_router)
    app.include_router(frames_router)
    app.include_router(roi_router)
    app.include_router(assets_router)
    app.include_router(corrections_router)
    app.include_router(stream_router)
    # Serve the static UI after API routes so `/api/*` keeps precedence.
    # Only explicit static paths are mounted: mounting "/" (even with
    # html=False) would swallow unknown /api/* paths into StaticFiles 404s
    # instead of FastAPI's JSON 404s. Unmatched /api/* therefore stays JSON.
    if STATIC_DIR.is_dir():
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles

        index_file = STATIC_DIR / "index.html"

        @app.get("/", include_in_schema=False)
        def _serve_index() -> FileResponse:
            return FileResponse(str(index_file))

        styles_css = STATIC_DIR / "styles.css"
        if styles_css.is_file():

            @app.get("/styles.css", include_in_schema=False)
            def _serve_styles_css() -> FileResponse:
                return FileResponse(str(styles_css))

        for sub in ("js", "styles"):
            subdir = STATIC_DIR / sub
            if subdir.is_dir():
                app.mount("/" + sub, StaticFiles(directory=str(subdir)), name="static-" + sub)
    else:

        @app.get("/")
        def _static_placeholder() -> JSONResponse:
            return JSONResponse(
                {"ok": True, "message": "frontend static bundle not present"}
            )

    return app


app = create_app()

__all__ = ["app", "create_app"]
