from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response

from ..services.card_art import _card_art_candidates, _card_art_dir

router = APIRouter()


@router.get("/api/card-icon")
def api_card_icon(name: str = Query(default="")) -> Response:
    options = _card_art_candidates(name)
    if not options:
        raise HTTPException(status_code=400, detail="card name is required")
    art_dir = _card_art_dir()
    if art_dir is None:
        raise HTTPException(status_code=404, detail="card art is not available")
    for option in options:
        path = art_dir / f"{option}.png"
        if path.is_file():
            return Response(
                content=path.read_bytes(),
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=86400"},
            )
    raise HTTPException(
        status_code=404, detail=f"no card art for {options[0]!r}"
    )


@router.get("/api/grid")
def api_grid() -> dict[str, Any]:
    """Serve the authoritative action-grid spec (mirrors ACTION_GRID)."""
    try:
        from cr_bot.features.action_space import (
            ACTION_GRID,
            BRIDGE_COLS,
            OWN_SIDE_FIRST_ROW,
            RIVER_ROWS,
        )
        from cr_bot.domain.constants import KING_TOWER_HP, PRINCESS_TOWER_HP
    except ImportError as error:
        raise HTTPException(
            status_code=404, detail="grid spec is not available"
        ) from error
    return {
        "cols": ACTION_GRID.cols,
        "rows": ACTION_GRID.rows,
        "x0": ACTION_GRID.x0,
        "y0": ACTION_GRID.y0,
        "x1": ACTION_GRID.x1,
        "y1": ACTION_GRID.y1,
        "river_rows": list(RIVER_ROWS),
        "bridge_cols": list(BRIDGE_COLS),
        "own_side_first_row": OWN_SIDE_FIRST_ROW,
        "tower_hp": {"princess": PRINCESS_TOWER_HP, "king": KING_TOWER_HP},
    }
