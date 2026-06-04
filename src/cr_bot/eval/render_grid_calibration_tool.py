from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
KATACR_ROOT = ROOT / "vendor/external/KataCR"
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from cr_bot.app.pipeline import normalize_frame
from cr_bot.eval.visualize_cells import arena_px_for
from cr_bot.features.action_space import ACTION_GRID


DEFAULT_VIDEO = (
    ROOT
    / "dataset_generation/data/video_clips/downloaded_videos/"
    / "HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].mp4"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/eval/grid_calibration_tool"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a browser tool for manually calibrating the action grid.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--image", type=Path, help="Use a still image instead of reading --frame-index from --video.")
    parser.add_argument("--frame-index", type=int, default=1387)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.image is not None:
        frame = read_image(args.image)
        source_label = str(args.image)
        image_stem = args.image.stem
    else:
        frame = read_frame(args.video, args.frame_index)
        source_label = f"{args.video} frame {args.frame_index}"
        image_stem = f"frame_{args.frame_index}"
    image = normalize_frame(frame)
    arena_px = arena_px_for(image)
    image_name = f"{image_stem}.png"
    cv2.imwrite(str(args.output_dir / image_name), image)

    initial_grid = current_grid_px(arena_px)
    html = build_html(
        image_name=image_name,
        frame_index=args.frame_index,
        source_label=source_label,
        image_width=image.shape[1],
        image_height=image.shape[0],
        arena_px=arena_px,
        initial_grid=initial_grid,
    )
    (args.output_dir / "index.html").write_text(html)
    print(args.output_dir / "index.html")
    print(f"image_size=({image.shape[1]}, {image.shape[0]})")
    print(f"arena_px={arena_px}")
    print(f"current_grid_px={initial_grid}")


def read_frame(video: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"could not open video: {video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"could not read frame {frame_index} from {video}")
        return frame
    finally:
        cap.release()


def read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return image


def current_grid_px(arena_px: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = arena_px
    return (
        round(ax + ACTION_GRID.x0 * aw),
        round(ay + ACTION_GRID.y0 * ah),
        round(ax + ACTION_GRID.x1 * aw),
        round(ay + ACTION_GRID.y1 * ah),
    )


def build_html(
    *,
    image_name: str,
    frame_index: int,
    source_label: str,
    image_width: int,
    image_height: int,
    arena_px: tuple[int, int, int, int],
    initial_grid: tuple[int, int, int, int],
) -> str:
    gx0, gy0, gx1, gy1 = initial_grid
    ax, ay, aw, ah = arena_px
    block_w = (gx1 - gx0) / ACTION_GRID.cols
    block_h = (gy1 - gy0) / ACTION_GRID.rows
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Action Grid Calibration</title>
  <style>
    :root {{ color-scheme: dark; }}
    html, body {{ height: 100%; overflow: hidden; }}
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #111; color: #f4f4f4; }}
    .app {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; height: 100vh; overflow: hidden; }}
    .stage {{ overflow: auto; padding: 14px; background: #181818; }}
    .canvas-wrap {{ min-width: max-content; display: flex; align-items: flex-start; justify-content: flex-start; }}
    canvas {{
      display: block;
      background: #000;
      width: auto;
      height: auto;
      box-shadow: 0 0 0 1px #333;
    }}
    aside {{ border-left: 1px solid #333; padding: 14px; background: #202020; overflow: hidden; }}
    h1 {{ font-size: 18px; margin: 0 0 12px; }}
    fieldset {{ border: 1px solid #3a3a3a; margin: 0 0 12px; padding: 10px; }}
    legend {{ padding: 0 5px; color: #ddd; }}
    label {{ display: grid; grid-template-columns: 1fr 100px; align-items: center; gap: 8px; margin: 7px 0; font-size: 13px; }}
    input, select, button, textarea {{ font: inherit; }}
    input, select, textarea {{ color: #f4f4f4; background: #111; border: 1px solid #555; border-radius: 4px; padding: 5px; }}
    button {{ color: #f4f4f4; background: #333; border: 1px solid #666; border-radius: 4px; padding: 7px 9px; cursor: pointer; }}
    button:hover {{ background: #3d3d3d; }}
    .buttons {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
    textarea {{ width: 100%; height: 170px; box-sizing: border-box; white-space: pre; }}
    .hint {{ font-size: 12px; color: #bbb; line-height: 1.35; margin: 7px 0 0; }}
  </style>
</head>
<body>
  <div class="app">
    <main class="stage">
      <div class="canvas-wrap">
        <canvas id="canvas" width="{image_width}" height="{image_height}"></canvas>
      </div>
    </main>
    <aside>
      <h1>Action Grid Calibration</h1>
      <p class="hint">Source: {source_label}</p>
      <fieldset>
        <legend>Draw Mode</legend>
        <label>Mode
          <select id="mode">
            <option value="grid">Drag full grid area</option>
            <option value="block">Drag one block</option>
            <option value="pan">Inspect only</option>
          </select>
        </label>
        <p class="hint">Drag on the image. Full grid area is cyan. Single block sample is orange.</p>
      </fieldset>
      <fieldset>
        <legend>Zoom</legend>
        <label>zoom %
          <input id="zoom" type="number" step="5" min="10" max="400" value="50">
        </label>
        <div class="buttons">
          <button id="zoomOut" type="button">Zoom Out</button>
          <button id="zoomIn" type="button">Zoom In</button>
          <button id="zoomFit" type="button">Fit</button>
          <button id="zoom100" type="button">100%</button>
        </div>
        <p class="hint">Mouse wheel over the image also zooms. Coordinates stay in original image pixels.</p>
      </fieldset>
      <fieldset>
        <legend>Grid Area</legend>
        <label>x <input id="gridX" type="number" step="0.1" value="{gx0}"></label>
        <label>y <input id="gridY" type="number" step="0.1" value="{gy0}"></label>
        <label>width <input id="gridW" type="number" step="0.1" value="{gx1 - gx0}"></label>
        <label>height <input id="gridH" type="number" step="0.1" value="{gy1 - gy0}"></label>
      </fieldset>
      <fieldset>
        <legend>Block Size</legend>
        <label>columns <input id="cols" type="number" step="1" value="{ACTION_GRID.cols}"></label>
        <label>rows <input id="rows" type="number" step="1" value="{ACTION_GRID.rows}"></label>
        <label>block width <input id="blockW" type="number" step="0.1" value="{block_w:.2f}"></label>
        <label>block height <input id="blockH" type="number" step="0.1" value="{block_h:.2f}"></label>
        <div class="buttons">
          <button id="fitFromBlock" type="button">Fit Area From Block</button>
          <button id="fitBlockFromArea" type="button">Fit Block From Area</button>
        </div>
      </fieldset>
      <fieldset>
        <legend>Context</legend>
        <label>arena x <input id="arenaX" type="number" step="0.1" value="{ax}"></label>
        <label>arena y <input id="arenaY" type="number" step="0.1" value="{ay}"></label>
        <label>arena width <input id="arenaW" type="number" step="0.1" value="{aw}"></label>
        <label>arena height <input id="arenaH" type="number" step="0.1" value="{ah}"></label>
      </fieldset>
      <fieldset>
        <legend>Output</legend>
        <textarea id="output" readonly></textarea>
        <div class="buttons">
          <button id="copy" type="button">Copy Output</button>
          <button id="reset" type="button">Reset Current Grid</button>
        </div>
      </fieldset>
    </aside>
  </div>
  <script>
    const image = new Image();
    image.src = {image_name!r};
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const stage = document.querySelector(".stage");
    const zoomInput = document.getElementById("zoom");
    const ids = ["gridX", "gridY", "gridW", "gridH", "cols", "rows", "blockW", "blockH", "arenaX", "arenaY", "arenaW", "arenaH"];
    const inputs = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
    const mode = document.getElementById("mode");
    const output = document.getElementById("output");
    const initial = {{ gridX: {gx0}, gridY: {gy0}, gridW: {gx1 - gx0}, gridH: {gy1 - gy0} }};
    let blockRect = null;
    let drag = null;

    function num(id) {{ return Number(inputs[id].value); }}
    function setNum(id, value) {{ inputs[id].value = Number(value).toFixed(2).replace(/\\.00$/, ""); }}
    function sortedRect(a, b) {{
      const x0 = Math.min(a.x, b.x);
      const y0 = Math.min(a.y, b.y);
      const x1 = Math.max(a.x, b.x);
      const y1 = Math.max(a.y, b.y);
      return {{ x: x0, y: y0, w: x1 - x0, h: y1 - y0 }};
    }}
    function mousePoint(event) {{
      const rect = canvas.getBoundingClientRect();
      return {{
        x: (event.clientX - rect.left) * canvas.width / rect.width,
        y: (event.clientY - rect.top) * canvas.height / rect.height,
      }};
    }}
    function currentScale() {{
      return Number(zoomInput.value) / 100;
    }}
    function applyZoom(percent, anchor = null) {{
      const oldScale = currentScale();
      const clamped = Math.min(400, Math.max(10, percent));
      const newScale = clamped / 100;
      let imageX = null;
      let imageY = null;
      let mouseStageX = null;
      let mouseStageY = null;
      if (anchor) {{
        const stageRect = stage.getBoundingClientRect();
        mouseStageX = anchor.clientX - stageRect.left;
        mouseStageY = anchor.clientY - stageRect.top;
        imageX = (stage.scrollLeft + mouseStageX - canvas.offsetLeft) / oldScale;
        imageY = (stage.scrollTop + mouseStageY - canvas.offsetTop) / oldScale;
      }}
      zoomInput.value = Math.round(clamped);
      canvas.style.width = `${{canvas.width * newScale}}px`;
      canvas.style.height = `${{canvas.height * newScale}}px`;
      requestAnimationFrame(() => {{
        if (anchor) {{
          stage.scrollLeft = imageX * newScale + canvas.offsetLeft - mouseStageX;
          stage.scrollTop = imageY * newScale + canvas.offsetTop - mouseStageY;
        }}
      }});
    }}
    function fitZoom() {{
      const pad = 34;
      const availableW = Math.max(100, stage.clientWidth - pad);
      const availableH = Math.max(100, stage.clientHeight - pad);
      const scale = Math.min(availableW / canvas.width, availableH / canvas.height);
      applyZoom(scale * 100);
    }}
    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0);
      drawGrid();
      if (blockRect) drawBlock();
      updateOutput();
    }}
    function drawGrid() {{
      const x = num("gridX"), y = num("gridY"), w = num("gridW"), h = num("gridH");
      const cols = Math.max(1, Math.round(num("cols")));
      const rows = Math.max(1, Math.round(num("rows")));
      ctx.save();
      ctx.strokeStyle = "rgba(0,255,255,0.95)";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.lineWidth = 1;
      for (let col = 0; col <= cols; col++) {{
        const gx = x + col / cols * w;
        ctx.beginPath(); ctx.moveTo(gx, y); ctx.lineTo(gx, y + h); ctx.stroke();
      }}
      for (let row = 0; row <= rows; row++) {{
        const gy = y + row / rows * h;
        ctx.beginPath(); ctx.moveTo(x, gy); ctx.lineTo(x + w, gy); ctx.stroke();
      }}
      ctx.restore();
    }}
    function drawBlock() {{
      ctx.save();
      ctx.strokeStyle = "rgba(255,165,0,0.98)";
      ctx.lineWidth = 4;
      ctx.strokeRect(blockRect.x, blockRect.y, blockRect.w, blockRect.h);
      ctx.restore();
    }}
    function updateOutput() {{
      const gx = num("gridX"), gy = num("gridY"), gw = num("gridW"), gh = num("gridH");
      const ax = num("arenaX"), ay = num("arenaY"), aw = num("arenaW"), ah = num("arenaH");
      const cols = Math.round(num("cols")), rows = Math.round(num("rows"));
      const bx = num("blockW"), by = num("blockH");
      const data = {{
        frame_index: {frame_index},
        source: {source_label!r},
        image_px: {{ width: canvas.width, height: canvas.height }},
        arena_px: {{ x: ax, y: ay, width: aw, height: ah }},
        grid_px: {{ x0: gx, y0: gy, x1: gx + gw, y1: gy + gh, width: gw, height: gh }},
        block_px: {{ width: bx, height: by }},
        grid_size: {{ cols, rows }},
        action_grid_normalized_to_arena: {{
          x0: (gx - ax) / aw,
          y0: (gy - ay) / ah,
          x1: (gx + gw - ax) / aw,
          y1: (gy + gh - ay) / ah,
        }},
        katacr_grid_xyxy_for_568x896: {{
          x0: (gx - ax) / aw * 568,
          y0: (gy - ay) / ah * 896,
          x1: (gx + gw - ax) / aw * 568,
          y1: (gy + gh - ay) / ah * 896,
        }},
      }};
      output.value = JSON.stringify(data, null, 2);
    }}
    canvas.addEventListener("mousedown", event => {{
      if (mode.value === "pan") return;
      drag = {{ start: mousePoint(event), current: mousePoint(event), mode: mode.value }};
    }});
    canvas.addEventListener("wheel", event => {{
      event.preventDefault();
      const direction = event.deltaY > 0 ? -1 : 1;
      const next = Number(zoomInput.value) * (direction > 0 ? 1.12 : 1 / 1.12);
      applyZoom(next, event);
    }}, {{ passive: false }});
    canvas.addEventListener("mousemove", event => {{
      if (!drag) return;
      drag.current = mousePoint(event);
      const rect = sortedRect(drag.start, drag.current);
      if (drag.mode === "grid") {{
        setNum("gridX", rect.x); setNum("gridY", rect.y); setNum("gridW", rect.w); setNum("gridH", rect.h);
      }} else {{
        blockRect = rect;
        setNum("blockW", rect.w); setNum("blockH", rect.h);
      }}
      draw();
    }});
    window.addEventListener("mouseup", () => {{ drag = null; }});
    function syncGridFromBlock() {{
      setNum("gridW", num("blockW") * Math.max(1, Math.round(num("cols"))));
      setNum("gridH", num("blockH") * Math.max(1, Math.round(num("rows"))));
    }}
    function handleInput(event) {{
      if (event.target.id === "blockW" || event.target.id === "blockH") {{
        syncGridFromBlock();
      }}
      draw();
    }}
    for (const input of Object.values(inputs)) input.addEventListener("input", handleInput);
    zoomInput.addEventListener("input", () => applyZoom(Number(zoomInput.value)));
    document.getElementById("zoomOut").addEventListener("click", () => applyZoom(Number(zoomInput.value) / 1.2));
    document.getElementById("zoomIn").addEventListener("click", () => applyZoom(Number(zoomInput.value) * 1.2));
    document.getElementById("zoomFit").addEventListener("click", fitZoom);
    document.getElementById("zoom100").addEventListener("click", () => applyZoom(100));
    document.getElementById("fitFromBlock").addEventListener("click", () => {{
      syncGridFromBlock();
      draw();
    }});
    document.getElementById("fitBlockFromArea").addEventListener("click", () => {{
      setNum("blockW", num("gridW") / Math.max(1, Math.round(num("cols"))));
      setNum("blockH", num("gridH") / Math.max(1, Math.round(num("rows"))));
      draw();
    }});
    document.getElementById("copy").addEventListener("click", async () => {{
      await navigator.clipboard.writeText(output.value);
    }});
    document.getElementById("reset").addEventListener("click", () => {{
      setNum("gridX", initial.gridX); setNum("gridY", initial.gridY); setNum("gridW", initial.gridW); setNum("gridH", initial.gridH);
      blockRect = null;
      draw();
    }});
    image.onload = () => {{ draw(); fitZoom(); }};
    window.addEventListener("resize", () => {{
      if (Number(zoomInput.value) < 100) fitZoom();
    }});
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
