from __future__ import annotations

import argparse
import csv
import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from urllib.parse import parse_qs, urlparse
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.vision.tower_hp_ocr import TowerHPOCR  # noqa: E402

DEFAULT_LABELS_CSV = ROOT / "outputs/tower_hp_ocr_crops/labels.csv"
DEFAULT_CHECKPOINT = ROOT / "outputs/models/tower_hp_crnn_candidate.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review tower HP OCR crop labels in a browser.")
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


class LabelStore:
    def __init__(self, csv_path: Path, checkpoint_path: Path, device: str | None = None) -> None:
        self.csv_path = csv_path
        self.ocr = TowerHPOCR(checkpoint_path, device=device)
        self.rows: list[dict[str, str]] = []
        self.fieldnames: list[str] = []
        self.model_ocr_cache: dict[int, str] = {}
        self.load()

    def load(self) -> None:
        with self.csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = [dict(row) for row in reader]
        for fieldname in ("readable", "label", "notes"):
            if fieldname not in self.fieldnames:
                self.fieldnames.append(fieldname)
                for row in self.rows:
                    row[fieldname] = ""

    def save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=self.csv_path.parent,
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        tmp_path.replace(self.csv_path)

    def filtered_indices(self, filter_name: str, frames: set[str] | None = None) -> list[int]:
        if filter_name == "labeled":
            indices = [idx for idx, row in enumerate(self.rows) if row.get("readable", "").strip()]
        elif filter_name == "unlabeled":
            indices = [idx for idx, row in enumerate(self.rows) if not row.get("readable", "").strip()]
        else:
            indices = list(range(len(self.rows)))

        if frames:
            indices = [
                idx for idx in indices
                if self.rows[idx].get("frame_index", "").strip() in frames
            ]
        return indices

    def model_ocr(self, idx: int) -> str:
        if idx in self.model_ocr_cache:
            return self.model_ocr_cache[idx]

        row = self.rows[idx]
        image = cv2.imread(str(resolve_path(row["image_path"])), cv2.IMREAD_COLOR)
        if image is None:
            value = ""
        else:
            tower_name = row.get("tower_name") or "enemy_support_left"
            prediction = self.ocr.predict_batch({tower_name: image}).get(tower_name)
            value = "" if prediction is None or prediction.value is None else str(prediction.value)
        self.model_ocr_cache[idx] = value
        return value

    def apply_updates(self, updates: list[dict]) -> int:
        changed = 0
        for update in updates:
            idx = int(update["idx"])
            if idx < 0 or idx >= len(self.rows):
                continue
            row = self.rows[idx]
            readable = str(update.get("readable", "")).strip().lower()
            label = str(update.get("label", "")).strip()
            notes = str(update.get("notes", "")).strip()
            if readable in {"true", "1", "yes", "y"}:
                readable = "true"
            elif readable in {"false", "0", "no", "n"}:
                readable = "false"
                label = ""
            elif label:
                readable = "true"
            else:
                readable = ""
                label = ""

            before = (row.get("readable", ""), row.get("label", ""), row.get("notes", ""))
            row["readable"] = readable
            row["label"] = label if readable == "true" else ""
            row["notes"] = notes
            after = (row.get("readable", ""), row.get("label", ""), row.get("notes", ""))
            if before != after:
                changed += 1
        if changed:
            self.save()
        return changed


def make_handler(store: LabelStore, default_page_size: int):
    class ReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.render_page(parsed)
            elif parsed.path == "/image":
                self.serve_image(parsed)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            changed = store.apply_updates(payload.get("updates", []))
            self.write_json({"changed": changed})

        def render_page(self, parsed) -> None:
            query = parse_qs(parsed.query)
            filter_name = query.get("filter", ["unlabeled"])[0]
            frames_text = query.get("frames", [""])[0]
            frames = parse_frame_filter(frames_text)
            page = max(0, int(query.get("page", ["0"])[0]))
            page_size = max(1, int(query.get("page_size", [str(default_page_size)])[0]))
            indices = store.filtered_indices(filter_name, frames=frames)
            total_pages = max(1, (len(indices) + page_size - 1) // page_size)
            page = min(page, total_pages - 1)
            visible = indices[page * page_size:(page + 1) * page_size]

            cards = "\n".join(self.render_card(idx) for idx in visible)
            body = PAGE_TEMPLATE.format(
                cards=cards,
                filter=html.escape(filter_name),
                page=page,
                page_plus_one=page + 1,
                page_size=page_size,
                frames=html.escape(frames_text),
                frames_url=html.escape(frames_text),
                total_pages=total_pages,
                total_rows=len(store.rows),
                filtered_rows=len(indices),
                labeled_rows=len(store.filtered_indices("labeled")),
                unlabeled_rows=len(store.filtered_indices("unlabeled")),
                prev_page=max(0, page - 1),
                next_page=min(total_pages - 1, page + 1),
            )
            self.write_html(body)

        def render_card(self, idx: int) -> str:
            row = store.rows[idx]
            model_ocr = store.model_ocr(idx)
            readable = row.get("readable", "")
            label = row.get("label", "")
            notes = row.get("notes", "")
            unreadable_checked = "checked" if readable.strip().lower() == "false" else ""
            return CARD_TEMPLATE.format(
                idx=idx,
                image_url=f"/image?idx={idx}",
                model_ocr=html.escape(model_ocr or "none"),
                model_ocr_value=html.escape(model_ocr),
                label=html.escape(label),
                notes=html.escape(notes),
                unreadable_checked=unreadable_checked,
                tower=html.escape(row.get("tower_name", "")),
                frame=html.escape(row.get("frame_index", "")),
                time=html.escape(row.get("video_time_s", "")),
                mode=html.escape(row.get("crop_mode", "")),
            )

        def serve_image(self, parsed) -> None:
            query = parse_qs(parsed.query)
            idx = int(query.get("idx", ["-1"])[0])
            if idx < 0 or idx >= len(store.rows):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = resolve_path(store.rows[idx]["image_path"])
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def write_html(self, text: str) -> None:
            data = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def write_json(self, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ReviewHandler


def parse_frame_filter(frames_text: str) -> set[str] | None:
    frames = {
        part.strip()
        for part in frames_text.replace("\n", ",").replace(" ", ",").split(",")
        if part.strip()
    }
    return frames or None


CARD_TEMPLATE = """
<div class="card" data-idx="{idx}">
  <img src="{image_url}" alt="crop {idx}" onclick="useModel(this.closest('.card'))">
  <div class="meta">#{idx} {tower} frame={frame} t={time}s {mode}</div>
  <button type="button" class="model" onclick="useModel(this.closest('.card'))">model OCR: {model_ocr}</button>
  <input class="label" inputmode="numeric" pattern="[0-9]*" value="{label}" placeholder="label">
  <label><input class="unreadable" type="checkbox" {unreadable_checked}> unreadable</label>
  <input class="notes" value="{notes}" placeholder="notes">
  <div class="actions">
    <button type="button" onclick="useModel(this.closest('.card'))" data-model="{model_ocr_value}">Use model</button>
    <button type="button" onclick="markUnreadable(this.closest('.card'))">Unreadable</button>
    <button type="button" onclick="clearCard(this.closest('.card'))">Clear</button>
  </div>
</div>
"""


PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Tower HP OCR Review</title>
<style>
body {{ font-family: sans-serif; margin: 16px; background: #111; color: #eee; }}
a {{ color: #8cf; }}
.toolbar {{ position: sticky; top: 0; z-index: 2; background: #111; padding: 12px 0; border-bottom: 1px solid #333; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; }}
.card {{ background: #1e1e1e; border: 1px solid #333; padding: 8px; border-radius: 8px; }}
.card img {{ width: 100%; height: 70px; object-fit: contain; image-rendering: pixelated; background: #000; cursor: pointer; }}
.meta {{ color: #aaa; font-size: 12px; min-height: 32px; }}
button, input, select {{ margin-top: 5px; }}
button {{ cursor: pointer; }}
.model {{ width: 100%; font-weight: bold; }}
.label, .notes {{ width: calc(100% - 8px); box-sizing: border-box; }}
.label {{ font-size: 20px; }}
.actions {{ display: flex; gap: 4px; flex-wrap: wrap; }}
.status {{ margin-left: 12px; color: #9f9; }}
</style>
</head>
<body>
<div class="toolbar">
  <form id="nav" method="get">
    filter:
    <select name="filter" onchange="this.form.submit()">
      <option value="unlabeled">unlabeled</option>
      <option value="all">all</option>
      <option value="labeled">labeled</option>
    </select>
    <input type="hidden" name="page" value="{page}">
    frames: <input name="frames" value="{frames}" placeholder="106680, 18360" size="24">
    page size: <input name="page_size" value="{page_size}" size="4">
    <button type="submit">Apply</button>
    <a href="/?filter={filter}&frames={frames_url}&page={prev_page}&page_size={page_size}">Prev</a>
    <a href="/?filter={filter}&frames={frames_url}&page={next_page}&page_size={page_size}">Next</a>
    page {page_plus_one}/{total_pages}
  </form>
  <button onclick="saveVisible()">Save visible</button>
  <span class="status" id="status"></span>
  <div>
    rows={total_rows}, filtered={filtered_rows}, labeled={labeled_rows}, unlabeled={unlabeled_rows}
  </div>
</div>
<div class="grid">
{cards}
</div>
<script>
document.querySelector('select[name="filter"]').value = "{filter}";
document.querySelectorAll('.unreadable').forEach(cb => {{
  cb.addEventListener('change', () => {{
    const card = cb.closest('.card');
    if (cb.checked) {{
      card.querySelector('.label').value = '';
    }}
  }});
}});
document.querySelectorAll('.label').forEach(input => {{
  input.addEventListener('input', () => {{
    if (input.value.trim()) {{
      input.closest('.card').querySelector('.unreadable').checked = false;
    }}
  }});
}});
function useModel(card) {{
  const model = card.querySelector('[data-model]').dataset.model;
  if (!model) return;
  card.querySelector('.label').value = model;
  card.querySelector('.unreadable').checked = false;
}}
function markUnreadable(card) {{
  card.querySelector('.label').value = '';
  card.querySelector('.unreadable').checked = true;
}}
function clearCard(card) {{
  card.querySelector('.label').value = '';
  card.querySelector('.unreadable').checked = false;
  card.querySelector('.notes').value = '';
}}
function collectVisible() {{
  return Array.from(document.querySelectorAll('.card')).map(card => {{
    let readable = '';
    if (card.querySelector('.unreadable').checked) readable = 'false';
    else if (card.querySelector('.label').value.trim()) readable = 'true';
    return {{
      idx: Number(card.dataset.idx),
      readable,
      label: card.querySelector('.label').value,
      notes: card.querySelector('.notes').value,
    }};
  }});
}}
async function saveVisible() {{
  const response = await fetch('/save', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{updates: collectVisible()}}),
  }});
  const payload = await response.json();
  document.getElementById('status').textContent = `saved ${{payload.changed}} changed rows`;
}}
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    store = LabelStore(args.labels_csv, args.checkpoint, device=args.device)
    handler = make_handler(store, args.page_size)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"reviewing {len(store.rows)} rows from {args.labels_csv}")
    print(f"open {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
