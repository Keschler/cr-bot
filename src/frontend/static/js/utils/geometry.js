/* Canvas/coordinate geometry: display rects, grid mapping, ROI rects.
 * Depends on the live <img> size (elements) and backend frame dims.
 */

import { img, els } from './elements.js';
import { GRID_COLS, GRID_ROWS, GRID_SPEC_FALLBACK, HAND_SLOT_ROIS, ROI_REF_W, ROI_REF_H, state } from '../state/store.js';
import { num } from './format.js';
import { roiSetFromPreview } from '../features/roi.js';
import { visualStateOf } from '../features/frames.js';

export function containRect() {
  const wrap = els['frame-wrap'];
  const w = wrap.clientWidth, h = wrap.clientHeight;
  const nw = img.naturalWidth || 0, nh = img.naturalHeight || 0;
  if (!nw || !nh) return { x: 0, y: 0, w, h, scaleX: 0, scaleY: 0 };
  const s = Math.min(w / nw, h / nh);
  const dw = nw * s, dh = nh * s;
  return { x: (w - dw) / 2, y: (h - dh) / 2, w: dw, h: dh, scaleX: dw / nw, scaleY: dh / nh };
}

export function frameDims(frame) {
  if (!frame || typeof frame !== 'object') return null;
  const w = num(frame.frame_width), h = num(frame.frame_height);
  if (w !== null && h !== null && w > 0 && h > 0) return { w, h };
  return null;
}

export function toDisplay(pt, rect, frame) {
  const nw = img.naturalWidth || 0, nh = img.naturalHeight || 0;
  let x = Number(pt[0]), y = Number(pt[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  // Prefer the backend-reported coordinate space (normalized frame pixels,
  // before JPEG downscaling). This keeps boxes aligned even though the
  // served JPEG is downscaled to max_width=720.
  const dims = frameDims(frame);
  let fx, fy;
  if (dims) {
    fx = x / dims.w; fy = y / dims.h;
  } else if (x >= 0 && x <= 1.5 && y >= 0 && y <= 1.5) {
    // Normalized fractions.
    fx = x; fy = y;
  } else if (nw && nh) {
    fx = x / nw; fy = y / nh;
  } else if (x <= 100 && y <= 100) {
    fx = x / 100; fy = y / 100;
  } else {
    return null;
  }
  if (!Number.isFinite(fx) || !Number.isFinite(fy)) return null;
  return { x: rect.x + fx * rect.w, y: rect.y + fy * rect.h };
}

export function fromDisplay(px, py, rect, frame) {
  const dims = frameDims(frame);
  if (!dims || !rect.w || !rect.h) return null;
  const fx = (px - rect.x) / rect.w, fy = (py - rect.y) / rect.h;
  if (!(fx >= 0 && fx <= 1 && fy >= 0 && fy <= 1)) return null;
  return [fx * dims.w, fy * dims.h];
}

export function parseCell(cell) {
  if (cell === null || cell === undefined) return null;
  if (Array.isArray(cell) && cell.length >= 2) {
    const c = num(cell[0]), r = num(cell[1]);
    return c === null || r === null ? null : { col: c, row: r };
  }
  if (typeof cell === 'object') {
    const c = num(cell.col !== undefined ? cell.col : cell.x);
    const r = num(cell.row !== undefined ? cell.row : cell.y);
    return c === null || r === null ? null : { col: c, row: r };
  }
  return null;
}

export function handSlotRect(slot, rect, frame) {
  const dims = frameDims(frame);
  if (!dims || slot === null || slot === undefined || slot < 0 || slot >= HAND_SLOT_ROIS.length) return null;
  // Prefer the reviewed/adapted hand ROIs (same video, probe-pixel space) so
  // the Inspect arrow starts on the real card for non-native videos. Falls
  // back to the fixed 1080x2400 reference fractions otherwise.
  try {
    const ra = state.roiAdapt;
    const pw = ra && ra.dims ? num(ra.dims.w) : null;
    const ph = ra && ra.dims ? num(ra.dims.h) : null;
    if (ra && ra.preview && pw && ph && pw > 0 && ph > 0) {
      const set = roiSetFromPreview(ra.preview, ra.edits);
      const hit = set && set['hand_card_slot_' + (slot + 1)];
      if (hit && hit.length >= 4) {
        const vals = hit.slice(0, 4).map(num);
        if (!vals.some((v) => v === null)) {
          return {
            x: rect.x + (vals[0] / pw) * rect.w,
            y: rect.y + (vals[1] / ph) * rect.h,
            w: (vals[2] / pw) * rect.w,
            h: (vals[3] / ph) * rect.h,
          };
        }
      }
    }
  } catch (e) { /* fall through to fixed ROIs */ }
  const roi = HAND_SLOT_ROIS[slot];
  const kx = dims.w / ROI_REF_W, ky = dims.h / ROI_REF_H;
  return {
    x: rect.x + (roi.x * kx / dims.w) * rect.w,
    y: rect.y + (roi.y * ky / dims.h) * rect.h,
    w: (roi.w * kx / dims.w) * rect.w,
    h: (roi.h * ky / dims.h) * rect.h,
  };
}

export function gridSpec() {
  const g = state.gridSpec;
  if (g && num(g.cols) === GRID_COLS && num(g.rows) === GRID_ROWS &&
      [g.x0, g.y0, g.x1, g.y1].every((v) => num(v) !== null)) {
    // River/bridge layout rides along when the backend serves it
    // (GET /api/grid also returns river_rows/bridge_cols); absent or
    // malformed values simply disable those layers.
    const rows = Array.isArray(g.river_rows) ? g.river_rows.map(num).filter((v) => v !== null) : null;
    const cols = Array.isArray(g.bridge_cols) ? g.bridge_cols.map(num).filter((v) => v !== null) : null;
    return {
      cols: GRID_COLS, rows: GRID_ROWS, x0: +g.x0, y0: +g.y0, x1: +g.x1, y1: +g.y1,
      riverRows: rows && rows.length ? rows : null,
      bridgeCols: cols && cols.length ? cols : null,
    };
  }
  return Object.assign({ riverRows: null, bridgeCols: null }, GRID_SPEC_FALLBACK);
}

export function arenaPxOf(frame) {
  // (ax, ay, aw, ah): arena origin + size in frame pixels, same tuple the
  // backend trackers pass to ACTION_GRID.cell_to_pixel_center.
  const vs = visualStateOf(frame);
  const a = vs.arena_px;
  if (!Array.isArray(a) || a.length < 4) return null;
  const vals = a.slice(0, 4).map(num);
  if (vals.some((v) => v === null) || vals[2] <= 0 || vals[3] <= 0) return null;
  return vals;
}

export function gridBounds(rect, frame) {
  // Display rect of the action grid: cells map into normalized arena coords
  // (grid spec), then through arena_px into frame pixels — exactly mirroring
  // ACTION_GRID.cell_to_pixel_center(col, row, arena_px).
  const spec = gridSpec();
  const arena = arenaPxOf(frame);
  const dims = frameDims(frame);
  if (!arena || !dims) return rect;
  const fx0 = (arena[0] + spec.x0 * arena[2]) / dims.w;
  const fy0 = (arena[1] + spec.y0 * arena[3]) / dims.h;
  const fx1 = (arena[0] + spec.x1 * arena[2]) / dims.w;
  const fy1 = (arena[1] + spec.y1 * arena[3]) / dims.h;
  return {
    x: rect.x + fx0 * rect.w,
    y: rect.y + fy0 * rect.h,
    w: (fx1 - fx0) * rect.w,
    h: (fy1 - fy0) * rect.h,
  };
}

export function cellToDisplay(cell, rect, frame) {
  const p = parseCell(cell);
  if (!p) return null;
  const spec = gridSpec();
  const arena = arenaPxOf(frame);
  const dims = frameDims(frame);
  if (arena && dims) {
    const nx = spec.x0 + ((p.col + 0.5) / spec.cols) * (spec.x1 - spec.x0);
    const ny = spec.y0 + ((p.row + 0.5) / spec.rows) * (spec.y1 - spec.y0);
    const fx = (arena[0] + nx * arena[2]) / dims.w;
    const fy = (arena[1] + ny * arena[3]) / dims.h;
    if (!Number.isFinite(fx) || !Number.isFinite(fy)) return null;
    return { x: rect.x + fx * rect.w, y: rect.y + fy * rect.h };
  }
  // Legacy fallback when arena/dims are unavailable: grid over full image.
  return {
    x: rect.x + ((p.col + 0.5) / GRID_COLS) * rect.w,
    y: rect.y + ((p.row + 0.5) / GRID_ROWS) * rect.h,
  };
}
