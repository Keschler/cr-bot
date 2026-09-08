/* Center column: badges, time label, and the canvas overlay layers
 * (grid, detection boxes, suggestion markers, inspect arrow).
 */

import { state, GRID_COLS, GRID_ROWS } from '../state/store.js';
import { els, img, canvas, ctx } from '../utils/elements.js';
import { num, suggestionCardName } from '../utils/format.js';
import { containRect, toDisplay, parseCell, handSlotRect, gridBounds, gridSpec, cellToDisplay } from '../utils/geometry.js';
import { visualStateOf, suggestionsOf, diagnosticsOf, currentFrame, topSuggestions } from './frames.js';
import { roiPreviewVisible, drawRoiPreviewBoxes } from './roi-editor.js';
import { imageMatchesFrame } from './session.js';
import { effectiveRows, draftUpdateFor } from './corrections.js';

export function renderCenter(frame) {
  const vs = visualStateOf(frame);
  const diag = diagnosticsOf(frame);
  const sug = suggestionsOf(frame);

  els['badge-ingame'].textContent = !frame ? '—' : frame.in_game ? 'in_game' : 'menu';
  els['badge-ingame'].classList.toggle('is-live', !!(frame && frame.in_game));

  const fps = num(diag.fps);
  const lat = num(diag.latency_ms) !== null ? num(diag.latency_ms)
    : num(diag.inference_ms) !== null ? num(diag.inference_ms) : null;
  let perf = '—';
  if (fps !== null && lat !== null) perf = fps.toFixed(0) + ' FPS · ' + lat.toFixed(0) + ' ms';
  else if (fps !== null) perf = fps.toFixed(0) + ' FPS';
  else if (lat !== null) perf = lat.toFixed(0) + ' ms';
  else if (diag.ocr_confidence !== undefined && diag.ocr_confidence !== null) {
    const ocr = num(diag.ocr_confidence);
    perf = ocr === null ? 'OCR —' : 'OCR ' + (ocr * 100).toFixed(1) + '%';
  }
  els['badge-perf'].textContent = perf;

  els['badge-arena'].textContent = diag.arena !== undefined && diag.arena !== null
    ? String(diag.arena) : (vs.arena !== undefined && vs.arena !== null ? String(vs.arena) : 'Arena —');

  if (!frame) {
    els['time-label'].textContent = '—';
  } else {
    els['time-label'].textContent = 'f' + frame.frame_index +
      (frame.timestamp_s !== undefined ? ' · ' + Number(frame.timestamp_s).toFixed(1) + 's' : '');
  }
  drawOverlay(frame, sug, diag);
  // Proposed ROIs under review render on top of the probe frame.
  if (roiPreviewVisible() && img.dataset.preview === 'roi') drawRoiPreviewBoxes();
}

export function drawOverlay(frame, suggestions, diagnostics) {
  const wrap = els['frame-wrap'];
  if (!wrap || !canvas || !ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = wrap.clientWidth, h = wrap.clientHeight;
  canvas.width = Math.max(1, Math.round(w * dpr));
  canvas.height = Math.max(1, Math.round(h * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!frame) return;

  const rect = containRect();
  const t = state.toggles;
  const vs = visualStateOf(frame);
  const sug = suggestions || suggestionsOf(frame);
  const diag = diagnostics || diagnosticsOf(frame);

  if (t.grid) drawGrid(rect, frame);
  // Detections, labels, and suggestion markers belong to the cursor frame,
  // so they draw only when the displayed pixels are that frame (the preview
  // tracks the cursor via per-frame image URLs; see refreshImage).
  if (!imageMatchesFrame(frame)) return;
  if (t.boxes) drawBoxes(vs, rect, t.labels, false, frame);
  else if (t.labels) drawBoxes(vs, rect, true, true, frame);
  drawSuggestionMarkers(sug, rect, frame);
}

export function drawSuggestionMarkers(sug, rect, frame) {
  const plays = topSuggestions(sug);
  plays.forEach((s, i) => {
    if (!s || String(s.kind || '').toLowerCase() !== 'play') return;
    const p = cellToDisplay(s.cell, rect, frame);
    if (!p) return;
    const selected = state.selectedRank === i;
    const r = selected ? 13 : 9;
    ctx.save();
    if (selected) {
      ctx.shadowColor = 'rgba(255,170,60,0.9)';
      ctx.shadowBlur = 12;
    }
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = selected ? 'rgba(255,150,50,0.95)' : 'rgba(5,8,15,0.85)';
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.lineWidth = 2;
    ctx.strokeStyle = selected ? '#ffd9a0' : 'rgba(46,166,255,0.9)';
    ctx.stroke();
    ctx.fillStyle = selected ? '#04121f' : '#dbe4f5';
    ctx.font = 'bold 11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), p.x, p.y + 0.5);
    ctx.restore();
  });
  const sel = state.selectedRank !== null ? plays[state.selectedRank] : null;
  if (sel && String(sel.kind || '').toLowerCase() === 'play') drawPlayArrow(sel, rect, frame);
}

export function drawPlayArrow(s, rect, frame) {
  // Curved "flick" from the played card's real position in the video frame
  // (its extractor hand-slot ROI) up to the placement cell, labelled with
  // the card name. Slot 1 starts bottom-left, slot 4 bottom-right.
  const p = cellToDisplay(s.cell, rect, frame);
  if (!p) return;
  const slot = num(s.card_slot);
  const cardRect = handSlotRect(slot, rect, frame);
  const start = cardRect
    ? { x: cardRect.x + cardRect.w / 2, y: cardRect.y + 8 }
    : { x: p.x, y: rect.y + rect.h - 4 };
  const name = suggestionCardName(s, visualStateOf(frame)) || 'card';
  const cell = parseCell(s.cell);
  const gb = gridBounds(rect, frame);
  const cw = gb.w / GRID_COLS, ch = gb.h / GRID_ROWS;
  ctx.save();
  if (cell) {
    ctx.strokeStyle = 'rgba(255,170,60,0.95)';
    ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(255,150,50,0.8)';
    ctx.shadowBlur = 8;
    ctx.strokeRect(gb.x + cell.col * cw, gb.y + cell.row * ch, cw, ch);
    ctx.shadowBlur = 0;
  }
  if (cardRect) {
    ctx.strokeStyle = 'rgba(255,180,80,0.9)';
    ctx.lineWidth = 2;
    ctx.strokeRect(cardRect.x, cardRect.y, cardRect.w, cardRect.h);
  }
  // Curve leaves the card horizontally (control shares the cell's x with the
  // card's y), arriving at the cell vertically.
  const cx = p.x, cy = start.y;
  const endX = p.x, endY = p.y + 16;
  ctx.strokeStyle = 'rgba(255,180,80,0.95)';
  ctx.fillStyle = 'rgba(255,180,80,0.95)';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.quadraticCurveTo(cx, cy, endX, endY);
  ctx.stroke();
  // Arrowhead oriented along the end tangent (end - control).
  const dx = endX - cx, dy = endY - cy;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len;
  const sz = 9;
  const bx = endX - ux * sz, by = endY - uy * sz;
  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(bx - uy * sz * 0.55, by + ux * sz * 0.55);
  ctx.lineTo(bx + uy * sz * 0.55, by - ux * sz * 0.55);
  ctx.closePath();
  ctx.fill();
  // Card label just above the origin card.
  ctx.font = '12px system-ui, sans-serif';
  const label = '▶ ' + name;
  const tw = ctx.measureText(label).width;
  const lx = Math.max(rect.x + 2, Math.min(rect.x + rect.w - tw - 12, start.x - tw / 2));
  const ly = Math.max(rect.y + 12, start.y - 8);
  ctx.fillStyle = 'rgba(5,8,15,0.85)';
  ctx.fillRect(lx - 4, ly - 13, tw + 8, 17);
  ctx.fillStyle = '#ffd9a0';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(label, lx, ly);
  ctx.restore();
}

export function drawGrid(rect, frame) {
  // Grid lines span the arena-mapped grid bounds, not the full image.
  // The river band and bridge columns shade in when the backend grid spec
  // serves them (GET /api/grid river_rows/bridge_cols).
  const gb = gridBounds(rect, frame);
  const spec = gridSpec();
  ctx.save();
  if (spec.riverRows) {
    const lo = Math.min.apply(null, spec.riverRows), hi = Math.max.apply(null, spec.riverRows);
    ctx.fillStyle = 'rgba(46,166,255,0.10)';
    ctx.fillRect(gb.x, gb.y + (lo / GRID_ROWS) * gb.h,
      gb.w, ((hi - lo + 1) / GRID_ROWS) * gb.h);
  }
  if (spec.bridgeCols) {
    ctx.fillStyle = 'rgba(255,180,80,0.12)';
    for (const c of spec.bridgeCols) {
      ctx.fillRect(gb.x + (c / GRID_COLS) * gb.w, gb.y,
        gb.w / GRID_COLS, gb.h);
    }
  }
  ctx.strokeStyle = 'rgba(46,166,255,0.35)';
  ctx.lineWidth = 1;
  for (let c = 1; c < GRID_COLS; c++) {
    const x = gb.x + (c / GRID_COLS) * gb.w;
    ctx.beginPath(); ctx.moveTo(x, gb.y); ctx.lineTo(x, gb.y + gb.h); ctx.stroke();
  }
  for (let r = 1; r < GRID_ROWS; r++) {
    const y = gb.y + (r / GRID_ROWS) * gb.h;
    ctx.beginPath(); ctx.moveTo(gb.x, y); ctx.lineTo(gb.x + gb.w, y); ctx.stroke();
  }
  ctx.restore();
}

export function drawBoxes(vs, rect, withLabels, labelsOnly, frame) {
  // Single detection layer, driven by retained true boxes. Edit states
  // (moved/relabeled, deleted, added, selection, rubber band) render on top;
  // selection and handles only in edit mode.
  const eff = effectiveRows(frame);
  if (!eff.rows.length) {
    drawCenterTicks(vs, rect, withLabels, labelsOnly, frame);
    return;
  }
  const inEdit = state.editMode && currentFrame() === frame;
  const draft = frame ? state.correctionDrafts[String(frame.frame_index)] : null;
  const stroke = (box, color, dashed, width) => {
    if (!Array.isArray(box) || box.length < 4) return null;
    const p0 = toDisplay([box[0], box[1]], rect, frame);
    const p1 = toDisplay([box[2], box[3]], rect, frame);
    if (!p0 || !p1) return null;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 2;
    if (dashed) ctx.setLineDash([6, 4]);
    ctx.strokeRect(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y);
    ctx.restore();
    return { x0: p0.x, y0: p0.y, x1: p1.x, y1: p1.y };
  };
  const tag = (box, text, color) => {
    const p1 = toDisplay([box[2], box[1]], rect, frame);
    if (!p1) return;
    ctx.save();
    ctx.font = '11px system-ui, sans-serif';
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = 'rgba(5,8,15,0.85)';
    ctx.fillRect(p1.x + 2, p1.y - 8, tw + 8, 16);
    ctx.fillStyle = color;
    ctx.fillText(text, p1.x + 6, p1.y + 4);
    ctx.restore();
  };
  for (const row of eff.rows) {
    const key = row.effKey;
    const isDel = eff.deleted.has(key);
    const isAdd = key.slice(0, 2) === 'a:';
    const upd = draft ? draftUpdateFor(draft, key) : null;
    let color = row.team === 'ally' ? '#3ddc84' : '#ff6b6b';
    let dashed = false;
    if (isAdd) { color = '#ffd166'; dashed = true; }
    else if (isDel) { color = 'rgba(255,107,107,0.75)'; dashed = true; }
    else if (upd) { color = '#ffa94d'; }
    const conf = row.confidence !== undefined && row.confidence !== null
      ? Number(row.confidence).toFixed(2) : null;
    const text = String(row.class_name !== undefined ? row.class_name : '?') +
      (conf === null ? '' : ' ' + conf);
    if (!labelsOnly) {
      const disp = stroke(row.box, color, dashed);
      if (withLabels) tag(row.box, text, color);
      if (inEdit && key === state.selection && disp) {
        ctx.save();
        ctx.strokeStyle = '#ffd166';
        ctx.lineWidth = 2;
        ctx.shadowColor = 'rgba(255,209,102,0.9)';
        ctx.shadowBlur = 10;
        ctx.strokeRect(disp.x0 - 3, disp.y0 - 3, (disp.x1 - disp.x0) + 6, (disp.y1 - disp.y0) + 6);
        ctx.restore();
        ctx.save();
        ctx.fillStyle = '#ffd166';
        const pts = [
          [disp.x0, disp.y0], [disp.x1, disp.y0], [disp.x0, disp.y1], [disp.x1, disp.y1],
        ];
        for (const [hx, hy] of pts) ctx.fillRect(hx - 4, hy - 4, 8, 8);
        ctx.restore();
      }
    } else if (withLabels) {
      tag(row.box, text, color);
    }
  }
  if (inEdit && state.dragBox) {
    const b = state.dragBox;
    ctx.save();
    ctx.strokeStyle = '#ffd166';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(Math.min(b.x0, b.x1), Math.min(b.y0, b.y1),
      Math.abs(b.x1 - b.x0), Math.abs(b.y1 - b.y0));
    ctx.restore();
  }
}

export function drawCenterTicks(vs, rect, withLabels, labelsOnly, frame) {
  // Fallback for frames without retained detection rows: center markers
  // from the display summary (no box extents available there).
  const ally = Array.isArray(vs.ally_units) ? vs.ally_units : [];
  const enemy = Array.isArray(vs.enemy_units) ? vs.enemy_units : [];
  const items = ally.map((u) => ({ u, team: 'ally' })).concat(enemy.map((u) => ({ u, team: 'enemy' })));
  for (const { u, team } of items) {
    if (!u || !Array.isArray(u.center_px)) continue;
    const p = toDisplay(u.center_px, rect, frame);
    if (!p) continue;
    const color = team === 'ally' ? '#3ddc84' : '#ff6b6b';
    const s = 14;
    if (!labelsOnly) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(p.x - s, p.y - s, s * 2, s * 2);
      ctx.beginPath();
      ctx.moveTo(p.x - s - 4, p.y); ctx.lineTo(p.x + s + 4, p.y);
      ctx.moveTo(p.x, p.y - s - 4); ctx.lineTo(p.x, p.y + s + 4);
      ctx.stroke();
      ctx.restore();
    }
    if (withLabels) {
      const label = (u.label !== undefined ? String(u.label) : '?') +
        (u.confidence !== undefined && u.confidence !== null ? ' ' + Number(u.confidence).toFixed(2) : '');
      ctx.save();
      ctx.font = '11px system-ui, sans-serif';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(5,8,15,0.85)';
      ctx.fillRect(p.x + s + 2, p.y - s - 8, tw + 8, 16);
      ctx.fillStyle = color;
      ctx.fillText(label, p.x + s + 6, p.y - s + 4);
      ctx.restore();
    }
  }
}
