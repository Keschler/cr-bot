/* ROI adaptation DOM wiring: proposal review in the middle player.
 * Pure payload helpers live in ./roi.js; this module owns the interactive
 * review (probe fetch, drag-to-adjust, meta line, event bindings).
 */

import { state } from '../state/store.js';
import { els, img, canvas, ctx } from '../utils/elements.js';
import { showError, toast } from '../utils/dom.js';
import { num } from '../utils/format.js';
import { apiGet } from '../api/client.js';
import { containRect } from '../utils/geometry.js';
import { ROI_PROBE_FRACTIONS, formatRoiPreviewMeta, roiAdaptNoticeText } from './roi.js';
import { renderCurrent } from './timeline.js';
import { refreshImage } from './session.js';

export function adaptRowLabel() {
  const box = els['check-adapt-rois'];
  return box && box.closest ? box.closest('label') : null;
}

export function clearRoiAdapt() {
  state.roiAdapt = { available: false, checked: false, preview: null,
    edits: {}, selRoi: null, dragRoi: null, image: null, dims: null,
    probeFrame: null, probeCursor: 0, showInMiddle: false, loading: false };
  hideRoiPreviewView();
  const row = adaptRowLabel();
  if (row) row.hidden = true;
  if (els['check-adapt-rois']) els['check-adapt-rois'].checked = false;
  if (els['adapt-notice']) { els['adapt-notice'].hidden = true; els['adapt-notice'].textContent = ''; }
  if (els['roi-preview-meta']) { els['roi-preview-meta'].hidden = true; els['roi-preview-meta'].textContent = ''; }
  if (els['btn-reset-rois']) els['btn-reset-rois'].hidden = true;
  if (els['btn-another-frame']) els['btn-another-frame'].hidden = true;
}

export function showRoiAdaptAvailable(width, height) {
  state.roiAdapt.available = true;
  state.roiAdapt.checked = true;
  state.roiAdapt.preview = null;
  state.roiAdapt.edits = {};
  state.roiAdapt.selRoi = null;
  state.roiAdapt.dragRoi = null;
  state.roiAdapt.image = null;
  state.roiAdapt.dims = null;
  state.roiAdapt.probeFrame = null;
  state.roiAdapt.probeCursor = 0;
  state.roiAdapt.showInMiddle = false;
  state.roiAdapt.loading = false;
  const row = adaptRowLabel();
  if (row) row.hidden = false;
  if (els['check-adapt-rois']) els['check-adapt-rois'].checked = true;
  if (els['adapt-notice']) {
    els['adapt-notice'].hidden = false;
    els['adapt-notice'].textContent = roiAdaptNoticeText(width, height);
  }
  if (els['roi-preview-meta']) { els['roi-preview-meta'].hidden = true; els['roi-preview-meta'].textContent = ''; }
  if (els['btn-reset-rois']) els['btn-reset-rois'].hidden = true;
  if (els['btn-another-frame']) els['btn-another-frame'].hidden = true;
  // Fetch the proposal immediately: review happens in the middle player,
  // and Start uses it automatically.
  previewRois();
}

export function renderRoiPreview(data) {
  const d = data || {};
  const rois = Array.isArray(d.rois) ? d.rois : [];
  state.roiAdapt.preview = {
    rois,
    native_size: d.native_size || null,
    warnings: Array.isArray(d.warnings) ? d.warnings.filter(Boolean) : [],
  };
  state.roiAdapt.image = (typeof d.image === 'string' && d.image) ? d.image : null;
  const ns = Array.isArray(d.native_size) ? d.native_size : null;
  state.roiAdapt.dims = (ns && ns.length >= 2 && num(ns[0]) > 0 && num(ns[1]) > 0)
    ? { w: num(ns[0]), h: num(ns[1]) } : null;
  state.roiAdapt.probeFrame = (d.probe_frame !== undefined && d.probe_frame !== null)
    ? d.probe_frame : null;
  state.roiAdapt.edits = {};
  state.roiAdapt.selRoi = null;
  state.roiAdapt.dragRoi = null;
  state.roiAdapt.showInMiddle = true;
  if (els['btn-another-frame']) els['btn-another-frame'].hidden = false;
  updateRoiAdaptMeta();
  showError('');
  refreshImage();
  renderCurrent();
}

export async function previewRois(probeIndex) {
  if (!state.uploadedVideoPath) {
    showError('Choose a video file first.');
    return;
  }
  if (state.roiAdapt.loading) return;
  state.roiAdapt.loading = true;
  const meta = els['roi-preview-meta'];
  if (meta) { meta.hidden = false; meta.textContent = 'Loading proposed ROIs…'; }
  try {
    // Raw probe frame (overlay=0): the review draws its own labeled boxes.
    // An explicit frame re-probes elsewhere in the video (default: middle).
    let url = '/api/roi-preview?path=' + encodeURIComponent(state.uploadedVideoPath) + '&overlay=0';
    if (probeIndex !== undefined && probeIndex !== null) {
      url += '&frame=' + encodeURIComponent(probeIndex);
    }
    const data = await apiGet(url);
    renderRoiPreview(data);
  } catch (err) {
    if (state.roiAdapt.preview) {
      // A failed re-probe keeps the previous proposal (and any
      // adjustments) instead of losing the review work.
      updateRoiAdaptMeta();
      toast('Re-probe failed — kept the previous proposal.', 'error');
    } else {
      state.roiAdapt.showInMiddle = false;
      if (meta) { meta.hidden = false; meta.textContent = 'ROI preview failed — see error above.'; }
      // Leave Another frame available so a bad probe frame can be skipped
      // without toggling Adapt off and on again.
      if (els['btn-another-frame']) els['btn-another-frame'].hidden = false;
      toast('ROI preview failed.', 'error', {
        action: { label: 'Retry', onClick: () => { previewRois(); } },
      });
    }
    showError(String(err && err.message || err));
  } finally {
    state.roiAdapt.loading = false;
  }
}

export async function anotherRoiFrame() {
  if (!state.uploadedVideoPath) {
    showError('Choose a video file first.');
    return;
  }
  if (state.roiAdapt.loading) return;
  // Re-checking via this button is the same as ticking Adapt: the new
  // proposal is reviewed in the player before Start uses it.
  if (!state.roiAdapt.checked) {
    state.roiAdapt.checked = true;
    if (els['check-adapt-rois']) els['check-adapt-rois'].checked = true;
  }
  let total = num(state.uploadedVideoFrames) || 0;
  if (!(total > 0)) {
    try {
      const info = await apiGet('/api/video/info?path=' + encodeURIComponent(state.uploadedVideoPath));
      total = num(info.frames) || 0;
      state.uploadedVideoFrames = total;
    } catch (err) { total = 0; }
  }
  if (!(total > 1)) {
    showError('Could not determine the video length — upload the file again.');
    return;
  }
  const cur = state.roiAdapt.probeFrame;
  let probe = null;
  for (let k = 0; k < ROI_PROBE_FRACTIONS.length; k++) {
    const frac = ROI_PROBE_FRACTIONS[(state.roiAdapt.probeCursor + k) % ROI_PROBE_FRACTIONS.length];
    const cand = Math.max(0, Math.min(total - 1, Math.floor(total * frac)));
    if (cand !== cur) {
      probe = cand;
      state.roiAdapt.probeCursor = (state.roiAdapt.probeCursor + k + 1) % ROI_PROBE_FRACTIONS.length;
      break;
    }
  }
  if (probe === null) {
    probe = (cur === 0 && total > 1) ? 1 : 0;
    if (probe === cur) {
      showError('No other frame to try in this video.');
      return;
    }
  }
  const btn = els['btn-another-frame'];
  if (btn) btn.disabled = true;
  try {
    await previewRois(probe);
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* ----- ROI preview review in the middle player ----- */

export function roiPreviewVisible() {
  const ra = state.roiAdapt;
  // showInMiddle is set only by a fresh pre-session preview and cleared on
  // Start/uncheck/upload, so the session always wins once it exists.
  return !!(ra && ra.checked && ra.preview && ra.showInMiddle);
}

export function roiPreviewImageReady() {
  return roiPreviewVisible() && img && img.dataset.preview === 'roi' &&
    !img.hidden && (img.naturalWidth || 0) > 0;
}

export function syncRoiPreviewClass(on) {
  const wrap = els['frame-wrap'];
  if (wrap) wrap.classList.toggle('roi-preview', !!on);
}

export function hideRoiPreviewView() {
  state.roiAdapt.showInMiddle = false;
  state.roiAdapt.selRoi = null;
  state.roiAdapt.dragRoi = null;
  syncRoiPreviewClass(false);
  if (canvas) canvas.style.cursor = '';
  if (img && img.dataset.preview === 'roi') {
    delete img.dataset.preview;
    delete img.dataset.previewFrame;
    delete img.dataset.src;
    delete img.dataset.fi;
    img.removeAttribute('src');
    img.hidden = true;
    if (els['frame-empty']) els['frame-empty'].style.display = '';
  }
}

export function roiPreviewDims() {
  const d = state.roiAdapt.dims;
  const w = d ? num(d.w) : null, h = d ? num(d.h) : null;
  return (w !== null && h !== null && w > 0 && h > 0) ? { w, h } : null;
}

// Proposed rects with manual adjustments applied. Boxes are
// [x0, y0, x1, y1] in probe-frame native pixels.
export function roiPreviewRects() {
  const pv = state.roiAdapt.preview;
  if (!pv || !Array.isArray(pv.rois)) return [];
  const out = [];
  for (const r of pv.rois) {
    if (!r || typeof r.name !== 'string' || !Array.isArray(r.rect) || r.rect.length < 4) continue;
    const vals = r.rect.slice(0, 4).map(num);
    if (vals.some((v) => v === null)) continue;
    let x = vals[0], y = vals[1], w = vals[2], h = vals[3];
    let edited = false;
    const e = state.roiAdapt.edits ? state.roiAdapt.edits[r.name] : null;
    if (Array.isArray(e) && e.length >= 4) {
      const ev = e.slice(0, 4).map(num);
      if (!ev.some((v) => v === null)) { x = ev[0]; y = ev[1]; w = ev[2]; h = ev[3]; edited = true; }
    }
    out.push({ name: r.name, box: [x, y, x + w, y + h], source: String(r.source || 'scaled'), edited });
  }
  return out;
}

export function hitRoi(x, y) {
  const dims = roiPreviewDims();
  if (!dims) return null;
  const rect = containRect();
  if (!rect.w || !rect.h) return null;
  const R = 10;
  let body = null;
  for (const r of roiPreviewRects()) {
    const b = r.box;
    const p0 = { x: rect.x + (b[0] / dims.w) * rect.w, y: rect.y + (b[1] / dims.h) * rect.h };
    const p1 = { x: rect.x + (b[2] / dims.w) * rect.w, y: rect.y + (b[3] / dims.h) * rect.h };
    const corners = { nw: p0, ne: { x: p1.x, y: p0.y }, sw: { x: p0.x, y: p1.y }, se: p1 };
    for (const part of ['nw', 'ne', 'sw', 'se']) {
      const c = corners[part];
      if (Math.abs(x - c.x) <= R && Math.abs(y - c.y) <= R) {
        return { name: r.name, part, box: b.slice() };
      }
    }
    if (x >= p0.x - 6 && x <= p1.x + 6 && y >= p0.y - 6 && y <= p1.y + 6) {
      const area = Math.max(1, (p1.x - p0.x) * (p1.y - p0.y));
      if (!body || area < body.area) body = { name: r.name, part: 'move', box: b.slice(), area };
    }
  }
  return body;
}

export function drawRoiPreviewBoxes() {
  const wrap = els['frame-wrap'];
  if (!wrap || !canvas || !ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = wrap.clientWidth, h = wrap.clientHeight;
  canvas.width = Math.max(1, Math.round(w * dpr));
  canvas.height = Math.max(1, Math.round(h * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const dims = roiPreviewDims();
  if (!dims || !img.naturalWidth) return;
  const rect = containRect();
  if (!rect.w || !rect.h) return;
  const drag = state.roiAdapt.dragRoi;
  const rows = roiPreviewRects().map((r) =>
    (drag && drag.name === r.name && drag.curBox) ? { name: r.name, box: drag.curBox.slice(), source: r.source, edited: r.edited } : r);
  for (const r of rows) {
    const b = r.box;
    const x0 = rect.x + (b[0] / dims.w) * rect.w, y0 = rect.y + (b[1] / dims.h) * rect.h;
    const x1 = rect.x + (b[2] / dims.w) * rect.w, y1 = rect.y + (b[3] / dims.h) * rect.h;
    const selected = state.roiAdapt.selRoi === r.name;
    const color = selected ? '#ffd166' : r.edited ? '#ffa94d'
      : (r.source === 'landmark' || r.source === 'native') ? '#3ddc84' : '#2ea6ff';
    ctx.save();
    if (selected) { ctx.shadowColor = 'rgba(255,209,102,0.9)'; ctx.shadowBlur = 10; }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    if (r.edited && !selected) ctx.setLineDash([6, 4]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.restore();
    // Name tags only on wide boxes or the selection: dozens of ROIs on a
    // small frame would otherwise bury each other. Click a box to identify
    // and resize it.
    if (selected || (x1 - x0) >= 64) {
      ctx.save();
      ctx.font = '11px system-ui, sans-serif';
      const tw = ctx.measureText(r.name).width;
      ctx.fillStyle = 'rgba(5,8,15,0.85)';
      ctx.fillRect(x1 + 2, y0 - 8, tw + 8, 16);
      ctx.fillStyle = color;
      ctx.fillText(r.name, x1 + 6, y0 + 4);
      ctx.restore();
    }
    if (selected) {
      ctx.save();
      ctx.fillStyle = '#ffd166';
      for (const [cx, cy] of [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) {
        ctx.fillRect(cx - 4, cy - 4, 8, 8);
      }
      ctx.restore();
    }
  }
}

export function updateRoiAdaptMeta() {
  const pv = state.roiAdapt.preview;
  const meta = els['roi-preview-meta'];
  if (!pv || !meta) return;
  const n = state.roiAdapt.edits ? Object.keys(state.roiAdapt.edits).length : 0;
  meta.hidden = false;
  meta.textContent = formatRoiPreviewMeta({
    probe_frame: state.roiAdapt.probeFrame, rois: pv.rois, warnings: pv.warnings || [],
  }) + (n ? ' · ' + n + ' adjusted' : '') +
    ' — review in the player, drag a box to move, a corner to resize.';
  const btn = els['btn-reset-rois'];
  if (btn) btn.hidden = n === 0;
}

export function resetRoiEdits() {
  state.roiAdapt.edits = {};
  state.roiAdapt.selRoi = null;
  updateRoiAdaptMeta();
  drawRoiPreviewBoxes();
}

export function roiCanvasMouseDown(ev) {
  ev.preventDefault();
  const r = canvas.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  const hit = hitRoi(x, y);
  if (hit) {
    state.roiAdapt.selRoi = hit.name;
    state.roiAdapt.dragRoi = {
      name: hit.name, part: hit.part,
      startX: x, startY: y, origBox: hit.box.slice(), curBox: hit.box.slice(),
      moved: false,
    };
  } else {
    state.roiAdapt.selRoi = null;
    state.roiAdapt.dragRoi = null;
  }
  drawRoiPreviewBoxes();
}

export function updateRoiDrag(x, y) {
  const drag = state.roiAdapt.dragRoi;
  if (!drag) return;
  const dims = roiPreviewDims();
  const rect = containRect();
  if (!dims || !rect.w || !rect.h) return;
  const kx = dims.w / rect.w, ky = dims.h / rect.h;
  const dx = (x - drag.startX) * kx;
  const dy = (y - drag.startY) * ky;
  if (Math.abs(x - drag.startX) + Math.abs(y - drag.startY) > 3) drag.moved = true;
  const o = drag.origBox;
  let b;
  switch (drag.part) {
    case 'nw': b = [o[0] + dx, o[1] + dy, o[2], o[3]]; break;
    case 'ne': b = [o[0], o[1] + dy, o[2] + dx, o[3]]; break;
    case 'sw': b = [o[0] + dx, o[1], o[2], o[3] + dy]; break;
    case 'se': b = [o[0], o[1], o[2] + dx, o[3] + dy]; break;
    default: b = [o[0] + dx, o[1] + dy, o[2] + dx, o[3] + dy]; break;
  }
  let [ax0, ay0, ax1, ay1] = [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])];
  ax0 = Math.max(0, Math.min(ax0, dims.w - 4));
  ay0 = Math.max(0, Math.min(ay0, dims.h - 4));
  ax1 = Math.max(ax0 + 4, Math.min(ax1, dims.w));
  ay1 = Math.max(ay0 + 4, Math.min(ay1, dims.h));
  drag.curBox = [ax0, ay0, ax1, ay1];
  drawRoiPreviewBoxes();
}

export function finishRoiDrag() {
  const drag = state.roiAdapt.dragRoi;
  state.roiAdapt.dragRoi = null;
  if (!drag) return;
  // Click without drag = select; a real drag commits an [x, y, w, h] edit.
  state.roiAdapt.selRoi = drag.name;
  if (drag.moved && drag.curBox) {
    const b = drag.curBox;
    state.roiAdapt.edits[drag.name] = [b[0], b[1], b[2] - b[0], b[3] - b[1]];
  }
  updateRoiAdaptMeta();
  drawRoiPreviewBoxes();
}

export function bindRoiEvents() {
  if (els['check-adapt-rois']) {
    els['check-adapt-rois'].addEventListener('change', () => {
      state.roiAdapt.checked = !!els['check-adapt-rois'].checked;
      if (state.roiAdapt.checked) {
        // Review the cached proposal, or fetch it automatically on first check.
        if (state.roiAdapt.preview) {
          state.roiAdapt.showInMiddle = true;
          if (els['btn-another-frame']) els['btn-another-frame'].hidden = false;
          updateRoiAdaptMeta();
          refreshImage();
          renderCurrent();
        } else {
          previewRois();
        }
      } else {
        if (els['btn-another-frame']) els['btn-another-frame'].hidden = true;
        if (els['btn-reset-rois']) els['btn-reset-rois'].hidden = true;
        if (els['roi-preview-meta']) els['roi-preview-meta'].hidden = true;
        hideRoiPreviewView();
        renderCurrent();
      }
    });
  }
  if (els['btn-reset-rois']) {
    els['btn-reset-rois'].addEventListener('click', resetRoiEdits);
  }
  if (els['btn-another-frame']) {
    els['btn-another-frame'].addEventListener('click', anotherRoiFrame);
  }
}
