/* ROI adaptation pure helpers (no DOM; testable without a browser). */

import { NATIVE_SIZE } from '../state/store.js';
import { num } from '../utils/format.js';

// Probe frames to cycle through with "Another frame", in order. The
// initial auto-preview uses the server default (middle of the video).
export const ROI_PROBE_FRACTIONS = [0.25, 0.75, 0.125, 0.375, 0.625, 0.875];

export function roiAdaptAvailableFromDims(width, height) {
  const w = num(width), h = num(height);
  if (w === null || h === null || w <= 0 || h <= 0) return false;
  return w !== NATIVE_SIZE[0] || h !== NATIVE_SIZE[1];
}

export function roiAdaptNoticeText(width, height) {
  const w = Math.round(Number(width)), h = Math.round(Number(height));
  return w + 'x' + h + ' detected — fixed ROIs assume 1080x2400.';
}

export function roiSetFromPreview(preview, edits) {
  if (!preview || !Array.isArray(preview.rois)) return null;
  const out = {};
  for (const r of preview.rois) {
    if (!r || typeof r.name !== 'string' || !Array.isArray(r.rect)) continue;
    out[r.name] = r.rect;
  }
  // Manual drag adjustments override the proposed rects (x, y, w, h in
  // probe-frame native pixels, same contract as the preview entries).
  if (edits && typeof edits === 'object') {
    for (const name of Object.keys(edits)) {
      if (!(name in out)) continue;
      const e = edits[name];
      if (!Array.isArray(e) || e.length < 4) continue;
      const vals = e.slice(0, 4).map(num);
      if (vals.some((v) => v === null)) continue;
      out[name] = vals;
    }
  }
  return Object.keys(out).length ? out : null;
}

export function buildVideoStartPayload(opts) {
  const o = opts || {};
  const ra = o.roiAdapt || { checked: false, preview: null };
  const checked = !!ra.checked;
  const preview = ra.preview || null;
  return {
    video_path: o.videoPath,
    frame_stride: o.stride,
    start_frame: o.startFrame,
    max_frames: o.maxFrames,
    checkpoint: o.checkpoint || null,
    device: o.device || 'auto',
    adapt_rois: checked,
    // Adapt checked + proposal reviewed in the player: Start is the
    // confirmation, so the proposal (plus drag adjustments) is always used.
    roi_set: (checked && preview) ? roiSetFromPreview(preview, ra.edits) : null,
  };
}

export function formatRoiPreviewMeta(data) {
  const d = data || {};
  const frame = d.probe_frame !== undefined && d.probe_frame !== null ? d.probe_frame : '—';
  const rois = Array.isArray(d.rois) ? d.rois : [];
  let landmark = 0, scaled = 0;
  for (const r of rois) {
    if (r && r.source === 'landmark') landmark++;
    else if (r && r.source === 'scaled') scaled++;
  }
  let meta = 'Frame ' + String(frame) + ' · ' + landmark + ' landmark / ' + scaled + ' scaled';
  const warnings = Array.isArray(d.warnings) ? d.warnings.filter(Boolean) : [];
  if (warnings.length) meta += ' · ' + warnings.map(String).join('; ');
  return meta;
}
