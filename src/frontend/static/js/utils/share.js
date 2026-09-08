/* Pure export/share helpers (no DOM, no state): CSV/JSON builders,
 * location-hash deep links, blob download. Testable without a browser.
 */

import { num } from './format.js';

export function csvCell(value) {
  const s = value === null || value === undefined ? '' : String(value);
  return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

// rows: [{side, card, fi, t, ts}] — all values already resolved by the caller.
export function buildActionsCsv(rows) {
  const head = 'side,card,frame_index,video_time_s,timestamp_s';
  const lines = [head];
  for (const r of rows || []) {
    lines.push([
      csvCell(r.side), csvCell(r.card), csvCell(r.fi),
      csvCell(r.t === null || r.t === undefined ? '' : r.t),
      csvCell(r.ts === null || r.ts === undefined ? '' : r.ts),
    ].join(','));
  }
  return lines.join('\n') + '\n';
}

export function buildSessionExport({ label, videoPath, videoName, params, cursorFrame, frames }) {
  return {
    app: 'arena-replay-analyst',
    version: 1,
    exported_at: new Date().toISOString(),
    label: label || '',
    video: { path: videoPath || '', name: videoName || '' },
    params: params || {},
    cursor_frame: cursorFrame !== undefined ? cursorFrame : null,
    frame_count: Array.isArray(frames) ? frames.length : 0,
    frames: Array.isArray(frames) ? frames : [],
  };
}

// Deep link: #f=<frame_index>&r=<rank>&v=<video name>. All parts optional.
export function encodeHash({ frame, rank, video }) {
  const parts = [];
  if (frame !== undefined && frame !== null && frame !== '') parts.push('f=' + encodeURIComponent(String(frame)));
  if (rank !== undefined && rank !== null && rank !== '') parts.push('r=' + encodeURIComponent(String(rank)));
  if (video) parts.push('v=' + encodeURIComponent(String(video)));
  return parts.length ? '#' + parts.join('&') : '#';
}

export function decodeHash(hash) {
  const out = { frame: null, rank: null, video: null };
  const raw = String(hash || '').replace(/^#/, '');
  if (!raw) return out;
  for (const part of raw.split('&')) {
    const eq = part.indexOf('=');
    if (eq < 0) continue;
    const key = part.slice(0, eq);
    let val = null;
    try { val = decodeURIComponent(part.slice(eq + 1)); } catch (e) { continue; }
    if (key === 'f') {
      const n = num(val);
      out.frame = n !== null ? Math.round(n) : null;
    } else if (key === 'r') {
      const n = num(val);
      out.rank = n !== null ? Math.round(n) : null;
    } else if (key === 'v') {
      out.video = val || null;
    }
  }
  return out;
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || 'download';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    try { document.body.removeChild(a); } catch (e) { /* gone */ }
    URL.revokeObjectURL(url);
  }, 1000);
}

export function downloadText(text, filename, mime) {
  downloadBlob(new Blob([text], { type: mime || 'text/plain;charset=utf-8' }), filename);
}
