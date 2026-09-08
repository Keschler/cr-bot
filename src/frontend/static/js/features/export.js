/* Export & share: session JSON / actions CSV downloads, frame JPEG save,
 * deep-link copy. Pure builders live in ../utils/share.js (no DOM).
 */

import { state } from '../state/store.js';
import { els, img } from '../utils/elements.js';
import { showError, toast } from '../utils/dom.js';
import { basename } from '../utils/format.js';
import {
  buildActionsCsv, buildSessionExport, downloadBlob, downloadText, encodeHash,
} from '../utils/share.js';
import { currentFrame } from './frames.js';
import { collectActionEntries } from './timeline.js';
import { currentVideoParams } from './session.js';

export function frameByFi(frameIndex) {
  if (frameIndex === undefined || frameIndex === null) return null;
  const want = String(frameIndex);
  for (const f of state.history) {
    if (f && String(f.frame_index) === want) return f;
  }
  return null;
}

function sessionFileStem() {
  const base = state.sessionLabel || state.uploadedVideoName || 'session';
  return basename(base).replace(/\.[A-Za-z0-9]+$/, '') || 'session';
}

export function exportSessionJson() {
  if (!state.history.length) {
    showError('Nothing to export — start a session first.');
    return;
  }
  const frame = currentFrame();
  const payload = buildSessionExport({
    label: state.sessionLabel,
    videoPath: state.uploadedVideoPath,
    videoName: state.uploadedVideoName,
    params: state.mode === 'video' ? currentVideoParams() : { mode: state.mode },
    cursorFrame: frame ? frame.frame_index : null,
    frames: state.history,
  });
  downloadText(JSON.stringify(payload, null, 2), sessionFileStem() + '-session.json', 'application/json');
  toast('Session exported: ' + state.history.length + ' frames', 'success');
}

export function exportActionsCsv() {
  const entries = collectActionEntries(state.history);
  if (!entries.length) {
    showError('No confirmed plays to export on this session.');
    return;
  }
  const rows = entries.map((e) => {
    const fr = frameByFi(e.fi);
    return {
      side: e.side, card: e.card, fi: e.fi, t: e.t,
      ts: fr ? fr.timestamp_s : null,
    };
  });
  downloadText(buildActionsCsv(rows), sessionFileStem() + '-actions.csv', 'text/csv;charset=utf-8');
  toast('Actions exported: ' + entries.length + ' plays', 'success');
}

export async function saveFrameJpeg() {
  const frame = currentFrame();
  const src = img && !img.hidden && img.currentSrc ? img.currentSrc : (img ? img.src : '');
  if (!frame || !src) {
    showError('No frame image to save — start a session first.');
    return;
  }
  try {
    const res = await fetch(src, { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const blob = await res.blob();
    downloadBlob(blob, sessionFileStem() + '-f' + frame.frame_index + '.jpg');
    toast('Frame f' + frame.frame_index + ' saved', 'success');
  } catch (err) {
    showError('Frame download failed: ' + String((err && err.message) || err));
  }
}

export function currentDeepLink() {
  const frame = currentFrame();
  return location.origin + location.pathname + encodeHash({
    frame: frame ? frame.frame_index : null,
    rank: state.selectedRank,
    video: state.uploadedVideoName || state.sessionLabel || null,
  });
}

export async function copyDeepLink() {
  if (!state.history.length) {
    showError('Nothing to link — start a session first.');
    return;
  }
  const url = currentDeepLink();
  try {
    await navigator.clipboard.writeText(url);
    showError('');
    toast('Link copied to clipboard', 'success');
    return;
  } catch (err) {
    // Clipboard API unavailable (permissions/insecure context): select-and-copy fallback.
    const ta = document.createElement('textarea');
    ta.value = url;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      if (!document.execCommand('copy')) throw new Error('copy rejected');
    } catch (fallbackErr) {
      document.body.removeChild(ta);
      showError('Copy failed — the link is in the address bar.');
      return;
    }
    document.body.removeChild(ta);
  }
  showError('');
  toast('Link copied to clipboard', 'success');
}

export function bindExportEvents() {
  if (els['btn-export-json']) els['btn-export-json'].addEventListener('click', exportSessionJson);
  if (els['btn-export-csv']) els['btn-export-csv'].addEventListener('click', exportActionsCsv);
  if (els['btn-save-frame']) els['btn-save-frame'].addEventListener('click', saveFrameJpeg);
  if (els['btn-copy-link']) els['btn-copy-link'].addEventListener('click', copyDeepLink);
}
