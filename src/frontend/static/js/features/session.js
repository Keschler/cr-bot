/* Session lifecycle + polling: status/frames loops, center image,
 * mode tabs, session visibility, checkpoints, upload/probe, video/live
 * start/stop. Rendering itself lives in panels/overlay/timeline.
 */

import { state, GRID_COLS, GRID_ROWS, loadPersistedSettings, persistSettings } from '../state/store.js';
import { els, img } from '../utils/elements.js';
import { showError, setPill, esc, toast } from '../utils/dom.js';
import { num, basename } from '../utils/format.js';
import { apiGet, apiPost, apiDelete } from '../api/client.js';
import { roiAdaptAvailableFromDims, buildVideoStartPayload } from './roi.js';
import { clearRoiAdapt, showRoiAdaptAvailable, roiPreviewVisible, syncRoiPreviewClass, drawRoiPreviewBoxes } from './roi-editor.js';
import { visualStateOf, suggestionsOf, diagnosticsOf, currentFrame, isLiveEdge } from './frames.js';
import { drawOverlay, renderCenter } from './overlay.js';
import { renderCurrent, seek, pausePlayback } from './timeline.js';
import { hideFloatbar, setCorrectionStatus } from './corrections.js';

/* ---------- status + frames polling (with reconnect backoff) ---------- */

let statusFails = 0;
let statusTimer = null;
let statusDownToast = null;

export async function pollStatus() {
  try {
    const s = await apiGet('/api/status');
    if (statusFails > 0 && statusDownToast) {
      statusDownToast.dismiss();
      statusDownToast = null;
      toast('Backend reconnected', 'success');
    }
    statusFails = 0;
    if (s && typeof s.summary === 'object' && s.summary !== null) {
      state.sessionSummary = s.summary; // devices/timing inspector source
    }
    if (s && s.error) {
      setPill('Error: ' + String(s.error).slice(0, 80), 'is-error', String(s.error));
      showError(String(s.error));
      return;
    }
    showError('');
    if (!s.running) closeStream(); // no producer: don't hold a dead stream
    if (s && s.running) {
      const mode = s.mode || state.mode;
      const n = s.frame_count !== undefined && s.frame_count !== null ? s.frame_count : state.history.length;
      setPill('● ' + mode + ' · ' + n + ' frames', 'is-running', JSON.stringify(s.summary || ''));
    } else {
      setPill(state.history.length ? 'Stopped · ' + state.history.length + ' frames' : 'No session', 'is-idle');
    }
    applySessionVisibility(s);
  } catch (err) {
    statusFails++;
    setPill('Backend unreachable — retrying… (click to retry)', 'is-error', String(err && err.message || err));
    if (!statusDownToast) {
      statusDownToast = toast('Backend unreachable — retrying…', 'error', {
        timeoutMs: 0,
        action: {
          label: 'Retry now',
          onClick: () => {
            statusDownToast = null;
            pollStatus();
            scheduleStatus();
          },
        },
      });
    }
  }
}

export function scheduleStatus() {
  // Fixed 2s cadence while healthy, exponential backoff (max 30s) while the
  // backend is unreachable. Clicking the pill retries immediately.
  if (statusTimer) clearTimeout(statusTimer);
  const step = async () => {
    await pollStatus();
    scheduleStatus();
  };
  const delay = statusFails > 0
    ? Math.min(30000, 2000 * Math.pow(2, Math.min(statusFails, 4)))
    : 2000;
  statusTimer = setTimeout(step, delay);
}

export function frameDelayMs() {
  // Polling interval is fixed: playback speed steps the cursor through
  // buffered frames (see playAcc below), it must not change network rate.
  return 1000;
}

let playAcc = 0;
let lastLiveRefresh = 0;
let pendingResumeFrame = null;
let pendingLink = null;
let lastLibrarySave = 0;
let framesFails = 0;

export function setPendingDeepLink(link) {
  pendingLink = link && typeof link === 'object' ? link : null;
}

export function scheduleFrames() {
  if (state.framesTimer) clearTimeout(state.framesTimer);
  state.framesTimer = setTimeout(async () => {
    // The SSE stream carries increments with ~0.25s latency when healthy;
    // polling stays as fallback (and re-syncs a silently stalled stream).
    // Either way the cursor stepping below advances playback at speed.
    if (!sseActive() || Date.now() - sseLastMsg > 10000) await pollFrames();
    if (state.playing && state.history.length) {
      if (state.cursor < state.history.length - 1) {
        // Step through buffered frames at the selected speed (1 frame per
        // tick at 1x; fractional speeds accumulate).
        playAcc += state.speed;
        const step = Math.floor(playAcc);
        if (step > 0) {
          playAcc -= step;
          state.cursor = Math.min(state.history.length - 1, state.cursor + step);
          refreshImage();
          renderCurrent();
        }
      } else {
        playAcc = 0;
      }
    } else {
      playAcc = 0;
    }
    scheduleFrames();
    // Back off while frame fetches fail (server down/restarting); the
    // status loop reports and recovers separately.
  }, Math.max(frameDelayMs(), framesFails > 0
    ? Math.min(15000, 1000 * Math.pow(2, Math.min(framesFails, 4)))
    : 0));
}

export async function pollFrames() {
  try {
    const data = await apiGet('/api/frames?since=' + encodeURIComponent(state.lastSince) + '&limit=50');
    const frames = data && Array.isArray(data.frames) ? data.frames : [];
    ingestFrames(frames);
    refreshImage();
    framesFails = 0;
  } catch (err) {
    framesFails++;
    setPill('Frames error: ' + String(err && err.message || err).slice(0, 80), 'is-error');
  }
}

// Shared ingest for polling and the SSE stream: dedup, sort, cap, follow,
// resume/link jumps. Idempotent per frame_index, so stream redeliveries and
// poll overlap are harmless.
export function ingestFrames(frames) {
  if (!Array.isArray(frames) || !frames.length) return;
  // Follow mode: only snap to the new edge when the cursor was already
  // at the edge (or empty). A scrubbed-back cursor must not be yanked
  // away by every poll; stepping in scheduleFrames advances it instead.
  const wasAtEdge = state.cursor < 0 || state.cursor >= state.history.length - 1;
  const seen = new Set(state.history.map((f) => f.frame_index));
  for (const f of frames) {
    if (f === null || f === undefined || f.frame_index === undefined) continue;
    if (seen.has(f.frame_index)) continue;
    seen.add(f.frame_index);
    state.history.push(f);
  }
  state.history.sort((a, b) => a.frame_index - b.frame_index);
  if (state.history.length > 2000) {
    const removed = state.history.length - 2000;
    state.history.splice(0, removed);
    // The splice shifts every retained index: a paused cursor must shift
    // with it (and clamp), or currentFrame() goes out of bounds and the
    // panels blank until the next play-follow.
    state.cursor = Math.max(0, Math.min(state.history.length - 1, state.cursor - removed));
  }
  state.lastSince = state.history[state.history.length - 1].frame_index;
  if (state.cursor < 0 || (state.playing && wasAtEdge)) {
    state.cursor = state.history.length - 1;
  }
  renderCurrent();
  // A resumed replay jumps back to its saved cursor once the
  // re-analysis has caught up (seek pauses, so the jump sticks).
      if (pendingResumeFrame !== null && pendingResumeFrame !== undefined) {
        const last = state.history[state.history.length - 1];
        if (last && last.frame_index >= pendingResumeFrame) {
          const target = state.history.findIndex((f) => f.frame_index >= pendingResumeFrame);
          pendingResumeFrame = null;
          seek(target >= 0 ? target : state.history.length - 1);
          // The jump is for inspection: stay paused even if it lands on the
          // edge (seek-to-edge otherwise resumes live follow).
          pausePlayback();
        }
      }
  // A pasted deep link (#f=&r=&v=) jumps once the frames arrive. A link
  // naming a different replay than the running one is dropped instead of
  // yanking the cursor somewhere surprising.
  if (pendingLink && state.history.length) {
    const last = state.history[state.history.length - 1];
    const linkVideo = pendingLink.video || null;
    const curVideo = state.uploadedVideoName || state.sessionLabel || '';
    if (linkVideo && curVideo && linkVideo !== curVideo) {
      pendingLink = null;
    } else if (pendingLink.frame === null || last.frame_index >= pendingLink.frame) {
      if (pendingLink.rank !== null && pendingLink.rank !== undefined) {
        state.selectedRank = pendingLink.rank;
      }
          const want = pendingLink.frame;
          pendingLink = null;
          if (want === null) {
            renderCurrent();
          } else {
            const target = state.history.findIndex((f) => f.frame_index >= want);
            seek(target >= 0 ? target : state.history.length - 1);
            // Same as resume jumps: a pasted link is for inspection, so
            // stay paused even when it lands on the edge.
            pausePlayback();
          }
    }
  }
}

/* ---------- SSE stream (primary) with polling fallback ---------- */

let sseSource = null;
let sseHealthy = false;
let sseFailed = false;
let sseLastMsg = 0;

export function sseActive() {
  return !!(sseSource && sseHealthy);
}

export function startStream() {
  // One stream per session: a reconnecting EventSource redelivers from the
  // connect-time cursor and ingestFrames() dedups, so no frames are lost.
  // The endpoint defaults since<=0 to the latest frame; pass the live
  // cursor so a reload mid-session only streams what polling missed.
  if (sseSource || sseFailed || typeof EventSource === 'undefined') return;
  let url = null;
  try {
    url = '/api/stream?since=' + encodeURIComponent(Math.max(0, state.lastSince));
    const source = new EventSource(url);
    sseSource = source;
    source.onmessage = (ev) => {
      let frame = null;
      try { frame = JSON.parse(ev.data); } catch (e) { return; }
      sseHealthy = true;
      sseLastMsg = Date.now();
      ingestFrames([frame]);
      refreshImage();
    };
    source.onerror = () => {
      // Quiet fallback: the polling loop keeps working and retries the
      // stream on the next session start.
      try { source.close(); } catch (e) { /* ignore */ }
      if (sseSource === source) sseSource = null;
      sseHealthy = false;
      sseFailed = true;
    };
  } catch (err) {
    sseFailed = true;
  }
}

export function closeStream() {
  const source = sseSource;
  sseSource = null;
  sseHealthy = false;
  if (source) {
    try { source.close(); } catch (e) { /* ignore */ }
  }
}

export function frameImageUrl(frame, atEdge) {
  // Immutable per-frame URLs are browser-cacheable; the live edge carries a
  // timestamp bust because its content can advance under the same frame
  // index (a constant `?t=<index>` would be served from cache, stalling live).
  if (!frame) return null;
  return atEdge
    ? '/api/frame/latest?fi=' + encodeURIComponent(frame.frame_index) + '&t=' + Date.now()
    : '/api/frame/' + encodeURIComponent(frame.frame_index);
}

export function refreshImage() {
  // A pre-session ROI preview takes over the middle player: the proposed
  // ROIs are reviewed (and adjusted) on the probe frame before Start.
  if (roiPreviewVisible()) {
    syncRoiPreviewClass(true);
    const url = state.roiAdapt.image;
    // Reload when the marker is absent (first show, polls) or when a
    // re-probe picked a different frame — otherwise the photo goes stale
    // while the boxes update underneath it.
    const tag = String(state.roiAdapt.probeFrame);
    if (url && (img.dataset.preview !== 'roi' || img.dataset.previewFrame !== tag)) {
      img.dataset.preview = 'roi';
      img.dataset.previewFrame = tag;
      img.dataset.src = 'roi-preview';
      delete img.dataset.fi;
      imgRequestId++; // cancel any in-flight session image fetch
      img.src = url;
    }
    return;
  }
  syncRoiPreviewClass(false);
  delete img.dataset.preview;
  delete img.dataset.previewFrame;
  // The session preview always tracks the cursor frame (history frames have
  // their own served JPEGs), so panels, image, and overlays stay aligned
  // while scrubbing. Images load via fetch so evicted (HTTP 204) frames are
  // detected explicitly instead of stalling silently on a bare <img> src.
  refreshEvictBadge();
  const frame = currentFrame();
  const atEdge = isLiveEdge();
  const now = Date.now();
  if (!frame) {
    // No frames yet: keep showing the live stream while playing, throttled
    // to one request per 2s so an idle page doesn't hammer /api/frame/latest.
    if (state.playing && now - lastLiveRefresh > 2000) {
      lastLiveRefresh = now;
      loadFrameImage('/api/frame/latest?t=' + now, null, true);
    }
    return;
  }
  const fi = String(frame.frame_index);
  if (atEdge) {
    // Live edge reloads when the index advances, or at most every 1.5s while
    // staying on one index (content may advance under the same index).
    if (img.dataset.fi !== fi || now - lastLiveRefresh > 1500) {
      lastLiveRefresh = now;
      loadFrameImage(frameImageUrl(frame, true), fi, true);
    }
    return;
  }
  const url = frameImageUrl(frame, false);
  if (img.dataset.src !== url) {
    img.dataset.src = url;
    loadFrameImage(url, fi, false);
  }
}

// Frames evicted from the bounded server history answer HTTP 204: no image
// arrives, so the UI keeps the last pixels and says so instead of looking
// frozen. Overlays stay withheld via imageMatchesFrame().
const evictedFrames = new Set();
let lastEvictToastFi = null;
let imgRequestId = 0;
let lastObjectUrl = null;

export function refreshEvictBadge() {
  const badge = els['history-badge'];
  if (!badge) return;
  const frame = currentFrame();
  if (frame && !isLiveEdge() && evictedFrames.has(String(frame.frame_index))) {
    badge.hidden = false;
    badge.textContent = 'frame evicted — showing last image';
    badge.title = 'This frame left the bounded server history; overlays withheld.';
  } else {
    badge.textContent = 'viewing history';
    badge.title = '';
    badge.hidden = !(frame && !isLiveEdge());
  }
}

async function loadFrameImage(url, fi, edge) {
  const id = ++imgRequestId;
  let res = null;
  try {
    res = await fetch(url, { cache: 'no-store' });
  } catch (err) {
    return; // network blip: keep last image; the status loop reports outages
  }
  if (id !== imgRequestId) return; // superseded by a newer scrub/poll
  if (!res.ok || res.status === 204) {
    if (!edge && fi !== null) {
      evictedFrames.add(String(fi));
      if (lastEvictToastFi !== String(fi)) {
        lastEvictToastFi = String(fi);
        toast('Frame f' + fi + ' left the server history — showing last image.', 'error');
      }
    }
    refreshEvictBadge();
    return;
  }
  let blob = null;
  try {
    blob = await res.blob();
  } catch (err) {
    return;
  }
  if (id !== imgRequestId || !blob || !blob.size) return;
  const objUrl = URL.createObjectURL(blob);
  if (lastObjectUrl) {
    try { URL.revokeObjectURL(lastObjectUrl); } catch (e) { /* gone */ }
  }
  lastObjectUrl = objUrl;
  if (fi !== null) {
    img.dataset.fi = String(fi);
    evictedFrames.delete(String(fi));
  } else {
    delete img.dataset.fi;
  }
  img.src = objUrl;
  refreshEvictBadge();
}

export function imageMatchesFrame(frame) {
  // Frame-bound overlays (detections, markers, arrows) may only draw when
  // the pixels on screen belong to the cursor frame.
  if (!frame || img.hidden) return false;
  return String(img.dataset.fi || '') === String(frame.frame_index);
}

export function bindImageEvents() {
  img.addEventListener('load', () => {
    img.hidden = false;
    els['frame-empty'].style.display = 'none';
    // ROI review draws its own boxes, not session overlays.
    if (roiPreviewVisible() && img.dataset.preview === 'roi') {
      drawRoiPreviewBoxes();
      return;
    }
    // Redraw the overlay only if the loaded pixels still belong to the
    // cursor frame: a scrub during image load must not paint new boxes on
    // old pixels (or vice versa). The next refreshImage() targets the new
    // cursor frame instead.
    const frame = currentFrame();
    if (frame && String(img.dataset.fi || '') === String(frame.frame_index)) {
      drawOverlay(frame, suggestionsOf(frame), diagnosticsOf(frame));
    }
  });

  img.addEventListener('error', () => {
    if (!currentFrame()) {
      img.hidden = true;
      els['frame-empty'].style.display = '';
      return;
    }
    // Hard image failure on a history frame: treat like an eviction so the
    // badge explains the stale pixels instead of showing them silently.
    const frame = currentFrame();
    if (frame && !isLiveEdge()) {
      evictedFrames.add(String(frame.frame_index));
      refreshEvictBadge();
    }
  });
}

/* ---------- session controls ---------- */

export function setMode(mode) {
  state.mode = mode;
  const isVideo = mode === 'video';
  els['tab-video'].classList.toggle('is-active', isVideo);
  els['tab-live'].classList.toggle('is-active', !isVideo);
  els['tab-video'].setAttribute('aria-selected', String(isVideo));
  els['tab-live'].setAttribute('aria-selected', String(!isVideo));
  els['panel-video'].hidden = !isVideo;
  els['panel-live'].hidden = isVideo;
}

export function resetSession(label) {
  closeStream();
  state.history = [];
  state.cursor = -1;
  state.lastSince = -1;
  playAcc = 0;
  lastLiveRefresh = 0;
  framesFails = 0;
  evictedFrames.clear();
  lastEvictToastFi = null;
  imgRequestId++; // cancel in-flight image fetches from the old session
  state.sessionLabel = label || '';
  state.selectedRank = null;
  state.sessionPin = null;
  state.showAllDetections = false;
  state.correctionDrafts = {};
  state.addSeq = 1;
  state.editMode = false;
  state.selection = null;
  state.dragBox = null;
  state.dragMove = null;
  hideFloatbar();
  const editBtn = els['btn-edit-labels'];
  if (editBtn) {
    editBtn.classList.remove('is-on');
    editBtn.setAttribute('aria-pressed', 'false');
  }
  const wrap = els['frame-wrap'];
  if (wrap) wrap.classList.remove('editing');
  setCorrectionStatus('');
  applySessionVisibility(null);
  renderCurrent();
}

export function applySessionVisibility(s) {
  // Auto-collapse the session setup once analysis is running with frames;
  // expand again when idle. A manual toggle pins the state until next start.
  const running = !!(s && s.running);
  const n = s && s.frame_count !== undefined && s.frame_count !== null
    ? s.frame_count : state.history.length;
  const collapsed = state.sessionPin !== null ? state.sessionPin : (running && n > 0);
  const card = document.querySelector('.card-session');
  if (card) card.classList.toggle('collapsed', !!collapsed);
  const tgl = els['btn-session-toggle'];
  if (tgl) {
    tgl.textContent = collapsed ? '+' : '–';
    tgl.setAttribute('aria-expanded', String(!collapsed));
  }
  const mini = els['session-mini'];
  if (mini) {
    if (collapsed) {
      mini.hidden = false;
      mini.textContent = (s && s.mode ? s.mode : state.mode) + ' · ' + n + ' frames';
    } else {
      mini.hidden = true;
    }
  }
  const stopMini = els['btn-stop-mini'];
  if (stopMini) stopMini.hidden = !(collapsed && running);
  // Prevent double-starting while a session runs; Stop stays available.
  if (els['btn-start-video']) els['btn-start-video'].disabled = running;
  if (els['btn-start-live']) els['btn-start-live'].disabled = running;
}

/* ---------- checkpoints + upload ---------- */

export function fillCheckpointSelect(selectEl, data) {
  if (!selectEl) return;
  selectEl.textContent = '';
  const items = (data && Array.isArray(data.checkpoints)) ? data.checkpoints : [];
  if (!items.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'prototype.pt (default)';
    selectEl.appendChild(opt);
    return;
  }
  for (const item of items) {
    const opt = document.createElement('option');
    opt.value = item.path || '';
    opt.textContent = item.name || item.path || '';
    if (item.default) opt.selected = true;
    selectEl.appendChild(opt);
  }
  // Restore the persisted checkpoint when it still exists server-side.
  try {
    const saved = loadPersistedSettings();
    const want = selectEl === els['select-checkpoint']
      ? saved && saved.videoCheckpoint
      : saved && saved.liveCheckpoint;
    if (want !== undefined && want !== null) {
      const match = Array.from(selectEl.options).some((o) => o.value === want);
      if (match) selectEl.value = want;
    }
  } catch (e) { /* keep server default */ }
}

export function applyPersistedSettings() {
  let saved = null;
  try { saved = loadPersistedSettings(); } catch (e) { saved = null; }
  if (!saved) return;
  if (saved.toggles && typeof saved.toggles === 'object') {
    for (const [id, key] of [['toggle-boxes', 'boxes'], ['toggle-grid', 'grid'], ['toggle-labels', 'labels']]) {
      if (typeof saved.toggles[key] !== 'boolean' || !els[id]) continue;
      state.toggles[key] = saved.toggles[key];
      els[id].classList.toggle('is-on', state.toggles[key]);
      els[id].setAttribute('aria-pressed', String(state.toggles[key]));
    }
  }
  if (saved.speed !== undefined && saved.speed !== null && els['speed-select']) {
    const v = parseFloat(saved.speed);
    const match = Array.from(els['speed-select'].options).some((o) => o.value === String(saved.speed));
    if (Number.isFinite(v) && v > 0 && match) {
      state.speed = v;
      els['speed-select'].value = String(saved.speed);
    }
  }
  for (const [id, key] of [['select-device', 'device'], ['select-live-device', 'liveDevice'],
      ['select-transport', 'transport']]) {
    const el = els[id];
    if (!el || saved[key] === undefined || saved[key] === null) continue;
    const match = Array.from(el.options).some((o) => o.value === String(saved[key]));
    if (match) el.value = String(saved[key]);
  }
}

export async function loadGrid() {
  try {
    const data = await apiGet('/api/grid');
    if (data && num(data.cols) === GRID_COLS && num(data.rows) === GRID_ROWS &&
        [data.x0, data.y0, data.x1, data.y1].every((v) => num(v) !== null)) {
      state.gridSpec = data;
    }
    if (data && data.tower_hp && typeof data.tower_hp === 'object') {
      state.towerMax = data.tower_hp;
    }
  } catch (err) {
    state.gridSpec = null; // GRID_SPEC_FALLBACK mirrors the same constants
    state.towerMax = null; // KING/PRINCESS_TOWER_MAX fallback consts apply
  }
  renderCurrent();
}

export async function loadCheckpoints() {
  try {
    const data = await apiGet('/api/checkpoints');
    fillCheckpointSelect(els['select-checkpoint'], data);
    fillCheckpointSelect(els['select-live-checkpoint'], data);
  } catch (err) {
    fillCheckpointSelect(els['select-checkpoint'], null);
    fillCheckpointSelect(els['select-live-checkpoint'], null);
  }
}

export async function loadServerCapabilities() {
  // Disable the GPU option when the server environment has no CUDA, so a
  // start cannot fail with a 400 after the user filled the whole form.
  let cuda = true;
  try {
    const data = await apiGet('/api/health');
    if (data && typeof data.cuda_available === 'boolean') cuda = data.cuda_available;
  } catch (err) { /* keep enabled on probe failure */ }
  if (cuda) return;
  for (const id of ['select-device', 'select-live-device']) {
    const select = els[id];
    if (!select) continue;
    const gpu = select.querySelector('option[value="cuda"]');
    if (gpu) {
      gpu.disabled = true;
      gpu.textContent = 'GPU (CUDA — unavailable)';
    }
    if (select.value === 'cuda') select.value = 'auto';
  }
}

export function uploadVideoFile(file, onProgress) {
  // XHR (not fetch) so large uploads report progress. onProgress(frac) is
  // best-effort: servers without a Content-Length simply never call it.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload');
    xhr.responseType = 'json';
    if (xhr.upload && onProgress) {
      xhr.upload.addEventListener('progress', (ev) => {
        if (ev.lengthComputable && ev.total > 0) {
          try { onProgress(ev.loaded / ev.total); } catch (e) { /* ignore */ }
        }
      });
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const data = xhr.response !== undefined && xhr.response !== null ? xhr.response : {};
        resolve(typeof data === 'object' ? data : {});
        return;
      }
      // FastAPI error bodies arrive parsed as {detail}; responseText throws
      // under responseType=json, so every access is guarded.
      let detail = '';
      try {
        const r = xhr.response;
        if (r && typeof r === 'object' && r.detail) detail = String(r.detail);
        else if (typeof r === 'string' && r) detail = r;
        else detail = xhr.responseText || '';
      } catch (e) { /* ignore */ }
      reject(new Error('Upload → HTTP ' + xhr.status + (detail ? ' ' + detail : '')));
    };
    xhr.onerror = () => reject(new Error('Upload failed: network error'));
    xhr.onabort = () => reject(new Error('Upload cancelled'));
    const form = new FormData();
    form.append('file', file, file.name);
    try {
      xhr.send(form);
    } catch (err) {
      reject(err);
    }
  });
}

// Shared upload wrapper: determinate progress toast + file-meta percent,
// success toast on completion. Errors propagate to the caller as before.
export async function uploadWithProgress(file) {
  const handle = toast('Uploading ' + file.name, 'info', { progress: true, timeoutMs: 0 });
  setVideoFileMeta('Uploading… 0%');
  try {
    const out = await uploadVideoFile(file, (frac) => {
      handle.setProgress(frac);
      setVideoFileMeta('Uploading… ' + Math.round(frac * 100) + '%');
    });
    handle.dismiss();
    toast('Upload complete: ' + (file.name || 'video'), 'success');
    return out;
  } catch (err) {
    handle.dismiss();
    throw err;
  }
}

export function setVideoFileLabel(name) {
  const label = els['video-file-label'];
  if (name) {
    label.textContent = name;
    label.classList.remove('is-empty');
    label.title = name;
  } else {
    label.textContent = 'No file chosen';
    label.classList.add('is-empty');
    label.title = '';
  }
}

export function setVideoFileMeta(text, isError) {
  const meta = els['video-file-meta'];
  if (!text) {
    meta.hidden = true;
    meta.textContent = '';
    return;
  }
  meta.hidden = false;
  meta.textContent = text;
  meta.style.color = isError ? 'var(--red)' : '';
}

export async function probeUploadedVideo(out) {
  // Shared probe after every upload: frame count / duration for the
  // start-frame range, plus the ROI-format check so a non-native video never
  // starts on fixed ROIs silently.
  try {
    const info = await apiGet('/api/video/info?path=' + encodeURIComponent(state.uploadedVideoPath));
    state.uploadedVideoFrames = num(info.frames) || 0;
    const bits = [];
    if (info.frames) bits.push(info.frames + ' frames');
    if (info.duration_s !== undefined && info.duration_s !== null) bits.push(Number(info.duration_s).toFixed(1) + 's');
    if (info.fps) bits.push(Number(info.fps).toFixed(0) + 'fps');
    setVideoFileMeta(bits.join(' · ') || null);
    els['video-file-label'].title = state.uploadedVideoName + ' → ' + ((out && out.path) || '');
    if (roiAdaptAvailableFromDims(info.width, info.height)) {
      showRoiAdaptAvailable(info.width, info.height);
    } else {
      clearRoiAdapt();
    }
  } catch (probeErr) {
    setVideoFileMeta(null);
    clearRoiAdapt();
  }
}

export async function onVideoFileChange() {
  const input = els['input-video-file'];
  const file = input && input.files && input.files[0];
  if (!file) return;
  clearRoiAdapt();
  state.uploadedVideoFrames = 0;
  setVideoFileLabel(file.name);
  try {
    const out = await uploadWithProgress(file);
    state.uploadedVideoPath = out.path || '';
    state.uploadedVideoName = out.filename || file.name;
    showError('');
    setVideoFileLabel(state.uploadedVideoName);
    await probeUploadedVideo(out);
  } catch (err) {
    state.uploadedVideoPath = '';
    clearRoiAdapt();
    setVideoFileLabel(null);
    setVideoFileMeta('Upload failed — try again.', true);
    showError(String(err && err.message || err));
    // Clear the input so re-picking the same file fires a change event.
    if (input) input.value = '';
  }
}

export async function startVideo() {
  let videoPath = state.uploadedVideoPath;
  const input = els['input-video-file'];
  const pending = input && input.files && input.files[0];
  if (!videoPath && pending) {
    try {
      setVideoFileLabel(pending.name);
      state.uploadedVideoFrames = 0;
      const out = await uploadWithProgress(pending);
      videoPath = out.path || '';
      state.uploadedVideoPath = videoPath;
      state.uploadedVideoName = out.filename || pending.name;
      setVideoFileLabel(state.uploadedVideoName);
      // Same probe as the file-picker path: without it a non-native video
      // would start on fixed ROIs with no format warning.
      await probeUploadedVideo(out);
    } catch (err) {
      setVideoFileMeta('Upload failed — try again.', true);
      showError(String(err && err.message || err));
      if (input) input.value = '';
      return;
    }
  }
  if (!videoPath) {
    showError('Choose a video file first.');
    return;
  }
  const stride = Math.max(1, parseInt(els['input-stride'].value, 10) || 1);
  const maxFrames = Math.max(1, parseInt(els['input-max-frames'].value, 10) || 500);
  const startFrame = Math.max(0, parseInt(els['input-start-frame'].value, 10) || 0);
  const checkpoint = els['select-checkpoint'] ? els['select-checkpoint'].value : '';
  // Sync the ROI checkbox into state (single source for the payload builder).
  if (els['check-adapt-rois']) state.roiAdapt.checked = !!els['check-adapt-rois'].checked;
  // Adapt on without a reviewed proposal: the fetch may still be running
  // or have failed — Start refuses rather than silently using fixed ROIs.
  if (state.roiAdapt.checked && !state.roiAdapt.preview) {
    showError(state.roiAdapt.loading
      ? 'The ROI proposal is still loading — wait for the player preview, then press Start.'
      : 'Adapt is on but no ROI proposal is ready — wait for the preview, try Another frame, or uncheck Adapt ROIs.');
    return;
  }
  const payload = buildVideoStartPayload({
    videoPath, stride, startFrame, maxFrames, checkpoint, roiAdapt: state.roiAdapt,
    device: els['select-device'] ? els['select-device'].value : 'auto',
  });
  try {
    showError('');
    await apiPost('/api/video/start', payload);
    // The session takes over the middle player from here on.
    state.roiAdapt.showInMiddle = false;
    syncRoiPreviewClass(false);
    resetSession(basename(state.uploadedVideoName || videoPath));
    state.playing = true;
    els['btn-play'].textContent = 'Pause';
    sseFailed = false;
    scheduleFrames();
    startStream();
    pollStatus();
    pollFrames();
    saveRecentSession();
  } catch (err) {
    setPill('Start failed', 'is-error', String(err && err.message || err));
    showError(String(err && err.message || err));
  }
}

export async function startLive() {
  const serial = els['input-serial'].value.trim();
  if (!serial) {
    showError('Device serial is required.');
    return;
  }
  const execute = els['check-execute'].checked;
  const confirmLive = els['check-confirm-live'].checked;
  if (execute && !confirmLive) {
    showError('Execute mode requires the confirmation checkbox.');
    return;
  }
  try {
    showError('');
    await apiPost('/api/live/start', {
      serial,
      transport: els['select-transport'].value,
      checkpoint: els['select-live-checkpoint'] ? (els['select-live-checkpoint'].value || null) : null,
      device: els['select-live-device'] ? els['select-live-device'].value : 'auto',
      calibration: els['input-calibration'].value.trim(),
      execute,
      confirm_live: confirmLive,
    });
    resetSession(serial);
    state.playing = true;
    els['btn-play'].textContent = 'Pause';
    sseFailed = false;
    scheduleFrames();
    startStream();
    pollStatus();
    pollFrames();
  } catch (err) {
    setPill('Start failed', 'is-error', String(err && err.message || err));
    showError(String(err && err.message || err));
  }
}

export async function stopSession() {
  closeStream();
  try {
    await apiPost('/api/stop', {});
  } catch (err) {
    showError(String(err && err.message || err));
  }
  noteCursorMoved(true);
  pollStatus();
}

/* ---------- recent replays (session library) ---------- */

export function currentVideoParams() {
  return {
    frame_stride: Math.max(1, parseInt(els['input-stride'].value, 10) || 1),
    start_frame: Math.max(0, parseInt(els['input-start-frame'].value, 10) || 0),
    max_frames: Math.max(1, parseInt(els['input-max-frames'].value, 10) || 500),
    checkpoint: els['select-checkpoint'] ? els['select-checkpoint'].value : '',
    device: els['select-device'] ? els['select-device'].value : 'auto',
  };
}

export async function saveRecentSession() {
  // Video sessions only (v1): live serials are not resumable replays.
  if (!state.uploadedVideoPath) return;
  const frame = currentFrame();
  try {
    await apiPost('/api/sessions', {
      name: state.uploadedVideoName || basename(state.uploadedVideoPath),
      video_path: state.uploadedVideoPath,
      filename: state.uploadedVideoName || basename(state.uploadedVideoPath),
      params: currentVideoParams(),
      cursor_frame: frame ? frame.frame_index : null,
      frame_count: state.history.length,
    });
    loadRecentSessions();
  } catch (err) { /* library is best-effort; never break the session */ }
}

export function noteCursorMoved(force) {
  // Seeks fire rapidly while dragging: throttle library writes, but always
  // persist on Stop so the resumed cursor is fresh.
  const now = Date.now();
  if (!force && now - lastLibrarySave < 2000) return;
  lastLibrarySave = now;
  saveRecentSession();
}

export async function loadRecentSessions() {
  const list = els['recent-sessions'];
  if (!list) return;
  let entries = [];
  try {
    const data = await apiGet('/api/sessions');
    if (data && Array.isArray(data.sessions)) entries = data.sessions;
  } catch (err) { /* keep previous list on probe failure */ }
  list.innerHTML = '';
  if (!entries.length) {
    list.innerHTML = '<li class="empty">No saved replays yet</li>';
    return;
  }
  for (const e of entries.slice(0, 20)) {
    const li = document.createElement('li');
    li.className = 'recent-row';
    const sub = [];
    if (e.frame_count !== undefined && e.frame_count !== null) sub.push(e.frame_count + ' fr');
    if (e.cursor_frame !== undefined && e.cursor_frame !== null) sub.push('f' + e.cursor_frame);
    li.innerHTML = '<span class="recent-main"><strong>' + esc(e.filename || e.name || 'replay') + '</strong>' +
      '<span class="muted">' + esc(sub.join(' · ') || '—') + '</span></span>' +
      '<span class="recent-actions"><button type="button" class="mini" data-resume="' + esc(e.name || '') +
      '" title="Re-run this replay and jump back">Resume</button>' +
      '<button type="button" class="mini" data-del="' + esc(e.name || '') + '" title="Forget this replay">✕</button></span>';
    list.appendChild(li);
  }
}

export async function resumeSession(name) {
  let entries = [];
  try {
    const data = await apiGet('/api/sessions');
    if (data && Array.isArray(data.sessions)) entries = data.sessions;
  } catch (err) {
    showError(String(err && err.message || err));
    return;
  }
  const entry = entries.find((e) => e && e.name === name);
  if (!entry) {
    showError('Saved replay not found: ' + name);
    return;
  }
  if (entry.params && entry.params.adapt_rois) {
    showError('This replay used adapted ROIs — re-check Adapt ROIs and Start manually.');
    return;
  }
  setMode('video');
  state.uploadedVideoPath = entry.video_path || '';
  state.uploadedVideoName = entry.filename || entry.name || '';
  setVideoFileLabel(state.uploadedVideoName);
  setVideoFileMeta(null);
  const p = entry.params || {};
  if (p.frame_stride !== undefined) els['input-stride'].value = p.frame_stride;
  if (p.start_frame !== undefined) els['input-start-frame'].value = p.start_frame;
  if (p.max_frames !== undefined) els['input-max-frames'].value = p.max_frames;
  if (p.checkpoint !== undefined && els['select-checkpoint']) {
    const match = Array.from(els['select-checkpoint'].options).some((o) => o.value === p.checkpoint);
    if (match) els['select-checkpoint'].value = p.checkpoint;
  }
  if (p.device !== undefined && els['select-device']) {
    const match = Array.from(els['select-device'].options).some((o) => o.value === p.device);
    if (match) els['select-device'].value = p.device;
  }
  pendingResumeFrame = entry.cursor_frame !== undefined && entry.cursor_frame !== null
    ? entry.cursor_frame : null;
  await startVideo();
  if (!state.uploadedVideoPath) pendingResumeFrame = null;
}

export async function deleteRecentSession(name) {
  try {
    await apiDelete('/api/sessions/' + encodeURIComponent(name));
  } catch (err) {
    showError(String(err && err.message || err));
    return;
  }
  loadRecentSessions();
}

export function bindToggle(id, key) {
  els[id].addEventListener('click', () => {
    state.toggles[key] = !state.toggles[key];
    els[id].classList.toggle('is-on', state.toggles[key]);
    els[id].setAttribute('aria-pressed', String(state.toggles[key]));
    persistSettings({ toggles: Object.assign({}, state.toggles) });
    renderCenter(currentFrame());
  });
}

export function bindPersistedSelect(id, key) {
  const el = els[id];
  if (!el) return;
  el.addEventListener('change', () => {
    persistSettings({ [key]: el.value });
  });
}

export function bindSessionEvents() {
  els['tab-video'].addEventListener('click', () => setMode('video'));
  els['tab-live'].addEventListener('click', () => setMode('live'));
  // The status pill doubles as a retry button when polling is failing.
  if (els['status-pill']) {
    els['status-pill'].addEventListener('click', () => {
      pollStatus();
      scheduleStatus();
      pollFrames();
      scheduleFrames();
    });
  }
  els['btn-open-replay'].addEventListener('click', () => {
    setMode('video');
    els['btn-browse-video'].focus();
  });
  els['btn-dashboard'].addEventListener('click', () => {
    setMode('live');
    els['input-serial'].focus();
  });
  els['btn-start-video'].addEventListener('click', startVideo);
  els['btn-stop-video'].addEventListener('click', stopSession);
  els['btn-start-live'].addEventListener('click', startLive);
  els['btn-stop-live'].addEventListener('click', stopSession);
  els['btn-stop-mini'].addEventListener('click', stopSession);
  els['btn-browse-video'].addEventListener('click', () => els['input-video-file'].click());
  els['btn-session-toggle'].addEventListener('click', () => {
    const card = document.querySelector('.card-session');
    const collapsed = card ? card.classList.contains('collapsed') : false;
    state.sessionPin = !collapsed;
    applySessionVisibility(null);
  });
  els['input-video-file'].addEventListener('change', onVideoFileChange);
  if (els['recent-sessions']) {
    // Delegated: rows re-render on every library load.
    els['recent-sessions'].addEventListener('click', (ev) => {
      const resume = ev.target && ev.target.closest ? ev.target.closest('[data-resume]') : null;
      if (resume) {
        resumeSession(resume.dataset.resume);
        return;
      }
      const del = ev.target && ev.target.closest ? ev.target.closest('[data-del]') : null;
      if (del) deleteRecentSession(del.dataset.del);
    });
  }

  els['check-execute'].addEventListener('change', () => {
    els['live-warning'].hidden = !els['check-execute'].checked;
  });

  els['btn-play'].addEventListener('click', () => {
    state.playing = !state.playing;
    playAcc = 0;
    els['btn-play'].textContent = state.playing ? 'Pause' : 'Play';
    els['btn-play'].setAttribute('aria-pressed', String(state.playing));
    if (state.playing && isLiveEdge()) {
      state.cursor = state.history.length - 1;
      renderCurrent();
    }
    scheduleFrames();
  });
  els['btn-prev'].addEventListener('click', () => seek(state.cursor - 1));
  els['btn-next'].addEventListener('click', () => seek(state.cursor + 1));
  if (els['btn-go-live']) {
    // Seeking to the edge resumes live follow (see seek()).
    els['btn-go-live'].addEventListener('click', () => seek(state.history.length - 1));
  }
  els['speed-select'].addEventListener('change', () => {
    const v = parseFloat(els['speed-select'].value);
    state.speed = Number.isFinite(v) && v > 0 ? v : 1;
    persistSettings({ speed: els['speed-select'].value });
    scheduleFrames();
  });

  bindPersistedSelect('select-device', 'device');
  bindPersistedSelect('select-live-device', 'liveDevice');
  bindPersistedSelect('select-transport', 'transport');
  bindPersistedSelect('select-checkpoint', 'videoCheckpoint');
  bindPersistedSelect('select-live-checkpoint', 'liveCheckpoint');

  bindToggle('toggle-boxes', 'boxes');
  bindToggle('toggle-grid', 'grid');
  bindToggle('toggle-labels', 'labels');
}
