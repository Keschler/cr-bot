/* Session lifecycle + polling: status/frames loops, center image,
 * mode tabs, session visibility, checkpoints, upload/probe, video/live
 * start/stop. Rendering itself lives in panels/overlay/timeline.
 */

import { state, GRID_COLS, GRID_ROWS, loadPersistedSettings, persistSettings } from '../state/store.js';
import { els, img } from '../utils/elements.js';
import { showError, setPill, esc } from '../utils/dom.js';
import { num, basename } from '../utils/format.js';
import { apiGet, apiPost, apiDelete } from '../api/client.js';
import { roiAdaptAvailableFromDims, buildVideoStartPayload } from './roi.js';
import { clearRoiAdapt, showRoiAdaptAvailable, roiPreviewVisible, syncRoiPreviewClass, drawRoiPreviewBoxes } from './roi-editor.js';
import { visualStateOf, suggestionsOf, diagnosticsOf, currentFrame, isLiveEdge } from './frames.js';
import { drawOverlay, renderCenter } from './overlay.js';
import { renderCurrent, seek } from './timeline.js';
import { hideFloatbar, setCorrectionStatus } from './corrections.js';

/* ---------- status + frames polling ---------- */

export async function pollStatus() {
  try {
    const s = await apiGet('/api/status');
    if (s && s.error) {
      setPill('Error: ' + String(s.error).slice(0, 80), 'is-error', String(s.error));
      showError(String(s.error));
      return;
    }
    showError('');
    if (s && s.running) {
      const mode = s.mode || state.mode;
      const n = s.frame_count !== undefined && s.frame_count !== null ? s.frame_count : state.history.length;
      setPill('● ' + mode + ' · ' + n + ' frames', 'is-running', JSON.stringify(s.summary || ''));
    } else {
      setPill(state.history.length ? 'Stopped · ' + state.history.length + ' frames' : 'No session', 'is-idle');
    }
    applySessionVisibility(s);
  } catch (err) {
    setPill('Backend unreachable', 'is-error', String(err && err.message || err));
  }
}

export function frameDelayMs() {
  // Polling interval is fixed: playback speed steps the cursor through
  // buffered frames (see playAcc below), it must not change network rate.
  return 1000;
}

let playAcc = 0;
let lastLiveRefresh = 0;
let pendingResumeFrame = null;
let lastLibrarySave = 0;


export function scheduleFrames() {
  if (state.framesTimer) clearTimeout(state.framesTimer);
  state.framesTimer = setTimeout(async () => {
    // Always poll so a paused timeline stays fresh; pollFrames only follows
    // the live edge when playing (follow mode), otherwise the cursor stays
    // where the user scrubbed and playback stepping advances it below.
    await pollFrames();
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
  }, frameDelayMs());
}

export async function pollFrames() {
  try {
    const data = await apiGet('/api/frames?since=' + encodeURIComponent(state.lastSince) + '&limit=50');
    const frames = data && Array.isArray(data.frames) ? data.frames : [];
    if (frames.length) {
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
        }
      }
    }
    refreshImage();
  } catch (err) {
    setPill('Frames error: ' + String(err && err.message || err).slice(0, 80), 'is-error');
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
      img.src = url;
    }
    return;
  }
  syncRoiPreviewClass(false);
  delete img.dataset.preview;
  delete img.dataset.previewFrame;
  // The session preview always tracks the cursor frame (history frames have
  // their own served JPEGs), so panels, image, and overlays stay aligned
  // while scrubbing. The badge only flags that you are off the live edge.
  const frame = currentFrame();
  const atEdge = isLiveEdge();
  els['history-badge'].hidden = !(frame && !atEdge);
  const now = Date.now();
  if (!frame) {
    // No frames yet: keep showing the live stream while playing, throttled
    // to one request per 2s so an idle page doesn't hammer /api/frame/latest.
    if (state.playing && now - lastLiveRefresh > 2000) {
      lastLiveRefresh = now;
      const liveUrl = '/api/frame/latest?t=' + now;
      img.dataset.src = liveUrl;
      delete img.dataset.fi;
      img.src = liveUrl;
    }
    return;
  }
  if (atEdge) {
    // Live edge reloads when the index advances, or at most every 1.5s while
    // staying on one index (content may advance under the same index).
    const fi = String(frame.frame_index);
    if (img.dataset.fi !== fi || now - lastLiveRefresh > 1500) {
      lastLiveRefresh = now;
      const url = frameImageUrl(frame, true);
      img.dataset.src = url;
      img.dataset.fi = fi;
      img.src = url;
    }
    return;
  }
  const url = frameImageUrl(frame, false);
  if (img.dataset.src !== url) {
    img.dataset.src = url;
    img.dataset.fi = String(frame.frame_index);
    img.src = url;
  }
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
  state.history = [];
  state.cursor = -1;
  state.lastSince = -1;
  playAcc = 0;
  lastLiveRefresh = 0;
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

export async function uploadVideoFile(file) {
  const form = new FormData();
  form.append('file', file, file.name);
  const res = await fetch('/api/upload', { method: 'POST', body: form });
  if (!res.ok) {
    let detail = '';
    try { detail = await res.text(); } catch (e) { /* ignore */ }
    throw new Error('Upload → HTTP ' + res.status + (detail ? ' ' + detail : ''));
  }
  return res.json();
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
  setVideoFileMeta('Uploading…');
  try {
    const out = await uploadVideoFile(file);
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
      setVideoFileMeta('Uploading…');
      state.uploadedVideoFrames = 0;
      const out = await uploadVideoFile(pending);
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
    scheduleFrames();
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
  if (execute && !els['input-calibration'].value.trim()) {
    showError('Live execution requires a calibration artifact path.');
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
    scheduleFrames();
    pollStatus();
    pollFrames();
  } catch (err) {
    setPill('Start failed', 'is-error', String(err && err.message || err));
    showError(String(err && err.message || err));
  }
}

export async function stopSession() {
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
