/* Session lifecycle + polling: status/frames loops, center image,
 * mode tabs, session visibility, checkpoints, upload/probe, video/live
 * start/stop. Rendering itself lives in panels/overlay/timeline.
 */

import { state, GRID_COLS, GRID_ROWS } from '../state/store.js';
import { els, img } from '../utils/elements.js';
import { showError, setPill } from '../utils/dom.js';
import { num, basename } from '../utils/format.js';
import { apiGet, apiPost } from '../api/client.js';
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
  pollStatus();
}

export function bindToggle(id, key) {
  els[id].addEventListener('click', () => {
    state.toggles[key] = !state.toggles[key];
    els[id].classList.toggle('is-on', state.toggles[key]);
    els[id].setAttribute('aria-pressed', String(state.toggles[key]));
    renderCenter(currentFrame());
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
    scheduleFrames();
  });

  bindToggle('toggle-boxes', 'boxes');
  bindToggle('toggle-grid', 'grid');
  bindToggle('toggle-labels', 'labels');
}
