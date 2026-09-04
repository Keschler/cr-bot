'use strict';

/* Arena Replay Analyst v1.2.0 — vanilla frontend.
 * All data comes from the backend API (see ../README.md). No mocks. */

const GRID_COLS = 18;
const GRID_ROWS = 32;
// Authoritative grid geometry lives in cr_bot/features/action_space.py
// (ACTION_GRID over the KataCR 568x896 arena crop). The backend serves it at
// GET /api/grid; this fallback mirrors those exact constants so the overlay
// keeps working if that fetch fails.
const GRID_SPEC_FALLBACK = {
  cols: 18, rows: 32,
  x0: -0.9320463320463317 / 568, y0: 72.54622356495467 / 896,
  x1: 569.2610038610038 / 568, y1: 879.9748640483384 / 896,
};
// Extractor hand-slot ROIs (src/cr_bot/domain/rois.py) as (x, y, w, h) in
// the 1080x2400 reference space. The Inspect arrow starts at the played
// card's real position in the video frame.
const ROI_REF_W = 1080, ROI_REF_H = 2400;
const NATIVE_SIZE = [1080, 2400];
const HAND_SLOT_ROIS = [
  { x: 230, y: 2020, w: 220, h: 300 },
  { x: 430, y: 2020, w: 220, h: 300 },
  { x: 630, y: 2020, w: 220, h: 300 },
  { x: 840, y: 2020, w: 220, h: 300 },
];
const KING_TOWER_MAX = 7032;
const PRINCESS_TOWER_MAX = 4424;
// Full-HP reference values mirror cr_bot/domain/constants.py
// (PRINCESS_TOWER_HP / KING_TOWER_HP). GET /api/grid also serves them and
// takes precedence when available (see towerMax / loadGrid).

const state = {
  mode: 'video', // 'video' | 'live'
  playing: true,
  speed: 1,
  toggles: { boxes: true, grid: true, labels: true },
  history: [], // frames from GET /api/frames, ascending by frame_index
  cursor: -1, // index into history; -1 = empty
  lastSince: 0,
  sessionLabel: '',
  framesTimer: null,
  uploadedVideoPath: '',
  uploadedVideoName: '',
  selectedRank: null, // inspected suggestion rank (0-based into top-3) or null
  gridSpec: null, // fetched from GET /api/grid; fallback used when missing
  towerMax: null, // {princess, king} from GET /api/grid; fallback consts above
  sessionPin: null, // manual session collapse override: true/false or null (auto)
  roiAdapt: { available: false, checked: false, preview: null, accepted: false },
};

const $ = (id) => document.getElementById(id);

const els = {};
[
  'status-pill', 'error-bar',
  'tab-video', 'tab-live', 'btn-open-replay', 'btn-dashboard',
  'panel-video', 'panel-live',
  'btn-session-toggle', 'btn-stop-mini', 'session-mini',
  'input-video-file', 'btn-browse-video', 'video-file-label', 'video-file-meta', 'select-checkpoint',
  'input-start-frame', 'input-stride', 'input-max-frames',
  'btn-start-video', 'btn-stop-video',
  'check-adapt-rois', 'adapt-notice', 'btn-preview-rois', 'roi-preview',
  'roi-preview-meta', 'accept-rois-row', 'check-accept-rois',
  'input-serial', 'select-transport', 'select-live-checkpoint', 'input-calibration',
  'check-execute', 'check-confirm-live', 'live-warning',
  'btn-start-live', 'btn-stop-live',
  'meta-replay-id', 'meta-arena', 'meta-tick', 'meta-phase',
  'hand-slots', 'next-card', 'own-elixir-text', 'own-elixir-fill',
  'enemy-elixir-text', 'enemy-elixir-fill',
  'tower-rows', 'detected-objects', 'detection-count',
  'badge-ingame', 'badge-perf', 'badge-arena',
  'toggle-boxes', 'toggle-grid', 'toggle-labels',
  'frame-wrap', 'center-frame', 'frame-overlay', 'frame-empty', 'history-badge',
  'btn-play', 'btn-prev', 'btn-next', 'time-label', 'speed-select',
  'suggestions', 'reason-text', 'reason-entropy', 'reason-mode-probs',
  'reason-hand-stable', 'reason-confidence',
  'timeline-label', 'timeline-track', 'timeline-rail', 'timeline-fill', 'timeline-markers',
  'timeline-cursor', 'timeline-tooltip',
].forEach((id) => { els[id] = $(id); });

const img = els['center-frame'];
const canvas = els['frame-overlay'];
const ctx = canvas.getContext('2d');

/* ---------- helpers ---------- */

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmtInt(v) {
  const n = num(v);
  return n === null ? '—' : Math.round(n).toLocaleString('en-US');
}

function basename(path) {
  if (path === null || path === undefined) return '';
  const s = String(path);
  const parts = s.split(/[\\/]/);
  return parts[parts.length - 1] || s;
}

function cardIconName(value) {
  if (value === null || value === undefined) return null;
  const raw = Array.isArray(value) ? value[0]
    : (typeof value === 'object' ? value.name : value);
  if (raw === null || raw === undefined) return null;
  const s = String(raw).trim();
  return s && s !== '—' ? s : null;
}

function cardIconUrl(value) {
  const name = cardIconName(value);
  return name === null ? null : '/api/card-icon?name=' + encodeURIComponent(name);
}

function suggestionCardName(s, vs) {
  // The policy observation carries no hand names, so a suggestion's
  // card_name is often null. Fall back to the frame's hand at card_slot.
  const direct = s ? cardIconName(s.card_name) : null;
  if (direct) return direct;
  const slot = s ? num(s.card_slot) : null;
  if (slot !== null && vs) {
    const entries = handEntries(vs);
    if (slot >= 0 && slot < entries.length) {
      const h = cardIconName(entries[slot]);
      if (h) return h;
    }
  }
  return null;
}

function showError(msg) {
  if (!msg) {
    els['error-bar'].hidden = true;
    els['error-bar'].textContent = '';
    return;
  }
  els['error-bar'].hidden = false;
  els['error-bar'].textContent = msg;
}

function setPill(text, kind, title) {
  const pill = els['status-pill'];
  pill.textContent = text;
  pill.className = 'status-pill ' + (kind || 'is-idle');
  if (title) pill.title = title;
}

async function apiGet(path) {
  const res = await fetch(path, { cache: 'no-store' });
  if (!res.ok) throw new Error('GET ' + path + ' → HTTP ' + res.status);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = '';
    try { detail = await res.text(); } catch (e) { /* ignore */ }
    throw new Error('POST ' + path + ' → HTTP ' + res.status + (detail ? ' ' + detail : ''));
  }
  try { return await res.json(); } catch (e) { return {}; }
}

/* ---------- ROI adaptation (pure helpers, testable without DOM) ---------- */

function roiAdaptAvailableFromDims(width, height) {
  const w = num(width), h = num(height);
  if (w === null || h === null || w <= 0 || h <= 0) return false;
  return w !== NATIVE_SIZE[0] || h !== NATIVE_SIZE[1];
}

function roiAdaptNoticeText(width, height) {
  const w = Math.round(Number(width)), h = Math.round(Number(height));
  return w + 'x' + h + ' detected — fixed ROIs assume 1080x2400.';
}

function roiSetFromPreview(preview) {
  if (!preview || !Array.isArray(preview.rois)) return null;
  const out = {};
  for (const r of preview.rois) {
    if (!r || typeof r.name !== 'string' || !Array.isArray(r.rect)) continue;
    out[r.name] = r.rect;
  }
  return Object.keys(out).length ? out : null;
}

function buildVideoStartPayload(opts) {
  const o = opts || {};
  const ra = o.roiAdapt || { checked: false, accepted: false, preview: null };
  const checked = !!ra.checked;
  const accepted = !!ra.accepted;
  const preview = ra.preview || null;
  return {
    video_path: o.videoPath,
    frame_stride: o.stride,
    start_frame: o.startFrame,
    max_frames: o.maxFrames,
    checkpoint: o.checkpoint || null,
    adapt_rois: checked,
    roi_set: (checked && accepted && preview) ? roiSetFromPreview(preview) : null,
  };
}

function formatRoiPreviewMeta(data) {
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

/* ---------- ROI adaptation (DOM wiring) ---------- */

function adaptRowLabel() {
  const box = els['check-adapt-rois'];
  return box && box.closest ? box.closest('label') : null;
}

function clearRoiAdapt() {
  state.roiAdapt = { available: false, checked: false, preview: null, accepted: false };
  const row = adaptRowLabel();
  if (row) row.hidden = true;
  if (els['check-adapt-rois']) els['check-adapt-rois'].checked = false;
  if (els['adapt-notice']) { els['adapt-notice'].hidden = true; els['adapt-notice'].textContent = ''; }
  if (els['btn-preview-rois']) els['btn-preview-rois'].hidden = true;
  if (els['roi-preview']) { els['roi-preview'].hidden = true; els['roi-preview'].removeAttribute('src'); }
  if (els['roi-preview-meta']) { els['roi-preview-meta'].hidden = true; els['roi-preview-meta'].textContent = ''; }
  if (els['accept-rois-row']) els['accept-rois-row'].hidden = true;
  if (els['check-accept-rois']) els['check-accept-rois'].checked = false;
}

function showRoiAdaptAvailable(width, height) {
  state.roiAdapt.available = true;
  state.roiAdapt.checked = true;
  state.roiAdapt.preview = null;
  state.roiAdapt.accepted = false;
  const row = adaptRowLabel();
  if (row) row.hidden = false;
  if (els['check-adapt-rois']) els['check-adapt-rois'].checked = true;
  if (els['adapt-notice']) {
    els['adapt-notice'].hidden = false;
    els['adapt-notice'].textContent = roiAdaptNoticeText(width, height);
  }
  if (els['btn-preview-rois']) els['btn-preview-rois'].hidden = false;
  if (els['roi-preview']) { els['roi-preview'].hidden = true; els['roi-preview'].removeAttribute('src'); }
  if (els['roi-preview-meta']) { els['roi-preview-meta'].hidden = true; els['roi-preview-meta'].textContent = ''; }
  if (els['accept-rois-row']) els['accept-rois-row'].hidden = true;
  if (els['check-accept-rois']) els['check-accept-rois'].checked = false;
}

function renderRoiPreview(data) {
  const d = data || {};
  const rois = Array.isArray(d.rois) ? d.rois : [];
  state.roiAdapt.preview = { rois, native_size: d.native_size || null };
  state.roiAdapt.accepted = false;
  if (els['check-accept-rois']) els['check-accept-rois'].checked = false;
  if (els['roi-preview']) {
    if (d.image) {
      els['roi-preview'].src = d.image;
      els['roi-preview'].hidden = false;
    } else {
      els['roi-preview'].hidden = true;
      els['roi-preview'].removeAttribute('src');
    }
  }
  if (els['roi-preview-meta']) {
    els['roi-preview-meta'].hidden = false;
    els['roi-preview-meta'].textContent = formatRoiPreviewMeta(d);
  }
  if (els['accept-rois-row']) els['accept-rois-row'].hidden = false;
}

async function previewRois() {
  if (!state.uploadedVideoPath) {
    showError('Choose a video file first.');
    return;
  }
  try {
    const data = await apiGet('/api/roi-preview?path=' + encodeURIComponent(state.uploadedVideoPath));
    renderRoiPreview(data);
    showError('');
  } catch (err) {
    showError(String(err && err.message || err));
  }
}

/* Frame shape (per contract):
 * {frame_index, timestamp_s, in_game, emitted,
 *  record: {visual_state, action, result},
 *  suggestions: [{kind, card_slot, cell, probability, log_prob, card_name}],
 *  diagnostics} */
function visualStateOf(frame) {
  if (!frame) return {};
  if (frame.record && frame.record.visual_state) return frame.record.visual_state;
  return frame.visual_state || {};
}

function suggestionsOf(frame) {
  if (!frame) return [];
  if (Array.isArray(frame.suggestions)) return frame.suggestions;
  if (frame.record && Array.isArray(frame.record.suggestions)) return frame.record.suggestions;
  return [];
}

function diagnosticsOf(frame) {
  if (!frame) return {};
  if (frame.diagnostics && typeof frame.diagnostics === 'object') return frame.diagnostics;
  if (frame.record && frame.record.diagnostics) return frame.record.diagnostics;
  return {};
}

function actionOf(frame) {
  if (!frame) return null;
  if (frame.record && frame.record.action) return frame.record.action;
  return frame.action || null;
}

function currentFrame() {
  if (state.cursor < 0 || state.cursor >= state.history.length) return null;
  return state.history[state.cursor];
}

function isLiveEdge() {
  return state.cursor === state.history.length - 1;
}

/* ---------- status + frames polling ---------- */

async function pollStatus() {
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

function frameDelayMs() {
  return Math.round(1000 / state.speed);
}

function scheduleFrames() {
  if (state.framesTimer) clearTimeout(state.framesTimer);
  state.framesTimer = setTimeout(async () => {
    if (state.playing) await pollFrames();
    scheduleFrames();
  }, frameDelayMs());
}

async function pollFrames() {
  try {
    const data = await apiGet('/api/frames?since=' + encodeURIComponent(state.lastSince) + '&limit=50');
    const frames = data && Array.isArray(data.frames) ? data.frames : [];
    if (frames.length) {
      const seen = new Set(state.history.map((f) => f.frame_index));
      for (const f of frames) {
        if (f === null || f === undefined || f.frame_index === undefined) continue;
        if (seen.has(f.frame_index)) continue;
        seen.add(f.frame_index);
        state.history.push(f);
      }
      state.history.sort((a, b) => a.frame_index - b.frame_index);
      if (state.history.length > 2000) {
        state.history.splice(0, state.history.length - 2000);
      }
      state.lastSince = state.history[state.history.length - 1].frame_index;
      if (state.playing || state.cursor < 0) {
        state.cursor = state.history.length - 1;
      }
      renderCurrent();
    }
    refreshImage();
  } catch (err) {
    setPill('Frames error: ' + String(err && err.message || err).slice(0, 80), 'is-error');
  }
}

function frameImageUrl(frame, atEdge) {
  // Immutable per-frame URLs are browser-cacheable; the live edge carries a
  // bust parameter because its content advances under the same frame index.
  if (!frame) return null;
  return atEdge
    ? '/api/frame/latest?t=' + encodeURIComponent(frame.frame_index)
    : '/api/frame/' + encodeURIComponent(frame.frame_index);
}

function refreshImage() {
  // The preview always tracks the cursor frame now (history frames have
  // their own served JPEGs), so panels, image, and overlays stay aligned
  // while scrubbing. The badge only flags that you are off the live edge.
  const frame = currentFrame();
  const atEdge = isLiveEdge();
  els['history-badge'].hidden = !(frame && !atEdge);
  if (!frame) {
    // No frames yet: keep showing the live stream while playing.
    if (state.playing && img.dataset.src !== '/api/frame/latest') {
      img.dataset.src = '/api/frame/latest';
      delete img.dataset.fi;
      img.src = '/api/frame/latest';
    }
    return;
  }
  const url = frameImageUrl(frame, atEdge);
  if (img.dataset.src !== url) {
    img.dataset.src = url;
    img.dataset.fi = String(frame.frame_index);
    img.src = url;
  }
}

function imageMatchesFrame(frame) {
  // Frame-bound overlays (detections, markers, arrows) may only draw when
  // the pixels on screen belong to the cursor frame.
  if (!frame || img.hidden) return false;
  return String(img.dataset.fi || '') === String(frame.frame_index);
}

img.addEventListener('load', () => {
  img.hidden = false;
  els['frame-empty'].style.display = 'none';
  // Redraw the overlay for the current cursor frame: the image element
  // reloads on every poll at the live edge, and the overlay must survive it.
  const frame = currentFrame();
  if (frame) drawOverlay(frame, suggestionsOf(frame), diagnosticsOf(frame));
});

img.addEventListener('error', () => {
  if (!currentFrame()) {
    img.hidden = true;
    els['frame-empty'].style.display = '';
  }
});

/* ---------- left panel ---------- */

function phaseOf(vs, inGame) {
  if (!inGame) return 'Not in game';
  if (vs.overtime) return 'Overtime';
  const t = num(vs.time_left_s);
  if (t === null) return '—';
  return t <= 60 ? 'Double elixir' : 'Single elixir';
}

function handEntries(vs) {
  const hand = Array.isArray(vs.hand) ? vs.hand.slice(0, 4) : [];
  while (hand.length < 4) hand.push(null);
  return hand;
}

function handName(entry) {
  if (entry === null || entry === undefined) return '—';
  if (Array.isArray(entry)) return entry[0] === null || entry[0] === undefined ? '—' : String(entry[0]);
  if (typeof entry === 'object') return entry.name !== undefined ? String(entry.name) : '—';
  return String(entry);
}

function handCost(entry) {
  if (Array.isArray(entry) && entry.length > 1 && entry[1] !== null && entry[1] !== undefined) {
    return String(entry[1]);
  }
  if (entry && typeof entry === 'object' && entry.cost !== undefined) return String(entry.cost);
  return '—';
}

function towerMax(isKing) {
  const t = state.towerMax;
  if (t && typeof t === 'object') {
    const v = num(isKing ? t.king : t.princess);
    if (v !== null && v > 0) return v;
  }
  return isKing ? KING_TOWER_MAX : PRINCESS_TOWER_MAX;
}

function kingState(vs, side) {
  // Real extractor flag first (state_builder: king HP below max), then HP
  // derivation: a damaged king or a fallen princess tower means activated.
  const flag = side === 'self' ? vs.own_king_active : vs.enemy_king_active;
  if (flag === true) return 'active';
  const arr = side === 'self' ? vs.tower_hp_self : vs.tower_hp_enemy;
  const t = Array.isArray(arr) ? arr : [];
  const k = num(t[1]);
  if (k === null) return 'unknown';
  if (k <= 0) return 'destroyed';
  if (k < towerMax(true)) return 'active';
  const l = num(t[0]), r = num(t[2]);
  if ((l !== null && l <= 0) || (r !== null && r <= 0)) return 'active';
  return 'dormant';
}

function kingStatusLabel(st) {
  if (st === 'active') return 'Active';
  if (st === 'dormant') return 'Sleeping';
  if (st === 'destroyed') return 'Destroyed';
  return 'Unknown';
}

function kingStateClass(st) {
  if (st === 'active') return 'kact';
  if (st === 'dormant') return 'kdor';
  if (st === 'destroyed') return 'kdes';
  return 'kunk';
}

function towerPct(hp, isKing) {
  const n = num(hp);
  if (n === null) return null;
  const max = towerMax(isKing);
  return Math.max(0, Math.min(100, (n / max) * 100));
}

function renderLeft(frame) {
  const vs = visualStateOf(frame);
  const has = !!frame;

  els['meta-replay-id'].textContent = state.sessionLabel || (has ? '#' + frame.frame_index : 'No session');
  const diag = diagnosticsOf(frame);
  els['meta-arena'].textContent = diag.arena !== undefined && diag.arena !== null
    ? String(diag.arena) : (vs.arena !== undefined && vs.arena !== null ? String(vs.arena) : '—');
  els['meta-tick'].textContent = frame ? fmtInt(frame.frame_index) : '—';
  els['meta-phase'].textContent = has ? phaseOf(vs, !!frame.in_game) : '—';

  // Hand: highlight the inspected suggestion's slot, else the top pick.
  const sug = suggestionsOf(frame).slice().sort((a, b) => (num(b.probability) || 0) - (num(a.probability) || 0));
  const sel = state.selectedRank !== null ? sug[state.selectedRank] : null;
  const top = sug[0] || null;
  const selSlot = num((sel || top || {}).card_slot);
  const slots = els['hand-slots'].querySelectorAll('.hand-slot');
  const entries = handEntries(vs);
  slots.forEach((slotEl, i) => {
    const nameEl = slotEl.querySelector('.card-name');
    const costEl = slotEl.querySelector('.card-cost');
    const iconEl = slotEl.querySelector('.hand-icon');
    nameEl.textContent = has ? handName(entries[i]) : '—';
    costEl.textContent = has ? handCost(entries[i]) : '—';
    const iconUrl = has ? cardIconUrl(entries[i]) : null;
    if (iconEl) {
      if (iconUrl) {
        if (iconEl.dataset.src !== iconUrl) {
          iconEl.dataset.src = iconUrl;
          iconEl.src = iconUrl;
        }
        iconEl.hidden = false;
        iconEl.alt = handName(entries[i]);
      } else {
        iconEl.hidden = true;
        iconEl.removeAttribute('src');
        delete iconEl.dataset.src;
      }
    }
    slotEl.classList.toggle('is-selected', selSlot === i);
  });
  els['next-card'].textContent = vs.next_card !== undefined && vs.next_card !== null ? String(vs.next_card) : '—';
  const elixir = num(vs.elixir);
  els['own-elixir-text'].textContent = elixir === null ? '—' : elixir.toFixed(1) + ' / 10';
  els['own-elixir-fill'].style.width = elixir === null ? '0' : Math.max(0, Math.min(100, (elixir / 10) * 100)) + '%';
  const ee = num(vs.enemy_elixir_est);
  els['enemy-elixir-text'].textContent = ee === null ? '—' : ee.toFixed(1) + ' / 10';
  els['enemy-elixir-fill'].style.width = ee === null ? '0' : Math.max(0, Math.min(100, (ee / 10) * 100)) + '%';

  // Towers: Left / King / Right rows, each split into a YOU side (left)
  // and an OPPONENT side (right) with its own HP number and health bar.
  // King activation is a separate status line, never merged into HP.
  const box = els['tower-rows'];
  box.innerHTML = '';
  const selfT = Array.isArray(vs.tower_hp_self) ? vs.tower_hp_self : [];
  const enemyT = Array.isArray(vs.tower_hp_enemy) ? vs.tower_hp_enemy : [];
  if (!has || (!selfT.length && !enemyT.length)) {
    box.innerHTML = '<p class="empty">No session</p>';
  } else {
    const head = document.createElement('div');
    head.className = 'tower-colheads';
    head.innerHTML = '<span>You</span><span>Opponent</span>';
    box.appendChild(head);
    const names = ['Left', 'King', 'Right'];
    for (let i = 0; i < 3; i++) {
      const s = num(selfT[i]);
      const e = num(enemyT[i]);
      const isKing = i === 1;
      const max = towerMax(isKing);
      const pctS = s === null ? null : Math.max(0, Math.min(100, (s / max) * 100));
      const pctE = e === null ? null : Math.max(0, Math.min(100, (e / max) * 100));
      const row = document.createElement('div');
      row.className = 'tower-row';
      const barCls = (pct) => pct === null ? '' : pct > 50 ? '' : pct > 25 ? 'warn' : 'bad';
      let statusLine = '';
      if (isKing) {
        const stS = kingState(vs, 'self'), stE = kingState(vs, 'enemy');
        statusLine =
          '<div class="king-states"><span class="kstate">You · <b class="' + kingStateClass(stS) + '">' +
          esc(kingStatusLabel(stS)) + '</b></span>' +
          '<span class="kstate">Opponent · <b class="' + kingStateClass(stE) + '">' +
          esc(kingStatusLabel(stE)) + '</b></span></div>';
      }
      row.innerHTML =
        '<div class="tower-name">' + esc(names[i]) + '</div>' +
        '<div class="tower-sides">' +
        '<div class="tower-side you"><div class="tower-hp">' +
        (s === null ? '—' : esc(fmtInt(s))) + '</div>' +
        '<div class="bar"><div class="' + barCls(pctS) + '" style="width:' +
        (pctS === null ? 0 : pctS.toFixed(1)) + '%" title="you ' +
        (pctS === null ? '—' : pctS.toFixed(0) + '%') + '"></div></div></div>' +
        '<div class="tower-side opponent"><div class="tower-hp">' +
        (e === null ? '—' : esc(fmtInt(e))) + '</div>' +
        '<div class="bar"><div class="' + barCls(pctE) + '" style="width:' +
        (pctE === null ? 0 : pctE.toFixed(1)) + '%" title="opponent ' +
        (pctE === null ? '—' : pctE.toFixed(0) + '%') + '"></div></div></div>' +
        '</div>' + statusLine;
      box.appendChild(row);
    }
  }

  // Detected objects.
  const list = els['detected-objects'];
  list.innerHTML = '';
  const ally = Array.isArray(vs.ally_units) ? vs.ally_units : [];
  const enemy = Array.isArray(vs.enemy_units) ? vs.enemy_units : [];
  const detCount = vs.detection_count !== undefined && vs.detection_count !== null
    ? Number(vs.detection_count) : ally.length + enemy.length;
  els['detection-count'].textContent = has ? '(' + detCount + ')' : '';
  const all = ally.map((u) => ({ u, team: 'ally' })).concat(enemy.map((u) => ({ u, team: 'enemy' })));
  if (!has || !all.length) {
    list.innerHTML = '<li class="empty">No session</li>';
    return;
  }
  for (const { u, team } of all) {
    const label = u && u.label !== undefined ? String(u.label) : '?';
    const c = u && Array.isArray(u.center_px) ? u.center_px : null;
    const xy = c && c.length >= 2
      ? 'x ' + Math.round(Number(c[0])) + ' / y ' + Math.round(Number(c[1])) : 'x — / y —';
    const conf = u && u.confidence !== undefined && u.confidence !== null ? Number(u.confidence).toFixed(2) : '—';
    const li = document.createElement('li');
    li.innerHTML = '<span><span class="team team-' + team + '">' + team + '</span>' +
      '<strong>' + esc(label) + '</strong> <span class="muted">' + esc(xy) + '</span></span>' +
      '<span>' + esc(conf) + '</span>';
    list.appendChild(li);
    if (list.children.length >= 6) break;
  }
  if (all.length > list.children.length) {
    const more = document.createElement('li');
    more.className = 'more';
    more.textContent = '+' + (all.length - list.children.length) + ' more';
    list.appendChild(more);
  }
}

/* ---------- center ---------- */

function renderCenter(frame) {
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
    perf = 'OCR ' + (Number(diag.ocr_confidence) * 100).toFixed(1) + '%';
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
}

function containRect() {
  const wrap = els['frame-wrap'];
  const w = wrap.clientWidth, h = wrap.clientHeight;
  const nw = img.naturalWidth || 0, nh = img.naturalHeight || 0;
  if (!nw || !nh) return { x: 0, y: 0, w, h, scaleX: 0, scaleY: 0 };
  const s = Math.min(w / nw, h / nh);
  const dw = nw * s, dh = nh * s;
  return { x: (w - dw) / 2, y: (h - dh) / 2, w: dw, h: dh, scaleX: dw / nw, scaleY: dh / nh };
}

function frameDims(frame) {
  if (!frame || typeof frame !== 'object') return null;
  const w = num(frame.frame_width), h = num(frame.frame_height);
  if (w !== null && h !== null && w > 0 && h > 0) return { w, h };
  return null;
}

function toDisplay(pt, rect, frame) {
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

function parseCell(cell) {
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

function handSlotRect(slot, rect, frame) {
  const dims = frameDims(frame);
  if (!dims || slot === null || slot === undefined || slot < 0 || slot >= HAND_SLOT_ROIS.length) return null;
  const roi = HAND_SLOT_ROIS[slot];
  const kx = dims.w / ROI_REF_W, ky = dims.h / ROI_REF_H;
  return {
    x: rect.x + (roi.x * kx / dims.w) * rect.w,
    y: rect.y + (roi.y * ky / dims.h) * rect.h,
    w: (roi.w * kx / dims.w) * rect.w,
    h: (roi.h * ky / dims.h) * rect.h,
  };
}

function gridSpec() {
  const g = state.gridSpec;
  if (g && num(g.cols) === GRID_COLS && num(g.rows) === GRID_ROWS &&
      [g.x0, g.y0, g.x1, g.y1].every((v) => num(v) !== null)) {
    return { cols: GRID_COLS, rows: GRID_ROWS, x0: +g.x0, y0: +g.y0, x1: +g.x1, y1: +g.y1 };
  }
  return GRID_SPEC_FALLBACK;
}

function arenaPxOf(frame) {
  // (ax, ay, aw, ah): arena origin + size in frame pixels, same tuple the
  // backend trackers pass to ACTION_GRID.cell_to_pixel_center.
  const vs = visualStateOf(frame);
  const a = vs.arena_px;
  if (!Array.isArray(a) || a.length < 4) return null;
  const vals = a.slice(0, 4).map(num);
  if (vals.some((v) => v === null) || vals[2] <= 0 || vals[3] <= 0) return null;
  return vals;
}

function gridBounds(rect, frame) {
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

function cellToDisplay(cell, rect, frame) {
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

function drawOverlay(frame, suggestions, diagnostics) {
  const wrap = els['frame-wrap'];
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

function drawSuggestionMarkers(sug, rect, frame) {
  const plays = (sug || []).slice(0, 3);
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

function drawPlayArrow(s, rect, frame) {
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

function drawGrid(rect, frame) {
  // Grid lines span the arena-mapped grid bounds, not the full image.
  const gb = gridBounds(rect, frame);
  ctx.save();
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

function drawBoxes(vs, rect, withLabels, labelsOnly, frame) {
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

/* ---------- right panel ---------- */

function actionText(s, fallbackName) {
  if (!s || typeof s !== 'object') return '—';
  const kind = String(s.kind || '').toLowerCase();
  if (kind === 'wait') return 'Wait';
  const raw = s.card_name !== undefined && s.card_name !== null ? String(s.card_name) : null;
  const name = raw || fallbackName || 'card';
  const cell = parseCell(s.cell);
  if (cell) return 'Place ' + name + ' @ (' + Math.round(cell.col) + ',' + Math.round(cell.row) + ')';
  return 'Play ' + name;
}

function probPct(s) {
  const p = num(s.probability);
  if (p === null) return '—';
  return (p <= 1 ? p * 100 : p).toFixed(0) + '%';
}

function probFrac(s) {
  const p = num(s.probability);
  if (p === null) return null;
  return p <= 1 ? p : p / 100;
}

function shouldAppendWaitRow(sug, diag) {
  // WAIT is one lumped hypothesis against ~2,300 split card×cell combos, so
  // a healthy P(wait) can sit just outside the top-3 cutoff. Return its
  // scored mass when no Wait entry is listed, else null.
  const hasWait = (sug || []).some(
    (s) => String((s && s.kind) || '').toLowerCase() === 'wait'
  );
  if (hasWait) return null;
  return finiteNum(diag && diag.mode_prob_wait);
}

function finiteNum(v) {
  // Like num(), but JSON null/undefined/'' mean "unknown", never 0
  // (Number(null) === 0 would silently turn unknown into zero).
  if (v === null || v === undefined || v === '') return null;
  return num(v);
}

function modeProbsText(diag) {
  const mpw = finiteNum(diag && diag.mode_prob_wait);
  const mpp = finiteNum(diag && diag.mode_prob_play);
  if (mpw !== null || mpp !== null) {
    return 'wait ' + (mpw === null ? '—' : (mpw * 100).toFixed(1) + '%') +
      ' · play ' + (mpp === null ? '—' : (mpp * 100).toFixed(1) + '%');
  }
  const legacy = diag ? diag.mode_probs : undefined;
  if (legacy !== undefined && legacy !== null) {
    return typeof legacy === 'object' ? JSON.stringify(legacy) : String(legacy);
  }
  return null;
}

function waitReason(frame) {
  // Human reason for a frame with no scored suggestions (backend `result`).
  const rec = (frame && frame.record) || {};
  switch (rec.result) {
    case 'hand-not-stable': return 'Waiting for a stable hand…';
    case 'observation-not-ready': return 'Waiting for a readable game state…';
    case 'not-in-game': return 'Not in game — no decision';
    case 'cooldown': return 'Wait — between plays';
    default: {
      const action = actionOf(frame);
      if (action && String(action.kind || '').toLowerCase() === 'wait') return 'Wait — holding elixir';
      return 'No suggestion for this frame';
    }
  }
}

function renderRight(frame) {
  const box = els['suggestions'];
  box.innerHTML = '';
  const sug = suggestionsOf(frame).slice()
    .sort((a, b) => (num(b.probability) || 0) - (num(a.probability) || 0))
    .slice(0, 3);
  const diag = diagnosticsOf(frame);

  if (!frame) {
    box.innerHTML = '<p class="empty">No session</p>';
  } else if (!sug.length) {
    // A frame with no scored suggestions is still a live analysis result —
    // typically a deliberate Wait (unstable hand, cooldown) — not "no session".
    box.innerHTML =
      '<div class="suggestion is-wait"><div class="suggestion-head">' +
      '<span class="suggestion-rank">–</span>' +
      '<span><span class="suggestion-title">Wait</span><br />' +
      '<span class="suggestion-prob">' + esc(waitReason(frame)) + '</span></span>' +
      '</div></div>';
  } else {
    sug.forEach((s, i) => {
      const card = document.createElement('div');
      card.className = 'suggestion' + (i === 0 ? ' is-top' : '') +
        (state.selectedRank === i ? ' is-selected' : '');
      card.dataset.rank = String(i);
      const ev = s.ev !== undefined && s.ev !== null ? s.ev
        : s.expected_value !== undefined ? s.expected_value : null;
      const lp = s.log_prob !== undefined && s.log_prob !== null ? Number(s.log_prob) : null;
      const cell = parseCell(s.cell);
      const slot = num(s.card_slot);
      const name = suggestionCardName(s, visualStateOf(frame)) || 'card';
      const iconUrl = String(s.kind || '').toLowerCase() === 'play' ? cardIconUrl(name === 'card' ? null : name) : null;
      let preview = '';
      if (state.selectedRank === i) {
        if (cell && slot !== null) {
          preview =
            '<div class="inspect-preview">' +
            '<div class="inspect-caption">Slot ' + (slot + 1) + ' · ' + esc(name) +
            ' → (' + Math.round(cell.col) + ',' + Math.round(cell.row) + ')<br />' +
            '<span class="muted">' + esc(probPct(s)) +
            (lp === null || !Number.isFinite(lp) ? '' : ' · logprob ' + lp.toFixed(2)) + '</span><br />' +
            '<button type="button" class="raw-toggle" data-raw="' + i + '">raw JSON</button></div></div>' +
            '<div class="inspect-raw" data-rawbox="' + i + '" hidden><pre>' +
            esc(JSON.stringify(s, null, 2)) + '</pre></div>';
        } else {
          preview =
            '<div class="inspect-preview"><div class="inspect-caption">Wait — hold elixir, no placement.<br />' +
            '<button type="button" class="raw-toggle" data-raw="' + i + '">raw JSON</button></div></div>' +
            '<div class="inspect-raw" data-rawbox="' + i + '" hidden><pre>' +
            esc(JSON.stringify(s, null, 2)) + '</pre></div>';
        }
      }
      card.innerHTML =
        '<div class="suggestion-head">' +
        (iconUrl
          ? '<img class="card-icon" src="' + iconUrl + '" alt="" loading="lazy" onerror="this.hidden=true" />'
          : '') +
        '<span class="suggestion-rank">' + (i + 1) + '</span>' +
        '<span><span class="suggestion-title">' + esc(actionText(s, name === 'card' ? null : name)) + '</span><br />' +
        '<span class="suggestion-prob">Probability ' + esc(probPct(s)) +
        (lp === null || !Number.isFinite(lp) ? '' : ' · logprob ' + lp.toFixed(2)) + '</span></span>' +
        (ev === null ? '' : '<span class="suggestion-ev">' + esc(String(ev)) + ' EV</span>') +
        '</div>' + preview +
        '<button type="button" class="inspect-btn' + (state.selectedRank === i ? ' is-on' : '') +
        '" data-inspect="' + i + '">' +
        (state.selectedRank === i ? 'Hide' : 'Inspect') + ' ↗</button>';
      box.appendChild(card);
    });
    const waitP = shouldAppendWaitRow(sug, diag);
    if (waitP !== null) {
      const waitCard = document.createElement('div');
      waitCard.className = 'suggestion is-wait';
      waitCard.innerHTML =
        '<div class="suggestion-head">' +
        '<span class="suggestion-rank">–</span>' +
        '<span><span class="suggestion-title">Wait</span><br />' +
        '<span class="suggestion-prob">Probability ' + esc(probPct({ probability: waitP })) +
        ' · outside top 3</span></span>' +
        '</div>';
      box.appendChild(waitCard);
    }
  }

  // Reasoning summary — only real backend values, never invented.
  if (!frame) {
    els['reason-text'].textContent = 'No session';
    els['reason-entropy'].textContent = '—';
    els['reason-mode-probs'].textContent = '—';
    els['reason-hand-stable'].textContent = '—';
    els['reason-confidence'].textContent = '—';
    els['reason-confidence'].className = '';
    return;
  }
  if (diag.summary !== undefined && diag.summary !== null) {
    els['reason-text'].textContent = String(diag.summary);
  } else if (sug.length) {
    els['reason-text'].textContent = 'Top action ' + actionText(sug[0], suggestionCardName(sug[0], visualStateOf(frame))) +
      ' at ' + probPct(sug[0]) + ' over ' + sug.length + ' suggestion(s).';
  } else {
    els['reason-text'].textContent = waitReason(frame) + '.';
  }
  const ent = num(diag.entropy);
  els['reason-entropy'].textContent = ent === null ? '—' : ent.toFixed(3);
  const modeProbs = modeProbsText(diag);
  els['reason-mode-probs'].textContent = modeProbs === null ? '—' : modeProbs;
  if (diag.hand_stable !== undefined && diag.hand_stable !== null) {
    els['reason-hand-stable'].textContent = typeof diag.hand_stable === 'boolean'
      ? (diag.hand_stable ? 'Yes' : 'No') : String(diag.hand_stable);
  } else {
    els['reason-hand-stable'].textContent = '—';
  }
  const topP = sug.length ? probFrac(sug[0]) : null;
  const conf = els['reason-confidence'];
  if (topP === null) {
    conf.textContent = '—';
    conf.className = '';
  } else if (topP >= 0.8) {
    conf.textContent = 'High';
    conf.className = 'is-high';
  } else if (topP >= 0.5) {
    conf.textContent = 'Medium';
    conf.className = 'is-med';
  } else {
    conf.textContent = 'Low';
    conf.className = 'is-low';
  }
}

/* ---------- timeline / transport ---------- */

function frameFlags(frame, prev) {
  let tower = false;
  if (frame && prev) {
    const a = visualStateOf(frame), b = visualStateOf(prev);
    const sum = (t) => (Array.isArray(t) ? t : []).reduce((acc, v) => acc + (num(v) || 0), 0);
    const now = sum(a.tower_hp_self) + sum(a.tower_hp_enemy);
    const before = sum(b.tower_hp_self) + sum(b.tower_hp_enemy);
    if (now > 0 && before > 0 && now < before) tower = true;
  }
  // Match boundaries from in-game transitions (first in-game frame counts
  // as a start even with no predecessor).
  const wasInGame = !!(prev && prev.in_game);
  const inGame = !!(frame && frame.in_game);
  return { tower, matchStart: inGame && !wasInGame, matchEnd: !inGame && wasInGame };
}

function indexForTime(t, fallback) {
  // Nearest history index by frame timestamp, so tracker events confirmed
  // late still land at their actual play time. Falls back to arrival index.
  const n = state.history.length;
  if (!Number.isFinite(t) || !n) return fallback;
  let best = fallback, bd = Infinity;
  for (let i = 0; i < n; i++) {
    const ts = num(state.history[i].timestamp_s);
    if (ts === null) continue;
    const d = Math.abs(ts - t);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function eventCard(ev) {
  if (!ev || typeof ev !== 'object') return null;
  const c = ev.card;
  return typeof c === 'string' && c ? c : null;
}

function renderTimeline() {
  const n = state.history.length;
  const track = els['timeline-track'];
  const markers = els['timeline-markers'];
  markers.innerHTML = '';
  track.setAttribute('aria-valuemax', String(Math.max(0, n - 1)));
  track.setAttribute('aria-valuenow', String(Math.max(0, state.cursor)));

  if (!n) {
    els['timeline-label'].textContent = 'No session';
    els['timeline-fill'].style.width = '0';
    els['timeline-cursor'].hidden = true;
    return;
  }
  const first = state.history[0], last = state.history[n - 1];
  const t0 = num(first.timestamp_s), t1 = num(last.timestamp_s);
  const span = t0 !== null && t1 !== null ? ' · ' + t0.toFixed(1) + 's → ' + t1.toFixed(1) + 's' : '';
  els['timeline-label'].textContent = 'f' + first.frame_index + ' → f' + last.frame_index +
    ' · ' + n + ' frames' + span + ' · cursor f' + (currentFrame() ? currentFrame().frame_index : '—');
  const frac = n > 1 ? state.cursor / (n - 1) : 1;
  els['timeline-fill'].style.width = (frac * 100).toFixed(2) + '%';
  const cur = els['timeline-cursor'];
  cur.hidden = false;
  cur.style.left = (frac * 100).toFixed(2) + '%';

  for (let i = 0; i < n; i++) {
    const fr = state.history[i];
    const flags = frameFlags(fr, i > 0 ? state.history[i - 1] : null);
    // kinds: {k, at, label} — real extracted plays land at their event time.
    const kinds = [];
    if (flags.matchStart) kinds.push({ k: 'match', at: i, label: 'match start' });
    if (flags.matchEnd) kinds.push({ k: 'match', at: i, label: 'match end' });
    if (flags.tower) kinds.push({ k: 'tower', at: i, label: 'tower damage' });
    const own = Array.isArray(fr.own_actions) ? fr.own_actions : null;
    const foe = Array.isArray(fr.enemy_plays) ? fr.enemy_plays : null;
    if (own !== null || foe !== null) {
      for (const ev of own || []) {
        const at = indexForTime(num(ev.video_time_s), i);
        const card = eventCard(ev);
        kinds.push({ k: 'user', at, label: 'your play' + (card ? ' ' + card : '') });
      }
      for (const ev of foe || []) {
        const at = indexForTime(num(ev.video_time_s), i);
        const card = eventCard(ev);
        kinds.push({ k: 'foe', at, label: 'opponent play' + (card ? ' ' + card : '') });
      }
    } else {
      // Legacy fallback for servers without tracker arrays: proposed plays.
      const action = actionOf(fr);
      if (action && String(action.kind || '').toLowerCase() === 'play') {
        kinds.push({ k: 'user', at: i, label: 'proposed play' });
      }
    }
    if (!kinds.length) continue;
    for (const { k, at, label } of kinds) {
      const d = document.createElement('div');
      d.className = 'marker marker-' + k + (n > 120 ? ' small' : '');
      d.style.left = (n > 1 ? (at / (n - 1)) * 100 : 0).toFixed(2) + '%';
      d.title = 'f' + state.history[at].frame_index + ' ' + label;
      markers.appendChild(d);
    }
  }
}

function trackIndexFromEvent(ev) {
  // Percentages resolve against the inset rail, so measure the rail —
  // dots, fill, and thumb then share one geometry, including at the edges.
  const r = els['timeline-rail'].getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (ev.clientX - r.left) / Math.max(1, r.width)));
  return Math.round(frac * (state.history.length - 1));
}

function frameTooltip(i) {
  const fr = state.history[i];
  if (!fr) return null;
  const head = 'f' + fr.frame_index +
    (fr.timestamp_s !== undefined ? ' · ' + Number(fr.timestamp_s).toFixed(1) + 's' : '');
  const parts = [];
  for (const ev of Array.isArray(fr.own_actions) ? fr.own_actions : []) {
    const card = eventCard(ev);
    if (card) parts.push('Your play ' + card);
  }
  for (const ev of Array.isArray(fr.enemy_plays) ? fr.enemy_plays : []) {
    const card = eventCard(ev);
    if (card) parts.push('Opponent play ' + card);
  }
  if ((!fr.own_actions || !fr.enemy_plays) && parts.length === 0) {
    const action = actionOf(fr);
    if (action && String(action.kind || '').toLowerCase() === 'play') parts.push('Proposed play');
  }
  const prev = i > 0 ? state.history[i - 1] : null;
  const ff = frameFlags(fr, prev);
  if (ff.tower) parts.push('Tower damage');
  if (ff.matchStart) parts.push('Match start');
  if (ff.matchEnd) parts.push('Match end');
  return { head, sub: parts.join(' · ') || 'No events' };
}

function showTrackTooltip(ev) {
  const tip = els['timeline-tooltip'];
  if (!state.history.length) { tip.hidden = true; return; }
  const i = trackIndexFromEvent(ev);
  const info = frameTooltip(i);
  if (!info) { tip.hidden = true; return; }
  const r = els['timeline-track'].getBoundingClientRect();
  const frac = state.history.length > 1 ? i / (state.history.length - 1) : 0;
  tip.innerHTML = '<div class="tt-head">' + esc(info.head) + '</div>' +
    '<div class="tt-sub">' + esc(info.sub) + '</div>';
  tip.style.left = (frac * 100).toFixed(2) + '%';
  tip.hidden = false;
}

function hideTrackTooltip() {
  els['timeline-tooltip'].hidden = true;
}

function renderCurrent() {
  const frame = currentFrame();
  renderLeft(frame);
  renderCenter(frame);
  renderRight(frame);
  renderTimeline();
}

function seek(i) {
  if (!state.history.length) return;
  state.cursor = Math.max(0, Math.min(state.history.length - 1, i));
  refreshImage();
  renderCurrent();
}

/* ---------- session controls ---------- */

function setMode(mode) {
  state.mode = mode;
  const isVideo = mode === 'video';
  els['tab-video'].classList.toggle('is-active', isVideo);
  els['tab-live'].classList.toggle('is-active', !isVideo);
  els['tab-video'].setAttribute('aria-selected', String(isVideo));
  els['tab-live'].setAttribute('aria-selected', String(!isVideo));
  els['panel-video'].hidden = !isVideo;
  els['panel-live'].hidden = isVideo;
}

function resetSession(label) {
  state.history = [];
  state.cursor = -1;
  state.lastSince = 0;
  state.sessionLabel = label || '';
  state.selectedRank = null;
  state.sessionPin = null;
  applySessionVisibility(null);
  renderCurrent();
}

function applySessionVisibility(s) {
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

function fillCheckpointSelect(selectEl, data) {
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

async function loadGrid() {
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

async function loadCheckpoints() {
  try {
    const data = await apiGet('/api/checkpoints');
    fillCheckpointSelect(els['select-checkpoint'], data);
    fillCheckpointSelect(els['select-live-checkpoint'], data);
  } catch (err) {
    fillCheckpointSelect(els['select-checkpoint'], null);
    fillCheckpointSelect(els['select-live-checkpoint'], null);
  }
}

async function uploadVideoFile(file) {
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

function setVideoFileLabel(name) {
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

function setVideoFileMeta(text, isError) {
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

async function onVideoFileChange() {
  const input = els['input-video-file'];
  const file = input && input.files && input.files[0];
  if (!file) return;
  clearRoiAdapt();
  setVideoFileLabel(file.name);
  setVideoFileMeta('Uploading…');
  try {
    const out = await uploadVideoFile(file);
    state.uploadedVideoPath = out.path || '';
    state.uploadedVideoName = out.filename || file.name;
    showError('');
    setVideoFileLabel(state.uploadedVideoName);
    // Probe the upload for frame count / duration so the start frame
    // has a known valid range.
    try {
      const info = await apiGet('/api/video/info?path=' + encodeURIComponent(state.uploadedVideoPath));
      const bits = [];
      if (info.frames) bits.push(info.frames + ' frames');
      if (info.duration_s !== undefined && info.duration_s !== null) bits.push(Number(info.duration_s).toFixed(1) + 's');
      if (info.fps) bits.push(Number(info.fps).toFixed(0) + 'fps');
      setVideoFileMeta(bits.join(' · ') || null);
      els['video-file-label'].title = state.uploadedVideoName + ' → ' + (out.path || '');
      if (roiAdaptAvailableFromDims(info.width, info.height)) {
        showRoiAdaptAvailable(info.width, info.height);
      } else {
        clearRoiAdapt();
      }
    } catch (probeErr) {
      setVideoFileMeta(null);
      clearRoiAdapt();
    }
  } catch (err) {
    state.uploadedVideoPath = '';
    clearRoiAdapt();
    setVideoFileLabel(null);
    setVideoFileMeta('Upload failed — try again.', true);
    showError(String(err && err.message || err));
  }
}

async function startVideo() {
  let videoPath = state.uploadedVideoPath;
  const input = els['input-video-file'];
  const pending = input && input.files && input.files[0];
  if (!videoPath && pending) {
    try {
      setVideoFileLabel(pending.name);
      setVideoFileMeta('Uploading…');
      const out = await uploadVideoFile(pending);
      videoPath = out.path || '';
      state.uploadedVideoPath = videoPath;
      state.uploadedVideoName = out.filename || pending.name;
      setVideoFileLabel(state.uploadedVideoName);
      setVideoFileMeta(null);
    } catch (err) {
      setVideoFileMeta('Upload failed — try again.', true);
      showError(String(err && err.message || err));
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
  // Sync ROI checkboxes into state (single source for the payload builder).
  if (els['check-adapt-rois']) state.roiAdapt.checked = !!els['check-adapt-rois'].checked;
  if (els['check-accept-rois']) {
    state.roiAdapt.accepted = !!els['check-accept-rois'].checked && !!state.roiAdapt.preview;
  }
  if (state.roiAdapt.checked && !state.roiAdapt.accepted) {
    showError('Preview and accept adapted ROIs first (or uncheck).');
    return;
  }
  const payload = buildVideoStartPayload({
    videoPath, stride, startFrame, maxFrames, checkpoint, roiAdapt: state.roiAdapt,
  });
  try {
    showError('');
    await apiPost('/api/video/start', payload);
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

async function startLive() {
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

async function stopSession() {
  try {
    await apiPost('/api/stop', {});
  } catch (err) {
    showError(String(err && err.message || err));
  }
  pollStatus();
}

/* ---------- events ---------- */

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
els['btn-stop-mini'].addEventListener('click', stopSession);
els['btn-browse-video'].addEventListener('click', () => els['input-video-file'].click());
els['btn-session-toggle'].addEventListener('click', () => {
  const card = document.querySelector('.card-session');
  const collapsed = card ? card.classList.contains('collapsed') : false;
  state.sessionPin = !collapsed;
  applySessionVisibility(null);
});
els['input-video-file'].addEventListener('change', onVideoFileChange);
if (els['check-adapt-rois']) {
  els['check-adapt-rois'].addEventListener('change', () => {
    state.roiAdapt.checked = !!els['check-adapt-rois'].checked;
  });
}
if (els['btn-preview-rois']) {
  els['btn-preview-rois'].addEventListener('click', previewRois);
}
if (els['check-accept-rois']) {
  els['check-accept-rois'].addEventListener('change', () => {
    state.roiAdapt.accepted = !!els['check-accept-rois'].checked && !!state.roiAdapt.preview;
  });
}

// Inspect toggles locate the suggestion on the arena (delegated: cards
// re-render on every poll). Raw JSON stays behind a secondary toggle.
els['suggestions'].addEventListener('click', (ev) => {
  const rawBtn = ev.target.closest('[data-raw]');
  if (rawBtn) {
    const box = els['suggestions'].querySelector('[data-rawbox="' + rawBtn.dataset.raw + '"]');
    if (box) box.hidden = !box.hidden;
    return;
  }
  const btn = ev.target.closest('[data-inspect]');
  if (!btn) return;
  const rank = parseInt(btn.dataset.inspect, 10) || 0;
  state.selectedRank = state.selectedRank === rank ? null : rank;
  renderCurrent();
});
els['btn-stop-video'].addEventListener('click', stopSession);
els['btn-start-live'].addEventListener('click', startLive);
els['btn-stop-live'].addEventListener('click', stopSession);

els['check-execute'].addEventListener('change', () => {
  els['live-warning'].hidden = !els['check-execute'].checked;
});

els['btn-play'].addEventListener('click', () => {
  state.playing = !state.playing;
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

function bindToggle(id, key) {
  els[id].addEventListener('click', () => {
    state.toggles[key] = !state.toggles[key];
    els[id].classList.toggle('is-on', state.toggles[key]);
    els[id].setAttribute('aria-pressed', String(state.toggles[key]));
    renderCenter(currentFrame());
  });
}
bindToggle('toggle-boxes', 'boxes');
bindToggle('toggle-grid', 'grid');
bindToggle('toggle-labels', 'labels');

let trackDragging = false;

els['timeline-track'].addEventListener('pointerdown', (ev) => {
  if (!state.history.length) return;
  trackDragging = true;
  els['timeline-track'].setPointerCapture(ev.pointerId);
  seek(trackIndexFromEvent(ev));
  showTrackTooltip(ev);
});
els['timeline-track'].addEventListener('pointermove', (ev) => {
  if (!state.history.length) return;
  if (trackDragging) seek(trackIndexFromEvent(ev));
  showTrackTooltip(ev);
});
els['timeline-track'].addEventListener('pointerup', (ev) => {
  trackDragging = false;
  hideTrackTooltip();
});
els['timeline-track'].addEventListener('pointercancel', () => {
  trackDragging = false;
  hideTrackTooltip();
});
els['timeline-track'].addEventListener('pointerleave', () => {
  if (!trackDragging) hideTrackTooltip();
});
els['timeline-track'].addEventListener('keydown', (ev) => {
  if (ev.key === 'ArrowLeft') { ev.preventDefault(); seek(state.cursor - (ev.shiftKey ? 10 : 1)); }
  else if (ev.key === 'ArrowRight') { ev.preventDefault(); seek(state.cursor + (ev.shiftKey ? 10 : 1)); }
  else if (ev.key === 'Home') { ev.preventDefault(); seek(0); }
  else if (ev.key === 'End') { ev.preventDefault(); seek(state.history.length - 1); }
  else if (ev.key === 'PageUp') { ev.preventDefault(); seek(state.cursor - 10); }
  else if (ev.key === 'PageDown') { ev.preventDefault(); seek(state.cursor + 10); }
});

window.addEventListener('resize', () => drawOverlay(currentFrame()));

/* ---------- init ---------- */

setMode('video');
renderCurrent();
loadCheckpoints();
loadGrid();
pollStatus();
setInterval(pollStatus, 2000);
scheduleFrames();
