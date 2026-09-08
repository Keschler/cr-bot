/* Timeline / transport: action history feed, timeline markers, scrubbing,
 * and renderCurrent() — the single fan-out that repaints all panels.
 */

import { state } from '../state/store.js';
import { els } from '../utils/elements.js';
import { esc } from '../utils/dom.js';
import { num } from '../utils/format.js';
import { encodeHash } from '../utils/share.js';
import { visualStateOf, actionOf, currentFrame } from './frames.js';
import { renderLeft, renderRight } from './panels.js';
import { renderCenter } from './overlay.js';
import { roiPreviewVisible } from './roi-editor.js';
import { hideFloatbar } from './corrections.js';
import { refreshImage, noteCursorMoved } from './session.js';

export function frameFlags(frame, prev) {
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

export function indexForTime(t, fallback) {
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

export function buildTimeIndex(history) {
  // Sorted [{t, i}] over frames with numeric timestamps: one O(n log n)
  // build per render, then O(log n) nearest lookups per event instead of an
  // O(n) scan per event.
  const idx = [];
  for (let i = 0; i < history.length; i++) {
    const t = num(history[i] && history[i].timestamp_s);
    if (t !== null) idx.push({ t, i });
  }
  idx.sort((a, b) => a.t - b.t);
  return idx;
}

export function indexForTimeIn(t, fallback, timeIndex) {
  if (!Number.isFinite(t) || !timeIndex || !timeIndex.length) return fallback;
  // Binary search for the insertion point, then take the nearest neighbor.
  let lo = 0, hi = timeIndex.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (timeIndex[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  let best = fallback, bd = Infinity;
  for (const k of [lo - 1, lo, lo + 1]) {
    const e = timeIndex[k];
    if (!e) continue;
    const d = Math.abs(e.t - t);
    if (d < bd) { bd = d; best = e.i; }
  }
  return best;
}

export function eventCard(ev) {
  if (!ev || typeof ev !== 'object') return null;
  const c = ev.card;
  return typeof c === 'string' && c ? c : null;
}

export function fmtClock(s) {
  const v = num(s);
  if (v === null || v < 0) return '—';
  return Math.floor(v / 60) + ':' + String(Math.floor(v % 60)).padStart(2, '0');
}

export function collectActionEntries(history) {
  // Pure: confirmed plays across frames, newest first. Testable via CDP.
  const entries = [];
  for (let i = 0; i < history.length; i++) {
    const fr = history[i];
    if (!fr) continue;
    for (const ev of Array.isArray(fr.own_actions) ? fr.own_actions : []) {
      const t = num(ev.video_time_s);
      entries.push({
        side: 'you', team: 'ally', card: eventCard(ev) || '?',
        t: t !== null ? t : num(fr.timestamp_s), fi: fr.frame_index, idx: i,
      });
    }
    for (const ev of Array.isArray(fr.enemy_plays) ? fr.enemy_plays : []) {
      const t = num(ev.video_time_s);
      entries.push({
        side: 'foe', team: 'enemy', card: eventCard(ev) || '?',
        t: t !== null ? t : num(fr.timestamp_s), fi: fr.frame_index, idx: i,
      });
    }
  }
  entries.sort((a, b) => ((b.t === null ? -1 : b.t) - (a.t === null ? -1 : a.t)) || (b.idx - a.idx));
  return entries;
}

export function renderActionHistory(overrideEntries) {
  // Newest-first feed of confirmed plays; click a row to jump to its frame.
  // overrideEntries (CDP/smoke use) renders the given entries instead.
  const list = els['action-history'];
  if (!list) return;
  const entries = Array.isArray(overrideEntries)
    ? overrideEntries
    : collectActionEntries(state.history);
  const hasSession = !!state.history.length || Array.isArray(overrideEntries);
  const you = entries.filter((e) => e.side === 'you').length;
  els['action-history-count'].textContent = hasSession
    ? '(' + you + ' you · ' + (entries.length - you) + ' foe)' : '';
  list.innerHTML = '';
  if (!hasSession) {
    list.innerHTML = '<li class="empty">No session</li>';
    return;
  }
  if (!entries.length) {
    list.innerHTML = '<li class="empty">No plays yet</li>';
    return;
  }
  for (const e of entries) {
    const li = document.createElement('li');
    // Seek target is the immutable frame_index: arrival indices shift when
    // the 2000-frame cap truncates history. Entries without one (CDP
    // overrides) fall back to the arrival index.
    li.dataset.seek = (e.fi !== undefined && e.fi !== null) ? String(e.fi) : String(e.idx);
    const fiLabel = (e.fi !== undefined && e.fi !== null) ? e.fi : e.idx;
    li.title = 'Jump to frame f' + fiLabel;
    li.innerHTML = '<span><span class="team team-' + e.team + '">' + e.side + '</span>' +
      '<strong>' + esc(e.card) + '</strong></span>' +
      '<span class="muted">' + esc(fmtClock(e.t)) + ' · f' + fiLabel + '</span>';
    list.appendChild(li);
  }
}

export function renderTimeline() {
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

  // One timestamp index per render; event markers then resolve in O(log n).
  const timeIndex = buildTimeIndex(state.history);
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
        const at = indexForTimeIn(num(ev.video_time_s), i, timeIndex);
        const card = eventCard(ev);
        kinds.push({ k: 'user', at, label: 'your play' + (card ? ' ' + card : '') });
      }
      for (const ev of foe || []) {
        const at = indexForTimeIn(num(ev.video_time_s), i, timeIndex);
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

export function trackIndexFromEvent(ev) {
  // Percentages resolve against the inset rail, so measure the rail —
  // dots, fill, and thumb then share one geometry, including at the edges.
  const r = els['timeline-rail'].getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (ev.clientX - r.left) / Math.max(1, r.width)));
  return Math.round(frac * (state.history.length - 1));
}

export function frameTooltip(i) {
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
  if (!fr.own_actions && !fr.enemy_plays && parts.length === 0) {
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

export function showTrackTooltip(ev) {
  const tip = els['timeline-tooltip'];
  if (!state.history.length) { tip.hidden = true; return; }
  const i = trackIndexFromEvent(ev);
  const info = frameTooltip(i);
  if (!info) { tip.hidden = true; return; }
  const r = els['timeline-rail'].getBoundingClientRect();
  const frac = state.history.length > 1 ? i / (state.history.length - 1) : 0;
  tip.innerHTML = '<div class="tt-head">' + esc(info.head) + '</div>' +
    '<div class="tt-sub">' + esc(info.sub) + '</div>';
  tip.hidden = false;
  // Viewport-anchored (the tooltip is position:fixed): 10px above the
  // rail, clamped so edge dots don't push it off-screen. The rail (not the
  // outer track) matches dots, fill, and thumb geometry.
  const w = tip.offsetWidth || 0;
  let x = r.left + frac * r.width;
  x = Math.max(w / 2 + 8, Math.min(window.innerWidth - w / 2 - 8, x));
  tip.style.left = x + 'px';
  tip.style.top = (r.top - 10) + 'px';
}

export function hideTrackTooltip() {
  els['timeline-tooltip'].hidden = true;
}

export function writeLocationHash() {
  // Deep link to the cursor frame (+ inspected rank): cheap replaceState so
  // scrubbing never spams the browser history. Read back at boot.
  try {
    const frame = currentFrame();
    window.history.replaceState(null, '', encodeHash({
      frame: frame ? frame.frame_index : null,
      rank: state.selectedRank,
      video: state.uploadedVideoName || state.sessionLabel || null,
    }));
  } catch (e) { /* file:// or restricted contexts */ }
}

export function renderCurrent() {
  // A pre-session ROI preview takes over the whole center column: session
  // panels fall back to "No session" while the proposal is reviewed.
  const frame = roiPreviewVisible() ? null : currentFrame();
  renderLeft(frame);
  renderCenter(frame);
  renderRight(frame);
  renderActionHistory();
  renderTimeline();
}

export function seek(i) {
  if (!state.history.length) return;
  // A manual seek pauses: otherwise the next arriving frames snap the cursor
  // back to the live edge and the seek is immediately yanked away.
  if (state.playing) {
    state.playing = false;
    if (els['btn-play']) {
      els['btn-play'].textContent = 'Play';
      els['btn-play'].setAttribute('aria-pressed', 'false');
    }
  }
  state.cursor = Math.max(0, Math.min(state.history.length - 1, i));
  state.selection = null;
  state.dragBox = null;
  state.dragMove = null;
  hideFloatbar();
  refreshImage();
  renderCurrent();
  noteCursorMoved(false);
  writeLocationHash();
}

let trackDragging = false;

export function bindTimelineEvents() {
  if (els['action-history']) {
    els['action-history'].addEventListener('click', (ev) => {
      const li = ev.target && ev.target.closest ? ev.target.closest('li[data-seek]') : null;
      if (!li) return;
      // data-seek holds a frame_index (stable across truncation); resolve it
      // to the current arrival index. Legacy arrival indices still work.
      const raw = String(li.dataset.seek);
      let idx = state.history.findIndex((f) => String(f.frame_index) === raw);
      if (idx < 0) {
        const legacy = parseInt(raw, 10);
        if (Number.isFinite(legacy)) idx = legacy;
      }
      if (idx >= 0) seek(idx);
    });
  }

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
}
