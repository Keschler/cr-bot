/* Label correction (what-if re-evaluation): draft edits, edit-mode canvas
 * gestures, floating toolbar, submit/revert against the backend.
 */

import { state } from '../state/store.js';
import { els, canvas } from '../utils/elements.js';
import { esc, showError } from '../utils/dom.js';
import { num } from '../utils/format.js';
import { toDisplay, containRect, frameDims } from '../utils/geometry.js';
import { apiGet, apiPost, apiDelete } from '../api/client.js';
import { currentFrame, suggestionsOf, diagnosticsOf } from './frames.js';
import { roiPreviewVisible, roiPreviewImageReady, roiCanvasMouseDown, updateRoiDrag, finishRoiDrag, hitRoi, drawRoiPreviewBoxes } from './roi-editor.js';
import { drawOverlay } from './overlay.js';
import { renderCurrent } from './timeline.js';

export function detectionRowsOf(frame) {
  if (!frame || !Array.isArray(frame.detections)) return [];
  return frame.detections;
}

export function rowKeyOfDetection(row) {
  if (!row) return '';
  if (row.track_id !== undefined && row.track_id !== null) return 't:' + String(row.track_id);
  const c = Array.isArray(row.center) ? row.center : null;
  return 'l:' + String(row.class_name !== undefined ? row.class_name : '?') + '@' +
    (c ? Math.round(Number(c[0])) + ',' + Math.round(Number(c[1])) : '?,?');
}

export function unitKeyOf(u) {
  if (!u) return '';
  if (u.isAdd !== undefined && u.isAdd !== null) return 'a:' + String(u.isAdd);
  if (u.track !== undefined && u.track !== null) return 't:' + String(u.track);
  const c = Array.isArray(u.center_px) ? u.center_px : null;
  return 'l:' + String(u.label !== undefined ? u.label : '?') + '@' +
    (c ? Math.round(Number(c[0])) + ',' + Math.round(Number(c[1])) : '?,?');
}

export function editKeyOf(target) {
  if (!target || typeof target !== 'object') return '';
  if (target.track !== undefined && target.track !== null) return 't:' + String(target.track);
  const c = Array.isArray(target.center) ? target.center : null;
  return 'l:' + String(target.label || '?') + '@' +
    (c ? Math.round(Number(c[0])) + ',' + Math.round(Number(c[1])) : '?,?');
}

export function targetForKey(key, frame) {
  const rows = detectionRowsOf(frame);
  for (const row of rows) {
    if (rowKeyOfDetection(row) !== key) continue;
    if (row.track_id !== undefined && row.track_id !== null) return { track: row.track_id };
    return {
      label: String(row.class_name || '?'),
      center: Array.isArray(row.center) ? [Number(row.center[0]), Number(row.center[1])] : [0, 0],
    };
  }
  return null;
}

export function draftFor(frameIndex) {
  const k = String(frameIndex);
  if (!state.correctionDrafts[k]) state.correctionDrafts[k] = { updates: [], deletes: [], adds: [] };
  return state.correctionDrafts[k];
}

// Frame pinned at gesture start (mouseup may fire after polls/seeks moved
// the cursor). Returns null when evicted.
export function frameByIndex(frameIndex) {
  if (frameIndex === undefined || frameIndex === null) return null;
  const want = String(frameIndex);
  for (const f of state.history) {
    if (f && String(f.frame_index) === want) return f;
  }
  return null;
}

export function draftUpdateFor(draft, key) {
  if (!draft) return null;
  return draft.updates.find((e) => editKeyOf(e.target) === key) || null;
}

// Draft adds carry stable ids ('a:<id>') so deleting one add never renumbers
// the keys of the others. Legacy adds without an id fall back to their array
// index.
export function addKeyOf(a, i) {
  return 'a:' + String(a && a.id !== undefined && a.id !== null ? a.id : i);
}

export function addByKey(draft, key) {
  if (!draft || typeof key !== 'string' || key.slice(0, 2) !== 'a:') return null;
  const id = key.slice(2);
  for (let i = 0; i < draft.adds.length; i++) {
    const a = draft.adds[i];
    if (!a) continue;
    if (a.id !== undefined && a.id !== null) {
      if (String(a.id) === id) return { add: a, index: i };
    } else if (String(i) === id) {
      return { add: a, index: i };
    }
  }
  return null;
}

export function draftSummary(draft) {
  const moves = draft.updates.filter((e) => e && e.box).length;
  const relabels = draft.updates.filter((e) => e && !e.box).length;
  const d = draft.deletes.length, a = draft.adds.length;
  if (!relabels && !moves && !d && !a) return 'No edits yet — drag a box, relabel, delete, or draw a box.';
  const bits = [];
  if (relabels) bits.push(relabels + ' relabel' + (relabels > 1 ? 's' : ''));
  if (moves) bits.push(moves + ' move' + (moves > 1 ? 's' : ''));
  if (d) bits.push(d + ' delete' + (d > 1 ? 's' : ''));
  if (a) bits.push(a + ' added box' + (a > 1 ? 'es' : ''));
  return bits.join(' · ') + ' — Re-evaluate to preview.';
}

export function setCorrectionStatus(text, isError) {
  const el = els['correction-status'];
  if (!el) return;
  if (!text) { el.hidden = true; el.textContent = ''; return; }
  el.hidden = false;
  el.textContent = text;
  el.style.color = isError ? 'var(--red)' : '';
}

export function isCorrectionStatusText(text) {
  return /re-evaluat|what-if|no edits yet|re-evaluating|revised|no frame selected|revert failed/i
    .test(String(text || ''));
}

export function clearStaleCorrectionStatus() {
  const el = els['correction-status'];
  if (!el || el.hidden) return;
  if (isCorrectionStatusText(el.textContent)) setCorrectionStatus('');
}

export function originalOf(frame, key) {
  const rows = detectionRowsOf(frame);
  for (const row of rows) {
    if (rowKeyOfDetection(row) === key) return row;
  }
  return null;
}

export function upsertUpdate(frame, key, patch) {
  const draft = draftFor(frame.frame_index);
  const target = targetForKey(key, frame);
  if (!target) return;
  const orig = originalOf(frame, key) || {};
  let entry = draftUpdateFor(draft, key);
  if (!entry) {
    entry = { target };
    draft.updates.push(entry);
  }
  if (patch.class_name !== undefined) {
    if (patch.class_name === orig.class_name) delete entry.class_name;
    else entry.class_name = patch.class_name;
  }
  if (patch.team !== undefined) {
    if (patch.team === orig.team) delete entry.team;
    else entry.team = patch.team;
  }
  if (patch.box !== undefined) {
    const same = Array.isArray(orig.box) && orig.box.length >= 4 &&
      orig.box.slice(0, 4).every((v, i) => Number(v) === Number(patch.box[i]));
    if (same) delete entry.box;
    else entry.box = patch.box.slice(0, 4).map(Number);
  }
  if (!entry.class_name && !entry.team && !entry.box) {
    draft.updates = draft.updates.filter((e) => e !== entry);
  }
  setCorrectionStatus(draftSummary(draft));
  renderCurrent();
}

export function toggleDelete(frame, key) {
  const draft = draftFor(frame.frame_index);
  if (key.slice(0, 2) === 'a:') {
    const found = addByKey(draft, key);
    if (found) {
      draft.adds.splice(found.index, 1);
      if (state.selection === key) { state.selection = null; hideFloatbar(); }
      setCorrectionStatus(draftSummary(draft));
      renderCurrent();
    }
    return;
  }
  const at = draft.deletes.findIndex((d) => editKeyOf(d.target) === key);
  if (at >= 0) draft.deletes.splice(at, 1);
  else {
    const target = targetForKey(key, frame);
    if (!target) return;
    draft.deletes.push({ target });
  }
  setCorrectionStatus(draftSummary(draft));
  renderCurrent();
}

export function setEditMode(on) {
  state.editMode = !!on;
  const btn = els['btn-edit-labels'];
  if (btn) {
    btn.classList.toggle('is-on', state.editMode);
    btn.setAttribute('aria-pressed', String(state.editMode));
  }
  const wrap = els['frame-wrap'];
  if (wrap) wrap.classList.toggle('editing', state.editMode);
  state.selection = null;
  state.dragBox = null;
  state.dragMove = null;
  hideFloatbar();
  if (canvas) canvas.style.cursor = '';
  if (state.editMode) {
    const frame = currentFrame();
    if (!frame || !frame.emitted || !frame.in_game) {
      setCorrectionStatus('Edit mode needs an emitted in-game frame — seek to one first.', true);
    } else {
      if (state.playing) {
        state.playing = false;
        if (els['btn-play']) {
          els['btn-play'].textContent = 'Play';
          els['btn-play'].setAttribute('aria-pressed', 'false');
        }
      }
      const n = detectionRowsOf(frame).length;
      setCorrectionStatus(
        'Edit mode: drag a box to move, drag a corner to resize, click to select, drag empty area to add.' +
        (n ? '' : ' No detections retained on this frame — you can still add boxes.')
      );
    }
  } else {
    setCorrectionStatus('');
  }
  renderCurrent();
}

export function editableFrame() {
  if (!state.editMode) return null;
  const frame = currentFrame();
  if (!frame || !frame.emitted || !frame.in_game) return null;
  return frame;
}

// Retained rows with the draft applied (updates override box/class/team,
// deletes flagged, adds appended). Every row carries effKey: the retained
// detection key, or 'a:<i>' for draft adds.
export function effectiveRows(frame) {
  const rows = detectionRowsOf(frame).map((r) => Object.assign({}, r));
  const byKey = {};
  rows.forEach((r) => {
    r.effKey = rowKeyOfDetection(r);
    byKey[r.effKey] = r;
  });
  const draft = frame ? state.correctionDrafts[String(frame.frame_index)] : null;
  const deleted = new Set();
  if (draft) {
    for (const e of draft.updates) {
      const r = byKey[editKeyOf(e.target)];
      if (!r) continue;
      if (e.class_name) r.class_name = e.class_name;
      if (e.team) r.team = e.team;
      if (e.box) {
        r.box = e.box.slice();
        r.center = [(e.box[0] + e.box[2]) / 2, (e.box[1] + e.box[3]) / 2];
      }
    }
    for (const d of draft.deletes) deleted.add(editKeyOf(d.target));
    draft.adds.forEach((a, i) => {
      if (!a || !Array.isArray(a.box)) return;
      const r = {
        class_name: a.class_name, team: a.team, confidence: 1,
        track_id: null, box: a.box.slice(),
        center: [(a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2],
        effKey: addKeyOf(a, i),
      };
      rows.push(r);
      byKey[r.effKey] = r;
    });
  }
  // In-progress drags preview on top of the draft.
  if (frame && state.dragMove && state.dragMove.curBox) {
    const r = byKey[state.dragMove.key];
    if (r && Array.isArray(r.box)) {
      r.box = state.dragMove.curBox.slice();
      r.center = [(r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2];
    }
  }
  return { rows, byKey, deleted };
}

export function hitDetection(frame, rect, x, y) {
  return hitDetectionIn(cachedEffectiveRows(frame), frame, rect, x, y);
}

// Cache + rAF coalescing for hover: mousemove fires far more often than
// paints, and effectiveRows copies every row per call.
let hoverCache = { key: null, eff: null };
let hoverRaf = 0;
let hoverPending = null;

export function effectiveRowsKey(frame) {
  const draft = frame ? state.correctionDrafts[String(frame.frame_index)] : null;
  let sig = '';
  try { sig = JSON.stringify(draft || null); } catch (e) { sig = ''; }
  return String(frame ? frame.frame_index : '?') + '|' + sig + '|' +
    (state.dragMove && state.dragMove.curBox
      ? state.dragMove.key + JSON.stringify(state.dragMove.curBox) : '-');
}

export function cachedEffectiveRows(frame) {
  const key = effectiveRowsKey(frame);
  if (hoverCache.key !== key) {
    hoverCache = { key, eff: effectiveRows(frame) };
  }
  return hoverCache.eff;
}

export function hitDetectionIn(eff, frame, rect, x, y) {
  const CORNER_R = 10;
  let bodyHit = null;
  for (const row of eff.rows) {
    const key = row.effKey;
    if (eff.deleted.has(key)) continue;
    if (!Array.isArray(row.box) || row.box.length < 4) continue;
    const p0 = toDisplay([row.box[0], row.box[1]], rect, frame);
    const p1 = toDisplay([row.box[2], row.box[3]], rect, frame);
    if (!p0 || !p1) continue;
    const corners = { nw: p0, ne: { x: p1.x, y: p0.y }, sw: { x: p0.x, y: p1.y }, se: p1 };
    for (const name of ['nw', 'ne', 'sw', 'se']) {
      const c = corners[name];
      if (Math.abs(x - c.x) <= CORNER_R && Math.abs(y - c.y) <= CORNER_R) {
        return { key, row, part: name, box: row.box.slice() };
      }
    }
    if (x >= p0.x - 6 && x <= p1.x + 6 && y >= p0.y - 6 && y <= p1.y + 6) {
      const area = Math.max(1, (p1.x - p0.x) * (p1.y - p0.y));
      if (!bodyHit || area < bodyHit.area) bodyHit = { key, row, part: 'move', box: row.box.slice(), area };
    }
  }
  return bodyHit;
}

export function upsertUpdateBox(frame, key, box) {
  if (key.slice(0, 2) === 'a:') {
    // Resize/move of a draft add edits the add in place.
    const draft = draftFor(frame.frame_index);
    const found = addByKey(draft, key);
    if (found && Array.isArray(found.add.box)) {
      found.add.box = box.slice();
      setCorrectionStatus(draftSummary(draft));
      renderCurrent();
    }
    return;
  }
  upsertUpdate(frame, key, { box: box.slice() });
}

export function updateAddLabel(frame, key, label) {
  const draft = draftFor(frame.frame_index);
  const found = addByKey(draft, key);
  if (!found || !label) return;
  found.add.class_name = label;
  setCorrectionStatus(draftSummary(draft));
  renderCurrent();
}

export function toggleTeamForKey(frame, key) {  if (key.slice(0, 2) === 'a:') {
    const draft = draftFor(frame.frame_index);
    const found = addByKey(draft, key);
    if (found) {
      found.add.team = found.add.team === 'ally' ? 'enemy' : 'ally';
      setCorrectionStatus(draftSummary(draft));
      renderCurrent();
    }
    return;
  }
  const rows = detectionRowsOf(frame);
  let orig = null;
  for (const row of rows) {
    if (rowKeyOfDetection(row) === key) { orig = row; break; }
  }
  const cur = draftUpdateFor(draftFor(frame.frame_index), key);
  const shown = (cur && cur.team) || (orig && orig.team) || 'enemy';
  upsertUpdate(frame, key, { team: shown === 'ally' ? 'enemy' : 'ally' });
}

export function ensureFloatbar() {
  let bar = document.getElementById('edit-floatbar');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'edit-floatbar';
    bar.className = 'edit-floatbar';
    bar.hidden = true;
    bar.addEventListener('click', onFloatbarAction);
    if (els['frame-wrap']) els['frame-wrap'].appendChild(bar);
  }
  return bar;
}

export function hideFloatbar() {
  const bar = document.getElementById('edit-floatbar');
  if (bar) bar.hidden = true;
}

export function showFloatbarFor(frame, key) {
  const bar = ensureFloatbar();
  const eff = effectiveRows(frame);
  const row = eff.byKey[key] || null;
  const label = row ? String(row.class_name || '?') : key;
  const team = row ? String(row.team || '?') : '?';
  const isDel = eff.deleted.has(key);
  bar.innerHTML =
    '<span class="elink">' + esc(label) + '</span>' +
    '<button type="button" class="mini" data-fact="team" title="Toggle team">' + esc(team) + '</button>' +
    '<button type="button" class="mini" data-fact="delete" title="Delete / restore">' + (isDel ? 'keep' : 'del') + '</button>' +
    '<button type="button" class="mini" data-fact="close" title="Deselect">×</button>';
  const rect = containRect();
  let pos = null;
  if (row && Array.isArray(row.box)) {
    pos = toDisplay([row.box[2], row.box[1]], rect, frame);
  }
  if (pos) {
    bar.style.left = Math.max(4, Math.min(pos.x + 8, rect.x + rect.w - 190)) + 'px';
    bar.style.top = Math.max(4, pos.y - 40) + 'px';
  } else {
    bar.style.left = (rect.x + 8) + 'px';
    bar.style.top = (rect.y + 8) + 'px';
  }
  bar.hidden = false;
}

export function onFloatbarAction(ev) {
  const el = ev.target && ev.target.closest ? ev.target.closest('[data-fact]') : null;
  if (!el) return;
  const frame = currentFrame();
  const key = state.selection;
  if (!frame || !key) return;
  const act = el.dataset.fact;
  if (act === 'team') toggleTeamForKey(frame, key);
  else if (act === 'delete') toggleDelete(frame, key);
  else if (act === 'close') {
    state.selection = null;
    hideFloatbar();
    renderCurrent();
    return;
  }
  // Refresh the toolbar contents (team label / del state may have flipped).
  showFloatbarFor(frame, state.selection);
}

export function selectDetection(frame, key, scroll) {
  state.selection = key;
  showFloatbarFor(frame, key);
  renderCurrent();
  if (scroll) {
    // Compare dataset values directly: CSS.escape() is for identifiers, not
    // quoted attribute strings, so a querySelector with an escaped key
    // (colons, @, commas, spaces) never matches.
    const rows = els['detected-objects']
      ? els['detected-objects'].querySelectorAll('li[data-ckey]')
      : [];
    for (const li of rows) {
      if (li.dataset && li.dataset.ckey === key) {
        if (li.scrollIntoView) li.scrollIntoView({ block: 'nearest' });
        break;
      }
    }
  }
}

export function onCorrectionControl(ev) {
  const expand = ev.target && ev.target.closest ? ev.target.closest('[data-expand]') : null;
  if (expand) {
    state.showAllDetections = expand.dataset.expand === 'more';
    renderCurrent();
    return;
  }
  const li = ev.target && ev.target.closest ? ev.target.closest('li[data-ckey]') : null;
  const frame = currentFrame();
  if (!li || !frame) return;
  const key = li.dataset.ckey;
  // Plain row click selects on canvas (entering edit mode if needed).
  const el = ev.target && ev.target.closest ? ev.target.closest('[data-cact]') : null;
  if (!el) {
    if (!state.editMode) setEditMode(true);
    if (!editableFrame()) return;
    selectDetection(frame, key, false);
    return;
  }
  const act = el.dataset.cact;
  if (act === 'label' && ev.type === 'change') {
    if (key.slice(0, 2) === 'a:') updateAddLabel(frame, key, el.value);
    else upsertUpdate(frame, key, { class_name: el.value });
  } else if (act === 'team') {
    if (key.slice(0, 2) === 'a:') {
      // Draft adds have no retained row: upsertUpdate's targetForKey would
      // find nothing and silently drop the toggle.
      toggleTeamForKey(frame, key);
      state.selection = key;
      showFloatbarFor(frame, key);
      return;
    }
    const orig = originalOf(frame, key);
    const cur = draftUpdateFor(draftFor(frame.frame_index), key);
    const shown = (cur && cur.team) || (orig && orig.team) || 'enemy';
    upsertUpdate(frame, key, { team: shown === 'ally' ? 'enemy' : 'ally' });
  } else if (act === 'delete') {
    toggleDelete(frame, key);
    return;
  } else {
    return;
  }
  state.selection = key;
  showFloatbarFor(frame, key);
}

export function refreshCorrectionControls(frame) {
  const btn = els['btn-revert-correction'];
  if (btn) btn.hidden = !(frame && frame.corrected);
  const draft = frame ? state.correctionDrafts[String(frame.frame_index)] : null;
  const pending = draft && (draft.updates.length || draft.deletes.length || draft.adds.length);
  if (pending) {
    // Unsent edits take status precedence over an active what-if.
    setCorrectionStatus(draftSummary(draft));
    return;
  }
  if (!frame || !frame.corrected) {
    // Neither draft nor what-if on this frame: drop the previous frame's
    // correction text instead of leaving it stale. Non-correction hints
    // (edit-mode guidance, draw prompts) are left alone.
    clearStaleCorrectionStatus();
    return;
  }
  const applied = frame.corrected.applied && frame.corrected.applied.counts;
  const bits = applied
    ? [applied.updated + ' relabel', applied.deleted + ' delete', applied.added + ' add']
      .filter((_, i) => [applied.updated, applied.deleted, applied.added][i] > 0).join(' · ')
    : '';
  setCorrectionStatus('What-if active' + (bits ? ' (' + bits + ')' : '') + ' — trackers & timeline unchanged.');
}

export async function submitCorrection() {
  const frame = currentFrame();
  if (!frame) { setCorrectionStatus('No frame selected.', true); return; }
  if (!frame.emitted || !frame.in_game) {
    setCorrectionStatus('Only emitted in-game frames can be revised.', true);
    return;
  }
  const draft = draftFor(frame.frame_index);
  if (!draft.updates.length && !draft.deletes.length && !draft.adds.length) {
    setCorrectionStatus('No edits yet — relabel, delete, or draw a box.', true);
    return;
  }
  setCorrectionStatus('Re-evaluating…');
  try {
    const data = await apiPost('/api/frame/' + frame.frame_index + '/reevaluate', {
      updates: draft.updates,
      deletes: draft.deletes,
      adds: draft.adds,
    });
    frame.corrected = data && data.corrected ? data.corrected : null;
    // The draft is now applied server-side: drop it so the status shows the
    // fresh what-if instead of "Re-evaluate to preview" over it.
    delete state.correctionDrafts[String(frame.frame_index)];
    setCorrectionStatus('What-if applied — trackers & timeline unchanged.');
    renderCurrent();
  } catch (err) {
    setCorrectionStatus('Re-evaluate failed: ' + String((err && err.message) || err), true);
    showError(String((err && err.message) || err));
  }
}

export async function revertCorrection() {
  const frame = currentFrame();
  if (!frame) return;
  try {
    await apiDelete('/api/frame/' + frame.frame_index + '/reevaluate');
  } catch (err) {
    setCorrectionStatus('Revert failed: ' + String((err && err.message) || err), true);
    return;
  }
  frame.corrected = null;
  delete state.correctionDrafts[String(frame.frame_index)];
  setCorrectionStatus('');
  renderCurrent();
}

export async function loadLabels() {
  try {
    const data = await apiGet('/api/labels');
    state.labels = data && Array.isArray(data.labels) ? data.labels : [];
  } catch (err) {
    state.labels = [];
  }
  const select = els['select-correct-label'];
  if (select) {
    select.textContent = '';
    for (const name of state.labels) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    }
  }
}

export function bindCorrectionEvents() {
  // Label correction controls (delegated: rows re-render on every poll).
  if (els['detected-objects']) {
    els['detected-objects'].addEventListener('change', onCorrectionControl);
    els['detected-objects'].addEventListener('click', onCorrectionControl);
  }
  if (els['btn-reevaluate']) {
    els['btn-reevaluate'].addEventListener('click', submitCorrection);
  }
  if (els['btn-revert-correction']) {
    els['btn-revert-correction'].addEventListener('click', revertCorrection);
  }
  if (els['btn-edit-labels']) {
    els['btn-edit-labels'].addEventListener('click', () => setEditMode(!state.editMode));
  }
  if (els['btn-add-box']) {
    // Explicit entry to box drawing: enters edit mode and tells the user to
    // drag on the frame. Drawing itself needs no arming — any empty-area drag
    // in edit mode creates the box with the picked label/team.
    els['btn-add-box'].addEventListener('click', () => {
      setEditMode(true);
      const label = els['select-correct-label'] ? els['select-correct-label'].value : '';
      setCorrectionStatus(
        'Drag a box on the frame for "' + (label || 'the picked label') + '". ' +
        'Drag a detection to move it, a corner to resize, click to select.'
      );
    });
  }
}

export function bindCanvasEvents() {
  if (!canvas) return;
  canvas.addEventListener('mousedown', (ev) => {
    // ROI review gestures take precedence while a proposal is displayed.
    if (ev.button === 0 && roiPreviewVisible() && roiPreviewImageReady()) {
      roiCanvasMouseDown(ev);
      return;
    }
    const frame = editableFrame();
    if (!frame || ev.button !== 0) return;
    // Block native image drag/selection first: otherwise the browser grabs
    // the underlying <img> and the drag never reaches the box logic.
    ev.preventDefault();
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    const rect = containRect();
    const hit = hitDetection(frame, rect, x, y);
    hideFloatbar();
    if (hit) {
      state.selection = hit.key;
      state.dragMove = {
        key: hit.key, part: hit.part,
        startX: x, startY: y, origBox: hit.box.slice(), curBox: hit.box.slice(),
        moved: false, fi: frame.frame_index,
      };
    } else {
      state.selection = null;
      state.dragBox = { x0: x, y0: y, x1: x, y1: y, fi: frame.frame_index };
    }
    renderCurrent();
  });
  canvas.addEventListener('mousemove', (ev) => {
    const frame = editableFrame();
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    if (state.roiAdapt.dragRoi) {
      if (roiPreviewImageReady()) updateRoiDrag(x, y);
      return;
    }
    if (state.dragMove && frame) {
      const dims = frameDims(frame);
      const rect = containRect();
      if (!dims || !rect.w || !rect.h) return;
      const kx = dims.w / rect.w, ky = dims.h / rect.h;
      const dx = (x - state.dragMove.startX) * kx;
      const dy = (y - state.dragMove.startY) * ky;
      if (Math.abs(x - state.dragMove.startX) + Math.abs(y - state.dragMove.startY) > 3) {
        state.dragMove.moved = true;
      }
      const o = state.dragMove.origBox;
      let b;
      switch (state.dragMove.part) {
        case 'nw': b = [o[0] + dx, o[1] + dy, o[2], o[3]]; break;
        case 'ne': b = [o[0], o[1] + dy, o[2] + dx, o[3]]; break;
        case 'sw': b = [o[0] + dx, o[1], o[2], o[3] + dy]; break;
        case 'se': b = [o[0], o[1], o[2] + dx, o[3] + dy]; break;
        default: b = [o[0] + dx, o[1] + dy, o[2] + dx, o[3] + dy]; break;
      }
      // Normalize corners (a corner may cross the opposite one), clamp
      // into the frame, enforce a 4px minimum.
      let [ax0, ay0, ax1, ay1] = [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])];
      ax0 = Math.max(0, Math.min(ax0, dims.w - 4));
      ay0 = Math.max(0, Math.min(ay0, dims.h - 4));
      ax1 = Math.max(ax0 + 4, Math.min(ax1, dims.w));
      ay1 = Math.max(ay0 + 4, Math.min(ay1, dims.h));
      state.dragMove.curBox = [ax0, ay0, ax1, ay1];
      drawOverlay(frame, suggestionsOf(frame), diagnosticsOf(frame));
      return;
    }
    if (state.dragBox && frame) {
      state.dragBox.x1 = x;
      state.dragBox.y1 = y;
      drawOverlay(frame, suggestionsOf(frame), diagnosticsOf(frame));
      return;
    }
    // Hover cursor over detections, rAF-coalesced: mousemove fires far more
    // often than paints, and each hit-test would otherwise copy all rows.
    if (frame) {
      hoverPending = { fi: frame.frame_index, x, y };
      if (!hoverRaf) {
        hoverRaf = requestAnimationFrame(() => {
          hoverRaf = 0;
          const pending = hoverPending;
          hoverPending = null;
          if (!pending) return;
          const f = editableFrame();
          if (!f || f.frame_index !== pending.fi) return;
          const hit = hitDetection(f, containRect(), pending.x, pending.y);
          canvas.style.cursor = hit ? (hit.part === 'move' ? 'move' : 'nwse-resize') : 'crosshair';
        });
      }
    } else if (roiPreviewVisible() && roiPreviewImageReady()) {
      const hit = hitRoi(x, y);
      canvas.style.cursor = hit ? (hit.part === 'move' ? 'move' : 'nwse-resize') : 'default';
    }
  });
  window.addEventListener('mouseup', () => {
    if (state.dragMove) {
      const drag = state.dragMove;
      state.dragMove = null;
      // Commit to the frame the drag started on, not whatever the cursor
      // happens to show now (polls can advance it mid-drag). Discard when
      // edit mode was left mid-drag or the frame was evicted.
      const frame = frameByIndex(drag.fi);
      if (!state.editMode || !frame) {
        renderCurrent();
        return;
      }
      if (drag.moved) {
        upsertUpdateBox(frame, drag.key, drag.curBox);
      } else {
        // Click without drag = select; show the floating actions.
        selectDetection(frame, drag.key, true);
      }
      renderCurrent();
      return;
    }
    if (state.roiAdapt.dragRoi) {
      finishRoiDrag();
      return;
    }
    if (!state.dragBox) return;
    const drag = state.dragBox;
    state.dragBox = null;
    // Same frame pinning as moves: a poll or seek mid-draw must not land the
    // new box on a different frame, and leaving edit mode cancels the draw.
    const frame = frameByIndex(drag.fi);
    if (!state.editMode || !frame) { renderCurrent(); return; }
    const rect = containRect();
    const dims = frameDims(frame);
    if (!dims || !rect.w || !rect.h) { renderCurrent(); return; }
    // A click without a drag is a no-op (selection is handled by mousedown
    // hit-testing), not an erroneous tiny box.
    if (Math.abs(drag.x1 - drag.x0) < 4 && Math.abs(drag.y1 - drag.y0) < 4) {
      renderCurrent();
      return;
    }
    const dx0 = Math.min(drag.x0, drag.x1), dx1 = Math.max(drag.x0, drag.x1);
    const dy0 = Math.min(drag.y0, drag.y1), dy1 = Math.max(drag.y0, drag.y1);
    // Wholly outside the frame image (e.g. in the letterbox gutters).
    if (dx1 < rect.x || dx0 > rect.x + rect.w || dy1 < rect.y || dy0 > rect.y + rect.h) {
      setCorrectionStatus('Box is outside the frame image — draw on the video.', true);
      renderCurrent();
      return;
    }
    // Clamp into the frame image so drags starting in the letterbox
    // gutters slide onto the video instead of being silently dropped.
    const clamp01 = (v) => Math.max(0, Math.min(1, v));
    const fx0 = (dx0 - rect.x) / rect.w;
    const fy0 = (dy0 - rect.y) / rect.h;
    const fx1 = (dx1 - rect.x) / rect.w;
    const fy1 = (dy1 - rect.y) / rect.h;
    if (![fx0, fy0, fx1, fy1].every(Number.isFinite)) { renderCurrent(); return; }
    const p0 = [clamp01(fx0) * dims.w, clamp01(fy0) * dims.h];
    const p1 = [clamp01(fx1) * dims.w, clamp01(fy1) * dims.h];
    if (p1[0] - p0[0] < 4 || p1[1] - p0[1] < 4) {
      setCorrectionStatus('Box is too small — drag a larger box on the video.', true);
      renderCurrent();
      return;
    }
    const label = els['select-correct-label'] ? els['select-correct-label'].value : '';
    const team = els['select-correct-team'] ? els['select-correct-team'].value : 'enemy';
    if (!label) {
      setCorrectionStatus('Cannot add a box — no label selected (label list unavailable?).', true);
      renderCurrent();
      return;
    }
    draftFor(frame.frame_index).adds.push({
      id: state.addSeq++,
      box: [p0[0], p0[1], p1[0], p1[1]],
      class_name: label,
      team: team || 'enemy',
    });
    setCorrectionStatus(draftSummary(draftFor(frame.frame_index)));
    renderCurrent();
  });
  window.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && state.roiAdapt.selRoi && roiPreviewVisible()) {
      state.roiAdapt.selRoi = null;
      drawRoiPreviewBoxes();
      return;
    }
    if (!state.editMode || !state.selection) return;
    const tag = ev.target && ev.target.tagName ? String(ev.target.tagName).toLowerCase() : '';
    if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
    const frame = currentFrame();
    if (!frame) return;
    if (ev.key === 'Delete' || ev.key === 'Backspace') {
      toggleDelete(frame, state.selection);
      ev.preventDefault();
    } else if (ev.key === 't' || ev.key === 'T') {
      toggleTeamForKey(frame, state.selection);
      ev.preventDefault();
    } else if (ev.key === 'Escape') {
      state.selection = null;
      hideFloatbar();
      renderCurrent();
    }
  });
}
