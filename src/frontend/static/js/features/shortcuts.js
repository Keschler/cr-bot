/* Global keyboard shortcuts + help modal. Capture-phase listener so an
 * open modal can swallow keys before canvas/slider handlers see them.
 * All shortcuts no-op while typing in inputs, and never fight the
 * timeline slider's own Arrow/Home/End handling when it is focused.
 *
 * Shortcut map: Space play/pause · ←/→ or j/l step (Shift = 10) ·
 * Home/End jump · e edit mode · 1/2/3 inspect rank · ? help · Esc close.
 */

import { state } from '../state/store.js';
import { els } from '../utils/elements.js';
import { seek } from './timeline.js';
import { scheduleFrames } from './session.js';
import { setEditMode } from './corrections.js';
import { toggleInspectRank } from './panels.js';

export function isHelpOpen() {
  const modal = els['help-modal'];
  return !!(modal && !modal.hidden);
}

export function openHelp() {
  const modal = els['help-modal'];
  if (!modal) return;
  modal.hidden = false;
  if (els['btn-help-close']) els['btn-help-close'].focus();
}

export function closeHelp() {
  const modal = els['help-modal'];
  if (!modal) return;
  modal.hidden = true;
  if (els['btn-help']) els['btn-help'].focus();
}

export function onShortcutKey(ev) {
  if (isHelpOpen()) {
    // Freeze the app behind the modal (never preventDefault here, so
    // browser keys like F5 keep working); Escape closes it.
    if (ev.key === 'Escape') {
      ev.preventDefault();
      ev.stopPropagation();
      closeHelp();
    } else {
      ev.stopPropagation();
    }
    return;
  }
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return; // keep browser shortcuts
  const target = ev.target;
  const tag = target && target.tagName ? String(target.tagName).toLowerCase() : '';
  const typing = tag === 'input' || tag === 'select' || tag === 'textarea' ||
    !!(target && target.isContentEditable);
  const key = ev.key;
  if (key === '?' && !typing) {
    ev.preventDefault();
    openHelp();
    return;
  }
  if (typing) return;
  if (tag === 'button' && (key === ' ' || key === 'Spacebar')) return; // let focused buttons activate
  const sliderFocused = document.activeElement === els['timeline-track'];
  const step = ev.shiftKey ? 10 : 1;
  if (key === ' ' || key === 'Spacebar') {
    if (!state.history.length) return;
    ev.preventDefault();
    if (state.playing) {
      seek(state.cursor); // seek pauses (polling continues, cursor sticks)
    } else {
      state.playing = true;
      if (els['btn-play']) {
        els['btn-play'].textContent = 'Pause';
        els['btn-play'].setAttribute('aria-pressed', 'true');
      }
      scheduleFrames();
    }
  } else if (key === 'ArrowLeft' || key === 'j' || key === 'J') {
    if (key.startsWith('Arrow') && sliderFocused) return; // slider handles it
    if (!state.history.length) return;
    ev.preventDefault();
    seek(state.cursor - step);
  } else if (key === 'ArrowRight' || key === 'l' || key === 'L') {
    if (key.startsWith('Arrow') && sliderFocused) return;
    if (!state.history.length) return;
    ev.preventDefault();
    seek(state.cursor + step);
  } else if (key === 'Home') {
    if (sliderFocused) return;
    if (!state.history.length) return;
    ev.preventDefault();
    seek(0);
  } else if (key === 'End') {
    if (sliderFocused) return;
    if (!state.history.length) return;
    ev.preventDefault();
    seek(state.history.length - 1);
  } else if (key === 'e' || key === 'E') {
    setEditMode(!state.editMode);
  } else if (key === '1' || key === '2' || key === '3') {
    if (!state.history.length) return;
    toggleInspectRank(parseInt(key, 10) - 1);
  }
}

export function bindShortcutEvents() {
  window.addEventListener('keydown', onShortcutKey, true);
  if (els['btn-help']) els['btn-help'].addEventListener('click', openHelp);
  if (els['btn-help-close']) els['btn-help-close'].addEventListener('click', closeHelp);
  const modal = els['help-modal'];
  if (modal) {
    modal.addEventListener('click', (ev) => {
      if (ev.target === modal) closeHelp(); // backdrop click
    });
  }
}
