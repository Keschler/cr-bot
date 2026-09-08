/* DOM helpers: escaping, status pill, error bar. */

import { els } from './elements.js';

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

export function cssEscapeAttr(s) {
  const v = String(s);
  if (typeof CSS !== 'undefined' && CSS && typeof CSS.escape === 'function') return CSS.escape(v);
  return v.replace(/["\\]/g, '\\$&');
}

export function showError(msg) {
  if (!msg) {
    els['error-bar'].hidden = true;
    els['error-bar'].textContent = '';
    return;
  }
  els['error-bar'].hidden = false;
  els['error-bar'].textContent = msg;
}

export function setPill(text, kind, title) {
  const pill = els['status-pill'];
  pill.textContent = text;
  pill.className = 'status-pill ' + (kind || 'is-idle');
  if (title) pill.title = title;
}
