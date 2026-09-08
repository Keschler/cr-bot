/* DOM helpers: escaping, status pill, error bar, toasts. */

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

export function ensureToastStack() {
  let stack = document.getElementById('toast-stack');
  if (!stack) {
    stack = document.createElement('div');
    stack.id = 'toast-stack';
    stack.className = 'toast-stack';
    stack.setAttribute('aria-live', 'polite');
    document.body.appendChild(stack);
  }
  return stack;
}

// Transient notification. kind: 'info' | 'success' | 'error'.
// opts: {timeoutMs (0 = sticky), action: {label, onClick}, progress: bool}.
// Returns {el, dismiss, setProgress(frac)}; click (outside action) dismisses.
export function toast(message, kind, opts) {
  const o = opts || {};
  const stack = ensureToastStack();
  const el = document.createElement('div');
  el.className = 'toast' + (kind ? ' toast-' + kind : '');
  el.setAttribute('role', 'status');
  const span = document.createElement('span');
  span.className = 'toast-msg';
  span.textContent = String(message === undefined || message === null ? '' : message);
  el.appendChild(span);
  let bar = null;
  if (o.progress) {
    const track = document.createElement('div');
    track.className = 'toast-progress';
    bar = document.createElement('div');
    track.appendChild(bar);
    el.appendChild(track);
  }
  let timer = 0;
  const dismiss = () => {
    if (timer) { clearTimeout(timer); timer = 0; }
    if (el.parentNode) el.parentNode.removeChild(el);
  };
  if (o.action && o.action.label) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'toast-action';
    btn.textContent = String(o.action.label);
    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      try { if (o.action.onClick) o.action.onClick(); } finally { dismiss(); }
    });
    el.appendChild(btn);
  }
  el.addEventListener('click', (ev) => {
    if (ev.target && ev.target.closest && ev.target.closest('.toast-action')) return;
    dismiss();
  });
  stack.appendChild(el);
  while (stack.children.length > 4) stack.removeChild(stack.firstChild);
  const timeout = o.timeoutMs !== undefined && o.timeoutMs !== null
    ? o.timeoutMs : (kind === 'error' ? 9000 : 5000);
  if (timeout > 0) timer = setTimeout(dismiss, timeout);
  return {
    el,
    dismiss,
    setProgress(frac) {
      if (!bar) return;
      const f = Number(frac);
      bar.style.width = (Number.isFinite(f) ? Math.max(0, Math.min(1, f)) * 100 : 0).toFixed(1) + '%';
    },
  };
}
