/* Shared DOM element handles. Evaluated once at startup (module scripts
 * are deferred, so the DOM is parsed before this runs). Canvas 2D context
 * included: overlay drawing modules share the single context.
 */

export const $ = (id) => document.getElementById(id);

export const els = {};
[
  'status-pill', 'error-bar', 'btn-help', 'help-modal', 'btn-help-close',
  'tab-video', 'tab-live', 'btn-open-replay', 'btn-dashboard',
  'panel-video', 'panel-live',
  'btn-session-toggle', 'btn-stop-mini', 'session-mini',
  'input-video-file', 'btn-browse-video', 'video-file-label', 'video-file-meta', 'select-checkpoint',
  'input-start-frame', 'input-stride', 'input-max-frames', 'select-device',
  'btn-start-video', 'btn-stop-video', 'recent-sessions',
  'check-adapt-rois', 'adapt-notice',
  'roi-preview-meta', 'btn-reset-rois',
  'btn-another-frame',
  'input-serial', 'select-transport', 'select-live-checkpoint', 'input-calibration',
  'select-live-device',
  'btn-edit-labels',
  'check-execute', 'check-confirm-live', 'live-warning',
  'btn-start-live', 'btn-stop-live',
  'meta-replay-id', 'meta-arena', 'meta-tick', 'meta-phase',
  'hand-slots', 'next-card', 'own-elixir-text', 'own-elixir-fill',
  'enemy-elixir-text', 'enemy-elixir-fill',
  'tower-rows', 'detected-objects', 'detection-count',
  'action-history', 'action-history-count',
  'select-correct-label', 'select-correct-team',
  'btn-add-box', 'btn-reevaluate', 'btn-revert-correction', 'correction-status',
  'badge-ingame', 'badge-perf', 'badge-arena',
  'toggle-boxes', 'toggle-grid', 'toggle-labels',
  'frame-wrap', 'center-frame', 'frame-overlay', 'frame-empty', 'history-badge',
  'btn-play', 'btn-prev', 'btn-next', 'time-label', 'speed-select',
  'btn-export-json', 'btn-export-csv', 'btn-save-frame', 'btn-copy-link',
  'suggestions', 'reason-text', 'reason-entropy', 'reason-mode-probs',
  'reason-hand-stable', 'reason-confidence', 'reason-timing', 'reason-devices',
  'timeline-label', 'timeline-track', 'timeline-rail', 'timeline-fill', 'timeline-markers',
  'timeline-cursor', 'timeline-tooltip',
].forEach((id) => { els[id] = $(id); });

export const img = els['center-frame'];
export const canvas = els['frame-overlay'];
// Guarded: a missing canvas element must not throw at import time and take
// the whole UI down with it; drawing modules check for null ctx.
export const ctx = canvas ? canvas.getContext('2d') : null;
