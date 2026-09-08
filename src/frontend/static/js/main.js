/* Arena Replay Analyst v1.2.0 — vanilla frontend, ES-module entry point.
 * All data comes from the backend API (see ../../README.md). No mocks.
 *
 * Module map:
 *   state/store.js        shared mutable UI state + constants
 *   utils/elements.js     cached DOM handles + canvas context
 *   utils/dom.js          escaping, status pill, error bar
 *   utils/format.js       pure value/card-name formatting
 *   utils/geometry.js     canvas/coordinate mapping
 *   api/client.js         fetch wrappers (GET/POST/DELETE)
 *   features/frames.js    frame-shape accessors + top-3 ordering
 *   features/roi.js       ROI-adapt pure payload helpers
 *   features/roi-editor.js ROI proposal review + drag-to-adjust
 *   features/session.js   polling, center image, session controls, video/live
 *   features/panels.js    left extracted-state + right suggestions
 *   features/overlay.js   center badges + canvas overlay layers
 *   features/corrections.js label-correction drafts, edit mode, canvas gestures
 *   features/timeline.js  history, markers, scrubbing, renderCurrent fan-out
 */

import { img } from './utils/elements.js';
import { decodeHash } from './utils/share.js';
import {
  setMode, bindImageEvents, bindSessionEvents, applyPersistedSettings,
  loadCheckpoints, loadServerCapabilities, loadGrid, loadRecentSessions,
  setPendingDeepLink,
  pollStatus, scheduleFrames, scheduleStatus,
} from './features/session.js';
import { bindExportEvents } from './features/export.js';
import { bindShortcutEvents } from './features/shortcuts.js';
import { bindRoiEvents, roiPreviewVisible, drawRoiPreviewBoxes } from './features/roi-editor.js';
import { bindCorrectionEvents, bindCanvasEvents, loadLabels } from './features/corrections.js';
import { bindPanelsEvents } from './features/panels.js';
import { bindTimelineEvents, renderCurrent } from './features/timeline.js';
import { drawOverlay } from './features/overlay.js';
import { currentFrame } from './features/frames.js';

bindImageEvents();
bindSessionEvents();
bindRoiEvents();
bindCorrectionEvents();
bindCanvasEvents();
bindPanelsEvents();
bindTimelineEvents();
bindExportEvents();
bindShortcutEvents();
applyPersistedSettings();
setPendingDeepLink(decodeHash(window.location.hash));

window.addEventListener('resize', () => {
  drawOverlay(currentFrame());
  // drawOverlay skips ROI review boxes: repaint them so a resize never
  // leaves a stale or missing proposal overlay.
  if (roiPreviewVisible() && img.dataset.preview === 'roi') drawRoiPreviewBoxes();
});

/* ---------- init ---------- */

setMode('video');
renderCurrent();
loadCheckpoints();
loadServerCapabilities();
loadLabels();
loadGrid();
loadRecentSessions();
pollStatus();
scheduleStatus();
scheduleFrames();
