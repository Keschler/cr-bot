/* Frame-shape accessors + shared ordering. Per contract a frame is:
 * {frame_index, timestamp_s, in_game, emitted,
 *  record: {visual_state, action, result},
 *  suggestions: [{kind, card_slot, cell, probability, log_prob, card_name}],
 *  diagnostics}
 */

import { state } from '../state/store.js';
import { num } from '../utils/format.js';

export function visualStateOf(frame) {
  if (!frame) return {};
  if (frame.record && frame.record.visual_state) return frame.record.visual_state;
  return frame.visual_state || {};
}

export function suggestionsOf(frame) {
  if (!frame) return [];
  // A what-if correction overwrites the displayed suggestions; the originals
  // stay on the frame object underneath (reverted via DELETE).
  if (frame.corrected && Array.isArray(frame.corrected.suggestions)) return frame.corrected.suggestions;
  if (Array.isArray(frame.suggestions)) return frame.suggestions;
  if (frame.record && Array.isArray(frame.record.suggestions)) return frame.record.suggestions;
  return [];
}

export function originalSuggestionsOf(frame) {
  // The pre-correction suggestions underneath a what-if (for diff display).
  if (!frame) return [];
  if (Array.isArray(frame.suggestions)) return frame.suggestions;
  if (frame.record && Array.isArray(frame.record.suggestions)) return frame.record.suggestions;
  return [];
}

export function diagnosticsOf(frame) {
  if (!frame) return {};
  if (frame.corrected && frame.corrected.diagnostics && typeof frame.corrected.diagnostics === 'object') return frame.corrected.diagnostics;
  if (frame.diagnostics && typeof frame.diagnostics === 'object') return frame.diagnostics;
  if (frame.record && frame.record.diagnostics) return frame.record.diagnostics;
  return {};
}

export function actionOf(frame) {
  if (!frame) return null;
  if (frame.record && frame.record.action) return frame.record.action;
  return frame.action || null;
}

export function currentFrame() {
  if (state.cursor < 0 || state.cursor >= state.history.length) return null;
  return state.history[state.cursor];
}

export function isLiveEdge() {
  return state.history.length > 0 && state.cursor === state.history.length - 1;
}

export function topSuggestions(sug) {
  // Single ordering for the top-3: canvas markers, side panel, and hand
  // highlight must agree even if the backend ever returns unsorted input.
  return (sug || []).slice()
    .sort((a, b) => (num(b && b.probability) || 0) - (num(a && a.probability) || 0))
    .slice(0, 3);
}
