/* Arena Replay Analyst — shared mutable UI state + constants.
 *
 * Single ownership point for mutable state: every feature module imports
 * { state } from here instead of keeping its own copy. ES-module live
 * bindings keep all importers in sync. All data comes from the backend
 * API (see ../../README.md). No mocks.
 */

export const GRID_COLS = 18;
export const GRID_ROWS = 32;
// Authoritative grid geometry lives in cr_bot/features/action_space.py
// (ACTION_GRID over the KataCR 568x896 arena crop). The backend serves it at
// GET /api/grid; this fallback mirrors those exact constants so the overlay
// keeps working if that fetch fails.
export const GRID_SPEC_FALLBACK = {
  cols: 18, rows: 32,
  x0: -0.9320463320463317 / 568, y0: 72.54622356495467 / 896,
  x1: 569.2610038610038 / 568, y1: 879.9748640483384 / 896,
};
// Extractor hand-slot ROIs (src/cr_bot/domain/rois.py) as (x, y, w, h) in
// the 1080x2400 reference space. The Inspect arrow starts at the played
// card's real position in the video frame.
export const ROI_REF_W = 1080, ROI_REF_H = 2400;
export const NATIVE_SIZE = [1080, 2400];
export const HAND_SLOT_ROIS = [
  { x: 230, y: 2020, w: 220, h: 300 },
  { x: 430, y: 2020, w: 220, h: 300 },
  { x: 630, y: 2020, w: 220, h: 300 },
  { x: 840, y: 2020, w: 220, h: 300 },
];
export const KING_TOWER_MAX = 7032;
export const PRINCESS_TOWER_MAX = 4424;
// Full-HP reference values mirror cr_bot/domain/constants.py
// (PRINCESS_TOWER_HP / KING_TOWER_HP). GET /api/grid also serves them and
// takes precedence when available (see towerMax / loadGrid).

export const state = {
  mode: 'video', // 'video' | 'live'
  playing: true,
  speed: 1,
  toggles: { boxes: true, grid: true, labels: true },
  history: [], // frames from GET /api/frames, ascending by frame_index
  cursor: -1, // index into history; -1 = empty
  lastSince: -1, // last seen frame_index for GET /api/frames (strictly-greater cursor; -1 fetches frame 0)
  sessionLabel: '',
  framesTimer: null,
  uploadedVideoPath: '',
  uploadedVideoName: '',
  uploadedVideoFrames: 0, // total frames from /api/video/info (0 = unknown)
  selectedRank: null, // inspected suggestion rank (0-based into top-3) or null
  gridSpec: null, // fetched from GET /api/grid; fallback used when missing
  towerMax: null, // {princess, king} from GET /api/grid; fallback consts above
  sessionSummary: null, // latest GET /api/status summary (devices, timing) for the inspector
  sessionPin: null, // manual session collapse override: true/false or null (auto)
  showAllDetections: false, // expanded detections list (toggled via "+N more")
  addSeq: 1, // stable per-session id source for draft adds ('a:<id>')
  roiAdapt: { available: false, checked: false, preview: null,
    edits: {}, selRoi: null, dragRoi: null, image: null, dims: null,
    probeFrame: null, probeCursor: 0, showInMiddle: false, loading: false },
};

state.correctionDrafts = {}; // frame_index -> {updates:[], deletes:[], adds:[]}
state.labels = []; // unit-class vocabulary from GET /api/labels
state.editMode = false; // label edit mode: drag to move/resize/add
state.selection = null; // detection key selected on canvas
state.dragBox = null; // display-px rubber band while drawing
state.dragMove = null; // {key, part, startX/Y, origBox, curBox, moved}

// Persisted UI settings (localStorage, versioned key so future schemas can
// migrate). Only plain UI preferences — never session frames or drafts.
export const SETTINGS_KEY = 'ara-settings-v1';

export function loadPersistedSettings() {
  try {
    const raw = window.localStorage.getItem(SETTINGS_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    return data && typeof data === 'object' ? data : null;
  } catch (e) {
    return null; // private mode / disabled storage: run with defaults
  }
}

export function persistSettings(patch) {
  try {
    const cur = loadPersistedSettings() || {};
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(Object.assign({}, cur, patch || {})));
  } catch (e) { /* storage unavailable: settings stay memory-only */ }
}
