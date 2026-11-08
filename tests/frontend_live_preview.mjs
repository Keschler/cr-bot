import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const source = await readFile(new URL('../src/frontend/static/js/features/live-preview.js', import.meta.url), 'utf8');
const { createLivePreviewPoller } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

const flush = () => new Promise((resolve) => setImmediate(resolve));
function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}
function response(index, status = 200) {
  return {
    ok: status < 400, status,
    headers: new Headers(index === null ? {} : {
      'X-Frame-Index': String(index), 'X-Capture-Age-Ms': '12', 'X-Analysis-Age-Ms': '1100',
    }),
    blob: async () => new Blob(['frame-' + index]),
  };
}
function harness(fetchFn, onFrame = async () => {}, options = {}) {
  const timers = new Map();
  let seq = 0;
  const following = { value: true };
  const poller = createLivePreviewPoller({
    fetchFn, onFrame, isFollowing: () => following.value,
    setTimeoutFn: (fn) => { const id = ++seq; timers.set(id, fn); return id; },
    clearTimeoutFn: (id) => timers.delete(id),
    ...options,
  });
  return {
    poller, timers, following,
    next() {
      const [id, fn] = timers.entries().next().value;
      timers.delete(id);
      fn();
    },
  };
}

test('slow fetch and decoding never queue old pictures', async () => {
  const first = deferred();
  const paint = deferred();
  const frames = [];
  let calls = 0;
  const h = harness(async (url, options) => {
    assert.equal(url, '/api/frame/preview');
    assert.equal(options.cache, 'no-store');
    calls++;
    return calls === 1 ? first.promise : response(300);
  }, async (frame) => {
    frames.push(frame);
    if (frame.frameIndex === 30) await paint.promise;
  });
  h.poller.start();
  h.poller.start();
  await flush();
  assert.equal(calls, 1);
  assert.equal(h.timers.size, 0);
  first.resolve(response(30));
  await flush();
  assert.equal(h.timers.size, 0);
  paint.resolve();
  await flush();
  assert.equal(h.timers.size, 1);
  h.next();
  await flush();
  assert.deepEqual(frames.map((frame) => frame.frameIndex), [30, 300]);
  assert.equal(frames[1].captureAgeMs, 12);
  assert.equal(frames[1].analysisAgeMs, 1100);
  h.poller.stop();
  assert.equal(h.timers.size, 0);
});

test('a paused or scrubbed view cannot be replaced by an in-flight live image', async () => {
  const request = deferred();
  const frames = [];
  const h = harness(() => request.promise, (frame) => frames.push(frame));
  h.poller.start();
  h.following.value = false;
  request.resolve(response(90));
  await flush();
  assert.deepEqual(frames, []);
  h.poller.stop();
});

test('stopping and restarting discards the previous session response', async () => {
  const old = deferred();
  const frames = [];
  let signal;
  let calls = 0;
  const h = harness((url, options) => {
    calls++;
    if (calls === 1) { signal = options.signal; return old.promise; }
    return Promise.resolve(response(0));
  }, (frame) => frames.push(frame.frameIndex));
  h.poller.start();
  h.poller.stop();
  assert.equal(signal.aborted, true);
  h.poller.start();
  await flush();
  old.resolve(response(120));
  await flush();
  assert.deepEqual(frames, [0]);
  assert.equal(h.timers.size, 1);
  h.poller.stop();
});

for (const failure of ['no-content', 'network-error', 'missing-frame-index']) {
  test('preview recovers after ' + failure, async () => {
    const frames = [];
    let calls = 0;
    const h = harness(async () => {
      if (++calls > 1) return response(99);
      if (failure === 'network-error') throw new Error('disconnected');
      return response(null, failure === 'no-content' ? 204 : 200);
    }, (frame) => frames.push(frame.frameIndex));
    h.poller.start();
    await flush();
    assert.deepEqual(frames, []);
    h.next();
    await flush();
    assert.deepEqual(frames, [99]);
    h.poller.stop();
  });
}

test('opt-in metrics separate network, decode, and changed-picture FPS', async () => {
  const clock = [0, 190, 190, 1190, 1290, 1480, 1490, 2250];
  const metrics = [];
  const h = harness(async () => {
    const result = response(70);
    result.headers.set('X-Live-Diagnostics', 'session-a');
    return result;
  }, async () => {}, { clockFn: () => clock.shift(), onMetrics: (sample) => metrics.push(sample) });
  h.poller.start();
  await flush();
  assert.equal(metrics.length, 0);
  h.next();
  await flush();
  assert.deepEqual(metrics, [{
    session_id: 'session-a', window_s: 2.25, displayed_frames: 2, changed_frames: 1,
    last_frame_index: 70, response_headers_ms: 190, body_ms: 10,
    decode_ms: 760, request_to_display_ms: 960,
  }]);
  h.poller.stop();
});

test('metrics reset when the backend session changes without a local restart', async () => {
  let clock = 0;
  let index = 70;
  let sessionId = 'old-session';
  const metrics = [];
  const h = harness(async () => {
    clock += 190;
    const result = response(index++);
    result.headers.set('X-Live-Diagnostics', sessionId);
    return result;
  }, async () => { clock += 1000; }, {
    clockFn: () => clock, onMetrics: (sample) => metrics.push(sample),
  });
  h.poller.start();
  await flush();
  sessionId = 'new-session';
  clock += 100;
  h.next();
  await flush();
  assert.equal(metrics.length, 0);
  clock += 100;
  h.next();
  await flush();
  assert.equal(metrics.length, 1);
  assert.equal(metrics[0].session_id, 'new-session');
  assert.equal(metrics[0].displayed_frames, 2);
  assert.equal(metrics[0].changed_frames, 2);
  assert.equal(metrics[0].window_s, 2.48);
  h.poller.stop();
});

// Load the real session/correction modules with a minimal DOM and mocked
// network. This exercises pause and image identity across their module boundary.
async function sessionHarness() {
  const base = new URL('../src/frontend/static/js/', import.meta.url);
  const paths = ['features/session.js', 'features/live-preview.js', 'features/frames.js',
    'features/corrections.js', 'utils/format.js'];
  const sources = new Map(await Promise.all(paths.map(async (path) => {
    const url = new URL(path, base);
    return [url.href, await readFile(url, 'utf8')];
  })));
  const imports = new Map();
  for (const [url, code] of sources) {
    for (const match of code.matchAll(/import\s*\{([^}]+)\}\s*from\s*['"]([^'"]+)['"]/g)) {
      const dependency = new URL(match[2], url).href;
      if (!imports.has(dependency)) imports.set(dependency, new Set());
      for (const name of match[1].split(',')) imports.get(dependency).add(name.trim());
    }
  }
  const nodes = new Map();
  function node() {
    return {
      dataset: {}, hidden: false, style: {}, value: '', options: [], handlers: {},
      classList: { toggle() {}, add() {}, remove() {} },
      setAttribute() {}, focus() {},
      addEventListener(event, handler) { this.handlers[event] = handler; },
    };
  }
  const els = new Proxy({}, { get(_, id) {
    if (!nodes.has(id)) nodes.set(id, node());
    return nodes.get(id);
  } });
  const image = els['center-frame'];
  image.dataset = { src: '/api/frame/100', fi: '100' };
  let decode = null;
  image.decode = () => decode ? decode.promise : Promise.resolve();
  const state = {
    history: [{ frame_index: 100, timestamp_s: 3, emitted: true, in_game: true }],
    cursor: 0, playing: true, mode: 'live', sessionRunning: true, sessionMode: 'live',
    roiAdapt: {}, toggles: {}, speed: 1, correctionDrafts: {},
    sessionPin: null,
  };
  const requests = [];
  const statusReplies = [];
  const apiCalls = [];
  const timers = new Map();
  let timerId = 0;
  let blobId = 0;
  const context = vm.createContext({
    AbortController, Blob, Headers, performance,
    URL: { createObjectURL: () => 'blob:' + (++blobId), revokeObjectURL() {} },
    fetch: (url, options) => {
      const result = deferred();
      requests.push({ url, options, result });
      return result.promise;
    },
    setTimeout: (fn) => { timers.set(++timerId, fn); return timerId; },
    clearTimeout: (id) => timers.delete(id),
    document: { querySelector: () => null, getElementById: () => null },
  });
  const values = {
    state, els, img: image, canvas: node(), GRID_COLS: 18, GRID_ROWS: 32,
    roiPreviewVisible: () => false,
    apiGet: async () => statusReplies.shift() || { running: true, mode: 'live', frame_count: 0 },
    apiPost: async (url, body) => { apiCalls.push({ url, body }); return {}; },
    setPill: (text, kind, title) => {
      els['status-pill'].textContent = text;
      els['status-pill'].className = kind;
      els['status-pill'].title = title || '';
    },
  };
  const modules = new Map();
  function moduleFor(url) {
    if (modules.has(url)) return modules.get(url);
    let mod;
    if (sources.has(url)) {
      mod = new vm.SourceTextModule(sources.get(url), { context, identifier: url });
    } else {
      const names = [...imports.get(url)];
      mod = new vm.SyntheticModule(names, function () {
        for (const name of names) this.setExport(name, values[name] ?? (() => {}));
      }, { context, identifier: url });
    }
    modules.set(url, mod);
    return mod;
  }
  const sessionModule = moduleFor(new URL('features/session.js', base).href);
  await sessionModule.link((specifier, parent) => moduleFor(new URL(specifier, parent.identifier).href));
  await sessionModule.evaluate();
  const session = sessionModule.namespace;
  session.bindSessionEvents();
  return {
    session, state, image, els, requests, apiCalls,
    queueStatus(reply) { statusReplies.push(reply); },
    corrections: moduleFor(new URL('features/corrections.js', base).href).namespace,
    deferDecode() { decode = deferred(); return decode; },
    clearDecode() { decode = null; },
  };
}

test('pause loads exact analyzed pixels immediately, even after resuming the same frame', async () => {
  const h = await sessionHarness();
  h.session.refreshImage();
  assert.equal(h.requests[0].url, '/api/frame/preview');
  h.requests[0].result.resolve(response(103));
  await flush();
  assert.equal(h.image.dataset.fi, '103');
  assert.equal(h.session.imageMatchesFrame(h.state.history[0]), false);
  assert.equal(h.image.dataset.src, undefined);
  assert.equal(h.els['history-badge'].textContent, 'Live · gameplay age unknown');

  h.els['btn-play'].handlers.click();
  assert.equal(h.state.playing, false);
  assert.equal(h.requests[1].url, '/api/frame/100');
  const decode = h.deferDecode();
  h.requests[1].result.resolve(response(100));
  await flush();
  assert.equal(h.session.imageMatchesFrame(h.state.history[0]), false);
  h.state.editMode = true;
  assert.equal(h.corrections.editableFrame(), null);
  decode.resolve();
  await flush();
  assert.equal(h.session.imageMatchesFrame(h.state.history[0]), true);
  assert.equal(h.corrections.editableFrame().frame_index, 100);

  h.clearDecode();
  h.els['btn-play'].handlers.click();
  assert.equal(h.requests[2].url, '/api/frame/preview');
  h.requests[2].result.resolve(response(106));
  await flush();
  h.els['btn-play'].handlers.click();
  assert.equal(h.requests[3].url, '/api/frame/100');
  h.requests[3].result.resolve(response(100));
  await flush();
  assert.equal(h.image.dataset.fi, '100');
});

test('an old session live image cannot replace a reset picture', async () => {
  const h = await sessionHarness();
  h.session.refreshImage();
  const old = h.requests[0];
  h.session.resetSession('new session');
  assert.equal(old.options.signal.aborted, true);
  assert.equal(h.image.hidden, true);
  assert.equal(h.image.dataset.src, undefined);
  assert.equal(h.image.dataset.fi, undefined);
  old.result.resolve(response(200));
  await flush();
  assert.notEqual(h.image.dataset.fi, '200');
});

test('frame counts update on ingest and stale status cannot lag or revert them', async () => {
  const h = await sessionHarness();
  h.session.ingestFrames([{ frame_index: 140, timestamp_s: 4 }]);
  assert.equal(h.state.history[h.state.cursor].frame_index, 140);
  assert.equal(h.els['status-pill'].textContent, '● live · 2 history frames');
  assert.equal(h.els['session-mini'].textContent, 'live · 2 history frames');
  // A status request started before the latest SSE frame can complete later.
  h.queueStatus({ running: true, mode: 'live', frame_count: 1 });
  await h.session.pollStatus();
  assert.equal(h.els['status-pill'].textContent, '● live · 2 history frames');
  assert.equal(h.els['session-mini'].textContent, 'live · 2 history frames');
  h.state.playing = false;
  h.session.ingestFrames([{ frame_index: 180, timestamp_s: 5 }]);
  assert.equal(h.state.history[h.state.cursor].frame_index, 140);
  assert.equal(h.els['status-pill'].textContent, '● live · 3 history frames');
  h.queueStatus({ running: false, mode: 'live', frame_count: 512 });
  await h.session.pollStatus();
  assert.equal(h.els['status-pill'].textContent, 'Stopped · 3 history frames');
  h.session.resetSession('new run');
});

test('live start sends the selected capture width, including native and default', async () => {
  for (const [selected, expected] of [['720', 720], ['480', 480], ['native', null], ['', null]]) {
    const h = await sessionHarness();
    h.els['input-serial'].value = 'test-device';
    h.els['select-transport'].value = 'stream';
    h.els['select-stream-width'].value = selected;
    await h.session.startLive();
    assert.equal(h.apiCalls.length, 1);
    assert.equal(h.apiCalls[0].url, '/api/live/start');
    assert.equal(h.apiCalls[0].body.stream_max_width_px, expected);
    h.session.resetSession('finished');
  }
});
