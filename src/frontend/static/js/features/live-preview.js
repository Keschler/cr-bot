/* Single-flight live picture polling, independent of analyzed frame arrival. */

export function createLivePreviewPoller({
  isFollowing, onFrame, fetchFn = (...args) => fetch(...args),
  intervalMs = 100, setTimeoutFn = setTimeout, clearTimeoutFn = clearTimeout,
  onMetrics = null, clockFn = () => performance.now(),
}) {
  let enabled = false;
  let generation = 0;
  let timer = null;
  let request = null;
  let metricsWindowStart = null;
  let displayedFrames = 0;
  let changedFrames = 0;
  let lastDisplayedIndex = null;
  let metricsSessionId = null;

  const headerNumber = (response, name) => {
    const value = response.headers.get(name);
    if (value === null || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  };

  async function tick(token) {
    timer = null;
    if (!enabled || token !== generation) return;
    try {
      if (!isFollowing()) return;
      const controller = new AbortController();
      request = controller;
      const startedMs = onMetrics ? clockFn() : null;
      const response = await fetchFn('/api/frame/preview', {
        cache: 'no-store', signal: controller.signal,
      });
      const headersMs = onMetrics ? clockFn() : null;
      if (!response.ok || response.status === 204) return;
      const frameIndex = headerNumber(response, 'X-Frame-Index');
      if (frameIndex === null) return;
      const blob = await response.blob();
      const bodyMs = onMetrics ? clockFn() : null;
      if (!blob.size || !enabled || token !== generation || !isFollowing()) return;
      // Wait for painting/decoding before asking for the newest image again.
      // There is no queue of obsolete pictures to catch up through.
      await onFrame({
        blob, frameIndex,
        captureAgeMs: headerNumber(response, 'X-Capture-Age-Ms'),
        analysisAgeMs: headerNumber(response, 'X-Analysis-Age-Ms'),
      });
      const diagnosticsId = response.headers.get('X-Live-Diagnostics');
      if (onMetrics && diagnosticsId && enabled && token === generation && isFollowing()) {
        const displayedMs = clockFn();
        if (metricsWindowStart === null || diagnosticsId !== metricsSessionId) {
          metricsSessionId = diagnosticsId;
          metricsWindowStart = startedMs;
          displayedFrames = changedFrames = 0;
          lastDisplayedIndex = null;
        }
        displayedFrames++;
        if (frameIndex !== lastDisplayedIndex) changedFrames++;
        lastDisplayedIndex = frameIndex;
        const windowMs = displayedMs - metricsWindowStart;
        if (windowMs >= 2000) {
          onMetrics({
            session_id: diagnosticsId, window_s: windowMs / 1000,
            displayed_frames: displayedFrames, changed_frames: changedFrames,
            last_frame_index: frameIndex,
            response_headers_ms: headersMs - startedMs,
            body_ms: bodyMs - headersMs,
            decode_ms: displayedMs - bodyMs,
            request_to_display_ms: displayedMs - startedMs,
          });
          metricsWindowStart = displayedMs;
          displayedFrames = changedFrames = 0;
        }
      }
    } catch (error) {
      // Status polling reports connection errors; the next tick retries.
    } finally {
      if (token === generation) {
        request = null;
        if (enabled) timer = setTimeoutFn(() => tick(token), intervalMs);
      }
    }
  }

  return {
    start() {
      if (enabled) return;
      enabled = true;
      metricsWindowStart = null;
      displayedFrames = changedFrames = 0;
      lastDisplayedIndex = null;
      metricsSessionId = null;
      void tick(++generation);
    },
    stop() {
      if (!enabled) return;
      enabled = false;
      generation++;
      if (timer !== null) clearTimeoutFn(timer);
      timer = null;
      if (request) request.abort();
      request = null;
    },
  };
}
