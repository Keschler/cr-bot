/** Resolve playback shortcuts while preserving native form interactions. */
export function playbackCommand(event, {page = 'watch', dialogOpen = false} = {}) {
  if (page !== 'watch' || dialogOpen || event.defaultPrevented || event.isComposing ||
      event.ctrlKey || event.metaKey || event.altKey) return null;

  const target = event.target;
  if (target?.isContentEditable || target?.closest?.('textarea, select')) return null;
  const input = target?.closest?.('input');
  const scrubber = input?.id === 'scrubber';
  if (input && !scrubber && input.type !== 'checkbox') return null;

  const delta = event.shiftKey ? 10 : 1;
  if (event.key === 'ArrowLeft') return {action: 'step', delta: -delta};
  if (event.key === 'ArrowRight') return {action: 'step', delta};
  if (event.key === 'Home') return {action: 'first'};
  if (event.key === 'End') return {action: 'last'};
  // Space still activates a focused button or toggles a checkbox once.
  if ((event.code === 'Space' || event.key === ' ') &&
      !target?.closest?.('button') && (!input || scrubber)) return {action: 'play'};
  if (event.key?.toLowerCase() === 'l') return {action: 'follow'};
  if (event.key === 'Escape') return {action: 'clear'};
  return null;
}
