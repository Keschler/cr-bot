/* Pure value formatting + card-name helpers (no DOM, no state). */

export function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export function fmtInt(v) {
  const n = num(v);
  return n === null ? '—' : Math.round(n).toLocaleString('en-US');
}

export function basename(path) {
  if (path === null || path === undefined) return '';
  const s = String(path);
  const parts = s.split(/[\\/]/);
  return parts[parts.length - 1] || s;
}

export function cardIconName(value) {
  if (value === null || value === undefined) return null;
  const raw = Array.isArray(value) ? value[0]
    : (typeof value === 'object' ? value.name : value);
  if (raw === null || raw === undefined) return null;
  const s = String(raw).trim();
  return s && s !== '—' ? s : null;
}

export function cardIconUrl(value) {
  const name = cardIconName(value);
  return name === null ? null : '/api/card-icon?name=' + encodeURIComponent(name);
}

export function handEntries(vs) {
  const hand = Array.isArray(vs.hand) ? vs.hand.slice(0, 4) : [];
  while (hand.length < 4) hand.push(null);
  return hand;
}

export function suggestionCardName(s, vs) {
  // The policy observation carries no hand names, so a suggestion's
  // card_name is often null. Fall back to the frame's hand at card_slot.
  const direct = s ? cardIconName(s.card_name) : null;
  if (direct) return direct;
  const slot = s ? num(s.card_slot) : null;
  if (slot !== null && vs) {
    const entries = handEntries(vs);
    if (slot >= 0 && slot < entries.length) {
      const h = cardIconName(entries[slot]);
      if (h) return h;
    }
  }
  return null;
}

export function handName(entry) {
  if (entry === null || entry === undefined) return '—';
  if (Array.isArray(entry)) return entry[0] === null || entry[0] === undefined ? '—' : String(entry[0]);
  if (typeof entry === 'object') return entry.name !== undefined ? String(entry.name) : '—';
  return String(entry);
}

export function handCost(entry) {
  if (Array.isArray(entry) && entry.length > 1 && entry[1] !== null && entry[1] !== undefined) {
    return String(entry[1]);
  }
  if (entry && typeof entry === 'object' && entry.cost !== undefined) return String(entry.cost);
  return '—';
}
