/* Side panels: left extracted-state (meta/hand/towers/detections) and
 * right bot-suggestions (top-3 + reasoning summary).
 */

import { state, KING_TOWER_MAX, PRINCESS_TOWER_MAX } from '../state/store.js';
import { els } from '../utils/elements.js';
import { esc } from '../utils/dom.js';
import { num, fmtInt, cardIconUrl, handEntries, handName, handCost, suggestionCardName } from '../utils/format.js';
import { parseCell } from '../utils/geometry.js';
import { visualStateOf, suggestionsOf, originalSuggestionsOf, diagnosticsOf, actionOf, topSuggestions } from './frames.js';
import { unitKeyOf, editKeyOf, draftFor, draftUpdateFor, refreshCorrectionControls } from './corrections.js';
import { renderCurrent, writeLocationHash } from './timeline.js';

// Card-art URLs that 404'd: remembered so polls don't re-request (and
// re-flash) them when the same card reappears.
const deadIcons = new Set();

export function formatEv(ev) {
  const v = num(ev);
  if (v === null) return null;
  return (v >= 0 ? '+' : '') + v.toFixed(2) + ' EV';
}

/* ---------- left panel ---------- */

export function phaseOf(vs, inGame) {
  if (!inGame) return 'Not in game';
  if (vs.overtime) return 'Overtime';
  const t = num(vs.time_left_s);
  if (t === null) return '—';
  return t <= 60 ? 'Double elixir' : 'Single elixir';
}

export function towerMax(isKing) {
  const t = state.towerMax;
  if (t && typeof t === 'object') {
    const v = num(isKing ? t.king : t.princess);
    if (v !== null && v > 0) return v;
  }
  return isKing ? KING_TOWER_MAX : PRINCESS_TOWER_MAX;
}

export function kingState(vs, side) {
  // Real extractor flag first (state_builder: king HP below max), then HP
  // derivation: a damaged king or a fallen princess tower means activated.
  const flag = side === 'self' ? vs.own_king_active : vs.enemy_king_active;
  if (flag === true) return 'active';
  const arr = side === 'self' ? vs.tower_hp_self : vs.tower_hp_enemy;
  const t = Array.isArray(arr) ? arr : [];
  const k = num(t[1]);
  if (k === null) return 'unknown';
  if (k <= 0) return 'destroyed';
  if (k < towerMax(true)) return 'active';
  const l = num(t[0]), r = num(t[2]);
  if ((l !== null && l <= 0) || (r !== null && r <= 0)) return 'active';
  return 'dormant';
}

export function kingStatusLabel(st) {
  if (st === 'active') return 'Active';
  if (st === 'dormant') return 'Sleeping';
  if (st === 'destroyed') return 'Destroyed';
  return 'Unknown';
}

export function kingStateClass(st) {
  if (st === 'active') return 'kact';
  if (st === 'dormant') return 'kdor';
  if (st === 'destroyed') return 'kdes';
  return 'kunk';
}

export function towerPct(hp, isKing) {
  const n = num(hp);
  if (n === null) return null;
  const max = towerMax(isKing);
  return Math.max(0, Math.min(100, (n / max) * 100));
}

export function renderLeft(frame) {
  const vs = visualStateOf(frame);
  const has = !!frame;

  els['meta-replay-id'].textContent = state.sessionLabel || (has ? '#' + frame.frame_index : 'No session');
  const diag = diagnosticsOf(frame);
  els['meta-arena'].textContent = diag.arena !== undefined && diag.arena !== null
    ? String(diag.arena) : (vs.arena !== undefined && vs.arena !== null ? String(vs.arena) : '—');
  els['meta-tick'].textContent = frame ? fmtInt(frame.frame_index) : '—';
  els['meta-phase'].textContent = has ? phaseOf(vs, !!frame.in_game) : '—';

  // Hand: highlight the inspected suggestion's slot, else the top pick.
  const sug = topSuggestions(suggestionsOf(frame));
  const sel = state.selectedRank !== null ? sug[state.selectedRank] : null;
  const top = sug[0] || null;
  const selSlot = num((sel || top || {}).card_slot);
  const slots = els['hand-slots'].querySelectorAll('.hand-slot');
  const entries = handEntries(vs);
  slots.forEach((slotEl, i) => {
    const nameEl = slotEl.querySelector('.card-name');
    const costEl = slotEl.querySelector('.card-cost');
    const iconEl = slotEl.querySelector('.hand-icon');
    nameEl.textContent = has ? handName(entries[i]) : '—';
    costEl.textContent = has ? handCost(entries[i]) : '—';
    const iconUrl = has ? cardIconUrl(entries[i]) : null;
    if (iconEl) {
      // Same broken-image handling as suggestion icons: a 404 hides the
      // slot icon instead of leaving a broken image. Failed URLs are
      // remembered in a module set so alternating failures don't re-flash.
      if (!iconEl.dataset.errbound) {
        iconEl.dataset.errbound = '1';
        iconEl.addEventListener('error', () => {
          if (iconEl.dataset.src) deadIcons.add(iconEl.dataset.src);
          iconEl.hidden = true;
        });
      }
      if (iconUrl) {
        if (iconEl.dataset.src !== iconUrl) {
          iconEl.dataset.src = iconUrl;
          iconEl.src = iconUrl;
        }
        iconEl.hidden = deadIcons.has(iconUrl);
        iconEl.alt = handName(entries[i]);
      } else {
        iconEl.hidden = true;
        iconEl.removeAttribute('src');
        delete iconEl.dataset.src;
      }
    }
    slotEl.classList.toggle('is-selected', selSlot === i);
  });
  els['next-card'].textContent = vs.next_card !== undefined && vs.next_card !== null ? String(vs.next_card) : '—';
  const elixir = num(vs.elixir);
  els['own-elixir-text'].textContent = elixir === null ? '—' : elixir.toFixed(1) + ' / 10';
  els['own-elixir-fill'].style.width = elixir === null ? '0' : Math.max(0, Math.min(100, (elixir / 10) * 100)) + '%';
  const ee = num(vs.enemy_elixir_est);
  els['enemy-elixir-text'].textContent = ee === null ? '—' : ee.toFixed(1) + ' / 10';
  els['enemy-elixir-fill'].style.width = ee === null ? '0' : Math.max(0, Math.min(100, (ee / 10) * 100)) + '%';

  // Towers: Left / King / Right rows, each split into a YOU side (left)
  // and an OPPONENT side (right) with its own HP number and health bar.
  // King activation is a separate status line, never merged into HP.
  const box = els['tower-rows'];
  box.innerHTML = '';
  const selfT = Array.isArray(vs.tower_hp_self) ? vs.tower_hp_self : [];
  const enemyT = Array.isArray(vs.tower_hp_enemy) ? vs.tower_hp_enemy : [];
  if (!has || (!selfT.length && !enemyT.length)) {
    box.innerHTML = '<p class="empty">No session</p>';
  } else {
    const head = document.createElement('div');
    head.className = 'tower-colheads';
    head.innerHTML = '<span>You</span><span>Opponent</span>';
    box.appendChild(head);
    const names = ['Left', 'King', 'Right'];
    for (let i = 0; i < 3; i++) {
      const s = num(selfT[i]);
      const e = num(enemyT[i]);
      const isKing = i === 1;
      const max = towerMax(isKing);
      const pctS = s === null ? null : Math.max(0, Math.min(100, (s / max) * 100));
      const pctE = e === null ? null : Math.max(0, Math.min(100, (e / max) * 100));
      const row = document.createElement('div');
      row.className = 'tower-row';
      const barCls = (pct) => pct === null ? '' : pct > 50 ? '' : pct > 25 ? 'warn' : 'bad';
      let statusLine = '';
      if (isKing) {
        const stS = kingState(vs, 'self'), stE = kingState(vs, 'enemy');
        statusLine =
          '<div class="king-states"><span class="kstate">You · <b class="' + kingStateClass(stS) + '">' +
          esc(kingStatusLabel(stS)) + '</b></span>' +
          '<span class="kstate">Opponent · <b class="' + kingStateClass(stE) + '">' +
          esc(kingStatusLabel(stE)) + '</b></span></div>';
      }
      row.innerHTML =
        '<div class="tower-name">' + esc(names[i]) + '</div>' +
        '<div class="tower-sides">' +
        '<div class="tower-side you"><div class="tower-hp">' +
        (s === null ? '—' : esc(fmtInt(s))) + '</div>' +
        '<div class="bar"><div class="' + barCls(pctS) + '" style="width:' +
        (pctS === null ? 0 : pctS.toFixed(1)) + '%" title="you ' +
        (pctS === null ? '—' : pctS.toFixed(0) + '%') + '"></div></div></div>' +
        '<div class="tower-side opponent"><div class="tower-hp">' +
        (e === null ? '—' : esc(fmtInt(e))) + '</div>' +
        '<div class="bar"><div class="' + barCls(pctE) + '" style="width:' +
        (pctE === null ? 0 : pctE.toFixed(1)) + '%" title="opponent ' +
        (pctE === null ? '—' : pctE.toFixed(0) + '%') + '"></div></div></div>' +
        '</div>' + statusLine;
      box.appendChild(row);
    }
  }

  // Detected objects.
  const list = els['detected-objects'];
  list.innerHTML = '';
  const ally = Array.isArray(vs.ally_units) ? vs.ally_units : [];
  const enemy = Array.isArray(vs.enemy_units) ? vs.enemy_units : [];
  const detCount = vs.detection_count !== undefined && vs.detection_count !== null
    ? Number(vs.detection_count) : ally.length + enemy.length;
  const draftAdds = frame && state.correctionDrafts[String(frame.frame_index)]
    ? state.correctionDrafts[String(frame.frame_index)].adds.filter((a) => a && Array.isArray(a.box)).length
    : 0;
  els['detection-count'].textContent = has
    ? '(' + detCount + (draftAdds ? ' +' + draftAdds + ' new' : '') + ')'
    : '';
  const all = ally.map((u) => ({ u, team: 'ally' })).concat(enemy.map((u) => ({ u, team: 'enemy' })));
  if (!has) {
    list.innerHTML = '<li class="empty">No session</li>';
    // No frame: the revert button must hide (it belongs to a what-if frame)
    // and a stale correction status must not linger.
    refreshCorrectionControls(frame);
    return;
  }
  // Draft adds are detections too: show them so new boxes can be
  // relabeled, re-teamed, or removed before submitting.
  const draft = frame ? draftFor(frame.frame_index) : null;
  if (draft) {
    draft.adds.forEach((a, i) => {
      if (!a || !Array.isArray(a.box)) return;
      all.push({
        u: {
          label: a.class_name, team: a.team,
          center_px: [(a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2],
          confidence: null, isAdd: (a.id !== undefined && a.id !== null ? a.id : i),
        },
        team: a.team,
      });
    });
  }
  if (!all.length) {
    list.innerHTML = '<li class="empty">No detections on this frame</li>';
    return;
  }
  for (const { u, team } of all) {
    const label = u && u.label !== undefined ? String(u.label) : '?';
    const c = u && Array.isArray(u.center_px) ? u.center_px : null;
    const cx = c && c.length >= 2 ? num(c[0]) : null;
    const cy = c && c.length >= 2 ? num(c[1]) : null;
    const xy = (cx === null || cy === null)
      ? 'x — / y —' : 'x ' + Math.round(cx) + ' / y ' + Math.round(cy);
    const conf = (u && u.isAdd !== undefined && u.isAdd !== null) ? 'new'
      : (u && u.confidence !== undefined && u.confidence !== null ? Number(u.confidence).toFixed(2) : '—');
    const li = document.createElement('li');
    const key = unitKeyOf(u);
    const upd = draft ? draftUpdateFor(draft, key) : null;
    const isDel = draft ? draft.deletes.some((d) => editKeyOf(d.target) === key) : false;
    const shownLabel = upd && upd.class_name ? upd.class_name : label;
    const shownTeam = upd && upd.team ? upd.team : team;
    const editable = !!(frame && frame.emitted && frame.in_game);
    let controls = '';
    if (editable) {
      const opts = [label].concat(state.labels.filter((l) => l !== label)).map((l) =>
        '<option value="' + esc(l) + '"' + (l === shownLabel ? ' selected' : '') + '>' + esc(l) + '</option>').join('');
      controls = ' <span class="cedit">' +
        '<select data-cact="label" title="Correct label">' + opts + '</select>' +
        '<button type="button" class="mini" data-cact="team" title="Toggle team">' + esc(shownTeam) + '</button>' +
        '<button type="button" class="mini" data-cact="delete" title="Toggle delete">' + (isDel ? 'keep' : 'del') + '</button>' +
        '</span>';
    }
    li.dataset.ckey = key;
    const classes = [];
    if (isDel) classes.push('is-deleted');
    else if (upd) classes.push('is-edited');
    if (key.slice(0, 2) === 'a:') classes.push('is-added');
    if (state.selection && state.selection === key) classes.push('is-selected');
    li.className = classes.join(' ');
    li.innerHTML = '<span><span class="team team-' + esc(shownTeam) + '">' + esc(shownTeam) + '</span>' +
      '<strong>' + esc(shownLabel) + '</strong> <span class="muted">' + esc(xy) + '</span></span>' +
      '<span>' + esc(conf) + controls + '</span>';
    list.appendChild(li);
    if (!state.showAllDetections && list.children.length >= 6) break;
  }
  const hidden = all.length - list.children.length;
  if (hidden > 0) {
    const more = document.createElement('li');
    more.className = 'more';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'mini';
    btn.dataset.expand = 'more';
    btn.textContent = '+' + hidden + ' more — show all';
    btn.title = 'Expand the detections list';
    more.appendChild(btn);
    list.appendChild(more);
  } else if (state.showAllDetections && all.length > 6) {
    const less = document.createElement('li');
    less.className = 'more';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'mini';
    btn.dataset.expand = 'less';
    btn.textContent = 'show less';
    btn.title = 'Collapse the detections list';
    less.appendChild(btn);
    list.appendChild(less);
  }
  refreshCorrectionControls(frame);
}

/* ---------- right panel ---------- */

export function actionText(s, fallbackName) {
  if (!s || typeof s !== 'object') return '—';
  const kind = String(s.kind || '').toLowerCase();
  if (kind === 'wait') return 'Wait';
  const raw = s.card_name !== undefined && s.card_name !== null ? String(s.card_name) : null;
  const name = raw || fallbackName || 'card';
  const cell = parseCell(s.cell);
  if (cell) return 'Place ' + name + ' @ (' + Math.round(cell.col) + ',' + Math.round(cell.row) + ')';
  return 'Play ' + name;
}

export function probPct(s) {
  const p = num(s.probability);
  if (p === null) return '—';
  return (p <= 1 ? p * 100 : p).toFixed(0) + '%';
}

export function probFrac(s) {
  const p = num(s.probability);
  if (p === null) return null;
  return p <= 1 ? p : p / 100;
}

export function shouldAppendWaitRow(sug, diag) {
  // WAIT is one lumped hypothesis against ~2,300 split card×cell combos, so
  // a healthy P(wait) can sit just outside the top-3 cutoff. Return its
  // scored mass when no Wait entry is listed, else null.
  const hasWait = (sug || []).some(
    (s) => String((s && s.kind) || '').toLowerCase() === 'wait'
  );
  if (hasWait) return null;
  return finiteNum(diag && diag.mode_prob_wait);
}

export function finiteNum(v) {
  // Like num(), but JSON null/undefined/'' mean "unknown", never 0
  // (Number(null) === 0 would silently turn unknown into zero).
  if (v === null || v === undefined || v === '') return null;
  return num(v);
}

export function modeProbsText(diag) {
  const mpw = finiteNum(diag && diag.mode_prob_wait);
  const mpp = finiteNum(diag && diag.mode_prob_play);
  if (mpw !== null || mpp !== null) {
    return 'wait ' + (mpw === null ? '—' : (mpw * 100).toFixed(1) + '%') +
      ' · play ' + (mpp === null ? '—' : (mpp * 100).toFixed(1) + '%');
  }
  const legacy = diag ? diag.mode_probs : undefined;
  if (legacy !== undefined && legacy !== null) {
    return typeof legacy === 'object' ? JSON.stringify(legacy) : String(legacy);
  }
  return null;
}

export function waitReason(frame) {
  // Human reason for a frame with no scored suggestions (backend `result`).
  const rec = (frame && frame.record) || {};
  switch (rec.result) {
    case 'hand-not-stable': return 'Waiting for a stable hand…';
    case 'observation-not-ready': return 'Waiting for a readable game state…';
    case 'not-in-game': return 'Not in game — no decision';
    case 'cooldown': return 'Wait — between plays';
    default: {
      const action = actionOf(frame);
      if (action && String(action.kind || '').toLowerCase() === 'wait') return 'Wait — holding elixir';
      return 'No suggestion for this frame';
    }
  }
}

export function renderRight(frame) {
  const box = els['suggestions'];
  box.innerHTML = '';
  if (frame && frame.corrected) {
    const badge = document.createElement('div');
    badge.className = 'correction-badge';
    badge.textContent = 'Corrected labels · what-if (trackers & timeline unchanged)';
    box.appendChild(badge);
    const diff = suggestionDiff(frame);
    if (diff) {
      const row = document.createElement('div');
      row.className = 'correction-diff' + (diff.changed ? '' : ' is-same');
      row.innerHTML = diff.html;
      box.appendChild(row);
    }
  }
  const sug = topSuggestions(suggestionsOf(frame));
  const diag = diagnosticsOf(frame);

  if (!frame) {
    box.innerHTML = '<p class="empty">No session</p>';
  } else if (!sug.length) {
    // A frame with no scored suggestions is still a live analysis result —
    // typically a deliberate Wait (unstable hand, cooldown) — not "no session".
    box.innerHTML =
      '<div class="suggestion is-wait"><div class="suggestion-head">' +
      '<span class="suggestion-rank">–</span>' +
      '<span><span class="suggestion-title">Wait</span><br />' +
      '<span class="suggestion-prob">' + esc(waitReason(frame)) + '</span></span>' +
      '</div></div>';
  } else {
    sug.forEach((s, i) => {
      const card = document.createElement('div');
      card.className = 'suggestion' + (i === 0 ? ' is-top' : '') +
        (state.selectedRank === i ? ' is-selected' : '');
      card.dataset.rank = String(i);
      const ev = s.ev !== undefined && s.ev !== null ? s.ev
        : s.expected_value !== undefined ? s.expected_value : null;
      const lp = s.log_prob !== undefined && s.log_prob !== null ? Number(s.log_prob) : null;
      const cell = parseCell(s.cell);
      const slot = num(s.card_slot);
      const name = suggestionCardName(s, visualStateOf(frame)) || 'card';
      const iconUrl = String(s.kind || '').toLowerCase() === 'play' ? cardIconUrl(name === 'card' ? null : name) : null;
      let preview = '';
      if (state.selectedRank === i) {
        if (cell && slot !== null) {
          preview =
            '<div class="inspect-preview">' +
            '<div class="inspect-caption">Slot ' + (slot + 1) + ' · ' + esc(name) +
            ' → (' + Math.round(cell.col) + ',' + Math.round(cell.row) + ')<br />' +
            '<span class="muted">' + esc(probPct(s)) +
            (lp === null || !Number.isFinite(lp) ? '' : ' · logprob ' + lp.toFixed(2)) + '</span><br />' +
            '<button type="button" class="raw-toggle" data-raw="' + i + '">raw JSON</button></div></div>' +
            '<div class="inspect-raw" data-rawbox="' + i + '" hidden><pre>' +
            esc(JSON.stringify(s, null, 2)) + '</pre></div>';
        } else {
          preview =
            '<div class="inspect-preview"><div class="inspect-caption">Wait — hold elixir, no placement.<br />' +
            '<button type="button" class="raw-toggle" data-raw="' + i + '">raw JSON</button></div></div>' +
            '<div class="inspect-raw" data-rawbox="' + i + '" hidden><pre>' +
            esc(JSON.stringify(s, null, 2)) + '</pre></div>';
        }
      }
      card.innerHTML =
        '<div class="suggestion-head">' +
        (iconUrl
          ? '<img class="card-icon" src="' + iconUrl + '" alt="" loading="lazy" onerror="this.hidden=true" />'
          : '') +
        '<span class="suggestion-rank">' + (i + 1) + '</span>' +
        '<span><span class="suggestion-title">' + esc(actionText(s, name === 'card' ? null : name)) + '</span><br />' +
        '<span class="suggestion-prob">Probability ' + esc(probPct(s)) +
        (lp === null || !Number.isFinite(lp) ? '' : ' · logprob ' + lp.toFixed(2)) + '</span></span>' +
        (ev === null || ev === undefined ? '' : '<span class="suggestion-ev">' + esc(formatEv(ev) || String(ev)) + '</span>') +
        '</div>' + preview +
        '<button type="button" class="inspect-btn' + (state.selectedRank === i ? ' is-on' : '') +
        '" data-inspect="' + i + '">' +
        (state.selectedRank === i ? 'Hide' : 'Inspect') + ' ↗</button>';
      box.appendChild(card);
    });
    const waitP = shouldAppendWaitRow(sug, diag);
    if (waitP !== null) {
      const waitCard = document.createElement('div');
      waitCard.className = 'suggestion is-wait';
      waitCard.innerHTML =
        '<div class="suggestion-head">' +
        '<span class="suggestion-rank">–</span>' +
        '<span><span class="suggestion-title">Wait</span><br />' +
        '<span class="suggestion-prob">Probability ' + esc(probPct({ probability: waitP })) +
        ' · outside top 3</span></span>' +
        '</div>';
      box.appendChild(waitCard);
    }
  }

  // Reasoning summary — only real backend values, never invented.
  if (!frame) {
    els['reason-text'].textContent = 'No session';
    els['reason-entropy'].textContent = '—';
    els['reason-mode-probs'].textContent = '—';
    els['reason-hand-stable'].textContent = '—';
    els['reason-timing'].textContent = '—';
    els['reason-timing'].title = '';
    els['reason-devices'].textContent = '—';
    els['reason-confidence'].textContent = '—';
    els['reason-confidence'].className = '';
    return;
  }
  if (diag.summary !== undefined && diag.summary !== null) {
    els['reason-text'].textContent = String(diag.summary);
  } else if (sug.length) {
    els['reason-text'].textContent = 'Top action ' + actionText(sug[0], suggestionCardName(sug[0], visualStateOf(frame))) +
      ' at ' + probPct(sug[0]) + ' over ' + sug.length + ' suggestion(s).';
  } else {
    els['reason-text'].textContent = waitReason(frame) + '.';
  }
  const ent = num(diag.entropy);
  els['reason-entropy'].textContent = ent === null ? '—' : ent.toFixed(3);
  const modeProbs = modeProbsText(diag);
  els['reason-mode-probs'].textContent = modeProbs === null ? '—' : modeProbs;
  if (diag.hand_stable !== undefined && diag.hand_stable !== null) {
    els['reason-hand-stable'].textContent = typeof diag.hand_stable === 'boolean'
      ? (diag.hand_stable ? 'Yes' : 'No') : String(diag.hand_stable);
  } else {
    els['reason-hand-stable'].textContent = '—';
  }
  const timing = timingText(diag);
  els['reason-timing'].textContent = timing === null ? '—' : timing;
  els['reason-timing'].title = timingTitle(diag);
  const devices = devicesText();
  els['reason-devices'].textContent = devices === null ? '—' : devices;
  const topP = sug.length ? probFrac(sug[0]) : null;
  const conf = els['reason-confidence'];
  if (topP === null) {
    conf.textContent = '—';
    conf.className = '';
  } else if (topP >= 0.8) {
    conf.textContent = 'High';
    conf.className = 'is-high';
  } else if (topP >= 0.5) {
    conf.textContent = 'Medium';
    conf.className = 'is-med';
  } else {
    conf.textContent = 'Low';
    conf.className = 'is-low';
  }
}

export function timingText(diag) {
  // Per-frame pipeline breakdown served as diagnostics.timing_ms
  // {fetch, normalize, ..., total}. The headline is the total; the full
  // breakdown is one hover away.
  const t = diag ? diag.timing_ms : null;
  if (t && typeof t === 'object') {
    const total = num(t.total);
    if (total !== null) return total.toFixed(0) + ' ms total';
  }
  return null;
}

export function timingTitle(diag) {
  const t = diag ? diag.timing_ms : null;
  if (!t || typeof t !== 'object') return '';
  return Object.keys(t).sort()
    .map((k) => k + ': ' + (num(t[k]) !== null ? num(t[k]).toFixed(1) + 'ms' : String(t[k])))
    .join(' · ');
}

export function devicesText() {
  // Session summary from GET /api/status (runners report resolved devices).
  const s = state.sessionSummary;
  const d = s && typeof s.devices === 'object' && s.devices !== null ? s.devices : null;
  if (!d) return null;
  const bits = [];
  for (const k of Object.keys(d).sort()) {
    if (d[k] !== undefined && d[k] !== null && d[k] !== '') bits.push(k + ' ' + String(d[k]));
  }
  return bits.length ? bits.join(' · ') : null;
}

export function suggestionDiff(frame) {
  // One-line before → after for the top action under a what-if correction,
  // so reviewers see what changed before Revert. Null when identical.
  if (!frame || !frame.corrected) return null;
  const vs = visualStateOf(frame);
  const before = topSuggestions(originalSuggestionsOf(frame))[0] || null;
  const after = topSuggestions(suggestionsOf(frame))[0] || null;
  const describe = (s) => s
    ? actionText(s, suggestionCardName(s, vs)) + ' @ ' + probPct(s) : '—';
  const a = describe(before), b = describe(after);
  if (a === b) return { changed: false, html: 'Top action unchanged by this correction.' };
  return {
    changed: true,
    html: '<span class="diff-before">' + esc(a) + '</span>' +
      '<span class="diff-arrow">→</span>' +
      '<span class="diff-after">' + esc(b) + '</span>',
  };
}

export function toggleInspectRank(rank) {
  const r = Number.isFinite(rank) ? rank : 0;
  state.selectedRank = state.selectedRank === r ? null : r;
  renderCurrent();
  writeLocationHash();
}

export function bindPanelsEvents() {
  // Inspect toggles locate the suggestion on the arena (delegated: cards
  // re-render on every poll). Raw JSON stays behind a secondary toggle.
  els['suggestions'].addEventListener('click', (ev) => {
    const rawBtn = ev.target.closest('[data-raw]');
    if (rawBtn) {
      const box = els['suggestions'].querySelector('[data-rawbox="' + rawBtn.dataset.raw + '"]');
      if (box) box.hidden = !box.hidden;
      return;
    }
    const btn = ev.target.closest('[data-inspect]');
    if (!btn) return;
    toggleInspectRank(parseInt(btn.dataset.inspect, 10) || 0);
  });
}
