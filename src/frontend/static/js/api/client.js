/* Backend API client (fetch wrappers only). */

export async function apiGet(path) {
  const res = await fetch(path, { cache: 'no-store' });
  if (!res.ok) {
    let detail = '';
    try { detail = await res.text(); } catch (e) { /* ignore */ }
    throw new Error('GET ' + path + ' → HTTP ' + res.status + (detail ? ' ' + detail : ''));
  }
  return res.json();
}

export async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = '';
    try { detail = await res.text(); } catch (e) { /* ignore */ }
    throw new Error('POST ' + path + ' → HTTP ' + res.status + (detail ? ' ' + detail : ''));
  }
  try { return await res.json(); } catch (e) { return {}; }
}

export function apiDelete(path) {
  return fetch(path, { method: 'DELETE' }).then(async (res) => {
    if (!res.ok) {
      let detail = '';
      try { detail = await res.text(); } catch (e) { /* ignore */ }
      throw new Error('DELETE ' + path + ' → HTTP ' + res.status + (detail ? ' ' + detail : ''));
    }
    try { return await res.json(); } catch (e) { return {}; }
  });
}
