// Fetch big model files once, keep them in the Cache API (survives reloads), report progress.
const CACHE = "omnivoice-models-v1";

export async function fetchBytes(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* private mode etc. */ }
  const hit = cache && await cache.match(url);
  if (hit) {
    const b = new Uint8Array(await hit.arrayBuffer());
    onProgress?.(b.length, b.length);
    return b;
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = Number(r.headers.get("content-length")) || 0;
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onProgress?.(got, total);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  try { await cache?.put(url, new Response(buf, { headers: { "content-length": String(got) } })); }
  catch { /* quota exceeded: still usable this session */ }
  return buf;
}
