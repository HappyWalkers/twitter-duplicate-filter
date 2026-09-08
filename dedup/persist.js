/**
 * Cross-session memory of posts already seen.
 *
 * The vectors live in IndexedDB on the extension origin, owned by the service worker
 * (see background.js). This module is the content-script half: it buffers what the
 * observer learns and ships it over in batches.
 *
 * Two things it is NOT:
 *   * not chrome.storage.local -- that is capped near 10MB, which would be a ~20,000-post
 *     limit enforced by silent write failures;
 *   * not IndexedDB opened here -- a content script's IndexedDB belongs to x.com's origin,
 *     where the site could read it and clearing site data would wipe it.
 *
 * There is NO entry cap. Age is the only bound, applied by the service worker on recall.
 *
 * Vectors are int8, not float32: measured cost on the shipped algorithm is a mean cosine
 * shift of 0.00045 (max 0.0027) with collapse behaviour unchanged -- 279 posts folded at
 * 95.1% precision either way -- for a quarter of the bytes.
 */
const FLUSH_MS = 10_000
const send = (msg) => new Promise((resolve) => {
  try {
    chrome.runtime.sendMessage(msg, (r) => { void chrome.runtime.lastError; resolve(r) })
  } catch { resolve(undefined) }
})

/** @param {Float32Array} v -> {q: Int8Array, s: scale} */
export function quantise(v) {
  let max = 0
  for (let i = 0; i < v.length; i++) { const a = Math.abs(v[i]); if (a > max) max = a }
  const s = (max || 1) / 127
  const q = new Int8Array(v.length)
  for (let i = 0; i < v.length; i++) q[i] = Math.max(-127, Math.min(127, Math.round(v[i] / s)))
  return { q, s }
}

export function dequantise(q, s) {
  const v = new Float32Array(q.length)
  let n = 0
  for (let i = 0; i < q.length; i++) { v[i] = q[i] * s; n += v[i] * v[i] }
  n = Math.sqrt(n) || 1
  for (let i = 0; i < q.length; i++) v[i] /= n     // renormalise: cosine assumes unit length
  return v
}

export class Persistence {
  constructor(transport = send) {
    this.send = transport
    this.buffer = []
    this.seen = new Set()          // ids already stored or queued, so we never re-send
    this.timer = undefined
  }

  /** @returns {Promise<Array<{statusId, vec, author}>>} */
  async load() {
    const r = await this.send({ type: 'dedup-recall' })
    if (!r?.ok || !Array.isArray(r.rows)) return []
    const out = []
    for (const row of r.rows) {
      this.seen.add(row.id)
      // Structured clone hands back an Int8Array; a plain object means a corrupt row,
      // which is not worth crashing the timeline over.
      const q = row.q instanceof Int8Array ? row.q : new Int8Array(Object.values(row.q || {}))
      if (q.length) out.push({ statusId: row.id, vec: dequantise(q, row.s), author: row.a })
    }
    return out
  }

  remember(statusId, vec, author) {
    if (!vec || this.seen.has(statusId)) return
    this.seen.add(statusId)
    const { q, s } = quantise(vec)
    this.buffer.push({ id: statusId, q, s, a: author, t: Date.now() })
    // Batch: one message per post would put a structured clone and a transaction on
    // every scroll tick.
    if (this.timer === undefined) this.timer = setTimeout(() => this.flush(), FLUSH_MS)
  }

  async flush() {
    if (this.timer !== undefined) { clearTimeout(this.timer); this.timer = undefined }
    if (!this.buffer.length) return
    const rows = this.buffer
    this.buffer = []
    const r = await this.send({ type: 'dedup-remember', rows })
    // Put them back if the worker was asleep or the write failed, rather than dropping
    // silently -- but only once, so a permanently broken store cannot grow forever.
    if (!r?.ok) for (const row of rows) this.seen.delete(row.id)
  }

  async clear() {
    this.buffer = []
    this.seen = new Set()
    await this.send({ type: 'dedup-forget' })
  }

  get size() { return this.seen.size }
}
