/**
 * On-disk memory of previously-seen posts, so duplicates are still recognised after a
 * reload. Lives in chrome.storage.local, which is the EXTENSION's storage even when
 * written from a content script -- IndexedDB from a content script would land in x.com's
 * own origin, where the site could read it and clearing site data would wipe it.
 *
 * Vectors are stored as int8, not float32. chrome.storage JSON-serialises, so a raw
 * Float32Array becomes ~8KB of decimal text per post; symmetric int8 plus a scale, base64
 * encoded, is ~512 bytes. Measured cost of that quantisation on the shipped algorithm:
 * mean cosine shift 0.00045 (max 0.0027), and collapse behaviour is unchanged --
 * 279 posts folded at 95.1% precision either way. Two orders of magnitude below the
 * threshold's own granularity.
 *
 * Bounded on purpose. This is a record of what the user has read, so it expires and it
 * is capped, and the popup can erase it.
 */
const KEY = 'dedupCache'
const VERSION = 1
export const MAX_ENTRIES = 4000
export const TTL_MS = 7 * 24 * 60 * 60 * 1000
const FLUSH_MS = 10_000

const B64 = (bytes) => {
  let s = ''
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i])
  return btoa(s)
}
const UNB64 = (str) => {
  const bin = atob(str)
  const out = new Int8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = (bin.charCodeAt(i) << 24) >> 24
  return out
}

/** @param {Float32Array} v */
export function quantise(v) {
  let max = 0
  for (let i = 0; i < v.length; i++) { const a = Math.abs(v[i]); if (a > max) max = a }
  const scale = (max || 1) / 127
  const q = new Int8Array(v.length)
  for (let i = 0; i < v.length; i++) q[i] = Math.max(-127, Math.min(127, Math.round(v[i] / scale)))
  return { b: B64(new Uint8Array(q.buffer, q.byteOffset, q.byteLength)), s: scale }
}

export function dequantise(b64, scale) {
  const q = UNB64(b64)
  const v = new Float32Array(q.length)
  let n = 0
  for (let i = 0; i < q.length; i++) { v[i] = q[i] * scale; n += v[i] * v[i] }
  n = Math.sqrt(n) || 1
  for (let i = 0; i < q.length; i++) v[i] /= n      // renormalise: cosine assumes unit length
  return v
}

export class Persistence {
  constructor(storage = globalThis.chrome?.storage?.local) {
    this.storage = storage
    /** statusId -> {b, s, a: author, t: seen-at} */
    this.entries = new Map()
    this.dirty = false
    this.timer = undefined
  }

  /** @returns {Promise<Array<{statusId, vec, author}>>} most recent first */
  async load() {
    if (!this.storage) return []
    let raw
    try {
      raw = (await this.storage.get(KEY))[KEY]
    } catch {
      return []                       // storage unavailable mid-update; run without memory
    }
    if (!raw || raw.v !== VERSION || !Array.isArray(raw.e)) return []

    const cutoff = Date.now() - TTL_MS
    const out = []
    for (const [statusId, b, s, a, t] of raw.e) {
      if (!t || t < cutoff) continue          // expired; simply not restored, and not rewritten
      this.entries.set(statusId, { b, s, a, t })
      out.push({ statusId, vec: dequantise(b, s), author: a })
    }
    if (out.length !== raw.e.length) this.dirty = true   // pruning happened; persist it
    return out.reverse()
  }

  remember(statusId, vec, author) {
    if (!this.storage || !vec || this.entries.has(statusId)) return
    const { b, s } = quantise(vec)
    this.entries.set(statusId, { b, s, a: author, t: Date.now() })
    this.dirty = true
    if (this.timer === undefined) this.timer = setTimeout(() => this.flush(), FLUSH_MS)
  }

  async flush() {
    if (this.timer !== undefined) { clearTimeout(this.timer); this.timer = undefined }
    if (!this.dirty || !this.storage) return
    this.dirty = false
    // Oldest first in the array, so the tail is what a truncation keeps.
    const all = [...this.entries.entries()].sort((x, y) => x[1].t - y[1].t)
    const keep = all.slice(-MAX_ENTRIES)
    this.entries = new Map(keep)
    const e = keep.map(([id, v]) => [id, v.b, v.s, v.a, v.t])
    try {
      await this.storage.set({ [KEY]: { v: VERSION, e } })
    } catch {
      // Over quota or storage gone. Drop half and let the next flush try again rather
      // than wedging: a cache that cannot be written is not worth breaking dedup over.
      this.entries = new Map(keep.slice(Math.floor(keep.length / 2)))
      this.dirty = true
    }
  }

  async clear() {
    this.entries = new Map()
    this.dirty = false
    try { await this.storage?.remove(KEY) } catch {}
  }

  get size() { return this.entries.size }
}
