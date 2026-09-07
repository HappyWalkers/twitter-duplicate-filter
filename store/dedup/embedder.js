/**
 * Embedding client: batching, caching, circuit breaker, fail-open.
 *
 * The transport is injected so this whole file is testable without chrome.* or a real
 * worker. `embed()` NEVER rejects -- on any failure it resolves null, and null means
 * "no vector, so never collapse". A dead model must degrade to a normal timeline, never
 * to a blank one.
 */
import { TUNING } from './config.js'

/** @typedef {(texts: string[]) => Promise<Float32Array[]>} Transport */

export class Embedder {
  /** @param {Transport} transport */
  constructor(transport) {
    this.transport = transport
    /** @type {Map<string, Float32Array>} */
    this.cache = new Map()
    /** @type {Map<string, Promise<Float32Array|null>>} */
    this.inflight = new Map()
    this.queue = []
    this.timer = undefined
    this.failures = 0
    this.mutedUntil = 0
  }

  get muted() { return Date.now() < this.mutedUntil }

  /** LRU touch on read so the window of live posts stays resident. */
  cacheGet(t) {
    const v = this.cache.get(t)
    if (v !== undefined) { this.cache.delete(t); this.cache.set(t, v) }
    return v
  }

  cacheSet(t, v) {
    this.cache.set(t, v)
    if (this.cache.size > TUNING.windowSize * 2) {
      const oldest = this.cache.keys().next().value
      if (oldest !== undefined) this.cache.delete(oldest)
    }
  }

  /** @returns {Promise<Float32Array|null>} null = unavailable, do not collapse. */
  embed(text) {
    const t = (text || '').trim()
    if (!t) return Promise.resolve(null)

    const hit = this.cacheGet(t)
    if (hit !== undefined) return Promise.resolve(hit)

    // Breaker: after repeated failures stop hammering a dead offscreen document.
    if (this.muted) return Promise.resolve(null)

    const dup = this.inflight.get(t)
    if (dup) return dup

    const p = new Promise((resolve) => {
      this.queue.push({ text: t, resolve })
      if (this.queue.length >= TUNING.batchMax) this.flush()
      else if (this.timer === undefined) {
        this.timer = setTimeout(() => this.flush(), TUNING.batchWindowMs)
      }
    })
    this.inflight.set(t, p)
    return p
  }

  flush() {
    if (this.timer !== undefined) { clearTimeout(this.timer); this.timer = undefined }
    const batch = this.queue.splice(0, TUNING.batchMax)
    if (!batch.length) return

    // One entry per distinct text, but every waiter for it gets resolved. A timeline
    // re-render asks for the same text many times before the first answer returns.
    const order = []
    const waiters = new Map()
    for (const { text, resolve } of batch) {
      const list = waiters.get(text)
      if (list) list.push(resolve)
      else { waiters.set(text, [resolve]); order.push(text) }
    }

    const settle = (vecs) => {
      order.forEach((text, i) => {
        const v = vecs && vecs[i] ? vecs[i] : null
        if (v) this.cacheSet(text, v)
        waiters.get(text)?.forEach((r) => r(v))
        this.inflight.delete(text)
      })
      if (this.queue.length) this.flush()
    }

    this.transport(order)
      .then((vecs) => { this.failures = 0; settle(vecs) })
      .catch(() => {
        this.failures++
        if (this.failures >= TUNING.breakerTrips) {
          this.mutedUntil = Date.now() + TUNING.breakerMuteMs
          this.failures = 0
        }
        settle(null)          // fail OPEN
      })
  }
}
