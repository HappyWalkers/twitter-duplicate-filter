/**
 * Content script -> background -> offscreen document -> worker.
 *
 * The hop through background.js exists because a content script cannot create or talk
 * to an offscreen document directly; only the service worker can. See background.js for
 * why the offscreen document is required at all (short version: X's connect-src does
 * not include huggingface.co, so nothing on the page origin can fetch the weights).
 *
 * Resolves an array of Float32Array, or REJECTS -- Embedder catches and fails open.
 */

/** @param {string[]} texts */
export function offscreenTransport(texts) {
  return new Promise((resolve, reject) => {
    let settled = false
    const done = (fn, v) => { if (!settled) { settled = true; fn(v) } }

    // A service worker that has been evicted mid-flight leaves sendMessage hanging.
    // Without this the whole batch's waiters never settle and those posts stay
    // permanently unclassified.
    const timer = setTimeout(() => done(reject, new Error('offscreen timeout')), 60_000)

    try {
      chrome.runtime.sendMessage({ type: 'dedup-embed', texts }, (resp) => {
        clearTimeout(timer)
        if (chrome.runtime.lastError) {
          return done(reject, new Error(chrome.runtime.lastError.message))
        }
        if (!resp?.ok) return done(reject, new Error(resp?.error || 'embed failed'))
        // Vectors cross the message boundary as plain arrays; rehydrate to typed arrays
        // so the cosine loop stays on a fast path.
        done(resolve, resp.vecs.map((v) => (v ? Float32Array.from(v) : null)))
      })
    } catch (err) {
      clearTimeout(timer)
      done(reject, err)
    }
  })
}
