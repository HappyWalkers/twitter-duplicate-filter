/**
 * Offscreen host: owns the worker, relays embed requests from the service worker.
 * Lazily creates the worker on first use so the model is loaded once per session
 * rather than per request.
 */
let worker
let nextId = 1
const pending = new Map()
let progress = { phase: 'idle', percent: 0 }

function getWorker() {
  if (worker) return worker
  worker = new Worker(chrome.runtime.getURL('dedup/dedup-worker.bundle.js'),
    { type: 'module' })
  worker.onmessage = (ev) => {
    const m = ev.data
    if (m.type === 'progress') {
      if (m.data.status === 'progress_total' && typeof m.data.progress === 'number') {
        progress = { phase: 'loading', percent: Math.round(m.data.progress) }
      } else if (m.data.status === 'ready') {
        progress = { phase: 'ready', percent: 100 }
      }
      chrome.runtime.sendMessage({ type: 'cpftdup-progress', progress }).catch(() => {})
      return
    }
    const w = pending.get(m.id)
    if (!w) return
    pending.delete(m.id)
    if (m.type === 'result') w.resolve(m.vecs)
    else w.reject(new Error(m.message))
  }
  worker.onerror = (e) => {
    // Fail every in-flight request rather than let them hang forever; the caller fails
    // open, so the timeline just stops collapsing instead of freezing.
    for (const [id, w] of pending) { w.reject(new Error(e.message || 'worker crashed')); pending.delete(id) }
    worker = undefined            // next call gets a fresh worker
    progress = { phase: 'error', percent: 0 }
  }
  return worker
}

chrome.runtime.onMessage.addListener((msg, _sender, respond) => {
  if (msg?.type !== 'cpftdup-offscreen-embed') return
  const id = nextId++
  pending.set(id, {
    resolve: (vecs) => respond({ ok: true, vecs: vecs.map((v) => Array.from(v)) }),
    reject: (err) => respond({ ok: false, error: String(err.message || err) }),
  })
  try {
    getWorker().postMessage({ type: 'embed', id, texts: msg.texts })
  } catch (err) {
    pending.delete(id)
    respond({ ok: false, error: String(err) })
  }
  return true                      // keep the message channel open for the async reply
})
