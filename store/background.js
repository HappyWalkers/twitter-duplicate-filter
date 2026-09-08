/**
 * Service worker. Owns the offscreen document's lifecycle and relays embed requests.
 *
 * This hop exists because a content script cannot create or talk to an offscreen
 * document -- only the service worker can. Inference lives in an offscreen document
 * rather than on the page because the page origin is x.com, whose CSP governs what its
 * workers may load; the extension's own origin is unconstrained and is where the
 * bundled model and ONNX Runtime actually live.
 */
const OFFSCREEN = 'offscreen.html'
let offscreenReady = null

async function ensureOffscreen() {
  if (offscreenReady) return offscreenReady
  offscreenReady = (async () => {
    // hasDocument() is not on every channel; fall back to getContexts.
    try {
      if (chrome.offscreen.hasDocument && await chrome.offscreen.hasDocument()) return
      if (chrome.runtime.getContexts) {
        const ctx = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })
        if (ctx.length) return
      }
    } catch {}
    try {
      await chrome.offscreen.createDocument({
        url: OFFSCREEN,
        reasons: ['WORKERS'],
        justification: 'Hosts the worker that runs the bundled sentence-embedding model '
          + 'used to detect posts about the same story. Runs on the extension origin so '
          + 'the model and its WebAssembly runtime load under the extension CSP.',
      })
    } catch (err) {
      // A concurrent createDocument loses this race; that is fine, the document exists.
      if (!String(err).includes('Only a single offscreen')) {
        offscreenReady = null
        throw err
      }
    }
  })()
  return offscreenReady
}

chrome.runtime.onMessage.addListener((msg, sender, respond) => {
  if (msg?.type !== 'dedup-embed') return
  ;(async () => {
    try {
      await ensureOffscreen()
      const resp = await chrome.runtime.sendMessage({
        type: 'dedup-offscreen-embed', texts: msg.texts,
      })
      respond(resp || { ok: false, error: 'no response from offscreen' })
    } catch (err) {
      offscreenReady = null        // rebuild it on the next attempt
      respond({ ok: false, error: String(err?.message || err) })
    }
  })()
  return true                      // async reply
})


/* ---------------------------------------------------------------------------
 * Remembered posts.
 *
 * IndexedDB on the EXTENSION origin, not chrome.storage.local. storage.local is capped
 * at ~10MB (QUOTA_BYTES) with no permission to raise it, which at 512 bytes per post is
 * a hard ceiling around 20,000 posts -- a count limit by the back door, enforced by
 * write failures rather than by anything the user or the code could see. IndexedDB draws
 * on the browser's storage pool instead, and stores typed arrays natively, so the vectors
 * skip base64 and JSON number-text entirely.
 *
 * It has to live here rather than in the content script: a content script's IndexedDB
 * belongs to x.com's origin, where the site could read it and clearing site data would
 * erase it.
 *
 * There is no entry cap. Age is the only bound.
 * --------------------------------------------------------------------------- */
const DB_NAME = 'dedup'
const STORE = 'posts'
/** Weekly. A month is a one-line change, but see the cost: a heavy reader at ~5k
 *  posts/day reaches ~35k posts in a week (13MB on disk, ~58MB of tab memory once
 *  recalled) and ~150k in a month (~58MB disk, ~250MB memory). Memory in the tab, not
 *  disk, is the binding constraint. */
const TTL_MS = 7 * 24 * 60 * 60 * 1000

let dbPromise
function db() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1)
      req.onupgradeneeded = () => {
        const s = req.result.createObjectStore(STORE, { keyPath: 'id' })
        s.createIndex('t', 't')          // for the age sweep and the recall range
      }
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    }).catch((e) => { dbPromise = null; throw e })
  }
  return dbPromise
}

const tx = async (mode, fn) => {
  const d = await db()
  return new Promise((resolve, reject) => {
    const t = d.transaction(STORE, mode)
    const out = fn(t.objectStore(STORE))
    t.oncomplete = () => resolve(out?.result ?? out)
    t.onerror = () => reject(t.error)
    t.onabort = () => reject(t.error)
  })
}

/** Everything still within the TTL. Expired rows are dropped on the way past, so the
 *  store stays bounded by age without a separate sweep task that a service worker
 *  shutdown could skip. */
async function recall() {
  const cutoff = Date.now() - TTL_MS
  const d = await db()
  return new Promise((resolve, reject) => {
    const t = d.transaction(STORE, 'readwrite')
    const s = t.objectStore(STORE)
    const out = []
    s.index('t').openCursor().onsuccess = (e) => {
      const c = e.target.result
      if (!c) return
      if (c.value.t < cutoff) c.delete()
      else out.push(c.value)
      c.continue()
    }
    t.oncomplete = () => resolve(out)
    t.onerror = () => reject(t.error)
  })
}

chrome.runtime.onMessage.addListener((msg, _sender, respond) => {
  if (msg?.type === 'dedup-recall') {
    recall().then((rows) => respond({ ok: true, rows })).catch((e) => respond({ ok: false, error: String(e) }))
    return true
  }
  if (msg?.type === 'dedup-remember') {
    tx('readwrite', (s) => { for (const r of msg.rows) s.put(r) })
      .then(() => respond({ ok: true })).catch((e) => respond({ ok: false, error: String(e) }))
    return true
  }
  if (msg?.type === 'dedup-forget') {
    tx('readwrite', (s) => s.clear())
      .then(() => respond({ ok: true })).catch((e) => respond({ ok: false, error: String(e) }))
    return true
  }
  if (msg?.type === 'dedup-count') {
    tx('readonly', (s) => s.count())
      .then((n) => respond({ ok: true, n })).catch(() => respond({ ok: false, n: 0 }))
    return true
  }
})
