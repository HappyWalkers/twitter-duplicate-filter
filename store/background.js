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
