const isSafari = location.protocol.startsWith('safari-web-extension:')

const enabledIcons = {
  16: 'icons/icon16.png',
  32: 'icons/icon32.png',
  48: 'icons/icon48.png',
  64: 'icons/icon64.png',
  96: 'icons/icon96.png',
  128: 'icons/icon128.png',
}

const disabledIcons = {
  16: 'icons/icon16-disabled.png',
  32: 'icons/icon32-disabled.png',
  48: 'icons/icon48-disabled.png',
  64: 'icons/icon64-disabled.png',
  96: 'icons/icon96-disabled.png',
  128: 'icons/icon128-disabled.png',
}

function updateToolbarIcon(enabled) {
  let title = chrome.i18n.getMessage(enabled ? 'extensionName' : 'extensionNameDisabled')
  if (chrome.runtime.getManifest().manifest_version == 3) {
    chrome.action.setTitle({title})
    if (!isSafari) {
      chrome.action.setIcon({path: enabled ? enabledIcons : disabledIcons})
    } else {
      chrome.action.setBadgeText({text: enabled ? '' : '⏻'})
    }
  } else {
    chrome.browserAction.setTitle({title})
    chrome.browserAction.setIcon({path: enabled ? enabledIcons : disabledIcons})
  }
}

// Update browser action icon to reflect enabled state
chrome.storage.local.get({enabled: true}, ({enabled}) => {
  updateToolbarIcon(enabled)
})

chrome.storage.local.onChanged.addListener((changes) => {
  if (changes.enabled) {
    updateToolbarIcon(changes.enabled.newValue)
  }
})
// ---------------------------------------------------------------------------
// Semantic dedup: offscreen inference host.
//
// A content script cannot create or address an offscreen document -- only the service
// worker can -- so this relays embed requests. The offscreen document is required
// because x.com's CSP sets connect-src WITHOUT huggingface.co, so a worker running on
// the page origin can start and then never fetch the model weights. On the extension's
// own origin that restriction does not apply, and WebGPU is available too.
// ---------------------------------------------------------------------------

const CPFTDUP_OFFSCREEN = 'offscreen.html'
let cpftDupOffscreenReady = null

async function cpftDupEnsureOffscreen() {
  if (cpftDupOffscreenReady) return cpftDupOffscreenReady
  cpftDupOffscreenReady = (async () => {
    // hasDocument() is not available on every channel; fall back to getContexts.
    try {
      if (chrome.offscreen.hasDocument && await chrome.offscreen.hasDocument()) return
      if (chrome.runtime.getContexts) {
        const ctx = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })
        if (ctx.length) return
      }
    } catch {}
    try {
      await chrome.offscreen.createDocument({
        url: CPFTDUP_OFFSCREEN,
        reasons: ['WORKERS'],
        justification: 'Runs the local sentence-embedding model used to collapse '
          + 'near-duplicate posts. Kept off the page origin because x.com CSP blocks '
          + 'fetching the model there.',
      })
    } catch (err) {
      // A concurrent createDocument loses this race; that is fine, the document exists.
      if (!String(err).includes('Only a single offscreen')) {
        cpftDupOffscreenReady = null
        throw err
      }
    }
  })()
  return cpftDupOffscreenReady
}

chrome.runtime.onMessage.addListener((msg, sender, respond) => {
  if (msg?.type !== 'cpftdup-embed') return
  ;(async () => {
    try {
      await cpftDupEnsureOffscreen()
      const resp = await chrome.runtime.sendMessage({
        type: 'cpftdup-offscreen-embed', texts: msg.texts,
      })
      respond(resp || { ok: false, error: 'no response from offscreen' })
    } catch (err) {
      cpftDupOffscreenReady = null      // rebuild it on the next attempt
      respond({ ok: false, error: String(err?.message || err) })
    }
  })()
  return true                            // async reply
})
