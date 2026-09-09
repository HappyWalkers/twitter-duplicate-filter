/**
 * Content script. Starts the timeline observer and nothing else.
 *
 * Runs in the ISOLATED world at document_start. It needs chrome.runtime to reach the
 * offscreen document where inference happens, which the page's main world cannot use,
 * so this is the only place the observer can live.
 *
 * Deliberately does NOT fight other extensions. Control Panel for Twitter marks posts
 * it hides with .HiddenTweet and this skips those, so the two cooperate when both are
 * installed: CPFT owns .HiddenTweet, we own .CpftDup, CSS unions them, and neither side
 * reads the other's bookkeeping.
 */
const KEY = 'dedupEnabled'
const TAU = 'dedupThreshold'
const SEEN = 'dedupHideSeen'

async function boot() {
  let enabled = true
  // Default matches config.js MODEL.threshold. PRESETS is the whitelist: an older build
  // offered 0.89/0.85/0.80, and those values survive in storage across an update. Without
  // this check a user who once chose "Eager" would silently keep running at 0.80 -- a
  // threshold this build never offers and whose precision it does not claim -- while the
  // popup's dropdown showed blank because the value matches no option.
  const PRESETS = [0.94, 0.92, 0.89]
  let threshold = 0.94
  // Off by default: it hides posts outright, and a reader who has not asked for that
  // should never have content disappear on an upgrade.
  let hideSeen = false
  try {
    const stored = await chrome.storage.local.get([KEY, TAU, SEEN])
    enabled = stored[KEY] !== false
    hideSeen = stored[SEEN] === true
    if (PRESETS.includes(stored[TAU])) threshold = stored[TAU]
    else if (stored[TAU]) chrome.storage.local.remove(TAU)   // stale: fall back to default
  } catch {
    // Storage can be unavailable during an extension update; default to on rather than
    // silently doing nothing.
  }

  const mod = await import(chrome.runtime.getURL('dedup/observer.js'))
  const api = mod.start({ enabled, threshold, hideSeen })

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local') return
    if (KEY in changes) api.setEnabled(changes[KEY].newValue !== false)
    if (TAU in changes && PRESETS.includes(changes[TAU].newValue)) {
      api.setThreshold(changes[TAU].newValue)
    }
    // "Forget" erases the stored copy, but this tab still holds one in memory and would
    // flush it straight back on the next timer -- which looks exactly like the button not
    // working. Drop the in-memory copy too, so the erase is real without a reload.
    if (SEEN in changes) api.setHideSeen(changes[SEEN].newValue === true)
    if ('dedupForgetAt' in changes) api.forgetAll()
  })

  // Publish stats for the popup. Polling rather than pushing on every mutation: the
  // observer fires on each timeline change, which on an active feed is many times a
  // second, and a storage write per change would be pure overhead for a number nobody
  // reads unless the popup is open.
  setInterval(() => {
    try { chrome.storage.local.set({ lastStats: api.stats() }) } catch {}
  }, 2000)
}

boot().catch((err) => console.error('[timeline-dedup] failed to start', err))
