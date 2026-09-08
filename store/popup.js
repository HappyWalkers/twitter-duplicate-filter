/**
 * Popup: on/off plus what the extension has actually done.
 *
 * Reads stats from chrome.storage rather than injecting a script into the tab. Doing it
 * with chrome.scripting.executeScript would work, but it costs the "scripting" and
 * "activeTab" permissions -- two more things to justify in review, and two more
 * capabilities on an extension whose whole pitch is that it needs almost nothing. The
 * content script already knows the numbers; letting it publish them is cheaper.
 */
const KEY = 'dedupEnabled'
const toggle = document.getElementById('toggle')

chrome.storage.local.get(KEY).then((s) => { toggle.checked = s[KEY] !== false })
toggle.addEventListener('change', () => chrome.storage.local.set({ [KEY]: toggle.checked }))

function render(st) {
  if (!st) return
  document.getElementById('posts').textContent = st.posts ?? '–'
  document.getElementById('collapsed').textContent = st.collapsed ?? '–'
  if (st.embedErrors > 0 && st.embedded === 0) {
    document.getElementById('foot').textContent =
      'The model could not start, so nothing is being folded.'
  }
}

chrome.storage.local.get('lastStats').then((s) => render(s.lastStats))
chrome.storage.onChanged.addListener((c, area) => {
  if (area === 'local' && c.lastStats) render(c.lastStats.newValue)
})

/* Sensitivity.
 *
 * This control exists because one number could not serve both cases. The threshold was
 * calibrated on a corpus that is 89% trending-topic pages, where 34% of posts repeat a
 * story already present; a home timeline is 9%. So the same setting that folds a useful
 * amount during breaking news folds roughly one post in 300 on a quiet Following feed.
 * Rather than pick for everyone, the options say plainly what each costs -- the rates
 * are measured on the home-timeline slice, the accuracy on the full labelled corpus.
 */
const TAU = 'dedupThreshold'
const sel = document.getElementById('tau')
const NOTE = {
  '0.89': 'About 19 in 20 folds are correct. Folds rarely on a quiet timeline — roughly one post in 300.',
  '0.85': 'About 9 in 10 folds are correct. Folds roughly one post in 190.',
  '0.80': 'About 6 in 7 folds are correct. Folds roughly one post in 110 — most useful during big news.',
}
function note() { document.getElementById('note').textContent = NOTE[sel.value] || '' }

chrome.storage.local.get(TAU).then((s) => {
  sel.value = String(s[TAU] || 0.89)
  note()
})
sel.addEventListener('change', () => {
  chrome.storage.local.set({ [TAU]: Number(sel.value) })
  note()
})

/* Remembered posts.
 *
 * Cross-session memory means the extension keeps a record of what has been read, so the
 * user gets a count and a way to erase it in the same place they turn the feature on.
 * A privacy control the user cannot find is not a control.
 */
const CACHE_KEY = 'dedupCache'
const forget = document.getElementById('forget')

function showRemembered(raw) {
  const n = raw && Array.isArray(raw.e) ? raw.e.length : 0
  document.getElementById('remembered').textContent = n
  forget.disabled = n === 0
}

chrome.storage.local.get(CACHE_KEY).then((s) => showRemembered(s[CACHE_KEY]))
chrome.storage.onChanged.addListener((c, area) => {
  if (area === 'local' && CACHE_KEY in c) showRemembered(c[CACHE_KEY].newValue)
})

forget.addEventListener('click', async () => {
  await chrome.storage.local.remove(CACHE_KEY)
  showRemembered(null)
  forget.textContent = 'Forgotten'
  // Open tabs drop their in-memory copy too (content.js watches this key), so the erase
  // is complete and the count will not creep back on the next flush.
  document.getElementById('foot').textContent = 'Erased from this device.'
})
