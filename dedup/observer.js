/**
 * Timeline observer: extract -> embed -> cluster -> collapse.
 *
 * Runs in the ISOLATED world alongside Control Panel for Twitter's own content script,
 * and deliberately does NOT touch CPFT's script.js. Three reasons that matters:
 *
 *  1. CPFT's onTimelineChange() is fully SYNCHRONOUS; embedding is async. Splicing an
 *     await into it would mean rewriting its core loop.
 *  2. script.js is a 305KB monolith that upstream changes weekly. Every line we add to
 *     it is a future merge conflict; we add none.
 *  3. CPFT runs in the MAIN world and has no chrome.* access. We need chrome.runtime to
 *     reach the offscreen document, so we must live over here anyway.
 *
 * Coordination is by CSS class only: CPFT owns .HiddenTweet, we own .CpftDup. We skip
 * anything CPFT already hid, and CSS unions the two. Neither side reads the other's
 * bookkeeping.
 */
import { MODEL, TUNING } from './config.js'
import { Embedder } from './embedder.js'
import { ClusterStore } from './cluster.js'
import { offscreenTransport } from './transport.js'

const ARTICLE = 'article[data-testid="tweet"]'
const MARK = 'data-cpftdup'          // stamped so a post is processed once per id

const embedder = new Embedder(offscreenTransport)
const store = new ClusterStore(MODEL.threshold)
let enabled = true
let debugScores = false
// Count of posts that received a REAL vector. This is the only trustworthy signal that
// the model actually loaded and ran: a network-request listener cannot see the offscreen
// document's traffic, so "0 huggingface requests" is a measurement gap, not evidence.
let embeddedCount = 0
let embedErrors = 0

/** Pull the fields we need out of one rendered post. Returns null for anything we must
 *  not touch (ads, unavailable posts, posts CPFT already hid). */
function extract(article) {
  const item = article.closest('[data-testid="cellInnerDiv"]') || article.parentElement
  if (!item) return null
  // Never fight CPFT: if it hid this, leave it hidden and do not count it as a cluster
  // member either -- a hidden post is not a duplicate the user can see.
  if (item.firstElementChild?.classList.contains('HiddenTweet')) return null

  const timeEl = article.querySelector('a[href*="/status/"] time')
  const m = timeEl?.closest('a')?.getAttribute('href')?.match(/\/([^/]+)\/status\/(\d+)/)
  if (!m) return null

  const text = article.querySelector('div[data-testid="tweetText"]')?.innerText || ''
  if (text.trim().length < TUNING.minTextLength) return null   // too short to judge

  // Exact signal: the opaque media id in pbs.twimg.com/media/<KEY>. Same image posted
  // by different accounts yields the same key.
  let exactKey = null
  const img = article.querySelector('img[src*="twimg.com/media/"]')
  if (img) exactKey = img.src.match(/\/media\/([A-Za-z0-9_-]+)/)?.[1] || null

  return { item, statusId: m[2], author: m[1], text, exactKey }
}

function render(item, info, statusId) {
  const first = item.firstElementChild
  if (!first) return
  const dup = info && !info.isRepresentative && info.size > 1
  first.classList.toggle('CpftDup', !!dup)

  if (info?.isRepresentative && info.size > 1) {
    let chip = item.querySelector('.CpftDupChip')
    if (!chip) {
      chip = document.createElement('button')
      chip.className = 'CpftDupChip'
      chip.type = 'button'
      chip.addEventListener('click', (e) => {
        e.stopPropagation(); e.preventDefault()
        const cid = chip.getAttribute('data-cluster')
        document.documentElement.classList.toggle(`CpftDupOpen-${cid}`)
        chip.setAttribute('aria-expanded',
          document.documentElement.classList.contains(`CpftDupOpen-${cid}`) ? 'true' : 'false')
      })
      first.appendChild(chip)
    }
    chip.setAttribute('data-cluster', info.clusterId)
    chip.setAttribute('aria-expanded', 'false')
    chip.textContent = `⌄ +${info.size - 1} similar post${info.size > 2 ? 's' : ''}`
  }
  if (dup) first.setAttribute('data-cpftdup-of', info.clusterId)
  if (debugScores && info) first.setAttribute('data-cpftdup-size', String(info.size))
}

/** Re-apply current cluster state to everything on screen. Cheap and idempotent, which
 *  is what makes virtualisation survivable: when X recycles a node back into view we
 *  simply paint it again from the id-keyed store. */
function repaint() {
  for (const article of document.querySelectorAll(ARTICLE)) {
    const info = extract(article)
    if (!info) continue
    const known = store.posts.get(info.statusId)
    if (known) render(info.item, store.view(known.clusterId, info.statusId), info.statusId)
  }
}

/** Mirror cluster state onto <html> as data attributes.
 *
 * Necessary because window.__cpftDup lives in the ISOLATED world, and anything running
 * in the page's main world -- including Playwright's page.evaluate() -- cannot see it.
 * DOM attributes cross that boundary, so this is what makes the extension observable
 * to an end-to-end test (and to devtools) without exposing anything to x.com's own JS
 * beyond inert attributes. */
function publishStats() {
  const s = store.stats()
  const el = document.documentElement
  el.setAttribute('data-cpftdup-posts', String(s.posts))
  el.setAttribute('data-cpftdup-clusters', String(s.clusters))
  el.setAttribute('data-cpftdup-multi', String(s.multi))
  el.setAttribute('data-cpftdup-collapsed', String(s.collapsed))
  el.setAttribute('data-cpftdup-model', MODEL.id)
  el.setAttribute('data-cpftdup-embedded', String(embeddedCount))
  el.setAttribute('data-cpftdup-embed-errors', String(embedErrors))
  el.setAttribute('data-cpftdup-ready', '1')
  // Ids of the posts we believe are collapsed. Without this, "store says 1 collapsed,
  // DOM shows 0 .CpftDup" is unfalsifiable: it reads identically whether the duplicate
  // scrolled out of the DOM (fine) or its class write was dropped (a bug). Publishing
  // the ids lets a test ask the only question that settles it -- is that specific post
  // still on the page, and if so is it marked? Inert, and devtools-friendly.
  el.setAttribute('data-cpftdup-dupids', store.duplicateIds().join(','))
}

async function scan() {
  if (!enabled) return
  const pending = []
  for (const article of document.querySelectorAll(ARTICLE)) {
    const info = extract(article)
    if (!info) continue
    if (store.posts.has(info.statusId)) {
      render(info.item, store.view(store.posts.get(info.statusId).clusterId, info.statusId),
        info.statusId)
      continue
    }
    if (article.getAttribute(MARK) === info.statusId) continue   // already queued
    article.setAttribute(MARK, info.statusId)
    pending.push(info)
  }
  if (!pending.length) return

  await Promise.all(pending.map(async (p) => {
    const vec = await embedder.embed(p.text)
    if (vec) embeddedCount++; else embedErrors++
    // vec === null means the model is unavailable. Fail OPEN: the post is simply not
    // clustered, so nothing collapses. A dead model must never blank the feed.
    const info = store.add(p.statusId, vec, p.author, p.exactKey)
    if (info) render(p.item, info, p.statusId)
  }))
  repaint()      // a late arrival can turn an existing post into a representative
  publishStats()
}

let scheduled = false
function schedule() {
  if (scheduled) return
  scheduled = true
  // rAF keeps DOM writes inside a frame. NOTE for automated testing: Chrome PAUSES
  // requestAnimationFrame in hidden/background tabs, so a headless run computes the
  // right clusters and then silently drops every write -- indistinguishable from a
  // broken filter. Assert document.visibilityState first when testing this.
  requestAnimationFrame(() => { scheduled = false; scan() })
}

export function start(opts = {}) {
  enabled = opts.enabled !== false
  debugScores = !!opts.debugScores
  // Observe documentElement, NOT document.body: the host content script runs at
  // document_start, where <body> does not exist yet and observe(null) throws. This was
  // intermittent -- a slow page load let body appear first and the bug hid -- which is
  // exactly the kind of race that ships and then fails on fast machines.
  const mo = new MutationObserver(schedule)
  mo.observe(document.documentElement, { childList: true, subtree: true })
  publishStats()          // mark presence immediately, before the first embed resolves
  schedule()
  return {
    stop() { mo.disconnect() },
    stats: () => store.stats(),
    setEnabled(v) {
      enabled = v
      if (!v) {
        for (const el of document.querySelectorAll('.CpftDup')) el.classList.remove('CpftDup')
        for (const el of document.querySelectorAll('.CpftDupChip')) el.remove()
      } else schedule()
    },
  }
}
