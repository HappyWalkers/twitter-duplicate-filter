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
import { Persistence } from './persist.js'

const ARTICLE = 'article[data-testid="tweet"]'
const MARK = 'data-cpftdup'          // stamped so a post is processed once per id

const embedder = new Embedder(offscreenTransport)
const store = new ClusterStore(MODEL.threshold)
const persist = new Persistence()
let enabled = true
let debugScores = false
// Count of posts that received a REAL vector. This is the only trustworthy signal that
// the model actually loaded and ran: a network-request listener cannot see the offscreen
// document's traffic, so "0 huggingface requests" is a measurement gap, not evidence.
let embeddedCount = 0
let embedErrors = 0
let seenCollapsed = 0

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
  // Empty only. There is no minimum length: a length cutoff is a rule deciding what may
  // be judged, and the model is what judges here. "plane cake" is ten characters and is a
  // post a reader meets over and over; a floor of 15 made it invisible to the extension
  // entirely. Measured on the corpus, removing the floor changes nothing (2.42% folded at
  // 92.7% precision either way), so it was costing that case for no gain.
  if (!text.trim()) return null

  return { item, statusId: m[2], author: m[1], text }
}

/** One reveal rule per collapsed post.
 *
 *  CSS cannot compare a class name to an attribute value, so a single static rule would
 *  reveal every seen post the moment any one of them was expanded. Injecting one narrow
 *  rule per post keeps expansion local. Rules are tiny and bounded by what is on screen.
 */
const seenStyles = new Set()
function ensureSeenStyle(statusId) {
  if (seenStyles.has(statusId)) return
  seenStyles.add(statusId)
  let el = document.getElementById('cpftdup-seen-styles')
  if (!el) {
    el = document.createElement('style')
    el.id = 'cpftdup-seen-styles'
    document.head?.appendChild(el)
  }
  el.sheet?.insertRule(
    `html.CpftDupOpen-${CSS.escape(statusId)} .CpftDupSeen[data-cpftdup-seen="${statusId}"]` +
    `{display:revert !important}`, el.sheet.cssRules.length)
}

/** Collapse a post whose story the model has already placed in a cluster from an earlier
 *  session, leaving a control in its place.
 *
 *  The chip goes on the ITEM, not on the collapsed element: the collapsed element is
 *  display:none, so a chip inside it would be invisible and the post would be gone with no
 *  way to bring it back. */
function renderSeen(item, statusId) {
  const first = item.firstElementChild
  if (!first || first.classList.contains('CpftDupSeen')) return
  first.classList.add('CpftDupSeen')
  first.setAttribute('data-cpftdup-seen', statusId)
  ensureSeenStyle(statusId)
  const chip = document.createElement('button')
  chip.className = 'CpftDupChip CpftDupSeenChip'
  chip.type = 'button'
  chip.textContent = '⌄ seen before'
  chip.addEventListener('click', (e) => {
    e.stopPropagation(); e.preventDefault()
    document.documentElement.classList.toggle(`CpftDupOpen-${statusId}`)
    chip.setAttribute('aria-expanded',
      document.documentElement.classList.contains(`CpftDupOpen-${statusId}`) ? 'true' : 'false')
  })
  // Before the post, not after it. Appending puts the control last, which looks right
  // while the post is hidden (the chip is all there is) and wrong the moment it is
  // expanded -- the control ends up beneath the post it belongs to, and the reader has to
  // scroll past the whole thing to collapse it again.
  item.insertBefore(chip, first)
  seenCollapsed++
}

function render(item, info, statusId) {
  const first = item.firstElementChild
  if (!first) return
  // A story carried over from an earlier session has already been shown, so every post in
  // it now is a repeat -- including the identical post served again, which the model
  // scores against its own remembered vector at 1.0. Those collapse individually, each
  // with its own control, because there is no representative on screen to hang one on.
  if (info?.fromMemory) { renderSeen(item, statusId); return }

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
    if (known && !known.prior) render(info.item, store.view(known.clusterId, info.statusId), info.statusId)
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
  // Posts recalled from an earlier session. Published so a test can prove cross-session
  // memory actually reached IndexedDB and came back, which no in-page number otherwise
  // distinguishes from a fresh start.
  el.setAttribute('data-cpftdup-remembered', String(s.remembered || 0))
  el.setAttribute('data-cpftdup-seen-collapsed', String(seenCollapsed))
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
    const known = store.posts.get(info.statusId)
    if (known) {
      // A post remembered from an earlier session is NOT already handled -- it has a
      // vector but no place on screen yet. Promote it instead of skipping it.
      const view = known.prior ? store.promote(info.statusId)
                               : store.view(known.clusterId, info.statusId)
      render(info.item, view, info.statusId)
      continue
    }
    if (article.getAttribute(MARK) === info.statusId) continue   // already queued
    article.setAttribute(MARK, info.statusId)
    pending.push(info)
  }
  if (!pending.length) return

  await Promise.all(pending.map(async (p) => {
    const vec = await embedder.embed(p.text)
    if (vec) { embeddedCount++; persist.remember(p.statusId, vec, p.author) }
    else embedErrors++
    // vec === null means the model is unavailable. Fail OPEN: the post is simply not
    // clustered, so nothing collapses. A dead model must never blank the feed.
    const info = store.add(p.statusId, vec, p.author)
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
  if (opts.threshold) store.threshold = opts.threshold
  // Observe documentElement, NOT document.body: the host content script runs at
  // document_start, where <body> does not exist yet and observe(null) throws. This was
  // intermittent -- a slow page load let body appear first and the bug hid -- which is
  // exactly the kind of race that ships and then fails on fast machines.
  const mo = new MutationObserver(schedule)
  mo.observe(document.documentElement, { childList: true, subtree: true })
  publishStats()          // mark presence immediately, before the first embed resolves
  schedule()

  // Restore what earlier sessions saw. Deliberately not awaited: the timeline should
  // start deduplicating immediately rather than waiting on storage, and a post embedded
  // before the restore lands simply misses the cross-session match once.
  if (opts.remember !== false) {
    persist.load()
      .then((prior) => { if (prior.length) { store.seedPrior(prior); repaint(); publishStats() } })
      .catch(() => {})
  }
  // Flush on the way out as well as on the timer: a tab closed 9 seconds into the debounce
  // would otherwise lose everything it just learned.
  addEventListener('pagehide', () => persist.flush(), { capture: true })
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') persist.flush()
  })
  return {
    stop() { mo.disconnect() },
    // Include the embed counters, not just cluster shape: "0 collapsed" is ambiguous
    // between a quiet timeline and a model that never started, and a UI showing that
    // number needs to tell those apart.
    stats: () => ({ ...store.stats(), embedded: embeddedCount, embedErrors: embedErrors }),
    /** Applies to comparisons made from now on. Posts already placed keep their
     *  cluster -- re-clustering the whole window would make posts appear and disappear
     *  under the reader mid-scroll, which is worse than waiting for a reload. */
    setThreshold(v) { if (v) store.threshold = v },
    forgetAll: () => persist.clear(),
    rememberedCount: () => persist.size,
    setEnabled(v) {
      enabled = v
      if (!v) {
        for (const el of document.querySelectorAll('.CpftDup')) el.classList.remove('CpftDup')
        for (const el of document.querySelectorAll('.CpftDupChip')) el.remove()
      } else schedule()
    },
  }
}
