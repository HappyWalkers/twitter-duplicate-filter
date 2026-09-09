/**
 * Cluster store: greedy first-match assignment over a rolling window.
 *
 * Deliberately the SAME algorithm the benchmark scores with `greedy_ari`, rather than
 * an idealised clustering. The first post seen for a story becomes its representative
 * and later posts join the first representative they exceed the threshold against.
 * That is order-dependent and scores worse than optimal clustering -- which is exactly
 * why the benchmark measures it, so the reported number describes what ships.
 *
 * All state is keyed by STATUS ID, never by element. X virtualises the timeline and
 * destroys/recycles article nodes on scroll, so a representative's DOM node is
 * routinely gone by the time its seventh duplicate arrives. Keying by id also makes
 * re-rendering idempotent for free: every observer pass can re-apply the current state
 * without tracking what it already did.
 */
import { MODEL, TUNING } from './config.js'

export class ClusterStore {
  constructor(threshold = MODEL.threshold) {
    this.threshold = threshold
    /** statusId -> {vec, clusterId, author, prior} */
    this.posts = new Map()
    /** clusterId -> {repId, members, live} */
    this.clusters = new Map()
    /** insertion order of LIVE ids */
    this.order = []
    /**
     * Representatives in insertion order, as a flat array of {id, vec, author}.
     *
     * Kept alongside `clusters` purely for the matching scan. With no window there can be
     * tens of thousands of representatives and this loop runs for every new post, so the
     * cost of iterating a Map and doing a `posts.get()` per entry stops being negligible:
     * a flat array keeps the hot path to an indexed walk over objects that are already
     * adjacent. Insertion order is preserved, so first-match semantics are unchanged.
     */
    this.reps = []
  }

  /**
   * Seed posts remembered from earlier sessions. They can be matched against but are not
   * on screen, so they are tracked separately from live ones.
   *
   * @param {Array<{statusId:string, vec:Float32Array, author:string}>} entries
   */
  seedPrior(entries) {
    for (const { statusId, vec, author } of entries) {
      if (!vec || this.posts.has(statusId)) continue
      this.posts.set(statusId, { vec, clusterId: statusId, author, prior: true })
      this.clusters.set(statusId, { repId: statusId, members: [statusId], live: [] })
      this.reps.push({ id: statusId, vec, author, clusterId: statusId })
    }
  }

  static cosine(a, b) {
    let s = 0
    for (let i = 0; i < a.length; i++) s += a[i] * b[i]
    return s        // vectors are L2-normalised at the source
  }

  /**
   * @returns {{clusterId:string, isRepresentative:boolean, size:number}|null}
   *   null when the post cannot be placed (no vector) -- caller must not collapse.
   *
   * The model is the ONLY thing that groups posts. There was once a rule-based pre-pass
   * that joined posts sharing an identical image (the opaque key in
   * pbs.twimg.com/media/<KEY>), on the theory that much of X's duplication is mechanical.
   * It was removed, for two reasons worth keeping written down:
   *
   *   * It did almost nothing -- 29 pairs caught out of 4.3M measured, recall 0.004.
   *   * It bypassed the threshold, so its mistakes were unbounded by any calibration. A
   *     quote-tweet's <article> contains the QUOTED post's image, so every quote-tweet of
   *     one source inherited that source's key and folded together regardless of what it
   *     actually said. "Welcome to the singularity" was hidden behind an OpenAI
   *     announcement it scored 0.36 against, on a threshold of 0.94.
   *
   * Removing it also makes the shipped behaviour match the calibration: collapse_sim.py
   * only ever modelled the vector path, so rule-based folds were extra ones no measured
   * precision figure covered.
   */
  add(statusId, vec, author) {
    const known = this.posts.get(statusId)
    if (known) return this.view(known.clusterId, statusId)
    if (!vec) return null

    let hit = null

    // First representative above threshold wins. Inlined rather than calling
    // cosine() per candidate, and bailing out of the dot product early is deliberately
    // NOT done -- a partial sum says nothing about the final one for signed vectors.
    if (!hit && vec) {
      const t = this.threshold
      const reps = this.reps
      for (let r = 0; r < reps.length; r++) {
        const rep = reps[r]
        // Self-threads and reply chains are legitimately repetitive; collapsing an
        // author against their own earlier post hides a thread, not a duplicate.
        if (TUNING.exemptSameAuthor && rep.author && rep.author === author) continue
        const b = rep.vec
        let sum = 0
        for (let i = 0; i < vec.length; i++) sum += vec[i] * b[i]
        if (sum >= t) { hit = rep.clusterId; break }
      }
    }

    if (!hit) {
      hit = statusId          // this post becomes its own representative
      this.clusters.set(hit, { repId: statusId, members: [], live: [] })
      this.reps.push({ id: statusId, vec, author, clusterId: hit })
    }
    const c = this.clusters.get(hit)
    // A cluster restored from disk has no member on screen. Folding into it would make
    // this post disappear with no chip to expand -- the user would lose the story
    // entirely and have no way to notice. So the first live post to rejoin a remembered
    // cluster BECOMES its representative and stays visible; only the ones after it fold.
    // Cross-session memory therefore turns "seen ten times" into "seen once", never into
    // "never seen".
    if (!c.live.length) c.repId = statusId
    c.members.push(statusId)
    c.live.push(statusId)

    this.posts.set(statusId, { vec, clusterId: hit, author, prior: false })
    this.order.push(statusId)
    this.evict()
    return this.view(hit, statusId)
  }

  /**
   * A remembered post has appeared on screen: make it a live member of its own cluster.
   *
   * Without this, cross-session memory silently disables the extension as it fills up.
   * seedPrior() puts remembered posts into `posts`, and the observer used to treat
   * anything in `posts` as already handled -- so every post the reader had seen before
   * was skipped entirely: not counted, not embedded, and unable to form or join a
   * cluster. After a few sessions most of the timeline was invisible to it.
   *
   * No re-embedding: the vector came back from storage with the post, so this is
   * bookkeeping only.
   */
  promote(statusId) {
    const p = this.posts.get(statusId)
    if (!p) return null
    if (!p.prior) return this.view(p.clusterId, statusId)
    const c = this.clusters.get(p.clusterId)
    if (!c) return null
    p.prior = false
    // Same rule as add(): the first live post in a remembered cluster becomes its
    // visible representative, so nothing is ever folded behind something off-screen.
    if (!c.live.length) c.repId = statusId
    c.live.push(statusId)
    this.order.push(statusId)
    return this.view(p.clusterId, statusId)
  }

  view(clusterId, statusId) {
    const c = this.clusters.get(clusterId)
    if (!c) return null
    return {
      clusterId,
      isRepresentative: c.repId === statusId,
      // Counts only posts present in THIS session: the chip promises "+N similar" and
      // expanding must reveal exactly N. Counting remembered posts would promise more
      // than the page can show.
      size: c.live.length,
      members: c.live,
    }
  }

  /** Drop the oldest posts once the window is exceeded.
   *
   *  Disabled by default (TUNING.windowSize = 0). A window was originally imposed to
   *  bound memory, but it was also silently discarding matches: measured on the labelled
   *  corpus, a 400-post window caught only 62% of true duplicate pairs, because the median
   *  distance between two posts about one story is 237 posts and the 90th percentile is
   *  954. Keeping everything for the session catches ~100% of them.
   *
   *  Set windowSize > 0 to re-enable. A cluster whose representative is evicted keeps its
   *  remaining members but stops accepting new ones -- preferable to promoting a new
   *  representative, which would silently change what later posts are compared against
   *  mid-session. */
  evict() {
    if (!TUNING.windowSize) return
    while (this.order.length > TUNING.windowSize) {
      const id = this.order.shift()
      const p = this.posts.get(id)
      if (!p) continue
      this.posts.delete(id)
      const c = this.clusters.get(p.clusterId)
      if (c) {
        c.members = c.members.filter((m) => m !== id)
        c.live = c.live.filter((m) => m !== id)
        // Keep the cluster while any remembered member remains: it is still a valid
        // thing for a future post to match against.
        if (!c.members.length) this.clusters.delete(p.clusterId)
      }
    }
  }

  /** Ids of every non-representative member of a multi-post cluster -- i.e. exactly the
   *  posts that should be carrying .CpftDup right now, for any of them still in the DOM. */
  duplicateIds() {
    const out = []
    for (const c of this.clusters.values()) {
      if (c.live.length < 2) continue
      for (const id of c.live) if (id !== c.repId) out.push(id)
    }
    return out
  }

  stats() {
    let multi = 0, collapsed = 0, remembered = 0
    for (const c of this.clusters.values()) {
      if (c.live.length > 1) { multi++; collapsed += c.live.length - 1 }
      if (!c.live.length) remembered++
    }
    // `order` holds exactly the ids seen on screen this session. Deriving the count from
    // posts.size minus remembered clusters double-counted every remembered post that
    // reappeared, which is the sort of arithmetic that makes a counter quietly wrong.
    return { posts: this.order.length, clusters: this.clusters.size - remembered,
             multi, collapsed, remembered }
  }
}
