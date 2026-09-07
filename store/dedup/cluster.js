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
    /** statusId -> {vec, clusterId, author} */
    this.posts = new Map()
    /** clusterId -> {repId, members: string[]} */
    this.clusters = new Map()
    /** insertion order of ids, for windowing */
    this.order = []
  }

  static cosine(a, b) {
    let s = 0
    for (let i = 0; i < a.length; i++) s += a[i] * b[i]
    return s        // vectors are L2-normalised at the source
  }

  /**
   * @returns {{clusterId:string, isRepresentative:boolean, size:number}|null}
   *   null when the post cannot be placed (no vector) -- caller must not collapse.
   */
  add(statusId, vec, author, exactKey = null) {
    const known = this.posts.get(statusId)
    if (known) return this.view(known.clusterId, statusId)
    if (!vec && !exactKey) return null

    let hit = null

    // Tier 1: exact signal (identical media). Free and precise (0.714) but almost no
    // recall (0.005 measured), so it is a cheap pre-pass, never the mechanism.
    if (TUNING.useExactSignals && exactKey) {
      for (const [cid, c] of this.clusters) {
        if (c.exactKeys && c.exactKeys.has(exactKey)) { hit = cid; break }
      }
    }

    // Tier 2: nearest representative above threshold, first match wins.
    if (!hit && vec) {
      for (const [cid, c] of this.clusters) {
        const rep = this.posts.get(c.repId)
        if (!rep?.vec) continue
        // Self-threads and reply chains are legitimately repetitive; collapsing an
        // author against their own earlier post hides a thread, not a duplicate.
        if (TUNING.exemptSameAuthor && rep.author && rep.author === author) continue
        if (ClusterStore.cosine(vec, rep.vec) >= this.threshold) { hit = cid; break }
      }
    }

    if (!hit) {
      hit = statusId          // this post becomes its own representative
      this.clusters.set(hit, { repId: statusId, members: [], exactKeys: new Set() })
    }
    const c = this.clusters.get(hit)
    c.members.push(statusId)
    if (exactKey) c.exactKeys.add(exactKey)

    this.posts.set(statusId, { vec, clusterId: hit, author })
    this.order.push(statusId)
    this.evict()
    return this.view(hit, statusId)
  }

  view(clusterId, statusId) {
    const c = this.clusters.get(clusterId)
    if (!c) return null
    return {
      clusterId,
      isRepresentative: c.repId === statusId,
      size: c.members.length,
      members: c.members,
    }
  }

  /** Drop the oldest posts once the window is exceeded. A cluster whose representative
   *  is evicted keeps its remaining members but stops accepting new ones -- preferable
   *  to promoting a new representative, which would silently change what later posts
   *  are compared against mid-session. */
  evict() {
    while (this.order.length > TUNING.windowSize) {
      const id = this.order.shift()
      const p = this.posts.get(id)
      if (!p) continue
      this.posts.delete(id)
      const c = this.clusters.get(p.clusterId)
      if (c) {
        c.members = c.members.filter((m) => m !== id)
        if (!c.members.length) this.clusters.delete(p.clusterId)
      }
    }
  }

  /** Ids of every non-representative member of a multi-post cluster -- i.e. exactly the
   *  posts that should be carrying .CpftDup right now, for any of them still in the DOM. */
  duplicateIds() {
    const out = []
    for (const c of this.clusters.values()) {
      if (c.members.length < 2) continue
      for (const id of c.members) if (id !== c.repId) out.push(id)
    }
    return out
  }

  stats() {
    let multi = 0, collapsed = 0
    for (const c of this.clusters.values()) {
      if (c.members.length > 1) { multi++; collapsed += c.members.length - 1 }
    }
    return { posts: this.posts.size, clusters: this.clusters.size, multi, collapsed }
  }
}
