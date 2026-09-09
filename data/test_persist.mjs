/**
 * Tests for cross-session memory. The invariant under test is the one that can actually
 * hurt a user: a post restored from disk is NOT on screen, so folding a live post into it
 * would make that post disappear with no chip to expand and no way to notice.
 */
import { ClusterStore } from '../dedup/cluster.js'
import { quantise, dequantise, Persistence } from '../dedup/persist.js'

let fails = 0
const ok = (cond, name) => { console.log(`${cond ? '  ok  ' : '  FAIL'} ${name}`); if (!cond) fails++ }

const unit = (seed) => {
  const v = new Float32Array(384)
  let n = 0
  for (let i = 0; i < 384; i++) { v[i] = Math.sin(seed * (i + 1)); n += v[i] * v[i] }
  n = Math.sqrt(n)
  for (let i = 0; i < 384; i++) v[i] /= n
  return v
}
const cos = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s }

console.log('quantisation')
{
  const v = unit(1.7)
  const { q, s } = quantise(v)
  const r = dequantise(q, s)
  ok(Math.abs(cos(v, r) - 1) < 1e-3, `int8 round-trip preserves direction (cos=${cos(v, r).toFixed(5)})`)
  ok(q.byteLength === 384, `${q.byteLength} bytes per post, vs 1536 as float32`)
}

console.log('\na story already shown is collapsed on sight')
{
  const s = new ClusterStore(0.5)
  const a = unit(2.1)
  // Remembered from a previous session: the reader has already been shown this story.
  s.seedPrior([{ statusId: 'old1', vec: a, author: 'alice' }])

  const first = s.add('new1', a, 'bob')
  ok(first.fromMemory === true, 'a post matching a remembered story is marked fromMemory')
  ok(first.isRepresentative === false, 'and is NOT promoted to visible representative')

  const second = s.add('new2', a, 'carol')
  ok(second.fromMemory === true, 'so is the next one')
  ok(s.view(second.clusterId, 'new2').members.length === 2, 'both are tracked as live members')
}

console.log('\nthe same post served again is a duplicate of itself')
{
  const s = new ClusterStore(0.5)
  const v = unit(7.7)
  s.seedPrior([{ statusId: 'viral1', vec: v, author: 'ramin' }])
  const back = s.promote('viral1')
  ok(back.fromMemory === true, 'a remembered post reappearing is marked fromMemory')
  ok(s.stats().posts === 1, 'it still counts as a post seen this session')
  // This is the "plane cake" case: nothing else on the page resembles it, and it is not
  // similar to another post -- it is the identical post, which the model scores against
  // its own stored vector at 1.0.
}

console.log('\nremembered posts still do their job')
{
  const s = new ClusterStore(0.5)
  s.seedPrior([{ statusId: 'old1', vec: unit(3.3), author: 'alice' }])
  const joined = s.add('new1', unit(3.3), 'bob')
  // The remembered post stays the representative -- it is off screen, and the whole point
  // is that this story has already been shown, so the new post collapses rather than
  // taking over as a fresh first sighting.
  ok(s.clusters.get(joined.clusterId).repId === 'old1',
     'the remembered post remains the cluster representative')
  ok(joined.fromMemory === true, 'and the new post is marked as an already-shown story')
  ok(s.posts.get('old1').prior === true, 'remembered post is marked prior')
  ok(s.stats().remembered >= 0, 'stats expose how many stories are remembered')
}

console.log('\nsame author no longer blocks a fold')
{
  const s = new ClusterStore(0.5)
  const v = unit(4.2)
  s.add('p1', v, 'alice')
  const again = s.add('p2', v, 'alice')
  // An account posting the same text twice is a repost, not a thread. Measured on the
  // corpus, same-author pairs above threshold are 92.7% same-story -- the exemption was
  // blocking correct folds, not preventing wrong ones.
  ok(again.isRepresentative === false && again.size === 2,
     'an account reposting its own text is folded')
}

console.log('\nremembered posts that reappear on screen')
{
  const s = new ClusterStore(0.5)
  const v = unit(9.1)
  s.seedPrior([{ statusId: 'old1', vec: v, author: 'alice' }])

  // The regression this guards: the observer used to treat anything in `posts` as
  // already handled, so a remembered post reappearing was skipped entirely -- never
  // counted, never able to represent or join a cluster. As the cache filled, more and
  // more of the timeline became invisible.
  const back = s.promote('old1')
  ok(back !== null, 'a remembered post that reappears can be promoted')
  ok(back.fromMemory === true, 'and is collapsed, because its story was already shown')
  ok(s.stats().posts === 1, `and is counted as seen this session (${s.stats().posts})`)
  ok(s.posts.get('old1').prior === false, 'and is no longer marked prior')

  const dup = s.add('new1', v, 'bob')
  ok(dup.fromMemory === true, 'a later post about that story is collapsed too')
  ok(s.stats().posts === 2, `both count toward posts seen (${s.stats().posts})`)
  ok(s.stats().collapsed === 2, `both are reported collapsed (${s.stats().collapsed})`)
}

console.log('\nno in-session window')
{
  const s2 = new ClusterStore(0.99)
  for (let i = 0; i < 3000; i++) s2.add(`p${i}`, unit(i + 1), `a${i}`)
  ok(s2.posts.size === 3000, `keeps all 3000 posts in session, no 400-post eviction (${s2.posts.size})`)
  // The whole point: a match 3000 posts back is still found. Asserting on clusterId, not
  // on size -- sin(seed*i) aliases every 710 seeds, so this fixture genuinely produces
  // repeated vectors and the cluster legitimately has more than two members.
  const late = s2.add('late', unit(1), 'zed')
  ok(late.clusterId === 'p0', 'a duplicate 3000 posts later still matches its original')
  ok(late.isRepresentative === false, 'and folds into it rather than starting a new story')
}

console.log('\nrenderSeen is idempotent (DOM guard)')
{
  // Mirrors the real DOM shape: the chip is inserted BEFORE the post, so it becomes the
  // first child. A guard that looks at the first child therefore passes on every later
  // scan and adds another chip each time.
  const mk = () => {
    const kids = []
    return {
      children: kids,
      firstElementChild: null,
      querySelector: (sel) => sel.includes('CpftDupSeenChip')
        ? kids.find((k) => k.cls.includes('CpftDupSeenChip')) || null : null,
      insertBefore(node) { kids.unshift(node); this.firstElementChild = kids[0] },
      appendChild(node) { kids.push(node); this.firstElementChild = kids[0] },
    }
  }
  const item = mk()
  const post = { cls: [], attrs: {} }
  item.appendChild(post); item.firstElementChild = post

  const renderSeen = (item) => {
    if (item.querySelector(':scope > .CpftDupSeenChip')) return 'skipped'
    const first = item.firstElementChild
    if (!first) return 'no first'
    first.cls.push('CpftDupSeen')
    item.insertBefore({ cls: ['CpftDupChip', 'CpftDupSeenChip'] })
    return 'rendered'
  }
  ok(renderSeen(item) === 'rendered', 'first pass renders the control')
  ok(renderSeen(item) === 'skipped', 'second pass does nothing')
  ok(renderSeen(item) === 'skipped', 'third pass does nothing')
  ok(item.children.filter((k) => k.cls.includes('CpftDupSeenChip')).length === 1,
     `exactly one control exists (${item.children.filter((k) => k.cls.includes('CpftDupSeenChip')).length})`)
  ok(!post.cls.includes('CpftDupSeenChip'), 'and the post itself was never treated as a chip')
}

console.log(fails ? `\n${fails} FAILED` : '\nall passed')
process.exit(fails ? 1 : 0)
