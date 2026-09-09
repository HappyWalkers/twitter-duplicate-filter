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

console.log('\nnever hide behind an invisible representative')
{
  const s = new ClusterStore(0.5)
  const a = unit(2.1)
  // Remembered from a previous session; nothing on screen.
  s.seedPrior([{ statusId: 'old1', vec: a, author: 'alice' }])

  const first = s.add('new1', a, 'bob')
  ok(first.isRepresentative === true, 'first live post rejoining a remembered story stays VISIBLE')
  ok(first.size === 1, 'and reports size 1, so no chip claims hidden posts that do not exist')

  const second = s.add('new2', a, 'carol')
  ok(second.isRepresentative === false, 'the SECOND live post folds')
  ok(second.size === 2, 'chip counts live posts only')
  // The cluster keeps the remembered post's id as its key even after a live post takes
  // over as representative -- the id is only a handle, and re-keying would mean rewriting
  // every member's clusterId for a cosmetic gain.
  ok(second.clusterId === first.clusterId, 'both live posts land in the same cluster')
  ok(s.view(second.clusterId, 'new1').members.length === 2,
     'members are the live ones, not the remembered one')
}

console.log('\nremembered posts still do their job')
{
  const s = new ClusterStore(0.5)
  s.seedPrior([{ statusId: 'old1', vec: unit(3.3), author: 'alice' }])
  const joined = s.add('new1', unit(3.3), 'bob')
  ok(s.clusters.get(joined.clusterId).repId === 'new1',
     'live post takes over the remembered cluster as representative')
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
  ok(back.isRepresentative === true, 'and becomes the visible representative of its story')
  ok(s.stats().posts === 1, `and is counted as seen this session (${s.stats().posts})`)
  ok(s.posts.get('old1').prior === false, 'and is no longer marked prior')

  const dup = s.add('new1', v, 'bob')
  ok(dup.isRepresentative === false && dup.size === 2,
     'a later post about that story then folds into it')
  ok(s.stats().posts === 2, `both count toward posts seen (${s.stats().posts})`)
  ok(s.stats().collapsed === 1, `one is reported collapsed (${s.stats().collapsed})`)
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

console.log(fails ? `\n${fails} FAILED` : '\nall passed')
process.exit(fails ? 1 : 0)
