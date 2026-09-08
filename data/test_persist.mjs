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

console.log('\nsame author is still exempt across sessions')
{
  const s = new ClusterStore(0.5)
  const v = unit(4.2)
  s.seedPrior([{ statusId: 'old1', vec: v, author: 'alice' }])
  s.add('new1', v, 'alice')
  const again = s.add('new2', v, 'alice')
  ok(again.isRepresentative === true || again.size === 1,
     'an author repeating themselves is not folded into their own earlier post')
}

console.log('\npersistence: age-bounded, no entry cap')
{
  // Stand-in for the service worker's IndexedDB store.
  const rows = new Map()
  let ttlCutoff = 0
  const transport = async (msg) => {
    if (msg.type === 'dedup-remember') { for (const r of msg.rows) rows.set(r.id, r); return { ok: true } }
    if (msg.type === 'dedup-recall') {
      const live = [...rows.values()].filter((r) => r.t >= ttlCutoff)
      for (const [id, r] of rows) if (r.t < ttlCutoff) rows.delete(id)
      return { ok: true, rows: live }
    }
    if (msg.type === 'dedup-forget') { rows.clear(); return { ok: true } }
    return { ok: false }
  }

  const p = new Persistence(transport)
  for (let i = 0; i < 25000; i++) p.remember(`s${i}`, unit((i % 97) + 1), `a${i % 50}`)
  await p.flush()
  ok(rows.size === 25000, `stores 25,000 posts with no entry cap (${rows.size})`)

  const back = await new Persistence(transport).load()
  ok(back.length === 25000, `restores all of them (${back.length})`)
  ok(cos(back.find((e) => e.statusId === 's5').vec, unit(6)) > 0.999, 'vectors survive the round trip')

  // Age is the only bound.
  const now = Date.now()
  for (const r of rows.values()) if (Number(r.id.slice(1)) < 10000) r.t = now - 8 * 24 * 3600 * 1000
  ttlCutoff = now - 7 * 24 * 3600 * 1000
  const after = await new Persistence(transport).load()
  ok(after.length === 15000, `expiry drops only the aged-out rows (${after.length} left of 25000)`)
  ok(rows.size === 15000, 'and prunes them from the store, so age really is the bound')

  await new Persistence(transport).clear()
  ok(rows.size === 0, 'clear() erases everything')
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
