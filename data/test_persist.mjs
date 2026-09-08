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
  const { b, s } = quantise(v)
  const r = dequantise(b, s)
  ok(Math.abs(cos(v, r) - 1) < 1e-3, `int8 round-trip preserves direction (cos=${cos(v, r).toFixed(5)})`)
  ok(b.length < 700, `encodes to ${b.length} bytes, not ~8KB of JSON floats`)
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

console.log('\npersistence bounds')
{
  const mem = new Map()
  const fake = {
    get: async (k) => (mem.has(k) ? { [k]: mem.get(k) } : {}),
    set: async (o) => { for (const [k, v] of Object.entries(o)) mem.set(k, v) },
    remove: async (k) => { mem.delete(k) },
  }
  const p = new Persistence(fake)
  for (let i = 0; i < 20; i++) p.remember(`s${i}`, unit(i + 1), `a${i}`)
  await p.flush()
  const p2 = new Persistence(fake)
  const back = await p2.load()
  ok(back.length === 20, `restores what it stored (${back.length}/20)`)
  ok(cos(back.find((e) => e.statusId === 's5').vec, unit(6)) > 0.999, 'vectors survive the round trip')
  await p2.clear()
  ok((await new Persistence(fake).load()).length === 0, 'clear() erases everything')
}

console.log(fails ? `\n${fails} FAILED` : '\nall passed')
process.exit(fails ? 1 : 0)
