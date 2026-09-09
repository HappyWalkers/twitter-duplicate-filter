# Semantic timeline dedup

An addition to [Control Panel for Twitter](https://github.com/insin/control-panel-for-twitter):
when a story breaks, every account you follow posts about it and the same news arrives ten
times in different words. This folds those into one.

Everything on the Chrome Web Store today is a **filter** — per post, hide or show — which
has to know in advance what it is hiding. This is a **clusterer**: it compares posts to
each other and collapses the matches, so it needs to know nothing about the story.

Upstream's own files are untouched apart from four small, rebase-safe edits (manifest,
one dynamic import in `content.js`, offscreen lifecycle in `background.js`, one CSS line).
`script.js` — a 305KB monolith that changes weekly — is not modified at all.

## What ships, and why

`dedup/config.js` is the swap point: model id, dtype, pooling, prompt, dim and threshold
in one object. Changing embedder is an edit there plus a worker rebuild.

**`hotchpotch/bekko-embedding-v1-a8m`, fp32, 384-dim, τ = 0.89, 130MB, MIT.**

At that threshold, on 14,201 labelled posts: **1.96% of the timeline folds, at 95.1%
collapse precision.**

### The benchmark

Eight multilingual embedders, shortlisted from the 398-model MTEB multilingual table by
STS + PairClassification + Clustering (the task types that match this problem) rather than
the headline mean, then filtered to open weights with real ONNX files on the Hub. Scored
on 14,201 posts / 30.3M adjudicated pairs / 343 macro-scoreable stories, using the same
ONNX weights the extension downloads — not fp32 PyTorch, because scoring one and shipping
the other makes every number a polite fiction.

| model | dtype | dim | pooled AUC | macro AUC | best F1 | greedy ARI | download |
|---|---|---|---|---|---|---|---|
| **bekko-a8m** | fp32 | 384 | 0.9174 | **0.9834** | **0.3159** | **0.2001** | **130MB** |
| gte-ml-base | q8 | 768 | **0.9346** | 0.9831 | 0.3012 | 0.1783 | 340MB |
| granite-97m | fp32 | 384 | 0.8941 | 0.9755 | 0.2766 | 0.1529 | 390MB |
| jina-v5-nano-clustering | q8 | 768 | 0.8905 | 0.9454 | 0.1032 | 0.0777 | cc-by-nc |
| embeddinggemma-300m | q8 | 768 | 0.8765 | 0.9609 | 0.2115 | 0.1063 | 309MB |
| mmini-l12 *(control)* | q8 | 384 | 0.8695 | 0.9569 | 0.1749 | 0.0681 | 118MB |
| harrier-270m | q8 | 640 | 0.8633 | 0.9714 | 0.2407 | 0.1272 | 344MB |
| f2llm-160m | q8 | 640 | 0.8549 | 0.9646 | 0.2117 | 0.0878 | 160MB |

**Macro AUC** weights every story equally; **pooled AUC** does not, and one large cluster
supplying half the positive pairs can carry a model to a good pooled number while it is
mediocre everywhere else. That is not hypothetical — on an earlier, more concentrated
version of this corpus `jina-v5-nano-clustering` ranked 2nd pooled and *last* on macro,
riding a single cluster. Its non-commercial licence turned out to cost nothing.

**MTEB rank did not predict this task.** bekko was 6th on the leaderboard shortlist and
1st here; embeddinggemma was 4th and finished 6th. The most-downloaded obvious pick,
`paraphrase-multilingual-MiniLM-L12-v2`, was included as a control precisely to measure
that, and it lands second from bottom.

### bekko vs gte, settled

The two are **statistically indistinguishable**: paired bootstrap over 343 stories gives
bekko **+0.0003, 95% CI [−0.0024, +0.0032]**, P = 0.594 — a coin flip centred on zero.
Stories, not pairs, are the resampling unit: pairs inside one story are anything but
independent, and bootstrapping them would report a CI perhaps an order of magnitude too
narrow.

This is now a conclusion rather than a lack of power, and getting there required
correcting an earlier reading. On a 110-story gold the same test gave **gte +0.0035, CI
[−0.0002, +0.0080], P = 0.968** — one hair from significance, and it was reported as
probably real. Tripling the stories **dissolved** that signal rather than confirming it:
it was noise. The lesson is that a near-miss on significance is not weak evidence of an
effect, and the only way to tell is more data.

So the decision falls to the axes that are not a coin flip. bekko wins best F1 and greedy
ARI outright, folds more of the timeline at matched ~95% collapse precision (1.96% vs
1.48%), downloads 10× smaller and runs 4× faster.

**granite is a cautionary case.** Its macro AUC is 0.9755, third of eight — respectable.
But it cannot reach acceptable collapse precision at *any* threshold, topping out at 83.6%
where the other two exceed 95%; at τ 0.86 it hides 45% of the timeline at 43% precision.
Its similarity distribution is too compressed for a threshold to be picked at all. A model
can rank well and still be unusable, which is why the collapse simulation exists alongside
the ranking metrics.

### q8 is a pessimisation, which is backwards

Measured on WebGPU (RTX 5060, Blackwell adapter): the same model runs **18–48× slower**
quantised — gte 508.9 ms/post at q8 against 10.5 at fp32; paraphrase-MiniLM 116.2 against
6.3. Dequantisation dominates and the GPU never touches its native float paths. So the
shipped model is fp32 on purpose, and "quantise for the browser" is exactly wrong here.

## Choosing the threshold

τ is **not** the best-F1 point, and this is the single most consequential decision in the
project. F1 weighs a false collapse and a missed duplicate equally; this product must not.
A miss costs the user one redundant post — the status quo. A wrong collapse **hides a post
they wanted, and they cannot see what they are missing to know it happened.**

τ comes instead from `data/collapse_sim.py`, which runs the shipped algorithm verbatim —
greedy assignment against cluster representatives, 400-post rolling window, same-author
exempt — and reports the only two numbers the user experiences:

| τ | timeline folded | collapse precision |
|---|---|---|
| 0.75 | 6.6% | 81.8% |
| 0.85 | 2.70% | 91.2% |
| 0.87 | 2.32% | 92.2% |
| **0.89** | **1.96%** | **95.1%** |
| 0.92 | 1.48% | 95.6% |

Dropping from 0.89 to 0.87 gains 0.36pp of coverage while nearly doubling the wrong
collapses — a bad trade when the error is invisible to the person it happens to.

**Two earlier values were wrong, each instructively.**

`τ = 0.50` came from maximising F1, and landed on the sweep's lower boundary — a
constrained optimum, not a real one. It hid 32.6% of the timeline. Sweeping 0.10–0.99 put
the true F1 peak at 0.49, where pairwise precision is 0.408.

`τ = 0.75` came from the right metric but a gold that could not see its own errors. Only
**32%** of collapses were adjudicated then, and it measured 93.4% precision. On the
rebuilt gold, where **63%** are adjudicated, that same threshold measures **81.8%**. The
number did not move because the model changed — it moved because the measurement stopped
being blind. **A precision figure is only as trustworthy as the share of collapses
actually judged**, so read `unjudged_share` in `collapse_sim.json` before believing a
headline number.

A third error was caught before it shipped: the first collapse simulation scored against
the merged clustering rather than the adjudicated pairs, counting every never-judged
collapse as a mistake. It reported 37.9% precision at τ 0.75 and made every threshold look
equally hopeless. The audit that caught it was reading 30 scored-wrong collapses by hand —
most were the model being **right** and the gold being fragmented, one viral wine-bottle
puzzle carrying three story ids.

Three things were measured and **rejected**: raising the 15-char minimum (no precision
gain, −40% coverage), best-match assignment instead of first-match (+1 point, not worth
diverging from what the benchmark scores), and rule-based exact-match signals of any kind
(see below).

### No rule-based signals

The design originally paired the model with a cheap exact-match pre-pass: X serves images
from `pbs.twimg.com/media/<KEY>`, so posts sharing an identical image could be grouped by
string equality before spending any model compute. It shipped, and has been removed.

It caught **29 pairs out of 4.3 million** — recall 0.004. And because it joined clusters
without consulting the threshold, its errors were bounded by no calibration at all. Live
testing found the failure: a quote-tweet's `<article>` contains the *quoted* post's image,
so every quote-tweet of one source inherited that source's media key and folded together
regardless of its own text. A four-word reaction was hidden behind an OpenAI announcement
it scored **0.36** against, on a threshold of **0.94**.

Removing it also closed a gap between what ships and what is measured: `collapse_sim.py`
only ever simulated the vector path, so rule-based folds were extra ones that no reported
precision figure covered. Grouping is now the model's job alone, and every fold is
governed by τ.

## Architecture

```
x.com page
├─ MAIN world ── CPFT script.js               ← untouched; all upstream filters keep working
└─ isolated world ── content.js
     └─ dedup/observer.js
          ├─ MutationObserver on documentElement
          ├─ embed via chrome.runtime → background.js
          │            └─ chrome.offscreen ({reasons:['WORKERS']})
          │                 └─ offscreen.html   ← extension origin: HF fetch + WebGPU allowed
          │                      └─ Worker(dedup-worker.bundle.js) → transformers.js → ORT
          ├─ ClusterStore keyed by status id, never by element
          └─ .CpftDup → CSS collapse + "+N similar" chip
```

**Why the offscreen document.** Measured from live x.com response headers: `worker-src
'self' blob:` and `'wasm-unsafe-eval'` are both permitted, so a worker and WASM are fine —
but **`connect-src` does not include huggingface.co**. A worker on the page origin starts
happily and can never download the weights. Inference has to run on the extension's own
origin, and only the service worker can create an offscreen document, which is why the
content script hops through `background.js` to reach it.

**Why not touch `onTimelineChange`.** It is fully synchronous, embedding is not, and it
lives in a file that changes weekly. Dedup runs as an independent observer applying its own
class; CPFT owns `.HiddenTweet`, we own `.CpftDup`, CSS unions them, neither side reads the
other's bookkeeping.

**State is keyed by status id, never by element.** X virtualises the timeline and recycles
`<article>` nodes on scroll, so a representative's DOM node is routinely gone by the time
its seventh duplicate arrives. Keying by id also makes repainting idempotent for free.

**Fail open.** `embed()` never rejects; on any failure it resolves `null`, and `null` means
never collapse. A dead model degrades to a normal timeline, never to a blank one. A circuit
breaker mutes the transport for 30s after 5 consecutive failures.

### ORT payload

ONNX Runtime ships four wasm variants and picks by feature detection, so the build began by
copying all four — 74MB — rather than guessing wrong and getting a bare "no available
backend found". The set is now pinned to what the built worker can actually request, read
out of the bundle itself:

```
grep -o 'ort-wasm-simd-threaded[a-z.]*\.wasm' dist-dedup/dedup/dedup-worker.bundle.js
```

which names only `asyncify` (WebGPU) and plain (CPU fallback). `jsep` and `jspi` are
referenced only by ORT entry points this build never imports — 41MB for nothing. **76MB →
37MB**, verified end-to-end rather than by grep alone. Re-run that command after bumping
`onnxruntime-web` or `@huggingface/transformers`.

## Data pipeline

| file | what it does |
|---|---|
| `data/capture.py` | CDP-attached Playwright scraper; per-trend and per-source caps |
| `data/cron_capture.sh` | self-healing hourly top-up; alternates trends and rotating query sets |
| `data/label.py` | codex / agy labelling with per-batch checkpointing and `--resume` |
| `data/make_chunks.py` | partitions unlabelled posts by trend for subagent labelling |
| `data/final_judge.py` | cross-chunk merges, ARI, agreed-pairs gold |
| `data/benchmark.py` | the eight-model sweep; pooled + macro AUC, τ sweep, greedy ARI |
| `data/bootstrap.py` | paired bootstrap over stories |
| `data/threshold.py` | wide τ sweep + precision-targeted operating points |
| `data/collapse_sim.py` | simulates the shipped algorithm; collapse rate and precision |
| `data/e2e_test.py` | live-timeline verification against a real logged-in session |

**Corpus**: 14,201 posts across 383 trends/queries, 9,686 gold stories, **343
macro-scoreable**, 11,699 posts double-labelled, **54,256 double-confirmed positive pairs**
(7.5× the previous gold). 24% non-Latin-script. The bottleneck was never post count — it was *distinct stories*, since
pairs scale as C(n,2) and deepening one cluster adds concentration without statistical
power. Retuning collection to 30 trends × cap 16 (from 16 × 30) took macro-scoreable
clusters from 53 to 110 and dropped the top cluster's share of positive pairs from 50.5%
to 21.4%.

**SemEval-2022 Task 8 was evaluated and rejected**: its Chinese side is machine-translated
UK tabloid copy, it yields only 2,432 unique pairs against an official ~9,866, 62 ids appear
in both splits, and its median document is 1,039 characters against our 150.

## Verification

`data/e2e_test.py` runs headed against a real logged-in timeline and checks, in the order
that matters — each gating the next: content script injected and CSS applied → offscreen
document reachable → model downloads on the extension origin → posts cluster and duplicates
collapse → state survives scrolling away and back.

Three traps it exists to avoid, each of which produced a confidently wrong result first:

- **Chrome pauses `requestAnimationFrame` in hidden tabs**, and the observer schedules its
  DOM writes inside rAF. A headless run computes the right clusters and silently drops
  every write — indistinguishable from a broken filter. It asserts `visibilityState` first.
- **Content scripts run in the isolated world; `page.evaluate()` runs in the main world.**
  Probing `window.__cpftDup` or `document.styleSheets` can never work and reported FAIL
  twice. State crosses the boundary as `data-cpftdup-*` attributes on `<html>` instead.
- **"Store says 1 collapsed, DOM shows 0" is unfalsifiable** without knowing *which* posts
  those are — it reads identically whether the duplicate scrolled out of the DOM (fine) or
  its class write was dropped (a bug). The store now publishes the ids, so the test asks
  the only question that settles it: is that specific post still on the page, and is it
  marked?

Chrome 152 has removed `--load-extension` entirely — it registers nothing and prints no
error — so the harness uses Playwright's bundled Chromium. Cookies must be exported over
CDP rather than by copying a profile directory: Chrome encrypts the cookie store with an
OS-keyring key another build cannot read, so the copy loads and lands on the login page
with no error at all.

## Labelling, and the defect that nearly poisoned the gold

Two independent labellers per post; the gold keeps **only pairs both agreed on**, because
each fails in the opposite direction — sonnet over-merges topically-related events, codex
isolates real stories into singletons — and the intersection excludes both failure modes.
Disputed pairs are **dropped, never counted as negatives**: nobody established their truth,
and they are exactly the near-duplicate pairs a model is most likely to be right about.

The prompt that drove labelling had two defects worth recording, because both were silent:

1. **Story descriptions were thrown away.** `labels_<name>.json` stores only
   `statusId → story id`, so a resumed run seeded its known-stories list as
   `- s_2095762088182378571: s_2095762088182378571`. The labeller was being asked to reuse
   ids it had no way to read.
2. **The candidate list was the 180 *biggest* stories, not the relevant ones.** Once the
   corpus held thousands of stories, the one a post actually belonged to was almost never
   on the list, so a fresh id got invented for it.

Both are fixed: descriptions persist to a sidecar (falling back to a story's first member
text), and candidates are retrieved per batch by char-n-gram TF-IDF. Retrieval is
deliberately **lexical, never an embedding model** — the embedders are what this gold
exists to benchmark, and using one to choose which stories the labeller may reuse would
shape the gold around that model's own notion of similarity and inflate its score.

Measured three ways:

| measure | before | after |
|---|---|---|
| story visible in the prompt when it already exists | 85.2% | **98.2%** |
| stories whose representative has a near-twin (fragmentation) | 4.76% | **1.67%** |

A third measure, inter-labeller ARI, moved around too much across coverage levels to
support a claim, and is reported here as inconclusive rather than as evidence.

**Labeller independence is load-bearing and easy to break by accident.** Seeding one
labeller's prompt with another's story names looks like a free win — shared ids, less
duplication — but the gold's whole value is that two labellers agreed *without seeing each
other's answers*. Anchoring inflates agreement into confirmation it never earned. This was
introduced and reverted once during development; the sidecar now reads only its own
labeller's descriptions, and the post-text fallback carries no other labeller's opinion.

For the same reason, second labellers are reconciled against codex **separately and then
unioned**, never merged into one map: sonnet, agy and each subagent invent ids in unrelated
namespaces, so a pair whose members were second-labelled by different labellers would be
recorded as "different story" no matter what it is. Three subagents produced three
different ids for the same Michigan–Western Michigan clock controversy; pooling them would
have manufactured false negatives on the largest story in the corpus.

### Pair storage

The adjudicated pair set is 30.3M pairs. Enumerating it produced a 1.4GB JSON that cost
several GB just to `json.load` and grew quadratically with the overlap, so
`gold_pairs_compact.json` stores what cannot be derived — the overlaps, the 54,256
positives and the 122,367 disputed — in **9MB**, and `benchmark.py` expands it with numpy.

The first version of that format had a real bug: without an explicit `disputed` list, an
expander must assume every non-positive pair in an overlap is a negative, silently
converting all disputed pairs into agreed negatives. A round-trip against the enumerated
file caught it — the expansion came back with precisely `disputed` extra rows. Both the
pair expansion and the collapse-simulation lookup are verified equivalent to the
enumerated form.
