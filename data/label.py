#!/usr/bin/env python3
"""
Ground-truth event clustering for the dedup benchmark, via two independent LLM labellers.

Two labellers, not one, and the agreement between them is measured and reported before
any of it is used. The bilibili project's numbers were only meaningful because its
teachers were scored on human-labelled data BEFORE being trusted to label 10k titles;
the same discipline applies here. A single labeller's opinion is not ground truth, and
presenting it as such would make every downstream AUC a polite fiction.

Outputs:
  labels_codex.json / labels_agy.json  - {statusId: storyId} per labeller
  gold.json                            - connected components of "both said same"
  gold_pairs.json                      - only pairs the two labellers AGREE on
  agreement.json                       - ARI, disagreement counts, adjudication sample

Usage:
  python data/label.py --labeller codex
  python data/label.py --labeller agy
  python data/label.py --reconcile
"""
import argparse, json, re, subprocess, sys, textwrap
from pathlib import Path

HERE = Path(__file__).parent
POSTS = HERE / "posts.jsonl"

# How many already-named stories the prompt carries. Retrieved-by-relevance first, then
# the biggest as filler -- see StoryIndex for why the old flat "180 biggest" fragmented.
# Tuned by measuring the only thing that matters here: for a post whose story ALREADY
# exists, is that story id on the list the labeller sees? On the labelled corpus the old
# 180-biggest showed it 85.2% of the time; this shows it 98.2%, cutting the misses that
# force a fresh id by 88%. Costs ~47KB of prompt, which is the trade being made.
RETRIEVE_PER_POST = 8
RETRIEVE_MAX = 400
BIGGEST_FILL = 60
ALL_POSTS = []

PROMPT = """You are grouping social-media posts by NEWS EVENT / STORY.

Two posts share a story if they are about the SAME specific real-world event, claim, or
announcement - even when the wording, language, framing or stance differ completely.
A post in Chinese and a post in English about the same event DO share a story.

Two posts do NOT share a story merely because they share a broad topic. "Two posts about
AI" is not a story. "Two posts about OpenAI shipping a specific model on a specific day"
is a story. Be strict: over-merging destroys the benchmark more than over-splitting does.

Posts that are personal opinions, jokes, memes, promos or chit-chat with no shared
specific event should each get their own unique story id.

Stories already identified (reuse these ids where they apply):
{known}

Now assign every post below. Reply with ONLY a JSON object mapping post number to an
object {{"story": "<id>", "label": "<short description, <=8 words>"}}.
Use an existing story id when it fits, otherwise invent a new short snake_case id.
No prose, no markdown fence, JSON only.

POSTS:
{posts}
"""


class QuotaExhausted(RuntimeError):
    """The CLI refused because the account is out of quota -- retrying is pointless."""


def _run(cmd, timeout: int) -> str:
    """Run a labeller CLI, surfacing stderr.

    Discarding stderr here cost a full 60-batch run: agy returned exit 1 with
    'Individual quota reached ... Resets in 1h52m' on stderr and an EMPTY stdout, which
    the caller logged as 'unparseable reply (0 chars)' 48 times in a row. A silent
    quota wall is indistinguishable from a broken parser unless stderr is read.
    """
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    err = (r.stderr or "").strip()
    if r.returncode != 0 or not r.stdout.strip():
        if re.search(r"quota|rate.?limit|upgrade your subscription", err, re.I):
            raise QuotaExhausted(err.splitlines()[0] if err else "quota exhausted")
        if err:
            print(f"    ! {cmd[0]} exit={r.returncode}: {err.splitlines()[0][:140]}",
                  file=sys.stderr)
    return r.stdout


def run_codex(prompt: str, timeout: int) -> str:
    return _run(["codex", "exec", "--skip-git-repo-check", prompt], timeout)


def run_agy(prompt: str, timeout: int) -> str:
    # flag order matters: -p swallows the next token, so the prompt must come last
    return _run(["agy", "--model", "gemini-3.8-flash-high",
                 "--print-timeout", "14m", "-p", prompt], timeout)


RUNNERS = {"codex": run_codex, "agy": run_agy}


def extract_json(s: str):
    """LLM CLIs wrap output in banners/reasoning; find the largest JSON object."""
    s = re.sub(r"```(?:json)?", "", s)
    best = None
    for m in re.finditer(r"\{", s):
        depth = 0
        for i in range(m.start(), len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    frag = s[m.start():i + 1]
                    try:
                        o = json.loads(frag)
                        if isinstance(o, dict) and (best is None or len(frag) > best[0]):
                            best = (len(frag), o)
                    except Exception:
                        pass
                    break
    return best[1] if best else None


class StoryIndex:
    """Which already-named stories to show the labeller for a given batch.

    Fixes the largest known defect in this gold. The prompt can only carry so many known
    stories, and it used to carry the 180 BIGGEST -- so once the corpus had thousands of
    stories, the one a post actually belonged to was almost never on the list and the
    labeller invented a fresh id for it. That is why one viral wine-bottle puzzle ended up
    under three ids and `cod_figure_nscale_gpu_deployment` / `cod_figure_nscale_100k_gpus`
    are the same announcement split in two. Showing stories RELEVANT to the batch instead
    of merely popular ones puts the right id in front of the labeller.

    Retrieval is deliberately LEXICAL (char n-gram TF-IDF), never an embedding model. The
    embedders are what this gold exists to benchmark; using one to decide which stories the
    labeller may reuse would shape the gold around that model's own notion of similarity
    and quietly inflate its score. Character n-grams also survive the typo-and-emoji
    variation that is most of the near-duplicate text here, and work across scripts.
    """

    def __init__(self):
        self.vec = self.mat = None
        self.ids = []
        self.fitted_on = -1

    def fit(self, desc):
        from sklearn.feature_extraction.text import TfidfVectorizer
        if len(desc) == self.fitted_on or not desc:
            return
        self.ids = list(desc)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                   max_features=200_000)
        self.mat = self.vec.fit_transform([desc[i] for i in self.ids])
        self.fitted_on = len(desc)

    def top_for(self, texts, k):
        """Nearest stories over the batch, ROUND-ROBIN by rank.

        Every post contributes its best candidate before any post contributes its second.
        Concatenating each post's top-k in turn instead looks equivalent but is not: the
        caller truncates to a fixed budget, so the first few posts in a batch consume all
        of it and the rest are shown nothing. That made recall fall as k rose -- more
        candidates, fewer posts served.
        """
        if self.mat is None or not texts:
            return []
        import numpy as np
        sims = (self.vec.transform(texts) @ self.mat.T).toarray()
        best = np.argsort(-sims, axis=1)[:, :k]
        seen, out = set(), []
        for rank in range(best.shape[1]):
            for row in range(best.shape[0]):
                j = best[row, rank]
                if sims[row, j] <= 0.0 or self.ids[j] in seen:
                    continue
                seen.add(self.ids[j]); out.append(self.ids[j])
        return out


def describe(out, by_id, desc):
    """story id -> short human description, so the known-stories list is READABLE.

    labels_<name>.json stores only statusId -> story id; the model's own one-line label is
    thrown away. On --resume that left the prompt seeded with lines like
    "- s_2095762088182378571: s_2095762088182378571", which tells the labeller nothing
    about what the story is and guarantees it cannot reuse the id. Any story still without
    a description falls back to its first member's text.
    """
    members = {}
    for status_id, sid in out.items():
        members.setdefault(sid, []).append(status_id)
    for sid, mem in members.items():
        if desc.get(sid) and desc[sid] != sid:
            continue
        for m in mem:
            post = by_id.get(m)
            if post and post.get("text"):
                desc[sid] = re.sub(r"\s+", " ", post["text"]).strip()[:90]
                break
        desc.setdefault(sid, sid)
    return desc, members


def label(name: str, posts, batch: int, timeout: int, resume: bool = False):
    """Label posts, optionally skipping ones already done.

    When resuming, the known-stories list is seeded from the EXISTING labels, biggest
    story first -- new posts overwhelmingly belong to stories that already have several
    members, and the prompt can only carry so many. Seeding with singletons instead
    would make the labeller re-invent ids for clusters it has already named, silently
    splitting stories across a corpus expansion.
    """
    runner = RUNNERS[name]
    by_id = {p["statusId"]: p for p in ALL_POSTS}
    desc_file = HERE / f"labels_{name}_desc.json"
    desc = json.loads(desc_file.read_text()) if desc_file.exists() else {}
    index, sizes = StoryIndex(), {}
    out = {}
    if resume:
        f = HERE / f"labels_{name}.json"
        if f.exists():
            out = json.loads(f.read_text())
            # ONLY this labeller's own descriptions, never another's. Seeding agy with
            # codex's story names looked like a free win -- shared ids, less duplication --
            # but it anchors one labeller on the other's judgement. The gold keeps only
            # pairs BOTH labellers independently agreed on, so anchoring would inflate that
            # agreement into confirmation it never earned and quietly degrade the gold.
            # The fallback below derives descriptions from post text, which is the data
            # itself and carries no other labeller's opinion.
            desc, _ = describe(out, by_id, desc)
            for sid in out.values():
                sizes[sid] = sizes.get(sid, 0) + 1
            done = {p["statusId"] for p in posts} & set(out)
            posts = [p for p in posts if p["statusId"] not in out]
            print(f"  resuming: {len(done)} already labelled, {len(posts)} to go, "
                  f"{len(desc)} known stories seeded "
                  f"({sum(1 for v in desc.values() if v and not v.startswith(('s_', 'cod_')))} "
                  f"with readable descriptions)")
    for i in range(0, len(posts), batch):
        chunk = posts[i:i + batch]
        listing = "\n".join(
            f"{k}. [{p['authorId']}] {re.sub(chr(10), ' / ', p['text'])[:260]}"
            for k, p in enumerate(chunk))
        # RELEVANT stories first, then the biggest as filler. Ordering matters: the
        # retrieved ones are the ids this batch is actually likely to reuse.
        index.fit(desc)
        picked = index.top_for([p["text"] for p in chunk], RETRIEVE_PER_POST)[:RETRIEVE_MAX]
        seen = set(picked)
        for sid, _ in sorted(sizes.items(), key=lambda kv: -kv[1]):
            if len(picked) >= RETRIEVE_MAX + BIGGEST_FILL:
                break
            if sid not in seen:
                seen.add(sid); picked.append(sid)
        kn = "\n".join(f"- {sid}: {desc.get(sid, sid)}" for sid in picked) or "(none yet)"
        prompt = PROMPT.format(known=kn, posts=listing)
        print(f"  [{name}] batch {i//batch+1}/{(len(posts)+batch-1)//batch} "
              f"({len(chunk)} posts, {len(desc)} known stories, "
              f"{len(picked)} shown)", flush=True)
        try:
            raw = runner(prompt, timeout)
        except subprocess.TimeoutExpired:
            print(f"    ! timeout, skipping batch", file=sys.stderr)
            continue
        except QuotaExhausted as e:
            print(f"\n  !! {name} quota exhausted: {e}\n"
                  f"  !! stopping here; {len(out)} labels checkpointed. "
                  f"Re-run with --resume once it resets.", file=sys.stderr)
            break
        except Exception as e:
            print(f"    ! runner failed ({type(e).__name__}: {e}), skipping batch",
                  file=sys.stderr)
            continue
        obj = extract_json(raw)
        if not obj:
            print(f"    ! unparseable reply ({len(raw)} chars), skipping", file=sys.stderr)
            continue
        hit = 0
        for k, v in obj.items():
            try:
                idx = int(re.sub(r"\D", "", str(k)))
            except ValueError:
                continue
            if idx >= len(chunk):
                continue
            sid = v.get("story") if isinstance(v, dict) else str(v)
            if not sid:
                continue
            out[chunk[idx]["statusId"]] = sid
            lbl = (v.get("label") if isinstance(v, dict) else "") or ""
            if lbl and (not desc.get(sid) or desc[sid] == sid):
                desc[sid] = lbl
            desc.setdefault(sid, re.sub(r"\s+", " ", chunk[idx]["text"]).strip()[:90])
            sizes[sid] = sizes.get(sid, 0) + 1
            hit += 1
        print(f"    -> {hit}/{len(chunk)} assigned", flush=True)
        # Checkpoint every batch. agy's CLI has exited mid-run twice with no traceback,
        # no non-zero status and no OOM -- intermittent and not diagnosable from here.
        # Rather than chase it, make it survivable: --resume then costs one batch, not
        # the whole run.
        (HERE / f"labels_{name}.json").write_text(json.dumps(out, indent=1))
        # Persist descriptions too. Without this the story labels are thrown away and a
        # resumed run cannot show the labeller what any existing story IS.
        desc_file.write_text(json.dumps(desc, indent=1, ensure_ascii=False))
    return out


def pairs_of(cl, ids):
    P = {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            x, y = ids[a], ids[b]
            if x in cl and y in cl:
                P[(x, y)] = cl[x] == cl[y]
    return P


def reconcile(ids):
    from sklearn.metrics import adjusted_rand_score
    A = json.loads((HERE / "labels_codex.json").read_text())
    B = json.loads((HERE / "labels_agy.json").read_text())
    both = [i for i in ids if i in A and i in B]
    print(f"both labellers covered {len(both)}/{len(ids)} posts")

    ari = adjusted_rand_score([A[i] for i in both], [B[i] for i in both])
    pa, pb = pairs_of(A, both), pairs_of(B, both)
    keys = sorted(set(pa) & set(pb))
    agree = [k for k in keys if pa[k] == pb[k]]
    both_same = [k for k in keys if pa[k] and pb[k]]
    disagree = [k for k in keys if pa[k] != pb[k]]

    print(f"ARI(codex, agy)      = {ari:.4f}")
    print(f"pairs compared       = {len(keys)}")
    print(f"pairs agreed         = {len(agree)} ({len(agree)/max(len(keys),1)*100:.2f}%)")
    print(f"pairs both-say-SAME  = {len(both_same)}")
    print(f"pairs disagreed      = {len(disagree)}  <- excluded from gold")

    # gold clusters = connected components of the "both said same" graph
    parent = {i: i for i in both}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for x, y in both_same:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    gold = {i: f"s_{find(i)[-8:]}" for i in both}
    sizes = {}
    for v in gold.values():
        sizes[v] = sizes.get(v, 0) + 1
    multi = {k: v for k, v in sizes.items() if v > 1}

    (HERE / "gold.json").write_text(json.dumps(gold, indent=1))
    (HERE / "gold_pairs.json").write_text(json.dumps(
        [[x, y, bool(pa[(x, y)])] for (x, y) in agree], indent=1))
    (HERE / "agreement.json").write_text(json.dumps({
        "ari": round(ari, 4), "posts": len(both), "pairs": len(keys),
        "agreed": len(agree), "disagreed": len(disagree),
        "both_same": len(both_same), "gold_stories": len(sizes),
        "multi_post_stories": len(multi),
        "largest_story": max(sizes.values()) if sizes else 0,
        "adjudication_sample": [list(k) for k in disagree[:60]],
    }, indent=1))
    print(f"\ngold: {len(sizes)} stories, {len(multi)} with >1 post, "
          f"largest = {max(sizes.values()) if sizes else 0} posts")
    if ari < 0.5:
        print("\n!! ARI is low - the two labellers substantially disagree. Treat every\n"
              "   downstream number as unreliable until this is adjudicated by hand.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeller", choices=["codex", "agy"])
    ap.add_argument("--reconcile", action="store_true")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip posts already present in labels_<labeller>.json")
    ap.add_argument("--exclude", default="",
                    help="comma-separated statusId->label maps whose posts to SKIP "
                         "(e.g. gold.json: those posts already have two labellers, so "
                         "a second pass over them buys nothing)")
    args = ap.parse_args()

    # Tolerate a torn read: capture.py truncates and rewrites posts.jsonl wholesale, so a
    # labeller starting mid-write sees a half-written final line. Skipping bad lines beats
    # crashing 77 batches into nothing.
    posts, skipped = [], 0
    for l in POSTS.read_text().split("\n"):
        if not l.strip():
            continue
        try:
            posts.append(json.loads(l))
        except json.JSONDecodeError:
            skipped += 1
    if skipped:
        print(f"  warning: skipped {skipped} unparseable line(s) in posts.jsonl",
              file=sys.stderr)
    # Keep the FULL set for description lookup before --limit trims what gets labelled:
    # a story's readable description may come from a post outside this run's slice.
    global ALL_POSTS
    ALL_POSTS = posts
    if args.exclude:
        skip = set()
        for f in args.exclude.split(","):
            q = HERE / f.strip()
            if q.exists():
                skip |= set(json.loads(q.read_text()))
        before = len(posts)
        posts = [p for p in posts if p["statusId"] not in skip]
        print(f"  --exclude: {before - len(posts)} posts already covered, "
              f"{len(posts)} remain")
    if args.limit:
        posts = posts[:args.limit]
    ids = [p["statusId"] for p in posts]

    if args.reconcile:
        return reconcile(ids)
    if not args.labeller:
        sys.exit("need --labeller or --reconcile")
    res = label(args.labeller, posts, args.batch, args.timeout, resume=args.resume)
    p = HERE / f"labels_{args.labeller}.json"
    p.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {len(res)} labels -> {p}")


if __name__ == "__main__":
    main()
