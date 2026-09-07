#!/usr/bin/env python3
"""
Paired bootstrap over STORIES for the macro-AUC differences between embedders.

Why this exists: on the old 53-story gold, gte-ml-base beat bekko-a8m by +0.0052 macro
AUC with a 95% CI of [-0.0041, +0.0210] -- i.e. the headline ranking was not
distinguishable from noise, and picking the top row of the table would have been picking
a coin flip. 53 stories was simply too small. The gold now has ~110 macro-scoreable
stories, which is the sample size that question always needed.

The resampling unit is the STORY, not the pair. Pairs within one story are anything but
independent -- 143 posts about one NBA story generate ~10k positive pairs that all rise
and fall together -- so bootstrapping pairs would report a CI perhaps an order of
magnitude too narrow and manufacture significance out of one big cluster. Stories are
the thing that varies between "the news cycle we sampled" and "some other news cycle",
so they are the unit that has to be resampled.

The negative pool is held FIXED across resamples. There are 4.29M negative pairs against
110 stories; their sampling variance is negligible next to the story-to-story variance,
and holding them fixed is what makes each resample O(stories) instead of O(pairs).

Usage:
    python data/bootstrap.py                        # top models by macro AUC
    python data/bootstrap.py --models bekko-a8m,gte-ml-base --iters 20000
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import (BY_KEY, GOLD, HERE, POSTS, embed_cached, pair_labels,  # noqa: E402
                       pair_sims)

RESULTS = HERE / "benchmark_results.json"


def per_story_auc(sim, I, Y, cluster_of, ids, min_pairs=3):
    """AUC of each story's positive pairs against the shared negative pool.

    Computed by rank rather than by calling roc_auc_score per story: each story would
    otherwise be scored against all 4.29M negatives separately, 110 times over. Sorting
    the negatives once and binary-searching turns that into a searchsorted per story.
    Ties contribute 0.5, matching roc_auc_score exactly (asserted in main()).
    """
    neg = np.sort(sim[Y == 0])
    n = len(neg)
    by = {}
    for k in np.where(Y == 1)[0]:
        by.setdefault(cluster_of.get(ids[I[k]]), []).append(sim[k])

    out = {}
    for c, pos in by.items():
        if len(pos) < min_pairs:
            continue
        pos = np.asarray(pos)
        lo = np.searchsorted(neg, pos, side="left")     # negatives strictly below
        hi = np.searchsorted(neg, pos, side="right")    # negatives at or below
        out[c] = float(np.mean((lo + (hi - lo) * 0.5) / n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated keys; default = top 4 by macro AUC in results")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    posts = [json.loads(l) for l in POSTS.read_text().split("\n") if l.strip()]
    cluster_of = json.loads(GOLD.read_text())
    posts = [p for p in posts if p["statusId"] in cluster_of]
    ids = [p["statusId"] for p in posts]
    texts = [p["text"] for p in posts]

    if args.models:
        keys = [(k, None) for k in args.models.split(",")]
    else:
        res = json.loads(RESULTS.read_text())["models"]
        top = sorted(res.items(), key=lambda kv: -kv[1].get("macro_auc", 0))[:4]
        keys = [(name.split(":")[0], r["dtype"]) for name, r in top]
    print("comparing:", ", ".join(k for k, _ in keys))

    I, J, Y = pair_labels(cluster_of, ids, HERE / "gold_pairs.json")

    story_auc = {}
    for k, dt in keys:
        m = BY_KEY.get(k)
        if not m:
            print(f"! unknown model {k}"); continue
        V, used, fname, secs = embed_cached(m, texts, dt or m.dtypes[0], args.max_len)
        sim = pair_sims(V, I, J)
        story_auc[k] = per_story_auc(sim, I, Y, cluster_of, ids)
        print(f"  {k:22} {used:5} {len(story_auc[k])} scoreable stories  {fname}")

    # Sanity-check the rank shortcut against sklearn on the largest story once. A silent
    # disagreement here would make every CI below wrong in a way nothing else catches.
    from sklearn.metrics import roc_auc_score
    k0 = keys[0][0]
    V, used, _, _ = embed_cached(BY_KEY[k0], texts, keys[0][1] or BY_KEY[k0].dtypes[0],
                                 args.max_len)
    sim0 = pair_sims(V, I, J)
    biggest = max(story_auc[k0], key=lambda c: sum(
        1 for k in np.where(Y == 1)[0] if cluster_of.get(ids[I[k]]) == c))
    pos = np.array([sim0[k] for k in np.where(Y == 1)[0]
                    if cluster_of.get(ids[I[k]]) == biggest])
    neg = sim0[Y == 0]
    ref = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
    assert abs(ref - story_auc[k0][biggest]) < 1e-9, (ref, story_auc[k0][biggest])
    print(f"  rank shortcut verified against roc_auc_score ({ref:.6f})\n")

    common = sorted(set.intersection(*(set(v) for v in story_auc.values())))
    print(f"{len(common)} stories scoreable by every model; {args.iters} resamples\n")
    A = {k: np.array([story_auc[k][c] for c in common]) for k in story_auc}

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(common), size=(args.iters, len(common)))

    order = sorted(A, key=lambda k: -A[k].mean())
    print("| model | macro AUC | vs next | 95% CI | P(better) |")
    print("|---|---|---|---|---|")
    out = {"n_stories": len(common), "iters": args.iters, "macro_auc": {}, "pairs": {}}
    for k in order:
        out["macro_auc"][k] = round(float(A[k].mean()), 4)

    for a, b in zip(order, order[1:]):
        d = A[a][idx].mean(1) - A[b][idx].mean(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "significant" if lo > 0 else "NOT significant"
        print(f"| {a} | {A[a].mean():.4f} | {a[:12]} - {b[:12]} = {d.mean():+.4f} | "
              f"[{lo:+.4f}, {hi:+.4f}] | {(d > 0).mean():.3f} {sig} |")
        out["pairs"][f"{a}-{b}"] = {"delta": round(float(d.mean()), 4),
                                    "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                    "p_better": round(float((d > 0).mean()), 3),
                                    "significant": bool(lo > 0)}
    print(f"| {order[-1]} | {A[order[-1]].mean():.4f} | | | |")

    # Best vs every other model, not just adjacent rows: a chain of individually
    # non-significant steps can still add up to a significant gap end to end.
    best = order[0]
    print(f"\n{best} vs each:")
    for k in order[1:]:
        d = A[best][idx].mean(1) - A[k][idx].mean(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  vs {k:22} {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"{'significant' if lo > 0 else 'NOT significant'}")
        out["pairs"][f"{best}-vs-{k}"] = {"delta": round(float(d.mean()), 4),
                                          "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                          "significant": bool(lo > 0)}

    (HERE / "bootstrap_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'bootstrap_results.json'}")


if __name__ == "__main__":
    main()
