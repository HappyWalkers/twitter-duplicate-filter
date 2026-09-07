#!/usr/bin/env python3
"""
Simulate what the user actually sees, and choose tau from that.

Pairwise precision/recall is the wrong axis to ship on, for two reasons:

  * The extension never evaluates all C(n,2) pairs. It compares each new post against
    CLUSTER REPRESENTATIVES ONLY, first match wins, inside a rolling window of
    TUNING.windowSize posts, skipping same-author. Most pairs the benchmark scores are
    never actually tested.
  * A cluster of k posts needs only k-1 correct links to fold completely, but pairwise
    recall counts all C(k,2). So pairwise recall systematically understates how much
    duplication the user stops seeing.

What the user experiences is exactly two numbers:
    collapse rate      -- share of posts hidden
    collapse precision -- share of hidden posts that really were the same story as the
                          representative they were folded into
The second one is the risk: a wrong collapse hides a post the user wanted, and they
cannot see what they are missing to know it happened. This reports both against tau,
running the shipped algorithm verbatim -- window, author exemption, first-match and all.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import BY_KEY, GOLD, HERE, POSTS, embed_cached

WINDOW = 400          # TUNING.windowSize
EXEMPT_SAME_AUTHOR = True


def agreed_lookup(ids, pairs_file):
    """How a given collapse was judged: same story, different story, or never adjudicated.

    Prefers gold_pairs_compact.json. That file stores the overlaps, the agreed positives
    and the disputed pairs, and everything else inside an overlap is an agreed negative --
    which means the answer needs three small sets, not the 29M-row expansion. The old
    enumerated file is still read if the compact spec is absent.

    Disputed pairs are UNJUDGED here, not negative. The labellers looked and disagreed, so
    counting a collapse of one against the model would penalise it for the cases nobody
    could settle -- and those are exactly the near-duplicate pairs it is most likely to be
    right about.
    """
    idx = {x: k for k, x in enumerate(ids)}
    n = len(ids)

    def key(a, b):
        return (a * n + b) if a < b else (b * n + a)

    compact = Path(pairs_file).parent / "gold_pairs_compact.json"
    if compact.exists():
        spec = json.loads(compact.read_text())
        def keyset(rows):
            out = set()
            for x, y in rows:
                a, b = idx.get(x), idx.get(y)
                if a is not None and b is not None:
                    out.add(key(a, b))
            return out
        pos = keyset(spec["positives"])
        dis = keyset(spec.get("disputed") or [])
        overlaps = [frozenset(idx[x] for x in m if x in idx)
                    for m in spec["overlaps"].values()]
        print(f"using {compact.name}: {len(pos)} positive, {len(dis)} disputed, "
              f"{len(overlaps)} overlap(s)")
        return ("compact", pos, dis, overlaps, n)

    rows = json.loads(Path(pairs_file).read_text())
    keys, same = [], []
    for x, y, s_ in rows:
        a, b = idx.get(x), idx.get(y)
        if a is None or b is None:
            continue
        keys.append(key(a, b)); same.append(1 if s_ else 0)
    order = np.argsort(keys)
    return ("enumerated", np.array(keys, dtype=np.int64)[order],
            np.array(same, dtype=np.int8)[order], n)


def judge(agreed, i, j):
    """1 same-story, 0 different-story, None never adjudicated."""
    if agreed[0] == "compact":
        _, pos, dis, overlaps, n = agreed
        k = (i * n + j) if i < j else (j * n + i)
        if k in pos:
            return 1
        if k in dis:
            return None
        # Only a pair both of whose members sat inside ONE labeller's overlap was
        # actually checked; anything spanning two overlaps was judged by nobody.
        for ov in overlaps:
            if i in ov and j in ov:
                return 0
        return None

    _, keys, same, n = agreed
    k = (i * n + j) if i < j else (j * n + i)
    p = int(np.searchsorted(keys, k))
    if p < len(keys) and keys[p] == k:
        return int(same[p])
    return None


def simulate(V, authors, gold, tau, window=WINDOW, rule="first", agreed=None):
    """Greedy assignment against representatives, in a rolling window.

    rule="first" mirrors dedup/cluster.js add() as shipped: reps in insertion order,
    the first one over tau wins. rule="best" takes the NEAREST rep over tau instead.
    Same candidate set and same threshold -- the only difference is which of several
    qualifying representatives the post joins."""
    rep_idx = []                       # indices of current representatives, in order
    order = []                         # insertion order for eviction
    member_of = {}                     # rep index -> [member indices]
    hidden, correct, wrong, unjudged = 0, 0, 0, 0
    for i in range(len(V)):
        hit = -1
        if rep_idx:
            R = V[rep_idx]
            sims = R @ V[i]
            ok = sims >= tau
            if EXEMPT_SAME_AUTHOR:
                ok &= np.array([authors[r] != authors[i] for r in rep_idx])
            if ok.any():
                if rule == "best":
                    hit = rep_idx[int(np.argmax(np.where(ok, sims, -np.inf)))]
                else:
                    hit = rep_idx[int(np.argmax(ok))]  # FIRST match, not the best one
        if hit < 0:
            rep_idx.append(i); member_of[i] = [i]
        else:
            member_of[hit].append(i)
            hidden += 1
            if agreed is not None:
                v = judge(agreed, i, hit)
                if v is None:
                    unjudged += 1
                elif v:
                    correct += 1
                else:
                    wrong += 1
            elif gold[i] is not None and gold[i] == gold[hit]:
                correct += 1
        order.append(i)
        if len(order) > window:                        # evict, as the extension does
            old = order.pop(0)
            if old in member_of:
                rep_idx.remove(old)
    judged = correct + wrong
    return {"hidden": hidden, "correct": correct, "wrong": wrong,
            "unjudged": unjudged, "rate": hidden / len(V),
            # Precision over ADJUDICATED collapses only. The unjudged share is reported
            # alongside so the number is never read as if it covered everything.
            "precision": (correct / judged) if judged else float("nan"),
            "unjudged_share": (unjudged / hidden) if hidden else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--taus", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    ap.add_argument("--rules", default="first", help="comma-separated: first,best")
    ap.add_argument("--window", type=int, default=WINDOW)
    args = ap.parse_args()

    posts = [json.loads(l) for l in POSTS.read_text().split("\n") if l.strip()]
    cluster_of = json.loads(GOLD.read_text())
    posts = [p for p in posts if p["statusId"] in cluster_of]
    texts = [p["text"] for p in posts]
    authors = [p.get("authorId") or p.get("author") or "" for p in posts]
    gold = [cluster_of.get(p["statusId"]) for p in posts]
    ids = [p["statusId"] for p in posts]

    agreed = agreed_lookup(ids, HERE / "gold_pairs.json")
    res = json.loads((HERE / "benchmark_results.json").read_text())["models"]
    want = args.models.split(",") if args.models else None
    names = [n for n in res if want is None or n.split(":")[0] in want]
    if not names:
        sys.exit(f"no benchmarked model matches {args.models}; have: {', '.join(res)}")

    out = {}
    for name in names:
        key, dt = name.split(":")
        V, *_ = embed_cached(BY_KEY[key], texts, dt, 192)
        rows = []
        for rule in args.rules.split(","):
            print(f"\n{name}  rule={rule}  (window={args.window}, "
                  f"same-author exempt={EXEMPT_SAME_AUTHOR})")
            print("   tau | hidden |  rate  | correct | wrong | unadjudicated | "
                  "precision (of judged)")
            for t in [float(x) for x in args.taus.split(",")]:
                r = simulate(V, authors, gold, t, window=args.window, rule=rule,
                             agreed=agreed)
                rows.append({"tau": t, "rule": rule, "window": args.window, **r})
                print(f"  {t:.2f} | {r['hidden']:>6} | {r['rate']*100:5.2f}% | "
                      f"{r['correct']:>7} | {r['wrong']:>5} | "
                      f"{r['unjudged']:>4} ({r['unjudged_share']*100:4.1f}%) | "
                      f"{r['precision']*100:5.1f}%")
        out[name] = rows
    (HERE / "collapse_sim.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE/'collapse_sim.json'}")


if __name__ == "__main__":
    main()
