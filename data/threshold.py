#!/usr/bin/env python3
"""
Pick the operating threshold on the axis the product actually cares about.

Two problems with reading tau off the benchmark's best-F1 column:

 1. The sweep ran 0.50-0.95, and bekko-a8m's best F1 landed exactly on 0.50 -- the
    boundary. A maximum at the edge of the search range is a constrained optimum, not
    a real one, and 0.50 was already shipping in config.js on that basis. This sweeps
    0.10-0.99 so the maximum is interior and therefore meaningful.

 2. F1 weighs a false collapse and a missed duplicate equally, and this product does
    not. A missed duplicate costs the user one redundant post -- the status quo they
    already live with. A false collapse HIDES A POST THEY WANTED, and they cannot see
    what they are missing to know it happened. So the shipped threshold is chosen for
    high precision, and recall is whatever it is at that point.

Reuses the cached embeddings written by benchmark.py, so this costs seconds.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import BY_KEY, GOLD, HERE, POSTS, embed_cached, pair_labels, pair_sims

TARGETS = (0.80, 0.90, 0.95)


def main():
    posts = [json.loads(l) for l in POSTS.read_text().split("\n") if l.strip()]
    cluster_of = json.loads(GOLD.read_text())
    posts = [p for p in posts if p["statusId"] in cluster_of]
    ids = [p["statusId"] for p in posts]
    texts = [p["text"] for p in posts]
    I, J, Y = pair_labels(cluster_of, ids, HERE / "gold_pairs.json")

    res = json.loads((HERE / "benchmark_results.json").read_text())["models"]
    out = {}
    for name, r in res.items():
        key, dt = name.split(":")
        V, used, fname, _ = embed_cached(BY_KEY[key], texts, dt, 192)
        sim = pair_sims(V, I, J)
        taus = np.arange(0.10, 0.995, 0.005)
        rows = []
        for t in taus:
            pred = sim >= t
            tp = int((pred & (Y == 1)).sum()); fp = int((pred & (Y == 0)).sum())
            fn = int((~pred & (Y == 1)).sum())
            p = tp / (tp + fp) if tp + fp else 1.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            rows.append({"tau": round(float(t), 3), "precision": round(p, 4),
                         "recall": round(rc, 4),
                         "f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0})
        best = max(rows, key=lambda r: r["f1"])
        edge = best["tau"] <= 0.105 or best["tau"] >= 0.99
        ops = {}
        for want in TARGETS:
            ok = [r for r in rows if r["precision"] >= want and r["recall"] > 0]
            ops[want] = min(ok, key=lambda r: r["tau"]) if ok else None
        out[name] = {"best_f1": best, "best_f1_at_edge": edge, "operating": ops}
        print(f"\n{name}")
        print(f"  best F1 {best['f1']:.4f} @ tau={best['tau']:.3f} "
              f"(P={best['precision']:.3f} R={best['recall']:.3f})"
              f"{'  <-- STILL AT EDGE' if edge else ''}")
        for want, r in ops.items():
            print(f"  P>={want:.2f}: " + (f"tau={r['tau']:.3f}  R={r['recall']:.4f}  "
                  f"F1={r['f1']:.4f}" if r else "unreachable at any tau"))
    (HERE / "threshold_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE/'threshold_results.json'}")


if __name__ == "__main__":
    main()
