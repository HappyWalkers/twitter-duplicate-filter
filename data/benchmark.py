#!/usr/bin/env python3
"""
Offline accuracy benchmark for the X-timeline dedup embedders.

Runs the SAME quantised ONNX weights the extension would ship, via onnxruntime --
not fp32 PyTorch. That parity is deliberate: scoring fp32 and shipping q8 would
make every number here a polite fiction.

Per-model pooling and prompt settings below are read from each repo's real
1_Pooling/config.json and config_sentence_transformers.json, not assumed. Three
different pooling modes appear across these eight models; defaulting everything to
mean pooling would silently turn lasttoken models into noise and produce a
confident, wrong conclusion that small models win.

Usage:
    python data/benchmark.py --models all
    python data/benchmark.py --models bekko,granite --dtype fp32
"""
import argparse, json, sys, time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
POSTS = HERE / "posts.jsonl"
GOLD = HERE / "gold.json"
OUT = HERE / "benchmark_results.json"


@dataclass
class M:
    key: str
    onnx_repo: str          # repo holding the .onnx files
    tok_repo: str           # repo holding the tokenizer (often the original, not the ONNX mirror)
    pooling: str            # 'mean' | 'cls' | 'lasttoken'   <- from 1_Pooling/config.json
    prompt: str = ""        # symmetric prefix applied to BOTH sides  <- from config_sentence_transformers.json
    dtypes: list = field(default_factory=lambda: ["q8", "fp32"])  # tried in order
    note: str = ""


# Verified against each repo's own config files (see README table).
MODELS = [
    M("granite-97m", "ibm-granite/granite-embedding-97m-multilingual-r2",
      "ibm-granite/granite-embedding-97m-multilingual-r2", "cls",
      dtypes=["fp32"], note="non-standard q8 filename (model_quint8_avx2) -> fp32 only"),
    M("bekko-a8m", "hotchpotch/bekko-embedding-v1-a8m",
      "hotchpotch/bekko-embedding-v1-a8m", "mean",
      dtypes=["fp32"], note="non-standard q8 filenames -> fp32 only (130MB anyway)"),
    M("mmini-l12", "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
      # NOT the Xenova mirror: its tokenizer_config declares BertTokenizer, so
      # "Fed cuts rates" tokenizes to "fed <unk> <unk>" and every embedding is noise.
      # The original ST repo has the correct XLM-R SentencePiece tokenizer.
      "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "mean",
      note="CONTROL - the obvious/most-downloaded pick"),
    M("f2llm-160m", "onnx-community/F2LLM-v2-160M-ONNX",
      "codefuse-ai/F2LLM-v2-160M", "lasttoken", prompt=""),
    M("jina-v5-nano-clu", "jinaai/jina-embeddings-v5-text-nano-clustering",
      "jinaai/jina-embeddings-v5-text-nano-clustering", "lasttoken", prompt="Document: ",
      note="cc-by-nc-4.0 - UPPER BOUND REFERENCE ONLY, cannot ship"),
    M("embeddinggemma-300m", "onnx-community/embeddinggemma-300m-ONNX",
      "onnx-community/embeddinggemma-300m-ONNX", "mean",
      prompt="task: sentence similarity | query: "),
    M("gte-ml-base", "onnx-community/gte-multilingual-base",
      "Alibaba-NLP/gte-multilingual-base", "cls"),
    M("harrier-270m", "onnx-community/harrier-oss-v1-270m-ONNX",
      "microsoft/harrier-oss-v1-270m", "lasttoken",
      prompt="Instruct: Retrieve semantically similar text\nQuery: ",
      note="ships an sts_query prompt that is exactly this task"),
]
BY_KEY = {m.key: m for m in MODELS}

DTYPE_FILES = {
    "q8":   ["onnx/model_quantized.onnx", "onnx/model_int8.onnx", "onnx/model_uint8.onnx"],
    "fp16": ["onnx/model_fp16.onnx"],
    "fp32": ["onnx/model.onnx"],
}


def load_session(m: M, dtype: str):
    """Download + open an ORT session. Returns (session, dtype_used, filename)."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    last = None
    for cand in DTYPE_FILES[dtype]:
        try:
            p = hf_hub_download(m.onnx_repo, cand)
        except Exception as e:
            last = e
            continue
        # models >2GB split weights into a sibling .onnx_data that ORT loads by
        # relative path -- it must sit next to the graph file, so fetch it too.
        for extra in (cand + "_data", cand.replace(".onnx", ".onnx_data")):
            try:
                hf_hub_download(m.onnx_repo, extra)
            except Exception:
                pass
        so = ort.SessionOptions()
        so.log_severity_level = 3
        return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]), dtype, cand
    raise RuntimeError(f"no {dtype} file for {m.onnx_repo}: {last}")


def pool(hidden: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "cls":
        return hidden[:, 0]
    if mode == "lasttoken":
        # index of the final non-pad token per row
        idx = mask.sum(axis=1).astype(np.int64) - 1
        return hidden[np.arange(hidden.shape[0]), idx]
    m = mask[..., None].astype(hidden.dtype)
    return (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)


def embed(m: M, texts, dtype: str, batch: int = 16, max_len: int = 128):
    from transformers import AutoTokenizer
    sess, used, fname = load_session(m, dtype)
    tok = AutoTokenizer.from_pretrained(m.tok_repo, trust_remote_code=False)
    want = {i.name for i in sess.get_inputs()}

    # Guard the failure mode that already bit once: a mismatched tokenizer maps most
    # of the input to <unk> and yields plausible-looking but meaningless vectors. A
    # silently-wrong control model would have skewed the entire comparison.
    probe = tok.decode(tok("Fed cuts interest rates by 50 basis points")["input_ids"])
    n_unk = probe.count("<unk>") + probe.count("[UNK]")
    if n_unk > 1:
        raise RuntimeError(
            f"tokenizer {type(tok).__name__} from {m.tok_repo} produced {n_unk} <unk> "
            f"on plain English -- wrong vocabulary, refusing to report numbers from it")

    texts = [m.prompt + t for t in texts]
    vecs, t0 = [], time.time()
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True,
                  max_length=max_len, return_tensors="np")
        feed = {k: v for k, v in enc.items() if k in want}
        if "token_type_ids" in want and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        if "position_ids" in want and "position_ids" not in feed:
            n = feed["input_ids"].shape[1]
            feed["position_ids"] = np.tile(np.arange(n, dtype=np.int64),
                                           (feed["input_ids"].shape[0], 1))
        out = sess.run(None, feed)[0]
        if out.ndim == 2:                      # model already pools internally
            v = out
        else:
            v = pool(out, enc["attention_mask"], m.pooling)
        vecs.append(v.astype(np.float32))
    V = np.concatenate(vecs, 0)
    V /= np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
    return V, used, fname, time.time() - t0


EMB_CACHE = HERE / "emb"


def embed_cached(m: M, texts, dtype: str, max_len: int):
    """embed(), memoised to data/emb/<key>-<dtype>.npy.

    Embedding all 7k posts on CPU is the expensive part of a sweep, and the paired
    bootstrap needs the exact same vectors afterwards. Caching them means the
    bootstrap is free and a re-run after a crash costs nothing.
    """
    EMB_CACHE.mkdir(exist_ok=True)
    f = EMB_CACHE / f"{m.key}-{dtype}-{len(texts)}-{max_len}.npy"
    if f.exists():
        return np.load(f), dtype, f"(cached {f.name})", 0.0
    V, used, fname, secs = embed(m, texts, dtype, max_len=max_len)
    np.save(EMB_CACHE / f"{m.key}-{used}-{len(texts)}-{max_len}.npy", V)
    return V, used, fname, secs


def pair_sims(V, I, J, chunk=250_000):
    """Cosine for every (I[k], J[k]) pair, in chunks.

    NOT np.einsum('ij,ij->i', V[I], V[J]): fancy-indexing 4.3M pairs materialises two
    (4.3M, dim) copies, which is 26GB at dim=768 against 21GB of RAM -- the run either
    gets OOM-killed or thrashes into swap, and from the outside that is indis-
    tinguishable from the process having been reaped. The old 53-cluster gold had 3.4M
    pairs and squeaked through; this one does not. Chunking caps it at ~1.5GB.
    """
    out = np.empty(len(I), dtype=np.float32)
    for a in range(0, len(I), chunk):
        b = min(a + chunk, len(I))
        out[a:b] = np.einsum("ij,ij->i", V[I[a:b]], V[J[a:b]])
    return out


# ---------------------------------------------------------------- metrics

def pair_labels(cluster_of, ids, pairs_file=None):
    """Pairs -> (i, j, same_story).

    Prefers data/gold_pairs.json when present: those are only the pairs BOTH
    labellers agreed on. Deriving every C(n,2) pair from the merged clustering
    instead would silently relabel the disagreed pairs as 'different' and quietly
    inflate every AUC below, so the agreed set is used whenever it exists.
    """
    idx = {x: k for k, x in enumerate(ids)}

    # Preferred: the compact spec (overlaps + positives), expanded here. Storing the
    # negatives explicitly is what made the old file 1.4GB; regenerating them costs a
    # couple of seconds of numpy and a fraction of the memory.
    compact = Path(pairs_file).parent / "gold_pairs_compact.json" if pairs_file else None
    if compact and compact.exists():
        spec = json.loads(compact.read_text())
        pos = set()
        for x, y in spec["positives"]:
            a, b = idx.get(x), idx.get(y)
            if a is not None and b is not None:
                pos.add((a, b) if a < b else (b, a))
        I, J = [], []
        for name, members in spec["overlaps"].items():
            m = np.array([idx[x] for x in members if x in idx], dtype=np.int64)
            if len(m) < 2:
                continue
            a, b = np.triu_indices(len(m), 1)
            I.append(m[a]); J.append(m[b])
        I = np.concatenate(I); J = np.concatenate(J)
        lo = np.minimum(I, J); hi = np.maximum(I, J)
        # One pair can sit in two overlaps; keep it once.
        key = lo.astype(np.int64) * len(ids) + hi
        _, uniq = np.unique(key, return_index=True)
        I, J = lo[uniq], hi[uniq]

        # Drop the pairs the labellers disagreed on. These are excluded from scoring, not
        # counted as negatives: a disputed pair is one nobody established the truth of.
        dis = spec.get("disputed") or []
        if len(dis):
            dk = []
            for x, y in dis:
                a, b = idx.get(x), idx.get(y)
                if a is not None and b is not None:
                    dk.append(min(a, b) * len(ids) + max(a, b))
            if dk:
                dk = np.sort(np.array(dk, dtype=np.int64))
                k2 = I.astype(np.int64) * len(ids) + J
                at = np.searchsorted(dk, k2)
                at[at >= len(dk)] = 0
                keep = dk[at] != k2
                I, J = I[keep], J[keep]

        Y = np.zeros(len(I), dtype=np.int8)
        if pos:
            pk = np.sort(np.array([a * len(ids) + b for a, b in pos], dtype=np.int64))
            k2 = I.astype(np.int64) * len(ids) + J
            p_at = np.searchsorted(pk, k2)
            p_at[p_at >= len(pk)] = 0
            Y[pk[p_at] == k2] = 1
        print(f"expanded {compact.name}: {len(I)} scored pairs, {int(Y.sum())} positive, "
              f"{len(dis)} disputed excluded")
        return I, J, Y

    if pairs_file and Path(pairs_file).exists():
        I, J, Y = [], [], []
        for x, y, same in json.loads(Path(pairs_file).read_text()):
            if x in idx and y in idx:
                I.append(idx[x]); J.append(idx[y]); Y.append(1 if same else 0)
        if I:
            print(f"using {len(I)} labeller-agreed pairs from {Path(pairs_file).name}")
            return np.array(I), np.array(J), np.array(Y)
    n = len(ids)
    I, J, Y = [], [], []
    for a in range(n):
        for b in range(a + 1, n):
            ca, cb = cluster_of.get(ids[a]), cluster_of.get(ids[b])
            if ca is None or cb is None:
                continue
            I.append(a); J.append(b); Y.append(1 if ca == cb else 0)
    print(f"using all {len(I)} derived pairs (no gold_pairs.json)")
    return np.array(I), np.array(J), np.array(Y)


def macro_auc(sim, I, J, Y, cluster_of, ids):
    """Per-story AUC, averaged with EACH STORY WEIGHTED EQUALLY.

    The pooled AUC is dominated by whichever story happens to be biggest: one NBA
    scandal cluster contributes ~40% of all positive pairs here, so a model that nails
    one large English sports story can win the pooled number while being mediocre
    everywhere else. This scores each story against the shared negative pool separately
    and averages, so a 4-post Japanese story counts as much as a 143-post one.
    """
    from sklearn.metrics import roc_auc_score
    neg = sim[Y == 0]
    by = {}
    for k in np.where(Y == 1)[0]:
        c = cluster_of.get(ids[I[k]])
        by.setdefault(c, []).append(sim[k])
    aucs = []
    for c, pos in by.items():
        if len(pos) < 3:            # too few pairs to estimate anything stable
            continue
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        v = np.concatenate([np.array(pos), neg])
        aucs.append(roc_auc_score(y, v))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def sweep(sim, y, taus):
    rows = []
    for t in taus:
        pred = sim >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"tau": round(float(t), 3), "precision": round(p, 4),
                     "recall": round(r, 4), "f1": round(f, 4),
                     "pair_fold_rate": round(float(pred.mean()), 4)})
    return rows


def greedy_ari(V, ids, cluster_of, tau):
    """Score the algorithm we ACTUALLY ship: first-seen post becomes the
    representative, each later post joins the first rep it exceeds tau against.
    This differs from ideal clustering and generally scores worse -- which is the
    point of measuring it rather than reporting an optimistic number."""
    from sklearn.metrics import adjusted_rand_score
    reps, assign = [], []
    for i in range(len(ids)):
        hit = -1
        for ci, r in enumerate(reps):
            if float(V[i] @ V[r]) >= tau:
                hit = ci
                break
        if hit < 0:
            reps.append(i); hit = len(reps) - 1
        assign.append(hit)
    gold = [cluster_of.get(x, f"__{k}") for k, x in enumerate(ids)]
    keep = [k for k, x in enumerate(ids) if x in cluster_of]
    return adjusted_rand_score([gold[k] for k in keep], [assign[k] for k in keep])


def tier1_baseline(posts, cluster_of, ids):
    """How much duplication do the FREE exact signals already catch?
    Reported before any embedding so the model's real contribution is visible."""
    by = {p["statusId"]: p for p in posts}
    I, J, Y = pair_labels(cluster_of, ids, HERE / "gold_pairs.json")

    # A URL carried by many posts is boilerplate -- an ad campaign, a link-in-bio, a
    # sponsor tag -- not a story marker. Measured on the first corpus, the raw URL rule
    # scored precision 0.035: ~20 promo posts shared one link and generated 191 false
    # pairs by themselves (C(20,2)=190). Media keys, by contrast, scored 0.839 untouched.
    # So URLs get a frequency guard and media keys do not.
    MAX_URL_POSTS = 4
    url_df = {}
    for p in posts:
        for u in set(p.get("urls") or []):
            url_df[u] = url_df.get(u, 0) + 1
    common = {u for u, n in url_df.items() if n > MAX_URL_POSTS}

    pred = np.zeros(len(Y), dtype=bool)
    for k, (a, b) in enumerate(zip(I, J)):
        pa, pb = by[ids[a]], by[ids[b]]
        if pa.get("quotedId") and pa["quotedId"] == pb.get("quotedId"):
            pred[k] = True; continue
        ua = set(pa.get("urls") or []) - common
        ub = set(pb.get("urls") or []) - common
        if ua & ub:
            pred[k] = True; continue
        if set(pa.get("mediaKeys") or []) & set(pb.get("mediaKeys") or []):
            pred[k] = True
    tp = int((pred & (Y == 1)).sum()); fp = int((pred & (Y == 0)).sum())
    fn = int((~pred & (Y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "pairs_caught": tp, "false_pairs": fp}


def main():
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--dtype", default=None, help="force one dtype")
    ap.add_argument("--max-len", type=int, default=192)
    args = ap.parse_args()

    if not POSTS.exists():
        sys.exit(f"missing {POSTS} - run data/capture.py first")
    posts = [json.loads(l) for l in POSTS.read_text().split("\n") if l.strip()]
    if not GOLD.exists():
        sys.exit(f"missing {GOLD} - run data/label.py first")
    cluster_of = json.loads(GOLD.read_text())

    posts = [p for p in posts if p["statusId"] in cluster_of]
    ids = [p["statusId"] for p in posts]
    texts = [p["text"] for p in posts]
    print(f"{len(ids)} labelled posts, {len(set(cluster_of.values()))} gold stories")

    I, J, Y = pair_labels(cluster_of, ids, HERE / "gold_pairs.json")
    print(f"{len(Y)} pairs, {int(Y.sum())} same-story ({Y.mean()*100:.2f}%)\n")

    base = tier1_baseline(posts, cluster_of, ids)
    print(f"TIER-1 exact signals alone: P={base['precision']:.3f} R={base['recall']:.3f} "
          f"F1={base['f1']:.3f}  ({base['pairs_caught']} pairs caught, {base['false_pairs']} false)\n")

    keys = [m.key for m in MODELS] if args.models == "all" else args.models.split(",")
    taus = np.arange(0.50, 0.96, 0.01)
    results = {"tier1": base, "n_posts": len(ids), "n_pairs": int(len(Y)), "models": {}}

    for k in keys:
        m = BY_KEY.get(k)
        if not m:
            print(f"! unknown model {k}"); continue
        dts = [args.dtype] if args.dtype else m.dtypes
        for dt in dts:
            try:
                V, used, fname, secs = embed_cached(m, texts, dt, args.max_len)
            except Exception as e:
                print(f"  {k:22} {dt:5} FAILED: {str(e)[:90]}")
                continue
            sim = pair_sims(V, I, J)
            auc = float(roc_auc_score(Y, sim))
            mauc, n_stories = macro_auc(sim, I, J, Y, cluster_of, ids)
            rows = sweep(sim, Y, taus)
            best = max(rows, key=lambda r: r["f1"])
            ari = float(greedy_ari(V, ids, cluster_of, best["tau"]))
            print(f"  {k:22} {used:5} dim={V.shape[1]:<5} AUC={auc:.4f} "
                  f"macroAUC={mauc:.4f}({n_stories}st)  "
                  f"bestF1={best['f1']:.4f}@tau={best['tau']:.2f}  ARI={ari:.4f}  "
                  f"({secs:.1f}s cpu)  {m.note}")
            results["models"][f"{k}:{used}"] = {
                "model": m.onnx_repo, "dtype": used, "file": fname, "dim": int(V.shape[1]),
                "pooling": m.pooling, "prompt": m.prompt, "auc": round(auc, 4),
                "macro_auc": round(mauc, 4), "macro_stories": n_stories,
                "best": best, "greedy_ari": round(ari, 4),
                "cpu_embed_secs": round(secs, 1), "note": m.note, "sweep": rows,
            }
            # Write after EVERY model, not just at the end: this sweep has now been
            # killed twice mid-run by session restarts, and each kill threw away the
            # models that had already finished. Partial results are worth keeping.
            OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            break  # first dtype that loads wins

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")

    ranked = sorted(results["models"].items(), key=lambda kv: -kv[1]["auc"])
    print(f"\n| model | dtype | dim | AUC | macro AUC | best F1 | tau | greedy ARI |")
    print(f"|---|---|---|---|---|---|---|---|")
    for name, r in ranked:
        print(f"| {name} | {r['dtype']} | {r['dim']} | {r['auc']:.4f} | "
              f"{r.get('macro_auc',float('nan')):.4f} | "
              f"{r['best']['f1']:.4f} | {r['best']['tau']:.2f} | {r['greedy_ari']:.4f} |")


if __name__ == "__main__":
    main()
