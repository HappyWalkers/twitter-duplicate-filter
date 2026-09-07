#!/usr/bin/env python3
"""
Partition unlabelled posts into chunks for parallel labelling by subagents.

Partitioned BY TREND, not by arbitrary slices. Independent labellers each invent their
own story ids, so if two posts about one event land in different chunks they get
different ids and the pair is silently recorded as a NEGATIVE -- which is precisely the
error class that made codex's first pass over-split. Posts from one trend page are
overwhelmingly about the same handful of events, so keeping a trend intact inside one
chunk keeps almost every true positive pair inside a single labeller's view.

Chunks are capped by post count; a trend larger than the cap gets its own chunk rather
than being split.
"""
import argparse, collections, json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="labels_codex.json",
                    help="skip posts already labelled here (comma-separated)")
    ap.add_argument("--size", type=int, default=220, help="target posts per chunk")
    ap.add_argument("--out", default="chunks")
    ap.add_argument("--all", action="store_true",
                    help="chunk ALL posts, not just unlabelled ones (for an independent pass)")
    args = ap.parse_args()

    posts = [json.loads(l) for l in (HERE / "posts.jsonl").read_text().split("\n") if l.strip()]

    done = set()
    if not args.all:
        for f in args.exclude.split(","):
            p = HERE / f.strip()
            if p.exists():
                done |= set(json.loads(p.read_text()))
    todo = [p for p in posts if p["statusId"] not in done]

    # group by trend (None -> its own bucket keyed by source)
    groups = collections.defaultdict(list)
    for p in todo:
        groups[p.get("trend") or f"__{p['source']}"].append(p)

    chunks, cur = [], []
    for _, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) >= args.size:
            chunks.append(members)          # big trend gets its own chunk, never split
            continue
        if len(cur) + len(members) > args.size and cur:
            chunks.append(cur); cur = []
        cur.extend(members)
    if cur:
        chunks.append(cur)

    outdir = HERE / args.out
    outdir.mkdir(exist_ok=True)
    for f in outdir.glob("chunk_*.json"):
        f.unlink()
    for i, ch in enumerate(chunks):
        (outdir / f"chunk_{i:02d}.json").write_text(json.dumps(
            [{"statusId": p["statusId"], "author": p["authorId"],
              "text": p["text"][:400], "trend": p.get("trend")} for p in ch],
            ensure_ascii=False, indent=1))
    print(f"{len(todo)} unlabelled posts -> {len(chunks)} chunks in {outdir}")
    for i, ch in enumerate(chunks):
        trends = sorted({p.get("trend") or p["source"] for p in ch})
        print(f"  chunk_{i:02d}: {len(ch):4} posts, {len(trends):2} trends "
              f"({', '.join(str(t)[:18] for t in trends[:4])}{'...' if len(trends)>4 else ''})")


if __name__ == "__main__":
    main()
