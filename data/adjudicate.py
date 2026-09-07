#!/usr/bin/env python3
"""
Claude's independent adjudication of codex's clustering -- the second labeller.

agy was meant to be the second opinion but hit its account quota ("Individual quota
reached ... Resets in 1h52m") after ~12 calls, three separate times. Rather than block,
I reviewed all 80 of codex's multi-post clusters myself and recorded the corrections.

WHAT I ACTUALLY REVIEWED, precisely: the 703 posts sitting inside codex's multi-post
clusters. Those generate 100% of the same-story pairs, so they are where a labelling
error changes the benchmark. The 794 singletons I did NOT individually review -- they
inherit "own story" from codex. So agreement on NEGATIVE pairs is largely inherited,
not independent, and only the positive side of this gold is genuinely double-labelled.
Stating that plainly because it bounds what the agreement number below means.

The dominant correction is that codex OVER-SPLITS: it repeatedly files one event under
several story ids (the same BTS setlist reveal as 3 stories, one Minneapolis shooting as
2, one Leagues Cup elimination as 3). A model that correctly groups those was being
punished for it.
"""
import json, collections
from pathlib import Path

HERE = Path(__file__).parent
A = json.loads((HERE / "labels_codex.json").read_text())
posts = {json.loads(l)["statusId"]: json.loads(l)
         for l in (HERE / "posts.jsonl").read_text().split("\n") if l.strip()}

c = collections.Counter(A.values())
multi = sorted([k for k, v in c.items() if v > 1], key=lambda k: -c[k])
byidx = {i: sid for i, sid in enumerate(multi)}

# Codex cluster indices that describe ONE event and must be merged.
MERGES = [
    ([0, 10, 38], "bts_la_d2_love_maze_baepsae"),      # same setlist reveal, 3 ids
    ([2, 26],     "minneapolis_highrise_shooting"),
    ([4, 31, 37], "america_eliminated_by_rayados_penalties"),
    ([5, 50],     "clippers_kawhi_cap_penalty"),        # penalty + the reporting on it
    ([6, 36],     "madonna_lourdes_viral_photo"),
    ([9, 19],     "ryan_reynolds_cuban_missile_quote"),
    ([14, 34],    "fulmer_allows_ballesteros_homer"),
    ([15, 16, 49],"wuwa_hsin_suoming_gameplay_reveal"),
    ([18, 22],    "tanner_scott_blows_dodgers_lead"),
    ([25, 39],    "rayados_reach_leagues_cup_final"),
    ([27, 42],    "meta_muse_spark_13_release"),
    ([32, 52],    "gemini_38_flash_release"),
    ([41, 59],    "xiaomi_ifa_2026_showcase"),
    ([1, 77],     "cato_institute_immigration_backlash"),
    ([70, 76],    "arabic_travel_discount_spam"),       # same spam campaign, 2 ids
]

# Deliberately NOT merged: distinct moments inside one sports game (Austin Wells' hit vs
# Schlittler's start vs Fulmer's homer; Mookie's homer vs strikeout vs lineout) and
# distinct moments at one concert. Codex's granularity there is consistent and defensible,
# and merging them would make the task easier than the real product problem.

# Individual posts that do not belong to the cluster they were filed under.
EVICT = {
    # Arabic travel-discount spam sitting inside the Clippers cluster -- it belongs with
    # the other Arabic spam, and its media/text share nothing with the NBA story.
    "@AnoukChelsea": "arabic_travel_discount_spam",
}

mine = dict(A)
merged = 0
for idxs, name in MERGES:
    for i in idxs:
        sid = byidx.get(i)
        if sid is None:
            continue
        for k, v in A.items():
            if v == sid:
                mine[k] = name
                merged += 1

evicted = 0
for k, p in posts.items():
    tag = "@" + p["authorId"]
    if tag in EVICT and k in mine and mine[k] != EVICT[tag]:
        # only evict if it is currently inside a multi-post cluster
        if c[A[k]] > 1:
            mine[k] = EVICT[tag]
            evicted += 1

(HERE / "labels_claude.json").write_text(json.dumps(mine, indent=1))
cm = collections.Counter(mine.values())
mm = {k: v for k, v in cm.items() if v > 1}
pairs = sum(v * (v - 1) // 2 for v in mm.values())
old = sum(v * (v - 1) // 2 for v in c.values() if v > 1)
print(f"reassigned {merged} posts across {len(MERGES)} merges; evicted {evicted}")
print(f"codex : {len(c)} stories, {sum(1 for v in c.values() if v>1)} multi-post, {old} same-story pairs")
print(f"claude: {len(cm)} stories, {len(mm)} multi-post, {pairs} same-story pairs")
print(f"\nlargest after merge: {[f'{k[:34]}={v}' for k,v in cm.most_common(8)]}")
