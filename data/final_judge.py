#!/usr/bin/env python3
"""
Final adjudication: consolidate the Sonnet chunk labels, then reconcile against codex.

Two jobs, in order.

1) CROSS-CHUNK MERGE. The chunks were partitioned by trend so that same-story posts
   stayed inside one labeller's view. That mostly worked, but one real-world event can
   span several trends -- the NBA ruling appeared under "Clippers", "Ballmer" AND
   "Uncle Dennis" -- so three different agents named the same story three different
   ways. Left alone, every pair across those ids is silently recorded as a NEGATIVE,
   which is the exact error that made codex's first pass over-split. Merges below are
   my judgement calls, each verified by reading the posts.

2) RECONCILE vs codex. codex is independently re-labelling the same posts, so the
   overlap gives a genuine two-labeller agreement number. This matters because my
   earlier adjudication pass was DERIVED from codex's output and so agreed with it by
   construction (~0.95); it could not measure anything. This one can.

Gold = pairs where both labellers agree. Disagreements are excluded and counted, never
silently recoded as negatives.
"""
import argparse, collections, glob, json
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent

# Verified by reading posts from each id before merging -- one real-world event named
# differently by different agents. Left unmerged, every pair across two ids is silently
# recorded as a NEGATIVE, which is the single largest labelling error this pipeline can
# make (it punishes exactly the models that correctly group the story).
MERGES = {
    # Same product launch; one agent filed the announcement, another the usage reactions.
    "gpt6_astra_launch":                              "gpt6_astra_release",
    "gpt6_astra_release_reactions":                   "gpt6_astra_release",
    # Same concert, same guest appearances (Big Sean / Twista / Future / 2 Chainz).
    "ye_chicago_concert_brings_out_big_sean":         "kanye_chicago_concert_guests",
    "kanye_chicago_night1_soldier_field_guests":      "kanye_chicago_concert_guests",
    # Same loss and its fallout.
    "rutgers_umass_upset_big_ten_fallout":            "rutgers_umass_loss_fallout",
    "rutgers_schiano_embarrassing_loss_job_security": "rutgers_umass_loss_fallout",
    # Same HOH reign -- the win and the nominations are one night's arc, and posts
    # cross-reference each other ("waking up to see Barrett being HoH").
    "bb28_barrett_wins_hoh":                          "bb28_barrett_hoh",
    "bb28_barrett_hoh_nomination_plans":              "bb28_barrett_hoh",
}
# Deliberately NOT merged:
#   kanye_soldier_field_chicago_ticket_resale -- people reselling tickets is a different
#     thing from the performance, despite the same event and day.
#   spam_arabic_perfume_discount_bot vs spam_arabic_discount_code_ads -- same advertiser,
#     but the templates' text differs substantially, so a model SHOULD keep them apart.
#   The Colorado/Georgia Tech game's six per-player stories, and Beyonce's birthday vs
#     her re-release -- distinct moments, per the labelling rules.


def _load(files):
    m = {}
    for f in files:
        m.update(json.loads(Path(f).read_text()))
    return {k: MERGES.get(v, v) for k, v in m.items()}


def load_sonnet():
    return _load(sorted(glob.glob(str(HERE / "chunks" / "labels_*.json"))))


# chunks_v2 was labelled by THREE separate subagents, and each invented its own ids: the
# Michigan-WMU clock finish came back as big_ten_michigan_wmu_clock_controversy,
# michigan_western_michigan_clock_controversy and michigan_wmu_hail_mary_clock_controversy
# from the three of them. Pooling their output into one map would declare every cross-agent
# pair of that story "different story" -- a false negative on the single largest story in
# the corpus. Each agent is therefore its own labeller, and a pair only counts as judged
# when one agent saw both of its members. (The original chunks/ pass has the same seam, but
# its agent boundaries were not recorded, so it stays a single namespace.)
V2_AGENTS = {"sonnet_v2a": ("00", "01", "02"),
             "sonnet_v2b": ("03", "04", "05"),
             "sonnet_v2c": ("06", "07", "08")}


def load_v2():
    out = []
    for name, nums in V2_AGENTS.items():
        files = [HERE / "chunks_v2" / f"labels_{n}.json" for n in nums]
        files = [f for f in files if f.exists()]
        if files:
            out.append((name, _load(files)))
    return out


def stats(name, lab):
    c = collections.Counter(lab.values())
    multi = {k: v for k, v in c.items() if v > 1}
    pairs = sum(v * (v - 1) // 2 for v in multi.values())
    print(f"{name:8} {len(lab):5} posts | {len(c):5} stories | {len(multi):4} multi-post "
          f"| {pairs:7} same-story pairs | largest {max(c.values()) if c else 0}")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-codex", type=int, default=0,
                    help="minimum codex labels before reconciling (0 = whatever exists)")
    args = ap.parse_args()

    son = load_sonnet()
    (HERE / "labels_sonnet.json").write_text(json.dumps(son, indent=1))
    stats("sonnet", son)

    agy = json.loads((HERE / "labels_agy.json").read_text()) \
        if (HERE / "labels_agy.json").exists() else {}
    if agy:
        stats("agy", agy)

    cod = json.loads((HERE / "labels_codex.json").read_text())
    stats("codex", cod)

    # TWO second labellers, reconciled SEPARATELY against codex and then unioned.
    #
    # sonnet covers the original corpus, agy the backlog. Merging them into one map first
    # would be wrong: they invent ids in unrelated namespaces, so a pair whose members were
    # second-labelled by different labellers would be recorded as "different story" no
    # matter what it is -- manufacturing false negatives at the seam between the two.
    # A pair is only genuinely double-checked when ONE second labeller saw BOTH its members,
    # so each overlap is scored on its own and the agreed pairs are unioned.
    from sklearn.metrics import adjusted_rand_score
    overlaps = [(name, lab) for name, lab in
                [("sonnet", son), ("agy", agy)] + load_v2() if lab]

    both_all, sa, sb, agree_pos, disputed_n = set(), set(), set(), set(), 0
    for name, lab in overlaps:
        ov = sorted(set(lab) & set(cod))
        if len(ov) < 2:
            print(f"\noverlap codex&{name}: {len(ov)} posts -- skipped")
            continue
        ari = adjusted_rand_score([cod[i] for i in ov], [lab[i] for i in ov])
        a = {q for q in combinations(ov, 2) if cod[q[0]] == cod[q[1]]}
        b = {q for q in combinations(ov, 2) if lab[q[0]] == lab[q[1]]}
        print(f"\noverlap codex&{name}: {len(ov)} posts | ARI = {ari:.4f}")
        print(f"  codex SAME: {len(a):6}  {name} SAME: {len(b):6}  both: {len(a & b):6}"
              f"  disputed: {len(a ^ b):6}")
        both_all |= set(ov); sa |= a; sb |= b
        agree_pos |= a & b
        disputed_n += len(a ^ b)

    both = sorted(both_all)
    print(f"\ntotal double-labelled: {len(both)} posts, "
          f"{len(agree_pos)} double-confirmed positive pairs")
    if len(both) < args.require_codex:
        print("  too few to reconcile yet; rerun once codex has covered more.")
        return

    # Gold clusters = connected components of "both labellers said same".
    parent = {i: i for i in both}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for x, y in agree_pos:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    gold = {i: f"s_{find(i)}" for i in both}

    # Posts only one labeller saw keep that labeller's assignment, so coverage is not
    # thrown away -- but they contribute no double-checked positive pairs.
    for i, v in son.items():
        gold.setdefault(i, f"son_{v}")
    for i, v in agy.items():
        gold.setdefault(i, f"agy_{v}")
    for name, lab in load_v2():
        for i, v in lab.items():
            gold.setdefault(i, f"{name}_{v}")
    for i, v in cod.items():
        gold.setdefault(i, f"cod_{v}")

    # The RIGOROUS eval set: only pairs inside the double-labelled overlap, and only
    # where both labellers agree. Adjudicating a sample showed each makes opposite errors
    # -- sonnet over-merges topically-related events, codex isolates real stories into
    # singletons -- so the intersection excludes both failure modes. benchmark.py prefers
    # this file over deriving pairs from the clustering.
    agreed = [[x, y, True] for x, y in sorted(agree_pos)]
    # Negatives only where a pair was actually CHECKED -- i.e. both members fell inside
    # the same labeller's overlap. combinations(both, 2) spans the seam between the two
    # second labellers, and a cross-seam pair was never judged by anyone; recording it as
    # an agreed negative would invent labels nobody assigned.
    # Negatives are DERIVABLE, so they are not stored. Enumerating them produced a 1.4GB
    # gold_pairs.json (22M rows) that costs several GB of RAM just to json.load, and it
    # grows quadratically with the overlap. The compact spec below says the same thing --
    # "these posts were checked by this labeller; among them these pairs are positive;
    # every other pair inside that overlap is a checked negative" -- in a few MB.
    # benchmark.py expands it with numpy. gold_pairs.json is still written for
    # compatibility, but only when it would be small enough to be worth having.
    # `disputed` is NOT optional. Without it an expander has to assume every non-positive
    # pair in an overlap is a negative, which silently converts all 41,866 pairs the two
    # labellers DISAGREED on into agreed negatives -- the exact pairs the whole
    # reconciliation exists to throw away, and the ones models are most likely to be
    # right about. Verified by round-tripping against the enumerated file: without this
    # key the expansion came back with precisely `disputed` extra rows.
    compact = {
        "overlaps": {name: sorted(set(lab) & set(cod)) for name, lab in overlaps},
        "positives": [[x, y] for x, y in sorted(agree_pos)],
        "disputed": [[x, y] for x, y in sorted(sa ^ sb)],
    }
    (HERE / "gold_pairs_compact.json").write_text(json.dumps(compact))
    n_all = sum(len(v) * (len(v) - 1) // 2 for v in compact["overlaps"].values())
    n_checked = n_all - len(compact["disputed"])
    print(f"\nwrote gold_pairs_compact.json: {len(compact['positives'])} positives, "
          f"{n_checked - len(compact['positives'])} derived negatives, "
          f"{len(compact['disputed'])} disputed excluded ({n_checked} scored pairs)")
    neg = []
    if n_checked <= 6_000_000:
        checked = set()
        for name, lab in overlaps:
            checked |= set(combinations(sorted(set(lab) & set(cod)), 2))
        neg = [[x, y, False] for x, y in sorted(checked)
               if (x, y) not in sa and (x, y) not in sb]
    if neg:
        (HERE / "gold_pairs.json").write_text(json.dumps(agreed + neg, indent=1))
        print(f"wrote gold_pairs.json: {len(agreed)} agreed-positive, "
              f"{len(neg)} agreed-negative")
    else:
        # Stale and now smaller than the compact spec it would contradict; leaving it
        # would let benchmark.py silently score an older, narrower pair set.
        stale = HERE / "gold_pairs.json"
        if stale.exists():
            stale.rename(HERE / "gold_pairs.superseded.json")
            print("gold_pairs.json too large to enumerate -> superseded by the compact spec")

    (HERE / "gold_final.json").write_text(json.dumps(gold, indent=1))
    p = stats("GOLD", gold)
    print(f"\nwrote gold_final.json  ({disputed_n} disputed pairs excluded, "
          f"{len(agree_pos)} double-confirmed positives inside the overlaps)")


if __name__ == "__main__":
    main()
