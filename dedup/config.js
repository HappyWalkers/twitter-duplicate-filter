/**
 * THE SWAP POINT.
 *
 * Everything model-specific lives in this one object. Changing embedder means editing
 * MODEL below and rebuilding the worker -- no other file needs to know which model is
 * in use, what shape its output is, or how it wants its input prepared.
 *
 * Three fields are NOT cosmetic and get a model silently wrong if mismatched:
 *
 *   pooling   'mean' | 'cls' | 'lasttoken'. Taken from the model's own
 *             1_Pooling/config.json, never assumed. Across the eight models
 *             benchmarked, two use cls, two mean and three lasttoken; applying mean
 *             pooling to a lasttoken model produces plausible-looking vectors that are
 *             noise, and nothing downstream will complain.
 *
 *   prompt    A symmetric prefix from the model's config_sentence_transformers.json,
 *             applied to BOTH sides of every comparison. Empty for most; e5-family and
 *             harrier want one.
 *
 *   threshold Cosine cutoff, calibrated PER MODEL and NOT by maximising F1. F1 weighs a
 *             false collapse and a missed duplicate equally, and this product does not:
 *             a miss costs the user one redundant post, while a wrong collapse hides a
 *             post they wanted and they cannot see what they are missing to notice. So
 *             tau is chosen from a simulation of the shipped algorithm (data/collapse_
 *             sim.py) at ~93% collapse precision. Best-F1 would put bekko at 0.49 --
 *             where 32% of the timeline is hidden and most of it wrongly. Calibrated
 *             thresholds span 0.75 to 0.94 across models; never carry one over.
 */

/** @typedef {{id:string, dtype:string, pooling:'mean'|'cls'|'lasttoken', prompt:string,
 *             dim:number, threshold:number, downloadMB:number, licence:string,
 *             evidence:object}} ModelSpec */

/** @type {ModelSpec} */
export const MODEL = {
  id: 'hotchpotch/bekko-embedding-v1-a8m',
  // fp32 ON PURPOSE, not for lack of a quantised build. Measured on this machine's
  // RTX 5060 via WebGPU, q8 is a PESSIMISATION in the browser: the same model runs
  // 18-48x slower quantised (gte-multilingual-base 508.9 ms/post at q8 vs 10.5 at
  // fp32; paraphrase-MiniLM 116.2 vs 6.3). Dequantisation overhead dominates and the
  // GPU cannot use its native float paths. This inverts the usual "quantise for the
  // browser" instinct, which is why it is written down here.
  dtype: 'fp32',
  pooling: 'mean',
  prompt: '',
  dim: 384,

  // Calibrated on the shipped algorithm, scored only over labeller-ADJUDICATED pairs, on
  // 30.3M scored pairs across 343 macro-scoreable stories. With windowSize 0 (compare
  // against every post this session) 0.94 folds 1.57% of posts at 91.7% precision.
  //
  // The window is not just a memory bound -- it is an accuracy device, and removing it
  // costs more than it gives. Measured at tau 0.89: a 400-post window folds 1.96% at
  // 95.1%; 1600 folds 2.37% at 89.9%; unbounded folds 2.55% at 88.2%. Precision falls
  // monotonically as the window grows while the fold rate barely moves, because each new
  // post is being compared against thousands more representatives and buys thousands more
  // chances to match one of them spuriously. Unbounded cannot reach 95% at ANY threshold;
  // it plateaus near 92% even at 0.96.
  //
  // Temporal proximity is real evidence: posts about one story arrive close together.
  // Restore the window by setting TUNING.windowSize back to 400 and this to 0.89.
  // (Caveat on the magnitude: the corpus is in capture order, where a trend page's posts
  // are adjacent, so the window's edge is probably overstated -- but every window size
  // measured ranks the same way, so the direction is not in doubt.)
  //
  // Two earlier values were wrong, each for a different reason worth remembering:
  //
  //   0.50  came from maximising F1, which weighs a false collapse and a missed duplicate
  //         equally -- and landed on the sweep's lower boundary, a constrained optimum
  //         rather than a real one. It hid 32.6% of the timeline.
  //   0.75  came from the right metric but a gold that could not see its own errors: only
  //         32% of collapses were adjudicated, so it reported 93.4% precision. On the
  //         rebuilt gold (63% adjudicated) that same threshold measures 81.8%. The number
  //         did not change because the model changed; it changed because the measurement
  //         stopped being blind.
  //
  // The lesson for anyone retuning this: a precision figure is only as trustworthy as the
  // share of collapses actually judged, so read `unjudged_share` in collapse_sim.json
  // before believing the headline.
  threshold: 0.94,

  downloadMB: 130,
  licence: 'MIT',

  evidence: {
    // Measured on 14,201 labelled posts / 343 macro-scoreable stories / 54,256
    // double-confirmed positive pairs. Earlier figures here came from 53- and
    // 110-story golds and are superseded; the absolute AUCs are LOWER than those not
    // because anything regressed but because a bigger, more varied gold is a harder
    // benchmark.
    macroAuc: 0.9834,      // rank 1 of 8; each story weighted equally
    pooledAuc: 0.9174,     // rank 2 of 8
    bestF1: 0.3159,        // rank 1 of 8
    greedyAri: 0.2001,     // rank 1 of 8 -- this metric matches what ships
    collapseRate: 0.0157,  // share of timeline hidden at tau, windowSize 0
    collapsePrecision: 0.917,
    msPerPost: 2.6,        // real WebGPU, NVIDIA Blackwell adapter
    note: 'Statistically indistinguishable from gte-multilingual-base, and now that ' +
          'is a conclusion rather than a lack of power: paired bootstrap over 343 ' +
          'stories gives +0.0003 for bekko, 95% CI [-0.0024, +0.0032], P = 0.594 -- a ' +
          'coin flip centred on zero. The 110-story gold had suggested gte was ahead ' +
          '(+0.0035, P = 0.968, one hair from significant); tripling the stories ' +
          'dissolved that signal instead of confirming it, so it was noise. bekko ' +
          'wins best F1 and greedy ARI outright, folds MORE of the timeline at ' +
          'matched ~95% collapse precision (1.96% vs 1.48%), and is 4x faster and ' +
          '10x smaller to download.',
  },
}

/**
 * Runners-up, kept configured so swapping is a one-line change rather than a rewrite.
 * Each threshold is calibrated to the SAME ~93% collapse precision as the shipped
 * model, which is the only basis on which their collapse rates are comparable.
 *
 * gte is statistically tied with the shipped model (CI [-0.0024, +0.0032] over 343
 * stories) and costs 4x the latency and 10x the download. It is the one genuine
 * alternative; promote it only if the size and speed stop mattering.
 *
 * granite is kept for reference but should NOT be promoted: on the rebuilt gold it
 * cannot reach acceptable collapse precision at ANY threshold, topping out at 83.6%
 * (tau 0.96) where the other two exceed 95%. Its similarity distribution is compressed
 * enough that at 0.86 it hides 45% of the timeline at 43% precision. A model can look
 * fine on AUC -- granite's macro AUC is 0.9755, third of eight -- and still be unusable
 * once a threshold has to be picked, which is the whole reason the collapse simulation
 * exists alongside the ranking metrics.
 */
export const ALTERNATES = {
  'gte-ml-base': {
    id: 'onnx-community/gte-multilingual-base',
    dtype: 'fp32', pooling: 'cls', prompt: '', dim: 768, threshold: 0.94,
    downloadMB: 1255, licence: 'Apache-2.0',
    evidence: { macroAuc: 0.9831, pooledAuc: 0.9346, greedyAri: 0.1783,
                collapseRate: 0.0148, collapsePrecision: 0.944, msPerPost: 10.5 },
  },
  'granite-97m': {
    id: 'ibm-granite/granite-embedding-97m-multilingual-r2',
    dtype: 'fp32', pooling: 'cls', prompt: '', dim: 384, threshold: 0.96,
    downloadMB: 390, licence: 'Apache-2.0',
    notRecommended: 'cannot reach 95% collapse precision at any threshold (83.6% max)',
    evidence: { macroAuc: 0.9755, pooledAuc: 0.8941, greedyAri: 0.1529,
                collapseRate: 0.0310, collapsePrecision: 0.836, msPerPost: 3.9 },
  },
}

export const TUNING = {
  /** Rolling window of posts held for comparison, or 0 for no window.
   *
   *  Now 0. The window existed to bound memory, but it was also throwing away matches:
   *  on the labelled corpus a 400-post window caught only 62.3% of true duplicate pairs,
   *  because the median gap between two posts about the same story is 237 posts and the
   *  90th percentile is 954. 800 would have reached 86%, 1600 reached 97.4%. Keeping the
   *  whole session costs roughly 1.7KB per post -- 17MB after 10,000 posts -- which is
   *  cheaper than the recall it was quietly costing. */
  windowSize: 0,
  /** Upper bound on the text -> vector cache, so repeated renders of the same post are
   *  not re-embedded. Independent of windowSize: it used to be windowSize * 2, and when
   *  the window was disabled (0) that became "evict whenever size > 0" -- the cache threw
   *  away every entry the moment it was written and every re-render paid for a fresh
   *  embed. A resource bound, not a rule about what may be compared. */
  cacheMax: 5000,
  /** Embedding requests are coalesced into one worker round trip. */
  batchMax: 32,
  batchWindowMs: 25,
  /** Consecutive transport failures before muting, and how long to mute for. */
  breakerTrips: 5,
  breakerMuteMs: 30_000,
  // There was a same-author exemption here, on the theory that self-threads and reply
  // chains are legitimately repetitive. It was removed after measuring what it actually
  // blocked: of the same-author pairs scoring above threshold, 115 were the same story
  // and 9 were not -- 92.7% precision, indistinguishable from the general rate. It was
  // costing ~115 correct folds and protecting against nothing, because the concern it
  // encoded applies at LOW similarity (a thread's parts are related but worded
  // differently) while the comparison only ever happens at 0.94 and above, where the
  // texts are near-identical. Observed live: one account posting the same 271-character
  // post three times, left unfolded by this rule.
  //
  // It was also the last rule-based override in the path -- grouping is the model's.
  // There is deliberately no rule-based signal here. An exact-media-match pre-pass was
  // tried and removed: 29 pairs caught out of 4.3M (recall 0.004), and because it
  // bypassed the threshold its errors were bounded by nothing. Grouping is the model's
  // job alone, so every fold is governed by `threshold` and covered by the calibration.
  // There is no minimum post length. There was one, at 15 characters, and it was the last
  // rule deciding what the model is allowed to judge. Measured at the shipped settings,
  // removing it changes nothing -- 2.42% folded at 92.7% precision with or without -- and
  // it was making short viral posts ("plane cake", ten characters) invisible to the
  // extension, which are exactly the posts a reader meets again and again. Only genuinely
  // empty text is skipped now, because there is nothing to embed.
}
