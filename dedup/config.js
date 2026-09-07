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

  // Calibrated on the shipped algorithm (greedy assignment against representatives in a
  // 400-post window), scored only over labeller-ADJUDICATED pairs, on 30.3M scored pairs
  // across 343 macro-scoreable stories. At 0.89: 1.96% of posts collapse at 95.1%
  // precision (176 correct / 9 wrong of the 185 judged).
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
  threshold: 0.89,

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
    collapseRate: 0.0196,  // share of timeline hidden at tau
    collapsePrecision: 0.951,
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
  /** Rolling window of posts held for comparison. X recycles DOM nodes aggressively,
   *  so this is keyed by status id and is the real memory bound. */
  windowSize: 400,
  /** Embedding requests are coalesced into one worker round trip. */
  batchMax: 32,
  batchWindowMs: 25,
  /** Consecutive transport failures before muting, and how long to mute for. */
  breakerTrips: 5,
  breakerMuteMs: 30_000,
  /** Never collapse a post whose author matches the representative's -- self-threads
   *  and reply chains are legitimately repetitive, not duplicates. */
  exemptSameAuthor: true,
  /** Tier-1 exact-signal pre-pass. Measured recall is 0.004 (29 of 4.3M agreed pairs),
   *  so this is nearly worthless on its own and must never be the primary mechanism --
   *  but it is free, precise (0.707), and catches identical reposted media before the
   *  model runs. Kept small and clearly secondary for that reason. */
  useExactSignals: true,
  /** Posts shorter than this are never clustered. Kept at 15 after measuring, not by
   *  intuition: short reactions ("Fetterman is finished", "Says boo Carter") are the
   *  largest class of genuine false collapses, so raising the floor looked obviously
   *  right -- but at tau=0.75 it buys nothing (precision 93.4% at 15 chars, 93.5% at
   *  30, 92.8% at 40) while cutting the posts folded by 40%. The threshold is already
   *  doing that work. Enforced in observer.js extract(). */
  minTextLength: 15,
}
