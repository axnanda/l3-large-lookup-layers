# A critical look at L3 (Large Lookup Layers) and at a spiking L3

This note records the design reasoning behind `l3lab`, what we think is right and fragile about the
L3 layer as published (Tseng and De Sa, "L3: Large Lookup Layers", arXiv 2601.21461), how it was
adapted to Fashion-MNIST, and what a spiking version can and cannot keep. The WeightWatcher
experiments in `results/` are designed around the predictions made here.

## 1. What an L3 layer actually computes

Per position, with hidden state `x` (dimension `h`) and token id `t` that owns a contiguous block of
`d_t` rows of two tables (keys `K`, shape `n_emb x h`; values `V`, shape `n_emb x d_emb`):

    p   = softmax over the d_t rows of  K_t . RMSNorm(x)
    c   = p . V_t                          (one d_emb vector, a convex combination of the token's values)
    out = W_mix [ RMSNorm(W_up c) ; x ]    (a gated residual MLP that mixes the looked-up value with x)

So the layer is an embedding table generalised to a *set* of embeddings per token, chosen by a
context-dependent softmax, followed by an ordinary two-matrix MLP. The only discrete operation is the
gather by token id; everything else is dense and differentiable, which is why the user's corrected port
trains without a straight-through estimator.

Our reference implementation (`l3lab/l3_ref.py`) does exactly this with a padded `[V, k]` index table and
a masked softmax, and matches the paper's blocked forward to 1e-6 (`tests/test_l3_ref.py`). The paper's
forward is a performance device (sort tokens, run block-diagonal masked GEMMs); it is correct, but it needs
`bb` to divide `batch x sequence`, and its FlexAttention variant is broken as written
(`docs/L3_REF_NOTES.md`). Those are the first two "improvements": a correct reference, and tests that pin
the semantics so the fast path can be changed safely.

## 2. Where the design is strong

* **Capacity without FLOPs.** Parameters scale with `n_emb` while compute scales with `k` (at most `k` keys
  and values are touched per position). This is the point of the paper and it survives our port.
* **Frequency-aware allocation.** Giving frequent tokens more entries (the LZW rule, or our
  `allocate_by_frequency`) spends capacity where gradient actually arrives. Rare tokens would never
  train more than one or two entries anyway.
* **A clean residual path.** `W_mix[.; x]` keeps `x` available, so a block can learn to ignore the
  lookup. This matters for the spiking variant (section 4).

## 3. Where it is fragile, and what we changed or instrumented

1. **Rows that never learn.** Only the rows of tokens present in a batch receive gradient, and within a
   token only the rows the softmax selects get a non-negligible update. Row norms therefore track token
   frequency and selection frequency, not learned structure. For WeightWatcher this is decisive: the
   eigenvalue spectrum of `K^T K` or `V^T V` carries a *frequency-induced* tail. A small alpha on `K` or
   `V` is expected and is not evidence of over-fitting. We therefore (a) log per-role alphas and never
   average lookup tables with dense layers, (b) read the "ideal alpha of 2" claim only on `W_up`, `W_mix`
   and the classifier head, and (c) record utilisation (`lookup_probs`, fraction of entries ever selected).
2. **Softmax over tiny sets.** With `d_t = 1` the softmax is the constant 1 and the layer degenerates to
   an embedding; with `d_t` small and keys near-parallel, selection is almost uniform and the value table
   trains as one averaged embedding per token. We expose `score_scale` and keep `d_t = 1` allowed so the
   degenerate case is visible in the data instead of hidden.
3. **Query collapse without position.** In the paper the hidden state `x` already contains position and
   context. On image tokens the hidden state at the first block is a pure function of the token id, so the
   lookup query is constant per token and the model collapses to a bag of tokens. Our classifier adds a
   learned patch-position embedding (measured: 79 % vs 74 % test accuracy after two epochs on 10 k images).
4. **Allocation feasibility.** `n_emb` must lie in `[V, V k]`; the sweep enforces this.
5. **Two RMSNorms, no bias.** `norm_out` normalises `W_up c` before mixing (the user's bug fix #2 was
   right: the norm has to live in `d_up`). Keeping all weight matrices bias-free also keeps WeightWatcher's
   `W^T W` analysis clean.

## 4. What "spiking L3" can mean

Two things in the layer are candidates for spiking: the *query* and the *selection*.

* `rate_query`: the hidden state is a spike train from a LIF population; keys are scored against 0/1 query
  vectors each time step; the softmax selection and the value aggregation stay continuous; the output passes
  through LIF units again. This keeps the lookup exact and tests whether binary, temporally integrated
  queries are enough to route.
* `spiking_memory`: every allocated entry is itself a LIF unit driven by the key score of its token's query;
  entries that cross threshold fire and their values are summed and normalised by the number of firing
  entries. This is content-addressed memory implemented by threshold units (winner-take-all without the
  softmax), the closest thing to a biologically plausible lookup.

Design facts that matter for interpretation (`docs/SNN_L3_DESIGN.md` has the table):

* A 0/1 query with rate `r` has squared norm `r h`, so key scores are already smaller than for an
  RMS-normalised dense query; the usual `1/sqrt(h)` scaling silences every memory unit (0.0 % firing at
  initialisation). The default `score_scale` is therefore 1.0. Any claim about spiking lookups has to
  state the score scale and the memory firing fraction.
* Memory units persist membrane potential across the `T` steps of a position, so selection has hysteresis:
  an entry can fire late in the window because evidence accumulates. That is a feature (temporal
  integration) and a hazard (entries selected by leak dynamics rather than content).
* Dense `W_up` / `W_mix` run per time step, so the spiking layer costs about `T` times the dense one.

## 5. Predictions for the WeightWatcher experiments

These follow from the earlier Hessian study on spiking networks (block upper-triangular surrogate
Jacobian, non-negative rate features with a dominant mean direction, dead units growing with depth,
threshold-crossing noise) and from how WeightWatcher fits alpha.

* P1. Spiking dense layers (`SNNMLP3.fc2`, spiking `W_up`/`W_mix`) show a rank-one spike along the
  input feature-mean direction. Projecting that direction out of `W` (`scripts/run_controls.py`,
  `ww_meanproj`) should raise alpha and remove the WeightWatcher 'over-trained' warning; centring the
  features during training (`SNNMLP3Centered`) should prevent it.
* P2. Alpha of lookup tables (`K`, `V`, embeddings) is set by selection frequency and is not comparable
  with alpha of dense layers; it should sit low and barely respond to training quality.
* P3. The average alpha over all layers is a poor predictor of validation accuracy for L3 and for spiking
  models because the layer types pull in opposite directions (lookup tables low, silent or under-driven
  layers high); per-role alpha of the dense layers should correlate better.
* P4. Alpha keeps drifting after validation accuracy has plateaued for spiking models (threshold noise keeps
  adding correlations), so "alpha trends down without improvement" should be the common case.
* P5. The surrogate slope rescales updates layer by layer, so alpha of the first spiking layer depends on
  the slope while the readout's does not.

The sweeps (`scripts/run_*_sweep.py`), trajectories (`scripts/run_trajectories.py`) and controls
(`scripts/run_controls.py`) are built to test exactly these five statements; the report in `README.md`
records which held.
