# WeightWatcher on L3 and on spiking L3: what alpha says about lookup layers and spiking layers

Status: draft written while the sweeps were still running; the numbers marked (partial) are updated in the
final section 9 and in `figures/final/`.

## 0. Summary

We trained many variations of three model families on Fashion-MNIST and ran WeightWatcher
(`weightwatcher` 0.7.7, `min_evals=50`, randomize and MP fits on) on every weight matrix:

* `MLP3`, the 784-h-h-10 ReLU network of Charles Martin's post, as the baseline;
* `L3Classifier`, the L3 lookup layer of Tseng and De Sa (arXiv 2601.21461) applied to k-means patch tokens,
  with the user's corrected port verified against a clean reference implementation;
* spiking versions of both (`SNNMLP3`, `SNNL3Classifier` with a spiking query or a fully spiking memory).

What we found, against the three observations in the post:

1. **"Average alpha < 2 means degraded or over-fit."** For the MLP3 baseline, lower average alpha goes with
   *better* validation accuracy across the sweep (Spearman rho about -0.3 to -0.5), and the runs with alpha < 2
   are the longest-trained Adam runs with a visible train/validation gap, so the "over-fit" half of the statement
   holds while the "degraded" half does not at this scale. For spiking MLPs the relation disappears
   (rho about 0): alpha sits at 1.6-2.2 whatever the accuracy. For spiking L3 it is strongly negative
   (rho about -0.8) because the under-driven spiking models stay random-like (high alpha) and bad.
2. **"alpha in [2, 3] can still be good."** True everywhere, and for L3 it is the typical state of a good
   model: the dense mixing layers of the best L3 models sit at alpha 2.0-2.2 while their lookup tables sit at
   2.6-4.7, so the average is above 2 for every good L3 model.
3. **"alpha can trend down without accuracy improving."** Common in every family (12-30 % of epoch-to-epoch
   alpha drops come with no validation gain), and the long trajectories show more: with Adam, alpha overshoots
   below 2 by epoch 10, then *rises back* above 2 as the network memorises the training set; with 30 % label
   noise alpha rises to 2.7-3.0 while validation accuracy falls, i.e. memorising noise makes the spectrum *less*
   heavy-tailed, not more. With plain SGD alpha decreases monotonically toward 2.0 over 40 epochs as accuracy
   improves, which is the textbook HTSR picture.

Spiking-specific results:

* Spiking does not change alpha of the dense layers: `SNNMLP3` and `MLP3` trained identically reach the same
  fc1 / fc2 alphas (1.73 / 1.89 vs 1.73 / 2.00) at the same accuracy. The spiking signature is in the
  *eigenvectors*: the top singular direction of spiking `W_up` / `W_mix` / key tables aligns with the mean of
  the non-negative input rates (overlap 0.3-0.98 vs 0.06-0.23 in the ANN), but that single direction carries
  only 1-4 % of the weight energy and projecting it out leaves alpha unchanged. Our earlier prediction that
  the feature-mean spike would *set* alpha for spiking layers is therefore wrong for weight spectra; it is a
  curvature effect, not a weight-tail effect.
* The surrogate slope is a hidden knob on alpha: fc1 alpha of `SNNMLP3` rises monotonically with slope
  (1.60, 1.76, 1.91, 2.14 for k = 1, 2, 4, 8) and a slope of 25 kills the network (94 % silent units,
  validation accuracy 0.20) while its alphas look healthy (3.2 / 4.0). Alpha cannot see a dead network.
* In spiking L3 the token embedding never leaves the random phase (alpha 17, 'under-trained') because its
  gradient has to pass through the spiking memory; the key tables learn a common direction aligned with the
  mean spike pattern; the dense mixing layers reach alpha 2.1-2.2 like their ANN twins.

L3-specific results:

* Lookup tables (embedding, keys, values) have *higher* alpha than dense layers, not lower as we predicted:
  their rows are updated in proportion to token frequency and softmax selection, which produces a few isolated
  large eigenvalues (fingers) on an otherwise random bulk; `fix_fingers='clip_xmax'` moves key-table alphas
  from 10-35 to 5-10 on under-trained models, and the embedding fit uses only the top 10-60 eigenvalues
  (`xmin` in the hundreds). Average alpha over all layers is therefore a poor summary of an L3 model
  (rho with validation accuracy -0.2, not significant); alpha of the dense mixing layers should be read instead.
* The best L3 classifier we trained (val 0.845, 2 layers, V = 512, n_emb = 4096, k = 8) has `W_mix` at
  alpha 2.00 / 2.01 and `W_up` at 2.07 / 2.17, i.e. the dense part of a good lookup model sits exactly where
  HTSR says a well-trained layer should.

## 1. Setup

Data: Fashion-MNIST, 50 k train / 10 k validation (fixed split) / 10 k test. L3 models see each image as 16
patch tokens (7x7 patches quantised by a k-means codebook of V codes fitted on 20 k training images) plus a
learned position embedding; the paper's LZW-style allocation becomes a frequency-proportional allocation of
`n_emb` embeddings over the V codes with at most k per code. Models and WeightWatcher harness:
`l3lab/` (see `docs/DESIGN_SPEC.md`, `docs/SNN_L3_DESIGN.md`, `docs/CRITICAL_DESIGN.md`).

WeightWatcher settings: `analyze(min_evals=50, randomize=True, mp_fit=True)`; layers below 50 eigenvalues
(the 10-way readout) are reported but excluded from averages (`alpha_min50_avg`); `alpha_default_avg` is the
plain mean over all analysed layers as in the post. Extras computed by us per matrix: top-singular-value
energy, overlap of the top singular vectors with the input-mean direction, dead-row fraction, effective rank,
Hill tail index of the entries. Every run also stores the per-epoch alpha trajectory when it trained for at most
10 epochs.

Sweeps (random configurations, one JSON per run, resumable): `scripts/run_mlp_sweep.py` (width 64-1024,
Adam/SGD, lr, weight decay, dropout, label noise 0/10/30 %, training subset 2 k / 10 k / 50 k, 1-20 epochs),
`scripts/run_l3_sweep.py` (V, n_emb, k, h, d_emb, depth 1-3, 1-8 epochs), `scripts/run_snn_sweep.py` (T, surrogate
slope, leak, spiking mode); `scripts/run_trajectories.py` (40-epoch runs with WeightWatcher every epoch);
`scripts/run_controls.py` (centred features, mean-direction projection, live-unit sub-matrices, finger
clipping, entry tail index). All on 4 CPU cores in about four hours, so the sweeps are 100-200 runs rather than
the post's 1000 and the spiking L3 arm uses 10 k-image subsets.

## 2. MLP3 baseline (partial: 86 of 160 runs)

Pooled over the MLP3 runs, Spearman(alpha_min50_avg, val acc) = -0.29 (p = 0.01), Spearman with train
accuracy -0.29; runs with dense-layer alpha < 2 (n = 25) have mean validation accuracy 0.80 versus 0.71 for
alpha >= 2 (n = 52) and never exceed 0.857, while the best run (0.885) has alpha slightly above 2. So lower
alpha means a better-trained model; alpha < 2 marks long Adam training with a train/validation gap, not
degradation. `figures/final/alpha_vs_acc__alpha_min50_avg.png`.

## 3. L3 classifiers (partial: 28 of 40 runs)

Per-role alphas of a good L3 model (controls run, val 0.845): embedding 4.7, keys 2.6-2.7, values 3.6,
`W_up` 2.1-2.2, `W_mix` 2.0. Across the sweep the correlation of average alpha with validation accuracy is
weak (rho -0.2, p = 0.3) and the dense-layer alpha alone is not better (rho -0.1), because most sweep runs are
short (1-8 epochs) and their lookup tables are still random-like (alpha 5-35, 'under-trained'). The classifier
reaches 0.80-0.84 validation accuracy with one or two L3 blocks, below the 0.89 of the MLP3 on raw pixels, as
expected for a bag of 16 quantised patches.

## 4. Spiking models (partial: 36 of 40 runs)

`SNNMLP3`: alphas 1.6-2.2 for every run; no correlation with accuracy (rho 0.04); fc1 alpha grows with the
surrogate slope and the dead-row fraction shrinks with it (0.12 at slope 1, 0 at slope 8).
`SNNL3Classifier`: rho(alpha, val acc) = -0.8 (n = 12); the models that train well (val 0.73-0.80) have
dense-layer alpha 1.9-2.2, the ones that do not (val 0.23-0.42) stay at 2.7-17. Spiking memory and spiking
query modes behave alike in alpha; the spiking memory fires 28-35 % of its entries at initialisation and
selects 2-3 entries per lookup.

## 5. Controls (`results/controls/summary.md`)

| model (10 epochs, Adam 1e-3) | val acc | fc1 alpha | fc2 alpha | fc1 alpha after removing the input-mean direction | fc1 top-vector overlap with input mean |
|---|---|---|---|---|---|
| MLP3 h = 512 | 0.891 | 1.73 | 2.00 | 1.73 | 0.47 |
| MLP3, centred features | 0.883 | 1.69 | 1.88 | 1.69 | 0.30 |
| SNNMLP3 h = 512, T = 8, slope 4 | 0.887 | 1.73 | 1.89 | 1.74 | 0.42 |
| SNNMLP3, centred spikes | 0.881 | 1.71 | 1.87 | 1.72 | 0.83 |
| SNNMLP3, slope 25 | 0.203 | 3.16 | 3.98 | 3.58 | 0.91 |

Alpha is insensitive to centring, to projecting out the mean direction and to removing dead units
(live-unit alpha within 0.1 of the full alpha); the Hill tail index of the weight *entries* is 3.5-4
(kurtosis 2.5-3.7) for trained dense layers and 5-13 for lookup tables.

## 6. Trajectories (`figures/trajectories_alpha.png`, `results/trajectories/summary.md`)

| run | alpha_avg at epochs 3 / 9 / 15 / 24 / 36 | best val (epoch) | final train / val |
|---|---|---|---|
| MLP3 h = 512, Adam, clean | 2.14 / 1.88 / 1.80 / 1.89 / 2.00 | 0.905 (26) | 0.980 / 0.904 |
| MLP3 h = 512, Adam, 30 % label noise | 2.28 / 1.98 / 2.24 / 2.38 / 3.04 | 0.882 (13) | 0.794 / 0.780 |
| MLP3 h = 1024, SGD 0.05, clean | 3.02 / 2.59 / 2.39 / 2.19 / 2.00 | 0.904 (40) | 0.973 / 0.901 |

With Adam, alpha undershoots 2 while validation accuracy is still improving, and the WeightWatcher fit's
`xmin` drifts from 0.4 to 21 (it ends up fitting only the top 67 eigenvalues) as the network memorises;
with SGD alpha approaches 2 from above and arrives there with the best accuracy; with label noise alpha
rises as memorisation proceeds.

## 7. Reading these results in HTSR terms

* The "ideal alpha = 2" statement is reproduced for the dense layers of good models trained with SGD or
  moderately with Adam, including the mixing layers of L3 and of spiking L3.
* Alpha < 2 is a signature of a particular optimiser regime (Adam, many epochs, small batches) rather than of
  bad accuracy; the trajectories show that memorisation of clean data first pushes alpha below 2 and then
  back up, and memorisation of noisy labels pushes it up, so the mapping from alpha to "over-fit" is not
  monotone.
* Lookup tables and embeddings are not in the heavy-tailed phase at all at this scale: they are random bulk
  plus frequency-driven fingers, and their alpha should not be averaged with dense layers.
* Spiking layers reach the same alpha as their ANN twins; what spiking changes is the alignment of the top
  singular vectors with the mean firing pattern (a feature-mean, data-like effect) and the dependence of alpha
  on the surrogate slope. Alpha cannot distinguish a healthy spiking network from one whose units are silent.
* Our earlier Hessian study showed that the surrogate-gradient update of a spiking network is not a gradient
  flow (block-triangular Jacobian, non-vanishing functional term). The weight spectra nevertheless look like
  those of ordinary networks, so whatever drives the heavy tails of W (noise plus correlation accumulation)
  does not need a potential.

## 8. Caveats

CPU-only budget: 100-200 runs per family instead of 1000, 1-20 epochs, widths up to 1024 for the MLP and 256
for L3; spiking L3 trained on 10 k-image subsets; single seed per configuration; WeightWatcher's own caveats
apply (fits with fewer than 50 eigenvalues excluded; `xmin` chosen by KS minimisation can jump between epochs;
failed fits reported as alpha = -1 are filtered out).

## 9. Final numbers

(filled in after the sweeps completed; see `figures/final/summary.md` and `figures/final/by_model.md`)
