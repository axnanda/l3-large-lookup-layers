# WeightWatcher on L3 and on spiking L3: what alpha says about lookup layers and spiking layers

## 0. Summary

We trained 208 random variations of three model families on Fashion-MNIST (plus 7 long trajectory runs and
7 control runs) and ran WeightWatcher (`weightwatcher` 0.7.7, `min_evals=50`, randomize and MP fits on) on
every weight matrix at the end of training and, for runs of at most 10 epochs, after every epoch:

* `MLP3` (122 runs), the 784-h-h-10 ReLU network of Charles Martin's post, as the baseline;
* `L3Classifier` (35 runs), the L3 lookup layer of Tseng and De Sa (arXiv 2601.21461) applied to k-means
  patch tokens, with the user's corrected port verified against a clean reference implementation;
* spiking versions of both: `SNNMLP3` (18 runs) and `SNNL3Classifier` (33 runs, spiking query or fully
  spiking memory).

Against the three observations in the post:

1. **"Average alpha < 2 means degraded or over-fit."** Pooled over all 208 runs, Spearman(average alpha,
   validation accuracy) = -0.47 (p = 6e-13): lower alpha goes with *better* accuracy, as in Martin's plots.
   For the MLP3 baseline the runs with dense-layer alpha < 2 (n = 44) average 0.78 validation accuracy versus
   0.70 for alpha >= 2 (n = 69), and the alpha < 2 group never reaches the best accuracy (0.877 vs 0.892): they
   are the long Adam runs with the largest train/validation gaps. So "over-fit" holds and "degraded" means
   "a little below the best", not "bad". For spiking MLPs the relation vanishes (rho = +0.11, n.s.); for
   spiking L3 it is strongest of all (rho = -0.67, p < 0.001), because spiking L3 models that fail to train
   stay random-like (alpha 3-17).
2. **"alpha in [2, 3] can still be good."** True in every family, and for L3 it is the typical state of a
   good model: the dense mixing layers of the best L3 models sit at alpha 2.0-2.2 while their lookup tables sit
   at 2.6-4.7, so the average is above 2 for every good L3 model. Only 2 of 24 MLP3 runs in [2, 3] are within
   0.03 of the best accuracy, but so are only 2 of 44 runs below 2.
3. **"alpha can trend down without accuracy improving."** 12-22 % of all epoch-to-epoch alpha drops in the
   sweeps came with no validation gain, and the 40-epoch trajectories show the full picture: with Adam, alpha
   overshoots below 2 by epoch 6-10 while validation accuracy is still rising, then climbs *back* above 2 as
   the network memorises the training set; with 30 % label noise alpha rises to 2.7-3.0 while validation
   accuracy falls, i.e. memorising noise makes the spectrum *less* heavy-tailed; with plain SGD alpha decreases
   monotonically toward 2.0 over 40 epochs and arrives there with the best accuracy, the textbook HTSR picture.

Spiking-specific results:

* Spiking does not change alpha of the dense layers: `SNNMLP3` and `MLP3` trained identically reach the same
  fc1 / fc2 alphas (1.73 / 1.89 vs 1.73 / 2.00) at the same accuracy (0.887 vs 0.891). The spiking signature is
  in the *eigenvectors*: the top singular direction of spiking `W_up` / `W_mix` / key tables aligns with the mean
  of the non-negative input rates (overlap 0.3-0.98 vs 0.06-0.23 in the ANN), but that direction carries only
  1-4 % of the weight energy and projecting it out leaves alpha unchanged. Our earlier prediction that the
  feature-mean spike would *set* alpha in spiking layers is therefore wrong for weight spectra: it is a curvature
  effect, not a weight-tail effect.
* The surrogate slope is a hidden knob on alpha: fc1 alpha of `SNNMLP3` rises with the slope (1.60, 1.89, 1.85,
  1.99 for k = 1, 2, 4, 8 in the sweep), and a slope of 25 kills the network (94 % silent units, validation
  accuracy 0.20) while its alphas look healthy (3.2 / 4.0). Alpha cannot see a dead network. Within the spiking
  MLP sweep, the fraction of dead rows is the strongest correlate of alpha (rho = -0.79): silent units lower
  alpha, independently of accuracy.
* In spiking L3 the token embedding never leaves the random phase (alpha 17, 'under-trained') because its
  gradient has to pass through the spiking memory; the key tables learn a common direction aligned with the mean
  spike pattern (overlap 0.98); the dense mixing layers reach alpha 2.1-2.2 like their ANN twins.

L3-specific results:

* Lookup tables (embedding, keys, values) have *higher* alpha than dense layers, not lower as we predicted:
  rows are updated in proportion to token frequency and softmax selection, giving a few isolated large
  eigenvalues ("fingers") on an otherwise random bulk; `fix_fingers='clip_xmax'` moves key-table alphas from
  10-35 to 5-10 on under-trained models, and the embedding fit uses only the top 10-60 eigenvalues
  (`xmin` in the hundreds).
* Across the L3 sweep the alpha of the *lookup tables* is the summary that tracks validation accuracy
  (rho = -0.39, p = 0.03), while the dense-layer alpha does not (rho = -0.12, n.s.) because every L3 run with more
  than a couple of epochs already has its dense layers near 2. The best L3 classifier (val 0.848, 2 blocks,
  V = 512, n_emb = 4096, k = 8) has `W_mix` at alpha 2.00 / 2.01 and `W_up` at 2.07 / 2.17: the dense part of a
  good lookup model sits exactly where HTSR says a well-trained layer should.
* In the 12-epoch L3 trajectory the best validation accuracy comes at epoch 3 (0.846) and alpha_avg then
  *rises* from 3.3 to 3.6 while the model over-fits mildly (train 0.90, val 0.83): for lookup models,
  over-fitting raises alpha.

## 1. Setup

Data: Fashion-MNIST, 50 k train / 10 k validation (fixed split) / 10 k test. L3 models see each image as 16
patch tokens (7x7 patches quantised by a k-means codebook of V codes fitted on 20 k training images) plus a
learned position embedding; the paper's LZW-style allocation becomes a frequency-proportional allocation of
`n_emb` embeddings over the V codes with at most k per code. Models and WeightWatcher harness: `l3lab/`
(see `docs/DESIGN_SPEC.md`, `docs/SNN_L3_DESIGN.md`, `docs/CRITICAL_DESIGN.md`).

WeightWatcher settings: `analyze(min_evals=50, randomize=True, mp_fit=True)`; layers below 50 eigenvalues
(the 10-way readout) are reported but excluded from averages (`alpha_min50_avg`); `alpha_default_avg` is the
plain mean over all analysed layers as in the post; failed fits (alpha = -1) are excluded. Extras computed by us
per matrix: top-singular-value energy, overlap of the top singular vectors with the input-mean direction,
dead-row fraction, effective rank, Hill tail index of the entries.

Sweeps (random configurations, one JSON per run, resumable): `scripts/run_mlp_sweep.py` (width 64-1024,
Adam/SGD, lr, weight decay, dropout, label noise 0/10/30 %, training subset 2 k / 10 k / 50 k, 1-20 epochs),
`scripts/run_l3_sweep.py` (V, n_emb, k, h, d_emb, depth 1-3, 1-8 epochs), `scripts/run_snn_sweep.py` (T,
surrogate slope, leak, spiking mode; spiking L3 on 2-10 k-image subsets, 1-4 epochs);
`scripts/run_trajectories.py` (long runs with WeightWatcher every epoch); `scripts/run_controls.py` (centred
features, mean-direction projection, live-unit sub-matrices, finger clipping, entry tail index). Everything ran
on 4 CPU cores in about 3.5 hours, so the sweeps are 18-122 runs per family rather than the post's 1000.
Analyses: `scripts/analyze_sweeps.py` (`figures/final/summary.md`, scatter plots), `scripts/analyze_by_model.py`
(`figures/final/by_model.md`), `scripts/analyze_trajectories.py`, `scripts/analyze_controls.py`.

## 2. Correlations by model family (`figures/final/by_model.md`)

| family | n | rho(alpha, val acc): all layers | dense layers | lookup tables | val acc, alpha < 2 vs >= 2 (n) |
|---|---|---|---|---|---|
| MLP3 | 122 | -0.34 (p < 0.001) | -0.34 | n/a | 0.778 (44) vs 0.700 (69) |
| L3Classifier | 35 | -0.32 (p = 0.07) | -0.12 (n.s.) | -0.39 (p = 0.03) | 0.766 (9) vs 0.749 (25) |
| SNNMLP3 | 18 | +0.11 (n.s.) | +0.11 | n/a | 0.767 (11) vs 0.804 (7) |
| SNNL3Classifier | 33 | -0.67 (p < 0.001) | -0.64 | -0.64 | 0.628 (10) vs 0.466 (21) |

Pooled over all 208 runs: rho(alpha_avg, val) = -0.47, rho(alpha_avg, train) = -0.29, rho(alpha_avg, gap) =
+0.12 (n.s.); `alpha_weighted` correlates with the *gap* (+0.55) rather than with accuracy, and
`log_spectral_norm` with both (+0.43 / +0.45). Scatter plots: `figures/final/alpha_vs_acc__alpha_min50_avg.png`
(one row per family), `figures/final/alpha_by_role_vs_val_acc.png`, `figures/final/alpha_hist_by_role.png`.

Within the spiking MLP sweep: fc1 alpha by surrogate slope 1 / 2 / 4 / 8 = 1.60 / 1.89 / 1.85 / 1.99; by
leak 0.5 / 0.9 = 1.77 / 1.98; by T = 4 / 8 / 16 = 1.96 / 1.79 / 1.82; dead-row fraction vs alpha rho = -0.79.
Within the spiking L3 sweep: dense alpha by slope 1 / 2 / 4 / 8 = 3.13 / 2.51 / 2.42 / 2.41 with validation
accuracy 0.33 / 0.53 / 0.56 / 0.57 (a slope of 1 under-drives the lookups); the two spiking modes are
indistinguishable in alpha (2.62 vs 2.51) and accuracy (0.50 both).

## 3. Controls (`results/controls/summary.md`)

| model (10 epochs, Adam 1e-3, h = 512) | train / val | fc1 alpha | fc2 alpha | fc1 alpha, input-mean removed | fc1 alpha, live units only | fc1 top-vector overlap with input mean |
|---|---|---|---|---|---|---|
| MLP3 | 0.919 / 0.891 | 1.73 | 2.00 | 1.73 | 1.70 | 0.47 |
| MLP3, centred features | 0.907 / 0.883 | 1.69 | 1.88 | 1.69 | 1.67 | 0.30 |
| SNNMLP3 (T = 8, slope 4) | 0.914 / 0.887 | 1.73 | 1.89 | 1.74 | 1.72 | 0.42 |
| SNNMLP3, centred spikes | 0.908 / 0.881 | 1.71 | 1.87 | 1.72 | 1.73 | 0.83 |
| SNNMLP3, slope 25 | 0.200 / 0.203 | 3.16 | 3.98 | 3.58 | n/a (94 % silent) | 0.91 |
| L3Classifier (6 epochs) | 0.897 / 0.845 | lookup 2.6-4.7, dense 2.0-2.2 | | | | keys 0.06-0.10 |
| SNNL3 spiking memory (4 epochs, 10 k) | 0.820 / 0.783 | embed 17, keys 3.2-3.3, dense 2.1-2.2 | | | | keys 0.98 |

Alpha is insensitive to centring, to projecting out the input-mean direction and to removing dead units;
the Hill tail index of the weight *entries* is 3.5-4 (excess kurtosis 2.5-3.7) for trained dense layers and
5-13 (kurtosis near 0) for lookup tables, i.e. heavy eigenvalue tails with light-tailed entries, as the HTMP
line of work describes.

## 4. Trajectories (`figures/trajectories_alpha.png`, `results/trajectories/summary.md`)

| run | alpha_avg at epochs 3 / 9 / 15 / 24 / 36 (or last) | best val (epoch) | final train / val | first epoch alpha < 2 |
|---|---|---|---|---|
| MLP3 h = 512, Adam, clean (40 ep) | 2.14 / 1.88 / 1.80 / 1.89 / 2.00 | 0.905 (26) | 0.980 / 0.904 | 6 |
| MLP3 h = 512, Adam, 30 % label noise | 2.28 / 1.98 / 2.24 / 2.38 / 3.04 | 0.882 (13) | 0.794 / 0.780 | 8 |
| MLP3 h = 1024, SGD 0.05, clean | 3.02 / 2.59 / 2.39 / 2.19 / 2.00 | 0.904 (30) | 0.973 / 0.901 | 35 |
| SNNMLP3 h = 512, Adam (20 ep) | 1.79 at best epoch, 1.79 final | 0.897 (18) | 0.934 / 0.888 | 4 |
| L3Classifier 2 blocks (12 ep) | 3.27 at best epoch, 3.62 final | 0.846 (3) | 0.900 / 0.833 | never |
| SNNL3 rate_query (6 ep, 10 k) | 2.99 final | 0.800 (6) | 0.859 / 0.800 | never |
| SNNL3 spiking_memory (1 ep, budget) | 5.17 | 0.643 | 0.649 / 0.643 | never |

With Adam the fit's `xmin` drifts from 0.4 to 21 during memorisation (the power law ends up fitted on the top
67 of 512 eigenvalues), which is why alpha climbs back; with SGD `xmin` stays small and alpha approaches 2
from above.

## 5. Reading these results in HTSR terms

* The "ideal alpha = 2" statement is reproduced for the dense layers of good models trained with SGD or
  moderately with Adam, including the mixing layers of L3 and of spiking L3.
* Alpha < 2 is a signature of an optimiser regime (Adam, many epochs, small batches) rather than of bad
  accuracy; memorisation of clean data first pushes alpha below 2 and then back up, and memorisation of noisy
  labels pushes it up, so the mapping from alpha to "over-fit" is not monotone. Alpha alone cannot separate
  "a well-trained model at its best" from "a model that has started to memorise".
* Lookup tables and embeddings are not in the heavy-tailed phase at this scale: they are a random bulk plus
  frequency-driven fingers. Their alpha should not be averaged with dense layers, but in the L3 sweep it is
  the lookup alpha, not the dense alpha, that still moves with accuracy, because the tables are the last part
  of the model to leave the random phase.
* Spiking layers reach the same alpha as their ANN twins; what spiking changes is the alignment of the top
  singular vectors with the mean firing pattern (a feature-mean, data-like effect), the dependence of alpha on
  the surrogate slope and on silent units, and the fact that a dead network can carry healthy-looking alphas.
* Our earlier Hessian study showed that the surrogate-gradient update of a spiking network is not a gradient
  flow (block-triangular Jacobian, non-vanishing functional term). The weight spectra nevertheless look like
  those of ordinary networks, so whatever drives the heavy tails of W (noise plus accumulated correlations)
  does not need a potential; the RG reading of alpha = 2 as a steady state is about the weight statistics, and
  those statistics are reached by spiking networks too.

## 6. Caveats

CPU-only budget: 18-122 runs per family instead of 1000, 1-20 epochs, widths up to 1024 for the MLP and 256
for L3; spiking L3 trained on 2-10 k-image subsets for 1-4 epochs, so many spiking L3 runs are under-trained
(which is what makes their alpha-accuracy correlation strong); single seed per configuration; the L3 and MLP
sweeps were cut by the time budget at 35 / 40 and 122 / 160 runs; the spiking-memory trajectory got one epoch.
WeightWatcher's own caveats apply (fits with fewer than 50 eigenvalues excluded; `xmin` chosen by KS
minimisation jumps between epochs; `fix_fingers` changes lookup-table alphas by up to 3x).

## 7. File map

| path | content |
|---|---|
| `results/mlp_sweep`, `results/l3_sweep`, `results/snn_sweep`, `results/snn_sweep2` | one JSON per run (config, history, WeightWatcher rows and summary, alpha trajectory) |
| `results/trajectories/runs` | 7 long runs with per-epoch WeightWatcher rows |
| `results/controls/runs`, `results/controls/summary.md` | control experiments (saved models in `results/controls/*.pt`, not committed) |
| `figures/final/` | scatter plots, per-role plots, histograms, `summary.md`, `by_model.md`, `runs_table.csv`, `correlations.csv` |
| `figures/trajectories_alpha.png` | alpha and accuracy trajectories |
