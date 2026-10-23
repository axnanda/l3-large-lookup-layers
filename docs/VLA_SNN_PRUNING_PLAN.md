# Plan: alpha-allocated SNN pruning of a VLA with a spectrum-aware optimizer (CVPR 2027 target)

Written 2026-10-23. Deadline assumed 2026-11-10 (18 days). Builds on docs/MULTIMODAL_STUDY.md (CLIP study)
and docs/WEIGHTWATCHER_STUDY.md (small-model study).

## 1. What AlphaPruning does, and what our data say about the sign

AlphaPruning (Lu et al., NeurIPS 2024, arXiv 2410.10912) fits the power-law exponent alpha of each layer's
weight spectrum (PL_Alpha_Hill) and assigns *lower* sparsity to layers with heavier tails (small alpha, "more
learned signal") and *higher* sparsity to light-tailed layers (large alpha). The per-layer sparsity is a
linear map of alpha between a minimum and a maximum, rescaled so the mean sparsity hits the global target.

Our CLIP study measured the same quantity for activation-level spiking conversion instead of weight pruning:

* single-block conversion damage is anti-correlated with alpha (Spearman -0.75 without bias correction,
  -0.59 with it) - low-alpha blocks are the ones that break;
* converting "closest to 2 first" is the worst order at every fraction; "farthest from 2 first" is within
  1-2 points of the best; damage compounds super-additively when low-alpha (early) blocks are converted.

So the correct spiking analogue of AlphaPruning is: **allocate the spiking budget by alpha in AlphaPruning's
direction** (high alpha -> aggressive conversion: few time steps, high threshold, converted first; low alpha
-> gentle or dense). The earlier hypothesis H2 ("near-2 first") had the sign backwards; the data and the
AlphaPruning rule agree with each other.

Second thing the CLIP study settled: alpha of a pretrained matrix is **inert** under the fine-tuning that
repairs a converted block (moves < 0.05 while accuracy recovers up to 9.5 points), and the surrogate slope
does not move it either. Therefore alpha belongs in the optimizer as a *static per-layer allocation signal*
(budgets, learning-rate temperatures), not as a training-time feedback signal. The training-time signals
that do carry information are the distillation loss and the firing statistics (silent / saturated fraction).

## 2. What "SNN pruning" means concretely (three forms; we do the first two)

1. **Activation conversion with an alpha-allocated budget.** Replace the MLP activation (and optionally the
   attention value path) of block i by a rate-coded IF population with time steps T_i and per-channel
   thresholds theta_i; T_i and the threshold percentile come from alpha_i through the AlphaPruning map.
   Compute is counted in synaptic operations (one accumulate per spike per fan-out) instead of MACs; the
   "pruning" is the dynamic activation sparsity of the spikes (CLIP blocks: 3-17 % of channels active).
2. **Alpha-allocated weight sparsity on top** (AlphaPruning proper, Wanda/SparseGPT scoring), so the paper
   has a direct comparison: same alpha allocation, weights vs activations vs both.
3. **A small spiking gate ("spiking pruning layer")**: a LIF population of width = number of tokens (or
   channels) whose 0/1 spikes mask tokens for the next block (DynamicViT with a spiking predictor), with the
   keep-rate per block set by alpha. This is the most literal reading of "small spiking pruning layer" and is
   the natural extension if time permits; it is cheap (tiny layer) and gives token-level pruning.

Required fixes from the CLIP study, carried over: per-channel thresholds (massive-activation channels) and
per-channel bias correction for the GELU negative lobe (without it a single block costs 20-38 points; with
it 0-2). VLA backbones (SigLIP / DINOv2 vision towers, Llama / SmolLM language model, GELU/SiLU MLPs) have
the same structure, so both fixes apply; SiLU needs the same mean-offset correction.

## 3. The optimizer (the paper's main contribution)

Working name: **SpikeBalance** - a layer-wise, spectrum-allocated, homeostatic optimizer for converting a
pretrained transformer into a hybrid SNN with minimal task loss. Components, each tied to evidence:

| component | rule | why (evidence) |
|---|---|---|
| A. budget allocation | T_i, threshold percentile p_i, conversion order from alpha_i via the AlphaPruning linear map; mean budget hits a global SOP target | E2a/E2b sign; AlphaPruning |
| B. layer temperatures | per-layer learning rate lr_i = lr * f(alpha_i), larger for high-alpha (light-tailed, under-trained) layers, smaller for low-alpha ones (TempBalance rule) | protects the signal-carrying layers during repair; alpha is static so compute once |
| C. objective | distillation to the dense model's hidden states and action logits (1 - cos on block outputs + KL on action tokens), optional task loss with small weight | E1: distillation repairs in one epoch, task-CE does not |
| D. homeostatic thresholds | thresholds and gains are parameters in their own group; a per-channel homeostatic term lowers theta of channels whose firing rate falls below r_min and raises it above r_max (target rate from the dense activation statistics) | dead-unit growth was the failure signature in the small-model study; firing fraction is the signal that moves |
| E. bias-correction refresh | the per-channel offset b is re-estimated from a running mean of GELU(x) - rate-code(x) every k steps (a moving-average optimizer state, not a parameter) | the dominant error term; keeps conversion unbiased as weights drift |
| F. surrogate schedule | surrogate slope fixed (k = 4); no annealing | slope had no effect on repair or alpha in E1; one less knob |
| G. schedule | convert blocks progressively in the alpha order with sequential calibration; repair each stage for a fixed step budget | E2b: sequential calibration is sub-additive for late blocks |

Baselines the optimizer must beat: uniform T and threshold with plain Adam + distillation (ours without A, B,
D, E); AlphaPruning weight sparsity at matched SOPs; SpikeZIP-TF-style full conversion + fine-tune;
depth-based allocation (late blocks first) to show alpha adds something beyond depth.

Ablations: drop A / B / D / E one at a time; alpha-reversed allocation (near-2 first) as the negative control.

## 4. Target model and evaluation

VLA candidates (need a GPU and Hugging Face access; neither is available in this container):

* **SmolVLA** (Hugging Face / LeRobot, ~450 M, SmolVLM backbone + flow-matching action expert): smallest
  real VLA, LIBERO evaluation supported in LeRobot, fits a single 24 GB GPU for fine-tuning.
* **OpenVLA-7B / OpenVLA-OFT** (LIBERO fine-tuned checkpoints released): the reference VLA, 7 B parameters;
  inference in bf16 on a 24 GB GPU, repair fine-tuning needs LoRA or a larger GPU.
* **pi0 / pi0-FAST** (openpi, 3 B): strong but heavier to evaluate.

Recommendation: SmolVLA as the main model (all experiments and ablations), OpenVLA-OFT as the scale check
(allocation + optimizer applied once, few seeds). Benchmark: LIBERO (four suites, success rate over 500
episodes per suite), plus the SimplerEnv Bridge tasks if time permits. Report success rate vs. spiking
fraction / SOP reduction / estimated energy (0.9 pJ per accumulate vs 4.6 pJ per MAC, the standard SNN
convention), alpha profile of the backbone, firing fractions, and the dense model as the ceiling.

Proxy that runs here (CPU, no HF): the CLIP ViT-B/32 harness in `l3lab/multimodal/` (caches, conversion,
distillation fine-tuning). The optimizer and the allocation can be implemented and ablated on it first
(zero-shot accuracy and retrieval as the task), then moved to the VLA unchanged.

## 5. Eighteen-day schedule

| days | what |
|---|---|
| 1-3 | implement SpikeBalance (A-G) and the SOP/energy accounting in l3lab; ablate A/B/D/E on CLIP (CPU, here); GPU machine: LeRobot + LIBERO + SmolVLA running, dense success rates reproduced |
| 4-6 | alpha profile of SmolVLA; cached hidden states; conversion + repair at 25/50/75/100 % of MLP blocks with the allocator; first success-rate curve |
| 7-10 | full grid: allocator (alpha / uniform / depth / reversed) x optimizer ablations; weight-sparsity comparison; seeds |
| 11-13 | OpenVLA-OFT scale check; spiking token gate if ahead of schedule |
| 14-17 | writing (CVPR template already in paper/), figures from the analysis scripts, limitations |
| 18 | buffer / submission |

## 6. Risks

* No GPU / no Hugging Face in this environment: the VLA half cannot run here. Needs a GPU box with internet
  (the user's machine or a cloud instance). The CPU proxy covers the optimizer ablations only.
* LIBERO evaluation cost: 4 suites x 500 episodes x ~50 steps x VLA forward; at 50 ms per forward on a GPU
  that is ~1.5 h per model variant; the grid in days 7-10 needs 2-3 GPUs or fewer episodes (200) for
  ablations and 500 for the headline rows.
* A hybrid SNN (spiking MLPs, dense attention / LayerNorm) is what the literature converts too, but reviewers
  will ask for the attention path; converting value/output projections is the stretch goal.
* "Minimal loss" must be quantified against the dense model with the same repair budget (a dense model
  fine-tuned by distillation for the same steps), not against the released checkpoint.

## 7. Statistic selection, data-driven (scripts/mm_select_statistics.py, results/mm/statistics_selection.md)

Every WeightWatcher column of every matrix in a block (alpha, alpha_weighted, log norms, spectral norm,
stable rank, MP soft rank, number of spikes, rand distance, KS distance D, xmin/xmax, entropy, effective
rank, top-singular-value energy, mean-direction overlap) plus the activation statistics (fraction of positive
pre-activations, MLP output share, pre-activation std, per-channel threshold median and max, silent fraction)
and depth were screened against the E2a damage targets on the 24 CLIP blocks, with leave-one-out validation
of 1- and 2-feature rank predictors. What came out:

| question | best predictor(s) (LOO Spearman) | reading |
|---|---|---|
| which block's conversion hurts the task (uncorrected) | depth +0.76; best weight stat fc1 alpha +0.41; depth + any weight stat +0.78-0.79 | position dominates; weight statistics add <= 0.03 over depth |
| which block's conversion hurts the task (bias-corrected, T = 4 / 16) | depth +0.51 / +0.53; mean MLP alpha +0.42 / +0.41; fc1 alpha + depth +0.52 / +0.58 | alpha is the best weight-only signal, in AlphaPruning's direction (low alpha = more damage); norm-type statistics (fc2 / attn_out / attn_v log norm, -0.56 to -0.65 pooled) are depth proxies through residual-stream growth |
| how large the block's own conversion error is | MLP output share +0.63; threshold max (massive-activation channels) +0.51; fc2 log spectral norm +0.42; share + fc2 spectral norm +0.69 | activation statistics, not weight spectra, predict the error magnitude |
| retrieval damage (COCO R@1) | silent fraction after conversion +0.51; mean-direction overlap of fc1 +0.51 | firing statistics are the monitor signal |

Implications for the optimizer and the statistics design:

1. **Allocation score = composite, not alpha alone.** Rank-average of depth, mean MLP alpha (AlphaPruning
   direction: higher alpha -> more aggressive budget), 1 - MLP output share, and the massive-activation
   ratio theta_max / theta_median (blocks with large ratios get a gentler budget). Alpha-only, depth-only,
   uniform and alpha-reversed are the ablations; the honest claim is "alpha is the best weight-only proxy
   and adds a little on top of depth", to be re-screened on the VLA backbone before the weights are fixed.
2. **Channel-level mixed conversion.** The massive-activation channels (theta ratio > ~10, a handful per
   block) are what a pooled threshold breaks; keep them dense and spike the rest. This is the literal "small
   spiking pruning layer": a per-channel mask chosen from the calibration statistics, with the dense
   fraction per block set by the allocation score.
3. **Calibration state in the optimizer**: per-channel threshold percentile, per-channel bias offset
   (mean GELU - rate-code error) and the output-share estimate are running statistics refreshed during
   repair, not parameters.
4. **Homeostasis on the silent fraction**: it is the one statistic that moves during training and predicts
   both classification and retrieval damage; the optimizer lowers thresholds of channels that go silent
   relative to their dense firing target.
5. **Layer temperatures from alpha** (TempBalance): alpha is static under repair (E1), so it is a safe
   per-layer learning-rate multiplier; norm-based temperatures would only re-encode depth.
6. Statistics that did **not** earn a place: alpha_weighted, KS distance D, xmin, MP soft rank, number of
   spikes, rand distance (inconsistent sign across towers), surrogate slope (no effect on repair or alpha).

Caveat: 24 blocks from one model; |rho| > 0.40 is the p < 0.05 line for the pooled ranks, so the composite
should be re-derived on the VLA backbone's own single-block screen (cheap: one forward per block on cached
hidden states) before the final allocation is fixed.
