# Spiking inserts in a pretrained vision-language model: can HTSR alpha certify, select and tune them?

Status: complete (E0, E1, E2a, E2b, E3; 2026-10-21).

## 1. Question

The hypothesis handed over from the Fashion-MNIST study (docs/WEIGHTWATCHER_STUDY.md) is that the HTSR /
WeightWatcher power-law exponent alpha of a layer can be used in three ways when a small spiking block is put
into an *existing* model:

* **H1 (certification).** After inserting a spiking block into a pretrained multimodal model and fine-tuning,
  the block's alpha moves toward the alpha of the dense layer it replaces (toward ~2 for well-trained layers),
  and its alpha trajectory plus firing fraction tracks whether the insert has learned, before and
  independently of the downstream metric.
* **H2 (selection / conversion order).** Converting dense layers to spiking in order of |alpha - 2| (closest
  first) retains more downstream accuracy at a given spiking fraction than other orders; layers with alpha
  far from 2 and lookup / embedding tables should stay dense.
* **H3 (performance).** A spiking lookup memory (spiking L3) added as an adapter adds useful capacity; the
  dense matrices around it settle at alpha 2.0-2.2.

## 2. Model, data, compute

* Model: open_clip `ViT-B-32-quickgelu` pretrained on LAION-400M (151 M parameters; 12 image blocks of width
  768 with MLP 768-3072-768 and QuickGELU, 12 text blocks of width 512). Weights were fetched from the
  open_clip GitHub release; Hugging Face is not reachable from this environment.
* Data (fast.ai S3 mirror; the usual dataset hosts are blocked): Imagenette-160 (10 ImageNet classes, 9 469
  train / 3 925 val), Oxford-IIIT Pets (37 classes, 3 680 trainval / 3 669 test), CIFAR-10 test, and the
  fast.ai COCO sample (21 837 images with boxes, no captions). Evaluation uses stratified 800-image subsets of
  Imagenette val, Pets test and CIFAR-10 test (zero-shot top-1 with 7 CLIP prompt templates) and 1 000 COCO
  images with pseudo-captions built from their object categories ("a photo of a person, a dog and a frisbee";
  73 distinct captions, image-to-text R@1 / R@5). `mean_acc` is the mean of the three zero-shot accuracies.
* Compute: 4 CPU cores, no GPU. All experiments run on cached block inputs (the input tokens of every image
  block for the evaluation sets; blocks 6, 8, 10 and 0 for Imagenette train), so a change at block k costs
  blocks k..11 only. Single seed unless stated.

Code: `l3lab/multimodal/` (`clipmodel.py` loader, WW layer map, spiking modules; `data.py`; `cache.py`;
`evalmm.py`; `finetune.py`) and `scripts/mm_*.py`; tests in `tests/test_multimodal.py`.

## 3. Spiking inserts

* **Replace-activation (conversion).** The QuickGELU of one block's MLP is replaced by a rate-coded
  integrate-and-fire population: constant input current I = pre-activation for T steps, soft reset, threshold
  theta from threshold balancing (99.9th percentile of the pre-activation on 256 calibration images), membrane
  initialised at theta/2, output = gain * mean spike rate with gain = theta per channel (learnable in
  fine-tuning). For T -> inf this equals clamp(I, 0, theta); the conversion error is the GELU-vs-ReLU shape
  error plus the 1/T quantisation. Backward pass: sigmoid surrogate with slope k_sg (straight-through spikes
  from `l3lab/snn.py`).
* **Spiking adapter (addition).** A bottleneck adapter W_up(SpikingAct(W_down(x))) with r = 64 added in
  parallel to a block's MLP, W_up zero-initialised, so the model is unchanged at insertion. Dense twin: the
  same adapter with QuickGELU.
* **Spiking L3 memory adapter (E3).** On the frozen 512-d image embedding: the embedding drives a LIF
  population whose spikes query a lookup memory (64 k-means tokens x 8 entries) in `spiking_memory` mode
  (threshold-selected entries) or `rate_query` mode; output mixed by W_mix (zero-initialised) and added to the
  embedding (CLIP-Adapter style). Dense controls: MLP adapter and dense L3 adapter with the same tables.

Fine-tuning objective (E1, E2b): text-anchored cross-entropy, logits = 100 cos(image embedding, class text
embedding) with the frozen zero-shot head on Imagenette train (3 000 images, 3 epochs, Adam). Only the
modified block's MLP (c_fc, c_proj, ln_2) or the adapter trains unless `scope=later`.

WeightWatcher: 0.7.7, power-law fits with `min_evals=50`, alpha reported per matrix; attention in_proj split
into q/k/v; embedding tables never averaged with dense layers.

## 4. Experiments

| id | what | status |
|---|---|---|
| E0 | alpha profile of the pretrained model per role and tower | done |
| E2a | single-block conversion damage (ReLU, clipped-ReLU + bias correction, spike + bias correction at T = 4/16) vs per-block alpha | done (96 configs) |
| E1 | fine-tuning with an insert: spike_replace (k_sg 1/4/10/25, T 4/8/16, leak, seed, lr, with/without bias correction), dense_replace, spike/dense adapter, block 10, later-blocks scope, CE objective | running (12/21 done) |
| E2b | cumulative conversion in six orders, 3/6/9/12 blocks, conversion-only | done (19 configs) |
| E3 | adapters on frozen embeddings: dense MLP, spiking MLP, dense L3, spiking L3 (memory / rate query); 16-shot and full Pets, 16-shot Imagenette; 3 seeds | done (45 runs) |

## 5. Results

### E0: alpha profile of the pretrained model (figures/mm/e0_alpha_profile.png)

148 matrices, WeightWatcher 0.7.7, 24 min on one core. No layer has alpha < 2 (minimum 2.24, image.b04.attn_q);
the model-wide mean is 6.3 and the median 5.5, i.e. by the HTSR reading the LAION-400M ViT-B/32 is
"under-trained" in most attention value/output projections and mid-to-late MLP output projections, and
"well-trained" (2 < alpha < 5) mainly in the MLP input projections (fc1), q/k projections of the first six
image blocks, and the last block's fc2 (alpha 3.56 with 459 spikes beyond the MP bulk).

| tower | role | mean alpha | min | max | n under-trained (alpha>6) |
|---|---|---|---|---|---|
| image | attn_k | 4.11 | 2.55 | 6.29 | 1/12 |
| image | attn_out | 7.36 | 4.40 | 12.59 | 6/12 |
| image | attn_q | 4.24 | 2.24 | 8.12 | 3/12 |
| image | attn_v | 8.20 | 3.89 | 18.35 | 7/12 |
| image | mlp_fc1 | 4.44 | 2.62 | 5.79 | 0/12 |
| image | mlp_fc2 | 5.81 | 3.31 | 8.82 | 5/12 |
| text | attn_k | 5.89 | 4.10 | 8.58 | 5/12 |
| text | attn_out | 7.64 | 3.40 | 15.87 | 7/12 |
| text | attn_q | 5.94 | 3.51 | 14.80 | 4/12 |
| text | attn_v | 9.76 | 3.41 | 25.00 | 7/12 |
| text | mlp_fc1 | 5.22 | 4.16 | 5.97 | 0/12 |
| text | mlp_fc2 | 7.25 | 4.33 | 10.31 | 7/12 |
| image | patch_embed | 5.71 | | | |
| image | proj_image | 6.58 | | | |
| text | token_embed | 5.93 | | | |
| text | proj_text | 8.54 | | | |

Per-block `mlp_fc1` alpha of the image tower rises almost monotonically with depth (2.62, 3.41, 4.68, 4.55,
4.22, 4.60, 4.64, 3.90, 4.79, 5.31, 5.79, 4.77 for blocks 0-11), so in this model "closest to 2" is confounded
with "early block"; the depth-order and random-order arms of E2b are there to separate the two.


### E2a: single-block conversion damage versus alpha (results/mm/e2a_summary.md, figures/mm/e2a_damage_vs_alpha.png)

96 configs: 24 blocks x {ReLU, clipped-ReLU + bias correction, spike + bias correction at T = 4 and 16}.
Clean model: mean_acc 0.9029, COCO R@1 0.224.

Two methodological results first, because they decide everything downstream.

* **Only 2-17 % of MLP pre-activations are positive** (image blocks 0.03-0.17, text blocks 0.02-0.08), so the
  MLP output of a CLIP block is dominated by the negative lobe of QuickGELU summed over thousands of channels.
  A ReLU-like rate code that discards it has a relative MLP output error of 0.7-1.3 and costs 21-38 points of
  zero-shot accuracy for a single early image block (block 3: -38 points), 13 points for text block 0.
* **Per-channel thresholds and a per-channel bias correction** (the calibration mean of GELU(x) - clipped
  ReLU(x), folded into the output offset) reduce the MLP error to 0.1-0.35 and the damage to 0-2 points per
  block (image mean 1.1 points at T = 4, 0.9 at T = 16 and in the clipped-ReLU limit; text mean 0.4 points).
  The spike quantisation itself (T = 4 vs the T -> inf limit) is worth 0.1-0.2 points. Pooled thresholds
  would clip the massive-activation channels (pre-activations up to 15 in image, 93 in text blocks, against a
  pooled 99.9th percentile of 0.6-2.3) and give relative errors near 1.

Does the pretrained alpha of a block predict its conversion damage?

| tower / variant | rho(drop, fc1 alpha) | rho(drop, fc2 alpha) | rho(drop, fc1 alpha_w) | rho(drop, frac. positive pre-act) | rho(drop, MLP share of block output) | mean drop |
|---|---|---|---|---|---|---|
| image, ReLU (no correction) | -0.75 (p 0.01) | -0.62 (p 0.03) | -0.90 (p < 0.01) | +0.72 | +0.55 | 0.134 |
| image, spike_bc T = 4 | -0.25 (p 0.43) | -0.59 (p 0.04) | -0.43 (p 0.16) | +0.42 | +0.12 | 0.011 |
| image, spike_bc T = 16 | -0.20 (p 0.53) | -0.53 (p 0.08) | -0.42 (p 0.17) | +0.34 | +0.02 | 0.009 |
| text, ReLU | -0.23 (p 0.46) | +0.13 | +0.35 | -0.48 | +0.19 | 0.027 |
| text, spike_bc T = 4 | -0.37 (p 0.23) | -0.28 | -0.04 | -0.49 | -0.08 | 0.004 |

* In the image tower the **direction is the opposite of H2**: the blocks whose MLP alpha is closest to 2
  (blocks 0, 1, 7: fc1 alpha 2.6, 3.4, 3.9) are the *most* damaged by conversion, strongly so without bias
  correction (Spearman -0.75 with fc1 alpha, -0.90 with alpha_weighted) and weakly after it (-0.59 with fc2
  alpha, p = 0.04; -0.25 with fc1 alpha, n.s.). "Well-trained" by the HTSR reading means "carries the
  signal", and the signal is what a lossy activation destroys. The relation is confounded with depth and with
  how much of the block's output the MLP contributes (share 0.7-0.8 in blocks 0-7, 0.3-0.4 in blocks 8-10);
  E2b separates the orders directly.
* In the text tower alpha does not predict damage (|rho| <= 0.44, all p > 0.15); the fc1 log spectral norm does
  weakly (+0.54 to +0.60, p ~ 0.05). Text block 0 is the fragile one (ReLU: -13 points).
* **Retrieval is more sensitive than classification and in different places.** With bias correction the
  zero-shot accuracy loses at most 2 points anywhere, but COCO R@1 loses 18 points when text block 0 is
  converted, 10 points for text block 1 and 6 points for image block 11 (the last block), against 1 point of
  mean_acc. Fine-grained text geometry (near-duplicate captions) lives in the first text blocks and the last
  image block.
* Silent units after conversion: 0.2-1.2 % of channels in image blocks 1-11 (37 % in block 0), 8-48 % in text
  blocks, i.e. the per-channel threshold leaves channels that are almost never positive silent.

### E1: alpha during fine-tuning of an insert (results/mm/e1_summary.md, figures/mm/e1_trajectories.png)

21 runs. Setting: image block 6 (or 10), insert, then 3 epochs of distillation fine-tuning on 2 000
Imagenette-train images (loss 1 - cos to the clean model's embedding), Adam 3e-5 unless stated, WW on the
trained matrices every 24 steps, metrics every epoch. Clean model 0.9029 / COCO R@1 0.224.

| run | insert | mean_acc at insertion -> after 3 epochs | fc1 alpha 0 -> end | fc2 alpha 0 -> end | insert-specific alpha | silent units |
|---|---|---|---|---|---|---|
| sr_b6_k4 / seed 1 | spike (T 8, k_sg 4, bias-corrected) | 0.894 -> 0.900 / 0.895 -> 0.900 | 4.64 -> 4.62 / 4.63 | 5.33 -> 5.31 / 5.32 | - | 4.1 / 4.0 % |
| sr_b6_k1 / k10 / k25 | k_sg 1 / 10 / 25 | 0.894 -> 0.900 / 0.899 / 0.902 | 4.64 -> 4.62 / 4.67 / 4.64 | 5.33 -> 5.31 / 5.31 / 5.30 | - | 4.4 / 4.1 / 4.1 % |
| sr_b6_T4 / T16 | T 4 / 16 | 0.892 -> 0.901 / 0.897 -> 0.899 | 4.64 -> 4.65 / 4.65 | 5.33 -> 5.31 / 5.31 | - | 4.1 % |
| sr_b6_leak | beta 0.8 | 0.887 -> 0.899 | 4.64 -> 4.59 | 5.33 -> 5.29 | - | 3.9 % |
| sr_b6_lr1e4 / lr3e4 | lr 1e-4 / 3e-4 | 0.894 -> 0.898 / 0.895 | 4.64 -> 4.66 / 4.66 | 5.33 -> 5.30 / 5.36 | - | 4.1 / 3.6 % |
| sr_b6_ce | text-anchored CE objective instead of distillation | 0.894 -> 0.894 (COCO R@1 0.216 -> 0.190) | 4.64 -> 4.65 | 5.33 -> 5.34 | - | 4.0 % |
| sr_b6_later | also train blocks 7-11 and the projection | 0.894 -> 0.897 | 4.64 -> 4.64 | 5.33 -> 5.32 | later blocks: mean abs. alpha change 0.15 | 4.2 % |
| sr_b6_nobc | no bias correction | 0.806 -> 0.893 | 4.64 -> 4.66 | 5.33 -> 5.28 | - | 3.8 % |
| sr_b6_nobc_lr1e4 / k1 / k25 | no bias correction, lr 1e-4 (k_sg 4 / 1 / 25) | 0.806 -> 0.899 / 0.900 / 0.902 | 4.64 -> 4.68 / 4.62 / 4.65 | 5.33 -> 5.11 / 5.11 / 5.03 | - | 4.0 / 5.0 / 3.6 % |
| sr_b10_k4 | spike in block 10 | 0.898 -> 0.898 | 5.79 -> 5.56 | 6.75 -> 6.71 | - | 1.2 % |
| dr_b6 / dr_b10 | dense twin (QuickGELU + gain), same training | 0.903 -> 0.902 / 0.903 | 4.64 -> 4.64 / 5.79 -> 5.78 | 5.33 -> 5.34 / 6.75 -> 6.79 | - | - |
| sa_b6 | spiking adapter r = 64, lr 1e-3 | 0.903 -> 0.903 | unchanged | unchanged | down 3.58 -> 5.56, up 0 -> 2.4 | 1.6 % |
| da_b6 | dense adapter r = 64, lr 1e-3 | 0.903 -> 0.903 | unchanged | unchanged | down 3.58 -> 5.00, up 0 -> 2.4 | - |

Findings.

* **The spectrum of a pretrained block is inert under repair fine-tuning.** Every bias-corrected spiking
  replacement recovers to within 0.1-0.4 points of the clean model and the uncorrected ones recover 8.7-9.5
  of their 9.7-point loss, while fc1 alpha of the converted block moves by at most 0.05 and fc2 alpha by at
  most 0.30 (the uncorrected, higher-LR runs; 0.02 otherwise) - the same range as the dense twin that had
  nothing to repair (0.00 / +0.01) and as block 10's dense control (-0.01 / +0.04). The repair is done by
  small weight changes that do not touch the power-law tail of a 3072 x 768 matrix trained for 32 epochs on
  400 M pairs. A within-run correlation between the block's alpha and the metric is therefore meaningless
  (it flips sign from seed to seed, -0.8 to +1.0, on alpha changes of 0.01). H1 in the "block alpha certifies
  the insert" form is **not supported** at this scale of fine-tuning.
* **Surrogate slope, time budget, leak and learning rate do not move alpha here.** In the small-MLP study fc1
  alpha rose monotonically with k_sg (1.60 -> 1.99 for k = 1 -> 10); in the pretrained block k_sg = 1, 4, 10,
  25 give fc1 alpha 4.62-4.67 and the same recovery (0.899-0.902), with or without bias correction. The
  slope matters when a layer is learned from scratch through the spikes, not when it is nudged.
* **What does carry the signal**: the distillation loss (0.009 at insertion with bias correction, 0.03
  without; both to ~0.001-0.01 within an epoch) and the firing statistics (silent fraction constant at
  3.6-5.0 % in block 6, 1.2 % in block 10, no saturated units); none of the runs shows the dead-unit growth
  that flagged failure in the small-MLP study.
* **The objective matters more than any spiking knob**: text-anchored CE on Imagenette (98.5 % zero-shot
  already) gives no gradient to repair with - mean_acc stays at 0.894 and COCO R@1 falls from 0.216 to 0.190
  - whereas distillation to the clean embedding repairs all three classification sets and retrieval.
* **Adapters under distillation are a null task**: the zero-initialised adapter has nothing to learn from a
  clean model, so mean_acc is unchanged and alpha of the random-initialised down-projection *rises* from
  3.58 to 5.0-5.6 (dense / spiking), the signature of an insert that is not needed. Adapters under a real
  learning objective are E3, where the down-projection alpha fell toward 2 as the adapter learned.

### E2b: cumulative conversion order (results/mm/e2b_summary.md, figures/mm/e2b_orders.png)

19 configs, conversion only (per-channel threshold balancing + bias correction, T = 8, sequential
calibration), no fine-tuning; clean model 0.9029, COCO R@1 0.224. Drops in points of mean zero-shot accuracy.

| order | 3 blocks | 6 blocks | 9 blocks | 12 blocks |
|---|---|---|---|---|
| alpha closest to 2 first (the H2 rule) | 8.0 (0, 11, 1) | 10.5 | 15.9 | |
| alpha farthest from 2 first | 2.0 (8, 9, 10) | 4.3 | 7.9 | 17.8 (all orders) |
| depth first | 9.3 (0, 1, 2) | 13.6 | 14.8 | |
| depth last | 1.3 (11, 10, 9) | 2.7 | 5.5 | |
| random 0 / random 1 | 2.9 / 1.4 | 5.1 / 4.2 | 8.8 / 12.0 | |
| sum of single-block drops (E2a) for the H2 order / its reverse | 3.8 / 2.2 | 6.8 / 4.5 | 9.1 / 7.5 | 11.3 |

* **H2 is contradicted at every fraction.** The "closest to 2 first" order is the worst or second-worst of
  the six orders at 3, 6 and 9 blocks (8.0 / 10.5 / 15.9 points), beaten only by depth-first at 3 and 6;
  its reverse (farthest from 2 first) is the second-best at every fraction, within 0.7-2.4 points of
  depth-last. In this model alpha near 2 marks the blocks that carry the signal and are the most expensive
  to convert; the two random orders fall in between and rank by how many early blocks they happen to contain.
* **Damage compounds in early blocks and is sub-additive in late ones.** For the early-block orders the
  cumulative drop is 1.5-2 x the sum of the E2a single-block drops (closest-to-2: 15.9 vs 9.1 at 9 blocks;
  depth-first 13.6 vs 6.9 at 6), for the late-block orders it is below the sum (depth-last 5.5 vs 6.5;
  farthest-from-2 7.9 vs 7.5), i.e. sequential calibration recovers part of the single-block loss when the
  converted blocks come late. Converting all twelve MLPs costs 17.8 points (0.903 -> 0.725; Imagenette
  0.942, Pets 0.624, CIFAR 0.609) without any fine-tuning.
* Retrieval does not follow the classification ranking: COCO R@1 after 3 late blocks (0.152-0.191) is as
  damaged as after 3 early blocks (0.123-0.140) and all orders end near 0.13-0.15 at 9 blocks, consistent
  with E2a's finding that the last image block matters most for R@1.
* Silent units: 5-13 % of channels whenever block 0 is among the converted blocks (it alone contributes
  37 %), 0.4-1.7 % otherwise.

### E3: adapters on frozen CLIP embeddings (results/mm/e3_summary.md, figures/mm/e3_adapters.png)

45 runs (5 adapters x 3 settings x 3 seeds, 40 epochs, Adam 1e-3, text-anchored CE on the adapted embedding).
Zero-shot references: Pets 0.834, Imagenette 0.985.

| setting | dense MLP | spiking MLP | dense L3 | spiking L3 (memory) | spiking L3 (rate query) |
|---|---|---|---|---|---|
| Pets 16-shot, test acc | 0.854 +- 0.011 | 0.842 +- 0.020 | 0.729 +- 0.002 | 0.744 +- 0.002 | 0.763 +- 0.010 |
| Pets 16-shot, transfer retained | 0.927 | 0.926 | 0.245 | 0.169 | 0.165 |
| Pets full, test acc | 0.867 +- 0.005 | 0.852 +- 0.005 | 0.809 +- 0.014 | 0.849 +- 0.004 | 0.848 +- 0.004 |
| Pets full, transfer retained / COCO R@1 | 0.875 / 0.204 | 0.886 / 0.087 | 0.179 / 0.049 | 0.447 / 0.149 | 0.404 / 0.128 |
| Imagenette 16-shot, test acc | 0.982 | 0.984 | 0.956 | 0.977 | 0.977 |
| end alpha, first matrix (adapter_down / W_up), Pets full | 1.85 | 2.09 | 5.34 | 2.75 | 2.74 |
| end alpha, output matrix (adapter_up / W_mix), Pets full | 1.35 | 1.41 | 1.51 | 1.54 | 1.55 |

Findings.

* A spiking MLP adapter matches the dense one on classification (within 1-1.5 points, overlapping seeds)
  and keeps transfer, but it costs image-to-text retrieval (COCO R@1 0.087 vs 0.204 in the full-shot
  setting): the rate-coded bottleneck is fine for the class decision and bad for the fine-grained geometry
  the retrieval head needs.
* The L3 lookup adapters as designed are harmful: routing by a k-means token of the embedding applies a
  training-set-specific offset to every input, so transfer collapses (0.17-0.25 of the zero-shot accuracy on
  the other sets) and the 16-shot runs end *below* zero-shot. The spiking memory version is consistently
  better than the dense L3 (full-shot 0.849 vs 0.809, transfer 0.45 vs 0.18): the threshold-selected,
  normalised memory read-out is a weaker, sparser perturbation than the softmax-weighted dense read-out.
  (Memory firing fraction 0.17-0.35.)
* Alpha. The first matrix of a well-working adapter converges to alpha ~2 (dense 1.85, spiking 2.09 with all
  Pets data) and stays at 2.7-2.9 with 16 shots and 4.1-4.5 on the easy Imagenette task where little is
  learned; alpha of the first matrix correlates with the accuracy *gain* over zero-shot at Spearman -0.73
  (p = 0.001, n = 18) for the MLP adapters, i.e. lower alpha = more learned. The output matrices sit at
  alpha 1.35-1.7 in every run, below the "ideal" 2, and across all 45 runs the output-matrix alpha correlates
  with transfer retained at Spearman -0.78 (p < 1e-4): the adapters whose output matrix became most
  heavy-tailed are the ones that over-specialised. For the H3 prediction "W_up / W_mix settle at 2.0-2.2":
  W_up of the spiking L3 lands at 2.4-2.9 (dense L3: 4-5), W_mix at 1.5-1.7; the direction (spiking mixing
  layers closer to 2 than dense ones) holds, the number does not.

## 6. Verdicts on H1-H3

| hypothesis | verdict | evidence |
|---|---|---|
| H1 certification: the insert's alpha converges toward the dense layer's alpha and its trajectory tracks recovery | **not supported** for converted pretrained blocks; **partly supported** for adapters that actually learn | E1: 21 runs, fc1 alpha of the converted block moves <= 0.05 while accuracy recovers 0.5-9.5 points; dense twin moves the same; no dependence on k_sg, T, leak, lr. E3: the first matrix of a learning adapter converges to alpha ~2 (1.85 dense, 2.09 spiking) and its alpha correlates with the gain over zero-shot at Spearman -0.73; an unneeded insert's alpha *rises* (E1 adapters, 3.6 -> 5.0-5.6). Alpha therefore reads "how much a fresh matrix has learned", not "whether a pretrained matrix has been repaired". |
| H2 selection: convert the blocks closest to alpha 2 first; keep lookup tables dense | **contradicted** on the ordering; **supported** on the tables | E2a: damage is anti-correlated with alpha in the image tower (Spearman -0.75 without, -0.59 with bias correction); E2b: the closest-to-2 order is the worst or second-worst at 3/6/9 blocks (8.0/10.5/15.9 points), its reverse is second-best (2.0/4.3/7.9); depth-last is best. The practical rule for this model is "convert late blocks first", and alpha only helps because it is anti-correlated with depth. E3: lookup tables (L3 keys/values) end at alpha 2.7-14 depending on the data and never predict anything. |
| H3 performance: a spiking lookup memory adds capacity; its mixing layers settle at alpha 2.0-2.2 | **not supported** as a performance claim; **directionally supported** on alpha | E3: spiking L3 memory < spiking MLP adapter on Pets (0.849 vs 0.852 full-shot, 0.744 vs 0.842 16-shot) and it destroys transfer (0.45 vs 0.89); it does beat the dense L3 memory (0.809, transfer 0.18). W_up alpha 2.4-2.9 (spiking) vs 4-5 (dense); W_mix 1.5-1.7 in both. |

What held up from the small-model study: a healthy spiking layer reaches the same alpha as its dense twin
(E3 MLP adapters 2.09 vs 1.85; E1 converted blocks 4.62-4.68 vs 4.64) and the output-matrix alpha below 2
flags over-specialisation (E3: Spearman -0.78 with transfer retained). What did not: the surrogate-slope
dependence of alpha and the "alpha tracks recovery" trajectory, both of which require the layer to be
learned through the spikes rather than nudged.

What the study adds that was not hypothesised: (i) CLIP MLPs run almost entirely in the negative lobe of
QuickGELU (2-17 % positive pre-activations), so any ReLU-like rate code needs a per-channel bias correction;
with it and per-channel thresholds a single block converts at a cost of 0-2 points and the full image
tower's MLPs at 17.8 points without fine-tuning; (ii) conversion damage compounds super-additively when early
blocks are converted and sub-additively when late ones are; (iii) retrieval and classification are damaged
by different blocks (first text blocks and the last image block for R@1); (iv) distillation to the clean
embedding repairs a converted block in one epoch, a task-CE objective does not.

## 7. Limitations

* CPU only: one pretrained model (ViT-B/32), small evaluation subsets (800 images per set), 3 000-image
  fine-tuning, mostly single seed. Differences below ~1 % accuracy on 800 images are within noise
  (binomial s.e. ~1.1 % at 90 % accuracy).
* The COCO "captions" are templated from box categories, not human captions.
* arXiv and WeightWatcher's own site could not be fetched from this environment; related work was read through search snippets (docs/MM_RELATED_WORK.md).
* E2b is conversion-only; the brief-fine-tune arm was not run for lack of time (the script supports it with `--finetune_epochs`).
