# Spiking inserts in a pretrained vision-language model: can HTSR alpha certify, select and tune them?

Status: experiments running; results sections are filled in as they complete (see the bottom for the log).

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
| E2a | single-block conversion damage (T = 4/8/16, ReLU and clipped-ReLU ablations) vs per-block alpha | pending |
| E1 | fine-tuning with an insert: spike_replace (k_sg 1/4/10/25, T 4/8/16, leak, seed), dense_replace, spike/dense adapter, block 10, later-blocks scope, 20 % label noise | pending |
| E2b | cumulative conversion in six orders, 3/6/9/12 blocks, conversion-only and after a brief fine-tune | pending |
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

(pending)

## 7. Limitations

* CPU only: one pretrained model (ViT-B/32), small evaluation subsets (800 images per set), 3 000-image
  fine-tuning, mostly single seed. Differences below ~1 % accuracy on 800 images are within noise
  (binomial s.e. ~1.1 % at 90 % accuracy).
* The COCO "captions" are templated from box categories, not human captions.
* The paper (arXiv 2602.00545 study) and WeightWatcher's own CLIP notebooks could not be fetched.
