# Related work for the multimodal spiking-alpha study (and what is new here)

Searched 2026-10-21 (web search only; arXiv itself is not reachable from this environment, so papers were
read through search snippets and secondary pages — verify details before citing in a submission).

## Heavy-tailed self-regularization (HTSR) as a per-layer signal

* Martin & Mahoney, *Heavy-tailed self-regularization* (JMLR 2021) and *Predicting trends in the quality of
  state-of-the-art neural networks without access to training or testing data* (Nat. Commun. 2021): the
  power-law exponent alpha of a layer's ESD; alpha ~ 2 "ideal", alpha < 2 over-fit / over-trained, alpha > 6
  under-trained; the WeightWatcher tool.
* **TempBalance** (Zhou, Pang, Liu, Martin, Mahoney, Yang; NeurIPS 2023 spotlight, arXiv 2312.00359):
  layer-wise learning rates set from each layer's HT-SR alpha; CNNs on CIFAR/SVHN/TinyImageNet.
* **AlphaPruning** (Lu, Zhou, Liu, Wang, Mahoney, Yang; NeurIPS 2024, arXiv 2410.10912): per-layer pruning
  ratios of LLMs allocated from ESD shape metrics (alpha); LLaMA-7B to 80 % sparsity. This is the closest
  prior use of alpha as a *layer selection* signal; it selects how much to prune, not which layers to convert
  to a different compute model.
* Layerwise Learning Rate for Transformers (2026 preprint) and the Bielik 11B report use WW for monitoring.
* No published alpha profile of CLIP / ViT-B/32 per role (q/k/v/out/fc1/fc2) was found; E0 provides one.

## Spiking transformers and spiking CLIP

* **Spikformer** (Zhou et al., ICLR 2023, arXiv 2209.15425) and **Spike-driven Transformer** (Yao et al.,
  NeurIPS 2023, arXiv 2307.01694; V2/V3 follow-ups): spiking ViTs trained from scratch, 70-77 % ImageNet.
* **SpikeCLIP** (Li et al., Fudan, arXiv 2310.06488, Neural Networks 2024/25): a fully spiking CLIP trained by
  alignment pre-training + dual-loss fine-tuning. Our setting is the opposite: a *pretrained dense* CLIP
  with a *small* spiking insert.
* ANN-to-SNN conversion of transformers: SpikeZIP-TF (You et al. 2024; replaces GELU/softmax/LN, fine-tunes),
  STA (Jiang et al. ICLR 2024; spatio-temporal approximation, training-free, higher latency), the
  MBE-neuron training-free framework (arXiv 2508.07710), *Towards high-performance spiking transformers from
  ANN-to-SNN conversion* (arXiv 2502.21193), *Efficient ANN-SNN conversion with error compensation learning*
  (ICML 2025; learnable clipping thresholds, dual-threshold neurons, membrane initialisation; CNNs).
  They all convert the *whole* network and attribute the loss to clipping, quantisation and non-uniform
  activations; none asks which blocks tolerate conversion or whether a weight-spectrum statistic predicts it.
* Hybrid ANN-SNN layer selection: no paper found that chooses the subset of transformer blocks to convert;
  the common heuristic is "keep softmax/LayerNorm dense, convert MLPs".

## Parameter-efficient spiking inserts

* SpikePEFT (point clouds, 2026 preprint) tunes LIF decay/threshold of a *spiking* backbone; test-time LoRA on
  spiking decoders; a "SpikeLoRA" appears only as a project summary. No work found on inserting a spiking
  adapter into a frozen dense vision-language model, nor on using alpha to certify such an insert.
* CLIP-Adapter (Gao et al. 2021) and Tip-Adapter (Zhang et al., ECCV 2022): residual adapters on frozen CLIP
  features for few-shot classification; our E3 is a CLIP-Adapter with dense / spiking / lookup-memory bodies.

## What this study adds

1. A per-role alpha profile of a pretrained CLIP and its use to *predict* which blocks tolerate ANN-to-SNN
   conversion of their MLP (E2a/E2b), i.e. AlphaPruning's idea transferred to conversion ordering.
2. Alpha as a certificate for a spiking insert: whether the alpha trajectory of the inserted/converted block
   during fine-tuning tracks the recovery of zero-shot and retrieval metrics, and how surrogate slope, T and
   silent units move it (E1), on a multimodal model rather than a toy MLP.
3. A spiking L3 lookup memory as a CLIP adapter, against dense MLP / dense L3 adapters (E3).
