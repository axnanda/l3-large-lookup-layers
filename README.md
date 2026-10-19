# L3: Large Lookup Layers — corrected port, reference implementation, spiking variants and a WeightWatcher study

This repository started as a corrected PyTorch port of the L3 layer from "L3: Large Lookup Layers"
(Tseng and De Sa, arXiv 2601.21461); the bug-fix notes for the paper's listing are kept below.
It now also contains `l3lab`, a small research package built on top of it:

* `l3lab/l3_ref.py` — a clean reference L3 layer (padded per-token index table + masked softmax) that is
  numerically equivalent to the paper's blocked forward (tested to 1e-6), plus allocation and tokenizer helpers
  so L3 can be used on images (k-means patch tokens) and text.
* `l3lab/models.py`, `l3lab/snn.py` — `MLP3`, `L3Classifier`, `L3CharLM` and spiking (LIF, surrogate-gradient)
  variants `SNNMLP3` and `SNNL3Classifier` with two spiking lookup designs (`rate_query`, `spiking_memory`).
* `l3lab/ww_analysis.py`, `l3lab/train.py`, `scripts/` — WeightWatcher harness (per-layer alpha and extra spectral
  diagnostics logged during training), random sweeps, trajectory and control experiments.
* Documents: `docs/CRITICAL_DESIGN.md` (what is strong and fragile in L3, how a spiking L3 can be built),
  `docs/SNN_L3_DESIGN.md`, `docs/L3_REF_NOTES.md`, `docs/DESIGN_SPEC.md`, and the study report
  `docs/WEIGHTWATCHER_STUDY.md` (alpha vs accuracy for MLP3, L3 and spiking models on Fashion-MNIST).

```
pip install -r requirements.txt            # torch (CPU is fine), weightwatcher, scikit-learn, ...
python3 -m pytest tests -q                 # 41 tests
python3 scripts/run_mlp_sweep.py --smoke --out results/smoke   # then drop --smoke for the real sweep
```
Fashion-MNIST idx files are expected under `data/fashion/` (the repo of zalandoresearch/fashion-mnist has them).

---

# Corrections to L3 Layer Implementation from the Paper

The originally published PyTorch implementation in the "L3: Large Lookup Layers" paper contained multiple syntactical and mathematical bugs affecting both `l3_layer.py` and the example allocation loop. We have corrected these issues in our provided repository codebase.

Here is a summary of all 6 bugs identified and fixed.

## Functional & Dimensionality Bugs

### 1. Dimensionality Mismatch in Slicing (l3_layer.py)
**Bug #1:** `A = self.norm_in(input[fw])` and `A.reshape(-1, d)[fw]`
**Why it failed:** The authors try to slice the 3D tensor `input` (shape `[batch, sequence, hidden_dim]`) using a flattened 1D sorting index array `fw` (length `batch * sequence`). This immediately duplicated the sequence dimension into `[batch*sequence]`, turning `A` into an impossibly large `[batch*sequence, sequence, hidden_dim]` array. They then tried to reshape and index it *again* using `[fw]`.
**Our Fix (Line 107):** Applied `norm_in` to `input` directly, flattened to 2D *first*, indexed by `[fw]`, and reshaped.
```python
A = self.norm_in(input).reshape(-1, d)[fw].reshape(-1, bb, d)
```

### 2. LayerNorm Shape Broadcasting Error (l3_layer.py)
**Bug #2:** `self.norm_out = LlamaRMSNorm(h)`
**Why it failed:** The formula $LayerNorm(W_{up}(...))$ dictates that the output of $W_{up}$ must be passed into `norm_out`. $W_{up}$ scales the embeddings up to the `d_up` dimension. Passing a tensor of `[..., d_up]` into a LayerNorm dimensioned as `h` causes a fatal PyTorch broadcasting mismatch.
**Our Fix (Line 60):** Initialize `norm_out` using the properly scaled dimension `d_up`.
```python
self.norm_out = LlamaRMSNorm(d_up)
```

### 3. Improper LZW Target Mapping (train.py script)
**Bug #3:** `torch.tensor(alloc)`
**Why it failed:** `train_lzw` produces an array of *counts* representing how many embeddings each token is allocated. However, `valid_collate_fn` requires a 1-to-1 array map denoting exactly which embedding ID corresponds to which token ID. 
**Our Fix:** Using a loop to explicitly expand the dictionary counts into a 1-to-1 index mapping array of size $v$.

---

## ⚠️ `forward_flexattn` Specific Bugs

The file included an alternative hardware-optimized implementation block called `forward_flexattn` that harbored identical logic bugs.

### 4. 3D Sequence Slicing in flex_attention (l3_layer.py)
**Bug #4:** `A = self.norm_in(input[fw]).gather(...)`
**Why it failed:** Same as Bug #1. `input[fw]` crashes PyTorch identically by treating `fw` as indexing `input` directly against a 3D tensor structure.
**Our Fix (Line 134):**
```python
A = self.norm_in(input).reshape(-1, d)[fw].reshape(b, t, d).unsqueeze(1)
```

### 5. Out of bounds emb_alloc Assignment (l3_layer.py)
**Bug #5:** `emb_alloc = emb_alloc[emb_alloc]`
**Why it failed:** Slicing `emb_alloc` by `emb_alloc` itself arrays out-of-bounds indices trivially. It's semantically invalid since `emb_alloc` maps embedding length $v$ to subset tokens, while the actual iterator masks rely on dynamically computed blocks `keep_cols`.
**Our Fix (Line 127):** 
```python
emb_alloc = emb_alloc[keep_cols]
```

---

## Minor Typos & Syntax

**Bug #6: Missing Parameter in train_lzw**
The paper's function signature was `def train_lzw(files, tok, k):` but the embedded `while` loop evaluated against a variable named `target` that was completely undefined. 
- **Fixed (Line 24):** Added `target` to the method signature `def train_lzw(files, tok, k, target):`.

**Bug #7: Undefined Variable inside Concat**
Both implementations of `forward` and `forward_flexattn` attempt to return using `torch.concat([... , inputs])`. `inputs` is undefined—the actual tensor param is named `input`.
- **Fixed (Lines 120 and 140):** Changed to `input`.