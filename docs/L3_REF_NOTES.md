# Notes on the paper's blocked L3 forward vs. `l3lab.l3_ref.L3Ref`

**Result of the equivalence check: no numerical bug was found in the user's corrected blocked forward.**
`L3Ref` (plain gather + per-token masked softmax over that token's allocated embeddings) matches
`l3_layer.L3.forward` called through `l3_layer.valid_collate_fn` to a max abs diff of ~1e-6 in float32
(`tests/test_l3_ref.py`, V=20, n_emb=60, k=5, h=16, d_emb=8, d_up=12, b=4, t=8, for bb ∈ {32, 16, 8},
also for degenerate batches: all-same token and all-distinct tokens; parameter gradients agree to 1e-4).

Why the blocked forward is right: `valid_collate_fn` sorts the b·t positions by token id, concatenates the
embedding ranges `[bounds[v], bounds[v+1])` of every token present (`keep_cols`), and records for every sorted
position the half-open range `[starts, ends)` of its token inside `keep_cols`. `forward` cuts the sorted
rows into blocks of `bb`, and for a block uses the union range `[min starts, max ends)` of its rows; the
`torch.where(seq_sort == last_token)` mask then removes the columns that belong to other tokens of the
block, so every row sees exactly its own token's embeddings, regardless of how blocks fall.

## Things to know (not bugs in the math, but easy to trip over)

1. **`bb` must divide `b·t`** (`A.reshape(-1, bb, d)`, `seq_sort.reshape(-1, bb)`); the default `bb=512`
   therefore fails for small batches. `L3Ref` has no such constraint.
2. **`out[i] = ...` in `mask_logits_gemm`** writes into a pre-allocated zeros tensor. Autograd handles it
   (CopySlices), gradients match `L3Ref`.
3. **`torch.compile(mode='max-autotune', fullgraph=True)`** on `mask_logits_gemm_` compiled the method at
   import time on every platform. It is now optional: set `L3_COMPILE=1` in the environment (checked at
   import) or `l3_layer.L3_COMPILE = True` before constructing an `L3`. This is the only edit to `l3_layer.py`.
4. **`forward_flexattn` is broken as written** (left untouched, it is not used by any test):
   `from torch.nn.attention.flex_attention import flex_attention` binds the *function*, so
   `flex_attention.create_block_mask` and `flex_attention.flex_attention` raise `AttributeError`
   (`create_block_mask` lives in the module `torch.nn.attention.flex_attention`). Also `mask_fn` indexes
   `seq_sort[b, q_idx]` although `seq_sort` is 1-D there.
5. **`train_lzw` edge cases** (the generic re-implementation `l3lab.allocation.train_lzw_generic`
   handles them, see its docstring): (a) when a line ends inside a phrase that is already in the
   dictionary, `lzw_counter[phrase] = 1` *resets* that phrase's count instead of incrementing it; (b) a
   trailing single token is never counted (`while cur < len(toks)` exits first); (c) if the sorted phrase
   list is exhausted before `target` embeddings are placed, the loop raises `IndexError`; (d) the paper's
   scan resumes *after* the newly inserted phrase, i.e. it is not textbook LZW (this one is kept as is).
6. `valid_collate_fn` builds `keep_cols/starts/ends` with python loops over the unique tokens — fine for
   tests, slow for big batches; `L3Ref` needs only `alloc.table[token_ids]`.

## Minimal repro of the equivalence

```python
import torch, l3_layer
from l3lab.l3_ref import L3Ref, AllocTable, paper_inputs
from l3lab.allocation import allocate_by_frequency
torch.manual_seed(0)
alloc = AllocTable.from_counts(allocate_by_frequency(torch.randint(1, 100, (20,)).tolist(), 60, 5))
paper = l3_layer.L3(16, 60, 8, 12); ref = L3Ref.load_from_paper_layer(paper)
x, seqs = torch.randn(4, 8, 16), torch.randint(0, 20, (4, 8))
inputs = paper_inputs(seqs, alloc)              # -> valid_collate_fn([(seq_i, emb_alloc, bounds) ...])
print((paper(x, *inputs[1:], bb=32) - ref(x, seqs, alloc)).abs().max())   # ~1e-6
```
