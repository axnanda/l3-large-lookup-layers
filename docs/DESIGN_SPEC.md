# l3lab: design spec (implementation contract for the two build agents)

Repository: /home/user/l3-large-lookup-layers (user's corrected PyTorch port of the L3 layer from
"L3: Large Lookup Layers", Chris De Sa's group). Existing files: README.md (bug-fix notes), l3_layer.py
(the L3 module with the paper's blocked forward + LZW allocation + collate). Keep them; do not change their
public behaviour except: make the `@torch.compile(...)` decorator on `L3.mask_logits_gemm_` OPTIONAL
(enabled only when env L3_COMPILE=1 or a module flag is set), because torch.compile max-autotune is
unavailable/slow on CPU and breaks tests.

Goal of the package: (1) a clean reference L3 layer that is numerically equivalent to the paper's blocked
implementation, (2) L3-based classifiers for Fashion-MNIST (patch-token input) and a char-LM for text,
(3) spiking (LIF, surrogate-gradient) variants, (4) a WeightWatcher (pip `weightwatcher` 0.7.7, already
installed) analysis harness that logs per-layer power-law exponents alpha etc. during training, and
(5) sweep scripts that train many variations and record alpha vs train/validation accuracy, in the style
of Charles Martin's "1000 variations of a 3-layer MLP on Fashion-MNIST" post.

Environment: CPU only (4 cores shared by several processes: ALWAYS torch.set_num_threads(1) in scripts),
no internet except PyPI; torch 2.14 installed; numpy, scipy, sklearn, matplotlib, pandas, weightwatcher
installed. Data on disk: Fashion-MNIST idx-gz files in
/tmp/claude-0/-home-user-tvm-ffi/cf719c09-d7e5-5f6b-bcb9-86a98cd733c6/scratchpad/ext/fashion-mnist/data/fashion/
(train-images-idx3-ubyte.gz etc.), tiny Shakespeare at
/tmp/claude-0/-home-user-tvm-ffi/cf719c09-d7e5-5f6b-bcb9-86a98cd733c6/scratchpad/ext/text/tinyshakespeare.txt.
Copy both into /home/user/l3-large-lookup-layers/data/ (data/ is git-ignored except a README).

## How the L3 layer works (from l3_layer.py)
Parameters: w_k: Linear(h -> n_emb) whose weight rows are n_emb keys; w_v: Linear(d_emb -> n_emb) whose
weight rows are n_emb values (d_emb each); w_up: Linear(d_emb -> d_up); w_mix: Linear(d_up + h -> h);
norm_in = RMSNorm(h); norm_out = RMSNorm(d_up). Each vocabulary token v owns a contiguous range of
embedding ids [bounds[v], bounds[v+1]) (allocation: at least 1 per token, at most k, more for frequent
tokens, total n_emb). Forward for a hidden state x (b,t,h) at a position whose token id is v:
  A = norm_in(x); scores = A . w_k.weight[ids(v)]^T ; p = softmax(scores over ids(v)); comb = p . w_v.weight[ids(v)]
  out = w_mix(concat[norm_out(w_up(comb)), x])     (shape h; the caller adds a residual if desired)
The paper's forward computes this with sorting + block-diagonal masked GEMMs (bb blocks) for speed.

## Package layout (all under /home/user/l3-large-lookup-layers)
l3lab/__init__.py
l3lab/l3_ref.py        [Agent A]  L3Ref module + AllocTable
l3lab/allocation.py    [Agent A]  allocate_by_frequency, train_lzw_generic
l3lab/tokenizers.py    [Agent A]  PatchVQTokenizer, CharTokenizer
l3lab/snn.py           [Agent A]  LIF primitives (straight-through surrogate spikes)
l3lab/models.py        [Agent A]  MLP3, SNNMLP3, L3Classifier, SNNL3Classifier, L3CharLM
l3lab/data.py          [Agent B]  load_fashion_mnist, load_tinyshakespeare, patchify
l3lab/train.py         [Agent B]  train_model(...) loop with checkpoints + eval + WW logging hooks
l3lab/ww_analysis.py   [Agent B]  analyze_model(...) -> per-layer rows + summaries + extras
scripts/run_mlp_sweep.py, scripts/run_l3_sweep.py, scripts/run_snn_sweep.py, scripts/analyze_sweeps.py  [Agent B]
tests/test_l3_ref.py, tests/test_models.py [Agent A]; tests/test_data_train_ww.py [Agent B]
requirements.txt, .gitignore (data/, results/, __pycache__)  [Agent B]

## Interfaces (both agents code against these exactly)

### l3lab/l3_ref.py  [A]
class AllocTable:  built from an allocation (list of ints: embeddings per token, sum = n_emb) via
  AllocTable.from_counts(counts) -> fields: bounds (LongTensor V+1), table (LongTensor [V, k] of embedding
  ids, padded with 0), mask (BoolTensor [V, k], True where valid), k (int), n_emb (int), V (int).
class L3Ref(nn.Module):
  __init__(self, h, n_emb, d_emb, d_up)   # SAME parameter names/shapes as l3_layer.L3 so state_dicts load both ways
  forward(self, x, token_ids, alloc) -> Tensor (..., h)   # x: (..., h) float, token_ids: (...) long, alloc: AllocTable
      implemented with gather over alloc.table[token_ids] (shape (..., k)) and masked softmax (mask -> -inf)
  attention(self, x, token_ids, alloc) -> (p (..., k), comb (..., d_emb))   # for diagnostics
  @staticmethod load_from_paper_layer(L3 instance) -> L3Ref (copy parameters)
Equivalence: for random inputs, L3Ref.forward must match l3_layer.L3.forward (called through
l3_layer.valid_collate_fn with bb chosen to divide b*t) to atol 1e-5 in float32 — this is a test.

### l3lab/allocation.py  [A]
allocate_by_frequency(counts: Sequence[int] (length V), n_emb: int, k: int) -> list[int]  (>=1 each, <=k, sum=n_emb;
  greedy: repeatedly give one more embedding to the most frequent token still below k, same spirit as train_lzw)
train_lzw_generic(token_sequences: Iterable[Sequence[int]], V: int, k: int, target: int) -> list[int]  (the paper's LZW
  allocation generalised to any integer token sequences)

### l3lab/tokenizers.py  [A]
class PatchVQTokenizer: __init__(patch=7, n_codes=512, seed=0); fit(images uint8 (n,28,28)) -> self (MiniBatchKMeans on
  flattened patches, normalised to [0,1]); encode(images (n,28,28)) -> LongTensor (n, 16) of code ids; attributes V=n_codes,
  n_patches=16; counts(images) -> np.ndarray[V] code frequencies.  Must be deterministic given seed.
class CharTokenizer: fit(text) ; encode(text)->LongTensor; decode; V.

### l3lab/snn.py  [A]
def surrogate_sigmoid(k): returns sg(v)=sigmoid(k v)
def spike(v, k): straight-through spike = heaviside(v).detach() + (sg(v) - sg(v).detach())   (forward exact 0/1, backward sg')
class LIF(nn.Module): __init__(beta=0.9, theta=1.0, k=4.0, reset="soft"); forward(current (T, ..., n)) -> spikes (T, ..., n), membranes
  u_t = beta*u_{t-1} + I_t - theta*s_{t-1} (soft reset); s_t = spike(u_t - theta, k).  No in-place ops on graph tensors.
Utility: rate(spikes) = spikes.mean(0).

### l3lab/models.py  [A]   (all models: forward(batch) -> logits (b, 10) for classifiers; .ww_layers() -> list of
(name, nn.Linear/nn.Embedding modules) that WeightWatcher should analyse, with a role tag string:
'mlp', 'l3_wk', 'l3_wv', 'l3_wup', 'l3_wmix', 'embed', 'head')
MLP3(h=256, d_in=784, n_out=10, dropout=0.0): fc1, fc2, fc3 with ReLU  (the post's model)
SNNMLP3(h=256, T=8, beta=0.9, theta=1.0, k=4.0): fc1 -> LIF -> fc2 -> LIF -> fc3 readout on rates; input x (b,784) in [0,1]
  encoded as constant current (direct encoding) for T steps; returns logits = fc3(mean_t s2_t).
L3Classifier(V, n_emb, k, h=256, d_emb=64, d_up=256, n_layers=2, n_patches=16, n_out=10, alloc: AllocTable):
  token embedding E (V, h) -> [x = x + L3Ref(x, ids, alloc)] * n_layers (pre-norm is inside L3Ref) -> mean over patches ->
  head Linear(h, n_out). forward(token_ids (b,16)) -> logits.
SNNL3Classifier(..., T=8, beta, theta, k_sg, mode in {"rate_query","spiking_memory"}):
  Both modes replace the dense hidden state with spike trains. Shared skeleton per time step t:
   x0 = E[ids] (constant current); s^0_t = LIF(x0)  (spiking token features)
   per L3 block: query = s_t (spikes, no RMSNorm; optionally scaled by 1/sqrt(h)),
     mode "rate_query": p = masked softmax(query . w_k[ids]^T), comb = p . w_v[ids]  (continuous selection, spiking query)
     mode "spiking_memory": memory LIF units, one per allocated entry: I_mem = query . w_k[ids]^T (masked entries get 0 input and
        never spike), s_mem_t = LIF_mem(I_mem); comb = s_mem_t . w_v[ids] / max(1, sum of s_mem_t)   (spike-selected values)
     then cur = w_mix(concat[w_up(comb), s_t]) ; s_{t} <- LIF(cur) + residual handled as current addition (cur + x0)
   readout: head(mean over patches of mean_t s_t)
  Expose spike-rate diagnostics (mean rate per LIF, fraction of silent units, for memory units the fraction of entries that ever fire).
L3CharLM(V, n_emb, k, h, d_emb, d_up, n_layers, context=64): embedding + positional embedding + L3 blocks (token id = current
  char) + head; forward(ids (b,t)) -> logits (b,t,V).  (A secondary, faithful-to-the-paper text arm.)
All models must run on CPU in float32, be deterministic given torch.manual_seed, and have NO torch.compile.

### l3lab/data.py  [B]
load_fashion_mnist(root) -> dict(x_train uint8 (60000,28,28), y_train int64, x_test, y_test); reads the idx .gz files.
make_split(x, y, n_val=10000, seed) -> train/val split of the 60k.
patchify(x uint8 (n,28,28), patch=7) -> float32 (n,16,49) in [0,1].
load_tinyshakespeare(path) -> str.

### l3lab/train.py  [B]
train_model(model, train_x, train_y, val_x, val_y, *, epochs, batch_size, lr, weight_decay, optimizer in {"adam","sgd"},
  momentum, seed, log_every_epoch=True, ww_every=1 (epochs), ww_kwargs=None, out_dir=None, test_x=None, test_y=None)
  -> history dict: per epoch train_loss, train_acc, val_acc, (test_acc), wall, and if ww_every: the ww rows for that epoch
  (list of dicts) + summary. Model inputs: for MLP3/SNNMLP3 pass flattened float x in [0,1] (b,784); for L3 classifiers pass
  token id tensors (b,16) — train_model should accept any tensor dataset and a `to_input` callable.
  Saves state_dict checkpoints (torch.save) at the end (and optionally per epoch) when out_dir is given.
evaluate(model, x, y, batch_size, to_input) -> (loss, acc).

### l3lab/ww_analysis.py  [B]
analyze_model(model, *, min_evals=50, randomize=True, mp_fit=True, roles=None, tag=None) -> (rows: list[dict], summary: dict)
  Uses weightwatcher.WeightWatcher(model=model).analyze(plot=False, randomize=..., mp_fit=..., min_evals=...).
  rows: one per analysed layer with keys: layer_id, name, role (from model.ww_layers() name-matching; fall back to module name),
  N, M, num_evals, alpha, alpha_weighted, log_norm, log_spectral_norm, log_alpha_norm, stable_rank, mp_softrank, num_spikes,
  num_rand_spikes, rand_distance, D, xmin, xmax, sigma, warning, plus EXTRAS computed by us from the weight matrix W:
  top_sv_energy = s_1^2 / sum s_i^2 ; mean_dir_overlap = |<u_1, mean of columns of W normalised>| and |<v_1, ones/sqrt(n)>|
  (does the top singular direction align with a feature-mean / all-ones direction?), frac_dead_rows (rows with norm < 1e-3 *
  median row norm), effective_rank (exp of entropy of normalised squared singular values), and spectral_gap = s_1/s_2.
  summary: mean alpha over ALL analysed layers, mean alpha over layers with num_evals >= 50, mean alpha per role, mean
  alpha_weighted, mean log_spectral_norm, number of layers flagged 'over-trained'/'under-trained', fraction of layers with alpha<2,
  with alpha in [2,3], alpha>6.
  Must silence weightwatcher logging (logging.getLogger('weightwatcher').setLevel(ERROR)) and tolerate layers WW skips.
  Note: weightwatcher supports nn.Linear and nn.Embedding; for custom parameters (if any model keeps raw nn.Parameter tables)
  provide a helper `analyze_matrices({name: 2-D tensor})` that wraps them into a throw-away nn.Module of nn.Linear layers.

### scripts  [B]
run_mlp_sweep.py --n_runs N --out results/mlp_sweep --seed S [--epochs_max 20]: samples N random configs
  (h in {64,128,256,512}, lr log-uniform [1e-4, 3e-2], weight_decay in {0, 1e-5, 1e-4, 1e-3, 1e-2}, batch in {32,64,128,256},
  optimizer adam/sgd(momentum 0.9), epochs uniform int [1, epochs_max], init_gain in {0.5,1,2}, dropout in {0,0.2,0.5},
  label_noise in {0, 0.1, 0.3}, train_subset in {60k, 10k, 2k}) — the point is to span under-trained, well-trained and
  over-fit regimes so alpha vs accuracy can be plotted like the post. Each run: train MLP3, WW at the end (and every epoch when
  epochs <= 10), save one JSON per run (config, history, final ww rows, summary) into out/runs/*.json; resumable (skip existing).
run_l3_sweep.py: same for L3Classifier (configs: V in {256,512,1024}, n_emb in {2048,4096,8192}, k in {4,8,16}, h in {128,256},
  d_emb in {32,64,128}, n_layers in {1,2,3}, lr, wd, epochs, train_subset); tokenizer fitted once per V and cached.
run_snn_sweep.py: same for SNNMLP3 and SNNL3Classifier (extra: T in {4,8,16}, k_sg in {1,2,4,8}, beta in {0.5,0.9}, mode).
analyze_sweeps.py --in results/<dir> --out figures/: scatter avg-alpha vs train acc and vs val acc (colour = generalization gap),
  per-role alpha vs val acc, alpha trajectories over epochs, histogram of alpha by role, and a markdown table of correlations
  (Spearman between each alpha summary and val acc / gap).
All scripts: argparse, torch.set_num_threads(1), deterministic seeds, write JSON incrementally, print progress lines.

## Tests (pytest, fast)
[A] L3Ref == paper L3 (atol 1e-5) for V=20, n_emb=60, k=5, h=16, d_emb=8, d_up=12, b=4, t=8 (bb=32); AllocTable invariants;
    allocate_by_frequency invariants; PatchVQTokenizer shapes/determinism; LIF spikes are exactly 0/1 and surrogate grads are
    finite and nonzero; all models forward/backward with tiny sizes; SNNL3Classifier both modes; state_dict round trip between
    L3 and L3Ref.
[B] fashion-mnist loader shapes & label range; train_model runs 1 epoch of MLP3 on 512 samples and returns history with ww rows;
    analyze_model on MLP3 returns rows with alpha for layers with >= 50 evals; extras computed; analyze_matrices works.
