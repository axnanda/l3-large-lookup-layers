# Spiking L3 (SNNL3Classifier): what spikes, what does not, and why

Code: `l3lab/snn.py` (primitives), `l3lab/models.py` (`SNNL3Block`, `SNNL3Classifier`, `SNNMLP3`).

## Spike primitive
`spike(v, k) = heaviside(v).detach() + (sigmoid(k v) - sigmoid(k v).detach())`. The forward value is
*exactly* 0 or 1 (the two surrogate terms cancel); the backward pass sees `d/dv sigmoid(k v)`. `k_sg`
sets the surrogate width. `LIF(beta, theta, k, reset)` integrates `u_t = beta u_{t-1} + I_t - theta s_{t-1}`
(soft reset; hard/none available) and emits `s_t = spike(u_t - theta, k)`; it is unrolled with lists and
`torch.stack`, no in-place ops, so autograd is clean. `LIF.step` exposes one time step so the L3 model
can interleave lookups with the membrane updates.

## What is spiking
* **Token features**: the embedding (+ position) vector is a *constant current* `x0` for `T` steps
  (direct encoding); `LIF_in` turns it into a spike train `s_in,t` (0/1 vectors of length `h`).
* **Queries**: every L3 lookup is driven by the spikes of the previous stage, not by an RMS-normalised
  dense vector. Scores are `score_scale * s . w_k[ids]^T`, restricted to the `k` embeddings allocated to the
  position's token.
* **Hidden state between blocks**: each block's output current `cur = w_mix([w_up(comb), s])` is added to
  the residual current (`residual="stream"`: `I_l = I_{l-1} + cur_l`, the analogue of the dense residual
  stream; `"input"`: `I_l = x0 + cur_l` as literally written in the spec; identical for one layer) and
  passed through `LIF_l`. Spikes, not real vectors, flow between blocks.
* **Memory selection, mode `spiking_memory`**: one LIF unit per allocated (position, entry) pair with
  membrane persisting over the `T` steps; input current = the score; padded entries get zero current and
  are additionally masked, so they never spike. The value read out is the *mean of the values of the
  entries that spiked* (`s_mem . w_v[ids] / max(1, sum s_mem)`); if nothing spikes the lookup returns 0.
  Selection is therefore binary and event-driven, the number of selected memories is data-dependent.
* **Mode `rate_query`**: same spiking query/hidden state, but the memory selection stays a masked
  softmax (continuous). This isolates the effect of spiking *queries* from spiking *selection*.

## What is not spiking (and why)
* Weight matrices (`w_k, w_v, w_up, w_mix`, embeddings, head) are ordinary float matrices: they are what
  WeightWatcher analyses, so they must be the same objects as in the dense model (plain `nn.Linear`).
* The readout is a rate decoder: `head(mean over patches of mean_t s_last,t)`. A spiking head would add a
  second surrogate stage with no benefit for the alpha-vs-accuracy question.
* No RMSNorm on the spike query (a 0/1 vector has a natural scale) and no `norm_out` on `w_up(comb)` (the
  LIF threshold plays the role of the normaliser).

## Numerical scale (`score_scale`)
The dense model feeds `RMSNorm(x)` (squared norm `h`) into `w_k` (init `U(±1/sqrt(h))`), giving score std
`≈ sqrt(1/3) = 0.58`. A spike query with firing rate `r` has squared norm `r h`, so with `score_scale = 1`
the spike scores are already *smaller*: std `≈ 0.58 sqrt(r)` (measured 0.28 at init, r ≈ 0.23, h = 256).
Dividing additionally by `sqrt(h)` (0.0625 for h=256) pushes scores to ~0.02; with `theta = 1` a memory LIF
then needs `I > theta (1-beta)` to ever fire (0.1 for beta=0.9), so the memory is silent and
`spiking_memory` degenerates to `comb = 0` at init. Measured at init (V=512, n_emb=4096, k=8, h=256, T=8,
batch 128), fraction of allocated entries that fire at least once in a lookup:

| beta | score_scale = 1 | score_scale = 1/sqrt(h) |
|------|-----------------|--------------------------|
| 0.9  | 0.285 (0.56 active entries / lookup / step) | 0.000 |
| 0.5  | 0.040 | 0.000 |

Hence the default is `score_scale = 1.0`; it is a constructor argument (`SNNL3Classifier(score_scale=...)`),
`h ** -0.5` can be swept. With `beta = 0.5` the memory is nearly silent at init too, so sweeps over `beta`
should watch the firing statistics below rather than assume the lookup is active.

## Statistics to watch (`model.last_stats` after every forward)
* `rate_in`, `rate_l{i}`, `rate_out`: mean firing rate of the input LIF / each block LIF (0 = dead, 1 =
  saturated; healthy values 0.05–0.4). `silent_in`, `silent_l{i}`: fraction of hidden units that never fire
  in the batch (dead units carry no gradient through the surrogate once far below threshold).
* `mem_rate_l{i}`: spikes per valid memory entry per step; `mem_frac_entries_fired_l{i}`: fraction of the
  (batch, patch, entry) memory units that fire at least once over `T` (the "firing fraction" of the
  memory); `mem_frac_emb_fired_l{i}`: fraction of the distinct embedding ids present in the batch that
  fired at least once (how much of the table is actually used); `mem_mean_active_l{i}`: mean number of
  selected memories per lookup per step (k = all, 0 = none).
* Cost: one forward+backward (batch 128, T=8, h=256, n_emb=4096, 2 layers, 1 thread) is ~1.0–1.2 s for
  either mode versus ~0.16–0.26 s for the dense `L3Classifier`: `w_mix`/`w_up` run once per time step, so
  the SNN is inherently ~T× the dense cost. Budget sweeps with `train_subset` and `T` accordingly.
