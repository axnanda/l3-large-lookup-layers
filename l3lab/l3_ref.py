"""Reference (gather-based) implementation of the L3 "Large Lookup Layer".

The paper's implementation (``l3_layer.L3``) sorts the tokens of a batch, gathers the keys/values of
every token that occurs in the batch and runs block-diagonal masked GEMMs over blocks of ``bb`` rows.
That is fast on a GPU but hard to read, needs a collate step, and requires ``bb`` to divide ``b*t``.

``L3Ref`` computes exactly the same function with a plain gather:

    A      = RMSNorm_in(x)                                   (..., h)
    ids    = alloc.table[token]                              (..., k)   embedding ids owned by `token`
    scores = A . w_k.weight[ids]^T                           (..., k)   masked with -inf where alloc.mask is False
    p      = softmax(scores, -1)                             (..., k)
    comb   = p . w_v.weight[ids]                             (..., d_emb)
    out    = w_mix( concat[ RMSNorm_out(w_up(comb)), x ] )   (..., h)

The caller adds the residual connection if desired (as in the paper's transformer block).

Parameter names and shapes are identical to ``l3_layer.L3`` so ``state_dict``s load in both directions
(``L3Ref.load_from_paper_layer`` copies them explicitly). The equivalence (atol 1e-5, float32) is
tested in ``tests/test_l3_ref.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Llama-style RMSNorm, numerically identical to ``l3_layer.LlamaRMSNorm`` (same parameter name)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


@dataclass
class AllocTable:
    """Allocation of embedding ids to vocabulary tokens, in gather-friendly form.

    Token ``v`` owns the contiguous embedding ids ``[bounds[v], bounds[v+1])``.

    Fields:
        bounds: LongTensor (V+1,)        bounds[0] = 0, bounds[V] = n_emb
        table:  LongTensor (V, k)        table[v, j] = bounds[v] + j for j < counts[v], padded with 0
        mask:   BoolTensor (V, k)        True where ``table`` holds a valid id
        k:      int                      width of the table (>= max(counts))
        n_emb:  int                      total number of embeddings (= sum(counts))
        V:      int                      vocabulary size
    """

    bounds: torch.Tensor
    table: torch.Tensor
    mask: torch.Tensor
    k: int
    n_emb: int
    V: int

    @staticmethod
    def from_counts(counts: Sequence[int], k: int | None = None) -> "AllocTable":
        """Build a table from per-token counts (each >= 1). ``k`` defaults to ``max(counts)``."""
        counts_t = torch.as_tensor(list(counts), dtype=torch.long)
        if counts_t.ndim != 1 or counts_t.numel() == 0:
            raise ValueError("counts must be a non-empty 1-D sequence")
        if int(counts_t.min()) < 1:
            raise ValueError("every token must own at least one embedding")
        V = counts_t.numel()
        kmax = int(counts_t.max())
        if k is None:
            k = kmax
        if k < kmax:
            raise ValueError(f"k={k} is smaller than max(counts)={kmax}")
        bounds = torch.zeros(V + 1, dtype=torch.long)
        bounds[1:] = torch.cumsum(counts_t, 0)
        offs = torch.arange(k, dtype=torch.long)
        mask = offs.view(1, k) < counts_t.view(V, 1)
        table = torch.where(mask, bounds[:-1].view(V, 1) + offs.view(1, k), torch.zeros(1, 1, dtype=torch.long))
        return AllocTable(bounds=bounds, table=table, mask=mask, k=int(k), n_emb=int(bounds[-1]), V=int(V))

    @property
    def counts(self) -> torch.Tensor:
        return self.bounds[1:] - self.bounds[:-1]

    def emb_alloc(self) -> torch.Tensor:
        """LongTensor (n_emb,) giving the owning token id of every embedding (the paper's ``emb_alloc``)."""
        return torch.repeat_interleave(torch.arange(self.V, dtype=torch.long), self.counts)

    def to(self, device) -> "AllocTable":
        return AllocTable(self.bounds.to(device), self.table.to(device), self.mask.to(device), self.k, self.n_emb, self.V)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """softmax over ``dim`` with ``mask == False`` entries sent to -inf (every row must have a True)."""
    neg_inf = torch.full((), -float("inf"), dtype=scores.dtype, device=scores.device)
    return torch.softmax(torch.where(mask, scores, neg_inf), dim=dim)


class L3Ref(nn.Module):
    """Gather-based L3 layer; see the module docstring. Same parameter names/shapes as ``l3_layer.L3``."""

    def __init__(self, h: int, n_emb: int, d_emb: int, d_up: int):
        super().__init__()
        self.h, self.n_emb, self.d_emb, self.d_up = h, n_emb, d_emb, d_up
        self.w_k = nn.Linear(h, n_emb, bias=False)      # rows = keys
        self.w_v = nn.Linear(d_emb, n_emb, bias=False)  # rows = values
        self.w_mix = nn.Linear(d_up + h, h, bias=False)
        self.w_up = nn.Linear(d_emb, d_up, bias=False)
        self.norm_in = RMSNorm(h)
        self.norm_out = RMSNorm(d_up)

    def attention(self, x: torch.Tensor, token_ids: torch.Tensor, alloc: AllocTable):
        """Return ``(p, comb)``: the softmax weights (..., k) and combined value (..., d_emb)."""
        if token_ids.shape != x.shape[:-1]:
            raise ValueError(f"token_ids shape {tuple(token_ids.shape)} must equal x.shape[:-1] {tuple(x.shape[:-1])}")
        ids = alloc.table[token_ids]            # (..., k)
        mask = alloc.mask[token_ids]            # (..., k)
        A = self.norm_in(x)                     # (..., h)
        keys = self.w_k.weight[ids]             # (..., k, h)
        vals = self.w_v.weight[ids]             # (..., k, d_emb)
        scores = torch.matmul(keys, A.unsqueeze(-1)).squeeze(-1)   # (..., k)
        p = masked_softmax(scores, mask, dim=-1)
        comb = torch.matmul(p.unsqueeze(-2), vals).squeeze(-2)     # (..., d_emb)
        return p, comb

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor, alloc: AllocTable) -> torch.Tensor:
        _, comb = self.attention(x, token_ids, alloc)
        up = self.norm_out(self.w_up(comb))
        return self.w_mix(torch.cat([up, x], dim=-1))

    @staticmethod
    def load_from_paper_layer(layer: nn.Module) -> "L3Ref":
        """Copy the parameters of an ``l3_layer.L3`` instance (or another ``L3Ref``) into a fresh ``L3Ref``."""
        h = layer.w_k.weight.shape[1]
        n_emb, d_emb = layer.w_v.weight.shape
        d_up = layer.w_up.weight.shape[0]
        ref = L3Ref(h, n_emb, d_emb, d_up)
        ref.load_state_dict({k: v.detach().clone() for k, v in layer.state_dict().items()}, strict=True)
        return ref


def paper_inputs(seqs: torch.Tensor, alloc: AllocTable):
    """Build the argument tuple for ``l3_layer.L3.forward`` from token ids ``seqs`` (b, t) and an allocation.

    Goes through ``l3_layer.valid_collate_fn`` exactly as the paper's data pipeline would. Returns
    ``(seqs, fw, bw, seq_sort, keep_cols, emb_alloc, starts, ends)``.
    """
    import l3_layer  # local import: the paper file pulls in tqdm, which l3lab itself does not need

    emb_alloc = alloc.emb_alloc()
    batch = [(seqs[i], emb_alloc, alloc.bounds) for i in range(seqs.shape[0])]
    return l3_layer.valid_collate_fn(batch)
