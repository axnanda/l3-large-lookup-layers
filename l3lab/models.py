"""Models for the l3lab experiments. All CPU/float32, no torch.compile, deterministic given torch.manual_seed.

Every model exposes ``ww_layers() -> list[(name, module, role)]`` where ``name`` is the dotted
``named_modules()`` path (WeightWatcher reports it as ``longname``), ``module`` is an ``nn.Linear`` or
``nn.Embedding`` (all weight matrices are plain Linear/Embedding so WeightWatcher finds them; no raw
``nn.Parameter`` tables) and ``role`` is one of
``'mlp', 'l3_wk', 'l3_wv', 'l3_wup', 'l3_wmix', 'embed', 'head'``.

Models
------
MLP3            the 3-layer ReLU MLP of the "1000 variations" post.
SNNMLP3         fc1 -> LIF -> fc2 -> LIF -> fc3 on firing rates (direct/constant-current input encoding).
L3Classifier    token embedding (+ learned patch-position embedding) -> n_layers x [x = x + L3Ref(x)] ->
                final RMSNorm -> mean over patches -> head.
SNNL3Classifier spiking version; see ``docs/SNN_L3_DESIGN.md`` and the class docstring.
L3CharLM        embedding + positional embedding + L3 blocks (token = current char) + head (no attention).

Allocation tables are stored as buffers (``alloc_bounds``, ``alloc_table``, ``alloc_mask``) so a
state_dict carries the allocation; ``model.alloc`` rebuilds the ``AllocTable`` view from them.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .allocation import allocate_by_frequency
from .l3_ref import AllocTable, L3Ref, RMSNorm, masked_softmax
from .snn import LIF

WW_ROLES = ("mlp", "l3_wk", "l3_wv", "l3_wup", "l3_wmix", "embed", "head")


def uniform_alloc(V: int, n_emb: int, k: int) -> AllocTable:
    """As-even-as-possible allocation (used when a model is built without an explicit ``alloc``)."""
    return AllocTable.from_counts(allocate_by_frequency([1] * V, n_emb, k), k=k)


def _scale_linear_weights(module: nn.Module, gain: float) -> None:
    if gain == 1.0:
        return
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, nn.Linear):
                m.weight.mul_(gain)


# --------------------------------------------------------------------------------------------------
# dense MLP and its spiking twin
# --------------------------------------------------------------------------------------------------
class MLP3(nn.Module):
    """fc1 -> ReLU -> (dropout) -> fc2 -> ReLU -> (dropout) -> fc3. ``init_gain`` rescales the default init."""

    def __init__(self, h: int = 256, d_in: int = 784, n_out: int = 10, dropout: float = 0.0, init_gain: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, n_out)
        self.dropout = nn.Dropout(dropout)
        _scale_linear_weights(self, init_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

    def ww_layers(self) -> List[Tuple[str, nn.Module, str]]:
        return [("fc1", self.fc1, "mlp"), ("fc2", self.fc2, "mlp"), ("fc3", self.fc3, "head")]


class SNNMLP3(nn.Module):
    """Spiking 3-layer MLP: x (b, 784) in [0,1] is injected as a constant current ``fc1(x)`` for ``T`` steps
    (direct encoding); LIF -> fc2 -> LIF; logits = fc3(mean_t s2_t) (rate readout, non-spiking).

    After every forward ``self.last_stats`` holds spike-rate diagnostics (rates and silent-unit fractions).
    """

    def __init__(self, h: int = 256, T: int = 8, beta: float = 0.9, theta: float = 1.0, k: float = 4.0,
                 d_in: int = 784, n_out: int = 10, reset: str = "soft", init_gain: float = 1.0):
        super().__init__()
        self.T = int(T)
        self.fc1 = nn.Linear(d_in, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, n_out)
        self.lif1 = LIF(beta, theta, k, reset)
        self.lif2 = LIF(beta, theta, k, reset)
        self.last_stats: Dict[str, float] = {}
        _scale_linear_weights(self, init_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        I1 = self.fc1(x).unsqueeze(0).expand(self.T, -1, -1)      # constant current, (T, b, h)
        s1, _ = self.lif1(I1)
        s2, _ = self.lif2(self.fc2(s1))
        r2 = s2.mean(0)
        with torch.no_grad():
            self.last_stats = {
                "rate_1": float(s1.mean()), "silent_1": float((s1.sum((0, 1)) == 0).float().mean()),
                "rate_2": float(s2.mean()), "silent_2": float((s2.sum((0, 1)) == 0).float().mean()),
            }
        return self.fc3(r2)

    def ww_layers(self) -> List[Tuple[str, nn.Module, str]]:
        return [("fc1", self.fc1, "mlp"), ("fc2", self.fc2, "mlp"), ("fc3", self.fc3, "head")]


# --------------------------------------------------------------------------------------------------
# shared allocation-buffer plumbing for the L3 models
# --------------------------------------------------------------------------------------------------
class _AllocMixin:
    def _register_alloc(self: nn.Module, alloc: AllocTable, V: int, n_emb: int) -> None:
        if alloc.V != V or alloc.n_emb != n_emb:
            raise ValueError(f"alloc has V={alloc.V}, n_emb={alloc.n_emb}; model expects V={V}, n_emb={n_emb}")
        self.register_buffer("alloc_bounds", alloc.bounds.clone())
        self.register_buffer("alloc_table", alloc.table.clone())
        self.register_buffer("alloc_mask", alloc.mask.clone())

    @property
    def alloc(self) -> AllocTable:
        t = self.alloc_table
        return AllocTable(bounds=self.alloc_bounds, table=t, mask=self.alloc_mask, k=int(t.shape[1]),
                          n_emb=int(self.alloc_bounds[-1]), V=int(t.shape[0]))


def _l3_ww_rows(prefix: str, layer: nn.Module) -> List[Tuple[str, nn.Module, str]]:
    return [(f"{prefix}.w_k", layer.w_k, "l3_wk"), (f"{prefix}.w_v", layer.w_v, "l3_wv"),
            (f"{prefix}.w_up", layer.w_up, "l3_wup"), (f"{prefix}.w_mix", layer.w_mix, "l3_wmix")]


# --------------------------------------------------------------------------------------------------
# dense L3 classifier
# --------------------------------------------------------------------------------------------------
class L3Classifier(_AllocMixin, nn.Module):
    """Patch-token classifier built from L3 layers.

    ``x = embed[ids] (+ pos[patch index])``; ``n_layers`` times ``x = x + L3Ref(x, ids, alloc)`` (pre-norm is
    inside L3Ref); final RMSNorm; mean over patches; ``head``. ``forward(token_ids (b, n_patches)) -> (b, n_out)``.

    The learned position embedding (``pos_embed=True``) is on by default: without it the query of every
    L3 lookup is a function of the token id alone, so the whole model collapses to a bag-of-tokens linear
    model and the lookups are trivial. With it the lookup selects among a token's ``k`` memories according to
    where the patch sits. ``alloc=None`` builds an as-even-as-possible allocation with ``k`` per token.
    """

    def __init__(self, V: int, n_emb: int, k: int, h: int = 256, d_emb: int = 64, d_up: int = 256, n_layers: int = 2,
                 n_patches: int = 16, n_out: int = 10, alloc: Optional[AllocTable] = None, pos_embed: bool = True):
        super().__init__()
        self.V, self.n_emb, self.k, self.h, self.n_patches = V, n_emb, k, h, n_patches
        self._register_alloc(alloc if alloc is not None else uniform_alloc(V, n_emb, k), V, n_emb)
        self.embed = nn.Embedding(V, h)
        self.pos = nn.Embedding(n_patches, h) if pos_embed else None
        self.layers = nn.ModuleList([L3Ref(h, n_emb, d_emb, d_up) for _ in range(n_layers)])
        self.norm_f = RMSNorm(h)
        self.head = nn.Linear(h, n_out)

    def features(self, token_ids: torch.Tensor, return_p: bool = False):
        if token_ids.ndim != 2 or token_ids.shape[1] != self.n_patches:
            raise ValueError(f"token_ids must have shape (b, {self.n_patches})")
        alloc = self.alloc
        x = self.embed(token_ids)
        if self.pos is not None:
            x = x + self.pos.weight[: token_ids.shape[1]].unsqueeze(0)
        ps = []
        for layer in self.layers:
            if return_p:
                p, _ = layer.attention(x, token_ids, alloc)
                ps.append(p)
            x = x + layer(x, token_ids, alloc)
        x = self.norm_f(x)
        return (x, ps) if return_p else x

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(token_ids).mean(1))

    def lookup_probs(self, token_ids: torch.Tensor) -> List[torch.Tensor]:
        """Softmax weights over each token's memories, one (b, n_patches, k) tensor per layer (diagnostics)."""
        return self.features(token_ids, return_p=True)[1]

    def ww_layers(self) -> List[Tuple[str, nn.Module, str]]:
        rows = [("embed", self.embed, "embed")]
        if self.pos is not None:
            rows.append(("pos", self.pos, "embed"))
        for i, layer in enumerate(self.layers):
            rows += _l3_ww_rows(f"layers.{i}", layer)
        rows.append(("head", self.head, "head"))
        return rows


# --------------------------------------------------------------------------------------------------
# spiking L3
# --------------------------------------------------------------------------------------------------
class SNNL3Block(nn.Module):
    """One spiking L3 lookup block (no RMSNorms; the query is a spike vector).

    Per time step, given the incoming spikes ``s`` (b, P, h) and the gathered keys/values of each
    position's token (gathered once per forward since token ids do not change over time):

        scores = score_scale * s . keys^T                                  (b, P, k)
        mode "rate_query":     p = masked softmax(scores); comb = p . vals            (continuous selection)
        mode "spiking_memory": I_mem = scores (0 on masked entries); s_mem = LIF_mem(I_mem)
                               comb = s_mem . vals / max(1, sum_k s_mem)              (spike-selected values)
        cur = w_mix(concat[w_up(comb), s])                                 (b, P, h)

    Memory LIF units are one per allocated (position, entry) pair; their membranes persist across the T
    steps. Masked (padding) entries receive zero current and therefore never cross a positive threshold;
    their spikes are additionally multiplied by the mask so this holds for any ``theta``.

    ``score_scale`` (default 1.0, see ``docs/SNN_L3_DESIGN.md``): a spike query with firing rate ``r`` has
    squared norm ``r*h`` while the dense model's RMS-normalised query has squared norm ``h``, so at
    initialisation spike scores are already *smaller* than dense scores by ``sqrt(r)``; 1/sqrt(h) is also
    accepted (pass ``score_scale=h**-0.5``) but it makes memory units essentially silent at init.
    """

    def __init__(self, h: int, n_emb: int, d_emb: int, d_up: int, mode: str = "rate_query", score_scale: float = 1.0,
                 beta: float = 0.9, theta: float = 1.0, k_sg: float = 4.0, reset: str = "soft"):
        super().__init__()
        if mode not in ("rate_query", "spiking_memory"):
            raise ValueError("mode must be 'rate_query' or 'spiking_memory'")
        self.mode, self.score_scale = mode, float(score_scale)
        self.w_k = nn.Linear(h, n_emb, bias=False)
        self.w_v = nn.Linear(d_emb, n_emb, bias=False)
        self.w_up = nn.Linear(d_emb, d_up, bias=False)
        self.w_mix = nn.Linear(d_up + h, h, bias=False)
        self.lif_mem = LIF(beta, theta, k_sg, reset) if mode == "spiking_memory" else None

    def gather(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.w_k.weight[ids], self.w_v.weight[ids]          # (b, P, k, h), (b, P, k, d_emb)

    def step(self, s: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor,
             mem_state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        """Returns ``(cur, new_mem_state, s_mem_or_None)``."""
        scores = torch.matmul(keys, (self.score_scale * s).unsqueeze(-1)).squeeze(-1)   # (b, P, k)
        if self.mode == "rate_query":
            p = masked_softmax(scores, mask, dim=-1)
            comb = torch.matmul(p.unsqueeze(-2), vals).squeeze(-2)
            s_mem = None
        else:
            I_mem = torch.where(mask, scores, torch.zeros((), dtype=scores.dtype))
            u_prev, s_prev = (None, None) if mem_state is None else mem_state
            s_mem, u_mem = self.lif_mem.step(I_mem, u_prev, s_prev)
            s_mem = s_mem * mask.to(s_mem.dtype)
            mem_state = (u_mem, s_mem)
            n_active = s_mem.sum(-1, keepdim=True).clamp(min=1.0)
            comb = torch.matmul(s_mem.unsqueeze(-2), vals).squeeze(-2) / n_active
        cur = self.w_mix(torch.cat([self.w_up(comb), s], dim=-1))
        return cur, mem_state, s_mem


class SNNL3Classifier(_AllocMixin, nn.Module):
    """Spiking L3 classifier. Per time step ``t`` (T steps, constant input current):

        x0   = embed[ids] (+ pos)                       current, (b, P, h)
        s_in = LIF_in(x0)                               spiking token features
        I = x0; s = s_in
        for each block l:  cur_l = block_l(s);  I = I + cur_l  (residual="stream") | I = x0 + cur_l (residual="input")
                           s = LIF_l(I)
        readout: head( mean over patches of mean_t s )  (rates of the last LIF; non-spiking head)

    With ``residual="stream"`` (default) the membrane current is the analogue of the dense residual stream
    ``x_l = x_{l-1} + L3(x_{l-1})``; for ``n_layers=1`` both choices coincide with the spec's ``cur + x0``.
    ``mode`` selects how the lookup is done (see ``SNNL3Block``). ``self.last_stats`` holds spike-rate
    diagnostics after each forward: ``rate_in``/``silent_in``, ``rate_l{i}``/``silent_l{i}`` per block LIF,
    and for spiking memory ``mem_rate_l{i}`` (mean spikes per valid entry per step),
    ``mem_frac_entries_fired_l{i}`` (fraction of valid (b, P, k) memory units firing at least once),
    ``mem_frac_emb_fired_l{i}`` (fraction of distinct embedding ids in the batch that fired at least once)
    and ``mem_mean_active_l{i}`` (mean number of spiking entries per lookup per step).
    """

    def __init__(self, V: int, n_emb: int, k: int, h: int = 256, d_emb: int = 64, d_up: int = 256, n_layers: int = 2,
                 n_patches: int = 16, n_out: int = 10, alloc: Optional[AllocTable] = None, T: int = 8,
                 beta: float = 0.9, theta: float = 1.0, k_sg: float = 4.0, mode: str = "rate_query",
                 score_scale: Optional[float] = None, reset: str = "soft", residual: str = "stream",
                 pos_embed: bool = True):
        super().__init__()
        if residual not in ("stream", "input"):
            raise ValueError("residual must be 'stream' or 'input'")
        self.V, self.n_emb, self.k, self.h, self.n_patches = V, n_emb, k, h, n_patches
        self.T, self.mode, self.residual = int(T), mode, residual
        self.score_scale = 1.0 if score_scale is None else float(score_scale)
        self._register_alloc(alloc if alloc is not None else uniform_alloc(V, n_emb, k), V, n_emb)
        self.embed = nn.Embedding(V, h)
        self.pos = nn.Embedding(n_patches, h) if pos_embed else None
        self.lif_in = LIF(beta, theta, k_sg, reset)
        self.blocks = nn.ModuleList([SNNL3Block(h, n_emb, d_emb, d_up, mode, self.score_scale, beta, theta, k_sg, reset)
                                     for _ in range(n_layers)])
        self.lifs = nn.ModuleList([LIF(beta, theta, k_sg, reset) for _ in range(n_layers)])
        self.head = nn.Linear(h, n_out)
        self.last_stats: Dict[str, float] = {}

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != self.n_patches:
            raise ValueError(f"token_ids must have shape (b, {self.n_patches})")
        alloc = self.alloc
        ids, mask = alloc.table[token_ids], alloc.mask[token_ids]
        x0 = self.embed(token_ids)
        if self.pos is not None:
            x0 = x0 + self.pos.weight[: token_ids.shape[1]].unsqueeze(0)
        gathered = [blk.gather(ids) for blk in self.blocks]
        L = len(self.blocks)
        u_in = s_in = None
        u_l: List[Optional[torch.Tensor]] = [None] * L
        s_l: List[Optional[torch.Tensor]] = [None] * L
        mem_state: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * L
        rec_in, rec_l, rec_mem = [], [[] for _ in range(L)], [[] for _ in range(L)]
        for _ in range(self.T):
            s_in, u_in = self.lif_in.step(x0, u_in, s_in)
            rec_in.append(s_in)
            s, I = s_in, x0
            for l, (blk, lif) in enumerate(zip(self.blocks, self.lifs)):
                keys, vals = gathered[l]
                cur, mem_state[l], s_mem = blk.step(s, keys, vals, mask, mem_state[l])
                I = I + cur if self.residual == "stream" else x0 + cur
                s, u_l[l] = lif.step(I, u_l[l], s_l[l])
                s_l[l] = s
                rec_l[l].append(s)
                if s_mem is not None:
                    rec_mem[l].append(s_mem)
        out_rates = torch.stack(rec_l[-1], 0).mean(0)                       # (b, P, h)
        self._record_stats(rec_in, rec_l, rec_mem, ids, mask)
        return self.head(out_rates.mean(1))

    @torch.no_grad()
    def _record_stats(self, rec_in, rec_l, rec_mem, ids, mask) -> None:
        def unit_stats(rec: List[torch.Tensor]) -> Tuple[float, float]:
            s = torch.stack(rec, 0)                                          # (T, b, P, h)
            return float(s.mean()), float((s.sum((0, 1, 2)) == 0).float().mean())

        st: Dict[str, float] = {}
        st["rate_in"], st["silent_in"] = unit_stats(rec_in)
        for l in range(len(self.blocks)):
            st[f"rate_l{l}"], st[f"silent_l{l}"] = unit_stats(rec_l[l])
            if rec_mem[l]:
                s = torch.stack(rec_mem[l], 0)                               # (T, b, P, k)
                valid = mask.unsqueeze(0).expand_as(s)
                n_valid = float(valid.sum()) / s.shape[0]
                st[f"mem_rate_l{l}"] = float(s.sum() / max(1.0, float(valid.sum())))
                fired = s.sum(0) > 0                                         # (b, P, k)
                st[f"mem_frac_entries_fired_l{l}"] = float(fired[mask].float().mean()) if n_valid > 0 else 0.0
                seen = torch.zeros(self.n_emb, dtype=torch.bool)
                seen[ids[mask]] = True
                hit = torch.zeros(self.n_emb, dtype=torch.bool)
                hit[ids[mask & fired]] = True
                st[f"mem_frac_emb_fired_l{l}"] = float(hit.sum()) / max(1, int(seen.sum()))
                st[f"mem_mean_active_l{l}"] = float(s.sum(-1).mean())
        st["rate_out"] = st[f"rate_l{len(self.blocks) - 1}"]
        self.last_stats = st

    def ww_layers(self) -> List[Tuple[str, nn.Module, str]]:
        rows = [("embed", self.embed, "embed")]
        if self.pos is not None:
            rows.append(("pos", self.pos, "embed"))
        for i, blk in enumerate(self.blocks):
            rows += _l3_ww_rows(f"blocks.{i}", blk)
        rows.append(("head", self.head, "head"))
        return rows


# --------------------------------------------------------------------------------------------------
# char LM
# --------------------------------------------------------------------------------------------------
class L3CharLM(_AllocMixin, nn.Module):
    """Character LM: ``x = embed[ids] + pos[0..t)``; ``n_layers`` x ``x = x + L3Ref(x, ids, alloc)`` (token id
    of a position = the current char, as in the paper); final RMSNorm; ``head`` -> logits (b, t, V).
    There is no attention/mixing across positions (a faithful-to-the-paper lookup arm, not a full LM)."""

    def __init__(self, V: int, n_emb: int, k: int, h: int = 128, d_emb: int = 32, d_up: int = 128, n_layers: int = 2,
                 context: int = 64, alloc: Optional[AllocTable] = None):
        super().__init__()
        self.V, self.n_emb, self.k, self.h, self.context = V, n_emb, k, h, context
        self._register_alloc(alloc if alloc is not None else uniform_alloc(V, n_emb, k), V, n_emb)
        self.embed = nn.Embedding(V, h)
        self.pos = nn.Embedding(context, h)
        self.layers = nn.ModuleList([L3Ref(h, n_emb, d_emb, d_up) for _ in range(n_layers)])
        self.norm_f = RMSNorm(h)
        self.head = nn.Linear(h, V)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        if t > self.context:
            raise ValueError(f"sequence length {t} exceeds context {self.context}")
        alloc = self.alloc
        x = self.embed(ids) + self.pos.weight[:t].unsqueeze(0)
        for layer in self.layers:
            x = x + layer(x, ids, alloc)
        return self.head(self.norm_f(x))

    def ww_layers(self) -> List[Tuple[str, nn.Module, str]]:
        rows = [("embed", self.embed, "embed"), ("pos", self.pos, "embed")]
        for i, layer in enumerate(self.layers):
            rows += _l3_ww_rows(f"layers.{i}", layer)
        rows.append(("head", self.head, "head"))
        return rows


def check_ww_layers(model: nn.Module) -> None:
    """Assert that ``ww_layers()`` names are real ``named_modules()`` paths of Linear/Embedding modules."""
    named = dict(model.named_modules())
    for name, module, role in model.ww_layers():
        assert role in WW_ROLES, role
        assert isinstance(module, (nn.Linear, nn.Embedding)), (name, type(module))
        assert named.get(name) is module, f"ww_layers name {name!r} is not the module's named_modules() path"
