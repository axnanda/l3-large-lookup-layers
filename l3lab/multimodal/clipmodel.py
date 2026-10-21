"""Pretrained CLIP (open_clip ViT-B/32, LAION-400M) with spiking inserts and a WeightWatcher layer map.

Everything here works on the open_clip ``CLIP`` class (``model.visual`` = VisionTransformer, text tower
= ``model.transformer`` + ``token_embedding`` + ``ln_final`` + ``text_projection``).

Spiking inserts (section 3 of docs/MULTIMODAL_STUDY.md):

* :class:`SpikingAct` replaces the QuickGELU of one transformer block's MLP by a rate-coded population
  of integrate-and-fire units driven for ``T`` steps by the (constant) pre-activation. Output is
  ``gain * mean_t s_t`` so that, with ``gain = theta`` and large ``T``, it reproduces ``relu`` of the input
  (standard ANN-to-SNN conversion identity with soft reset).  ``k_sg`` is the sigmoid surrogate slope used
  in the backward pass (straight-through spikes from :mod:`l3lab.snn`).
* :class:`SpikingAdapter` is a bottleneck adapter ``W_up(SpikingAct(W_down(x)))`` added in parallel to a
  block's MLP with ``W_up`` initialised to zero (so the model is unchanged at insertion).  Its dense twin
  :class:`DenseAdapter` uses QuickGELU.

Firing diagnostics (mean rate, fraction of silent units, fraction of saturated units) are accumulated on the
module while ``track`` is on.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..snn import spike

DEFAULT_WEIGHTS = os.environ.get(
    "CLIP_WEIGHTS",
    "/tmp/claude-0/-home-user/cf719c09-d7e5-5f6b-bcb9-86a98cd733c6/scratchpad/dl/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
)
MODEL_NAME = "ViT-B-32-quickgelu"


def load_clip(weights: str = DEFAULT_WEIGHTS, device: str = "cpu"):
    """Return (model, preprocess, tokenizer) for the pretrained ViT-B/32."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=weights, device=device)
    tok = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    return model, preprocess, tok


# ------------------------------------------------------------------ tower pieces
def image_blocks(model) -> nn.ModuleList:
    return model.visual.transformer.resblocks


def text_blocks(model) -> nn.ModuleList:
    return model.transformer.resblocks


def embed_images(model, x: torch.Tensor) -> torch.Tensor:
    """Patch + cls + pos embedding + ln_pre: the input of image block 0 (N, 50, 768)."""
    return model.visual._embeds(x)


def run_image_blocks(model, tokens: torch.Tensor, start: int, end: Optional[int] = None) -> torch.Tensor:
    blocks = image_blocks(model)
    end = len(blocks) if end is None else end
    for i in range(start, end):
        tokens = blocks[i](tokens)
    return tokens


def pool_image(model, tokens: torch.Tensor) -> torch.Tensor:
    """ln_post on the cls token and projection -> (N, 512) un-normalised image embedding."""
    v = model.visual
    pooled = v.ln_post(tokens[:, 0])
    return pooled @ v.proj


def image_embed_from_block(model, tokens: torch.Tensor, start: int) -> torch.Tensor:
    return pool_image(model, run_image_blocks(model, tokens, start))


def embed_text(model, text: torch.Tensor) -> torch.Tensor:
    x = model.token_embedding(text)
    return x + model.positional_embedding


def run_text_blocks(model, x: torch.Tensor, start: int, end: Optional[int] = None) -> torch.Tensor:
    blocks = text_blocks(model)
    end = len(blocks) if end is None else end
    for i in range(start, end):
        x = blocks[i](x, attn_mask=model.attn_mask)
    return x


def pool_text(model, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    x = model.ln_final(x)
    pooled = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    return pooled @ model.text_projection


def text_embed(model, text: torch.Tensor) -> torch.Tensor:
    return pool_text(model, run_text_blocks(model, embed_text(model, text), 0), text)


# ------------------------------------------------------------------ WeightWatcher matrices
def clip_matrices(model, towers=("image", "text")) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """All 2-D weight matrices of the model with a role each.

    Roles: patch_embed, token_embed, attn_q, attn_k, attn_v, attn_out, mlp_fc1, mlp_fc2, proj_image, proj_text.
    Names: ``image.b03.mlp_fc1`` etc.  Attention in_proj is split into q/k/v.  Embedding tables are returned
    too but should never be averaged with the dense layers.
    """
    mats: Dict[str, torch.Tensor] = {}
    roles: Dict[str, str] = {}

    def add(name, W, role):
        mats[name] = W.detach().float().reshape(W.shape[0], -1)
        roles[name] = role

    if "image" in towers:
        v = model.visual
        add("image.patch_embed", v.conv1.weight, "patch_embed")
        for i, b in enumerate(image_blocks(model)):
            _add_block(add, f"image.b{i:02d}", b)
        add("image.proj", v.proj.t(), "proj_image")  # (512, 768): rows = outputs
    if "text" in towers:
        add("text.token_embed", model.token_embedding.weight, "token_embed")
        for i, b in enumerate(text_blocks(model)):
            _add_block(add, f"text.b{i:02d}", b)
        add("text.proj", model.text_projection.t(), "proj_text")
    return mats, roles


def _add_block(add, prefix, b):
    w = b.attn.in_proj_weight
    d = w.shape[0] // 3
    add(f"{prefix}.attn_q", w[:d], "attn_q")
    add(f"{prefix}.attn_k", w[d:2 * d], "attn_k")
    add(f"{prefix}.attn_v", w[2 * d:], "attn_v")
    add(f"{prefix}.attn_out", b.attn.out_proj.weight, "attn_out")
    mlp = b.mlp.mlp if isinstance(b.mlp, MLPWithAdapter) else b.mlp
    add(f"{prefix}.mlp_fc1", mlp.c_fc.weight, "mlp_fc1")
    add(f"{prefix}.mlp_fc2", mlp.c_proj.weight, "mlp_fc2")


def block_matrices(block: nn.Module, prefix: str = "b") -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    mats, roles = {}, {}

    def add(name, W, role):
        mats[name] = W.detach().float().reshape(W.shape[0], -1)
        roles[name] = role

    _add_block(add, prefix, block)
    return mats, roles


# ------------------------------------------------------------------ spiking modules
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class SpikingAct(nn.Module):
    """Rate-coded (leaky) integrate-and-fire activation driven by a constant current for ``T`` steps.

    u_t = beta * u_{t-1} + I - theta * s_{t-1}   (soft reset; ``reset='hard'`` multiplies by (1 - s))
    s_t = spike(u_t - theta, k_sg)               exact 0/1 forward, sigmoid-surrogate backward
    out = gain * mean_t s_t                       (gain per channel, init = theta)

    ``u0_frac * theta`` initialises the membrane (0.5 halves the conversion quantisation bias).
    ``theta`` is set by :meth:`calibrate` (percentile of the pre-activation) or given.
    """

    def __init__(self, n: int, T: int = 8, theta: float = 1.0, beta: float = 1.0, k_sg: float = 4.0,
                 reset: str = "soft", u0_frac: float = 0.5, learn_gain: bool = True, learn_theta: bool = False,
                 detach_reset: bool = False):
        super().__init__()
        self.n, self.T, self.beta, self.k_sg, self.reset, self.u0_frac = n, T, beta, k_sg, reset, u0_frac
        self.detach_reset = detach_reset
        self.log_theta = nn.Parameter(torch.tensor(math.log(theta)), requires_grad=learn_theta)
        self.gain = nn.Parameter(torch.full((n,), float(theta)), requires_grad=learn_gain)
        self.track = False
        self.reset_stats()

    @property
    def theta(self) -> torch.Tensor:
        return self.log_theta.exp()

    def set_theta(self, theta: float):
        with torch.no_grad():
            self.log_theta.fill_(math.log(max(theta, 1e-6)))
            self.gain.fill_(float(theta))

    def reset_stats(self):
        self._rate_sum = torch.zeros(self.n)
        self._active_sum = torch.zeros(self.n)
        self._sat_sum = torch.zeros(self.n)
        self._count = 0

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        theta = self.theta
        u = torch.full_like(I, 0.0) + self.u0_frac * theta
        s = torch.zeros_like(I)
        acc = torch.zeros_like(I)
        for _ in range(self.T):
            s_prev = s.detach() if self.detach_reset else s
            if self.reset == "hard":
                u = (self.beta * u + I) * (1 - s_prev)
            else:
                u = self.beta * u + I - theta * s_prev
            s = spike(u - theta, self.k_sg)
            acc = acc + s
        rate = acc / self.T
        if self.track:
            with torch.no_grad():
                flat = rate.detach().reshape(-1, self.n)
                self._rate_sum += flat.mean(0)
                self._active_sum += (flat > 0).float().mean(0)
                self._sat_sum += (flat >= 1 - 1e-6).float().mean(0)
                self._count += 1
        return rate * self.gain

    def stats(self) -> Dict[str, float]:
        if self._count == 0:
            return {"mean_rate": None, "frac_silent_units": None, "frac_saturated_units": None, "frac_active": None}
        r = self._rate_sum / self._count
        a = self._active_sum / self._count
        sat = self._sat_sum / self._count
        return {"mean_rate": float(r.mean()), "frac_silent_units": float((a < 1e-3).float().mean()),
                "frac_saturated_units": float((sat > 0.5).float().mean()), "frac_active": float(a.mean())}

    def extra_repr(self):
        return f"n={self.n}, T={self.T}, beta={self.beta}, k_sg={self.k_sg}, reset={self.reset}"


class DenseAct(nn.Module):
    """QuickGELU with a learnable per-channel gain (dense twin of :class:`SpikingAct`)."""

    def __init__(self, n: int, learn_gain: bool = True):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n), requires_grad=learn_gain)

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x) * self.gain


class _Adapter(nn.Module):
    def __init__(self, d: int, r: int, act: nn.Module):
        super().__init__()
        self.down = nn.Linear(d, r, bias=True)
        self.act = act
        self.up = nn.Linear(r, d, bias=True)
        nn.init.normal_(self.down.weight, std=d ** -0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class SpikingAdapter(_Adapter):
    def __init__(self, d: int, r: int, **spk):
        super().__init__(d, r, SpikingAct(r, **spk))


class DenseAdapter(_Adapter):
    def __init__(self, d: int, r: int):
        super().__init__(d, r, QuickGELU())


class MLPWithAdapter(nn.Module):
    """``mlp(x) + adapter(x)``; wraps a block's ``mlp`` so the block code stays untouched."""

    def __init__(self, mlp: nn.Module, adapter: nn.Module):
        super().__init__()
        self.mlp, self.adapter = mlp, adapter

    def forward(self, x):
        return self.mlp(x) + self.adapter(x)


# ------------------------------------------------------------------ install / query inserts
def get_block(model, tower: str, idx: int) -> nn.Module:
    return (image_blocks(model) if tower == "image" else text_blocks(model))[idx]


def install_spiking_act(model, tower: str, idx: int, **spk) -> SpikingAct:
    """Replace the MLP activation of block ``idx`` by a :class:`SpikingAct` (returns it)."""
    b = get_block(model, tower, idx)
    n = b.mlp.c_fc.out_features
    act = SpikingAct(n, **spk)
    b.mlp.gelu = act
    return act


def install_dense_act(model, tower: str, idx: int, learn_gain: bool = True) -> DenseAct:
    b = get_block(model, tower, idx)
    act = DenseAct(b.mlp.c_fc.out_features, learn_gain=learn_gain)
    b.mlp.gelu = act
    return act


def restore_act(model, tower: str, idx: int):
    get_block(model, tower, idx).mlp.gelu = QuickGELU()


def install_adapter(model, tower: str, idx: int, r: int = 64, spiking: bool = True, **spk) -> _Adapter:
    b = get_block(model, tower, idx)
    d = b.mlp.c_fc.in_features
    adapter = SpikingAdapter(d, r, **spk) if spiking else DenseAdapter(d, r)
    if isinstance(b.mlp, MLPWithAdapter):
        b.mlp = b.mlp.mlp
    b.mlp = MLPWithAdapter(b.mlp, adapter)
    return adapter


def spiking_modules(model) -> List[Tuple[str, SpikingAct]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, SpikingAct)]


def set_tracking(model, on: bool):
    for _, m in spiking_modules(model):
        m.track = on
        if on:
            m.reset_stats()


def firing_stats(model) -> Dict[str, Dict[str, float]]:
    return {n: m.stats() for n, m in spiking_modules(model)}


@torch.no_grad()
def calibrate_theta(block: nn.Module, tokens: torch.Tensor, percentile: float = 99.9, attn_mask=None) -> float:
    """Threshold balancing: percentile of the block's MLP pre-activation on ``tokens`` (its input)."""
    x = tokens
    x = x + block.ls_1(block.attention(q_x=block.ln_1(x), attn_mask=attn_mask))
    pre = block.mlp.c_fc(block.ln_2(x))
    q = torch.quantile(pre.reshape(-1).float()[:: max(1, pre.numel() // 2_000_000)], percentile / 100)
    return float(q)
