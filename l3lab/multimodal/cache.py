"""Token caches: the input of every image block (fp16, memmap on disk) plus final embeddings, so that
experiments that modify block k only run blocks k..11.  Also text-side caches (class prompt embeddings).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import clipmodel as cm
from .data import ImageSet, load_images

CACHE_ROOT = Path(os.environ.get("MM_CACHE_ROOT", "/tmp/claude-0/-home-user/cf719c09-d7e5-5f6b-bcb9-86a98cd733c6/scratchpad/mmcache"))


def _mm_path(name: str, blk) -> Path:
    return CACHE_ROOT / f"{name}_blk{blk}.f16"


@torch.no_grad()
def build_image_cache(model, preprocess, ds: ImageSet, blocks: Sequence[int], batch: int = 64, log=print):
    """Store the input tokens of each block in ``blocks`` (and the final 512-d embedding) for ``ds``."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    N, L, D = len(ds), 50, 768
    meta_path = CACHE_ROOT / f"{ds.name}_meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        if all(b in meta["blocks"] for b in blocks) and meta["n"] == N:
            log(f"cache {ds.name}: exists"); return meta
    mms = {b: np.memmap(_mm_path(ds.name, b), dtype=np.float16, mode="w+", shape=(N, L, D)) for b in blocks}
    emb = np.zeros((N, 512), np.float32)
    nblk = len(cm.image_blocks(model))
    i0 = 0
    for x in load_images(ds.paths, preprocess, batch):
        t = cm.embed_images(model, x)
        for k in range(nblk):
            if k in mms:
                mms[k][i0:i0 + len(x)] = t.half().numpy()
            t = cm.image_blocks(model)[k](t)
        emb[i0:i0 + len(x)] = cm.pool_image(model, t).numpy()
        i0 += len(x)
        if (i0 // batch) % 10 == 0:
            log(f"cache {ds.name}: {i0}/{N}")
    for m in mms.values():
        m.flush()
    np.save(CACHE_ROOT / f"{ds.name}_emb.npy", emb)
    np.save(CACHE_ROOT / f"{ds.name}_labels.npy", ds.labels)
    meta = {"n": N, "blocks": sorted(set(list(blocks) + (json.load(open(meta_path))["blocks"] if meta_path.exists() else []))), "class_names": ds.class_names}
    json.dump(meta, open(meta_path, "w"))
    return meta


def load_tokens(name: str, blk: int, idx: Optional[np.ndarray] = None) -> np.memmap:
    meta = json.load(open(CACHE_ROOT / f"{name}_meta.json"))
    mm = np.memmap(_mm_path(name, blk), dtype=np.float16, mode="r", shape=(meta["n"], 50, 768))
    return mm if idx is None else mm[idx]


def load_labels(name: str) -> np.ndarray:
    return np.load(CACHE_ROOT / f"{name}_labels.npy")


def load_emb(name: str) -> np.ndarray:
    return np.load(CACHE_ROOT / f"{name}_emb.npy")


@torch.no_grad()
def text_classifier(model, tok, prompts: List[List[str]], batch: int = 256) -> torch.Tensor:
    """Average normalised prompt embeddings per class -> (C, 512) normalised classifier weights."""
    flat = [p for ps in prompts for p in ps]
    embs = []
    for i in range(0, len(flat), batch):
        e = cm.text_embed(model, tok(flat[i:i + batch]))
        embs.append(torch.nn.functional.normalize(e, dim=-1))
    e = torch.cat(embs)
    k = len(prompts[0])
    w = e.reshape(len(prompts), k, -1).mean(1)
    return torch.nn.functional.normalize(w, dim=-1)


def zero_shot_acc(img_emb: torch.Tensor, labels: torch.Tensor, W: torch.Tensor) -> float:
    pred = (torch.nn.functional.normalize(img_emb, dim=-1) @ W.t()).argmax(-1)
    return float((pred == labels).float().mean())


def retrieval_recall(img_emb: torch.Tensor, labels: torch.Tensor, W: torch.Tensor, ks=(1, 5)) -> Dict[str, float]:
    sims = torch.nn.functional.normalize(img_emb, dim=-1) @ W.t()
    rank = (sims > sims[torch.arange(len(labels)), labels][:, None]).sum(-1)
    return {f"R@{k}": float((rank < k).float().mean()) for k in ks}
