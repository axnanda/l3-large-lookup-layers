"""Evaluation helpers for the multimodal study: zero-shot accuracy on three image sets and image-to-text
retrieval on COCO pseudo-captions, computed from cached block inputs (image-side changes) or cached image
embeddings (text-side changes)."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import torch

from . import clipmodel as cm
from . import cache, data

VAL_SETS = ["imagenette_val_sub800", "pets_test_sub800", "cifar10_test_sub800", "coco_sample"]


def class_names_of(name: str) -> List[str]:
    return json.load(open(cache.CACHE_ROOT / f"{name}_meta.json"))["class_names"]


def prompts_of(name: str) -> List[List[str]]:
    names = class_names_of(name)
    if name == "coco_sample":
        return [[c] for c in names]
    return data.class_prompts(names)


@torch.no_grad()
def text_heads(model, tok) -> Dict[str, torch.Tensor]:
    return {n: cache.text_classifier(model, tok, prompts_of(n)) for n in VAL_SETS}


@torch.no_grad()
def image_embeddings(model, name: str, start_block: int, batch: int = 100, idx: Optional[np.ndarray] = None) -> torch.Tensor:
    """Recompute embeddings of ``name`` by running blocks ``start_block..`` on the cached block input."""
    mm = cache.load_tokens(name, start_block)
    n = mm.shape[0] if idx is None else len(idx)
    out = []
    for i in range(0, n, batch):
        sl = slice(i, i + batch) if idx is None else idx[i:i + batch]
        t = torch.from_numpy(np.asarray(mm[sl])).float()
        out.append(cm.image_embed_from_block(model, t, start_block))
    return torch.cat(out)


@torch.no_grad()
def evaluate(model, heads: Dict[str, torch.Tensor], start_block: Optional[int] = None,
             image_emb: Optional[Dict[str, torch.Tensor]] = None, sets=VAL_SETS) -> Dict[str, float]:
    """Metrics on all sets.  If ``start_block`` is given, image embeddings are recomputed from that block
    (image-side modification); otherwise cached clean embeddings (or ``image_emb``) are used."""
    res = {}
    for n in sets:
        y = torch.from_numpy(cache.load_labels(n))
        if image_emb is not None and n in image_emb:
            e = image_emb[n]
        elif start_block is not None:
            e = image_embeddings(model, n, start_block)
        else:
            e = torch.from_numpy(cache.load_emb(n))
        if n == "coco_sample":
            r = cache.retrieval_recall(e, y, heads[n])
            res["coco_R@1"], res["coco_R@5"] = r["R@1"], r["R@5"]
        else:
            res[n.split("_")[0] + "_acc"] = cache.zero_shot_acc(e, y, heads[n])
    accs = [v for k, v in res.items() if k.endswith("_acc")]
    res["mean_acc"] = float(np.mean(accs)) if accs else None
    return res
