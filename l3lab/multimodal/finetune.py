"""Fine-tuning of a CLIP image block with a spiking (or dense) insert on cached block inputs, with
per-step WeightWatcher trajectories, firing diagnostics and zero-shot / retrieval metrics.

The objective is text-anchored cross-entropy: logits = 100 * cos(image_emb, class_text_emb) with the frozen
zero-shot head of the (clean) text tower on Imagenette train, i.e. the image tower is fine-tuned to re-align
with the language embedding after the insert damaged it.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import clipmodel as cm
from . import cache, evalmm
from ..ww_analysis import analyze_matrices


def ww_matrices_for(model, tower, k, adapter: Optional[nn.Module], scope: str):
    blk = cm.get_block(model, tower, k)
    mats, roles = cm.block_matrices(blk, f"{tower}.b{k:02d}")
    if scope == "block":  # only the MLP trains; drop the frozen attention matrices
        mats = {n: w for n, w in mats.items() if "mlp" in n}; roles = {n: roles[n] for n in mats}
    if scope == "later":
        blocks = cm.image_blocks(model) if tower == "image" else cm.text_blocks(model)
        for j in range(k + 1, len(blocks)):
            m2, r2 = cm.block_matrices(blocks[j], f"{tower}.b{j:02d}"); mats.update(m2); roles.update(r2)
    if adapter is not None:
        mats[f"{tower}.b{k:02d}.adapter_down"] = adapter.down.weight; roles[f"{tower}.b{k:02d}.adapter_down"] = "adapter_down"
        mats[f"{tower}.b{k:02d}.adapter_up"] = adapter.up.weight; roles[f"{tower}.b{k:02d}.adapter_up"] = "adapter_up"
    return mats, roles


def run_insert(cfg: Dict, log=print) -> Dict:
    """cfg keys: tower('image'), block, variant (spike_replace|dense_replace|spike_adapter|dense_adapter),
    T, k_sg, beta, reset, r (adapter width), scope (block|later), train_set, n_train, epochs, lr, batch, seed,
    ww_every (steps), eval_every (steps), label_noise (fraction), threads, objective (distill = 1 - cos to the
    clean model's embedding of the same image, the standard conversion-repair objective; ce = text-anchored
    cross-entropy; both = ce + 10 * distill)."""
    torch.set_num_threads(cfg.get("threads", 2))
    torch.manual_seed(cfg.get("seed", 0)); rng = np.random.default_rng(cfg.get("seed", 0))
    t0 = time.time()
    model, pre, tok = cm.load_clip()
    tower, k, variant = cfg.get("tower", "image"), cfg["block"], cfg["variant"]
    assert tower == "image", "text-side fine-tuning is not implemented in this script"
    heads = evalmm.text_heads(model, tok)
    train_name = cfg.get("train_set", "imagenette_train")
    mm = cache.load_tokens(train_name, k); y_all = cache.load_labels(train_name)
    n_train = min(cfg.get("n_train", 3000), mm.shape[0])
    idx = np.sort(rng.choice(mm.shape[0], n_train, replace=False))
    y = y_all[idx].copy()
    if cfg.get("label_noise", 0) > 0:
        flip = rng.random(n_train) < cfg["label_noise"]; y[flip] = rng.integers(0, y.max() + 1, flip.sum())
    head = heads["imagenette_val_sub800"]  # same class names as imagenette train
    clean_emb = F.normalize(torch.from_numpy(cache.load_emb(train_name)), dim=-1)  # distillation targets
    objective = cfg.get("objective", "distill")
    # ---- insert
    blk = cm.get_block(model, tower, k)
    calib = torch.from_numpy(np.asarray(mm[idx[:256]])).float()
    theta = cm.calibrate_theta(blk, calib, 99.9)
    adapter = None
    spk = dict(T=cfg.get("T", 8), k_sg=cfg.get("k_sg", 4.0), beta=cfg.get("beta", 1.0), reset=cfg.get("reset", "soft"))
    if variant == "spike_replace":
        cm.install_spiking_act(model, tower, k, theta=theta, learn_theta=cfg.get("learn_theta", False), **spk)
    elif variant == "dense_replace":
        cm.install_dense_act(model, tower, k)
    elif variant == "spike_adapter":
        adapter = cm.install_adapter(model, tower, k, r=cfg.get("r", 64), spiking=True, theta=1.0, **spk)
    elif variant == "dense_adapter":
        adapter = cm.install_adapter(model, tower, k, r=cfg.get("r", 64), spiking=False)
    else:
        raise ValueError(variant)
    if adapter is not None and variant == "spike_adapter":
        with torch.no_grad():  # calibrate adapter threshold on its own pre-activation
            pre_a = adapter.down(blk.ln_2(calib)); adapter.act.set_theta(float(torch.quantile(pre_a.reshape(-1), 0.999)))
    # ---- trainable parameters
    for p in model.parameters():
        p.requires_grad_(False)
    scope = cfg.get("scope", "block")
    params: List[nn.Parameter] = []
    if adapter is not None:
        params += list(adapter.parameters())
        if cfg.get("train_block_with_adapter", False):
            params += list(blk.mlp.mlp.parameters()) + list(blk.ln_2.parameters())
    else:
        params += list(blk.mlp.parameters()) + list(blk.ln_2.parameters())
    if scope == "later":
        blocks = cm.image_blocks(model)
        for j in range(k + 1, len(blocks)):
            params += list(blocks[j].parameters())
        params += list(model.visual.ln_post.parameters()) + [model.visual.proj]
    params = [p for p in params if p.dim() > 0 or True]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.get("lr", 1e-4), weight_decay=0.0)
    n_params = sum(p.numel() for p in params)
    # ---- logging closures
    history = {"step": [], "epoch": [], "train_loss": [], "train_acc": [], "ww": [], "eval": [], "firing": []}
    bs = cfg.get("batch", 64); steps_per_epoch = math.ceil(n_train / bs); total_steps = steps_per_epoch * cfg.get("epochs", 3)
    ww_every = cfg.get("ww_every", 32); eval_every = cfg.get("eval_every", steps_per_epoch)

    def do_ww(step):
        mats, roles = ww_matrices_for(model, tower, k, adapter, scope)
        rows, summ = analyze_matrices(mats, roles=roles, min_evals=50, randomize=False, mp_fit=False)
        slim = [{kk: r.get(kk) for kk in ("name", "role", "N", "M", "num_evals", "alpha", "alpha_weighted", "log_spectral_norm", "log_norm", "xmin", "xmax", "D", "num_spikes", "status", "warning", "frac_dead_rows", "frac_dead_cols", "top_sv_energy", "effective_rank", "stable_rank", "mean_dir_overlap_right")} for r in rows]
        history["ww"].append({"step": step, "rows": slim})
        return {r["name"].split(".")[-1]: (None if r["alpha"] is None else round(r["alpha"], 3)) for r in slim}

    def do_eval(step):
        model.eval(); cm.set_tracking(model, True)
        m = evalmm.evaluate(model, heads, start_block=k)
        fs = cm.firing_stats(model); cm.set_tracking(model, False); model.train()
        history["eval"].append({"step": step, **m}); history["firing"].append({"step": step, **(list(fs.values())[0] if fs else {})})
        return m, fs

    model.train()
    a0 = do_ww(0); m0, f0 = do_eval(0)
    log(f"[{time.time()-t0:6.0f}s] step 0: alpha {a0} eval {m0['mean_acc']:.4f} coco {m0['coco_R@1']:.3f} firing {list(f0.values())[0] if f0 else '-'} theta={theta:.3f} n_params={n_params}")
    step = 0
    for ep in range(cfg.get("epochs", 3)):
        perm = rng.permutation(n_train)
        for b in range(steps_per_epoch):
            bi = np.sort(perm[b * bs:(b + 1) * bs])
            x = torch.from_numpy(np.asarray(mm[idx[bi]])).float(); yb = torch.from_numpy(y[bi]).long()
            emb = cm.image_embed_from_block(model, x, k)
            logits = 100.0 * F.normalize(emb, dim=-1) @ head.t()
            ce = F.cross_entropy(logits, yb)
            distill = (1 - (F.normalize(emb, dim=-1) * clean_emb[idx[bi]]).sum(-1)).mean()
            loss = ce if objective == "ce" else distill if objective == "distill" else ce + 10 * distill
            opt.zero_grad(); loss.backward()
            if cfg.get("clip_grad"):
                torch.nn.utils.clip_grad_norm_(params, cfg["clip_grad"])
            opt.step(); step += 1
            history["step"].append(step); history["epoch"].append(ep + (b + 1) / steps_per_epoch)
            history["train_loss"].append(float(loss)); history["train_acc"].append(float((logits.argmax(-1) == yb).float().mean()))
            history.setdefault("distill_loss", []).append(float(distill)); history.setdefault("ce_loss", []).append(float(ce))
            if step % ww_every == 0 or step == total_steps:
                a = do_ww(step)
                log(f"[{time.time()-t0:6.0f}s] step {step} ep {ep+1} loss {float(loss):.3f} acc {history['train_acc'][-1]:.2f} alpha {a}")
            if step % eval_every == 0 or step == total_steps:
                m, fs = do_eval(step)
                log(f"[{time.time()-t0:6.0f}s] step {step} eval mean_acc {m['mean_acc']:.4f} imagenette {m['imagenette_acc']:.4f} pets {m['pets_acc']:.4f} cifar {m['cifar10_acc']:.4f} coco R@1 {m['coco_R@1']:.3f} firing {list(fs.values())[0] if fs else '-'}")
    return {"config": cfg, "theta": theta, "n_params_trainable": n_params, "history": history, "seconds": time.time() - t0}
