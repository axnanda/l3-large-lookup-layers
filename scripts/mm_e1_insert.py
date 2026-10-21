"""E1: spiking insert fine-tuning on pretrained CLIP (image tower), resumable.
Usage: PYTHONPATH=. python3 scripts/mm_e1_insert.py [--only id1,id2] [--threads 2] [--out results/mm/e1]"""
import argparse, json, sys, time
from pathlib import Path
from l3lab.multimodal.finetune import run_insert

BASE = dict(tower="image", block=6, variant="spike_replace", T=8, k_sg=4.0, beta=1.0, reset="soft", r=64, scope="block",
            n_train=3000, epochs=3, lr=3e-4, batch=64, seed=0, ww_every=24, label_noise=0.0, objective="distill")
CONFIGS = {
    "sr_b6_k4":      dict(),
    "dr_b6":         dict(variant="dense_replace"),
    "sr_b6_k1":      dict(k_sg=1.0),
    "sr_b6_k10":     dict(k_sg=10.0),
    "sr_b6_k25":     dict(k_sg=25.0),
    "sr_b6_T4":      dict(T=4),
    "sr_b6_T16":     dict(T=16),
    "sa_b6":         dict(variant="spike_adapter", lr=1e-3),
    "da_b6":         dict(variant="dense_adapter", lr=1e-3),
    "sr_b6_k4_s1":   dict(seed=1),
    "sr_b10_k4":     dict(block=10),
    "dr_b10":        dict(block=10, variant="dense_replace"),
    "sr_b6_later":   dict(scope="later", lr=3e-5),
    "sr_b6_ce":      dict(objective="ce"),
    "dr_b6_ce":      dict(variant="dense_replace", objective="ce"),
    "sr_b6_noise":   dict(objective="ce", label_noise=0.2),
    "sr_b6_leak":    dict(beta=0.8),
    "sr_b6_nobc":    dict(bias_correct=False),
}
ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--out", default="results/mm/e1")
args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
ids = [i for i in args.only.split(",") if i] or list(CONFIGS)
for rid in ids:
    p = out / f"{rid}.json"
    if p.exists(): print("skip", rid); continue
    cfg = {**BASE, **CONFIGS[rid], "threads": args.threads, "run_id": rid}
    print("=== run", rid, cfg, flush=True)
    res = run_insert(cfg, log=lambda m: print(f"{rid} {m}", flush=True))
    json.dump(res, open(p, "w"))
    print("saved", p, f"{res['seconds']:.0f}s", flush=True)
