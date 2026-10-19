#!/usr/bin/env python3
"""Random sweep of L3Classifier (patch-token input) on Fashion-MNIST with WeightWatcher logging.

Config space (spec): V in {256,512,1024}, n_emb in {2048,4096,8192}, k in {4,8,16} (with V <= n_emb <= V*k),
h in {128,256}, d_emb in {32,64,128}, n_layers in {1,2,3}, lr log-uniform [1e-4,1e-2], weight decay, batch,
epochs uniform [1, --epochs_max], train_subset in {60k,10k,2k}.  The PatchVQTokenizer is fitted once per V
(cached under <out>/tokenizers/).  One JSON per run under <out>/runs/, resumable.

    python3 scripts/run_l3_sweep.py --n_runs 60 --out results/l3_sweep --seed 0 --max_minutes 180
    python3 scripts/run_l3_sweep.py --smoke --out results/smoke_l3
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from l3lab import data as ldata  # noqa: E402
from l3lab.sweeps import (L3_SPACE, Budget, RunStore, construct, import_models, log, make_alloc,  # noqa: E402
                          make_payload, prepare_tokenized, run_id_for, sample_l3_arch, sample_l3_training,
                          set_single_thread, subsample_indices, summarize_run_line)
from l3lab.train import train_model, write_json  # noqa: E402

SPACE = dict(L3_SPACE)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_runs", type=int, default=60)
    p.add_argument("--out", default="results/l3_sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs_max", type=int, default=10)
    p.add_argument("--max_minutes", type=float, default=0.0, help="wall-clock budget (0 = unlimited)")
    p.add_argument("--smoke", action="store_true", help="2 runs, 512 training samples, 1 epoch, tiny tokenizer fit")
    p.add_argument("--model_module", default="l3lab.models", help="dotted module or .py path providing L3Classifier")
    p.add_argument("--data_root", default=None)
    p.add_argument("--n_val", type=int, default=10000)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--tok_fit_images", type=int, default=20000, help="images used to fit the VQ tokenizer")
    p.add_argument("--tok_seed", type=int, default=0)
    p.add_argument("--ww_every_max_epochs", type=int, default=10)
    p.add_argument("--ww_every", type=int, default=-1)
    p.add_argument("--min_evals", type=int, default=50)
    p.add_argument("--no_randomize", action="store_true")
    p.add_argument("--no_mp_fit", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--save_models", action="store_true")
    p.add_argument("--quiet_epochs", action="store_true")
    return p.parse_args(argv)


def sample_config(rng: np.random.Generator, i: int, args) -> dict:
    cfg = dict(index=int(i), model="L3Classifier")
    cfg.update(sample_l3_arch(rng, SPACE))
    cfg.update(sample_l3_training(rng, args.epochs_max, SPACE))
    cfg["seed"] = int(args.seed * 100003 + i)
    if args.smoke:
        cfg.update(epochs=1, train_subset=512, V=256, n_emb=1024, k=8, n_layers=min(cfg["n_layers"], 2))
    return cfg


def load_images(args):
    d = ldata.load_fashion_mnist(args.data_root)
    xt, yt, xv, yv = ldata.make_split(d["x_train"], d["y_train"], args.n_val, args.split_seed)
    xte, yte = d["x_test"], d["y_test"]
    if args.smoke:
        xv, yv, xte, yte = xv[:1000], yv[:1000], xte[:1000], yte[:1000]
    return dict(x_train=xt, y_train=yt, x_val=xv, y_val=yv, x_test=xte, y_test=yte)


class TokenCache:
    """Per-V tokenised data (fit once, cache in memory + on disk)."""

    def __init__(self, imgs: dict, args, cache_dir):
        self.imgs, self.args, self.cache_dir, self.mem = imgs, args, Path(cache_dir), {}

    def get(self, V: int) -> dict:
        if V not in self.mem:
            fit_images = min(self.args.tok_fit_images, 2000) if self.args.smoke else self.args.tok_fit_images
            self.mem[V] = prepare_tokenized(V, self.imgs["x_train"], self.imgs["x_val"], self.imgs["x_test"],
                                            seed=self.args.tok_seed, fit_images=fit_images,
                                            cache_dir=self.cache_dir, split_seed=self.args.split_seed)
        return self.mem[V]


def build_l3_model(L3Classifier, cfg: dict, counts):
    alloc, alloc_stats = make_alloc(counts, cfg["n_emb"], cfg["k"])
    torch.manual_seed(cfg["seed"])
    model = construct(L3Classifier, V=cfg["V"], n_emb=cfg["n_emb"], k=cfg["k"], h=cfg["h"], d_emb=cfg["d_emb"],
                      d_up=cfg["d_up"], n_layers=cfg["n_layers"], n_patches=cfg.get("n_patches", 16), n_out=10,
                      alloc=alloc)
    return model, alloc_stats


def run_one(run_id: str, cfg: dict, L3Classifier, tokens: TokenCache, imgs: dict, args, store, budget) -> dict:
    tk = tokens.get(cfg["V"])
    rng = np.random.default_rng(cfg["seed"])
    idx = subsample_indices(len(tk["ids_train"]), cfg["train_subset"], rng)
    X, Y = tk["ids_train"][idx], ldata.to_tensor(imgs["y_train"][idx])
    XV, YV = tk["ids_val"], ldata.to_tensor(imgs["y_val"])
    XT, YT = tk["ids_test"], ldata.to_tensor(imgs["y_test"])
    model, alloc_stats = build_l3_model(L3Classifier, cfg, tk["counts"])
    ww_every = args.ww_every if args.ww_every >= 0 else (1 if cfg["epochs"] <= args.ww_every_max_epochs else 0)
    ww_kwargs = dict(min_evals=args.min_evals, randomize=not args.no_randomize, mp_fit=not args.no_mp_fit)
    out_dir = store.out_dir / "models" / run_id if args.save_models else None
    t0 = time.time()
    hist = train_model(model, X, Y, XV, YV, epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                       weight_decay=cfg["weight_decay"], optimizer=cfg["optimizer"], momentum=cfg["momentum"],
                       seed=cfg["seed"], log_every_epoch=not args.quiet_epochs, ww_every=ww_every,
                       ww_at_init=(ww_every == 1), ww_final=True, ww_kwargs=ww_kwargs, out_dir=out_dir,
                       test_x=XT, test_y=YT, stop_fn=budget.exhausted, log_fn=log)
    extra = dict(n_train_actual=int(len(idx)), alloc_stats=alloc_stats, tokenizer_fit_time=tk.get("fit_time"),
                 run_wall=time.time() - t0, ww_every_used=ww_every)
    return make_payload(run_id, "L3Classifier", cfg, hist, extra)


def main(argv=None):
    args = parse_args(argv)
    set_single_thread()
    if args.smoke:
        args.n_runs = min(args.n_runs, 2) if args.n_runs != parse_args([]).n_runs else 2
        args.epochs_max = 1
    try:
        models = import_models(args.model_module, allow_standin=False)
        L3Classifier = getattr(models, "L3Classifier")
    except (ImportError, AttributeError) as e:
        log(f"ERROR: cannot load L3Classifier from {args.model_module}: {e}")
        return 2
    log(f"models module: {getattr(models, '__file__', models)}")
    store = RunStore(args.out)
    budget = Budget(args.max_minutes)
    write_json(store.out_dir / "sweep_args.json", dict(args=vars(args), space=SPACE, script="run_l3_sweep.py"))
    t0 = time.time()
    imgs = load_images(args)
    log(f"data: train={len(imgs['x_train'])} val={len(imgs['x_val'])} test={len(imgs['x_test'])} ({time.time() - t0:.1f}s)")
    tokens = TokenCache(imgs, args, store.out_dir / "tokenizers")
    rng = np.random.default_rng(args.seed)
    configs = [sample_config(rng, i, args) for i in range(args.n_runs)]
    n_done = n_skipped = 0
    last_run_sec = 0.0
    for i, cfg in enumerate(configs):
        if i < args.start:
            continue
        run_id = run_id_for(args.seed, i)
        if store.exists(run_id):
            n_skipped += 1
            continue
        if budget.exhausted(last_run_sec):
            log(f"budget of {args.max_minutes} min reached after {n_done} new runs; stopping")
            break
        log(f"=== {run_id} ({i + 1}/{args.n_runs}) " + " ".join(
            f"{k}={v if not isinstance(v, float) else round(v, 6)}" for k, v in cfg.items() if k not in ("index", "seed")))
        t_run = time.time()
        payload = run_one(run_id, cfg, L3Classifier, tokens, imgs, args, store, budget)
        store.save(run_id, payload)
        last_run_sec = time.time() - t_run
        n_done += 1
        log(summarize_run_line(run_id, cfg, payload["history"], keys=("V", "n_emb", "k", "h", "n_layers", "lr", "epochs", "train_subset")))
    log(f"finished: {n_done} new runs, {n_skipped} skipped (existing), elapsed {budget.elapsed_min():.1f} min, runs dir {store.runs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
