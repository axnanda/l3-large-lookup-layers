#!/usr/bin/env python3
"""Random sweep of spiking models (SNNMLP3 and SNNL3Classifier) on Fashion-MNIST with WeightWatcher logging.

Extra config dimensions (spec): T in {4,8,16}, k_sg (surrogate slope) in {1,2,4,8}, beta in {0.5,0.9},
mode in {rate_query, spiking_memory} (SNNL3Classifier only).  SNNMLP3 samples h, lr, wd, batch, epochs,
train_subset; SNNL3Classifier uses the L3 config space (V, n_emb, k, h, d_emb, n_layers, ...).
One JSON per run under <out>/runs/ (field "model" says which), resumable.

    python3 scripts/run_snn_sweep.py --n_runs 60 --out results/snn_sweep --model_type both
    python3 scripts/run_snn_sweep.py --smoke --out results/smoke_snn
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
from l3lab.sweeps import (L3_SPACE, Budget, RunStore, choice, construct, import_models, log, log_uniform,  # noqa: E402
                          make_alloc, make_payload, prepare_tokenized, run_id_for, sample_l3_arch,
                          sample_l3_training, set_single_thread, spike_diagnostics, subsample_indices,
                          summarize_run_line)
from l3lab.train import train_model, write_json  # noqa: E402

SNN_SPACE = dict(T=[4, 8, 16], k_sg=[1.0, 2.0, 4.0, 8.0], beta=[0.5, 0.9], theta=1.0,
                 mode=["rate_query", "spiking_memory"])
MLP_SPACE = dict(h=[64, 128, 256, 512], lr=(1e-4, 1e-2), weight_decay=[0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                 batch_size=[32, 64, 128, 256], train_subset=[60000, 10000, 2000])
SPACE = dict(snn=SNN_SPACE, snn_mlp3=MLP_SPACE, snn_l3=L3_SPACE)
MODEL_TYPES = ("snn_mlp3", "snn_l3")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_runs", type=int, default=60)
    p.add_argument("--out", default="results/snn_sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs_max", type=int, default=10)
    p.add_argument("--max_minutes", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true", help="2 runs (one of each model), 512 samples, 1 epoch")
    p.add_argument("--model_type", choices=("both",) + MODEL_TYPES, default="both")
    p.add_argument("--model_module", default="l3lab.models")
    p.add_argument("--data_root", default=None)
    p.add_argument("--n_val", type=int, default=10000)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--tok_fit_images", type=int, default=20000)
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
    if args.model_type == "both":
        mtype = MODEL_TYPES[i % 2] if args.smoke else str(choice(rng, MODEL_TYPES))
    else:
        mtype = args.model_type
    cfg = dict(index=int(i), model_type=mtype, T=int(choice(rng, SNN_SPACE["T"])),
               k_sg=float(choice(rng, SNN_SPACE["k_sg"])), beta=float(choice(rng, SNN_SPACE["beta"])),
               theta=float(SNN_SPACE["theta"]))
    if mtype == "snn_mlp3":
        cfg.update(model="SNNMLP3", h=int(choice(rng, MLP_SPACE["h"])), lr=log_uniform(rng, *MLP_SPACE["lr"]),
                   weight_decay=float(choice(rng, MLP_SPACE["weight_decay"])),
                   batch_size=int(choice(rng, MLP_SPACE["batch_size"])), optimizer="adam", momentum=0.9,
                   epochs=int(rng.integers(1, int(args.epochs_max) + 1)),
                   train_subset=int(choice(rng, MLP_SPACE["train_subset"])))
    else:
        cfg.update(model="SNNL3Classifier", mode=str(choice(rng, SNN_SPACE["mode"])))
        # budget: spiking L3 costs ~96 s per 10k samples per epoch (T=8) on one CPU thread, so the
        # spiking-L3 arm uses smaller training subsets, fewer epochs and T in {4, 8} (overridable by args)
        cfg.update(T=int(choice(rng, [4, 8])))
        cfg.update(sample_l3_arch(rng, L3_SPACE))
        l3_space = dict(L3_SPACE, train_subset=[10000, 5000, 2000])
        cfg.update(sample_l3_training(rng, min(int(args.epochs_max), 4), l3_space))
    cfg["seed"] = int(args.seed * 100003 + i)
    if args.smoke:
        cfg.update(epochs=1, train_subset=512, T=4)
        if mtype == "snn_l3":
            cfg.update(V=256, n_emb=1024, k=8, n_layers=1)
        else:
            cfg.update(h=min(cfg["h"], 256))
    return cfg


def load_images(args):
    d = ldata.load_fashion_mnist(args.data_root)
    xt, yt, xv, yv = ldata.make_split(d["x_train"], d["y_train"], args.n_val, args.split_seed)
    xte, yte = d["x_test"], d["y_test"]
    if args.smoke:
        xv, yv, xte, yte = xv[:1000], yv[:1000], xte[:1000], yte[:1000]
    return dict(x_train=xt, y_train=yt, x_val=xv, y_val=yv, x_test=xte, y_test=yte)


class TokenCache:
    def __init__(self, imgs: dict, args, cache_dir):
        self.imgs, self.args, self.cache_dir, self.mem = imgs, args, Path(cache_dir), {}

    def get(self, V: int) -> dict:
        if V not in self.mem:
            fit_images = min(self.args.tok_fit_images, 2000) if self.args.smoke else self.args.tok_fit_images
            self.mem[V] = prepare_tokenized(V, self.imgs["x_train"], self.imgs["x_val"], self.imgs["x_test"],
                                            seed=self.args.tok_seed, fit_images=fit_images,
                                            cache_dir=self.cache_dir, split_seed=self.args.split_seed)
        return self.mem[V]


def run_one(run_id: str, cfg: dict, models, tokens: TokenCache, imgs: dict, args, store, budget) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    extra = {}
    if cfg["model_type"] == "snn_mlp3":
        idx = subsample_indices(len(imgs["x_train"]), cfg["train_subset"], rng)
        X = ldata.to_tensor(ldata.flatten_images(imgs["x_train"][idx]))
        XV, XT = ldata.to_tensor(ldata.flatten_images(imgs["x_val"])), ldata.to_tensor(ldata.flatten_images(imgs["x_test"]))
        torch.manual_seed(cfg["seed"])
        model = construct(models.SNNMLP3, h=cfg["h"], T=cfg["T"], beta=cfg["beta"], theta=cfg["theta"], k=cfg["k_sg"],
                          d_in=784, n_out=10)
    else:
        tk = tokens.get(cfg["V"])
        idx = subsample_indices(len(tk["ids_train"]), cfg["train_subset"], rng)
        X, XV, XT = tk["ids_train"][idx], tk["ids_val"], tk["ids_test"]
        alloc, alloc_stats = make_alloc(tk["counts"], cfg["n_emb"], cfg["k"])
        extra["alloc_stats"] = alloc_stats
        torch.manual_seed(cfg["seed"])
        model = construct(models.SNNL3Classifier, V=cfg["V"], n_emb=cfg["n_emb"], k=cfg["k"], h=cfg["h"],
                          d_emb=cfg["d_emb"], d_up=cfg["d_up"], n_layers=cfg["n_layers"],
                          n_patches=cfg.get("n_patches", 16), n_out=10, alloc=alloc, T=cfg["T"], beta=cfg["beta"],
                          theta=cfg["theta"], k_sg=cfg["k_sg"], mode=cfg["mode"])
    Y, YV, YT = ldata.to_tensor(imgs["y_train"][idx]), ldata.to_tensor(imgs["y_val"]), ldata.to_tensor(imgs["y_test"])
    ww_every = args.ww_every if args.ww_every >= 0 else (1 if cfg["epochs"] <= args.ww_every_max_epochs else 0)
    ww_kwargs = dict(min_evals=args.min_evals, randomize=not args.no_randomize, mp_fit=not args.no_mp_fit)
    out_dir = store.out_dir / "models" / run_id if args.save_models else None
    t0 = time.time()
    hist = train_model(model, X, Y, XV, YV, epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                       weight_decay=cfg["weight_decay"], optimizer=cfg["optimizer"], momentum=cfg["momentum"],
                       seed=cfg["seed"], log_every_epoch=not args.quiet_epochs, ww_every=ww_every,
                       ww_at_init=(ww_every == 1), ww_final=True, ww_kwargs=ww_kwargs, out_dir=out_dir,
                       test_x=XT, test_y=YT, stop_fn=budget.exhausted, log_fn=log)
    # spike-rate diagnostics after a forward pass on a validation batch (best effort)
    try:
        model.eval()
        with torch.no_grad():
            model(XV[:256])
        extra["spike_diagnostics"] = spike_diagnostics(model)
    except Exception as e:  # noqa: BLE001
        extra["spike_diagnostics"] = {"error": repr(e)}
    extra.update(n_train_actual=int(len(idx)), run_wall=time.time() - t0, ww_every_used=ww_every)
    return make_payload(run_id, cfg["model"], cfg, hist, extra)


def main(argv=None):
    args = parse_args(argv)
    set_single_thread()
    if args.smoke:
        args.n_runs = min(args.n_runs, 2) if args.n_runs != parse_args([]).n_runs else 2
        args.epochs_max = 1
    try:
        models = import_models(args.model_module, allow_standin=False)
        for name in ("SNNMLP3", "SNNL3Classifier"):
            getattr(models, name)
    except (ImportError, AttributeError) as e:
        log(f"ERROR: cannot load spiking models from {args.model_module}: {e}")
        return 2
    log(f"models module: {getattr(models, '__file__', models)}")
    store = RunStore(args.out)
    budget = Budget(args.max_minutes)
    write_json(store.out_dir / "sweep_args.json", dict(args=vars(args), space=SPACE, script="run_snn_sweep.py"))
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
        payload = run_one(run_id, cfg, models, tokens, imgs, args, store, budget)
        store.save(run_id, payload)
        last_run_sec = time.time() - t_run
        n_done += 1
        log(summarize_run_line(run_id, cfg, payload["history"], keys=("model", "h", "T", "k_sg", "beta", "lr", "epochs", "train_subset")))
    log(f"finished: {n_done} new runs, {n_skipped} skipped (existing), elapsed {budget.elapsed_min():.1f} min, runs dir {store.runs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
