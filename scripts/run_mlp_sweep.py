#!/usr/bin/env python3
"""Random sweep of 3-layer MLPs (MLP3) on Fashion-MNIST with WeightWatcher logging.

In the spirit of Charles Martin's "1000 variations of a 3-layer MLP" post: each run samples
h, lr, weight decay, batch size, optimizer, epochs, init gain, dropout, label noise and the
training-subset size, trains MLP3, runs WeightWatcher at the end (and every epoch, incl. at init,
when epochs <= --ww_every_max_epochs) and writes one JSON per run to <out>/runs/.  Resumable.

    python3 scripts/run_mlp_sweep.py --n_runs 200 --out results/mlp_sweep --seed 0 --max_minutes 120
    python3 scripts/run_mlp_sweep.py --smoke --out results/smoke_mlp          # 2 tiny runs
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import inspect
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from l3lab import data as ldata  # noqa: E402
from l3lab.sweeps import (Budget, RunStore, apply_init_gain, choice, construct, corrupt_labels,  # noqa: E402
                          import_models, log, log_uniform, make_payload, run_id_for, set_single_thread,
                          subsample_indices, summarize_run_line)
from l3lab.train import evaluate, train_model, write_json  # noqa: E402

SPACE = dict(
    h=[64, 128, 256, 512, 1024], lr=(1e-4, 3e-2), weight_decay=[0.0, 1e-5, 1e-4, 1e-3, 1e-2],
    batch_size=[32, 64, 128, 256], optimizer=["adam", "sgd"], init_gain=[0.5, 1.0, 2.0],
    dropout=[0.0, 0.2, 0.5], label_noise=[0.0, 0.1, 0.3], train_subset=[60000, 10000, 2000],
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_runs", type=int, default=100)
    p.add_argument("--out", default="results/mlp_sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs_max", type=int, default=20)
    p.add_argument("--max_minutes", type=float, default=0.0, help="wall-clock budget (0 = unlimited)")
    p.add_argument("--smoke", action="store_true", help="2 runs, 512 training samples, 1 epoch")
    p.add_argument("--model_module", default="l3lab.models",
                   help="dotted module or .py path providing MLP3 (smoke falls back to tests/standin_models.py)")
    p.add_argument("--data_root", default=None, help="dir with the Fashion-MNIST idx-gz files")
    p.add_argument("--n_val", type=int, default=10000)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--ww_every_max_epochs", type=int, default=10, help="WW every epoch when epochs <= this")
    p.add_argument("--ww_every", type=int, default=-1, help="override: WW every N epochs (0 = final only)")
    p.add_argument("--min_evals", type=int, default=50)
    p.add_argument("--no_randomize", action="store_true", help="skip WW randomize (faster)")
    p.add_argument("--no_mp_fit", action="store_true", help="skip WW MP fit (faster)")
    p.add_argument("--start", type=int, default=0, help="first run index")
    p.add_argument("--save_models", action="store_true", help="save final state_dicts under <out>/models/")
    p.add_argument("--quiet_epochs", action="store_true", help="do not print per-epoch lines")
    return p.parse_args(argv)


def sample_config(rng: np.random.Generator, i: int, args) -> dict:
    cfg = dict(
        index=int(i), model="MLP3",
        h=int(choice(rng, SPACE["h"])), lr=log_uniform(rng, *SPACE["lr"]),
        weight_decay=float(choice(rng, SPACE["weight_decay"])), batch_size=int(choice(rng, SPACE["batch_size"])),
        optimizer=str(choice(rng, SPACE["optimizer"])), momentum=0.9,
        epochs=int(rng.integers(1, int(args.epochs_max) + 1)),
        init_gain=float(choice(rng, SPACE["init_gain"])), dropout=float(choice(rng, SPACE["dropout"])),
        label_noise=float(choice(rng, SPACE["label_noise"])), train_subset=int(choice(rng, SPACE["train_subset"])),
        seed=int(args.seed * 100003 + i),
    )
    if args.smoke:
        cfg.update(epochs=1, train_subset=512, h=min(cfg["h"], 256))
    return cfg


def load_data(args):
    d = ldata.load_fashion_mnist(args.data_root)
    xt, yt, xv, yv = ldata.make_split(d["x_train"], d["y_train"], args.n_val, args.split_seed)
    D = dict(X=ldata.to_tensor(ldata.flatten_images(xt)), Y=ldata.to_tensor(yt),
             XV=ldata.to_tensor(ldata.flatten_images(xv)), YV=ldata.to_tensor(yv),
             XT=ldata.to_tensor(ldata.flatten_images(d["x_test"])), YT=ldata.to_tensor(d["y_test"]))
    if args.smoke:
        D["XV"], D["YV"], D["XT"], D["YT"] = D["XV"][:1000], D["YV"][:1000], D["XT"][:1000], D["YT"][:1000]
    return D


def run_one(run_id: str, cfg: dict, MLP3, D: dict, args, store: RunStore, budget: Budget) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    idx = subsample_indices(len(D["X"]), cfg["train_subset"], rng)
    X, Y_clean = D["X"][idx], D["Y"][idx]
    y_noisy, noisy_mask = corrupt_labels(Y_clean.numpy(), cfg["label_noise"], rng, n_classes=10)
    Y = torch.from_numpy(y_noisy)
    torch.manual_seed(cfg["seed"])
    # init_gain: pass it to the constructor when the model supports it (l3lab.models.MLP3 does),
    # otherwise rescale the default-initialised Linear weights ourselves (stand-in model).
    accepts_gain = "init_gain" in inspect.signature(MLP3).parameters
    model = construct(MLP3, h=cfg["h"], d_in=784, n_out=10, dropout=cfg["dropout"],
                      **({"init_gain": cfg["init_gain"]} if accepts_gain else {}))
    if not accepts_gain:
        apply_init_gain(model, cfg["init_gain"])
    if args.ww_every >= 0:
        ww_every = args.ww_every
    else:
        ww_every = 1 if cfg["epochs"] <= args.ww_every_max_epochs else 0
    ww_kwargs = dict(min_evals=args.min_evals, randomize=not args.no_randomize, mp_fit=not args.no_mp_fit)
    out_dir = store.out_dir / "models" / run_id if args.save_models else None
    t0 = time.time()
    hist = train_model(model, X, Y, D["XV"], D["YV"], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                       lr=cfg["lr"], weight_decay=cfg["weight_decay"], optimizer=cfg["optimizer"],
                       momentum=cfg["momentum"], seed=cfg["seed"], log_every_epoch=not args.quiet_epochs,
                       ww_every=ww_every, ww_at_init=(ww_every == 1), ww_final=True, ww_kwargs=ww_kwargs,
                       out_dir=out_dir, test_x=D["XT"], test_y=D["YT"], stop_fn=budget.exhausted, log_fn=log)
    clean_loss, clean_acc = evaluate(model, X, Y_clean)
    hist["final"]["train_acc_clean"] = clean_acc
    hist["final"]["train_loss_clean"] = clean_loss
    extra = dict(n_train_actual=int(len(idx)), n_noisy_labels=int(noisy_mask.sum()),
                 run_wall=time.time() - t0, ww_every_used=ww_every)
    return make_payload(run_id, "MLP3", cfg, hist, extra)


def main(argv=None):
    args = parse_args(argv)
    set_single_thread()
    if args.smoke:
        args.n_runs = min(args.n_runs, 2) if args.n_runs != parse_args([]).n_runs else 2
        args.epochs_max = 1
    try:
        models = import_models(args.model_module, allow_standin=args.smoke)
    except ImportError as e:
        log(f"ERROR: {e}")
        return 2
    MLP3 = getattr(models, "MLP3")
    log(f"models module: {getattr(models, '__file__', models)}  MLP3={MLP3.__module__}.{MLP3.__name__}")
    store = RunStore(args.out)
    budget = Budget(args.max_minutes)
    write_json(store.out_dir / "sweep_args.json", dict(args=vars(args), space=SPACE, script="run_mlp_sweep.py"))
    t0 = time.time()
    D = load_data(args)
    log(f"data: train={len(D['X'])} val={len(D['XV'])} test={len(D['XT'])} ({time.time() - t0:.1f}s)")
    rng = np.random.default_rng(args.seed)
    configs = [sample_config(rng, i, args) for i in range(args.n_runs)]  # deterministic given --seed
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
        log(f"=== {run_id} ({i + 1}/{args.n_runs}) " + " ".join(f"{k}={v if not isinstance(v, float) else round(v, 6)}"
                                                           for k, v in cfg.items() if k not in ("index", "seed")))
        t_run = time.time()
        payload = run_one(run_id, cfg, MLP3, D, args, store, budget)
        store.save(run_id, payload)
        last_run_sec = time.time() - t_run
        n_done += 1
        log(summarize_run_line(run_id, cfg, payload["history"], keys=("h", "lr", "epochs", "train_subset", "label_noise")))
    log(f"finished: {n_done} new runs, {n_skipped} skipped (existing), elapsed {budget.elapsed_min():.1f} min, "
        f"runs dir {store.runs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
