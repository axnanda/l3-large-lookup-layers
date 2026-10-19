"""Alpha trajectories: train a few representative models for many epochs with WeightWatcher every epoch.

Runs (each saved as results/trajectories/runs/<name>.json):
  mlp3_h512_clean      MLP3 h=512, 50k train, 40 epochs, Adam 1e-3            (well-trained -> over-trained?)
  mlp3_h512_noise30    same with 30% label noise                                 (over-fit regime)
  mlp3_h1024_sgd       MLP3 h=1024, SGD 0.05, 40 epochs                          (slow, under- to well-trained)
  l3_2layer            L3Classifier V=512 n_emb=4096 k=8 h=256 d_emb=64, 2 layers, 50k, 12 epochs
  snnmlp3_h512         SNNMLP3 h=512 T=8 k_sg=4, 50k, 20 epochs
  snnl3_rate_query     SNNL3Classifier rate_query  V=512 n_emb=4096 k=8 h=256, 10k subset, T=8, 6 epochs
  snnl3_spiking_memory SNNL3Classifier spiking_memory (same)
Usage: python3 scripts/run_trajectories.py --out results/trajectories [--only name ...] [--max_minutes M]
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from l3lab.sweeps import (Budget, RunStore, corrupt_labels, import_models, log, make_alloc, make_payload,
                          prepare_tokenized, set_single_thread, spike_diagnostics, subsample_indices)
from l3lab.data import load_fashion_mnist, make_split, flatten_images, to_tensor
from l3lab.train import train_model

RUNS = {
    "mlp3_h512_clean": dict(model="MLP3", h=512, lr=1e-3, optimizer="adam", epochs=40, batch_size=128, weight_decay=0.0, train_subset=60000, label_noise=0.0),
    "mlp3_h512_noise30": dict(model="MLP3", h=512, lr=1e-3, optimizer="adam", epochs=40, batch_size=128, weight_decay=0.0, train_subset=60000, label_noise=0.3),
    "mlp3_h1024_sgd": dict(model="MLP3", h=1024, lr=0.05, optimizer="sgd", epochs=40, batch_size=128, weight_decay=0.0, train_subset=60000, label_noise=0.0),
    "l3_2layer": dict(model="L3Classifier", V=512, n_emb=4096, k=8, h=256, d_emb=64, d_up=256, n_layers=2, lr=1e-3, optimizer="adam", epochs=12, batch_size=128, weight_decay=0.0, train_subset=60000),
    "snnmlp3_h512": dict(model="SNNMLP3", h=512, T=8, beta=0.9, theta=1.0, k_sg=4.0, lr=1e-3, optimizer="adam", epochs=20, batch_size=128, weight_decay=0.0, train_subset=60000, label_noise=0.0),
    "snnl3_rate_query": dict(model="SNNL3Classifier", mode="rate_query", V=512, n_emb=4096, k=8, h=256, d_emb=64, d_up=256, n_layers=2, T=8, beta=0.9, theta=1.0, k_sg=4.0, lr=1e-3, optimizer="adam", epochs=6, batch_size=128, weight_decay=0.0, train_subset=10000),
    "snnl3_spiking_memory": dict(model="SNNL3Classifier", mode="spiking_memory", V=512, n_emb=4096, k=8, h=256, d_emb=64, d_up=256, n_layers=2, T=8, beta=0.9, theta=1.0, k_sg=4.0, lr=1e-3, optimizer="adam", epochs=6, batch_size=128, weight_decay=0.0, train_subset=10000),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/trajectories"); ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max_minutes", type=float, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_mp_fit", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 512 samples per run, for testing")
    args = ap.parse_args()
    set_single_thread()
    store = RunStore(args.out); budget = Budget(args.max_minutes)
    models = import_models("l3lab.models")
    d = load_fashion_mnist(); x_tr, y_tr, x_val, y_val = make_split(d["x_train"], d["y_train"], n_val=10000, seed=0)
    X, XV, XT = to_tensor(flatten_images(x_tr)), to_tensor(flatten_images(x_val)), to_tensor(flatten_images(d["x_test"]))
    Y, YV, YT = torch.as_tensor(y_tr), torch.as_tensor(y_val), torch.as_tensor(d["y_test"])
    tokens = {}
    for name, cfg in RUNS.items():
        if args.only and name not in args.only: continue
        if store.exists(name): log(f"skip {name} (exists)"); continue
        if budget.exhausted(): log("budget exhausted"); break
        torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
        if args.smoke: cfg = dict(cfg, epochs=1, train_subset=512, h=min(cfg.get("h", 256), 256))
        ww_kwargs = dict(min_evals=50, randomize=True, mp_fit=not args.no_mp_fit)
        t0 = time.time()
        if cfg["model"] in ("MLP3", "SNNMLP3"):
            idx = subsample_indices(len(X), cfg["train_subset"], rng)
            Xs, Ys = X[idx], Y[idx]
            if cfg.get("label_noise", 0) > 0:
                _cl = corrupt_labels(Ys.numpy(), cfg["label_noise"], rng, n_classes=10)
                Ys = torch.as_tensor(np.asarray(_cl[0] if isinstance(_cl, tuple) else _cl))
            kw = dict(h=cfg["h"]) if cfg["model"] == "MLP3" else dict(h=cfg["h"], T=cfg["T"], beta=cfg["beta"], theta=cfg["theta"], k=cfg["k_sg"])
            model = getattr(models, cfg["model"])(**kw)
            to_input = None
        else:
            V = cfg["V"]
            if V not in tokens:
                tokens[V] = prepare_tokenized(V, x_tr, x_val, d["x_test"], seed=0, fit_images=20000, cache_dir=Path(args.out) / "tokenizers")
            tk = tokens[V]
            idx = subsample_indices(len(tk["ids_train"]), cfg["train_subset"], rng)
            Xs, Ys = tk["ids_train"][idx], Y[idx]; XVt, XTt = tk["ids_val"], tk["ids_test"]
            alloc, alloc_stats = make_alloc(tk["counts"], cfg["n_emb"], cfg["k"])
            common = dict(V=V, n_emb=cfg["n_emb"], k=cfg["k"], h=cfg["h"], d_emb=cfg["d_emb"], d_up=cfg["d_up"], n_layers=cfg["n_layers"], n_patches=16, n_out=10, alloc=alloc)
            if cfg["model"] == "L3Classifier": model = models.L3Classifier(**common)
            else: model = models.SNNL3Classifier(**common, T=cfg["T"], beta=cfg["beta"], theta=cfg["theta"], k_sg=cfg["k_sg"], mode=cfg["mode"])
            to_input = None
        val_x, val_y = (XV, YV) if cfg["model"] in ("MLP3", "SNNMLP3") else (XVt, YV)
        test_x, test_y = (XT, YT) if cfg["model"] in ("MLP3", "SNNMLP3") else (XTt, YT)
        hist = train_model(model, Xs, Ys, val_x, val_y, epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"], weight_decay=cfg["weight_decay"], optimizer=cfg["optimizer"], seed=args.seed, ww_every=1, ww_kwargs=ww_kwargs, test_x=test_x, test_y=test_y, to_input=to_input, ww_at_init=True, stop_fn=(lambda *a, **k: budget.exhausted()))
        payload = dict(run_id=name, model=cfg["model"], config=cfg, history=hist, wall=time.time() - t0)
        try: payload["spike_diagnostics"] = spike_diagnostics(model)
        except Exception: pass
        store.save(name, payload)
        f = hist.get("final", {}); log(f"done {name}: train {f.get('train_acc')} val {f.get('val_acc')} test {f.get('test_acc')} in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
