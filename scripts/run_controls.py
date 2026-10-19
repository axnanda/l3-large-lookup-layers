"""Control experiments for reading WeightWatcher alpha on lookup / spiking layers.

Trains a small matched set of models (weights saved), then for every analysed weight matrix reports:
  * standard WeightWatcher alpha (min_evals=50, randomize, mp_fit)            -> 'ww'
  * alpha with fix_fingers='clip_xmax' and the number of clipped 'fingers'   -> 'ww_clip'
  * truncated power-law fit (fit='TPL', pl_package='powerlaw')               -> 'ww_tpl'
  * alpha after removing the feature-mean direction of the layer INPUT:
        W' = W (I - m m^T), m = unit vector along the mean input feature       -> 'ww_meanproj'
  * alpha on the live-unit sub-matrix (dead input columns / output rows removed) -> 'ww_live'
  * element tail index (Hill estimator on |W| entries, top 5%) and excess kurtosis
  * the same for the untrained (initial) model                                -> 'init_*'
Model set: MLP3 and a feature-centred MLP3, SNNMLP3 and a centred variant, SNNMLP3 with a steep
surrogate (k_sg=25), L3Classifier and SNNL3Classifier(spiking_memory).
Usage: python3 scripts/run_controls.py --out results/controls [--smoke]
"""
import argparse, json, sys, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from l3lab.sweeps import (RunStore, import_models, log, make_alloc, prepare_tokenized, set_single_thread,
                          subsample_indices, spike_diagnostics)
from l3lab.data import load_fashion_mnist, make_split, flatten_images, to_tensor
from l3lab.train import train_model, to_jsonable
from l3lab.ww_analysis import analyze_model, analyze_matrices, compute_extras

models = import_models("l3lab.models")


# ----------------------------------------------------------------------------- centred variants
class MLP3Centered(models.MLP3):
    """MLP3 whose hidden ReLU features are centred (detached batch mean subtracted) before the next layer."""
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        a1 = F.relu(self.fc1(x)); a1 = a1 - a1.mean(0, keepdim=True).detach()
        a2 = F.relu(self.fc2(a1)); a2 = a2 - a2.mean(0, keepdim=True).detach()
        return self.fc3(a2)


class SNNMLP3Centered(models.SNNMLP3):
    """SNNMLP3 whose spike trains are centred (detached batch mean per time step) before fc2 / fc3."""
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        I1 = self.fc1(x).unsqueeze(0).expand(self.T, -1, -1)
        s1, _ = self.lif1(I1)
        s1c = s1 - s1.mean(1, keepdim=True).detach()
        s2, _ = self.lif2(self.fc2(s1c))
        r2 = s2.mean(0); r2c = r2 - r2.mean(0, keepdim=True).detach()
        self.last_stats = {"rate_1": float(s1.mean()), "rate_2": float(s2.mean()),
                           "silent_1": float((s1.mean((0, 1)) == 0).float().mean()), "silent_2": float((s2.mean((0, 1)) == 0).float().mean())}
        return self.fc3(r2c)


# ----------------------------------------------------------------------------- feature statistics
@torch.no_grad()
def layer_inputs_mlp(model, X, spiking):
    """Return {layer_name: (input features (n, d_in), output activity (n, d_out))} for fc1, fc2."""
    x = X.reshape(X.shape[0], -1)
    if not spiking:
        a1 = F.relu(model.fc1(x)); a2 = F.relu(model.fc2(a1))
        return {"fc1": (x, a1), "fc2": (a1, a2)}
    I1 = model.fc1(x).unsqueeze(0).expand(model.T, -1, -1)
    s1, _ = model.lif1(I1); r1 = s1.mean(0)
    s2, _ = model.lif2(model.fc2(s1)); r2 = s2.mean(0)
    return {"fc1": (x, r1), "fc2": (r1, r2)}


def hill_tail_index(W: torch.Tensor, frac: float = 0.05) -> float:
    a = W.detach().abs().reshape(-1).double().sort(descending=True).values
    k = max(10, int(frac * a.numel()))
    top = a[:k]; xk = a[k]
    if xk <= 0: return float("nan")
    return float(1.0 / (torch.log(top / xk).mean() + 1e-12))


def excess_kurtosis(W: torch.Tensor) -> float:
    w = W.detach().reshape(-1).double(); w = w - w.mean()
    return float((w ** 4).mean() / ((w ** 2).mean() ** 2) - 3.0)


def ww_rows(mats, **kw):
    try:
        rows, summ = analyze_matrices(mats, min_evals=50, randomize=False, mp_fit=False, extras=False, **kw)
        return {r["name"]: {k: r.get(k) for k in ("alpha", "xmin", "D", "num_pl_spikes", "num_fingers", "raw_alpha", "Lambda", "status", "warning", "num_evals")} for r in rows}
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}


def analyses_for_matrices(mats: dict, inputs: dict, acts: dict) -> dict:
    """mats: {name: W (out,in)}; inputs: {name: input features (n,in)}; acts: {name: output activity (n,out)}."""
    out = {}
    out["ww"] = ww_rows(mats)
    out["ww_clip"] = ww_rows(mats, fix_fingers="clip_xmax")
    tpl = ww_rows(mats, fit="TPL", pl_package="powerlaw")
    if not tpl or "error" in tpl or all((v.get("status") != "success") for v in tpl.values() if isinstance(v, dict)):
        tpl = ww_rows(mats, fit="TPL")  # fall back to WeightWatcher's own fitter
    out["ww_tpl"] = tpl
    proj, live, extra = {}, {}, {}
    for name, W in mats.items():
        W = W.detach().double()
        extra[name] = dict(hill_mu=hill_tail_index(W), excess_kurtosis=excess_kurtosis(W), **compute_extras(W.float()))
        if name in inputs:
            m = inputs[name].double().mean(0); mn = m.norm()
            if mn > 0:
                mh = m / mn
                proj[name] = (W - (W @ mh)[:, None] * mh[None, :]).float()
                extra[name]["input_mean_frac_second_moment"] = float((mn ** 2) / (inputs[name].double() ** 2).sum(1).mean())
                extra[name]["energy_along_input_mean"] = float(((W @ mh) ** 2).sum() / (W ** 2).sum())
            act_in = (inputs[name].double() != 0).float().mean(0) if name != "fc1" else torch.ones(W.shape[1])
            live_in = act_in > 0.01
            act_out = (acts[name].double() != 0).float().mean(0) if name in acts else torch.ones(W.shape[0])
            live_out = act_out > 0.01
            extra[name]["frac_live_in"] = float(live_in.float().mean()); extra[name]["frac_live_out"] = float(live_out.float().mean())
            if live_in.sum() >= 50 and live_out.sum() >= 50:
                live[name] = W[live_out][:, live_in].float()
    out["ww_meanproj"] = ww_rows(proj) if proj else {}
    out["ww_live"] = ww_rows(live) if live else {}
    out["extras"] = extra
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/controls"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(); set_single_thread(); torch.manual_seed(args.seed)
    store = RunStore(args.out)
    d = load_fashion_mnist(); x_tr, y_tr, x_val, y_val = make_split(d["x_train"], d["y_train"], n_val=10000, seed=0)
    X, XV = to_tensor(flatten_images(x_tr)), to_tensor(flatten_images(x_val)); Y, YV = torch.as_tensor(y_tr), torch.as_tensor(y_val)
    XT, YT = to_tensor(flatten_images(d["x_test"])), torch.as_tensor(d["y_test"])
    n_feat = 512 if args.smoke else 4000
    ep = (lambda e: 1 if args.smoke else e); sub = (lambda n: 512 if args.smoke else n); h = 128 if args.smoke else 512
    RUNS = {
        "mlp3": dict(kind="mlp", cls=models.MLP3, kw=dict(h=h), epochs=ep(10), subset=sub(60000)),
        "mlp3_centered": dict(kind="mlp", cls=MLP3Centered, kw=dict(h=h), epochs=ep(10), subset=sub(60000)),
        "snnmlp3": dict(kind="snn", cls=models.SNNMLP3, kw=dict(h=h, T=8, beta=0.9, theta=1.0, k=4.0), epochs=ep(10), subset=sub(60000)),
        "snnmlp3_centered": dict(kind="snn", cls=SNNMLP3Centered, kw=dict(h=h, T=8, beta=0.9, theta=1.0, k=4.0), epochs=ep(10), subset=sub(60000)),
        "snnmlp3_k25": dict(kind="snn", cls=models.SNNMLP3, kw=dict(h=h, T=8, beta=0.9, theta=1.0, k=25.0), epochs=ep(10), subset=sub(60000)),
        "l3_classifier": dict(kind="l3", cls=models.L3Classifier, kw=dict(V=512, n_emb=4096, k=8, h=256, d_emb=64, d_up=256, n_layers=2), epochs=ep(6), subset=sub(60000)),
        "snnl3_spiking_memory": dict(kind="l3", cls=models.SNNL3Classifier, kw=dict(V=512, n_emb=4096, k=8, h=256, d_emb=64, d_up=256, n_layers=2, T=8, beta=0.9, theta=1.0, k_sg=4.0, mode="spiking_memory"), epochs=ep(4), subset=sub(10000)),
    }
    tokens = {}
    for name, R in RUNS.items():
        if args.only and name not in args.only: continue
        if store.exists(name): log(f"skip {name}"); continue
        t0 = time.time(); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
        if R["kind"] in ("mlp", "snn"):
            idx = subsample_indices(len(X), R["subset"], rng); Xs, Ys = X[idx], Y[idx]; val = (XV, YV); test = (XT, YT)
            model = R["cls"](**R["kw"])
        else:
            V = R["kw"]["V"]
            if V not in tokens: tokens[V] = prepare_tokenized(V, x_tr, x_val, d["x_test"], seed=0, fit_images=2000 if args.smoke else 20000, cache_dir=Path(args.out) / "tokenizers")
            tk = tokens[V]; idx = subsample_indices(len(tk["ids_train"]), R["subset"], rng)
            Xs, Ys = tk["ids_train"][idx], Y[idx]; val = (tk["ids_val"], YV); test = (tk["ids_test"], YT)
            alloc, _ = make_alloc(tk["counts"], R["kw"]["n_emb"], R["kw"]["k"])
            model = R["cls"](**R["kw"], n_patches=16, n_out=10, alloc=alloc)
        init_model = copy.deepcopy(model)
        hist = train_model(model, Xs, Ys, val[0], val[1], epochs=R["epochs"], batch_size=128, lr=1e-3, weight_decay=0.0, optimizer="adam",
                           seed=args.seed, ww_every=0, test_x=test[0], test_y=test[1], log_every_epoch=False)
        payload = dict(run_id=name, kind=R["kind"], config=dict(R["kw"], epochs=R["epochs"], subset=R["subset"], cls=R["cls"].__name__),
                       final=hist.get("final"), history_epochs=[{k: v for k, v in e.items() if k in ("epoch", "train_acc", "val_acc", "test_acc", "train_loss")} for e in hist.get("epochs", hist.get("history", []))] if isinstance(hist.get("epochs", hist.get("history", [])), list) else None)
        # standard WW rows on trained and initial model
        for tag, m in (("trained", model), ("init", init_model)):
            rows, summ = analyze_model(m, min_evals=50, randomize=True, mp_fit=True)
            payload[f"ww_{tag}_rows"] = rows; payload[f"ww_{tag}_summary"] = summ
        # control analyses on fc1/fc2 (mlp/snn) or on all L3 matrices
        if R["kind"] in ("mlp", "snn"):
            feats = layer_inputs_mlp(model, X[:n_feat], spiking=(R["kind"] == "snn"))
            mats = {"fc1": model.fc1.weight, "fc2": model.fc2.weight}
            inputs = {k: v[0] for k, v in feats.items()}; acts = {k: v[1] for k, v in feats.items()}
            payload["controls_trained"] = analyses_for_matrices(mats, inputs, acts)
            feats0 = layer_inputs_mlp(init_model, X[:n_feat], spiking=(R["kind"] == "snn"))
            payload["controls_init"] = analyses_for_matrices({"fc1": init_model.fc1.weight, "fc2": init_model.fc2.weight},
                                                              {k: v[0] for k, v in feats0.items()}, {k: v[1] for k, v in feats0.items()})
            if R["kind"] == "snn":
                payload["rates"] = {k: dict(mean_rate=float(v[1].mean()), silent=float((v[1].mean(0) == 0).float().mean())) for k, v in feats.items()}
        else:
            mats = {n: mod.weight for n, mod, role in model.ww_layers() if mod.weight.dim() == 2 and min(mod.weight.shape) >= 50}
            payload["controls_trained"] = analyses_for_matrices(mats, {}, {})
            try: payload["spike_diagnostics"] = spike_diagnostics(model)
            except Exception: pass
        torch.save(model.state_dict(), Path(args.out) / f"{name}_final.pt"); torch.save(init_model.state_dict(), Path(args.out) / f"{name}_init.pt")
        store.save(name, to_jsonable(payload))
        f = hist.get("final", {}); log(f"done {name}: val {f.get('val_acc')} test {f.get('test_acc')} alpha_avg {payload['ww_trained_summary'].get('alpha_avg')} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
