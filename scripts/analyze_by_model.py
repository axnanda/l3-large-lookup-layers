"""Per-model and per-layer-role analysis of the WeightWatcher sweeps (complements analyze_sweeps.py).

Writes <out>/by_model.md with, for every model type:
  * Spearman rho of alpha summaries (all layers / dense layers only / lookup tables only) vs train, val acc and gap
  * accuracy and gap of runs with average alpha < 2 vs >= 2 (Martin's observation 1)
  * fraction of runs in [2, 3] that are 'good' (val acc within 0.03 of the model's best)   (observation 2)
  * for runs with per-epoch WW: how often alpha falls while val acc does not rise         (observation 3)
  * SNN-specific: alpha of fc1/fc2 vs surrogate slope k_sg, T, beta; dead units vs alpha
Usage: python3 scripts/analyze_by_model.py --in results/mlp_sweep results/l3_sweep results/snn_sweep --out figures/final
"""
import argparse, glob, json, math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

DENSE = {"mlp", "l3_wup", "l3_wmix"}; LOOKUP = {"embed", "l3_wk", "l3_wv"}


def role_alpha(rows, roles, min_evals=50):
    a = [r["alpha"] for r in rows if r.get("role") in roles and (r.get("num_evals") or 0) >= min_evals
         and r.get("alpha") is not None and r.get("status", "success") == "success" and 0 < r["alpha"] < 8]
    return float(np.mean(a)) if a else None


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5: return (float("nan"), float("nan"), int(m.sum()))
    r, p = spearmanr(x[m], y[m]); return (float(r), float(p), int(m.sum()))


def fmt(v, nd=2):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", nargs="+", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    runs = []
    for d in args.inp:
        for f in sorted(glob.glob(str(Path(d) / "runs" / "*.json"))):
            r = json.load(open(f)); rows = r.get("ww_rows", []); fin = r.get("final", {}) or {}
            cfg = r.get("config", {})
            runs.append(dict(model=r.get("model"), run_id=r.get("run_id"), cfg=cfg, train=fin.get("train_acc"), val=fin.get("val_acc"), test=fin.get("test_acc"),
                             gap=fin.get("gap"), a_all=role_alpha(rows, DENSE | LOOKUP | {"head"}), a_dense=role_alpha(rows, DENSE),
                             a_lookup=role_alpha(rows, LOOKUP), a_fc1=role_alpha([x for x in rows if x.get("name") == "fc1"], {"mlp"}),
                             a_fc2=role_alpha([x for x in rows if x.get("name") == "fc2"], {"mlp"}),
                             dead=np.mean([x.get("frac_dead_rows") or 0 for x in rows]) if rows else None,
                             traj=r.get("alpha_trajectory"), hist=r.get("history"), spikes=r.get("spike_diagnostics")))
    lines = [f"# Per-model analysis ({len(runs)} runs)", ""]
    for model in sorted(set(x["model"] for x in runs)):
        R = [x for x in runs if x["model"] == model]
        lines += [f"## {model} (n = {len(R)})", "",
                  "| alpha summary | rho vs train acc (p) | rho vs val acc (p) | rho vs gap (p) | n |", "|---|---|---|---|---|"]
        for key, label in [("a_all", "all analysed layers"), ("a_dense", "dense layers (mlp, w_up, w_mix)"), ("a_lookup", "lookup tables (embed, w_k, w_v)"), ("a_fc1", "fc1"), ("a_fc2", "fc2")]:
            xs = [x[key] for x in R]
            if all(v is None for v in xs): continue
            rt = rho(xs, [x["train"] for x in R]); rv = rho(xs, [x["val"] for x in R]); rg = rho(xs, [x["gap"] for x in R])
            lines.append(f"| {label} | {fmt(rt[0])} ({fmt(rt[1], 3)}) | {fmt(rv[0])} ({fmt(rv[1], 3)}) | {fmt(rg[0])} ({fmt(rg[1], 3)}) | {rv[2]} |")
        # observation 1: alpha < 2 vs >= 2 (dense-layer alpha)
        key = "a_dense" if any(x["a_dense"] is not None for x in R) else "a_all"
        lo = [x for x in R if x[key] is not None and x[key] < 2]; hi = [x for x in R if x[key] is not None and x[key] >= 2]
        best = max((x["val"] for x in R if x["val"] is not None), default=float("nan"))
        def stats(S):
            if not S: return "n=0"
            v = np.array([x["val"] for x in S], float); g = np.array([x["gap"] for x in S], float); t = np.array([x["train"] for x in S], float)
            return f"n={len(S)}, val {np.nanmean(v):.3f} (min {np.nanmin(v):.3f}, max {np.nanmax(v):.3f}), train {np.nanmean(t):.3f}, gap {np.nanmean(g):+.3f} (max {np.nanmax(g):+.3f})"
        lines += ["", f"Observation 1 (alpha < 2 => degraded / over-fit), using {key} (best val acc of this model: {best:.3f}):",
                  f"- alpha < 2: {stats(lo)}", f"- alpha >= 2: {stats(hi)}"]
        mid = [x for x in R if x[key] is not None and 2 <= x[key] <= 3]
        good_mid = [x for x in mid if x["val"] is not None and x["val"] >= best - 0.03]
        lo_good = [x for x in lo if x["val"] is not None and x["val"] >= best - 0.03]
        lines += [f"Observation 2 (alpha in [2,3] can still be good): {len(good_mid)}/{len(mid)} runs in [2,3] are within 0.03 of the best val acc; {len(lo_good)}/{len(lo)} runs with alpha < 2 are."]
        # observation 3: alpha falls while val does not improve (per-epoch trajectories)
        n_tr, n_fall_noimp, n_fall = 0, 0, 0
        for x in R:
            tr, h = x["traj"], x["hist"]
            if not tr or not h or len(tr) < 4 or "val_acc" not in h: continue
            n_tr += 1
            al = [e.get("alpha_min50_avg") for e in tr if e.get("epoch", 0) >= 1]; ep = [e["epoch"] for e in tr if e.get("epoch", 0) >= 1]
            va = h["val_acc"]
            for i in range(1, len(al)):
                if al[i] is None or al[i - 1] is None: continue
                if al[i] < al[i - 1] - 0.02:
                    n_fall += 1
                    e0, e1 = ep[i - 1] - 1, ep[i] - 1
                    if 0 <= e0 < len(va) and 0 <= e1 < len(va) and va[e1] <= va[e0] + 0.002: n_fall_noimp += 1
        if n_tr: lines.append(f"Observation 3 (alpha falls without val improvement): among {n_fall} epoch-to-epoch alpha drops (> 0.02) in {n_tr} runs with per-epoch WW, {n_fall_noimp} ({100 * n_fall_noimp / max(n_fall, 1):.0f} %) came with no val-acc gain.")
        # SNN specifics
        if model.startswith("SNN"):
            lines += ["", "| factor | value | n | mean alpha_dense | mean alpha fc1 | mean val acc | mean dead-row frac |", "|---|---|---|---|---|---|---|"]
            for fac in ("k_sg", "T", "beta", "mode", "h"):
                vals = sorted(set(x["cfg"].get(fac) for x in R if x["cfg"].get(fac) is not None), key=lambda v: (str(type(v)), v))
                for v in vals:
                    S = [x for x in R if x["cfg"].get(fac) == v]
                    ad = [x["a_dense"] for x in S if x["a_dense"] is not None]; a1 = [x["a_fc1"] for x in S if x["a_fc1"] is not None]
                    lines.append(f"| {fac} | {v} | {len(S)} | {fmt(np.mean(ad) if ad else None)} | {fmt(np.mean(a1) if a1 else None)} | {fmt(np.mean([x['val'] for x in S]), 3)} | {fmt(np.mean([x['dead'] for x in S if x['dead'] is not None]) if S else None, 3)} |")
            rd = rho([x["dead"] for x in R], [x["a_dense"] for x in R]); lines.append(f"\nSpearman(dead-row fraction, alpha_dense) = {fmt(rd[0])} (p {fmt(rd[1], 3)}, n {rd[2]})")
        lines.append("")
    (out / "by_model.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
