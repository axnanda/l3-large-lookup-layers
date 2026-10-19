"""Summarise results/controls/runs/*.json into results/controls/summary.md (+ csv)."""
import json, glob, sys, math
from pathlib import Path
out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/controls")
def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d
def fmt(x, nd=2):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))
lines = ["# Control experiments: what moves alpha?", "",
         "Columns: alpha at init / after training (WeightWatcher, min_evals=50), warning, alpha after clipping fingers (fix_fingers=clip_xmax), ",
         "alpha after projecting out the layer's input-mean direction (W(I - m m^T)), alpha of the live-unit sub-matrix, energy of W along the input mean, ",
         "overlap of the top right-singular vector with the column-mean direction, top-singular-value energy, fraction of live input units, Hill tail index of |W| entries, excess kurtosis of entries, xmin and number of eigenvalues used by the PL fit.", "",
         "| model | val acc | train acc | layer | a_init | a_trained | warning | a_clip | a_meanproj | a_live | E_along_mean | ovl_mean | top_sv_E | live_in | hill_mu | kurt | xmin | n_pl |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
rows_csv = ["model,val_acc,train_acc,layer,alpha_init,alpha_trained,warning,alpha_clip,alpha_meanproj,alpha_live,energy_along_input_mean,mean_dir_overlap_right,top_sv_energy,frac_live_in,hill_mu,excess_kurtosis,xmin,num_pl_spikes"]
for f in sorted(glob.glob(str(out_dir / "runs" / "*.json"))):
    d = json.load(open(f)); name = d["run_id"]; val = g(d, "final", "val_acc"); tr = g(d, "final", "train_acc")
    rows_tr = {r["name"]: r for r in d["ww_trained_rows"]}; rows_in = {r["name"]: r for r in d["ww_init_rows"]}; ct = d.get("controls_trained", {})
    for lname, r in rows_tr.items():
        if (r.get("num_evals") or 0) < 50: continue
        ex = g(ct, "extras", lname, default={}) or {}
        vals = [name, fmt(val, 3), fmt(tr, 3), lname, fmt(g(rows_in, lname, "alpha")), fmt(r.get("alpha")), r.get("warning") or "", fmt(g(ct, "ww_clip", lname, "alpha")),
                fmt(g(ct, "ww_meanproj", lname, "alpha")), fmt(g(ct, "ww_live", lname, "alpha")), fmt(ex.get("energy_along_input_mean"), 3), fmt(ex.get("mean_dir_overlap_right")),
                fmt(ex.get("top_sv_energy"), 3), fmt(ex.get("frac_live_in")), fmt(ex.get("hill_mu"), 1), fmt(ex.get("excess_kurtosis")), fmt(r.get("xmin"), 3), fmt(r.get("num_pl_spikes"), 0)]
        lines.append("| " + " | ".join(vals) + " |"); rows_csv.append(",".join(v.replace(",", ";") for v in vals))
    if "rates" in d: lines.append(f"| {name} |  |  | spike rates | " + "; ".join(f"{k}: rate {v['mean_rate']:.2f}, silent {v['silent']:.2f}" for k, v in d["rates"].items()) + " | | | | | | | | | | | | | |")
(out_dir / "summary.md").write_text("\n".join(lines) + "\n"); (out_dir / "summary.csv").write_text("\n".join(rows_csv) + "\n")
print("\n".join(lines[:4])); print(f"wrote {out_dir/'summary.md'} with {len(lines)-7} rows")
