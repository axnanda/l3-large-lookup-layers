"""Which statistic best predicts a block's conversion damage / repairability? Screens every WeightWatcher
column and the activation statistics against the E2a damage targets, per tower and pooled (rank-based),
with leave-one-out validation of 1- and 2-feature rank predictors. Writes results/mm/statistics_selection.md"""
import glob, json, itertools
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

prof = json.load(open("results/mm/e0_profile.json"))["rows"]
WW = ["alpha", "alpha_weighted", "log_norm", "log_spectral_norm", "log_alpha_norm", "stable_rank", "mp_softrank", "num_spikes", "num_rand_spikes", "rand_distance", "D", "xmin", "xmax", "sigma", "entropy", "rank_loss", "top_sv_energy", "effective_rank", "spectral_gap", "mean_dir_overlap_right", "frac_dead_rows"]
ROLES = ["mlp_fc1", "mlp_fc2", "attn_q", "attn_k", "attn_v", "attn_out"]
feat = {}
for r in prof:
    if ".b" not in r["name"]: continue
    tower, blk, role = r["name"].split("."); b = int(blk[1:])
    for k in WW:
        v = r.get(k)
        if v is not None: feat.setdefault((tower, b), {})[f"{role}.{k}"] = float(v)
# block-level combos
for key, d in feat.items():
    for k in WW:
        a, c = d.get(f"mlp_fc1.{k}"), d.get(f"mlp_fc2.{k}")
        if a is not None and c is not None: d[f"mlp_mean.{k}"] = (a + c) / 2
    d["depth"] = key[1]
# activation stats + targets from E2a
tgt = {}
for f in glob.glob("results/mm/e2a/*.json"):
    if "baseline" in f: continue
    r = json.load(open(f)); key = (r["tower"], r["block"]); t = tgt.setdefault(key, {})
    if r["variant"] == "relu": t["damage_relu"] = r["drop"]["mean_acc"]; t["mlperr_relu"] = r["rel_err_mlp"]
    if r["variant"] == "spike_bc" and r["T"] == 4:
        t["damage_spike4"] = r["drop"]["mean_acc"]; t["coco_drop_spike4"] = r["drop"]["coco_R@1"]; t["mlperr_spike4"] = r["rel_err_mlp"]
        feat[key]["act.frac_pos"] = r["frac_pos_preact"]; feat[key]["act.preact_std"] = r["preact_std"]; feat[key]["act.mlp_out_share"] = r["mlp_out_share"]
        feat[key]["act.theta_median"] = r["theta"]; feat[key]["act.theta_max"] = r.get("theta_max", np.nan); feat[key]["act.silent_frac"] = r["firing"]["frac_silent_units"]
    if r["variant"] == "spike_bc" and r["T"] == 16: t["damage_spike16"] = r["drop"]["mean_acc"]
TARGETS = ["damage_relu", "damage_spike4", "damage_spike16", "mlperr_spike4", "coco_drop_spike4"]
keys = sorted(k for k in feat if k in tgt)
fnames = sorted(set(n for k in keys for n in feat[k]))
lines = ["# Statistic selection: what predicts a block's conversion damage? (24 CLIP blocks: 12 image + 12 text)", "",
         "Spearman rank correlation between each per-block statistic and the damage of converting that block's MLP (E2a). 'pooled' ranks within each tower first (so tower offsets cancel) and then correlates over 24 blocks. Positive = larger statistic, larger damage.", ""]
rows = []
for n in fnames:
    rec = {"feature": n}
    for tname in TARGETS:
        for tower in ("image", "text", "pooled"):
            ks = [k for k in keys if (tower == "pooled" or k[0] == tower) and n in feat[k] and tname in tgt[k]]
            if len(ks) < 6: rec[(tname, tower)] = np.nan; continue
            x = np.array([feat[k][n] for k in ks]); y = np.array([tgt[k][tname] for k in ks])
            if tower == "pooled":
                xr, yr = np.zeros(len(ks)), np.zeros(len(ks))
                for tw in ("image", "text"):
                    idx = [i for i, k in enumerate(ks) if k[0] == tw]
                    if len(idx) > 1: xr[idx] = rankdata(x[idx]) / len(idx); yr[idx] = rankdata(y[idx]) / len(idx)
                x, y = xr, yr
            rec[(tname, tower)] = spearmanr(x, y)[0] if np.std(x) > 0 else np.nan
    rows.append(rec)
hdr = "| feature | " + " | ".join(f"{t} ({tw})" for t in TARGETS for tw in ("img", "txt", "pool")) + " |"
lines += [hdr, "|" + "---|" * (1 + 3 * len(TARGETS))]
def score(rec): return np.nanmean([abs(rec[(t, "pooled")]) for t in ("damage_relu", "damage_spike4", "damage_spike16", "mlperr_spike4")])
rows.sort(key=lambda r: -np.nan_to_num(score(r)))
for rec in rows:
    lines.append(f"| {rec['feature']} | " + " | ".join("" if np.isnan(rec[(t, tw)]) else f"{rec[(t, tw)]:+.2f}" for t in TARGETS for tw in ("image", "text", "pooled")) + " |")
# leave-one-out validation of 1- and 2-feature rank-regression predictors for damage_spike4 (pooled) and damage_relu
lines += ["", "## Leave-one-out validation (rank regression, pooled 24 blocks)", "", "Predict the within-tower damage rank from 1 or 2 feature ranks with least squares; score = Spearman between held-out predictions and truth over the 24 blocks.", ""]
def loo(tname, fs):
    ks = [k for k in keys if all(f in feat[k] for f in fs) and tname in tgt[k]]
    if len(ks) < 12: return np.nan
    X = np.zeros((len(ks), len(fs))); y = np.zeros(len(ks))
    for tw in ("image", "text"):
        idx = [i for i, k in enumerate(ks) if k[0] == tw]
        for j, f in enumerate(fs): X[idx, j] = rankdata([feat[ks[i]][f] for i in idx]) / len(idx)
        y[idx] = rankdata([tgt[ks[i]][tname] for i in idx]) / len(idx)
    Xb = np.hstack([X, np.ones((len(ks), 1))]); pred = np.zeros(len(ks))
    for i in range(len(ks)):
        m = np.ones(len(ks), bool); m[i] = False
        w, *_ = np.linalg.lstsq(Xb[m], y[m], rcond=None); pred[i] = Xb[i] @ w
    return spearmanr(pred, y)[0]
cands = ["mlp_fc1.alpha", "mlp_fc2.alpha", "mlp_mean.alpha", "mlp_fc1.alpha_weighted", "mlp_fc1.log_spectral_norm", "mlp_fc2.log_spectral_norm", "mlp_fc1.stable_rank", "mlp_fc2.stable_rank", "mlp_fc1.mp_softrank", "mlp_fc1.num_spikes", "mlp_fc1.rand_distance", "mlp_fc1.entropy", "mlp_fc2.entropy", "mlp_fc1.D", "act.frac_pos", "act.mlp_out_share", "act.preact_std", "act.theta_max", "depth"]
for tname in ("damage_relu", "damage_spike4", "damage_spike16", "mlperr_spike4"):
    res1 = sorted(((loo(tname, [f]), f) for f in cands), key=lambda t: -np.nan_to_num(t[0]))[:6]
    res2 = sorted(((loo(tname, list(p)), p) for p in itertools.combinations(cands, 2)), key=lambda t: -np.nan_to_num(t[0]))[:6]
    lines += [f"### target {tname}", "", "1 feature: " + ", ".join(f"{f} {s:+.2f}" for s, f in res1), "", "2 features: " + ", ".join(f"{p[0]}+{p[1]} {s:+.2f}" for s, p in res2), ""]
# E1 repairability: nothing to select (alpha inert) -> report firing / loss signals
lines += ["## Notes", "", "* Targets: damage_* = drop in mean zero-shot accuracy when that block alone is converted (relu: no bias correction; spike4/16: per-channel thresholds + bias correction, T = 4 / 16); mlperr = relative MLP output error; coco_drop = R@1 drop.", "* 12 blocks per tower: |rho| > 0.58 is p < 0.05 for one tower; the pooled rank correlation over 24 blocks has p < 0.05 at |rho| > 0.40.", "* 'depth' is included as the confound every weight statistic must beat."]
Path("results/mm/statistics_selection.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
