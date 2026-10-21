"""Analyse E2a: single-block conversion damage versus per-block alpha (and competing predictors).
Writes figures/mm/e2a_damage_vs_alpha.png, figures/mm/e2a_damage_per_block.png and results/mm/e2a_summary.md"""
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

E2A = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mm/e2a"); out = Path("figures/mm"); out.mkdir(parents=True, exist_ok=True)
prof = json.load(open("results/mm/e0_profile.json"))["rows"]
alpha = {}
for r in prof:
    if ".b" in r["name"]:
        tower, blk, role = r["name"].split("."); alpha.setdefault((tower, int(blk[1:])), {})[role] = r
runs = [json.load(open(f)) for f in sorted(glob.glob(str(E2A / "*.json"))) if "baseline" not in f]
base = json.load(open(E2A / "baseline.json"))
lines = [f"# E2a summary ({len(runs)} configs; baseline mean_acc {base['mean_acc']:.4f}, coco R@1 {base['coco_R@1']:.3f})", ""]
METRICS = [("mean_acc", "mean zero-shot acc drop"), ("coco_R@1", "COCO R@1 drop"), ("rel_err_mlp", "relative MLP output error"), ("rel_err_block", "relative block output error")]
PRED = [("mlp_fc1", "alpha"), ("mlp_fc2", "alpha"), ("mlp_fc1", "alpha_weighted"), ("mlp_fc2", "alpha_weighted"), ("mlp_fc1", "log_spectral_norm"), ("mlp_fc2", "log_spectral_norm"),
        ("mlp_fc1", "stable_rank"), ("mlp_fc2", "stable_rank"), ("attn_out", "alpha")]
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for tower, axrow in zip(("image", "text"), axes):
    for variant, T, ax in ((("spike", 8), axrow[0]), (("spike", 4), axrow[1])) if False else []:
        pass
for ti, tower in enumerate(("image", "text")):
    R = [r for r in runs if r["tower"] == tower]
    if not R: continue
    variants = sorted(set((r["variant"], r["T"]) for r in R), key=lambda v: (v[0], v[1] or 0))
    lines += [f"## {tower} tower", "", "| variant | T | " + " | ".join(f"Spearman(drop mean_acc, {p[0]}.{p[1]})" for p in PRED[:6]) + " | rho(mlp_err, fc1 alpha) | rho(drop, frac_pos_preact) | rho(drop, mlp_out_share) | mean drop |", "|" + "---|" * (len(PRED[:6]) + 6)]
    ax = axes[ti][0]
    for variant, T in variants:
        S = sorted([r for r in R if r["variant"] == variant and r["T"] == T], key=lambda r: r["block"])
        blocks = [r["block"] for r in S]; drop = np.array([r["drop"]["mean_acc"] for r in S]); err = np.array([r["rel_err_mlp"] for r in S])
        cells = []
        for role, key in PRED[:6]:
            x = np.array([alpha[(tower, b)][role].get(key) or np.nan for b in blocks], float)
            m = np.isfinite(x); rho, p = spearmanr(x[m], drop[m]) if m.sum() > 4 else (np.nan, np.nan); cells.append(f"{rho:+.2f} (p={p:.2f})")
        x1 = np.array([alpha[(tower, b)]["mlp_fc1"].get("alpha") or np.nan for b in blocks], float)
        r_err = spearmanr(x1, err)[0]; r_pos = spearmanr([r["frac_pos_preact"] for r in S], drop)[0]; r_share = spearmanr([r["mlp_out_share"] for r in S], drop)[0]
        lines.append(f"| {variant} | {T or '-'} | " + " | ".join(cells) + f" | {r_err:+.2f} | {r_pos:+.2f} | {r_share:+.2f} | {drop.mean():+.4f} |")
        lab = f"{variant}" + (f" T={T}" if T else "")
        ax.plot(blocks, drop, marker="o", ms=4, label=lab)
    ax.set_xlabel("converted block"); ax.set_ylabel("drop in mean zero-shot acc"); ax.set_title(f"{tower}: single-block conversion damage"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax2 = axes[ti][1]
    S = sorted([r for r in R if r["variant"] == "spike" and r["T"] == 8], key=lambda r: r["block"])
    if S:
        x = [alpha[(tower, r["block"])]["mlp_fc1"]["alpha"] for r in S]; y = [r["drop"]["mean_acc"] for r in S]
        ax2.scatter(x, y, c=[r["block"] for r in S], cmap="viridis")
        for r, xi, yi in zip(S, x, y): ax2.annotate(str(r["block"]), (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        rho, p = spearmanr(x, y); ax2.set_title(f"{tower}: drop (T=8) vs fc1 alpha, Spearman {rho:+.2f} (p={p:.2f})")
        ax2.set_xlabel("pretrained alpha of mlp_fc1"); ax2.set_ylabel("drop in mean zero-shot acc"); ax2.grid(alpha=0.3)
    lines.append("")
    # per-block table
    lines += ["| block | fc1 alpha | fc2 alpha | fc1 lsn | theta | frac_pos | mlp share | relu drop | cliprelu drop | " + " | ".join(f"spike T={T} drop" for T in (4, 8, 16)) + " | spike T=8 mlp err | silent frac |", "|" + "---|" * 14]
    for b in range(12):
        g = {(r["variant"], r["T"]): r for r in R if r["block"] == b}
        a = alpha[(tower, b)]; s8 = g.get(("spike", 8))
        def d(v): return "-" if v is None else f"{v['drop']['mean_acc']:+.4f}"
        lines.append(f"| {b} | {a['mlp_fc1']['alpha']:.2f} | {a['mlp_fc2']['alpha']:.2f} | {a['mlp_fc1']['log_spectral_norm']:.2f} | {s8['theta'] if s8 else float('nan'):.2f} | {s8['frac_pos_preact'] if s8 else float('nan'):.2f} | {s8['mlp_out_share'] if s8 else float('nan'):.2f} | {d(g.get(('relu', None)))} | {d(g.get(('cliprelu', None)))} | " + " | ".join(d(g.get(("spike", T))) for T in (4, 8, 16)) + f" | {s8['rel_err_mlp'] if s8 else float('nan'):.3f} | {s8['firing']['frac_silent_units'] if s8 else float('nan'):.3f} |")
    lines.append("")
fig.tight_layout(); fig.savefig(out / "e2a_damage_vs_alpha.png", dpi=130)
Path("results/mm/e2a_summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
