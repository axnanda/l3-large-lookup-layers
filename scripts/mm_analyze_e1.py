"""Analyse E1: alpha trajectories of spiking / dense inserts during fine-tuning versus recovery of the
zero-shot and retrieval metrics.  Writes figures/mm/e1_trajectories.png and results/mm/e1_summary.md"""
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

E1 = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mm/e1"); out = Path("figures/mm"); out.mkdir(parents=True, exist_ok=True)
runs = {Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(E1 / "*.json")))}
base = json.load(open("results/mm/e2a/baseline.json")) if Path("results/mm/e2a/baseline.json").exists() else None


def alpha_series(r, name_suffix):
    """alpha trajectory of the matrix ``<tower>.b<block>.<name_suffix>`` of the insert block only."""
    blk = r["config"]["block"]; tower = r["config"].get("tower", "image"); full = f"{tower}.b{blk:02d}.{name_suffix}"
    xs, ys = [], []
    for w in r["history"]["ww"]:
        for row in w["rows"]:
            if row["name"] == full and row.get("alpha") is not None and row["alpha"] > 0:
                xs.append(w["step"]); ys.append(row["alpha"])
    return np.array(xs), np.array(ys)


def later_alpha_change(r):
    """For scope=later runs: mean |alpha change| of the later blocks' matrices."""
    if r["config"].get("scope") != "later": return None
    first, last = r["history"]["ww"][0]["rows"], r["history"]["ww"][-1]["rows"]
    a0 = {x["name"]: x["alpha"] for x in first if x.get("alpha")}; a1 = {x["name"]: x["alpha"] for x in last if x.get("alpha")}
    d = [abs(a1[k] - a0[k]) for k in a0 if k in a1 and a0[k] > 0 and a1[k] > 0]
    return (float(np.mean(d)), float(np.max(d)), len(d)) if d else None


def eval_series(r, key):
    return np.array([e["step"] for e in r["history"]["eval"]]), np.array([e[key] for e in r["history"]["eval"]])


lines = [f"# E1 summary ({len(runs)} runs)" + (f"; clean baseline mean_acc {base['mean_acc']:.4f}, coco R@1 {base['coco_R@1']:.3f}" if base else ""), "",
         "| run | variant | block | T | k_sg | scope | objective | mean_acc step0 | mean_acc final | imagenette final | pets final | coco R@1 final | fc1 alpha 0 -> end | fc2 alpha 0 -> end | insert alpha 0 -> end | rho(alpha_insert, mean_acc) | silent frac end | mean rate end | minutes |",
         "|" + "---|" * 19]
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
for rid, r in runs.items():
    c = r["config"]; ev0, evN = r["history"]["eval"][0], r["history"]["eval"][-1]
    s1, a1 = alpha_series(r, "mlp_fc1"); s2, a2 = alpha_series(r, "mlp_fc2")
    ins_name = "adapter_down" if "adapter" in c["variant"] else "mlp_fc1"
    si, ai = alpha_series(r, ins_name)
    se, me = eval_series(r, "mean_acc")
    # correlate alpha at eval steps with mean_acc (nearest alpha measurement)
    rho = np.nan
    if len(si) > 3 and len(se) > 3:
        ai_at = np.interp(se, si, ai); rho = spearmanr(ai_at, me)[0]
    f = r["history"]["firing"][-1] if r["history"]["firing"] else {}
    def fmt(a): return f"{a[0]:.2f} -> {a[-1]:.2f}" if len(a) else "-"
    lc = later_alpha_change(r)
    if lc: lines.append(f"| ({rid}: later blocks' alpha changes: mean {lc[0]:.3f}, max {lc[1]:.3f} over {lc[2]} matrices) |" + " |" * 18)
    lines.append(f"| {rid} | {c['variant']} | {c['block']} | {c['T']} | {c['k_sg']} | {c['scope']} | {c.get('objective', 'distill')} | {ev0['mean_acc']:.4f} | {evN['mean_acc']:.4f} | {evN['imagenette_acc']:.4f} | {evN['pets_acc']:.4f} | {evN['coco_R@1']:.3f} | {fmt(a1)} | {fmt(a2)} | {fmt(ai)} | {rho:+.2f} | {f.get('frac_silent_units', float('nan')):.3f} | {f.get('mean_rate', float('nan')):.3f} | {r['seconds']/60:.0f} |")
    col = "C3" if c["variant"].startswith("dense") else None
    axes[0][0].plot(s1, a1, label=rid, color=col, alpha=0.8); axes[0][1].plot(s2, a2, label=rid, color=col, alpha=0.8)
    axes[0][2].plot(se, me, label=rid, color=col, alpha=0.8, marker="o", ms=3)
    if "adapter" in c["variant"]:
        axes[1][0].plot(si, ai, label=rid, marker=".", alpha=0.8)
    axes[1][1].plot([x["step"] for x in r["history"]["firing"]], [x.get("frac_silent_units") or 0 for x in r["history"]["firing"]], label=rid, marker="o", ms=3, alpha=0.8)
    axes[1][2].plot(r["history"]["step"], np.convolve(r["history"]["train_loss"], np.ones(8) / 8, mode="same"), label=rid, alpha=0.7)
for ax, t in zip(axes.flat, ("alpha of mlp_fc1 (block with the insert)", "alpha of mlp_fc2", "mean zero-shot acc (3 sets)", "alpha of adapter_down (adapter runs)", "fraction of silent spiking units", "train loss (smoothed)")):
    ax.set_title(t, fontsize=10); ax.grid(alpha=0.3); ax.set_xlabel("step")
axes[0][0].legend(fontsize=6, ncol=2)
fig.suptitle("E1: spiking / dense inserts in pretrained CLIP image block, fine-tuned on Imagenette (distillation to the clean embedding)")
fig.tight_layout(); fig.savefig(out / "e1_trajectories.png", dpi=130)
# k_sg dependence table
lines += ["", "## Surrogate slope and T dependence (spike_replace, block 6)", "", "| run | k_sg | T | fc1 alpha end | fc2 alpha end | fc1 alpha change | mean_acc final | recovery (final - step0) | silent frac |", "|---|---|---|---|---|---|---|---|---|"]
for rid, r in runs.items():
    c = r["config"]
    if c["variant"] != "spike_replace" or c["block"] != 6: continue
    _, a1 = alpha_series(r, "mlp_fc1"); _, a2 = alpha_series(r, "mlp_fc2"); ev0, evN = r["history"]["eval"][0], r["history"]["eval"][-1]; f = r["history"]["firing"][-1]
    lines.append(f"| {rid} | {c['k_sg']} | {c['T']} | {a1[-1]:.2f} | {a2[-1]:.2f} | {a1[-1]-a1[0]:+.2f} | {evN['mean_acc']:.4f} | {evN['mean_acc']-ev0['mean_acc']:+.4f} | {f.get('frac_silent_units', float('nan')):.3f} |")
Path("results/mm/e1_summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
