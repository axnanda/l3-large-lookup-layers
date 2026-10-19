"""Plot alpha trajectories (per layer) with train/val accuracy from results/trajectories/runs/*.json."""
import json, glob, sys, math
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
in_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/trajectories"); out = Path(sys.argv[2] if len(sys.argv) > 2 else "figures"); out.mkdir(exist_ok=True)
files = sorted(glob.glob(str(in_dir / "runs" / "*.json")))
if not files: sys.exit("no runs")
fig, axes = plt.subplots(2, len(files), figsize=(4.2 * len(files), 7), squeeze=False)
summary = ["| run | epochs | best val acc (epoch) | final train / val acc | alpha_avg at best-val epoch | alpha_avg final | epoch alpha_avg first < 2 | layers |", "|---|---|---|---|---|---|---|---|"]
for j, f in enumerate(files):
    d = json.load(open(f)); h = d["history"]; name = d["run_id"]
    ep = h["epoch"]; tr = h["train_acc"]; va = h["val_acc"]
    ww = h.get("ww", [])
    ax = axes[0, j]; ax.plot(ep, tr, label="train acc"); ax.plot(ep, va, label="val acc")
    if "test_acc" in h: ax.plot(ep, h["test_acc"], label="test acc", ls=":")
    ax.set_title(name, fontsize=9); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax = axes[1, j]
    per_layer = {}
    for e in ww:
        for r in e["rows"]:
            if (r.get("num_evals") or 0) >= 50 and r.get("alpha") is not None and 0 < r["alpha"] < 12:
                per_layer.setdefault(r["name"], []).append((e["epoch"], r["alpha"]))
    for lname, pts in per_layer.items():
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=".", label=lname)
    avg = [(e["epoch"], e["summary"].get("alpha_avg")) for e in ww if e["summary"].get("alpha_avg") is not None]
    if avg: ax.plot([a[0] for a in avg], [a[1] for a in avg], "k-", lw=2, label="alpha_avg")
    ax.axhline(2, color="gray", ls="--", lw=1); ax.set_xlabel("epoch"); ax.set_ylabel("alpha"); ax.set_ylim(0, 8); ax.grid(alpha=0.3); ax.legend(fontsize=6)
    best_i = max(range(len(va)), key=lambda i: va[i]); best_ep = ep[best_i]
    a_best = next((a for e, a in avg if e == best_ep), None); a_final = avg[-1][1] if avg else None
    first_lt2 = next((e for e, a in avg if a is not None and a < 2), None)
    summary.append(f"| {name} | {len(ep)} | {va[best_i]:.3f} ({best_ep}) | {tr[-1]:.3f} / {va[-1]:.3f} | {a_best if a_best is None else round(a_best,2)} | {None if a_final is None else round(a_final,2)} | {first_lt2} | {', '.join(per_layer.keys())} |")
plt.tight_layout(); plt.savefig(out / "trajectories_alpha.png", dpi=110)
(in_dir / "summary.md").write_text("\n".join(summary) + "\n"); print("\n".join(summary))
