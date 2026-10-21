"""Analyse E2b: cumulative conversion in different orders (conversion-only and, if present, after fine-tune).
Writes figures/mm/e2b_orders.png and results/mm/e2b_summary.md"""
import glob, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

runs = [json.load(open(f)) for f in sorted(glob.glob("results/mm/e2b/*.json"))]
base = runs[0]["baseline"] if runs else None
e2a = {}
for f in glob.glob("results/mm/e2a/image_*_spike_bc_T16.json"):
    r = json.load(open(f)); e2a[r["block"]] = r["drop"]["mean_acc"]
orders = sorted(set(r["order"] for r in runs))
lines = [f"# E2b summary ({len(runs)} configs, T = {runs[0]['T'] if runs else '-'}; clean mean_acc {base['mean_acc']:.4f}, COCO R@1 {base['coco_R@1']:.3f})", "",
         "| order | m | blocks | mean block alpha | conv mean_acc | drop | sum of single-block drops (E2a, T=16) | COCO R@1 | imagenette | pets | cifar | mean silent frac |" + (" ft mean_acc | ft drop |" if any("finetuned" in r for r in runs) else ""),
         "|" + "---|" * (12 + (2 if any("finetuned" in r for r in runs) else 0))]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for order in orders:
    R = sorted([r for r in runs if r["order"] == order], key=lambda r: r["m"])
    for r in R:
        c = r["conversion_only"]; sil = np.mean([v["frac_silent_units"] for v in r["firing"].values()]) if r["firing"] else float("nan")
        add = sum(e2a.get(b, 0) for b in r["blocks"])
        row = f"| {order} | {r['m']} | {r['blocks']} | {np.mean(r['block_alpha']):.2f} | {c['mean_acc']:.4f} | {r['drop_conversion']:+.4f} | {add:+.4f} | {c['coco_R@1']:.3f} | {c['imagenette_acc']:.3f} | {c['pets_acc']:.3f} | {c['cifar10_acc']:.3f} | {sil:.3f} |"
        if "finetuned" in r: row += f" {r['finetuned']['mean_acc']:.4f} | {r['drop_finetuned']:+.4f} |"
        lines.append(row)
    ms = [0] + [r["m"] for r in R]; acc = [base["mean_acc"]] + [r["conversion_only"]["mean_acc"] for r in R]; coco = [base["coco_R@1"]] + [r["conversion_only"]["coco_R@1"] for r in R]
    axes[0].plot(ms, acc, marker="o", label=order); axes[1].plot(ms, coco, marker="o", label=order)
axes[0].set_xlabel("number of converted image blocks"); axes[0].set_ylabel("mean zero-shot acc (3 sets)"); axes[0].set_title("cumulative conversion, by order"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
axes[1].set_xlabel("number of converted image blocks"); axes[1].set_ylabel("COCO image-to-text R@1"); axes[1].grid(alpha=0.3)
fig.tight_layout(); Path("figures/mm").mkdir(parents=True, exist_ok=True); fig.savefig("figures/mm/e2b_orders.png", dpi=130)
Path("results/mm/e2b_summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
