"""Analyse E3 adapters: mean +- std over seeds of in-domain test accuracy, transfer retained and end-of-training
alpha per matrix; alpha-vs-accuracy correlations.  Writes results/mm/e3_summary.md, figures/mm/e3_adapters.png"""
import glob, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

runs = [json.load(open(f)) for f in sorted(glob.glob("results/mm/e3/*.json"))]
groups = {}
for r in runs:
    groups.setdefault((r["train"], r["shots"], r["variant"]), []).append(r)


def final_alpha(r):
    return {x["name"]: x["alpha"] for x in r["history"]["ww"][-1]["rows"] if x["alpha"] is not None and x["alpha"] > 0}


def transfer(r):
    t = r["history"]["transfer"][-1]; key = r["test"].split("_")[0] + "_acc"
    accs = [v for k, v in t.items() if k.endswith("_acc") and k != key]
    return float(np.mean(accs)), t["coco_R@1"]


lines = [f"# E3 summary ({len(runs)} runs, 3 seeds, 40 epochs, Adam 1e-3)", "",
         "| train set | shots | adapter | params | zero-shot test | test acc (mean +- std) | best epoch acc | transfer acc retained (other 2 sets) | COCO R@1 | end alphas (mean over seeds) | memory firing |", "|" + "---|" * 11]
rows_for_corr = []
for (tr, shots, var), R in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
    test = np.array([r["history"]["test_acc"][-1] for r in R]); best = np.array([max(r["history"]["test_acc"]) for r in R])
    zs = np.array([r["zero_shot"][r["test"].split("_")[0] + "_acc"] for r in R]); tra = np.array([transfer(r)[0] for r in R]); coco = np.array([transfer(r)[1] for r in R])
    al = {}
    for r in R:
        for k, v in final_alpha(r).items(): al.setdefault(k, []).append(v)
    mem = [r["history"]["mem_rate"][-1] for r in R if r["history"]["mem_rate"][-1] is not None]
    lines.append(f"| {tr} | {shots or 'all'} | {var} | {R[0]['n_params']/1e3:.0f}k | {zs.mean():.3f} | {test.mean():.3f} +- {test.std():.3f} | {best.mean():.3f} | {tra.mean():.3f} | {coco.mean():.3f} | " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in al.items()) + f" | {np.mean(mem):.2f} |" if mem else f"| {tr} | {shots or 'all'} | {var} | {R[0]['n_params']/1e3:.0f}k | {zs.mean():.3f} | {test.mean():.3f} +- {test.std():.3f} | {best.mean():.3f} | {tra.mean():.3f} | {coco.mean():.3f} | " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in al.items()) + " | - |")
    for r in R:
        fa = final_alpha(r); rows_for_corr.append(dict(var=var, test=r["history"]["test_acc"][-1], gain=r["history"]["test_acc"][-1] - r["zero_shot"][r["test"].split("_")[0] + "_acc"], transfer=transfer(r)[0], a_first=fa.get("adapter_down", fa.get("l3_wup")), a_last=fa.get("adapter_up", fa.get("l3_wmix")), a_mix=fa.get("l3_wmix")))
lines += ["", "## Alpha vs accuracy across runs", ""]
for fam, sel in (("MLP adapters", lambda x: x["var"].startswith("mlp")), ("L3 adapters", lambda x: x["var"].startswith("l3")), ("all", lambda x: True)):
    S = [x for x in rows_for_corr if sel(x)]
    for key, lab in (("a_first", "alpha of first matrix (adapter_down / l3_wup)"), ("a_last", "alpha of output matrix (adapter_up / l3_wmix)")):
        xs = np.array([x[key] for x in S], float); m = np.isfinite(xs)
        for tgt in ("test", "gain", "transfer"):
            ys = np.array([x[tgt] for x in S], float); rho, p = spearmanr(xs[m], ys[m])
            lines.append(f"- {fam}, {lab} vs {tgt}: Spearman {rho:+.2f} (p={p:.3f}, n={m.sum()})")
# figure: alpha trajectories and test acc per variant (pets full-shot)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for (tr, shots, var), R in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
    if tr != "pets_train" or shots is not None: continue
    r = R[0]; ep = r["history"]["epoch"]
    axes[0].plot(ep, r["history"]["test_acc"], marker="o", ms=3, label=var)
    axes[1].plot(ep, [transfer_i for transfer_i in [np.mean([v for k, v in t.items() if k.endswith("_acc") and k != "pets_acc"]) for t in r["history"]["transfer"]]], marker="o", ms=3, label=var)
    first = "adapter_down" if var.startswith("mlp") else "l3_wup"
    axes[2].plot(ep, [next((x["alpha"] for x in w["rows"] if x["name"] == first), np.nan) for w in r["history"]["ww"]], marker="o", ms=3, label=var)
axes[0].set_title("Pets full-shot: test accuracy"); axes[1].set_title("transfer retained (Imagenette + CIFAR zero-shot)"); axes[2].set_title("alpha of adapter_down / l3_wup")
for ax in axes: ax.grid(alpha=0.3); ax.set_xlabel("epoch"); ax.legend(fontsize=7)
fig.tight_layout(); Path("figures/mm").mkdir(exist_ok=True, parents=True); fig.savefig("figures/mm/e3_adapters.png", dpi=130)
Path("results/mm/e3_summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
