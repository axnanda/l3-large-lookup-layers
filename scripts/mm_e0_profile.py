"""E0: WeightWatcher alpha profile of pretrained CLIP ViT-B/32 (LAION-400M), per role and per tower.
Writes results/mm/e0_profile.json and figures/mm/e0_alpha_profile.png."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from l3lab.multimodal import clipmodel as cm
from l3lab.ww_analysis import analyze_matrices
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
out = Path("results/mm"); out.mkdir(parents=True, exist_ok=True)
model, _, _ = cm.load_clip()
mats, roles = cm.clip_matrices(model)
t0 = time.time()
rows, summary = analyze_matrices(mats, roles=roles, min_evals=50, randomize=True, mp_fit=True)
print(f"WW on {len(rows)} matrices in {time.time()-t0:.0f}s", flush=True)
json.dump({"rows": rows, "summary": summary}, open(out / "e0_profile.json", "w"), indent=1)
# figure: alpha per block for mlp_fc1 / mlp_fc2 / attn_* in each tower
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, tower in zip(axes, ("image", "text")):
    for role, mk in (("mlp_fc1", "o"), ("mlp_fc2", "s"), ("attn_q", "^"), ("attn_k", "v"), ("attn_v", "<"), ("attn_out", ">")):
        xs, ys = [], []
        for r in rows:
            if r["name"].startswith(tower + ".b") and r["role"] == role and r.get("alpha") is not None:
                xs.append(int(r["name"].split(".")[1][1:])); ys.append(r["alpha"])
        ax.plot(xs, ys, marker=mk, ms=4, lw=1, label=role)
    ax.axhline(2, color="k", ls="--", lw=0.8); ax.axhspan(2, 6, color="g", alpha=0.06)
    ax.set_title(f"{tower} tower"); ax.set_xlabel("block"); ax.grid(alpha=0.3)
axes[0].set_ylabel("WeightWatcher alpha"); axes[1].legend(fontsize=8, ncol=2)
fig.suptitle("E0: HTSR alpha per layer of pretrained CLIP ViT-B/32 (LAION-400M)")
fig.tight_layout(); Path("figures/mm").mkdir(parents=True, exist_ok=True); fig.savefig("figures/mm/e0_alpha_profile.png", dpi=130)
for r in rows:
    print(f"{r['name']:24s} {r['role']:12s} N={r['N']:6} M={r['M']:5} alpha={r['alpha'] if r['alpha'] is None else round(r['alpha'],2)} aw={r['alpha_weighted'] and round(r['alpha_weighted'],2)} lsn={r['log_spectral_norm'] and round(r['log_spectral_norm'],2)} xmin={r['xmin'] and round(r['xmin'],3)} spikes={r['num_spikes']} warn={r['warning']}")
print(json.dumps({k: v for k, v in summary.items() if 'alpha' in k}, indent=0)[:1500])
