"""E2a: single-block ANN-to-SNN conversion damage versus per-block alpha.

For every tower (image/text), block k and time budget T, the QuickGELU of block k's MLP is replaced by a
threshold-balanced integrate-and-fire population (theta = 99.9th percentile of the pre-activation on
calibration data) and the zero-shot / retrieval metrics are recomputed.  Dense ablations (ReLU, clipped ReLU)
separate the activation-shape error from the spike quantisation error.  Resumable: one JSON per config.
Usage: PYTHONPATH=. python3 scripts/mm_e2a_conversion.py [threads] [--T 4,8,16] [--towers image,text]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from l3lab.multimodal import clipmodel as cm, cache, data, evalmm

ap = argparse.ArgumentParser(); ap.add_argument("threads", type=int, nargs="?", default=2)
ap.add_argument("--T", default="4,8,16"); ap.add_argument("--towers", default="image,text"); ap.add_argument("--out", default="results/mm/e2a")
ap.add_argument("--calib", type=int, default=256); ap.add_argument("--pct", type=float, default=99.9)
args = ap.parse_args(); torch.set_num_threads(args.threads)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)
model, pre, tok = cm.load_clip()
Ts = [int(t) for t in args.T.split(",")]
heads_clean = evalmm.text_heads(model, tok)
base_path = out / "baseline.json"
if not base_path.exists():
    json.dump(evalmm.evaluate(model, heads_clean), open(base_path, "w"), indent=1)
base = json.load(open(base_path)); log(f"baseline {base}")
# calibration data: block inputs of 256 imagenette train images (image), all prompts (text)
rng = np.random.default_rng(0)
calib_idx = np.sort(rng.choice(9469, args.calib, replace=False))
all_prompts = [p for n in evalmm.VAL_SETS for ps in evalmm.prompts_of(n) for p in ps]
calib_text = tok(all_prompts[:args.calib])


class ReLUAct(torch.nn.Module):
    def __init__(self, clip=None, offset=None): super().__init__(); self.clip = clip; self.offset = offset
    def forward(self, x):
        y = torch.relu(x) if self.clip is None else torch.minimum(torch.relu(x), self.clip)
        return y if self.offset is None else y + self.offset


def block_input(tower, k):
    if tower == "image":
        mm = cache.load_tokens("imagenette_train", 6)  # run 6..k on cached block-6 input (k >= 6) else from images
        if k >= 6:
            t = torch.from_numpy(np.asarray(mm[calib_idx])).float(); return cm.run_image_blocks(model, t, 6, k)
        ds = data.imagenette("train"); paths = [ds.paths[i] for i in calib_idx]
        t = torch.cat([cm.embed_images(model, x) for x in data.load_images(paths, pre, 64)]); return cm.run_image_blocks(model, t, 0, k)
    x = cm.embed_text(model, calib_text); return cm.run_text_blocks(model, x, 0, k)


@torch.no_grad()
def run(tower, k, variant, T):
    tag = f"{tower}_b{k:02d}_{variant}" + (f"_T{T}" if T else "")
    p = out / f"{tag}.json"
    if p.exists(): return
    blk = cm.get_block(model, tower, k); mask = model.attn_mask if tower == "text" else None
    x_in = block_input(tower, k)
    theta = cm.calibrate_theta(blk, x_in, args.pct, attn_mask=mask)
    ref_out = blk(x_in, attn_mask=mask) if tower == "text" else blk(x_in)
    pre_act = blk.mlp.c_fc(blk.ln_2(x_in + blk.ls_1(blk.attention(q_x=blk.ln_1(x_in), attn_mask=mask))))
    bc = cm.bias_correction(pre_act, theta)
    if variant == "spike":
        act = cm.install_spiking_act(model, tower, k, T=T, theta=theta, k_sg=4.0); cm.set_tracking(model, True)
    elif variant == "spike_bc":
        act = cm.install_spiking_act(model, tower, k, T=T, theta=theta, k_sg=4.0); act.offset.copy_(bc); cm.set_tracking(model, True)
    elif variant == "relu":
        blk.mlp.gelu = ReLUAct()
    elif variant == "cliprelu":
        blk.mlp.gelu = ReLUAct(theta)
    elif variant == "cliprelu_bc":
        blk.mlp.gelu = ReLUAct(theta, bc)
    new_out = blk(x_in, attn_mask=mask) if tower == "text" else blk(x_in)
    rel_err = float(((new_out - ref_out).norm() / ref_out.norm()))
    mlp_ref = blk.ls_2(cm.QuickGELU()(pre_act) @ blk.mlp.c_proj.weight.t() + blk.mlp.c_proj.bias)
    mlp_new = blk.ls_2(blk.mlp.gelu(pre_act) @ blk.mlp.c_proj.weight.t() + blk.mlp.c_proj.bias)
    mlp_rel_err = float((mlp_new - mlp_ref).norm() / mlp_ref.norm())
    fs = cm.firing_stats(model) if variant.startswith("spike") else {}
    if tower == "image":
        metrics = evalmm.evaluate(model, heads_clean, start_block=k)
    else:
        metrics = evalmm.evaluate(model, evalmm.text_heads(model, tok))
    cm.restore_act(model, tower, k)
    rec = dict(tower=tower, block=k, variant=variant, T=T, theta=float(theta.median()), theta_max=float(theta.max()), theta_min=float(theta.min()), rel_err_block=rel_err, rel_err_mlp=mlp_rel_err,
               frac_pos_preact=float((pre_act > 0).float().mean()), preact_std=float(pre_act.std()),
               mlp_out_share=float(mlp_ref.norm() / ref_out.norm()), firing=list(fs.values())[0] if fs else None,
               metrics=metrics, drop={m: base[m] - metrics[m] for m in metrics if metrics[m] is not None})
    json.dump(rec, open(p, "w"), indent=1)
    log(f"{tag}: theta={float(theta.median()):.2f} relerr_mlp={mlp_rel_err:.3f} mean_acc={metrics['mean_acc']:.4f} (drop {rec['drop']['mean_acc']:+.4f}) coco R@1 drop {rec['drop']['coco_R@1']:+.4f}")


for tower in args.towers.split(","):
    for k in range(12):
        for variant in ("relu", "cliprelu_bc"):
            run(tower, k, variant, None)
        for T in Ts:
            run(tower, k, "spike_bc", T)
log("done")
