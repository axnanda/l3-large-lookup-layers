"""E2b: cumulative ANN-to-SNN conversion of image-tower MLP activations in different orders
(|alpha-2| ascending / descending, depth first / last, random), with sequential threshold calibration,
conversion-only metrics and an optional brief fine-tune of the converted blocks' MLPs.
Usage: PYTHONPATH=. python3 scripts/mm_e2b_order.py [--threads 2] [--T 8] [--fractions 3,6,9,12] [--finetune_epochs 0]"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from l3lab.multimodal import clipmodel as cm, cache, data, evalmm
from l3lab.ww_analysis import analyze_matrices

ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--T", type=int, default=8)
ap.add_argument("--fractions", default="3,6,9,12"); ap.add_argument("--finetune_epochs", type=int, default=0); ap.add_argument("--n_train", type=int, default=2000)
ap.add_argument("--orders", default="alpha_asc,alpha_desc,depth_first,depth_last,random0,random1"); ap.add_argument("--out", default="results/mm/e2b")
ap.add_argument("--alpha_key", default="mlp_mean")
args = ap.parse_args(); torch.set_num_threads(args.threads); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
model, pre, tok = cm.load_clip(); heads = evalmm.text_heads(model, tok)
prof = json.load(open("results/mm/e0_profile.json"))["rows"]
alpha = {}
for r in prof:
    if r["name"].startswith("image.b") and r["role"] in ("mlp_fc1", "mlp_fc2") and r.get("alpha") is not None:
        alpha.setdefault(int(r["name"].split(".")[1][1:]), {})[r["role"]] = r["alpha"]
score = {b: (np.mean(list(v.values())) if args.alpha_key == "mlp_mean" else v[args.alpha_key]) for b, v in alpha.items()}
log(f"block mlp alpha: { {b: round(s, 2) for b, s in sorted(score.items())} }")
ORDERS = {"alpha_asc": sorted(score, key=lambda b: abs(score[b] - 2)), "alpha_desc": sorted(score, key=lambda b: -abs(score[b] - 2)),
          "depth_first": list(range(12)), "depth_last": list(range(11, -1, -1)),
          "random0": list(np.random.default_rng(0).permutation(12)), "random1": list(np.random.default_rng(1).permutation(12))}
calib_idx = np.sort(np.random.default_rng(0).choice(9469, 256, replace=False))
mm0 = cache.load_tokens("imagenette_train", 0); y_all = cache.load_labels("imagenette_train")
calib0 = torch.from_numpy(np.asarray(mm0[calib_idx])).float()
base = evalmm.evaluate(model, heads); log(f"baseline {base}")
sd = {k: v.clone() for k, v in model.state_dict().items()}


def reset_model():
    for b in range(12): cm.restore_act(model, "image", b)
    model.load_state_dict(sd)
    for p in model.parameters(): p.requires_grad_(False)


@torch.no_grad()
def convert(blocks):
    thetas = {}
    for b in blocks:  # sequential calibration on the partly converted model
        x = cm.run_image_blocks(model, calib0, 0, b); pre = cm.block_preact(cm.get_block(model, "image", b), x); th = cm.theta_from_preact(pre, 99.9)
        act = cm.install_spiking_act(model, "image", b, T=args.T, theta=th, k_sg=4.0); act.offset.copy_(cm.bias_correction(pre, th)); thetas[b] = float(th.median())
    return thetas


def finetune(blocks, epochs):
    params = []
    for b in blocks: blk = cm.get_block(model, "image", b); params += list(blk.mlp.parameters()) + list(blk.ln_2.parameters())
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=1e-4); head = heads["imagenette_val_sub800"]; rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(9469, args.n_train, replace=False)); bs = 64; model.train(); losses = []
    for ep in range(epochs):
        perm = rng.permutation(len(idx))
        for i in range(0, len(idx), bs):
            bi = np.sort(idx[perm[i:i + bs]]); x = torch.from_numpy(np.asarray(mm0[bi])).float(); y = torch.from_numpy(y_all[bi]).long()
            loss = F.cross_entropy(100 * F.normalize(cm.image_embed_from_block(model, x, 0), dim=-1) @ head.t(), y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
    model.eval()
    for p in params: p.requires_grad_(False)
    return losses


for m in [int(f) for f in args.fractions.split(",")]:
    for name in args.orders.split(","):
        if m == 12 and name != "depth_first": continue
        rid = f"{name}_m{m}_T{args.T}" + (f"_ft{args.finetune_epochs}" if args.finetune_epochs else ""); p = out / f"{rid}.json"
        if p.exists(): continue
        reset_model(); blocks = [int(b) for b in ORDERS[name][:m]]
        thetas = convert(blocks); cm.set_tracking(model, True)
        conv = evalmm.evaluate(model, heads, start_block=min(blocks)); fs = cm.firing_stats(model); cm.set_tracking(model, False)
        rec = dict(order=name, m=m, T=args.T, blocks=blocks, thetas=thetas, block_alpha=[score[b] for b in blocks], conversion_only=conv,
                   firing=fs, drop_conversion=base["mean_acc"] - conv["mean_acc"], baseline=base)
        log(f"{rid}: blocks {blocks} conv mean_acc {conv['mean_acc']:.4f} (drop {rec['drop_conversion']:+.4f}) coco R@1 {conv['coco_R@1']:.3f}")
        if args.finetune_epochs:
            losses = finetune(blocks, args.finetune_epochs); cm.set_tracking(model, True)
            ft = evalmm.evaluate(model, heads, start_block=min(blocks)); fs2 = cm.firing_stats(model); cm.set_tracking(model, False)
            mats, roles = {}, {}
            for b in blocks: m2, r2 = cm.block_matrices(cm.get_block(model, "image", b), f"image.b{b:02d}"); mats.update({k: v for k, v in m2.items() if "mlp" in k}); roles.update({k: r2[k] for k in m2 if "mlp" in k})
            rows, _ = analyze_matrices(mats, roles=roles, min_evals=50, randomize=False, mp_fit=False)
            rec.update(finetuned=ft, firing_ft=fs2, train_loss=losses, drop_finetuned=base["mean_acc"] - ft["mean_acc"],
                       ww_after=[{k: r.get(k) for k in ("name", "role", "alpha", "alpha_weighted", "log_spectral_norm", "frac_dead_rows")} for r in rows])
            log(f"{rid}: after ft mean_acc {ft['mean_acc']:.4f} (drop {rec['drop_finetuned']:+.4f})")
        json.dump(rec, open(p, "w"), indent=1)
log("done")
