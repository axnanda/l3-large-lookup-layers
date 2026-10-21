"""E3: adapters on frozen CLIP image embeddings (CLIP-Adapter style) with dense MLP, spiking MLP, dense L3
lookup memory and spiking L3 memory; few-shot and full-shot training; alpha trajectories of the adapter
matrices; in-domain accuracy and zero-shot transfer retained on the other datasets.
Usage: PYTHONPATH=. python3 scripts/mm_e3_memory.py [--threads 1] [--out results/mm/e3]"""
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from l3lab.multimodal import clipmodel as cm, cache, evalmm
from l3lab.l3_ref import L3Ref, AllocTable
from l3lab.models import SNNL3Block, uniform_alloc
from l3lab.snn import LIF
from l3lab.ww_analysis import analyze_matrices

ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=1); ap.add_argument("--out", default="results/mm/e3")
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--epochs", type=int, default=40)
args = ap.parse_args(); torch.set_num_threads(args.threads); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
model, pre, tok = cm.load_clip(); heads = evalmm.text_heads(model, tok)
D = 512


class MLPAdapter(nn.Module):
    def __init__(self, r=256, spiking=False, T=8, k_sg=4.0):
        super().__init__(); self.down = nn.Linear(D, r); self.up = nn.Linear(r, D)
        self.act = cm.SpikingAct(r, T=T, theta=1.0, k_sg=k_sg) if spiking else cm.QuickGELU(); nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, e, ids=None): return self.up(self.act(self.down(e)))
    def calibrate(self, e):
        if isinstance(self.act, cm.SpikingAct):
            with torch.no_grad(): self.act.set_theta(float(torch.quantile(self.down(e).reshape(-1), 0.999)))
    def ww(self): return {"adapter_down": self.down.weight, "adapter_up": self.up.weight}, {"adapter_down": "mlp", "adapter_up": "mlp"}


class L3Adapter(nn.Module):
    """Dense L3 lookup memory keyed by a k-means token of the embedding; w_mix zero-initialised."""
    def __init__(self, V=64, k=8, d_emb=64, d_up=256):
        super().__init__(); self.alloc = uniform_alloc(V, V * k, k); self.l3 = L3Ref(D, self.alloc.n_emb, d_emb, d_up); nn.init.zeros_(self.l3.w_mix.weight)
    def forward(self, e, ids): return self.l3(e, ids, self.alloc)
    def calibrate(self, e): pass
    def ww(self):
        l = self.l3; return ({"l3_wk": l.w_k.weight, "l3_wv": l.w_v.weight, "l3_wup": l.w_up.weight, "l3_wmix": l.w_mix.weight},
                             {"l3_wk": "l3_wk", "l3_wv": "l3_wv", "l3_wup": "l3_wup", "l3_wmix": "l3_wmix"})


class SpikingL3Adapter(nn.Module):
    """Spiking L3 memory: the embedding drives an input LIF population; its spikes query the memory each step
    (mode spiking_memory = threshold-selected entries; rate_query = softmax over scores of the spike query)."""
    def __init__(self, V=64, k=8, d_emb=64, d_up=256, T=8, mode="spiking_memory", k_sg=4.0, in_scale=None, score_scale=1.0):
        super().__init__(); self.alloc = uniform_alloc(V, V * k, k); self.T = T
        self.blk = SNNL3Block(D, self.alloc.n_emb, d_emb, d_up, mode=mode, score_scale=score_scale, beta=1.0, theta=1.0, k_sg=k_sg)
        nn.init.zeros_(self.blk.w_mix.weight); self.lif_in = LIF(1.0, 1.0, k_sg, "soft"); self.in_scale = in_scale or math.sqrt(D)
        self.last_mem_rate = None
    def forward(self, e, ids):
        I = (e * self.in_scale).unsqueeze(1)  # (b, 1, h) one position
        keys, vals = self.blk.gather(self.alloc.table[ids].unsqueeze(1)); mask = self.alloc.mask[ids].unsqueeze(1)
        u = torch.full_like(I, 0.5); s = torch.zeros_like(I); mem = None; acc = 0; mem_acc = 0
        for _ in range(self.T):
            s, u = self.lif_in.step(I, u, s)
            cur, mem, s_mem = self.blk.step(s, keys, vals, mask, mem); acc = acc + cur
            if s_mem is not None: mem_acc = mem_acc + s_mem
        if torch.is_tensor(mem_acc): self.last_mem_rate = float((mem_acc / self.T)[mask].mean())
        return (acc / self.T).squeeze(1)
    def calibrate(self, e): pass
    def ww(self):
        l = self.blk; return ({"l3_wk": l.w_k.weight, "l3_wv": l.w_v.weight, "l3_wup": l.w_up.weight, "l3_wmix": l.w_mix.weight},
                             {"l3_wk": "l3_wk", "l3_wv": "l3_wv", "l3_wup": "l3_wup", "l3_wmix": "l3_wmix"})


def kmeans_ids(train_e, V, seed):
    from sklearn.cluster import KMeans
    km = KMeans(V, n_init=3, random_state=seed).fit(train_e.numpy()); return km


def run(train_name, test_name, shots, variant, seed, epochs):
    rid = f"{train_name.split('_')[0]}_{shots or 'full'}shot_{variant}_s{seed}"; p = out / f"{rid}.json"
    if p.exists(): return
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    e_tr = F.normalize(torch.from_numpy(cache.load_emb(train_name)), dim=-1); y_tr = torch.from_numpy(cache.load_labels(train_name)).long()
    if shots:
        idx = np.concatenate([rng.permutation(np.where(y_tr.numpy() == c)[0])[:shots] for c in range(int(y_tr.max()) + 1)]); e_tr, y_tr = e_tr[idx], y_tr[idx]
    head = heads[test_name]
    others = {n: (F.normalize(torch.from_numpy(cache.load_emb(n)), dim=-1), torch.from_numpy(cache.load_labels(n))) for n in evalmm.VAL_SETS}
    V = 64
    if variant == "mlp_dense": ad = MLPAdapter(spiking=False)
    elif variant == "mlp_spike": ad = MLPAdapter(spiking=True)
    elif variant == "l3_dense": ad = L3Adapter(V=V)
    elif variant == "l3_spike_mem": ad = SpikingL3Adapter(V=V, mode="spiking_memory")
    elif variant == "l3_spike_rate": ad = SpikingL3Adapter(V=V, mode="rate_query")
    else: raise ValueError(variant)
    km = kmeans_ids(e_tr, V, seed) if variant.startswith("l3") else None
    ids_of = (lambda e: torch.from_numpy(km.predict(e.numpy())).long()) if km else (lambda e: None)
    ad.calibrate(e_tr); ids_tr = ids_of(e_tr); ids_te = {n: ids_of(o[0]) for n, o in others.items()}
    opt = torch.optim.Adam(ad.parameters(), lr=1e-3); bs = 64; n = len(e_tr); hist = {"epoch": [], "train_loss": [], "test_acc": [], "transfer": [], "ww": [], "mem_rate": []}
    def evaluate(ep):
        ad.eval()
        with torch.no_grad():
            res = {}
            for nme, (e, y) in others.items():
                e2 = F.normalize(e + ad(e, ids_te[nme]), dim=-1)
                if nme == "coco_sample": r = cache.retrieval_recall(e2, y, heads[nme]); res["coco_R@1"] = r["R@1"]
                else: res[nme.split("_")[0] + "_acc"] = cache.zero_shot_acc(e2, y, heads[nme])
        ad.train(); hist["epoch"].append(ep); hist["test_acc"].append(res[test_name.split("_")[0] + "_acc"]); hist["transfer"].append(res)
        hist["mem_rate"].append(getattr(ad, "last_mem_rate", None))
        mats, roles = ad.ww(); rows, _ = analyze_matrices(mats, roles=roles, min_evals=50, randomize=False, mp_fit=False)
        hist["ww"].append({"epoch": ep, "rows": [{k: r.get(k) for k in ("name", "role", "N", "M", "num_evals", "alpha", "alpha_weighted", "log_spectral_norm", "frac_dead_rows", "status", "warning")} for r in rows]})
        return res
    r0 = evaluate(0)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n); tl = 0
        for b in range(0, n, bs):
            bi = perm[b:b + bs]; e2 = F.normalize(e_tr[bi] + ad(e_tr[bi], None if ids_tr is None else ids_tr[bi]), dim=-1)
            loss = F.cross_entropy(100 * e2 @ head.t(), y_tr[bi]); opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss) * len(bi)
        hist["train_loss"].append(tl / n)
        if ep % 5 == 0 or ep == epochs:
            r = evaluate(ep); a = {x["name"]: x["alpha"] and round(x["alpha"], 2) for x in hist["ww"][-1]["rows"]}
            log(f"{rid} ep {ep} loss {tl/n:.3f} test {r[test_name.split('_')[0]+'_acc']:.4f} transfer {[round(v,3) for k,v in r.items()]} alpha {a} mem_rate {hist['mem_rate'][-1]}")
    json.dump({"run_id": rid, "train": train_name, "test": test_name, "shots": shots, "variant": variant, "seed": seed, "n_train": n,
               "n_params": sum(p.numel() for p in ad.parameters()), "zero_shot": r0, "history": hist}, open(p, "w"))


for seed in [int(s) for s in args.seeds.split(",")]:
    for train_name, test_name, shots in [("pets_train", "pets_test_sub800", 16), ("pets_train", "pets_test_sub800", None), ("imagenette_train", "imagenette_val_sub800", 16)]:
        for variant in ("mlp_dense", "mlp_spike", "l3_dense", "l3_spike_mem", "l3_spike_rate"):
            run(train_name, test_name, shots, variant, seed, args.epochs)
log("done")
