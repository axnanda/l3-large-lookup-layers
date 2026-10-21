"""Build block-input token caches and embeddings for the multimodal study (run once; resumable)."""
import sys, time, torch
from l3lab.multimodal import clipmodel as cm, data, cache
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)
model, pre, tok = cm.load_clip()
ALL = list(range(12))
jobs = [
    (data.imagenette("val").subset(800), ALL),
    (data.pets("test").subset(800), ALL),
    (data.cifar10("test").subset(800), ALL),
    (data.coco_captions(1000)[0], ALL),
    (data.imagenette("train"), [6, 8, 10]),
    (data.pets("train"), [6, 10]),
]
for ds, blocks in jobs:
    ds.name = ds.name + (f"_sub{len(ds)}" if ds.name.endswith(("val", "test")) else "")
    log(f"start {ds.name} n={len(ds)} blocks={blocks}")
    cache.build_image_cache(model, pre, ds, blocks, log=log)
log("done")
