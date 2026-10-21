"""Image datasets reachable from the fast.ai S3 mirror (Imagenette-160, Oxford-IIIT Pets, CIFAR-10, COCO sample).

Every dataset yields (paths, labels, class_names) plus prompt captions.  COCO sample has no captions; we build
pseudo-captions from the set of object categories in each image ("a photo of a person, a dog and a frisbee")
and use them for image-to-text retrieval over the distinct captions.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

DATA_ROOT = Path(os.environ.get("MM_DATA_ROOT", "/tmp/claude-0/-home-user/cf719c09-d7e5-5f6b-bcb9-86a98cd733c6/scratchpad/dl/data"))

IMAGENETTE = {"n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
              "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
              "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball", "n03888257": "parachute"}
SIMPLE_TEMPLATES = ["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.",
                    "a {} in a video game.", "art of the {}.", "a photo of the small {}."]


class ImageSet:
    def __init__(self, name: str, paths: Sequence[str], labels: Sequence[int], class_names: Sequence[str]):
        self.name, self.paths, self.labels, self.class_names = name, list(paths), np.asarray(labels, int), list(class_names)

    def __len__(self):
        return len(self.paths)

    def subset(self, n: int, seed: int = 0) -> "ImageSet":
        if n is None or n >= len(self):
            return self
        rng = np.random.default_rng(seed)
        # stratified: equal share per class as far as possible
        idx = []
        per = max(1, n // len(self.class_names))
        for c in range(len(self.class_names)):
            cidx = np.where(self.labels == c)[0]
            idx.extend(rng.permutation(cidx)[:per].tolist())
        if len(idx) < n:
            rest = np.setdiff1d(np.arange(len(self)), idx)
            idx.extend(rng.permutation(rest)[: n - len(idx)].tolist())
        idx = sorted(idx[:n])
        return ImageSet(self.name, [self.paths[i] for i in idx], self.labels[idx], self.class_names)


def imagenette(split: str = "train") -> ImageSet:
    root = DATA_ROOT / "imagenette2-160" / split
    names = list(IMAGENETTE.values())
    paths, labels = [], []
    for ci, wnid in enumerate(IMAGENETTE):
        for f in sorted((root / wnid).iterdir()):
            paths.append(str(f)); labels.append(ci)
    return ImageSet(f"imagenette_{split}", paths, labels, names)


def pets(split: str = "test") -> ImageSet:
    root = DATA_ROOT / "oxford-iiit-pet"
    lines = [l.split() for l in (root / "annotations" / f"{'trainval' if split == 'train' else 'test'}.txt").read_text().splitlines() if l and not l.startswith("#")]
    names_raw = sorted(set(l[0].rsplit("_", 1)[0] for l in lines))
    names = [n.replace("_", " ").lower() for n in names_raw]
    idx = {n: i for i, n in enumerate(names_raw)}
    paths = [str(root / "images" / f"{l[0]}.jpg") for l in lines]
    labels = [idx[l[0].rsplit("_", 1)[0]] for l in lines]
    return ImageSet(f"pets_{split}", paths, labels, names)


def cifar10(split: str = "test") -> ImageSet:
    root = DATA_ROOT / "cifar10" / split
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    paths, labels = [], []
    for ci, c in enumerate(names):
        for f in sorted((root / c).iterdir()):
            paths.append(str(f)); labels.append(ci)
    return ImageSet(f"cifar10_{split}", paths, labels, names)


def coco_captions(n_images: int = 1000, seed: int = 0, max_cats: int = 3) -> Tuple[ImageSet, List[str]]:
    """COCO sample with pseudo-captions from object categories.  Returns (ImageSet with label = caption id,
    captions).  Only images with 1..max_cats distinct categories are used; captions list the categories by
    decreasing box area."""
    root = DATA_ROOT / "coco_sample"
    d = json.load(open(root / "annotations" / "train_sample.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    files = {im["id"]: im["file_name"] for im in d["images"]}
    per_img: Dict[int, Dict[int, float]] = {}
    for a in d["annotations"]:
        per_img.setdefault(a["image_id"], {}).setdefault(a["category_id"], 0.0)
        per_img[a["image_id"]][a["category_id"]] += a["bbox"][2] * a["bbox"][3]
    rng = random.Random(seed)
    ids = [i for i, cs in per_img.items() if 1 <= len(cs) <= max_cats]
    rng.shuffle(ids)
    ids = ids[:n_images]
    captions, cap_idx, paths, labels = [], {}, [], []
    for i in ids:
        names = [cats[c] for c, _ in sorted(per_img[i].items(), key=lambda kv: -kv[1])]
        cap = "a photo of " + (names[0] if len(names) == 1 else ", ".join("a " + n for n in names[:-1]) + " and a " + names[-1]) + "."
        if cap not in cap_idx:
            cap_idx[cap] = len(captions); captions.append(cap)
        paths.append(str(root / "train_sample" / files[i])); labels.append(cap_idx[cap])
    return ImageSet("coco_sample", paths, labels, captions), captions


def class_prompts(class_names: Sequence[str], templates: Sequence[str] = SIMPLE_TEMPLATES) -> List[List[str]]:
    return [[t.format(c) for t in templates] for c in class_names]


def load_images(paths: Sequence[str], preprocess, batch: int = 64):
    """Yield preprocessed image batches (N, 3, 224, 224)."""
    for i in range(0, len(paths), batch):
        ims = [preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]
        yield torch.stack(ims)
