"""Tokenizers that turn raw data into integer token sequences for the L3 models.

* ``PatchVQTokenizer``: 28x28 images -> 16 non-overlapping 7x7 patches -> nearest of ``n_codes`` k-means
  centroids (a visual "vocabulary"). Fitting uses ``sklearn.cluster.MiniBatchKMeans`` with a fixed
  ``random_state`` on a seeded subsample of patches, so it is deterministic given ``seed``.
* ``CharTokenizer``: character-level vocabulary for text.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def patchify(images: np.ndarray, patch: int = 7) -> np.ndarray:
    """uint8/float images (n, H, W) -> float32 (n, (H/patch)*(W/patch), patch*patch) in [0, 1]."""
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError("images must have shape (n, H, W)")
    n, H, W = images.shape
    if H % patch or W % patch:
        raise ValueError(f"image size {H}x{W} not divisible by patch={patch}")
    x = images.astype(np.float32)
    if images.dtype == np.uint8:
        x = x / 255.0
    gh, gw = H // patch, W // patch
    x = x.reshape(n, gh, patch, gw, patch).transpose(0, 1, 3, 2, 4).reshape(n, gh * gw, patch * patch)
    return np.ascontiguousarray(x)


class PatchVQTokenizer:
    """Vector-quantise image patches with k-means; ``V = n_codes`` codes, ``n_patches`` tokens per image."""

    def __init__(self, patch: int = 7, n_codes: int = 512, seed: int = 0, max_fit_patches: int = 200_000,
                 image_size: int = 28):
        self.patch, self.n_codes, self.seed = int(patch), int(n_codes), int(seed)
        self.max_fit_patches = int(max_fit_patches)
        self.image_size = int(image_size)
        self.V = self.n_codes
        self.n_patches = (self.image_size // self.patch) ** 2
        self.centroids: Optional[np.ndarray] = None   # (n_codes, patch*patch) float32

    def _patches(self, images) -> np.ndarray:
        images = np.asarray(images)
        if images.ndim == 3 and images.shape[-1] == self.patch * self.patch and images.shape[1] == self.n_patches:
            return images.astype(np.float32)          # already patchified (n, n_patches, p*p)
        return patchify(images, self.patch)

    def fit(self, images) -> "PatchVQTokenizer":
        from sklearn.cluster import MiniBatchKMeans

        pts = self._patches(images).reshape(-1, self.patch * self.patch)
        rng = np.random.RandomState(self.seed)
        if pts.shape[0] > self.max_fit_patches:
            pts = pts[rng.choice(pts.shape[0], self.max_fit_patches, replace=False)]
        km = MiniBatchKMeans(n_clusters=self.n_codes, random_state=self.seed, batch_size=4096, n_init=3,
                             max_iter=100, reassignment_ratio=0.01)
        km.fit(pts.astype(np.float64))
        self.centroids = km.cluster_centers_.astype(np.float32)
        return self

    def encode(self, images) -> torch.Tensor:
        """images (n, 28, 28) (or patches (n, 16, 49)) -> LongTensor (n, n_patches) of code ids."""
        if self.centroids is None:
            raise RuntimeError("call fit() first")
        pts = torch.from_numpy(self._patches(images))                  # (n, P, d)
        n, P, d = pts.shape
        flat = pts.reshape(-1, d)
        C = torch.from_numpy(self.centroids)                            # (V, d)
        c2 = (C * C).sum(1)
        codes = torch.empty(flat.shape[0], dtype=torch.long)
        for i in range(0, flat.shape[0], 65536):
            x = flat[i:i + 65536]
            dist = c2.view(1, -1) - 2.0 * (x @ C.T)                     # ||x||^2 omitted: constant per row
            codes[i:i + 65536] = dist.argmin(1)
        return codes.view(n, P)

    def counts(self, images) -> np.ndarray:
        """Code frequencies (length V) over the given images."""
        ids = self.encode(images).reshape(-1).numpy()
        return np.bincount(ids, minlength=self.V)

    def state_dict(self) -> dict:
        return {"patch": self.patch, "n_codes": self.n_codes, "seed": self.seed, "image_size": self.image_size,
                "max_fit_patches": self.max_fit_patches, "centroids": None if self.centroids is None else self.centroids.copy()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "PatchVQTokenizer":
        tok = cls(patch=d["patch"], n_codes=d["n_codes"], seed=d["seed"], max_fit_patches=d.get("max_fit_patches", 200_000),
                  image_size=d.get("image_size", 28))
        tok.centroids = None if d["centroids"] is None else np.asarray(d["centroids"], dtype=np.float32)
        return tok


class CharTokenizer:
    """Character-level tokenizer: ``fit(text)`` builds a sorted vocabulary; unknown chars at encode -> error."""

    def __init__(self):
        self.chars: list[str] = []
        self.stoi: dict[str, int] = {}

    @property
    def V(self) -> int:
        return len(self.chars)

    def fit(self, text: str) -> "CharTokenizer":
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        return self

    def encode(self, text: str) -> torch.Tensor:
        try:
            return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        except KeyError as e:
            raise ValueError(f"character {e} not in vocabulary") from None

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)
