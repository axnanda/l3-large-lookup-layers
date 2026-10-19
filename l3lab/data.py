"""Dataset loading for l3lab (Agent B).

* Fashion-MNIST is read directly from the idx ``.gz`` files (no torchvision).
  Default location: ``<repo>/data/fashion/`` (``train-images-idx3-ubyte.gz`` etc.).
* Tiny Shakespeare is a plain text file at ``<repo>/data/tinyshakespeare.txt``.

All functions return numpy arrays; use :func:`to_tensor` / :func:`flatten_images`
to build the float inputs expected by ``l3lab.train.train_model``.
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FASHION_DIR = DATA_DIR / "fashion"
TINYSHAKESPEARE_PATH = DATA_DIR / "tinyshakespeare.txt"

FASHION_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
N_CLASSES = 10

_IDX_DTYPES = {
    0x08: np.uint8, 0x09: np.int8, 0x0B: np.int16,
    0x0C: np.int32, 0x0D: np.float32, 0x0E: np.float64,
}


def read_idx(path) -> np.ndarray:
    """Parse one idx(-gz) file (the MNIST/Fashion-MNIST binary format)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        raw = f.read()
    zero, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    if zero != 0 or dtype_code not in _IDX_DTYPES:
        raise ValueError(f"{path}: not an idx file (magic={raw[:4]!r})")
    shape = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    dtype = _IDX_DTYPES[dtype_code]
    count = int(np.prod(shape))
    arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder(">"),
                        count=count, offset=4 + 4 * ndim)
    return arr.reshape(shape).astype(dtype)  # native byte order, writeable copy


def load_fashion_mnist(root=None) -> Dict[str, np.ndarray]:
    """Return dict(x_train uint8 (60000,28,28), y_train int64 (60000,), x_test, y_test)."""
    root = Path(root) if root is not None else FASHION_DIR
    out = {}
    for key, fname in FASHION_FILES.items():
        p = root / fname
        if not p.exists():
            raise FileNotFoundError(f"Fashion-MNIST file missing: {p}")
        arr = read_idx(p)
        out[key] = arr.astype(np.int64) if key.startswith("y") else arr
    assert out["x_train"].shape == (60000, 28, 28), out["x_train"].shape
    assert out["x_test"].shape == (10000, 28, 28), out["x_test"].shape
    assert out["y_train"].shape == (60000,) and out["y_test"].shape == (10000,)
    return out


def make_split(x, y, n_val: int = 10000, seed: int = 0
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train/val split: returns (x_train, y_train, x_val, y_val)."""
    n = len(x)
    if not 0 <= n_val < n:
        raise ValueError(f"n_val={n_val} must be in [0, {n})")
    perm = np.random.default_rng(seed).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def patchify(x: np.ndarray, patch: int = 7) -> np.ndarray:
    """uint8 (n,28,28) -> float32 (n, (28/patch)^2, patch*patch) in [0,1], row-major patch order."""
    x = np.asarray(x)
    n, H, W = x.shape
    if H % patch or W % patch:
        raise ValueError(f"image size {H}x{W} not divisible by patch={patch}")
    gh, gw = H // patch, W // patch
    out = x.reshape(n, gh, patch, gw, patch).transpose(0, 1, 3, 2, 4)
    out = out.reshape(n, gh * gw, patch * patch)
    return (out.astype(np.float32) / 255.0) if x.dtype == np.uint8 else out.astype(np.float32)


def flatten_images(x: np.ndarray) -> np.ndarray:
    """uint8 (n,28,28) -> float32 (n,784) in [0,1] (input format of MLP3 / SNNMLP3)."""
    x = np.asarray(x)
    flat = x.reshape(len(x), -1).astype(np.float32)
    return flat / 255.0 if x.dtype == np.uint8 else flat


def to_tensor(a: np.ndarray) -> torch.Tensor:
    """numpy -> torch (int labels become int64, floats float32)."""
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.integer):
        return torch.from_numpy(np.ascontiguousarray(a.astype(np.int64)))
    return torch.from_numpy(np.ascontiguousarray(a.astype(np.float32)))


def load_tinyshakespeare(path=None) -> str:
    path = Path(path) if path is not None else TINYSHAKESPEARE_PATH
    if not path.exists():
        raise FileNotFoundError(f"tiny Shakespeare missing: {path}")
    return path.read_text(encoding="utf-8")


__all__ = [
    "read_idx", "load_fashion_mnist", "make_split", "patchify", "flatten_images",
    "to_tensor", "load_tinyshakespeare", "FASHION_CLASSES", "N_CLASSES",
    "DATA_DIR", "FASHION_DIR", "TINYSHAKESPEARE_PATH", "REPO_ROOT",
]
