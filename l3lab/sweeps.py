"""Shared helpers for the sweep scripts (Agent B): run store, budget guard, config sampling,
label noise / subsets, model-module import with a stand-in fallback, payload assembly."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .train import to_jsonable, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDIN_PATH = REPO_ROOT / "tests" / "standin_models.py"


def set_single_thread():
    """Spec: CPU with 4 shared cores -> one thread for torch and BLAS."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    torch.set_num_threads(1)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def choice(rng: np.random.Generator, options):
    """rng.choice that returns plain Python scalars."""
    options = list(options)
    v = options[int(rng.integers(len(options)))]
    return v.item() if isinstance(v, np.generic) else v


# ------------------------------------------------------------------------ run store
def run_id_for(seed: int, index: int) -> str:
    return f"run_s{seed}_{index:04d}"


class RunStore:
    """One JSON per run under ``<out>/runs/run_<id>.json`` (resumable: skip existing)."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.runs_dir = self.out_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def exists(self, run_id: str) -> bool:
        return self.path(run_id).exists()

    def save(self, run_id: str, payload: dict):
        write_json(self.path(run_id), payload)

    def load(self, run_id: str) -> dict:
        with open(self.path(run_id)) as f:
            return json.load(f)

    def load_all(self) -> List[dict]:
        out = []
        for p in sorted(self.runs_dir.glob("*.json")):
            try:
                with open(p) as f:
                    out.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                log(f"skipping unreadable {p}")
        return out


class Budget:
    """Wall-clock budget guard (``max_minutes`` <= 0 means unlimited)."""

    def __init__(self, max_minutes: float = 0.0):
        self.max_minutes = float(max_minutes or 0.0)
        self.t0 = time.time()

    def elapsed_min(self) -> float:
        return (time.time() - self.t0) / 60.0

    def remaining_min(self) -> float:
        return float("inf") if self.max_minutes <= 0 else self.max_minutes - self.elapsed_min()

    def exhausted(self, next_estimate_sec: float = 0.0) -> bool:
        if self.max_minutes <= 0:
            return False
        return self.elapsed_min() + next_estimate_sec / 60.0 >= self.max_minutes


# ------------------------------------------------------------------ data utilities
def subsample_indices(n_total: int, n_sub: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if n_sub is None or n_sub >= n_total:
        return np.arange(n_total)
    return np.sort(rng.choice(n_total, size=int(n_sub), replace=False))


def corrupt_labels(y: np.ndarray, frac: float, rng: np.random.Generator, n_classes: int = 10):
    """Relabel a ``frac`` fraction of samples with a uniformly random *different* class."""
    y = np.asarray(y).astype(np.int64).copy()
    n_noisy = int(round(float(frac) * len(y)))
    if n_noisy <= 0:
        return y, np.zeros(len(y), dtype=bool)
    idx = rng.choice(len(y), size=n_noisy, replace=False)
    shift = rng.integers(1, n_classes, size=n_noisy)
    y[idx] = (y[idx] + shift) % n_classes
    mask = np.zeros(len(y), dtype=bool)
    mask[idx] = True
    return y, mask


@torch.no_grad()
def apply_init_gain(model: nn.Module, gain: float, include_embeddings: bool = False):
    """Scale the (default-initialised) weights of every nn.Linear (optionally nn.Embedding) by ``gain``."""
    if gain is None or float(gain) == 1.0:
        return model
    for m in model.modules():
        if isinstance(m, nn.Linear) or (include_embeddings and isinstance(m, nn.Embedding)):
            m.weight.mul_(float(gain))
    return model


# ------------------------------------------------------------------ model modules
def load_module_from_path(path) -> Any:
    path = Path(path)
    name = "l3lab_ext_" + hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def import_models(spec: str = "l3lab.models", allow_standin: bool = False) -> Any:
    """Import the models module by dotted name or file path.

    If the real ``l3lab.models`` is missing and ``allow_standin`` is set (``--smoke``), fall back
    to the stand-in in ``tests/standin_models.py`` (MLP3 only) with a loud warning.
    """
    spec = str(spec)
    if spec.endswith(".py") or os.sep in spec:
        return load_module_from_path(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        return importlib.import_module(spec)
    except ModuleNotFoundError as e:
        if allow_standin and e.name in (spec, spec.split(".")[0]):
            log(f"WARNING: {spec} not importable ({e}); using stand-in {STANDIN_PATH}")
            return load_module_from_path(STANDIN_PATH)
        raise ImportError(f"cannot import models module {spec!r}: {e}. Pass --model_module "
                          f"(dotted name or .py path), e.g. --model_module {STANDIN_PATH}") from e


def construct(cls: Callable, **kwargs):
    """Instantiate ``cls`` passing only the kwargs its signature accepts (logs the dropped ones)."""
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls(**kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return cls(**kwargs)
    allowed = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(allowed))
    if dropped:
        log(f"  note: {getattr(cls, '__name__', cls)} does not accept {dropped}; dropped")
    return cls(**allowed)


def spike_diagnostics(model: nn.Module) -> Optional[dict]:
    """Best-effort call of a spiking model's rate diagnostics (several plausible method names)."""
    for name in ("spike_stats", "spike_diagnostics", "rate_stats", "get_spike_stats", "diagnostics"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                return to_jsonable(fn())
            except Exception as e:  # noqa: BLE001
                return {"error": f"{name}: {e!r}"}
    for name in ("last_stats", "last_rates", "spike_rates", "rates"):
        v = getattr(model, name, None)
        if v is not None and not callable(v):
            return to_jsonable(v)
    return None


# ----------------------------------------------------------------------- payloads
def make_payload(run_id: str, model_name: str, config: dict, history: dict, extra: Optional[dict] = None) -> dict:
    """Assemble the per-run JSON: config, history, final WW rows + summary, final metrics."""
    ww_entries = history.get("ww") or []
    last = ww_entries[-1] if ww_entries else None
    payload = {
        "run_id": run_id,
        "model": model_name,
        "config": to_jsonable(config),
        "final": history.get("final"),
        "ww_rows": last["rows"] if last else [],
        "ww_summary": last["summary"] if last else {},
        "ww_epoch": last["epoch"] if last else None,
        "alpha_trajectory": [{"epoch": e["epoch"], **{k: e["summary"].get(k) for k in (
            "alpha_avg", "alpha_default_avg", "alpha_min50_avg", "alpha_weighted_avg",
            "log_spectral_norm_avg")}} for e in ww_entries],
        "history": history,
        "n_params": history.get("n_params"),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(to_jsonable(extra))
    return payload


def summarize_run_line(run_id: str, config: dict, history: dict, keys=("h", "lr", "epochs", "train_subset")) -> str:
    final = history.get("final") or {}
    ww = (history.get("ww") or [{}])[-1].get("summary", {}) if history.get("ww") else {}
    cfg = " ".join(f"{k}={config[k]}" for k in keys if k in config)
    parts = [f"{run_id} done: {cfg}",
             f"train_acc={final.get('train_acc', float('nan')):.4f}",
             f"val_acc={final.get('val_acc', float('nan')):.4f}"]
    if "test_acc" in final:
        parts.append(f"test_acc={final['test_acc']:.4f}")
    for k in ("alpha_avg", "alpha_default_avg"):
        if ww.get(k) is not None:
            parts.append(f"{k}={ww[k]:.3f}")
    parts.append(f"wall={final.get('wall', 0.0):.1f}s")
    return " ".join(parts)


__all__ = ["set_single_thread", "log", "log_uniform", "choice", "run_id_for", "RunStore", "Budget",
           "subsample_indices", "corrupt_labels", "apply_init_gain", "import_models", "construct",
           "spike_diagnostics", "make_payload", "summarize_run_line", "load_module_from_path",
           "REPO_ROOT", "STANDIN_PATH"]


# ================================================================ L3-specific helpers
# Config space shared by run_l3_sweep.py and run_snn_sweep.py (spec section "scripts").
L3_SPACE = dict(
    V=[256, 512, 1024], n_emb=[2048, 4096, 8192], k=[4, 8, 16], h=[128, 256],
    d_emb=[32, 64, 128], n_layers=[1, 2, 3],
    weight_decay=[0.0, 1e-5, 1e-4, 1e-3, 1e-2], batch_size=[64, 128, 256],
    train_subset=[60000, 10000, 2000], lr=(1e-4, 1e-2),
)


def sample_l3_arch(rng: np.random.Generator, space: dict = L3_SPACE, max_tries: int = 1000) -> dict:
    """Sample (V, n_emb, k, h, d_emb, n_layers) with the allocation constraint V <= n_emb <= V*k."""
    for _ in range(max_tries):
        V, n_emb, k = choice(rng, space["V"]), choice(rng, space["n_emb"]), choice(rng, space["k"])
        if V <= n_emb <= V * k:
            break
    else:  # pragma: no cover
        raise RuntimeError("could not sample a valid (V, n_emb, k)")
    h = choice(rng, space["h"])
    return dict(V=int(V), n_emb=int(n_emb), k=int(k), h=int(h), d_emb=int(choice(rng, space["d_emb"])),
                d_up=int(h), n_layers=int(choice(rng, space["n_layers"])), n_patches=16)


def sample_l3_training(rng: np.random.Generator, epochs_max: int, space: dict = L3_SPACE) -> dict:
    return dict(lr=log_uniform(rng, *space["lr"]), weight_decay=float(choice(rng, space["weight_decay"])),
                batch_size=int(choice(rng, space["batch_size"])), optimizer="adam", momentum=0.9,
                epochs=int(rng.integers(1, int(epochs_max) + 1)),
                train_subset=int(choice(rng, space["train_subset"])))


def prepare_tokenized(V: int, x_train, x_val, x_test, *, seed: int, fit_images: int,
                      cache_dir, split_seed: int = 0, patch: int = 7) -> dict:
    """Fit PatchVQTokenizer once per V (cached on disk + returned dict), encode train/val/test images
    and compute train-token counts for the allocation.  Returns
    {V, ids_train, ids_val, ids_test (LongTensor (n,16)), counts (np.ndarray[V]), fit_time}."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"vq_V{V}_p{patch}_s{seed}_f{fit_images}_split{split_seed}_n{len(x_train)}_{len(x_val)}_{len(x_test)}.pt"
    path = cache_dir / key
    if path.exists():
        try:
            d = torch.load(path, weights_only=False)
            if int(d.get("V", -1)) == int(V):
                log(f"tokenizer cache hit {path.name}")
                return d
        except Exception as e:  # noqa: BLE001
            log(f"tokenizer cache unreadable ({e!r}); refitting")
    from l3lab.tokenizers import PatchVQTokenizer  # Agent A
    t0 = time.time()
    tok = PatchVQTokenizer(patch=patch, n_codes=int(V), seed=int(seed)).fit(x_train[:int(fit_images)])
    d = {"V": int(V), "patch": patch, "fit_images": int(fit_images), "seed": int(seed)}
    d["ids_train"] = torch.as_tensor(tok.encode(x_train)).long()
    d["ids_val"] = torch.as_tensor(tok.encode(x_val)).long()
    d["ids_test"] = torch.as_tensor(tok.encode(x_test)).long()
    counts = tok.counts(x_train) if hasattr(tok, "counts") else np.bincount(d["ids_train"].reshape(-1).numpy(), minlength=int(V))
    d["counts"] = np.asarray(counts, dtype=np.int64)
    d["fit_time"] = time.time() - t0
    torch.save(d, path)
    log(f"tokenizer V={V} fitted on {min(int(fit_images), len(x_train))} images in {d['fit_time']:.1f}s -> {path.name}")
    return d


def make_alloc(counts, n_emb: int, k: int):
    """allocate_by_frequency + AllocTable (Agent A) -> (alloc_table, stats dict)."""
    from l3lab.allocation import allocate_by_frequency
    from l3lab.l3_ref import AllocTable
    alloc = allocate_by_frequency([int(c) for c in counts], int(n_emb), int(k))
    table = AllocTable.from_counts(alloc)
    a = np.asarray(alloc)
    stats = dict(n_tokens=int(len(a)), n_emb=int(a.sum()), k=int(k), min=int(a.min()), max=int(a.max()),
                 mean=float(a.mean()), frac_at_k=float((a >= k).mean()), frac_at_1=float((a <= 1).mean()))
    return table, stats


__all__ += ["L3_SPACE", "sample_l3_arch", "sample_l3_training", "prepare_tokenized", "make_alloc"]
