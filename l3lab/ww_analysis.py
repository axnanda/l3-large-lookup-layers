"""WeightWatcher (0.7.7) analysis harness (Agent B).

``analyze_model(model, ...)`` -> (rows, summary)

* rows: one dict per layer WeightWatcher analysed, with the WW columns listed in
  :data:`WW_COLUMNS` (missing ones are ``None``), a ``role`` taken from
  ``model.ww_layers()`` (list of ``(name, module[, role])``) by matching the module
  object, and EXTRAS computed here from the weight matrix with ``torch.linalg.svd``.
* summary: alpha averages (all analysed layers / layers meeting ``min_evals`` /
  layers with >= 50 evals / per role), mean alpha_weighted, log norms, counts of
  over-/under-trained layers and alpha-range fractions.

WeightWatcher's logging is silenced and its stdout chatter ("SMALL N PL FIT", ...)
is captured. Layers WW skips (too few evals, unsupported type) are tolerated.

How rows are matched to modules: WW iterates ``model.named_modules()`` and assigns
``layer_id`` = enumeration index (root module = 0) and ``longname`` = module path.
We resolve ``longname`` in ``dict(model.named_modules())`` (checked against ``layer_id``
and the (N, M) shape); if that fails we fall back to order-based (N, M) matching over the
nn.Linear / nn.Embedding modules in ``named_modules()`` order.
"""
from __future__ import annotations

import contextlib
import io
import logging
import math
import time
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

for _name in ("weightwatcher", "powerlaw", "matplotlib"):
    logging.getLogger(_name).setLevel(logging.ERROR)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import weightwatcher as ww  # prints "Defaulting to SciPy for SVD" etc. at import

try:
    from weightwatcher import constants as _wwc
    WW_DEFAULT_MIN_EVALS = int(getattr(_wwc, "DEFAULT_MIN_EVALS", 8))
    WW_DEFAULT_MAX_EVALS = int(getattr(_wwc, "DEFAULT_MAX_EVALS", 15000))
    OVER_TRAINED_THRESH = float(getattr(_wwc, "OVER_TRAINED_THRESH", 2.0))
    UNDER_TRAINED_THRESH = float(getattr(_wwc, "UNDER_TRAINED_THRESH", 6.0))
except Exception:  # pragma: no cover - defensive
    WW_DEFAULT_MIN_EVALS, WW_DEFAULT_MAX_EVALS = 8, 15000
    OVER_TRAINED_THRESH, UNDER_TRAINED_THRESH = 2.0, 6.0

WW_VERSION = getattr(ww, "__version__", "unknown")

# Columns we export from WW's details frame (None if WW did not produce them).
WW_COLUMNS = [
    "layer_id", "name", "longname", "layer_type", "N", "M", "num_evals",
    "alpha", "alpha_weighted", "log_norm", "log_spectral_norm", "log_alpha_norm",
    "stable_rank", "mp_softrank", "num_spikes", "num_rand_spikes", "rand_distance",
    "D", "xmin", "xmax", "sigma", "warning", "status",
    "matrix_rank", "rank_loss", "entropy", "spectral_norm", "norm", "num_pl_spikes",
    "sigma_mp", "rand_mp_softrank",
]
# WW's own spelling for some of the columns (first existing alias wins).
WW_COLUMN_ALIASES = {
    "num_rand_spikes": ("num_rand_spikes", "rand_num_spikes"),
    "name": ("name",),
}
EXTRA_COLUMNS = [
    "top_sv_energy", "mean_dir_overlap_left", "mean_dir_overlap_right",
    "ones_overlap_left", "mean_dir_overlap_right_rowmean",
    "frac_dead_rows", "frac_dead_cols", "effective_rank", "spectral_gap",
    "sv_top", "sv_second", "n_rows", "n_cols",
]
ROW_COLUMNS = ["role", "meets_min_evals"] + WW_COLUMNS + EXTRA_COLUMNS

_WW_TYPES = (nn.Linear, nn.Embedding)


# ----------------------------------------------------------------------------- utils
@contextlib.contextmanager
def silence_ww():
    """Swallow WW/powerlaw stdout+stderr chatter and numpy/sklearn warnings."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            yield


def _py(v):
    """Convert numpy/torch scalars to plain Python; NaN/inf -> None; leave strings."""
    if v is None:
        return None
    if isinstance(v, (str, bool)):
        return v
    if isinstance(v, torch.Tensor):
        if v.numel() != 1:
            return [_py(x) for x in v.flatten().tolist()]
        v = v.item()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (int,)):
        return int(v)
    try:
        if v != v:  # pandas NA / NaN-like
            return None
    except Exception:
        pass
    return v


def _finite(vals: Iterable) -> List[float]:
    out = []
    for v in vals:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _mean(vals: Iterable) -> Optional[float]:
    f = _finite(vals)
    return float(np.mean(f)) if f else None


def _median(vals: Iterable) -> Optional[float]:
    f = _finite(vals)
    return float(np.median(f)) if f else None


# ------------------------------------------------------------------- weight extras
def _weight_matrix(module: nn.Module) -> Optional[torch.Tensor]:
    w = getattr(module, "weight", None)
    if w is None or not torch.is_tensor(w):
        return None
    w = w.detach()
    if w.dim() < 2:
        return None
    if w.dim() > 2:
        w = w.reshape(w.shape[0], -1)
    return w.float()


@torch.no_grad()
def compute_extras(W: torch.Tensor, dead_rel: float = 1e-3) -> Dict[str, Optional[float]]:
    """Spectral extras from a 2-D weight matrix W (rows = output features for nn.Linear).

    top_sv_energy      = s_1^2 / sum_i s_i^2
    mean_dir_overlap_left  = |<u_1, mean of the columns of W (normalised)>|   (output space)
    mean_dir_overlap_right = |<v_1, ones/sqrt(n_cols)>|                      (input space)
    ones_overlap_left      = |<u_1, ones/sqrt(n_rows)>|
    mean_dir_overlap_right_rowmean = |<v_1, mean of the rows of W (normalised)>|
    frac_dead_rows/cols = fraction of rows/cols with norm < dead_rel * median norm
    effective_rank     = exp(entropy of s_i^2 / sum s_i^2)
    spectral_gap       = s_1 / s_2
    """
    W = W.detach().float()
    if W.dim() != 2:
        W = W.reshape(W.shape[0], -1)
    n_rows, n_cols = W.shape
    out: Dict[str, Optional[float]] = {k: None for k in EXTRA_COLUMNS}
    out["n_rows"], out["n_cols"] = int(n_rows), int(n_cols)
    if min(n_rows, n_cols) < 1 or not torch.isfinite(W).all():
        return out
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except Exception:  # pragma: no cover - numerical failure
        return out
    s2 = S ** 2
    tot = float(s2.sum())
    if tot <= 0:
        out["frac_dead_rows"], out["frac_dead_cols"] = 1.0, 1.0
        return out
    p = s2 / tot
    pp = p[p > 0]
    ent = float(-(pp * pp.log()).sum())
    out["top_sv_energy"] = float(p[0])
    out["effective_rank"] = float(math.exp(ent))
    out["sv_top"] = float(S[0])
    out["sv_second"] = float(S[1]) if S.numel() > 1 else None
    out["spectral_gap"] = float(S[0] / S[1]) if S.numel() > 1 and float(S[1]) > 0 else None
    u1, v1 = U[:, 0], Vh[0]
    col_mean = W.mean(dim=1)   # (n_rows,) mean of the columns -> output space
    row_mean = W.mean(dim=0)   # (n_cols,) mean of the rows -> input space
    cm_n, rm_n = float(col_mean.norm()), float(row_mean.norm())
    out["mean_dir_overlap_left"] = float(abs(torch.dot(u1, col_mean / cm_n))) if cm_n > 0 else None
    out["mean_dir_overlap_right_rowmean"] = float(abs(torch.dot(v1, row_mean / rm_n))) if rm_n > 0 else None
    out["ones_overlap_left"] = float(abs(u1.sum()) / math.sqrt(n_rows))
    out["mean_dir_overlap_right"] = float(abs(v1.sum()) / math.sqrt(n_cols))
    rn = W.norm(dim=1)
    cn = W.norm(dim=0)
    med_r, med_c = float(rn.median()), float(cn.median())
    out["frac_dead_rows"] = float((rn < dead_rel * med_r).float().mean()) if med_r > 0 else 1.0
    out["frac_dead_cols"] = float((cn < dead_rel * med_c).float().mean()) if med_c > 0 else 1.0
    return out


# ---------------------------------------------------------------- role/module maps
def _role_map(model: nn.Module, roles=None) -> Dict[int, Tuple[str, str]]:
    """id(module) -> (name, role) from model.ww_layers() (+ optional overrides)."""
    out: Dict[int, Tuple[str, str]] = {}
    entries = []
    if hasattr(model, "ww_layers") and callable(model.ww_layers):
        try:
            entries = list(model.ww_layers())
        except Exception:
            entries = []
    if isinstance(roles, (list, tuple)):  # same format as ww_layers()
        entries = list(entries) + list(roles)
    for ent in entries:
        if len(ent) == 3:
            name, mod, role = ent
        elif len(ent) == 2:
            name, mod = ent
            role = None
        else:
            continue
        if not isinstance(mod, nn.Module):
            continue
        out[id(mod)] = (str(name), str(role) if role is not None else None)
    if isinstance(roles, dict):  # {name or module: role} overrides
        by_name = {v[0]: k for k, v in out.items()}
        named = dict(model.named_modules())
        for key, role in roles.items():
            if isinstance(key, nn.Module):
                mid = id(key)
                name = out.get(mid, (None,))[0] or _module_path(model, key)
            elif key in by_name:
                mid, name = by_name[key], key
            elif key in named:
                mid, name = id(named[key]), key
            else:
                continue
            out[mid] = (name, str(role))
    return out


def _module_path(model: nn.Module, module: nn.Module) -> str:
    for n, m in model.named_modules():
        if m is module:
            return n
    return type(module).__name__


def _shape_NM(module: nn.Module) -> Optional[Tuple[int, int]]:
    W = getattr(module, "weight", None)
    if W is None or not torch.is_tensor(W) or W.dim() < 2:
        return None
    if W.dim() == 2:
        a, b = W.shape
    else:  # conv: WW uses (out, in) of each receptive-field slice
        a, b = W.shape[0], W.shape[1]
    return (max(a, b), min(a, b))


def match_rows_to_modules(model: nn.Module, details) -> List[Optional[nn.Module]]:
    """For every row of WW's details frame, return the module it describes (or None)."""
    named = list(model.named_modules())
    named_dict = dict(named)
    candidates = [(n, m) for n, m in named if isinstance(m, _WW_TYPES) or _shape_NM(m) is not None]
    used = set()
    matched: List[Optional[nn.Module]] = []
    for _, row in details.iterrows():
        N, M = _py(row.get("N")), _py(row.get("M"))
        want = (int(N), int(M)) if N is not None and M is not None else None
        mod = None
        longname = row.get("longname") if "longname" in details.columns else None
        if isinstance(longname, str) and longname in named_dict:
            cand = named_dict[longname]
            if want is None or _shape_NM(cand) == want:
                mod = cand
        if mod is None:
            lid = _py(row.get("layer_id"))
            if lid is not None and 0 <= int(lid) < len(named):
                cand = named[int(lid)][1]
                if want is None or _shape_NM(cand) == want:
                    mod = cand
        if mod is None and want is not None:  # order-based fallback
            for n, m in candidates:
                if id(m) in used:
                    continue
                if _shape_NM(m) == want:
                    mod = m
                    break
        if mod is not None:
            used.add(id(mod))
        matched.append(mod)
    return matched


# ----------------------------------------------------------------------- analysis
def _details_to_rows(details, model: nn.Module, roles, min_evals: int, extras: bool) -> List[dict]:
    rows: List[dict] = []
    if details is None or len(details) == 0:
        return rows
    role_map = _role_map(model, roles)
    modules = match_rows_to_modules(model, details)
    cols = set(details.columns)
    for (_, rec), mod in zip(details.iterrows(), modules):
        row: dict = {}
        for c in WW_COLUMNS:
            val = None
            for alias in WW_COLUMN_ALIASES.get(c, (c,)):
                if alias in cols:
                    val = rec[alias]
                    break
            row[c] = _py(val)
        row["ww_name"] = row["name"]
        if mod is not None and id(mod) in role_map:
            name, role = role_map[id(mod)]
            row["name"] = name
            row["role"] = role if role is not None else "unknown"
        else:
            row["name"] = row.get("longname") or row.get("name")
            row["role"] = "unknown"
        row["module_path"] = _module_path(model, mod) if mod is not None else None
        row["matched"] = mod is not None
        ne = row.get("num_evals")
        row["meets_min_evals"] = bool(ne is not None and ne >= min_evals)
        if extras and mod is not None:
            W = _weight_matrix(mod)
            row.update(compute_extras(W) if W is not None else {k: None for k in EXTRA_COLUMNS})
        else:
            row.update({k: None for k in EXTRA_COLUMNS})
        rows.append(row)
    return rows


def summarize_rows(rows: Sequence[dict], min_evals: int = 50, tag=None) -> dict:
    """Aggregate per-layer rows into the summary dict described in the spec."""
    def a(r):
        return r.get("alpha")

    with_alpha = [r for r in rows if _finite([a(r)])]
    meets = [r for r in with_alpha if (r.get("num_evals") or 0) >= min_evals]
    ge50 = [r for r in with_alpha if (r.get("num_evals") or 0) >= 50]
    alphas = [a(r) for r in meets]
    n_meet = len(meets)

    def frac(pred):
        return float(sum(1 for v in alphas if pred(v)) / n_meet) if n_meet else None

    roles_all = sorted({r.get("role") or "unknown" for r in rows})
    s: dict = {
        "tag": tag,
        "min_evals": int(min_evals),
        "ww_version": WW_VERSION,
        "n_layers": len(rows),
        "n_layers_alpha": len(with_alpha),
        "n_layers_min_evals": n_meet,
        "n_layers_min50": len(ge50),
        # alpha averages -------------------------------------------------------
        "alpha_avg": _mean(alphas),                       # layers meeting min_evals
        "alpha_default_avg": _mean(a(r) for r in with_alpha),  # all layers WW analysed (its default min_evals)
        "alpha_min50_avg": _mean(a(r) for r in ge50),
        "alpha_median": _median(alphas),
        "alpha_min": min(_finite(alphas)) if n_meet else None,
        "alpha_max": max(_finite(alphas)) if n_meet else None,
        "alpha_by_role": {},
        "alpha_by_role_all": {},
        # other WW averages (layers meeting min_evals) -------------------------
        "alpha_weighted_avg": _mean(r.get("alpha_weighted") for r in meets),
        "log_norm_avg": _mean(r.get("log_norm") for r in meets),
        "log_spectral_norm_avg": _mean(r.get("log_spectral_norm") for r in meets),
        "log_alpha_norm_avg": _mean(r.get("log_alpha_norm") for r in meets),
        "stable_rank_avg": _mean(r.get("stable_rank") for r in meets),
        "mp_softrank_avg": _mean(r.get("mp_softrank") for r in meets),
        "rand_distance_avg": _mean(r.get("rand_distance") for r in meets),
        "D_avg": _mean(r.get("D") for r in meets),
        "num_spikes_total": int(sum(_finite(r.get("num_spikes") for r in meets))) if meets else None,
        # flags -----------------------------------------------------------------
        "n_over_trained": int(sum(1 for v in alphas if v < OVER_TRAINED_THRESH)),
        "n_under_trained": int(sum(1 for v in alphas if v > UNDER_TRAINED_THRESH)),
        "n_ww_warn_over_trained": int(sum(1 for r in rows if (r.get("warning") or "") == "over-trained")),
        "n_ww_warn_under_trained": int(sum(1 for r in rows if (r.get("warning") or "") == "under-trained")),
        "frac_alpha_lt2": frac(lambda v: v < 2.0),
        "frac_alpha_2_3": frac(lambda v: 2.0 <= v <= 3.0),
        "frac_alpha_2_4": frac(lambda v: 2.0 <= v <= 4.0),
        "frac_alpha_gt6": frac(lambda v: v > 6.0),
        # extras ------------------------------------------------------------------
        "top_sv_energy_avg": _mean(r.get("top_sv_energy") for r in meets),
        "mean_dir_overlap_left_avg": _mean(r.get("mean_dir_overlap_left") for r in meets),
        "mean_dir_overlap_right_avg": _mean(r.get("mean_dir_overlap_right") for r in meets),
        "effective_rank_avg": _mean(r.get("effective_rank") for r in meets),
        "spectral_gap_avg": _mean(r.get("spectral_gap") for r in meets),
        "frac_dead_rows_avg": _mean(r.get("frac_dead_rows") for r in meets),
    }
    for role in roles_all:
        m_role = _mean(a(r) for r in meets if (r.get("role") or "unknown") == role)
        all_role = _mean(a(r) for r in with_alpha if (r.get("role") or "unknown") == role)
        s["alpha_by_role"][role] = m_role
        s["alpha_by_role_all"][role] = all_role
        s[f"alpha_role_{role}"] = m_role
        s[f"alpha_role_all_{role}"] = all_role
    return s


def analyze_model(model: nn.Module, *, min_evals: int = 50, randomize: bool = True,
                  mp_fit: bool = True, roles=None, tag=None, extras: bool = True,
                  max_evals: Optional[int] = None, ww_min_evals: Optional[int] = None,
                  **ww_extra) -> Tuple[List[dict], dict]:
    """Run WeightWatcher on ``model`` and return (rows, summary).

    WW itself is run with ``ww_min_evals`` (default: ``min(min_evals, WW default=8)``) so
    that every layer WW would analyse by default is present in ``rows``; each row carries
    ``meets_min_evals`` (num_evals >= ``min_evals``).  ``summary['alpha_avg']`` averages
    the layers meeting ``min_evals``; ``summary['alpha_default_avg']`` averages *all*
    analysed layers (what "mean alpha of fc1, fc2, fc3" means in Martin's post);
    ``summary['alpha_min50_avg']`` averages layers with num_evals >= 50.
    """
    t0 = time.time()
    was_training = model.training
    model.eval()
    eff_min = int(ww_min_evals) if ww_min_evals is not None else min(int(min_evals), WW_DEFAULT_MIN_EVALS)
    kwargs = dict(plot=False, randomize=bool(randomize), mp_fit=bool(mp_fit), min_evals=eff_min)
    if max_evals is not None:
        kwargs["max_evals"] = int(max_evals)
    kwargs.update(ww_extra)
    details = None
    with silence_ww():
        try:
            watcher = ww.WeightWatcher(model=model)
            details = watcher.analyze(**kwargs)
        except Exception as e:  # tolerate WW failures (e.g. no supported layers)
            details = None
            err = repr(e)
        else:
            err = None
    rows = _details_to_rows(details, model, roles, int(min_evals), extras)
    summary = summarize_rows(rows, min_evals=int(min_evals), tag=tag)
    summary["time_sec"] = time.time() - t0
    summary["ww_min_evals_used"] = eff_min
    summary["randomize"], summary["mp_fit"] = bool(randomize), bool(mp_fit)
    if err:
        summary["error"] = err
    model.train(was_training)
    return rows, summary


class _MatrixBag(nn.Module):
    """Throw-away module wrapping raw 2-D tensors as nn.Linear layers (for WW)."""

    def __init__(self, mats: Dict[str, torch.Tensor], roles: Optional[Dict[str, str]] = None):
        super().__init__()
        self._names: List[str] = []
        self._roles = dict(roles or {})
        self.layers = nn.ModuleList()
        for name, W in mats.items():
            W = torch.as_tensor(W).detach().float()
            if W.dim() != 2:
                W = W.reshape(W.shape[0], -1)
            lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
            with torch.no_grad():
                lin.weight.copy_(W)
            self.layers.append(lin)
            self._names.append(str(name))

    def ww_layers(self):
        return [(n, m, self._roles.get(n, "matrix")) for n, m in zip(self._names, self.layers)]


def analyze_matrices(mats: Dict[str, torch.Tensor], *, roles: Optional[Dict[str, str]] = None,
                     **kwargs) -> Tuple[List[dict], dict]:
    """Analyse raw matrices {name: 2-D tensor} (e.g. nn.Parameter tables) with WeightWatcher."""
    bag = _MatrixBag(mats, roles)
    return analyze_model(bag, **kwargs)


def rows_to_dataframe(rows: Sequence[dict]):
    import pandas as pd
    return pd.DataFrame(list(rows))


def format_rows(rows: Sequence[dict], cols: Sequence[str] = ("name", "role", "N", "M", "num_evals",
                                                            "alpha", "alpha_weighted", "log_spectral_norm",
                                                            "stable_rank", "top_sv_energy", "effective_rank",
                                                            "warning")) -> str:
    """Small fixed-width text table of the rows (for logs)."""
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)
    widths = [max(len(c), *(len(fmt(r.get(c))) for r in rows)) if rows else len(c) for c in cols]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths))]
    for r in rows:
        lines.append("  ".join(fmt(r.get(c)).ljust(w) for c, w in zip(cols, widths)))
    return "\n".join(lines)


__all__ = [
    "analyze_model", "analyze_matrices", "compute_extras", "summarize_rows",
    "match_rows_to_modules", "rows_to_dataframe", "format_rows", "silence_ww",
    "WW_COLUMNS", "EXTRA_COLUMNS", "ROW_COLUMNS", "WW_DEFAULT_MIN_EVALS", "WW_VERSION",
    "OVER_TRAINED_THRESH", "UNDER_TRAINED_THRESH",
]
