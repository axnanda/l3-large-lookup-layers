#!/usr/bin/env python3
"""Figures + markdown summary for sweep results written by run_{mlp,l3,snn}_sweep.py.

    python3 scripts/analyze_sweeps.py --in results/mlp_sweep [results/l3_sweep ...] --out figures/mlp

Outputs (in --out):
  runs_table.csv                 one row per run: config, final metrics, WW summary scalars
  alpha_vs_acc__<metric>.png     avg-alpha vs train acc and vs val acc (colour = generalization gap)
  alpha_by_role_vs_val_acc.png   per-role mean alpha vs val acc (small multiples)
  alpha_trajectories.png         alpha over epochs for runs with per-epoch WW (colour = final val acc)
  alpha_hist_by_role.png         histogram of per-layer alpha by role
  summary.md                     counts, config coverage, Spearman correlation table, top runs
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from l3lab.sweeps import RunStore, log  # noqa: E402

# dataviz reference palette (fixed categorical order, single-hue sequential for magnitude)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_CMAP = "Blues"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d7d2"

ALPHA_METRICS = ["alpha_avg", "alpha_default_avg", "alpha_min50_avg", "alpha_median", "alpha_weighted_avg",
                 "log_spectral_norm_avg", "log_norm_avg", "log_alpha_norm_avg", "stable_rank_avg", "mp_softrank_avg",
                 "rand_distance_avg", "top_sv_energy_avg", "mean_dir_overlap_left_avg", "mean_dir_overlap_right_avg",
                 "effective_rank_avg", "spectral_gap_avg", "frac_dead_rows_avg", "frac_alpha_lt2", "frac_alpha_2_4",
                 "n_over_trained", "n_under_trained"]
TARGETS = ["train_acc", "val_acc", "test_acc", "gap"]
ALPHA_GUIDE_METRICS = {"alpha_avg", "alpha_default_avg", "alpha_min50_avg", "alpha_median"}
CONFIG_SKIP = {"index", "seed", "model"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inputs", nargs="+", required=True, help="sweep result dirs (with runs/*.json)")
    p.add_argument("--out", default="figures", help="output directory for figures + summary.md")
    p.add_argument("--x_metric", default="alpha_default_avg",
                   help="alpha summary used for the main scatter (also drawn for alpha_min50_avg, alpha_weighted_avg)")
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--min_runs_corr", type=int, default=5)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- loading
def load_payloads(in_dirs):
    payloads = []
    for d in in_dirs:
        d = Path(d)
        runs_dir = d if d.name == "runs" else d / "runs"
        if not runs_dir.exists():
            log(f"no runs dir in {d}; skipping")
            continue
        store = RunStore(runs_dir.parent)
        ps = store.load_all()
        for p in ps:
            p["_source"] = str(d)
        payloads += ps
        log(f"{d}: {len(ps)} runs")
    return payloads


def flatten_payload(p: dict) -> dict:
    row = {"run_id": p.get("run_id"), "model": p.get("model"), "source": p.get("_source"), "n_params": p.get("n_params")}
    for k, v in (p.get("config") or {}).items():
        if k in CONFIG_SKIP:
            continue
        row[f"cfg_{k}"] = v
    for k, v in (p.get("final") or {}).items():
        row[k] = v
    for k, v in (p.get("ww_summary") or {}).items():
        if isinstance(v, (int, float)) or v is None:
            row[k] = v
    row["n_ww_epochs"] = len(p.get("alpha_trajectory") or [])
    row["epochs_run"] = (p.get("final") or {}).get("epochs_run")
    return row


def build_table(payloads) -> pd.DataFrame:
    df = pd.DataFrame([flatten_payload(p) for p in payloads])
    if "gap" not in df.columns and {"train_acc", "val_acc"} <= set(df.columns):
        df["gap"] = df["train_acc"] - df["val_acc"]
    return df


def role_columns(df: pd.DataFrame):
    return sorted(c for c in df.columns if c.startswith("alpha_role_") and not c.startswith("alpha_role_all_"))


# ----------------------------------------------------------------------------- style
def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
        return float("nan"), float("nan"), n
    r = stats.spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue), n


def savefig(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {path}")


# --------------------------------------------------------------------------- figures
def fig_alpha_vs_acc(df: pd.DataFrame, metric: str, out: Path, dpi: int):
    if metric not in df.columns or df[metric].notna().sum() < 2:
        log(f"skip {metric}: not enough data")
        return None
    models = sorted(df["model"].dropna().unique()) if "model" in df.columns else ["all"]
    multi = len(models) > 1
    targets = [t for t in ("train_acc", "val_acc") if t in df.columns]
    row_labels = (["all"] + models) if multi else ["all"]
    nrows = len(row_labels)
    fig, axes = plt.subplots(nrows, len(targets), figsize=(4.6 * len(targets), 3.6 * nrows), squeeze=False,
                             layout="constrained")
    gap = df["gap"].to_numpy(dtype=float) if "gap" in df.columns else np.zeros(len(df))
    finite_gap = gap[np.isfinite(gap)]
    vmin = min(0.0, float(finite_gap.min())) if finite_gap.size else 0.0
    vmax = max(float(finite_gap.max()) if finite_gap.size else 1.0, 1e-3)
    last_sc = None
    for r, model in enumerate(row_labels):
        sub = df if model == "all" else df[df["model"] == model]
        for c, target in enumerate(targets):
            ax = axes[r][c]
            x, y, g = sub[metric].to_numpy(float), sub[target].to_numpy(float), sub["gap"].to_numpy(float) if "gap" in sub else None
            sc = ax.scatter(x, y, c=g, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, s=18, linewidths=0.5,
                            edgecolors="#ffffff", alpha=0.95)
            last_sc = sc
            rho, p, n = spearman(x, y)
            ax.set_xlabel(metric)
            ax.set_ylabel(target)
            title = f"{target} vs {metric}" + (f" [{model}]" if multi else "")
            ax.set_title(f"{title}\nSpearman rho={rho:.2f} (p={p:.2g}, n={n})", fontsize=9)
            if metric in ALPHA_GUIDE_METRICS:  # WW heuristics: alpha < 2 over-trained, > 6 under-trained
                for xv in (2.0, 6.0):
                    ax.axvline(xv, color=GRID, linewidth=1, linestyle="--")
            style_axes(ax)
    if last_sc is not None:
        cb = fig.colorbar(last_sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
        cb.set_label("generalization gap (train acc - val acc)", color=TEXT_SECONDARY, fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)
    path = out / f"alpha_vs_acc__{metric}.png"
    savefig(fig, path, dpi)
    return path


def fig_alpha_by_role(df: pd.DataFrame, out: Path, dpi: int):
    cols = role_columns(df)
    cols = [c for c in cols if df[c].notna().sum() >= 2]
    if not cols or "val_acc" not in df.columns:
        log("skip per-role figure: no role columns")
        return None
    n = len(cols)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False, layout="constrained")
    for i, c in enumerate(cols):
        ax = axes[i // ncols][i % ncols]
        role = c[len("alpha_role_"):]
        x, y = df[c].to_numpy(float), df["val_acc"].to_numpy(float)
        ax.scatter(x, y, s=18, color=CATEGORICAL[i % len(CATEGORICAL)], linewidths=0.5, edgecolors="#ffffff", label=role)
        rho, p, nn = spearman(x, y)
        ax.set_title(f"role '{role}': rho={rho:.2f} (p={p:.2g}, n={nn})", fontsize=9)
        ax.set_xlabel("mean alpha (layers meeting min_evals)")
        ax.set_ylabel("val_acc")
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        style_axes(ax)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    path = out / "alpha_by_role_vs_val_acc.png"
    savefig(fig, path, dpi)
    return path


def fig_trajectories(payloads, out: Path, dpi: int, key: str = "alpha_avg"):
    trajs = []
    for p in payloads:
        tr = p.get("alpha_trajectory") or []
        pts = [(t["epoch"], t.get(key)) for t in tr if t.get(key) is not None]
        if len(pts) >= 2:
            trajs.append((pts, (p.get("final") or {}).get("val_acc"), p.get("run_id")))
    if not trajs:
        log("skip trajectories: no run has >= 2 WW epochs")
        return None
    accs = np.array([t[1] if t[1] is not None else np.nan for t in trajs], dtype=float)
    vmin, vmax = np.nanmin(accs), np.nanmax(accs)
    if not np.isfinite(vmin) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    cmap = plt.get_cmap(SEQ_CMAP)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    for (pts, acc, rid), a in zip(trajs, accs):
        xs, ys = zip(*pts)
        col = cmap(0.25 + 0.75 * (a - vmin) / (vmax - vmin)) if np.isfinite(a) else "#999999"
        ax.plot(xs, ys, color=col, linewidth=1.2, alpha=0.85, marker="o", markersize=3)
    for yv in (2.0, 6.0):
        ax.axhline(yv, color=GRID, linewidth=1, linestyle="--")
    ax.set_xlabel("epoch (0 = initialisation)")
    ax.set_ylabel(key)
    ax.set_title(f"{key} over training ({len(trajs)} runs); colour = final val acc", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("final val_acc", color=TEXT_SECONDARY, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)
    style_axes(ax)
    path = out / "alpha_trajectories.png"
    savefig(fig, path, dpi)
    return path


def layer_rows(payloads) -> pd.DataFrame:
    recs = []
    for p in payloads:
        for r in p.get("ww_rows") or []:
            recs.append({"run_id": p.get("run_id"), "model": p.get("model"), "name": r.get("name"),
                         "role": r.get("role") or "unknown", "alpha": r.get("alpha"), "num_evals": r.get("num_evals"),
                         "N": r.get("N"), "M": r.get("M"), "alpha_weighted": r.get("alpha_weighted"),
                         "log_spectral_norm": r.get("log_spectral_norm"), "meets_min_evals": r.get("meets_min_evals")})
    return pd.DataFrame(recs)


def fig_alpha_hist_by_role(layers: pd.DataFrame, out: Path, dpi: int):
    if layers.empty or layers["alpha"].notna().sum() < 2:
        log("skip histogram: no layer rows")
        return None
    roles = sorted(layers["role"].dropna().unique())
    vals = layers["alpha"].dropna().to_numpy(float)
    hi = float(np.nanpercentile(vals, 99)) if len(vals) > 10 else float(vals.max())
    bins = np.linspace(0.0, max(hi, 2.5), 40)
    n = len(roles)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False, sharex=True,
                             layout="constrained")
    for i, role in enumerate(roles):
        ax = axes[i // ncols][i % ncols]
        v = layers.loc[layers["role"] == role, "alpha"].dropna().to_numpy(float)
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, color=CATEGORICAL[i % len(CATEGORICAL)], edgecolor="#ffffff",
                linewidth=0.5, label=f"{role} (n={len(v)})")
        for xv in (2.0, 6.0):
            ax.axvline(xv, color=GRID, linewidth=1, linestyle="--")
        ax.set_xlabel("alpha (per layer, final epoch)")
        ax.set_ylabel("layers")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        style_axes(ax)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    path = out / "alpha_hist_by_role.png"
    savefig(fig, path, dpi)
    return path


# --------------------------------------------------------------------------- summary
def correlation_table(df: pd.DataFrame, min_runs: int) -> pd.DataFrame:
    metrics = [m for m in ALPHA_METRICS if m in df.columns] + role_columns(df)
    targets = [t for t in TARGETS if t in df.columns]
    recs = []
    for m in metrics:
        if df[m].notna().sum() < min_runs:
            continue
        rec = {"metric": m, "n": int(df[m].notna().sum())}
        for t in targets:
            rho, p, n = spearman(df[m], df[t])
            rec[f"rho_{t}"] = rho
            rec[f"p_{t}"] = p
        recs.append(rec)
    return pd.DataFrame(recs)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "_(empty)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if not np.isfinite(v) else (f"{v:.2g}" if c.startswith("p_") else floatfmt.format(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_summary(df: pd.DataFrame, layers: pd.DataFrame, corr: pd.DataFrame, figs, out: Path, args):
    lines = ["# Sweep analysis", "", f"Inputs: {', '.join(args.inputs)}  ", f"Runs: {len(df)}", ""]
    if "model" in df.columns:
        lines += ["## Runs per model", "", md_table(df.groupby("model").size().reset_index(name="runs")), ""]
    cfg_cols = [c for c in df.columns if c.startswith("cfg_")]
    if cfg_cols:
        lines += ["## Config coverage", ""]
        cov = []
        for c in cfg_cols:
            vals = df[c].dropna()
            if vals.empty:
                continue
            if pd.api.types.is_numeric_dtype(vals) and vals.nunique() > 8:
                desc = f"min={vals.min():.4g} median={vals.median():.4g} max={vals.max():.4g}"
            else:
                def _fmt(k):
                    return int(k) if isinstance(k, float) and k.is_integer() else k
                desc = ", ".join(f"{_fmt(k)}:{v}" for k, v in vals.value_counts().sort_index().items())
            cov.append({"config": c[4:], "values (count)": desc})
        lines += [md_table(pd.DataFrame(cov)), ""]
    for t in ("train_acc", "val_acc", "test_acc", "gap", "alpha_avg", "alpha_default_avg", "alpha_min50_avg"):
        if t in df.columns and df[t].notna().any():
            v = df[t].astype(float)
            lines.append(f"- **{t}**: mean={v.mean():.4f} min={v.min():.4f} max={v.max():.4f} (n={int(v.notna().sum())})")
    lines += ["", "## Spearman correlations (each WW summary vs accuracy / gap)", "",
              "rho_<target> = Spearman rho of the metric with <target>; p_<target> its p-value; n = runs with the metric.",
              "Martin's post reports *negative* correlation of average alpha with train accuracy (lower alpha = better fit)", ""]
    lines += [md_table(corr), ""]
    if not layers.empty and layers["alpha"].notna().any():
        g = layers.dropna(subset=["alpha"]).groupby("role")["alpha"].agg(["count", "mean", "median", "min", "max"]).reset_index()
        lines += ["## Per-layer alpha by role (final epoch, all runs)", "", md_table(g), ""]
    top_cols = [c for c in ("run_id", "model", "val_acc", "train_acc", "gap", "alpha_avg", "alpha_default_avg",
                            "alpha_min50_avg", "cfg_h", "cfg_lr", "cfg_epochs", "cfg_train_subset", "cfg_label_noise") if c in df.columns]
    if "val_acc" in df.columns:
        lines += ["## Top runs by val_acc", "", md_table(df.sort_values("val_acc", ascending=False)[top_cols].head(10), floatfmt="{:.4g}"), ""]
    lines += ["## Figures", ""] + [f"- `{Path(f).name}`" for f in figs if f] + [""]
    (out / "summary.md").write_text("\n".join(lines))
    log(f"wrote {out / 'summary.md'}")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payloads = load_payloads(args.inputs)
    if not payloads:
        log("no runs found")
        return 1
    df = build_table(payloads)
    df.to_csv(out / "runs_table.csv", index=False)
    log(f"table: {len(df)} runs x {len(df.columns)} columns -> {out / 'runs_table.csv'}")
    layers = layer_rows(payloads)
    figs = []
    metrics = [args.x_metric] + [m for m in ("alpha_min50_avg", "alpha_weighted_avg") if m != args.x_metric]
    for m in metrics:
        figs.append(fig_alpha_vs_acc(df, m, out, args.dpi))
    figs.append(fig_alpha_by_role(df, out, args.dpi))
    figs.append(fig_trajectories(payloads, out, args.dpi))
    figs.append(fig_alpha_hist_by_role(layers, out, args.dpi))
    corr = correlation_table(df, args.min_runs_corr)
    corr.to_csv(out / "correlations.csv", index=False)
    write_summary(df, layers, corr, figs, out, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
