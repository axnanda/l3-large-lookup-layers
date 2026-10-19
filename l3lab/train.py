"""Generic training loop with per-epoch evaluation and WeightWatcher logging (Agent B).

``train_model(model, train_x, train_y, val_x, val_y, *, epochs, batch_size, lr, ...)``
returns a JSON-serialisable ``history`` dict (see :func:`train_model`).  The model is
called as ``model(to_input(x_batch))`` so the same loop serves MLP3 / SNNMLP3 (float
inputs (b,784)), the L3 classifiers (token ids (b,16)) and the char-LM (ids (b,t) ->
logits (b,t,V); loss/accuracy flatten the sequence dimension).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ww_analysis import analyze_model


# ----------------------------------------------------------------------------- utils
def to_jsonable(obj):
    """Recursively convert numpy / torch objects to plain Python; non-finite floats -> None."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        return to_jsonable(obj.item() if obj.numel() == 1 and obj.dim() == 0 else obj.tolist())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def write_json(path, obj, indent=1):
    """Atomic JSON write (tmp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(to_jsonable(obj), f, indent=indent)
    os.replace(tmp, path)


def _identity(x):
    return x


def ce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy that also handles sequence logits (b,t,V) vs targets (b,t)."""
    if logits.dim() > 2:
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
    return F.cross_entropy(logits, y)


def n_correct(logits: torch.Tensor, y: torch.Tensor) -> int:
    if logits.dim() > 2:
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
    return int((logits.argmax(dim=-1) == y).sum().item())


def n_targets(y: torch.Tensor) -> int:
    return int(y.numel())


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return int(sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)))


def make_optimizer(model: nn.Module, optimizer: str = "adam", lr: float = 1e-3,
                   weight_decay: float = 0.0, momentum: float = 0.9) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optimizer.lower()
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {optimizer!r} (use 'adam', 'adamw' or 'sgd')")


@torch.no_grad()
def evaluate(model: nn.Module, x, y, batch_size: int = 512, to_input: Optional[Callable] = None,
             max_samples: Optional[int] = None, loss_fn: Callable = ce_loss):
    """Mean loss and accuracy of ``model`` on (x, y) in eval mode -> (loss, acc)."""
    to_input = to_input or _identity
    was_training = model.training
    model.eval()
    n = len(x) if max_samples is None else min(len(x), int(max_samples))
    tot_loss, correct, count = 0.0, 0, 0
    for i in range(0, n, batch_size):
        xb, yb = x[i:min(i + batch_size, n)], y[i:min(i + batch_size, n)]
        logits = model(to_input(xb))
        tot_loss += float(loss_fn(logits, yb).item()) * n_targets(yb)
        correct += n_correct(logits, yb)
        count += n_targets(yb)
    model.train(was_training)
    if count == 0:
        return float("nan"), float("nan")
    return tot_loss / count, correct / count


# ------------------------------------------------------------------------ training
def train_model(model: nn.Module, train_x, train_y, val_x, val_y, *,
                epochs: int, batch_size: int, lr: float, weight_decay: float = 0.0,
                optimizer: str = "adam", momentum: float = 0.9, seed: int = 0,
                log_every_epoch: bool = True, ww_every: int = 1, ww_kwargs: Optional[dict] = None,
                out_dir=None, test_x=None, test_y=None,
                to_input: Optional[Callable] = None, loss_fn: Callable = ce_loss,
                ww_at_init: bool = False, ww_final: bool = True,
                save_every_epoch: bool = False, train_eval_max: Optional[int] = 10000,
                eval_batch_size: int = 512, grad_clip: Optional[float] = None,
                max_batches_per_epoch: Optional[int] = None, stop_fn: Optional[Callable[[], bool]] = None,
                log_fn: Callable[[str], None] = print, drop_last: bool = False) -> dict:
    """Train ``model`` and return a JSON-serialisable history.

    history keys
      config        : the training hyper-parameters
      n_params      : parameter count
      epoch, train_loss (running mean over the epoch), train_acc_running,
      train_acc / train_loss_eval (eval-mode pass over <= train_eval_max training samples),
      val_loss, val_acc, [test_loss, test_acc], wall (cumulative seconds), epoch_time
      ww            : list of {"epoch", "rows", "summary", "time_sec"} (epoch 0 = init when ww_at_init);
                      WW runs every ``ww_every`` epochs (0 disables) and after the last epoch when ww_final
      final         : full-train / val / test loss+acc after training, gap = train_acc - val_acc, total wall
      stopped_early : True when ``stop_fn()`` returned True (training stops after that epoch)
    When ``out_dir`` is given, ``final.pt`` (state_dict), ``history.json`` and optionally
    ``epoch_XXX.pt`` are written there.
    """
    to_input = to_input or _identity
    ww_kwargs = dict(ww_kwargs or {})
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = make_optimizer(model, optimizer, lr, weight_decay, momentum)
    out_dir = Path(out_dir) if out_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    n = len(train_x)
    has_test = test_x is not None and test_y is not None
    history: dict = {
        "config": dict(epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
                       weight_decay=float(weight_decay), optimizer=optimizer, momentum=float(momentum),
                       seed=int(seed), ww_every=int(ww_every), ww_kwargs=to_jsonable(ww_kwargs),
                       n_train=int(n), n_val=int(len(val_x)), n_test=int(len(test_x)) if has_test else 0,
                       ww_at_init=bool(ww_at_init), ww_final=bool(ww_final)),
        "n_params": count_params(model),
        "epoch": [], "train_loss": [], "train_acc_running": [], "train_acc": [], "train_loss_eval": [],
        "val_loss": [], "val_acc": [], "wall": [], "epoch_time": [], "ww": [], "stopped_early": False,
    }
    if has_test:
        history["test_loss"], history["test_acc"] = [], []

    t_start = time.time()

    def run_ww(epoch: int):
        kw = dict(ww_kwargs)
        kw.setdefault("tag", f"epoch{epoch}")
        t0 = time.time()
        rows, summary = analyze_model(model, **kw)
        entry = {"epoch": int(epoch), "rows": rows, "summary": summary, "time_sec": time.time() - t0}
        history["ww"].append(entry)
        return entry

    def fmt_ww(entry):
        s = entry["summary"]
        parts = []
        for key, label in (("alpha_avg", "alpha"), ("alpha_default_avg", "alpha_def"),
                           ("alpha_weighted_avg", "alpha_w")):
            v = s.get(key)
            if v is not None:
                parts.append(f"{label}={v:.3f}")
        return " ".join(parts) + f" ww={entry['time_sec']:.1f}s"

    if ww_at_init and ww_every >= 0:
        entry = run_ww(0)
        if log_every_epoch:
            log_fn(f"  epoch 0 (init) {fmt_ww(entry)}")

    last_ww_epoch = 0 if ww_at_init else None
    for epoch in range(1, int(epochs) + 1):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot_loss, correct, count, n_batches = 0.0, 0, 0, 0
        for bi, i in enumerate(range(0, n, batch_size)):
            idx = perm[i:i + batch_size]
            if drop_last and len(idx) < batch_size:
                break
            if max_batches_per_epoch is not None and bi >= max_batches_per_epoch:
                break
            xb, yb = train_x[idx], train_y[idx]
            logits = model(to_input(xb))
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot_loss += float(loss.item()) * n_targets(yb)
            correct += n_correct(logits.detach(), yb)
            count += n_targets(yb)
            n_batches += 1
        train_time = time.time() - t_ep
        tr_loss_eval, tr_acc = evaluate(model, train_x, train_y, eval_batch_size, to_input,
                                        max_samples=train_eval_max, loss_fn=loss_fn)
        va_loss, va_acc = evaluate(model, val_x, val_y, eval_batch_size, to_input, loss_fn=loss_fn)
        history["epoch"].append(epoch)
        history["train_loss"].append(tot_loss / max(count, 1))
        history["train_acc_running"].append(correct / max(count, 1))
        history["train_acc"].append(tr_acc)
        history["train_loss_eval"].append(tr_loss_eval)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        msg = (f"  epoch {epoch}/{epochs} loss={tot_loss / max(count, 1):.4f} "
               f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
        if has_test:
            te_loss, te_acc = evaluate(model, test_x, test_y, eval_batch_size, to_input, loss_fn=loss_fn)
            history["test_loss"].append(te_loss)
            history["test_acc"].append(te_acc)
            msg += f" test_acc={te_acc:.4f}"
        is_last = epoch == int(epochs)
        do_ww = (ww_every > 0 and epoch % ww_every == 0) or (ww_final and is_last)
        stop = bool(stop_fn()) if stop_fn is not None else False
        if stop and ww_final and not do_ww:
            do_ww = True
        if do_ww:
            entry = run_ww(epoch)
            last_ww_epoch = epoch
            msg += " " + fmt_ww(entry)
        history["epoch_time"].append(time.time() - t_ep)
        history["train_time"] = history.get("train_time", []) + [train_time]
        history["wall"].append(time.time() - t_start)
        msg += f" ({history['epoch_time'][-1]:.1f}s)"
        if log_every_epoch:
            log_fn(msg)
        if out_dir is not None:
            if save_every_epoch:
                torch.save(model.state_dict(), out_dir / f"epoch_{epoch:03d}.pt")
            write_json(out_dir / "history.json", history)
        if stop:
            history["stopped_early"] = True
            if log_every_epoch:
                log_fn(f"  stop_fn() requested stop after epoch {epoch}")
            break

    if int(epochs) == 0 and ww_final and not history["ww"]:
        run_ww(0)

    # final full evaluation -------------------------------------------------------
    tr_loss, tr_acc = evaluate(model, train_x, train_y, eval_batch_size, to_input, loss_fn=loss_fn)
    va_loss, va_acc = evaluate(model, val_x, val_y, eval_batch_size, to_input, loss_fn=loss_fn)
    final = {"train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc,
             "gap": tr_acc - va_acc, "epochs_run": len(history["epoch"]),
             "wall": time.time() - t_start, "last_ww_epoch": last_ww_epoch}
    if has_test:
        te_loss, te_acc = evaluate(model, test_x, test_y, eval_batch_size, to_input, loss_fn=loss_fn)
        final["test_loss"], final["test_acc"] = te_loss, te_acc
    history["final"] = final
    if out_dir is not None:
        torch.save(model.state_dict(), out_dir / "final.pt")
        write_json(out_dir / "history.json", history)
    return to_jsonable(history)


def final_ww(history: dict) -> Optional[dict]:
    """The last WW entry of a history (or None)."""
    ww = history.get("ww") or []
    return ww[-1] if ww else None


__all__ = ["train_model", "evaluate", "make_optimizer", "to_jsonable", "write_json",
           "ce_loss", "n_correct", "count_params", "final_ww"]
