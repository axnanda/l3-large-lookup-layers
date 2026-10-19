"""Spiking-neuron primitives: straight-through surrogate spikes and a leaky integrate-and-fire (LIF) cell.

Design:
* ``spike(v, k)`` is exactly 0/1 in the forward pass (Heaviside of ``v``) and uses the derivative of the
  sigmoid surrogate ``sg(v) = sigmoid(k v)`` in the backward pass. It is built with the straight-through
  trick ``heaviside(v).detach() + (sg(v) - sg(v).detach())`` so no custom autograd Function is needed and
  the forward value is *exactly* the Heaviside value (the surrogate terms cancel to 0 in forward).
* ``LIF`` unrolls the recurrence over the leading time axis with python lists + ``torch.stack``; there are
  no in-place operations on tensors that take part in autograd.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


def surrogate_sigmoid(k: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``sg(v) = sigmoid(k v)``; its derivative ``k sg (1 - sg)`` is the surrogate for the spike."""

    def sg(v: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(k * v)

    return sg


def heaviside(v: torch.Tensor) -> torch.Tensor:
    """Exact 0/1 spike: 1 where ``v > 0`` (a membrane exactly at threshold does not fire)."""
    return (v > 0).to(v.dtype)


def spike(v: torch.Tensor, k: float = 4.0) -> torch.Tensor:
    """Straight-through spike: forward value is exactly ``heaviside(v)`` (0.0 or 1.0), backward uses
    ``d/dv sigmoid(k v)``.
    """
    sg = torch.sigmoid(k * v)
    return heaviside(v).detach() + (sg - sg.detach())


def rate(spikes: torch.Tensor) -> torch.Tensor:
    """Mean firing rate over the leading time axis: ``spikes.mean(0)``."""
    return spikes.mean(0)


class LIF(nn.Module):
    """Leaky integrate-and-fire layer with surrogate-gradient spikes.

    Dynamics (``current`` has shape (T, ..., n)):

        u_t = beta * u_{t-1} + I_t - theta * s_{t-1}    (reset="soft", the default)
        u_t = (beta * u_{t-1} + I_t) * (1 - s_{t-1})    (reset="hard")
        u_t = beta * u_{t-1} + I_t                      (reset="none")
        s_t = spike(u_t - theta, k)                     exact 0/1 forward, sigmoid-surrogate backward

    with ``u_{-1} = s_{-1} = 0``. ``forward`` returns ``(spikes, membranes)``, both (T, ..., n); ``step``
    advances a single time step so callers can interleave other computation (used by the L3 SNN model).
    If ``detach_reset`` the reset term carries no gradient (a common stabilisation trick); default False.
    """

    def __init__(self, beta: float = 0.9, theta: float = 1.0, k: float = 4.0, reset: str = "soft",
                 detach_reset: bool = False):
        super().__init__()
        if reset not in ("soft", "hard", "none"):
            raise ValueError("reset must be 'soft', 'hard' or 'none'")
        self.beta, self.theta, self.k, self.reset, self.detach_reset = float(beta), float(theta), float(k), reset, detach_reset

    def step(self, current: torch.Tensor, u_prev: Optional[torch.Tensor] = None,
             s_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """One time step. Returns ``(s_t, u_t)``. ``u_prev``/``s_prev`` default to zeros."""
        if u_prev is None:
            u = current
        else:
            u = self.beta * u_prev + current
        if s_prev is not None and self.reset != "none":
            s_r = s_prev.detach() if self.detach_reset else s_prev
            if self.reset == "soft":
                u = u - self.theta * s_r
            else:
                u = u * (1.0 - s_r)
        s = spike(u - self.theta, self.k)
        return s, u

    def forward(self, current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if current.ndim < 2:
            raise ValueError("current must have shape (T, ..., n)")
        spikes, mems = [], []
        u = s = None
        for t in range(current.shape[0]):
            s, u = self.step(current[t], u, s)
            spikes.append(s)
            mems.append(u)
        return torch.stack(spikes, 0), torch.stack(mems, 0)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, theta={self.theta}, k={self.k}, reset={self.reset!r}"


def spike_stats(spikes: torch.Tensor, time_dim: int = 0, unit_dim: int = -1) -> dict:
    """Diagnostics for a spike tensor: mean rate, fraction of units (last dim) that never fire."""
    with torch.no_grad():
        s = spikes.detach()
        n_units = s.shape[unit_dim]
        per_unit = s.movedim(unit_dim, -1).reshape(-1, n_units).sum(0)
        return {"rate": float(s.mean()), "frac_silent_units": float((per_unit == 0).float().mean()),
                "n_units": int(n_units)}
