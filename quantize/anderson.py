"""Anderson acceleration for batched fixed-point iterations.

This module implements the type-II Anderson mixing scheme of
Walker & Ni (2011).  Given an iteration :math:`x_{k+1} = G(x_k)`, the
accelerator stores the last :math:`m` residuals
:math:`r_k = G(x_k) - x_k` and combines them through

.. math::

    \\gamma^k = \\arg\\min_{\\gamma}
    \\left\\| \\sum_i \\gamma_i r_i \\right\\|_2^2
    \\quad \\text{s.t.} \\quad \\sum_i \\gamma_i = 1,

which is reduced to an unconstrained least-squares problem in the basis
of residual differences.  The next iterate is the damped combination

.. math::

    x_{k+1} = \\sum_i \\gamma_i \\left( x_i + \\beta\\, r_i \\right),

with damping factor :math:`\\beta \\in (0, 1]`.  The least-squares
problem is solved independently for every leading batch dimension
through regularised normal equations :math:`(D^\\top D + \\lambda I)\\,
\\alpha = D^\\top r_k`, so the whole routine is batched on CPU and GPU.

References:
    Walker, H. F., & Ni, P. (2011). Anderson acceleration for fixed-point
    iterations. *SIAM Journal on Numerical Analysis*, 49(4), 1715-1735.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch import Tensor

__all__ = [
    "AndersonConfig",
    "AndersonState",
    "FixedPointResult",
    "anderson_fixed_point",
]


@dataclass
class AndersonConfig:
    """Configuration for Anderson mixing.

    Attributes:
        m: History depth (number of residuals kept, ``m >= 1``).
        beta: Damping factor in ``(0, 1]``; ``1.0`` is undamped.
        reg: Tikhonov regularisation added to the normal equations.
        safeguard: Reset the history when the residual norm grows by more
            than this factor in a single step.
    """

    m: int = 5
    beta: float = 1.0
    reg: float = 1e-8
    safeguard: float = 1e6

    def __post_init__(self) -> None:
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m}")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {self.beta}")
        if self.reg < 0.0:
            raise ValueError(f"reg must be non-negative, got {self.reg}")
        if self.safeguard <= 1.0:
            raise ValueError(f"safeguard must be > 1, got {self.safeguard}")


class AndersonState:
    """Batched Anderson mixing state.

    The state holds the last ``m + 1`` pairs ``(x_i, r_i)`` where
    ``r_i = G(x_i) - x_i``.  :meth:`step` returns the accelerated next
    iterate; call :meth:`reset` to drop the history (e.g. after a
    divergence safeguard).

    Args:
        cfg: Anderson configuration.
    """

    def __init__(self, cfg: AndersonConfig) -> None:
        self.cfg = cfg
        self.reset_count = 0
        self._x: List[Tensor] = []
        self._r: List[Tensor] = []

    def reset(self) -> None:
        """Drop the stored history."""
        self._x.clear()
        self._r.clear()

    @property
    def history(self) -> int:
        """Number of stored residuals."""
        return len(self._r)

    def step(self, x: Tensor, gx: Tensor) -> Tensor:
        """Return the next iterate given ``x`` and ``G(x)``.

        Args:
            x: Current iterate, shape ``(..., D)``.
            gx: Evaluation of the fixed-point map, shape ``(..., D)``.

        Returns:
            Accelerated next iterate with the same shape as ``x``.

        Raises:
            ValueError: If ``x`` and ``gx`` have different shapes.
        """
        if x.shape != gx.shape:
            raise ValueError(f"shape mismatch: x {tuple(x.shape)} vs G(x) {tuple(gx.shape)}")
        r = gx - x
        self._x.append(x)
        self._r.append(r)
        if len(self._r) > self.cfg.m + 1:
            self._x.pop(0)
            self._r.pop(0)
        if len(self._r) == 1:
            return x + self.cfg.beta * r
        return self._mix()

    def _mix(self) -> Tensor:
        """Solve the Anderson least-squares problem and mix the iterates."""
        r_k = self._r[-1]
        diffs = torch.stack([r_i - r_k for r_i in self._r[:-1]], dim=-1)  # (..., D, k)
        dt = diffs.transpose(-1, -2)
        ata = dt @ diffs  # (..., k, k)
        atb = -(dt @ r_k.unsqueeze(-1))  # solve min ||r_k + D alpha||
        k = ata.shape[-1]
        eye = torch.eye(k, dtype=ata.dtype, device=ata.device)
        scale = ata.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1)
        ata = ata + (self.cfg.reg * scale.clamp_min(1e-30)) * eye
        try:
            alpha = torch.linalg.solve(ata, atb).squeeze(-1)  # (..., k)
        except RuntimeError:
            self.reset_count += 1
            return self._x[-1] + self.cfg.beta * r_k
        if not torch.isfinite(alpha).all():
            self.reset_count += 1
            return self._x[-1] + self.cfg.beta * r_k
        gamma = torch.cat([alpha, 1.0 - alpha.sum(-1, keepdim=True)], dim=-1)  # (..., k+1)
        xs = torch.stack(self._x, dim=-2)  # (..., k+1, D)
        rs = torch.stack(self._r, dim=-2)
        return (gamma.unsqueeze(-1) * (xs + self.cfg.beta * rs)).sum(-2)


@dataclass
class FixedPointResult:
    """Outcome of :func:`anderson_fixed_point`.

    Attributes:
        x: Final iterate.
        iters: Number of map evaluations performed.
        converged: Whether the stopping criterion was met.
        residual: Per-batch residual norms at the final iterate.
        trace: Residual norms (max over batch) per iteration; empty when a
            custom ``callback`` controls stopping.
        resets: Number of history resets triggered by safeguards.
    """

    x: Tensor
    iters: int
    converged: bool
    residual: Tensor
    trace: List[float] = field(default_factory=list)
    resets: int = 0


def _align(mask: Tensor, x: Tensor) -> Tensor:
    """Make a batch mask broadcastable against ``x``."""
    return mask.unsqueeze(-1) if x.dim() > 1 else mask


def anderson_fixed_point(
    G: Callable[[Tensor], Tensor],
    x0: Tensor,
    cfg: Optional[AndersonConfig] = None,
    tol: float = 1e-6,
    max_iter: int = 100,
    callback: Optional[Callable[[Tensor, Tensor], Optional[bool]]] = None,
) -> FixedPointResult:
    """Iterate ``x = G(x)`` with Anderson acceleration.

    Args:
        G: Fixed-point map; must preserve shape.
        x0: Initial iterate, shape ``(..., D)``.
        cfg: Anderson configuration (defaults to ``AndersonConfig()``).
        tol: Stop when ``max_batch ||G(x) - x||_2 < tol``.  Ignored when a
            ``callback`` is given and returns True.
        max_iter: Maximum number of map evaluations.
        callback: Optional hook ``callback(x, G(x)) -> bool``; returning
            True stops the iteration (used by the Sinkhorn wrapper to stop
            on marginal violation instead of the raw residual).

    Returns:
        A :class:`FixedPointResult`.  When stopping via ``callback`` the
        returned iterate is ``x`` (whose image was just evaluated), not
        the unfinished accelerated step.
    """
    cfg = cfg or AndersonConfig()
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    acc = AndersonState(cfg)
    x = x0
    gx = G(x)
    evals = 1
    rn = (gx - x).norm(dim=-1)
    trace: List[float] = [float(rn.max())]
    converged = False
    for _ in range(max_iter):
        if callback is not None:
            if callback(x, gx):
                converged = True
                break
        elif float(rn.max()) < tol:
            converged = True
            break
        x_new = acc.step(x, gx)
        gx_new = G(x_new)
        evals += 1
        rn_new = (gx_new - x_new).norm(dim=-1)
        bad = ~torch.isfinite(rn_new) | (rn_new > cfg.safeguard * rn.clamp_min(1e-30))
        if bool(bad.any()):
            acc.reset()
            x_new = torch.where(_align(bad, x_new), x + cfg.beta * (gx - x), x_new)
            gx_new = G(x_new)
            evals += 1
            rn_new = (gx_new - x_new).norm(dim=-1)
        x, gx, rn = x_new, gx_new, rn_new
        trace.append(float(rn.max()))
    return FixedPointResult(x, evals, converged, rn, trace, acc.reset_count)
