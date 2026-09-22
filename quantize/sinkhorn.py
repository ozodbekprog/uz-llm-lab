"""Log-domain Sinkhorn-Knopp algorithm for entropy-regularised OT.

Given source/target probability vectors :math:`a \\in \\Delta^{n-1}`,
:math:`b \\in \\Delta^{m-1}` and a cost matrix :math:`C \\in
\\mathbb{R}^{n \\times m}`, the module solves

.. math::

    P^\\star = \\arg\\min_{P \\in U(a,b)}
    \\; \\langle P, C \\rangle - \\varepsilon H(P),
    \\qquad
    U(a,b) = \\{P \\ge 0 : P\\mathbf{1} = a, \\; P^\\top\\mathbf{1} = b\\},

where :math:`H(P) = -\\sum_{ij} P_{ij}(\\log P_{ij} - 1)` is the entropy
and :math:`\\varepsilon > 0` the regularisation strength.  The dual
potentials :math:`(f, g)` are updated in log-space,

.. math::

    f_i = \\varepsilon\\left(\\log a_i - \\log\\sum_j
        e^{(g_j - C_{ij})/\\varepsilon}\\right), \\qquad
    g_j = \\varepsilon\\left(\\log b_j - \\log\\sum_i
        e^{(f_i - C_{ij})/\\varepsilon}\\right),

which is numerically stable for arbitrary cost magnitudes (contrary to
the explicit-kernel iteration :math:`u = a / Kv`).  The plain fixed-point
iteration can be replaced by the Anderson-mixed version from
:mod:`quantize.anderson`, which typically reaches the same tolerance in
far fewer iterations.

All routines are batched over arbitrary leading dimensions and support
both CPU and CUDA tensors.

References:
    Peyre, G., & Cuturi, M. (2019). Computational Optimal Transport.
    *Foundations and Trends in Machine Learning*, 11(5-6), 355-607.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .anderson import AndersonConfig, anderson_fixed_point

__all__ = [
    "SinkhornConfig",
    "SinkhornResult",
    "sinkhorn_log",
    "plan_from_duals",
    "marginal_error",
]

_EPS_MASS = 1e-12


@dataclass
class SinkhornConfig:
    """Configuration of :func:`sinkhorn_log`.

    Attributes:
        eps: Entropic regularisation strength, ``eps > 0``.
        max_iter: Maximum number of simultaneous dual updates.
        tol: Stop when the maximum marginal violation (or, for the plain
            iteration, the maximum change of the duals) is below ``tol``.
        anderson_m: History depth for Anderson mixing; ``0`` disables it
            and runs the classic Sinkhorn iteration.
        anderson_beta: Damping factor passed to :class:`AndersonConfig`.
        anderson_reg: Tikhonov regularisation of the Anderson least squares.
    """

    eps: float = 1e-2
    max_iter: int = 300
    tol: float = 1e-6
    anderson_m: int = 0
    anderson_beta: float = 1.0
    anderson_reg: float = 1e-8

    def __post_init__(self) -> None:
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.tol <= 0.0:
            raise ValueError(f"tol must be positive, got {self.tol}")
        if self.anderson_m < 0:
            raise ValueError(f"anderson_m must be >= 0, got {self.anderson_m}")


@dataclass
class SinkhornResult:
    """Result of :func:`sinkhorn_log`.

    Attributes:
        f: Row dual potentials, shape ``(..., n)``.
        g: Column dual potentials, shape ``(..., m)``.
        iters: Number of fixed-point updates performed.
        converged: Whether the stopping tolerance was reached.
        marginal_err: Final maximum marginal violation
            :math:`\\max_k \\|P\\mathbf{1} - a\\|_\\infty` (and the
            analogous column error).
        trace: Stopping metric per iteration (marginal violation).
    """

    f: Tensor
    g: Tensor
    iters: int
    converged: bool
    marginal_err: float
    trace: List[float] = field(default_factory=list)


def _validate(a: Tensor, b: Tensor, c: Tensor, cfg: SinkhornConfig) -> None:
    """Validate shapes, finiteness and probability constraints."""
    if a.ndim < 1 or b.ndim < 1:
        raise ValueError("a and b must have at least one dimension")
    if a.shape[:-1] != b.shape[:-1]:
        raise ValueError(f"batch shape mismatch: a {tuple(a.shape)} vs b {tuple(b.shape)}")
    if c.shape[:-2] != a.shape[:-1] or c.shape[-1] != b.shape[-1] or c.shape[-2] != a.shape[-1]:
        raise ValueError(
            "cost shape mismatch: expected "
            f"{tuple(a.shape[:-1]) + (a.shape[-1], b.shape[-1])}, got {tuple(c.shape)}"
        )
    for name, v in (("a", a), ("b", b)):
        if not torch.isfinite(v).all():
            raise ValueError(f"{name} contains non-finite values")
        if (v < 0).any():
            raise ValueError(f"{name} contains negative entries")
        total = v.sum(-1)
        if not torch.allclose(total, torch.ones_like(total), atol=1e-3):
            raise ValueError(
                f"{name} must sum to 1 along the last axis "
                f"(range [{float(total.min()):.4f}, {float(total.max()):.4f}])"
            )
    if not torch.isfinite(c).all():
        raise ValueError("cost matrix contains non-finite values")


def marginal_error(f: Tensor, g: Tensor, c: Tensor, eps: float, a: Tensor, b: Tensor) -> float:
    """Return the maximum marginal violation of the plan implied by ``(f, g)``.

    Args:
        f: Row duals, shape ``(..., n)``.
        g: Column duals, shape ``(..., m)``.
        c: Cost matrix, shape ``(..., n, m)``.
        eps: Entropic regularisation strength.
        a: Source histogram, shape ``(..., n)``.
        b: Target histogram, shape ``(..., m)``.

    Returns:
        ``max(max_i |(P 1)_i - a_i|, max_j |(P^T 1)_j - b_j|)`` with
        ``P_ij = exp((f_i + g_j - C_ij) / eps)``.
    """
    log_p = (f.unsqueeze(-1) + g.unsqueeze(-2) - c) / eps
    p = log_p.clamp(max=60.0).exp()
    err_rows = (p.sum(-1) - a).abs().max()
    err_cols = (p.sum(-2) - b).abs().max()
    return float(torch.maximum(err_rows, err_cols))


def plan_from_duals(f: Tensor, g: Tensor, c: Tensor, eps: float) -> Tensor:
    """Materialise the transport plan ``P = exp((f + g - C) / eps)``."""
    return ((f.unsqueeze(-1) + g.unsqueeze(-2) - c) / eps).clamp(max=60.0).exp()


def _sinkhorn_map(log_a: Tensor, log_b: Tensor, c: Tensor, eps: float):
    """Build the batched fixed-point map ``G`` on the concatenated duals.

    The state is ``x = [f; g]`` (last axis) and one application of ``G``
    performs a full row update followed by a full column update.
    """
    n = log_a.shape[-1]

    def g_map(x: Tensor) -> Tensor:
        f = x[..., :n]
        g = x[..., n:]
        f = eps * (log_a - torch.logsumexp((g.unsqueeze(-2) - c) / eps, dim=-1))
        g = eps * (log_b - torch.logsumexp((f.unsqueeze(-1) - c) / eps, dim=-2))
        return torch.cat([f, g], dim=-1)

    return g_map


def sinkhorn_log(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    cfg: Optional[SinkhornConfig] = None,
    init: Optional[Tuple[Tensor, Tensor]] = None,
) -> SinkhornResult:
    """Solve entropic OT in log-space, optionally Anderson-accelerated.

    Args:
        a: Source histogram, shape ``(..., n)``, entries non-negative,
            last axis summing to 1.
        b: Target histogram, shape ``(..., m)``, same constraints.
        c: Cost matrix, shape ``(..., n, m)``.
        cfg: Solver configuration (defaults to :class:`SinkhornConfig`).
        init: Optional warm start ``(f0, g0)`` for the dual potentials.

    Returns:
        A :class:`SinkhornResult` holding the duals, iteration count and
        final marginal violation.

    Raises:
        ValueError: On malformed inputs or invalid configuration.

    Example:
        >>> a = torch.full((4,), 0.25)
        >>> b = torch.full((3,), 1 / 3)
        >>> c = torch.rand(4, 3)
        >>> res = sinkhorn_log(a, b, c, SinkhornConfig(eps=0.1))
        >>> res.marginal_err < 1e-5
        True
    """
    cfg = cfg or SinkhornConfig()
    b = b.to(a.dtype)
    c = c.to(a.dtype)
    _validate(a, b, c, cfg)
    log_a = a.clamp_min(_EPS_MASS).log()
    log_b = b.clamp_min(_EPS_MASS).log()
    if init is None:
        f0 = torch.zeros_like(a)
        g0 = torch.zeros_like(b)
    else:
        f0, g0 = init
        f0 = f0.to(a.dtype)
        g0 = g0.to(a.dtype)
    eps = cfg.eps
    trace: List[float] = []

    if cfg.anderson_m == 0:
        f, g = f0, g0
        converged = False
        iters = 0
        for iters in range(1, cfg.max_iter + 1):
            f = eps * (log_a - torch.logsumexp((g.unsqueeze(-2) - c) / eps, dim=-1))
            g = eps * (log_b - torch.logsumexp((f.unsqueeze(-1) - c) / eps, dim=-2))
            err = marginal_error(f, g, c, eps, a, b)
            trace.append(err)
            if err < cfg.tol:
                converged = True
                break
    else:
        n = a.shape[-1]
        state = torch.cat([f0, g0], dim=-1)

        def callback(x: Tensor, _gx: Tensor) -> bool:
            f = x[..., :n]
            g = x[..., n:]
            err = marginal_error(f, g, c, eps, a, b)
            trace.append(err)
            return err < cfg.tol

        acc_cfg = AndersonConfig(
            m=cfg.anderson_m, beta=cfg.anderson_beta, reg=cfg.anderson_reg
        )
        fp = anderson_fixed_point(
            _sinkhorn_map(log_a, log_b, c, eps),
            state,
            acc_cfg,
            tol=0.0,
            max_iter=cfg.max_iter,
            callback=callback,
        )
        f, g = fp.x[..., :n], fp.x[..., n:]
        iters = max(fp.iters, len(trace))
        converged = bool(trace and trace[-1] < cfg.tol)

    err = marginal_error(f, g, c, eps, a, b)
    return SinkhornResult(f=f, g=g, iters=iters, converged=converged, marginal_err=err, trace=trace)
