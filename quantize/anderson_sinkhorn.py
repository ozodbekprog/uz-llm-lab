"""Anderson-Sinkhorn ternary quantisation (MATRIX-DTO, layer 1).

For every weight block :math:`W \\in \\mathbb{R}^{B \\times n}` (row,
group or the flattened tensor) the quantiser solves an entropic optimal
transport problem between the empirical weight distribution

.. math::
    \\mu_F = \\frac{1}{n} \\sum_{i=1}^{n} \\delta_{w_i}

and the ternary measure supported on
:math:`\\{-\\alpha, 0, +\\alpha\\}`,

.. math::
    \\nu_\\alpha = p_- \\delta_{-\\alpha} + p_0 \\delta_{0}
                 + p_+ \\delta_{+\\alpha},

with masses ``(p_-, p_0, p_+)`` induced by nearest-neighbour
thresholding at :math:`\\alpha / 2`.  The scale :math:`\\alpha` is chosen
to minimise the per-block nearest-neighbour distortion

.. math::
    \\alpha^\\star = \\arg\\min_{\\alpha \\ge 0} \\;
    \\frac{1}{n} \\sum_i \\min\\left(w_i^2, (|w_i| - \\alpha)^2\\right),

which is a strictly better choice than the classical
``0.7 * E|w|`` heuristic and is found here with a two-stage grid search
(coarse logarithmic grid plus per-row ternary refinement).

The Sinkhorn coupling :math:`P^\\star` (see :mod:`quantize.sinkhorn`,
optionally Anderson-accelerated) is condensed into the barycentric
projection

.. math::
    \\tilde w_i = \\frac{\\sum_j P^\\star_{ij} t_j}
                       {\\sum_j P^\\star_{ij}},
    \\qquad t = (-\\alpha, 0, +\\alpha),

and finally snapped to the nearest support point.  The cost matrix is
normalised by :math:`\\alpha^2` so that the entropic regularisation
``eps`` is scale-free.

Methods:
    ``ot``: full pipeline above.
    ``round``: optimal scale + direct nearest rounding (ablation).
    ``absmean``: legacy ``q = clamp(round(w / E|w|), -1, 1) * E|w|``
        (the baseline used before MATRIX-DTO).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor

from .repro import resolve_device
from .sinkhorn import SinkhornConfig, plan_from_duals, sinkhorn_log

__all__ = [
    "TernaryOTConfig",
    "TernaryStats",
    "optimal_scale",
    "ternary_nearest",
    "quantize_batched",
    "quantize_matrix",
]

_METHODS = ("ot", "round", "absmean")
_GRANULARITIES = ("tensor", "row", "group")


@dataclass
class TernaryOTConfig:
    """Configuration of the ternary OT quantiser.

    Attributes:
        method: ``"ot"``, ``"round"`` or ``"absmean"``.
        granularity: ``"tensor"``, ``"row"`` or ``"group"``; controls the
            partition of the weight matrix into independently quantised
            blocks.
        group_size: Block width for ``granularity="group"``.
        eps: Scale-free entropic regularisation strength.
        max_iter: Maximum Sinkhorn iterations per block.
        tol: Marginal-violation tolerance of the Sinkhorn solver.
        anderson_m: Anderson history depth; ``0`` disables acceleration.
        anderson_beta: Damping factor of the Anderson mixing.
        alpha_coarse: Number of points in the coarse scale grid.
        alpha_refine: Number of ternary-search refinement steps.
        alpha_span: Relative bracket half-width around the coarse optimum.
        alpha_min_factor: Lower edge of the coarse scale grid (relative to
            ``E|w|``).
        alpha_max_factor: Upper edge of the coarse scale grid.
        min_alpha: Floor for the scale to avoid division by zero.
        eps_mass: Floor for target masses to keep them strictly positive.
        dtype: Computation dtype (``torch.float32`` recommended; use
            ``torch.float64`` for very tight tolerances).
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
    """

    method: str = "ot"
    granularity: str = "row"
    group_size: int = 128
    eps: float = 0.05
    max_iter: int = 200
    tol: float = 1e-6
    anderson_m: int = 5
    anderson_beta: float = 1.0
    alpha_coarse: int = 33
    alpha_refine: int = 16
    alpha_span: float = 0.2
    alpha_min_factor: float = 0.25
    alpha_max_factor: float = 1.75
    min_alpha: float = 1e-8
    eps_mass: float = 1e-9
    dtype: torch.dtype = torch.float32
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {self.method!r}")
        if self.granularity not in _GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {_GRANULARITIES}, got {self.granularity!r}"
            )
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.alpha_coarse < 3 or self.alpha_refine < 1:
            raise ValueError("alpha_coarse >= 3 and alpha_refine >= 1 are required")


@dataclass
class TernaryStats:
    """Per-block quantisation statistics.

    Attributes:
        mse: Mean squared error ``E[(w - q)^2]``.
        rel_err: Relative Frobenius error ``||W - Q||_F / ||W||_F``.
        alpha_mean: Mean ternary scale over blocks.
        sinkhorn_iters: Iterations used by the Sinkhorn solver (0 for
            non-OT methods).
        sinkhorn_converged: Whether every block met ``tol`` (True for
            non-OT methods).
        marginal_err: Final maximum marginal violation (0 for non-OT).
        n_blocks: Number of independently quantised blocks.
        time_s: Wall-clock time of the block quantisation.
        name: Optional layer name.
    """

    mse: float
    rel_err: float
    alpha_mean: float
    sinkhorn_iters: int
    sinkhorn_converged: bool
    marginal_err: float
    n_blocks: int
    time_s: float
    name: str = ""


def _as_2d(w: Tensor, granularity: str, group_size: int) -> Tensor:
    """Flatten ``w`` into a ``(n_blocks, block_size)`` view.

    Args:
        w: Weight tensor of arbitrary shape.
        granularity: One of ``"tensor"``, ``"row"``, ``"group"``.
        group_size: Block width for ``"group"``.

    Returns:
        A 2-D tensor sharing memory with ``w`` when possible.  For
        ``"group"`` the flat weight vector is zero-padded to a multiple of
        ``group_size`` (padded zeros quantise to zero and are harmlessly
        ignored downstream).
    """
    if granularity == "row":
        return w.reshape(-1, w.shape[-1])
    if granularity == "tensor":
        return w.reshape(1, -1)
    flat = w.reshape(-1)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.reshape(-1, group_size)


def _mse_for_alpha(w: Tensor, alpha: Tensor) -> Tensor:
    """Nearest-rounding MSE of a ternary quantisation with scale ``alpha``.

    Args:
        w: Blocks, shape ``(B, n)``.
        alpha: Per-block scales, shape ``(B,)``.

    Returns:
        Per-block MSE, shape ``(B,)``.
    """
    half = (alpha / 2.0)[:, None]
    abs_w = w.abs()
    small = abs_w <= half
    err_small = (w * small).square().sum(-1)
    err_big = (((abs_w - alpha[:, None]) * ~small).square()).sum(-1)
    return (err_small + err_big) / w.shape[-1]


def optimal_scale(
    w: Tensor,
    coarse: int = 33,
    refine: int = 16,
    span: float = 0.2,
    min_factor: float = 0.25,
    max_factor: float = 1.75,
    min_alpha: float = 1e-8,
) -> Tensor:
    """Per-block MSE-optimal nearest-rounding scale.

    Minimises the piecewise-quadratic objective :func:`_mse_for_alpha` by
    evaluating a logarithmic grid around ``E|w|`` and refining the best
    bracket with ternary search.

    Args:
        w: Blocks, shape ``(B, n)``.
        coarse: Number of coarse grid points.
        refine: Number of ternary refinement iterations.
        span: Relative half-width of the refinement bracket.
        min_factor: Lower edge of the coarse grid (times ``E|w|``).
        max_factor: Upper edge of the coarse grid.
        min_alpha: Floor for the returned scale.

    Returns:
        Optimal per-block scales, shape ``(B,)``.
    """
    base = w.abs().mean(-1).clamp_min(min_alpha)
    factors = torch.logspace(
        math.log(min_factor), math.log(max_factor), coarse, dtype=w.dtype, device=w.device
    )
    mses = torch.stack([_mse_for_alpha(w, base * f) for f in factors], dim=-1)
    best = factors[mses.argmin(-1)]
    lo = (best * (1.0 - span)).clamp_min(min_factor)
    hi = (best * (1.0 + span)).clamp_max(max_factor)
    for _ in range(refine):
        c = lo + (hi - lo) / 3.0
        d = hi - (hi - lo) / 3.0
        mc = _mse_for_alpha(w, base * c)
        md = _mse_for_alpha(w, base * d)
        take_c = mc <= md
        hi = torch.where(take_c, d, hi)
        lo = torch.where(take_c, lo, c)
    return (base * (lo + hi) / 2.0).clamp_min(min_alpha)


def ternary_nearest(w: Tensor, alpha: Tensor) -> Tensor:
    """Snap ``w`` to ``{-alpha, 0, +alpha}`` with threshold ``alpha / 2``."""
    half = (alpha / 2.0)[:, None]
    pos = alpha[:, None].expand_as(w)
    return torch.where(w > half, pos, torch.where(w < -half, -pos, torch.zeros_like(w)))


def _target_masses(w: Tensor, alpha: Tensor, eps_mass: float) -> Tensor:
    """Ternary masses ``(p_-, p_0, p_+)`` from thresholding at ``alpha / 2``."""
    half = (alpha / 2.0)[:, None]
    p_pos = (w > half).to(w.dtype).mean(-1)
    p_neg = (w < -half).to(w.dtype).mean(-1)
    p_zero = (1.0 - p_pos - p_neg).clamp_min(0.0)
    masses = torch.stack([p_neg, p_zero, p_pos], dim=-1).clamp_min(eps_mass)
    return masses / masses.sum(-1, keepdim=True)


def _scale_free_cost(w: Tensor, support: Tensor, alpha: Tensor) -> Tensor:
    """``C_ij = (w_i - t_j)^2 / alpha^2`` to make ``eps`` scale-free."""
    return (w.unsqueeze(-1) - support.unsqueeze(-2)).square() / alpha.square()[:, None, None]


def quantize_batched(w: Tensor, cfg: TernaryOTConfig) -> Tuple[Tensor, Tensor, dict]:
    """Quantise a batch of weight blocks.

    Args:
        w: Blocks, shape ``(B, n)``, already on the target device/dtype.
        cfg: Quantiser configuration.

    Returns:
        Tuple ``(q, alpha, info)`` where ``q`` are the quantised blocks
        (same shape as ``w``), ``alpha`` the per-block scales and ``info``
        Sinkhorn diagnostics.
    """
    if w.ndim != 2:
        raise ValueError(f"expected 2-D blocks, got shape {tuple(w.shape)}")
    info = {"iters": 0, "converged": True, "marginal_err": 0.0}
    if cfg.method == "absmean":
        alpha = w.abs().mean(-1).clamp_min(cfg.min_alpha)
        q = torch.clamp(torch.round(w / alpha[:, None]), -1.0, 1.0) * alpha[:, None]
        return q, alpha, info
    alpha = optimal_scale(
        w,
        coarse=cfg.alpha_coarse,
        refine=cfg.alpha_refine,
        span=cfg.alpha_span,
        min_factor=cfg.alpha_min_factor,
        max_factor=cfg.alpha_max_factor,
        min_alpha=cfg.min_alpha,
    )
    if cfg.method == "round":
        return ternary_nearest(w, alpha), alpha, info
    zeros = torch.zeros_like(alpha)
    support = torch.stack([-alpha, zeros, alpha], dim=-1)  # (B, 3)
    masses = _target_masses(w, alpha, cfg.eps_mass)
    cost = _scale_free_cost(w, support, alpha)
    source = torch.full_like(w, 1.0 / w.shape[-1])
    res = sinkhorn_log(
        source,
        masses,
        cost,
        SinkhornConfig(
            eps=cfg.eps,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            anderson_m=cfg.anderson_m,
            anderson_beta=cfg.anderson_beta,
        ),
    )
    plan = plan_from_duals(res.f, res.g, cost, cfg.eps)
    bary = (plan * support.unsqueeze(-2)).sum(-1) / plan.sum(-1).clamp_min(1e-30)
    q = ternary_nearest(bary, alpha)
    info = {"iters": res.iters, "converged": res.converged, "marginal_err": res.marginal_err}
    return q, alpha, info


def quantize_matrix(w: Tensor, cfg: TernaryOTConfig, name: str = "") -> Tuple[Tensor, TernaryStats]:
    """Quantise a weight matrix with the configured granularity.

    Args:
        w: Weight matrix of arbitrary shape.
        cfg: Quantiser configuration.
        name: Optional layer name for logging/reporting.

    Returns:
        Tuple ``(q, stats)``: the quantised tensor (same shape and dtype
        as ``w``, detached) and its :class:`TernaryStats`.

    Raises:
        ValueError: If ``w`` is not finite.
    """
    if not torch.isfinite(w).all():
        raise ValueError(f"weight tensor contains non-finite values ({name or 'unnamed'})")
    t0 = time.perf_counter()
    original_dtype = w.dtype
    device = resolve_device(cfg.device)
    blocks = _as_2d(w.detach().to(device=device, dtype=cfg.dtype), cfg.granularity, cfg.group_size)
    q_blocks, alpha, info = quantize_batched(blocks, cfg)
    q = q_blocks.reshape(-1)[: w.numel()].reshape(w.shape)
    q = q.to(device=w.device, dtype=original_dtype).detach()
    elapsed = time.perf_counter() - t0
    diff = (w.detach().to(torch.float32).reshape(-1) - q.to(torch.float32).reshape(-1))
    ref = w.detach().to(torch.float32).reshape(-1).norm().clamp_min(1e-30)
    stats = TernaryStats(
        mse=float(diff.square().mean()),
        rel_err=float(diff.norm() / ref),
        alpha_mean=float(alpha.to(torch.float32).mean()),
        sinkhorn_iters=int(info["iters"]),
        sinkhorn_converged=bool(info["converged"]),
        marginal_err=float(info["marginal_err"]),
        n_blocks=int(blocks.shape[0]),
        time_s=elapsed,
        name=name,
    )
    return q.to(original_dtype).detach(), stats
