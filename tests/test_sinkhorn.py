"""Tests for :mod:`quantize.sinkhorn` (log-domain entropic OT)."""
from __future__ import annotations

import math

import pytest
import torch

from quantize.sinkhorn import (
    SinkhornConfig,
    marginal_error,
    plan_from_duals,
    sinkhorn_log,
)

DT = torch.float64


def _ot_cost_1d(x: torch.Tensor, a: torch.Tensor, y: torch.Tensor, b: torch.Tensor) -> float:
    """Exact 1-D OT cost with quadratic ground metric (merge algorithm)."""
    x, ix = torch.sort(x)
    a = a[ix]
    y, iy = torch.sort(y)
    b = b[iy]
    i = j = 0
    rx, ry = float(a[0]), float(b[0])
    cost = 0.0
    while True:
        m = min(rx, ry)
        cost += m * float((x[i] - y[j]) ** 2)
        rx -= m
        ry -= m
        if rx <= 1e-12:
            i += 1
            if i >= x.numel():
                break
            rx = float(a[i])
        if ry <= 1e-12:
            j += 1
            if j >= y.numel():
                break
            ry = float(b[j])
    return cost


def _random_problem(n: int, m: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g, dtype=DT)
    y = torch.randn(m, generator=g, dtype=DT)
    c = (x[:, None] - y[None, :]) ** 2
    a = torch.rand(n, generator=g, dtype=DT)
    a = a / a.sum()
    b = torch.rand(m, generator=g, dtype=DT)
    b = b / b.sum()
    return x, y, c, a, b


def test_marginals_are_satisfied() -> None:
    _, _, c, a, b = _random_problem(64, 24, seed=0)
    res = sinkhorn_log(a, b, c, SinkhornConfig(eps=0.05, tol=1e-10, max_iter=2000))
    assert res.converged
    assert res.marginal_err < 1e-8
    p = plan_from_duals(res.f, res.g, c, 0.05)
    assert torch.allclose(p.sum(-1), a, atol=1e-8)
    assert torch.allclose(p.sum(-2), b, atol=1e-8)


def test_matches_closed_form_1d_ot() -> None:
    """With small regularisation the entropic plan approaches the exact 1-D OT cost."""
    g = torch.Generator().manual_seed(7)
    x = torch.linspace(-2.0, 2.0, 9, dtype=DT) + 0.05 * torch.randn(9, generator=g, dtype=DT)
    y = torch.tensor([-1.0, 0.0, 1.0], dtype=DT)
    a = torch.rand(9, generator=g, dtype=DT) + 0.2
    a = a / a.sum()
    b = torch.tensor([0.3, 0.4, 0.3], dtype=DT)
    c = (x[:, None] - y[None, :]) ** 2
    res = sinkhorn_log(a, b, c, SinkhornConfig(eps=0.05, tol=1e-12, max_iter=2000))
    assert res.marginal_err < 1e-10
    p = plan_from_duals(res.f, res.g, c, 0.05)
    cost = float((p * c).sum())
    exact = _ot_cost_1d(x, a, y, b)
    assert math.isclose(cost, exact, rel_tol=1e-3)


def test_batch_matches_single_problem() -> None:
    c1 = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=DT)
    c2 = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.4, 0.6]], dtype=DT)
    a = torch.tensor([0.5, 0.3, 0.2], dtype=DT)
    b = torch.tensor([0.6, 0.4], dtype=DT)
    r1 = sinkhorn_log(a, b, c1, SinkhornConfig(eps=0.05, tol=1e-12, max_iter=3000))
    r2 = sinkhorn_log(a, b, c2, SinkhornConfig(eps=0.05, tol=1e-12, max_iter=3000))
    rb = sinkhorn_log(
        a.expand(2, -1).contiguous(),
        b.expand(2, -1).contiguous(),
        torch.stack([c1, c2]),
        SinkhornConfig(eps=0.05, tol=1e-12, max_iter=3000),
    )
    assert torch.allclose(r1.f, rb.f[0], atol=1e-8)
    assert torch.allclose(r2.f, rb.f[1], atol=1e-8)
    assert torch.allclose(r1.g, rb.g[0], atol=1e-8)
    assert torch.allclose(r2.g, rb.g[1], atol=1e-8)


def test_large_cost_stability() -> None:
    """Huge costs relative to eps must not overflow the log-domain iteration."""
    _, _, c, a, b = _random_problem(32, 8, seed=3)
    res = sinkhorn_log(a, b, c * 1e6, SinkhornConfig(eps=1.0, tol=1e-8, max_iter=1000))
    assert torch.isfinite(res.f).all() and torch.isfinite(res.g).all()
    assert math.isfinite(res.marginal_err)


def test_large_cost_with_matched_eps_converges() -> None:
    """Scaling costs and eps together keeps the problem well-posed."""
    _, _, c, a, b = _random_problem(32, 8, seed=3)
    res = sinkhorn_log(a, b, c * 1e6, SinkhornConfig(eps=1e6, tol=1e-10, max_iter=2000))
    assert res.marginal_err < 1e-8


def test_long_unconverged_runs_stay_finite() -> None:
    """Iterating far past the useful regime must not produce NaN/Inf."""
    _, _, c, a, b = _random_problem(16, 5, seed=5)
    res = sinkhorn_log(a, b, c, SinkhornConfig(eps=1e-3, tol=1e-16, max_iter=400))
    assert torch.isfinite(res.f).all() and torch.isfinite(res.g).all()
    assert all(math.isfinite(v) for v in res.trace)


def test_invalid_inputs_raise() -> None:
    g = torch.Generator().manual_seed(1)
    a = torch.rand(4, generator=g, dtype=DT)
    b = torch.rand(3, generator=g, dtype=DT)
    c = torch.rand(4, 3, dtype=DT)
    with pytest.raises(ValueError):
        sinkhorn_log(a, b, c)  # a does not sum to 1
    with pytest.raises(ValueError):
        sinkhorn_log(-a, b, c)
    with pytest.raises(ValueError):
        sinkhorn_log(a / a.sum(), b / b.sum(), c[:2, :])  # shape mismatch
    with pytest.raises(ValueError):
        sinkhorn_log(a / a.sum(), b / b.sum(), c, SinkhornConfig(eps=0.0))
    bad = c.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        sinkhorn_log(a / a.sum(), b / b.sum(), bad)
