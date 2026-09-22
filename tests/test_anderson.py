"""Tests for :mod:`quantize.anderson` and the accelerated Sinkhorn."""
from __future__ import annotations

import torch

from quantize.anderson import AndersonConfig, AndersonState, anderson_fixed_point
from quantize.sinkhorn import SinkhornConfig, plan_from_duals, sinkhorn_log

DT = torch.float64


def _linear_system(n: int = 60, rho: float = 0.85, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=DT))
    d = torch.linspace(0.2, rho, n, dtype=DT)
    if rand_bool(g):
        d = -d
    a = q @ torch.diag(d) @ q.T
    b = torch.randn(n, generator=g, dtype=DT)
    x_star = torch.linalg.solve(torch.eye(n, dtype=DT) - a, b)
    return a, b, x_star


def rand_bool(g: torch.Generator) -> bool:
    return bool(torch.randint(0, 2, (1,), generator=g).item())


def _plain_iterations(g, x0, b, tol: float, max_iter: int = 20_000):
    x = x0
    for k in range(1, max_iter + 1):
        gx = g(x)
        if float((gx - x).norm()) < tol:
            return x, k
        x = gx
    return x, max_iter


def test_anderson_beats_plain_on_linear_system() -> None:
    a, b, x_star = _linear_system()
    g = lambda x: x @ a.T + b
    x0 = torch.zeros_like(b)
    x_plain, it_plain = _plain_iterations(g, x0, b, tol=1e-10)
    res = anderson_fixed_point(g, x0, AndersonConfig(m=5), tol=1e-10, max_iter=2000)
    assert res.converged
    assert torch.allclose(res.x, x_star, atol=1e-7)
    assert res.iters < it_plain
    assert it_plain > 0


def test_anderson_batched_systems() -> None:
    g = torch.Generator().manual_seed(11)
    a = torch.stack([0.5 * torch.eye(8, dtype=DT), 0.3 * torch.eye(8, dtype=DT)])
    b = torch.randn(2, 8, generator=g, dtype=DT)
    G = lambda x: (a @ x.unsqueeze(-1)).squeeze(-1) + b
    x_star = torch.linalg.solve(torch.eye(8, dtype=DT) - a, b.unsqueeze(-1)).squeeze(-1)
    res = anderson_fixed_point(G, torch.zeros_like(b), AndersonConfig(m=3), tol=1e-12)
    assert res.converged
    assert torch.allclose(res.x, x_star, atol=1e-9)
    assert res.residual.max() < 1e-12


def test_safeguard_keeps_iterates_finite() -> None:
    """A diverging map must not produce NaN/Inf, even without a fixed point."""
    G = lambda x: 1.7 * x
    res = anderson_fixed_point(
        G, torch.ones(4, dtype=DT), AndersonConfig(m=3, safeguard=1.5), tol=1e-12, max_iter=40
    )
    assert torch.isfinite(res.x).all()
    assert res.resets >= 0


def test_state_reset_clears_history() -> None:
    st = AndersonState(AndersonConfig(m=3))
    x = torch.zeros(4, dtype=DT)
    gx = torch.ones(4, dtype=DT)
    st.step(x, gx)
    st.step(x + 0.5, gx)
    assert st.history == 2
    st.reset()
    assert st.history == 0


def test_state_validation() -> None:
    st = AndersonState(AndersonConfig(m=2))
    x = torch.zeros(3, dtype=DT)
    gx = torch.ones(3, dtype=DT)
    x1 = st.step(x, gx)
    assert torch.allclose(x1, gx)
    try:
        st.step(torch.zeros(2, dtype=DT), torch.zeros(3, dtype=DT))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        AndersonConfig(m=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def _ot_problem(n: int = 96, m: int = 24, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g, dtype=DT)
    y = torch.randn(m, generator=g, dtype=DT)
    c = (x[:, None] - y[None, :]) ** 2
    a = torch.full((n,), 1.0 / n, dtype=DT)
    b = torch.full((m,), 1.0 / m, dtype=DT)
    return c, a, b


def test_anderson_accelerated_sinkhorn_matches_and_is_faster() -> None:
    c, a, b = _ot_problem()
    tol = 1e-10
    plain = sinkhorn_log(a, b, c, SinkhornConfig(eps=0.05, tol=tol, max_iter=3000, anderson_m=0))
    acc = sinkhorn_log(a, b, c, SinkhornConfig(eps=0.05, tol=tol, max_iter=3000, anderson_m=5))
    assert plain.marginal_err < 1e-8
    assert acc.marginal_err < 1e-8
    p_plain = plan_from_duals(plain.f, plain.g, c, 0.05)
    p_acc = plan_from_duals(acc.f, acc.g, c, 0.05)
    assert torch.allclose(p_plain, p_acc, atol=1e-6)
    assert acc.iters < plain.iters, f"Anderson {acc.iters} !< plain {plain.iters}"
