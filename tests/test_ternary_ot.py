"""Tests for :mod:`quantize.anderson_sinkhorn` (ternary OT quantiser)."""
from __future__ import annotations

import pytest
import torch

from quantize.anderson_sinkhorn import (
    TernaryOTConfig,
    _as_2d,
    _mse_for_alpha,
    optimal_scale,
    quantize_batched,
    quantize_matrix,
    ternary_nearest,
)

DT = torch.float64


def _blocks(n_blocks: int = 6, n: int = 256, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n_blocks):
        if i % 3 == 0:
            rows.append(torch.randn(n, generator=g, dtype=DT) * 0.03)
        elif i % 3 == 1:
            u = torch.rand(n, generator=g, dtype=DT).clamp_min(1e-12)
            sign = torch.where(torch.rand(n, generator=g, dtype=DT) < 0.5, -1.0, 1.0)
            rows.append(sign * 0.02 * torch.log(u))
        else:
            sparse = torch.randn(n, generator=g, dtype=DT) * 0.05
            mask = torch.rand(n, generator=g, dtype=DT) < 0.7
            rows.append(sparse * mask.to(DT))
    return torch.stack(rows)


def test_output_values_are_ternary() -> None:
    w = _blocks()
    cfg = TernaryOTConfig(method="ot", granularity="row")
    q, alpha, _ = quantize_batched(w, cfg)
    for b in range(w.shape[0]):
        allowed = {-float(alpha[b]), 0.0, float(alpha[b])}
        unique = set(q[b].tolist())
        assert unique <= allowed


def test_round_beats_absmean() -> None:
    w = _blocks()
    q_round, _, _ = quantize_batched(w, TernaryOTConfig(method="round"))
    q_abs, _, _ = quantize_batched(w, TernaryOTConfig(method="absmean"))
    mse_round = float((w - q_round).square().mean())
    mse_abs = float((w - q_abs).square().mean())
    assert mse_round <= mse_abs * 1.001


def test_ot_on_par_with_optimal_rounding() -> None:
    w = _blocks()
    q_ot, _, info = quantize_batched(
        w,
        TernaryOTConfig(
            method="ot", eps=0.05, anderson_m=5, tol=1e-6, max_iter=1000, dtype=DT
        ),
    )
    q_round, _, _ = quantize_batched(w, TernaryOTConfig(method="round", dtype=DT))
    mse_ot = float((w - q_ot).square().mean())
    mse_round = float((w - q_round).square().mean())
    assert info["converged"], f"Sinkhorn did not converge (err={info['marginal_err']:.2e})"
    assert mse_ot <= mse_round * 1.05, f"OT MSE {mse_ot:.3e} vs round {mse_round:.3e}"


def test_optimal_scale_minimises_mse() -> None:
    w = _blocks(n_blocks=4, n=512, seed=5)
    alpha = optimal_scale(w)
    mse = _mse_for_alpha(w, alpha)
    base = w.abs().mean(-1)
    mse_heuristic = _mse_for_alpha(w, 0.7 * base)
    assert bool((mse <= mse_heuristic + 1e-18).all())
    fine = torch.linspace(0.2, 1.6, 71, dtype=DT)
    grid = torch.stack([_mse_for_alpha(w, base * f) for f in fine], dim=-1)
    mse_grid_best = grid.min(-1).values
    assert bool((mse <= mse_grid_best * 1.02).all())


def test_zero_and_tiny_blocks_are_safe() -> None:
    w = torch.zeros(3, 64, dtype=DT)
    w[1] = 1e-12
    q, alpha, _ = quantize_batched(w, TernaryOTConfig(method="ot"))
    assert torch.isfinite(q).all()
    assert float(q[0].abs().max()) == 0.0
    assert bool((alpha > 0).all())


def test_shapes_for_all_granularities() -> None:
    g = torch.Generator().manual_seed(9)
    w = torch.randn(37, 53, generator=g, dtype=DT)
    for granularity in ("tensor", "row", "group"):
        cfg = TernaryOTConfig(
            method="ot", granularity=granularity, group_size=64, dtype=DT, device="cpu"
        )
        q, stats = quantize_matrix(w, cfg)
        assert q.shape == w.shape
        assert q.dtype == w.dtype
        assert stats.n_blocks >= 1
        if granularity == "group":
            assert stats.n_blocks == -(-w.numel() // 64)


def test_group_padding_is_dropped() -> None:
    w = torch.arange(1.0, 8.0, dtype=DT).reshape(7, 1)
    view = _as_2d(w, "group", 4)
    assert view.shape == (2, 4)
    q, _ = quantize_matrix(
        w, TernaryOTConfig(granularity="group", group_size=4, dtype=DT, device="cpu")
    )
    assert q.shape == w.shape


def test_ternary_nearest_threshold() -> None:
    w = torch.tensor([[-0.9, -0.4, -0.1, 0.1, 0.4, 0.9]], dtype=DT)
    alpha = torch.tensor([1.0], dtype=DT)
    q = ternary_nearest(w, alpha)
    assert q.tolist() == [[-1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        TernaryOTConfig(method="nope")
    with pytest.raises(ValueError):
        TernaryOTConfig(granularity="nope")
    with pytest.raises(ValueError):
        quantize_matrix(torch.full((2, 2), float("nan")), TernaryOTConfig())
