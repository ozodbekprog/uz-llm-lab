"""Tests for :mod:`train.dto_layer` (MATRIX-DTO layer 2 core)."""
from __future__ import annotations

import pytest
import torch

from quantize.anderson_sinkhorn import optimal_scale, ternary_nearest
from train.dto_layer import DTOConfig, DTOTernaryLinear


def _ternary_layer(out: int = 6, inn: int = 8) -> DTOTernaryLinear:
    g = torch.Generator().manual_seed(0)
    w = torch.randn(out, inn, generator=g) * 0.03
    alpha = optimal_scale(w)
    q = ternary_nearest(w, alpha)
    return DTOTernaryLinear(q, alpha, bias=torch.zeros(out), cfg=DTOConfig())


def test_init_hard_weight_is_ternary() -> None:
    layer = _ternary_layer()
    q = layer.hard_weight()
    for i in range(q.shape[0]):
        allowed = {-float(layer.alpha[i]), 0.0, float(layer.alpha[i])}
        assert set(q[i].tolist()) <= allowed


def test_soft_weight_approaches_hard_at_low_temperature() -> None:
    layer = _ternary_layer()
    layer.temperature.fill_(1e-4)
    soft = layer.soft_weight()
    hard = layer.hard_weight()
    assert torch.allclose(soft, hard, atol=1e-3)


def test_ste_forward_equals_hard() -> None:
    layer = _ternary_layer()
    x = torch.randn(2, 8)
    ste = layer(x)
    manual = torch.nn.functional.linear(x, layer.hard_weight().to(x.dtype), layer.bias)
    assert torch.allclose(ste, manual, atol=1e-6)


def test_gradients_flow_to_latent_only() -> None:
    layer = _ternary_layer()
    x = torch.randn(4, 8)
    loss = layer(x).square().mean()
    loss.backward()
    assert layer.latent.grad is not None
    assert layer.latent.grad.abs().sum() > 0
    assert layer.alpha.grad is None
    if layer.deadzone_bias is not None:
        assert layer.deadzone_bias.grad is None


def test_soft_mode_forward_uses_soft_values() -> None:
    layer = _ternary_layer()
    layer.forward_mode = "soft"
    layer.temperature.fill_(0.02)
    x = torch.randn(2, 8)
    out = layer(x)
    manual = torch.nn.functional.linear(x, layer.soft_weight().to(x.dtype), layer.bias)
    assert torch.allclose(out, manual, atol=1e-6)


def test_deadzone_steers_zero_fraction() -> None:
    g = torch.Generator().manual_seed(1)
    w = torch.randn(64, 128, generator=g) * 0.03
    alpha = optimal_scale(w)
    q = ternary_nearest(w, alpha)
    layer = DTOTernaryLinear(q, alpha, cfg=DTOConfig(deadzone=True, deadzone_lr=0.5))
    base = layer.zero_fraction()
    with torch.no_grad():
        layer.deadzone_bias.fill_(2.0)
        for _ in range(5):
            layer.update_deadzone()
    assert layer.zero_fraction() >= base


def test_deadzone_disabled() -> None:
    layer = _ternary_layer()
    layer.deadzone_bias = None
    layer.forward_mode = "ste"
    x = torch.randn(2, 8)
    assert torch.isfinite(layer(x)).all()


def test_to_linear_exports_hard_weight() -> None:
    layer = _ternary_layer()
    with torch.no_grad():
        layer.latent.add_(0.01)  # move latent away from ternary grid
    linear = layer.to_linear()
    assert torch.allclose(linear.weight, layer.hard_weight())
    x = torch.randn(3, 8)
    layer.forward_mode = "hard"
    assert torch.allclose(linear(x), layer(x), atol=1e-6)


def test_invalid_config() -> None:
    with pytest.raises(ValueError):
        DTOConfig(temperature=0.0)
    with pytest.raises(ValueError):
        DTOConfig(forward_mode="nope")
    with pytest.raises(ValueError):
        DTOConfig(temp_start=0.01, temp_end=0.1)
    with pytest.raises(ValueError):
        DTOConfig(deadzone_momentum=1.0)
