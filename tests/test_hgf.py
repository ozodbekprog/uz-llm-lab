"""Tests for :mod:`quality.hgf_correction` (MATRIX-DTO layer 3)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from quality.hgf_correction import (
    HGFConfig,
    HGFLinear,
    attach_hgf,
    hgf_state_dict,
    load_hgf,
    merge_hgf,
    save_hgf,
    trainable_parameter_count,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=True)
        self.mlp = nn.ModuleDict({"down_proj": nn.Linear(16, 16, bias=False)})
        self.lm_head = nn.Linear(16, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp["down_proj"](torch.relu(self.q_proj(x)))
        return self.lm_head(x)


def _cfg(**kw) -> HGFConfig:
    base = dict(rank=4, alpha=8.0, target_modules=("q_proj", "down_proj"))
    base.update(kw)
    return HGFConfig(**base)


def test_init_equals_quantised_base() -> None:
    m = TinyModel()
    cfg = _cfg()
    hgf = HGFLinear(m.q_proj, cfg)
    x = torch.randn(3, 16)
    assert torch.allclose(hgf(x), m.q_proj(x), atol=1e-6)


def test_bias_is_preserved() -> None:
    m = TinyModel()
    hgf = HGFLinear(m.q_proj, _cfg())
    assert hgf.bias is not None
    assert torch.allclose(hgf.bias, m.q_proj.bias)


def test_attach_freezes_base() -> None:
    m = TinyModel()
    attach_hgf(m, _cfg())
    assert isinstance(m.q_proj, HGFLinear)
    assert isinstance(m.mlp["down_proj"], HGFLinear)
    assert isinstance(m.lm_head, nn.Linear)
    assert m.lm_head.weight.requires_grad is False
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all(("lora_a" in n or "lora_b" in n or "gate" in n) for n in trainable)
    q = m.q_proj
    expected = q.lora_a.numel() + q.lora_b.numel() + q.gate.numel()
    d = m.mlp["down_proj"]
    expected += d.lora_a.numel() + d.lora_b.numel() + d.gate.numel()
    assert trainable_parameter_count(m) == expected


def test_gradients_flow_only_to_hgf() -> None:
    m = TinyModel()
    attach_hgf(m, _cfg())
    x = torch.randn(2, 16)
    loss = m(x).square().mean()
    loss.backward()
    assert m.q_proj.lora_a.grad is not None
    assert m.q_proj.lora_b.grad is not None
    assert m.q_proj.gate.grad is not None
    assert m.q_proj.weight_q.grad is None
    assert m.lm_head.weight.grad is None


def test_merge_preserves_forward() -> None:
    m = TinyModel()
    attach_hgf(m, _cfg())
    with torch.no_grad():
        m.q_proj.lora_b.normal_(0, 0.02)
        m.q_proj.gate.uniform_(0.5, 1.5)
        m.mlp["down_proj"].lora_b.normal_(0, 0.02)
    x = torch.randn(5, 16)
    before = m(x)
    n = merge_hgf(m)
    after = m(x)
    assert n == 2
    assert torch.allclose(before, after, atol=1e-6)
    module = m.q_proj
    assert torch.allclose(module.lora_b, torch.zeros_like(module.lora_b))


def test_merge_restore_linear_saves_correct_keys() -> None:
    m = TinyModel()
    attach_hgf(m, _cfg())
    with torch.no_grad():
        m.q_proj.lora_b.normal_(0, 0.02)
    x = torch.randn(5, 16)
    before = m(x)
    merge_hgf(m, restore_linear=True)
    assert isinstance(m.q_proj, nn.Linear)
    assert isinstance(m.mlp["down_proj"], nn.Linear)
    assert "q_proj.weight" in m.state_dict()
    assert not any("weight_q" in k for k in m.state_dict())
    assert torch.allclose(m(x), before, atol=1e-6)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    import copy

    cfg = _cfg()
    m1 = TinyModel()
    attach_hgf(m1, cfg)
    m2 = copy.deepcopy(m1)
    with torch.no_grad():
        m1.q_proj.lora_b.normal_(0, 0.05)
        m1.q_proj.gate.uniform_(0.5, 1.5)
        m1.mlp["down_proj"].lora_b.normal_(0, 0.05)
    path = tmp_path / "hgf.pt"
    save_hgf(m1, path, base_model="tiny", cfg=cfg)
    load_hgf(m2, path)
    x = torch.randn(4, 16)
    assert torch.allclose(m1(x), m2(x), atol=1e-6)
    sd1, sd2 = hgf_state_dict(m1), hgf_state_dict(m2)
    assert set(sd1) == set(sd2)


def test_state_dict_is_detached() -> None:
    m = TinyModel()
    attach_hgf(m, _cfg())
    sd = hgf_state_dict(m)
    assert not sd["q_proj"]["lora_a"].requires_grad


def test_attach_without_match_raises() -> None:
    with pytest.raises(ValueError):
        attach_hgf(TinyModel(), _cfg(target_modules=("nonexistent",)))


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        HGFConfig(rank=0)
    with pytest.raises(ValueError):
        HGFConfig(alpha=0)
    with pytest.raises(ValueError):
        HGFConfig(dropout=1.0)
