"""MATRIX-DTO layer 3: HGF (Hybrid Gated Flow) correction.

The ternary base weight :math:`W_t` stays frozen; a trainable low-rank
correction with a per-output gate restores quality:

.. math::

    y = W_t x + \\operatorname{diag}(g)\\, \\frac{\\alpha}{r} B A x,
    \\qquad B \\in \\mathbb{R}^{m \\times r}, \\; A \\in \\mathbb{R}^{r \\times n},

where :math:`g \\in \\mathbb{R}^{m}` is the gate (initialised to ones) and
:math:`\\alpha / r` the LoRA-style scaling.  ``B`` is initialised to zero,
so at step 0 the layer is exactly the quantised one (y = W_t x); training
then learns how far to "flow" away from the ternary constraint.  The
correction can be merged back,

.. math::

    W_{\\text{eff}} = W_t + \\frac{\\alpha}{r}\\,
        \\bigl(\\operatorname{diag}(g) B\\bigr) A,

which keeps inference cost identical to the ternary base.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "HGFConfig",
    "HGFLinear",
    "attach_hgf",
    "hgf_parameters",
    "hgf_state_dict",
    "load_hgf",
    "merge_hgf",
    "save_hgf",
    "trainable_parameter_count",
]

_DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class HGFConfig:
    """Configuration of the HGF correction.

    Attributes:
        rank: Low-rank dimension ``r``.
        alpha: LoRA-style scaling numerator; effective scale is ``alpha / r``.
        gate_init: Initial value of the per-output gate.
        dropout: Dropout applied to the low-rank branch.
        target_modules: Module name suffixes that receive an HGF branch.
        freeze_base: Whether the (already ternary) base weights stay frozen.
    """

    rank: int = 16
    alpha: float = 32.0
    gate_init: float = 1.0
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = _DEFAULT_TARGETS
    freeze_base: bool = True

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class HGFLinear(nn.Module):
    """Frozen ternary/quantised linear layer plus a gated low-rank branch.

    Args:
        base: Source linear layer; its weight becomes the frozen base.
        cfg: HGF configuration.
    """

    def __init__(self, base: nn.Linear, cfg: HGFConfig) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = cfg.rank
        self.scaling = cfg.alpha / cfg.rank
        self.dropout_p = cfg.dropout

        weight = base.weight.detach().clone()
        self.register_buffer("weight_q", weight, persistent=True)
        if base.bias is not None:
            self.register_buffer("bias", base.bias.detach().clone(), persistent=True)
        else:
            self.bias = None

        a = torch.empty(cfg.rank, self.in_features)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_a = nn.Parameter(a.to(weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, cfg.rank, dtype=weight.dtype))
        self.gate = nn.Parameter(torch.full((self.out_features,), cfg.gate_init, dtype=weight.dtype))

        if cfg.freeze_base:
            self.weight_q.requires_grad_(False)
            if self.bias is not None:
                self.bias.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``W_q x + diag(gate) * scaling * B A x``."""
        base = F.linear(x, self.weight_q, self.bias)
        branch = F.linear(x, self.lora_a.to(x.dtype))
        if self.dropout_p > 0.0 and self.training:
            branch = F.dropout(branch, p=self.dropout_p)
        branch = F.linear(branch, self.lora_b.to(x.dtype))
        return base + self.gate.to(x.dtype) * (self.scaling * branch)

    def merged_weight(self) -> Tensor:
        """Return ``W_eff`` including the folded correction."""
        delta = (self.lora_b * self.gate.unsqueeze(1)) @ self.lora_a
        return (self.weight_q + self.scaling * delta).detach()

    @torch.no_grad()
    def merge_(self) -> None:
        """Fold the branch into the frozen weight and reset the branch."""
        self.weight_q.copy_(self.merged_weight())
        self.lora_b.zero_()
        self.gate.fill_(1.0)

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        """Return a plain :class:`nn.Linear` holding the merged weight.

        The result is directly compatible with ``save_pretrained`` (unlike
        the wrapper, whose frozen weight lives in the ``weight_q`` buffer).
        """
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight_q.device,
            dtype=self.weight_q.dtype,
        )
        linear.weight.copy_(self.merged_weight())
        if self.bias is not None:
            linear.bias.copy_(self.bias)
        return linear


def _iter_targets(model: nn.Module, targets: Iterable[str]) -> List[Tuple[str, nn.Module, str]]:
    """Yield ``(full_name, parent, child)`` for Linear layers matching targets."""
    suffixes = tuple(targets)
    found: List[Tuple[str, nn.Module, str]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.endswith(suffixes):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        found.append((name, parent, child))
    return found


def attach_hgf(model: nn.Module, cfg: HGFConfig) -> List[str]:
    """Replace matching Linear layers with :class:`HGFLinear`.

    Args:
        model: Causal LM (typically already ternary-quantised).
        cfg: HGF configuration.

    Returns:
        Names of the replaced modules.
    """
    replaced: List[str] = []
    for name, parent, child in _iter_targets(model, cfg.target_modules):
        if name.endswith("lm_head"):
            continue
        base = getattr(parent, child)
        setattr(parent, child, HGFLinear(base, cfg))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"no modules matched targets={cfg.target_modules}")
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, HGFLinear):
            for p in module.parameters():
                p.requires_grad_(True)
    return replaced


def hgf_parameters(model: nn.Module) -> List[nn.Parameter]:
    """All trainable HGF parameters."""
    return [p for m in model.modules() if isinstance(m, HGFLinear) for p in m.parameters() if p.requires_grad]


def trainable_parameter_count(model: nn.Module) -> int:
    """Number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def hgf_state_dict(model: nn.Module) -> Dict[str, dict]:
    """Extract a portable state dict of all HGF branches."""
    out: Dict[str, dict] = {}
    for name, module in model.named_modules():
        if isinstance(module, HGFLinear):
            out[name] = {
                "lora_a": module.lora_a.detach().cpu(),
                "lora_b": module.lora_b.detach().cpu(),
                "gate": module.gate.detach().cpu(),
                "scaling": module.scaling,
                "rank": module.rank,
            }
    return out


def save_hgf(model: nn.Module, path: Path, base_model: str, cfg: HGFConfig) -> None:
    """Save HGF branches plus metadata to ``path``."""
    payload = {"base_model": base_model, "config": asdict(cfg), "modules": hgf_state_dict(model)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_hgf(model: nn.Module, path: Path) -> List[str]:
    """Load HGF branches into a model that already has them attached.

    Args:
        model: Model with :class:`HGFLinear` layers attached.
        path: File written by :func:`save_hgf`.

    Returns:
        Names of the modules that were restored.

    Raises:
        KeyError: If a saved module is missing from the model.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    loaded: List[str] = []
    modules = dict(model.named_modules())
    for name, sd in payload["modules"].items():
        if name not in modules or not isinstance(modules[name], HGFLinear):
            raise KeyError(f"HGF module not found in model: {name}")
        target = modules[name]
        with torch.no_grad():
            target.lora_a.copy_(sd["lora_a"])
            target.lora_b.copy_(sd["lora_b"])
            target.gate.copy_(sd["gate"])
        loaded.append(name)
    return loaded


@torch.no_grad()
def merge_hgf(model: nn.Module, restore_linear: bool = False) -> int:
    """Fold every HGF branch into its frozen weight (inference form).

    Args:
        model: Model with :class:`HGFLinear` layers attached.
        restore_linear: When True, replace every :class:`HGFLinear` with a
            plain :class:`nn.Linear` (required before ``save_pretrained``).

    Returns:
        Number of merged modules.
    """
    hits: List[Tuple[str, nn.Module, str]] = []
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, HGFLinear):
            continue
        module.merge_()
        count += 1
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        hits.append((name, parent, child))
    if restore_linear:
        for _, parent, child in hits:
            setattr(parent, child, getattr(parent, child).to_linear())
    return count
