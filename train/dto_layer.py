"""MATRIX-DTO layer 2: DTO with HESTIA relaxation.

DTO (Discrete Ternary Optimization) keeps full-precision *latent* weights
:math:`W` and derives the deployed ternary weights from them.  Following
HESTIA, the discrete assignment is relaxed with a temperature-scaled
softmax over the support :math:`t = (-\\alpha, 0, +\\alpha)`,

.. math::

    p_j = \\operatorname{softmax}_j\\left(
        \\frac{w\\, t_j - t_j^2 / 2}{\\tau} \\right), \\qquad
    q_{\\text{soft}} = \\sum_j p_j\\, t_j,

which is exactly the entropic-OT barycentric projection with a single
dual fixed at :math:`t_j^2/2`.  The temperature :math:`\\tau` is annealed
during training so that :math:`q_{\\text{soft}} \\to q_{\\text{hard}}`
(hard ternary).  Gradients flow through the softmax (STE-style: hard
values in forward, soft gradients in backward):

.. math::

    q_{\\text{ste}} = q_{\\text{hard}} +
        \\bigl(q_{\\text{soft}} - q_{\\text{soft}}\\bigr)_{\\text{detach}}.

Tequila's dynamic deadzone bias adds a trainable offset :math:`b` to the
zero-target logit, steered every step to keep the fraction of zero
weights at its initial level,

.. math::

    b \\leftarrow b + \\eta\\,
        \\bigl(\\rho^{\\star} - \\rho(q_{\\text{hard}})\\bigr),

with target dead fraction :math:`\\rho^{\\star}`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from quantize.anderson_sinkhorn import ternary_nearest

__all__ = ["DTOConfig", "DTOTernaryLinear"]

_FORWARD_MODES = ("ste", "soft", "hard")


@dataclass
class DTOConfig:
    """Configuration of the DTO layer.

    Attributes:
        temperature: Current relaxation temperature ``tau > 0``.
        temp_start: Temperature at the first training step.
        temp_end: Final temperature of the annealing schedule.
        forward_mode: ``"ste"`` (hard forward, soft backward), ``"soft"``
            or ``"hard"``.
        deadzone: Enable the Tequila dynamic deadzone bias.
        deadzone_lr: Step size of the deadzone feedback.
        deadzone_momentum: EMA factor for the dead fraction error.
        target_zero_frac: Target fraction of zero weights; ``None`` keeps
            the fraction measured at initialisation.
    """

    temperature: float = 1.0
    temp_start: float = 1.0
    temp_end: float = 0.01
    forward_mode: str = "ste"
    deadzone: bool = True
    deadzone_lr: float = 5e-3
    deadzone_momentum: float = 0.9
    target_zero_frac: Optional[float] = None
    checkpoint: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0 or self.temp_start <= 0 or self.temp_end <= 0:
            raise ValueError("temperatures must be positive")
        if self.temp_end > self.temp_start:
            raise ValueError("temp_end must not exceed temp_start")
        if self.forward_mode not in _FORWARD_MODES:
            raise ValueError(f"forward_mode must be one of {_FORWARD_MODES}")
        if self.deadzone_lr < 0:
            raise ValueError("deadzone_lr must be non-negative")
        if not 0.0 <= self.deadzone_momentum < 1.0:
            raise ValueError("deadzone_momentum must lie in [0, 1)")
        if self.target_zero_frac is not None and not 0.0 <= self.target_zero_frac <= 1.0:
            raise ValueError("target_zero_frac must lie in [0, 1]")


class DTOTernaryLinear(nn.Module):
    """Ternary linear layer trained in the latent space (DTO + HESTIA).

    Args:
        weight: Initial ternary weight matrix, shape ``(out, in)``.
        alpha: Per-output scale, shape ``(out,)`` or scalar.
        bias: Optional bias vector.
        cfg: DTO configuration.
    """

    def __init__(
        self,
        weight: Tensor,
        alpha: Tensor,
        bias: Optional[Tensor] = None,
        cfg: Optional[DTOConfig] = None,
    ) -> None:
        super().__init__()
        cfg = cfg or DTOConfig()
        self.cfg = cfg
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.forward_mode = cfg.forward_mode
        self.deadzone_lr = cfg.deadzone_lr
        self.deadzone_momentum = cfg.deadzone_momentum
        self.checkpoint = cfg.checkpoint

        self.latent = nn.Parameter(weight.detach().clone().float())
        alpha = alpha.detach().reshape(-1).float()
        if alpha.numel() == 1:
            alpha = alpha.expand(self.out_features).clone()
        if alpha.numel() != self.out_features:
            raise ValueError(f"alpha must have {self.out_features} entries, got {alpha.numel()}")
        self.register_buffer("alpha", alpha)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.register_buffer("temperature", torch.tensor(float(cfg.temperature)))
        self.register_buffer("_deadzone_ema", torch.zeros(()))

        with torch.no_grad():
            q0 = self.hard_weight()
            zero_frac = float((q0 == 0).float().mean())
            target = cfg.target_zero_frac if cfg.target_zero_frac is not None else zero_frac
            self.register_buffer("target_zero_frac", torch.tensor(float(target)))
        if cfg.deadzone:
            self.deadzone_bias = nn.Parameter(torch.zeros(()))
        else:
            self.deadzone_bias = None

    def _targets(self) -> Tensor:
        """Support values ``(-alpha, 0, +alpha)`` per output row, shape (out, 3)."""
        zeros = torch.zeros_like(self.alpha)
        return torch.stack([-self.alpha, zeros, self.alpha], dim=-1)

    def soft_weight(self) -> Tensor:
        """HESTIA soft ternary assignment, shape ``(out, in)``.

        The temperature is dimensionless: it is measured relative to the
        squared scale :math:`\\alpha^2`, so ``temperature=1`` gives a soft
        relaxation while ``temperature -> 0`` recovers hard ternary.
        """
        deadzone = self.deadzone_bias.detach() if self.deadzone_bias is not None else None
        return self._soft_from(self.latent, deadzone)

    def _soft_from(self, latent: Tensor, deadzone_bias: Optional[Tensor]) -> Tensor:
        """Soft assignment as a pure function of ``latent`` (checkpointable)."""
        targets = self._targets()  # (out, 3)
        scale = self.alpha.square().clamp_min(1e-12).unsqueeze(-1)  # (out, 1)
        logits = (
            latent.unsqueeze(-1) * targets.unsqueeze(1)
            - 0.5 * targets.unsqueeze(1).square()
        ) / (self.temperature * scale).unsqueeze(1)
        if deadzone_bias is not None:
            logits = torch.cat(
                [logits[..., :1], logits[..., 1:2] + deadzone_bias, logits[..., 2:]],
                dim=-1,
            )
        probs = torch.softmax(logits, dim=-1)
        return (probs * targets.unsqueeze(1)).sum(-1)

    @torch.no_grad()
    def hard_weight(self) -> Tensor:
        """Hard ternary weight, shape ``(out, in)``."""
        return ternary_nearest(self.latent.detach(), self.alpha)

    def quantized_weight(self) -> Tensor:
        """Weight used in the forward pass (mode dependent)."""
        if self.forward_mode == "hard":
            return self.hard_weight()
        soft = self.soft_weight()
        if self.forward_mode == "soft":
            return soft
        hard = self.hard_weight()
        return hard + (soft - soft.detach())

    def forward(self, x: Tensor) -> Tensor:
        """Apply the relaxed ternary linear map (optionally weight-checkpointed)."""
        if self.forward_mode == "hard":
            weight = self.hard_weight()
        else:
            if self.training and self.checkpoint and torch.is_grad_enabled():
                deadzone = (
                    self.deadzone_bias.detach() if self.deadzone_bias is not None else None
                )
                soft = checkpoint(self._soft_from, self.latent, deadzone, use_reentrant=False)
            else:
                soft = self.soft_weight()
            if self.forward_mode == "soft":
                weight = soft
            else:
                weight = self.hard_weight() + (soft - soft.detach())
        work_dtype = torch.float32 if x.dtype != torch.float32 else x.dtype
        out = F.linear(x.to(work_dtype), weight, self.bias)
        return out.to(x.dtype)

    @torch.no_grad()
    def update_deadzone(self) -> float:
        """Tequila feedback step; returns the current zero fraction."""
        if self.deadzone_bias is None:
            return float((self.hard_weight() == 0).float().mean())
        zero_frac = (self.hard_weight() == 0).float().mean()
        err = self.target_zero_frac - zero_frac
        self._deadzone_ema.mul_(self.deadzone_momentum).add_((1 - self.deadzone_momentum) * err)
        self.deadzone_bias.add_(self.deadzone_lr * self._deadzone_ema)
        return float(zero_frac)

    @torch.no_grad()
    def zero_fraction(self) -> float:
        """Fraction of hard-zero weights."""
        return float((self.hard_weight() == 0).float().mean())

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        """Export a plain :class:`nn.Linear` with the hard ternary weight."""
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.latent.device,
            dtype=self.latent.dtype,
        )
        linear.weight.copy_(self.hard_weight())
        if self.bias is not None:
            linear.bias.copy_(self.bias)
        return linear
