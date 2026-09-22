"""Reproducibility helpers: seeding and device resolution."""
from __future__ import annotations

import random
from typing import Union

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs used by the pipeline.

    Args:
        seed: Non-negative seed value.
        deterministic: If True also selects deterministic cuDNN kernels
            (slower, but bit-reproducible on the same hardware).
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, otherwise CPU.

    Args:
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or a full device string.

    Returns:
        A :class:`torch.device` instance that is usable on this machine.
    """
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(device)
