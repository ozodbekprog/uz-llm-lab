#!/usr/bin/env python
"""MATRIX-DTO layer 2 runner: DTO/HESTIA quantisation-aware training (QAT).

Unlike the HGF hybrid (which only bolts a correction onto a fixed ternary
base), this trainer makes the *ternary weights themselves* learnable in
the last ``K`` transformer layers while the earlier layers stay frozen
ternary.  Every selected ``nn.Linear`` becomes a
:class:`train.dto_layer.DTOTernaryLinear`:

* latent fp32 weights initialised from the OT quantiser,
* HESTIA softmax relaxation with a cosine-annealed temperature,
* straight-through hard ternary in the forward pass,
* Tequila dynamic deadzone bias keeping the zero fraction at its
  initial level.

Memory fits the 4 GB GTX 1650 because gradients/optimiser state exist
only for the last ``K`` layers.

Usage:
    .venv/bin/python train/dto_qat.py --layers 6 --steps 800 --merge-out models/qwen05b-dtoqat
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "train"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import torch
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dto_trainer import CausalCollator, build_dataset, eval_nll, load_messages
from dto_layer import DTOConfig, DTOTernaryLinear

from quantize.anderson_sinkhorn import optimal_scale, ternary_nearest
from quantize.repro import set_seed


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MATRIX-DTO DTO/HESTIA QAT trainer")
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct", help="fp32 source model")
    ap.add_argument("--out", default="models/qwen05b-dtoqat")
    ap.add_argument("--data", default="data/uzbek-instruct.jsonl")
    ap.add_argument("--layers", type=int, default=6, help="last K transformer layers to DTO-train")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--max-samples", type=int, default=6000)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optim", choices=["adamw", "adamw8bit", "sgd"], default="adamw8bit")
    ap.add_argument("--base-dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--temp-start", type=float, default=1.0)
    ap.add_argument("--temp-end", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-samples", type=int, default=24)
    ap.add_argument("--eval-max-len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge-out", default="", help="optional export path (defaults to --out)")
    return ap.parse_args(argv)


def _is_in_last_layers(name: str, n_layers: int, k: int) -> bool:
    """Whether a module name belongs to one of the last ``k`` blocks."""
    if not name.startswith("model.layers."):
        return False
    try:
        idx = int(name.split(".")[2])
    except (IndexError, ValueError):
        return False
    return idx >= n_layers - k


@torch.no_grad()
def convert_to_dto(
    model: torch.nn.Module,
    k_layers: int,
    cfg: DTOConfig,
    base_dtype: str = "float16",
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Quantise every Linear and turn the last-K ones into DTO layers.

    Args:
        model: fp32 causal LM, modified in place.
        k_layers: Number of last transformer blocks to keep trainable.
        cfg: DTO configuration.
        base_dtype: Dtype of the frozen base (``"float16"`` halves VRAM at
            the cost of slower fp16 matmul on GTX-class GPUs).

    Returns:
        Tuple ``(originals, dto_names)`` where ``originals`` maps layer
        names to their fp32 source weights (for error reporting) and
        ``dto_names`` lists the DTO-converted layers.
    """
    n_layers = model.config.num_hidden_layers
    originals: Dict[str, torch.Tensor] = {}
    dto_names: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight.data
        blocks = weight.reshape(-1, weight.shape[-1]).float()
        alpha = optimal_scale(blocks)
        q = ternary_nearest(blocks, alpha).reshape(weight.shape)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if name.endswith("lm_head") or not _is_in_last_layers(name, n_layers, k_layers):
            module.weight.data.copy_(q.to(weight.dtype))
            module.weight.requires_grad_(False)
            continue
        originals[name] = weight.detach().clone().float()
        dto = DTOTernaryLinear(
            q,
            alpha,
            bias=module.bias.data if module.bias is not None else None,
            cfg=cfg,
        )
        setattr(parent, child, dto)
        dto_names.append(name)
    if base_dtype == "float16":
        for module in model.modules():
            is_norm = "RMSNorm" in type(module).__name__ or isinstance(
                module, (torch.nn.LayerNorm,)
            )
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)) or is_norm:
                module.to(torch.float16)
    return originals, dto_names


@torch.no_grad()
def export_model(model: torch.nn.Module, out_dir: Path, tokenizer) -> int:
    """Replace DTO layers with hard ternary Linear layers and save."""
    model.to(torch.float32)
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, DTOTernaryLinear):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, module.to_linear())
        count += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return count


def lr_at(step: int, steps: int, base_lr: float, warmup: int = 20) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, steps - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    logger.add(ROOT / "logs" / "dto_qat.log", level="INFO")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.config.use_cache = False

    cfg = DTOConfig(temp_start=args.temp_start, temp_end=args.temp_end)
    originals, dto_names = convert_to_dto(model, args.layers, cfg, base_dtype=args.base_dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"DTO layers: {len(dto_names)} | trainable: {trainable/1e6:.1f}M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = load_messages(ROOT / args.data, 0)
    train_rows = all_rows[: args.max_samples] if args.max_samples else all_rows
    eval_rows = all_rows[-args.eval_samples :] if args.eval_samples else []
    eval_rows = [r for r in eval_rows if id(r) not in {id(t) for t in train_rows}]
    loss_before = eval_nll(model, tokenizer, eval_rows, args.eval_max_len, device) if eval_rows else float("nan")
    logger.info(f"loss before QAT (held-out n={len(eval_rows)}): {loss_before:.4f}")

    ds = build_dataset(train_rows, tokenizer, args.max_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=CausalCollator(tokenizer.pad_token_id),
        num_workers=2,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info(f"train samples: {len(ds)} | steps: {args.steps} | accum: {args.grad_accum}")

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(params, lr=args.lr)
            logger.info("using bitsandbytes AdamW8bit")
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning(f"AdamW8bit unavailable ({exc}); falling back to AdamW")
            optimizer = torch.optim.AdamW(params, lr=args.lr)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    dto_modules = [m for m in model.modules() if isinstance(m, DTOTernaryLinear)]

    data_iter = iter(loader)
    running, t0 = 0.0, time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        (loss / args.grad_accum).backward()
        running += float(loss)

        temp = args.temp_start * (args.temp_end / args.temp_start) ** (step / args.steps)
        for module in dto_modules:
            module.temperature.fill_(temp)
            module.update_deadzone()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, args.steps, args.lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == 1:
            zeros = sum(m.zero_fraction() for m in dto_modules) / len(dto_modules)
            mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            logger.info(
                f"step {step}/{args.steps} loss={running/10 if step % 10 == 0 else float(loss):.4f} "
                f"temp={temp:.4f} zero={zeros:.3f} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"vram={mem:.2f}GB"
            )
            if step % 10 == 0:
                running = 0.0

    loss_after = eval_nll(model, tokenizer, eval_rows, args.eval_max_len, device) if eval_rows else float("nan")
    logger.info(f"loss after QAT: {loss_after:.4f} (before: {loss_before:.4f})")

    model.to("cpu")
    rel_errors: Dict[str, float] = {}
    for name, module in model.named_modules():
        if isinstance(module, DTOTernaryLinear) and name in originals:
            ref = originals[name]
            q = module.hard_weight()
            rel_errors[name] = float((ref - q).norm() / ref.norm().clamp_min(1e-30))
    mean_rel = sum(rel_errors.values()) / max(len(rel_errors), 1)
    logger.info(f"DTO layers mean rel_err vs fp32: {mean_rel:.4f}")

    out_dir = ROOT / (args.merge_out or args.out)
    exported = export_model(model, out_dir, tokenizer)
    summary = {
        "source_model": args.model,
        "out": str(out_dir),
        "dto_layers": len(dto_names),
        "trainable_params": trainable,
        "steps": args.steps,
        "layers_k": args.layers,
        "optim": args.optim,
        "lr": args.lr,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "mean_rel_err_dto": mean_rel,
        "exported_modules": exported,
        "duration_s": time.perf_counter() - t0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qat_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"exported to {out_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
