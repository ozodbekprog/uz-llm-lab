#!/usr/bin/env python
"""MATRIX-DTO, layer 1 CLI: ternary quantisation via Anderson-Sinkhorn OT.

Quantises all ``nn.Linear`` weights of a HF causal LM with the OT
quantiser from :mod:`quantize.anderson_sinkhorn`, optionally compares
against legacy absmean baselines, measures the LM loss on Uzbek
instruction data and micro-benchmarks Anderson acceleration against the
plain Sinkhorn iteration on real weight rows.

Examples:
    .venv/bin/python scripts/quantize_ot.py --config configs/matrix_dto.yaml
    .venv/bin/python scripts/quantize_ot.py --method absmean --granularity tensor --tag legacy
    .venv/bin/python scripts/quantize_ot.py --granularity group --group-size 128
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.anderson_sinkhorn import (
    TernaryOTConfig,
    _scale_free_cost,
    _target_masses,
    optimal_scale,
    quantize_matrix,
)
from quantize.repro import resolve_device, set_seed
from quantize.sinkhorn import SinkhornConfig, sinkhorn_log

_DTYPES = {"float32": torch.float32, "float64": torch.float64}
_METHOD_LABELS = {"ot": "OT(Anderson-Sinkhorn)", "round": "optimal-round", "absmean": "absmean"}


def load_items(path: Path, limit: int) -> List[dict]:
    """Load up to ``limit`` JSONL items."""
    items: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if len(items) >= limit:
                break
    return items


def build_text(tokenizer, item: dict) -> str:
    """Render a chat item into a single prompt string."""
    messages = item.get("messages")
    if messages:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return item.get("text", "")


@torch.no_grad()
def lm_loss(model, tokenizer, items: Iterable[dict], max_len: int) -> Tuple[float, int]:
    """Token-weighted mean NLL over ``items``.

    Args:
        model: Causal LM in eval mode, already on the target device.
        tokenizer: Matching tokenizer.
        items: Chat items with a ``messages`` field.
        max_len: Truncation length.

    Returns:
        Tuple ``(mean_nll, num_tokens)``; ``(nan, 0)`` when no tokens.
    """
    total, tokens = 0.0, 0
    for item in items:
        text = build_text(tokenizer, item)
        batch = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_len
        ).to(model.device)
        if batch["input_ids"].shape[-1] < 2:
            continue
        out = model(**batch, labels=batch["input_ids"])
        n = int(batch["input_ids"].numel())
        total += float(out.loss) * n
        tokens += n
    return (total / tokens if tokens else float("nan")), tokens


def linear_modules(model, include_lm_head: bool, limit: int = 0) -> List[Tuple[str, torch.nn.Linear]]:
    """Collect quantisable ``nn.Linear`` layers in model order."""
    out: List[Tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name == "lm_head" and not include_lm_head:
            continue
        out.append((name, module))
        if limit and len(out) >= limit:
            break
    return out


def quantize_model(
    model, cfg: TernaryOTConfig, include_lm_head: bool, limit: int = 0
) -> Tuple[List[dict], float]:
    """Quantise every linear layer in place and return per-layer stats."""
    layers = linear_modules(model, include_lm_head, limit)
    records: List[dict] = []
    t0 = time.perf_counter()
    for i, (name, module) in enumerate(layers, 1):
        q, stats = quantize_matrix(module.weight.data, cfg, name=name)
        module.weight.data.copy_(q)
        rec = asdict(stats)
        rec["shape"] = list(module.weight.shape)
        records.append(rec)
        if i % 20 == 0 or i == len(layers):
            logger.info(f"[{i}/{len(layers)}] {name} rel_err={stats.rel_err:.4f}")
    return records, time.perf_counter() - t0


def sinkhorn_microbench(
    weight: torch.Tensor,
    n_rows: int = 128,
    scenarios: Optional[List[dict]] = None,
) -> List[dict]:
    """Compare plain vs Anderson-accelerated Sinkhorn on real weight rows.

    Two scenarios by default: the production setting of the quantiser and a
    deliberately hard problem (small ``eps``, tight tolerance, float64).
    """
    scenarios = scenarios or [
        {"name": "quant eps=0.05 tol=1e-6 f32", "eps": 0.05, "tol": 1e-6, "max_iter": 1000, "dtype": torch.float32},
        {"name": "hard  eps=0.01 tol=1e-10 f64", "eps": 0.01, "tol": 1e-10, "max_iter": 4000, "dtype": torch.float64},
    ]
    rows: List[dict] = []
    for sc in scenarios:
        blocks = weight.reshape(-1, weight.shape[-1])[:n_rows].to(sc["dtype"])
        alpha = optimal_scale(blocks)
        support = torch.stack([-alpha, torch.zeros_like(alpha), alpha], dim=-1)
        masses = _target_masses(blocks, alpha, 1e-9)
        cost = _scale_free_cost(blocks, support, alpha)
        source = torch.full_like(blocks, 1.0 / blocks.shape[-1])
        for label, m in (("plain", 0), ("anderson", 5)):
            t0 = time.perf_counter()
            res = sinkhorn_log(
                source,
                masses,
                cost,
                SinkhornConfig(eps=sc["eps"], max_iter=sc["max_iter"], tol=sc["tol"], anderson_m=m),
            )
            rows.append(
                {
                    "scenario": sc["name"],
                    "variant": label,
                    "iters": res.iters,
                    "time_s": round(time.perf_counter() - t0, 4),
                    "marginal_err": res.marginal_err,
                    "converged": res.converged,
                }
            )
    return rows


def module_type(name: str) -> str:
    """Map a layer name to its projection type."""
    return name.split(".")[-1]


def print_table(header: List[str], rows: List[List[str]]) -> None:
    """Print a simple aligned table."""
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MATRIX-DTO layer 1 quantisation CLI")
    ap.add_argument("--config", default="configs/matrix_dto.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="ot")
    ap.add_argument("--method", choices=sorted(_METHOD_LABELS), default=None)
    ap.add_argument("--granularity", choices=["tensor", "row", "group"], default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--anderson-m", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto | cpu | cuda")
    ap.add_argument("--dtype", choices=["float32", "float64"], default=None)
    ap.add_argument("--include-lm-head", action="store_true")
    ap.add_argument("--limit-layers", type=int, default=0)
    ap.add_argument("--eval-samples", type=int, default=None, help="0 disables loss evaluation")
    ap.add_argument("--eval-max-len", type=int, default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--baselines", default="tensor,row", help="absmean granularities, 'none' to skip")
    ap.add_argument("--results", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save", action="store_true", help="save the quantised model to --out")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    ycfg: dict = {}
    cfg_path = ROOT / args.config
    if cfg_path.exists():
        ycfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    qy = ycfg.get("quantize", {})
    my = ycfg.get("model", {})
    ey = ycfg.get("eval", {})
    oy = ycfg.get("output", {})

    def pick(cli, d: dict, key: str, default):
        if cli is not None:
            return cli
        return d.get(key, default)

    seed = pick(args.seed, ycfg, "seed", 42)
    set_seed(seed)
    model_name = pick(args.model, my, "name", "models/Qwen2.5-0.5B-Instruct")
    out_dir = pick(args.out, oy, "model_dir", "models/qwen05b-ot-ternary")
    results_path = pick(args.results, oy, "results", f"logs/results_ot_quant_{args.tag}.json")
    eval_samples = pick(args.eval_samples, ey, "samples", 24)
    eval_max_len = pick(args.eval_max_len, ey, "max_len", 192)
    data_path = ROOT / pick(args.data, ey, "data", "data/uzbek-instruct.jsonl")
    dtype_key = pick(args.dtype, my, "dtype", "float32")

    cfg = TernaryOTConfig(
        method=pick(args.method, qy, "method", "ot"),
        granularity=pick(args.granularity, qy, "granularity", "row"),
        group_size=pick(args.group_size, qy, "group_size", 128),
        eps=pick(args.eps, qy, "eps", 0.05),
        max_iter=pick(args.max_iter, qy, "max_iter", 200),
        tol=pick(args.tol, qy, "tol", 1e-6),
        anderson_m=pick(args.anderson_m, qy, "anderson_m", 5),
        alpha_coarse=pick(None, qy, "alpha_coarse", 33),
        alpha_refine=pick(None, qy, "alpha_refine", 16),
        dtype=_DTYPES[dtype_key],
        device=pick(args.device, my, "device", "auto"),
    )
    include_lm_head = args.include_lm_head or bool(my.get("include_lm_head", False))

    log_path = ROOT / "logs" / f"quantize_{args.tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, level="INFO")
    logger.info(f"cfg={cfg}")
    logger.info(f"model={model_name} out={out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model loaded: {n_params/1e6:.1f}M params")

    eval_device = resolve_device(cfg.device)
    items = load_items(data_path, eval_samples) if eval_samples else []
    losses: Dict[str, float] = {}
    if items:
        model.to(eval_device)
        t0 = time.perf_counter()
        losses["fp32"] = lm_loss(model, tokenizer, items, eval_max_len)[0]
        logger.info(f"fp32 loss={losses['fp32']:.4f} ({time.perf_counter()-t0:.1f}s)")
        model.to("cpu")
        torch.cuda.empty_cache() if eval_device.type == "cuda" else None

    pristine = copy.deepcopy(model) if args.baselines != "none" else None
    bench_weight = next(
        m.weight.data.clone() for n, m in linear_modules(model, False) if n.endswith("q_proj")
    )
    records, total_time = quantize_model(model, cfg, include_lm_head, args.limit_layers)
    logger.info(
        f"quantised {len(records)} layers in {total_time:.1f}s "
        f"(mean rel_err={sum(r['rel_err'] for r in records)/len(records):.4f})"
    )

    if items:
        model.to(eval_device)
        losses[cfg.method] = lm_loss(model, tokenizer, items, eval_max_len)[0]
        logger.info(f"{cfg.method} loss={losses[cfg.method]:.4f}")
        model.to("cpu")
        torch.cuda.empty_cache() if eval_device.type == "cuda" else None

    baseline_records: Dict[str, List[dict]] = {}
    baseline_losses: Dict[str, float] = {}
    if pristine is not None:
        for gran in [g.strip() for g in args.baselines.split(",") if g.strip()]:
            bcfg = TernaryOTConfig(method="absmean", granularity=gran, device=cfg.device)
            copy_model = copy.deepcopy(pristine)
            recs, dt = quantize_model(copy_model, bcfg, include_lm_head, args.limit_layers)
            baseline_records[f"absmean-{gran}"] = recs
            logger.info(
                f"baseline absmean-{gran}: mean rel_err="
                f"{sum(r['rel_err'] for r in recs)/len(recs):.4f} ({dt:.1f}s)"
            )
            if items:
                copy_model.to(eval_device)
                baseline_losses[f"absmean-{gran}"] = lm_loss(
                    copy_model, tokenizer, items, eval_max_len
                )[0]
                logger.info(f"absmean-{gran} loss={baseline_losses[f'absmean-{gran}']:.4f}")
            del copy_model
            torch.cuda.empty_cache() if eval_device.type == "cuda" else None

    micro = sinkhorn_microbench(bench_weight)

    if args.save:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        logger.info(f"model saved: {out_dir}")

    methods = [cfg.method] + [m for m in baseline_records]
    per_type: Dict[str, Dict[str, float]] = {}
    for m in methods:
        recs = records if m == cfg.method else baseline_records[m]
        agg: Dict[str, List[float]] = {}
        for r in recs:
            agg.setdefault(module_type(r["name"]), []).append(r["rel_err"])
        per_type[m] = {k: sum(v) / len(v) for k, v in sorted(agg.items())}

    report = {
        "config": {**asdict(cfg), "dtype": dtype_key, "seed": seed},
        "model": model_name,
        "params_m": n_params / 1e6,
        "layers": [r["name"] for r in records],
        "loss": {**losses, **baseline_losses},
        "microbench": micro,
        "aggregate": {},
        "per_module_type": per_type,
        "per_layer": [
            {
                "name": r["name"],
                "rel_err": r["rel_err"],
                "mse": r["mse"],
                "alpha_mean": r["alpha_mean"],
                "shape": r["shape"],
                "sinkhorn_iters": r["sinkhorn_iters"],
                "time_s": r["time_s"],
            }
            for r in records
        ],
    }
    for m in methods:
        recs = records if m == cfg.method else baseline_records[m]
        report["aggregate"][m] = {
            "layers": len(recs),
            "mean_rel_err": sum(r["rel_err"] for r in recs) / len(recs),
            "mean_mse": sum(r["mse"] for r in recs) / len(recs),
            "total_time_s": sum(r["time_s"] for r in recs),
        }
        if m != cfg.method:
            report["aggregate"][m]["loss"] = baseline_losses.get(m)
    if items:
        report["eval"] = {"samples": len(items), "max_len": eval_max_len, "data": str(data_path)}
    out_path = ROOT / results_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print(f"MATRIX-DTO | layer 1 natijalari | model: {model_name} | tag: {args.tag}")
    print(f"granularity={cfg.granularity} eps={cfg.eps} anderson_m={cfg.anderson_m} "
          f"method={cfg.method} | {len(records)} qatlam, {n_params/1e6:.1f}M param")
    print("=" * 78)
    print("\n[1] Kvantlash sifati (rel_err = ||W-Q||_F/||W||_F, o'rtacha)")
    print_table(
        ["metod", "qatlam", "rel_err", "MSE", "vaqt,s"],
        [
            [
                m,
                str(report["aggregate"][m]["layers"]),
                f"{report['aggregate'][m]['mean_rel_err']:.4f}",
                f"{report['aggregate'][m]['mean_mse']:.3e}",
                f"{report['aggregate'][m]['total_time_s']:.1f}",
            ]
            for m in methods
        ],
    )
    if "sinkhorn_iters" in records[0]:
        iters = [r["sinkhorn_iters"] for r in records if r["sinkhorn_iters"]]
        if iters:
            print(f"\nSinkhorn iteratsiyalari (OT qatlamlari): "
                  f"min={min(iters)} mean={sum(iters)/len(iters):.1f} max={max(iters)}")
    print("\n[2] Loss (o'zbekcha instruct, n={})".format(len(items)))
    if report["loss"]:
        print_table(["model", "NLL"], [[k, f"{v:.4f}"] for k, v in report["loss"].items()])
    print("\n[3] Anderson mikrobenchmark (q_proj, birinchi 128 qator)")
    print_table(
        ["ssenariy", "variant", "iter", "vaqt,s", "marginal_err", "converged"],
        [
            [
                r["scenario"],
                r["variant"],
                str(r["iters"]),
                f"{r['time_s']:.3f}",
                f"{r['marginal_err']:.2e}",
                str(r["converged"]),
            ]
            for r in micro
        ],
    )
    speed_lines = []
    for sc in sorted({r["scenario"] for r in micro}):
        p = next(r for r in micro if r["scenario"] == sc and r["variant"] == "plain")
        a = next(r for r in micro if r["scenario"] == sc and r["variant"] == "anderson")
        sp = p["iters"] / a["iters"] if a["iters"] else float("nan")
        speed_lines.append(f"{sc}: {sp:.1f}x")
    print("Anderson iteratsiya tejamkorligi -> " + "; ".join(speed_lines))
    print("\n[4] Modul turlari bo'yicha rel_err")
    types = sorted({k for v in per_type.values() for k in v})
    print_table(
        ["modul"] + methods,
        [[t] + [f"{per_type[m].get(t, float('nan')):.4f}" for m in methods] for t in types],
    )
    print(f"\nJSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
