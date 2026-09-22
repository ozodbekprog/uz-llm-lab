#!/usr/bin/env python
"""MATRIX-DTO layer 2+3 hybrid trainer (DTO base + HGF correction).

Practical variant of the DTO/HESTIA plan for a 4 GB GTX 1650: the
ternary base stays frozen and only the HGF branches (low-rank + per-output
gate, see :mod:`quality.hgf_correction`) are optimised against the causal
LM loss on Uzbek instruction data.  The gated low-rank flow plays the role
of HESTIA's soft relaxation: instead of relaxing the ternary constraint
itself, it learns a continuous correction whose magnitude is controlled
per output neuron.

Usage:
    .venv/bin/python train/dto_trainer.py --model models/qwen05b-ot-ternary \
        --out models/hgf-qwen05b-ot --steps 500 --eval-samples 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from quality.hgf_correction import (
    HGFConfig,
    attach_hgf,
    merge_hgf,
    save_hgf,
    trainable_parameter_count,
)


class CausalCollator:
    """Right-pads batches and masks the padding in the labels."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(list(f["labels"]) + [-100] * pad)
            attention.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def load_messages(path: Path, max_samples: int) -> List[dict]:
    """Read JSONL chat items with a ``messages`` field."""
    rows: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not msgs:
                continue
            rows.append({"messages": msgs})
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def build_dataset(rows: List[dict], tokenizer, max_len: int) -> Dataset:
    """Tokenise chat items into a causal-LM dataset."""

    def to_texts(batch):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]}

    ds = Dataset.from_list(rows).map(to_texts, batched=True, remove_columns=["messages"])

    def tokenize(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)
        enc["labels"] = [ids[:] for ids in enc["input_ids"]]
        return enc

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds.filter(lambda x: len(x["input_ids"]) > 16)


@torch.no_grad()
def eval_nll(model, tokenizer, rows: List[dict], max_len: int, device: torch.device) -> float:
    """Token-weighted NLL over chat items."""
    model.to(device).eval()
    total, tokens = 0.0, 0
    for item in rows:
        text = tokenizer.apply_chat_template(item["messages"], tokenize=False)
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        if batch["input_ids"].shape[-1] < 2:
            continue
        out = model(**batch, labels=batch["input_ids"])
        n = int(batch["input_ids"].numel())
        total += float(out.loss) * n
        tokens += n
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return total / max(tokens, 1)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MATRIX-DTO DTO/HGF hybrid trainer")
    ap.add_argument("--model", default="models/qwen05b-ot-ternary")
    ap.add_argument("--data", default="data/uzbek-instruct.jsonl")
    ap.add_argument("--out", default="models/hgf-qwen05b-ot")
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=0, help="max_steps override (smoke runs)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--eval-samples", type=int, default=16)
    ap.add_argument("--eval-max-len", type=int, default=192)
    ap.add_argument("--merge-out", default="", help="optionally save the merged fp32 model here")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    logger.add(ROOT / "logs" / "dto_hgf_trainer.log", level="INFO")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.config.use_cache = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = load_messages(ROOT / args.data, 0)
    train_rows = all_rows[: args.max_samples] if args.max_samples else all_rows
    eval_rows = all_rows[-args.eval_samples :] if args.eval_samples else []
    train_ids = {id(r) for r in train_rows}
    eval_rows = [r for r in eval_rows if id(r) not in train_ids]
    loss_before = eval_nll(model, tokenizer, eval_rows, args.eval_max_len, device) if eval_rows else float("nan")
    logger.info(f"loss before HGF: {loss_before:.4f} (held-out n={len(eval_rows)})")

    cfg = HGFConfig(
        rank=args.rank,
        alpha=args.alpha,
        target_modules=tuple(args.target_modules.split(",")),
    )
    replaced = attach_hgf(model, cfg)
    logger.info(f"HGF attached to {len(replaced)} layers, trainable params: {trainable_parameter_count(model)}")

    ds = build_dataset(train_rows, tokenizer, args.max_len)
    logger.info(f"train samples: {len(ds)}")

    training_args = TrainingArguments(
        output_dir=args.out + "-checkpoints",
        num_train_epochs=args.epochs,
        max_steps=args.steps if args.steps > 0 else -1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=args.grad_checkpoint,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        dataloader_num_workers=2,
        use_cpu=device.type == "cpu",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    trainer.train()

    loss_after = eval_nll(model, tokenizer, eval_rows, args.eval_max_len, device) if eval_rows else float("nan")
    logger.info(f"loss after HGF: {loss_after:.4f} (before: {loss_before:.4f})")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    save_hgf(model, out_dir / "hgf_weights.pt", base_model=args.model, cfg=cfg)
    logger.info(f"HGF weights saved: {out_dir / 'hgf_weights.pt'}")

    if args.merge_out:
        merged = merge_hgf(model, restore_linear=True)
        merged_dir = ROOT / args.merge_out
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"{merged} branches merged and saved: {merged_dir}")

    summary = {
        "base_model": args.model,
        "out": str(out_dir),
        "rank": args.rank,
        "alpha": args.alpha,
        "steps": args.steps or "epoch-based",
        "trainable_params": trainable_parameter_count(model),
        "loss_before": loss_before,
        "loss_after": loss_after,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
