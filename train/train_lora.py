import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


class CausalCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
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


def load_messages(path, max_samples):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data/uzbek-instruct.jsonl")
    ap.add_argument("--out", default="models/lora-qwen05b-uz")
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--load-4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--optim", default="adamw_torch")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_messages(args.data, args.max_samples)
    print(f"namunalar: {len(rows)}")

    def to_texts(batch):
        texts = []
        for msgs in batch["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
            texts.append(text)
        return {"text": texts}

    ds = Dataset.from_list(rows).map(to_texts, batched=True, remove_columns=["messages"])

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
            padding=False,
        )
        enc["labels"] = [ids[:] for ids in enc["input_ids"]]
        return enc

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds = ds.filter(lambda x: len(x["input_ids"]) > 16)
    print(f"tokendan keyin: {len(ds)}")

    use_4bit = args.load_4bit and torch.cuda.is_available()
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto"
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.grad_checkpoint
        )
    else:
        dt = torch.float32 if args.dtype == "fp32" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt)
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.out + "-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        fp16=args.dtype == "fp16" and torch.cuda.is_available(),
        optim=args.optim,
        gradient_checkpointing=args.grad_checkpoint,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        dataloader_num_workers=2,
    )

    collator = CausalCollator(tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"sag'landi: {args.out}")


if __name__ == "__main__":
    main()
