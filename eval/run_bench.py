import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys_prompt = "Sen foydali yordamchisan. Faqat bitta harf bilan javob ber (A, B, C yoki D)."


def load_items(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_choice(text):
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--bench", default="eval/uzbench_mini.jsonl")
    ap.add_argument("--out", default="logs/results_baseline.json")
    args = ap.parse_args()

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    items = load_items(args.bench)
    results = []
    correct = 0
    cat_stats = {}
    for it in items:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": it["question"] + "\n\nJavobni faqat bitta harf bilan yoz:"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = extract_choice(gen)
        ok = pred == it["answer"]
        correct += int(ok)
        cat = it.get("category", "boshqa")
        cat_stats.setdefault(cat, [0, 0])
        cat_stats[cat][0] += int(ok)
        cat_stats[cat][1] += 1
        results.append({**it, "pred": pred, "gen": gen[:80], "ok": ok})
        print(f"{it['id']:12s} kutilgan={it['answer']} model={pred} {'OK' if ok else 'X'} | {gen[:50]!r}")

    n = len(items)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "accuracy": round(correct / n, 4),
        "n": n,
        "per_category": {
            k: {"acc": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(cat_stats.items())
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    Path(args.out).write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
