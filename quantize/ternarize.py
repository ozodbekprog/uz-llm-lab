import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def ternarize(weight: torch.Tensor) -> torch.Tensor:
    alpha = weight.abs().mean()
    return torch.clamp(torch.round(weight / (alpha + 1e-8)), -1, 1) * alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="models/qwen05b-ternary")
    ap.add_argument("--include-lm-head", action="store_true")
    args = ap.parse_args()

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    layer_names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not args.include_lm_head and name == "lm_head":
            continue
        module.weight.data = ternarize(module.weight.data.float()).to(module.weight.dtype)
        layer_names.append(name)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"ternary: {len(layer_names)} qatlam -> {args.out}")
    print("qatlamlar:", ", ".join(layer_names[:6]), "..." if len(layer_names) > 6 else "")


if __name__ == "__main__":
    main()
