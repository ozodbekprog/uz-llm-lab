# uz-llm-lab

Arzon temirda (GTX 1650 4GB + 16GB RAM laptop) ternary (1.58-bit) LLM lar bilan tajribalar.

## Natijalar (2026-09-21)

| Model | Parametrlar | Fayl hajmi | Backend | Tezlik |
|---|---|---|---|---|
| BitNet b1.58 2B4T | 2.41B | 1.10 GB | CPU | **21.6 tok/s** |
| Falcon3-7B-Instruct-1.58bit | 7.46B | 3.05 GB | CPU | **11.0 tok/s** |
| Falcon3-10B-Instruct-1.58bit | 10.31B | 3.71 GB | CPU | **8.67 tok/s** |

- 7B model odatda Q4 kvantlashda ~5 GB VRAM talab qiladi (4GB laptopda ishlamaydi).
- Ternary formatda 3 GB, GPU'siz, CPU'da 11 tok/s; 10B model ham 3.7 GB bilan ishlaydi.

## Sifat (uzbench_mini, n=30)

| Model | Aniqlik |
|---|---|
| Qwen2.5-0.5B-Instruct (FP16) | 23.3% |
| Falcon3-7B-Instruct-1.58bit | 6.7% |
| Qwen2.5-0.5B (absmean ternary) | 0.0% |
| Qwen2.5-0.5B (ternary + LoRA) | 0.0% (loss 11.9 -> 6.1) |

Xulosa: katta model o'zbek tilida avtomatik kuchli emas; moslashtirish (fine-tune) kerak.
Post-hoc absmean ternary kichik modelni buzadi, oddiy LoRA tiklay olmaydi -
ilg'or yondashuvlar kerak (HESTIA, Tequila, CAT-Q, BitDistill).

## Optimallashtirishlar

- 16k kontekst + q8_0 KV-cache + flash-attn auto (llama-server)
- systemd user xizmati: `systemctl --user status falcon-chat`
- Energiya o'lchovi: `eval/energy_bench.py` (RAPL; `sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj`)

## Tuzilma

- `quantize/ternarize.py` - absmean ternary kvantlash
- `train/train_lora.py` - LoRA / QLoRA fine-tune (fp32 / 4-bit)
- `eval/run_bench.py` - uzbench_mini benchmark (transformers)
- `eval/run_bench_api.py` - uzbench_mini benchmark (llama-server API)
- `eval/energy_bench.py` - energiya o'lchovi (RAPL)
- `data/uzbek-instruct.jsonl` - o'zbekcha instruct dataset (14k namuna)
- `data/UzbekMMLU.parquet`, `eval/uzbench_mini.jsonl` - baholash to'plamlari
- `start_falcon_chat.sh` - llama-server chat (MODEL/PORT/CTX env orqali sozlanadi)

## Reproduksiya

```bash
python -m venv .venv && .venv/bin/pip install torch transformers peft datasets bitsandbytes accelerate
# ternary kvantlash
.venv/bin/python quantize/ternarize.py --model Qwen/Qwen2.5-0.5B-Instruct
# LoRA o'qitish (fp32; GTX 1650 da fp16 sekin)
.venv/bin/python train/train_lora.py --model models/qwen05b-ternary --no-load-4bit --dtype fp32
# benchmark
.venv/bin/python eval/run_bench.py --model models/qwen05b-ternary --adapter models/lora-qwen05b-ternary-uz
```

bitnet.cpp (Microsoft) inference uchun: gcc bilan build qilinadi (setup_env.py da clang -> gcc).

## Muhit

- Laptop: i5-12450H, GTX 1650 4GB, 16GB RAM, Fedora 44
- Muhim: bu tizimda fp16 matmul sekin (163 GFLOPS) vs fp32 (859 GFLOPS) - o'qitish fp32 da.
