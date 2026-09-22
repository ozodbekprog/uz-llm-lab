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

## MATRIX-DTO (BOSQICH 1-2, 2026-09-21)

Ternary (1.58-bit) o'qitish/inference tizimi. Hozircha 1-qatlam (Anderson-Sinkhorn
OT kvantlash) va 2-qatlamga poydevor qo'yildi.

**Modullar:**

- `quantize/sinkhorn.py` - log-domain Sinkhorn (entropik OT), barqaror, batched
- `quantize/anderson.py` - Anderson acceleration (type-II mixing, batched normal equations)
- `quantize/anderson_sinkhorn.py` - ternary OT kvantizator: MSE-optimal `alpha`
  (grid + ternary search), `{-α,0,+α}` support, barycentric projection
- `quantize/repro.py` - seed/device yordamchilar
- `scripts/check_env.py` - muhit auditi (GPU, CUDA, paketlar, tezliklar)
- `scripts/quantize_ot.py` - model kvantlash CLI + loss + mikrobenchmark
- `tests/` - 22 pytest testi (marginallar, 1D OT aniq yechimi, Anderson tezligi,
  ternary xossalari)
- `configs/matrix_dto.yaml` - global konfiguratsiya

**Ishga tushirish:**

```bash
.venv/bin/python scripts/check_env.py
.venv/bin/python -m pytest tests/
.venv/bin/python scripts/quantize_ot.py --config configs/matrix_dto.yaml \
    --eval-samples 24 --baselines tensor,row --tag full --save
```

**Natijalar (Qwen2.5-0.5B, 168 Linear qatlam):**

| Variant | rel_err | MSE | Vaqt |
|---|---|---|---|
| OT (row) | **0.4640** | **1.06e-04** | 106 s |
| OT (group=128) | 0.4574 | 1.03e-04 | 637 s |
| OT (tensor) | 0.4899 | 1.28e-04 | 31 s |
| absmean row | 0.5466 | 1.47e-04 | 0.6 s |
| absmean tensor (eski usul) | 0.5780 | 1.78e-04 | 0.6 s |

Anderson (iteratsiya): production sozlamasida **6.2x** kam (80 -> 13),
qiyin rejimda **154x** kam (4000 -> 26).

**Muhim xulosa:** post-hoc ternary (OT ham) 0.5B modelni buzadi: NLL 4.57 -> 13.53,
uzbench_mini 23.3% -> 0.0%. Kvantlash sifati yaxshilandi, lekin imkoniyatni
saqlash uchun o'qitish/kompensatsiya (BOSQICH 3-4: DTO/HESTIA, HGF) shart.

## MATRIX-DTO BOSQICH 3: DTO/HGF gibrid (2026-09-21)

4GB VRAM uchun amaliy variant: ternary baza muzlatilgan, faqat HGF (low-rank +
per-output gate, `quality/hgf_correction.py`) `train/dto_trainer.py` bilan
o'qitiladi. Gradient checkpointing + batch 1 x accum 8 sig'adi.

| Metrika | Qiymat |
|---|---|
| Trainable param | 9.1M / 494M (1.8%) |
| O'qitish | 500 qadam, 4000 namuna, ~19 daqiqa (GTX 1650) |
| NLL (held-out 24) | ternary 13.78 -> **HGF 4.65** (fp32: 4.57) |
| \|\|delta_HGF\|\| / \|\|xato_ternary\|\| | median **2.9%** (1-10%) |
| uzbench_mini | **0.0%** (fp32: 23.3%) |

Xulosa: HGF NLL ni deyarli to'liq tiklaydi va juda kichik (2.9%) korreksiya
bilan buni qiladi (xato strukturali), lekin savol-javob imkoniyati qaytmaydi -
NLL yaxshilanishi imkoniyatga teng emas. Keyingi qadam: ternary og'irliklarni
o'zi o'qitadigan DTO/QAT (softmax relaxation) yoki kattaroq model (7B-10B
xatoga chidamliroq).

Artefaktlar: `models/hgf-qwen05b-ot/hgf_weights.pt`,
`models/qwen05b-hgf-merged/`, `logs/hgf_full.stdout`,
`logs/results_hgf_bench.json`.

## MATRIX-DTO BOSQICH 4: SPRT baholash (2026-09-21)

`eval/sprt_benchmark.py` - Waldning ketma-ket ehtimollik nisbati testi
(p0=0.25, p1=0.50, alpha=beta=0.05). 10 unit test.

| Model | n | aniqlik | SPRT to'xtadi | Qaror | Namuna tejash |
|---|---|---|---|---|---|
| Qwen2.5-0.5B fp32 | 30 | 23.3% | 16 | chance | 46.7% |
| OT ternary | 30 | 0% | 8 | chance | 73.3% |
| HGF merged | 30 | 0% | 8 | chance | 73.3% |
| Falcon3-7B ternary | 30 | 10% | 13 | chance | 56.7% |

Simulyatsiya: haqiqiy aniqlik 50% bo'lsa o'rtacha 20.6 namuna (fixed-n 42 ->
51% tejash), 70% bo'lsa 9 namuna (79% tejash). I tur xatoligi 3.4% <= 5%.

## MATRIX-DTO BOSQICH 2 (real DTO/HESTIA QAT) va BOSQICH 5 (bitnet.cpp)

**DTO/HESTIA yadro** (`train/dto_layer.py`, `train/dto_qat.py`): latent fp32
og'irliklar, HESTIA softmax relaxation (temperatura anneling), straight-through
hard ternary, Tequila deadzone bias (zero ulushini saqlaydi), weight
checkpointing. 9 unit test.

Natija (oxirgi 6 qatlam, 800 qadam, 15 daqiqa, 8-bit AdamW, fp16 muzlatilgan
baza): NLL 13.46 -> 7.32, rel_err 0.501, uzbench 0%, generatsiya degeneratsiya.
Xulosa: faqat oxirgi qatlamlarni tuzatish yetarli emas - zarar barcha
qatlamlarda taqsimlangan; HGF (barcha qatlamlar) shu sababli yaxshiroq (4.65).

**Layer 5 (GGUF + bitnet.cpp) quvuri ishladi:**
`HF -> convert_hf_to_gguf.py -> llama-quantize TQ1_0 -> llama-server`.
0.5B model 301 MB (TQ1_0), CPU'da **62.5 tok/s**. Yo'lda uchragan muammolar:
bitnet.cpp'dagi gguf-py eski (DATASETS yo'q), gguf 0.19.0 da `add_size_label`
yo'q - ikkalasi patch qilindi (`gguf-py` -> `gguf-py.bak`, metadata pyc
tozalandi).

**Falcon3-10B ternary** (uzbench_mini): **10%** (7B: 6.7%, 0.5B ternary: 0%) -
kattaroq model kvantlashga chidamliroq.

## Tuzilma

- `quantize/ternarize.py` - absmean ternary kvantlash
- `quantize/sinkhorn.py` - log-domain Sinkhorn (entropik OT)
- `quantize/anderson.py` - Anderson acceleration
- `quantize/anderson_sinkhorn.py` - OT ternary kvantizator (layer 1)
- `train/train_lora.py` - LoRA / QLoRA fine-tune (fp32 / 4-bit)
- `train/dto_layer.py` - DTO/HESTIA ternary qatlam (layer 2)
- `train/dto_qat.py` - DTO QAT trener (oxirgi K qatlam)
- `train/dto_trainer.py` - HGF gibrid trener (layer 2/3)
- `quality/hgf_correction.py` - HGF low-rank tuzatish (layer 3)
- `eval/sprt_benchmark.py` - SPRT ketma-ket baholash (layer 4)
- `scripts/check_env.py`, `scripts/quantize_ot.py` - muhit va kvantlash CLI
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

## KRONOS: Vulkan GPU gibrid (2026-09-22)

- CUDA backendida ternary (I2_S/TQ1_0/TQ2_0) kernellar **yo'q**; Vulkan'da bor
  (`mul_mat_vec_tq2_0.comp`) - shuning uchun GTX 1650 Vulkan orqali ishlatildi.
- `kronos/i2s_to_tq2_0.py` - lossless I2_S->TQ2_0 transkoder (tritlar aynan;
  bit-maydon permutatsiyasi; scale fp32->fp16), 30 s, 3.8 -> 4.07 GB.
- Natija: CPU-only 8.67 -> **14.54-14.80 tok/s** (+68-71%).
  `-ngl` sweep tg32: 0=7.33, 16=9.54, 24=11.74, 32=13.32, 36=14.03, 40=**14.80**;
  real 4k/q8_0 KV server: ngl 40 = 14.54 tok/s, VRAM 3573 MB (ngl 36: 13.71, 3306 MB).
- Sifat: uzbench_mini **16.7%** (I2_S baseline 10%) - regressiya yo'q.
- Batafsil: `kronos/BOSQICH0_hisobot.md`, `kronos/BOSQICH234_natijalar.md`;
  server: `kronos/start_falcon10b_vulkan.sh`; build: `tools/llama-vulkan/bin/`.
