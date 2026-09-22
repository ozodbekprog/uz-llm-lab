# KRONOS BOSQICH 2–4 natijalari — Vulkan GPU gibrid ishga tushdi

**Sana:** 2026-09-22 · **Asos:** BOSQICH 0 hisoboti (`BOSQICH0_hisobot.md`)

## Qisqa natija

**8.67 tok/s (CPU-only I2_S) → 14.54–14.80 tok/s (TQ2_0 + Vulkan GPU gibrid) — +68…71%.**
Kutilgan 14–50 tok/s oralig'ining quyi qismiga real sharoitda erishildi. 50 tok/s bu GPU'da
fizik jihatdan mumkin emas (nazariy cap ≈ 52 ideal holda; ~47 GB/s ≈ 28–36% band kenglik
ishlatilyapti — optimallashtirish zaxirasi bor).

## Bajarilgan ishlar

| # | Ish | Fayl / natija |
|---|---|---|
| 1 | I2_S→TQ2_0 transkoder (lossless: tritlar aynan, bit maydon permutatsiyasi; scale fp32→fp16) | `kronos/i2s_to_tq2_0.py`, 30 s, 3.8→4.07 GB, 280 tensor |
| 2 | llama.cpp mainline Vulkan build (commit 7ab4ee7; glslc + spirv-headers) | `/tmp/opencode/llama.cpp-main/build-vulkan` |
| 3 | Format validatsiyasi: CPU va GPU bir xil matn berdi (" Paris, not Rome...") | ✅ |
| 4 | `-ngl` sweep (llama-bench, q8_0 KV, Vulkan1 = GTX 1650) | quyidagi jadval |
| 5 | Real kontekst (c=4096, q8 KV) server testi + VRAM monitoring | ngl40: 14.54 tok/s, 3573 MB |
| 6 | Sifat testi (uzbench_mini, API) | `logs/results_falcon10b_tq2_vulkan.json` |

## `-ngl` sweep (llama-bench: pp64 / tg32, q8_0 KV, Vulkan1)

| -ngl | pp64 t/s | tg32 tok/s | izoh |
|---:|---:|---:|---|
| 0 | 59.0 | 7.33 | sof CPU (TQ2_0 CPU kerneli I2_S'dan ~15% sekin) |
| 8 | 81.0 | 7.74 | |
| 16 | 87.9 | 9.54 | |
| 24 | 95.8 | 11.74 | |
| 32 | 104.6 | 13.32 | xavfsiz zona |
| 36 | 109.1 | 14.03 | VRAM 3306 MB (real 4k) |
| **40** | **114.8** | **14.80** | VRAM 3573 MB (real 4k) — to'liq offload sig'adi |

## Nega ishladi (kalit kashfiyot)

llama.cpp mainline **CUDA** backendida TQ1_0/TQ2_0/I2_S kernellari **yo'q**, lekin
**Vulkan** backendida bor: `mul_mat_vec_tq2_0.comp`, `dequant_tq1_0.comp` va h.k.
GTX 1650 Vulkan orqali ishlaydi (D2D ~167 GB/s vs DDR4 ~26–30 GB/s → ~5x).

Bandwidth modeli:

$$t_{tok} = \frac{B_{cpu}}{S_{cpu}} + \frac{B_{gpu}}{S_{gpu}}, \quad
B_{gpu} = 2.43\,GB\ (TQ2\_0\ body) + 0.805\,GB\ (F16\ head)$$

Hozirgi o'lchov: $3.24/0.0145 \approx 223\,GB/s$?? — yo'q: $3.24\,GB \times 14.54 /s =
47\,GB/s$ effective (VRAM 128–167 GB/s dan ~28–36%). Shuning uchun keyingi optimizatsiya
(shader tuning, zero-skip) nazariy jihatdan yana ~1.5–2x berishi mumkin.

## Buyruqlar (ishlaydigan)

```bash
# Transkod (bir marta) - allaqachon bajarilgan, model: models/gguf/Falcon3-10B-Instruct-1.58bit-tq2_0.gguf
cd uz-llm-lab
.venv/bin/python kronos/i2s_to_tq2_0.py \
  --input bitnet/models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf \
  --output models/gguf/Falcon3-10B-Instruct-1.58bit-tq2_0.gguf

# Server (14.5 tok/s, 4k kontekst, q8 KV) - doimiy o'rnatilgan build bilan
kronos/start_falcon10b_vulkan.sh          # PORT/CTX/NGL/DEV env orqali

# yoki qo'lda:
tools/llama-vulkan/bin/llama-server -m models/gguf/Falcon3-10B-Instruct-1.58bit-tq2_0.gguf \
  --host 127.0.0.1 --port 8080 -c 4096 -t 8 \
  -ngl 40 -dev Vulkan1 -ctk q8_0 -ctv q8_0 -fa auto

# Sweep
tools/llama-vulkan/bin/llama-bench -m models/gguf/Falcon3-10B-Instruct-1.58bit-tq2_0.gguf \
  -p 64 -n 32 -t 8 -ngl 0,8,16,24,32,36,40 -dev Vulkan1 -ctk q8_0 -ctv q8_0
```

Qulay skript: `kronos/start_falcon10b_vulkan.sh` (o'zgaruvchilar: MODEL, NGL, CTX, PORT).
Build `tools/llama-vulkan/bin/` da (RUNPATH `$ORIGIN`, `/tmp` ga bog'liq emas; commit 7ab4ee7).

## Cheklovlar / keyingi qadamlar

1. **-ngl 40 VRAM 3573 MB** — brauzer/desktop GPU ishlatsa OOM xavfi; zaxira: `-ngl 36` (3306 MB, 13.7 tok/s).
2. **Vulkan TQ2_0 GEMV samaradorligi ~30%** — shader tuning (workgroup, vector width) potensial +30–60%.
3. **Zero-skip (BITCOS uslubi)** — Vulkan shaderda nollarni bitmap orqali skip qilish; izoh: Sparse-BitNet
   g'oyasi Ampere TC talab qiladi, lekin bitmap+compact sign vektori GPU'da zero-skip imkonini beradi
   (qo'shimcha 1.15–1.3x, original ish).
4. **SASORI / Kronecker** — yakuniy xulosa: bu pipeline uchun yaroqsiz (BOSQICH 0).
5. 30B-A3B (MoE) sinovi: ternary MoE ~7–8 GB; `--n-cpu-moe` bilan gibrid mumkin (kelajak).
