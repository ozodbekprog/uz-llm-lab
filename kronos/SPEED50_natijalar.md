# SPEED-50 — natijalar va kunlik taqqoslash (2026-09-22)

## 🏆 Bugungi asosiy yutuq

**8.67 tok/s → 67.06 tok/s (7.7x o'sish, bir kunda)** — Maple-Preview 20B-A1B
ternary MoE, GTX 1650 (4GB) + i5-12450H (16GB) da Vulkan GPU gibrid rejimda.

## Kecha (2026-09-21) vs Bugun (2026-09-22)

| Model / ko'rsatkich | Kecha | Bugun | O'sish |
|---|---|---|---|
| BitNet b1.58 2B4T | 21.6 tok/s (CPU) | — | — |
| Falcon3-7B-1.58bit | 11.0 tok/s (CPU) | — | — |
| Falcon3-10B-1.58bit | 8.67 tok/s (CPU) | **14.80 tok/s** (Vulkan TQ2_0) | **+71%** |
| Maple-Preview 20B-A1B | — | **67.06 tok/s** (GPU gibrid) | yangi |
| Maple (64k kontekst) | — | 56.9 tok/s | yangi |
| Eng tez model | 21.6 tok/s (2B) | **67.06 tok/s (20B)** | **3.1x** |
| uzbench (10B) | 10% | 16.7% (TQ2_0, regressiyasiz) | +6.7pp |
| Kontekst (server) | 16k | **64k** (Maple) | 4x |

## Bugungi bosqichlar

### 1. KRONOS BOSQICH 0 — da'volar tekshiruvi
- **SASORI** (real, HF FardoX): 6.19 bpw, 14B da 1.7–5.8 tok/s, kernelda zero-skip yo'q,
  custom type id 43 → tezlik bermaydi, ishlatilmadi
- **KroneckerBERT**: 19x (21x emas), faqat BERT+KD → LLM'da post-hoc yaroqsiz (probe bilan isbotlandi)
- **Zero-skip**: Sparse-BitNet 1.30x faqat Ampere+ (TC); BITCOS AVX2 1.18–1.28x (kod yo'q);
  PR #23439 faqat d==0 bloklarni skip qiladi
- **Layer sharding**: CUDA'da ternary kernel **yo'q**, **Vulkan'da bor** (`mul_mat_vec_tq2_0.comp`,
  `mul_mat_vec_id_tq2_0` — MoE ham) → asosiy kashfiyot

### 2. Kronecker probe (BOSQICH 1)
`kronos/kronecker_probe.py` — haqiqiy qatlamlarda NKK vs SVD: teng parametrda **SVD har doim yutadi**
(salbiy natija, 1-qatlam arxivga).

### 3. I2_S → TQ2_0 lossless transkoder
`kronos/i2s_to_tq2_0.py` — bit-maydon permutatsiyasi, tritlar aynan, scale fp32→fp16.
3.8GB ni 30 soniyada o'giradi. Falcon3-10B: 8.67 → 14.54 tok/s.

### 4. SPEED-50 — Maple-Preview 20B-A1B (deepgrove, MIT)
20B jami / ~1B aktiv, 24 qatlam, 256 ekspert (8 aktiv), SWA-512, 131k kontekst.
GGUF: `maple-preview-TQ2_0-head-Q4_K.gguf` 5.49 GiB (HF'dan).

**`--n-cpu-moe` sweep (tg32, Vulkan1):**

| ncmoe | tok/s |
|---:|---:|
| CPU-only | 36.97 |
| 24 | 48.73 |
| 20 | 52.06 |
| 16 | 57.89 |
| 12 | 64.99 |
| **8** | **67.06** |
| 6 / 4 | OOM (VRAM chegarasi) |

**Ishlamagan optimizatsiyalar:** spec decode (Qwen2.5-0.5B draft) 56 tok/s — sekinlashtirdi;
threads t6/t10/t12 — t8 optimal; `-fa 1` neytral; 2-GPU split (NVIDIA+Intel) — OOM.

### 5. Server konfiguratsiyalari
- Tezlik: `kronos/start_maple_vulkan.sh` (ngl 99, ncmoe 8, c4096) — **65 tok/s real**
- Kontekst: `CTX=65536 NCMOE=12` — **56.9 tok/s, 64k kontekst** (uzun reasoning uchun)

### 6. O'zbek tili xulosasi
Maple o'zbek tiliga **moslashmagan** (ruscha o'ylaydi, so'zlarni xato tushunadi, tarjimasi buzilgan).
O'zbek uchun: `uzlm/alloma-8B-Instruct` (Q4 GGUF tayyor: `kmamaroziqov/alloma-8b-q4`, 4.92GB).

## Fayllar
- `kronos/i2s_to_tq2_0.py`, `kronos/kronecker_probe.py`
- `kronos/BOSQICH0_hisobot.md`, `kronos/BOSQICH234_natijalar.md`, `kronos/SPEED50_natijalar.md`
- `kronos/start_falcon10b_vulkan.sh`, `kronos/start_maple_vulkan.sh`
- Modellar (gitignored): `models/gguf/Falcon3-10B-Instruct-1.58bit-tq2_0.gguf`,
  `models/gguf/maple-preview-TQ2_0-head-Q4_K.gguf`
- Build (gitignored): `tools/llama-vulkan/` (llama.cpp mainline 7ab4ee7 + Vulkan)
