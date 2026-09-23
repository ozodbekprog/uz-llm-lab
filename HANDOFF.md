# HANDOFF — uz-llm-lab (boshqa agentlar uchun brifing)

**Sana:** 2026-09-23 · **Mashina:** i5-12450H (8C/12T) + GTX 1650 4GB (sm_75) + 15.3GB DDR4 + NVMe · Fedora 44

## 0. Birdaniga

**Maqsad:** 4GB VRAM laptopda yirik ternary (1.58-bit) modellarni tez ishlatish, o'zbek tiliga
moslashtirish yo'lini ochish va universitetga klaster olish uchun jonli demo tayyorlash.

**Asosiy papka:** `/home/tony/Documents/Default Project/uz-llm-lab`
**GitHub (private):** https://github.com/ozodbekprog/uz-llm-lab

## 1. Papkalar xaritasi

| Papka | Nima bor |
|---|---|
| `bitnet/` | Microsoft BitNet (bitnet.cpp) clone + CPU build. `build/bin/`: llama-bench, llama-server, llama-cli, llama-quantize. Ichida `models/` — i2_s GGUF lar (BitNet-2B, Falcon3-1B/3B/7B/10B) |
| `tools/llama-vulkan/` | Mainline llama.cpp **Vulkan** build (TQ2_0 GPU kernellari bor). Maple/Falcon TQ2_0 serverlari shu bilan ishlaydi |
| `models/` | Eksport qilingan modellar: `qwen05b-ternary`, `lora-qwen05b-ternary-uz`, `qwen05b-dtoqat`, HGF (`hgf-qwen05b-ot`, `qwen05b-hgf-merged`), `gguf/` — `Falcon3-10B-...-tq2_0.gguf`, `maple-preview-TQ2_0-head-Q4_K.gguf`, `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (yuklash chala) |
| `kronos/` | KRONOS ishlari: `i2s_to_tq2_0.py` (lossless transkoder), `start_falcon10b_vulkan.sh`, `start_maple_vulkan.sh`, hisobotlar (`BOSQICH0_hisobot.md`, `BOSQICH234_natijalar.md`) |
| `train/` | `train_lora.py` (bizning), DTO/HGF: `dto_layer.py`, `dto_qat.py`, `dto_trainer.py` |
| `quantize/` | `ternarize.py` (absmean), Sinkhorn/Anderson: `sinkhorn.py`, `anderson_sinkhorn.py` |
| `eval/` | `run_bench.py` (transformers), `run_bench_api.py` (llama-server API), `energy_bench.py` (RAPL), `sprt_benchmark.py` |
| `data/` | `uzbek-instruct.jsonl` (14k namuna), `UzbekMMLU.parquet`; `eval/uzbench_mini.jsonl` (30 savol, A/B/C/D) |
| `logs/` | `results_*.json` (benchmark natijalari), o'qitish loglari |
| `configs/` | `matrix_dto.yaml` |

## 2. O'lchangan natijalar

### Tezlik (llama-bench, tg32, t=8)

| Model | Konfiguratsiya | tok/s |
|---|---|---|
| BitNet b1.58 2B4T | CPU, i2_s | 21.6 |
| Falcon3-7B-1.58bit | CPU, i2_s | 11.0 |
| Falcon3-10B-1.58bit | CPU, i2_s | 8.67 |
| **Falcon3-10B (TQ2_0)** | **Vulkan GPU gibrid, -ngl 40** | **14.54–14.80** (VRAM 3573MB) |
| Maple-Preview 20B-A1B (MoE, TQ2_0) | CPU | 36.97 |
| **Maple 20B-A1B** | **Vulkan gibrid, ncmoe=8** | **67.06** (64k ctx: 56.9) |

### Sifat (uzbench_mini, n=30)

| Model | Aniqlik |
|---|---|
| Qwen2.5-0.5B-Instruct (FP16) | 23.3% |
| Falcon3-7B (i2_s) | 6.7% |
| Falcon3-10B (TQ2_0) | 16.7% |
| Qwen2.5-0.5B absmean ternary | 0.0% |
| Qwen2.5-0.5B ternary + LoRA (1 epoch) | 0.0% (train loss 11.9→6.1) |
| DTO-QAT (qwen05b-dtoqat) | **hali o'lchanmagan** |

### Muhim topilmalar

- **fp16 bu tizimda buzuq/sekin** (163 GFLOPS vs fp32 859 GFLOPS) → o'qitishda `--dtype fp32` ishlatilsin.
- 10B CPU'da memory-bandwidth chegarasida (eff. 68%); thread tuning / spec-decoding foyda bermadi (spec-decoding 6.55 — sekinlashtirdi).
- **CUDA backendda TQ2_0/I2_S kernellari yo'q; Vulkan backendda bor** → GPU uchun yagona yo'l: TQ2_0 + Vulkan.
- Post-hoc absmean ternary 0.5B'ni buzadi; oddiy LoRA tiklamaydi → HESTIA/Tequila/SINQ kerak (MATRIX-DTO yo'nalishi tasdiqlandi).
- Kronecker probe: LLM'da post-hoc yaroqsiz (BOSQICH0'da batafsil).

## 3. Ishlaydigan buyruqlar

```bash
cd "/home/tony/Documents/Default Project/uz-llm-lab"

# Falcon3-10B chat (CPU i2_s, 8.7 tok/s) — systemd xizmati bilan
systemd-run --user --unit=falcon-chat --collect \
  --setenv=MODEL="models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf" \
  "$PWD/start_falcon_chat.sh"

# Falcon3-10B chat (TQ2_0 + Vulkan, 14.5 tok/s) — tavsiya etiladi
kronos/start_falcon10b_vulkan.sh              # PORT/CTX/NGL/DEV env orqali

# Maple 20B MoE chat (Vulkan gibrid, 67 tok/s)
kronos/start_maple_vulkan.sh

# Transkod (i2_s -> TQ2_0, lossless, ~30 s)
.venv/bin/python kronos/i2s_to_tq2_0.py \
  --input bitnet/models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf \
  --output models/gguf/Falcon3-10B-Instruct-1.58bit-tq2_0.gguf

# Tezlik o'lchash (CPU)
bitnet/build/bin/llama-bench -m bitnet/models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf -p 64 -n 32 -t 8

# Sifat o'lchash (server ishlab turganda)
.venv/bin/python eval/run_bench_api.py --url http://127.0.0.1:8080 --out logs/results_x.json

# LoRA o'qitish (fp32 majburiy; 0.5B, batch1, grad-accum 16)
.venv/bin/python train/train_lora.py --model models/qwen05b-ternary \
  --out models/lora-qwen05b-ternary-uz --no-load-4bit --dtype fp32 \
  --max-samples 4000 --max-len 256 --batch 1 --grad-accum 16
```

## 4. Tuzoqlar (tajribadan)

1. **Xotira:** 15.3GB umumiy; 10B + KV ≈ 5GB. Boshqa og'ir ish (o'qitish, STT) bilan birga ishlasa
   systemd-oomd chat serverni o'ldiradi. `--parallel 1` majburiy, CTX ni ehtiyot bilan tanlang.
2. **GPU fp16 buzuq** — o'qitishda fp32. CUDA toolkit o'rnatilmagan (kerak bo'lsa: `cuda-toolkit-13-0`, fedora44 repo).
3. `llama-cli -cnv` conversation rejimiga tushib qoladi — o'lchov uchun `llama-bench` yoki API ishlatilsin.
4. `pkill -f "pattern"` o'z-o'zini o'ldirishi mumkin — bracket trick: `pkill -f "llama-benc[h]"`.
5. git: `models/`, `.venv/`, `bitnet/` `.gitignore` da — commit qilmang; repo private.
6. HF cache: `/home/tony/.fariza/cache/huggingface`.

## 5. Ochiq ishlar (keyingi qadamlar)

- [ ] **DTO-QAT modelini o'lchash**: `models/qwen05b-dtoqat` → uzbench (baseline/LoRA bilan solishtirish)
- [ ] **Litespark** benchmark: model 4.6/4.83GB yuklangan, to'xtagan → tugatib, 2B natijasini bitnet.cpp (21.6) bilan solishtirish
- [ ] **Energiya**: RAPL o'qish uchun `sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj`
- [ ] **Ternary-Bonsai-2-27B-PTQ1_0** yuklash chala (4.0GB/~6+) → tugatib Vulkan/CPU sinash
- [ ] **KRONOS optimallashtirish**: shader tuning / zero-skip (~1.5–2x nazariy zaxira), alloma-8B (o'zbek uchun baza)
- [ ] **Universitet hisoboti**: raqamlar tayyor, matn yozish va demo skript
- [ ] CUDA toolkit (ixtiyoriy) — bitnet.cpp GPU kernel faqat 2B uchun

## 6. Parallel ishlash qoidalari

- Bir vaqtda bir nechta agent ishlaydi: ish boshlashdan oldin `git status` va `pgrep -af llama-serve[r]` tekshiring.
- Chat serverlarni to'xtatishdan oldin kim ishlatayotgani noma'lum bo'lsa — qayta ishga tushirish buyrug'ini tayyorlab qo'ying.
- Uzoq ishlarni `setsid --fork ... > log 2>&1` bilan fonda yuriting (shell timeout o'ldirmasin).
- Natijalarni `logs/*.json` ga yozing va `kronos/*.md`/README'ga qisqa hisobot qo'shing.
