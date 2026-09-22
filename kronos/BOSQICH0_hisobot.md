# KRONOS — BOSQICH 0 hisoboti (Feasibility)

**Sana:** 2026-09-22 · **Mashina:** GTX 1650 4GB (sm_75, D2D ~167 GB/s), i5-12450H (8C), 15.3GB DDR4 (~26-30 GB/s), NVMe.
**Baza:** `uz-llm-lab` (bitnet.cpp b9918 CPU-only build, Falcon3-10B-Instruct-1.58bit I2_S = 3.8GB, CPU'da 8.67 tok/s).

---

## 1. Xulosa (TL;DR)

| # | Texnologiya | Realmi? | Bizning maqsadga yaroqlimi? |
|---|---|---|---|
| 1 | KroneckerBERT (19x) | ✅ Ha (arXiv 2109.06243, NAACL 2022) | ❌ Yo'q — BERT, qayta o'qitish (KD) talab; LLM'da post-hoc yaroqsiz (quyida probe) |
| 2 | SASORI (TQ2P/TQ3P) | ✅ Ha (HF FardoX) | ❌ Yo'q — 6.19 bpw, sekinroq (14B: 1.7–5.8 tok/s), zero-skip yo'q |
| 3 | Zero-skipping | ✅ Ha (MSR Sparse-BitNet 1.30x; Intel BITCOS 1.18–1.28x) | ⚠️ Qisman — GPU yo'li Turing'da yo'q (TC yo'q); CPU'da o'zimiz yozsak 1.15–1.25x |
| 4 | Layer sharding | ✅ Ha (llama.cpp -ngl/-ot) | ⚠️ Faqat **Vulkan TQ2_0** orqali (CUDA'da ternary kernel yo'q) — eng istiqbolli yo'l |

**Muhim:** Asl KRONOS rejasidagi "Kronecker → SASORI → zero-skip → sharding" zanjiri o'rniga amaliy zanjir:
**I2_S→TQ2_0 transkod → llama.cpp Vulkan offload (-ngl sweep) → (bonus) CPU zero-density/AVX2 kernel.**
Kutilma: CPU-only 8.67 → **15–25 tok/s** (to'liq yoki qismli offload). 50 tok/s fizik jihatdan bu GPU'da mumkin emas.

---

## 2. Tekshiruv jadvali (har bir da'vo — manba bilan)

| Da'vo | Manba (tekshirilgan) | Holat |
|---|---|---|
| SASORI K-trit plane, 14B TQ3P 6.1875 bpw | huggingface.co/FardoX/sasori-qwen3-14b-tq3p | ✅ Real; K=1 "reasoning'da yiqiladi", K=3 sifat qaytadi; EF512 = GPTQ-uslub kalibrlash |
| "SASORI 30B ~14 tok/s CPU" | topilmadi (HF'da 30B yo'q; karta 14B: 1.7–5.8 tok/s, 8-yadro AMD) | ❌ Tasdiqlanmadi |
| SASORI kernel zero-skip qiladi | SASORI kartasi: "dense maddubs, **no zero-skipping**, value-independent" | ❌ Yo'q |
| SASORI integratsiya | custom ggml type id 43 (upstream 41 gacha), llama.cpp @6f3a9f3 + patch, runtime tar ichida; GitHub repo topilmadi | ⚠️ Yopiq/sinmagan |
| KroneckerBERT 21x | arXiv 2109.06243: **19x (BERT_BASE'ning 5%)**, KD bilan | ⚠️ 21x emas, 19x |
| Kronecker LLM'da | faqat BERT (encoder) + o'qitish; decoder-LLM dalil yo'q | ❌ |
| Nollar ~50–70% | Intel BITCOS (arXiv 2609.16338): 29 model, maks **51.5%**; BitNet ~42% | ⚠️ Yuqori chegara 51.5% |
| Zero-skip tezlik | Sparse-BitNet (arXiv 2603.05168): 1.30x — **maxsus sparse tensor core**; spbitnet: CC 8.0+ (Ampere) | ❌ GTX 1650 (sm_75, TC yo'q) |
| CPU zero-skip | BITCOS: AVX2/AVX-512 unpack, 1.18–1.28x GEMV; kod publ. ko'rinmadi | ⚠️ O'zimiz yozishimiz kerak |
| Layer sharding flaglari | `-ngl`, `-ot`, `-ctk/-ctv` (q8_0), `--n-cpu-moe` — lokal bitnet.cpp build'da tasdiqlandi | ✅ |
| CUDA ternary kernel | mainline llama.cpp `ggml-cuda/` da TQ1_0/TQ2_0 **0 ta havola**; ggml type: TQ1_0=34, TQ2_0=35, (fork) I2_S=36 | ❌ CUDA'da yo'q |
| Vulkan ternary kernel | mainline: `mul_mat_vec_tq2_0.comp`, `dequant_tq1_0.comp`, TQ1_0/TQ2_0 shaderlari | ✅ **Bor** |
| Falcon3-10B arxitekturasi | GGUF: 40 qatlam, hidden 3072, vocab 131072, FFN 23040, 12/4 head, KV hajmi 80 KB/tok (q8) | ✅ |
| Model tarkibi | body (I2_S) ≈ 2.2GB + token_embd F16 805MB + output F16 805MB + F32 normlar = 3.8GB | ✅ |
| GTX 1650 fp16 compute | env_report: fp32 1484 GFLOPS, fp16 **315 GFLOPS** (buzuq); lekin matvec memory-bound → bandwidth muhim (167 GB/s D2D) | ⚠️ fp16 o'qitish yo'q |

---

## 3. Matematik chegaralar (nega 50 tok/s mumkin emas)

Decode = memory-bandwidth-bound. Token uchun o'qiladigan baytlar:

$$t_{tok} = \frac{B_{cpu}}{S_{cpu}} + \frac{B_{gpu}}{S_{gpu}}, \qquad B_{tok} = \underbrace{B_{body}}_{2.2\,GB} + \underbrace{B_{head}}_{0.805\,GB} \approx 3.0\,GB$$

**CPU-only (hozir):** $3.0/0.026 \approx 115\,ms \Rightarrow 8.7$ tok/s — o'lchovga mos ✅. Nazariy shift $3.0/0.030 = 100\,ms \Rightarrow 10$ tok/s. **14+ tok/s CPU-only — faqat format kichraysa (zero-density/bitcos yoki F16 head'ni q8_0 ga tushirish).**

**TQ2_0 + to'liq Vulkan (nazariy eng yaxshi):**

$$B = \underbrace{9.44B \times \tfrac{2.0625}{8}}_{2.43\,GB\ body} + \underbrace{0.40\,GB}_{q8\_0\ head} + \underbrace{0.33\,GB}_{KV\ q8,\ 4k} \approx 3.16\,GB$$

$$t_{tok} \ge 3.16/0.167 = 19\,ms \Rightarrow \le 52\ \text{tok/s (ideal, nol overhead)}$$

Real Vulkan GEMV samaradorligi 35–60% → **15–25 tok/s**. 50+ tok/s uchun SRAM-bandwidth nisbati 1 ga yaqin bo'lishi kerak — amalda yo'q.

**Qismli sharding** ($n$ qatlam GPU'da):

$$t_{tok} \approx \frac{B_{cpu}(n)}{0.026} + \frac{B_{gpu}(n)}{S_{gpu}^{eff}}, \qquad S_{gpu}^{eff} \approx 0.06\!-\!0.10\ \text{GB/s}\ \text{(per 1 GB) tushuniladi}$$

Masalan, 32/40 qatlam GPU'da: $B_{gpu}\approx 2.4\,GB$, $B_{cpu}\approx 0.49\,GB$ → $0.019 + 0.024 = 43\,ms \Rightarrow \sim 23$ tok/s.

---

## 4. BOSQICH 1 — Kronecker probe natijalari (haqiqiy qatlamlar)

Kod: `kronos/kronecker_probe.py` (self-test OK: Van Loan konvensiyasi, $A\otimes B$ → err1=$10^{-16}$).
Model: Qwen2.5-0.5B F16 GGUF (`models/gguf/qwen05b-ot-f16.gguf`). Batafsil: `kronos/results_qwen05b.json`.

Model: $W \approx \sum_{k=1}^{r} A_k \otimes B_k$, $A_k: m_1{\times}m_2$, $B_k: n_1{\times}n_2$, parametr $= r(m_1m_2+n_1n_2)$.

| Tensor | r | split | Param | Siqish | Kron err | SVD@teng-param | SVD err |
|---|---:|---:|---:|---:|---:|---:|---:|
| attn_q 896×896 | 16 | 14×14 | 68 672 | 11.7x | 0.816 | 38 | **0.493** |
| attn_q | 64 | 14×14 | 274 688 | 2.9x | 0.555 | 153 | **0.302** |
| attn_q | 128 | 14×14 | 549 376 | 1.5x | 0.314 | 307 | **0.179** |
| attn_output | 64 | 14×14 | 274 688 | 2.9x | 0.642 | 153 | **0.482** |
| ffn_gate 4864×896 | 64 | 32×14 | 651 264 | 6.7x | 0.897 | 113 | **0.815** |
| ffn_gate | 128 | 32×14 | 1 302 528 | 3.3x | 0.795 | 226 | **0.707** |

**Xulosa:** teng (va hatto kamroq) parametr budjetida **SVD har doim Kron'dan yaxshi** yoki unga teng. Kronecker post-hoc (qayta o'qitishsiz) decoder-LLM chiziqli qatlamlarida yaroqsiz. KroneckerBERT'ning 19x natijasi BERT + uzoq KD o'qitish bilan olingan; 4GB GPU'da 10B KD o'qitish bu mashinada mumkin emas. → **1-qatlam arxivga** (faqat maqola/ilmiy probe sifatida qoldiriladi).

---

## 5. Revize qilingan KRONOS rejasi

| Bosqich | Ish | Kut. natija |
|---|---|---|
| B0 ✅ | Feasibility (shu hisobot) | — |
| B1 ⏸ | Kronecker probe (bajarildi, arxiv) | Salbiy natija hujjatlashtirildi |
| B2 🔜 | **I2_S→TQ2_0 transkoder** (streaming, ~10 daqiqa) + llama.cpp Vulkan build | TQ2_0 model 2.43GB body |
| B3 🔜 | `-ngl` sweep (0/8/16/24/32/40), `-ctk/-ctv q8_0`, `-fa on`; TTFT/tok/s/VRAM log | Optimal sharding nuqtasi |
| B4 | q8_0/`-ot` bilan F16 head'ni GPU'ga (yoki q8_0 ga) — CPU-only zaxira yo'l | CPU-only +10% |
| B5 | I2_S nol-zichligi tahlili (`kronos/zero_density.py`) + AVX2 BITCOS-uslub eksperiment | +1.15–1.25x (ixtiyoriy) |
| B6 | SASORI kuzatuv (patch/kod chiqsa) | — |
| B7 | MoE 30B-A3B sinovi (--n-cpu-moe) | 30B-sinf, ~3B aktiv |

**Darhol bajariladigan arzon tajriba (B4.0):** mavjud fork bilan F16 head'ni q8_0 ga requantizatsiya (`llama-quantize --allow-requantize --output-tensor-type q8_0`) → CPU-only ~9.5–10 tok/s bo'lishi kerak.

---

## 6. Manbalar

- KroneckerBERT: arXiv:2109.06243; NAACL 2022 (aclanthology.org/2022.naacl-main.154)
- SASORI: huggingface.co/FardoX/sasori-qwen3-14b-tq3p (TQ3P, K=3, g=256, 6.1875 bpw, EF512)
- Sparse-BitNet: arXiv:2603.05168 (MSR; 1.30x, sparse tensor core; kod: github.com/AAzdi/Sparse-BitNet)
- BITCOS: arXiv:2609.16338 (Intel; nol 51.5% gacha, 1.485 bpw, AVX2/AVX-512, 1.28x)
- spbitnet: github.com/Artemarius/spbitnet (CC 8.0+ talab — bizga yaramaydi)
- bitnet.cpp: github.com/microsoft/BitNet (b9918; GPU kernel = Triton/PyTorch yo'li)
- llama.cpp mainline (klon: /tmp/opencode/llama.cpp-main): Vulkan TQ2_0 shader bor, CUDA'da yo'q
