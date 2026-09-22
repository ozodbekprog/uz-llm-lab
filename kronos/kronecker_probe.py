#!/usr/bin/env python3
"""KRONOS BOSQICH 1 - Kronecker (NKK) decomposition probe (haqiqiy F16 GGUF qatlamlarida).

Model:  W (m x n) ~ sum_{k=1..r} A_k (x) B_k ,   A_k: m1 x m2,  B_k: n1 x n2
        W = np.kron(A, B)  <=>  m = m1*n1, n = m2*n2  (Van Loan-Pitsianis)

Shuffle (R operator):
    R(W)[(i1,i2),(j1,j2)] = W[i1*n1 + j1, i2*n2 + j2] = A[i1,i2] * B[j1,j2]
    R = W.reshape(m1, n1, m2, n2).transpose(0, 2, 1, 3).reshape(m1*m2, n1*n2)

Xato (Frobenius, eng yaxshi r ta had):
    err_kron(r) = sqrt(sum_{k>r} s_k(R)^2) / ||W||_F
SVD-baseline (teng parametr budjetida):
    err_svd(p)  = sqrt(sum_{k>p} s_k(W)^2) / ||W||_F
Parametr soni (bitta had): kron: m1*m2 + n1*n2 ; svd: m + n

Ishga tushirish:
    .venv/bin/python kronos/kronecker_probe.py \
        --model models/gguf/qwen05b-ot-f16.gguf \
        --tensors blk.0.attn_q.weight,blk.0.attn_output.weight,blk.0.ffn_gate.weight
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def load_tensor(path: str, name: str) -> np.ndarray:
    reader = GGUFReader(path)
    for t in reader.tensors:
        if t.name != name:
            continue
        ttype = int(t.tensor_type)
        dtype = {0: np.float32, 1: np.float16}.get(ttype)
        if dtype is None:
            raise ValueError(f"{name}: turi {ttype} (faqat F16/F32); "
                             "I2_S/IQ uchun avval dequantizatsiya kerak")
        ne = [int(x) for x in t.shape]
        w = np.frombuffer(t.data, dtype=dtype).astype(np.float32)
        return w.reshape(ne[1], ne[0])
    raise KeyError(f"{name} topilmadi: {path}")


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def candidates(n: int, max_picks: int = 3) -> list[int]:
    base = math.isqrt(n)
    ds = divisors(n)
    picks: list[int] = []
    for k in (1.0, 0.5, 2.0):
        d = min(ds, key=lambda x: abs(x - base * k))
        if d not in picks:
            picks.append(d)
    return picks[:max_picks]


def kron_spectrum(W: np.ndarray, m1: int, m2: int) -> np.ndarray:
    m, n = W.shape
    if m % m1 or n % m2:
        raise ValueError("m % m1 == 0 va n % m2 == 0 bo'lishi shart")
    n1, n2 = m // m1, n // m2
    R = W.reshape(m1, n1, m2, n2).transpose(0, 2, 1, 3).reshape(m1 * m2, n1 * n2)
    return np.linalg.svd(R, compute_uv=False)


def kron_reconstruct(W: np.ndarray, m1: int, m2: int, r: int) -> np.ndarray:
    m, n = W.shape
    n1, n2 = m // m1, n // m2
    R = W.reshape(m1, n1, m2, n2).transpose(0, 2, 1, 3).reshape(m1 * m2, n1 * n2)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    Rh = (U[:, :r] * S[:r]) @ Vt[:r]
    return Rh.reshape(m1, m2, n1, n2).transpose(0, 2, 1, 3).reshape(m, n)


def err_from_spectrum(s: np.ndarray, r: int) -> float:
    tail = s[r:]
    if tail.size == 0:
        return 0.0
    return float(np.sqrt(np.sum(tail ** 2)) / np.sqrt(np.sum(s ** 2)))


def self_test() -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 8))
    B = rng.standard_normal((6, 5))
    W = np.kron(A, B)
    s = kron_spectrum(W, 4, 8)
    err1 = err_from_spectrum(s, 1)
    if err1 > 1e-5:
        raise AssertionError(f"Van Loan konvensiyasi xato: err1={err1}")
    W2 = rng.standard_normal((64, 96))
    s2 = kron_spectrum(W2, 8, 12)
    full = min(8 * 12, (64 // 8) * (96 // 12))
    e_full = err_from_spectrum(s2, full)
    e_rec = float(np.linalg.norm(W2 - kron_reconstruct(W2, 8, 12, full))
                  / np.linalg.norm(W2))
    if e_full > 1e-6 or e_rec > 1e-5:
        raise AssertionError(f"rekonstruksiya xato: full={e_full} rec={e_rec}")
    print(f"[self-test] A(x)B konvensiya OK (err1={err1:.2e}); "
          f"64x96 full-rank rec={e_rec:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensors", required=True)
    ap.add_argument("--ranks", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    self_test()
    ranks = [int(x) for x in args.ranks.split(",")]
    out: dict = {"model": args.model, "tensors": {}}

    for name in args.tensors.split(","):
        name = name.strip()
        W = load_tensor(args.model, name)
        m, n = W.shape
        fro = np.linalg.norm(W)
        s_svd = np.linalg.svd(W, compute_uv=False)
        splits = [(m1, m2) for m1 in candidates(m) for m2 in candidates(n)]
        spectra = {sp: kron_spectrum(W, *sp) for sp in splits}

        print(f"\n=== {name}  W={m}x{n}  (out x in) ===")
        hdr = f"{'r':>4s} {'split':>8s} {'param':>8s} {'siqish':>8s} " \
              f"{'kron_err':>9s} {'svd@r':>6s} {'svd_err':>9s}"
        print(hdr)
        print("-" * len(hdr))
        rec = {"shape": [m, n], "kron": {}, "svd": {}}
        for r in ranks:
            best_sp, best_err, best_par = None, float("inf"), None
            for sp, s in spectra.items():
                if r > min(sp[0] * sp[1], (m // sp[0]) * (n // sp[1])):
                    continue
                e = err_from_spectrum(s, r)
                par = r * (sp[0] * sp[1] + (m // sp[0]) * (n // sp[1]))
                if e < best_err:
                    best_sp, best_err, best_par = sp, e, par
            if best_sp is None:
                continue
            n1, n2 = m // best_sp[0], n // best_sp[1]
            comp = (m * n) / best_par
            r_svd = max(1, round(best_par / (m + n)))
            e_svd = err_from_spectrum(s_svd, min(r_svd, min(m, n)))
            rec["kron"][r] = {"split": f"{best_sp[0]}x{best_sp[1]}",
                              "params": best_par, "err": round(best_err, 5),
                              "compression": round(comp, 1)}
            rec["svd"][r_svd] = round(e_svd, 5)
            print(f"{r:>4d} {f'{best_sp[0]}x{best_sp[1]}':>8s} {best_par:>8d} "
                  f"{comp:>7.1f}x {best_err:>9.4f} {r_svd:>6d} {e_svd:>9.4f}")
        out["tensors"][name] = rec

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nJSON: {args.json}")


if __name__ == "__main__":
    main()
