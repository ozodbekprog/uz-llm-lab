#!/usr/bin/env python
"""MATRIX-DTO, BOSQICH 1: environment audit.

Checks the machine against the MATRIX-DTO requirements:

* Python / PyTorch / CUDA and the physical GPU (VRAM, compute capability,
  fp16/bf16 support, measured matmul throughput and memory bandwidth);
* build toolchain (nvcc, gcc, cmake, git) needed for bitnet.cpp;
* Python package inventory with required/optional flags;
* RAM, disk and the RAPL energy counter used by ``eval/energy_bench.py``.

The report is printed as a table and stored in ``logs/env_report.json``.

Usage:
    .venv/bin/python scripts/check_env.py
    .venv/bin/python scripts/check_env.py --json logs/env_report.json --no-bench
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

REQUIRED_PACKAGES: Dict[str, str] = {
    "torch": ">=2.7",
    "transformers": ">=4.45",
    "accelerate": ">=1.0",
    "numpy": ">=1.26",
    "scipy": ">=1.13",
    "ot": "POT >=0.9",
    "loguru": ">=0.7",
    "yaml": "PyYAML >=6",
    "pytest": ">=8.0",
}
OPTIONAL_PACKAGES: Dict[str, str] = {
    "peft": "LoRA fine-tuning",
    "datasets": "dataset tooling",
    "bitsandbytes": "4-bit QLoRA",
    "kfac": "K-FAC Fisher (BOSQICH 4)",
    "skopt": "Bayesian search (BOSQICH 5)",
    "wandb": "experiment tracking",
    "matplotlib": "plots",
    "pandas": "tables",
    "sympy": "symbolic checks",
}


@dataclass
class EnvReport:
    """Structured environment report."""

    python: str = ""
    executable: str = ""
    platform: str = ""
    cpu: str = ""
    cpu_count: int = 0
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    torch: str = ""
    torch_cuda: Optional[str] = None
    cuda_available: bool = False
    cudnn: Optional[str] = None
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_capability: str = ""
    fp16_supported: bool = False
    bf16_supported: bool = False
    matmul_gflops: Dict[str, float] = field(default_factory=dict)
    memcpy_gbps: float = 0.0
    tools: Dict[str, str] = field(default_factory=dict)
    packages: Dict[str, str] = field(default_factory=dict)
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        return "MISSING"
    try:
        out = subprocess.run(
            [name, "--version"], capture_output=True, text=True, timeout=10
        ).stdout.splitlines()
        return out[0].strip() if out else path
    except Exception:
        return path


def _ram_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return 0.0


def _matmul_gflops(dtype: torch.dtype, n: int = 2048, iters: int = 20) -> float:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return 2.0 * n**3 * iters / elapsed / 1e9


def _memcpy_gbps(size_mb: int = 128, iters: int = 20) -> float:
    n = size_mb * 1024**2 // 4
    a = torch.empty(n, device="cuda", dtype=torch.float32)
    b = torch.empty(n, device="cuda", dtype=torch.float32)
    for _ in range(3):
        a.copy_(b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a.copy_(b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return 2.0 * n * 4 * iters / elapsed / 1e9


def _rapl() -> str:
    path = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    if not path.exists():
        return "MISSING"
    try:
        with open(path, encoding="utf-8") as fh:
            fh.read()
        return "readable"
    except PermissionError:
        return "permission denied (sudo chmod a+r)"


def collect(bench: bool = True) -> EnvReport:
    """Collect the environment report."""
    rep = EnvReport(
        python=sys.version.split()[0],
        executable=sys.executable,
        platform=f"{platform.system()} {platform.release()}",
        cpu=platform.processor() or "-",
        cpu_count=torch.get_num_threads(),
        ram_gb=round(_ram_gb(), 1),
        disk_free_gb=round(shutil.disk_usage(ROOT).free / 1024**3, 1),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        tools={
            "nvcc": _tool("nvcc"),
            "gcc": _tool("gcc"),
            "cmake": _tool("cmake"),
            "git": _tool("git"),
            "rapl": _rapl(),
        },
    )
    for name, spec in REQUIRED_PACKAGES.items():
        found = importlib.util.find_spec(name) is not None
        rep.packages[name] = f"OK ({spec})" if found else f"MISSING ({spec})"
        if not found:
            rep.missing_required.append(name)
    for name, purpose in OPTIONAL_PACKAGES.items():
        found = importlib.util.find_spec(name) is not None
        rep.packages[name] = f"OK ({purpose})" if found else f"missing ({purpose})"
        if not found:
            rep.missing_optional.append(name)
    if rep.cuda_available:
        props = torch.cuda.get_device_properties(0)
        rep.gpu_name = props.name
        rep.gpu_vram_gb = round(props.total_memory / 1024**3, 2)
        rep.gpu_capability = f"sm_{props.major}{props.minor}"
        rep.fp16_supported = True
        rep.bf16_supported = bool(torch.cuda.is_bf16_supported())
        if bench:
            rep.matmul_gflops["fp32"] = round(_matmul_gflops(torch.float32), 1)
            rep.matmul_gflops["fp16"] = round(_matmul_gflops(torch.float16), 1)
            rep.memcpy_gbps = round(_memcpy_gbps(), 1)
    else:
        rep.warnings.append("CUDA is not available: training/inference will run on CPU")
    if rep.gpu_vram_gb and rep.gpu_vram_gb < 5:
        rep.warnings.append(
            f"GPU has {rep.gpu_vram_gb} GB VRAM: 1B+ models need gradient checkpointing "
            "or CPU offload (MATRIX-DTO ternary path fits, fp16 Adam does not)"
        )
    if rep.tools["nvcc"] == "MISSING":
        rep.warnings.append(
            "nvcc missing: fine for the PyTorch pipeline, required to rebuild bitnet.cpp"
        )
    try:
        import os

        page = os.sysconf("SC_PAGE_SIZE")
        phys = os.sysconf("SC_PHYS_PAGES")
        rep.ram_gb = round(page * phys / 1024**3, 1)
    except (ValueError, OSError):
        pass
    return rep


def print_report(rep: EnvReport) -> None:
    """Pretty-print the report."""
    def row(k: str, v: object) -> None:
        print(f"  {k:<28} {v}")

    print("=" * 72)
    print("MATRIX-DTO | BOSQICH 1: muhit hisoboti")
    print("=" * 72)
    print("[tizim]")
    row("Python", f"{rep.python}  ({rep.executable})")
    row("Platforma", f"{rep.platform} | CPU: {rep.cpu} x{rep.cpu_count}")
    row("RAM", f"{rep.ram_gb} GB")
    row("Disk (bo'sh)", f"{rep.disk_free_gb} GB")
    print("[torch / GPU]")
    row("PyTorch", f"{rep.torch} (CUDA {rep.torch_cuda}, cuDNN {rep.cudnn})")
    if rep.cuda_available:
        row("GPU", f"{rep.gpu_name} | {rep.gpu_vram_gb} GB | {rep.gpu_capability}")
        row("fp16 / bf16", f"{rep.fp16_supported} / {rep.bf16_supported}")
        if rep.matmul_gflops:
            row("matmul GFLOPS", ", ".join(f"{k}={v}" for k, v in rep.matmul_gflops.items()))
            row("memcpy", f"{rep.memcpy_gbps} GB/s")
    else:
        row("GPU", "yo'q (CPU rejimi)")
    print("[asboblar]")
    for k, v in rep.tools.items():
        row(k, v)
    print("[paketlar]")
    for k, v in rep.packages.items():
        row(k, v)
    if rep.warnings:
        print("[ogohlantirishlar]")
        for w in rep.warnings:
            row("-", w)
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="MATRIX-DTO environment audit")
    ap.add_argument("--json", default="logs/env_report.json", help="where to store the JSON report")
    ap.add_argument("--no-bench", action="store_true", help="skip GPU benchmarks")
    args = ap.parse_args()
    rep = collect(bench=not args.no_bench)
    print_report(rep)
    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")
    print(f"JSON saqlandi: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
