"""Tests for :mod:`eval.sprt_benchmark` (MATRIX-DTO layer 4)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.sprt_benchmark import (
    SPRTConfig,
    analyze_results_file,
    boundaries,
    fixed_n_required,
    run_sprt,
    simulate_sprt,
)


def test_boundaries_math() -> None:
    cfg = SPRTConfig(alpha=0.05, beta=0.05)
    a, b = boundaries(cfg)
    assert math.isclose(a, math.log(0.95 / 0.05))
    assert math.isclose(b, math.log(0.05 / 0.95))
    assert a > 0 > b


def test_all_correct_accepts_h1_quickly() -> None:
    cfg = SPRTConfig(p0=0.25, p1=0.5, alpha=0.05, beta=0.05, max_n=200)
    res = run_sprt([True] * 100, cfg)
    assert res.decision == "accept_h1"
    assert res.n <= 15
    assert res.trace[-1] >= res.boundary_a


def test_all_wrong_accepts_h0_quickly() -> None:
    cfg = SPRTConfig(p0=0.25, p1=0.5, alpha=0.05, beta=0.05, max_n=200)
    res = run_sprt([False] * 100, cfg)
    assert res.decision == "accept_h0"
    assert res.n <= 20


def test_inconclusive_when_capped() -> None:
    cfg = SPRTConfig(p0=0.25, p1=0.5, alpha=0.01, beta=0.01, max_n=3)
    res = run_sprt([True, False, True], cfg)
    assert res.decision == "inconclusive"
    assert res.n == 3


def test_sprt_saves_samples_on_average() -> None:
    cfg = SPRTConfig(p0=0.25, p1=0.5, alpha=0.05, beta=0.05, max_n=400)
    sim = simulate_sprt(0.8, cfg, n_sim=400, seed=1)
    fixed = sim["fixed_n"]
    assert fixed is not None and fixed > 0
    assert sim["mean_n"] < fixed
    assert sim["decision_rates"]["accept_h1"] > 0.95


def test_type_one_error_is_controlled() -> None:
    cfg = SPRTConfig(p0=0.25, p1=0.5, alpha=0.05, beta=0.05, max_n=400)
    sim = simulate_sprt(0.25, cfg, n_sim=800, seed=2)
    assert sim["decision_rates"]["accept_h1"] <= 0.08


def test_simulation_is_reproducible() -> None:
    cfg = SPRTConfig()
    a = simulate_sprt(0.5, cfg, n_sim=100, seed=7)
    b = simulate_sprt(0.5, cfg, n_sim=100, seed=7)
    assert a == b


def test_analyze_results_file(tmp_path: Path) -> None:
    payload = {
        "summary": {"model": "dummy"},
        "results": [{"ok": i < 20} for i in range(30)],
    }
    path = tmp_path / "res.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    out = analyze_results_file(path, SPRTConfig(p0=0.25, p1=0.5, max_n=30))
    assert out["n_total"] == 30
    assert math.isclose(out["accuracy"], 20 / 30, rel_tol=1e-9)
    assert out["sprt_decision"] == "accept_h1"
    assert 0 <= out["sample_saving"] <= 1


def test_fixed_n_reasonable() -> None:
    cfg = SPRTConfig(p0=0.25, p1=0.5, alpha=0.05, beta=0.05)
    n = fixed_n_required(cfg)
    assert n is not None and 10 <= n <= 60


def test_invalid_config() -> None:
    with pytest.raises(ValueError):
        SPRTConfig(p0=0.5, p1=0.5)
    with pytest.raises(ValueError):
        SPRTConfig(p0=0.0)
    with pytest.raises(ValueError):
        SPRTConfig(alpha=1.5)
