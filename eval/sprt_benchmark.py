"""MATRIX-DTO layer 4: SPRT (Sequential Probability Ratio Test) benchmarking.

Instead of a fixed sample count, Wald's SPRT decides after every question
whether the model's accuracy is at chance level (:math:`H_0: p = p_0`) or
above it (:math:`H_1: p = p_1`).  The log-likelihood ratio

.. math::

    \\Lambda_n = \\sum_{i=1}^{n} \\log \\frac{P(x_i \\mid p_1)}
                                            {P(x_i \\mid p_0)}
    = \\sum_i \\begin{cases}
        \\log \\frac{p_1}{p_0}, & x_i = 1 \\\\
        \\log \\frac{1-p_1}{1-p_0}, & x_i = 0
      \\end{cases}

is compared against Wald boundaries

.. math::

    A = \\log \\frac{1-\\beta}{\\alpha}, \\qquad
    B = \\log \\frac{\\beta}{1-\\alpha},

stopping with "accept :math:`H_1`" when :math:`\\Lambda_n \\ge A` and with
"accept :math:`H_0`" when :math:`\\Lambda_n \\le B`.  The test controls the
error rates at :math:`(\\alpha, \\beta)` while typically needing far fewer
samples than a fixed-:math:`n` design.

Reference:
    Wald, A. (1945). Sequential tests of statistical hypotheses.
    *The Annals of Mathematical Statistics*, 16(2), 117-186.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SPRTConfig",
    "SPRTResult",
    "boundaries",
    "fixed_n_required",
    "run_sprt",
    "simulate_sprt",
    "analyze_results_file",
]


@dataclass
class SPRTConfig:
    """Hypotheses and error rates of the sequential test.

    Attributes:
        p0: Accuracy under :math:`H_0` (typically chance level).
        p1: Accuracy under :math:`H_1` (interesting level).
        alpha: Type I error budget.
        beta: Type II error budget.
        max_n: Hard cap on the number of questions.
    """

    p0: float = 0.25
    p1: float = 0.50
    alpha: float = 0.05
    beta: float = 0.05
    max_n: int = 200

    def __post_init__(self) -> None:
        if not 0.0 < self.p0 < 1.0 or not 0.0 < self.p1 < 1.0:
            raise ValueError("p0 and p1 must lie in (0, 1)")
        if self.p0 == self.p1:
            raise ValueError("p0 and p1 must differ")
        if not 0.0 < self.alpha < 1.0 or not 0.0 < self.beta < 1.0:
            raise ValueError("alpha and beta must lie in (0, 1)")
        if self.max_n < 1:
            raise ValueError("max_n must be >= 1")


@dataclass
class SPRTResult:
    """Outcome of a sequential run.

    Attributes:
        n: Number of outcomes consumed.
        decision: ``"accept_h1"``, ``"accept_h0"`` or ``"inconclusive"``.
        llr: Final log-likelihood ratio.
        trace: Running LLR after every outcome.
        boundary_a: Upper boundary.
        boundary_b: Lower boundary.
    """

    n: int
    decision: str
    llr: float
    trace: List[float] = field(default_factory=list)
    boundary_a: float = 0.0
    boundary_b: float = 0.0


def boundaries(cfg: SPRTConfig) -> Tuple[float, float]:
    """Return Wald boundaries ``(A, B)``."""
    a = math.log((1.0 - cfg.beta) / cfg.alpha)
    b = math.log(cfg.beta / (1.0 - cfg.alpha))
    return a, b


def _step(correct: bool, cfg: SPRTConfig) -> float:
    if correct:
        return math.log(cfg.p1 / cfg.p0)
    return math.log((1.0 - cfg.p1) / (1.0 - cfg.p0))


def run_sprt(outcomes: Iterable[bool], cfg: Optional[SPRTConfig] = None) -> SPRTResult:
    """Run the sequential test over a stream of binary outcomes.

    Args:
        outcomes: Iterable of booleans (True = correct answer).
        cfg: Test configuration.

    Returns:
        :class:`SPRTResult` with the stopping point and decision.
    """
    cfg = cfg or SPRTConfig()
    a, b = boundaries(cfg)
    llr = 0.0
    trace: List[float] = []
    decision = "inconclusive"
    n = 0
    for outcome in outcomes:
        llr += _step(bool(outcome), cfg)
        trace.append(llr)
        n += 1
        if llr >= a:
            decision = "accept_h1"
            break
        if llr <= b:
            decision = "accept_h0"
            break
        if n >= cfg.max_n:
            break
    return SPRTResult(n=n, decision=decision, llr=llr, trace=trace, boundary_a=a, boundary_b=b)


def _binomial_pmf(n: int, p: float) -> List[float]:
    """Stable binomial pmf via recurrence anchored at the mode."""
    mode = min(n, max(0, int((n + 1) * p)))
    pmf = [0.0] * (n + 1)
    pmf[mode] = 1.0
    ratio_up = p / (1.0 - p)
    for k in range(mode, n):
        pmf[k + 1] = pmf[k] * (n - k) / (k + 1) * ratio_up
    ratio_down = (1.0 - p) / p
    for k in range(mode, 0, -1):
        pmf[k - 1] = pmf[k] * k / (n - k + 1) * ratio_down
    total = sum(pmf)
    return [v / total for v in pmf]


def _tail_from(pmf: List[float], k: int) -> float:
    """``P(X >= k)`` from a pmf list."""
    if k <= 0:
        return 1.0
    if k >= len(pmf):
        return 0.0
    return sum(pmf[k:])


def fixed_n_required(cfg: Optional[SPRTConfig] = None, max_n: int = 400) -> Optional[int]:
    """Smallest fixed sample size reaching the same power as the SPRT.

    Chooses the smallest ``n`` for which a one-sided binomial test at
    level ``alpha`` has power at least ``1 - beta`` when the true accuracy
    is ``p1`` (chance ``p0``).

    Returns:
        Required ``n`` or None if it exceeds ``max_n``.
    """
    cfg = cfg or SPRTConfig()
    for n in range(1, max_n + 1):
        pmf0 = _binomial_pmf(n, cfg.p0)
        k = None
        for crit in range(n, -1, -1):
            if _tail_from(pmf0, crit) > cfg.alpha:
                k = crit + 1
                break
        if k is None or k > n:
            continue
        pmf1 = _binomial_pmf(n, cfg.p1)
        if _tail_from(pmf1, k) >= 1.0 - cfg.beta:
            return n
    return None


def simulate_sprt(
    true_acc: float,
    cfg: Optional[SPRTConfig] = None,
    n_sim: int = 2000,
    seed: int = 0,
    max_n: Optional[int] = None,
) -> dict:
    """Monte-Carlo estimate of the SPRT sample size and decision mix.

    Args:
        true_acc: Ground-truth accuracy used to generate outcomes.
        cfg: Test configuration.
        n_sim: Number of simulated benchmark runs.
        seed: RNG seed for reproducibility.
        max_n: Optional cap override (defaults to ``cfg.max_n``).

    Returns:
        Dict with ``mean_n``, ``median_n``, ``decision_rates`` and the
        fixed-``n`` requirement for reference.
    """
    import random

    cfg = cfg or SPRTConfig()
    rng = random.Random(seed)
    sizes: List[int] = []
    counts = {"accept_h1": 0, "accept_h0": 0, "inconclusive": 0}
    cap = max_n or cfg.max_n
    for _ in range(n_sim):
        outcomes = [rng.random() < true_acc for _ in range(cap)]
        res = run_sprt(outcomes, cfg)
        sizes.append(res.n)
        counts[res.decision] += 1
    sizes_sorted = sorted(sizes)
    return {
        "true_acc": true_acc,
        "mean_n": sum(sizes) / len(sizes),
        "median_n": sizes_sorted[len(sizes_sorted) // 2],
        "p90_n": sizes_sorted[int(0.9 * (len(sizes_sorted) - 1))],
        "decision_rates": {k: v / n_sim for k, v in counts.items()},
        "fixed_n": fixed_n_required(cfg),
    }


def analyze_results_file(path: Path, cfg: Optional[SPRTConfig] = None) -> dict:
    """Run SPRT on the per-question outcomes of a saved benchmark file.

    Args:
        path: JSON produced by ``eval/run_bench.py`` (must contain a
            ``results`` list with an ``ok`` flag per question).
        cfg: Test configuration.

    Returns:
        Dict with the model name, raw accuracy, SPRT decision and the
        number of questions the SPRT consumed versus the full set.
    """
    cfg = cfg or SPRTConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    results = payload.get("results", [])
    outcomes = [bool(r.get("ok")) for r in results]
    res = run_sprt(outcomes, cfg)
    n_total = len(outcomes)
    return {
        "file": str(path),
        "model": payload.get("summary", {}).get("model", "-"),
        "n_total": n_total,
        "accuracy": (sum(outcomes) / n_total) if n_total else float("nan"),
        "sprt_n": res.n,
        "sprt_decision": res.decision,
        "sample_saving": 1.0 - res.n / n_total if n_total else float("nan"),
        "boundary_a": res.boundary_a,
        "boundary_b": res.boundary_b,
        "llr": res.llr,
    }


def _main() -> int:
    """CLI: analyse saved benchmark files and/or simulate SPRT savings."""
    import argparse

    ap = argparse.ArgumentParser(description="MATRIX-DTO SPRT benchmark analysis")
    ap.add_argument("--results", nargs="*", default=None, help="benchmark JSON files")
    ap.add_argument("--p0", type=float, default=0.25)
    ap.add_argument("--p1", type=float, default=0.50)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--max-n", type=int, default=200)
    ap.add_argument("--simulate", action="store_true", help="run Monte-Carlo simulations")
    args = ap.parse_args()
    cfg = SPRTConfig(p0=args.p0, p1=args.p1, alpha=args.alpha, beta=args.beta, max_n=args.max_n)

    if args.results:
        print(f"{'model':<42} {'n':>3} {'acc':>6} {'sprt_n':>6} {'decision':>12} {'tejash':>7}")
        for f in args.results:
            out = analyze_results_file(Path(f), cfg)
            print(
                f"{out['model'][-42:]:<42} {out['n_total']:>3} {out['accuracy']:>6.3f} "
                f"{out['sprt_n']:>6} {out['sprt_decision']:>12} {out['sample_saving']*100:>6.1f}%"
            )
    if args.simulate:
        fixed = fixed_n_required(cfg)
        print(f"\nFixed-n talab (p0={cfg.p0}, p1={cfg.p1}, alpha={cfg.alpha}, beta={cfg.beta}): {fixed}")
        print(f"{'true_acc':>8} {'mean_n':>7} {'median_n':>8} {'p90_n':>6} {'H1%':>6} {'H0%':>6}")
        for acc in (0.1, 0.25, 0.35, 0.5, 0.7, 0.9):
            sim = simulate_sprt(acc, cfg, n_sim=2000, seed=42)
            print(
                f"{acc:>8.2f} {sim['mean_n']:>7.1f} {sim['median_n']:>8} {sim['p90_n']:>6} "
                f"{sim['decision_rates']['accept_h1']*100:>5.1f} "
                f"{sim['decision_rates']['accept_h0']*100:>5.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
