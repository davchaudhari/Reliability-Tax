"""Metrics: pass@k (unbiased), bootstrap CIs, reliability-per-dollar.

References
----------
pass@k unbiased estimator: Chen et al. 2021, "Evaluating Large Language Models Trained on Code"
(the HumanEval paper). For a task with n samples of which c are correct:

    pass@k = 1 - C(n - c, k) / C(n, k)

averaged over tasks. Computed in a numerically stable product form to avoid huge binomials.
This is unbiased for the probability that at least one of k i.i.d. samples is correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# pass@k
# ---------------------------------------------------------------------------
def pass_at_k_single(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for ONE task: n total samples, c correct, draw k.

    Stable product form (Chen et al. 2021):
        1 - prod_{i=n-c+1}^{n} (1 - k/i)  ... computed as 1 - C(n-c,k)/C(n,k).
    """
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n}) for an unbiased estimate.")
    if c <= 0:
        return 0.0
    if n - c < k:
        # Fewer than k incorrect samples remain -> any draw of k must include a correct one.
        return 1.0
    # 1 - C(n-c, k) / C(n, k), product form.
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def pass_at_k(num_samples: Sequence[int], num_correct: Sequence[int], k: int) -> float:
    """Mean unbiased pass@k over tasks. `num_samples[i]` and `num_correct[i]` per task."""
    vals = [
        pass_at_k_single(n, c, k) for n, c in zip(num_samples, num_correct) if n >= k
    ]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def pass_at_1(num_samples: Sequence[int], num_correct: Sequence[int]) -> float:
    """pass@1 = mean per-task success rate = mean(c/n)."""
    rates = [c / n for n, c in zip(num_samples, num_correct) if n > 0]
    return float(np.mean(rates)) if rates else float("nan")


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
@dataclass
class CI:
    point: float
    low: float
    high: float
    level: float = 0.95

    def to_dict(self) -> dict:
        return {"point": self.point, "low": self.low, "high": self.high, "level": self.level}


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
    statistic=np.mean,
) -> CI:
    """Percentile bootstrap CI for a statistic of `values` (default: the mean).

    Resamples tasks with replacement. For success rates, pass per-task 0/1 (or 0..1) values.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return CI(point=float("nan"), low=float("nan"), high=float("nan"), level=level)
    rng = np.random.default_rng(seed)
    point = float(statistic(arr))
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = statistic(arr[idx], axis=1)
    alpha = (1 - level) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])
    return CI(point=point, low=float(low), high=float(high), level=level)


# ---------------------------------------------------------------------------
# Reliability-per-dollar
# ---------------------------------------------------------------------------
@dataclass
class ReliabilityPerDollar:
    success_rate: float
    cost_usd: float
    n_tasks: int
    cost_view: str  # "measured_gpu" or "normalized_token"

    @property
    def successes(self) -> float:
        return self.success_rate * self.n_tasks

    @property
    def reliability_per_dollar(self) -> Optional[float]:
        """Successful tasks per dollar. None if cost is zero (mock / undefined)."""
        if self.cost_usd <= 0:
            return None
        return self.successes / self.cost_usd

    def to_dict(self) -> dict:
        return {
            "success_rate": self.success_rate,
            "cost_usd": self.cost_usd,
            "n_tasks": self.n_tasks,
            "cost_view": self.cost_view,
            "successes": self.successes,
            "reliability_per_dollar": self.reliability_per_dollar,
        }


def marginal_cost_per_extra_success(
    strategy_success_rate: float,
    strategy_cost: float,
    baseline_success_rate: float,
    baseline_cost: float,
    n_tasks: int,
) -> Optional[float]:
    """USD of additional cost per ADDITIONAL successful task vs. baseline.

    = (cost_strategy - cost_baseline) / (successes_strategy - successes_baseline).
    None if the strategy does not add successes (denominator <= 0) — meaning you paid more for
    no reliability gain, which we report as 'no marginal benefit' rather than a misleading number.
    """
    extra_successes = (strategy_success_rate - baseline_success_rate) * n_tasks
    extra_cost = strategy_cost - baseline_cost
    if extra_successes <= 0:
        return None
    return extra_cost / extra_successes


# ---------------------------------------------------------------------------
# Compounding-error curve (accuracy vs number of steps)
# ---------------------------------------------------------------------------
def compounding_curve(per_step_accuracy: float, max_steps: int) -> list[tuple[int, float]]:
    """Theoretical end-to-end success if each of s independent steps must succeed:
    acc(s) = per_step_accuracy ** s. The classic '99%/step -> ~60% after 50 steps' point.
    Returned for overlay against EMPIRICAL multi-step success measured from real runs."""
    return [(s, per_step_accuracy**s) for s in range(0, max_steps + 1)]
