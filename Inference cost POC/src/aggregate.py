"""Aggregate raw TaskResults into per-strategy metrics with CIs and cost views.

This is the bridge from raw run logs -> the numbers the plots and leaderboard consume. It groups
by strategy, computes success rate (with abstentions handled), pass@1/pass@k across seeds, both
cost views, reliability-per-dollar, and bootstrap CIs over tasks.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .cost import CostBreakdown, GpuPrice, TokenPrice, summarize_cost
from .instrument import TaskResult
from .metrics import (
    CI,
    ReliabilityPerDollar,
    bootstrap_ci,
    marginal_cost_per_extra_success,
    pass_at_1,
    pass_at_k,
)


@dataclass
class StrategySummary:
    strategy: str
    n_tasks: int
    n_seeds: int
    success_rate: float
    success_ci: CI
    abstention_rate: float
    pass_at_1: float
    cost: CostBreakdown
    rpd_measured: ReliabilityPerDollar
    rpd_normalized: ReliabilityPerDollar
    avg_calls_per_task: float
    avg_tokens_per_task: float
    pass_at_k_curve: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n_tasks": self.n_tasks,
            "n_seeds": self.n_seeds,
            "success_rate": self.success_rate,
            "success_ci": self.success_ci.to_dict(),
            "abstention_rate": self.abstention_rate,
            "pass_at_1": self.pass_at_1,
            "cost": self.cost.to_dict(),
            "reliability_per_dollar_measured": self.rpd_measured.to_dict(),
            "reliability_per_dollar_normalized": self.rpd_normalized.to_dict(),
            "avg_calls_per_task": self.avg_calls_per_task,
            "avg_tokens_per_task": self.avg_tokens_per_task,
            "pass_at_k_curve": self.pass_at_k_curve,
        }


def results_to_frame(results: list[TaskResult]) -> pd.DataFrame:
    """Flatten TaskResults to a tidy DataFrame (one row per (strategy, seed, task))."""
    rows = []
    for r in results:
        rows.append(
            {
                "strategy": r.strategy,
                "seed": r.seed,
                "task_id": r.task_id,
                "success": int(r.success),
                "abstained": int(r.abstained),
                "num_calls": r.num_calls,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "wall_clock_s": r.wall_clock_s,
            }
        )
    return pd.DataFrame(rows)


def _dict_results_to_frame(results: list[dict]) -> pd.DataFrame:
    """Same as results_to_frame but from saved JSON dicts (load_run output)."""
    rows = []
    for r in results:
        rows.append(
            {
                "strategy": r["strategy"],
                "seed": r["seed"],
                "task_id": r["task_id"],
                "success": int(r["success"]),
                "abstained": int(r["abstained"]),
                "num_calls": r["num_calls"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["total_tokens"],
                "wall_clock_s": r["wall_clock_s"],
            }
        )
    return pd.DataFrame(rows)


def summarize_strategy(
    results: list[TaskResult],
    strategy: str,
    *,
    gpu_price: GpuPrice,
    token_price: TokenPrice,
    k_values: Optional[list[int]] = None,
    boot_seed: int = 0,
) -> StrategySummary:
    """Compute the full metric bundle for one strategy across all its seeds/tasks."""
    sres = [r for r in results if r.strategy == strategy]
    if not sres:
        raise ValueError(f"No results for strategy '{strategy}'.")

    seeds = sorted({r.seed for r in sres})
    task_ids = sorted({r.task_id for r in sres})
    n_tasks = len(task_ids)
    n_seeds = len(seeds)

    # Success rate: fraction of all (task, seed) attempts that succeeded. Abstentions count as
    # non-success (already reflected in r.success=False).
    success_flags = np.array([int(r.success) for r in sres], dtype=float)
    success_rate = float(success_flags.mean())
    abstention_rate = float(np.mean([int(r.abstained) for r in sres]))

    # Bootstrap over TASKS (resample task-level mean success, averaged over seeds) for an honest
    # CI that respects task as the unit of analysis.
    per_task_mean = []
    by_task = defaultdict(list)
    for r in sres:
        by_task[r.task_id].append(int(r.success))
    for tid in task_ids:
        per_task_mean.append(float(np.mean(by_task[tid])))
    success_ci = bootstrap_ci(per_task_mean, seed=boot_seed)

    # pass@1 / pass@k: treat the `n_seeds` rollouts per task as the n samples.
    num_samples, num_correct = [], []
    for tid in task_ids:
        flags = by_task[tid]
        num_samples.append(len(flags))
        num_correct.append(int(sum(flags)))
    p_at_1 = pass_at_1(num_samples, num_correct)
    k_values = k_values or list(range(1, n_seeds + 1))
    pak = {k: pass_at_k(num_samples, num_correct, k) for k in k_values if k <= n_seeds}

    cost = summarize_cost(sres, gpu_price=gpu_price, token_price=token_price)

    # reliability-per-dollar uses the SAME success_rate but the two cost views.
    rpd_measured = ReliabilityPerDollar(
        success_rate=success_rate,
        cost_usd=cost.measured_gpu_usd,
        n_tasks=len(sres),
        cost_view="measured_gpu",
    )
    rpd_normalized = ReliabilityPerDollar(
        success_rate=success_rate,
        cost_usd=cost.normalized_token_usd,
        n_tasks=len(sres),
        cost_view="normalized_token",
    )

    avg_calls = float(np.mean([r.num_calls for r in sres]))
    avg_tokens = float(np.mean([r.total_tokens for r in sres]))

    return StrategySummary(
        strategy=strategy,
        n_tasks=n_tasks,
        n_seeds=n_seeds,
        success_rate=success_rate,
        success_ci=success_ci,
        abstention_rate=abstention_rate,
        pass_at_1=p_at_1,
        cost=cost,
        rpd_measured=rpd_measured,
        rpd_normalized=rpd_normalized,
        avg_calls_per_task=avg_calls,
        avg_tokens_per_task=avg_tokens,
        pass_at_k_curve=pak,
    )


def summarize_all(
    results: list[TaskResult],
    *,
    gpu_price: GpuPrice,
    token_price: TokenPrice,
    boot_seed: int = 0,
) -> list[StrategySummary]:
    strategies = []
    seen = set()
    for r in results:  # preserve first-seen order
        if r.strategy not in seen:
            seen.add(r.strategy)
            strategies.append(r.strategy)
    return [
        summarize_strategy(
            results, s, gpu_price=gpu_price, token_price=token_price, boot_seed=boot_seed
        )
        for s in strategies
    ]
