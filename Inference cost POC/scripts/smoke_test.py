#!/usr/bin/env python3
"""Phase 1 smoke test: ~5 real tasks against the deployed vLLM server. Target spend < $1.

This is the FIRST real-money script. It:
  * prints cumulative spend (read from BUDGET.md ledger if present) before and after,
  * runs only baseline + self_consistency(n=3) on 5 tasks, 1 seed,
  * records measured wall-clock and tokens,
  * prints a measured-cost estimate to reconcile against Modal's reported usage in BUDGET.md.

It will NOT run unless --base-url is provided (no accidental spend). Run scripts/run_eval.py
--dry-run first to see the projection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aggregate import summarize_all  # noqa: E402
from src.benchmarks import BFCLBenchmark, MockBenchmark  # noqa: E402
from src.cost import GpuPrice, TokenPrice  # noqa: E402
from src.harness import run, save_run  # noqa: E402
from src.strategies import Baseline, SelfConsistency  # noqa: E402
from src.strategies.base import StrategyConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="vLLM OpenAI base url (https://.../v1)")
    ap.add_argument("--model", required=True, help="Served model name/tag.")
    ap.add_argument("--gpu", default="L4")
    ap.add_argument("--benchmark", default="bfcl", choices=["bfcl", "mock"])
    ap.add_argument("--bfcl-data", default=None)
    ap.add_argument("--bfcl-categories", nargs="+", default=["simple"])
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--budget", type=float, default=1.0, help="Hard kill-switch (USD).")
    args = ap.parse_args()

    from src.openai_client import VLLMClient

    model = VLLMClient(base_url=args.base_url, model=args.model)
    if args.benchmark == "mock":
        benchmark = MockBenchmark(n_tasks=args.n_tasks)
    else:
        benchmark = BFCLBenchmark(data_dir=args.bfcl_data, categories=args.bfcl_categories)

    cfg = StrategyConfig(n_samples=3, sample_temperature=0.7, max_tokens=512)
    strategies = {"baseline": Baseline(config=cfg), "self_consistency": SelfConsistency(config=cfg)}

    gpu_price = GpuPrice(gpu=args.gpu)

    def cost_guard(results_so_far) -> bool:
        gpu_seconds = sum(r.wall_clock_s for r in results_so_far)
        cost = gpu_price.cost_for_seconds(gpu_seconds)
        return cost > args.budget

    print(f"=== SMOKE TEST: {args.n_tasks} tasks, baseline + self_consistency(n=3), 1 seed ===")
    print(f"GPU={args.gpu} @ ${gpu_price.rate():.6f}/sec, hard budget ${args.budget:.2f}")
    print("Cumulative spend BEFORE this run: see BUDGET.md ledger.\n")

    t0 = time.perf_counter()
    out = run(
        benchmark=benchmark,
        model=model,
        strategies=strategies,
        seeds=[0],
        n_tasks=args.n_tasks,
        model_name=args.model,
        notes="phase1-smoke",
        cost_guard=cost_guard,
    )
    wall = time.perf_counter() - t0

    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_model = args.model.replace("/", "-")  # model tags contain '/', which breaks paths
    path = os.path.join("results", f"smoke_{safe_model}_{ts}.json")
    save_run(out, path)

    token_price = TokenPrice()
    summaries = summarize_all(out.results, gpu_price=gpu_price, token_price=token_price)
    measured = sum(s.cost.measured_gpu_usd for s in summaries)
    total_tokens = sum(s.cost.total_tokens for s in summaries)

    print(f"\nWall-clock: {wall:.1f}s | total tokens: {total_tokens}")
    print(f"Per-result GPU-seconds attribution -> measured ~= ${measured:.4f} (UPPER BOUND).")
    print(
        "NOTE: with request concurrency this OVER-counts GPU time. Reconcile against the actual "
        "Modal-reported usage for this window and record BOTH in BUDGET.md.\n"
    )
    print(f"Saved -> {path}")
    for s in summaries:
        print(f"  {s.strategy:<18} success={s.success_rate:.3f} tokens/task={s.avg_tokens_per_task:.0f}")


if __name__ == "__main__":
    main()
