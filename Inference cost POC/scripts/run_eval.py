#!/usr/bin/env python3
"""Main entrypoint: run a strategy sweep over a benchmark and save logs + a summary.

Examples
--------
# $0 mock sweep, all strategies, 3 seeds, 40 tasks:
python scripts/run_eval.py --benchmark mock --model mock --strategies all --seeds 0 1 2 --n-tasks 40

# Real served model (after Modal deploy), with a hard budget kill-switch:
python scripts/run_eval.py --benchmark bfcl --bfcl-data /path/to/bfcl_eval/data \
    --model qwen2.5-1.5b --base-url https://...modal.run/v1 \
    --strategies baseline self_consistency --seeds 0 1 2 --n-tasks 40 \
    --gpu L4 --budget 5.0

The --dry-run flag prints a projected cost (tasks x strategies x seeds x est calls x est tokens)
and EXITS without calling any model. Always dry-run before a GPU phase.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Allow running from repo root without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aggregate import summarize_all  # noqa: E402
from src.benchmarks import BFCLBenchmark, MockBenchmark  # noqa: E402
from src.cost import GpuPrice, TokenPrice  # noqa: E402
from src.harness import run, save_run  # noqa: E402
from src.strategies import REGISTRY  # noqa: E402
from src.strategies.base import StrategyConfig  # noqa: E402


# Per-strategy expected model calls (for dry-run cost projection). Rough; documented as estimates.
def _expected_calls(strategy: str, cfg: StrategyConfig) -> float:
    return {
        "baseline": 1,
        "self_consistency": cfg.n_samples,
        "self_correct": 1 + 2 * cfg.max_iters,  # worst case: critique+revise each iter
        "abstain": 1,
        "verifier_rerank": 2 * cfg.n_samples,  # n candidates + n verifier calls
    }.get(strategy, 1)


def build_benchmark(args):
    if args.benchmark == "mock":
        return MockBenchmark(n_tasks=max(args.n_tasks + args.task_offset, args.n_tasks))
    if args.benchmark == "bfcl":
        return BFCLBenchmark(
            data_dir=args.bfcl_data,
            categories=args.bfcl_categories,
            limit=None,
        )
    raise ValueError(f"Unknown benchmark: {args.benchmark}")


def build_model(args):
    if args.model_kind == "mock":
        from src.mock_model import MockModel

        return MockModel()
    # real
    from src.openai_client import VLLMClient

    if not args.base_url:
        raise SystemExit("--base-url is required for a real model.")
    return VLLMClient(base_url=args.base_url, model=args.model)


def make_strategies(names: list[str], cfg: StrategyConfig) -> dict:
    if names == ["all"]:
        names = list(REGISTRY.keys())
    out = {}
    for n in names:
        if n not in REGISTRY:
            raise SystemExit(f"Unknown strategy '{n}'. Known: {sorted(REGISTRY)}")
        out[n] = REGISTRY[n](config=cfg)
    return out


def project_cost(args, strategies, cfg) -> dict:
    """Dry-run cost projection using the ALIVE-TIME model calibrated from the Phase 1 smoke test.

    Phase 1 taught us the token/throughput model is wrong for self-hosting: the real Modal bill is
    GPU *rental* time, which = cold start + serving wall-clock + idle-until-scaledown. So we model:

        serving_wall_s = total_calls * per_call_latency_s / max_workers   (concurrency batches)
        alive_s        = cold_start_s + serving_wall_s + idle_s
        measured_usd   = alive_s * gpu_$/s

    The normalized (reference-token) view is independent of all this — just tokens * price.
    Defaults are calibrated from Phase 1 (1.5B/L4: ~0.37 s/call sequential round-trip, ~150 s cold
    start). For the 7B, raise per_call_latency. All overridable from the CLI.
    """
    gpu_price = GpuPrice(gpu=args.gpu)
    token_price = TokenPrice(
        prompt_per_mtok=args.prompt_price, completion_per_mtok=args.completion_price
    )

    total_calls = 0.0
    for name in strategies:
        total_calls += _expected_calls(name, cfg) * args.n_tasks * len(args.seeds)
    total_tokens = total_calls * args.est_tokens_per_call

    workers = max(1, args.max_workers)
    # Concurrency speedup saturates at the server's throughput ceiling; cap effective parallelism.
    effective_workers = min(workers, args.max_effective_concurrency)
    serving_wall_s = total_calls * args.per_call_latency_s / effective_workers
    alive_s = args.cold_start_s + serving_wall_s + args.idle_s
    measured = gpu_price.cost_for_seconds(alive_s)
    normalized = token_price.cost(int(total_tokens * 0.6), int(total_tokens * 0.4))

    return {
        "total_model_calls": total_calls,
        "est_total_tokens": total_tokens,
        "max_workers": workers,
        "effective_concurrency": effective_workers,
        "est_serving_wall_s": round(serving_wall_s, 1),
        "est_cold_start_s": args.cold_start_s,
        "est_alive_s": round(alive_s, 1),
        "gpu": args.gpu,
        "gpu_usd_per_sec": gpu_price.rate(),
        "projected_measured_usd": round(measured, 4),
        "projected_normalized_usd": round(normalized, 4),
        "assumptions": {
            "per_call_latency_s": args.per_call_latency_s,
            "est_tokens_per_call": args.est_tokens_per_call,
            "cold_start_s": args.cold_start_s,
            "idle_s": args.idle_s,
            "max_effective_concurrency": args.max_effective_concurrency,
            "note": "Alive-time model calibrated from Phase 1. Reconcile vs Modal dashboard.",
        },
    }


def main():
    ap = argparse.ArgumentParser(description="reliability-tax eval runner")
    ap.add_argument("--benchmark", default="mock", choices=["mock", "bfcl"])
    ap.add_argument("--bfcl-data", default=None, help="Path to bfcl_eval/data dir (real BFCL).")
    ap.add_argument("--bfcl-categories", nargs="+", default=["simple"])
    ap.add_argument("--model", default="mock", help="Model name/tag for logging.")
    ap.add_argument(
        "--model-kind",
        default=None,
        choices=["mock", "real"],
        help="mock (default if --model=mock) or real (vLLM HTTP).",
    )
    ap.add_argument("--base-url", default=None, help="vLLM OpenAI base url, e.g. https://.../v1")
    ap.add_argument("--strategies", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument("--task-offset", type=int, default=0)
    # strategy knobs
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=2)
    ap.add_argument("--sample-temperature", type=float, default=0.7)
    ap.add_argument("--logprob-threshold", type=float, default=-0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    # cost / budget
    ap.add_argument("--gpu", default="L4")
    ap.add_argument("--budget", type=float, default=None, help="Hard USD kill-switch (measured).")
    ap.add_argument("--prompt-price", type=float, default=0.20, help="USD / 1M prompt tokens.")
    ap.add_argument("--completion-price", type=float, default=0.60, help="USD / 1M completion tok.")
    ap.add_argument("--est-tokens-per-call", type=float, default=350.0)
    ap.add_argument("--est-throughput", type=float, default=1500.0, help="(unused; legacy)")
    # alive-time projection knobs (calibrated from Phase 1)
    ap.add_argument("--per-call-latency-s", type=float, default=0.37, help="sec/call round-trip.")
    ap.add_argument("--cold-start-s", type=float, default=150.0, help="one-time cold start sec.")
    ap.add_argument("--idle-s", type=float, default=60.0, help="idle-until-scaledown sec.")
    ap.add_argument("--max-effective-concurrency", type=int, default=16,
                    help="cap on useful parallelism (server throughput ceiling).")
    # serving concurrency (real runs): batch requests so vLLM keeps the GPU busy
    ap.add_argument("--max-workers", type=int, default=1,
                    help="concurrent in-flight requests. >1 for real runs only (mock stays 1).")
    ap.add_argument("--dry-run", action="store_true", help="Print cost projection and exit.")
    ap.add_argument("--out", default=None, help="Output JSON path (default: results/<auto>.json)")
    args = ap.parse_args()

    if args.model_kind is None:
        args.model_kind = "mock" if args.model == "mock" else "real"

    cfg = StrategyConfig(
        temperature=0.0,
        max_tokens=args.max_tokens,
        n_samples=args.n_samples,
        max_iters=args.max_iters,
        sample_temperature=args.sample_temperature,
        logprob_threshold=args.logprob_threshold,
    )
    strategy_names = list(REGISTRY.keys()) if args.strategies == ["all"] else args.strategies
    strategies = make_strategies(strategy_names, cfg)

    # ---- dry-run cost projection ----
    projection = project_cost(args, list(strategies.keys()), cfg)
    print("\n=== COST PROJECTION (planning estimate) ===")
    print(json.dumps(projection, indent=2))
    if args.budget is not None and projection["projected_measured_usd"] > args.budget:
        print(
            f"\n!! Projected measured cost ${projection['projected_measured_usd']:.2f} exceeds "
            f"--budget ${args.budget:.2f}. Refusing to start. Shrink tasks/seeds/strategies or "
            f"use a smaller model.\n"
        )
        if not args.dry_run:
            sys.exit(2)
    if args.dry_run:
        print("\n(dry-run) Not calling any model. Exiting.\n")
        return

    if args.model_kind == "real":
        print(
            "\n*** This will spend REAL money on Modal GPU time. "
            "Confirm the projection above before proceeding. ***\n"
        )

    model = build_model(args)
    benchmark = build_benchmark(args)

    # Budget kill-switch as a cost_guard over measured cost so far.
    gpu_price = GpuPrice(gpu=args.gpu)
    budget = args.budget

    def cost_guard(results_so_far) -> bool:
        if budget is None:
            return False
        gpu_seconds = sum(r.wall_clock_s for r in results_so_far)
        return gpu_price.cost_for_seconds(gpu_seconds) > budget

    out = run(
        benchmark=benchmark,
        model=model,
        strategies=strategies,
        seeds=args.seeds,
        n_tasks=args.n_tasks,
        task_offset=args.task_offset,
        model_name=args.model,
        notes=f"strategies={strategy_names} gpu={args.gpu} workers={args.max_workers}",
        cost_guard=cost_guard,
        max_workers=args.max_workers,
    )

    # Save raw run.
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_model = args.model.replace("/", "-")  # model tags contain '/', which breaks paths
    out_path = args.out or os.path.join("results", f"run_{args.benchmark}_{safe_model}_{ts}.json")
    save_run(out, out_path)
    print(f"\nSaved raw run -> {out_path}")

    # Summarize.
    token_price = TokenPrice(
        prompt_per_mtok=args.prompt_price,
        completion_per_mtok=args.completion_price,
        label=f"ref({args.prompt_price}/{args.completion_price} per Mtok)",
    )
    summaries = summarize_all(out.results, gpu_price=gpu_price, token_price=token_price)
    summary_path = out_path.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in summaries], f, indent=2)
    print(f"Saved summary  -> {summary_path}\n")

    # Print a compact table.
    print(f"{'strategy':<18} {'success':>8} {'95% CI':>16} {'calls/task':>11} {'tok/task':>10}")
    for s in summaries:
        ci = f"[{s.success_ci.low:.2f},{s.success_ci.high:.2f}]"
        print(
            f"{s.strategy:<18} {s.success_rate:>8.3f} {ci:>16} "
            f"{s.avg_calls_per_task:>11.1f} {s.avg_tokens_per_task:>10.0f}"
        )


if __name__ == "__main__":
    main()
