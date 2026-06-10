#!/usr/bin/env python3
"""Phase 3 headline: run reliability POLICIES over tau-bench (multi-turn, tool-use).

Each (task, policy, seed) is an independent conversation episode; episodes run concurrently so the
self-hosted vLLM server batches the agent + user-simulator requests. Success = tau-bench reward
(DB-state match + required outputs). Cost is every recorded agent and user-sim call.

ALWAYS --dry-run first to print the alive-time cost projection. The real run spends Modal GPU time.

Example (after deploying the 7B with tool-use-capable settings):
  python scripts/run_taubench.py --base-url https://.../v1 --model Qwen/Qwen2.5-7B-Instruct \
     --env airline --n-tasks 12 --seeds 0 1 2 --policies baseline self_consistency self_correct \
     --max-workers 12 --gpu A10G --budget 3 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agentic.policy import PolicyConfig  # noqa: E402
from src.agentic.policies import MT_REGISTRY  # noqa: E402
from src.agentic.taubench import TauBenchConfig, run_episode  # noqa: E402
from src.aggregate import summarize_all  # noqa: E402
from src.cost import GpuPrice, TokenPrice  # noqa: E402
from src.harness import save_run, RunConfig, RunOutput  # noqa: E402


def _calls_per_turn(policy: str, cfg: PolicyConfig) -> float:
    return {
        "baseline": 1,
        "self_consistency": cfg.n_samples,
        "self_correct": 1 + 2 * cfg.max_iters,
    }.get(policy, 1)


def project_cost(args, policies, cfg) -> dict:
    """Alive-time projection (calibrated from Phases 1-2), adapted to the multi-turn call pattern:
    per episode ≈ est_turns × agent_calls_per_turn + est_user_sim_calls (one per respond turn)."""
    gpu_price = GpuPrice(gpu=args.gpu)
    token_price = TokenPrice(
        prompt_per_mtok=args.prompt_price, completion_per_mtok=args.completion_price
    )
    total_calls = 0.0
    for p in policies:
        agent_calls = args.est_turns * _calls_per_turn(p, cfg)
        user_calls = args.est_user_turns
        total_calls += (agent_calls + user_calls) * args.n_tasks * len(args.seeds)
    total_tokens = total_calls * args.est_tokens_per_call
    workers = max(1, args.max_workers)
    eff = min(workers, args.max_effective_concurrency)
    serving_wall_s = total_calls * args.per_call_latency_s / eff
    alive_s = args.cold_start_s + serving_wall_s + args.idle_s
    measured = gpu_price.cost_for_seconds(alive_s)
    normalized = token_price.cost(int(total_tokens * 0.7), int(total_tokens * 0.3))
    return {
        "policies": policies,
        "n_tasks": args.n_tasks,
        "seeds": args.seeds,
        "est_total_model_calls": round(total_calls),
        "est_total_tokens": round(total_tokens),
        "max_workers": workers,
        "est_serving_wall_s": round(serving_wall_s, 1),
        "est_cold_start_s": args.cold_start_s,
        "est_alive_s": round(alive_s, 1),
        "gpu": args.gpu,
        "projected_measured_usd": round(measured, 4),
        "projected_normalized_usd": round(normalized, 4),
        "assumptions": {
            "est_turns": args.est_turns,
            "est_user_turns": args.est_user_turns,
            "per_call_latency_s": args.per_call_latency_s,
            "est_tokens_per_call": args.est_tokens_per_call,
            "cold_start_s": args.cold_start_s,
            "note": "Multi-turn alive-time estimate. 7B prompts are large (wiki+tools+history).",
        },
    }


def main():
    ap = argparse.ArgumentParser(description="tau-bench reliability-policy runner")
    ap.add_argument("--base-url", default=None, help="vLLM OpenAI base url (real run).")
    ap.add_argument("--model", default="mock", help="agent model tag.")
    ap.add_argument("--user-model", default=None, help="user-sim model tag (default: --model).")
    ap.add_argument("--env", default="airline", choices=["airline", "retail"])
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--task-offset", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--policies", nargs="+", default=["baseline", "self_consistency", "self_correct"])
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=1)
    ap.add_argument("--sample-temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-workers", type=int, default=12)
    # cost / budget
    ap.add_argument("--gpu", default="A10G")
    ap.add_argument("--budget", type=float, default=None, help="hard USD kill-switch (measured).")
    ap.add_argument("--prompt-price", type=float, default=0.20)
    ap.add_argument("--completion-price", type=float, default=0.60)
    # projection knobs (multi-turn)
    ap.add_argument("--est-turns", type=float, default=12.0, help="agent turns/episode.")
    ap.add_argument("--est-user-turns", type=float, default=6.0, help="user-sim calls/episode.")
    ap.add_argument("--per-call-latency-s", type=float, default=1.2, help="7B sec/call.")
    ap.add_argument("--est-tokens-per-call", type=float, default=2500.0, help="big prompts.")
    ap.add_argument("--cold-start-s", type=float, default=240.0, help="7B cold start.")
    ap.add_argument("--idle-s", type=float, default=60.0)
    ap.add_argument("--max-effective-concurrency", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = PolicyConfig(
        max_tokens=args.max_tokens, n_samples=args.n_samples, max_iters=args.max_iters,
        sample_temperature=args.sample_temperature,
    )
    for p in args.policies:
        if p not in MT_REGISTRY:
            raise SystemExit(f"Unknown policy '{p}'. Known: {sorted(MT_REGISTRY)}")

    projection = project_cost(args, args.policies, cfg)
    print("\n=== TAU-BENCH COST PROJECTION (planning estimate) ===")
    print(json.dumps(projection, indent=2))
    if args.budget is not None and projection["projected_measured_usd"] > args.budget:
        print(f"\n!! Projected ${projection['projected_measured_usd']} exceeds --budget "
              f"${args.budget}. Refusing to start. Shrink tasks/seeds/policies.\n")
        if not args.dry_run:
            sys.exit(2)
    if args.dry_run:
        print("\n(dry-run) Not calling any model. Exiting.\n")
        return

    if args.model == "mock" or not args.base_url:
        raise SystemExit("Real run needs --base-url and a served --model.")

    print("\n*** This spends REAL Modal GPU time. Confirm the projection above. ***\n")
    from src.openai_client import VLLMClient

    agent_model = VLLMClient(base_url=args.base_url, model=args.model, timeout_s=180)
    user_tag = args.user_model or args.model
    user_model = VLLMClient(base_url=args.base_url, model=user_tag, timeout_s=180)

    tcfg = TauBenchConfig(env_name=args.env, max_steps=args.max_steps)
    policies = {p: MT_REGISTRY[p](config=cfg) for p in args.policies}
    task_indices = list(range(args.task_offset, args.task_offset + args.n_tasks))

    # Episodes are independent -> run concurrently so vLLM batches agent + user-sim requests.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    gpu_price = GpuPrice(gpu=args.gpu)
    results = []
    lock = threading.Lock()
    aborted = {"v": False}

    def _ep(pname, policy, seed, ti):
        return run_episode(
            task_index=ti, policy=policy, agent_model=agent_model,
            user_model=user_model, seed=seed, cfg=tcfg,
        )

    units = [
        (pname, policy, seed, ti)
        for pname, policy in policies.items()
        for seed in args.seeds
        for ti in task_indices
    ]
    total = len(units)
    print(f"Running {total} episodes ({len(policies)} policies x {len(args.seeds)} seeds x "
          f"{len(task_indices)} tasks), {args.max_workers} concurrent...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_ep, *u): u for u in units}
        done = 0
        for fut in as_completed(futs):
            try:
                tr = fut.result()
            except Exception as e:  # an episode error shouldn't kill the run
                print(f"  episode error: {e}")
                continue
            with lock:
                results.append(tr)
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  [{done}/{total}] reward so far: "
                          f"{sum(r.success for r in results)}/{len(results)}")
                if args.budget is not None:
                    gpu_seconds = sum(r.wall_clock_s for r in results)
                    if gpu_price.cost_for_seconds(gpu_seconds) > args.budget:
                        print("  !! budget kill-switch tripped — cancelling remaining episodes.")
                        aborted["v"] = True
                        for f in futs:
                            f.cancel()
                        break
    wall = time.perf_counter() - t0

    # Save raw + summary.
    out = RunOutput(
        config=RunConfig(
            benchmark_name=f"taubench-{args.env}", model_name=args.model,
            strategies=list(policies.keys()), seeds=args.seeds, n_tasks=args.n_tasks,
            notes=f"taubench {args.env} workers={args.max_workers}"
            + (" [ABORTED]" if aborted["v"] else ""),
        ),
        results=results, task_ids=[f"{args.env}_{i}" for i in task_indices], wall_clock_s=wall,
    )
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe = args.model.replace("/", "-")
    path = args.out or os.path.join("results", f"taubench_{args.env}_{safe}_{ts}.json")
    save_run(out, path)
    print(f"\nWall {wall:.0f}s | saved -> {path}")

    token_price = TokenPrice(args.prompt_price, args.completion_price, "ref")
    summaries = summarize_all(results, gpu_price=gpu_price, token_price=token_price)
    with open(path.replace(".json", "_summary.json"), "w") as f:
        json.dump([s.to_dict() for s in summaries], f, indent=2)
    print(f"\n{'policy':<18}{'success':>9}{'95% CI':>16}{'calls/task':>12}{'tok/task':>10}")
    for s in summaries:
        ci = f"[{s.success_ci.low:.2f},{s.success_ci.high:.2f}]"
        print(f"{s.strategy:<18}{s.success_rate:>9.3f}{ci:>16}{s.avg_calls_per_task:>12.1f}"
              f"{s.avg_tokens_per_task:>10.0f}")


if __name__ == "__main__":
    main()
