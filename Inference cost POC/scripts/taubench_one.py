#!/usr/bin/env python3
"""Diagnostic: run ONE tau-bench episode against the real model with a full verbose transcript.

Purpose: before any sweep, SEE what the model actually does — does it emit parseable JSON actions,
do tools execute, does the conversation terminate, what reward. Cheap (~1 episode).

  python scripts/taubench_one.py --base-url https://.../v1 --model Qwen/Qwen2.5-7B-Instruct \
      --task 0 --policy baseline --max-steps 15
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agentic.policy import PolicyConfig
from src.agentic.policies import MT_REGISTRY
from src.agentic.taubench import TauBenchConfig, run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--user-model", default=None)
    ap.add_argument("--env", default="airline")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="baseline", choices=list(MT_REGISTRY))
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=1)
    args = ap.parse_args()

    from src.openai_client import VLLMClient

    agent = VLLMClient(base_url=args.base_url, model=args.model, timeout_s=180)
    user = VLLMClient(base_url=args.base_url, model=args.user_model or args.model, timeout_s=180)

    cfg = PolicyConfig(n_samples=args.n_samples, max_iters=args.max_iters)
    policy = MT_REGISTRY[args.policy](config=cfg)
    tcfg = TauBenchConfig(env_name=args.env, max_steps=args.max_steps)

    res = run_episode(
        task_index=args.task, policy=policy, agent_model=agent, user_model=user,
        seed=args.seed, cfg=tcfg, verbose=True,
    )
    print("\n--- summary ---")
    print(f"reward={res.meta['reward']} success={res.success} turns={res.meta['turns']} "
          f"agent_calls={res.meta['n_agent_calls']} user_sim_calls={res.meta['n_user_sim_calls']} "
          f"tokens={res.total_tokens}")


if __name__ == "__main__":
    main()
