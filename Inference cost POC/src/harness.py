"""Harness: run strategies over a benchmark with matched task sets and multiple seeds.

Guarantees that make the comparison fair:
  * MATCHED TASK SETS — every strategy sees the exact same task subset (same ids, same order),
    selected deterministically from the benchmark. No strategy gets an easier slice.
  * MULTIPLE SEEDS — each (strategy, task) is run under every seed in `seeds`; metrics aggregate
    across seeds and bootstrap over tasks.
  * SUCCESS COMES FROM THE BENCHMARK CHECKER, never from the strategy. The strategy returns a
    `final_answer`; the harness runs `benchmark.check(...)` to set `success`. Abstentions never
    count as success.
  * EVERY CALL IS ACCOUNTED via the strategy's TaskResult. The harness adds nothing untracked.

Budget kill-switch: the harness accepts an optional cost guard that aborts mid-run if a live cost
estimate exceeds a cap. In mock mode cost is ~0 so the guard never trips; it matters on GPU runs.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .benchmarks.base import Benchmark, Task
from .instrument import TaskResult
from .model_client import ModelClient
from .strategies.base import Strategy


@dataclass
class RunConfig:
    benchmark_name: str
    model_name: str
    strategies: list[str]
    seeds: list[int]
    n_tasks: int
    task_offset: int = 0
    notes: str = ""


@dataclass
class RunOutput:
    config: RunConfig
    results: list[TaskResult] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    wall_clock_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "config": {
                "benchmark": self.config.benchmark_name,
                "model": self.config.model_name,
                "strategies": self.config.strategies,
                "seeds": self.config.seeds,
                "n_tasks": self.config.n_tasks,
                "task_offset": self.config.task_offset,
                "notes": self.config.notes,
            },
            "task_ids": self.task_ids,
            "wall_clock_s": self.wall_clock_s,
            "results": [r.to_dict() for r in self.results],
        }


def select_tasks(benchmark: Benchmark, n_tasks: int, offset: int = 0) -> list[Task]:
    """Deterministically select a matched subset: a contiguous, stable slice of the ordered list.

    Using a stable slice (not random sampling) guarantees byte-for-byte identical task sets across
    strategies and across reruns. If you want a different subset, change offset/n explicitly.
    """
    all_tasks = benchmark.tasks()
    if not all_tasks:
        return []
    chosen = all_tasks[offset : offset + n_tasks]
    return chosen


def _score(benchmark: Benchmark, task: Task, tr: TaskResult) -> TaskResult:
    """Apply the benchmark checker — the ONLY source of `success`. Abstentions are non-success."""
    if tr.abstained:
        tr.success = False
    else:
        verdict = benchmark.check(task, tr.final_answer)
        tr.success = verdict.success
        if verdict.used_judge:
            tr.meta["used_judge"] = True
    return tr


def run(
    *,
    benchmark: Benchmark,
    model: ModelClient,
    strategies: dict[str, Strategy],
    seeds: Sequence[int],
    n_tasks: int,
    task_offset: int = 0,
    model_name: str = "mock",
    notes: str = "",
    cost_guard: Optional[Callable[[list[TaskResult]], bool]] = None,
    progress: bool = True,
    max_workers: int = 1,
) -> RunOutput:
    """Execute the full (strategy x task x seed) grid on a MATCHED task set.

    `cost_guard(results_so_far) -> True` aborts the run (used as a budget kill-switch on GPU runs).

    `max_workers > 1` runs work units concurrently against the model server. This is the serving-
    efficiency lever: a self-hosted vLLM server batches concurrent requests (continuous batching),
    so the GPU is kept busy instead of idling on sequential network round-trips — cutting alive-time
    (and thus measured cost) dramatically. Use it ONLY with a real, thread-safe HTTP client; the
    deterministic mock shares per-call state and must stay at max_workers=1 (keeps results stable).
    Aggregation sorts by (task, seed), so the nondeterministic completion ORDER under concurrency
    does not affect metrics.
    """
    tasks = select_tasks(benchmark, n_tasks, task_offset)
    if not tasks:
        raise RuntimeError(
            f"Benchmark '{benchmark.name}' returned no tasks for offset={task_offset}, "
            f"n_tasks={n_tasks}. (For BFCL ensure data_dir points at local data.)"
        )

    cfg = RunConfig(
        benchmark_name=benchmark.name,
        model_name=model_name,
        strategies=list(strategies.keys()),
        seeds=list(seeds),
        n_tasks=len(tasks),
        task_offset=task_offset,
        notes=notes,
    )
    out = RunOutput(config=cfg, task_ids=[t.task_id for t in tasks])

    # Build the matched work-unit list once: (strategy_name, strategy, seed, task).
    units = [
        (sname, strat, seed, task)
        for sname, strat in strategies.items()
        for seed in seeds
        for task in tasks
    ]
    total = len(units)
    start = time.perf_counter()
    aborted = False

    if max_workers <= 1:
        # Sequential path (default): preserves ordering and mock determinism.
        for i, (sname, strat, seed, task) in enumerate(units, 1):
            tr = _score(benchmark, task, strat.run(task, model, seed=seed))
            out.results.append(tr)
            if progress and (i % 50 == 0 or i == total):
                print(f"  [{i}/{total}] {sname} seed={seed} task={task.task_id}")
            if cost_guard is not None and cost_guard(out.results):
                print("  !! cost_guard tripped — aborting run to respect budget.")
                aborted = True
                break
    else:
        # Concurrent path: many units in flight so vLLM batches them. Real client only.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        lock = threading.Lock()

        def _work(unit):
            sname, strat, seed, task = unit
            return _score(benchmark, task, strat.run(task, model, seed=seed))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_work, u): u for u in units}
            done = 0
            for fut in as_completed(futures):
                tr = fut.result()
                with lock:
                    out.results.append(tr)
                    done += 1
                    if progress and (done % 50 == 0 or done == total):
                        print(f"  [{done}/{total}] completed (concurrent, {max_workers} workers)")
                    if cost_guard is not None and cost_guard(out.results):
                        print("  !! cost_guard tripped — cancelling remaining units.")
                        aborted = True
                        # Cancel not-yet-started futures; in-flight ones finish.
                        for f in futures:
                            f.cancel()
                        break

    out.wall_clock_s = time.perf_counter() - start
    out.config.notes += " [ABORTED by cost_guard]" if aborted else ""
    if max_workers > 1:
        out.config.notes += f" [concurrent x{max_workers}]"
    return out


def save_run(out: RunOutput, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out.to_dict(), f, indent=2)


def load_run(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
