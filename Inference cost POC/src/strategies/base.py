"""Strategy interface.

A Strategy turns a Task + model client into a TaskResult, recording EVERY model call so cost is
honest. The benchmark's checker is injected so a strategy can (optionally) verify candidates —
but only `verifier_rerank` uses a *model* verifier; correctness verdicts for scoring always come
from the benchmark checker in the harness, never from the strategy self-grading (that would be
circular). Strategies may call the checker only when the benchmark says checking is cheap and
non-circular; by default they do NOT, to avoid leaking ground truth into the strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from ..benchmarks.base import Task
from ..instrument import CallRecord, TaskResult, timed_call
from ..model_client import ModelClient, ModelResponse


@dataclass
class StrategyConfig:
    """Sampling + strategy knobs. Sweepable from the CLI."""

    temperature: float = 0.0
    max_tokens: int = 512
    # self_consistency / verifier_rerank
    n_samples: int = 5
    # self_correct
    max_iters: int = 2
    # abstain
    logprob_threshold: float = -0.8  # abstain if mean_logprob below this
    sample_temperature: float = 0.7  # temperature used when a strategy needs diversity


class Strategy(Protocol):
    name: str
    config: StrategyConfig

    def run(self, task: Task, model: ModelClient, *, seed: int) -> TaskResult:
        ...


def call_model(
    model: ModelClient,
    messages: list[dict[str, str]],
    *,
    result: TaskResult,
    role: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    logprobs: bool = False,
    **extra: Any,
) -> ModelResponse:
    """Make one model call, timing it and appending a CallRecord to `result`.

    `extra` carries mock-only context (task_id, prior_correct, is_revision); real backends
    ignore unknown kwargs. This is the single chokepoint where calls get accounted, so no
    strategy can accidentally make an unrecorded call.

    Wall-clock precedence: `timed_call` measures real elapsed time and writes it on context
    exit. For the deterministic mock we PREFER the backend-reported synthetic latency (so mock
    plots are stable across machines); we therefore overwrite AFTER the context manager exits,
    not inside it (otherwise the `finally` clobbers our value back to the timer).
    """
    with timed_call(role) as rec:
        resp = model.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=logprobs,
            **extra,
        )
    # --- after context exit: rec.wall_clock_s now holds measured elapsed time ---
    rec.prompt_tokens = resp.prompt_tokens
    rec.completion_tokens = resp.completion_tokens
    rec.mean_logprob = resp.mean_logprob
    if resp.ttft_s is not None:
        rec.ttft_s = resp.ttft_s
    # Backend-reported synthetic wall-clock (mock) wins over the timer for reproducibility.
    if resp.raw and "wall_clock_s" in resp.raw:
        rec.wall_clock_s = resp.raw["wall_clock_s"]
    result.add_call(rec)
    return resp


@dataclass
class BaseStrategy:
    """Convenience base providing config + name plumbing; concrete strategies subclass."""

    config: StrategyConfig = field(default_factory=StrategyConfig)
    name: str = "base"

    def _new_result(self, task: Task, seed: int) -> TaskResult:
        return TaskResult(task_id=task.task_id, strategy=self.name, seed=seed)
