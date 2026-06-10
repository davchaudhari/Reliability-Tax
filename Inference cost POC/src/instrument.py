"""Honest instrumentation for LLM agent runs.

We separate three layers carefully because conflating them is the most common way
eval cost numbers become dishonest:

  1. CallRecord    — one model call. Tokens, wall-clock, TTFT, success-of-the-call.
  2. TaskResult    — one (strategy, task, seed) attempt. Aggregates the calls it made
                     plus whether the TASK ultimately succeeded / abstained.
  3. cost.py       — turns token + GPU-time accounting into dollars (two views).

Design notes
------------
* `ttft_s` (time to first token) reuses the AsyncLLMEngine TTFT idea from vLLM serving:
  the latency until the first streamed token, distinct from total latency. For non-streamed
  or mock calls we record it as None rather than faking it.
* We never infer tokens we didn't measure. A field is None if the source didn't report it.
* `abstained` is tracked explicitly and is NEITHER success nor failure for the success rate,
  but it DOES still incur cost — that asymmetry is the whole point of the `abstain` strategy.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator, Optional


@dataclass
class CallRecord:
    """A single model call's accounting."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_clock_s: float = 0.0
    ttft_s: Optional[float] = None  # time-to-first-token; None if not measured (mock/non-stream)
    # Free-form tag so strategies can label what a call was for (e.g. "candidate", "critique").
    role: str = "generate"
    # Optional: model-reported mean logprob of the completion, used by `abstain`.
    mean_logprob: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskResult:
    """The outcome of running one Strategy on one task under one seed.

    `success` is the benchmark verdict (did the final answer pass the checker?).
    `abstained` means the strategy declined to answer; when True, `success` is False by
    convention but metrics treat abstentions separately (non-failure-but-no-success).
    """

    task_id: str
    strategy: str
    seed: int
    success: bool = False
    abstained: bool = False
    calls: list[CallRecord] = field(default_factory=list)
    # The answer the strategy commits to; the harness runs the benchmark checker on this to
    # set `success`. Strategies set this; they do NOT set `success` themselves (avoids circular
    # self-grading). Empty string for an abstention.
    final_answer: str = ""
    # Anything a strategy wants to surface for debugging / plots (n, iterations, votes...).
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- aggregates over calls ----
    @property
    def num_calls(self) -> int:
        return len(self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def wall_clock_s(self) -> float:
        return sum(c.wall_clock_s for c in self.calls)

    @property
    def first_ttft_s(self) -> Optional[float]:
        """TTFT of the first call that reported one (proxy for user-perceived latency)."""
        for c in self.calls:
            if c.ttft_s is not None:
                return c.ttft_s
        return None

    def add_call(self, rec: CallRecord) -> None:
        self.calls.append(rec)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "seed": self.seed,
            "success": self.success,
            "abstained": self.abstained,
            "final_answer": self.final_answer,
            "num_calls": self.num_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "wall_clock_s": self.wall_clock_s,
            "first_ttft_s": self.first_ttft_s,
            "meta": self.meta,
            "calls": [c.to_dict() for c in self.calls],
        }
        return d


def task_result_from_dict(d: dict[str, Any]) -> "TaskResult":
    """Reconstruct a TaskResult from its saved JSON dict (inverse of to_dict)."""
    calls = [
        CallRecord(
            prompt_tokens=c.get("prompt_tokens", 0),
            completion_tokens=c.get("completion_tokens", 0),
            wall_clock_s=c.get("wall_clock_s", 0.0),
            ttft_s=c.get("ttft_s"),
            role=c.get("role", "generate"),
            mean_logprob=c.get("mean_logprob"),
        )
        for c in d.get("calls", [])
    ]
    return TaskResult(
        task_id=d["task_id"],
        strategy=d["strategy"],
        seed=d["seed"],
        success=d.get("success", False),
        abstained=d.get("abstained", False),
        final_answer=d.get("final_answer", ""),
        calls=calls,
        meta=d.get("meta", {}),
    )


@contextmanager
def timed_call(role: str = "generate") -> Iterator[CallRecord]:
    """Time a model call and return a CallRecord to be filled in by the caller.

    Usage:
        with timed_call("candidate") as rec:
            resp = client.generate(...)
            rec.prompt_tokens = resp.prompt_tokens
            rec.completion_tokens = resp.completion_tokens
            rec.ttft_s = resp.ttft_s
    The wall-clock is filled automatically on exit.
    """
    rec = CallRecord(role=role)
    start = time.perf_counter()
    try:
        yield rec
    finally:
        rec.wall_clock_s = time.perf_counter() - start
