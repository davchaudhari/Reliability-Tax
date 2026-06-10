"""Benchmark adapter interface.

A Benchmark exposes a list of Tasks and a `check(task, final_answer)` verdict. Strategies and
the harness depend only on this interface, so swapping BFCL <-> tau-bench <-> mock is a one-line
change. Keeping the surface tiny is deliberate: the project's claim is about reliability-per-
dollar across strategies, and that claim must not depend on a particular benchmark.

We prefer PROGRAMMATIC checking (AST / state / exact match). If a benchmark truly needs an LLM
judge, the adapter is responsible for invoking a self-hosted judge and the harness flags it as a
caveat — but none of the default adapters do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class Task:
    """One benchmark task.

    `messages` is the initial chat context handed to a strategy. `tools` is the available
    function schema list (OpenAI tool format) when the benchmark is tool-use; may be empty.
    `payload` carries adapter-private data the checker needs (ground truth, expected state...).
    """

    task_id: str
    messages: list[dict[str, str]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    # Optional human-facing category (e.g. "multi_turn_base") for slicing results.
    category: str = "default"


@dataclass
class Verdict:
    """Outcome of checking a final answer against a task."""

    success: bool
    detail: str = ""
    # Set True only if the adapter used an LLM judge (so the harness can flag the caveat).
    used_judge: bool = False


class Benchmark(Protocol):
    name: str

    def tasks(self) -> list[Task]:
        """Return the (ordered, deterministic) task list. The harness slices a matched subset."""
        ...

    def check(self, task: Task, final_answer: str, *, transcript: Optional[list] = None) -> Verdict:
        """Check a strategy's final answer (and optionally full transcript) for a task."""
        ...
