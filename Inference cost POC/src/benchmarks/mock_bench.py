"""Mock benchmark for the $0 pipeline.

Generates N synthetic tasks with stable ids. The checker reads the mock model's correctness
signal embedded in the final answer string ("FINAL: CORRECT" / "FINAL: WRONG"), so the mock
benchmark and mock model agree without a side channel. This lets the whole harness/metrics/plots
stack run with zero network and zero GPU, exercising exactly the same code paths a real run uses.
"""
from __future__ import annotations

from typing import Optional

from .base import Benchmark, Task, Verdict


class MockBenchmark:
    name = "mock"

    def __init__(self, n_tasks: int = 40, category: str = "mock") -> None:
        self.n_tasks = n_tasks
        self.category = category

    def tasks(self) -> list[Task]:
        out: list[Task] = []
        for i in range(self.n_tasks):
            tid = f"mock_task_{i:03d}"
            messages = [
                {"role": "system", "content": "You are a tool-use agent. Solve the task."},
                {
                    "role": "user",
                    "content": (
                        f"Task {tid}: call the right tools and produce a final answer. "
                        "End your reply with 'FINAL: <answer>'."
                    ),
                },
            ]
            out.append(Task(task_id=tid, messages=messages, category=self.category))
        return out

    def check(
        self, task: Task, final_answer: str, *, transcript: Optional[list] = None
    ) -> Verdict:
        # The mock model encodes ground-truth correctness in the answer text.
        text = (final_answer or "").upper()
        if "FINAL: CORRECT" in text:
            return Verdict(success=True, detail="mock-correct")
        return Verdict(success=False, detail="mock-wrong")
