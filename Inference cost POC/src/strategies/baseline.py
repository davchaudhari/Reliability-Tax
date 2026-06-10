"""baseline — a single greedy pass. The reference point everything else pays to beat."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..benchmarks.base import Task
from ..instrument import TaskResult
from ..model_client import ModelClient
from .base import BaseStrategy, StrategyConfig, call_model


@dataclass
class Baseline(BaseStrategy):
    config: StrategyConfig = field(default_factory=StrategyConfig)
    name: str = "baseline"

    def run(self, task: Task, model: ModelClient, *, seed: int) -> TaskResult:
        result = self._new_result(task, seed)
        resp = call_model(
            model,
            task.messages,
            result=result,
            role="answer",
            seed=seed,
            temperature=0.0,  # greedy
            max_tokens=self.config.max_tokens,
            task_id=task.task_id,
        )
        result.final_answer = resp.text
        result.meta["n_calls_planned"] = 1
        return result
