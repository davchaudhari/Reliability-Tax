"""self_consistency — sample n rollouts, majority-vote the answer.

Pays n model calls to buy a reliability change. The sign of that change is NOT always positive:
when the per-task correctness probability is below 0.5, the majority vote concentrates on the
WRONG answer and self-consistency underperforms baseline. Surfacing exactly when voting helps vs.
hurts (as a function of cost) is a core deliverable, so we record the vote distribution.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from ..benchmarks.base import Task
from ..instrument import TaskResult
from ..model_client import ModelClient
from ._answer import default_answer_key
from .base import BaseStrategy, StrategyConfig, call_model


@dataclass
class SelfConsistency(BaseStrategy):
    config: StrategyConfig = field(default_factory=StrategyConfig)
    name: str = "self_consistency"
    answer_key: Callable[[str], str] = default_answer_key

    def run(self, task: Task, model: ModelClient, *, seed: int) -> TaskResult:
        result = self._new_result(task, seed)
        n = self.config.n_samples
        temp = self.config.sample_temperature

        texts: list[str] = []
        for i in range(n):
            resp = call_model(
                model,
                task.messages,
                result=result,
                role="vote_sample",
                # Distinct sub-seed per sample so rollouts differ deterministically.
                seed=seed * 1000 + i,
                temperature=temp,
                max_tokens=self.config.max_tokens,
                task_id=task.task_id,
            )
            texts.append(resp.text)

        keys = [self.answer_key(t) for t in texts]
        counts = Counter(keys)
        winner_key, winner_votes = counts.most_common(1)[0]
        # Commit to one representative full text whose key is the winner.
        winner_text = next(t for t, k in zip(texts, keys) if k == winner_key)

        result.final_answer = winner_text
        result.meta.update(
            {
                "n_samples": n,
                "winner_votes": winner_votes,
                "vote_distribution": dict(counts),
                "unique_answers": len(counts),
            }
        )
        return result
