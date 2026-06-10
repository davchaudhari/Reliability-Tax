"""abstain — answer, but decline (escalate) when confidence is low.

Confidence signal: the mean logprob of the committed answer (a real, cheap signal from vLLM via
`logprobs`). If below `logprob_threshold`, the strategy ABSTAINS.

Accounting convention (this is the whole point): an abstention is NEITHER a success NOR a plain
failure. It still COSTS (you paid for the call(s) that produced the low-confidence answer), but it
does not count as a solved task. So in reliability-per-dollar, abstaining trades raw success rate
for fewer confident-but-wrong answers — useful when wrong answers are expensive downstream. We
record abstentions explicitly so success rate, abstention rate, and cost all reflect them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..benchmarks.base import Task
from ..instrument import TaskResult
from ..model_client import ModelClient
from .base import BaseStrategy, StrategyConfig, call_model


@dataclass
class Abstain(BaseStrategy):
    config: StrategyConfig = field(default_factory=StrategyConfig)
    name: str = "abstain"

    def run(self, task: Task, model: ModelClient, *, seed: int) -> TaskResult:
        result = self._new_result(task, seed)
        resp = call_model(
            model,
            task.messages,
            result=result,
            role="answer",
            seed=seed,
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            logprobs=True,  # we need the confidence signal
            task_id=task.task_id,
        )
        conf = resp.mean_logprob
        threshold = self.config.logprob_threshold

        if conf is not None and conf < threshold:
            # Abstain / escalate. No committed answer; success stays False, abstained True.
            result.abstained = True
            result.final_answer = ""
            result.meta.update(
                {"confidence_logprob": conf, "threshold": threshold, "decision": "abstain"}
            )
        else:
            result.final_answer = resp.text
            result.meta.update(
                {"confidence_logprob": conf, "threshold": threshold, "decision": "answer"}
            )
        return result
