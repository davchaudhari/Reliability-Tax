"""self_correct — reflexion-style act -> critique -> revise loop.

Each iteration spends TWO calls (a critique and a revise) on top of the initial answer, so cost
grows ~linearly in max_iters. Reflexion can fix a wrong first answer, but revising a correct
answer sometimes breaks it — the mock encodes both effects so the cost/reliability trade is real
rather than monotonic. We do NOT consult the benchmark checker to decide when to stop (that would
leak ground truth); stopping is driven by the model's own critique verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..benchmarks.base import Task
from ..instrument import TaskResult
from ..model_client import ModelClient
from .base import BaseStrategy, StrategyConfig, call_model


@dataclass
class SelfCorrect(BaseStrategy):
    config: StrategyConfig = field(default_factory=StrategyConfig)
    name: str = "self_correct"

    def run(self, task: Task, model: ModelClient, *, seed: int) -> TaskResult:
        result = self._new_result(task, seed)

        # Initial attempt (greedy).
        resp = call_model(
            model,
            task.messages,
            result=result,
            role="initial",
            seed=seed,
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            task_id=task.task_id,
        )
        current_text = resp.text
        current_correct = bool(resp.raw.get("correct")) if resp.raw else None

        iters_done = 0
        for it in range(self.config.max_iters):
            # Critique step.
            critique_msgs = task.messages + [
                {"role": "assistant", "content": current_text},
                {
                    "role": "user",
                    "content": (
                        "Critique the answer above. If it is correct and complete, reply exactly "
                        "'VERDICT: OK'. Otherwise reply 'VERDICT: REVISE' and explain the flaw."
                    ),
                },
            ]
            crit = call_model(
                model,
                critique_msgs,
                result=result,
                role="critique",
                seed=seed * 7 + it,
                temperature=self.config.sample_temperature,
                max_tokens=self.config.max_tokens,
                task_id=task.task_id,
                # Critique "confidence" tracks whether current answer is actually correct.
                prior_correct=current_correct,
            )
            # The model's self-judgment: in mock mode we derive a verdict from its correctness
            # signal with noise; in real mode we'd parse 'VERDICT:'. We parse text first, then
            # fall back to the mock signal.
            verdict_ok = _parse_verdict(crit.text)
            if verdict_ok is None and crit.raw is not None:
                # mock fallback: a correct current answer is judged OK ~most of the time.
                verdict_ok = bool(crit.raw.get("correct"))
            if verdict_ok:
                break

            # Revise step.
            revise_msgs = critique_msgs + [
                {"role": "assistant", "content": crit.text},
                {"role": "user", "content": "Now produce a corrected final answer."},
            ]
            rev = call_model(
                model,
                revise_msgs,
                result=result,
                role="revise",
                seed=seed * 13 + it,
                temperature=self.config.sample_temperature,
                max_tokens=self.config.max_tokens,
                task_id=task.task_id,
                prior_correct=current_correct,
                is_revision=True,
            )
            current_text = rev.text
            current_correct = bool(rev.raw.get("correct")) if rev.raw else current_correct
            iters_done = it + 1

        result.final_answer = current_text
        result.meta.update({"max_iters": self.config.max_iters, "iters_done": iters_done})
        return result


def _parse_verdict(text: str):
    """Return True if 'VERDICT: OK', False if 'VERDICT: REVISE', None if not parseable."""
    if not text:
        return None
    up = text.upper()
    if "VERDICT: OK" in up:
        return True
    if "VERDICT: REVISE" in up:
        return False
    return None
