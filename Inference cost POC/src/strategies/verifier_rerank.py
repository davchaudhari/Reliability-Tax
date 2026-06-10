"""verifier_rerank — generate n candidates, score each with a cheap verifier prompt (SAME model),
pick the best.

Cost = n generation calls + n verifier calls (the verifier is another forward pass of the same
served model — no separate model, so no extra deployment). The verifier is a self-judge, which is
imperfect: a model that's wrong may also mis-score. That ceiling is real and we let it show. We do
NOT use the benchmark checker to rank (that would be oracle reranking and dishonest); ranking uses
only the model's own verifier score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..benchmarks.base import Task
from ..instrument import TaskResult
from ..model_client import ModelClient
from ._answer import default_answer_key
from .base import BaseStrategy, StrategyConfig, call_model


@dataclass
class VerifierRerank(BaseStrategy):
    config: StrategyConfig = field(default_factory=StrategyConfig)
    name: str = "verifier_rerank"
    answer_key: Callable[[str], str] = default_answer_key

    def run(self, task: Task, model: ModelClient, *, seed: int) -> TaskResult:
        result = self._new_result(task, seed)
        n = self.config.n_samples

        candidates: list[str] = []
        for i in range(n):
            resp = call_model(
                model,
                task.messages,
                result=result,
                role="candidate",
                seed=seed * 1000 + i,
                temperature=self.config.sample_temperature,
                max_tokens=self.config.max_tokens,
                task_id=task.task_id,
            )
            candidates.append(resp.text)

        # Score each candidate with a verifier prompt. The verifier's self-reported confidence
        # (mean logprob on a 'is this correct?' judgement) is our ranking signal.
        scores: list[float] = []
        for i, cand in enumerate(candidates):
            verify_msgs = task.messages + [
                {"role": "assistant", "content": cand},
                {
                    "role": "user",
                    "content": (
                        "You are a verifier. Does the answer correctly and completely solve the "
                        "task? Reply 'SCORE: <0-10>' with a brief justification."
                    ),
                },
            ]
            v = call_model(
                model,
                verify_msgs,
                result=result,
                role="verify",
                seed=seed * 17 + i,
                temperature=0.0,
                max_tokens=64,
                logprobs=True,
                task_id=task.task_id,
                # Pass the candidate's true correctness so the mock verifier's confidence
                # (mean_logprob) imperfectly tracks it — a real-but-noisy ranking signal.
                # On a real backend these mock-only kwargs are ignored.
                is_verification=True,
                prior_correct=_candidate_is_correct(cand),
            )
            # Parse an explicit numeric score if present; else fall back to verifier logprob.
            s = _parse_score(v.text)
            if s is None:
                s = v.mean_logprob if v.mean_logprob is not None else 0.0
            scores.append(s)

        best_idx = max(range(n), key=lambda i: scores[i])
        result.final_answer = candidates[best_idx]
        result.meta.update(
            {
                "n_samples": n,
                "scores": scores,
                "best_idx": best_idx,
                "unique_answers": len({self.answer_key(c) for c in candidates}),
            }
        )
        return result


def _candidate_is_correct(candidate_text: str) -> bool:
    """Mock helper: the candidate text encodes its own correctness ('FINAL: CORRECT').
    On a real backend this is unused (the verifier judges from content)."""
    return "FINAL: CORRECT" in (candidate_text or "").upper()


def _parse_score(text: str):
    import re

    if not text:
        return None
    m = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None
