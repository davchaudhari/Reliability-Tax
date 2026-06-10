"""The three multi-turn policies used in the tau-bench headline: baseline, self_consistency,
self_correct. Each picks the next action; cost is whatever inference it spends to do so."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..instrument import TaskResult
from ..model_client import ModelClient
from .policy import ActionChoice, PolicyConfig, _one_call
from .prompt import action_key, parse_action


@dataclass
class BaselinePolicy:
    config: PolicyConfig = field(default_factory=PolicyConfig)
    name: str = "baseline"

    def select_action(self, messages, tools, model: ModelClient, *, seed, result: TaskResult) -> ActionChoice:
        text = _one_call(
            model, messages, result=result, role="action",
            seed=seed, temperature=0.0, max_tokens=self.config.max_tokens,
        )
        tool, kwargs = parse_action(text)
        return ActionChoice(tool=tool, kwargs=kwargs, raw_text=text)


@dataclass
class SelfConsistencyPolicy:
    """Sample n candidate actions; majority-vote on the action (by canonical key)."""

    config: PolicyConfig = field(default_factory=PolicyConfig)
    name: str = "self_consistency"

    def select_action(self, messages, tools, model: ModelClient, *, seed, result: TaskResult) -> ActionChoice:
        n = self.config.n_samples
        cands = []
        for i in range(n):
            text = _one_call(
                model, messages, result=result, role="action_sample",
                seed=seed * 1000 + i, temperature=self.config.sample_temperature,
                max_tokens=self.config.max_tokens,
            )
            tool, kwargs = parse_action(text)
            cands.append((action_key(tool, kwargs), tool, kwargs, text))
        counts = Counter(k for k, *_ in cands)
        winner_key, votes = counts.most_common(1)[0]
        _, tool, kwargs, text = next(c for c in cands if c[0] == winner_key)
        return ActionChoice(
            tool=tool, kwargs=kwargs, raw_text=text,
            meta={"n_samples": n, "winner_votes": votes, "unique": len(counts)},
        )


@dataclass
class SelfCorrectPolicy:
    """Propose an action, critique it against the policy, revise if the critique says so."""

    config: PolicyConfig = field(default_factory=PolicyConfig)
    name: str = "self_correct"

    def select_action(self, messages, tools, model: ModelClient, *, seed, result: TaskResult) -> ActionChoice:
        text = _one_call(
            model, messages, result=result, role="action",
            seed=seed, temperature=0.0, max_tokens=self.config.max_tokens,
        )
        tool, kwargs = parse_action(text)
        iters_done = 0
        for it in range(self.config.max_iters):
            critique_msgs = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Review the JSON action you just proposed against the policy and the "
                        "conversation. If it is the correct next action, reply exactly 'VERDICT: OK'. "
                        "Otherwise reply 'VERDICT: REVISE' and briefly say why."
                    ),
                },
            ]
            crit = _one_call(
                model, critique_msgs, result=result, role="critique",
                seed=seed * 7 + it, temperature=self.config.sample_temperature,
                max_tokens=self.config.max_tokens,
            )
            if "VERDICT: OK" in crit.upper() or "VERDICT: REVISE" not in crit.upper():
                break
            revise_msgs = critique_msgs + [
                {"role": "assistant", "content": crit},
                {"role": "user", "content": "Now output the corrected JSON action (only the JSON)."},
            ]
            text = _one_call(
                model, revise_msgs, result=result, role="revise",
                seed=seed * 13 + it, temperature=self.config.sample_temperature,
                max_tokens=self.config.max_tokens,
            )
            tool, kwargs = parse_action(text)
            iters_done = it + 1
        return ActionChoice(
            tool=tool, kwargs=kwargs, raw_text=text, meta={"iters_done": iters_done}
        )


MT_REGISTRY = {
    "baseline": BaselinePolicy,
    "self_consistency": SelfConsistencyPolicy,
    "self_correct": SelfCorrectPolicy,
}
