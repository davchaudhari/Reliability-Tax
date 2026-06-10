"""Multi-turn action policies: the reliability strategies applied per-turn in an agentic loop.

In the single-turn setting (BFCL) a Strategy produces one final answer. In the multi-turn setting
(tau-bench) a strategy instead chooses the NEXT ACTION at each turn of a conversation. Same thesis,
applied at the turn level: each policy may spend several model calls to pick one action, and we
record every call for honest cost accounting.

A policy returns an `ActionChoice` (tool name + kwargs + the raw text it committed to). The runner
appends every CallRecord the policy made to the episode's TaskResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ..instrument import CallRecord, TaskResult, timed_call
from ..model_client import ModelClient
from .prompt import action_key, parse_action


@dataclass
class ActionChoice:
    tool: str
    kwargs: dict[str, Any]
    raw_text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyConfig:
    temperature: float = 0.0
    max_tokens: int = 512
    n_samples: int = 3  # self_consistency
    max_iters: int = 1  # self_correct
    sample_temperature: float = 0.7


def _one_call(
    model: ModelClient,
    messages: list[dict[str, str]],
    *,
    result: TaskResult,
    role: str,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> str:
    """Make one accounted model call; append its CallRecord to the episode result; return text."""
    with timed_call(role) as rec:
        resp = model.generate(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed
        )
    rec.prompt_tokens = resp.prompt_tokens
    rec.completion_tokens = resp.completion_tokens
    rec.mean_logprob = resp.mean_logprob
    if resp.ttft_s is not None:
        rec.ttft_s = resp.ttft_s
    if resp.raw and "wall_clock_s" in resp.raw:
        rec.wall_clock_s = resp.raw["wall_clock_s"]
    result.add_call(rec)
    return resp.text


class ActionPolicy(Protocol):
    name: str
    config: PolicyConfig

    def select_action(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        model: ModelClient,
        *,
        seed: int,
        result: TaskResult,
    ) -> ActionChoice:
        ...
