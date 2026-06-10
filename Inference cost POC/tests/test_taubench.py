"""Mock-first validation of the tau-bench integration — runs entirely offline ($0, no GPU).

Skipped automatically if the tau-bench package isn't installed. Two checks:
  1. Plumbing: the episode loop runs, executes real tools, calls the user-sim, records every call
     (agent + user_sim) into the TaskResult, and computes a reward.
  2. Reward path: an ORACLE agent that replays the task's ground-truth actions then responds gets
     reward 1.0 — proving the DB-state reward wiring is correct end to end.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

tau_bench = pytest.importorskip("tau_bench")  # skip whole module if not installed

from src.agentic.policy import PolicyConfig
from src.agentic.policies import BaselinePolicy
from src.agentic.taubench import TauBenchConfig, run_episode, load_env
from src.model_client import ModelResponse


class ScriptedModel:
    """Mock ModelClient: returns a fixed sequence of texts (cycling on the last). Counts tokens."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def generate(self, messages, *, temperature=0.0, max_tokens=512, seed=None, logprobs=False, **kw):
        text = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return ModelResponse(
            text=text,
            prompt_tokens=sum(len(m.get("content", "")) // 4 for m in messages),
            completion_tokens=max(1, len(text) // 4),
            ttft_s=0.01,
            raw={"wall_clock_s": 0.02},
        )


class StopUser:
    """Mock user model: first message states intent, then immediately STOP."""

    def __init__(self):
        self.i = 0

    def generate(self, messages, *, temperature=0.0, max_tokens=512, seed=None, logprobs=False, **kw):
        self.i += 1
        text = "I'd like some help." if self.i == 1 else "###STOP###"
        return ModelResponse(text=text, prompt_tokens=10, completion_tokens=3, raw={"wall_clock_s": 0.01})


def test_episode_plumbing_records_all_calls():
    cfg = TauBenchConfig(env_name="airline", max_steps=4)
    # Agent: a harmless read tool, then respond -> user STOPs.
    agent = ScriptedModel([
        json.dumps({"tool": "get_user_details", "arguments": {"user_id": "mia_li_3668"}}),
        json.dumps({"tool": "respond", "content": "All set, anything else?"}),
    ])
    res = run_episode(
        task_index=0, policy=BaselinePolicy(config=PolicyConfig()),
        agent_model=agent, user_model=StopUser(), seed=0, cfg=cfg,
    )
    assert res.num_calls >= 2, "should have recorded agent + user-sim calls"
    assert res.meta["n_user_sim_calls"] >= 1, "user simulator calls must be accounted"
    assert res.meta["n_agent_calls"] >= 1
    assert res.meta["turns"] >= 1
    assert isinstance(res.meta["reward"], (int, float))
    assert res.total_tokens > 0


def test_oracle_replay_gets_reward_one():
    cfg = TauBenchConfig(env_name="airline", max_steps=10)
    # Pull the ground-truth actions for task 0 and have the agent replay them verbatim.
    env = load_env(0, cfg)
    gt = env.tasks[0].actions
    script = [json.dumps({"tool": a.name, "arguments": a.kwargs}) for a in gt]
    script.append(json.dumps({"tool": "respond", "content": "Your request is complete."}))
    agent = ScriptedModel(script)
    res = run_episode(
        task_index=0, policy=BaselinePolicy(config=PolicyConfig()),
        agent_model=agent, user_model=StopUser(), seed=0, cfg=cfg,
    )
    assert res.meta["reward"] == 1.0, f"oracle replay should score 1.0, got {res.meta['reward']}"
    assert res.success is True
