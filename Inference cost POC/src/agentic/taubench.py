"""tau-bench integration: instrumented user-simulator + episode runner.

We use tau-bench's `Env` for everything that must be faithful (task data, tool execution, the
user-simulator's behavior, and the programmatic DB-state reward) but drive it with our own prompted
agent loop so the reliability *policies* choose each action and EVERY model call (agent AND
user-simulator) is accounted through our instrumented client.

Honest-instrumentation notes:
- tau-bench's stock user-simulator calls models via litellm and reports a litellm dollar cost that
  is meaningless for a self-hosted endpoint. We replace it with `InstrumentedUser`, which calls our
  `ModelClient` and records token/latency CallRecords into the SAME episode TaskResult — so agent
  and user-sim costs are both captured and separable by `role`.
- Reward is tau-bench's own `calculate_reward` (DB-state hash vs replayed ground-truth actions +
  required output substrings). No LLM judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..instrument import CallRecord, TaskResult, timed_call
from ..model_client import ModelClient
from .policy import ActionPolicy
from .prompt import RESPOND, action_to_assistant_text, build_system_prompt


# tau-bench's user system prompt (mirrors LLMUserSimulationEnv.build_system_prompt), so behavior
# matches the benchmark even though the call goes through our client.
_USER_SYS_TEMPLATE = """You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""


class InstrumentedUser:
    """Drop-in replacement for tau-bench's user-sim that routes through our ModelClient.

    Implements the same interface the Env calls: reset(instruction)->str, step(content)->str,
    get_total_cost()->float. Records each call into `record_sink` (the episode's result.add_call).
    """

    def __init__(self, model: ModelClient, record_sink: Callable[[CallRecord], None], *, seed: int):
        self.model = model
        self.record_sink = record_sink
        self.seed = seed
        self.messages: list[dict[str, str]] = []
        self._turn = 0

    def _gen(self) -> str:
        self._turn += 1
        with timed_call("user_sim") as rec:
            resp = self.model.generate(
                self.messages, temperature=0.0, max_tokens=256, seed=self.seed * 31 + self._turn
            )
        rec.prompt_tokens = resp.prompt_tokens
        rec.completion_tokens = resp.completion_tokens
        if resp.ttft_s is not None:
            rec.ttft_s = resp.ttft_s
        if resp.raw and "wall_clock_s" in resp.raw:
            rec.wall_clock_s = resp.raw["wall_clock_s"]
        self.record_sink(rec)
        self.messages.append({"role": "assistant", "content": resp.text})
        return resp.text

    def reset(self, instruction: Optional[str] = None) -> str:
        disp = ("\n\nInstruction: " + instruction + "\n") if instruction else ""
        self.messages = [
            {"role": "system", "content": _USER_SYS_TEMPLATE.format(instruction_display=disp)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self._gen()

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self._gen()

    def get_total_cost(self) -> float:
        return 0.0  # cost is accounted via recorded tokens, not litellm


@dataclass
class TauBenchConfig:
    env_name: str = "airline"
    task_split: str = "test"
    max_steps: int = 25


def load_env(task_index: int, cfg: TauBenchConfig):
    """Construct a tau-bench env OFFLINE (human user-strategy avoids a model call in the ctor);
    the caller injects an InstrumentedUser before reset()."""
    from tau_bench.envs import get_env

    return get_env(
        cfg.env_name,
        user_strategy="human",  # no model call at construction; we replace env.user per episode
        user_model="placeholder",
        task_split=cfg.task_split,
        task_index=task_index,
    )


def run_episode(
    *,
    task_index: int,
    policy: ActionPolicy,
    agent_model: ModelClient,
    user_model: ModelClient,
    seed: int,
    cfg: TauBenchConfig,
    verbose: bool = False,
) -> TaskResult:
    """Run one (task, policy, seed) conversation; return a TaskResult with reward + all calls."""
    from tau_bench.types import Action

    def _vlog(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    env = load_env(task_index, cfg)
    result = TaskResult(
        task_id=f"{cfg.env_name}_{task_index}", strategy=policy.name, seed=seed
    )
    # Inject our instrumented user (records into this episode's result).
    env.user = InstrumentedUser(user_model, result.add_call, seed=seed)

    reset = env.reset(task_index=task_index)
    obs = reset.observation
    system = build_system_prompt(env.wiki, env.tools_info)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": obs},
    ]

    _vlog(f"\n=== task {task_index} | policy {policy.name} | seed {seed} ===")
    _vlog(f"[user→agent] {obs[:300]}")
    reward = 0.0
    steps = 0
    done = False
    stop_reason = "max_steps"
    # Loop guard: a weak model can repeat the same failing tool call forever (we saw `calculate`
    # loop 9x). Track consecutive identical failing actions and bail out — this both saves cost and
    # stops runaway context growth that would otherwise hit the context-length limit.
    last_fail_key: Optional[str] = None
    fail_streak = 0
    for step in range(cfg.max_steps):
        steps = step + 1
        try:
            choice = policy.select_action(
                messages, env.tools_info, agent_model, seed=seed * 1000 + step, result=result
            )
        except Exception as e:  # e.g. context-length BadRequestError on a long conversation
            stop_reason = f"agent_error:{type(e).__name__}"
            _vlog(f"[step {steps}] agent call failed: {e}")
            break
        action = Action(name=choice.tool, kwargs=choice.kwargs)
        env_resp = env.step(action)
        reward = env_resp.reward
        obs_str = str(env_resp.observation)
        _vlog(
            f"[step {steps}] action={choice.tool} args={str(choice.kwargs)[:160]}\n"
            f"          obs={obs_str[:240]}"
        )
        # Loop guard bookkeeping (only non-respond actions; an errored tool obs starts with "Error").
        is_error = obs_str.startswith("Error")
        from .prompt import action_key as _akey
        cur_key = _akey(choice.tool, choice.kwargs)
        if is_error and cur_key == last_fail_key:
            fail_streak += 1
        elif is_error:
            last_fail_key, fail_streak = cur_key, 1
        else:
            last_fail_key, fail_streak = None, 0
        # Record the assistant's committed action, then the resulting observation.
        messages.append(
            {"role": "assistant", "content": action_to_assistant_text(choice.tool, choice.kwargs)}
        )
        if choice.tool == RESPOND:
            messages.append({"role": "user", "content": env_resp.observation})
        else:
            messages.append(
                {"role": "user", "content": f"[tool:{choice.tool} result] {env_resp.observation}"}
            )
        if env_resp.done:
            done = True
            stop_reason = "env_done"
            break
        if fail_streak >= 3:
            stop_reason = "loop_guard"
            _vlog(f"[step {steps}] loop guard: same failing action x{fail_streak} — ending episode")
            break
    _vlog(f"=== done={done} reward={reward} turns={steps} stop={stop_reason} "
          f"calls={result.num_calls} ===")

    result.success = reward >= 1.0
    result.final_answer = ""  # n/a for multi-turn; success comes from reward
    result.meta.update(
        {
            "reward": reward,
            "turns": steps,
            "done": done,
            "stop_reason": stop_reason,
            "n_user_sim_calls": sum(1 for c in result.calls if c.role == "user_sim"),
            "n_agent_calls": sum(1 for c in result.calls if c.role != "user_sim"),
        }
    )
    return result
