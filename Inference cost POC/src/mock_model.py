"""Deterministic mock model for the $0 end-to-end pipeline.

Goal: prove the *science* (strategies, metrics, plots) before spending a cent, AND make the
mock behave the way real models do so the plots are meaningful rehearsals — not noise.

Latent-correctness design
--------------------------
Each task carries a latent per-task `p_correct` in [0, 1] (its "difficulty"). A single sampled
rollout from the mock is correct with probability `p_correct`, drawn deterministically from a
hash of (task_id, seed, temperature, call_index). Consequences that mirror reality:

  * `baseline` (greedy, temp 0) is correct iff p_correct >= 0.5  (the "most likely" answer).
  * `self_consistency` over n samples succeeds if the majority of rollouts are correct —
    this beats baseline exactly when p_correct is moderately high, and HURTS when p_correct
    is below 0.5 (majority of wrong answers). That asymmetry is real and shows up in plots.
  * `pass@k` rises with k because P(at least one correct) = 1 - (1 - p_correct)^k.
  * `mean_logprob` is generated to correlate with correctness, so `abstain` can threshold on it.

Token counts scale with a per-task latent "length" plus mild per-call jitter, so cost
accounting is non-degenerate. Everything is a pure function of its inputs — no global RNG
state — so runs are reproducible and matched across strategies.

This file deliberately knows nothing about strategies; it only answers `generate()`.
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

from .model_client import ModelResponse


def _hash_unit_float(*parts: Any) -> float:
    """Deterministic float in [0, 1) from arbitrary hashable parts."""
    h = hashlib.sha256("||".join(str(p) for p in parts).encode()).digest()
    # Take 8 bytes -> uint64 -> scale to [0,1).
    (val,) = struct.unpack("<Q", h[:8])
    return val / 2**64


def _hash_int(low: int, high: int, *parts: Any) -> int:
    """Deterministic int in [low, high] from parts."""
    u = _hash_unit_float(*parts)
    return low + int(u * (high - low + 1))


@dataclass
class MockConfig:
    """Knobs for the synthetic workload."""

    # Spread of per-task difficulty. We sample p_correct from a Beta-like shape via two
    # hash draws so the task set has easy, medium, and hard tasks (not all 0.5).
    min_p_correct: float = 0.15
    max_p_correct: float = 0.92
    # Token scale: prompt and completion base sizes (tokens), plus jitter range.
    prompt_base: int = 220
    completion_base: int = 90
    token_jitter: int = 40
    # Latency model (seconds) — only used so wall-clock/TTFT fields are non-empty in mock.
    # These are NOT used for dollar cost (that comes from real GPU time in cost.py); they let
    # latency-vs-cost plots render in mock mode and get overwritten by real measurements later.
    ttft_base_s: float = 0.05
    sec_per_token: float = 0.004
    # A tiny "self-correction can help" effect: a revise call nudges p upward when the prior
    # attempt was wrong, capturing that reflexion sometimes fixes things (and sometimes can't).
    correct_revise_bonus: float = 0.25


@dataclass
class MockModel:
    """A deterministic, seed-stable fake model implementing the ModelClient protocol."""

    config: MockConfig = field(default_factory=MockConfig)
    # Monotonic per-instance call counter ensures repeated identical requests in a single
    # strategy (e.g. n samples at the same temperature) get DIFFERENT draws, like real sampling.
    _call_index: int = 0
    name: str = "mock-qwen"

    # ---- latent task properties (pure functions of task_id) ----
    def p_correct(self, task_id: str) -> float:
        """Latent probability a single sampled rollout is correct for this task."""
        # Two draws multiplied -> skews toward harder tasks, gives a realistic spread.
        a = _hash_unit_float("pc-a", task_id)
        b = _hash_unit_float("pc-b", task_id)
        u = math.sqrt(a * b)  # in [0,1), skewed
        lo, hi = self.config.min_p_correct, self.config.max_p_correct
        return lo + u * (hi - lo)

    def _length(self, task_id: str) -> int:
        """Latent per-task completion length scale (tokens)."""
        return _hash_int(40, 160, "len", task_id)

    # ---- the core contract ----
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        logprobs: bool = False,
        task_id: Optional[str] = None,
        prior_correct: Optional[bool] = None,
        is_revision: bool = False,
        is_verification: bool = False,
        **kwargs: Any,
    ) -> ModelResponse:
        """Produce a deterministic mock completion.

        Extra (mock-only) kwargs let the harness/strategies pass task context:
          task_id         — which task this call concerns (drives latent correctness).
          prior_correct   — for self-correct: was the previous attempt correct? For a
                            verifier call, the true correctness of the candidate being judged.
          is_revision     — marks a reflexion revise step (applies a small bonus/penalty).
          is_verification — marks a verifier call; its confidence (mean_logprob) is made to
                            track `prior_correct` (the candidate's true correctness) but
                            IMPERFECTLY, modelling a noisy self-judge.

        Real backends ignore these extras; strategies pass them via **kwargs harmlessly.
        """
        self._call_index += 1
        call_idx = self._call_index
        tid = task_id or _infer_task_id(messages)

        p = self.p_correct(tid)
        # A verifier call's "correctness" = did it judge the candidate right? It agrees with the
        # candidate's true correctness with a fixed accuracy, so rerank gets a real but noisy
        # signal (a perfect verifier would be oracle reranking — explicitly avoided).
        if is_verification and prior_correct is not None:
            judge_accuracy = 0.75
            agree = _hash_unit_float("judge", tid, seed, call_idx) < judge_accuracy
            # 'correct' here means the verifier's logprob should reflect candidate correctness.
            correct = prior_correct if agree else (not prior_correct)
            draw = 0.0
        # Greedy (temp ~0) returns the modal answer: correct iff p >= 0.5, no sampling noise.
        elif temperature <= 1e-6 and not is_revision:
            correct = p >= 0.5
            draw = 0.0
        else:
            # Sampled rollout: correct with probability p (revisions get a context-dependent bonus).
            p_eff = p
            if is_revision and prior_correct is False:
                p_eff = min(1.0, p + self.config.correct_revise_bonus)
            elif is_revision and prior_correct is True:
                # Revising a correct answer can occasionally break it (real risk of self-correction).
                p_eff = max(0.0, p - 0.10)
            draw = _hash_unit_float("draw", tid, seed, round(temperature, 4), call_idx)
            correct = draw < p_eff

        # Token accounting.
        cfg = self.config
        prompt_tokens = sum(_approx_tokens(m.get("content", "")) for m in messages)
        if prompt_tokens == 0:
            prompt_tokens = cfg.prompt_base
        comp_jitter = _hash_int(-cfg.token_jitter, cfg.token_jitter, "ctok", tid, seed, call_idx)
        completion_tokens = max(8, self._length(tid) + cfg.completion_base // 2 + comp_jitter)
        completion_tokens = min(completion_tokens, max_tokens)

        # Latency model (mock only).
        ttft = cfg.ttft_base_s + _hash_unit_float("ttft", tid, call_idx) * 0.05
        wall = ttft + cfg.sec_per_token * completion_tokens

        # Logprob signal correlated with correctness so `abstain` can threshold on confidence.
        # Correct answers get higher (less negative) mean logprob; add small deterministic noise.
        noise = (_hash_unit_float("lp", tid, seed, call_idx) - 0.5) * 0.4
        mean_logprob = (-0.25 if correct else -1.1) + noise

        text = _render_answer(tid, correct, call_idx)

        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft if logprobs or True else None,
            mean_logprob=mean_logprob if logprobs else mean_logprob,
            raw={
                "correct": correct,
                "p_correct": p,
                "draw": draw,
                "wall_clock_s": wall,
                "task_id": tid,
                "is_revision": is_revision,
            },
        )


def _approx_tokens(text: str) -> int:
    """Rough token count (~4 chars/token) for mock prompt accounting."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _infer_task_id(messages: list[dict[str, str]]) -> str:
    """Best-effort task id extraction if not passed explicitly (keeps mock self-contained)."""
    blob = " ".join(m.get("content", "") for m in messages)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _render_answer(task_id: str, correct: bool, call_idx: int) -> str:
    """A deterministic answer string. The token CORRECT/WRONG lets the mock benchmark
    adapter check success without re-deriving latent state."""
    tag = "CORRECT" if correct else "WRONG"
    return f"[mock answer for {task_id} | call {call_idx}] FINAL: {tag}"
