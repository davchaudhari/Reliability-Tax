"""Real model client: talks to a vLLM OpenAI-compatible endpoint over HTTP.

Implements the same `ModelClient` protocol as the mock, so strategies are unchanged. Key honest-
instrumentation details:

  * Token counts come from the server's `usage` field (authoritative), not estimated locally.
  * TTFT is measured by STREAMING the completion and timestamping the first content chunk — the
    same idea as AsyncLLMEngine TTFT instrumentation in vLLM serving. We fall back to None if the
    server didn't stream.
  * `mean_logprob` is computed from the returned per-token logprobs when `logprobs=True`, giving
    the `abstain` strategy a real confidence signal.

Retries: transient HTTP/connection errors are retried with exponential backoff via tenacity. We do
NOT retry on 4xx (a bad request shouldn't be hammered).

This module imports `openai` lazily so the $0 mock pipeline never needs it installed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from .model_client import ModelResponse


@dataclass
class VLLMClient:
    """OpenAI-compatible client pointed at a self-hosted vLLM server."""

    base_url: str
    model: str
    api_key: str = "EMPTY"  # vLLM ignores the key but the OpenAI SDK requires one
    timeout_s: float = 120.0
    max_retries: int = 4
    stream_for_ttft: bool = True

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI  # lazy import
        except ImportError as e:  # pragma: no cover - only hit without `serve` extra
            raise ImportError(
                "openai is required for VLLMClient. Install with: pip install '.[serve]'"
            ) from e
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        logprobs: bool = False,
        **kwargs: Any,
    ) -> ModelResponse:
        # Drop mock-only kwargs (task_id, prior_correct, is_revision, is_verification...).
        return self._with_retry(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=logprobs,
        )

    def _with_retry(self, *args, **kwargs) -> ModelResponse:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        # Retry only on connection/timeout style errors; let 4xx propagate.
        try:
            from openai import APIConnectionError, APITimeoutError, InternalServerError

            transient = (APIConnectionError, APITimeoutError, InternalServerError)
        except ImportError:  # pragma: no cover
            transient = (Exception,)

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(transient),
        )
        def _call() -> ModelResponse:
            return self._generate_once(*args, **kwargs)

        return _call()

    def _generate_once(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        logprobs: bool,
    ) -> ModelResponse:
        if self.stream_for_ttft:
            return self._generate_streaming(messages, temperature, max_tokens, seed, logprobs)
        return self._generate_blocking(messages, temperature, max_tokens, seed, logprobs)

    def _generate_blocking(self, messages, temperature, max_tokens, seed, logprobs) -> ModelResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=logprobs,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        mean_lp = _mean_logprob_from_choice(choice) if logprobs else None
        return ModelResponse(
            text=text,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            ttft_s=None,
            mean_logprob=mean_lp,
            raw={"finish_reason": choice.finish_reason},
        )

    def _generate_streaming(self, messages, temperature, max_tokens, seed, logprobs) -> ModelResponse:
        start = time.perf_counter()
        ttft: Optional[float] = None
        chunks: list[str] = []
        token_logprobs: list[float] = []
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=logprobs,
            stream=True,
            stream_options={"include_usage": True},
        )
        prompt_tokens = 0
        completion_tokens = 0
        for event in stream:
            if event.usage is not None:
                prompt_tokens = event.usage.prompt_tokens
                completion_tokens = event.usage.completion_tokens
            if not event.choices:
                continue
            delta = event.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - start
                chunks.append(content)
            # Collect logprobs if the server streams them.
            lp = getattr(event.choices[0], "logprobs", None)
            if lp and getattr(lp, "content", None):
                for tok in lp.content:
                    if tok.logprob is not None:
                        token_logprobs.append(tok.logprob)

        text = "".join(chunks)
        mean_lp = (sum(token_logprobs) / len(token_logprobs)) if token_logprobs else None
        # If usage wasn't streamed, fall back to a rough local estimate (flagged in raw).
        usage_estimated = prompt_tokens == 0 and completion_tokens == 0
        if usage_estimated:
            completion_tokens = max(1, len(text) // 4)
        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft,
            mean_logprob=mean_lp,
            raw={"usage_estimated": usage_estimated},
        )


def _mean_logprob_from_choice(choice: Any) -> Optional[float]:
    lp = getattr(choice, "logprobs", None)
    if not lp or not getattr(lp, "content", None):
        return None
    vals = [t.logprob for t in lp.content if t.logprob is not None]
    return (sum(vals) / len(vals)) if vals else None
