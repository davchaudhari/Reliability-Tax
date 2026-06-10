"""Model client contract shared by the mock and the real (vLLM/OpenAI) backend.

Strategies depend ONLY on this interface, never on a concrete backend. That is what lets
the entire pipeline run for $0 against `mock_model.py` and then swap to a served Qwen model
by changing one constructor call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass
class ModelResponse:
    """One completion plus the accounting we need for honest cost numbers."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: Optional[float] = None
    mean_logprob: Optional[float] = None
    # Backend-specific extras (finish_reason, raw tool calls, etc.)
    raw: dict[str, Any] | None = None


class ModelClient(Protocol):
    """Minimal surface strategies rely on.

    `generate` takes a chat-style message list and sampling params and returns a
    ModelResponse. Implementations are responsible for token accounting and (when
    available) TTFT. Determinism under a given seed is required for reproducibility.
    """

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
        ...
