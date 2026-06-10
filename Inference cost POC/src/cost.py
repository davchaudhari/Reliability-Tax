"""Two honest cost views, kept rigorously separate.

The single most important methodological point of this project: there are TWO different
dollar numbers for the same run, and conflating them is dishonest.

  (a) MEASURED Modal GPU-time cost — the real bill. You rent a GPU by the second; the cost of a
      run is (GPU-seconds consumed) x (GPU $/sec). This is what actually leaves your wallet. It
      depends on YOUR batch efficiency, cold starts, and idle scaledown — it does NOT generalize
      to anyone else's setup.

  (b) NORMALIZED reference-token cost — what the same token volume would cost at a published
      per-token API price (e.g. a hosted Qwen/GPT price sheet). This generalizes across setups
      because it's just tokens x price, but it IGNORES batching/throughput entirely.

These two diverge, sometimes by an order of magnitude, because self-hosting amortizes a fixed
GPU rental across however many tokens you can push through it. We report both and document the
gap as a first-class caveat rather than picking the flattering one.

All prices here are overridable. The Modal GPU rates and reference token prices we looked up
live in BUDGET.md with their source URLs and lookup date; the defaults below mirror them but are
intended to be passed in explicitly at run time so the numbers are never silently stale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .instrument import TaskResult


# ---------------------------------------------------------------------------
# Modal GPU pricing (per-second). Verified from modal.com/pricing — see BUDGET.md
# for lookup date + source URL. Per-second billing; $0 while scaled to zero.
# ---------------------------------------------------------------------------
MODAL_GPU_USD_PER_SEC: dict[str, float] = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10G": 0.000306,  # listed as "A10" on the pricing page; requested as "A10G"
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}


# ---------------------------------------------------------------------------
# Reference API token prices (USD per 1M tokens). These are CONFIGURABLE placeholders
# so others can plug in their own provider's sheet. Defaults are a plausible hosted
# small-model price; OVERRIDE per run and record the source in BUDGET.md.
# ---------------------------------------------------------------------------
@dataclass
class TokenPrice:
    """USD per 1,000,000 tokens, split by prompt vs completion (they differ in practice)."""

    prompt_per_mtok: float = 0.20
    completion_per_mtok: float = 0.60
    label: str = "reference-default"

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1_000_000 * self.prompt_per_mtok
            + completion_tokens / 1_000_000 * self.completion_per_mtok
        )


@dataclass
class GpuPrice:
    """Measured-cost model: a GPU rented at `usd_per_sec`."""

    gpu: str = "L4"
    usd_per_sec: Optional[float] = None  # if None, looked up from MODAL_GPU_USD_PER_SEC

    def rate(self) -> float:
        if self.usd_per_sec is not None:
            return self.usd_per_sec
        if self.gpu not in MODAL_GPU_USD_PER_SEC:
            raise KeyError(
                f"Unknown GPU '{self.gpu}'. Known: {sorted(MODAL_GPU_USD_PER_SEC)}. "
                "Pass usd_per_sec explicitly or update MODAL_GPU_USD_PER_SEC from BUDGET.md."
            )
        return MODAL_GPU_USD_PER_SEC[self.gpu]

    def cost_for_seconds(self, gpu_seconds: float) -> float:
        return gpu_seconds * self.rate()


@dataclass
class CostBreakdown:
    """Both cost views for a set of results, plus the inputs that produced them."""

    measured_gpu_usd: float
    normalized_token_usd: float
    gpu_seconds: float
    prompt_tokens: int
    completion_tokens: int
    num_calls: int
    gpu: str
    gpu_usd_per_sec: float
    token_price_label: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def measured_to_normalized_ratio(self) -> Optional[float]:
        if self.normalized_token_usd == 0:
            return None
        return self.measured_gpu_usd / self.normalized_token_usd

    def to_dict(self) -> dict:
        return {
            "measured_gpu_usd": self.measured_gpu_usd,
            "normalized_token_usd": self.normalized_token_usd,
            "gpu_seconds": self.gpu_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "num_calls": self.num_calls,
            "gpu": self.gpu,
            "gpu_usd_per_sec": self.gpu_usd_per_sec,
            "token_price_label": self.token_price_label,
            "measured_to_normalized_ratio": self.measured_to_normalized_ratio,
        }


def _result_gpu_seconds(r: TaskResult) -> float:
    """GPU-seconds attributable to a result.

    In MOCK mode the only latency signal is each call's wall_clock_s (synthetic). In REAL mode
    the harness fills wall_clock_s from measured request latency; with concurrent batching that
    OVER-counts (requests overlap on one GPU), so the harness ALSO records a run-level wall-clock
    and we reconcile against the true Modal bill in BUDGET.md. Per-result GPU-seconds is therefore
    an UPPER-BOUND attribution, flagged as a caveat. We keep it for relative comparisons.
    """
    return r.wall_clock_s


def summarize_cost(
    results: Iterable[TaskResult],
    *,
    gpu_price: GpuPrice,
    token_price: TokenPrice,
) -> CostBreakdown:
    """Aggregate both cost views over a collection of TaskResults."""
    results = list(results)
    gpu_seconds = sum(_result_gpu_seconds(r) for r in results)
    prompt_tokens = sum(r.prompt_tokens for r in results)
    completion_tokens = sum(r.completion_tokens for r in results)
    num_calls = sum(r.num_calls for r in results)

    measured = gpu_price.cost_for_seconds(gpu_seconds)
    normalized = token_price.cost(prompt_tokens, completion_tokens)

    return CostBreakdown(
        measured_gpu_usd=measured,
        normalized_token_usd=normalized,
        gpu_seconds=gpu_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        num_calls=num_calls,
        gpu=gpu_price.gpu,
        gpu_usd_per_sec=gpu_price.rate(),
        token_price_label=token_price.label,
    )
