"""Reliability strategies — the core IP.

Each strategy implements `run(task, model_client) -> TaskResult`, may issue multiple model
calls, and records every call for honest cost accounting. The thesis: each of these buys
reliability by spending inference, and we measure the exchange rate.
"""
from .base import Strategy, StrategyConfig
from .baseline import Baseline
from .self_consistency import SelfConsistency
from .self_correct import SelfCorrect
from .abstain import Abstain
from .verifier_rerank import VerifierRerank

# Registry so the CLI can select strategies by name.
REGISTRY: dict[str, type[Strategy]] = {
    "baseline": Baseline,
    "self_consistency": SelfConsistency,
    "self_correct": SelfCorrect,
    "abstain": Abstain,
    "verifier_rerank": VerifierRerank,
}

__all__ = [
    "Strategy",
    "StrategyConfig",
    "Baseline",
    "SelfConsistency",
    "SelfCorrect",
    "Abstain",
    "VerifierRerank",
    "REGISTRY",
]
