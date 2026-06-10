"""End-to-end smoke test of the $0 mock pipeline: harness runs all strategies, checker scores,
cost aggregation works, and the deterministic mock is reproducible."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks import MockBenchmark
from src.mock_model import MockModel
from src.strategies import REGISTRY
from src.strategies.base import StrategyConfig
from src.harness import run
from src.cost import GpuPrice, TokenPrice
from src.aggregate import summarize_all


def _run(seeds=(0, 1, 2), n_tasks=12):
    cfg = StrategyConfig(n_samples=3, max_iters=2)
    strategies = {name: cls(config=cfg) for name, cls in REGISTRY.items()}
    return run(
        benchmark=MockBenchmark(n_tasks=n_tasks),
        model=MockModel(),
        strategies=strategies,
        seeds=list(seeds),
        n_tasks=n_tasks,
        model_name="mock",
        progress=False,
    )


def test_pipeline_runs_all_strategies():
    out = _run()
    got = {r.strategy for r in out.results}
    assert got == set(REGISTRY.keys())
    # Every result recorded at least one call and a final answer or an abstention.
    for r in out.results:
        assert r.num_calls >= 1
        assert r.final_answer != "" or r.abstained


def test_matched_task_sets():
    out = _run()
    # Each strategy must have seen the same set of task ids.
    by_strat = {}
    for r in out.results:
        by_strat.setdefault(r.strategy, set()).add(r.task_id)
    task_sets = list(by_strat.values())
    assert all(s == task_sets[0] for s in task_sets)


def test_determinism():
    a = _run()
    b = _run()
    sa = sorted((r.strategy, r.seed, r.task_id, r.success) for r in a.results)
    sb = sorted((r.strategy, r.seed, r.task_id, r.success) for r in b.results)
    assert sa == sb


def test_abstain_has_abstentions():
    out = _run()
    abst = [r for r in out.results if r.strategy == "abstain"]
    # The mock makes many tasks low-confidence; abstain should decline at least once.
    assert any(r.abstained for r in abst)
    # Abstentions never count as success.
    assert all((not r.abstained) or (not r.success) for r in out.results)


def test_summaries_have_cost_and_ci():
    out = _run()
    summaries = summarize_all(
        out.results, gpu_price=GpuPrice(gpu="L4"), token_price=TokenPrice()
    )
    for s in summaries:
        assert 0.0 <= s.success_rate <= 1.0
        assert s.success_ci.low <= s.success_rate <= s.success_ci.high + 1e-9
        assert s.cost.normalized_token_usd > 0
        assert s.avg_calls_per_task >= 1
