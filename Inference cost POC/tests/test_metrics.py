"""Tests for the load-bearing math: pass@k estimator and cost views."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

from src.metrics import pass_at_k_single, pass_at_k, pass_at_1, compounding_curve
from src.cost import GpuPrice, TokenPrice


def test_pass_at_k_single_edges():
    # No correct samples -> 0.
    assert pass_at_k_single(n=10, c=0, k=1) == 0.0
    # All correct -> 1 for any k <= n.
    assert pass_at_k_single(n=10, c=10, k=5) == 1.0
    # If fewer than k incorrect remain, any k-draw includes a correct one.
    assert pass_at_k_single(n=5, c=4, k=2) == 1.0  # only 1 wrong, draw 2 -> must hit a right


def test_pass_at_k_single_known_value():
    # n=4, c=1, k=1 -> probability a single draw is the 1 correct = 1/4.
    assert math.isclose(pass_at_k_single(4, 1, 1), 0.25, rel_tol=1e-9)
    # n=4, c=1, k=2 -> P(at least one correct in 2 draws) = 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5.
    assert math.isclose(pass_at_k_single(4, 1, 2), 0.5, rel_tol=1e-9)
    # n=4, c=2, k=2 -> 1 - C(2,2)/C(4,2) = 1 - 1/6.
    assert math.isclose(pass_at_k_single(4, 2, 2), 1 - 1 / 6, rel_tol=1e-9)


def test_pass_at_k_monotonic_in_k():
    n, c = 8, 3
    vals = [pass_at_k_single(n, c, k) for k in range(1, n + 1)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))  # non-decreasing in k


def test_pass_at_1_equals_mean_rate():
    # Two tasks: 1/2 correct and 2/2 correct -> mean rate = (0.5 + 1.0)/2 = 0.75.
    assert math.isclose(pass_at_1([2, 2], [1, 2]), 0.75, rel_tol=1e-9)


def test_pass_at_k_raises_when_k_gt_n():
    try:
        pass_at_k_single(3, 1, 5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cost_views_distinct():
    gp = GpuPrice(gpu="L4")
    tp = TokenPrice(prompt_per_mtok=0.20, completion_per_mtok=0.60)
    # 1M prompt + 1M completion tokens.
    assert math.isclose(tp.cost(1_000_000, 1_000_000), 0.80, rel_tol=1e-9)
    # GPU cost for 100 seconds on L4.
    assert math.isclose(gp.cost_for_seconds(100), 100 * 0.000222, rel_tol=1e-9)


def test_compounding_curve_canonical():
    curve = dict(compounding_curve(0.99, 50))
    assert math.isclose(curve[50], 0.99**50, rel_tol=1e-12)
    assert curve[0] == 1.0
