"""Regression tests for the BFCL AST checker — locking in fixes the Phase 1 smoke test surfaced.

The smoke test revealed the checker under-counted real model outputs ~4x because it:
  - only parsed keyword args (models often emit positional calls), and
  - stripped namespace-qualified names (GT uses fully-qualified names like math.factorial).
These tests pin the corrected behavior.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.base import Task
from src.benchmarks.bfcl import BFCLBenchmark, _parse_call, _func_name


def _task(name, params, gt, required=None):
    tools = [
        {
            "name": name,
            "description": "test fn",
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "integer"} for p in params},
                "required": required or [],
            },
        }
    ]
    return Task(
        task_id="t",
        messages=[],
        tools=tools,
        payload={"category": "simple_python", "ground_truth": gt},
        category="simple_python",
    )


B = BFCLBenchmark()  # no data needed; check() works on a Task we build


def test_keyword_args_match():
    t = _task("calculate_triangle_area", ["base", "height", "unit"],
              [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}])
    assert B.check(t, "calculate_triangle_area(base=10, height=5)").success  # optional unit omitted
    assert B.check(t, 'calculate_triangle_area(base=10, height=5, unit="units")').success


def test_positional_args_mapped_via_schema():
    # Model emits positional; checker must map 1->a, -3->b, 2->c using param order.
    t = _task("algebra.quadratic_roots", ["a", "b", "c"],
              [{"algebra.quadratic_roots": {"a": [1], "b": [-3], "c": [2]}}],
              required=["a", "b", "c"])
    assert B.check(t, "algebra.quadratic_roots(1, -3, 2)").success
    assert not B.check(t, "algebra.quadratic_roots(9, -3, 2)").success  # wrong a


def test_namespace_qualified_name_match():
    t = _task("math.factorial", ["number"], [{"math.factorial": {"number": [5]}}],
              required=["number"])
    assert B.check(t, "math.factorial(5)").success      # full name + positional
    assert B.check(t, "factorial(number=5)").success    # suffix-name match + kwarg


def test_optional_arg_present_or_absent():
    t = _task("math.hypot", ["x", "y", "z"],
              [{"math.hypot": {"x": [4], "y": [5], "z": ["", 0]}}], required=["x", "y"])
    assert B.check(t, "math.hypot(4, 5)").success        # z omitted (optional via "")
    assert B.check(t, "math.hypot(x=4, y=5, z=0)").success  # z=0 also allowed


def test_wrong_function_name_fails():
    t = _task("solve_quadratic_equation", ["a", "b", "c"],
              [{"solve_quadratic_equation": {"a": [2], "b": [6], "c": [5]}}],
              required=["a", "b", "c"])
    # Right args but wrong name (the smoke test's task_4 placeholder bug) must fail.
    assert not B.check(t, "function_name(2, 6, 5)").success


def test_parse_call_returns_positional_and_dotted_name():
    name, kwargs, pos = _parse_call("math.factorial(5)")
    assert name == "math.factorial"
    assert kwargs == {}
    assert pos == [5]


def test_func_name_full_dotted():
    import ast as _ast
    node = _ast.parse("a.b.c(1)", mode="eval").body
    assert _func_name(node.func) == "a.b.c"
