"""Benchmark adapters: pluggable so the choice of benchmark isn't load-bearing."""

from .base import Benchmark, Task, Verdict
from .mock_bench import MockBenchmark
from .bfcl import BFCLBenchmark

__all__ = ["Benchmark", "Task", "Verdict", "MockBenchmark", "BFCLBenchmark"]
