"""Tests for benchmarking failure handling."""
from __future__ import annotations

import networkx as nx
import pytest

from nf_bench.models import BenchmarkGraph
from nf_bench.runner import benchmark_flow_algorithms, run_benchmarks


def _toy_graph() -> BenchmarkGraph:
    g = nx.DiGraph()
    g.add_edge("s", "a", capacity=3)
    g.add_edge("a", "t", capacity=3)
    return BenchmarkGraph(
        name="toy",
        graph=g,
        source="s",
        sink="t",
        size_label="test",
        density_label="test",
    )


def _failing(*_args, **_kwargs):
    raise RuntimeError("boom")


def test_run_benchmarks_continue_on_error_records_failures() -> None:
    graph = _toy_graph()
    registry = {"bad": "tests.test_runner_failures._failing"}
    df = run_benchmarks(
        graphs=[graph],
        repetitions=1,
        algorithm_registry=registry,
        algorithm_names=["bad"],
        continue_on_error=True,
    )
    failures = df.attrs.get("failures", [])
    assert failures, "continue_on_error should record at least one failure"
    assert any("boom" in str(f.get("error", "")) for f in failures)


def test_run_benchmarks_default_raises_on_error() -> None:
    graph = _toy_graph()
    registry = {"bad": "tests.test_runner_failures._failing"}
    with pytest.raises(RuntimeError, match="boom"):
        run_benchmarks(
            graphs=[graph],
            repetitions=1,
            algorithm_registry=registry,
            algorithm_names=["bad"],
        )


def test_benchmark_flow_algorithms_tolerance_kwargs_accepted() -> None:
    graph = _toy_graph().graph
    summary = benchmark_flow_algorithms(graph, source="s", sink="t", rel_tol=1e-4, abs_tol=1e-9)
    assert summary.flow_value == 3
