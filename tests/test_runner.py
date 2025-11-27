from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from nf_bench.generators import build_demo_graph
from nf_bench.models import BenchmarkGraph
from nf_bench.runner import benchmark_flow_algorithms, run_benchmarks


@pytest.fixture()
def simple_test_graph() -> BenchmarkGraph:
    graph = nx.DiGraph()
    graph.add_edge("s", "a", capacity=7)
    graph.add_edge("s", "b", capacity=5)
    graph.add_edge("a", "b", capacity=2)
    graph.add_edge("a", "t", capacity=4)
    graph.add_edge("b", "t", capacity=6)

    return BenchmarkGraph(
        name="simple",
        graph=graph,
        source="s",
        sink="t",
        size_label="test",
        density_label="test",
    )


def test_benchmark_flow_algorithms_matches_networkx() -> None:
    graph = build_demo_graph()
    summary = benchmark_flow_algorithms(graph, source="s", sink="t")

    expected_flow = nx.algorithms.flow.maximum_flow_value(graph, "s", "t")
    assert pytest.approx(expected_flow) == summary.flow_value
    assert all(metric.flow_value == summary.flow_value for metric in summary.metrics.values())



def test_run_benchmarks_returns_expected_dataframe(simple_test_graph: BenchmarkGraph) -> None:
    results = run_benchmarks(graphs=[simple_test_graph], repetitions=2)

    assert isinstance(results, pd.DataFrame)
    assert set(results["flow_value"].unique()) == {results.iloc[0]["flow_value"]}
    assert set(results["algorithm"]) == {
        "boykov_kolmogorov",
        "dinitz",
        "edmonds_karp",
        "preflow_push",
        "shortest_augmenting_path",
    }
    assert results["repetition"].nunique() == 2


def test_run_benchmarks_accepts_algorithm_subset(simple_test_graph: BenchmarkGraph) -> None:
    subset = ["edmonds_karp", "dinitz"]
    results = run_benchmarks(
        graphs=[simple_test_graph],
        repetitions=1,
        algorithm_names=subset,
    )

    assert set(results["algorithm"].unique()) == set(subset)


def test_run_benchmarks_respects_max_nodes() -> None:
    results = run_benchmarks(max_nodes=500, repetitions=1)

    assert results["node_count"].max() <= 500


def test_run_benchmarks_supports_custom_registry(simple_test_graph: BenchmarkGraph) -> None:
    registry_override = {"custom": "networkx.algorithms.flow.edmonds_karp"}
    results = run_benchmarks(
        graphs=[simple_test_graph],
        repetitions=1,
        algorithm_registry=registry_override,
        algorithm_names=["custom"],
    )

    assert set(results["algorithm"].unique()) == {"custom"}


def test_run_benchmarks_concurrent_threads(simple_test_graph: BenchmarkGraph) -> None:
    progress_updates = []

    df = run_benchmarks(
        graphs=[simple_test_graph],
        repetitions=3,
        concurrent_workers=2,
        progress_callback=lambda update: progress_updates.append(update),
    )

    assert not df.empty
    assert progress_updates
    assert progress_updates[-1]["completed"] == 3
    assert progress_updates[-1]["total"] == 3
