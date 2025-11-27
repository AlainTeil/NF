from __future__ import annotations

import networkx as nx
import pytest

from nf_bench.generators import build_test_graphs, list_graph_family_names


def test_build_test_graphs_produces_connected_graphs() -> None:
    graphs = build_test_graphs(
        seed=7,
        node_sizes={"tiny": 12},
        density_probabilities={"medium": 0.2},
        family_names=["erdos_renyi"],
    )

    assert len(graphs) == 1
    test_graph = graphs[0]

    assert nx.has_path(nx.Graph(test_graph.graph), test_graph.source, test_graph.sink)

    for _, _, data in test_graph.graph.edges(data=True):
        assert data["capacity"] > 0


def test_build_test_graphs_filtering() -> None:
    graphs = build_test_graphs(
        seed=11,
        size_labels=["small"],
        density_labels=["sparse"],
        family_names=["erdos_renyi"],
    )

    assert len(graphs) == 1
    assert graphs[0].size_label == "small"
    assert graphs[0].density_label == "sparse"


def test_build_test_graphs_invalid_filter_raises() -> None:
    try:
        build_test_graphs(size_labels=["invalid"])
    except ValueError as exc:
        assert "Unknown size labels" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown size label")


def test_build_test_graphs_max_nodes_filters() -> None:
    graphs = build_test_graphs(max_nodes=500)

    assert all(graph.graph.number_of_nodes() <= 500 for graph in graphs)


def test_build_test_graphs_max_nodes_invalid() -> None:
    try:
        build_test_graphs(max_nodes=1)
    except ValueError as exc:
        assert "max_nodes" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid max_nodes")


def test_build_test_graphs_invalid_family() -> None:
    with pytest.raises(ValueError):
        build_test_graphs(family_names=["invalid_family"])


def test_layered_dag_family_generates_unique_names() -> None:
    graphs = build_test_graphs(
        seed=17,
        node_sizes={"tiny": 20},
        density_probabilities={"medium": 0.2},
        family_names=["layered_dag"],
    )

    assert graphs
    assert all(graph.name.startswith("layered_dag_") for graph in graphs)


def test_list_graph_family_names_includes_defaults() -> None:
    families = list_graph_family_names()
    assert "erdos_renyi" in families
    assert "layered_dag" in families
