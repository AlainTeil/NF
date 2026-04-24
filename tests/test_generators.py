from __future__ import annotations

import networkx as nx
import pytest

from nf_bench.generators import (
    DEFAULT_DENSITY_PROBABILITIES,
    DEFAULT_NODE_SIZES,
    build_test_graphs,
    list_graph_family_names,
)


@pytest.mark.parametrize("family", list_graph_family_names())
@pytest.mark.parametrize("size", ["small"])
@pytest.mark.parametrize("density", sorted(DEFAULT_DENSITY_PROBABILITIES))
def test_build_test_graphs_matrix(family: str, size: str, density: str) -> None:
    """Every (family, size, density) triple yields a connected, source/sink-reachable graph."""

    graphs = build_test_graphs(
        seed=3,
        size_labels=[size],
        density_labels=[density],
        family_names=[family],
    )
    assert graphs, f"No graphs produced for {family}/{size}/{density}"
    for bg in graphs:
        assert bg.size_label == size
        assert bg.density_label == density
        # Source must reach sink in the directed view (post fallback if needed).
        assert nx.has_path(bg.graph, bg.source, bg.sink)
        for _, _, data in bg.graph.edges(data=True):
            assert data["capacity"] > 0


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
    with pytest.raises(ValueError, match="Unknown size labels"):
        build_test_graphs(size_labels=["invalid"])


@pytest.mark.slow
def test_build_test_graphs_max_nodes_filters() -> None:
    graphs = build_test_graphs(max_nodes=500)

    assert all(graph.graph.number_of_nodes() <= 500 for graph in graphs)


def test_build_test_graphs_max_nodes_invalid() -> None:
    with pytest.raises(ValueError, match="max_nodes"):
        build_test_graphs(max_nodes=1)


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


def test_default_node_sizes_are_known() -> None:
    # Sanity check that the size labels referenced by other tests exist.
    assert "small" in DEFAULT_NODE_SIZES
