"""Property-based tests for graph generators using Hypothesis.

These tests assert *invariants* that must hold for every (family, size, density,
seed) combination, complementing the example-based tests in
``test_generators.py``. They favour small graphs and few examples to keep the
suite fast — Hypothesis is most valuable here as a regression net rather than
exhaustive search.
"""
from __future__ import annotations

import networkx as nx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nf_bench.generators import (
    DEFAULT_DENSITY_PROBABILITIES,
    DEFAULT_NODE_SIZES,
    build_test_graphs,
    list_graph_family_names,
)

_FAMILIES = sorted(list_graph_family_names())
_SIZES = sorted(DEFAULT_NODE_SIZES)
_DENSITIES = sorted(DEFAULT_DENSITY_PROBABILITIES)

_PROFILE = settings(
    max_examples=25,
    deadline=None,  # generators allocate full graphs; jitter is OK
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@_PROFILE
@given(
    family=st.sampled_from(_FAMILIES),
    size=st.sampled_from(_SIZES),
    density=st.sampled_from(_DENSITIES),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
def test_generated_graphs_satisfy_core_invariants(
    family: str, size: str, density: str, seed: int
) -> None:
    """Every produced graph satisfies the structural contract the runner relies on."""

    graphs = build_test_graphs(
        seed=seed,
        size_labels=[size],
        density_labels=[density],
        family_names=[family],
    )
    assert graphs, f"No graphs produced for {family}/{size}/{density}/seed={seed}"

    for bg in graphs:
        g = bg.graph

        # Identity / labelling
        assert bg.size_label == size
        assert bg.density_label == density
        assert bg.source != bg.sink, "source must differ from sink"
        assert bg.source in g, "source must be a node in the graph"
        assert bg.sink in g, "sink must be a node in the graph"

        # Structure
        assert g.number_of_nodes() >= 2, "graphs must have at least 2 nodes"
        assert isinstance(g, nx.DiGraph), "max-flow runner expects a directed graph"
        assert not any(u == v for u, v in g.edges()), "self-loops break flow algorithms"

        # Connectivity (undirected view — matches the runner's tolerance: some
        # families are intentionally allowed to leave the directed s-t path
        # empty; the runner reports 0 flow rather than failing).
        assert nx.has_path(nx.Graph(g), bg.source, bg.sink), (
            f"source/sink unreachable in undirected view for {bg.name}"
        )

        # Capacities — every edge must carry a strictly positive integer capacity
        # so flow algorithms never see zero/negative capacities.
        for _, _, data in g.edges(data=True):
            cap = data.get("capacity")
            assert isinstance(cap, int), f"capacity must be int, got {type(cap).__name__}"
            assert cap > 0, f"capacity must be positive, got {cap}"


@_PROFILE
@given(
    family=st.sampled_from(_FAMILIES),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
def test_generation_is_deterministic_for_fixed_seed(family: str, seed: int) -> None:
    """Two runs with the same seed produce identical graphs (edge sets + capacities)."""

    kwargs = dict(
        size_labels=["small"],
        density_labels=["sparse"],
        family_names=[family],
    )
    first = build_test_graphs(seed=seed, **kwargs)  # type: ignore[arg-type]
    second = build_test_graphs(seed=seed, **kwargs)  # type: ignore[arg-type]

    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert a.name == b.name
        assert a.source == b.source and a.sink == b.sink
        # Compare edges + capacities as a stable, hashable representation.
        edges_a = sorted((u, v, d.get("capacity")) for u, v, d in a.graph.edges(data=True))
        edges_b = sorted((u, v, d.get("capacity")) for u, v, d in b.graph.edges(data=True))
        assert edges_a == edges_b, f"non-deterministic generation for {family}@{seed}"


@_PROFILE
@given(max_nodes=st.integers(min_value=2, max_value=600))
def test_max_nodes_is_respected(max_nodes: int) -> None:
    """No produced graph exceeds ``max_nodes``; raises if no size fits."""

    try:
        graphs = build_test_graphs(
            seed=0,
            size_labels=None,
            density_labels=["sparse"],
            family_names=["erdos_renyi"],
            max_nodes=max_nodes,
        )
    except ValueError:
        # All known size labels exceeded ``max_nodes`` — acceptable behaviour.
        assert max_nodes < min(DEFAULT_NODE_SIZES.values())
        return

    for bg in graphs:
        assert bg.graph.number_of_nodes() <= max_nodes


def test_invalid_max_nodes_raises() -> None:
    """Boundary check pairs with the property test above."""

    with pytest.raises(ValueError, match="max_nodes"):
        build_test_graphs(seed=0, max_nodes=1)
