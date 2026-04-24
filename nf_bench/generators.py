"""Graph construction helpers for network-flow benchmarking."""
from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import networkx as nx

from .models import BenchmarkGraph

_logger = logging.getLogger(__name__)

# Default upper bound on retry attempts for connectivity-driven graph generation.
# Surfaced via :class:`GraphBuilderContext` so callers can override per run.
DEFAULT_MAX_GENERATION_ATTEMPTS = 20


DEFAULT_NODE_SIZES: dict[str, int] = {
    "small": 400,
    "medium": 800,
    "large": 1600,
}

DEFAULT_DENSITY_PROBABILITIES: dict[str, float] = {
    "sparse": 0.01,
    "medium": 0.03,
    "dense": 0.08,
}


GraphBuilder = Callable[["GraphBuilderContext"], list[BenchmarkGraph]]
GRAPH_REGISTRY: dict[str, GraphBuilder] = {}


@dataclass
class GraphBuilderContext:
    """Shared inputs passed to each registered graph builder."""

    family: str
    rng: random.Random
    capacity_range: tuple[int, int]
    node_sizes: dict[str, int]
    density_probabilities: dict[str, float]
    edge_probability_overrides: dict[str, float] | None
    capacity_distribution: str
    max_generation_attempts: int = DEFAULT_MAX_GENERATION_ATTEMPTS
    fallback_log: list[str] = field(default_factory=list)

    def probability_for_density(self, density_label: str) -> float:
        """Return resolved edge probability for the requested density label."""

        base = self.density_probabilities[density_label]
        override = _resolve_probability_override(
            self.edge_probability_overrides, self.family, density_label
        )
        return override if override is not None else base


def register_graph_family(name: str, builder: GraphBuilder) -> None:
    """Register a new graph family builder under the provided name."""

    GRAPH_REGISTRY[name] = builder


def list_graph_family_names() -> list[str]:
    """Return sorted graph family names currently registered."""

    return sorted(GRAPH_REGISTRY.keys())


def build_test_graphs(
    *,
    seed: int = 42,
    capacity_range: tuple[int, int] = (1, 50),
    node_sizes: dict[str, int] | None = None,
    density_probabilities: dict[str, float] | None = None,
    size_labels: list[str] | None = None,
    density_labels: list[str] | None = None,
    family_names: list[str] | None = None,
    max_nodes: int | None = None,
    edge_probability_overrides: dict[str, float] | None = None,
    capacity_distribution: str = "uniform",
) -> list[BenchmarkGraph]:
    """Create directed graphs of varying sizes and densities for benchmarking.

    Parameters
    ----------
    seed:
        Base seed used to initialise the random graph generator.
    capacity_range:
        Inclusive range of integer capacities assigned to edges.
    node_sizes:
        Optional mapping overriding the default node counts per size label.
    density_probabilities:
        Optional mapping overriding default edge probability per density label.
    size_labels:
        Optional subset of size labels to generate. Defaults to all available labels.
    density_labels:
        Optional subset of density labels to generate. Defaults to all available labels.
    family_names:
        Optional subset of registered graph families to generate. Defaults to all
        registered families.
    max_nodes:
        Optional upper bound on the number of nodes per generated graph. Labels
        exceeding the bound are discarded.
    edge_probability_overrides:
        Optional mapping of overrides for edge probabilities. Keys can reference
        `density`, `family`, or `family.density` to provide targeted overrides.
    capacity_distribution:
        Label describing the edge capacity distribution. Currently informational
        but propagated for future extensions.
    """

    node_sizes = dict(DEFAULT_NODE_SIZES) if node_sizes is None else dict(node_sizes)
    density_probabilities = (
        dict(DEFAULT_DENSITY_PROBABILITIES)
        if density_probabilities is None
        else dict(density_probabilities)
    )

    if size_labels is not None:
        missing_sizes = set(size_labels) - node_sizes.keys()
        if missing_sizes:
            raise ValueError(f"Unknown size labels requested: {sorted(missing_sizes)}")
        node_sizes = {label: node_sizes[label] for label in size_labels}

    if density_labels is not None:
        missing_densities = set(density_labels) - density_probabilities.keys()
        if missing_densities:
            raise ValueError(f"Unknown density labels requested: {sorted(missing_densities)}")
        density_probabilities = {
            label: density_probabilities[label] for label in density_labels
        }

    if max_nodes is not None:
        if max_nodes < 2:
            raise ValueError("max_nodes must be at least 2")
        node_sizes = {
            label: count for label, count in node_sizes.items() if count <= max_nodes
        }

    if not node_sizes:
        raise ValueError("No node sizes available after applying filters.")

    low, high = capacity_range
    if low <= 0 or high < low:
        raise ValueError("capacity_range must define positive integers in ascending order")

    available_families = list_graph_family_names()
    if not available_families:
        raise ValueError("No graph families registered.")

    selected_families = family_names or available_families
    missing = set(selected_families) - GRAPH_REGISTRY.keys()
    if missing:
        raise ValueError(f"Unknown graph families requested: {sorted(missing)}")

    rng = random.Random(seed)
    test_graphs: list[BenchmarkGraph] = []

    for family in selected_families:
        builder = GRAPH_REGISTRY[family]
        context = GraphBuilderContext(
            family=family,
            rng=random.Random(rng.randint(0, 2**32 - 1)),
            capacity_range=(low, high),
            node_sizes=dict(node_sizes),
            density_probabilities=dict(density_probabilities),
            edge_probability_overrides=edge_probability_overrides,
            capacity_distribution=capacity_distribution,
        )
        test_graphs.extend(builder(context))

    return test_graphs


def build_demo_graph() -> nx.DiGraph:
    """Construct a small demo graph with capacities for quick sanity checks."""

    g = nx.DiGraph()
    g.add_edge("s", "a", capacity=10)
    g.add_edge("s", "b", capacity=5)
    g.add_edge("a", "b", capacity=15)
    g.add_edge("a", "c", capacity=10)
    g.add_edge("b", "d", capacity=10)
    g.add_edge("c", "d", capacity=5)
    g.add_edge("c", "t", capacity=10)
    g.add_edge("d", "t", capacity=15)
    return g


def _generate_connected_digraph(
    *,
    n_nodes: int,
    edge_probability: float,
    rng: random.Random,
    capacity_range: tuple[int, int],
    source: int,
    sink: int,
    max_attempts: int = DEFAULT_MAX_GENERATION_ATTEMPTS,
) -> tuple[nx.DiGraph, bool]:
    """Generate a directed graph weakly connected between source and sink.

    Returns ``(graph, fallback)`` where ``fallback`` is ``True`` when the
    retry budget was exhausted and the linear-chain fallback was used. The
    fallback graph is no longer Erdos-Renyi-distributed; callers should record
    this in their metadata so downstream benchmarks aren't misinterpreted.
    """

    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2 to define distinct source and sink")
    if not 0 <= source < n_nodes or not 0 <= sink < n_nodes or source == sink:
        raise ValueError("source and sink must be distinct nodes within graph size")

    low, high = capacity_range

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))

    for _ in range(max_attempts):
        attempt_seed = rng.randint(0, 2**32 - 1)
        candidate = nx.gnp_random_graph(
            n=n_nodes,
            p=edge_probability,
            seed=attempt_seed,
            directed=True,
        )

        graph.clear()
        graph.add_nodes_from(candidate.nodes)
        graph.add_edges_from(candidate.edges)

        _assign_positive_capacities(graph, rng, low, high)
        _ensure_source_sink_connection(graph, source, sink, rng, low, high)

        if nx.has_path(nx.Graph(graph), source, sink):
            return graph, False

    _logger.warning(
        "Falling back to linear-chain graph for n=%s p=%s after %s attempts",
        n_nodes,
        edge_probability,
        max_attempts,
    )
    graph.clear()
    graph.add_nodes_from(range(n_nodes))
    for node in range(n_nodes - 1):
        if node == sink:
            continue
        graph.add_edge(node, node + 1, capacity=rng.randint(low, high))
    graph.add_edge(n_nodes - 2, sink, capacity=rng.randint(low, high))
    return graph, True


def _assign_positive_capacities(
    graph: nx.DiGraph,
    rng: random.Random,
    low: int,
    high: int,
) -> None:
    """Ensure every edge carries a positive integer capacity."""

    for u, v in graph.edges:
        graph[u][v]["capacity"] = rng.randint(low, high)


def _ensure_source_sink_connection(
    graph: nx.DiGraph,
    source: int,
    sink: int,
    rng: random.Random,
    low: int,
    high: int,
) -> None:
    """Add edges until the underlying undirected graph is connected.

    Uses ``nx.connected_components`` once per iteration; each iteration adds at
    least one cross-component edge, so the loop terminates in O(C) iterations
    where C is the initial number of components.
    """

    while True:
        undirected = graph.to_undirected(as_view=True)
        components = list(nx.connected_components(undirected))
        if len(components) <= 1:
            break
        left = rng.choice(tuple(components[0]))
        right = rng.choice(tuple(components[1]))
        graph.add_edge(left, right, capacity=rng.randint(low, high))
        if rng.random() < 0.5:
            graph.add_edge(right, left, capacity=rng.randint(low, high))

    if graph.out_degree(source) == 0:
        target_candidates = [node for node in graph.nodes if node != source]
        target = rng.choice(target_candidates)
        graph.add_edge(source, target, capacity=rng.randint(low, high))

    if graph.in_degree(sink) == 0:
        tail_candidates = [node for node in graph.nodes if node != sink]
        tail = rng.choice(tail_candidates)
        graph.add_edge(tail, sink, capacity=rng.randint(low, high))


def _resolve_probability_override(
    overrides: dict[str, float] | None,
    family: str,
    density_label: str,
) -> float | None:
    if not overrides:
        return None
    keys = (
        f"{family}.{density_label}",
        family,
        density_label,
    )
    for key in keys:
        if key in overrides:
            return overrides[key]
    return None


def _split_layers(n_nodes: int) -> list[list[int]]:
    """Partition ``range(n_nodes)`` into 2..6 contiguous layers.

    Targets roughly 150 nodes per layer; clamped to a minimum of 2 layers (so
    every layered DAG actually has a notion of source/sink layer) and a
    maximum of 6 (to keep small graphs from becoming too deep).
    """

    if n_nodes < 2:
        raise ValueError("Layered graphs require at least two nodes")

    target_layers = max(2, n_nodes // 150)
    layer_count = min(target_layers, 6, n_nodes)

    base, remainder = divmod(n_nodes, layer_count)
    layers: list[list[int]] = []
    current = 0
    for idx in range(layer_count):
        size = max(1, base + (1 if idx < remainder else 0))
        layers.append(list(range(current, current + size)))
        current += size
    if layers[-1][-1] != n_nodes - 1:
        layers[-1][-1] = n_nodes - 1
    return layers


def _graph_family(name: str):
    def decorator(func: GraphBuilder) -> GraphBuilder:
        register_graph_family(name, func)
        return func

    return decorator


@_graph_family("erdos_renyi")
def _build_erdos_renyi_graphs(context: GraphBuilderContext) -> list[BenchmarkGraph]:
    graphs: list[BenchmarkGraph] = []
    low, high = context.capacity_range

    for size_label, n_nodes in context.node_sizes.items():
        source = 0
        sink = n_nodes - 1

        for density_label in context.density_probabilities:
            edge_prob = context.probability_for_density(density_label)
            graph, fallback = _generate_connected_digraph(
                n_nodes=n_nodes,
                edge_probability=edge_prob,
                rng=context.rng,
                capacity_range=(low, high),
                source=source,
                sink=sink,
                max_attempts=context.max_generation_attempts,
            )
            name = f"erdos_renyi_{size_label}_{density_label}"
            if fallback:
                name += "_fallback"
                context.fallback_log.append(name)
            graphs.append(
                BenchmarkGraph(
                    name=name,
                    graph=graph,
                    source=source,
                    sink=sink,
                    size_label=size_label,
                    density_label=density_label,
                    fallback=fallback,
                )
            )

    return graphs


@_graph_family("layered_dag")
def _build_layered_dag_graphs(context: GraphBuilderContext) -> list[BenchmarkGraph]:
    graphs: list[BenchmarkGraph] = []
    low, high = context.capacity_range

    for size_label, n_nodes in context.node_sizes.items():
        layers = _split_layers(n_nodes)
        source = layers[0][0]
        sink = layers[-1][-1]

        for density_label in context.density_probabilities:
            prob = context.probability_for_density(density_label)
            g = nx.DiGraph()
            g.add_nodes_from(range(n_nodes))
            local_rng = random.Random(context.rng.randint(0, 2**32 - 1))

            # Spine guarantees a source -> ... -> sink path even when the
            # randomised inter-layer edges miss the sink entirely.
            spine_nodes = [layer[0] for layer in layers[:-1]] + [sink]
            for u, v in itertools.pairwise(spine_nodes):
                g.add_edge(u, v)

            for idx in range(len(layers) - 1):
                current_layer = layers[idx]
                next_layer = layers[idx + 1]
                edges_added = 0
                for u in current_layer:
                    for v in next_layer:
                        if local_rng.random() < prob:
                            g.add_edge(u, v)
                            edges_added += 1
                if edges_added == 0:
                    g.add_edge(
                        local_rng.choice(current_layer),
                        local_rng.choice(next_layer),
                    )

            if g.out_degree(source) == 0:
                g.add_edge(source, layers[1][0])
            if g.in_degree(sink) == 0:
                g.add_edge(layers[-2][-1], sink)

            _assign_positive_capacities(g, local_rng, low, high)
            graphs.append(
                BenchmarkGraph(
                    name=f"layered_dag_{size_label}_{density_label}",
                    graph=g,
                    source=source,
                    sink=sink,
                    size_label=size_label,
                    density_label=density_label,
                )
            )

    return graphs
