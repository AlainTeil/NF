"""Core benchmarking logic for maximum-flow algorithms."""
from __future__ import annotations

import importlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd

from .generators import build_test_graphs
from .models import BenchmarkGraph, FlowBenchmarkSummary, FlowMetric


FlowAlgorithm = Callable[..., object]
EventLogger = Optional[Callable[[str, Dict[str, object]], None]]
ProgressCallback = Optional[Callable[[Dict[str, object]], None]]

FLOW_ALGORITHM_REGISTRY: Dict[str, FlowAlgorithm] = {}


def register_flow_algorithm(name: str, flow_func: FlowAlgorithm) -> None:
    """Register (or replace) a max-flow algorithm callable by name."""

    FLOW_ALGORITHM_REGISTRY[name] = flow_func


def list_flow_algorithm_names() -> List[str]:
    """Return sorted algorithm names available in the registry."""

    return sorted(FLOW_ALGORITHM_REGISTRY.keys())


def resolve_flow_algorithms(
    algorithm_names: Optional[Iterable[str]] = None,
    overrides: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, FlowAlgorithm]:
    """Resolve algorithm callables applying optional dotted-path overrides."""

    registry: Dict[str, FlowAlgorithm] = dict(FLOW_ALGORITHM_REGISTRY)

    if overrides:
        for name, target in overrides.items():
            if not target:
                registry.pop(name, None)
                continue
            registry[name] = _import_from_string(target)

    if not registry:
        raise ValueError("No flow algorithms registered. Provide at least one entry.")

    if algorithm_names is None:
        return registry

    selected = list(algorithm_names)
    missing = set(selected) - registry.keys()
    if missing:
        raise ValueError(f"Unknown algorithms requested: {sorted(missing)}")

    return {name: registry[name] for name in selected}


def benchmark_flow_algorithms(
    graph: nx.DiGraph,
    source: int,
    sink: int,
    *,
    capacity: str = "capacity",
    algorithms: Optional[Dict[str, FlowAlgorithm]] = None,
) -> FlowBenchmarkSummary:
    """Benchmark multiple NetworkX max-flow algorithms on the same graph."""

    if algorithms is None:
        algorithms = resolve_flow_algorithms()

    metrics: Dict[str, FlowMetric] = {}
    reference_flow: Optional[float] = None

    for name, flow_func in algorithms.items():
        start = perf_counter()
        flow_value = nx.algorithms.flow.maximum_flow_value(
            graph,
            source,
            sink,
            capacity=capacity,
            flow_func=flow_func,
        )
        elapsed = perf_counter() - start
        metrics[name] = FlowMetric(flow_value=flow_value, elapsed_seconds=elapsed)

        if reference_flow is None:
            reference_flow = flow_value
        elif not math.isclose(flow_value, reference_flow, rel_tol=1e-9, abs_tol=0.0):
            raise ValueError(
                f"Flow value mismatch: {name} returned {flow_value}, expected {reference_flow}."
            )

    assert reference_flow is not None, "No algorithms were executed."
    return FlowBenchmarkSummary(flow_value=reference_flow, metrics=metrics)


def run_benchmarks(
    *,
    graphs: Optional[List[BenchmarkGraph]] = None,
    repetitions: int = 1,
    seed: Optional[int] = None,
    algorithm_names: Optional[Iterable[str]] = None,
    max_nodes: Optional[int] = None,
    event_logger: EventLogger = None,
    algorithm_registry: Optional[Dict[str, Optional[str]]] = None,
    concurrent_workers: Optional[int] = None,
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """Execute the full benchmark suite and return results as a DataFrame.

    Parameters
    ----------
    graphs:
        Optional list of pre-built graphs to benchmark. When provided other
        graph-generation options are ignored.
    repetitions:
        Number of times to repeat each graph benchmark.
    seed:
        Random seed used when generating graphs on demand.
    algorithm_names:
        Optional iterable restricting which NetworkX max-flow algorithms are
        executed. Defaults to all supported algorithms.
    max_nodes:
        Optional upper bound on node count for generated graphs when ``graphs``
        is omitted.
    event_logger:
        Optional callback notified after each graph benchmark completes.
    algorithm_registry:
        Optional mapping of algorithm name to dotted import path. ``None`` or empty
        values remove an algorithm from the registry.
    concurrent_workers:
        Optional number of worker threads to use for concurrent benchmark tasks.
    progress_callback:
        Optional callback invoked with ``{"completed": int, "total": int, ...}``
        after each benchmark task finishes.
    """

    if repetitions < 1:
        raise ValueError("repetitions must be a positive integer")

    if graphs is None:
        graphs = build_test_graphs(
            seed=seed if seed is not None else 42,
            max_nodes=max_nodes,
        )

    algorithms = resolve_flow_algorithms(algorithm_names, algorithm_registry)

    if concurrent_workers is not None and concurrent_workers < 1:
        raise ValueError("concurrent_workers must be a positive integer")

    tasks: List[Tuple[BenchmarkGraph, int]] = [
        (graph, repetition)
        for graph in graphs
        for repetition in range(repetitions)
    ]

    if not tasks:
        raise ValueError("No benchmark results were produced.")

    records: List[Dict[str, object]] = []
    total_tasks = len(tasks)
    completed = 0

    def _handle_completion(graph: BenchmarkGraph, repetition: int, wall_clock: float) -> None:
        nonlocal completed
        completed += 1
        if event_logger is not None:
            event_logger(
                "benchmark_completed",
                {
                    "graph": graph.name,
                    "repetition": repetition,
                    "wall_clock_seconds": wall_clock,
                },
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "completed": completed,
                    "total": total_tasks,
                    "graph": graph.name,
                    "repetition": repetition,
                }
            )

    if concurrent_workers and total_tasks > 1:
        # Thread pool lets NetworkX release the GIL during heavy work, so
        # overlapping tasks shortens wall-clock time without changing semantics.
        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            future_map = {
                executor.submit(
                    _execute_benchmark_task,
                    graph,
                    repetition,
                    algorithms,
                ): (graph, repetition)
                for graph, repetition in tasks
            }
            for future in as_completed(future_map):
                graph, repetition = future_map[future]
                task_records, wall_clock = future.result()
                records.extend(task_records)
                _handle_completion(graph, repetition, wall_clock)
    else:
        for graph, repetition in tasks:
            task_records, wall_clock = _execute_benchmark_task(graph, repetition, algorithms)
            records.extend(task_records)
            _handle_completion(graph, repetition, wall_clock)

    return pd.DataFrame.from_records(records)


def _execute_benchmark_task(
    graph: BenchmarkGraph,
    repetition: int,
    algorithms: Dict[str, FlowAlgorithm],
) -> Tuple[List[Dict[str, object]], float]:
    run_start = perf_counter()
    summary = benchmark_flow_algorithms(
        graph.graph,
        graph.source,
        graph.sink,
        algorithms=algorithms,
    )
    run_elapsed = perf_counter() - run_start

    records: List[Dict[str, object]] = []
    for algo_name, metric in summary.metrics.items():
        if not math.isclose(metric.flow_value, summary.flow_value, rel_tol=1e-9, abs_tol=0.0):
            raise ValueError(
                f"Flow mismatch for {graph.name} via {algo_name}: "
                f"{metric.flow_value} != {summary.flow_value}"
            )

        records.append(
            {
                "graph": graph.name,
                "size_label": graph.size_label,
                "density_label": graph.density_label,
                "node_count": graph.graph.number_of_nodes(),
                "edge_count": graph.graph.number_of_edges(),
                "algorithm": algo_name,
                "flow_value": metric.flow_value,
                "elapsed_seconds": metric.elapsed_seconds,
                "benchmark_wall_clock": run_elapsed,
                "repetition": repetition,
            }
        )

    return records, run_elapsed


def _import_from_string(path: str) -> FlowAlgorithm:
    module_path, _, attribute = path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Algorithm import path '{path}' must include both module and attribute"
        )
    module = importlib.import_module(module_path)
    try:
        func = getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - defensive programming
        raise ImportError(f"Cannot import '{attribute}' from '{module_path}'") from exc
    return func


def _register_default_algorithms() -> None:
    register_flow_algorithm("edmonds_karp", nx.algorithms.flow.edmonds_karp)
    register_flow_algorithm(
        "shortest_augmenting_path", nx.algorithms.flow.shortest_augmenting_path
    )
    register_flow_algorithm("preflow_push", nx.algorithms.flow.preflow_push)
    register_flow_algorithm("dinitz", nx.algorithms.flow.dinitz)
    register_flow_algorithm("boykov_kolmogorov", nx.algorithms.flow.boykov_kolmogorov)


_register_default_algorithms()
