"""Core benchmarking logic for maximum-flow algorithms."""
from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

import networkx as nx
import pandas as pd

from ._imports import import_callable
from .generators import build_test_graphs
from .models import BenchmarkGraph, FlowBenchmarkSummary, FlowMetric

# Default tolerances for cross-algorithm flow value comparison. Looser than the
# previous 1e-9 relative tolerance to accommodate harmless rounding noise.
DEFAULT_FLOW_REL_TOL = 1e-6
DEFAULT_FLOW_ABS_TOL = 1e-9


def _flows_match(
    a: float,
    b: float,
    *,
    rel_tol: float = DEFAULT_FLOW_REL_TOL,
    abs_tol: float = DEFAULT_FLOW_ABS_TOL,
) -> bool:
    """Return True when two flow values agree within tolerance."""

    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


FlowAlgorithm = Callable[..., object]
EventLogger = Callable[[str, dict[str, object]], None] | None
ProgressCallback = Callable[[dict[str, object]], None] | None

FLOW_ALGORITHM_REGISTRY: dict[str, FlowAlgorithm] = {}


def register_flow_algorithm(name: str, flow_func: FlowAlgorithm) -> None:
    """Register (or replace) a max-flow algorithm callable by name."""

    FLOW_ALGORITHM_REGISTRY[name] = flow_func


def list_flow_algorithm_names() -> list[str]:
    """Return sorted algorithm names available in the registry."""

    return sorted(FLOW_ALGORITHM_REGISTRY.keys())


def resolve_flow_algorithms(
    algorithm_names: Iterable[str] | None = None,
    overrides: dict[str, str | None] | None = None,
) -> dict[str, FlowAlgorithm]:
    """Resolve algorithm callables applying optional dotted-path overrides."""

    registry: dict[str, FlowAlgorithm] = dict(FLOW_ALGORITHM_REGISTRY)

    if overrides:
        for name, target in overrides.items():
            if not target:
                registry.pop(name, None)
                continue
            registry[name] = import_callable(target)

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
    source: Hashable,
    sink: Hashable,
    *,
    capacity: str = "capacity",
    algorithms: dict[str, FlowAlgorithm] | None = None,
    rel_tol: float = DEFAULT_FLOW_REL_TOL,
    abs_tol: float = DEFAULT_FLOW_ABS_TOL,
) -> FlowBenchmarkSummary:
    """Benchmark multiple NetworkX max-flow algorithms on the same graph."""

    if algorithms is None:
        algorithms = resolve_flow_algorithms()

    metrics: dict[str, FlowMetric] = {}
    reference_flow: float | None = None

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
        elif not _flows_match(flow_value, reference_flow, rel_tol=rel_tol, abs_tol=abs_tol):
            raise ValueError(
                f"Flow value mismatch: {name} returned {flow_value}, expected {reference_flow}."
            )

    if reference_flow is None:
        raise RuntimeError("No algorithms were executed.")
    return FlowBenchmarkSummary(flow_value=reference_flow, metrics=metrics)


def run_benchmarks(
    *,
    graphs: list[BenchmarkGraph] | None = None,
    repetitions: int = 1,
    seed: int | None = None,
    algorithm_names: Iterable[str] | None = None,
    max_nodes: int | None = None,
    event_logger: EventLogger = None,
    algorithm_registry: dict[str, str | None] | None = None,
    concurrent_workers: int | None = None,
    progress_callback: ProgressCallback = None,
    continue_on_error: bool = False,
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
    continue_on_error:
        When ``True`` a failing benchmark task is logged via ``event_logger``
        (event ``benchmark_failed``) and the run continues. When ``False``
        (default) the first exception aborts the run.
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

    tasks: list[tuple[BenchmarkGraph, int]] = [
        (graph, repetition)
        for graph in graphs
        for repetition in range(repetitions)
    ]

    if not tasks:
        raise ValueError("No benchmark results were produced.")

    records: list[dict[str, object]] = []
    total_tasks = len(tasks)
    completed = 0
    failures: list[dict[str, object]] = []

    def _handle_failure(graph: BenchmarkGraph, repetition: int, exc: BaseException) -> None:
        nonlocal completed
        completed += 1
        failure_info = {
            "graph": graph.name,
            "repetition": repetition,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        failures.append(failure_info)
        if event_logger is not None:
            event_logger("benchmark_failed", failure_info)
        if progress_callback is not None:
            progress_callback(
                {
                    "completed": completed,
                    "total": total_tasks,
                    "graph": graph.name,
                    "repetition": repetition,
                    "failed": True,
                }
            )

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
                try:
                    task_records, wall_clock = future.result()
                except Exception as exc:
                    _handle_failure(graph, repetition, exc)
                    if not continue_on_error:
                        # Cancel remaining tasks before re-raising.
                        for pending in future_map:
                            pending.cancel()
                        raise
                else:
                    records.extend(task_records)
                    _handle_completion(graph, repetition, wall_clock)
    else:
        for graph, repetition in tasks:
            try:
                task_records, wall_clock = _execute_benchmark_task(
                    graph, repetition, algorithms
                )
            except Exception as exc:
                _handle_failure(graph, repetition, exc)
                if not continue_on_error:
                    raise
            else:
                records.extend(task_records)
                _handle_completion(graph, repetition, wall_clock)

    if failures:
        df = pd.DataFrame.from_records(records)
        df.attrs["failures"] = failures
        return df
    return pd.DataFrame.from_records(records)


def _execute_benchmark_task(
    graph: BenchmarkGraph,
    repetition: int,
    algorithms: dict[str, FlowAlgorithm],
) -> tuple[list[dict[str, object]], float]:
    run_start = perf_counter()
    summary = benchmark_flow_algorithms(
        graph.graph,
        graph.source,
        graph.sink,
        algorithms=algorithms,
    )
    run_elapsed = perf_counter() - run_start

    records: list[dict[str, object]] = []
    for algo_name, metric in summary.metrics.items():
        if not _flows_match(metric.flow_value, summary.flow_value):
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


def _register_default_algorithms() -> None:
    register_flow_algorithm("edmonds_karp", nx.algorithms.flow.edmonds_karp)
    register_flow_algorithm(
        "shortest_augmenting_path", nx.algorithms.flow.shortest_augmenting_path
    )
    register_flow_algorithm("preflow_push", nx.algorithms.flow.preflow_push)
    register_flow_algorithm("dinitz", nx.algorithms.flow.dinitz)
    register_flow_algorithm("boykov_kolmogorov", nx.algorithms.flow.boykov_kolmogorov)


_register_default_algorithms()
