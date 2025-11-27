"""Public API for the network-flow benchmarking package."""
from .models import BenchmarkConfig, BenchmarkGraph, FlowBenchmarkSummary, FlowMetric
from .generators import (
    DEFAULT_DENSITY_PROBABILITIES,
    DEFAULT_NODE_SIZES,
    build_demo_graph,
    build_test_graphs,
)
from .runner import (
    benchmark_flow_algorithms,
    list_flow_algorithm_names,
    register_flow_algorithm,
    resolve_flow_algorithms,
    run_benchmarks,
)
from .reporting import ReportManager, build_report_metadata, format_summary_table

__all__ = [
    "BenchmarkConfig",
    "FlowBenchmarkSummary",
    "FlowMetric",
    "BenchmarkGraph",
    "DEFAULT_NODE_SIZES",
    "DEFAULT_DENSITY_PROBABILITIES",
    "build_demo_graph",
    "build_test_graphs",
    "benchmark_flow_algorithms",
    "register_flow_algorithm",
    "list_flow_algorithm_names",
    "resolve_flow_algorithms",
    "run_benchmarks",
    "ReportManager",
    "build_report_metadata",
    "format_summary_table",
]
