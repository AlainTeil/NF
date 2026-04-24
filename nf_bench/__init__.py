"""Public API for the network-flow benchmarking package."""
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .generators import (
    DEFAULT_DENSITY_PROBABILITIES,
    DEFAULT_NODE_SIZES,
    build_demo_graph,
    build_test_graphs,
)
from .models import BenchmarkConfig, BenchmarkGraph, FlowBenchmarkSummary, FlowMetric
from .reporting import ReportManager, build_report_metadata, format_summary_table
from .runner import (
    benchmark_flow_algorithms,
    list_flow_algorithm_names,
    register_flow_algorithm,
    resolve_flow_algorithms,
    run_benchmarks,
)

try:
    __version__ = _pkg_version("nf-bench")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "DEFAULT_DENSITY_PROBABILITIES",
    "DEFAULT_NODE_SIZES",
    "BenchmarkConfig",
    "BenchmarkGraph",
    "FlowBenchmarkSummary",
    "FlowMetric",
    "ReportManager",
    "__version__",
    "benchmark_flow_algorithms",
    "build_demo_graph",
    "build_report_metadata",
    "build_test_graphs",
    "format_summary_table",
    "list_flow_algorithm_names",
    "register_flow_algorithm",
    "resolve_flow_algorithms",
    "run_benchmarks",
]
