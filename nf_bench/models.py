"""Data models used across the network-flow benchmarking suite."""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass(frozen=True)
class FlowMetric:
    """Stores the resulting max-flow value and elapsed wall-clock time."""

    flow_value: float
    elapsed_seconds: float


@dataclass(frozen=True)
class FlowBenchmarkSummary:
    """Aggregates a shared flow value and per-algorithm timing metrics."""

    flow_value: float
    metrics: dict[str, FlowMetric]


@dataclass(frozen=True)
class BenchmarkGraph:
    """Descriptors for generated test graphs.

    The wrapped ``graph`` itself is a mutable :class:`networkx.DiGraph`; the
    dataclass is frozen only to prevent accidental rebinding of the descriptor
    fields.
    """

    name: str
    graph: nx.DiGraph
    source: Hashable
    sink: Hashable
    size_label: str
    density_label: str
    fallback: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    """Runtime configuration for the CLI benchmark run."""

    repetitions: int
    seed: int | None
    output_dir: Path
    graph_sizes: list[str] | None = None
    graph_densities: list[str] | None = None
    algorithms: list[str] | None = None
    graph_families: list[str] | None = None
    max_nodes: int | None = None
    capacity_distribution: str = "uniform"
    edge_probability_overrides: dict[str, float] | None = None
    write_csv: bool = True
    write_plots: bool = True
    write_markdown: bool = False
    write_json: bool = False
    algorithm_registry: dict[str, str | None] | None = None
    concurrent_workers: int | None = None
    extra_reporters: list[str] | None = None
    log_level: str = "INFO"
    log_format: str = "text"
    enable_tracemalloc: bool = False
    enable_psutil: bool = False
    flow_rel_tol: float = 1e-6
    flow_abs_tol: float = 1e-9
