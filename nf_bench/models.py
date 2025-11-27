"""Data models used across the network-flow benchmarking suite."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    metrics: Dict[str, FlowMetric]


@dataclass(frozen=True)
class BenchmarkGraph:
    """Descriptors for generated test graphs."""

    name: str
    graph: nx.DiGraph
    source: int
    sink: int
    size_label: str
    density_label: str


@dataclass(frozen=True)
class BenchmarkConfig:
    """Runtime configuration for the CLI benchmark run."""

    repetitions: int
    seed: Optional[int]
    output_dir: Path
    graph_sizes: Optional[List[str]] = None
    graph_densities: Optional[List[str]] = None
    algorithms: Optional[List[str]] = None
    graph_families: Optional[List[str]] = None
    max_nodes: Optional[int] = None
    capacity_distribution: str = "uniform"
    edge_probability_overrides: Optional[Dict[str, float]] = None
    write_csv: bool = True
    write_plots: bool = True
    write_markdown: bool = False
    write_json: bool = False
    algorithm_registry: Optional[Dict[str, str]] = None
    concurrent_workers: Optional[int] = None
    extra_reporters: Optional[List[str]] = None
    log_level: str = "INFO"
    log_format: str = "text"
    enable_tracemalloc: bool = False
    enable_psutil: bool = False
