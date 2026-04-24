"""Command-line interface for the network-flow benchmarking suite."""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from . import __version__
from ._encoding import safe_dumps
from ._imports import DynamicImportError
from .config import ConfigError, load_config
from .generators import (
    DEFAULT_DENSITY_PROBABILITIES,
    DEFAULT_NODE_SIZES,
    build_demo_graph,
    build_test_graphs,
    list_graph_family_names,
)
from .reporting import (
    ReportManager,
    build_report_metadata,
    format_summary_table,
)
from .runner import (
    benchmark_flow_algorithms,
    list_flow_algorithm_names,
    run_benchmarks,
)
from .telemetry import TelemetryCollector

LoggerCallback = Callable[[str, dict], None]

# Process exit codes returned by :func:`main`.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_RUNTIME = 4

_logger = logging.getLogger(__name__)


class JsonLogFormatter(logging.Formatter):
    """Simple JSON formatter for structured log lines."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return safe_dumps(payload, sort_keys=True)


def configure_logging(level_name: str = "INFO", log_format: str = "text") -> None:
    """Configure module-wide logging output with optional JSON formatting."""

    level = getattr(logging, level_name.upper(), logging.INFO)
    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def log_event(event: str, data: dict | None = None) -> None:
    """Emit a structured log line with JSON payload."""

    if data is None:
        data = {}
    payload = safe_dumps(data, sort_keys=True)
    logging.getLogger(__name__).info("%s | %s", event, payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments without resolving configuration defaults."""

    parser = argparse.ArgumentParser(
        description="Benchmark NetworkX maximum flow algorithms across generated graphs.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"nf-bench {__version__}",
    )
    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="List available max-flow algorithms and exit.",
    )
    parser.add_argument(
        "--list-graph-families",
        action="store_true",
        help="List available graph families and exit.",
    )
    parser.add_argument(
        "--list-graph-sizes",
        action="store_true",
        help="List available graph size labels and exit.",
    )
    parser.add_argument(
        "--list-graph-densities",
        action="store_true",
        help="List available graph density labels and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and exit without running benchmarks.",
    )
    parser.add_argument(
        "--demo",
        dest="run_demo",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run the demo graph before the benchmark suite (default: enabled).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML or YAML config file (default: None).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Number of times to repeat each benchmarked graph (default: config value or 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible graph generation (default: None).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saving benchmark artefacts (default: reports).",
    )
    parser.add_argument(
        "--graph-sizes",
        nargs="+",
        help=(
            "Subset of graph size labels to benchmark (default: all). Known: "
            f"{', '.join(sorted(DEFAULT_NODE_SIZES.keys()))}."
        ),
    )
    parser.add_argument(
        "--graph-densities",
        nargs="+",
        help=(
            "Subset of graph density labels to benchmark (default: all). Known: "
            f"{', '.join(sorted(DEFAULT_DENSITY_PROBABILITIES.keys()))}."
        ),
    )
    parser.add_argument(
        "--graph-families",
        nargs="+",
        help=(
            "Subset of graph family names to include (default: all). Known: "
            f"{', '.join(sorted(list_graph_family_names()))}."
        ),
    )
    available_algorithms = ", ".join(list_flow_algorithm_names())
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help=(
            "Subset of max-flow algorithms to benchmark (default: all). Known options: "
            f"{available_algorithms}"
        ),
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Upper bound on node count for generated graphs (default: unlimited).",
    )
    parser.add_argument(
        "--concurrent-workers",
        type=int,
        default=None,
        help="Number of parallel worker threads to use for benchmarks (default: serial).",
    )
    parser.add_argument(
        "--capacity-distribution",
        type=str,
        default=None,
        help="Capacity distribution label to use for edge weights (default: uniform).",
    )
    parser.add_argument(
        "--edge-probability",
        dest="edge_probabilities",
        action="append",
        metavar="FAMILY=P",
        help="Override edge probability per graph family, e.g. layered=0.3.",
    )
    parser.add_argument(
        "--write-csv",
        dest="write_csv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable CSV export (default: enabled).",
    )
    parser.add_argument(
        "--write-plots",
        dest="write_plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable PNG plot export (default: enabled).",
    )
    parser.add_argument(
        "--write-markdown",
        dest="write_markdown",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Markdown summary export (default: disabled).",
    )
    parser.add_argument(
        "--write-json",
        dest="write_json",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable JSON summary export (default: disabled).",
    )
    parser.add_argument(
        "--extra-reporter",
        dest="extra_reporters",
        action="append",
        metavar="MODULE:CALLABLE",
        help="Dotted path to additional reporter callable (can repeat).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Logging level to use (default: INFO).",
    )
    parser.add_argument(
        "--log-format",
        choices=["text", "json"],
        default=None,
        help="Logging format to use (default: text).",
    )
    parser.add_argument(
        "--enable-tracemalloc",
        dest="enable_tracemalloc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Python tracemalloc telemetry collection.",
    )
    parser.add_argument(
        "--enable-psutil",
        dest="enable_psutil",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable psutil-based telemetry (requires psutil).",
    )

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    log_callback: LoggerCallback | None = None,
) -> int:
    """CLI entry point for running the benchmark workflow.

    Returns the process exit code (0 success, 2 usage, 3 config, 4 runtime).
    """

    try:
        args = parse_args(argv)
    except SystemExit as exc:  # argparse already printed; propagate code
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE

    # Informational, exit-immediately flags.
    listing_handled = _handle_listing_flags(args)
    if listing_handled:
        return EXIT_OK

    try:
        config = load_config(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    configure_logging(config.log_level, config.log_format)
    _validate_known_choices(args)

    # Instantiate the report manager up front so a bad ``--extra-reporter``
    # surfaces immediately rather than after a long benchmark run.
    try:
        report_manager = ReportManager.from_config(config)
    except DynamicImportError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    _emit(
        log_callback,
        "configuration_loaded",
        {
            "repetitions": config.repetitions,
            "seed": config.seed,
            "output_dir": str(config.output_dir),
            "graph_sizes": config.graph_sizes,
            "graph_densities": config.graph_densities,
            "graph_families": config.graph_families,
            "algorithms": config.algorithms,
            "max_nodes": config.max_nodes,
            "capacity_distribution": config.capacity_distribution,
            "concurrent_workers": config.concurrent_workers,
            "extra_reporters": config.extra_reporters,
            "log_level": config.log_level,
            "log_format": config.log_format,
            "enable_tracemalloc": config.enable_tracemalloc,
            "enable_psutil": config.enable_psutil,
            "write_csv": config.write_csv,
            "write_plots": config.write_plots,
            "write_markdown": config.write_markdown,
            "write_json": config.write_json,
        },
    )

    if getattr(args, "dry_run", False):
        _logger.info("Dry run requested; configuration resolved successfully.")
        return EXIT_OK

    telemetry = TelemetryCollector(
        enable_tracemalloc=config.enable_tracemalloc,
        enable_psutil=config.enable_psutil,
    )
    telemetry.start()

    run_demo = True if getattr(args, "run_demo", None) is None else bool(args.run_demo)
    if run_demo:
        demo_graph = build_demo_graph()
        result = benchmark_flow_algorithms(demo_graph, source="s", sink="t")
        _emit(log_callback, "demo_completed", {"flow_value": result.flow_value})
        _logger.info("Demo result:\n%s", _format_metrics(result))

    _logger.info("Running benchmark suite across generated graphs...")

    graph_seed = config.seed if config.seed is not None else 42
    _emit(log_callback, "seed_initialized", {"seed": graph_seed})
    graphs = build_test_graphs(
        seed=graph_seed,
        size_labels=config.graph_sizes,
        density_labels=config.graph_densities,
        family_names=config.graph_families,
        max_nodes=config.max_nodes,
        edge_probability_overrides=config.edge_probability_overrides,
        capacity_distribution=config.capacity_distribution,
    )
    progress_tracker = {"last_percent": -1}

    def handle_progress(update: dict) -> None:
        completed = int(update.get("completed", 0))
        total = max(int(update.get("total", 1)), 1)
        percent = int((completed / total) * 100)
        if percent != progress_tracker["last_percent"]:
            progress_tracker["last_percent"] = percent
            _logger.info("Progress: %d%% (%d/%d)", percent, completed, total)
        _emit(log_callback, "progress_update", update)

    telemetry_data: dict[str, object] = {}
    summary_text = ""
    results = None
    try:
        results = run_benchmarks(
            graphs=graphs,
            repetitions=config.repetitions,
            algorithm_names=config.algorithms,
            max_nodes=config.max_nodes,
            event_logger=lambda event, data: _emit(log_callback, event, data),
            algorithm_registry=config.algorithm_registry,
            concurrent_workers=config.concurrent_workers,
            progress_callback=handle_progress,
        )
        _emit(log_callback, "benchmark_suite_completed", {"record_count": len(results)})
        summary_text = format_summary_table(results)
        _logger.info("Benchmark summary:\n%s", summary_text)
    finally:
        telemetry_data = telemetry.stop()

    if results is None:
        raise RuntimeError("Benchmark run did not produce a results frame.")
    metadata = build_report_metadata(
        config,
        row_count=len(results),
        graph_count=len(graphs),
        telemetry=telemetry_data,
    )
    artifacts = report_manager.emit(
        results,
        summary_text=summary_text,
        metadata=metadata,
    )
    _emit(log_callback, "telemetry_summary", telemetry_data)
    _emit(log_callback, "artifacts_saved", artifacts)
    for key, value in artifacts.items():
        if isinstance(value, list):
            for idx, item in enumerate(value, start=1):
                _logger.info("Saved %s[%d] to %s", key, idx, item)
        else:
            _logger.info("Saved %s to %s", key, value)
    return EXIT_OK


def _handle_listing_flags(args: argparse.Namespace) -> bool:
    """Print introspection listings if requested. Returns True if any printed."""

    listed = False
    if getattr(args, "list_algorithms", False):
        for name in list_flow_algorithm_names():
            print(name)
        listed = True
    if getattr(args, "list_graph_families", False):
        for name in list_graph_family_names():
            print(name)
        listed = True
    if getattr(args, "list_graph_sizes", False):
        for name in sorted(DEFAULT_NODE_SIZES):
            print(name)
        listed = True
    if getattr(args, "list_graph_densities", False):
        for name in sorted(DEFAULT_DENSITY_PROBABILITIES):
            print(name)
        listed = True
    return listed


def _validate_known_choices(args: argparse.Namespace) -> None:
    """Warn (do not abort) when CLI lists reference unknown labels."""

    _warn_unknown(args.graph_sizes, set(DEFAULT_NODE_SIZES), "graph size")
    _warn_unknown(args.graph_densities, set(DEFAULT_DENSITY_PROBABILITIES), "graph density")
    _warn_unknown(args.graph_families, set(list_graph_family_names()), "graph family")
    _warn_unknown(args.algorithms, set(list_flow_algorithm_names()), "algorithm")


def _warn_unknown(values: Sequence[str] | None, known: set, label: str) -> None:
    if not values:
        return
    for entry in values:
        if entry not in known:
            _logger.warning("Unknown %s: %s (known: %s)", label, entry, sorted(known))


def _emit(callback: LoggerCallback | None, event: str, data: dict) -> None:
    """Send structured logs to both stdlib logging and optional callback."""

    log_event(event, data)
    if callback is not None:
        callback(event, data)


def _format_metrics(summary) -> str:
    """Render benchmark results into a readable multiline string."""

    lines = [f"Max flow value: {summary.flow_value}", "Algorithm timings (s):"]
    for name, metric in summary.metrics.items():
        lines.append(f"  - {name:>24}: {metric.elapsed_seconds:.6f}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
