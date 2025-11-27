# nf-bench

Network Flow Bench (nf-bench) is a small benchmarking harness for comparing NetworkX maximum-flow algorithms across a catalogue of synthetic graphs.

## Features

- Generates directed graphs of multiple scales and densities with reproducible seeds
- Benchmarks Edmonds–Karp, Dinitz, Boykov–Kolmogorov, Shortest Augmenting Path, and Preflow-Push implementations
- Summarises runtimes in tabular output and persistent CSV/PNG artefacts
- Exposes ergonomic CLI flags for selecting graph sizes, densities, algorithms, repetitions, and maximum node counts
- Provides first-class observability: configurable log level/format plus optional telemetry collected via tracemalloc and psutil
- Offers a Python API for reuse in notebooks or custom pipelines

## Installation

```bash
# developer workflow (includes pytest + ruff)
pip install -e .[dev]

# add psutil-powered telemetry collectors
pip install -e .[dev,telemetry]

# opt into plotting stack separately when trimming dependencies
pip install -e .[dev,plots]
```

The editable install pulls in the runtime stack (NetworkX, pandas, seaborn, matplotlib) and developer extras (pytest, ruff). The `telemetry` extra installs `psutil` so that enhanced resource metrics can be captured, while the optional `plots` extra lets downstream automation defer heavyweight plotting libraries when they are not required.

## Quick Start

```bash
# Benchmark all default graphs and algorithms
python -m nf_bench.cli --repetitions 1 --seed 42 --output-dir reports

# Or use the packaged console script (after installation)
nf-bench --graph-sizes small medium --graph-densities sparse dense \
         --algorithms edmonds_karp dinitz --max-nodes 600 \
         --repetitions 3 --seed 99 --output-dir reports
```

Outputs include:

- Structured logs describing configuration and per-graph runtimes
- A human-readable summary table printed to stdout
- A timestamped CSV file and seaborn bar chart saved under `reports/`

## Observability Controls

nf-bench ships with instrumentation so large campaigns remain debuggable.

- `--log-level` and `--log-format` let you switch between human-friendly text logs and structured JSON, ensuring the same controls also work via config files and environment variables.
- Telemetry collection is opt-in. Pass `--enable-tracemalloc` to gather Python allocator stats and `--enable-psutil` (with the optional dependency installed) to add detailed CPU/memory/process metrics. Captured telemetry is embedded in the saved metadata so later analyses remain reproducible.
- Custom reporters can be layered on top of the built-ins with `--extra-reporter module:callable`. Each callable receives the full results DataFrame plus the generated metadata, making it easy to publish bespoke dashboards.

Example CLI invocation that enables all instrumentation:

```bash
nf-bench --seed 7 --output-dir reports \
         --log-format json --log-level DEBUG \
         --enable-tracemalloc --enable-psutil \
         --extra-reporter my_project.reporters:export_results
```

The same options can live in a TOML/YAML config file:

```toml
log_level = "INFO"
log_format = "json"
enable_tracemalloc = true
enable_psutil = true
```

## Configuration Workflow

Configuration follows a clear precedence order so that long-running campaigns stay reproducible:

1. **Config file** – point to a TOML or YAML manifest with `--config path` or by setting `NF_BENCH_CONFIG=/path/to/config.toml`.
2. **Environment variables** – any option can be prefixed with `NF_BENCH_`, e.g. `export NF_BENCH_GRAPH_SIZES="small,medium"`.
3. **CLI flags** – final overrides for ad-hoc experimentation.

A sample config (`configs/demo_config.toml`) ships with the repo:

```toml
[benchmark]
repetitions = 1
concurrent_workers = 2
output_dir = "reports"

[graph]
graph_sizes = ["small"]
graph_densities = ["sparse", "medium"]

[algorithms]
algorithms = ["edmonds_karp", "dinitz"]

[logging]
log_level = "INFO"
log_format = "text"
```

Run it via `nf-bench --config configs/demo_config.toml` or set `NF_BENCH_CONFIG` before invoking the CLI; additional CLI switches still take precedence when you need a quick override.

## Registry & Concurrency Notes

- **Algorithm registry:** the runner exposes a registry of NetworkX callables. Supply `--algorithms name1 name2` to filter, or map names to custom dotted paths under the `algorithms.registry` section of your config. This enables experimentation with patched or downstream implementations without editing the codebase.
- **Custom reporters:** any dotted `--extra-reporter module:callable` gets the results DataFrame + metadata, so integration with dashboards or artifact stores stays trivial.
- **Concurrency caveats:** `--concurrent-workers` leverages a thread pool to overlap graph runs. Because NetworkX releases the GIL during heavy linear-algebra work, moderate parallelism (≤ number of physical cores) can reduce wall-clock time, but each worker still builds full graphs in-memory. Keep an eye on RSS usage in telemetry, and prefer a small worker count when benchmarking particularly dense or user-defined generators.

## Python API

```python
from nf_bench import build_test_graphs, run_benchmarks, format_summary_table

graphs = build_test_graphs(seed=123, size_labels=["small"], density_labels=["sparse"], max_nodes=500)
results = run_benchmarks(graphs=graphs, repetitions=2, algorithm_names=["edmonds_karp", "dinitz"])
print(format_summary_table(results))
```

See the docstrings in `nf_bench/generators.py`, `nf_bench/runner.py`, and `nf_bench/reporting.py` for more details.

## Development

Run the automated test suite:

```bash
pytest
```

The project targets Python 3.11+. Static analysis via `ruff` (installed with the `dev` extra) keeps modules consistent; other formatters such as `black` can be layered on if desired.
