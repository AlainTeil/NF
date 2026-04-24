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
# developer workflow (pytest, pytest-cov, ruff, mypy, pre-commit, hypothesis)
pip install -e .[dev]

# add psutil-powered telemetry collectors
pip install -e .[dev,telemetry]

# enable YAML config files
pip install -e .[dev,yaml]
```

The editable install pulls in the runtime stack (NetworkX, pandas, seaborn, matplotlib) and developer extras (pytest, pytest-cov, ruff, mypy, pre-commit, hypothesis). The `telemetry` extra installs `psutil` so enhanced resource metrics can be captured, and `yaml` adds PyYAML for `*.yaml` configuration files.

## Quick Start

```bash
# Benchmark all default graphs and algorithms
python3 -m nf_bench.cli --repetitions 1 --seed 42 --output-dir reports

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

Example CLI invocation that enables all instrumentation (omit `--extra-reporter` unless you have a real reporter to register):

```bash
nf-bench --seed 7 --output-dir reports \
         --log-format json --log-level DEBUG \
         --enable-tracemalloc --enable-psutil
# To register your own reporter, replace ``my_project.reporters:export_results``
# with a real dotted path and whitelist its prefix:
#   export NF_BENCH_DYNAMIC_IMPORT_PREFIXES="my_project."
#   nf-bench ... --extra-reporter my_project.reporters:export_results
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

Run the automated test suite (pytest collects coverage by default; the 80% gate is enforced in CI):

```bash
pytest                       # full suite + coverage report
pytest --cov-fail-under=80   # also enforce the 80% gate locally
pytest -m "not slow"        # skip the slow-marked tests
pytest -m integration       # only end-to-end CLI tests (subset; gate not enforced)
ruff check .                 # lint
mypy nf_bench                # static type-check
pre-commit run --all-files  # run all hooks locally
```

The project targets Python 3.11+. `ruff`, `mypy`, `pytest-cov`, and `pre-commit` ship with the `dev` extra; install hooks once with `pre-commit install`. CI runs the same gates on every push/PR via `.github/workflows/ci.yml`.

## Exit Codes

`nf-bench` returns the following process exit codes:

| Code | Meaning                                                |
| ---- | ------------------------------------------------------ |
| `0`  | Success.                                               |
| `2`  | Argparse usage error (e.g. unknown flag).              |
| `3`  | Configuration error (file not found, invalid TOML…).   |
| `4`  | Benchmark runtime failure not handled by the runner.   |

When `--continue-on-error` mode is exposed via the Python API
(`run_benchmarks(..., continue_on_error=True)`), per-task failures are
collected on `df.attrs["failures"]` instead of aborting the run.

## Reproducibility & Metadata

Every run writes a `flow_metadata_<run_id>.json` file alongside the CSV/PNG
artifacts. The schema is intentionally stable so downstream tooling can rely
on it:

```jsonc
{
  "timestamp_utc": "2025-01-02T03:04:05+00:00",
  "run_id": "20250102T030405000000+0000",
  "git_commit": "abc1234",            // best-effort; "unknown" if not in a git tree
  "config_hash": "sha256-…",          // hash of the resolved BenchmarkConfig
  "config": { /* snapshot used for the run */ },
  "row_count": 42,
  "graph_count": 6,
  "telemetry": { /* optional, when --enable-* flags set */ },
  "artifacts": { /* paths written for this run */ }
}
```

All artifact files are written via temp-file + `os.replace`, so external
watchers never observe partial writes.

## Security Note: Dynamic Imports

Both `--extra-reporter` and the algorithm registry accept dotted paths that
are imported at runtime. To limit accidental code execution, nf-bench gates
imports through an allow-list:

- Anything under the built-in prefixes `networkx.` and `nf_bench.` is always
  allowed.
- To allow additional prefixes, set
  `NF_BENCH_DYNAMIC_IMPORT_PREFIXES="myorg.,otherpkg."`.
- To disable the gate entirely (not recommended in production), set
  `NF_BENCH_ALLOW_DYNAMIC_IMPORT=1` — a warning will be logged for every
  dynamic import so usage stays auditable.

Attempts to load a callable that fails the gate raise `DynamicImportError`.

## Troubleshooting

- **`ConfigError: pyyaml is required to load YAML configuration files`** —
  install the optional extra: `pip install -e .[yaml]`.
- **Plot is missing when running on a server** — nf-bench forces the
  matplotlib `Agg` backend before importing pyplot, but ensure your CI image
  has the matplotlib font cache writable (`MPLCONFIGDIR`).
- **`DynamicImportError` for an `--extra-reporter`** — see the security
  note above; whitelist your module via `NF_BENCH_DYNAMIC_IMPORT_PREFIXES`.
- **Unexpected results on Python `-O`** — assertions have been replaced by
  explicit `RuntimeError`s; `-O` is therefore safe to use.
- **`Unknown environment variable: NF_BENCH_…`** warning — indicates a
  typo in an `NF_BENCH_*` env var; check the option name.

## Data Flow

```
       ┌─────────────┐    ┌─────────────┐   ┌──────────────┐   ┌────────────────┐
CLI ──▶│ load_config │──▶ │ generators  │─▶ │  runner.run  │──▶│ ReportManager  │──▶ artifacts/
       │ (file/env/  │    │ build_test_ │   │ benchmarks() │   │ + metadata     │
       │  CLI merge) │    │ graphs()    │   │              │   │                │
       └─────────────┘    └─────────────┘   └──────┬───────┘   └────────────────┘
                                                   │
                                                   ▼
                                            ┌──────────────┐
                                            │  telemetry   │
                                            │ (tracemalloc │
                                            │  + psutil)   │
                                            └──────────────┘
```
