"""End-to-end integration tests for the CLI ``main`` entry point.

These exercise the full benchmark pipeline (config → generators → runner →
reporting) through ``main()`` with a deliberately tiny workload so the suite
stays fast while filling the coverage gap in ``nf_bench/cli.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nf_bench.cli import EXIT_OK, EXIT_USAGE, main

_MIN_ARGS = [
    "--no-demo",
    "--graph-sizes",
    "small",
    "--graph-densities",
    "sparse",
    "--graph-families",
    "erdos_renyi",
    "--algorithms",
    "edmonds_karp",
    "--repetitions",
    "1",
    "--max-nodes",
    "500",
    "--seed",
    "7",
]


def _run(tmp_path: Path, *extra: str) -> int:
    return main([*_MIN_ARGS, "--output-dir", str(tmp_path), *extra])


@pytest.mark.integration
def test_main_full_run_writes_csv_and_metadata(tmp_path: Path) -> None:
    code = _run(tmp_path)
    assert code == EXIT_OK

    files = list(tmp_path.iterdir())
    csvs = [p for p in files if p.suffix == ".csv"]
    metadata_files = [p for p in files if p.name.startswith("flow_metadata_")]
    assert csvs, f"expected a CSV artifact, got: {files}"
    assert metadata_files, f"expected a metadata JSON, got: {files}"

    payload = json.loads(metadata_files[0].read_text())
    assert payload["row_count"] >= 1
    assert payload["graph_count"] >= 1
    assert "config" in payload and "config_hash" in payload


@pytest.mark.integration
def test_main_demo_branch_runs(tmp_path: Path) -> None:
    code = main(
        [
            "--demo",
            "--dry-run",  # exit before the suite, but after the demo gate path
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == EXIT_OK


@pytest.mark.integration
def test_main_disable_writers_produces_no_csv(tmp_path: Path) -> None:
    code = _run(
        tmp_path,
        "--no-write-csv",
        "--no-write-plots",
        "--no-write-markdown",
        "--no-write-json",
    )
    assert code == EXIT_OK
    assert [p for p in tmp_path.iterdir() if p.suffix == ".csv"] == []


@pytest.mark.integration
def test_main_with_config_file(tmp_path: Path) -> None:
    config = tmp_path / "cfg.toml"
    config.write_text(
        """
        [benchmark]
        repetitions = 1
        seed = 11
        output_dir = "{out}"

        [graph]
        graph_sizes = ["small"]
        graph_densities = ["sparse"]
        graph_families = ["erdos_renyi"]
        max_nodes = 500

        [algorithms]
        algorithms = ["edmonds_karp"]
        """.format(out=str(tmp_path).replace("\\", "\\\\"))
    )
    code = main(["--no-demo", "--config", str(config)])
    assert code == EXIT_OK
    assert [p for p in tmp_path.iterdir() if p.suffix == ".csv"], list(tmp_path.iterdir())


@pytest.mark.integration
def test_main_list_graph_sizes(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--list-graph-sizes"])
    assert code == EXIT_OK
    assert "small" in capsys.readouterr().out


@pytest.mark.integration
def test_main_list_graph_densities(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--list-graph-densities"])
    assert code == EXIT_OK
    assert "sparse" in capsys.readouterr().out


@pytest.mark.integration
def test_main_unknown_flag_returns_usage_exit_code() -> None:
    assert main(["--definitely-not-a-flag"]) == EXIT_USAGE


@pytest.mark.integration
def test_main_log_callback_receives_events(tmp_path: Path) -> None:
    events: list[tuple[str, dict]] = []

    code = main(
        [*_MIN_ARGS, "--output-dir", str(tmp_path)],
        log_callback=lambda event, data: events.append((event, dict(data))),
    )
    assert code == EXIT_OK

    names = [e for e, _ in events]
    assert "configuration_loaded" in names
    assert "benchmark_suite_completed" in names
    assert "artifacts_saved" in names
