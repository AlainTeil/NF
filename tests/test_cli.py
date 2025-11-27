from __future__ import annotations

from pathlib import Path

from nf_bench.cli import parse_args


def test_parse_args_defaults(tmp_path: Path) -> None:
    args = parse_args([])

    assert args.repetitions is None
    assert args.seed is None
    assert args.output_dir is None
    assert args.graph_sizes is None
    assert args.graph_densities is None
    assert args.graph_families is None
    assert args.algorithms is None
    assert args.max_nodes is None
    assert args.capacity_distribution is None
    assert args.edge_probabilities is None
    assert args.write_csv is None
    assert args.write_plots is None
    assert args.write_markdown is None
    assert args.write_json is None
    assert args.concurrent_workers is None
    assert args.extra_reporters is None
    assert args.log_level is None
    assert args.log_format is None
    assert args.enable_tracemalloc is None
    assert args.enable_psutil is None


def test_parse_args_custom(tmp_path: Path) -> None:
    target_dir = tmp_path / "out"
    args = parse_args([
        "--config",
        str(tmp_path / "config.toml"),
        "--repetitions",
        "5",
        "--seed",
        "13",
        "--output-dir",
        str(target_dir),
        "--graph-sizes",
        "small",
        "medium",
        "--graph-densities",
        "sparse",
        "dense",
        "--graph-families",
        "erdos_renyi",
        "--algorithms",
        "edmonds_karp",
        "dinitz",
        "--max-nodes",
        "500",
        "--capacity-distribution",
        "triangular",
        "--edge-probability",
        "grid=0.4",
        "--edge-probability",
        "dense=0.9",
        "--write-csv",
        "--no-write-plots",
        "--write-markdown",
        "--write-json",
        "--concurrent-workers",
        "4",
        "--extra-reporter",
        "tests.test_reporting.custom_reporter",
        "--log-level",
        "DEBUG",
        "--log-format",
        "json",
        "--enable-tracemalloc",
        "--enable-psutil",
    ])

    assert args.config == tmp_path / "config.toml"
    assert args.repetitions == 5
    assert args.seed == 13
    assert args.output_dir == target_dir
    assert args.graph_sizes == ["small", "medium"]
    assert args.graph_densities == ["sparse", "dense"]
    assert args.graph_families == ["erdos_renyi"]
    assert args.algorithms == ["edmonds_karp", "dinitz"]
    assert args.max_nodes == 500
    assert args.capacity_distribution == "triangular"
    assert args.edge_probabilities == ["grid=0.4", "dense=0.9"]
    assert args.write_csv is True
    assert args.write_plots is False
    assert args.write_markdown is True
    assert args.write_json is True
    assert args.concurrent_workers == 4
    assert args.extra_reporters == ["tests.test_reporting.custom_reporter"]
    assert args.log_level == "DEBUG"
    assert args.log_format == "json"
    assert args.enable_tracemalloc is True
    assert args.enable_psutil is True
