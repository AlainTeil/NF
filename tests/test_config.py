from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from nf_bench.config import CONFIG_ENV_VAR, load_config


def _ns(**overrides):
    defaults = dict(
        config=None,
        repetitions=None,
        seed=None,
        output_dir=None,
        graph_sizes=None,
        graph_densities=None,
        graph_families=None,
        algorithms=None,
        max_nodes=None,
        capacity_distribution=None,
        edge_probabilities=None,
        write_csv=None,
        write_plots=None,
        write_markdown=None,
        write_json=None,
        concurrent_workers=None,
        extra_reporters=None,
        log_level=None,
        log_format=None,
        enable_tracemalloc=None,
        enable_psutil=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_load_config_defaults() -> None:
    config = load_config(_ns())

    assert config.repetitions == 1
    assert config.seed is None
    assert config.output_dir == Path("reports")
    assert config.capacity_distribution == "uniform"
    assert config.write_csv is True
    assert config.write_plots is True
    assert config.write_markdown is False
    assert config.write_json is False
    assert config.concurrent_workers is None
    assert config.algorithm_registry is None
    assert config.extra_reporters is None
    assert config.log_level == "INFO"
    assert config.log_format == "text"
    assert config.enable_tracemalloc is False
    assert config.enable_psutil is False


def test_load_config_from_file(tmp_path):
    config_path = tmp_path / "bench.toml"
    config_path.write_text(
        """
        [benchmark]
        repetitions = 3
        seed = 11
        output_dir = "artifacts"
        max_nodes = 250
        concurrent_workers = 4

        [graph]
        graph_sizes = ["small", "medium"]
        graph_densities = ["sparse"]
        graph_families = ["layered_dag"]

        [algorithms]
        algorithms = ["edmonds_karp"]
        registry = { custom = "networkx.algorithms.flow.edmonds_karp", preflow_push = "" }

        [capacity]
        distribution = "triangular"

        [outputs]
        write_csv = false
        write_plots = true
        write_markdown = true
        write_json = true
        extra_reporters = ["tests.test_reporting.custom_reporter"]

        [edges]
        probabilities = { demo = 0.25 }

        [logging]
        log_level = "DEBUG"
        log_format = "json"

        [telemetry]
        enable_tracemalloc = true
        enable_psutil = true
        """.strip()
    )

    config = load_config(_ns(config=config_path))

    assert config.repetitions == 3
    assert config.seed == 11
    assert config.output_dir == Path("artifacts")
    assert config.graph_sizes == ["small", "medium"]
    assert config.graph_densities == ["sparse"]
    assert config.graph_families == ["layered_dag"]
    assert config.algorithms == ["edmonds_karp"]
    assert config.max_nodes == 250
    assert config.capacity_distribution == "triangular"
    assert config.edge_probability_overrides == {"demo": 0.25}
    assert config.write_csv is False
    assert config.write_markdown is True
    assert config.write_json is True
    assert config.concurrent_workers == 4
    assert config.algorithm_registry == {
        "custom": "networkx.algorithms.flow.edmonds_karp",
        "preflow_push": "",
    }
    assert config.extra_reporters == ["tests.test_reporting.custom_reporter"]
    assert config.log_level == "DEBUG"
    assert config.log_format == "json"
    assert config.enable_tracemalloc is True
    assert config.enable_psutil is True


def test_load_config_precedence(tmp_path):
    file_path = tmp_path / "bench.toml"
    file_path.write_text("repetitions = 2")

    env = {
        CONFIG_ENV_VAR: str(file_path),
        "NF_BENCH_REPETITIONS": "5",
        "NF_BENCH_EDGE_PROBABILITIES": "demo=0.33",
    }

    config = load_config(_ns(repetitions=7, edge_probabilities=["demo=0.5"]), env=env)

    assert config.repetitions == 7  # CLI wins
    assert config.edge_probability_overrides == {"demo": 0.5}
