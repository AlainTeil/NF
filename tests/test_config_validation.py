"""Validation/edge-case tests for the configuration loader."""
from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

import pytest

from nf_bench.config import ConfigError, load_config


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


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(_ns(config=tmp_path / "nope.toml"))


def test_malformed_toml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("not = valid = toml\n[")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(_ns(config=path))


def test_unknown_top_level_key_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "ok.toml"
    path.write_text('repetitions = 1\nbogus_key = "x"\n')
    with caplog.at_level(logging.WARNING, logger="nf_bench.config"):
        load_config(_ns(config=path))
    assert any("bogus_key" in m for m in caplog.messages)


def test_unknown_env_var_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="nf_bench.config"):
        load_config(_ns(), env={"NF_BENCH_NOT_A_REAL_OPTION": "1"})
    assert any("NF_BENCH_NOT_A_REAL_OPTION" in m for m in caplog.messages)


def test_known_env_var_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="nf_bench.config"):
        cfg = load_config(_ns(), env={"NF_BENCH_REPETITIONS": "4"})
    assert cfg.repetitions == 4
    assert not any("NF_BENCH_REPETITIONS" in m for m in caplog.messages)


def test_flow_tolerances_round_trip_from_env() -> None:
    cfg = load_config(
        _ns(),
        env={"NF_BENCH_FLOW_REL_TOL": "1e-3", "NF_BENCH_FLOW_ABS_TOL": "1e-7"},
    )
    assert cfg.flow_rel_tol == pytest.approx(1e-3)
    assert cfg.flow_abs_tol == pytest.approx(1e-7)


def test_seed_none_literal_from_env() -> None:
    cfg = load_config(_ns(), env={"NF_BENCH_SEED": "none"})
    assert cfg.seed is None
