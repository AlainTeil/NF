"""Smoke tests for the CLI ``main`` entry point."""
from __future__ import annotations

from pathlib import Path

import pytest

from nf_bench.cli import EXIT_CONFIG, EXIT_OK, main


def test_main_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--version"])
    assert code == EXIT_OK
    captured = capsys.readouterr()
    # argparse writes ``--version`` text to stdout in Python 3.4+.
    assert "nf-bench" in (captured.out + captured.err)


def test_main_list_algorithms_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--list-algorithms"])
    assert code == EXIT_OK
    captured = capsys.readouterr()
    assert "edmonds_karp" in captured.out


def test_main_list_graph_families_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--list-graph-families"])
    assert code == EXIT_OK
    captured = capsys.readouterr()
    assert "erdos_renyi" in captured.out


def test_main_dry_run_exits_zero(tmp_path: Path) -> None:
    code = main(["--dry-run", "--no-demo", "--output-dir", str(tmp_path)])
    assert code == EXIT_OK
    # Dry run must not write any artifact files.
    assert list(tmp_path.iterdir()) == []


def test_main_invalid_config_returns_config_exit_code(tmp_path: Path) -> None:
    bad = tmp_path / "broken.toml"
    bad.write_text("this is = not valid toml = at all\n[")
    code = main(["--config", str(bad)])
    assert code == EXIT_CONFIG


def test_main_bad_extra_reporter_fails_fast(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reporter that fails the dynamic-import gate must abort before benchmarks run."""

    code = main(
        [
            "--no-demo",
            "--dry-run",  # would normally exit OK, but reporter check happens first
            "--output-dir",
            str(tmp_path),
            "--extra-reporter",
            "nonexistent_pkg.reporters:export",
        ]
    )
    assert code == EXIT_CONFIG
    err = capsys.readouterr().err
    assert "nonexistent_pkg" in err
    # No artifacts must be produced when the reporter is rejected.
    assert list(tmp_path.iterdir()) == []
