from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from nf_bench.models import BenchmarkConfig
from nf_bench.reporting import (
    ReportManager,
    build_report_metadata,
    format_summary_table,
)


def custom_reporter(
    df: pd.DataFrame, output_dir: Path, metadata: Dict[str, Any], summary_text: str
) -> Path:
    path = output_dir / "custom_report.txt"
    path.write_text("custom reporter invoked\n" + summary_text.splitlines()[0])
    return path


def _sample_config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        repetitions=1,
        seed=1,
        output_dir=tmp_path,
    )


def test_build_report_metadata_contains_expected_fields(tmp_path: Path) -> None:
    config = _sample_config(tmp_path)
    metadata = build_report_metadata(config, row_count=10, graph_count=3)

    assert "config_hash" in metadata
    assert metadata["row_count"] == 10
    assert metadata["graph_count"] == 3
    assert metadata["config"]["repetitions"] == 1


def test_report_manager_emits_selected_outputs(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "size_label": ["small", "small"],
            "density_label": ["sparse", "sparse"],
            "algorithm": ["edmonds_karp", "dinitz"],
            "elapsed_seconds": [0.1, 0.2],
        }
    )
    metadata = {
        "timestamp_utc": "2025-01-01T00:00:00Z",
        "run_id": "20250101T000000",
        "git_commit": "abc123",
        "config_hash": "xyz",
        "config": {},
        "row_count": len(df),
        "graph_count": 1,
    }

    manager = ReportManager(
        output_dir=tmp_path,
        write_csv=True,
        write_plots=False,
        write_markdown=True,
        write_json=True,
        extra_reporters=["tests.test_reporting.custom_reporter"],
    )
    summary = format_summary_table(df)
    artefacts = manager.emit(df, summary_text=summary, metadata=dict(metadata))

    csv_path = Path(artefacts["csv_path"])
    assert csv_path.exists()

    metadata_path = Path(artefacts["metadata_path"])
    metadata_payload = metadata_path.read_text()
    assert "artifacts" in metadata_payload

    markdown_path = Path(artefacts["markdown_path"])
    assert markdown_path.exists()

    json_path = Path(artefacts["json_path"])
    assert json_path.exists()

    custom_outputs = artefacts.get("custom_reporters")
    assert custom_outputs and Path(custom_outputs[0]).exists()