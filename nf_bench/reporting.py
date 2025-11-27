"""Reporting utilities for presenting benchmarking results."""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .models import BenchmarkConfig


ReporterFunc = Callable[[pd.DataFrame, Path, Dict[str, Any], str], Optional[Path]]


def format_summary_table(df: pd.DataFrame) -> str:
    """Generate a text table summarizing mean runtimes per algorithm/graph group."""

    if df.empty:
        return "No benchmark results available."

    summary = (
        df.groupby(["size_label", "density_label", "algorithm"])["elapsed_seconds"]
        .mean()
        .reset_index()
    )

    pivot = summary.pivot_table(
        index=["size_label", "density_label"],
        columns="algorithm",
        values="elapsed_seconds",
    ).sort_index()

    algorithms = sorted(summary["algorithm"].unique())
    pivot = pivot.reindex(columns=algorithms)

    table_rows: List[List[str]] = []
    header = ["size", "density"] + algorithms
    table_rows.append(header)

    for (size_label, density_label), values in pivot.iterrows():
        row = [size_label, density_label]
        for algo in algorithms:
            value = values.get(algo)
            row.append(f"{value:.6f}" if pd.notna(value) else "-")
        table_rows.append(row)

    col_widths = [max(len(row[idx]) for row in table_rows) for idx in range(len(header))]

    lines = []
    for row_idx, row in enumerate(table_rows):
        formatted = " | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(row))
        lines.append(formatted)
        if row_idx == 0:
            lines.append("-+-".join("-" * width for width in col_widths))

    return "\n".join(lines)


class ReportManager:
    """Co-ordinates benchmark artefact generation across multiple formats."""

    def __init__(
        self,
        *,
        output_dir: Path,
        write_csv: bool = True,
        write_plots: bool = True,
        write_markdown: bool = False,
        write_json: bool = False,
        extra_reporters: Optional[Iterable[str]] = None,
    ) -> None:
        self.output_dir = output_dir
        self.write_csv = write_csv
        self.write_plots = write_plots
        self.write_markdown = write_markdown
        self.write_json = write_json
        self._extra_reporters = [
            _import_reporter(path) for path in (extra_reporters or [])
        ]

    @classmethod
    def from_config(cls, config: BenchmarkConfig) -> "ReportManager":
        """Build a manager from :class:`BenchmarkConfig` settings."""

        return cls(
            output_dir=config.output_dir,
            write_csv=config.write_csv,
            write_plots=config.write_plots,
            write_markdown=config.write_markdown,
            write_json=config.write_json,
            extra_reporters=config.extra_reporters,
        )

    def emit(
        self,
        df: pd.DataFrame,
        *,
        summary_text: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Persist artefacts for the given benchmark results and metadata."""

        if df.empty:
            raise ValueError("No benchmark results available for reporting.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artefacts: Dict[str, Any] = {}

        if self.write_csv:
            artefacts["csv_path"] = str(self._save_results_csv(df, metadata))
        if self.write_plots:
            artefacts["plot_path"] = str(self._plot_runtime_bar_chart(df, metadata))
        if self.write_markdown:
            artefacts["markdown_path"] = str(
                self._write_markdown_summary(summary_text, metadata)
            )
        if self.write_json:
            artefacts["json_path"] = str(self._write_json_summary(df, metadata, summary_text))

        custom_outputs: List[str] = []
        for reporter in self._extra_reporters:
            result = reporter(df, self.output_dir, metadata, summary_text)
            if result is not None:
                custom_outputs.append(str(result))
        if custom_outputs:
            artefacts["custom_reporters"] = custom_outputs

        meta_path = self._write_metadata_file(metadata, artefacts)
        artefacts["metadata_path"] = str(meta_path)
        return artefacts

    def _timestamp_prefix(self, metadata: Dict[str, Any]) -> str:
        run_id = metadata.get("run_id")
        if isinstance(run_id, str):
            return run_id
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    def _save_results_csv(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Path:
        path = self.output_dir / f"flow_benchmark_{self._timestamp_prefix(metadata)}.csv"
        df.to_csv(path, index=False)
        return path

    def _plot_runtime_bar_chart(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Path:
        plot_df = (
            df.groupby(["size_label", "algorithm"])["elapsed_seconds"].mean().reset_index()
        )
        if plot_df.empty:
            raise ValueError("No data to plot.")

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=plot_df,
            x="algorithm",
            y="elapsed_seconds",
            hue="size_label",
            errorbar=None,
        )
        ax.set_ylabel("Mean runtime (seconds)")
        ax.set_xlabel("Algorithm")
        ax.set_title("NetworkX Maximum Flow Algorithm Runtimes")
        plt.legend(title="Graph size")
        plt.tight_layout()
        path = self.output_dir / f"flow_runtime_bar_{self._timestamp_prefix(metadata)}.png"
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def _write_markdown_summary(self, summary_text: str, metadata: Dict[str, Any]) -> Path:
        lines = [
            "# Network Flow Benchmark Summary",
            "",
            "## Metadata",
        ]
        for key in ("run_id", "timestamp_utc", "git_commit", "config_hash"):
            value = metadata.get(key, "unknown")
            lines.append(f"- **{key}**: {value}")
        lines.extend(
            [
                "",
                "## Summary Table",
                "```",
                summary_text,
                "```",
            ]
        )

        path = self.output_dir / f"flow_summary_{self._timestamp_prefix(metadata)}.md"
        path.write_text("\n".join(lines))
        return path

    def _write_json_summary(
        self, df: pd.DataFrame, metadata: Dict[str, Any], summary_text: str
    ) -> Path:
        summary_df = (
            df.groupby(["size_label", "density_label", "algorithm"])[
                "elapsed_seconds"
            ]
            .mean()
            .reset_index()
        )
        payload = {
            "metadata": metadata,
            "summary": summary_df.to_dict(orient="records"),
            "summary_table": summary_text,
        }
        path = self.output_dir / f"flow_summary_{self._timestamp_prefix(metadata)}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    def _write_metadata_file(
        self, metadata: Dict[str, Any], artefacts: Dict[str, Any]
    ) -> Path:
        payload = dict(metadata)
        payload["artifacts"] = artefacts
        path = self.output_dir / f"flow_metadata_{self._timestamp_prefix(metadata)}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path


def build_report_metadata(
    config: BenchmarkConfig,
    *,
    row_count: int,
    graph_count: int,
    telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct reproducibility metadata for a benchmark run."""

    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot = _config_snapshot(config)
    config_hash = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True).encode("utf-8")
    ).hexdigest()

    metadata: Dict[str, Any] = {
        "timestamp_utc": timestamp,
        "run_id": timestamp.replace(":", "").replace("-", ""),
        "git_commit": _detect_git_commit(),
        "config_hash": config_hash,
        "config": snapshot,
        "row_count": row_count,
        "graph_count": graph_count,
    }
    if telemetry:
        metadata["telemetry"] = telemetry
    return metadata


def _config_snapshot(config: BenchmarkConfig) -> Dict[str, Any]:
    snapshot = asdict(config)
    snapshot["output_dir"] = str(config.output_dir)
    return snapshot


def _detect_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:  # pragma: no cover - fallback path
        return "unknown"
    output = result.stdout.strip()
    return output or "unknown"


def _import_reporter(path: str) -> ReporterFunc:
    module_name, _, attr = path.partition(":")
    if not attr:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    reporter = getattr(module, attr)
    if not callable(reporter):
        raise TypeError(f"Reporter '{path}' is not callable")
    return reporter
