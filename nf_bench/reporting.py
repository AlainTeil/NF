"""Reporting utilities for presenting benchmarking results."""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

# Force a non-interactive backend before importing pyplot so that headless
# environments (CI, servers without DISPLAY) never trigger GUI machinery.
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ._encoding import safe_dumps
from ._imports import import_callable
from .models import BenchmarkConfig

_logger = logging.getLogger(__name__)

# pyplot is not thread-safe; serialise plotting across all ReportManager
# instances so concurrent emit() calls cannot corrupt figures.
_PLOT_LOCK = threading.Lock()


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via a same-directory temp file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup; ignore if temp file already moved.
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


ReporterFunc = Callable[[pd.DataFrame, Path, dict[str, Any], str], Path | None]


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

    table_rows: list[list[str]] = []
    header = ["size", "density", *algorithms]
    table_rows.append(header)

    for index, values in pivot.iterrows():
        size_label = str(index[0])  # type: ignore[index]
        density_label = str(index[1])  # type: ignore[index]
        row = [size_label, density_label]
        for algo in algorithms:
            value = values.get(algo)
            row.append(f"{float(value):.6f}" if pd.notna(value) else "-")  # type: ignore[arg-type]
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
    """Co-ordinates benchmark artifact generation across multiple formats."""

    def __init__(
        self,
        *,
        output_dir: Path,
        write_csv: bool = True,
        write_plots: bool = True,
        write_markdown: bool = False,
        write_json: bool = False,
        extra_reporters: Iterable[str] | None = None,
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
    def from_config(cls, config: BenchmarkConfig) -> ReportManager:
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
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist artifacts for the given benchmark results and metadata."""

        if df.empty:
            raise ValueError("No benchmark results available for reporting.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Any] = {}

        if self.write_csv:
            artifacts["csv_path"] = str(self._save_results_csv(df, metadata))
        if self.write_plots:
            artifacts["plot_path"] = str(self._plot_runtime_bar_chart(df, metadata))
        if self.write_markdown:
            artifacts["markdown_path"] = str(
                self._write_markdown_summary(summary_text, metadata)
            )
        if self.write_json:
            artifacts["json_path"] = str(self._write_json_summary(df, metadata, summary_text))

        custom_outputs: list[str] = []
        for reporter in self._extra_reporters:
            result = reporter(df, self.output_dir, metadata, summary_text)
            if result is not None:
                custom_outputs.append(str(result))
        if custom_outputs:
            artifacts["custom_reporters"] = custom_outputs

        meta_path = self._write_metadata_file(metadata, artifacts)
        artifacts["metadata_path"] = str(meta_path)
        return artifacts

    def _timestamp_prefix(self, metadata: dict[str, Any]) -> str:
        run_id = metadata.get("run_id")
        if isinstance(run_id, str):
            return run_id
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    def _save_results_csv(self, df: pd.DataFrame, metadata: dict[str, Any]) -> Path:
        path = self.output_dir / f"flow_benchmark_{self._timestamp_prefix(metadata)}.csv"
        _atomic_write_text(path, df.to_csv(index=False))
        return path

    def _plot_runtime_bar_chart(self, df: pd.DataFrame, metadata: dict[str, Any]) -> Path:
        plot_df = (
            df.groupby(["size_label", "algorithm"])["elapsed_seconds"].mean().reset_index()
        )
        if plot_df.empty:
            raise ValueError("No data to plot.")

        # Serialise pyplot access; matplotlib.pyplot is not thread-safe.
        with _PLOT_LOCK:
            sns.set_theme(style="whitegrid")
            n_algos = max(plot_df["algorithm"].nunique(), 1)
            width = max(10, n_algos * 1.2)
            fig = plt.figure(figsize=(width, 6))
            try:
                ax = sns.barplot(
                    data=plot_df,
                    x="algorithm",
                    y="elapsed_seconds",
                    hue="size_label",
                    errorbar=None,
                )
                ax.set_ylabel("Mean runtime (seconds)")
                ax.set_xlabel("Algorithm")
                run_id = metadata.get("run_id", "")
                git_commit = metadata.get("git_commit", "")
                title = "NetworkX Maximum Flow Algorithm Runtimes"
                if run_id or git_commit:
                    title += f"\n(run_id={run_id} commit={git_commit})"
                ax.set_title(title)
                plt.legend(title="Graph size")
                plt.tight_layout()
                path = self.output_dir / f"flow_runtime_bar_{self._timestamp_prefix(metadata)}.png"
                # Render to a sibling temp file then atomically move into place
                # so concurrent readers never see a partial PNG. Pass ``format``
                # explicitly because matplotlib otherwise infers it from the
                # ``.tmp`` suffix and rejects the unknown extension.
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                plt.savefig(tmp_path, dpi=300, format="png")
                os.replace(tmp_path, path)
            finally:
                plt.close(fig)
        return path

    def _write_markdown_summary(self, summary_text: str, metadata: dict[str, Any]) -> Path:
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
        _atomic_write_text(path, "\n".join(lines))
        return path

    def _write_json_summary(
        self, df: pd.DataFrame, metadata: dict[str, Any], summary_text: str
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
        _atomic_write_text(path, safe_dumps(payload, indent=2, sort_keys=True))
        return path

    def _write_metadata_file(
        self, metadata: dict[str, Any], artifacts: dict[str, Any]
    ) -> Path:
        payload = dict(metadata)
        payload["artifacts"] = artifacts
        path = self.output_dir / f"flow_metadata_{self._timestamp_prefix(metadata)}.json"
        _atomic_write_text(path, safe_dumps(payload, indent=2, sort_keys=True))
        return path


def build_report_metadata(
    config: BenchmarkConfig,
    *,
    row_count: int,
    graph_count: int,
    telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct reproducibility metadata for a benchmark run."""

    timestamp = datetime.now(UTC).isoformat()
    snapshot = _config_snapshot(config)
    config_hash = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True).encode("utf-8")
    ).hexdigest()

    metadata: dict[str, Any] = {
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


def _config_snapshot(config: BenchmarkConfig) -> dict[str, Any]:
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
    except (OSError, subprocess.SubprocessError) as exc:
        _logger.warning("Could not run git to detect commit: %s", exc)
        return "unknown"
    if result.returncode != 0:
        _logger.debug(
            "git rev-parse exited %s: %s", result.returncode, result.stderr.strip()
        )
    output = result.stdout.strip()
    return output or "unknown"


def _import_reporter(path: str) -> ReporterFunc:
    reporter = import_callable(path)
    if not callable(reporter):
        raise TypeError(f"Reporter '{path}' is not callable")
    return reporter
