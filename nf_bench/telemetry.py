"""Telemetry helpers for capturing runtime diagnostics."""
from __future__ import annotations

import logging
import os
import tracemalloc
from time import perf_counter
from typing import Any

try:  # Optional dependency for richer telemetry
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - executed when psutil unavailable
    psutil = None  # type: ignore


class TelemetryCollector:
    """Collects wall-clock/memory telemetry for a benchmark run.

    .. note::
        ``tracemalloc`` and ``psutil`` measurements are process-global. When
        ``run_benchmarks`` is invoked with ``concurrent_workers > 1`` the
        captured peak/RSS values reflect the union of all worker threads, not
        any individual benchmark task. Treat the numbers as run-level rather
        than per-task in that mode.
    """

    def __init__(
        self,
        *,
        enable_tracemalloc: bool = False,
        enable_psutil: bool = False,
    ) -> None:
        self.enable_tracemalloc = enable_tracemalloc
        self.enable_psutil = enable_psutil and psutil is not None
        self._psutil_requested = enable_psutil
        self._started = False
        self._start_time = 0.0
        self._start_cpu: Any | None = None
        self._start_rss: int | None = None
        self._process: Any | None = None

        if enable_psutil and psutil is None:
            logging.getLogger(__name__).warning(
                "psutil telemetry requested but psutil is not installed; disabling psutil telemetry"
            )

    def start(self) -> None:
        """Begin telemetry collection for the surrounding benchmark run."""

        if self._started:
            return
        self._started = True
        self._start_time = perf_counter()
        if self.enable_tracemalloc:
            tracemalloc.start()
        if self.enable_psutil and psutil is not None:
            self._process = psutil.Process(os.getpid())
            with self._process.oneshot():
                mem = self._process.memory_info()
                self._start_rss = mem.rss
                self._start_cpu = self._process.cpu_times()

    def stop(self) -> dict[str, Any]:
        """Stop telemetry collection and return the captured metrics."""

        if not self._started:
            return {}

        wall_clock = perf_counter() - self._start_time
        self._started = False

        telemetry: dict[str, Any] = {
            "wall_clock_seconds": wall_clock,
            "tracemalloc_enabled": self.enable_tracemalloc,
            "psutil_requested": self._psutil_requested,
            "psutil_enabled": self.enable_psutil,
        }

        if self.enable_tracemalloc:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            telemetry.update(
                {
                    "tracemalloc_current_bytes": current,
                    "tracemalloc_peak_bytes": peak,
                }
            )

        if self.enable_psutil and psutil is not None and self._process is not None:
            with self._process.oneshot():
                mem = self._process.memory_info()
                cpu_times = self._process.cpu_times()
            telemetry.update(
                {
                    "psutil_rss_bytes": mem.rss,
                    "psutil_vms_bytes": mem.vms,
                    "psutil_rss_delta_bytes": (
                        mem.rss - self._start_rss if self._start_rss is not None else None
                    ),
                    "psutil_cpu_user_seconds": _diff_cpu(cpu_times.user, self._start_cpu, "user"),
                    "psutil_cpu_system_seconds": _diff_cpu(
                        cpu_times.system, self._start_cpu, "system"
                    ),
                }
            )

        return telemetry


def _diff_cpu(current: float, start_cpu: Any | None, attr: str) -> float:
    """Compute cpu time deltas even when psutil start snapshots are missing."""

    if start_cpu is None:
        return current
    return current - getattr(start_cpu, attr, 0.0)
