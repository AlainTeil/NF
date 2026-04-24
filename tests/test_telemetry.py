from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nf_bench import telemetry as telemetry_mod
from nf_bench.telemetry import TelemetryCollector, _diff_cpu


def test_telemetry_collector_tracemalloc() -> None:
    collector = TelemetryCollector(enable_tracemalloc=True, enable_psutil=False)
    collector.start()
    data = collector.stop()

    assert data["tracemalloc_enabled"] is True
    assert "wall_clock_seconds" in data
    assert "tracemalloc_peak_bytes" in data


def test_telemetry_collector_psutil_flag() -> None:
    collector = TelemetryCollector(enable_psutil=True)
    collector.start()
    data = collector.stop()

    if data["psutil_enabled"]:
        assert "psutil_rss_bytes" in data
    else:
        assert data["psutil_requested"] is True
        assert "psutil_rss_bytes" not in data or data["psutil_rss_bytes"] is None


def test_stop_without_start_returns_empty() -> None:
    assert TelemetryCollector().stop() == {}


def test_double_start_is_idempotent() -> None:
    collector = TelemetryCollector()
    collector.start()
    # Second call should be a no-op (covers the ``if self._started: return`` guard).
    collector.start()
    data = collector.stop()
    assert "wall_clock_seconds" in data


def test_psutil_disabled_when_module_missing(caplog: pytest.LogCaptureFixture) -> None:
    """Requesting psutil when the dep is absent should warn and disable cleanly."""

    with patch.object(telemetry_mod, "psutil", None), caplog.at_level(logging.WARNING):
        collector = TelemetryCollector(enable_psutil=True)
        assert collector.enable_psutil is False
        assert collector._psutil_requested is True
        assert any("psutil telemetry requested" in rec.message for rec in caplog.records)

        data = collector.start() or collector.stop()
        assert data["psutil_enabled"] is False
        assert data["psutil_requested"] is True
        assert "psutil_rss_bytes" not in data


def _make_fake_psutil() -> SimpleNamespace:
    """Return a stub object exposing the subset of psutil we exercise."""

    @contextmanager
    def _oneshot() -> object:
        yield None

    state = {"rss": 100, "vms": 200, "user": 1.0, "system": 0.5}

    class _Process:
        def __init__(self, _pid: int) -> None:
            pass

        def oneshot(self):
            return _oneshot()

        def memory_info(self) -> SimpleNamespace:
            snap = SimpleNamespace(rss=state["rss"], vms=state["vms"])
            # advance for the next (stop) call so we get non-zero deltas
            state["rss"] += 50
            state["vms"] += 75
            return snap

        def cpu_times(self) -> SimpleNamespace:
            snap = SimpleNamespace(user=state["user"], system=state["system"])
            state["user"] += 0.25
            state["system"] += 0.10
            return snap

    return SimpleNamespace(Process=_Process)


def test_psutil_branch_with_fake_module() -> None:
    """Inject a fake psutil to exercise the start/stop psutil code paths."""

    fake = _make_fake_psutil()
    with patch.object(telemetry_mod, "psutil", fake):
        collector = TelemetryCollector(enable_tracemalloc=False, enable_psutil=True)
        assert collector.enable_psutil is True
        collector.start()
        data = collector.stop()

    assert data["psutil_enabled"] is True
    assert data["psutil_rss_bytes"] == 150  # second snapshot
    assert data["psutil_vms_bytes"] == 275
    assert data["psutil_rss_delta_bytes"] == 50
    assert data["psutil_cpu_user_seconds"] == pytest.approx(0.25)
    assert data["psutil_cpu_system_seconds"] == pytest.approx(0.10)


def test_diff_cpu_handles_missing_start_snapshot() -> None:
    # No start snapshot → returns the current value verbatim.
    assert _diff_cpu(3.5, None, "user") == 3.5
    # With a snapshot, computes the delta against the named attribute.
    snap = SimpleNamespace(user=1.0, system=0.25)
    assert _diff_cpu(2.5, snap, "user") == pytest.approx(1.5)
    assert _diff_cpu(0.75, snap, "system") == pytest.approx(0.5)
