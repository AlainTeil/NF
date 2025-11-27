from __future__ import annotations

from nf_bench.telemetry import TelemetryCollector


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
