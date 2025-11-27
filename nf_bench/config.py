"""Configuration loading utilities for nf_bench."""
from __future__ import annotations

import json
import os
import tomllib
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

try:  # Optional dependency used when YAML config files are provided.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised only when YAML unavailable
    yaml = None  # type: ignore

from .models import BenchmarkConfig

CONFIG_ENV_VAR = "NF_BENCH_CONFIG"
ENV_PREFIX = "NF_BENCH_"


class ConfigError(RuntimeError):
    """Raised when configuration input cannot be parsed."""


def load_config(
    args: Namespace,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> BenchmarkConfig:
    """Load configuration by layering file, environment, and CLI overrides."""

    env_mapping: Mapping[str, str] = env or os.environ
    file_overrides: Dict[str, Any] = {}

    config_path = _determine_config_path(args, env_mapping)
    if config_path is not None:
        file_overrides = _load_file(Path(config_path))

    env_overrides = _load_env_overrides(env_mapping)
    cli_overrides = _load_cli_overrides(args)

    merged: Dict[str, Any] = {}
    for layer in (file_overrides, env_overrides, cli_overrides):
        merged.update({k: v for k, v in layer.items() if v is not None})

    return _build_config(merged)


def _determine_config_path(
    args: Namespace, env_mapping: Mapping[str, str]
) -> Optional[Path]:
    """Return the configuration path derived from CLI args or environment."""

    if getattr(args, "config", None):
        return Path(args.config)
    env_value = env_mapping.get(CONFIG_ENV_VAR)
    return Path(env_value) if env_value else None


def _load_file(path: Path) -> Dict[str, Any]:
    """Load a TOML or YAML config file and normalise the structure."""

    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    data = path.read_bytes()
    lower_name = path.name.lower()
    if lower_name.endswith(('.yaml', '.yml')):
        if yaml is None:
            raise ConfigError("pyyaml is required to load YAML configuration files")
        loaded = yaml.safe_load(data) or {}
    else:
        loaded = tomllib.loads(data.decode())
    if not isinstance(loaded, dict):
        raise ConfigError("Top-level config structure must be a mapping")
    return _flatten_config_dict(loaded)


def _flatten_config_dict(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten nested config sections into a shallow dictionary."""

    flattened: Dict[str, Any] = {}
    direct_keys = {
        "repetitions",
        "seed",
        "output_dir",
        "graph_sizes",
        "graph_densities",
        "algorithms",
        "graph_families",
        "max_nodes",
        "capacity_distribution",
        "edge_probability_overrides",
        "write_csv",
        "write_plots",
        "write_markdown",
        "write_json",
        "concurrent_workers",
        "extra_reporters",
        "log_level",
        "log_format",
        "enable_tracemalloc",
        "enable_psutil",
    }
    for key in direct_keys:
        if key in raw:
            flattened[key] = raw[key]

    section_map = {
        "benchmark": {
            "repetitions",
            "seed",
            "output_dir",
            "max_nodes",
            "concurrent_workers",
        },
        "graph": {
            "graph_sizes",
            "graph_densities",
            "graph_families",
        },
        "algorithms": {"algorithms", "registry"},
        "capacity": {"capacity_distribution", "distribution"},
        "edges": {"edge_probability_overrides", "probabilities"},
        "outputs": {
            "write_csv",
            "write_plots",
            "write_markdown",
            "write_json",
            "extra_reporters",
        },
        "logging": {"log_level", "log_format"},
        "telemetry": {"enable_tracemalloc", "enable_psutil"},
    }

    for section, keys in section_map.items():
        section_value = raw.get(section)
        if not isinstance(section_value, Mapping):
            continue
        for key in keys:
            if key in section_value:
                if key == "probabilities":
                    target_key = "edge_probability_overrides"
                elif section == "capacity" and key == "distribution":
                    target_key = "capacity_distribution"
                elif section == "algorithms" and key == "registry":
                    target_key = "algorithm_registry"
                else:
                    target_key = key
                flattened[target_key] = section_value[key]

    return flattened


def _load_env_overrides(env_mapping: Mapping[str, str]) -> Dict[str, Any]:
    """Parse NF_BENCH_* environment variables into config overrides."""

    overrides: Dict[str, Any] = {}
    for key in (
        "repetitions",
        "seed",
        "output_dir",
        "graph_sizes",
        "graph_densities",
        "algorithms",
        "graph_families",
        "max_nodes",
        "capacity_distribution",
        "write_csv",
        "write_plots",
        "write_markdown",
        "write_json",
        "concurrent_workers",
        "extra_reporters",
        "log_level",
        "log_format",
        "enable_tracemalloc",
        "enable_psutil",
    ):
        value = env_mapping.get(f"{ENV_PREFIX}{key.upper()}")
        if value is not None:
            overrides[key] = _coerce_value(key, value)

    edge_env = env_mapping.get(f"{ENV_PREFIX}EDGE_PROBABILITIES")
    if edge_env is not None:
        overrides["edge_probability_overrides"] = _coerce_edge_mapping(edge_env)
    return overrides


def _load_cli_overrides(args: Namespace) -> Dict[str, Any]:
    """Convert parsed CLI args into config override values."""

    overrides = {
        "repetitions": getattr(args, "repetitions", None),
        "seed": getattr(args, "seed", None),
        "output_dir": getattr(args, "output_dir", None),
        "graph_sizes": getattr(args, "graph_sizes", None),
        "graph_densities": getattr(args, "graph_densities", None),
        "algorithms": getattr(args, "algorithms", None),
        "graph_families": getattr(args, "graph_families", None),
        "max_nodes": getattr(args, "max_nodes", None),
        "capacity_distribution": getattr(args, "capacity_distribution", None),
        "write_csv": getattr(args, "write_csv", None),
        "write_plots": getattr(args, "write_plots", None),
        "write_markdown": getattr(args, "write_markdown", None),
        "write_json": getattr(args, "write_json", None),
        "concurrent_workers": getattr(args, "concurrent_workers", None),
        "extra_reporters": getattr(args, "extra_reporters", None),
        "log_level": getattr(args, "log_level", None),
        "log_format": getattr(args, "log_format", None),
        "enable_tracemalloc": getattr(args, "enable_tracemalloc", None),
        "enable_psutil": getattr(args, "enable_psutil", None),
    }
    edge_flags = getattr(args, "edge_probabilities", None)
    if edge_flags:
        overrides["edge_probability_overrides"] = _coerce_edge_mapping(edge_flags)
    return overrides


def _coerce_value(key: str, value: Any) -> Any:
    """Normalise scalar env/CLI values into their target Python type."""

    if key in {"graph_sizes", "graph_densities", "algorithms", "graph_families", "extra_reporters"}:
        return _coerce_list(value)
    if key in {"write_csv", "write_plots", "write_markdown", "write_json", "enable_tracemalloc", "enable_psutil"}:
        return _coerce_bool(value)
    if key in {"repetitions", "max_nodes", "concurrent_workers"}:
        return int(value)
    if key == "seed":
        return None if value in {"", "none", "None"} else int(value)
    if key == "output_dir":
        return Path(value)
    return value


def _coerce_edge_mapping(raw: Any) -> Dict[str, float]:
    """Parse FAMILY=probability pairs passed via env/CLI overrides."""

    if isinstance(raw, Mapping):
        return {str(k): float(v) for k, v in raw.items()}
    if isinstance(raw, list):
        entries = raw
    else:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, Mapping):
                return {str(k): float(v) for k, v in parsed.items()}
            if isinstance(parsed, list):
                entries = parsed
            else:
                entries = None
        except json.JSONDecodeError:
            entries = None
        if entries is None:
            pieces = [p for p in str(raw).split(",") if p]
            mapping: Dict[str, float] = {}
            for piece in pieces:
                if "=" not in piece:
                    continue
                key, val = piece.split("=", 1)
                mapping[key.strip()] = float(val)
            return mapping
    mapping = {}
    if entries is None:
        return mapping
    for entry in entries:
        if isinstance(entry, str) and "=" in entry:
            key, val = entry.split("=", 1)
            mapping[key.strip()] = float(val)
    return mapping


def _coerce_list(value: Any) -> Optional[list[str]]:
    """Convert comma-delimited strings or iterables into a list of strings."""

    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _coerce_bool(value: Any) -> bool:
    """Interpret a truthy CLI/env value, respecting common string literals."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _build_config(overrides: MutableMapping[str, Any]) -> BenchmarkConfig:
    """Instantiate :class:`BenchmarkConfig` from merged override values."""

    output_dir = overrides.get("output_dir", Path("reports"))
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    graph_sizes = _coerce_list(overrides.get("graph_sizes"))
    graph_densities = _coerce_list(overrides.get("graph_densities"))
    algorithms = _coerce_list(overrides.get("algorithms"))

    return BenchmarkConfig(
        repetitions=int(overrides.get("repetitions", 1)),
        seed=overrides.get("seed"),
        output_dir=output_dir,
        graph_sizes=graph_sizes,
        graph_densities=graph_densities,
        algorithms=algorithms,
        graph_families=_coerce_list(overrides.get("graph_families")),
        max_nodes=overrides.get("max_nodes"),
        capacity_distribution=overrides.get("capacity_distribution", "uniform"),
        edge_probability_overrides=overrides.get("edge_probability_overrides"),
        write_csv=_coerce_bool(overrides.get("write_csv", True)),
        write_plots=_coerce_bool(overrides.get("write_plots", True)),
        write_markdown=_coerce_bool(overrides.get("write_markdown", False)),
        write_json=_coerce_bool(overrides.get("write_json", False)),
        algorithm_registry=overrides.get("algorithm_registry"),
        concurrent_workers=overrides.get("concurrent_workers"),
        extra_reporters=_coerce_list(overrides.get("extra_reporters")),
        log_level=str(overrides.get("log_level", "INFO")),
        log_format=str(overrides.get("log_format", "text")),
        enable_tracemalloc=_coerce_bool(overrides.get("enable_tracemalloc", False)),
        enable_psutil=_coerce_bool(overrides.get("enable_psutil", False)),
    )