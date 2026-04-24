"""Configuration loading utilities for nf_bench.

Precedence (highest first): CLI flags > environment variables > config file.
File parsing is intentionally permissive about *missing* keys but strict about
structure: top-level mappings must be a known section, and unknown
``NF_BENCH_*`` env vars are reported as warnings to surface user typos.
"""
from __future__ import annotations

import json
import logging
import os
import tomllib
from argparse import Namespace
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Optional dependency used when YAML config files are provided.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised only when YAML unavailable
    yaml = None  # type: ignore

from .models import BenchmarkConfig

CONFIG_ENV_VAR = "NF_BENCH_CONFIG"
ENV_PREFIX = "NF_BENCH_"
MAX_CONFIG_DEPTH = 6
MAX_CONFIG_NODES = 5_000

_logger = logging.getLogger(__name__)

# Top-level keys accepted directly at the document root (no section nesting).
_DIRECT_KEYS: frozenset[str] = frozenset(
    {
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
        "flow_rel_tol",
        "flow_abs_tol",
    }
)

# Recognised env-var suffixes (after stripping ``NF_BENCH_``). Used to warn the
# user about typos like ``NF_BENCH_REPETITION``.
_KNOWN_ENV_SUFFIXES: frozenset[str] = frozenset(
    {key.upper() for key in _DIRECT_KEYS}
    | {"CONFIG", "EDGE_PROBABILITIES", "DYNAMIC_IMPORT_PREFIXES", "ALLOW_DYNAMIC_IMPORT"}
)


class ConfigError(RuntimeError):
    """Raised when configuration input cannot be parsed."""


def load_config(
    args: Namespace,
    *,
    env: Mapping[str, str] | None = None,
) -> BenchmarkConfig:
    """Load configuration by layering file, environment, and CLI overrides."""

    env_mapping: Mapping[str, str] = env or os.environ
    file_overrides: dict[str, Any] = {}

    config_path = _determine_config_path(args, env_mapping)
    if config_path is not None:
        file_overrides = _load_file(Path(config_path))

    env_overrides = _load_env_overrides(env_mapping)
    cli_overrides = _load_cli_overrides(args)

    merged: dict[str, Any] = {}
    for layer in (file_overrides, env_overrides, cli_overrides):
        merged.update({k: v for k, v in layer.items() if v is not None})

    return _build_config(merged)


def _determine_config_path(
    args: Namespace, env_mapping: Mapping[str, str]
) -> Path | None:
    """Return the configuration path derived from CLI args or environment."""

    if getattr(args, "config", None):
        return Path(args.config)
    env_value = env_mapping.get(CONFIG_ENV_VAR)
    return Path(env_value) if env_value else None


def _load_file(path: Path) -> dict[str, Any]:
    """Load a TOML or YAML config file and normalise the structure."""

    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    data = path.read_bytes()
    lower_name = path.name.lower()
    if lower_name.endswith((".yaml", ".yml")):
        if yaml is None:
            raise ConfigError("pyyaml is required to load YAML configuration files")
        try:
            loaded = yaml.safe_load(data) or {}
        except yaml.YAMLError as exc:  # type: ignore[union-attr]
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    else:
        try:
            loaded = tomllib.loads(data.decode("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ConfigError(f"Config file {path} is not valid UTF-8: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError("Top-level config structure must be a mapping")
    _enforce_size_caps(loaded)
    return _flatten_config_dict(loaded)


def _enforce_size_caps(payload: Mapping[str, Any], depth: int = 0) -> int:
    if depth > MAX_CONFIG_DEPTH:
        raise ConfigError(
            f"Config nesting exceeds maximum depth of {MAX_CONFIG_DEPTH}"
        )
    count = len(payload)
    for value in payload.values():
        if isinstance(value, Mapping):
            count += _enforce_size_caps(value, depth + 1)
        elif isinstance(value, list):
            count += len(value)
        if count > MAX_CONFIG_NODES:
            raise ConfigError(
                f"Config exceeds maximum entry count of {MAX_CONFIG_NODES}"
            )
    return count


def _flatten_config_dict(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested config sections into a shallow dictionary.

    Unknown top-level keys are reported as warnings rather than rejected
    outright so that users can layer in-house extensions without a hard error.
    """

    flattened: dict[str, Any] = {}
    for key in _DIRECT_KEYS:
        if key in raw:
            flattened[key] = raw[key]

    section_map = {
        "benchmark": {
            "repetitions",
            "seed",
            "output_dir",
            "max_nodes",
            "concurrent_workers",
            "flow_rel_tol",
            "flow_abs_tol",
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

    known_top_level = _DIRECT_KEYS | section_map.keys()
    for key in raw:
        if key not in known_top_level:
            _logger.warning("Ignoring unknown top-level config key: %s", key)

    for section, keys in section_map.items():
        section_value = raw.get(section)
        if section_value is None:
            continue
        if not isinstance(section_value, Mapping):
            _logger.warning(
                "Config section '%s' must be a mapping, got %s",
                section,
                type(section_value).__name__,
            )
            continue
        for key in section_value:
            if key not in keys:
                _logger.warning(
                    "Ignoring unknown key '%s' in config section '%s'", key, section
                )
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


def _coerce_list(value: Any) -> list[str] | None:
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


def _coerce_int_optional(value: Any) -> int | None:
    return None if value is None else int(value)


def _coerce_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def _coerce_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _coerce_str(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


@dataclass(frozen=True)
class _Option:
    """Single source of truth for one ``BenchmarkConfig`` field.

    ``coerce`` accepts the raw value coming from any of the three layers
    (file / env / CLI) and normalises it to the dataclass field's expected
    type. ``default`` is the value used when no layer supplies the option.
    """

    key: str
    coerce: Callable[[Any], Any]
    default: Any


_OPTIONS: tuple[_Option, ...] = (
    _Option("repetitions", lambda v: int(v), 1),
    _Option("seed", _coerce_seed, None),
    _Option("output_dir", _coerce_path, Path("reports")),
    _Option("graph_sizes", _coerce_list, None),
    _Option("graph_densities", _coerce_list, None),
    _Option("algorithms", _coerce_list, None),
    _Option("graph_families", _coerce_list, None),
    _Option("max_nodes", _coerce_int_optional, None),
    _Option("capacity_distribution", _coerce_str, "uniform"),
    _Option("write_csv", _coerce_bool, True),
    _Option("write_plots", _coerce_bool, True),
    _Option("write_markdown", _coerce_bool, False),
    _Option("write_json", _coerce_bool, False),
    _Option("concurrent_workers", _coerce_int_optional, None),
    _Option("extra_reporters", _coerce_list, None),
    _Option("log_level", _coerce_str, "INFO"),
    _Option("log_format", _coerce_str, "text"),
    _Option("enable_tracemalloc", _coerce_bool, False),
    _Option("enable_psutil", _coerce_bool, False),
    _Option("flow_rel_tol", lambda v: float(v), 1e-6),
    _Option("flow_abs_tol", lambda v: float(v), 1e-9),
)

_OPTION_BY_KEY: dict[str, _Option] = {opt.key: opt for opt in _OPTIONS}


def _load_env_overrides(env_mapping: Mapping[str, str]) -> dict[str, Any]:
    """Parse NF_BENCH_* environment variables into config overrides."""

    # Warn on prefix typos so users notice misspelled vars early.
    for env_key in env_mapping:
        if not env_key.startswith(ENV_PREFIX):
            continue
        suffix = env_key[len(ENV_PREFIX):]
        if suffix not in _KNOWN_ENV_SUFFIXES:
            _logger.warning("Ignoring unknown environment variable: %s", env_key)

    overrides: dict[str, Any] = {}
    for opt in _OPTIONS:
        value = env_mapping.get(f"{ENV_PREFIX}{opt.key.upper()}")
        if value is not None:
            overrides[opt.key] = opt.coerce(value)

    edge_env = env_mapping.get(f"{ENV_PREFIX}EDGE_PROBABILITIES")
    if edge_env is not None:
        overrides["edge_probability_overrides"] = _coerce_edge_mapping(edge_env)
    return overrides


def _load_cli_overrides(args: Namespace) -> dict[str, Any]:
    """Convert parsed CLI args into config override values."""

    overrides: dict[str, Any] = {}
    for opt in _OPTIONS:
        value = getattr(args, opt.key, None)
        if value is not None:
            overrides[opt.key] = value
    edge_flags = getattr(args, "edge_probabilities", None)
    if edge_flags:
        overrides["edge_probability_overrides"] = _coerce_edge_mapping(edge_flags)
    return overrides


def _coerce_value(key: str, value: Any) -> Any:
    """Normalise a raw env/CLI scalar via the canonical option coercer."""

    opt = _OPTION_BY_KEY.get(key)
    if opt is None:
        return value
    return opt.coerce(value)


def _coerce_edge_mapping(raw: Any) -> dict[str, float]:
    """Parse FAMILY=probability pairs passed via env/CLI overrides."""

    if isinstance(raw, Mapping):
        return {str(k): float(v) for k, v in raw.items()}
    entries: list[Any] | None
    if isinstance(raw, list):
        entries = raw
    else:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, Mapping):
                return {str(k): float(v) for k, v in parsed.items()}
            entries = parsed if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            entries = None
        if entries is None:
            pieces = [p for p in str(raw).split(",") if p]
            mapping: dict[str, float] = {}
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


def _build_config(overrides: MutableMapping[str, Any]) -> BenchmarkConfig:
    """Instantiate :class:`BenchmarkConfig` from merged override values."""

    kwargs: dict[str, Any] = {}
    for opt in _OPTIONS:
        if opt.key in overrides:
            kwargs[opt.key] = opt.coerce(overrides[opt.key])
        else:
            kwargs[opt.key] = opt.default
    kwargs["algorithm_registry"] = overrides.get("algorithm_registry")
    kwargs["edge_probability_overrides"] = overrides.get("edge_probability_overrides")
    return BenchmarkConfig(**kwargs)
