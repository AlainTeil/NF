"""Safe dynamic-import helpers shared by runner and reporting.

Both the algorithm registry and the extra-reporters mechanism need to load
arbitrary user-provided dotted paths. This module centralises:

* A unified parser that accepts both ``"package.module:attribute"`` and
  ``"package.module.attribute"`` forms.
* An optional allow-list gate so that production deployments can refuse
  arbitrary code execution unless explicitly opted-in.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Any

ALLOW_DYNAMIC_IMPORT_ENV = "NF_BENCH_ALLOW_DYNAMIC_IMPORT"
ALLOW_DYNAMIC_IMPORT_PREFIXES_ENV = "NF_BENCH_DYNAMIC_IMPORT_PREFIXES"

_logger = logging.getLogger(__name__)


class DynamicImportError(ImportError):
    """Raised when a dynamic import path cannot be resolved or is disallowed."""


def _split_path(path: str) -> tuple[str, str]:
    """Parse ``module:attr`` or ``module.attr`` into ``(module, attribute)``."""

    if not isinstance(path, str) or not path.strip():
        raise DynamicImportError("Import path must be a non-empty string")

    candidate = path.strip()
    if ":" in candidate:
        module_name, _, attr = candidate.partition(":")
    else:
        module_name, _, attr = candidate.rpartition(".")

    if not module_name or not attr:
        raise DynamicImportError(
            f"Import path '{path}' must include both module and attribute"
        )
    return module_name, attr


def _check_allowed(module_name: str, env: dict[str, str] | None = None) -> None:
    """Enforce the optional allow-list gate for dynamic imports."""

    env_mapping = env if env is not None else os.environ
    raw_prefixes = env_mapping.get(ALLOW_DYNAMIC_IMPORT_PREFIXES_ENV, "")
    prefixes = tuple(p.strip() for p in raw_prefixes.split(",") if p.strip())
    allow_all = env_mapping.get(ALLOW_DYNAMIC_IMPORT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if allow_all:
        _logger.warning(
            "Loading dynamic import '%s' because %s is set; this allows arbitrary "
            "code execution.",
            module_name,
            ALLOW_DYNAMIC_IMPORT_ENV,
        )
        return

    if prefixes and any(
        module_name == p or module_name.startswith(p + ".") for p in prefixes
    ):
        return

    if prefixes or allow_all:  # defensive: above branches are exhaustive
        return  # pragma: no cover

    if _is_builtin_path(module_name):
        # First-party callables (e.g. networkx.algorithms.flow.*) are always
        # safe to load because they're already required runtime dependencies.
        return

    raise DynamicImportError(
        f"Refusing to import '{module_name}': dynamic imports are disabled. "
        f"Set {ALLOW_DYNAMIC_IMPORT_ENV}=1 to allow any module, or set "
        f"{ALLOW_DYNAMIC_IMPORT_PREFIXES_ENV}=pkg1,pkg2 to allow specific prefixes."
    )


_BUILTIN_PREFIXES: tuple[str, ...] = ("networkx.", "nf_bench.")


def _is_builtin_path(module_name: str) -> bool:
    return any(
        module_name == p.rstrip(".") or module_name.startswith(p)
        for p in _BUILTIN_PREFIXES
    )


def import_callable(path: str, *, env: dict[str, str] | None = None) -> Any:
    """Resolve ``path`` to a callable, applying the allow-list policy."""

    module_name, attr = _split_path(path)
    _check_allowed(module_name, env=env)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise DynamicImportError(
            f"Cannot import module '{module_name}' for path '{path}': {exc}"
        ) from exc
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise DynamicImportError(
            f"Module '{module_name}' has no attribute '{attr}' for path '{path}'"
        ) from exc
    return obj
