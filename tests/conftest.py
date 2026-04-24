"""Shared pytest configuration for nf-bench tests."""
from __future__ import annotations

import os

# Allow ``tests.*`` reporter modules to be loaded by the dynamic-import gate.
# Production users must opt-in explicitly via NF_BENCH_DYNAMIC_IMPORT_PREFIXES
# or NF_BENCH_ALLOW_DYNAMIC_IMPORT; tests configure it implicitly.
os.environ.setdefault("NF_BENCH_DYNAMIC_IMPORT_PREFIXES", "tests")
