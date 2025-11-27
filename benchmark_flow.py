"""Shim module delegating to the package CLI entry point."""
from __future__ import annotations

from nf_bench.cli import main


if __name__ == "__main__":
    main()
