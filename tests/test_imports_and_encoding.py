"""Tests for the dynamic-import gate and SafeJsonEncoder."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from nf_bench._encoding import SafeJsonEncoder, safe_dumps
from nf_bench._imports import DynamicImportError, import_callable


def test_import_callable_accepts_module_colon_attr() -> None:
    func = import_callable("networkx.algorithms.flow:edmonds_karp")
    assert callable(func)


def test_import_callable_accepts_module_dot_attr() -> None:
    func = import_callable("networkx.algorithms.flow.edmonds_karp")
    assert callable(func)


def test_import_callable_rejects_unallowed_prefix_without_env() -> None:
    # 'os.path.join' is not in the builtin allow-list and no env opt-in is set.
    with pytest.raises(DynamicImportError):
        import_callable("os.path.join", env={})


def test_import_callable_allows_via_prefix_env() -> None:
    func = import_callable(
        "os.path.join", env={"NF_BENCH_DYNAMIC_IMPORT_PREFIXES": "os."}
    )
    assert callable(func)


def test_import_callable_allows_via_global_env() -> None:
    func = import_callable("os.path.join", env={"NF_BENCH_ALLOW_DYNAMIC_IMPORT": "1"})
    assert callable(func)


def test_import_callable_invalid_path_raises() -> None:
    with pytest.raises(DynamicImportError):
        import_callable("not_a_valid_path")


def test_safe_json_encoder_handles_path_and_datetime() -> None:
    payload = {
        "path": Path("/tmp/x"),
        "ts": dt.datetime(2025, 1, 2, 3, 4, 5),
        "date": dt.date(2025, 1, 2),
        "set": {1, 2, 3},
    }
    encoded = json.loads(safe_dumps(payload, sort_keys=True))
    assert encoded["path"] == "/tmp/x"
    assert encoded["ts"].startswith("2025-01-02")
    assert encoded["date"] == "2025-01-02"
    assert sorted(encoded["set"]) == [1, 2, 3]


def test_safe_json_encoder_handles_dataclass() -> None:
    @dataclass
    class Item:
        name: str
        value: int

    encoded = json.loads(safe_dumps({"item": Item("a", 1)}))
    assert encoded["item"] == {"name": "a", "value": 1}


def test_safe_json_encoder_falls_back_to_repr() -> None:
    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    encoded = json.loads(safe_dumps({"w": Weird()}))
    assert encoded["w"] == "<weird>"


def test_safe_json_encoder_class_works_directly() -> None:
    out = json.dumps({"p": Path("/a")}, cls=SafeJsonEncoder)
    assert "/a" in out
