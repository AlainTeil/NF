"""JSON encoding helpers tolerant of pandas/numpy/datetime/Path types."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


class SafeJsonEncoder(json.JSONEncoder):
    """JSON encoder that gracefully handles non-standard scalar types.

    Falls back to ``repr(obj)`` for anything otherwise unencodable so that
    structured logs or report files never crash on a stray exotic type.
    """

    def default(self, obj: Any) -> Any:  # type: ignore[override]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        # numpy scalars / pandas timestamps expose .item()
        item = getattr(obj, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:  # pragma: no cover - exotic numpy edge case
                pass
        # pandas / numpy arrays expose .tolist()
        tolist = getattr(obj, "tolist", None)
        if callable(tolist):
            try:
                return tolist()
            except Exception:  # pragma: no cover
                pass
        if isinstance(obj, set):
            return sorted(obj, key=repr)
        return repr(obj)


def safe_dumps(payload: Any, **kwargs: Any) -> str:
    """``json.dumps`` that never raises; logs at WARNING and returns ``repr`` on failure."""

    try:
        return json.dumps(payload, cls=SafeJsonEncoder, **kwargs)
    except Exception as exc:  # pragma: no cover - SafeJsonEncoder should handle most
        _logger.warning("safe_dumps fell back to repr() due to: %s", exc)
        return json.dumps({"__repr__": repr(payload)}, **kwargs)
