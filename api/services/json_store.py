"""mtime-cached loaders for the pipeline's results/*.json files.

A file is re-read only when its mtime changes, so the API always serves
current pipeline output without re-parsing JSON on every request.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

_cache: dict[str, tuple[float, Any]] = {}
_lock = threading.Lock()


def load_json(filename: str, default: Any = None) -> Any:
    """Load results/<filename>, cached on file mtime. Returns `default`
    ({} unless overridden) when the file is missing or unparseable."""
    if default is None:
        default = {}
    path = RESULTS_DIR / filename
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return default

    with _lock:
        cached = _cache.get(filename)
        if cached and cached[0] == mtime:
            return cached[1]

    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return default

    with _lock:
        _cache[filename] = (mtime, data)
    return data
