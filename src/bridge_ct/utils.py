"""Small shared helpers: logging, seeding, directories, JSON and hashing."""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np


def get_logger(name: str = "bridge_ct") -> logging.Logger:
    """Return a logger that prints timestamped messages once (no duplicate handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seed Python's and NumPy's global RNGs (module code additionally uses explicit Generators)."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj: Any) -> Any:
    """Make numpy / pandas scalars JSON serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def write_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Checksum a file (used in the data-retrieval manifest)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
