"""Training-free baselines: the score is simply one feature (higher = more likely to link)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

BASELINE_COLUMNS: Dict[str, str] = {
    "preferential_attachment": "pref_attach",   # degree product
    "common_neighbours": "common_neighbors",
    "adamic_adar": "adamic_adar",
    "jaccard": "jaccard",
    "semantic_only": "semantic_author",         # content-only baseline (TF-IDF / topic cosine)
}


def baseline_scores(df: pd.DataFrame, seed: int = 0) -> Dict[str, np.ndarray]:
    """Score arrays for every baseline plus a seeded ``random`` control."""
    out = {name: df[col].to_numpy(dtype=float) for name, col in BASELINE_COLUMNS.items()}
    out["random"] = np.random.default_rng(seed).random(len(df))
    return out
