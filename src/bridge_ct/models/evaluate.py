"""Evaluation: AUPRC, ROC-AUC, Precision/Recall@K, calibration, bootstrap CIs and paired model comparison.

Ties (very common for count baselines such as common neighbours = 0) are broken by a fixed random key, so a
baseline is never rewarded or punished by the row order of the table.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _order(score: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Indices sorted by descending score, ties broken by ``tiebreak``."""
    return np.lexsort((tiebreak, -score))


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int, tiebreak: Optional[np.ndarray] = None) -> float:
    """Fraction of true future links among the K highest-scored candidates (K capped at n)."""
    k = min(k, len(y))
    if k == 0:
        return float("nan")
    tb = tiebreak if tiebreak is not None else np.random.default_rng(0).random(len(y))
    return float(y[_order(score, tb)[:k]].mean())


def recall_at_k(y: np.ndarray, score: np.ndarray, k: int, tiebreak: Optional[np.ndarray] = None) -> float:
    """Fraction of ALL true future links recovered within the top K (review-budget view)."""
    total = y.sum()
    if total == 0:
        return float("nan")
    tb = tiebreak if tiebreak is not None else np.random.default_rng(0).random(len(y))
    return float(y[_order(score, tb)[:min(k, len(y))]].sum() / total)


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC, or NaN (never silently 0.5) when only one class is present."""
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _safe_ap(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table with quantile bins: mean predicted vs observed frequency."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=min(n_bins, max(1, len(df))), labels=False)
    out = df.groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size")).reset_index()
    return out


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Sample-weighted mean |observed - predicted| over quantile bins."""
    t = calibration_table(y, p, n_bins)
    return float((t["n"] * (t["frac_pos"] - t["mean_pred"]).abs()).sum() / t["n"].sum())


def evaluate_scores(y: np.ndarray, score: np.ndarray, ks: Sequence[int] = (10, 50), n_boot: int = 300,
                    seed: int = 42, is_probability: bool = False, cutoffs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """All metrics for one score vector.

    Returns AUPRC, ROC-AUC, pooled Precision@K / Recall@K, per-cutoff mean Precision@K (if ``cutoffs`` given),
    bootstrap 95% CIs for AUPRC / ROC-AUC / pooled Precision@K, and Brier + ECE when ``is_probability``.
    The bootstrap resamples candidate rows (it ignores dependence between rows of one snapshot, so treat the CIs
    as optimistic and read them together with the per-cutoff results).
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    s = np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) if np.isfinite(s).any() else 0.0)
    rng = np.random.default_rng(seed)
    tb = rng.random(len(y))
    res: Dict[str, float] = {"n": int(len(y)), "n_pos": int(y.sum()), "base_rate": float(y.mean()) if len(y) else float("nan"),
                             "auprc": _safe_ap(y, s), "roc_auc": _safe_auc(y, s)}
    for k in ks:
        res[f"precision_at_{k}"] = precision_at_k(y, s, k, tb)
        res[f"recall_at_{k}"] = recall_at_k(y, s, k, tb)
        if cutoffs is not None:
            per = [precision_at_k(y[m], s[m], k, tb[m]) for c in np.unique(cutoffs) for m in [cutoffs == c]]
            res[f"precision_at_{k}_per_cutoff"] = float(np.nanmean(per))
    if is_probability:
        res["brier"] = float(brier_score_loss(y, np.clip(s, 0, 1)))
        res["ece"] = expected_calibration_error(y, s)
        res["brier_base_rate"] = float(brier_score_loss(y, np.full(len(y), y.mean())))
    boots: Dict[str, list] = {"auprc": [], "roc_auc": [], **{f"precision_at_{k}": [] for k in ks}}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        boots["auprc"].append(_safe_ap(yb, sb)); boots["roc_auc"].append(_safe_auc(yb, sb))
        for k in ks:
            boots[f"precision_at_{k}"].append(precision_at_k(yb, sb, k, tb[idx]))
    for name, vals in boots.items():
        if vals:
            res[f"{name}_ci_lo"], res[f"{name}_ci_hi"] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return res


def paired_bootstrap_diff(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_boot: int = 500,
                          seed: int = 42) -> Dict[str, float]:
    """AUPRC(A) - AUPRC(B) on the SAME resamples: difference, 95% CI and share of resamples where A wins."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    n, diffs = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        diffs.append(_safe_ap(y[idx], score_a[idx]) - _safe_ap(y[idx], score_b[idx]))
    d = np.array(diffs)
    return {"diff_auprc": float(_safe_ap(y, score_a) - _safe_ap(y, score_b)),
            "ci_lo": float(np.percentile(d, 2.5)) if len(d) else float("nan"),
            "ci_hi": float(np.percentile(d, 97.5)) if len(d) else float("nan"),
            "p_a_better": float((d > 0).mean()) if len(d) else float("nan")}
