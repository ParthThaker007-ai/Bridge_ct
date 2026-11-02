"""Interpretable logistic regression (the blueprint's combined-feature model)."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LogisticModel:
    """sklearn pipeline: median imputation -> standardisation -> L2 logistic regression.

    Parameters
    ----------
    C : inverse regularisation strength (smaller = stronger shrinkage).
    class_weight : ``"balanced"`` re-weights the rare class, which helps ranking but distorts the *probability
        scale* - use the Bayesian model (no re-weighting) when calibrated probabilities matter.
    max_iter : solver iterations (lbfgs).
    """

    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced", max_iter: int = 2000, seed: int = 42) -> None:
        self.C, self.class_weight, self.max_iter, self.seed = C, class_weight, max_iter, seed
        self.columns: List[str] = []
        self.pipe = None

    def fit(self, X: np.ndarray, y: np.ndarray, columns: List[str]) -> "LogisticModel":
        if len(np.unique(y)) < 2:
            raise ValueError("training labels contain a single class; widen the training period")
        self.columns = list(columns)
        self.pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=self.C, class_weight=self.class_weight,
                                                     max_iter=self.max_iter, random_state=self.seed))
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(X)[:, 1]

    def coef_table(self) -> pd.DataFrame:
        coef = self.pipe[-1].coef_.ravel()
        return pd.DataFrame({"feature": self.columns, "coef": coef}).sort_values("coef", key=np.abs, ascending=False)
