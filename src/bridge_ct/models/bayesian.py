"""Bayesian logistic regression with a Laplace (Gaussian) posterior approximation.

Model (blueprint section 4.5):  y_i ~ Bernoulli(p_i),  logit(p_i) = beta_0 + x_i . beta,
beta_j ~ Normal(0, prior_scale^2),  beta_0 ~ Normal(0, intercept_scale^2).

Fitting: Newton-Raphson (IRLS) with backtracking finds the MAP; the posterior is approximated by
N(beta_MAP, H^-1) where H is the Hessian of the negative log posterior. Predictions integrate the logistic
function over this Gaussian with Gauss-Hermite quadrature, and credible intervals use the exact monotone
transform of the Gaussian quantiles of the logit. Nothing is re-weighted, so probabilities stay calibrated to
the training base rate. (A full MCMC alternative is PyMC; this closed-form route is deterministic, fast and
dependency-free, which suits reproducible benchmarking.)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


class BayesianLogisticRegression:
    """Parameters: ``prior_scale`` (std of Normal prior on standardised coefficients), ``intercept_scale``,
    ``max_iter`` / ``tol`` (Newton stopping), ``n_quad`` (Gauss-Hermite nodes)."""

    def __init__(self, prior_scale: float = 1.0, intercept_scale: float = 10.0, max_iter: int = 100,
                 tol: float = 1e-8, n_quad: int = 40) -> None:
        if prior_scale <= 0 or intercept_scale <= 0:
            raise ValueError("prior scales must be positive")
        self.prior_scale, self.intercept_scale = prior_scale, intercept_scale
        self.max_iter, self.tol, self.n_quad = max_iter, tol, n_quad
        self.columns: List[str] = []
        self.converged_ = False
        self.n_iter_ = 0

    # ------------------------------------------------------------------ preprocessing (fitted on TRAIN only)
    def _fit_transform_stats(self, X: np.ndarray) -> None:
        self.median_ = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.median_ = np.where(np.isnan(self.median_), 0.0, self.median_)
        Xi = self._impute(X)
        self.mean_ = Xi.mean(axis=0)
        std = Xi.std(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)  # constant columns stay at 0 after centring

    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float, copy=True)
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.broadcast_to(self.median_, X.shape)[bad]
        return X

    def _design(self, X: np.ndarray) -> np.ndarray:
        Xs = (self._impute(X) - self.mean_) / self.std_
        return np.hstack([np.ones((Xs.shape[0], 1)), Xs])

    # ------------------------------------------------------------------ fitting
    def fit(self, X: np.ndarray, y: np.ndarray, columns: List[str]) -> "BayesianLogisticRegression":
        y = np.asarray(y, dtype=float)
        if len(np.unique(y)) < 2:
            raise ValueError("training labels contain a single class; widen the training period")
        self.columns = list(columns)
        self._fit_transform_stats(np.asarray(X, dtype=float))
        Xb = self._design(X)
        d = Xb.shape[1]
        prec = np.r_[1.0 / self.intercept_scale ** 2, np.full(d - 1, 1.0 / self.prior_scale ** 2)]

        def objective(b: np.ndarray) -> float:
            z = Xb @ b
            return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(prec * b * b))

        beta = np.zeros(d)
        beta[0] = np.log(y.mean() / (1 - y.mean()))  # start at the base rate
        f0 = objective(beta)
        for it in range(1, self.max_iter + 1):
            p = expit(Xb @ beta)
            grad = Xb.T @ (p - y) + prec * beta
            H = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(prec)
            step = np.linalg.solve(H, grad)
            t, improved = 1.0, False
            while t > 1e-10:  # backtracking keeps the objective monotone even with separable data
                cand = beta - t * step
                fc = objective(cand)
                if fc <= f0 + 1e-12:
                    improved = True
                    break
                t *= 0.5
            if not improved:
                self.converged_, self.n_iter_ = True, it  # no descent direction left: at the optimum numerically
                break
            delta = np.max(np.abs(cand - beta))
            beta, f0 = cand, fc
            self.n_iter_ = it
            if delta < self.tol:
                self.converged_ = True
                break
        p = expit(Xb @ beta)
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(prec)
        cov = np.linalg.inv(H)
        self.beta_ = beta
        self.cov_ = (cov + cov.T) / 2.0
        self._gh_x, self._gh_w = np.polynomial.hermite.hermgauss(self.n_quad)
        return self

    # ------------------------------------------------------------------ prediction
    def _logit_moments(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xb = self._design(X)
        mean = Xb @ self.beta_
        var = np.einsum("ij,jk,ik->i", Xb, self.cov_, Xb)
        return mean, np.sqrt(np.clip(var, 0.0, None))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior predictive probability E[sigmoid(z)], z ~ N(mean, sd^2), via Gauss-Hermite quadrature."""
        m, s = self._logit_moments(X)
        z = m[:, None] + np.sqrt(2.0) * s[:, None] * self._gh_x[None, :]
        return (expit(z) @ self._gh_w) / np.sqrt(np.pi)

    def predict_interval(self, X: np.ndarray, level: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """Central credible interval for the candidate's link probability (parameter uncertainty only)."""
        m, s = self._logit_moments(X)
        q = norm.ppf(0.5 + level / 2.0)
        return expit(m - q * s), expit(m + q * s)

    def contributions(self, X: np.ndarray) -> pd.DataFrame:
        """Per-feature contribution to the logit: standardised value x posterior-mean coefficient."""
        Xs = self._design(X)[:, 1:]
        return pd.DataFrame(Xs * self.beta_[1:], columns=self.columns)

    def coef_table(self, level: float = 0.94) -> pd.DataFrame:
        """Posterior mean, sd and central interval of each standardised coefficient (intercept excluded)."""
        sd = np.sqrt(np.diag(self.cov_))
        q = norm.ppf(0.5 + level / 2.0)
        mean = self.beta_
        df = pd.DataFrame({"feature": ["intercept"] + self.columns, "mean": mean, "sd": sd,
                           "lo": mean - q * sd, "hi": mean + q * sd})
        return df.iloc[1:].sort_values("mean", key=np.abs, ascending=False).reset_index(drop=True)
