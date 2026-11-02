import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from bridge_ct.models.bayesian import BayesianLogisticRegression
from bridge_ct.models.evaluate import evaluate_scores, expected_calibration_error, precision_at_k, recall_at_k
from bridge_ct.models.splits import make_splits


def test_precision_recall_at_k():
    y = np.array([1, 0, 1, 0, 0]); s = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert precision_at_k(y, s, 2) == 0.5 and recall_at_k(y, s, 3) == 1.0


def test_single_class_auc_is_nan_not_hidden():
    r = evaluate_scores(np.zeros(20, dtype=int), np.random.default_rng(0).random(20), n_boot=5)
    assert np.isnan(r["roc_auc"])


def test_perfect_ranking():
    y = np.array([1] * 5 + [0] * 95); r = evaluate_scores(y, y.astype(float), ks=[5], n_boot=10)
    assert r["auprc"] == 1.0 and r["precision_at_5"] == 1.0


def test_bayes_matches_unregularised_logistic():
    rng = np.random.default_rng(0); X = rng.normal(size=(4000, 3))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(-1 + X @ [1.0, -0.5, 0.0])))).astype(int)
    b = BayesianLogisticRegression(prior_scale=1e3, intercept_scale=1e3).fit(X, y, list("abc"))
    sk = LogisticRegression(C=1e8, max_iter=2000).fit((X - X.mean(0)) / X.std(0), y)
    np.testing.assert_allclose(b.beta_[1:], sk.coef_.ravel(), atol=1e-2)
    assert b.converged_ and ece(y, b.predict_proba(X)) < 0.02
    lo, hi = b.predict_interval(X[:50]); p = b.predict_proba(X[:50])
    assert ((lo <= p) & (p <= hi)).all()


def ece(y, p):
    return expected_calibration_error(y, p)


def test_splits_purge_prevents_label_overlap():
    sp = make_splits(list(range(2009, 2020)), 3, 2013, 2016, purge=True)
    assert max(sp["train"]) + 3 <= 2013 and max(sp["refit"]) + 3 <= 2016 and min(sp["test"]) == 2016
    with pytest.raises(ValueError):
        make_splits([2015, 2016], 3, 2015, 2016)
