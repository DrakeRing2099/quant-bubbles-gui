from __future__ import annotations

import warnings
from pathlib import Path

from joblib import load
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression


def load_joblib_with_sklearn_compat(path: str | Path):
    """
    Load a persisted estimator and backfill attributes expected by older
    scikit-learn runtimes when newer LogisticRegression artifacts are loaded.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        estimator = load(path)

    # Models saved under newer sklearn releases can omit this attribute, while
    # sklearn 1.7.x still reads it during predict_proba for binary classifiers.
    if isinstance(estimator, LogisticRegression) and not hasattr(estimator, "multi_class"):
        estimator.multi_class = "auto"

    return estimator
