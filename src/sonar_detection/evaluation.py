"""Evaluation metrics and interpretability helpers for sonar models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

RANDOM_STATE = 42


def evaluate_model(
    model: ClassifierMixin,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute headline test-set metrics for a fitted classifier.

    Parameters
    ----------
    model:
        A fitted classifier exposing ``predict`` and ``predict_proba``.
    x_test:
        Scaled test feature matrix.
    y_test:
        True test labels.

    Returns
    -------
    dict[str, float]
        Keys: ``accuracy``, ``roc_auc``, ``precision``, ``recall``, ``f1``
        (precision/recall/f1 are for the positive "Mine" class).
    """
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def cross_val_auc(
    model: ClassifierMixin,
    x: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> float:
    """Mean ROC-AUC across stratified cross-validation folds.

    Parameters
    ----------
    model:
        Unfitted estimator (cloned internally by scikit-learn).
    x:
        Feature matrix.
    y:
        Target vector.
    cv_splits:
        Number of stratified folds (default 5).
    random_state:
        Reproducibility seed.

    Returns
    -------
    float
        Mean cross-validated ROC-AUC.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, x, y, cv=cv, scoring="roc_auc")
    return float(scores.mean())


def compare_models(
    models: dict[str, ClassifierMixin],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Fit each candidate model and tabulate comparison metrics.

    Parameters
    ----------
    models:
        Mapping of model name → unfitted estimator (see
        :func:`sonar_detection.models.build_models`).
    x_train, x_test:
        Scaled train/test feature matrices.
    y_train, y_test:
        Train/test target vectors.

    Returns
    -------
    pd.DataFrame
        One row per model with columns ``accuracy``, ``roc_auc``, ``cv_auc``,
        ``precision``, ``recall``, ``f1``, sorted by ``cv_auc`` descending.
    """
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        metrics["cv_auc"] = cross_val_auc(model, x_train, y_train)
        metrics["model"] = name
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("model")
    cols = ["accuracy", "roc_auc", "cv_auc", "precision", "recall", "f1"]
    return df[cols].sort_values("cv_auc", ascending=False)


def confusion(
    model: ClassifierMixin,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    """Return the 2x2 confusion matrix (rows = true, cols = predicted).

    Parameters
    ----------
    model:
        Fitted classifier.
    x_test:
        Scaled test features.
    y_test:
        True test labels.

    Returns
    -------
    np.ndarray
        Confusion matrix with label order ``[0 (Rock), 1 (Mine)]``.
    """
    y_pred = model.predict(x_test)
    return confusion_matrix(y_test, y_pred, labels=[0, 1])


def roc_points(
    model: ClassifierMixin,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve points and AUC.

    Parameters
    ----------
    model:
        Fitted classifier exposing ``predict_proba``.
    x_test:
        Scaled test features.
    y_test:
        True test labels.

    Returns
    -------
    (fpr, tpr, auc) : tuple
        False-positive rates, true-positive rates, and the ROC-AUC.
    """
    y_proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = float(roc_auc_score(y_test, y_proba))
    return fpr, tpr, auc


def permutation_importance_auc(
    model: ClassifierMixin,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 30,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Permutation feature importance measured by drop in ROC-AUC.

    Each feature column is shuffled ``n_repeats`` times; the mean decrease in
    ROC-AUC relative to the unpermuted baseline quantifies its importance.

    Parameters
    ----------
    model:
        Fitted classifier.
    x_test:
        Scaled test feature matrix.
    y_test:
        True test labels.
    feature_names:
        Column names aligned with the columns of ``x_test``.
    n_repeats:
        Number of shuffles per feature (default 30).
    random_state:
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame
        Columns ``feature``, ``importance`` (mean AUC drop), ``std``,
        sorted by importance descending.
    """
    rng = np.random.default_rng(random_state)
    baseline = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    n_features = x_test.shape[1]
    drops = np.zeros((n_features, n_repeats))
    for feat in range(n_features):
        for r in range(n_repeats):
            x_perm = x_test.copy()
            x_perm[:, feat] = rng.permutation(x_perm[:, feat])
            drops[feat, r] = baseline - roc_auc_score(y_test, model.predict_proba(x_perm)[:, 1])
    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": drops.mean(axis=1),
            "std": drops.std(axis=1),
        }
    )
    return result.sort_values("importance", ascending=False).reset_index(drop=True)
