"""Model builders and hyperparameter tuning for sonar classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 42

# Best Extra Trees configuration found by randomized search (see tune_extra_trees).
BEST_ET_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_features": 0.4,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
}


def build_models(random_state: int = RANDOM_STATE) -> dict[str, ClassifierMixin]:
    """Instantiate the candidate classifiers compared in this project.

    Parameters
    ----------
    random_state:
        Seed applied to every estimator for reproducibility.

    Returns
    -------
    dict[str, ClassifierMixin]
        Mapping of model name → unfitted estimator. Includes Random Forest,
        Extra Trees, Gradient Boosting, and a small Neural Network.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "Neural Net": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=random_state,
        ),
    }


def build_extra_trees(
    params: dict[str, Any] | None = None,
    random_state: int = RANDOM_STATE,
) -> ExtraTreesClassifier:
    """Build the tuned Extra Trees classifier.

    Parameters
    ----------
    params:
        Hyperparameter overrides. Defaults to :data:`BEST_ET_PARAMS`.
    random_state:
        Reproducibility seed.

    Returns
    -------
    ExtraTreesClassifier
        Unfitted classifier configured with the given (or best) parameters.
    """
    cfg = dict(BEST_ET_PARAMS)
    if params:
        cfg.update(params)
    return ExtraTreesClassifier(random_state=random_state, n_jobs=-1, **cfg)


def tune_extra_trees(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 80,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    """Run a randomized hyperparameter search over Extra Trees.

    Optimizes ROC-AUC via stratified k-fold cross-validation.

    Parameters
    ----------
    x_train:
        Scaled training feature matrix.
    y_train:
        Training target vector.
    n_iter:
        Number of random parameter configurations to sample (default 80).
    cv_splits:
        Number of stratified CV folds (default 5).
    random_state:
        Reproducibility seed.

    Returns
    -------
    RandomizedSearchCV
        The fitted search object (``best_estimator_``, ``best_params_``,
        ``best_score_``, and ``cv_results_`` available).
    """
    param_distributions = {
        "n_estimators": [200, 400, 600],
        "max_features": ["sqrt", "log2", 0.4, 0.6],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        ExtraTreesClassifier(random_state=random_state, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search
