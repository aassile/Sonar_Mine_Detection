"""End-to-end sonar mine-vs-rock classification pipeline."""

from __future__ import annotations

from typing import Any

from sonar_detection.data import FEATURE_COLS, load_sonar
from sonar_detection.evaluation import (
    compare_models,
    confusion,
    cross_val_auc,
    evaluate_model,
    permutation_importance_auc,
)
from sonar_detection.models import build_extra_trees, build_models
from sonar_detection.preprocessing import (
    encode_target,
    scale_features,
    split_features_target,
    train_test_split_sonar,
)


def run_pipeline(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Load, preprocess, train, tune, and evaluate the sonar classifier.

    Runs the full workflow:

    1. Load the labeled sonar CSV.
    2. Encode the target and split features/target.
    3. Stratified train/test split, then standard-scale features.
    4. Compare candidate models (Random Forest, Extra Trees, Gradient
       Boosting, Neural Net) by cross-validated ROC-AUC.
    5. Fit the tuned Extra Trees model and evaluate it on the test set.
    6. Compute permutation feature importance.

    Parameters
    ----------
    data_path:
        Path to the labeled sonar CSV (``data/sonar.csv``).
    test_size:
        Fraction held out for testing (default 0.2).
    random_state:
        Reproducibility seed (default 42).

    Returns
    -------
    dict with keys:
        ``comparison`` (DataFrame), ``model`` (fitted Extra Trees),
        ``scaler``, ``metrics`` (test-set dict), ``cv_auc`` (float),
        ``confusion`` (ndarray), ``importance`` (DataFrame),
        ``splits`` (dict of the scaled arrays).
    """
    # --- Load & encode ---
    df = load_sonar(data_path)
    df = encode_target(df)
    x, y = split_features_target(df)

    # --- Split & scale ---
    x_train, x_test, y_train, y_test = train_test_split_sonar(
        x, y, test_size=test_size, random_state=random_state
    )
    x_train_sc, x_test_sc, scaler = scale_features(x_train, x_test)

    # --- Model comparison ---
    comparison = compare_models(build_models(random_state), x_train_sc, x_test_sc, y_train, y_test)

    # --- Tuned Extra Trees ---
    model = build_extra_trees(random_state=random_state)
    model.fit(x_train_sc, y_train)
    metrics = evaluate_model(model, x_test_sc, y_test)
    cv_auc = cross_val_auc(model, x_train_sc, y_train, random_state=random_state)
    cm = confusion(model, x_test_sc, y_test)

    # --- Interpretability ---
    importance = permutation_importance_auc(
        model, x_test_sc, y_test, FEATURE_COLS, random_state=random_state
    )

    return {
        "comparison": comparison,
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "cv_auc": cv_auc,
        "confusion": cm,
        "importance": importance,
        "splits": {
            "x_train": x_train_sc,
            "x_test": x_test_sc,
            "y_train": y_train,
            "y_test": y_test,
        },
    }
