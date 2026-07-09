"""Sonar mine-vs-rock classification on the UCI Connectionist Bench dataset."""

__version__ = "0.1.0"

from sonar_detection.data import (
    FEATURE_COLS,
    LABEL_COL,
    TARGET_COL,
    load_sonar,
    load_sonar_raw,
)
from sonar_detection.evaluation import (
    compare_models,
    confusion,
    cross_val_auc,
    evaluate_model,
    permutation_importance_auc,
    roc_points,
)
from sonar_detection.models import (
    BEST_ET_PARAMS,
    build_extra_trees,
    build_models,
    tune_extra_trees,
)
from sonar_detection.pipeline import run_pipeline
from sonar_detection.preprocessing import (
    encode_target,
    scale_features,
    split_features_target,
    train_test_split_sonar,
)

__all__ = [
    "FEATURE_COLS",
    "LABEL_COL",
    "TARGET_COL",
    "load_sonar",
    "load_sonar_raw",
    "encode_target",
    "split_features_target",
    "train_test_split_sonar",
    "scale_features",
    "build_models",
    "build_extra_trees",
    "tune_extra_trees",
    "BEST_ET_PARAMS",
    "evaluate_model",
    "cross_val_auc",
    "compare_models",
    "confusion",
    "roc_points",
    "permutation_importance_auc",
    "run_pipeline",
]
