"""Tests for sonar_detection.evaluation module."""

from sonar_detection.data import FEATURE_COLS
from sonar_detection.evaluation import (
    compare_models,
    confusion,
    cross_val_auc,
    evaluate_model,
    permutation_importance_auc,
    roc_points,
)
from sonar_detection.models import build_extra_trees, build_models
from sonar_detection.preprocessing import (
    encode_target,
    scale_features,
    split_features_target,
    train_test_split_sonar,
)
from tests.conftest import make_raw_df


def _fitted(n=140):
    df = encode_target(make_raw_df(n=n))
    x, y = split_features_target(df)
    x_tr, x_te, y_tr, y_te = train_test_split_sonar(x, y)
    x_tr_sc, x_te_sc, _ = scale_features(x_tr, x_te)
    model = build_extra_trees().fit(x_tr_sc, y_tr)
    return model, x_tr_sc, x_te_sc, y_tr, y_te


def test_evaluate_model():
    model, _, x_te, _, y_te = _fitted()
    m = evaluate_model(model, x_te, y_te)
    assert set(m) == {"accuracy", "roc_auc", "precision", "recall", "f1"}
    assert all(0.0 <= v <= 1.0 for v in m.values())


def test_cross_val_auc():
    model, x_tr, _, y_tr, _ = _fitted()
    auc = cross_val_auc(model, x_tr, y_tr, cv_splits=3)
    assert 0.0 <= auc <= 1.0


def test_compare_models():
    _, x_tr, x_te, y_tr, y_te = _fitted()
    df = compare_models(build_models(), x_tr, x_te, y_tr, y_te)
    assert len(df) == 4
    assert list(df.columns) == ["accuracy", "roc_auc", "cv_auc", "precision", "recall", "f1"]
    # sorted by cv_auc descending
    assert df["cv_auc"].is_monotonic_decreasing


def test_confusion_shape():
    model, _, x_te, _, y_te = _fitted()
    cm = confusion(model, x_te, y_te)
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y_te)


def test_roc_points():
    model, _, x_te, _, y_te = _fitted()
    fpr, tpr, auc = roc_points(model, x_te, y_te)
    assert len(fpr) == len(tpr)
    assert 0.0 <= auc <= 1.0


def test_permutation_importance():
    model, _, x_te, _, y_te = _fitted()
    imp = permutation_importance_auc(model, x_te, y_te, FEATURE_COLS, n_repeats=3)
    assert len(imp) == 60
    assert list(imp.columns) == ["feature", "importance", "std"]
    assert imp["importance"].is_monotonic_decreasing
