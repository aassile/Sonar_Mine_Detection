"""Tests for sonar_detection.models module."""

from sklearn.ensemble import ExtraTreesClassifier

from sonar_detection.models import (
    BEST_ET_PARAMS,
    build_extra_trees,
    build_models,
    tune_extra_trees,
)
from sonar_detection.preprocessing import (
    encode_target,
    scale_features,
    split_features_target,
    train_test_split_sonar,
)
from tests.conftest import make_raw_df


def _prepared(n=120):
    df = encode_target(make_raw_df(n=n))
    x, y = split_features_target(df)
    x_tr, x_te, y_tr, y_te = train_test_split_sonar(x, y)
    x_tr_sc, x_te_sc, _ = scale_features(x_tr, x_te)
    return x_tr_sc, x_te_sc, y_tr, y_te


def test_build_models():
    models = build_models()
    assert set(models) == {"Random Forest", "Extra Trees", "Gradient Boosting", "Neural Net"}


def test_build_extra_trees_defaults():
    model = build_extra_trees()
    assert isinstance(model, ExtraTreesClassifier)
    assert model.n_estimators == BEST_ET_PARAMS["n_estimators"]
    assert model.max_features == BEST_ET_PARAMS["max_features"]


def test_build_extra_trees_override():
    model = build_extra_trees(params={"n_estimators": 50})
    assert model.n_estimators == 50


def test_tune_extra_trees():
    x_tr, _, y_tr, _ = _prepared()
    search = tune_extra_trees(x_tr, y_tr, n_iter=4, cv_splits=3)
    assert hasattr(search, "best_estimator_")
    assert 0.0 <= search.best_score_ <= 1.0
    assert isinstance(search.best_params_, dict)
