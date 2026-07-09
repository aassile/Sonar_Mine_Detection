"""Tests for sonar_detection.preprocessing module."""

import numpy as np

from sonar_detection.preprocessing import (
    encode_target,
    scale_features,
    split_features_target,
    train_test_split_sonar,
)
from tests.conftest import make_raw_df


def test_encode_target():
    df = make_raw_df(n=20)
    out = encode_target(df)
    assert "target" in out.columns
    assert "label" not in out.columns
    assert set(out["target"].unique()) <= {0, 1}
    # M -> 1, R -> 0
    n_mine = (df["label"] == "M").sum()
    assert out["target"].sum() == n_mine


def test_split_features_target():
    df = encode_target(make_raw_df(n=30))
    x, y = split_features_target(df)
    assert x.shape == (30, 60)
    assert y.shape == (30,)


def test_train_test_split_stratified():
    df = encode_target(make_raw_df(n=100))
    x, y = split_features_target(df)
    x_tr, x_te, y_tr, y_te = train_test_split_sonar(x, y, test_size=0.2)
    assert len(x_tr) == 80
    assert len(x_te) == 20
    # stratification keeps class balance roughly intact
    assert abs(y_tr.mean() - y.mean()) < 0.1


def test_scale_features():
    df = encode_target(make_raw_df(n=60))
    x, y = split_features_target(df)
    x_tr, x_te, _, _ = train_test_split_sonar(x, y)
    x_tr_sc, x_te_sc, scaler = scale_features(x_tr, x_te)
    # training columns are standardized
    assert np.allclose(x_tr_sc.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(x_tr_sc.std(axis=0), 1, atol=1e-6)
    assert x_te_sc.shape == x_te.shape
