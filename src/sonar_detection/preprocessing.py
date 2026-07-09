"""Cleaning, encoding, and train/test splitting for sonar data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sonar_detection.data import FEATURE_COLS, LABEL_COL, TARGET_COL


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the string label into a numeric binary target.

    Maps ``"M"`` (mine) → 1 and ``"R"`` (rock) → 0, adds a ``target`` column,
    and drops the original string ``label`` column.

    Parameters
    ----------
    df:
        Input DataFrame containing a ``label`` column.

    Returns
    -------
    pd.DataFrame
        Copy with a numeric ``target`` column and no ``label`` column.
    """
    df = df.copy()
    df[TARGET_COL] = (df[LABEL_COL] == "M").astype(int)
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float)
    df = df.drop(columns=[LABEL_COL])
    return df


def split_features_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Separate the feature matrix and target vector.

    Parameters
    ----------
    df:
        DataFrame containing the 60 frequency columns and a ``target`` column.

    Returns
    -------
    (X, y) : tuple of np.ndarray
        Feature matrix of shape (n_samples, 60) and target vector.
    """
    x = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=int)
    return x, y


def train_test_split_sonar(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split.

    Parameters
    ----------
    x:
        Feature matrix.
    y:
        Target vector.
    test_size:
        Fraction held out for testing (default 0.2).
    random_state:
        Reproducibility seed (default 42).

    Returns
    -------
    (X_train, X_test, y_train, y_test) : tuple of np.ndarray
    """
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on the training data and transform both sets.

    Parameters
    ----------
    x_train:
        Training feature matrix (scaler is fit on this).
    x_test:
        Test feature matrix (transformed with the fitted scaler).

    Returns
    -------
    (X_train_scaled, X_test_scaled, scaler) : tuple
        The scaled matrices and the fitted :class:`StandardScaler`.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler
