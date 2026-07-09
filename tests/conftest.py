"""Shared fixtures for the sonar_detection test suite."""

import numpy as np
import pandas as pd
import pytest

from sonar_detection.data import FEATURE_COLS, LABEL_COL


def make_raw_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic sonar-shaped DataFrame with a learnable signal.

    Mines and rocks are given slightly different band means so that models
    can achieve above-chance performance in tests.
    """
    rng = np.random.default_rng(seed)
    n_mine = n // 2
    n_rock = n - n_mine
    mine = rng.uniform(0.0, 1.0, size=(n_mine, 60)) * 0.6 + 0.2
    rock = rng.uniform(0.0, 1.0, size=(n_rock, 60)) * 0.6 + 0.1
    x = np.vstack([mine, rock])
    labels = ["M"] * n_mine + ["R"] * n_rock
    df = pd.DataFrame(x, columns=FEATURE_COLS)
    df[LABEL_COL] = labels
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_df()
