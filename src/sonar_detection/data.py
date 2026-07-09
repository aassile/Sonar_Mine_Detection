"""Data loading utilities for the UCI Sonar (Mines vs. Rocks) dataset."""

from __future__ import annotations

import pandas as pd

FEATURE_COLS = [f"freq_{i + 1}" for i in range(60)]
LABEL_COL = "label"
TARGET_COL = "target"


def load_sonar(path: str) -> pd.DataFrame:
    """Load the sonar dataset from a labeled CSV.

    The CSV is expected to contain 60 numeric frequency-band columns
    (``freq_1`` … ``freq_60``) and a string ``label`` column with values
    ``"M"`` (mine) or ``"R"`` (rock).

    Parameters
    ----------
    path:
        Path to the sonar CSV file.

    Returns
    -------
    pd.DataFrame
        Raw sonar table with 60 float feature columns and a string label.
    """
    df = pd.read_csv(path)
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if LABEL_COL in df.columns:
        df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.upper()
    return df


def load_sonar_raw(path: str) -> pd.DataFrame:
    """Load the original UCI ``sonar.all-data`` file (no header).

    The raw file has 60 unnamed feature columns followed by a single label
    column containing ``"M"`` or ``"R"``. This helper assigns the canonical
    column names used throughout the project.

    Parameters
    ----------
    path:
        Path to the raw ``sonar.all-data`` file.

    Returns
    -------
    pd.DataFrame
        Table with columns ``freq_1`` … ``freq_60`` and ``label``.
    """
    df = pd.read_csv(path, header=None)
    df.columns = FEATURE_COLS + [LABEL_COL]
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.upper()
    return df
