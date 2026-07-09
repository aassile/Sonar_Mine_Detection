"""Tests for sonar_detection.data module."""

import pandas as pd

from sonar_detection.data import FEATURE_COLS, load_sonar, load_sonar_raw

RAW_LINES = "\n".join(",".join(["0.5"] * 60 + [lab]) for lab in ["M", "R", "M"])


def test_load_sonar_raw(tmp_path):
    p = tmp_path / "sonar.all-data"
    p.write_text(RAW_LINES + "\n")
    df = load_sonar_raw(str(p))
    assert df.shape == (3, 61)
    assert list(df.columns) == FEATURE_COLS + ["label"]
    assert set(df["label"].unique()) <= {"M", "R"}


def test_load_sonar_csv(tmp_path):
    df_in = pd.DataFrame(
        [[0.1] * 60 + ["m"], [0.2] * 60 + [" r "]], columns=FEATURE_COLS + ["label"]
    )
    p = tmp_path / "sonar.csv"
    df_in.to_csv(p, index=False)
    df = load_sonar(str(p))
    assert len(df) == 2
    # labels are upper-cased and stripped
    assert set(df["label"].unique()) == {"M", "R"}
    assert pd.api.types.is_numeric_dtype(df["freq_1"])
