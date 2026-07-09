"""Tests for sonar_detection.pipeline module."""

from sonar_detection.pipeline import run_pipeline
from tests.conftest import make_raw_df


def test_run_pipeline(tmp_path):
    # Write a synthetic sonar CSV the pipeline can consume end-to-end.
    df = make_raw_df(n=160)
    p = tmp_path / "sonar.csv"
    df.to_csv(p, index=False)

    result = run_pipeline(str(p))

    assert set(result) >= {
        "comparison",
        "model",
        "scaler",
        "metrics",
        "cv_auc",
        "confusion",
        "importance",
        "splits",
    }
    assert len(result["comparison"]) == 4
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert 0.0 <= result["cv_auc"] <= 1.0
    assert result["confusion"].shape == (2, 2)
    assert len(result["importance"]) == 60
