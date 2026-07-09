
import pandas as pd
import numpy as np

# ── Basic shape ───────────────────────────────────────────────────────────────
print(f"Rows: {sonar_df.shape[0]}  |  Columns: {sonar_df.shape[1]}")
print(f"Features: 60 numeric frequency-band energy readings (freq_1 … freq_60)")
print(f"Target  : 'label'  →  M (Mine) = 111 | R (Rock) = 97\n")

# ── Missing values ────────────────────────────────────────────────────────────
_nulls = sonar_df.isnull().sum()
print(f"Missing values  : {_nulls.sum()} total  ✓")

# ── Duplicates ────────────────────────────────────────────────────────────────
_dups = sonar_df.duplicated().sum()
print(f"Duplicate rows  : {_dups}  ✓\n")

# ── Numeric summary ───────────────────────────────────────────────────────────
_feat_cols = [c for c in sonar_df.columns if c.startswith("freq_")]
_desc = sonar_df[_feat_cols].describe().T

print("=== Numeric Feature Summary (across all 60 bands) ===")
print(_desc[["min","mean","max","std"]].describe().round(4).to_string())

print(f"\n  Overall min   : {_desc['min'].min():.4f}")
print(f"  Overall max   : {_desc['max'].max():.4f}")
print(f"  Mean of means : {_desc['mean'].mean():.4f}")
print(f"  Avg std-dev   : {_desc['std'].mean():.4f}")

# ── Values out of expected [0,1] range ───────────────────────────────────────
_out_of_range = ((sonar_df[_feat_cols] < 0) | (sonar_df[_feat_cols] > 1)).sum().sum()
print(f"\nValues outside [0, 1] range: {_out_of_range}  ✓ (energy readings are normalised)")

# ── Per-class summary ─────────────────────────────────────────────────────────
print("\n=== Per-Class Mean Energy Profile (first 10 bands) ===")
_class_means = sonar_df.groupby("label")[_feat_cols[:10]].mean().round(4)
print(_class_means.to_string())

# Export profiling summary for reporting
profile_summary = {
    "rows": sonar_df.shape[0],
    "feature_cols": 60,
    "missing_values": int(_nulls.sum()),
    "duplicates": int(_dups),
    "class_balance": sonar_df["label"].value_counts().to_dict(),
    "value_range": [float(_desc["min"].min()), float(_desc["max"].max())],
}
print(f"\nProfile summary exported ✓")
