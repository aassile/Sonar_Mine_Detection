
import pandas as pd
import numpy as np

# ── Starting point ────────────────────────────────────────────────────────────
# sonar_df already has no nulls, no duplicates, and all values in [0,1]
# Cleaning steps: encode label, verify types, confirm ready for ML

# 1. Encode label: M=1 (Mine), R=0 (Rock)
cleaned_df = sonar_df.copy()
cleaned_df["target"] = (cleaned_df["label"] == "M").astype(int)

# 2. Confirm all feature columns are float64
_feat_cols = [c for c in cleaned_df.columns if c.startswith("freq_")]
cleaned_df[_feat_cols] = cleaned_df[_feat_cols].astype(float)

# 3. Drop the raw string label (replaced by numeric target)
cleaned_df = cleaned_df.drop(columns=["label"])

# ── Verify ────────────────────────────────────────────────────────────────────
print("=== Cleaned Dataset ===")
print(f"Shape   : {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
print(f"Dtypes  : {cleaned_df.dtypes.value_counts().to_dict()}")
print(f"Nulls   : {cleaned_df.isnull().sum().sum()}")
print(f"Target  : 1=Mine ({cleaned_df['target'].sum()}) | 0=Rock ({(cleaned_df['target']==0).sum()})")
print(f"\nColumn range: {list(cleaned_df.columns[:4])} … {list(cleaned_df.columns[-3:])}")
print(f"\nReady for analysis ✓")
