
import pandas as pd
import numpy as np

_feat_cols = [c for c in cleaned_df.columns if c.startswith("freq_")]

# ── 1. Mean energy profile per class across frequency bands ───────────────────
mine_means = cleaned_df[cleaned_df["target"] == 1][_feat_cols].mean()
rock_means  = cleaned_df[cleaned_df["target"] == 0][_feat_cols].mean()

print("=== Mean Energy Profile (Mine vs Rock) — first 15 bands ===")
_compare = pd.DataFrame({"Mine": mine_means, "Rock": rock_means})
_compare["Δ (Mine-Rock)"] = (_compare["Mine"] - _compare["Rock"]).round(4)
print(_compare.head(15).round(4).to_string())

# ── 2. Which bands discriminate most? (largest mean difference) ───────────────
_compare["abs_diff"] = _compare["Δ (Mine-Rock)"].abs()
top_bands = _compare.nlargest(10, "abs_diff")
print(f"\n=== Top 10 Most Discriminating Frequency Bands ===")
print(top_bands[["Mine", "Rock", "Δ (Mine-Rock)"]].round(4).to_string())

# ── 3. Band-level variance (how spread out are readings for each band?) ────────
_var = cleaned_df[_feat_cols].var()
print(f"\n=== Feature Variance ===")
print(f"Highest variance band: {_var.idxmax()} ({_var.max():.4f})")
print(f"Lowest  variance band: {_var.idxmin()} ({_var.min():.4f})")

# ── 4. Correlation among features (average pairwise) ─────────────────────────
_corr_matrix = cleaned_df[_feat_cols].corr()
_upper = _corr_matrix.where(np.triu(np.ones_like(_corr_matrix, dtype=bool), k=1))
_avg_corr = _upper.stack().mean()
_high_corr_pairs = (_upper.abs() > 0.90).sum().sum()
print(f"\n=== Feature Correlations ===")
print(f"Average pairwise correlation  : {_avg_corr:.3f}")
print(f"Pairs with |corr| > 0.90      : {int(_high_corr_pairs)}")

# Export for visualisation block
eda_mine_means = mine_means
eda_rock_means  = rock_means
eda_top_bands   = top_bands
eda_corr_matrix = _corr_matrix

print("\nEDA complete ✓")
