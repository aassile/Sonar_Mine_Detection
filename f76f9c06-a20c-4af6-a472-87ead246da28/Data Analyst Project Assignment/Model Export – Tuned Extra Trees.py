import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Rebuild tuned Extra Trees (same seed as all downstream blocks) ────────────
_feature_cols = [c for c in cleaned_df.columns if c != 'target']
_X = cleaned_df[_feature_cols].values
_y = cleaned_df['target'].values

_X_train, _X_test, _y_train, _y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=42, stratify=_y
)

app_scaler = StandardScaler()
_X_train_sc = app_scaler.fit_transform(_X_train)

app_model = ExtraTreesClassifier(
    n_estimators=200, max_features=0.4, max_depth=None,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
app_model.fit(_X_train_sc, _y_train)

# ── Feature names ─────────────────────────────────────────────────────────────
app_feature_names = [f'freq_{i+1}' for i in range(60)]

# ── Real class means (unscaled) for profile comparison in app ─────────────────
_df = cleaned_df.copy()
app_mine_means = _df[_df['target'] == 1][app_feature_names].mean().values
app_rock_means = _df[_df['target'] == 0][app_feature_names].mean().values

# ── Training stats for synthetic sample generation ────────────────────────────
app_train_mean = _df[app_feature_names].mean().values
app_train_std  = _df[app_feature_names].std().values

# ── Top permutation-important bands (from Chart 13 findings) ─────────────────
app_top_bands = ['freq_37', 'freq_45', 'freq_27', 'freq_12', 'freq_11']

print("=== Model Export Summary ===")
print(f"Model      : ExtraTreesClassifier (tuned)")
print(f"Features   : {len(app_feature_names)} frequency bands (freq_1 … freq_60)")
print(f"Scaler     : StandardScaler (fitted on 80% training data)")
print(f"Mine means : min={app_mine_means.min():.4f}, max={app_mine_means.max():.4f}")
print(f"Rock means : min={app_rock_means.min():.4f}, max={app_rock_means.max():.4f}")
print(f"Top bands  : {app_top_bands}")
print("\n✅ All app variables exported: app_model, app_scaler, app_feature_names,")
print("   app_mine_means, app_rock_means, app_train_mean, app_train_std, app_top_bands")
