
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

_feat_cols = [c for c in cleaned_df.columns if c.startswith("freq_")]
_X = cleaned_df[_feat_cols].values
_y = cleaned_df["target"].values

# ── Train / Test split (80/20, stratified) ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=42, stratify=_y
)

# Scale
_scaler = StandardScaler()
X_train_sc = _scaler.fit_transform(X_train)
X_test_sc  = _scaler.transform(X_test)

# ── Random Forest ─────────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)

y_pred  = rf.predict(X_test_sc)
y_proba = rf.predict_proba(X_test_sc)[:, 1]

# ── Metrics ───────────────────────────────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_proba)
cv_scores = cross_val_score(rf, _scaler.fit_transform(_X), _y, cv=5, scoring="accuracy")

print("=== Random Forest Results ===")
print(f"Test Accuracy      : {accuracy:.3f}  ({accuracy*100:.1f}%)")
print(f"ROC-AUC            : {auc:.3f}")
print(f"5-Fold CV Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rock", "Mine"]))

# ── Confusion matrix values ───────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:")
print(f"             Predicted Rock  Predicted Mine")
print(f"Actual Rock       {cm[0,0]:>5}           {cm[0,1]:>5}")
print(f"Actual Mine       {cm[1,0]:>5}           {cm[1,1]:>5}")

# ── Feature importance ────────────────────────────────────────────────────────
_importances = pd.Series(rf.feature_importances_, index=_feat_cols).sort_values(ascending=False)
print(f"\nTop 10 Most Important Features (Random Forest):")
print(_importances.head(10).round(4).to_string())

# Export for charts
model_accuracy   = accuracy
model_auc        = auc
model_cv_scores  = cv_scores
model_cm         = cm
feat_importances = _importances
model_y_test     = y_test
model_y_proba    = y_proba
