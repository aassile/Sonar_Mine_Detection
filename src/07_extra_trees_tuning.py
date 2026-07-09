import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ── Reuse train/test split from upstream ─────────────────────────────────────
# X_train_sc, X_test_sc, y_train, y_test all flow in from RF block

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Baseline Extra Trees (default params) ─────────────────────────────────────
et_base = ExtraTreesClassifier(n_estimators=300, random_state=42)
et_base.fit(X_train_sc, y_train)
base_acc  = accuracy_score(y_test, et_base.predict(X_test_sc))
base_auc  = roc_auc_score(y_test, et_base.predict_proba(X_test_sc)[:, 1])
base_cv   = cross_val_score(et_base, X_train_sc, y_train, cv=cv, scoring='roc_auc').mean()
print(f"Baseline Extra Trees → Acc: {base_acc:.4f} | AUC: {base_auc:.4f} | CV AUC: {base_cv:.4f}")

# ── Grid Search ──────────────────────────────────────────────────────────────
# Key parameters for ExtraTreesClassifier (highest-impact first)
param_grid = {
    'n_estimators':    [200, 400, 600],
    'max_features':    ['sqrt', 'log2', 0.4, 0.6],
    'max_depth':       [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'class_weight':    [None, 'balanced'],
}

# Total combinations = 3×4×4×3×3×2 = 864 — use RandomizedSearchCV subset for speed
from sklearn.model_selection import RandomizedSearchCV

et_search = RandomizedSearchCV(
    ExtraTreesClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=80,               # 80 random configurations
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=0,
    refit=True,
)
et_search.fit(X_train_sc, y_train)

# ── Best model evaluation ─────────────────────────────────────────────────────
et_best        = et_search.best_estimator_
tuned_pred     = et_best.predict(X_test_sc)
tuned_proba    = et_best.predict_proba(X_test_sc)[:, 1]
tuned_acc      = accuracy_score(y_test, tuned_pred)
tuned_auc      = roc_auc_score(y_test, tuned_proba)
tuned_cv       = cross_val_score(et_best, X_train_sc, y_train, cv=cv, scoring='roc_auc').mean()
tuned_cv_acc   = cross_val_score(et_best, X_train_sc, y_train, cv=cv, scoring='accuracy').mean()

print(f"\n=== Tuned Extra Trees Results ===")
print(f"Best CV AUC (search)  : {et_search.best_score_:.4f}")
print(f"Test Accuracy         : {tuned_acc:.4f}  ({tuned_acc*100:.1f}%)")
print(f"Test AUC              : {tuned_auc:.4f}")
print(f"5-Fold CV AUC         : {tuned_cv:.4f}")
print(f"5-Fold CV Accuracy    : {tuned_cv_acc:.4f}")

print(f"\nBest Parameters:")
for k, v in et_search.best_params_.items():
    print(f"  {k:20s}: {v}")

print(f"\nClassification Report (Tuned):")
print(classification_report(y_test, tuned_pred, target_names=['Rock', 'Mine']))

# ── Delta vs baseline ─────────────────────────────────────────────────────────
print(f"\n=== Improvement over Baseline ===")
print(f"Accuracy  : {base_acc:.4f} → {tuned_acc:.4f}  (Δ {tuned_acc - base_acc:+.4f})")
print(f"AUC       : {base_auc:.4f} → {tuned_auc:.4f}  (Δ {tuned_auc - base_auc:+.4f})")
print(f"CV AUC    : {base_cv:.4f} → {tuned_cv:.4f}  (Δ {tuned_cv - base_cv:+.4f})")

# ── Top 10 search results ─────────────────────────────────────────────────────
results_df = pd.DataFrame(et_search.cv_results_)
top10_df = (
    results_df[['param_n_estimators','param_max_features','param_max_depth',
                'param_min_samples_split','param_min_samples_leaf',
                'param_class_weight','mean_test_score','std_test_score','rank_test_score']]
    .sort_values('rank_test_score')
    .head(10)
    .reset_index(drop=True)
)
top10_df.columns = ['n_est','max_feat','max_depth','min_split','min_leaf','class_wt','CV_AUC','CV_std','rank']
print(f"\n=== Top 10 Configurations ===")
print(top10_df.to_string(index=False))

# Export for downstream chart
et_tuned_acc    = tuned_acc
et_tuned_auc    = tuned_auc
et_tuned_cv_auc = tuned_cv
et_tuned_proba  = tuned_proba
et_tuned_pred   = tuned_pred
et_top10        = top10_df
