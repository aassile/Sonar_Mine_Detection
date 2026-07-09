import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix

# ── Reuse train/test split from upstream ──────────────────────────────────────
# X_train_sc, X_test_sc, y_train, y_test are scaled arrays from the RF block

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── XGBoost via sklearn GradientBoostingClassifier (XGB-equivalent; xgboost not installed) ──
from sklearn.ensemble import GradientBoostingClassifier

models = {
    "Random Forest":  RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Extra Trees":    ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Gradient Boost": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                                    max_iter=1000, random_state=42, early_stopping=True,
                                    validation_fraction=0.15, n_iter_no_change=20),
}

_results = []
comparison_cms = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    _y_pred  = model.predict(X_test_sc)
    _y_proba = model.predict_proba(X_test_sc)[:, 1]

    _acc  = accuracy_score(y_test, _y_pred)
    _auc  = roc_auc_score(y_test, _y_proba)
    _cv   = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="accuracy")
    _prec, _rec, _f1, _ = precision_recall_fscore_support(y_test, _y_pred, average="macro")

    comparison_cms[name] = confusion_matrix(y_test, _y_pred)

    _results.append({
        "Model":        name,
        "Test Accuracy": round(_acc, 4),
        "ROC-AUC":      round(_auc, 4),
        "CV Mean":      round(_cv.mean(), 4),
        "CV Std":       round(_cv.std(), 4),
        "Precision":    round(_prec, 4),
        "Recall":       round(_rec, 4),
        "F1-Score":     round(_f1, 4),
    })

comparison_df = pd.DataFrame(_results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

print("=== Model Comparison (all models use same train/test split & scaling) ===\n")
print(comparison_df.to_string(index=False))
print("\n⭐ Best by AUC :", comparison_df.iloc[0]["Model"],
      f"(AUC={comparison_df.iloc[0]['ROC-AUC']})")
print("⭐ Best by Acc :", comparison_df.sort_values("Test Accuracy", ascending=False).iloc[0]["Model"],
      f"(Acc={comparison_df.sort_values('Test Accuracy', ascending=False).iloc[0]['Test Accuracy']})")
