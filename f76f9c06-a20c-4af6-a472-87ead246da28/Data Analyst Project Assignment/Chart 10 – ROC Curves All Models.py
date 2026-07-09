import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# Refit all models to get per-model ROC curves (same split as comparison block)
_models = {
    "Random Forest":  RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Extra Trees":    ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Gradient Boost": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                                    max_iter=1000, random_state=42, early_stopping=True,
                                    validation_fraction=0.15, n_iter_no_change=20),
}

_palette = ["#A1C9F4", "#8DE5A1", "#FFB482", "#FF9F9B"]

fig_roc_comparison = go.Figure()

for (name, model), color in zip(_models.items(), _palette):
    model.fit(X_train_sc, y_train)
    _proba = model.predict_proba(X_test_sc)[:, 1]
    _fpr, _tpr, _ = roc_curve(y_test, _proba)
    _auc = roc_auc_score(y_test, _proba)
    fig_roc_comparison.add_trace(go.Scatter(
        x=_fpr, y=_tpr, mode="lines",
        name=f"{name} (AUC={_auc:.3f})",
        line=dict(color=color, width=2.5),
    ))

# Baseline
fig_roc_comparison.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    name="Random Baseline",
    line=dict(color="#909094", width=1.5, dash="dash"),
))

fig_roc_comparison.update_layout(
    title=dict(text="ROC Curves — All Models", font=dict(color="#fbfbff", size=18), x=0.5),
    plot_bgcolor="#1D1D20",
    paper_bgcolor="#1D1D20",
    font=dict(color="#fbfbff"),
    xaxis=dict(title="False Positive Rate", gridcolor="#333", tickfont=dict(color="#fbfbff")),
    yaxis=dict(title="True Positive Rate", gridcolor="#333", tickfont=dict(color="#fbfbff")),
    legend=dict(bgcolor="#1D1D20", font=dict(color="#fbfbff")),
    height=500,
    margin=dict(t=80, b=60, l=60, r=30),
)

fig_roc_comparison
