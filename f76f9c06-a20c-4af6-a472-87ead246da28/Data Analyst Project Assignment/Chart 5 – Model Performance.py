
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.metrics import roc_curve

# ── ROC Curve ─────────────────────────────────────────────────────────────────
_fpr, _tpr, _ = roc_curve(model_y_test, model_y_proba)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=_fpr, y=_tpr,
    mode="lines",
    name=f"Random Forest (AUC = {model_auc:.3f})",
    line=dict(color="#8DE5A1", width=3),
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    name="Random Baseline",
    line=dict(color="#555", dash="dash"),
))
fig_roc.update_layout(
    title=dict(text="ROC Curve – Sonar Mine vs. Rock Classifier", font=dict(size=17, color="#fbfbff")),
    xaxis=dict(title="False Positive Rate", gridcolor="#333", color="#909094"),
    yaxis=dict(title="True Positive Rate", gridcolor="#333", color="#909094"),
    plot_bgcolor="#1D1D20", paper_bgcolor="#1D1D20",
    font=dict(color="#fbfbff"),
    legend=dict(bgcolor="#1D1D20", bordercolor="#555", borderwidth=1),
    height=420,
)
fig_roc.show()
