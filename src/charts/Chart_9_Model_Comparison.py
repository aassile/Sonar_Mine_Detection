import plotly.graph_objects as go
import pandas as pd

# ── Data ──────────────────────────────────────────────────────────────────────
_model_names = comparison_df["Model"].tolist()
_metrics = {
    "Test Accuracy": comparison_df["Test Accuracy"].tolist(),
    "ROC-AUC":       comparison_df["ROC-AUC"].tolist(),
    "CV Mean":       comparison_df["CV Mean"].tolist(),
    "F1-Score":      comparison_df["F1-Score"].tolist(),
}

_colors = ["#A1C9F4", "#8DE5A1", "#FFB482", "#FF9F9B"]
_metric_names = list(_metrics.keys())

fig_model_comparison = go.Figure()

for i, (metric, values) in enumerate(_metrics.items()):
    fig_model_comparison.add_trace(go.Bar(
        name=metric,
        x=_model_names,
        y=values,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#fbfbff", size=11),
        marker_color=_colors[i],
        marker_line=dict(color="#1D1D20", width=1),
    ))

fig_model_comparison.update_layout(
    barmode="group",
    title=dict(text="Model Comparison — Accuracy, AUC, CV Score & F1", font=dict(color="#fbfbff", size=18), x=0.5),
    plot_bgcolor="#1D1D20",
    paper_bgcolor="#1D1D20",
    font=dict(color="#fbfbff"),
    legend=dict(bgcolor="#1D1D20", font=dict(color="#fbfbff"), orientation="h", y=-0.2, x=0.5, xanchor="center"),
    xaxis=dict(title="Model", gridcolor="#333", tickfont=dict(color="#fbfbff")),
    yaxis=dict(title="Score", gridcolor="#333", range=[0, 1.12], tickfont=dict(color="#fbfbff")),
    height=520,
    margin=dict(t=80, b=120, l=60, r=30),
)

fig_model_comparison
