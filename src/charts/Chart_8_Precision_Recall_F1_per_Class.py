
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    model_y_test, y_pred, labels=[0, 1]
)

classes   = ["Rock (0)", "Mine (1)"]
metrics   = {"Precision": precision, "Recall": recall, "F1-Score": f1}
colors    = {"Precision": "#A1C9F4", "Recall": "#8DE5A1", "F1-Score": "#FFB482"}

x = np.arange(len(classes))
bar_width = 0.22

fig_metrics = go.Figure()

for i, (metric_name, values) in enumerate(metrics.items()):
    fig_metrics.add_trace(go.Bar(
        name=metric_name,
        x=[c + f"  ({metric_name})" for c in classes],
        y=np.round(values, 4),
        marker_color=colors[metric_name],
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#fbfbff", size=13),
        width=0.55,
    ))

# Grouped layout
fig_metrics = go.Figure()
for i, (metric_name, values) in enumerate(metrics.items()):
    fig_metrics.add_trace(go.Bar(
        name=metric_name,
        x=classes,
        y=np.round(values, 4),
        marker_color=colors[metric_name],
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#fbfbff", size=14),
        offsetgroup=i,
    ))

fig_metrics.update_layout(
    barmode="group",
    title=dict(
        text="Precision, Recall & F1-Score per Class",
        font=dict(color="#fbfbff", size=18),
        x=0.5,
    ),
    paper_bgcolor="#1D1D20",
    plot_bgcolor="#1D1D20",
    font=dict(color="#fbfbff"),
    xaxis=dict(
        title="Class",
        tickfont=dict(color="#fbfbff", size=13),
        gridcolor="#333",
        linecolor="#555",
    ),
    yaxis=dict(
        title="Score",
        tickfont=dict(color="#fbfbff", size=12),
        gridcolor="#333",
        linecolor="#555",
        range=[0, 1.12],
    ),
    legend=dict(
        font=dict(color="#fbfbff"),
        bgcolor="#1D1D20",
        bordercolor="#555",
        borderwidth=1,
    ),
    bargap=0.25,
    bargroupgap=0.05,
    margin=dict(t=80, b=60, l=60, r=40),
)

# Print table
print("=== Per-Class Classification Metrics ===\n")
print(f"{'Metric':<14} {'Rock':>8} {'Mine':>8}")
print("-" * 32)
for metric_name, values in metrics.items():
    print(f"{metric_name:<14} {values[0]:>8.3f} {values[1]:>8.3f}")
print("-" * 32)
print(f"{'Support':<14} {int(support[0]):>8}   {int(support[1]):>8}")
