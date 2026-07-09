import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ── Subplot: Elbow (left) + Silhouette (right) ───────────────────────────────
fig_elbow_silhouette = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Elbow Curve (Inertia)', 'Silhouette Score'),
    horizontal_spacing=0.14
)

_k_vals = cluster_k_range
_colors = ['#ffd400' if k == cluster_optimal_k else '#A1C9F4' for k in _k_vals]

# Elbow
fig_elbow_silhouette.add_trace(
    go.Scatter(
        x=_k_vals, y=cluster_inertias,
        mode='lines+markers',
        line=dict(color='#A1C9F4', width=2),
        marker=dict(color=_colors, size=10, line=dict(color='#fbfbff', width=1)),
        name='Inertia',
        hovertemplate='k=%{x}<br>Inertia=%{y:.1f}<extra></extra>'
    ), row=1, col=1
)

# Silhouette
fig_elbow_silhouette.add_trace(
    go.Bar(
        x=_k_vals, y=cluster_sil_scores,
        marker_color=_colors,
        name='Silhouette',
        text=[f'{s:.4f}' for s in cluster_sil_scores],
        textposition='outside',
        textfont=dict(color='#fbfbff', size=10),
        hovertemplate='k=%{x}<br>Silhouette=%{y:.4f}<extra></extra>'
    ), row=1, col=2
)

# Annotate optimal k on both panels
for col in [1, 2]:
    fig_elbow_silhouette.add_vline(
        x=cluster_optimal_k, line_dash='dash',
        line_color='#ffd400', line_width=1.5,
        row=1, col=col
    )

fig_elbow_silhouette.add_annotation(
    x=cluster_optimal_k, y=max(cluster_inertias) * 0.98,
    text=f'<b>k={cluster_optimal_k} (optimal)</b>',
    showarrow=True, arrowhead=2, arrowcolor='#ffd400',
    font=dict(color='#ffd400', size=11),
    xref='x', yref='y', ax=40, ay=-30
)

fig_elbow_silhouette.update_layout(
    title=dict(
        text='Chart 17 – Optimal Cluster Selection: Elbow & Silhouette',
        font=dict(size=16, color='#fbfbff'), x=0.5, xanchor='center'
    ),
    plot_bgcolor='#1D1D20', paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff'),
    showlegend=False,
    height=430, margin=dict(t=80, b=60, l=60, r=40),
)
fig_elbow_silhouette.update_xaxes(
    tickvals=_k_vals, tickfont=dict(color='#fbfbff'),
    title_font=dict(color='#909094'), title_text='Number of Clusters (k)',
    gridcolor='#2d2d30'
)
fig_elbow_silhouette.update_yaxes(
    tickfont=dict(color='#fbfbff'), title_font=dict(color='#909094'),
    gridcolor='#2d2d30'
)

print(f"Chart 17 rendered. Optimal k={cluster_optimal_k}, silhouette={max(cluster_sil_scores):.4f}")
