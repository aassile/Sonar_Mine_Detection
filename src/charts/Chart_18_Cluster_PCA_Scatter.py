import plotly.graph_objects as go
import numpy as np

# ── Colour by cluster, symbol by true label ──────────────────────────────────
_cluster_palette = ['#FFB482', '#8DE5A1', '#D0BBFF', '#FF9F9B', '#A1C9F4']
_symbol_map = {1: 'circle', 0: 'diamond'}   # Mine=circle, Rock=diamond
_label_name  = {1: 'Mine', 0: 'Rock'}

_x = cluster_pca_X[:, 0]
_y = cluster_pca_X[:, 1]
_c = [int(l) for l in (cluster_labels if hasattr(cluster_labels, '__iter__') else [])]
_t = [int(l) for l in cluster_true_labels]

fig_cluster_pca = go.Figure()

for clust in sorted(set(_c)):
    _cc = _cluster_palette[clust % len(_cluster_palette)]
    for true_lab in [1, 0]:
        _mask = [i for i, (ci, ti) in enumerate(zip(_c, _t)) if ci == clust and ti == true_lab]
        if not _mask:
            continue
        fig_cluster_pca.add_trace(go.Scatter(
            x=[_x[i] for i in _mask],
            y=[_y[i] for i in _mask],
            mode='markers',
            marker=dict(
                color=_cc,
                symbol=_symbol_map[true_lab],
                size=8,
                opacity=0.85,
                line=dict(color='#1D1D20', width=0.5)
            ),
            name=f'Cluster {clust} / {_label_name[true_lab]}',
            hovertemplate=(
                f'Cluster {clust} | True: {_label_name[true_lab]}<br>'
                'PC1=%{x:.2f}  PC2=%{y:.2f}<extra></extra>'
            )
        ))

# Cluster centroids in PCA space (project KMeans centroids)
from sklearn.decomposition import PCA as _PCA
_pca_obj = _PCA(n_components=2, random_state=42)
_pca_obj.fit(X_scaled)  # refit on same scaled data for centroid projection

from sklearn.cluster import KMeans as _KMeans
_km2 = _KMeans(n_clusters=cluster_optimal_k, random_state=42, n_init=20)
_km2.fit(X_scaled)
_centroids_pca = _pca_obj.transform(_km2.cluster_centers_)

for ci, cp in enumerate(_centroids_pca):
    fig_cluster_pca.add_trace(go.Scatter(
        x=[cp[0]], y=[cp[1]],
        mode='markers',
        marker=dict(symbol='x', size=16, color='#ffd400',
                    line=dict(color='#fbfbff', width=2)),
        name=f'Centroid {ci}',
        hovertemplate=f'Centroid {ci}<br>PC1={cp[0]:.2f}  PC2={cp[1]:.2f}<extra></extra>'
    ))

_pv = cluster_pca_var
fig_cluster_pca.update_layout(
    title=dict(
        text='Chart 18 – KMeans Clusters in PCA Space (shape = true label)',
        font=dict(size=16, color='#fbfbff'), x=0.5, xanchor='center'
    ),
    xaxis=dict(
        title=f'PC1 ({_pv[0]:.1%} variance)',
        tickfont=dict(color='#fbfbff'), title_font=dict(color='#909094'),
        gridcolor='#2d2d30', zeroline=False
    ),
    yaxis=dict(
        title=f'PC2 ({_pv[1]:.1%} variance)',
        tickfont=dict(color='#fbfbff'), title_font=dict(color='#909094'),
        gridcolor='#2d2d30', zeroline=False
    ),
    plot_bgcolor='#1D1D20', paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff'),
    legend=dict(bgcolor='rgba(29,29,32,0.8)', font=dict(color='#fbfbff', size=10),
                bordercolor='#2d2d30', borderwidth=1),
    height=500, margin=dict(t=80, b=60, l=70, r=40),
)

print(f"Chart 18 rendered. k={cluster_optimal_k} — circles=Mine, diamonds=Rock, ✕=centroid.")
