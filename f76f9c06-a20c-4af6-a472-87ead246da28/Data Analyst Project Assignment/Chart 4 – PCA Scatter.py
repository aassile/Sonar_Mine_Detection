
import plotly.graph_objects as go
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_feat_cols = [c for c in cleaned_df.columns if c.startswith("freq_")]

# Scale and reduce to 2D
_X = StandardScaler().fit_transform(cleaned_df[_feat_cols])
_pca = PCA(n_components=2, random_state=42)
_coords = _pca.fit_transform(_X)

_labels = cleaned_df["target"].map({1: "Mine", 0: "Rock"})
_var_explained = _pca.explained_variance_ratio_ * 100

fig_pca = go.Figure()

for cls, color in [("Mine", "#FF9F9B"), ("Rock", "#A1C9F4")]:
    _mask = _labels == cls
    fig_pca.add_trace(go.Scatter(
        x=_coords[_mask, 0],
        y=_coords[_mask, 1],
        mode="markers",
        name=cls,
        marker=dict(color=color, size=8, opacity=0.8, line=dict(width=0.5, color="#1D1D20")),
    ))

fig_pca.update_layout(
    title=dict(
        text=f"PCA Projection: Mines vs. Rocks (2D)<br><sup>PC1 {_var_explained[0]:.1f}% · PC2 {_var_explained[1]:.1f}% of variance explained</sup>",
        font=dict(size=17, color="#fbfbff"),
    ),
    xaxis=dict(title=f"PC1 ({_var_explained[0]:.1f}%)", gridcolor="#333", color="#909094"),
    yaxis=dict(title=f"PC2 ({_var_explained[1]:.1f}%)", gridcolor="#333", color="#909094"),
    plot_bgcolor="#1D1D20",
    paper_bgcolor="#1D1D20",
    font=dict(color="#fbfbff"),
    legend=dict(bgcolor="#1D1D20", bordercolor="#555", borderwidth=1),
    height=440,
)
fig_pca.show()
