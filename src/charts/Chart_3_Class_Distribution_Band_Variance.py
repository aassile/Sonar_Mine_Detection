
import plotly.graph_objects as go
import pandas as pd

_feat_cols = [c for c in cleaned_df.columns if c.startswith("freq_")]
_band_nums = list(range(1, 61))
_var = cleaned_df[_feat_cols].var().values

fig_variance = go.Figure(go.Bar(
    x=_band_nums,
    y=_var,
    marker=dict(
        color=_var,
        colorscale=[[0, "#1F77B4"], [0.5, "#D0BBFF"], [1, "#ffd400"]],
        showscale=True,
        colorbar=dict(title="Variance", tickfont=dict(color="#909094"), titlefont=dict(color="#909094")),
    ),
))

fig_variance.update_layout(
    title=dict(
        text="Feature Variance Across All 60 Frequency Bands",
        font=dict(size=17, color="#fbfbff"),
    ),
    xaxis=dict(
        title="Frequency Band",
        tickvals=list(range(5, 61, 5)),
        gridcolor="#333",
        color="#909094",
    ),
    yaxis=dict(
        title="Variance",
        gridcolor="#333",
        color="#909094",
    ),
    plot_bgcolor="#1D1D20",
    paper_bgcolor="#1D1D20",
    font=dict(color="#fbfbff"),
    height=400,
)
fig_variance.show()
