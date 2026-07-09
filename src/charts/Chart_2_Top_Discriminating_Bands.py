
import plotly.graph_objects as go
import pandas as pd

# Sort top bands by their delta value (positive = Mine > Rock, negative = Rock > Mine)
_top = eda_top_bands.sort_values("Δ (Mine-Rock)", ascending=True)
_colors = ["#FF9F9B" if v > 0 else "#A1C9F4" for v in _top["Δ (Mine-Rock)"]]

fig_top_bands = go.Figure(go.Bar(
    x=_top["Δ (Mine-Rock)"].values,
    y=_top.index.tolist(),
    orientation="h",
    marker_color=_colors,
    text=[f"{v:+.4f}" for v in _top["Δ (Mine-Rock)"].values],
    textposition="outside",
    textfont=dict(color="#fbfbff", size=11),
))

fig_top_bands.update_layout(
    title=dict(
        text="Top 10 Frequency Bands by Mine vs. Rock Separation",
        font=dict(size=17, color="#fbfbff"),
    ),
    xaxis=dict(
        title="Mean Energy Difference (Mine − Rock)",
        gridcolor="#333",
        color="#909094",
        zeroline=True,
        zerolinecolor="#666",
    ),
    yaxis=dict(
        title="Frequency Band",
        color="#909094",
        gridcolor="#333",
    ),
    plot_bgcolor="#1D1D20",
    paper_bgcolor="#1D1D20",
    font=dict(color="#fbfbff"),
    height=420,
    margin=dict(l=80, r=80),
)
fig_top_bands.show()
