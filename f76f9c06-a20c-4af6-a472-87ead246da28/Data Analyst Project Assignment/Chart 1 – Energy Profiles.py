
import plotly.graph_objects as go
import pandas as pd

_feat_cols = [c for c in cleaned_df.columns if c.startswith("freq_")]
_band_nums = list(range(1, 61))

fig_energy_profiles = go.Figure()

fig_energy_profiles.add_trace(go.Scatter(
    x=_band_nums, y=eda_mine_means.values,
    mode="lines+markers",
    name="Mine (M)",
    line=dict(color="#FF9F9B", width=2.5),
    marker=dict(size=4),
))

fig_energy_profiles.add_trace(go.Scatter(
    x=_band_nums, y=eda_rock_means.values,
    mode="lines+markers",
    name="Rock (R)",
    line=dict(color="#A1C9F4", width=2.5),
    marker=dict(size=4),
))

fig_energy_profiles.update_layout(
    title=dict(
        text="Mean Sonar Energy Profile: Mines vs. Rocks",
        font=dict(size=18, color="#fbfbff"),
    ),
    xaxis=dict(
        title="Frequency Band",
        tickvals=list(range(0, 61, 5))[1:],
        gridcolor="#333",
        color="#909094",
    ),
    yaxis=dict(
        title="Mean Energy (normalised)",
        gridcolor="#333",
        color="#909094",
    ),
    plot_bgcolor="#1D1D20",
    paper_bgcolor="#1D1D20",
    font=dict(color="#fbfbff"),
    legend=dict(bgcolor="#1D1D20", bordercolor="#555", borderwidth=1),
    height=420,
)
fig_energy_profiles.show()
