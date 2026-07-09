import plotly.graph_objects as go
import numpy as np

_palette = ['#FFB482', '#8DE5A1', '#D0BBFF', '#FF9F9B', '#A1C9F4']
_bands   = list(range(1, 61))
_band_labels = [f'f{b}' for b in _bands]

fig_cluster_profiles = go.Figure()

for clust_id in cluster_means_df.index:
    _means = cluster_means_df.loc[clust_id].values
    _comp  = cluster_comp.loc[clust_id]
    _label = (
        f'Cluster {clust_id}  '
        f'(n={int(_comp["Total"])} | '
        f'{_comp["%Mine"]:.0f}% Mine / '
        f'{_comp["%Rock"]:.0f}% Rock)'
    )
    fig_cluster_profiles.add_trace(go.Scatter(
        x=_band_labels, y=_means,
        mode='lines',
        name=_label,
        line=dict(color=_palette[clust_id % len(_palette)], width=2.5),
        hovertemplate='Band=%{x}<br>Mean Energy=%{y:.4f}<extra>' + _label + '</extra>'
    ))

# Reference: overall grand mean
_grand_mean = cluster_means_df.mean(axis=0).values
fig_cluster_profiles.add_trace(go.Scatter(
    x=_band_labels, y=_grand_mean,
    mode='lines',
    name='Overall Mean',
    line=dict(color='#909094', width=1.5, dash='dot'),
    opacity=0.6,
    hovertemplate='Band=%{x}<br>Grand Mean=%{y:.4f}<extra></extra>'
))

# Mark top 5 permutation-important bands
for _bi in [37, 45, 27, 12, 11]:
    fig_cluster_profiles.add_vline(
        x=f'f{_bi}', line_dash='dash',
        line_color='#ffd400', line_width=1, opacity=0.5
    )

fig_cluster_profiles.add_annotation(
    x='f37', y=max(_grand_mean) * 1.08,
    text='Top importance bands', showarrow=False,
    font=dict(color='#ffd400', size=10), xanchor='center'
)

fig_cluster_profiles.update_layout(
    title=dict(
        text='Chart 19 – Cluster Mean Energy Profiles Across 60 Frequency Bands',
        font=dict(size=16, color='#fbfbff'), x=0.5, xanchor='center'
    ),
    xaxis=dict(
        title='Frequency Band',
        tickfont=dict(color='#fbfbff', size=9),
        title_font=dict(color='#909094'),
        tickmode='array',
        tickvals=[f'f{b}' for b in range(1, 61, 5)],
        ticktext=[f'f{b}' for b in range(1, 61, 5)],
        gridcolor='#2d2d30'
    ),
    yaxis=dict(
        title='Mean Energy Reading',
        tickfont=dict(color='#fbfbff'), title_font=dict(color='#909094'),
        gridcolor='#2d2d30'
    ),
    plot_bgcolor='#1D1D20', paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff'),
    legend=dict(bgcolor='rgba(29,29,32,0.8)', font=dict(color='#fbfbff', size=11),
                bordercolor='#2d2d30', borderwidth=1),
    height=470, margin=dict(t=80, b=60, l=70, r=40),
)

print("Chart 19 rendered — cluster mean energy profiles across all 60 bands.")
