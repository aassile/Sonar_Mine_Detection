
import plotly.graph_objects as go

_top15 = feat_importances.head(15).sort_values(ascending=True)
_colors = ["#ffd400" if i in ["freq_11", "freq_10", "freq_12"] else "#D0BBFF"
           for i in _top15.index]

fig_feat_importance = go.Figure(go.Bar(
    x=_top15.values,
    y=_top15.index.tolist(),
    orientation="h",
    marker_color=_colors,
    text=[f"{v:.4f}" for v in _top15.values],
    textposition="outside",
    textfont=dict(color="#fbfbff", size=10),
))

fig_feat_importance.update_layout(
    title=dict(
        text="Top 15 Feature Importances – Random Forest<br><sup>Bands 10–12 are the strongest predictors</sup>",
        font=dict(size=17, color="#fbfbff"),
    ),
    xaxis=dict(title="Mean Decrease in Impurity (importance)", gridcolor="#333", color="#909094"),
    yaxis=dict(title="Frequency Band", color="#909094", gridcolor="#333"),
    plot_bgcolor="#1D1D20",
    paper_bgcolor="#1D1D20",
    font=dict(color="#fbfbff"),
    height=480,
    margin=dict(r=80),
)
fig_feat_importance.show()
