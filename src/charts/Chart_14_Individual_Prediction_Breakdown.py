import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── rebuild tuned ET model & test set (same seed) ───────────────────────────
_feature_cols = [c for c in cleaned_df.columns if c != 'target']
_X = cleaned_df[_feature_cols].values
_y = cleaned_df['target'].values

_X_train, _X_test, _y_train, _y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=42, stratify=_y
)
_scaler = StandardScaler()
_X_train_sc = _scaler.fit_transform(_X_train)
_X_test_sc  = _scaler.transform(_X_test)

_et = ExtraTreesClassifier(
    n_estimators=200, max_features=0.4, max_depth=None,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
_et.fit(_X_train_sc, _y_train)

# ── pick one correctly-predicted Mine and one correctly-predicted Rock ────────
_preds  = _et.predict(_X_test_sc)
_probas = _et.predict_proba(_X_test_sc)[:, 1]  # P(Mine)

_mine_idxs = np.where((_y_test == 1) & (_preds == 1))[0]
_rock_idxs = np.where((_y_test == 0) & (_preds == 0))[0]
_mine_idx  = _mine_idxs[np.argmax(_probas[_mine_idxs])]  # most confident Mine
_rock_idx  = _rock_idxs[np.argmin(_probas[_rock_idxs])]  # most confident Rock

# ── marginal contributions via mean substitution ─────────────────────────────
# For each feature: contribution = p(sample) - p(sample with feature set to mean)
_feat_names  = [f'freq_{i+1}' for i in range(_X_test_sc.shape[1])]
_train_means = _X_train_sc.mean(axis=0)

def _contributions(sample_sc, n_top=12):
    _base_p = _et.predict_proba(sample_sc.reshape(1, -1))[0, 1]
    _contribs = np.zeros(len(_feat_names))
    for _f in range(len(_feat_names)):
        _x_mod = sample_sc.copy()
        _x_mod[_f] = _train_means[_f]
        _contribs[_f] = _base_p - _et.predict_proba(_x_mod.reshape(1, -1))[0, 1]
    # keep top n_top by absolute value
    _top_idx   = np.argsort(np.abs(_contribs))[-n_top:][::-1]
    _top_idx   = _top_idx[np.argsort(np.abs(_contribs[_top_idx]))[::-1]]
    return _base_p, _contribs, _top_idx

_mine_base_p, _mine_contribs, _mine_top = _contributions(_X_test_sc[_mine_idx])
_rock_base_p, _rock_contribs, _rock_top = _contributions(_X_test_sc[_rock_idx])

print(f"Sample A — True: Mine | Predicted: Mine | P(Mine) = {_mine_base_p:.3f}")
print(f"Sample B — True: Rock | Predicted: Rock | P(Mine) = {_rock_base_p:.3f}\n")
print("Top contributing bands for Sample A (Mine):")
for _i in _mine_top:
    _sign = '+' if _mine_contribs[_i] > 0 else ''
    print(f"  {_feat_names[_i]:10s}  {_sign}{_mine_contribs[_i]:.4f}")
print("\nTop contributing bands for Sample B (Rock):")
for _i in _rock_top:
    _sign = '+' if _rock_contribs[_i] > 0 else ''
    print(f"  {_feat_names[_i]:10s}  {_sign}{_rock_contribs[_i]:.4f}")

# ── waterfall-style chart ─────────────────────────────────────────────────────
def _make_waterfall(contribs, top_idx, feat_names, base_p, title, start_val=0.5):
    _vals  = [contribs[i] for i in top_idx]
    _names = [feat_names[i] for i in top_idx]
    _colors = ['#8DE5A1' if v > 0 else '#FF9F9B' for v in _vals]
    return go.Bar(
        x=_vals,
        y=_names,
        orientation='h',
        marker_color=_colors,
        name=title,
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:+.4f}<extra></extra>'
    ), _vals, _names

_mine_bar, _mv, _mn = _make_waterfall(_mine_contribs, _mine_top, _feat_names, _mine_base_p, 'Mine')
_rock_bar, _rv, _rn = _make_waterfall(_rock_contribs, _rock_top, _feat_names, _rock_base_p, 'Rock')

fig_prediction_breakdown = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        f'Sample A — Mine (P={"Mine"} {_mine_base_p:.2f})',
        f'Sample B — Rock (P={"Mine"} {_rock_base_p:.2f})'
    ],
    horizontal_spacing=0.18
)

fig_prediction_breakdown.add_trace(_mine_bar, row=1, col=1)
fig_prediction_breakdown.add_trace(_rock_bar, row=1, col=2)

# zero-lines
for _col in [1, 2]:
    fig_prediction_breakdown.add_vline(
        x=0, line_color='#909094', line_width=1, col=_col, row=1
    )

fig_prediction_breakdown.update_layout(
    title=dict(
        text='Chart 14 – Individual Prediction Breakdown<br>'
             '<sup>Top 12 bands driving each prediction — green pushes toward Mine, red toward Rock</sup>',
        font=dict(color='#fbfbff', size=16)
    ),
    plot_bgcolor='#1D1D20', paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff'),
    height=520, margin=dict(l=110, r=60, t=110, b=80),
    showlegend=False
)
fig_prediction_breakdown.update_xaxes(
    title_text='Contribution to P(Mine)', title_font=dict(color='#909094', size=11),
    tickfont=dict(color='#909094'), gridcolor='#2a2a2f', zeroline=False
)
fig_prediction_breakdown.update_yaxes(
    tickfont=dict(color='#909094'), autorange='reversed'
)
# subtitle annotations
for _col_n, (_label, _p) in enumerate([('Mine', _mine_base_p), ('Rock', _rock_base_p)], 1):
    fig_prediction_breakdown.layout.annotations[_col_n - 1].font = dict(color='#fbfbff', size=13)

fig_prediction_breakdown.add_annotation(
    text="<i>Method: marginal contribution = P(Mine | sample) − P(Mine | feature replaced by training mean)</i>",
    xref='paper', yref='paper', x=0.5, y=-0.13, showarrow=False,
    font=dict(size=11, color='#909094'), xanchor='center'
)
