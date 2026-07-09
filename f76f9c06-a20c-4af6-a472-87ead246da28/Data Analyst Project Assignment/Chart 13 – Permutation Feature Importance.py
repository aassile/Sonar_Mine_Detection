import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score

# ── reproduce the tuned ET model & data (same seed as tuning block) ──────────
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# ── permutation importance ────────────────────────────────────────────────────
_rng        = np.random.default_rng(42)
_n_repeats  = 30
_base_auc   = roc_auc_score(_y_test, _et.predict_proba(_X_test_sc)[:, 1])

_importances = np.zeros((_X_test_sc.shape[1], _n_repeats))
for _feat in range(_X_test_sc.shape[1]):
    for _r in range(_n_repeats):
        _X_perm = _X_test_sc.copy()
        _X_perm[:, _feat] = _rng.permutation(_X_perm[:, _feat])
        _importances[_feat, _r] = _base_auc - roc_auc_score(
            _y_test, _et.predict_proba(_X_perm)[:, 1]
        )

_perm_means = _importances.mean(axis=1)
_perm_stds  = _importances.std(axis=1)
_feat_names = [f'freq_{i+1}' for i in range(_X_test_sc.shape[1])]

_perm_df = pd.DataFrame({
    'feature': _feat_names,
    'importance': _perm_means,
    'std': _perm_stds
}).sort_values('importance', ascending=False).head(15)

print(f"Baseline AUC: {_base_auc:.4f}")
print("\nTop 15 Permutation-Important Bands (AUC drop when shuffled):")
print(_perm_df[['feature', 'importance', 'std']].to_string(index=False))

# ── chart ─────────────────────────────────────────────────────────────────────
_colors = ['#ffd400' if v == _perm_df['importance'].max() else '#A1C9F4'
           for v in _perm_df['importance']]

fig_perm_importance = go.Figure()
fig_perm_importance.add_trace(go.Bar(
    x=_perm_df['importance'],
    y=_perm_df['feature'],
    orientation='h',
    marker_color=_colors,
    error_x=dict(type='data', array=_perm_df['std'].tolist(), visible=True,
                 color='#909094', thickness=1.5, width=4),
    hovertemplate='<b>%{y}</b><br>AUC Drop: %{x:.4f}<extra></extra>'
))

fig_perm_importance.update_layout(
    title=dict(
        text='Chart 13 – Permutation Feature Importance<br>'
             '<sup>AUC drop when each band is randomly shuffled (30 repeats) — higher = more important</sup>',
        font=dict(color='#fbfbff', size=16)
    ),
    xaxis=dict(
        title='Mean AUC Drop (± std)', title_font=dict(color='#909094'),
        tickfont=dict(color='#909094'), gridcolor='#2a2a2f', zeroline=True,
        zerolinecolor='#909094', zerolinewidth=1
    ),
    yaxis=dict(
        title='Frequency Band', title_font=dict(color='#909094'),
        tickfont=dict(color='#909094'), autorange='reversed'
    ),
    plot_bgcolor='#1D1D20', paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff'),
    height=520, margin=dict(l=100, r=60, t=100, b=60),
    showlegend=False
)
plt_note = (
    f"<i style='color:#909094;font-size:11px'>"
    f"Baseline AUC: {_base_auc:.4f} | Yellow bar = most critical band | "
    f"Error bars = std across 30 shuffles</i>"
)
fig_perm_importance.add_annotation(
    text=plt_note, xref='paper', yref='paper',
    x=0, y=-0.12, showarrow=False, align='left',
    font=dict(size=11, color='#909094')
)
import matplotlib; matplotlib.use('Agg')  # suppress any implicit display
