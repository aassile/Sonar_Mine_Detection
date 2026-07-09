import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

# ── Rebuild tuned model from upstream data ──────────────────────────────────
_features = [c for c in cleaned_df.columns if c != 'target']
_X = cleaned_df[_features].values
_y = cleaned_df['target'].values

np.random.seed(42)
_idx = np.random.permutation(len(_X))
_split = int(0.8 * len(_X))
_train_idx, _test_idx = _idx[:_split], _idx[_split:]

_X_train, _X_test = _X[_train_idx], _X[_test_idx]
_y_train, _y_test = _y[_train_idx], _y[_test_idx]

_scaler = StandardScaler()
_X_train_sc = _scaler.fit_transform(_X_train)

_et = ExtraTreesClassifier(
    n_estimators=200, max_features=0.4,
    max_depth=None, min_samples_split=5,
    min_samples_leaf=2, random_state=42
)
_et.fit(_X_train_sc, _y_train)

# ── Compute training statistics for synthetic sample generation ─────────────
_mine_df   = cleaned_df[cleaned_df['target'] == 1][_features]
_rock_df   = cleaned_df[cleaned_df['target'] == 0][_features]
_mine_mean = _mine_df.mean().values
_rock_mean = _rock_df.mean().values
_mine_std  = _mine_df.std().values
_rock_std  = _rock_df.std().values
_global_min = cleaned_df[_features].min().values
_global_max = cleaned_df[_features].max().values

def _make_sample(profile, noise_scale=0.8, seed=0):
    """Create a synthetic sample near a given mean profile with added noise."""
    rng = np.random.RandomState(seed)
    mean, std = profile
    raw = mean + rng.randn(60) * std * noise_scale
    return np.clip(raw, _global_min, _global_max)

def _predict(sample):
    sc = _scaler.transform(sample.reshape(1, -1))
    prob = _et.predict_proba(sc)[0]
    label = 'Mine' if prob[1] >= 0.5 else 'Rock'
    return label, prob[1]

# ── Generate 3 synthetic samples ────────────────────────────────────────────
np.random.seed(7)
samples = {
    'Synth-A\n(Mine-like)':  _make_sample((_mine_mean, _mine_std), noise_scale=0.7, seed=11),
    'Synth-B\n(Rock-like)':  _make_sample((_rock_mean, _rock_std), noise_scale=0.7, seed=22),
    'Synth-C\n(Ambiguous)':  _make_sample(
        ((_mine_mean + _rock_mean) / 2, (_mine_std + _rock_std) / 2),
        noise_scale=1.2, seed=33
    ),
}

results = {}
for name, s in samples.items():
    label, p_mine = _predict(s)
    results[name] = {'sample': s, 'label': label, 'p_mine': p_mine, 'p_rock': 1 - p_mine}

print("=== Synthetic Sonar Sample Predictions ===\n")
for name, r in results.items():
    conf = max(r['p_mine'], r['p_rock'])
    clean_name = name.replace('\n', ' ')
    print(f"{clean_name:25s} → {r['label']:4s}  |  P(Mine)={r['p_mine']:.3f}  P(Rock)={r['p_rock']:.3f}  Confidence={conf:.1%}")

# ── Chart 15-A: Prediction gauge bars ───────────────────────────────────────
_names      = list(results.keys())
_p_mine_vals = [results[n]['p_mine']  for n in _names]
_p_rock_vals = [results[n]['p_rock']  for n in _names]
_labels      = [results[n]['label']   for n in _names]
_bar_colors  = ['#FF9F9B' if l == 'Mine' else '#A1C9F4' for l in _labels]

fig_synth_predictions = go.Figure()

fig_synth_predictions.add_trace(go.Bar(
    name='P(Mine)', x=_names, y=_p_mine_vals,
    marker_color='#FF9F9B', text=[f"{v:.3f}" for v in _p_mine_vals],
    textposition='inside', textfont=dict(color='#1D1D20', size=13, family='monospace'),
))
fig_synth_predictions.add_trace(go.Bar(
    name='P(Rock)', x=_names, y=_p_rock_vals,
    marker_color='#A1C9F4', text=[f"{v:.3f}" for v in _p_rock_vals],
    textposition='inside', textfont=dict(color='#1D1D20', size=13, family='monospace'),
))

for i, (n, r) in enumerate(results.items()):
    fig_synth_predictions.add_annotation(
        x=n, y=1.06,
        text=f"<b>→ {r['label']}</b>",
        showarrow=False, font=dict(size=13, color='#ffd400'),
        xanchor='center'
    )

fig_synth_predictions.add_hline(
    y=0.5, line_dash='dash', line_color='#909094', line_width=1.5,
    annotation_text='Decision boundary (0.50)', annotation_font_color='#909094',
    annotation_position='top right'
)

fig_synth_predictions.update_layout(
    title=dict(text='Chart 15 – Tuned Extra Trees: Synthetic Sonar Sample Predictions',
               font=dict(size=16, color='#fbfbff'), x=0.5, xanchor='center'),
    barmode='stack',
    xaxis=dict(title='Synthetic Sample', tickfont=dict(color='#fbfbff', size=12), title_font=dict(color='#909094')),
    yaxis=dict(title='Predicted Probability', range=[0, 1.15],
               tickfont=dict(color='#fbfbff'), title_font=dict(color='#909094'),
               tickvals=[0, 0.25, 0.5, 0.75, 1.0],
               ticktext=['0.00', '0.25', '0.50', '0.75', '1.00']),
    plot_bgcolor='#1D1D20', paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff'),
    legend=dict(bgcolor='#1D1D20', font=dict(color='#fbfbff')),
    height=480, margin=dict(t=80, b=60, l=60, r=40),
)
plt_note = "Chart 15 rendered."

# ── Chart 16 – Energy profiles of synthetic samples vs real class means ──────
_bands = list(range(1, 61))
_band_labels = [f"f{b}" for b in _bands]

fig_synth_profiles = go.Figure()

fig_synth_profiles.add_trace(go.Scatter(
    x=_band_labels, y=_mine_mean, name='Real Mine Mean',
    line=dict(color='#FF9F9B', width=2, dash='dot'), opacity=0.7,
))
fig_synth_profiles.add_trace(go.Scatter(
    x=_band_labels, y=_rock_mean, name='Real Rock Mean',
    line=dict(color='#A1C9F4', width=2, dash='dot'), opacity=0.7,
))

_synth_colors = ['#FFB482', '#8DE5A1', '#D0BBFF']
for (sname, r), col in zip(results.items(), _synth_colors):
    clean = sname.replace('\n', ' ')
    fig_synth_profiles.add_trace(go.Scatter(
        x=_band_labels, y=r['sample'],
        name=f"{clean} → {r['label']} ({r['p_mine']:.2f})",
        line=dict(color=col, width=1.5), opacity=0.9,
    ))

# Highlight top-5 permutation-important bands
_top_bands_idx = [37, 45, 27, 12, 11]
for _bi in _top_bands_idx:
    fig_synth_profiles.add_vline(
        x=f"f{_bi}", line_dash='dash', line_color='#ffd400', line_width=1, opacity=0.4,
    )

fig_synth_profiles.add_annotation(
    x='f37', y=max(_mine_mean) * 1.05,
    text='Top importance bands', showarrow=False,
    font=dict(color='#ffd400', size=10), xanchor='center'
)

fig_synth_profiles.update_layout(
    title=dict(text='Chart 16 – Synthetic Sample Energy Profiles vs. Real Class Means',
               font=dict(size=16, color='#fbfbff'), x=0.5, xanchor='center'),
    xaxis=dict(title='Frequency Band', tickfont=dict(color='#fbfbff', size=9),
               title_font=dict(color='#909094'),
               tickmode='array',
               tickvals=[f"f{b}" for b in range(1, 61, 5)],
               ticktext=[f"f{b}" for b in range(1, 61, 5)]),
    yaxis=dict(title='Energy Reading', tickfont=dict(color='#fbfbff'), title_font=dict(color='#909094')),
    plot_bgcolor='#1D1D20', paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff'),
    legend=dict(bgcolor='#1D1D20', font=dict(color='#fbfbff', size=10)),
    height=480, margin=dict(t=80, b=60, l=60, r=40),
)
print("\nChart 15 & 16 rendered.")
