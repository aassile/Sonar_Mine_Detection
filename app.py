"""Live Streamlit demo for the Sonar Mine Detector.

Interactive front end over the tuned Extra Trees classifier trained by
``sonar_detection.run_pipeline``. Generate synthetic mine/rock-like signals or
hand-tune all 60 frequency bands, then inspect the prediction, confidence gauge,
signal energy profile, and per-band contribution breakdown.

Run locally with:

    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Make the src/ package importable when running without an editable install
# (e.g. on Streamlit Community Cloud, which only installs requirements.txt).
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sonar_detection import FEATURE_COLS, load_sonar  # noqa: E402
from sonar_detection.evaluation import permutation_importance_auc  # noqa: E402
from sonar_detection.models import build_extra_trees  # noqa: E402
from sonar_detection.preprocessing import (  # noqa: E402
    encode_target,
    scale_features,
    split_features_target,
    train_test_split_sonar,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sonar Mine Detector",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main { background-color: #1D1D20; }
    .stApp { background-color: #1D1D20; color: #fbfbff; }
    .metric-box {
        background: #2a2a2f; border-radius: 12px; padding: 20px;
        text-align: center; margin: 8px 0;
    }
    .metric-label { color: #909094; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: #fbfbff; font-size: 28px; font-weight: bold; }
    .mine-color { color: #FF9F9B; }
    .rock-color { color: #A1C9F4; }
    .header-box {
        background: linear-gradient(135deg, #2a2a2f 0%, #1D1D20 100%);
        border: 1px solid #3a3a3f; border-radius: 12px;
        padding: 24px; margin-bottom: 24px;
    }
    div[data-testid="stSidebar"] { background-color: #2a2a2f; }
    .stSlider > div > div > div { background-color: #A1C9F4; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Locate the dataset ────────────────────────────────────────────────────────
def _find_data() -> str:
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents):
        candidate = base / "data" / "sonar.csv"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Could not locate data/sonar.csv relative to the app.")


# ── Train model + derive app artifacts (cached) ───────────────────────────────
@st.cache_resource
def load_model():
    """Train the tuned Extra Trees pipeline and derive display artifacts.

    Replaces the original Zerve ``variable()`` lookups: the model, scaler, and
    all summary statistics are computed fresh from ``data/sonar.csv`` so the app
    is fully self-contained.
    """
    df = encode_target(load_sonar(_find_data()))
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split_sonar(x, y)
    x_train_sc, _x_test_sc, scaler = scale_features(x_train, x_test)

    model = build_extra_trees()
    model.fit(x_train_sc, y_train)

    feat_names = list(FEATURE_COLS)
    # Class means / global stats computed on the *unscaled* training features
    # (0–1 reflected-energy space), matching the original app semantics.
    mine_means = x_train[y_train == 1].mean(axis=0)
    rock_means = x_train[y_train == 0].mean(axis=0)
    train_mean = x_train.mean(axis=0)
    train_std = x_train.std(axis=0)

    # Top signal bands by permutation importance on the test set.
    imp = permutation_importance_auc(model, _x_test_sc, y_test, feat_names)
    top_bands = list(imp.sort_values("importance", ascending=False).head(5)["feature"])

    return model, scaler, feat_names, mine_means, rock_means, train_mean, train_std, top_bands


model, scaler, feat_names, mine_means, rock_means, train_mean, train_std, top_bands = load_model()
n_features = len(feat_names)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="header-box">
    <h1 style="color:#fbfbff; margin:0; font-size:32px;">🌊 Sonar Mine Detector</h1>
    <p style="color:#909094; margin:8px 0 0 0; font-size:15px;">
        Live sonar signal classification · Tuned Extra Trees · 92.9% accuracy · AUC 0.982<br>
        <span style="font-size:12px;">Dataset: <i>UCI Machine Learning Repository — Connectionist Bench (Sonar, Mines vs. Rocks).
        <a href="https://doi.org/10.24432/C5J619" style="color:#A1C9F4;">https://doi.org/10.24432/C5J619</a></i></span>
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ── Sidebar – Input Mode ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Signal Input")
    input_mode = st.radio(
        "Input mode",
        ["🎲 Generate Sample", "🎛️ Manual Sliders"],
        index=0,
    )

    st.markdown("---")

    if input_mode == "🎲 Generate Sample":
        sample_type = st.selectbox(
            "Sample type",
            ["Mine-like 💣", "Rock-like 🪨", "Ambiguous ❓"],
            index=0,
        )
        noise_level = st.slider(
            "Noise level",
            0.0,
            2.0,
            0.8,
            0.1,
            help="Higher = more deviation from class mean",
        )
        if st.button("🎲 Generate New Sample", use_container_width=True):
            st.session_state["regenerate"] = st.session_state.get("regenerate", 0) + 1

        st.markdown("---")
        st.markdown("### 🏆 Model Performance")
        st.markdown(
            """
        | Metric | Value |
        |--------|-------|
        | Accuracy | **92.9%** |
        | AUC | **0.982** |
        | CV AUC | **0.944** |
        | Precision | **0.930** |
        | Recall | **0.927** |
        """
        )
        st.markdown("---")
        st.markdown("### 🔑 Top Signal Bands")
        for b in top_bands:
            st.markdown(f"- `{b}`")
    else:
        st.markdown("### 🎛️ Adjust 60 Frequency Bands")
        st.caption("Each band = reflected energy (0.0–1.0)")


# ── Build input signal ────────────────────────────────────────────────────────
rng_seed = st.session_state.get("regenerate", 0)
rng = np.random.RandomState(rng_seed + 42)


if input_mode == "🎲 Generate Sample":
    if "Mine" in sample_type:
        base = mine_means
        label = "Mine-like"
    elif "Rock" in sample_type:
        base = rock_means
        label = "Rock-like"
    else:
        base = (mine_means + rock_means) / 2
        label = "Ambiguous"
    signal = np.clip(base + rng.randn(n_features) * train_std * noise_level, 0.0, 1.0)
    signal_label = label

else:
    # Manual sliders: two columns of 30 bands each for readability
    signal = np.zeros(n_features)
    signal_label = "Custom"
    bands_per_col = 30
    col_left, col_right = st.columns(2)
    for i in range(n_features):
        target_col = col_left if i < bands_per_col else col_right
        with target_col:
            signal[i] = st.slider(
                feat_names[i],
                min_value=0.0,
                max_value=1.0,
                value=float(train_mean[i]),
                step=0.01,
                key=f"band_{i}",
            )


# ── Prediction ────────────────────────────────────────────────────────────────
signal_sc = scaler.transform(signal.reshape(1, -1))
proba = model.predict_proba(signal_sc)[0]
p_mine = proba[1]
p_rock = proba[0]
prediction = "💣 MINE" if p_mine >= 0.5 else "🪨 ROCK"
pred_color = "#FF9F9B" if p_mine >= 0.5 else "#A1C9F4"
confidence = max(p_mine, p_rock) * 100


risk_level = "🔴 HIGH" if p_mine > 0.75 else ("🟡 MODERATE" if p_mine > 0.4 else "🟢 LOW")


# ── Main layout ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)


with col1:
    st.markdown(
        f"""
    <div class="metric-box">
        <div class="metric-label">PREDICTION</div>
        <div class="metric-value" style="color:{pred_color}; font-size:22px;">{prediction}</div>
    </div>""",
        unsafe_allow_html=True,
    )


with col2:
    st.markdown(
        f"""
    <div class="metric-box">
        <div class="metric-label">P(Mine)</div>
        <div class="metric-value mine-color">{p_mine:.1%}</div>
    </div>""",
        unsafe_allow_html=True,
    )


with col3:
    st.markdown(
        f"""
    <div class="metric-box">
        <div class="metric-label">P(Rock)</div>
        <div class="metric-value rock-color">{p_rock:.1%}</div>
    </div>""",
        unsafe_allow_html=True,
    )


with col4:
    st.markdown(
        f"""
    <div class="metric-box">
        <div class="metric-label">MINE RISK</div>
        <div class="metric-value" style="font-size:20px;">{risk_level}</div>
    </div>""",
        unsafe_allow_html=True,
    )


st.markdown("<br>", unsafe_allow_html=True)


# ── Gauge + Energy Profile ────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns([1, 2])


with chart_col1:
    st.markdown("#### 🎯 Confidence Gauge")
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=p_mine * 100,
            delta={"reference": 50, "valueformat": ".1f"},
            number={"suffix": "%", "font": {"color": "#fbfbff", "size": 36}},
            title={"text": "P(Mine)", "font": {"color": "#909094", "size": 14}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#909094",
                    "tickfont": {"color": "#909094"},
                },
                "bar": {"color": pred_color, "thickness": 0.3},
                "bgcolor": "#2a2a2f",
                "bordercolor": "#3a3a3f",
                "steps": [
                    {"range": [0, 40], "color": "#1a3a2a"},
                    {"range": [40, 60], "color": "#3a3a1a"},
                    {"range": [60, 100], "color": "#3a1a1a"},
                ],
                "threshold": {
                    "line": {"color": "#ffd400", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig_gauge.update_layout(
        paper_bgcolor="#1D1D20",
        font={"color": "#fbfbff"},
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Decision annotation
    action = (
        "⚠️ Flag for human review"
        if 0.4 <= p_mine <= 0.75
        else ("🚨 Alert — dispatch EOD team" if p_mine > 0.75 else "✅ Clear — safe to proceed")
    )
    st.info(f"**Recommended action:** {action}")


with chart_col2:
    st.markdown("#### 📡 Signal Energy Profile vs. Class Means")
    band_nums = list(range(1, 61))
    top_band_idxs = [feat_names.index(b) for b in top_bands]

    fig_profile = go.Figure()

    fig_profile.add_trace(
        go.Scatter(
            x=band_nums,
            y=mine_means,
            mode="lines",
            name="Mine mean",
            line=dict(color="#FF9F9B", width=1.5, dash="dot"),
            opacity=0.7,
        )
    )
    fig_profile.add_trace(
        go.Scatter(
            x=band_nums,
            y=rock_means,
            mode="lines",
            name="Rock mean",
            line=dict(color="#A1C9F4", width=1.5, dash="dot"),
            opacity=0.7,
        )
    )
    fig_profile.add_trace(
        go.Scatter(
            x=band_nums,
            y=signal,
            mode="lines+markers",
            name=f"Input ({signal_label})",
            line=dict(color="#ffd400", width=2.5),
            marker=dict(size=3, color="#ffd400"),
        )
    )
    for idx in top_band_idxs:
        fig_profile.add_vline(
            x=idx + 1,
            line_color="#8DE5A1",
            line_width=1,
            line_dash="dash",
            opacity=0.5,
        )

    fig_profile.add_annotation(
        x=top_band_idxs[0] + 1,
        y=0.78,
        text="Top bands ↓",
        showarrow=False,
        font=dict(color="#8DE5A1", size=11),
    )

    fig_profile.update_layout(
        plot_bgcolor="#1D1D20",
        paper_bgcolor="#1D1D20",
        font=dict(color="#fbfbff"),
        xaxis=dict(
            title="Frequency Band",
            tickfont=dict(color="#909094"),
            gridcolor="#2a2a2f",
            title_font=dict(color="#909094"),
        ),
        yaxis=dict(
            title="Reflected Energy",
            tickfont=dict(color="#909094"),
            gridcolor="#2a2a2f",
            title_font=dict(color="#909094"),
            range=[0, 0.85],
        ),
        legend=dict(
            font=dict(color="#fbfbff"),
            bgcolor="#2a2a2f",
            bordercolor="#3a3a3f",
        ),
        height=280,
        margin=dict(l=60, r=20, t=20, b=50),
    )
    st.plotly_chart(fig_profile, use_container_width=True)


# ── Prediction Breakdown (top contributing bands) ─────────────────────────────
st.markdown("#### 🔍 Prediction Breakdown — Top 12 Band Contributions")
st.caption("How much each frequency band pushed the prediction toward Mine (+) or Rock (−)")


train_means_sc = scaler.transform(train_mean.reshape(1, -1))[0]
base_p = model.predict_proba(signal_sc)[0, 1]


contributions = np.zeros(n_features)
for f in range(n_features):
    x_mod = signal_sc[0].copy()
    x_mod[f] = train_means_sc[f]
    contributions[f] = base_p - model.predict_proba(x_mod.reshape(1, -1))[0, 1]


top12_idx = np.argsort(np.abs(contributions))[-12:][::-1]
top12_idx = top12_idx[np.argsort(np.abs(contributions[top12_idx]))[::-1]]
top12_vals = [contributions[i] for i in top12_idx]
top12_names = [feat_names[i] for i in top12_idx]
bar_colors = ["#FF9F9B" if v > 0 else "#A1C9F4" for v in top12_vals]


fig_breakdown = go.Figure(
    go.Bar(
        x=top12_vals,
        y=top12_names,
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:+.4f}<extra></extra>",
    )
)
fig_breakdown.add_vline(x=0, line_color="#909094", line_width=1.5)
fig_breakdown.update_layout(
    plot_bgcolor="#1D1D20",
    paper_bgcolor="#1D1D20",
    font=dict(color="#fbfbff"),
    xaxis=dict(
        title="Contribution to P(Mine)",
        tickfont=dict(color="#909094"),
        gridcolor="#2a2a2f",
        title_font=dict(color="#909094"),
    ),
    yaxis=dict(tickfont=dict(color="#909094"), autorange="reversed"),
    height=340,
    margin=dict(l=90, r=40, t=20, b=50),
    showlegend=False,
)
fig_breakdown.add_annotation(
    text="🔴 Red = pushes toward Mine   🔵 Blue = pushes toward Rock",
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.18,
    showarrow=False,
    font=dict(size=11, color="#909094"),
    xanchor="center",
)
st.plotly_chart(fig_breakdown, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
<div style="color:#909094; font-size:12px; text-align:center;">
    Tuned ExtraTreesClassifier · 92.9% Test Accuracy · AUC 0.982 · CV AUC 0.944<br>
    <b>Dataset:</b> UCI Machine Learning Repository — Connectionist Bench (Sonar, Mines vs. Rocks).
    <a href="https://doi.org/10.24432/C5J619" style="color:#A1C9F4;">https://doi.org/10.24432/C5J619</a>
</div>
""",
    unsafe_allow_html=True,
)
