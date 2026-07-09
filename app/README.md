# Sonar Mine Detector — Streamlit demo

An interactive front end over the tuned Extra Trees classifier from
`sonar_detection`. The app trains the model on first load (cached), then lets you:

- **Generate synthetic signals** — mine-like, rock-like, or ambiguous, with an
  adjustable noise level.
- **Hand-tune all 60 frequency bands** with sliders.
- Inspect the **prediction**, **confidence gauge**, **signal energy profile**
  against class means, and a **per-band contribution breakdown**.

No external services are required — the model, scaler, and summary statistics
are computed fresh from `data/sonar.csv`.

## Run locally

From the repository root:

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

## Deploy

The app is a standard single-file Streamlit app and deploys as-is on
[Streamlit Community Cloud](https://streamlit.io/cloud):

1. Point the app at `app/streamlit_app.py` on the `master` branch.
2. Set the dependencies file to `app/requirements.txt`.

> Model performance (test set): **92.9% accuracy · 0.982 ROC-AUC · 0.944 CV AUC**.

Dataset: UCI Machine Learning Repository — Connectionist Bench (Sonar, Mines vs.
Rocks), https://doi.org/10.24432/C5J619.
