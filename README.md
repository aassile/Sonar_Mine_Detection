# Sonar Mine Detection — Mines vs. Rocks Classification

[![CI](https://github.com/aassile/Sonar_Mine_Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/aassile/Sonar_Mine_Detection/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Objective

Can a machine distinguish an underwater sea **mine** from a **rock** using only reflected
sound? A sonar device pings the seabed and records the energy reflected back across 60
frequency bands. To the eye, mine and rock returns look nearly identical. This project
builds an end-to-end classification pipeline that makes the distinction automatically:

1. **Compare** four candidate models (Random Forest, Extra Trees, Gradient Boosting, Neural Net).
2. **Tune** the winning Extra Trees classifier via randomized search on ROC-AUC.
3. **Interpret** the model with permutation feature importance.

All logic lives in an importable package (`src/sonar_detection/`); a single reproducible
notebook (`notebooks/sonar_analysis.ipynb`) tells the story end to end.

## Dataset

| Property | Value |
|---|---|
| Source | [UCI ML Repository #151](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) (https://doi.org/10.24432/C5J619) |
| Samples | 208 sonar returns |
| Features | 60 frequency-band energy readings (`freq_1` … `freq_60`), each in [0, 1] |
| Target | Binary — `M` (mine) = 111, `R` (rock) = 97 |

The dataset is tiny, so the labeled CSV is committed directly as `data/sonar.csv` — no
download required for the notebook or CI. See [`data/README.md`](data/README.md) for the
schema and regeneration steps.

## Project Structure

```
Sonar_Mine_Detection/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── README.md              # schema + provenance
│   └── sonar.csv              # 208-row labeled dataset (committed)
├── docs/
│   ├── project-report.md      # full case-study report (EDA → deployment)
│   └── why-this-matters.md    # real-world context & applications
├── notebooks/
│   └── sonar_analysis.ipynb   # reproducible end-to-end narrative
├── src/sonar_detection/
│   ├── __init__.py
│   ├── data.py                # loading (CSV + raw UCI file)
│   ├── preprocessing.py       # encode target, split, scale
│   ├── models.py              # model builders + hyperparameter tuning
│   ├── evaluation.py          # metrics, CV, confusion, ROC, importance
│   └── pipeline.py            # run_pipeline() — full workflow in one call
└── tests/
    ├── conftest.py            # synthetic-data fixtures
    ├── test_data.py
    ├── test_preprocessing.py
    ├── test_models.py
    ├── test_evaluation.py
    └── test_pipeline.py
```

## Methods

- **Supervised learning:** Random Forest, Extra Trees, Gradient Boosting, and a small
  Neural Network (`MLPClassifier`), all via scikit-learn.
- **Preprocessing:** target encoding (`M`→1, `R`→0), stratified train/test split, standard scaling.
- **Model selection:** stratified 5-fold cross-validation scored on ROC-AUC.
- **Hyperparameter tuning:** randomized search over Extra Trees parameters (`tune_extra_trees`).
- **Evaluation:** accuracy, ROC-AUC, confusion matrix, precision/recall/F1.
- **Interpretability:** permutation feature importance (mean AUC drop over 30 shuffles).
- **Visualization:** interactive Plotly charts in the notebook.

## Results

Model comparison, ranked by 5-fold cross-validated ROC-AUC (the reliable metric given the
42-sample test set):

| Model | Accuracy | ROC-AUC | CV AUC | F1 (Mine) |
|---|---|---|---|---|
| **Extra Trees** | **0.929** | **0.982** | **0.940** | **0.933** |
| Neural Net | 0.881 | 0.952 | 0.907 | 0.889 |
| Gradient Boosting | 0.857 | 0.925 | 0.914 | 0.864 |
| Random Forest | 0.833 | 0.924 | 0.908 | 0.844 |

The tuned Extra Trees model reaches **~0.944 cross-validated ROC-AUC**. The most
discriminative frequency bands are `freq_37`, `freq_45`, `freq_27`, `freq_12`, and `freq_11`.

> ⚠️ The held-out test set is only 42 samples, so cross-validation metrics are the more
> reliable estimate of generalization. In an operational setting this model would act as a
> **first-pass filter** on sonar returns — narrowing the set that needs expert review, not
> making standalone safety-critical decisions.

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/aassile/Sonar_Mine_Detection.git
cd Sonar_Mine_Detection
pip install -r requirements.txt
pip install -e .

# 2. Run the full pipeline in one call
python -c "from sonar_detection import run_pipeline; \
    r = run_pipeline('data/sonar.csv'); print(r['comparison'].round(4))"

# 3. Or open the reproducible notebook
jupyter lab notebooks/sonar_analysis.ipynb
```

Using the package directly:

```python
from sonar_detection import (
    load_sonar, encode_target, split_features_target,
    train_test_split_sonar, scale_features, build_extra_trees, evaluate_model,
)

df = encode_target(load_sonar("data/sonar.csv"))
X, y = split_features_target(df)
X_tr, X_te, y_tr, y_te = train_test_split_sonar(X, y)
X_tr, X_te, scaler = scale_features(X_tr, X_te)

model = build_extra_trees().fit(X_tr, y_tr)
print(evaluate_model(model, X_te, y_te))
```

## Testing

```bash
pytest tests/ -v
```

The suite covers data loading, preprocessing, model construction/tuning, evaluation
metrics, and the end-to-end pipeline using synthetic fixtures (no dataset download needed).
CI runs lint (ruff), tests on Python 3.10 & 3.11, and executes the notebook on every push.

## Tech Stack

- **Language:** Python 3.10+
- **ML:** scikit-learn
- **Data:** pandas, NumPy
- **Visualization:** Plotly, matplotlib
- **Tooling:** pytest, ruff, GitHub Actions

## License & Attribution

Released under the [MIT License](LICENSE).

Dataset: Connectionist Bench (Sonar, Mines vs. Rocks), UCI Machine Learning Repository —
https://doi.org/10.24432/C5J619 (R. Paul Gorman & Terrence J. Sejnowski).

See [`docs/project-report.md`](docs/project-report.md) for the full case-study report
(EDA, model comparison, tuning, interpretability, clustering, and the operational
deployment framework) and [`docs/why-this-matters.md`](docs/why-this-matters.md) for
extended real-world context.
