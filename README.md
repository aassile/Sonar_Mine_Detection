# 🌊 Sonar Mine Detection — Acoustic Signal Classification with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Built%20on-Zerve-6C63FF)](https://zerve.ai)
[![Model](https://img.shields.io/badge/Best%20Model-Extra%20Trees-brightgreen)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
[![Accuracy](https://img.shields.io/badge/Accuracy-92.9%25-success)](https://zerve.ai)
[![AUC](https://img.shields.io/badge/AUC-0.982-success)](https://zerve.ai)
[![Live App](https://img.shields.io/badge/Live%20App-sonar--mine--detector.zerve.app-orange)](https://sonar-mine-detector.zerve.app)

A full end-to-end data science project that classifies underwater sonar returns as either **sea mines** or **rocks** using acoustic frequency-band energy signatures — built entirely on [Zerve](https://zerve.ai).

> **Live Demo:** [sonar-mine-detector.zerve.app](https://sonar-mine-detector.zerve.app)

---

## 📌 Objective

Can a machine reliably distinguish an underwater sea mine from a rock — using only reflected sound?

A sonar device pings the seabed and records the energy reflected back across **60 frequency bands**. To the human eye, Mine and Rock returns look nearly identical. This project trains and deploys a machine learning model to make that distinction automatically, instantly, and with 92.9% accuracy — enabling real-time classification on autonomous underwater vehicles (AUVs) without sending human divers into danger.

---

## 🌍 Why This Matters

| Domain | Application |
|---|---|
| 🪖 **Military & Defence** | AUV mine sweeping before ships enter contested or mined waters |
| 🚢 **Commercial Shipping** | Certifying trade routes through historically mined straits (Hormuz, Suez, South China Sea) |
| 🤿 **Humanitarian Demining** | ~11M active sea mines remain from past conflicts; automation reduces diver exposure |
| 🔬 **Ocean Science** | Pipeline inspection, geological surveys, seabed habitat mapping |

The model outputs a three-tier risk decision:
- 🔴 **HIGH** (P(Mine) > 0.70) → Dispatch EOD team
- 🟡 **MODERATE** (0.40–0.70) → Flag for human review
- 🟢 **LOW** (P(Mine) < 0.40) → Clear to proceed

---

## 📦 Dataset

**Undocumented [Dataset]. UCI Machine Learning Repository.**
https://doi.org/10.24432/C5J619

| Property | Value |
|---|---|
| Samples | 208 sonar returns |
| Features | 60 continuous frequency-band energy readings (freq_1 → freq_60) |
| Target | Binary — M (Mine) = 111 samples \| R (Rock) = 97 samples |
| Missing values | 0 |
| Duplicates | 0 |

Originally collected by Terry Sejnowski and R. Paul Gorman. Sonar signals were bounced off objects at various angles and distances; each band captures how much energy is reflected at a specific acoustic frequency.

---

## 🔄 Workflow

The project runs as an ordered pipeline. Each stage lives in `src/`:

1. **Load** the raw sonar returns and assign frequency-band column names.
2. **Clean** — encode the `M`/`R` label to a numeric `target`, verify types, confirm no nulls/duplicates.
3. **Profile** the data — class balance, per-band statistics, value ranges.
4. **Explore** — class-specific energy profiles, top discriminating bands, PCA projection.
5. **Model** — train a Random Forest baseline.
6. **Compare** — Random Forest vs Extra Trees vs Gradient Boosting vs Neural Network across accuracy, ROC–AUC, cross-validation, and F1.
7. **Tune** — randomized hyperparameter search on the winning Extra Trees model.
8. **Cluster** — unsupervised KMeans on the sonar signals (elbow/silhouette, PCA scatter, cluster energy profiles).
9. **Interpret & export** — permutation importance, individual prediction breakdowns, synthetic-sample tests, and final model export.

---

## 📁 Repository Structure

```text
Sonar_Mine_Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md            # dataset description + download instructions
├── docs/
│   └── why-this-matters.md  # real-world context & applications
├── src/
│   ├── 01_load_data.py
│   ├── 02_data_cleaning.py
│   ├── 03_data_profiling.py
│   ├── 04_exploratory_analysis.py
│   ├── 05_model_random_forest.py
│   ├── 06_model_comparison.py
│   ├── 07_extra_trees_tuning.py
│   ├── 08_clustering_kmeans.py
│   ├── 09_model_export.py
│   └── charts/              # Plotly visualization scripts (Chart 1–19)
├── notebooks/               # optional exploratory notebooks
└── outputs/
    └── figures/             # generated charts (git-ignored)
```

---

## 🧠 Methods

- **Supervised learning:** Random Forest, Extra Trees, Gradient Boosting, and a Neural Network (`MLPClassifier`), all via scikit-learn.
- **Model selection:** stratified 5-fold cross-validation scored on ROC–AUC.
- **Hyperparameter tuning:** randomized search over Extra Trees parameters.
- **Evaluation:** accuracy, ROC–AUC, confusion matrix, precision/recall/F1 per class.
- **Interpretability:** permutation feature importance and per-prediction marginal-contribution breakdowns.
- **Unsupervised analysis:** KMeans clustering with elbow/silhouette diagnostics and PCA visualization.
- **Visualization:** interactive Plotly charts throughout.

---

## 📊 Results

The tuned **Extra Trees** classifier is the best performer on this dataset:

| Metric | Score |
|---|---|
| Test accuracy | 92.9% |
| ROC–AUC | 0.982 |
| 5-fold CV AUC | ~0.944 |

> ⚠️ The held-out test set is small (42 samples), so cross-validation metrics are the more reliable estimate of generalization. In an operational setting this model would act as a **first-pass filter** that narrows the set of sonar returns needing expert review — not a standalone safety-critical decision maker.

---

## ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/aassile/Sonar_Mine_Detection.git
cd Sonar_Mine_Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset (see data/README.md) into data/

# 4. Run the pipeline stages in order
python src/01_load_data.py
python src/02_data_cleaning.py
# ... through src/09_model_export.py
```

> Note: the scripts were originally authored on the [Zerve](https://zerve.ai) platform, where blocks share a persistent kernel and pass variables between stages. To run them as standalone Python files you may need to chain the stages in a single session or notebook so downstream variables (e.g. `cleaned_df`, train/test splits) are available.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **ML:** scikit-learn
- **Data:** pandas, NumPy
- **Visualization:** Plotly, matplotlib
- **Original platform:** Zerve

---

## 🚀 Future Improvements

- Refactor shared pipeline logic into importable modules and package a single reproducible notebook.
- Add automated tests and a `make`/script entry point that runs the full pipeline end-to-end.
- Persist the trained model artifact and add an inference script.
- Expand beyond the benchmark dataset with additional data and domain validation before any real-world use.

---

## 📄 License & Attribution

Dataset: Connectionist Bench (Sonar, Mines vs. Rocks), UCI Machine Learning Repository — https://doi.org/10.24432/C5J619 (R. Paul Gorman & Terrence J. Sejnowski).

See `docs/why-this-matters.md` for extended real-world context.
