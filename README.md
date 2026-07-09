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
