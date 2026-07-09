# Dataset

This project uses the **Connectionist Bench (Sonar, Mines vs. Rocks)** dataset from the UCI Machine Learning Repository.

- **Source:** UCI Machine Learning Repository — https://doi.org/10.24432/C5J619
- **Original authors:** R. Paul Gorman and Terrence J. Sejnowski
- **Samples:** 208 sonar returns
- **Features:** 60 continuous frequency-band energy readings (`freq_1` … `freq_60`), each in the range [0, 1]
- **Target:** binary label — `M` (Mine) = 111 samples, `R` (Rock) = 97 samples
- **Missing values:** 0
- **Duplicates:** 0

## How to obtain the data

The raw data file is **not committed** to this repository (see `.gitignore`). Download it from the UCI repository and place it in this `data/` folder before running the pipeline:

1. Visit https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
2. Download `sonar.all-data` (or the zipped bundle).
3. Save it into this `data/` directory.

The loading script (`src/01_load_data.py`) expects the raw file here and assigns column names `freq_1` … `freq_60` plus a `label` column.

## Column description

| Column | Description |
|---|---|
| `freq_1` … `freq_60` | Energy reflected back at each of 60 acoustic frequency bands, normalized to [0, 1] |
| `label` | `M` = metal cylinder (mine), `R` = rock |

During cleaning (`src/02_data_cleaning.py`) the string `label` is encoded to a numeric `target` column (`M` → 1, `R` → 0).
