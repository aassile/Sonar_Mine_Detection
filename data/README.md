# Data

## Dataset

This project uses the **Connectionist Bench (Sonar, Mines vs. Rocks)** dataset from the
UCI Machine Learning Repository.

| Property | Value |
|---|---|
| Source | [UCI ML Repository #151](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) — https://doi.org/10.24432/C5J619 |
| Original authors | R. Paul Gorman & Terrence J. Sejnowski |
| Samples | 208 sonar returns |
| Features | 60 continuous frequency-band energy readings (`freq_1` … `freq_60`), each in [0, 1] |
| Target | Binary — `M` (mine) = 111, `R` (rock) = 97 |
| Missing values | 0 |
| Duplicates | 0 |

## Committed file

Because the dataset is tiny (~85 KB), the labeled CSV is **committed directly** as
[`data/sonar.csv`](sonar.csv). No download is required — the notebook and CI runs read it
straight from the repo. It has 60 float columns `freq_1 … freq_60` plus a string `label`
column (`M` / `R`).

## Regenerating from the raw UCI file

If you want to rebuild the CSV from the original archive:

```python
from sonar_detection import load_sonar_raw

# sonar.all-data comes from the UCI zip linked above
df = load_sonar_raw("sonar.all-data")
df.to_csv("data/sonar.csv", index=False)
```

## Schema

| Column | Description |
|---|---|
| `freq_1` … `freq_60` | Energy reflected back at each of 60 acoustic frequency bands, normalized to [0, 1] |
| `label` | `M` = metal cylinder (mine), `R` = rock |

During preprocessing (`sonar_detection.preprocessing.encode_target`) the string `label`
is encoded into a numeric `target` column (`M` → 1, `R` → 0) and the string column is dropped.
