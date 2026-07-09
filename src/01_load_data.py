
import zipfile
import pandas as pd

# Unzip
with zipfile.ZipFile("undocumented.zip", "r") as z:
    z.extractall("undocumented/")

# ── Load Sonar dataset ────────────────────────────────────────────────────────
# 208 sonar returns × 60 frequency-band energy readings + label (M=Mine, R=Rock)
sonar_df = pd.read_csv(
    "undocumented/connectionist-bench/sonar/sonar.all-data",
    header=None
)
sonar_df.columns = [f"freq_{i+1}" for i in range(60)] + ["label"]

# ── Citation ──────────────────────────────────────────────────────────────────
print("Dataset Citation:")
print("Undocumented [Dataset]. UCI Machine Learning Repository.")
print("https://doi.org/10.24432/C5J619")
print()

print("=== Sonar: Mines vs. Rocks ===")
print(f"Shape   : {sonar_df.shape[0]} rows × {sonar_df.shape[1]} columns")
print(f"Classes : {sonar_df['label'].value_counts().to_dict()}  (M=Mine, R=Rock)")
print(f"Dtypes  : {sonar_df.dtypes.value_counts().to_dict()}")
print(f"Nulls   : {sonar_df.isnull().sum().sum()} missing values")
print(f"\nSample (first 3 rows, first 8 features):")
print(sonar_df.iloc[:3, list(range(7)) + [60]].to_string())
