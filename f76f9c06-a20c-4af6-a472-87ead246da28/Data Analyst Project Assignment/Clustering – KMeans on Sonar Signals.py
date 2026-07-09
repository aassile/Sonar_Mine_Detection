import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ── Features & scaling ───────────────────────────────────────────────────────
_features = [c for c in cleaned_df.columns if c != 'target']
_X = cleaned_df[_features].values
_y = cleaned_df['target'].values  # true labels (for alignment check only)

cluster_scaler = StandardScaler()
X_scaled = cluster_scaler.fit_transform(_X)

# ── Elbow + Silhouette over k=2..8 ──────────────────────────────────────────
k_range = range(2, 9)
inertias, sil_scores = [], []

for k in k_range:
    _km = KMeans(n_clusters=k, random_state=42, n_init=20)
    _labels = _km.fit_predict(X_scaled)
    inertias.append(_km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, _labels))

print("=== Elbow & Silhouette Scores ===")
print(f"{'k':>3}  {'Inertia':>10}  {'Silhouette':>12}")
for k, ine, sil in zip(k_range, inertias, sil_scores):
    marker = " ◀ best" if sil == max(sil_scores) else ""
    print(f"{k:>3}  {ine:>10.1f}  {sil:>12.4f}{marker}")

# ── Optimal k: highest silhouette score ──────────────────────────────────────
optimal_k = list(k_range)[sil_scores.index(max(sil_scores))]
print(f"\nOptimal k = {optimal_k}  (silhouette = {max(sil_scores):.4f})")

# ── Fit final KMeans with optimal k ─────────────────────────────────────────
km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = km_final.fit_predict(X_scaled)

# ── PCA for visualisation (2D) ───────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pca_var = pca.explained_variance_ratio_

# ── Cluster profiles ─────────────────────────────────────────────────────────
cluster_df = cleaned_df.copy()
cluster_df['cluster'] = cluster_labels
cluster_df['true_label'] = _y  # 1=Mine, 0=Rock

print(f"\n=== Cluster Composition (k={optimal_k}) ===")
_comp = cluster_df.groupby('cluster')['true_label'].agg(
    Total='count',
    Mines=lambda x: (x == 1).sum(),
    Rocks=lambda x: (x == 0).sum()
)
_comp['%Mine'] = (_comp['Mines'] / _comp['Total'] * 100).round(1)
_comp['%Rock'] = (_comp['Rocks'] / _comp['Total'] * 100).round(1)
print(_comp.to_string())

# ── Cluster mean energy profiles ─────────────────────────────────────────────
cluster_means = cluster_df.groupby('cluster')[_features].mean()

# ── Export variables for chart block ─────────────────────────────────────────
cluster_k_range   = list(k_range)
cluster_inertias  = inertias
cluster_sil_scores = sil_scores
cluster_optimal_k = optimal_k
cluster_pca_X     = X_pca
cluster_pca_var   = pca_var
cluster_comp      = _comp
cluster_means_df  = cluster_means
cluster_true_labels = _y
cluster_band_names  = _features

print(f"\nPCA variance explained: PC1={pca_var[0]:.1%}, PC2={pca_var[1]:.1%}")
print("Clustering complete ✓")
