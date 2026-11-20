import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path

import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RANDOM_STATE = 42
df = pd.read_parquet("year_MNSENA_results_cut/qtransform_features.parquet")

feature_columns = [col for col in df.columns if col.startswith("feature_")]
if len(feature_columns) == 0:
    raise ValueError("No feature_* columns found.")

print(f"Number of features: {len(feature_columns)}")

X = df[feature_columns].to_numpy(dtype="float64")
print(f"X.shape = {X.shape}")

X_max = X.max(axis=1, keepdims=True)

X_norm = X / X_max

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)
print(f"Scaled matrix: {X_scaled.shape}") 

PCA_COMPONENTS = 460

n_components = min(PCA_COMPONENTS,X_scaled.shape[0] - 1,X_scaled.shape[1])
pca = PCA(n_components=n_components,random_state=RANDOM_STATE)

X_pca = pca.fit_transform(X_scaled)
explained_variance = np.sum(pca.explained_variance_ratio_)

print(f"PCA components: {n_components}")
print(f"Explained variance: {explained_variance:.4f}")

tsne = TSNE(n_components=2,perplexity=400, learning_rate=200, init="pca",learning_rate="auto",random_state=RANDOM_STATE, metric='euclidean', n_jobs=-1)
X_tsne = tsne.fit_transform(X_pca)
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],s=5,alpha=0.5)
ax.set_title(f"Perplexity = 400 | Learning Rate = 200 | PCA Components = {PCA_COMPONENTS} | Metric = euclidean")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("tsne_mnsena_euclidean.pdf", dpi=300)

tsne = TSNE(n_components=2,perplexity=400,learning_rate=300, init="pca",learning_rate="auto",random_state=RANDOM_STATE, metric='euclidean', n_jobs=-1)
X_tsne = tsne.fit_transform(X_pca)
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],s=5,alpha=0.5)
ax.set_title(f"Perplexity = 400 | Learning Rate = 300 | PCA Components = {PCA_COMPONENTS} | Metric = euclidean")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("tsne_mnsena_euclidean1.pdf", dpi=300)

tsne = TSNE(n_components=2,perplexity=400,learning_rate=400,init="pca",learning_rate="auto",random_state=RANDOM_STATE, metric='euclidean', n_jobs=-1)
X_tsne = tsne.fit_transform(X_pca)
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],s=5,alpha=0.5)
ax.set_title(f"Perplexity = 400 | Learning Rate = 400 | PCA Components = {PCA_COMPONENTS} | Metric = euclidean")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("tsne_mnsena_euclidean2.pdf", dpi=300)

tsne = TSNE(n_components=2,perplexity=400,learning_rate=500,init="pca",learning_rate="auto",random_state=RANDOM_STATE, metric='euclidean', n_jobs=-1)
X_tsne = tsne.fit_transform(X_pca)
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],s=5,alpha=0.5)
ax.set_title(f"Perplexity = 400 | Learning Rate = 500 | PCA Components = {PCA_COMPONENTS} | Metric = euclidean")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("tsne_mnsena_euclidean3.pdf", dpi=300)