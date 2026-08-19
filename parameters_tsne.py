import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplcursors

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATA_DIR = Path("/home/rauls/Desktop/VirgoBRET/cut_parquet_data")
PARQUET_FILES = sorted(DATA_DIR.glob("**/*/qtransform_features.parquet"))

RANDOM_STATE = 42
MONTH_ORDER = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

PERPLEXITY = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
EARLY_EXAGGERATION = [10, 12, 14, 20, 100, 300, 1000]
PCA_COMPONENTS = [2, 10, 20, 40]
METRIC = ['euclidean', 'manhattan', 'cosine']

if len(PARQUET_FILES) == 0:
    raise FileNotFoundError(
        f"No parquet files found inside: {DATA_DIR}")

print(f"Found {len(PARQUET_FILES)} parquet files.")

dataframes = []

parquet_files = list(DATA_DIR.glob("*/*/qtransform_features.parquet"))
parquet_files.sort(key=lambda p: (int(p.parent.parent.name), MONTH_ORDER[p.parent.name]))

for file in parquet_files:
    print(f"Loading: {file}")

    df = pd.read_parquet(file)
    folder = file.parent

    image_paths = [str(folder / f"qtransform_{i:06d}.pdf")for i in range(len(df))]
    missing_images = [path for path in image_paths if not Path(path).exists()]

    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} PDF files are missing in {folder}. First missing file: {missing_images[0]}")

    df["image_path"] = image_paths
    df["source_file"] = file.name
    df["source_path"] = str(file)

    dataframes.append(df)

df_all = pd.concat(dataframes,ignore_index=True)

feature_columns = [col for col in df_all.columns if col.startswith("feature_")]
if len(feature_columns) == 0:
    raise ValueError("No feature_* columns found.")

X = df_all[feature_columns].to_numpy(dtype=np.float64)
X_norm = X / 40

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)
print(f"Scaled matrix: {X_scaled.shape}") 

for p in PERPLEXITY:
    tsne = TSNE(n_components=2,perplexity=p,init="pca",learning_rate="auto",random_state=RANDOM_STATE)
    X_tsne = tsne.fit_transform(X_norm)

    plt.figure(figsize=(12, 10))
    plt.scatter(X_tsne[:, 0],X_tsne[:, 1],s=5,alpha=0.5)
    plt.title(f"Perplexity={p}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    plt.tight_layout()
    plt.savefig(f"tsne_par/perplexity_{p}.pdf.pdf")

for m in METRIC:
    tsne = TSNE(n_components=2,perplexity=40,init="pca",learning_rate="auto",random_state=RANDOM_STATE, metric=m)
    X_tsne = tsne.fit_transform(X_norm)

    plt.figure(figsize=(12, 10))
    plt.scatter(X_tsne[:, 0],X_tsne[:, 1],s=5,alpha=0.5)
    plt.title(f"Perplexity = 40 | Metric = {m}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    plt.tight_layout()
    plt.savefig(f"tsne_par/metric_{m}.pdf")

for e in EARLY_EXAGGERATION:
    tsne = TSNE(n_components=2,perplexity=40,early_exaggeration=e,init="pca",learning_rate="auto",random_state=RANDOM_STATE)
    X_tsne = tsne.fit_transform(X_norm)

    plt.figure(figsize=(12, 10))
    plt.scatter(X_tsne[:, 0],X_tsne[:, 1],s=5,alpha=0.5)
    plt.title(f"Perplexity = 40 | Early Exaggeration = {e}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    plt.tight_layout()
    plt.savefig(f"tsne_par/early_exaggeration_{e}.pdf")

PCA_COMPONENTS = 20

n_components = min(PCA_COMPONENTS,X_scaled.shape[0] - 1,X_scaled.shape[1])
pca = PCA(n_components=n_components,random_state=RANDOM_STATE)

X_pca = pca.fit_transform(X_scaled)
explained_variance = np.sum(pca.explained_variance_ratio_)

print(f"PCA components: {n_components}")
print(f"Explained variance: {explained_variance:.4f}")

for p in PERPLEXITY:
    tsne = TSNE(n_components=2,perplexity=p,init="pca",learning_rate="auto",random_state=RANDOM_STATE)
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(12, 10))
    plt.scatter(X_tsne[:, 0],X_tsne[:, 1],s=5,alpha=0.5)
    plt.title(f"Perplexity={p}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    plt.tight_layout()
    plt.savefig(f"tsne_par/pca_perplexity_{p}.pdf")

for m in METRIC:
    tsne = TSNE(n_components=2,perplexity=40,init="pca",learning_rate="auto",random_state=RANDOM_STATE, metric=m)
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(12, 10))
    plt.scatter(X_tsne[:, 0],X_tsne[:, 1],s=5,alpha=0.5)
    plt.title(f"Perplexity = 40 | Metric = {m}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    plt.tight_layout()
    plt.savefig(f"tsne_par/pca_metric_{m}.pdf")

for e in EARLY_EXAGGERATION:
    tsne = TSNE(n_components=2,perplexity=40,early_exaggeration=e,init="pca",learning_rate="auto",random_state=RANDOM_STATE)
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(12, 10))
    plt.scatter(X_tsne[:, 0],X_tsne[:, 1],s=5,alpha=0.5)
    plt.title(f"Perplexity = 40 | Early Exaggeration = {e}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    plt.tight_layout()
    plt.savefig(f"tsne_par/pca_early_exaggeration_{e}.pdf")