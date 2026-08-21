import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap

DATA_DIR = Path("/home/rauls/Desktop/VirgoBRET/cut_parquet_data")
PARQUET_FILES = sorted(DATA_DIR.glob("**/*/qtransform_features.parquet"))

MONTH_ORDER = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

N_NEIGHBORS = [20, 50, 100, 200, 1000] #troubles with low number of neighbors
MIN_DIST = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
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

print(f"Number of features: {len(feature_columns)}")

X = df_all[feature_columns].to_numpy(dtype=np.float64)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

for n in N_NEIGHBORS:
    reducer = umap.UMAP(n_neighbors=n) 
    embedded = reducer.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1],s=5,alpha=0.5)
    ax.set_title(f"UMAP using n_neighbors={n}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(f"umap_par/UMAP_n_neighbors={n}.pdf")

for m in METRIC:
    reducer = umap.UMAP(metric=m) 
    embedded = reducer.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1],s=5,alpha=0.5)
    ax.set_title(f"UMAP using metric={m}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(f"umap_par/UMAP_metric={m}.pdf")

for d in MIN_DIST:
    reducer = umap.UMAP(min_dist=d) 
    embedded = reducer.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1],s=5,alpha=0.5)
    ax.set_title(f"UMAP using min_dist={d}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(f"umap_par/UMAP_min_dist={d}.pdf")

#Normalized data

X = df_all[feature_columns].to_numpy(dtype=np.float64)
X_norm = X / 40

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)
reducer = umap.UMAP() 

embedded = reducer.fit_transform(X_scaled)

for n in N_NEIGHBORS:
    reducer = umap.UMAP(n_neighbors=n) 
    embedded = reducer.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1],s=5,alpha=0.5)
    ax.set_title(f"UMAP using n_neighbors={n}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(f"umap_par/norm_UMAP_n_neighbors={n}.pdf")

for m in METRIC:
    reducer = umap.UMAP(metric=m) 
    embedded = reducer.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1],s=5,alpha=0.5)
    ax.set_title(f"UMAP using metric={m}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(f"umap_par/norm_UMAP_metric={m}.pdf")

for d in MIN_DIST:
    reducer = umap.UMAP(min_dist=d) 
    embedded = reducer.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1],s=5,alpha=0.5)
    ax.set_title(f"UMAP using min_dist={d}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(f"umap_par/norm_UMAP_min_dist={d}.pdf")