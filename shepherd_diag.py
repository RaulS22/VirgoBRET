import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

DATA_DIR = Path("/home/rauls/Desktop/VirgoBRET/cut_parquet_data")
PARQUET_FILES = sorted(DATA_DIR.glob("**/*/qtransform_features.parquet"))
RANDOM_STATE = 42

MONTH_ORDER = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

N_PAIRS = 500000

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

tsne = TSNE(n_components=2,perplexity=40,early_exaggeration=500,init="pca",learning_rate="auto",random_state=RANDOM_STATE, metric='manhattan')
X_tsne = tsne.fit_transform(X)

rng = np.random.default_rng(RANDOM_STATE)
N = X.shape[0]

idx1 = rng.integers(0, N, size=N_PAIRS)
idx2 = rng.integers(0, N, size=N_PAIRS)

mask = idx1 != idx2
idx1 = idx1[mask]
idx2 = idx2[mask]

d_original = np.linalg.norm(X[idx1] - X[idx2],axis=1)
d_tsne = np.linalg.norm(X_tsne[idx1] - X_tsne[idx2],axis=1)

rho, pvalue = spearmanr(d_original, d_tsne)
print(f"Spearman correlation = {rho:.4f}")
print(f"p-value = {pvalue:.4e}")

plt.figure(figsize=(9, 7))
plt.scatter(d_original,d_tsne,s=4,alpha=0.2)
plt.xlabel("Distance in original feature space")
plt.ylabel("Distance in t-SNE space")
plt.title("Shepard Diagram — t-SNE")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("shepred_diagram.pdf")

PCA_COMPONENTS = 20
n_components = min(PCA_COMPONENTS,X_scaled.shape[0] - 1,X_scaled.shape[1])
pca = PCA(n_components=n_components,random_state=RANDOM_STATE)

X_pca = pca.fit_transform(X_scaled)
explained_variance = np.sum(pca.explained_variance_ratio_)

print(f"PCA components: {n_components}")
print(f"Explained variance: {explained_variance:.4f}")

tsne = TSNE(n_components=2,perplexity=40,early_exaggeration=500,init="pca",learning_rate="auto",random_state=RANDOM_STATE, metric='manhattan')
X_tsne = tsne.fit_transform(X_pca)
rng = np.random.default_rng(RANDOM_STATE)
N = X.shape[0]

idx1 = rng.integers(0, N, size=N_PAIRS)
idx2 = rng.integers(0, N, size=N_PAIRS)

mask = idx1 != idx2
idx1 = idx1[mask]
idx2 = idx2[mask]

d_original = np.linalg.norm(X_pca[idx1] - X_pca[idx2],axis=1)
d_tsne = np.linalg.norm(X_tsne[idx1] - X_tsne[idx2],axis=1)

rho, pvalue = spearmanr(d_original, d_tsne)
print(f"Spearman correlation = {rho:.4f}")
print(f"p-value = {pvalue:.4e}")

plt.figure(figsize=(9, 7))
plt.scatter(d_original,d_tsne,s=4,alpha=0.2)
plt.xlabel("Distance in PC feature space")
plt.ylabel("Distance in t-SNE space")
plt.title("Shepard Diagram — t-SNE")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("shepred_diagram_pca.pdf")
