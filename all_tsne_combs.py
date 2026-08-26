import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import product
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

#import time

#TODO: Review Loops

# ============================================================
# CONFIGURATION
# ============================================================

#time.sleep(7200)

DATA_DIR = Path("/home/rauls/Desktop/VirgoBRET/cut_parquet_data")
RANDOM_STATE = 42
MONTH_ORDER = {"jan": 1,"feb": 2,"mar": 3,"apr": 4,"may": 5,"jun": 6,
               "jul": 7,"ago": 8,"sep": 9,"oct": 10,"nov": 11,"dec": 12}

# ============================================================
# t-SNE PARAMETERS
# ============================================================

PERPLEXITY = [10, 20, 30, 40, 50,60, 70, 80, 90, 100]
EARLY_EXAGGERATION = [10, 12, 14, 20,100, 300, 1000]
PCA_COMPONENTS = [2, 10, 20, 40]
METRIC = ["euclidean","manhattan","cosine"]

# ============================================================
# SHEPARD PARAMETERS
# ============================================================

# Number of randomly selected point pairs used
# in each Shepard diagram.
N_PAIRS = 100000


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = Path("tsne_shepard_results")
TSNE_DIR = OUTPUT_DIR / "pdf"
CSV_DIR = OUTPUT_DIR / "csv"
TSNE_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND PARQUET FILES
# ============================================================

PARQUET_FILES = list(DATA_DIR.glob("*/*/qtransform_features.parquet"))
PARQUET_FILES.sort(key=lambda p: (int(p.parent.parent.name),MONTH_ORDER[p.parent.name]))

if len(PARQUET_FILES) == 0:
    raise FileNotFoundError(f"No parquet files found inside: {DATA_DIR}")

print("=" * 70)
print("LOADING DATA")
print("=" * 70)
print(f"Found {len(PARQUET_FILES)} parquet files.")


# ============================================================
# LOAD DATA
# ============================================================

dataframes = []
for file in PARQUET_FILES:
    print(f"Loading: {file}")
    df = pd.read_parquet(file)
    folder = file.parent
    # --------------------------------------------------------
    # q-transform PDF associated with each trigger
    # --------------------------------------------------------
    image_paths = [str(folder / f"qtransform_{i:06d}.pdf")for i in range(len(df))]
    missing_images = [path for path in image_paths if not Path(path).exists()]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} PDF files are missing in {folder}. First missing file: {missing_images[0]}")

    df["image_path"] = image_paths
    df["source_file"] = file.name
    df["source_path"] = str(file)

    dataframes.append(df)


# ============================================================
# COMBINE DATA
# ============================================================

df_all = pd.concat(dataframes,ignore_index=True)

feature_columns = [col for col in df_all.columns if col.startswith("feature_")]
if len(feature_columns) == 0:
    raise ValueError("No feature_* columns found.")


X = df_all[feature_columns].to_numpy(dtype=np.float64)
X_norm = X / 40.0
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)

# ============================================================
# FIXED RANDOM PAIRS FOR SHEPARD DIAGRAM
# ============================================================

# IMPORTANT:
# The same pairs are used for every configuration.
# This makes the Shepard values directly comparable.

rng = np.random.default_rng(RANDOM_STATE)
N = X.shape[0]

idx1 = rng.integers(0,N,size=N_PAIRS)
idx2 = rng.integers(0,N,size=N_PAIRS)
mask = idx1 != idx2
idx1 = idx1[mask]
idx2 = idx2[mask]
print(f"Number of Shepard pairs: {len(idx1)}")


print("=" * 70)
print("CALCULATING INPUT-SPACE DISTANCES")
print("=" * 70)
distance_cache = {}
for n_pca in PCA_COMPONENTS:
    n_components = min(n_pca,X_scaled.shape[0] - 1,X_scaled.shape[1])
    print(f"Preparing PCA = {n_components}")
    pca = PCA(n_components=n_components,random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Explained variance = {explained_variance:.6f}")

    for metric in METRIC:
        print(f"Calculating distances: PCA={n_components}, metric={metric}")
        X1 = X_pca[idx1]
        X2 = X_pca[idx2]
        if metric == "euclidean":
            d_original = np.linalg.norm(X1 - X2,axis=1)

        elif metric == "manhattan":
            d_original = np.sum(np.abs(X1 - X2),axis=1)

        elif metric == "cosine":
            norm1 = np.linalg.norm(X1,axis=1)
            norm2 = np.linalg.norm(X2,axis=1)
            denominator = norm1 * norm2

            # Prevent division by zero
            denominator = np.where(denominator == 0,1e-12,denominator)
            cosine_similarity = np.sum(X1 * X2,axis=1) / denominator
            d_original = 1.0 - cosine_similarity

        else:
            raise ValueError(f"Unknown metric: {metric}")
        distance_cache[(n_components, metric)] = (X_pca,d_original,explained_variance)


results = []

total_combinations = (len(PERPLEXITY)* len(EARLY_EXAGGERATION)* len(PCA_COMPONENTS)* len(METRIC))
print()
print("=" * 70)
print("t-SNE PARAMETER SWEEP")
print("=" * 70)

print(f"Total combinations: {total_combinations}")
combination_number = 0


# ============================================================
# ALL PARAMETER COMBINATIONS
# ============================================================

for (p,e,n_pca,metric) in product(PERPLEXITY,EARLY_EXAGGERATION,PCA_COMPONENTS,METRIC):
    combination_number += 1
    print()
    print("-" * 70)
    print(f"[{combination_number}/{total_combinations}] Perplexity={p} |Early Exaggeration={e} | PCA Components={n_pca} | Metric={metric}")

    (X_pca,d_original,explained_variance) = distance_cache[(n_pca, metric)]


    # ========================================================
    # t-SNE
    # ========================================================

    tsne = TSNE(n_components=2,perplexity=p,early_exaggeration=e,init="pca",learning_rate="auto",random_state=RANDOM_STATE,metric=metric, n_jobs=-1)
    try:

        X_tsne = tsne.fit_transform(X_pca)

    except Exception as exc:
        print(f"ERROR in t-SNE: {exc}")
        results.append({"perplexity": p,"early_exaggeration": e,"pca_components": n_components,"metric": metric,
                        "explained_variance": explained_variance,"spearman_rho": np.nan,"p_value": np.nan,"status": "FAILED","error": str(exc)})
        continue


    d_tsne = np.linalg.norm(X_tsne[idx1] - X_tsne[idx2],axis=1)

    # =======================================================
    # SPEARMAN CORRELATIO
    # =======================================================

    rho, pvalue = spearmanr(d_original,d_tsne)
    print(f"Spearman rho = {rho:.6f}")
    print(f"p-value      = {pvalue:.6e}")


    # ========================================================
    # FILE NAME
    # ========================================================

    pdf_name = (f"tsne_shepard _p{p}_e{e}_pca{n_components}_{metric}.pdf")
    pdf_path = TSNE_DIR / pdf_name


    # ========================================================
    # PDF
    # ========================================================

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(12, 10))
        plt.scatter(X_tsne[:, 0],X_tsne[:, 1],s=5,alpha=0.5)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE\n Perplexity = {p} | Early Exaggeration = {e} |" \
        f" PCA Components = {n_components} | Metric = {metric}")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig,bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 8))
        plt.scatter(d_original,d_tsne,s=4,alpha=0.20)
        plt.xlabel(f"Distance in PCA feature space ({metric})")
        plt.ylabel("Distance in t-SNE space")
        plt.title(f"Shepard Diagram\n Perplexity = {p} | Early Exaggeration = {e} | PCA Components = {n_components} | Metric = {metric}\nSpearman $\\rho$ = {rho:.6f} | p-value = {pvalue:.3e}")

        plt.grid(alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig,bbox_inches="tight")
        plt.close(fig)


    # ========================================================
    # SAVE NUMERICAL RESULTS
    # ========================================================

    results.append({"perplexity": p,"early_exaggeration": e,"pca_components": n_components,
                    "metric": metric,"explained_variance": explained_variance,"spearman_rho": rho,
                    "p_value": pvalue,"n_pairs": len(idx1),"pdf": str(pdf_path),"status": "OK","error": ""})
    
# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)
results_csv = (CSV_DIR /"tsne_shepard_results.csv")
results_df.to_csv(results_csv,index=False)

# ============================================================
# SORT BY SPEARMAN CORRELATION
# ============================================================

results_sorted = results_df.sort_values(by="spearman_rho",ascending=False)
best_csv = (CSV_DIR /"tsne_shepard_results_sorted.csv")
results_sorted.to_csv(best_csv,index=False)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print("FINISHED")
print("=" * 70)
print(f"Results directory:\n"f"{OUTPUT_DIR.resolve()}")
print(f"PDF files:\n"f"{TSNE_DIR.resolve()}")
print(f"CSV results:\n"f"{results_csv.resolve()}")
print(f"Sorted results:\n {best_csv.resolve()}")


# ============================================================
# BEST CONFIGURATION
# ============================================================

valid_results = results_df[results_df["status"] == "OK"]


if len(valid_results) > 0:
    best = valid_results.loc[valid_results["spearman_rho"].idxmax()]
    print("=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(best[["perplexity","early_exaggeration","pca_components","metric","spearman_rho","p_value"]])