import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import product
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
import umap


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("/home/rauls/Desktop/VirgoBRET/cut_parquet_data")
RANDOM_STATE = 42
MONTH_ORDER = {"jan": 1,"feb": 2,"mar": 3,"apr": 4,"may": 5,"jun": 6,
               "jul": 7,"ago": 8,"sep": 9,"oct": 10,"nov": 11,"dec": 12}

# ============================================================
# t-SNE PARAMETERS
# ============================================================

N_NEIGHBORS = [2, 5, 20, 50, 100, 200, 1000] 
MIN_DIST = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
METRIC = ['euclidean', 'manhattan', 'cosine']
REPULSION_STRENGHT = [0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]


# ============================================================
# SHEPARD PARAMETERS
# ============================================================

# Number of randomly selected point pairs used
# in each Shepard diagram.
N_PAIRS = 100000


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = Path("umap_shepard_results")
UMAP_DIR = OUTPUT_DIR / "pdf"
CSV_DIR = OUTPUT_DIR / "csv"
UMAP_DIR.mkdir(parents=True, exist_ok=True)
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
reducer = umap.UMAP() 

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

distance_cache = []

for metric in METRIC:
    X1 = X[idx1]
    X2 = X[idx2]
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


print("=" * 70)
print("CALCULATING INPUT-SPACE DISTANCES")
print("=" * 70)
distance_cache = {}
results = []
total_combinations = (len(N_NEIGHBORS)* len(MIN_DIST)* len(REPULSION_STRENGHT)* len(METRIC))
print()
print("=" * 70)
print("UMAP PARAMETER SWEEP")
print("=" * 70)

print(f"Total combinations: {total_combinations}")
combination_number = 0


# ============================================================
# ALL PARAMETER COMBINATIONS
# ============================================================

for (n,m,r,metric) in product(N_NEIGHBORS,MIN_DIST,REPULSION_STRENGHT,METRIC):
    combination_number += 1
    print()
    print("-" * 70)
    print(f"[{combination_number}/{total_combinations}] N_Neighbors={n} | Min_dist={m} | Repulsion_str={r} | Metric={metric}")

    # ========================================================
    # UMAP
    # ========================================================

    try: embedded = reducer.fit_transform(X_scaled)

    except Exception as exc:
        print(f"ERROR in UMAP: {exc}")
        results.append({"N_Neighbors": n,"Min_dist": m,"Repulsion_str": r,"metric": metric,
                        "spearman_rho": np.nan,"p_value": np.nan,"status": "FAILED","error": str(exc)})
        continue


    d_umap = np.linalg.norm(embedded[idx1] - embedded[idx2],axis=1)

    # =======================================================
    # SPEARMAN CORRELATIO
    # =======================================================

    rho, pvalue = spearmanr(d_original,d_umap)
    print(f"Spearman rho = {rho:.6f}")
    print(f"p-value      = {pvalue:.6e}")


    # ========================================================
    # FILE NAME
    # ========================================================

    pdf_name = (f"umap_shepard _n{n}_m{m}_repstr{r}_{metric}.pdf")
    pdf_path = UMAP_DIR / pdf_name


    # ========================================================
    # PDF
    # ========================================================

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(12, 10))
        plt.scatter(embedded[:, 0],embedded[:, 1],s=5,alpha=0.5)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"UMAP\n N_Neighbors={n} | Min_dist={m} | Repulsion_str={r} | Metric = {metric}")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig,bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 8))
        plt.scatter(d_original,d_umap,s=4,alpha=0.20)
        plt.xlabel(f"Distance in feature space ({metric})")
        plt.ylabel("Distance in UMAP space")
        plt.title(f"Shepard Diagram\n N_Neighbors={n} | Min_dist={m} | Repulsion_str={r} | Metric = {metric} \nSpearman $\\rho$ = {rho:.6f} | p-value = {pvalue:.3e}")

        plt.grid(alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig,bbox_inches="tight")
        plt.close(fig)


    # ========================================================
    # SAVE NUMERICAL RESULTS
    # ========================================================

    results.append({"N_Neighbors": n,"Min_dist": m,"Repulsion_str": r,"metric": metric,"spearman_rho": rho,
                    "p_value": pvalue,"n_pairs": len(idx1),"pdf": str(pdf_path),"status": "OK","error": ""})
    
# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)
results_csv = (CSV_DIR /"umap_shepard_results.csv")
results_df.to_csv(results_csv,index=False)

# ============================================================
# SORT BY SPEARMAN CORRELATION
# ============================================================

results_sorted = results_df.sort_values(by="spearman_rho",ascending=False)
best_csv = (CSV_DIR /"umap_shepard_results_sorted.csv")
results_sorted.to_csv(best_csv,index=False)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print("FINISHED")
print("=" * 70)
print(f"Results directory:\n"f"{OUTPUT_DIR.resolve()}")
print(f"PDF files:\n"f"{UMAP_DIR.resolve()}")
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
    print(best[["N_Neighbors","Min_dist","Repulsion_str","metric","spearman_rho","p_value"]])