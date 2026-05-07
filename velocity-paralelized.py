# blrms_mpi.py

import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_inventory
from pathlib import Path
from collections import defaultdict
from mpi4py import MPI
import gc

# ==========================================================
# MPI SETUP
# ==========================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ==========================================================
# PARAMETERS
# ==========================================================
INPUT_DIR = Path("/home/rauls/Desktop/GithubITA/VirgoBRET/SENA-mseed")
INVENTORY_PATH = "/home/rauls/Desktop/GithubITA/VirgoBRET/fdsn_station.xml"

WINDOW_SEC = 60

BANDS = {
    "microseism": (0.1, 0.3),
    "secondary_microseism": (0.3, 1.0),
    "low": (1.0, 3.0),
    "mid": (3.0, 10.0),
    "high": (10.0, 30.0),
}

# ==========================================================
# CORE FUNCTIONS
# ==========================================================
def compute_windowed_rms(data, fs, window_sec):
    n = int(window_sec * fs)
    n_windows = len(data) // n

    if n_windows == 0:
        return np.array([])

    trimmed = data[:n_windows * n]
    reshaped = trimmed.reshape(n_windows, n)

    return np.sqrt(np.mean(reshaped**2, axis=1))


def process_trace(tr, inv):
    results = {}

    tr.remove_response(inventory=inv, output="VEL", water_level=60)
    fs = tr.stats.sampling_rate

    for name, (fmin, fmax) in BANDS.items():
        tr_band = tr.copy()

        tr_band.filter(
            "bandpass",
            freqmin=fmin,
            freqmax=fmax,
            corners=4,
            zerophase=True
        )

        data = tr_band.data * 1e9
        data = data[np.isfinite(data)]

        if len(data) == 0:
            continue

        rms = compute_windowed_rms(data, fs, WINDOW_SEC)

        if len(rms) > 0:
            results[name] = rms

        del tr_band

    return results


def process_file(file, inv):
    print(f"[Rank {rank}] Processing {file.name}")

    st = read(file, dtype=np.float32)
    st.merge(method=1, fill_value="interpolate")

    file_results = []

    for tr in st:
        try:
            res = process_trace(tr, inv)
            date = tr.stats.starttime.date
            file_results.append((date, res))

        except Exception as e:
            print(f"[Rank {rank}] Error in {file.name}: {e}")

    del st
    gc.collect()

    return file_results


# ==========================================================
# DISTRIBUTE FILES
# ==========================================================
if rank == 0:
    files = sorted(INPUT_DIR.glob("*.mseed"))
    print(f"Total files: {len(files)}")

    # Split files across ranks
    chunks = np.array_split(files, size)
else:
    chunks = None

# Scatter file lists
local_files = comm.scatter(chunks, root=0)

# ==========================================================
# LOAD INVENTORY (per rank)
# ==========================================================
inv = read_inventory(INVENTORY_PATH)

# ==========================================================
# LOCAL PROCESSING
# ==========================================================
local_results = []

for file in local_files:
    local_results.extend(process_file(file, inv))

# ==========================================================
# GATHER RESULTS
# ==========================================================
all_results = comm.gather(local_results, root=0)

# ==========================================================
# AGGREGATION (rank 0 only)
# ==========================================================
if rank == 0:
    flat_results = [item for sublist in all_results for item in sublist]

    daily = defaultdict(lambda: defaultdict(list))

    for date, band_dict in flat_results:
        for band, values in band_dict.items():
            daily[date][band].extend(values)

    final = {}

    for date in sorted(daily.keys()):
        final[date] = {}
        for band in daily[date]:
            final[date][band] = np.median(daily[date][band])

    # ======================================================
    # PLOT
    # ======================================================
    dates = sorted(final.keys())
    bands = list(next(iter(final.values())).keys())

    plt.figure(figsize=(12, 6))

    for band in bands:
        vals = [final[d].get(band, np.nan) for d in dates]
        plt.plot(dates, vals, marker='o', label=band)

    plt.xlabel("Date")
    plt.ylabel("BLRMS (nm/s)")
    plt.title("Daily BLRMS per Band (MPI)")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()