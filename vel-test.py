import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_inventory
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import gc

# ==========================================================
# PARAMETERS
# ==========================================================
INPUT_DIR = Path("/home/rauls/Desktop/GithubITA/VirgoBRET/SENA-mseed")
INVENTORY_PATH = "/home/rauls/Desktop/GithubITA/VirgoBRET/fdsn_station.xml"

WINDOW_SEC = 20   # BLRMS window (seconds)
NPROC = 4         # number of parallel processes

# Virgo-like frequency bands (Hz)
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

    rms = np.sqrt(np.mean(reshaped**2, axis=1))
    return rms


def process_trace(tr, inv, bands, window_sec):
    results = {}

    # Remove instrument response (to velocity)
    tr.remove_response(inventory=inv, output="VEL", water_level=60)

    fs = tr.stats.sampling_rate

    for name, (fmin, fmax) in bands.items():
        tr_band = tr.copy()

        tr_band.filter(
            "bandpass",
            freqmin=fmin,
            freqmax=fmax,
            corners=4,
            zerophase=True
        )

        data = tr_band.data * 1e9  # convert to nm/s
        data = data[np.isfinite(data)]

        if len(data) == 0:
            continue

        rms = compute_windowed_rms(data, fs, window_sec)

        if len(rms) > 0:
            results[name] = rms

        del tr_band

    return results


def process_file(file, inv, bands, window_sec):
    print(f"Processing {file.name}")

    st = read(file, dtype=np.float32)
    st.merge(method=1, fill_value="interpolate")

    file_results = []

    for tr in st:
        try:
            res = process_trace(tr, inv, bands, window_sec)
            date = tr.stats.starttime.date

            file_results.append((date, res))

        except Exception as e:
            print(f"Error in {file.name}: {e}")

    del st
    gc.collect()

    return file_results


def run_pipeline(files, inv, bands, window_sec, nproc=4):
    func = partial(process_file,
                   inv=inv,
                   bands=bands,
                   window_sec=window_sec)

    results = []

    with ProcessPoolExecutor(max_workers=nproc) as exe:
        for out in exe.map(func, files):
            results.extend(out)

    return results


def aggregate(results):
    daily = defaultdict(lambda: defaultdict(list))

    for date, band_dict in results:
        for band, values in band_dict.items():
            daily[date][band].extend(values)

    final = {}

    for date in sorted(daily.keys()):
        final[date] = {}
        for band in daily[date]:
            final[date][band] = np.median(daily[date][band])

    return final


def plot_results(final):
    dates = sorted(final.keys())
    bands = list(next(iter(final.values())).keys())

    plt.figure(figsize=(12, 6))

    for band in bands:
        vals = [final[d].get(band, np.nan) for d in dates]
        plt.plot(dates, vals, marker='o', label=band)

    plt.xlabel("Date")
    plt.ylabel("BLRMS (nm/s)")
    plt.title("Daily BLRMS per Band")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("blrms_results.png")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob("*.mseed"))

    print(f"Found {len(files)} files")

    inv = read_inventory(INVENTORY_PATH)

    results = run_pipeline(
        files,
        inv,
        BANDS,
        window_sec=WINDOW_SEC,
        nproc=NPROC
    )

    final = aggregate(results)

    plot_results(final)