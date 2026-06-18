import numpy as np
import matplotlib.pyplot as plt
import gc

from obspy import read
from PyEMD import CEEMDAN #use pip install EMD-signal


# ==========================================================
# USER PARAMETERS
# ==========================================================

MSEED_FILE = "22-02-25-Raul.mseed"
OUTPUT_FILE = "CEEMDAN_reconstructed.mseed"

FMIN = 0.3
FMAX = 30.0

# IMF indices to keep (Python indexing starts at 0)
IMFS_TO_KEEP = [2, 3, 4, 5]
WINDOW_HOURS = 1.0
TRIALS = 10
RANDOM_SEED = 42


# ==========================================================
# READ DATA
# ==========================================================

print("Reading MiniSEED...")

st = read(MSEED_FILE)

print(st)

tr = st[0].copy()

print(f"Sampling rate: {tr.stats.sampling_rate:.2f} Hz")
print(f"NPTS: {tr.stats.npts:,}")
print(f"Duration: {tr.stats.endtime - tr.stats.starttime}")

# Optional downsampling
TARGET_FS = 50.0

if tr.stats.sampling_rate > TARGET_FS:
    print(f"Resampling to {TARGET_FS} Hz")
    tr.resample(TARGET_FS)

fs = tr.stats.sampling_rate
dt = tr.stats.delta


# ==========================================================
# PREPROCESSING
# ==========================================================

print("Preprocessing...")

tr.detrend("demean")
tr.detrend("linear")
tr.taper(max_percentage=0.05)

if FMIN is not None and FMAX is not None:
    tr.filter(
        "bandpass",
        freqmin=FMIN,
        freqmax=FMAX,
        corners=4,
        zerophase=True
    )

data = tr.data.astype(np.float32)

print(
    f"Signal size in memory: "
    f"{data.nbytes/1024**2:.2f} MB"
)

# ==========================================================
# WINDOW DEFINITION
# ==========================================================

window_samples = int(WINDOW_HOURS * 3600 * fs)

print(
    f"Window length: {WINDOW_HOURS} h "
    f"({window_samples:,} samples)"
)

reconstructed_full = []

n_windows = int(np.ceil(len(data) / window_samples))

print(f"Number of windows: {n_windows}")


# ==========================================================
# CEEMDAN OBJECT
# ==========================================================

ceemdan = CEEMDAN(trials=TRIALS)
ceemdan.noise_seed(RANDOM_SEED)


# ==========================================================
# PROCESS WINDOWS
# ==========================================================

for i in range(n_windows):

    start = i * window_samples
    end = min((i + 1) * window_samples, len(data))

    chunk = data[start:end]

    if len(chunk) < 1000:
        continue

    print(
        f"Window {i+1}/{n_windows} "
        f"({len(chunk):,} samples)"
    )

    try:

        IMFs = ceemdan(chunk)

        n_imfs = IMFs.shape[0]

        reconstructed = np.zeros_like(chunk)

        for idx in IMFS_TO_KEEP:

            if idx < n_imfs:
                reconstructed += IMFs[idx]

        reconstructed_full.append(
            reconstructed.astype(np.float32)
        )

        # Optional IMF energy output
        total_energy = np.sum(chunk**2)

        print("IMF energies:")

        for j in range(n_imfs):

            energy = np.sum(IMFs[j]**2)

            frac = 100 * energy / total_energy

            print(
                f"   IMF {j+1:2d}: "
                f"{frac:6.2f}%"
            )

        del IMFs
        del reconstructed

        gc.collect()

    except Exception as e:

        print(
            f"Failed window {i+1}: {e}"
        )


# ==========================================================
# MERGE WINDOWS
# ==========================================================

print("Merging windows...")

reconstructed_full = np.concatenate(
    reconstructed_full
)

print(
    f"Final size: "
    f"{len(reconstructed_full):,}"
)

# ==========================================================
# SAVE MINISEED
# ==========================================================

tr_out = tr.copy()

tr_out.data = reconstructed_full.astype(
    np.float32
)

tr_out.write(
    OUTPUT_FILE,
    format="MSEED"
)

print(
    f"Saved reconstructed signal to "
    f"{OUTPUT_FILE}"
)

# ==========================================================
# PLOT EXAMPLE SEGMENT
# ==========================================================

plot_samples = min(
    int(600 * fs),  # first 10 minutes
    len(reconstructed_full)
)

time = np.arange(plot_samples) / fs

plt.figure(figsize=(14,5))

plt.plot(
    time,
    data[:plot_samples],
    label="Original",
    alpha=0.5
)

plt.plot(
    time,
    reconstructed_full[:plot_samples],
    label="Reconstructed",
    linewidth=1
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.title(
    "First 10 minutes"
)

plt.tight_layout()

plt.savefig(
    "CEEMDAN_example.pdf",
    dpi=300
)

plt.show()

print("Done.")