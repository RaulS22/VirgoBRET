import numpy as np
import matplotlib.pyplot as plt

from obspy import read
from scipy.signal import hilbert
from PyEMD import CEEMDAN #use pip install EMD-signal


# ==========================================================
# USER PARAMETERS
# ==========================================================

MSEED_FILE = "22-02-25-Raul.mseed"

# Set to None to disable filtering
FMIN = 0.01     # Hz
FMAX = 30.0     # Hz

# IMF indices to reconstruct
# Python indexing starts at 0
# Example: IMF3–IMF6 -> [2,3,4,5]
IMFS_TO_KEEP = [2, 3, 4, 5]

SAVE_RECONSTRUCTED = True
OUTPUT_FILE = "EMD_signal.mseed"


# ==========================================================
# READ SEISMIC DATA
# ==========================================================

print("Reading MiniSEED...")
st = read(MSEED_FILE)
print(st)
tr = st[0].copy()
fs = tr.stats.sampling_rate
dt = tr.stats.delta

print(f"Sampling rate: {fs:.3f} Hz")
print(f"NPTS: {tr.stats.npts}")

data = tr.data.astype(np.float64)
time = tr.times()


# ==========================================================
# PREPROCESSING
# ==========================================================

print("Preprocessing...")
tr.detrend("demean")
tr.detrend("linear")
tr.taper(max_percentage=0.05)

if FMIN is not None and FMAX is not None:
    tr.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=4, zerophase=True)
data = tr.data.astype(np.float64)


# ==========================================================
# CEEMDAN DECOMPOSITION
# ==========================================================

print("Running CEEMDAN decomposition...")
ceemdan = CEEMDAN()
# Optional tuning parameters
ceemdan.noise_seed(42)
IMFs = ceemdan(data)
n_imfs = IMFs.shape[0]
print(f"Number of IMFs: {n_imfs}")

# ==========================================================
# PLOT ORIGINAL SIGNAL + IMFs
# ==========================================================

fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(14, 2*(n_imfs + 1)), sharex=True)
axes[0].plot(time, data, linewidth=0.8)
axes[0].set_title("Original Signal")

for i in range(n_imfs):
    axes[i + 1].plot(time, IMFs[i], linewidth=0.8)
    axes[i + 1].set_ylabel(f"IMF {i+1}")
axes[-1].set_xlabel("Time (s)")

plt.tight_layout()
plt.show()


# ==========================================================
# RECONSTRUCTION
# ==========================================================

print("Reconstructing signal...")
reconstructed = np.zeros_like(data)
for idx in IMFS_TO_KEEP:
    if idx < n_imfs:
        reconstructed += IMFs[idx]

plt.figure(figsize=(14,5))
plt.plot(time, data, label="Original", alpha=0.5)
plt.plot(time, reconstructed, label="Reconstructed", linewidth=1)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Signal Reconstruction")
plt.show()


# ==========================================================
# HILBERT ANALYSIS
# ==========================================================

print("Computing instantaneous frequency...")

# Example: analyze first reconstructed IMF
imf_index = IMFS_TO_KEEP[0]
analytic_signal = hilbert(IMFs[imf_index])
amplitude = np.abs(analytic_signal)
phase = np.unwrap(np.angle(analytic_signal))
inst_freq = np.diff(phase) / (2*np.pi*dt)
time_freq = time[:-1]
fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True)
ax[0].plot(time, IMFs[imf_index])
ax[0].set_title(f"IMF {imf_index+1}")

ax[1].plot(time_freq, inst_freq)
ax[1].set_ylabel("Frequency (Hz)")
ax[1].set_xlabel("Time (s)")
ax[1].set_title("Instantaneous Frequency")

plt.tight_layout()
plt.show()


# ==========================================================
# SAVE RECONSTRUCTED TRACE
# ==========================================================

if SAVE_RECONSTRUCTED:
    tr_new = tr.copy()
    tr_new.data = reconstructed.astype(np.float32)
    tr_new.write(OUTPUT_FILE, format="MSEED")
    print(f"Saved reconstructed trace: {OUTPUT_FILE}")

# ==========================================================
# IMF ENERGY ANALYSIS
# ==========================================================

print("\nIMF energy distribution:\n")
total_energy = np.sum(data**2)
for i in range(n_imfs):
    energy = np.sum(IMFs[i]**2)
    percentage = (100 * energy / total_energy)
    print(f"IMF {i+1:2d}: {percentage:6.2f}%")
print("\nDone.")