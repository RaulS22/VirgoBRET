import os

# ============================================================
# OPENMP / BLAS THREADING
# ============================================================

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy.clients.fdsn import Client
from obspy import read, read_inventory, UTCDateTime

# ============================================================
# PATHS
# ============================================================

here = "/home/rauls/Desktop/VirgoBRET/Paralel"

mseed_file = (
    "SENA-files/2021/"
    "eida_response_MN-SENA_20211201000000_20211231235959.mseed"
)

xml_file = "fdsn_station.xml"

# ============================================================
# LOAD DATA
# ============================================================

print("Reading MiniSEED file...")

st = read(mseed_file)

tr = st[0]

print(tr)

# ============================================================
# LOAD RESPONSE
# ============================================================

client = Client("INGV")

inv = client.get_stations(
    network="MN",
    station="SENA",
    starttime=UTCDateTime("2022-12-01"),
    endtime=UTCDateTime("2022-12-31"),
    level="response"
)

inv.write(
    xml_file,
    format="STATIONXML",
    validate=True
)
print(f"StationXML saved to: {xml_file}")

'''
try:

    print("Loading local StationXML...")

    inv = read_inventory(xml_file)

except:

    print("Downloading StationXML from INGV...")

    client = Client("INGV")

    inv = client.get_stations(
        network="MN",
        station="SENA",
        starttime=UTCDateTime("2021-06-21"),
        endtime=UTCDateTime("2025-12-31"),
        level="response"
    )

    inv.write(
        xml_file,
        format="STATIONXML",
        validate=True
    )

    print(f"StationXML saved to: {xml_file}")
'''

# ============================================================
# OPTIONAL DECIMATION
# ============================================================

print("Original sampling rate:", tr.stats.sampling_rate)

if tr.stats.sampling_rate > 50:

    tr.decimate(2)

print("New sampling rate:", tr.stats.sampling_rate)

# ============================================================
# RESPONSE REMOVAL
# ============================================================

print("\nRemoving instrument response...\n")

tr.remove_response(
    inventory=inv,
    output="VEL",
    pre_filt=[0.001, 0.005, 40, 60],
    water_level=60
)

print("Response removal completed.")

# ============================================================
# CONVERT TO NUMPY ARRAY
# ============================================================

velocity = tr.data.astype(np.float32)

fs = tr.stats.sampling_rate

dt = tr.stats.delta

starttime = tr.stats.starttime.datetime

print("\nArray information:")
print("Samples:", len(velocity))
print("Sampling rate:", fs)

# ============================================================
# MATRIX CONSTRUCTION
# ============================================================

chunk_duration = 600  # seconds

samples_per_window = int(fs * chunk_duration)

n_windows = len(velocity) // samples_per_window

print("\nBuilding matrix...")

velocity = velocity[:n_windows * samples_per_window]

matrix = velocity.reshape(
    n_windows,
    samples_per_window
)

print("Matrix shape:", matrix.shape)

# ============================================================
# DETREND MATRIX
# ============================================================

print("\nDetrending windows...")

matrix = matrix - np.mean(
    matrix,
    axis=1,
    keepdims=True
)

# ============================================================
# RMS OF EACH WINDOW
# ============================================================

print("Computing RMS...")

rms = np.sqrt(
    np.mean(
        matrix**2,
        axis=1
    )
)

# ============================================================
# FFT OF ALL WINDOWS
# ============================================================

print("Computing FFTs...")

spectra = np.fft.rfft(
    matrix,
    axis=1
)

# ============================================================
# PSD MATRIX
# ============================================================

print("Computing PSD matrix...")

psd = np.abs(spectra)**2

# ============================================================
# FREQUENCY AXIS
# ============================================================

freqs = np.fft.rfftfreq(
    samples_per_window,
    d=dt
)

# ============================================================
# BLRMS BAND
# ============================================================

fmin = 0.1
fmax = 1.0

mask = (
    (freqs >= fmin) &
    (freqs <= fmax)
)

# ============================================================
# BLRMS COMPUTATION
# ============================================================

print("Computing BLRMS...")

blrms = np.sqrt(
    np.mean(
        psd[:, mask],
        axis=1
    )
)

# ============================================================
# TIME AXIS FOR WINDOWS
# ============================================================

window_times = np.array([
    starttime + np.timedelta64(
        int(i * chunk_duration),
        's'
    )
    for i in range(n_windows)
])

# ============================================================
# PLOT RMS
# ============================================================

print("Generating RMS plot...")
plt.figure(figsize=(18, 5))

plt.plot(
    window_times,
    rms * 1e9,
    linewidth=1
)

plt.title("Window RMS Ground Velocity")

plt.xlabel("UTC Date")
plt.ylabel("RMS Velocity (nm/s)")

plt.grid(True)

ax = plt.gca()

ax.xaxis.set_major_locator(
    mdates.DayLocator(interval=2)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y-%m-%d')
)

plt.xticks(rotation=45)

plt.tight_layout()

output_file = (
    f"{here}/rms_matrix.pdf"
)

plt.savefig(output_file, dpi=300)

# ============================================================
# PLOT BLRMS
# ============================================================

print("Generating BLRMS plot...")

plt.figure(figsize=(18, 5))

plt.plot(
    window_times,
    blrms,
    linewidth=1
)

plt.title(
    f"BLRMS ({fmin}-{fmax} Hz)"
)

plt.xlabel("UTC Date")
plt.ylabel("BLRMS")

plt.grid(True)

ax = plt.gca()

ax.xaxis.set_major_locator(
    mdates.DayLocator(interval=2)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y-%m-%d')
)

plt.xticks(rotation=45)
plt.tight_layout()

output_file = (
    f"{here}/blrms_matrix.pdf"
)
plt.savefig(output_file, dpi=300)
print("\nFinished successfully.")