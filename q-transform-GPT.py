import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_inventory
from gwpy.timeseries import TimeSeries

#TODO: Improve thios code based on qT-test.py

# ==========================================================
# FILES
# ==========================================================

mseed_file = "22-02-25-Raul.mseed"
inventory_file = "fdsn_station.xml"

# ==========================================================
# READ DATA
# ==========================================================

print("Reading waveform...")
st = read(mseed_file)
print("Reading inventory...")
inv = read_inventory(inventory_file)

tr_raw = st[0].copy()

print(f"Station: {tr_raw.id}")
print(f"Sampling rate: {tr_raw.stats.sampling_rate:.2f} Hz")
print(f"Duration: {tr_raw.stats.npts / tr_raw.stats.sampling_rate / 3600:.2f} hours")

# ==========================================================
# PREPROCESS RAW TRACE
# ==========================================================

tr_raw.detrend("demean")
tr_raw.detrend("linear")
tr_raw.taper(max_percentage=0.05)

# ==========================================================
# CREATE VELOCITY TRACE
# ==========================================================

print("Removing instrument response...")
tr_vel = tr_raw.copy()
tr_vel.remove_response(inventory=inv,output="VEL",pre_filt=(0.005, 0.01, 45.0, 50.0),water_level=60)
print("Response removed.")

# ==========================================================
# CONVERT TO GWPY TIMESERIES
# ==========================================================

ts_raw = TimeSeries(tr_raw.data,sample_rate=tr_raw.stats.sampling_rate,t0=tr_raw.stats.starttime.timestamp)
ts_vel = TimeSeries(tr_vel.data,sample_rate=tr_vel.stats.sampling_rate,t0=tr_vel.stats.starttime.timestamp)

# ==========================================================
# COMPUTE Q-TRANSFORMS
# ==========================================================

print("Computing Q-transform (raw)...")
q_raw = ts_raw.q_transform(frange=(0.1, 3),qrange=(4, 64),whiten=False)
print("Computing Q-transform (velocity)...")
q_vel = ts_vel.q_transform(frange=(0.1, 3),qrange=(4, 64),whiten=False)
print("Done.")

# ==========================================================
# EXTRACT ARRAYS (OPTIONAL)
# ==========================================================

power_raw = q_raw.value
power_vel = q_vel.value
times_raw = q_raw.times.value
times_vel = q_vel.times.value
freqs_raw = q_raw.frequencies.value
freqs_vel = q_vel.frequencies.value
print("Raw Q-transform shape:", power_raw.shape)
print("Velocity Q-transform shape:", power_vel.shape)

# ==========================================================
# PLOT RAW Q-TRANSFORM
# ==========================================================

fig1 = q_raw.plot()
ax1 = fig1.gca()
ax1.set_title("Q-transform - Raw Data")
ax1.set_ylabel("Frequency [Hz]")
ax1.set_yscale("log")
plt.savefig("qT-raw.pdf")

# ==========================================================
# PLOT VELOCITY Q-TRANSFORM
# ==========================================================

fig2 = q_vel.plot()
ax2 = fig2.gca()
ax2.set_title("Q-transform - Velocity")
ax2.set_ylabel("Frequency [Hz]")
ax2.set_yscale("log")
plt.savefig("qT-velocity.pdf")

# ==========================================================
# COMPARISON FIGURE
# ==========================================================

fig, axes = plt.subplots(2,1,figsize=(14, 10),sharex=True)
im1 = axes[0].pcolormesh(times_raw, freqs_raw,power_raw.T, shading="auto")
axes[0].set_yscale("log")
axes[0].set_title("Raw Waveform Q-transform")
axes[0].set_ylabel("Frequency [Hz]")
plt.colorbar(im1, ax=axes[0], label="Energy")

im2 = axes[1].pcolormesh(times_vel,freqs_vel,power_vel.T,shading="auto")
axes[1].set_yscale("log")
axes[1].set_title("Velocity Q-transform")
axes[1].set_ylabel("Frequency [Hz]")
axes[1].set_xlabel("Time [s]")
plt.colorbar(im2, ax=axes[1], label="Energy")
plt.tight_layout()
plt.savefig("qT-comparison.pdf")