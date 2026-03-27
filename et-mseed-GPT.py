import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from pathlib import Path



# =========================
# Paths
# =========================
input_dir = Path("/home/rauls/Desktop/VirgoBRET/ET-mseed/")
output_dir = Path("ET-mseed-plots-test")
output_dir.mkdir(exist_ok=True)

# =========================
# File
# =========================
et_file = input_dir / "eida_response_3M-FAB00_20241101000000_20251102000000.mseed"

# =========================
# Read
# =========================
st = read(str(et_file))

# Optional: sort by channel for consistency (EHE, EHN, EHZ)
st.sort(keys=["channel"])

print(st)

# =========================
# Preprocess
# =========================
for tr in st:
    tr.detrend("linear")
    tr.detrend("demean")

# =========================
# Select only 3 components (if available)
# =========================
channels = ["HHE", "HHN", "HHZ"]
selected_traces = []

for ch in channels:
    tr = st.select(channel=ch)
    if len(tr) > 0:
        selected_traces.append(tr[0])

# =========================
# Plot (3 stacked subplots)
# =========================
n = len(selected_traces)

fig, axes = plt.subplots(n, 1, figsize=(12, 8), sharex=True)

if n == 1:
    axes = [axes]  # ensure iterable

for ax, tr in zip(axes, selected_traces):
    t = np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts)

    ax.plot(t, tr.data)
    ax.set_ylabel(tr.stats.channel)
    ax.grid()

axes[-1].set_xlabel("Time (s)")

# Title
fig.suptitle(et_file.stem)

# =========================
# Save
# =========================
outfile = output_dir / f"{et_file.stem}_3components.pdf"

plt.savefig(outfile)
plt.close(fig)

print(f"Saved: {outfile}")