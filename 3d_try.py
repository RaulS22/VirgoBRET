import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from obspy import read_events
from obspy.clients.fdsn import Client
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# Plot Style (LaTeX-quality)
# ==========================================================

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "figure.figsize": (8,5),
})

# ==========================================================
# Paths
# ==========================================================

qml_path = Path("20240101_20240430__1__INGV__QML/202404")
qml_files = sorted(qml_path.glob("*.qml"))

print("Total events:", len(qml_files))

# ==========================================================
# Spectral cube parameters
# ==========================================================

NDAYS = 30
NHOURS = 24
NFREQ = 128

spectra = np.zeros((NDAYS, NHOURS, NFREQ))
counts = np.zeros((NDAYS, NHOURS))

# ==========================================================
# Client
# ==========================================================

client = Client("INGV")

# ==========================================================
# Event Loop
# ==========================================================

for event_file in qml_files:

    print("Processing:", event_file.name)

    try:

        catalog = read_events(event_file)
        event = catalog[0]

        origin = event.preferred_origin() or event.origins[0]
        origin_time = origin.time

        day = origin_time.day - 1
        hour = origin_time.hour

        # --------------------------
        # Select first P pick
        # --------------------------

        p_picks = [p for p in event.picks if p.phase_hint == "P"]

        if len(p_picks) == 0:
            continue

        pick = p_picks[0]

        network = pick.waveform_id.network_code or "IV"
        station = pick.waveform_id.station_code or "CESX"
        channel = pick.waveform_id.channel_code or "HHZ"
        location = pick.waveform_id.location_code or " "

        arrival_time = pick.time

        # --------------------------
        # Download waveform
        # --------------------------

        st = client.get_waveforms(
            network="IV",
            station="CESX",
            location=" ",
            channel="HHE",
            starttime=arrival_time - 60,
            endtime=arrival_time + 120
        )

        tr = st[0]

        # --------------------------
        # Preprocess
        # --------------------------

        tr.detrend("linear")
        tr.taper(max_percentage=0.05)
        tr.filter("bandpass", freqmin=1.0, freqmax=20.0)

        tr_trimmed = tr.copy().trim(
            starttime=arrival_time - 5,
            endtime=arrival_time + 20
        )

        data = tr_trimmed.data
        dt = tr_trimmed.stats.delta

        # --------------------------
        # FFT
        # --------------------------

        spectrum = np.abs(np.fft.rfft(data))
        freq = np.fft.rfftfreq(len(data), dt)

        spectrum = spectrum[:NFREQ]

        spectra[day, hour] += spectrum
        counts[day, hour] += 1

    except Exception as e:

        print("Failed:", e)
        continue

# ==========================================================
# Average spectra
# ==========================================================

counts[counts == 0] = 1
spectra /= counts[:, :, None]

# ==========================================================
# Choose frequency band to visualize
# ==========================================================

freq_bin = 20

Z = spectra[:, :, freq_bin]

days = np.arange(1, NDAYS+1)
hours = np.arange(NHOURS)

D, H = np.meshgrid(days, hours)

# ==========================================================
# 3D Surface Plot
# ==========================================================

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    D,
    H,
    Z.T,
    cmap="viridis",
    edgecolor="none"
)

ax.set_xlabel("Day of Month")
ax.set_ylabel("Hour of Day")
ax.set_zlabel("Spectral Amplitude")

ax.set_title("3D Seismic Spectrogram (April)")

fig.colorbar(surf, shrink=0.6, label="Power")

# ==========================================================
# Save figure
# ==========================================================

# Directory for figures
fig_dir = Path("3d_plots")
fig_dir.mkdir(exist_ok=True)

outfile = fig_dir / "april_3D_spectral_cube.pdf"

plt.savefig(outfile, format="pdf", dpi=300, bbox_inches="tight")
plt.close()

print("\nFigure saved to:", outfile)

print("Saved:", outfile)
