from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from obspy import read_events
from obspy.clients.fdsn import Client

# ==========================================================
# Global Style for LaTeX-quality figures
# ==========================================================

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    
    # Font sizes
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

    # Figure size (good for LaTeX)
    "figure.figsize": (7, 4.5),

    # Lines
    "lines.linewidth": 2,

    # Grid
    "grid.alpha": 0.35,
    "grid.linestyle": "--",

    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,

    # Layout
    "figure.autolayout": True
})

##########################################################

"""
Italian Seismic Bulletin, January-April 2024
https://terremoti.ingv.it/en/bsi?id=10.13127/BSI/202401

Battelli, P., Berardi, M., Marchetti, A., Misiti, V., Modica, G., Nardi, A., … 
Malagnini, A. (2025). Bollettino Sismico Italiano (BSI), I quadrimestre 2024 (Version 1)
[Data set]. Istituto Nazionale di Geofisica e Vulcanologia (INGV). https://doi.org/10.13127/BSI/202401
"""


# Paths

qml_path = Path("20240101_20240430__1__INGV__QML/202401/")
output_dir = Path("spectrograms")
output_dir.mkdir(exist_ok=True)

qml_files = sorted(qml_path.glob("*.qml"))

print("Total January events:", len(qml_files))


# Client

client = Client("INGV") 

# Loop over events

for event_file in qml_files:

    print("\nProcessing:", event_file.name)

    try:

        catalog = read_events(event_file)
        event = catalog[0]

        origin = event.preferred_origin() or event.origins[0]
        origin_time = origin.time

        # -----------------------------
        # Select P pick
        # -----------------------------

        p_picks = [p for p in event.picks if p.phase_hint == "P"]

        if len(p_picks) == 0:
            print("No P pick found")
            continue

        pick = p_picks[0]

        network = pick.waveform_id.network_code or "IV"
        station = pick.waveform_id.station_code or "CESX"
        channel = pick.waveform_id.channel_code or "HHZ"
        location = pick.waveform_id.location_code or " "

        arrival_time = pick.time

        print(f"Station: {network}.{station}.{channel}")

        # -----------------------------
        # Download waveform
        # -----------------------------

        st = client.get_waveforms(
            network="IV",
            station="CESX",
            location=" ",
            channel="HHE",
            starttime=arrival_time - 60,
            endtime=arrival_time + 120
        )

        tr = st[0]

        # -----------------------------
        # Preprocessing
        # -----------------------------

        tr.detrend("linear")
        tr.taper(max_percentage=0.05)
        #tr.filter("bandpass", freqmin=1.0, freqmax=20.0)

        # -----------------------------
        # Window around P arrival
        # -----------------------------

        tr_trimmed = tr.copy().trim(
            starttime=arrival_time - 60, # -5,
            endtime=arrival_time + 120 # +20
        )

        data = tr_trimmed.data
        dt = tr_trimmed.stats.delta

        # -----------------------------
        # Spectrogram
        # -----------------------------

        fig, ax = plt.subplots()

        Pxx, freqs, bins, im = ax.specgram(
            data,
            NFFT=256,
            Fs=1/dt,
            noverlap=200,
            cmap="viridis"
        )

        ax.set_xlabel(r"Time (s)")
        ax.set_ylabel(r"Frequency (Hz)")
        ax.set_title(event_file.stem)

        ax.set_ylim(0, 30)

        cbar = fig.colorbar(im)
        cbar.set_label("Power")

        # -----------------------------
        # Save figure
        # -----------------------------

        outfile = output_dir / f"{event_file.stem}.pdf"
        plt.savefig(outfile)
        plt.close()

        print("Saved:", outfile)

    except Exception as e:

        print("Failed:", e)
        continue

print("\nFinished processing January events.")