import numpy as np
import matplotlib.pyplot as plt
from obspy import read_events
from obspy.clients.fdsn import Client

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

##########################################################

"""
Italian Seismic Bulletin, January-April 2024
https://terremoti.ingv.it/en/bsi?id=10.13127/BSI/202401

HYPO71 documentation: https://pubs.usgs.gov/publication/ofr72224
"""

# Path

path = "/home/rauls/Desktop/VirgoBRET/20240101_20240430__1__INGV__QML"
jan = path + "/202401/"
print(jan)

event_name = jan + "20240101-013511__37256531__INGV-EVENT.qml"

# Read

catalog = read_events(event_name)
event = catalog[0]

origin = event.preferred_origin() or event.origins[0]
origin_time = origin.time

print("Event time:", origin_time)

# Pick P

p_picks = [p for p in event.picks if p.phase_hint == "P"]

if len(p_picks) == 0:
    raise ValueError("Nenhum pick P encontrado.")

pick = p_picks[0]

network = pick.waveform_id.network_code
station = pick.waveform_id.station_code
channel = pick.waveform_id.channel_code or "HHZ"
location = pick.waveform_id.location_code or "*"

arrival_time = pick.time

print(f"Using station: {network}.{station}.{channel}")
print("Arrival time:", arrival_time)

# dados do próprio .qml (Ctrl+F "waveformID")

client = Client("INGV")

st = client.get_waveforms(
    network="IV",
    station="CESX",
    location=" ",
    channel="HHE",
    starttime=arrival_time - 60,
    endtime=arrival_time + 120
)

tr = st[0]

# Pré-processamento

tr.detrend("linear")
tr.taper(max_percentage=0.05)
tr.filter("bandpass", freqmin=1.0, freqmax=20.0)

# Recorte para análise espectral


tr_trimmed = tr.copy().trim(
    starttime=arrival_time - 5,
    endtime=arrival_time + 20
)

# FFT

data = tr_trimmed.data
dt = tr_trimmed.stats.delta

freq = np.fft.rfftfreq(len(data), dt)
spectrum = np.abs(np.fft.rfft(data))

# Plot

fig, ax = plt.subplots()
ax.plot(freq, spectrum, color="black")
ax.set_xlabel(r"Frequency (Hz)")
ax.set_ylabel(r"Amplitude")
ax.set_title(r"Amplitude Spectrum")
ax.set_xlim(0, 30)
ax.grid(True)

plt.show()

# Spectrogram

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
ax.set_title(r"Spectrogram of event 20240101-013511__37256531")
ax.set_ylim(0, 30)
cbar = fig.colorbar(im)
cbar.set_label("Power")
plt.show()
