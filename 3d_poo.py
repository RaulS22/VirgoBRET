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

"""
Início: path 
Para o mês: # dias

Infrmações sobre o canal da Itália:
    network = pick.waveform_id.network_code or "IV"
    station = pick.waveform_id.station_code or "CESX"
    channel = pick.waveform_id.channel_code or "HHZ"
    location = pick.waveform_id.location_code or " "

    

Pecisa selecionar uma banda:
    freq_bin = 20

    Z = spectra[:, :, freq_bin]

    days = np.arange(1, NDAYS+1)
    hours = np.arange(NHOURS)

    D, H = np.meshgrid(days, hours)

Plot, algumas opções:
    - Salvar o PDF 3d
    - Visualizar o gráfico (paia porque se esquecer de salvar demora)
    - Salvar os PDFs de cada evento solto
"""