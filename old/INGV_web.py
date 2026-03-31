import numpy as np
import matplotlib.pyplot as plt
from obspy import read_events
from obspy.clients.fdsn import Client
from pathlib import Path

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


'''
Read: 
https://data.ingv.it/dataset/1174#additional-metadata
https://data.ingv.it/docs/index_en.html
https://github.com/INGV/data-repository?tab=MIT-1-ov-file
https://terremoti.ingv.it/en/webservices_and_software
https://data.ingv.it/metadata/web_service_eng
'''