import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_inventory, UTCDateTime
from pathlib import Path
import gc

# =========================
# Paths
# =========================
input_dir = Path("/home/rauls/Desktop/VirgoBRET/SENA-mseed")
output_dir = Path("sena-jan24-velocity")
output_dir.mkdir(exist_ok=True)

inv = read_inventory("fdsn_station.xml")

