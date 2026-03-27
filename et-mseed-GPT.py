import numpy as np
from obspy import read
from pathlib import Path

# =========================
# Paths
# =========================
input_dir = Path("/home/rauls/Desktop/VirgoBRET/ET-mseed/")
output_dir = Path("ET-mseed-plots")
output_dir.mkdir(exist_ok=True)

# =========================
# Select file (single file case)
# =========================
et_file = input_dir / "eida_response_3M-FAB00_20241101000000_20251102000000.mseed"

# =========================
# Read data
# =========================
st = read(str(et_file))

# Optional: inspect content
print(st)

# =========================
# Process each trace
# =========================
for tr in st:
    tr_proc = tr.copy()

    # Minimal preprocessing (no bandpass)
    tr_proc.detrend("linear")
    tr_proc.detrend("demean")

    # =========================
    # Build filenames
    # =========================
    # Include channel name to avoid overwriting if multiple traces exist
    base_name = f"{et_file.stem}_{tr_proc.stats.channel}"

    outfile = output_dir / f"{base_name}.pdf"
    outfile_rel = output_dir / f"{base_name}_relative.pdf"

    # =========================
    # Save plots (ObsPy-native)
    # =========================
    tr_proc.plot(outfile=str(outfile))
    tr_proc.plot(type="relative", outfile=str(outfile_rel))

    print(f"Saved: {outfile}")
    print(f"Saved: {outfile_rel}")