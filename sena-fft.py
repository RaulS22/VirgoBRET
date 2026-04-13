import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime, read_inventory
from pathlib import Path
import gc

# =========================
# Paths
# =========================
input_dir = Path("/home/rauls/Desktop/VirgoBRET/SENA-mseed")
output_dir = Path("sena-jan24-spectrogram-100Hz")
output_dir.mkdir(exist_ok=True)

et_file = input_dir / "eida_response_MN-SENA_20240101000000_20240131000000.mseed"
print(f"Processing: {et_file}")

inv = read_inventory("fdsn_station.xml")

# =========================
# Time window
# =========================
start_date = UTCDateTime("2024-01-01")
end_date   = UTCDateTime("2024-01-31")

current_day = start_date

# =========================
# STFT parameters
# =========================
window_length_sec = 600        # 10 minutes
overlap_fraction  = 0.5

# =========================
# Main loop
# =========================
while current_day < end_date:
    next_day = current_day + 86400
    print(f"Processing day: {current_day.date}")

    st_day = read(
        str(et_file),
        starttime=current_day,
        endtime=next_day
    )

    if len(st_day) == 0:
        current_day = next_day
        continue

    # =========================
    # Preprocessing
    # =========================
    for tr in st_day:
        tr.detrend("linear")
        tr.detrend("demean")
        tr.resample(210.0)
        tr.data = tr.data.astype(np.float32)
        tr.filter("bandpass", freqmin=0.1, freqmax=100.0)

    # =========================
    # Select HHZ
    # =========================
    tr_sel = st_day.select(channel="HHZ")

    if len(tr_sel) == 0:
        del st_day
        gc.collect()
        current_day = next_day
        continue

    tr = tr_sel[0].copy()

    if not tr.stats.location:
        tr.stats.location = "00"

    data = tr.data
    fs = tr.stats.sampling_rate

    # =========================
    # Build STFT
    # =========================
    nperseg = int(window_length_sec * fs)
    step = int(nperseg * (1 - overlap_fraction))

    if len(data) < nperseg:
        print("Trace too short for spectrogram")
        del st_day, tr
        gc.collect()
        current_day = next_day
        continue

    window = np.hanning(nperseg)

    spectrogram = []
    times = []

    for start in range(0, len(data) - nperseg, step):
        segment = data[start:start + nperseg]

        # Demean each segment (important)
        segment = segment - np.mean(segment)

        # Apply window
        segment = segment * window

        # FFT
        fft_vals = np.fft.rfft(segment)

        # Amplitude → Power (better for spectrograms)
        power = np.abs(fft_vals) ** 2

        spectrogram.append(power)

        # Time axis (center of window)
        times.append((start + nperseg // 2) / fs)

    spectrogram = np.array(spectrogram).T  # shape: (freq, time)

    freqs = np.fft.rfftfreq(nperseg, d=1/fs)
    times = np.array(times) / 3600.0  # convert to hours

    # Convert to dB
    spectrogram_db = 10 * np.log10(spectrogram + 1e-20)

    # =========================
    # Plot
    # =========================
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.pcolormesh(
        times,
        freqs,
        spectrogram_db,
        shading="auto"
    )

    ax.set_yscale("log")
    ax.set_ylim(0.01, 101.0)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram (HHZ) - {current_day.date}")

    fig.colorbar(im, ax=ax, label="Power (dB)")

    # =========================
    # Save
    # =========================
    outfile = output_dir / f"{et_file.stem}_{current_day.date}_spectrogram_HHZ.pdf"
    plt.savefig(outfile)
    plt.close(fig)

    print(f"Saved: {outfile}")

    # Cleanup
    del st_day, tr, spectrogram
    gc.collect()

    current_day = next_day