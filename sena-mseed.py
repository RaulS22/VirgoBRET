import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from pathlib import Path
from obspy import UTCDateTime

# Debugging
import gc
'''This module provides an interface to the optional garbage collector. 
It provides the ability to disable the collector, tune the collection frequency,
and set debugging options.'''


# =========================
# Paths
# =========================
input_dir = Path("/home/rauls/Desktop/GithubITA/VirgoBRET/SENA-mseed")
output_dir = Path("sena-ago21-filtered-fft-test")
output_dir.mkdir(exist_ok=True)

# =========================
# File
# =========================
et_file = input_dir / "eida_response_MN-SENA_20210801000000_20210831235959.mseed"
print(f"Processing: {et_file}")

start_date = UTCDateTime("2021-08-01")
end_date   = UTCDateTime("2021-08-31")

current_day = start_date

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

    st_day.sort(keys=["channel"])

    # Preprocess
    for tr in st_day:
        tr.detrend("linear")
        tr.detrend("demean")
        tr.resample(2.1) # do this before filtering to reduce size, the first test was done with 10.0Hz
        # do not use 1Hz at the resample, it is too close to the Nyquist frequency
        tr.data = tr.data.astype(np.float32)
        tr.filter("bandpass", freqmin=0.1, freqmax=1.0)

    # Select channels
    channels = ["HHE", "HHN", "HHZ"] # ["BHE", "BHN", "BHZ", "HHE", "HHN", "HHZ"]
    selected_traces = []

    for ch in channels:
        tr_sel = st_day.select(channel=ch)
        if len(tr_sel) > 0:
            selected_traces.append(tr_sel[0])

    if len(selected_traces) == 0:
        current_day = next_day
        continue

    # Plot
    fig, axes = plt.subplots(len(selected_traces), 1, figsize=(12, 8), sharex=True)

    if len(selected_traces) == 1:
        axes = [axes]

    for ax, tr in zip(axes, selected_traces):
        ax.plot(tr.times(), tr.data)
        ax.set_ylabel(tr.stats.channel)
        ax.grid()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{et_file.stem} - {current_day.date}")

    outfile = output_dir / f"{et_file.stem}_{current_day.date}.pdf"
    plt.savefig(outfile)
    plt.close(fig)

    print(f"Saved: {outfile}")

    del st_day
    del selected_traces
    gc.collect()

    current_day = next_day