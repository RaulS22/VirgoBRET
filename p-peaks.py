'''
This method is the first step to implement an analysis of the SNR. 

The logical structural of this code will be similar to the one developed at: 
https://quakecoresoft.canterbury.ac.nz/docs/nzgmdb.calculation.snr.html in 
which we have a structure like Waveform + P-arrival -> SNR(f)
that basically transforms raw waveform data into:
FAS (Fourier Amplitude Spectrum)
SNR (Signal-to-Noise Ratio) as a function of frequency

########

Inputs and outputs (class interface):

Inputs
Waveform files (.mseed)
Phase arrival table (contains tp, the P-wave index)
Station metadata (Inventory)

Outputs
snr_fas.csv per event
Skipped-records log (diagnostics)

The output is frequency-dependent:
SNR(f)
FAS_signal(f), FAS_noise(f)
'''

import numpy as np
from obspy import read, read_inventory, Trace
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from concurrent.futures import ProcessPoolExecutor
import os

# ==========================================================
#   GLOBAL (per-process) inventory
# ==========================================================
_GLOBAL_INV = None


def _init_worker(inventory_file):
    global _GLOBAL_INV
    _GLOBAL_INV = read_inventory(inventory_file)


# ==========================================================
#   Core signal processing
# ==========================================================
def preprocess_trace_inplace(tr):
    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(max_percentage=0.05)

    tr.remove_response(
        inventory=_GLOBAL_INV,
        output="VEL",
        water_level=60
    )
    return tr


def detect_picks_in_trace(tr, window_sec):

    tr = preprocess_trace_inplace(tr)

    df = tr.stats.sampling_rate
    npts_window = int(window_sec * df)

    data = tr.data
    starttime = tr.stats.starttime

    picks = []

    # Precompute STA/LTA window sizes
    sta_n = int(1.0 * df)
    lta_n = int(10.0 * df)

    for i in range(0, len(data) - npts_window, npts_window):

        # 👉 zero-copy slice (no Trace.copy)
        segment_data = data[i:i + npts_window]

        cft = classic_sta_lta(segment_data, sta_n, lta_n)
        triggers = trigger_onset(cft, 3.5, 1.0)

        if len(triggers) == 0:
            continue

        idx = triggers[0][0]
        tp = starttime + (i + idx) / df

        picks.append({
            "station": tr.stats.station,
            "channel": tr.stats.channel,
            "time": tp
        })

    return picks


# ==========================================================
#   Worker
# ==========================================================
def _process_trace_worker(args):
    tr, window_sec = args
    return detect_picks_in_trace(tr, window_sec)


# ==========================================================
#   Main parallel driver
# ==========================================================
def process_month_optimized(
    mseed_file,
    inventory_file,
    window_sec=20,
    nproc=None
):

    if nproc is None:
        nproc = max(1, os.cpu_count() - 1)

    st = read(mseed_file)

    args = [(tr, window_sec) for tr in st]

    all_picks = []

    with ProcessPoolExecutor(
        max_workers=nproc,
        initializer=_init_worker,
        initargs=(inventory_file,)
    ) as executor:

        for result in executor.map(_process_trace_worker, args, chunksize=1):
            all_picks.extend(result)

    return all_picks


# ==========================================================
#   RUN
# ==========================================================
if __name__ == "__main__":

    mseed_file = "/home/rauls/Desktop/VirgoBRET/SENA-files/2022/eida_response_MN-SENA_20220101000000_20220131235959.mseed"

    picks = process_month_optimized(
        mseed_file,
        "fdsn_station.xml",
        window_sec=20,
        nproc=6
    )

    print(f"Detected {len(picks)} P arrivals")

    for p in picks[:5]:
        print(p)