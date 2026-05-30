import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import multiprocessing as mp
from pathlib import Path
from obspy import read
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset, z_detect, ar_pick

#TODO: fix this code

# ==========================================================
# CONFIG
# ==========================================================

MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250101000000_20250131235959.mseed"

NETWORK = "MN"
STATION = "SENA"

WATER_LEVEL = 20

# Frequency bands
BANDS = [
    (0.03, 0.1),
    (0.1, 1.0),
    (1.0, 3.0)
]

# STA/LTA parameters
STA = 5
LTA = 600

ON_THRESHOLD = 2.5
OFF_THRESHOLD = 1.5

# Chunk duration (seconds)
CHUNK_DURATION = 3600  # 1 hour

# Padding for stable response removal
PADDING = 600  # seconds

# ==========================================================
# PROCESS CHUNK
# ==========================================================

def process_chunk(args):

    tr_chunk, inventory, actual_start, actual_end = args

    try:

        tr_chunk = tr_chunk.copy()

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------

        tr_chunk.detrend("demean")
        tr_chunk.detrend("linear")

        # IMPORTANT:
        # no taper because you already verified
        # it hurts the SNR significantly

        # --------------------------------------------------
        # Remove instrument response
        # --------------------------------------------------

        tr_chunk.remove_response(
            inventory=inventory,
            output="VEL",
            water_level=WATER_LEVEL
        )

        # --------------------------------------------------
        # Trim padding away
        # --------------------------------------------------

        tr_chunk.trim(
            starttime=actual_start,
            endtime=actual_end
        )

        # --------------------------------------------------
        # Band filtering
        # --------------------------------------------------

        filtered_traces = []

        for fmin, fmax in BANDS:

            tr_band = tr_chunk.copy()

            tr_band.filter(
                "bandpass",
                freqmin=fmin,
                freqmax=fmax,
                corners=4,
                zerophase=True
            )

            filtered_traces.append(tr_band)

        tr_low  = filtered_traces[0]
        tr_mid  = filtered_traces[1]
        tr_high = filtered_traces[2]

        # --------------------------------------------------
        # Detection uses mid-band
        # --------------------------------------------------

        tr_detect = tr_mid

        df = tr_detect.stats.sampling_rate

        # ==================================================
        # STA/LTA
        # ==================================================

        cft_sta = recursive_sta_lta(
            tr_detect.data,
            int(STA * df),
            int(LTA * df)
        )

        triggers_sta = trigger_onset(
            cft_sta,
            ON_THRESHOLD,
            OFF_THRESHOLD
        )

        sta_times = []

        for trig in triggers_sta:

            sample = trig[0]

            sta_times.append(
                tr_detect.stats.starttime + sample / df
            )

        # ==================================================
        # Z-DETECT
        # ==================================================

        z_window = 10

        cft_z = z_detect(
            tr_detect.data,
            int(z_window * df)
        )

        triggers_z = trigger_onset(
            cft_z,
            ON_THRESHOLD,
            OFF_THRESHOLD
        )

        z_times = []

        for trig in triggers_z:

            sample = trig[0]

            z_times.append(
                tr_detect.stats.starttime + sample / df
            )

        return {
            "trace": tr_detect,
            "sta_times": sta_times,
            "z_times": z_times,
            "cft_sta": cft_sta,
            "cft_z": cft_z
        }

    except Exception as e:

        print(f"Chunk failed: {e}")

        return None

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # Read waveform
    # ------------------------------------------------------

    print("Reading waveform...")

    st = read(MSEED_FILE)

    print("\nOriginal stream:")
    print(st)

    # ------------------------------------------------------
    # Sort traces chronologically
    # ------------------------------------------------------

    st.sort()

    # ------------------------------------------------------
    # Merge traces
    # ------------------------------------------------------

    st.merge(
        method=1,
        fill_value=0
    )

    print("\nMerged stream:")
    print(st)

    # ------------------------------------------------------
    # Handle multiple remaining traces
    # ------------------------------------------------------

    if len(st) > 1:

        print("\nMultiple traces remain after merge.")
        print("Concatenating manually...")

        full_data = np.concatenate(
            [tr.data for tr in st]
        )

        tr = st[0].copy()

        tr.data = full_data

        tr.stats.npts = len(full_data)

    else:

        tr = st[0]

    # ------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------

    print("\nFinal trace:")
    print(tr)

    print("Starttime:", tr.stats.starttime)
    print("Endtime:", tr.stats.endtime)

    duration_days = (
        tr.stats.endtime - tr.stats.starttime
    ) / 86400

    print(f"Trace duration: {duration_days:.2f} days")

    # ------------------------------------------------------
    # Download inventory
    # ------------------------------------------------------

    print("\nDownloading station inventory...")

    client = Client("INGV")

    inv = client.get_stations(
        network=NETWORK,
        station=STATION,
        starttime=tr.stats.starttime,
        endtime=tr.stats.endtime,
        level="response"
    )

    # ======================================================
    # CREATE PADDED CHUNKS
    # ======================================================

    print("\nCreating padded chunks...")

    chunk_args = []

    t0 = tr.stats.starttime
    endtime = tr.stats.endtime

    while t0 < endtime:

        t1 = t0 + CHUNK_DURATION

        actual_start = t0
        actual_end   = min(t1, endtime)

        # --------------------------------------------------
        # Padded interval
        # --------------------------------------------------

        padded_start = max(
            tr.stats.starttime,
            actual_start - PADDING
        )

        padded_end = min(
            endtime,
            actual_end + PADDING
        )

        chunk = tr.slice(
            padded_start,
            padded_end
        ).copy()

        if len(chunk.data) > 0:

            chunk_args.append(
                (
                    chunk,
                    inv,
                    actual_start,
                    actual_end
                )
            )

        t0 = t1

    print(f"Number of chunks: {len(chunk_args)}")

    # ======================================================
    # MULTIPROCESSING
    # ======================================================

    nproc = max(1, mp.cpu_count() // 2)

    print(f"\nUsing {nproc} processes")

    with mp.Pool(processes=nproc) as pool:

        results = pool.map(
            process_chunk,
            chunk_args
        )

    # ------------------------------------------------------
    # Remove failed chunks
    # ------------------------------------------------------

    results = [r for r in results if r is not None]

    print(f"Processed chunks: {len(results)}")

    # ======================================================
    # MERGE DETECTIONS
    # ======================================================

    sta_all = []
    z_all = []

    for r in results:

        sta_all.extend(r["sta_times"])
        z_all.extend(r["z_times"])

    print(f"\nTotal STA/LTA triggers: {len(sta_all)}")
    print(f"Total Z-Detect triggers: {len(z_all)}")

    # ======================================================
    # PLOT FIRST CHUNK ONLY
    # ======================================================

    first = results[0]

    tr_plot = first["trace"]

    time_array = tr_plot.times("matplotlib")

    # ------------------------------------------------------
    # STA/LTA plot
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True
    )

    ax[0].plot(
        time_array,
        tr_plot.data * 1e9,
        linewidth=0.8
    )

    for pt in first["sta_times"]:

        ax[0].axvline(
            pt.datetime,
            color="r",
            linestyle="--",
            alpha=0.3
        )

    ax[0].set_ylabel("Velocity [nm/s]")

    ax[1].plot(
        time_array,
        first["cft_sta"],
        linewidth=0.8
    )

    ax[1].axhline(
        ON_THRESHOLD,
        color="g",
        linestyle="--"
    )

    ax[1].axhline(
        OFF_THRESHOLD,
        color="r",
        linestyle="--"
    )

    ax[1].set_ylabel("STA/LTA")

    ax[1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d %H:%M:%S")
    )

    fig.autofmt_xdate()

    plt.savefig(
        "STA_LTA_parallel.pdf",
        dpi=300,
        bbox_inches="tight"
    )

    # ======================================================
    # Z-DETECT plot
    # ======================================================

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True
    )

    ax[0].plot(
        time_array,
        tr_plot.data * 1e9,
        linewidth=0.8
    )

    for pt in first["z_times"]:

        ax[0].axvline(
            pt.datetime,
            color="r",
            linestyle="--",
            alpha=0.3
        )

    ax[0].set_ylabel("Velocity [nm/s]")

    ax[1].plot(
        time_array,
        first["cft_z"],
        linewidth=0.8
    )

    ax[1].axhline(
        ON_THRESHOLD,
        color="g",
        linestyle="--"
    )

    ax[1].axhline(
        OFF_THRESHOLD,
        color="r",
        linestyle="--"
    )

    ax[1].set_ylabel("Z-Detect")

    ax[1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d %H:%M:%S")
    )

    fig.autofmt_xdate()

    plt.savefig(
        "Z_DETECT_parallel.pdf",
        dpi=300,
        bbox_inches="tight"
    )

    print("\nDone.")