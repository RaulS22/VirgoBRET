import numpy as np
import matplotlib.pyplot as plt

from obspy import read
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from pathlib import Path
from gwpy.timeseries import TimeSeries

# ==========================================================
# USER PARAMETERS
# ==========================================================

MSEED_FILE = "14-08-25-Fabi.mseed"

# Window half-widths (seconds)
WINDOWS = [1, 2, 5, 10, 20, 30, 40]

# Frequency and Q ranges for Q-transform
FRANGE = (5, 30)
QRANGE = (4, 10)

Fmin, Fmax = FRANGE[0], FRANGE[1]

# STA/LTA parameters
STA = 5      # seconds
LTA = 600    # seconds

ON_THRESHOLD = 5.0
OFF_THRESHOLD = 1.5

# Peak recentering
CENTER_ON_PEAK = True
PEAK_SEARCH_WINDOW = 5     # seconds

# Minimum samples required by gwpy q_transform
MINIMUM_SAMPLES = 1200

# Preprocessing
DETREND = True
WHITEN = True

# Plot
PLOT_RESULTS = True

OUTPUT_DIR = Path("qTransform")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# READ DATA
# ==========================================================

print("Reading waveform...")

st = read(MSEED_FILE)
tr = st[0].copy()
tr = tr.resample(sampling_rate=2*Fmax)
df = tr.stats.sampling_rate

print(f"Sampling rate: {df:.2f} Hz")
print(f"Trace duration: {(tr.stats.endtime-tr.stats.starttime)/3600:.2f} hours")

# ==========================================================
# PREPROCESS
# ==========================================================

if DETREND:
    tr.detrend("demean")
    tr.detrend("linear")

# Create bandpassed copy for STA/LTA
tr_band = tr.copy()

tr_band.filter(
    "bandpass",
    freqmin=FRANGE[0],
    freqmax=FRANGE[1],
    corners=4,
    zerophase=True
)

# ==========================================================
# STA/LTA TRIGGER
# ==========================================================

print("Running STA/LTA...")

sta_samples = int(STA*df)
lta_samples = int(LTA*df)

cft = recursive_sta_lta(
    tr_band.data,
    sta_samples,
    lta_samples
)

triggers = trigger_onset(
    cft,
    ON_THRESHOLD,
    OFF_THRESHOLD
)

print(f"Number of triggers: {len(triggers)}")

if len(triggers) == 0:
    raise SystemExit("No triggers found.")

# ==========================================================
# CONVERT TO TIMES
# ==========================================================

trigger_times = []

for onset, offset in triggers:

    t = tr.stats.starttime + onset/df
    trigger_times.append(t)

# ==========================================================
# MAIN Q-TRANSFORM LOOP
# ==========================================================

results = []

for itrigger, trigger_time in enumerate(trigger_times):

    center_time = trigger_time

    # ------------------------------------------------------
    # Optional recentering around local peak
    # ------------------------------------------------------

    if CENTER_ON_PEAK:

        search_start = trigger_time - PEAK_SEARCH_WINDOW
        search_end   = trigger_time + PEAK_SEARCH_WINDOW

        if (
            search_start >= tr.stats.starttime
            and
            search_end <= tr.stats.endtime
        ):

            search_trace = tr.slice(
                starttime=search_start,
                endtime=search_end
            )

            if len(search_trace.data) > 0:

                peak_index = np.argmax(
                    np.abs(search_trace.data)
                )

                center_time = (
                    search_trace.stats.starttime
                    +
                    peak_index/df
                )

    # ======================================================
    # LOOP OVER WINDOWS
    # ======================================================

    for half_width in WINDOWS:

        print(
            f"\nTrigger {itrigger+1}/{len(trigger_times)}"
            f" | Window ±{half_width}s"
        )

        # --------------------------------------------------
        # Number of samples in window
        # --------------------------------------------------

        npts = int(2*half_width*df)

        if npts < MINIMUM_SAMPLES:

            print(
                f"Skipping: "
                f"N={npts} < {MINIMUM_SAMPLES}"
            )

            continue

        # --------------------------------------------------
        # Build symmetric window
        # --------------------------------------------------

        center_index = int(
            (center_time-tr.stats.starttime)*df
        )

        nwin = int(
            half_width*df
        )

        i0 = center_index-nwin
        i1 = center_index+nwin

        if i0 < 0:
            continue

        if i1 >= tr.stats.npts:
            continue

        data = tr.data[i0:i1]

        if len(data)==0:
            continue

        # --------------------------------------------------
        # Create TimeSeries
        # --------------------------------------------------

        ts = TimeSeries(
            data.astype(np.float64),
            sample_rate=df,
            t0=0
        )

        # --------------------------------------------------
        # Q-transform
        # --------------------------------------------------

        try:

            qspec = ts.q_transform(
                frange=FRANGE,
                qrange=QRANGE,
                whiten=WHITEN
            )

            qspec.xindex = (
                qspec.xindex.value-half_width
            )

        except Exception as e:

            print("\nQ-transform failed:")
            print(e)

            continue

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        power = qspec.value

        peak_energy = np.nanmax(power)
        mean_energy = np.nanmean(power)

        results.append({

            "trigger_time":trigger_time,
            "center_time":center_time,
            "window":half_width,
            "peak":peak_energy,
            "mean":mean_energy

        })

        print(
            f"Peak={peak_energy:.3e}"
            f" Mean={mean_energy:.3e}"
        )

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        if PLOT_RESULTS:

            fig = qspec.plot()

            ax = fig.gca()

            ax.set_title(
                f"Q-transform\n"
                f"Trigger={center_time}\n"
                f"Window=±{half_width}s"
            )

            ax.set_xlabel(
                "Time relative to trigger [s]"
            )

            ax.set_ylabel(
                "Frequency [Hz]"
            )

            ax.set_yscale("log")

            ax.set_xlim(
                -half_width,
                half_width
            )

            ax.axvline(
                0,
                linestyle="--",
                linewidth=1.5
            )

            mappable = ax.collections[0]

            cbar = fig.colorbar(
                mappable,
                ax=ax
            )

            cbar.set_label(
                "Q-transform intensity"
            )

            filename = (
                f"trigger_{itrigger:04d}"
                f"_window_{half_width}s.pdf"
            )

            fig.savefig(
                OUTPUT_DIR/filename,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

# ==========================================================
# SUMMARY
# ==========================================================

print("\n")
print("="*60)
print("SUMMARY")
print("="*60)

for r in results:

    print(
        f"Trigger={r['trigger_time']} "
        f"| Window=±{r['window']}s "
        f"| Peak={r['peak']:.3e} "
        f"| Mean={r['mean']:.3e}"
    )