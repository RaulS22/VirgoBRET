import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from pathlib import Path
from gwpy.timeseries import TimeSeries

#TODO: Understand the qTransform
#TODO: Check if the peals are actually peaks

# ==========================================================
# USER INPUTS
# ==========================================================

MSEED_FILE = "14-08-25-Fabi.mseed"

WINDOWS = [1, 2, 5, 10, 20, 30, 40]      # seconds on each side
FRANGE = (3, 30)
QRANGE = (8, 64)

CENTER_ON_PEAK = True
PEAK_SEARCH_WINDOW = 1    # seconds

PLOT_RESULTS = True

OUTPUT_DIR = Path("qTransform")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# READ DATA
# ==========================================================
st = read(MSEED_FILE)
tr = st[0]
starttime = tr.stats.starttime
endtime = tr.stats.endtime

# # ==========================================================
# # DOWNLOAD RESPONSE
# # ==========================================================
# client = Client("INGV")
# inv = client.get_stations(
#     network="MN",
#     station="SENA",
#     starttime=starttime,
#     endtime=endtime,
#     level="response"
# )
# 
# # ==========================================================
# # PREPROCESS
# # ==========================================================
# 
# tr.detrend("demean")
# tr.detrend("linear")
# tr.remove_response(inventory=inv,output="VEL",water_level=20)

# ==========================================================
# STA/LTA
# ==========================================================

sta = 5
lta = 600

df = tr.stats.sampling_rate
tr_original = tr.copy()

fmin = FRANGE[0]
fmax = FRANGE[1]

tr_band = tr_original.copy()
tr_band.filter("bandpass", freqmin=fmin, freqmax=fmax)
df = tr.stats.sampling_rate

cft = recursive_sta_lta(tr_band.data, int(sta * df), int(lta * df))
on_threshold = 5.0
off_threshold = 1.5

triggers = trigger_onset(cft, on_threshold, off_threshold)
print(f"\nNumber of triggers: {len(triggers)}")

# ==========================================================
# CONVERT TRIGGERS TO UTCDateTime
# ==========================================================

trigger_times = []

for onset, offset in triggers:
    trigger_time = (tr.stats.starttime +onset / tr.stats.sampling_rate)
    trigger_times.append(trigger_time)
    #print(f"Trigger: {trigger_time} (sample {onset})")

if len(trigger_times) == 0:
    print("No triggers found.")
    raise SystemExit

# ==========================================================
# Q-TRANSFORM ANALYSIS
# ==========================================================

results = []
for i, trigger_time in enumerate(trigger_times):
    center_time = trigger_time

    # ------------------------------------------------------
    # Optional peak centering
    # ------------------------------------------------------
    if CENTER_ON_PEAK:
        search_start = trigger_time - PEAK_SEARCH_WINDOW
        search_end = trigger_time + PEAK_SEARCH_WINDOW

        if (search_start >= tr.stats.starttime and search_end <= tr.stats.endtime):
            search_trace = tr.slice(starttime=search_start, endtime=search_end)
            if len(search_trace.data) > 0:
                imax = np.argmax(np.abs(search_trace.data))
                center_time = (search_trace.stats.starttime + imax / search_trace.stats.sampling_rate)

    # ======================================================
    # LOOP OVER WINDOW SIZES
    # ======================================================

    for half_width in WINDOWS:
        total_duration = 2 * half_width
        min_freq = FRANGE[0]
        if total_duration < 4.0 / min_freq:
            continue

        # --------------------------------------------------
        # Build perfectly symmetric window
        # --------------------------------------------------

        df = tr.stats.sampling_rate
        center_idx = int((center_time - tr.stats.starttime)* df)
        nwin = int(half_width * df)
        i0 = center_idx - nwin
        i1 = center_idx + nwin

        if i0 < 0 or i1 >= tr.stats.npts:
            continue

        data_event = tr.data[i0:i1]

        if len(data_event) == 0:
            continue

        # --------------------------------------------------
        # Relative time:
        # trigger is exactly t = 0
        # --------------------------------------------------

        ts = TimeSeries(data_event.astype(np.float64),sample_rate=df,t0=0)

        # --------------------------------------------------
        # Q-transform
        # --------------------------------------------------

        try:
            qspec = ts.q_transform(frange=FRANGE,qrange=QRANGE,whiten=True)
            qspec.xindex = qspec.xindex.value - half_width

        except Exception as e:
            print(f"Q-transform failed:\n{e}")
            continue

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        power = qspec.value
        peak_energy = np.nanmax(power)
        mean_energy = np.nanmean(power)

        results.append({

            "trigger_time": trigger_time,
            "center_time": center_time,
            "half_width": half_width,
            "peak_energy": peak_energy,
            "mean_energy": mean_energy

        })

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        if PLOT_RESULTS:
            fig = qspec.plot()
            ax = fig.gca()
            ax.set_title("Q-transform\n" f"Trigger = {center_time}\n" f"Window = ±{half_width} s")
            ax.set_xlabel("Time relative to trigger [s]")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_yscale("log")
            ax.set_xlim(-half_width,half_width)
            ax.axvline(0,color="red",linestyle="--",linewidth=1.5,alpha=0.8)
            mappable = ax.collections[0]
            cbar = fig.colorbar(mappable, ax=ax)
            cbar.set_label("Q-transform intensity")
            filename = (f"trigger_{i:04d}_window_{half_width}s.pdf")
            fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for r in results:
    print(f"Trigger={r['trigger_time']} | Window=±{r['half_width']}s | Peak={r['peak_energy']:.4e} | Mean={r['mean_energy']:.4e}")