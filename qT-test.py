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

MSEED_FILE = "22-02-25-Raul.mseed"

WINDOWS = [10, 20, 30]      # seconds on each side
FRANGE = (1, 20)
QRANGE = (4, 64)

CENTER_ON_PEAK = True
PEAK_SEARCH_WINDOW = 20    # seconds

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

# ==========================================================
# DOWNLOAD RESPONSE
# ==========================================================
client = Client("INGV")
inv = client.get_stations(
    network="MN",
    station="SENA",
    starttime=starttime,
    endtime=endtime,
    level="response"
)

# ==========================================================
# PREPROCESS
# ==========================================================

tr.detrend("demean")
tr.detrend("linear")
tr.remove_response(
    inventory=inv,
    output="VEL",
    water_level=20
)

# ==========================================================
# STA/LTA
# ==========================================================

sta = 5
lta = 600

df = tr.stats.sampling_rate

cft = recursive_sta_lta(
    tr.data,
    int(sta * df),
    int(lta * df)
)

on_threshold = 2.5
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
    #print(f"Trigger: {trigger_time} "f"(sample {onset})")

if len(trigger_times) == 0:
    print("No triggers found.")
    raise SystemExit

# ==========================================================
# Q-TRANSFORM ANALYSIS
# ==========================================================

results = []

for i, trigger_time in enumerate(trigger_times):
    #print("\n" + "=" * 60)
    #print(f"Trigger {i+1}/{len(trigger_times)}")
    #print(f"STA/LTA time: {trigger_time}")

    center_time = trigger_time

    # ------------------------------------------------------
    # Recenter on maximum amplitude
    # ------------------------------------------------------

    if CENTER_ON_PEAK:
        search_start = trigger_time - PEAK_SEARCH_WINDOW
        search_end = trigger_time + PEAK_SEARCH_WINDOW

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

                imax = np.argmax(
                    np.abs(search_trace.data)
                )

                center_time = (
                    search_trace.stats.starttime +
                    imax /
                    search_trace.stats.sampling_rate
                )

                print(
                    f"Peak-centered time: "
                    f"{center_time}"
                )

    # ------------------------------------------------------
    # Test different windows
    # ------------------------------------------------------

    for half_width in WINDOWS:
        total_duration = 2 * half_width
        #print(f"\nProcessing ±{half_width}s "f"(duration={total_duration}s)")

        # --------------------------------------------------
        # Ensure window is inside trace
        # --------------------------------------------------

        start_event = center_time - half_width
        end_event = center_time + half_width

        if (start_event < tr.stats.starttime or end_event > tr.stats.endtime):
            print("Window outside trace. Skipping.")
            continue

        # --------------------------------------------------
        # Frequency sanity check
        # --------------------------------------------------

        min_freq = FRANGE[0]

        # Need several cycles of lowest frequency
        if total_duration < 4.0 / min_freq:
            print(f"Window too short for "f"fmin={min_freq} Hz")
            print(f"Need at least "f"{4.0/min_freq:.1f}s")
            continue

        # --------------------------------------------------
        # Extract event
        # --------------------------------------------------

        tr_event = tr.slice(starttime=start_event,endtime=end_event)
        #print(np.min(tr_event.data))
        #print(np.max(tr_event.data))
        #print(np.std(tr_event.data))
        duration = (tr_event.stats.npts / tr_event.stats.sampling_rate)
        #print(f"Samples={tr_event.stats.npts} " f"Duration={duration:.2f}s")

        if len(tr_event.data) == 0:
            continue

        # --------------------------------------------------
        # Convert to GWpy TimeSeries
        # --------------------------------------------------

        


        ts = TimeSeries(tr_event.data.astype(np.float64), sample_rate=tr_event.stats.sampling_rate, t0=0)

        # --------------------------------------------------
        # Q-transform
        # --------------------------------------------------

        try:

            qspec = ts.q_transform(
                frange=FRANGE,
                qrange=QRANGE,
                whiten=False
            )

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

        #print(f"Peak energy = {peak_energy:.4e}")
        #print(f"Mean energy = {mean_energy:.4e}")

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        if PLOT_RESULTS:
            fig = qspec.plot()
            ax = fig.gca()
            ax.set_title(f"Q-transform\n Center = {center_time}\n Window = ±{half_width}s")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_yscale("log")
            filename = (f"trigger_{i:04d}_window_{half_width}s.pdf")
            fig.savefig(OUTPUT_DIR / filename,dpi=300,bbox_inches="tight")
            plt.close(fig)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for r in results:
    print(
        f"Trigger={r['trigger_time']} | "
        f"Window=±{r['half_width']}s | "
        f"Peak={r['peak_energy']:.4e} | "
        f"Mean={r['mean_energy']:.4e}"
    )